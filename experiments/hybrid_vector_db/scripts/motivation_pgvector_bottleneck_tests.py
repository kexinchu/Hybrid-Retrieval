from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from common_pg import pg_config_from_env, require_psycopg


TABLE = "amazon_grocery_reviews_10m_pgvector"
HNSW_INDEX = "amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx"

PREDICATES: list[tuple[str, str, str]] = [
    ("popular_ge1000", "50%", "item_rating_number >= 1000"),
    ("price_10_to_20", "20%", "has_price AND price > 10 AND price <= 20"),
    ("helpful_ge20", "0.5%", "helpful_vote >= 20"),
    ("grocery_long500", "0.2%", "main_category = 'Grocery' AND review_text_len >= 500"),
]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def timed_stream_fetch(cur: Any, sql: str, params: tuple[Any, ...] = ()) -> tuple[int, float]:
    start = now_ms()
    cur.execute(sql, params)
    rows = 0
    while True:
        batch = cur.fetchmany(10000)
        if not batch:
            break
        rows += len(batch)
    return rows, now_ms() - start


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def walk_plan(node: dict[str, Any], out: list[dict[str, Any]]) -> None:
    out.append(node)
    for child in node.get("Plans", []) or []:
        walk_plan(child, out)


def summarize_plan(plan_doc: Any) -> dict[str, Any]:
    plan_doc = parse_json(plan_doc)
    top = plan_doc[0]
    nodes: list[dict[str, Any]] = []
    walk_plan(top["Plan"], nodes)
    node_names = []
    index_names = []
    rows_removed = 0.0
    shared_hit = 0.0
    shared_read = 0.0
    shared_dirtied = 0.0
    temp_read = 0.0
    temp_written = 0.0

    for node in nodes:
        name = str(node.get("Node Type", ""))
        if node.get("Index Name"):
            name += f":{node['Index Name']}"
            index_names.append(str(node["Index Name"]))
        if node.get("Relation Name"):
            name += f":{node['Relation Name']}"
        node_names.append(name)
        rows_removed += float(node.get("Rows Removed by Filter", 0) or 0)
        shared_hit += float(node.get("Shared Hit Blocks", 0) or 0)
        shared_read += float(node.get("Shared Read Blocks", 0) or 0)
        shared_dirtied += float(node.get("Shared Dirtied Blocks", 0) or 0)
        temp_read += float(node.get("Temp Read Blocks", 0) or 0)
        temp_written += float(node.get("Temp Written Blocks", 0) or 0)

    return {
        "planning_ms": float(top.get("Planning Time", 0.0)),
        "execution_ms": float(top.get("Execution Time", 0.0)),
        "root_node": top["Plan"].get("Node Type", ""),
        "actual_rows": float(top["Plan"].get("Actual Rows", 0) or 0),
        "node_chain": " > ".join(node_names),
        "indexes": ",".join(dict.fromkeys(index_names)),
        "rows_removed_by_filter": rows_removed,
        "shared_hit_blocks": shared_hit,
        "shared_read_blocks": shared_read,
        "shared_dirtied_blocks": shared_dirtied,
        "temp_read_blocks": temp_read,
        "temp_written_blocks": temp_written,
    }


def explain_analyze(cur: Any, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
    return summarize_plan(cur.fetchone()[0])


def try_profile(cur: Any, fn: str) -> dict[str, Any]:
    try:
        cur.execute(f"SELECT {fn}()")
        value = cur.fetchone()[0]
        if value is None:
            return {}
        return parse_json(value)
    except Exception:
        return {}


def flatten_hnsw_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "hnsw_profile_valid": bool(profile.get("valid", False)),
        "hnsw_vector_ms": float(profile.get("vector_search_ms", 0.0) or 0.0),
        "hnsw_visited": float(profile.get("visited_tuples", 0.0) or 0.0),
        "hnsw_returned": float(profile.get("returned_tuples", 0.0) or 0.0),
    }


def flatten_qual_profile(profile: dict[str, Any]) -> dict[str, Any]:
    entries = profile.get("entries", []) or []
    true_count = 0.0
    false_count = 0.0
    for entry in entries:
        true_count += float(entry.get("true", 0.0) or 0.0)
        false_count += float(entry.get("false", 0.0) or 0.0)
    return {
        "qual_ms": float(profile.get("qual_ms", 0.0) or 0.0),
        "qual_calls": float(profile.get("qual_calls", 0.0) or 0.0),
        "qual_true": true_count,
        "qual_false": false_count,
    }


def reset_profiles(cur: Any, have_hnsw_profile: bool, have_qual_profile: bool) -> None:
    if have_hnsw_profile:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    if have_qual_profile:
        cur.execute("SELECT hybrid_qual_profile_reset()")


def load_query_vectors(cur: Any, queries: int) -> list[tuple[int, str]]:
    cur.execute(
        f"""
        SELECT id, embedding::text
        FROM {TABLE}
        WHERE embedding IS NOT NULL
        ORDER BY id
        LIMIT %s
        """,
        (queries,),
    )
    return [(int(row[0]), str(row[1])) for row in cur.fetchall()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[int(0.95 * (len(values) - 1))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["case"])), []).append(row)

    out: list[dict[str, Any]] = []
    order = {name: i for i, (name, _, _) in enumerate(PREDICATES)}
    for (filter_name, case), items in sorted(groups.items(), key=lambda x: (order.get(x[0][0], 999), x[0][1])):
        elapsed = [float(row["elapsed_ms"]) for row in items if row.get("elapsed_ms") not in ("", None)]
        exec_ms = [float(row["execution_ms"]) for row in items if row.get("execution_ms") not in ("", None)]
        rows_count = [float(row["rows"]) for row in items if row.get("rows") not in ("", None)]
        out.append(
            {
                "filter_name": filter_name,
                "target": items[0]["target"],
                "case": case,
                "runs": len(items),
                "rows_mean": mean(rows_count),
                "elapsed_mean_ms": mean(elapsed),
                "elapsed_p95_ms": p95(elapsed),
                "execution_mean_ms": mean(exec_ms),
                "rows_removed_mean": mean([float(row["rows_removed_by_filter"]) for row in items]),
                "shared_hit_blocks_mean": mean([float(row["shared_hit_blocks"]) for row in items]),
                "shared_read_blocks_mean": mean([float(row["shared_read_blocks"]) for row in items]),
                "hnsw_vector_ms_mean": mean([float(row["hnsw_vector_ms"]) for row in items]),
                "hnsw_visited_mean": mean([float(row["hnsw_visited"]) for row in items]),
                "qual_ms_mean": mean([float(row["qual_ms"]) for row in items]),
                "qual_calls_mean": mean([float(row["qual_calls"]) for row in items]),
                "indexes": items[0]["indexes"],
                "node_chain": items[0]["node_chain"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Motivation tests for PostgreSQL + pgvector hybrid-search bottlenecks.")
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/motivation_pgvector_bottleneck_tests.csv"))
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["strict_order", "relaxed_order", "off"])
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    parser.add_argument("--skip-full-export", action="store_true")
    parser.add_argument("--jit", choices=["on", "off"], default="off")
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    rows: list[dict[str, Any]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
            cur.execute(f"SET jit = {args.jit}")
            cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
            cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
            if args.iterative_scan == "off":
                cur.execute("SET hnsw.iterative_scan = off")
            else:
                cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")

            query_vectors = load_query_vectors(cur, args.queries)
            cur.execute("SELECT to_regprocedure('vector_hnsw_last_scan_profile()'), to_regprocedure('hybrid_qual_profile_last()')")
            have_hnsw_profile, have_qual_profile = cur.fetchone()

            for query_id, vector in query_vectors:
                print(f"[hnsw] no_filter q={query_id}", flush=True)
                sql = f"SELECT id FROM {TABLE} ORDER BY embedding <-> %s::vector LIMIT {int(args.k)}"
                start = now_ms()
                cur.execute(sql, (vector,))
                result_ids = [int(row[0]) for row in cur.fetchall()]
                elapsed_ms = now_ms() - start
                reset_profiles(cur, bool(have_hnsw_profile), bool(have_qual_profile))
                plan = explain_analyze(cur, sql, (vector,))
                hnsw_profile = try_profile(cur, "vector_hnsw_last_scan_profile") if have_hnsw_profile else {}
                qual_profile = try_profile(cur, "hybrid_qual_profile_last") if have_qual_profile else {}
                rows.append(
                    {
                        "filter_name": "no_filter",
                        "target": "100%",
                        "case": "pgvector_hnsw_baseline",
                        "query_id": query_id,
                        "rows": len(result_ids),
                        "elapsed_ms": elapsed_ms,
                        **plan,
                        **flatten_hnsw_profile(hnsw_profile),
                        **flatten_qual_profile(qual_profile),
                    }
                )

            for filter_name, target, predicate in PREDICATES:
                print(f"[sql] {filter_name}", flush=True)
                for case, sql in [
                    ("sql_limit_500", f"SELECT id FROM {TABLE} WHERE {predicate} LIMIT 500"),
                    ("sql_limit_50000", f"SELECT id FROM {TABLE} WHERE {predicate} LIMIT 50000"),
                ]:
                    rows_count, elapsed_ms = timed_stream_fetch(cur, sql)
                    plan = explain_analyze(cur, sql)
                    rows.append(
                        {
                            "filter_name": filter_name,
                            "target": target,
                            "case": case,
                            "query_id": "",
                            "rows": rows_count,
                            "elapsed_ms": elapsed_ms,
                            **plan,
                            **flatten_hnsw_profile({}),
                            **flatten_qual_profile({}),
                        }
                    )

                if not args.skip_full_export:
                    for case, sql in [
                        ("sql_full_export", f"SELECT id FROM {TABLE} WHERE {predicate}"),
                        ("sql_full_export_order_id", f"SELECT id FROM {TABLE} WHERE {predicate} ORDER BY id"),
                    ]:
                        rows_count, elapsed_ms = timed_stream_fetch(cur, sql)
                        plan = explain_analyze(cur, sql)
                        rows.append(
                            {
                                "filter_name": filter_name,
                                "target": target,
                                "case": case,
                                "query_id": "",
                                "rows": rows_count,
                                "elapsed_ms": elapsed_ms,
                                **plan,
                                **flatten_hnsw_profile({}),
                                **flatten_qual_profile({}),
                            }
                        )

                for query_id, vector in query_vectors:
                    print(f"[hnsw] {filter_name} q={query_id}", flush=True)
                    sql = (
                        f"SELECT id FROM {TABLE} "
                        f"WHERE {predicate} "
                        "ORDER BY embedding <-> %s::vector "
                        f"LIMIT {int(args.k)}"
                    )
                    start = now_ms()
                    cur.execute(sql, (vector,))
                    result_ids = [int(row[0]) for row in cur.fetchall()]
                    elapsed_ms = now_ms() - start
                    reset_profiles(cur, bool(have_hnsw_profile), bool(have_qual_profile))
                    plan = explain_analyze(cur, sql, (vector,))
                    hnsw_profile = try_profile(cur, "vector_hnsw_last_scan_profile") if have_hnsw_profile else {}
                    qual_profile = try_profile(cur, "hybrid_qual_profile_last") if have_qual_profile else {}
                    rows.append(
                        {
                            "filter_name": filter_name,
                            "target": target,
                            "case": "pgvector_filtered_hnsw",
                            "query_id": query_id,
                            "rows": len(result_ids),
                            "elapsed_ms": elapsed_ms,
                            **plan,
                            **flatten_hnsw_profile(hnsw_profile),
                            **flatten_qual_profile(qual_profile),
                        }
                    )

    write_csv(args.out, rows)
    summary = summarize_rows(rows)
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    write_csv(summary_out, summary)
    print(f"wrote {args.out}")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
