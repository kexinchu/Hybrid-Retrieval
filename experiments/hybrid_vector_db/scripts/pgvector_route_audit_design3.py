from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS
from pgvector_sql_complexity_selectivity import COMPLEXITY_SUFFIXES, combined_predicate, load_truth


TABLE = "amazon_grocery_reviews_10m_pgvector"
ROUTES = ["default_planner", "force_hnsw_postfilter", "force_prefilter_exact"]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def vector_literal(value: object) -> str:
    if isinstance(value, str):
        return value
    return "[" + ",".join(f"{float(x):.7g}" for x in value) + "]"


def load_query_vectors(cur: psycopg.Cursor, query_ids: list[int]) -> dict[int, str]:
    cur.execute(
        f"""
        SELECT id, embedding
        FROM {TABLE}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    rows = cur.fetchall()
    vectors = {int(row[0]): vector_literal(row[1]) for row in rows}
    missing = [qid for qid in query_ids if qid not in vectors]
    if missing:
        raise RuntimeError(f"missing query vectors: {missing[:5]}")
    return vectors


def configure_common(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET jit = off")


def configure_route(cur: psycopg.Cursor, args: argparse.Namespace, route: str) -> None:
    configure_common(cur, args)
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_indexonlyscan = on")
    cur.execute("SET enable_bitmapscan = on")
    cur.execute("SET enable_sort = on")

    if route == "force_hnsw_postfilter":
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_bitmapscan = off")
        cur.execute("SET enable_sort = off")
    elif route == "force_prefilter_exact":
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_indexonlyscan = off")
        cur.execute("SET enable_sort = on")
    elif route != "default_planner":
        raise ValueError(route)


def flatten_plan(plan: dict) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    def walk(node: dict, depth: int) -> None:
        out.append(
            {
                "depth": depth,
                "node_type": node.get("Node Type", ""),
                "relation": node.get("Relation Name", ""),
                "index": node.get("Index Name", ""),
                "startup_cost": node.get("Startup Cost", 0.0),
                "total_cost": node.get("Total Cost", 0.0),
                "plan_rows": node.get("Plan Rows", 0),
                "filter": node.get("Filter", ""),
                "sort_key": "; ".join(node.get("Sort Key", []) or []),
            }
        )
        for child in node.get("Plans", []) or []:
            walk(child, depth + 1)

    walk(plan, 0)
    return out


def classify_plan(nodes: list[dict[str, object]]) -> str:
    indexes = " ".join(str(n["index"]) for n in nodes)
    node_types = " > ".join(str(n["node_type"]) for n in nodes)
    has_sort = any(n["node_type"] in ("Sort", "Incremental Sort", "Top-N heapsort") or n["sort_key"] for n in nodes)
    if "hnsw" in indexes.lower() or "embedding_hnsw" in indexes.lower():
        return "hnsw_post_filter"
    if has_sort:
        return "filter_then_exact_sort"
    if "Seq Scan" in node_types or "Bitmap Heap Scan" in node_types:
        return "filter_scan_no_vector_index"
    return "other"


def explain_route(cur: psycopg.Cursor, table: str, predicate: str, query_vector: str) -> tuple[str, str, float, float, int]:
    cur.execute(
        f"""
        EXPLAIN (FORMAT JSON)
        SELECT id
        FROM {table}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT 10
        """,
        (query_vector,),
    )
    raw = cur.fetchone()[0]
    plan_doc = json.loads(raw) if isinstance(raw, str) else raw
    plan = plan_doc[0]["Plan"]
    nodes = flatten_plan(plan)
    plan_text = " > ".join(
        f"{n['node_type']}:{n['index'] or n['relation']}".rstrip(":") for n in nodes
    )
    return classify_plan(nodes), plan_text, float(plan.get("Startup Cost", 0.0)), float(plan.get("Total Cost", 0.0)), int(plan.get("Plan Rows", 0))


def load_hnsw_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    text = cur.fetchone()[0]
    return json.loads(text) if isinstance(text, str) else text


def run_route(cur: psycopg.Cursor, table: str, predicate: str, query_vector: str, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {table}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT {int(k)}
        """,
        (query_vector,),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    return ids, load_hnsw_profile(cur)


def warmup_case(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_vectors: dict[int, str], query_ids: list[int]) -> None:
    if args.warmup_queries <= 0:
        return
    for qid in query_ids[: args.warmup_queries]:
        for route in ROUTES:
            try:
                configure_route(cur, args, route)
                run_route(cur, args.table, predicate, query_vectors[qid], args.k)
            except Exception:
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
                configure_common(cur, args)


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ids[:k]) & set(truth[:k])) / min(k, len(truth))


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    summary = out.with_name(out.stem + "_summary.csv")
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["complexity"]), str(row["route"])), []).append(row)

    by_case: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for (filter_name, complexity, route), items in groups.items():
        ok = [r for r in items if not r["error"]]
        first = items[0]
        lat = [float(r["latency_ms"]) for r in ok]
        row = {
            "filter": first["filter"],
            "filter_name": filter_name,
            "complexity": complexity,
            "route": route,
            "plan_class": first["plan_class"],
            "ok": len(ok),
            "errors": len(items) - len(ok),
            "latency_mean_ms": statistics.fmean(lat) if lat else 0.0,
            "latency_p50_ms": statistics.median(lat) if lat else 0.0,
            "recall_mean": statistics.fmean(float(r["recall"]) for r in ok) if ok else 0.0,
            "visited_mean": statistics.fmean(float(r["visited_tuples"]) for r in ok) if ok else 0.0,
            "returned_tuples_mean": statistics.fmean(float(r["returned_tuples"]) for r in ok) if ok else 0.0,
            "planner_total_cost": first["planner_total_cost"],
            "plan_text": first["plan_text"],
        }
        by_case.setdefault((filter_name, complexity), {})[route] = row

    fields = [
        "filter",
        "filter_name",
        "complexity",
        "route",
        "plan_class",
        "ok",
        "errors",
        "latency_mean_ms",
        "latency_p50_ms",
        "recall_mean",
        "visited_mean",
        "returned_tuples_mean",
        "planner_total_cost",
            "fastest_route",
            "fastest_plan_class",
            "default_route",
            "default_plan_matches_fastest",
            "oracle_speedup_vs_default",
            "plan_text",
        ]
    out_rows: list[dict[str, object]] = []
    for case, route_rows in by_case.items():
        fastest = min(
            (r for r in route_rows.values() if float(r["latency_mean_ms"]) > 0),
            key=lambda r: float(r["latency_mean_ms"]),
            default=None,
        )
        default = route_rows.get("default_planner")
        fastest_route = str(fastest["route"]) if fastest else ""
        fastest_plan_class = str(fastest["plan_class"]) if fastest else ""
        default_route = str(default["plan_class"]) if default else ""
        default_ms = float(default["latency_mean_ms"]) if default else 0.0
        fastest_ms = float(fastest["latency_mean_ms"]) if fastest else 0.0
        for route in ROUTES:
            row = route_rows.get(route)
            if not row:
                continue
            item = dict(row)
            item["fastest_route"] = fastest_route
            item["fastest_plan_class"] = fastest_plan_class
            item["default_route"] = default_route
            item["default_plan_matches_fastest"] = default_route == fastest_plan_class
            item["oracle_speedup_vs_default"] = default_ms / fastest_ms if default_ms and fastest_ms else 0.0
            out_rows.append(item)

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PostgreSQL/pgvector route choices for Design 3 runtime profile cache.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--complexities", nargs="*", default=["simple"])
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--statement-timeout-ms", type=int, default=10000)
    parser.add_argument("--warmup-queries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-queries", type=int, default=5)
    args = parser.parse_args()

    unknown = [name for name in args.complexities if name not in COMPLEXITY_SUFFIXES]
    if unknown:
        raise SystemExit(f"unknown complexities: {unknown}")

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    query_ids = [query_by_no[qno] for qno in query_nos]
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    rows: list[dict[str, object]] = []
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filter",
        "filter_name",
        "complexity",
        "query_no",
        "query_id",
        "repeat",
        "route",
        "plan_class",
        "plan_text",
        "planner_startup_cost",
        "planner_total_cost",
        "planner_rows",
        "latency_ms",
        "recall",
        "returned",
        "visited_tuples",
        "returned_tuples",
        "vector_search_ms",
        "error",
    ]

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        configure_common(cur, args)
        query_vectors = load_query_vectors(cur, query_ids)

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for filter_name, target, base_predicate in filters:
                for complexity in args.complexities:
                    predicate = combined_predicate(base_predicate, complexity)
                    warmup_case(cur, args, predicate, query_vectors, query_ids)
                    for idx, qno in enumerate(query_nos, start=1):
                        qid = query_by_no[qno]
                        qvec = query_vectors[qid]
                        for repeat in range(args.repeats):
                            route_order = ROUTES[:]
                            rng.shuffle(route_order)
                            for route in route_order:
                                error = ""
                                ids: list[int] = []
                                profile: dict[str, object] = {}
                                latency_ms = 0.0
                                try:
                                    configure_route(cur, args, route)
                                    plan_class, plan_text, startup_cost, total_cost, planner_rows = explain_route(cur, args.table, predicate, qvec)
                                    (ids, profile), latency_ms = timed_ms(lambda: run_route(cur, args.table, predicate, qvec, args.k))
                                except errors.QueryCanceled as exc:
                                    error = exc.__class__.__name__
                                    cur.execute("SET statement_timeout = 0")
                                    plan_class, plan_text, startup_cost, total_cost, planner_rows = "", "", 0.0, 0.0, 0
                                except Exception as exc:  # noqa: BLE001
                                    error = exc.__class__.__name__
                                    try:
                                        cur.execute("ROLLBACK")
                                    except Exception:
                                        pass
                                    plan_class, plan_text, startup_cost, total_cost, planner_rows = "", "", 0.0, 0.0, 0
                                valid_hnsw_profile = bool(profile.get("valid", False))
                                row = {
                                    "filter": target,
                                    "filter_name": filter_name,
                                    "complexity": complexity,
                                    "query_no": qno,
                                    "query_id": qid,
                                    "repeat": repeat,
                                    "route": route,
                                    "plan_class": plan_class,
                                    "plan_text": plan_text,
                                    "planner_startup_cost": startup_cost,
                                    "planner_total_cost": total_cost,
                                    "planner_rows": planner_rows,
                                    "latency_ms": latency_ms,
                                    "recall": recall_at_k(ids, truth[(filter_name, qno)], args.k) if not error else 0.0,
                                    "returned": len(ids),
                                    "visited_tuples": profile.get("visited_tuples", 0) if valid_hnsw_profile else 0,
                                    "returned_tuples": profile.get("returned_tuples", 0) if valid_hnsw_profile else 0,
                                    "vector_search_ms": profile.get("vector_search_ms", 0.0) if valid_hnsw_profile else 0.0,
                                    "error": error,
                                }
                                rows.append(row)
                                writer.writerow(row)
                                f.flush()
                        if args.progress_queries and idx % args.progress_queries == 0:
                            recent = [r for r in rows if r["filter_name"] == filter_name and r["complexity"] == complexity and not r["error"]]
                            parts = []
                            for route in ROUTES:
                                vals = [float(r["latency_ms"]) for r in recent if r["route"] == route]
                                if vals:
                                    parts.append(f"{route}={statistics.fmean(vals):.2f}ms")
                            print(f"progress filter={filter_name} complexity={complexity} queries={idx}/{len(query_nos)} " + " ".join(parts), flush=True)

    print(f"wrote {args.out}", flush=True)
    summarize(rows, args.out)


if __name__ == "__main__":
    main()
