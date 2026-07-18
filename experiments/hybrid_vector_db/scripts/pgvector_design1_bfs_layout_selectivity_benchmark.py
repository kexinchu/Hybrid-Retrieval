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
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
from pgvector_predicate_guidance_benchmark import FILTER_ATOMS, load_truth


SOURCE_TABLE = "amazon_grocery_reviews_10m_pgvector_vector_clustered_10m"
INSERTION_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_insert"
INSERTION_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_insert_hnsw"
BFS_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs"
BFS_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs_hnsw"
MODES = ["original", "design1_bloom", "design1_bloom_bfs_layout"]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def ensure_functions(cur: psycopg.Cursor) -> None:
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in functions:
        cur.execute(sql)
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def prepare_same_graph_layouts(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    """Build twin tables and HNSW indexes where the logical graph is reproducible."""
    source = qident(args.source_table)
    insertion_table = qident(args.insertion_table)
    bfs_table = qident(args.bfs_table)
    insertion_index = qident(args.insertion_index)
    bfs_index = qident(args.bfs_index)
    order_sql = args.copy_order_by.strip()
    order_clause = f" ORDER BY {order_sql}" if order_sql else ""

    print("preparing same-graph twin tables", flush=True)
    cur.execute("SET statement_timeout = 0")
    cur.execute(f"DROP TABLE IF EXISTS {insertion_table} CASCADE")
    cur.execute(f"DROP TABLE IF EXISTS {bfs_table} CASCADE")
    logged = "" if args.logged_tables else "UNLOGGED "
    cur.execute(f"CREATE {logged}TABLE {insertion_table} AS SELECT * FROM {source}{order_clause}")
    cur.execute(f"CREATE {logged}TABLE {bfs_table} AS SELECT * FROM {source}{order_clause}")
    cur.execute(f"CREATE INDEX {qident(args.insertion_table + '_id_idx')} ON {insertion_table} (id)")
    cur.execute(f"CREATE INDEX {qident(args.bfs_table + '_id_idx')} ON {bfs_table} (id)")
    cur.execute(f"ANALYZE {insertion_table}")
    cur.execute(f"ANALYZE {bfs_table}")

    cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
    cur.execute("SET max_parallel_maintenance_workers = 0")
    cur.execute("SET hnsw.build_page_order = insertion")
    print("building insertion-layout HNSW index", flush=True)
    cur.execute(f"CREATE INDEX {insertion_index} ON {insertion_table} USING hnsw (embedding vector_l2_ops)")
    cur.execute("SET hnsw.build_page_order = bfs")
    print("building BFS-layout HNSW index from the same deterministic graph", flush=True)
    cur.execute(f"CREATE INDEX {bfs_index} ON {bfs_table} USING hnsw (embedding vector_l2_ops)")
    cur.execute("SET hnsw.build_page_order = insertion")
    cur.execute(f"ANALYZE {insertion_table}")
    cur.execute(f"ANALYZE {bfs_table}")


def configure_base(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(args.metadata_cache_max_mb)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def mode_table_index(args: argparse.Namespace, mode: str) -> tuple[str, str]:
    if mode == "design1_bloom_bfs_layout":
        return args.bfs_table, args.bfs_index
    return args.insertion_table, args.insertion_index


def activate_mode(cur: psycopg.Cursor, args: argparse.Namespace, mode: str, filter_name: str, preload: bool = False) -> dict[str, object]:
    configure_base(cur, args)
    if preload:
        cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    table, index = mode_table_index(args, mode)
    if mode == "original":
        return {"table": table, "index": index}

    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'bloom')",
        (index, FILTER_ATOMS[filter_name]),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    profile = json.loads(cur.fetchone()[0])
    profile["table"] = table
    profile["index"] = index
    return profile


def run_query(cur: psycopg.Cursor, table: str, predicate: str, query_id: int, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {table}
        WHERE {predicate}
        ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
        LIMIT {int(k)}
        """,
        (int(query_id),),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def run_unfiltered_query(cur: psycopg.Cursor, table: str, query_id: int, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {table}
        ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
        LIMIT {int(k)}
        """,
        (int(query_id),),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def verify_same_logical_graph(cur: psycopg.Cursor, args: argparse.Namespace, query_nos, query_by_no) -> None:
    if not args.verify_same_graph:
        return

    print("verifying same logical HNSW graph before benchmark", flush=True)
    configure_base(cur, args)
    mismatches = []
    for qno in query_nos[: args.verify_queries]:
        qid = query_by_no[qno]
        insert_ids, insert_profile = run_unfiltered_query(cur, args.insertion_table, qid, args.verify_k)
        bfs_ids, bfs_profile = run_unfiltered_query(cur, args.bfs_table, qid, args.verify_k)
        insert_visited = int(insert_profile.get("visited_tuples", -1))
        bfs_visited = int(bfs_profile.get("visited_tuples", -1))
        if insert_ids != bfs_ids or insert_visited != bfs_visited:
            mismatches.append(
                {
                    "query_no": qno,
                    "query_id": qid,
                    "insert_visited": insert_visited,
                    "bfs_visited": bfs_visited,
                    "insert_ids": insert_ids,
                    "bfs_ids": bfs_ids,
                }
            )

    if mismatches:
        sample = json.dumps(mismatches[:3], ensure_ascii=False)
        raise RuntimeError(
            "same-graph verification failed; insertion and BFS indexes differ logically. "
            f"sample={sample}"
        )
    print(f"same-graph verification passed for {min(len(query_nos), args.verify_queries)} queries", flush=True)


def warmup(cur: psycopg.Cursor, args: argparse.Namespace, filters, query_nos, query_by_no) -> None:
    if args.warmup_queries <= 0:
        return
    warm_nos = query_nos[: args.warmup_queries]
    warm_filter_names = set(args.warmup_filter_names or [])
    for filter_name, _, predicate in filters:
        if warm_filter_names and filter_name not in warm_filter_names:
            continue
        for mode in MODES:
            profile = activate_mode(cur, args, mode, filter_name)
            table = str(profile["table"])
            for qno in warm_nos:
                try:
                    print(f"warmup mode={mode} filter={filter_name} q={qno}", flush=True)
                    run_query(cur, table, predicate, query_by_no[qno], args.k)
                except Exception:
                    cur.execute("ROLLBACK")
                    configure_base(cur, args)
            cur.execute("SELECT vector_hnsw_guidance_reset()")


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    summary = out.with_name(out.stem + "_summary.csv")
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["mode"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    mode_order = {mode: i for i, mode in enumerate(MODES)}

    def mean(items, key):
        vals = [float(r[key]) for r in items]
        return statistics.fmean(vals) if vals else 0.0

    def p95(items, key):
        vals = sorted(float(r[key]) for r in items)
        return vals[int(0.95 * (len(vals) - 1))] if vals else 0.0

    fields = [
        "filter",
        "filter_name",
        "mode",
        "table",
        "index",
        "ok",
        "errors",
        "recall_mean",
        "end_to_end_mean_ms",
        "end_to_end_p95_ms",
        "query_latency_mean_ms",
        "activation_mean_ms",
        "vector_search_mean_ms",
        "visited_tuples_mean",
        "returned_tuples_mean",
        "guidance_checks_mean",
        "guidance_skips_mean",
        "index_element_runs_mean",
        "index_element_distinct_pages_mean",
        "speedup_vs_original",
        "speedup_vs_design1",
    ]
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    for (filter_name, mode), items in groups.items():
        ok = [r for r in items if not r["error"]]
        first = items[0]
        summaries[(filter_name, mode)] = {
            "filter": first["filter"],
            "filter_name": filter_name,
            "mode": mode,
            "table": first["table"],
            "index": first["index"],
            "ok": len(ok),
            "errors": len(items) - len(ok),
            "recall_mean": mean(ok, "recall"),
            "end_to_end_mean_ms": mean(ok, "end_to_end_ms"),
            "end_to_end_p95_ms": p95(ok, "end_to_end_ms"),
            "query_latency_mean_ms": mean(ok, "query_latency_ms"),
            "activation_mean_ms": mean(ok, "activation_ms"),
            "vector_search_mean_ms": mean(ok, "vector_search_ms"),
            "visited_tuples_mean": mean(ok, "visited_tuples"),
            "returned_tuples_mean": mean(ok, "returned_tuples"),
            "guidance_checks_mean": mean(ok, "guidance_checks"),
            "guidance_skips_mean": mean(ok, "guidance_skips"),
            "index_element_runs_mean": mean(ok, "index_page_element_runs"),
            "index_element_distinct_pages_mean": mean(ok, "index_page_element_distinct_pages"),
            "speedup_vs_original": 0.0,
            "speedup_vs_design1": 0.0,
        }

    for filter_name in {key[0] for key in summaries}:
        original = summaries.get((filter_name, "original"))
        design1 = summaries.get((filter_name, "design1_bloom"))
        if not original:
            continue
        base = float(original["end_to_end_mean_ms"])
        d1 = float(design1["end_to_end_mean_ms"]) if design1 else 0.0
        for mode in MODES:
            item = summaries.get((filter_name, mode))
            if not item:
                continue
            val = float(item["end_to_end_mean_ms"])
            item["speedup_vs_original"] = base / val if val else 0.0
            item["speedup_vs_design1"] = d1 / val if d1 and val else 0.0

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, item in sorted(summaries.items(), key=lambda kv: (order.get(kv[0][0], 999), mode_order.get(kv[0][1], 999))):
            writer.writerow(item)
    print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare insertion-order HNSW with Design 1 and BFS-layout Design 2.")
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--insertion-table", default=INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument("--prepare-same-graph-layouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-order-by", default="id")
    parser.add_argument("--maintenance-work-mem", default="32GB")
    parser.add_argument("--logged-tables", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-same-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-queries", type=int, default=8)
    parser.add_argument("--verify-k", type=int, default=20)
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--warmup-filter-names", nargs="*", default=["popular_ge1000"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--metadata-cache-max-mb", type=int, default=1024)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-queries", type=int, default=10)
    args = parser.parse_args()

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filter",
        "filter_name",
        "mode",
        "table",
        "index",
        "query_no",
        "query_id",
        "repeat",
        "run_order",
        "recall",
        "activation_ms",
        "query_latency_ms",
        "end_to_end_ms",
        "vector_search_ms",
        "visited_tuples",
        "returned_tuples",
        "guidance_checks",
        "guidance_skips",
        "index_page_element_runs",
        "index_page_element_distinct_pages",
        "returned",
        "ids",
        "error",
    ]

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        if args.prepare_same_graph_layouts:
            prepare_same_graph_layouts(cur, args)
        configure_base(cur, args)
        verify_same_logical_graph(cur, args, query_nos, query_by_no)

        # Build/load Design 1 fragments for both indexes before timing.
        for filter_name, _, _ in filters:
            for mode in ["design1_bloom", "design1_bloom_bfs_layout"]:
                activate_mode(cur, args, mode, filter_name, preload=True)
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        configure_base(cur, args)

        print("warming up", flush=True)
        warmup(cur, args, filters, query_nos, query_by_no)

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for filter_name, target_rate, predicate in filters:
                for idx, qno in enumerate(query_nos, start=1):
                    qid = query_by_no[qno]
                    for repeat in range(args.repeats):
                        run_modes = MODES[:]
                        rng.shuffle(run_modes)
                        for run_order, mode in enumerate(run_modes):
                            error = ""
                            activation_profile: dict[str, object] = {}
                            ids: list[int] = []
                            profile: dict[str, object] = {}
                            activation_ms = 0.0
                            query_ms = 0.0
                            try:
                                activation_profile, activation_ms = timed_ms(lambda m=mode: activate_mode(cur, args, m, filter_name))
                                table = str(activation_profile["table"])
                                index = str(activation_profile["index"])
                                (ids, profile), query_ms = timed_ms(lambda: run_query(cur, table, predicate, qid, args.k))
                            except errors.QueryCanceled as exc:
                                error = exc.__class__.__name__
                                cur.execute("SET statement_timeout = 0")
                                table, index = mode_table_index(args, mode)
                            except Exception as exc:  # noqa: BLE001
                                error = exc.__class__.__name__
                                try:
                                    cur.execute("ROLLBACK")
                                except Exception:
                                    pass
                                table, index = mode_table_index(args, mode)
                            row = {
                                "filter": target_rate,
                                "filter_name": filter_name,
                                "mode": mode,
                                "table": table,
                                "index": index,
                                "query_no": qno,
                                "query_id": qid,
                                "repeat": repeat,
                                "run_order": run_order,
                                "recall": recall_at_k(ids, truth[(filter_name, qno)], args.k) if not error else 0.0,
                                "activation_ms": activation_ms,
                                "query_latency_ms": query_ms,
                                "end_to_end_ms": activation_ms + query_ms,
                                "vector_search_ms": profile.get("vector_search_ms", 0.0),
                                "visited_tuples": profile.get("visited_tuples", 0),
                                "returned_tuples": profile.get("returned_tuples", 0),
                                "guidance_checks": profile.get("guidance_checks", 0),
                                "guidance_skips": profile.get("guidance_skips", 0),
                                "index_page_element_runs": profile.get("index_page_element_runs", 0),
                                "index_page_element_distinct_pages": profile.get("index_page_element_distinct_pages", 0),
                                "returned": len(ids),
                                "ids": ",".join(str(x) for x in ids),
                                "error": error,
                            }
                            rows.append(row)
                            writer.writerow(row)
                            f.flush()
                    if args.progress_queries and idx % args.progress_queries == 0:
                        recent = [r for r in rows if r["filter_name"] == filter_name and not r["error"]]
                        parts = []
                        for mode in MODES:
                            vals = [float(r["end_to_end_ms"]) for r in recent if r["mode"] == mode]
                            if vals:
                                parts.append(f"{mode}={statistics.fmean(vals):.2f}ms")
                        print(f"progress filter={filter_name} queries={idx}/{len(query_nos)} " + " ".join(parts), flush=True)

    print(f"wrote {args.out}", flush=True)
    summarize(rows, args.out)


if __name__ == "__main__":
    main()
