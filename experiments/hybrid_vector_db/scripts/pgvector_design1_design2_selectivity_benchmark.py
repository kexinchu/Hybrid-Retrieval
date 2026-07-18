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
from pgvector_predicate_guidance_benchmark import FILTER_ATOMS, INDEX, TABLE, load_truth


MODES = ["original", "design1_bloom", "design1_bloom_design2_page"]


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


def configure_base(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def mode_table(args: argparse.Namespace, mode: str) -> str:
    if mode == "design1_bloom_design2_page":
        return args.design2_table
    return args.table


def mode_index(args: argparse.Namespace, mode: str) -> str:
    if mode == "design1_bloom_design2_page":
        return args.design2_index
    return args.index


def configure_mode(cur: psycopg.Cursor, args: argparse.Namespace, mode: str, filter_name: str) -> dict[str, object]:
    configure_base(cur, args)
    cur.execute("SELECT vector_hnsw_guidance_reset()")

    if mode == "original":
        cur.execute("SET hnsw.page_access = off")
        cur.execute("SET hnsw.index_page_access = off")
        return {}

    if mode == "design1_bloom":
        cur.execute("SET hnsw.page_access = off")
        cur.execute("SET hnsw.index_page_access = off")
    elif mode == "design1_bloom_design2_page":
        cur.execute(f"SET hnsw.page_access = {args.design2_page_access}")
        if args.design2_page_access != "off":
            cur.execute(f"SET hnsw.page_window = {int(args.page_window)}")
            cur.execute(f"SET hnsw.page_prefetch_min_items = {int(args.page_prefetch_min_items)}")
            cur.execute(f"SET hnsw.page_disable_after_no_merge = {int(args.page_disable_after_no_merge)}")
        cur.execute(f"SET hnsw.index_page_access = {args.index_page_access}")
    else:
        raise ValueError(f"unknown mode {mode}")

    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'bloom')",
        (mode_index(args, mode), FILTER_ATOMS[filter_name]),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    return json.loads(cur.fetchone()[0])


def run_query(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_id: int) -> tuple[list[int], dict[str, object]]:
    table = args.current_table
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {table}
        WHERE {predicate}
        ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
        LIMIT {int(args.k)}
        """,
        (int(query_id),),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile = json.loads(cur.fetchone()[0])
    return ids, profile


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    summary = out.with_name(out.stem + "_summary.csv")
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["mode"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    mode_order = {mode: i for i, mode in enumerate(MODES)}
    fields = [
        "filter",
        "filter_name",
        "mode",
        "ok",
        "errors",
        "recall_mean",
        "query_latency_mean_ms",
        "query_latency_p50_ms",
        "query_latency_p95_ms",
        "end_to_end_mean_ms",
        "vector_search_mean_ms",
        "visited_tuples_mean",
        "returned_tuples_mean",
        "guidance_checks_mean",
        "guidance_skips_mean",
        "page_batches_mean",
        "page_candidates_mean",
        "page_prefetches_mean",
        "page_distinct_pages_mean",
        "index_prefetches_mean",
        "speedup_query_vs_original",
        "speedup_e2e_vs_original",
    ]

    summaries: dict[tuple[str, str], dict[str, object]] = {}

    def mean(items, key):
        vals = [float(r[key]) for r in items]
        return statistics.fmean(vals) if vals else 0.0

    def p50(items, key):
        vals = [float(r[key]) for r in items]
        return statistics.median(vals) if vals else 0.0

    def p95(items, key):
        vals = sorted(float(r[key]) for r in items)
        return vals[int(0.95 * (len(vals) - 1))] if vals else 0.0

    for (filter_name, mode), items in groups.items():
        ok = [r for r in items if not r["error"]]
        first = items[0]
        summaries[(filter_name, mode)] = {
            "filter": first["filter"],
            "filter_name": filter_name,
            "mode": mode,
            "ok": len(ok),
            "errors": len(items) - len(ok),
            "recall_mean": mean(ok, "recall"),
            "query_latency_mean_ms": mean(ok, "query_latency_ms"),
            "query_latency_p50_ms": p50(ok, "query_latency_ms"),
            "query_latency_p95_ms": p95(ok, "query_latency_ms"),
            "end_to_end_mean_ms": mean(ok, "end_to_end_ms"),
            "vector_search_mean_ms": mean(ok, "vector_search_ms"),
            "visited_tuples_mean": mean(ok, "visited_tuples"),
            "returned_tuples_mean": mean(ok, "returned_tuples"),
            "guidance_checks_mean": mean(ok, "guidance_checks"),
            "guidance_skips_mean": mean(ok, "guidance_skips"),
            "page_batches_mean": mean(ok, "page_access_batches"),
            "page_candidates_mean": mean(ok, "page_access_candidates"),
            "page_prefetches_mean": mean(ok, "page_access_prefetches"),
            "page_distinct_pages_mean": mean(ok, "page_access_distinct_pages"),
            "index_prefetches_mean": mean(ok, "index_page_prefetches"),
            "speedup_query_vs_original": 0.0,
            "speedup_e2e_vs_original": 0.0,
        }

    for filter_name in {key[0] for key in summaries}:
        base = summaries.get((filter_name, "original"))
        if not base:
            continue
        base_query = float(base["query_latency_mean_ms"])
        base_e2e = float(base["end_to_end_mean_ms"])
        for mode in MODES:
            item = summaries.get((filter_name, mode))
            if not item:
                continue
            q = float(item["query_latency_mean_ms"])
            e = float(item["end_to_end_mean_ms"])
            item["speedup_query_vs_original"] = base_query / q if q else 0.0
            item["speedup_e2e_vs_original"] = base_e2e / e if e else 0.0

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, item in sorted(summaries.items(), key=lambda kv: (order.get(kv[0][0], 999), mode_order.get(kv[0][1], 999))):
            writer.writerow(item)
    print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original pgvector, Design 1, and Design 1 + Design 2 across selectivities.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--design2-table", default=TABLE)
    parser.add_argument("--design2-index", default=INDEX)
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_design1_design2_selectivity.csv"))
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--page-window", type=int, default=128)
    parser.add_argument("--page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--design2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-queries", type=int, default=5)
    args = parser.parse_args()

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    rng = random.Random(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filter",
        "filter_name",
        "mode",
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
        "guidance_matches",
        "guidance_skips",
        "page_access_batches",
        "page_access_candidates",
        "page_access_prefetches",
        "page_access_distance_runs",
        "page_access_distinct_pages",
        "index_page_prefetches",
        "activation_build_ms",
        "activation_rows",
        "activation_memory_bytes",
        "returned",
        "ids",
        "error",
    ]
    rows: list[dict[str, object]] = []

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        configure_base(cur, args)

        # Warm/build Design 1 fragments before timed runs.
        for filter_name, _, _ in filters:
            configure_mode(cur, args, "design1_bloom", filter_name)
            configure_mode(cur, args, "design1_bloom_design2_page", filter_name)
        cur.execute("SELECT vector_hnsw_guidance_reset()")

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
                                activation_profile, activation_ms = timed_ms(lambda m=mode: configure_mode(cur, args, m, filter_name))
                                args.current_table = mode_table(args, mode)
                                (ids, profile), query_ms = timed_ms(lambda: run_query(cur, args, predicate, qid))
                            except errors.QueryCanceled as exc:
                                error = exc.__class__.__name__
                                cur.execute("SET statement_timeout = 0")
                            except Exception as exc:  # noqa: BLE001
                                error = exc.__class__.__name__
                                try:
                                    cur.execute("ROLLBACK")
                                except Exception:
                                    pass
                            truth_ids = truth[(filter_name, qno)]
                            row = {
                                "filter": target_rate,
                                "filter_name": filter_name,
                                "mode": mode,
                                "query_no": qno,
                                "query_id": qid,
                                "repeat": repeat,
                                "run_order": run_order,
                                "recall": recall_at_k(ids, truth_ids, args.k) if not error else 0.0,
                                "activation_ms": activation_ms,
                                "query_latency_ms": query_ms,
                                "end_to_end_ms": activation_ms + query_ms,
                                "vector_search_ms": profile.get("vector_search_ms", 0.0),
                                "visited_tuples": profile.get("visited_tuples", 0),
                                "returned_tuples": profile.get("returned_tuples", 0),
                                "guidance_checks": profile.get("guidance_checks", 0),
                                "guidance_matches": profile.get("guidance_matches", 0),
                                "guidance_skips": profile.get("guidance_skips", 0),
                                "page_access_batches": profile.get("page_access_batches", 0),
                                "page_access_candidates": profile.get("page_access_candidates", 0),
                                "page_access_prefetches": profile.get("page_access_prefetches", 0),
                                "page_access_distance_runs": profile.get("page_access_distance_runs", 0),
                                "page_access_distinct_pages": profile.get("page_access_distinct_pages", 0),
                                "index_page_prefetches": profile.get("index_page_prefetches", 0),
                                "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
                                "activation_rows": activation_profile.get("last_cache_rows", 0),
                                "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
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
