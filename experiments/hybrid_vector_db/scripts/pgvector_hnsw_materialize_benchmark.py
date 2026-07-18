from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from pgvector_hnsw_page_access_group_benchmark import load_query_ids, load_query_vectors
from pgvector_prefilter_10m import TABLE


def restart_container(container: str) -> None:
    import subprocess

    subprocess.run(["docker", "restart", container], check=True, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 60.0
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"PostgreSQL did not become ready after restarting {container}: {last_error}")


def configure_session(
    cur: psycopg.Cursor,
    ef_search: int,
    iterative_scan: str,
    max_scan_tuples: int,
    statement_timeout_ms: int,
) -> None:
    cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(max_scan_tuples)}")
    cur.execute("SET hnsw.page_access = off")


def run_standard(
    cur: psycopg.Cursor,
    table: str,
    query_vector: str,
    k: int,
) -> tuple[list[int], float, str | None]:
    start = time.perf_counter()
    try:
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (query_vector,),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
        error = None
    except errors.QueryCanceled as exc:
        ids = []
        error = exc.__class__.__name__
    return ids, (time.perf_counter() - start) * 1000.0, error


def run_materialized(
    cur: psycopg.Cursor,
    index_name: str,
    query_vector: str,
    k: int,
    candidate_limit: int,
) -> tuple[list[int], float, dict[str, object], str | None]:
    start = time.perf_counter()
    try:
        cur.execute(
            """
            SELECT id
            FROM vector_hnsw_page_materialize(%s::regclass, %s::vector, %s, %s)
            ORDER BY rank
            """,
            (index_name, query_vector, int(k), int(candidate_limit)),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
        error = None
    except errors.QueryCanceled as exc:
        ids = []
        error = exc.__class__.__name__

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    try:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SELECT vector_hnsw_page_materialize_profile()")
        profile = json.loads(cur.fetchone()[0])
    except Exception:
        profile = {}
    return ids, elapsed_ms, profile, error


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if not row["standard_error"] and not row["materialized_error"]]
    standard = [float(row["standard_ms"]) for row in ok_rows]
    materialized = [float(row["materialized_ms"]) for row in ok_rows]
    speedups = [s / m for s, m in zip(standard, materialized) if m > 0]
    ratios = [
        float(row["distance_order_page_runs"]) / float(row["distinct_heap_pages"])
        for row in ok_rows
        if float(row["distinct_heap_pages"] or 0) > 0
    ]

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def median(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    def p95(values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        return sorted_values[int(0.95 * (len(sorted_values) - 1))]

    return {
        "queries": len(rows),
        "ok_queries": len(ok_rows),
        "same_ordered_ids": sum(1 for row in ok_rows if row["same_ordered_ids"]),
        "standard_mean_ms": mean(standard),
        "standard_p50_ms": median(standard),
        "standard_p95_ms": p95(standard),
        "materialized_mean_ms": mean(materialized),
        "materialized_p50_ms": median(materialized),
        "materialized_p95_ms": p95(materialized),
        "mean_speedup": mean(speedups),
        "median_speedup": median(speedups),
        "mean_candidates": mean([float(row["candidates"]) for row in ok_rows]),
        "mean_visible": mean([float(row["visible"]) for row in ok_rows]),
        "mean_heap_pages": mean([float(row["distinct_heap_pages"]) for row in ok_rows]),
        "mean_page_runs": mean([float(row["distance_order_page_runs"]) for row in ok_rows]),
        "mean_run_page_ratio": mean(ratios),
        "mean_index_ms": mean([float(row["index_ms"]) for row in ok_rows]),
        "mean_page_fetch_ms": mean([float(row["page_fetch_ms"]) for row in ok_rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standard pgvector HNSW scan with C page-materialized HNSW scan.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", required=True)
    parser.add_argument("--query-id-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="relaxed_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument("--isolate-cache", action="store_true")
    parser.add_argument("--container", default="hybrid-pgvector")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    query_ids = load_query_ids(args.query_id_csv, args.queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SELECT vector_dims('[1,2]'::vector)")
        vectors = load_query_vectors(cur, args.table, query_ids)

    if args.isolate_cache:
        restart_container(args.container)

    standard_results: dict[int, tuple[list[int], float, str | None]] = {}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure_session(cur, args.ef_search, args.iterative_scan, args.max_scan_tuples, args.statement_timeout_ms)
        for query_no, query_id in enumerate(query_ids):
            ids, elapsed_ms, error = run_standard(cur, args.table, vectors[query_id], args.k)
            standard_results[query_id] = (ids, elapsed_ms, error)
            if args.stream:
                print(f"standard q={query_no} id={query_id} ms={elapsed_ms:.2f} err={error or ''}", flush=True)

    if args.isolate_cache:
        restart_container(args.container)

    fieldnames = [
        "table",
        "index",
        "query_no",
        "query_id",
        "k",
        "candidate_limit",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "statement_timeout_ms",
        "standard_ms",
        "materialized_ms",
        "speedup",
        "standard_error",
        "materialized_error",
        "same_ordered_ids",
        "standard_ids",
        "materialized_ids",
        "valid",
        "candidates",
        "visible",
        "returned",
        "distance_order_page_runs",
        "distinct_heap_pages",
        "index_ms",
        "page_fetch_ms",
    ]

    rows: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure_session(cur, args.ef_search, args.iterative_scan, args.max_scan_tuples, args.statement_timeout_ms)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for query_no, query_id in enumerate(query_ids):
                standard_ids, standard_ms, standard_error = standard_results[query_id]
                materialized_ids, materialized_ms, profile, materialized_error = run_materialized(
                    cur,
                    args.index,
                    vectors[query_id],
                    args.k,
                    args.candidate_limit,
                )
                speedup = standard_ms / materialized_ms if materialized_ms > 0 else 0.0
                row = {
                    "table": args.table,
                    "index": args.index,
                    "query_no": query_no,
                    "query_id": query_id,
                    "k": args.k,
                    "candidate_limit": args.candidate_limit,
                    "ef_search": args.ef_search,
                    "iterative_scan": args.iterative_scan,
                    "max_scan_tuples": args.max_scan_tuples,
                    "statement_timeout_ms": args.statement_timeout_ms,
                    "standard_ms": standard_ms,
                    "materialized_ms": materialized_ms,
                    "speedup": speedup,
                    "standard_error": standard_error or "",
                    "materialized_error": materialized_error or "",
                    "same_ordered_ids": standard_ids == materialized_ids,
                    "standard_ids": ",".join(str(x) for x in standard_ids),
                    "materialized_ids": ",".join(str(x) for x in materialized_ids),
                    "valid": profile.get("valid", False),
                    "candidates": profile.get("candidates", 0),
                    "visible": profile.get("visible", 0),
                    "returned": profile.get("returned", 0),
                    "distance_order_page_runs": profile.get("distance_order_page_runs", 0),
                    "distinct_heap_pages": profile.get("distinct_heap_pages", 0),
                    "index_ms": profile.get("index_ms", 0.0),
                    "page_fetch_ms": profile.get("page_fetch_ms", 0.0),
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if args.stream:
                    print(
                        f"materialized q={query_no} id={query_id} ms={materialized_ms:.2f} "
                        f"speedup={speedup:.2f} same={standard_ids == materialized_ids} "
                        f"runs={row['distance_order_page_runs']} pages={row['distinct_heap_pages']}",
                        flush=True,
                    )

    summary = summarize(rows)
    if args.summary_out:
        with args.summary_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
