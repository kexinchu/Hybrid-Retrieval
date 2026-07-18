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


FILTERS: dict[str, str] = {
    "helpful_ge20": "helpful_vote >= 20",
    "grocery_long500": "main_category = 'Grocery' AND review_text_len >= 500",
    "grocery_helpful": "main_category = 'Grocery' AND helpful_vote >= 1",
    "rating5_price_le10": "has_price AND price <= 10 AND rating = 5",
}


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def configure(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute("SET hnsw.page_access = off")
    cur.execute(f"SET hnsw.index_page_access = {args.index_page_access}")
    cur.execute("SET jit = off")


def run_standard(cur: psycopg.Cursor, table: str, predicate: str, query_vector: str, k: int):
    try:
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
        error = ""
    except errors.QueryCanceled as exc:
        ids = []
        error = exc.__class__.__name__
    cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile = json.loads(cur.fetchone()[0])
    return ids, profile, error


def run_cache(cur: psycopg.Cursor, index: str, filter_name: str, query_vector: str, k: int, candidate_limit: int):
    try:
        cur.execute(
            """
            SELECT id
            FROM vector_hnsw_metadata_filter_search(%s::regclass, %s::vector, %s, %s, %s)
            ORDER BY rank
            """,
            (index, query_vector, int(k), int(candidate_limit), filter_name),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
        error = ""
    except errors.QueryCanceled as exc:
        ids = []
        error = exc.__class__.__name__
    cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT vector_hnsw_metadata_filter_profile()")
    profile = json.loads(cur.fetchone()[0])
    return ids, profile, error


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok = [row for row in rows if not row["standard_error"] and not row["cache_error"]]
    standard = [float(row["standard_ms"]) for row in ok]
    cache = [float(row["cache_ms"]) for row in ok]
    speedups = [s / c for s, c in zip(standard, cache) if c > 0]

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def median(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    def p95(values: list[float]) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        return values[int(0.95 * (len(values) - 1))]

    return {
        "rows": len(rows),
        "ok": len(ok),
        "same_ordered_ids": sum(1 for row in ok if row["same_ordered_ids"]),
        "same_set_ids": sum(1 for row in ok if row["same_set_ids"]),
        "standard_mean_ms": mean(standard),
        "standard_p50_ms": median(standard),
        "standard_p95_ms": p95(standard),
        "cache_mean_ms": mean(cache),
        "cache_p50_ms": median(cache),
        "cache_p95_ms": p95(cache),
        "mean_speedup": mean(speedups),
        "median_speedup": median(speedups),
        "p05_speedup": sorted(speedups)[int(0.05 * (len(speedups) - 1))] if speedups else 0.0,
        "p95_speedup": p95(speedups),
        "mean_standard_visited": mean([float(row["standard_visited_tuples"]) for row in ok]),
        "mean_standard_returned_tids": mean([float(row["standard_returned_tuples"]) for row in ok]),
        "mean_cache_candidates": mean([float(row["cache_candidates"]) for row in ok]),
        "mean_cache_matches": mean([float(row["cache_matches"]) for row in ok]),
        "mean_cache_search_ms": mean([float(row["cache_search_ms"]) for row in ok]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standard pgvector filtered HNSW with backend-local metadata cache filtering.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--filter-name", required=True, choices=sorted(FILTERS))
    parser.add_argument("--query-id-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200000)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    predicate = FILTERS[args.filter_name]
    query_ids = load_query_ids(args.query_id_csv, args.queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args)
        vectors = load_query_vectors(cur, args.table, query_ids)

        _, build_ms = timed_ms(
            lambda: cur.execute(
                "SELECT vector_hnsw_metadata_cache_build(%s::regclass, %s)",
                (args.index, args.filter_name),
            ).fetchone()
        )
        cur.execute("SELECT vector_hnsw_metadata_filter_profile()")
        build_profile = json.loads(cur.fetchone()[0])

        fieldnames = [
            "table",
            "index",
            "filter_name",
            "query_no",
            "query_id",
            "k",
            "candidate_limit",
            "ef_search",
            "iterative_scan",
            "max_scan_tuples",
            "index_page_access",
            "cache_build_wall_ms",
            "cache_build_ms",
            "cache_rows",
            "standard_ms",
            "cache_ms",
            "speedup",
            "standard_error",
            "cache_error",
            "standard_returned",
            "cache_returned",
            "same_ordered_ids",
            "same_set_ids",
            "standard_ids",
            "cache_ids",
            "standard_vector_ms",
            "standard_visited_tuples",
            "standard_returned_tuples",
            "standard_index_page_element_loads",
            "cache_candidates",
            "cache_checks",
            "cache_matches",
            "cache_search_ms",
        ]

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for query_no, query_id in enumerate(query_ids):
                query_vector = vectors[query_id]
                configure(cur, args)
                (standard_ids, standard_profile, standard_error), standard_ms = timed_ms(
                    lambda: run_standard(cur, args.table, predicate, query_vector, args.k)
                )
                configure(cur, args)
                (cache_ids, cache_profile, cache_error), cache_ms = timed_ms(
                    lambda: run_cache(cur, args.index, args.filter_name, query_vector, args.k, args.candidate_limit)
                )
                speedup = standard_ms / cache_ms if cache_ms > 0 else 0.0
                row = {
                    "table": args.table,
                    "index": args.index,
                    "filter_name": args.filter_name,
                    "query_no": query_no,
                    "query_id": query_id,
                    "k": args.k,
                    "candidate_limit": args.candidate_limit,
                    "ef_search": args.ef_search,
                    "iterative_scan": args.iterative_scan,
                    "max_scan_tuples": args.max_scan_tuples,
                    "index_page_access": args.index_page_access,
                    "cache_build_wall_ms": build_ms,
                    "cache_build_ms": build_profile.get("cache_build_ms", 0.0),
                    "cache_rows": build_profile.get("cache_rows", 0),
                    "standard_ms": standard_ms,
                    "cache_ms": cache_ms,
                    "speedup": speedup,
                    "standard_error": standard_error,
                    "cache_error": cache_error,
                    "standard_returned": len(standard_ids),
                    "cache_returned": len(cache_ids),
                    "same_ordered_ids": standard_ids == cache_ids,
                    "same_set_ids": set(standard_ids) == set(cache_ids),
                    "standard_ids": ",".join(str(x) for x in standard_ids),
                    "cache_ids": ",".join(str(x) for x in cache_ids),
                    "standard_vector_ms": standard_profile.get("vector_search_ms", 0.0),
                    "standard_visited_tuples": standard_profile.get("visited_tuples", 0),
                    "standard_returned_tuples": standard_profile.get("returned_tuples", 0),
                    "standard_index_page_element_loads": standard_profile.get("index_page_element_loads", 0),
                    "cache_candidates": cache_profile.get("candidates", 0),
                    "cache_checks": cache_profile.get("cache_checks", 0),
                    "cache_matches": cache_profile.get("cache_matches", 0),
                    "cache_search_ms": cache_profile.get("search_ms", 0.0),
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                if args.stream:
                    print(
                        f"q={query_no} id={query_id} std={standard_ms:.2f} "
                        f"cache={cache_ms:.2f} speedup={speedup:.2f} "
                        f"same={row['same_ordered_ids']} ret={len(standard_ids)}/{len(cache_ids)}",
                        flush=True,
                    )

    summary = summarize(rows)
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
