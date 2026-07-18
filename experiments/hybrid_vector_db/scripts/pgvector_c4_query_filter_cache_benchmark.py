from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from pgvector_hnsw_page_access_group_benchmark import load_query_vectors


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def sql_key(predicate: str) -> str:
    return "sql:" + predicate


def price_bucket(row: dict[str, str]) -> str:
    if row["has_price"] != "True":
        return "NOT has_price"
    price = float(row["price"] or 0)
    if price <= 10:
        return "has_price AND price <= 10"
    if price <= 20:
        return "has_price AND price > 10 AND price <= 20"
    if price <= 50:
        return "has_price AND price > 20 AND price <= 50"
    return "has_price AND price > 50"


def price_label(row: dict[str, str]) -> str:
    if row["has_price"] != "True":
        return "p_missing"
    price = float(row["price"] or 0)
    if price <= 10:
        return "p_le10"
    if price <= 20:
        return "p_10_20"
    if price <= 50:
        return "p_20_50"
    return "p_gt50"


def popularity_bucket(row: dict[str, str]) -> str:
    count = int(float(row["item_rating_number"] or 0))
    if count >= 1000:
        return "item_rating_number >= 1000"
    if count >= 100:
        return "item_rating_number >= 100 AND item_rating_number < 1000"
    return "item_rating_number < 100"


def popularity_label(row: dict[str, str]) -> str:
    count = int(float(row["item_rating_number"] or 0))
    if count >= 1000:
        return "pop_high"
    if count >= 100:
        return "pop_mid"
    return "pop_low"


def query_predicate(row: dict[str, str], mode: str) -> str:
    rating = int(float(row.get("ori_rating") or row.get("table_rating") or 5))
    rating = max(1, min(5, rating))
    if mode == "price":
        return price_bucket(row)
    if mode == "popularity":
        return popularity_bucket(row)
    if mode == "mixed":
        return f"rating = {rating} AND {price_bucket(row)} AND {popularity_bucket(row)}"
    if mode == "mixed_category":
        category = (row.get("main_category") or "Grocery & Gourmet Food").replace("'", "''")
        return f"main_category = '{category}' AND rating = {rating} AND {price_bucket(row)} AND {popularity_bucket(row)}"
    raise ValueError(f"unknown mode {mode}")


def query_cache_key(row: dict[str, str], mode: str, predicate: str) -> str:
    rating = int(float(row.get("ori_rating") or row.get("table_rating") or 5))
    rating = max(1, min(5, rating))
    if mode == "mixed" and rating == 5:
        return f"c4_mixed_5_{price_label(row)}_{popularity_label(row)}"
    return sql_key(predicate)


def load_workload(path: Path, limit: int, mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    **row,
                    "query_id_int": int(row["query_id"]),
                    "predicate": query_predicate(row, mode),
                }
            )
            rows[-1]["cache_key"] = query_cache_key(row, mode, str(rows[-1]["predicate"]))
            if len(rows) >= limit:
                break
    return rows


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


def run_cache(cur: psycopg.Cursor, table: str, index: str, cache_key: str, query_id: int, k: int, candidate_limit: int):
    try:
        cur.execute(
            f"""
            SELECT id
            FROM vector_hnsw_metadata_filter_search(
                %s::regclass,
                (SELECT embedding FROM {table} WHERE id = %s),
                %s,
                %s,
                %s
            )
            ORDER BY rank
            """,
            (index, int(query_id), int(k), int(candidate_limit), cache_key),
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


def run_candidate_cache(
    cur: psycopg.Cursor,
    table: str,
    index: str,
    predicate: str,
    cache_key: str,
    query_id: int,
    k: int,
    candidate_limit: int,
    cache_kind: str,
    match_limit: int,
):
    candidate_function = {
        "page": "vector_hnsw_metadata_page_filter_candidates",
        "bloom": "vector_hnsw_metadata_bloom_filter_candidates",
    }[cache_kind]
    if cache_kind == "bloom" and match_limit > 0:
        candidate_sql = f"""
                SELECT rank, ctid::tid AS candidate_ctid
                FROM vector_hnsw_metadata_bloom_filter_candidates_limited(
                    %s::regclass,
                    (SELECT embedding FROM {table} WHERE id = %s),
                    %s,
                    %s,
                    %s
                )
        """
        params = (index, int(query_id), int(candidate_limit), int(match_limit), cache_key)
    else:
        candidate_sql = f"""
                SELECT rank, ctid::tid AS candidate_ctid
                FROM {candidate_function}(
                    %s::regclass,
                    (SELECT embedding FROM {table} WHERE id = %s),
                    %s,
                    %s
                )
        """
        params = (index, int(query_id), int(candidate_limit), cache_key)
    try:
        cur.execute(
            f"""
            WITH candidates AS MATERIALIZED (
                {candidate_sql}
            )
            SELECT t.id
            FROM candidates c
            JOIN {table} t ON t.ctid = c.candidate_ctid
            WHERE {predicate}
            ORDER BY c.rank
            LIMIT {int(k)}
            """,
            params,
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


def summarize(
    rows: list[dict[str, object]],
    distinct_filters: int,
    cached_filters: int,
    cached_queries: int,
    build_ms: float,
    cache_rows_total: int,
) -> dict[str, object]:
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

    total_standard = sum(standard)
    total_cache_hot = sum(cache)
    total_cache_with_build = total_cache_hot + build_ms
    return {
        "rows": len(rows),
        "ok": len(ok),
        "distinct_filters": distinct_filters,
        "cached_filters": cached_filters,
        "cached_queries": cached_queries,
        "cache_rows_total": cache_rows_total,
        "cache_build_total_ms": build_ms,
        "same_ordered_ids": sum(1 for row in ok if row["same_ordered_ids"]),
        "same_set_ids": sum(1 for row in ok if row["same_set_ids"]),
        "standard_mean_ms": mean(standard),
        "standard_p50_ms": median(standard),
        "standard_p95_ms": p95(standard),
        "cache_hot_mean_ms": mean(cache),
        "cache_hot_p50_ms": median(cache),
        "cache_hot_p95_ms": p95(cache),
        "mean_hot_speedup": mean(speedups),
        "median_hot_speedup": median(speedups),
        "p05_hot_speedup": sorted(speedups)[int(0.05 * (len(speedups) - 1))] if speedups else 0.0,
        "p95_hot_speedup": p95(speedups),
        "total_standard_ms": total_standard,
        "total_cache_hot_ms": total_cache_hot,
        "total_cache_with_build_ms": total_cache_with_build,
        "hot_total_speedup": total_standard / total_cache_hot if total_cache_hot > 0 else 0.0,
        "amortized_total_speedup": total_standard / total_cache_with_build if total_cache_with_build > 0 else 0.0,
        "mean_standard_returned_tids": mean([float(row["standard_returned_tuples"]) for row in ok]),
        "mean_cache_candidates": mean([float(row["cache_candidates"]) for row in ok]),
        "mean_cache_matches": mean([float(row["cache_matches"]) for row in ok]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Amazon-C4 query-derived filter workload with pgvector metadata cache.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200000)
    parser.add_argument("--match-limit", type=int, default=0)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    parser.add_argument("--max-cache-rows", type=int, default=200000)
    parser.add_argument("--cache-kind", default="exact", choices=["exact", "page", "bloom"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    workload = load_workload(args.query_csv, args.queries, args.mode)
    query_ids = [int(row["query_id_int"]) for row in workload]
    predicate_counts = collections.Counter(str(row["predicate"]) for row in workload)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    cache_rows_total = 0
    build_total_ms = 0.0
    filter_rows: list[dict[str, object]] = []

    with (
        psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as standard_conn,
        psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as cache_conn,
    ):
        standard_cur = standard_conn.cursor()
        cache_cur = cache_conn.cursor()
        configure(standard_cur, args)
        configure(cache_cur, args)
        vectors = load_query_vectors(standard_cur, args.table, query_ids)

        cache_key_by_predicate = {str(row["predicate"]): str(row["cache_key"]) for row in workload}
        predicate_rows_by_predicate: dict[str, int] = {}
        cacheable_predicates: set[str] = set()
        cached_predicates: set[str] = set()
        for predicate, frequency in predicate_counts.items():
            standard_cur.execute(f"SELECT count(*) FROM {args.table} WHERE {predicate}")
            predicate_rows = int(standard_cur.fetchone()[0])
            predicate_rows_by_predicate[predicate] = predicate_rows
            if predicate_rows > args.max_cache_rows:
                filter_rows.append(
                    {
                        "predicate": predicate,
                        "frequency": frequency,
                        "predicate_rows": predicate_rows,
                        "cached": False,
                        "cache_rows": 0,
                        "build_wall_ms": 0.0,
                        "build_profile_ms": 0.0,
                    }
                )
                continue

            cacheable_predicates.add(predicate)
            filter_rows.append(
                {
                    "predicate": predicate,
                    "frequency": frequency,
                    "predicate_rows": predicate_rows,
                    "cached": True,
                    "cache_rows": "",
                    "build_wall_ms": "",
                    "build_profile_ms": "",
                }
            )

        fieldnames = [
            "query_no",
            "query_id",
            "c4_query_no",
            "predicate",
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
            "cache_candidates",
            "cache_checks",
            "cache_matches",
            "cache_pages",
            "cache_kind",
            "cache_search_ms",
            "used_cache",
        ]
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for query_no, row in enumerate(workload):
                query_id = int(row["query_id_int"])
                predicate = str(row["predicate"])
                query_vector = vectors[query_id]
                configure(standard_cur, args)
                (standard_ids, standard_profile, standard_error), standard_ms = timed_ms(
                    lambda: run_standard(standard_cur, args.table, predicate, query_vector, args.k)
                )
                if predicate in cacheable_predicates:
                    if predicate not in cached_predicates:
                        configure(cache_cur, args)
                        build_function = "vector_hnsw_metadata_cache_build"
                        if args.cache_kind == "page":
                            build_function = "vector_hnsw_metadata_page_cache_build"
                        elif args.cache_kind == "bloom":
                            build_function = "vector_hnsw_metadata_bloom_cache_build"
                        _, build_ms = timed_ms(
                            lambda predicate=predicate, build_function=build_function: cache_cur.execute(
                                f"SELECT {build_function}(%s::regclass, %s)",
                                (args.index, cache_key_by_predicate[predicate]),
                            ).fetchone()
                        )
                        cache_cur.execute("SELECT vector_hnsw_metadata_filter_profile()")
                        build_profile = json.loads(cache_cur.fetchone()[0])
                        cache_rows_total += int(build_profile.get("cache_rows", 0))
                        build_total_ms += build_ms
                        cached_predicates.add(predicate)
                        for filter_row in filter_rows:
                            if filter_row["predicate"] == predicate:
                                filter_row["cache_rows"] = int(build_profile.get("cache_rows", 0))
                                filter_row["build_wall_ms"] = build_ms
                                filter_row["build_profile_ms"] = build_profile.get("cache_build_ms", 0.0)
                                break
                    configure(cache_cur, args)
                    if args.cache_kind in {"page", "bloom"}:
                        (cache_ids, cache_profile, cache_error), cache_ms = timed_ms(
                            lambda: run_candidate_cache(
                                cache_cur,
                                args.table,
                                args.index,
                                predicate,
                                str(row["cache_key"]),
                                query_id,
                                args.k,
                                args.candidate_limit,
                                args.cache_kind,
                                args.match_limit,
                            )
                        )
                    else:
                        (cache_ids, cache_profile, cache_error), cache_ms = timed_ms(
                            lambda: run_cache(cache_cur, args.table, args.index, str(row["cache_key"]), query_id, args.k, args.candidate_limit)
                        )
                    used_cache = True
                else:
                    cache_ids = standard_ids
                    cache_profile = {}
                    cache_error = standard_error
                    cache_ms = standard_ms
                    used_cache = False
                out_row = {
                    "query_no": query_no,
                    "query_id": query_id,
                    "c4_query_no": row["query_no"],
                    "predicate": predicate,
                    "standard_ms": standard_ms,
                    "cache_ms": cache_ms,
                    "speedup": standard_ms / cache_ms if cache_ms > 0 else 0.0,
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
                    "cache_candidates": cache_profile.get("candidates", 0),
                    "cache_checks": cache_profile.get("cache_checks", 0),
                    "cache_matches": cache_profile.get("cache_matches", 0),
                    "cache_pages": cache_profile.get("cache_pages", 0),
                    "cache_kind": cache_profile.get("cache_kind", ""),
                    "cache_search_ms": cache_profile.get("search_ms", 0.0),
                    "used_cache": used_cache,
                }
                rows.append(out_row)
                writer.writerow(out_row)
                f.flush()
                if args.stream:
                    print(
                        f"q={query_no} predfreq={predicate_counts[predicate]} "
                        f"std={standard_ms:.2f} cache={cache_ms:.2f} "
                        f"speedup={out_row['speedup']:.2f} same={out_row['same_ordered_ids']}",
                        flush=True,
                    )

    filters_out = args.filters_out or args.out.with_name(args.out.stem + "_filters.csv")
    with filters_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["predicate", "frequency", "predicate_rows", "cached", "cache_rows", "build_wall_ms", "build_profile_ms"])
        writer.writeheader()
        writer.writerows(sorted(filter_rows, key=lambda r: (-int(r["frequency"]), str(r["predicate"]))))

    summary = summarize(
        rows,
        len(predicate_counts),
        len(cached_predicates),
        sum(1 for row in workload if str(row["predicate"]) in cached_predicates),
        build_total_ms,
        cache_rows_total,
    )
    summary.update({"mode": args.mode, "cache_kind": args.cache_kind, "table": args.table, "index": args.index})
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {filters_out}")
    print(f"wrote {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
