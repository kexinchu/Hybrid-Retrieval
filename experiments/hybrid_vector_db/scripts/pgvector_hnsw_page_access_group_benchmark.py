from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from pgvector_prefilter_10m import TABLE


def load_query_ids(path: Path, limit: int) -> list[int]:
    ids: list[int] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["query_id"]))
            if len(ids) >= limit:
                break
    return ids


def load_query_vectors(cur: psycopg.Cursor, table: str, query_ids: list[int]) -> dict[int, str]:
    cur.execute(
        f"""
        SELECT id, embedding::text
        FROM {table}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    vectors = {int(row[0]): str(row[1]) for row in cur.fetchall()}
    missing = [query_id for query_id in query_ids if query_id not in vectors]
    if missing:
        raise RuntimeError(f"missing {len(missing)} query vectors from {table}")
    return vectors


def run_query(
    cur: psycopg.Cursor,
    table: str,
    query_vector: str,
    mode: str,
    index_page_access: str,
    ef_search: int,
    iterative_scan: str,
    max_scan_tuples: int,
    page_window: int,
    k: int,
    statement_timeout_ms: int,
) -> tuple[list[int], float, dict[str, object], str | None]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(max_scan_tuples)}")
    cur.execute(f"SET hnsw.page_window = {int(page_window)}")
    cur.execute(f"SET hnsw.page_access = {mode}")
    cur.execute(f"SET hnsw.index_page_access = {index_page_access}")

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

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    profile: dict[str, object]
    try:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        profile = json.loads(cur.fetchone()[0])
    except Exception:
        profile = {}

    return ids, elapsed_ms, profile, error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one C-level pgvector HNSW page-access mode over a query group.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-id-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--iterative-scan", default="relaxed_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=50000)
    parser.add_argument("--page-window", type=int, default=128)
    parser.add_argument("--mode", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    query_ids = load_query_ids(args.query_id_csv, args.queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "table",
        "query_no",
        "query_id",
        "mode",
        "index_page_access",
        "k",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "page_window",
        "statement_timeout_ms",
        "elapsed_ms",
        "error",
        "returned",
        "ids",
        "valid",
        "vector_search_ms",
        "visited_tuples",
        "returned_tuples",
        "page_access_batches",
        "page_access_candidates",
        "page_access_prefetches",
        "page_access_distance_runs",
        "page_access_distinct_pages",
        "guidance_checks",
        "guidance_matches",
        "guidance_skips",
        "index_page_neighbor_loads",
        "index_page_neighbor_runs",
        "index_page_neighbor_distinct_pages",
        "index_page_element_loads",
        "index_page_element_runs",
        "index_page_element_distinct_pages",
        "index_page_prefetches",
    ]

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SELECT vector_dims('[1,2]'::vector)")
        vectors = load_query_vectors(cur, args.table, query_ids)

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for query_no, query_id in enumerate(query_ids):
                ids, elapsed_ms, profile, error = run_query(
                    cur,
                    args.table,
                    vectors[query_id],
                    args.mode,
                    args.index_page_access,
                    args.ef_search,
                    args.iterative_scan,
                    args.max_scan_tuples,
                    args.page_window,
                    args.k,
                    args.statement_timeout_ms,
                )

                row = {
                    "table": args.table,
                    "query_no": query_no,
                    "query_id": query_id,
                    "mode": args.mode,
                    "index_page_access": args.index_page_access,
                    "k": args.k,
                    "ef_search": args.ef_search,
                    "iterative_scan": args.iterative_scan,
                    "max_scan_tuples": args.max_scan_tuples,
                    "page_window": args.page_window,
                    "statement_timeout_ms": args.statement_timeout_ms,
                    "elapsed_ms": elapsed_ms,
                    "error": error or "",
                    "returned": len(ids),
                    "ids": ",".join(str(x) for x in ids),
                    **profile,
                }
                writer.writerow(row)
                f.flush()

                if args.stream:
                    print(
                        f"q={query_no} id={query_id} mode={args.mode} "
                        f"idxpage={args.index_page_access} "
                        f"ms={elapsed_ms:.2f} err={error or ''} "
                        f"ret={len(ids)} runs={profile.get('page_access_distance_runs')} "
                        f"pages={profile.get('page_access_distinct_pages')}",
                        flush=True,
                    )

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
