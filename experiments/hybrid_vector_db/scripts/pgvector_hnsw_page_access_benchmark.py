from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import psycopg

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
    rows = cur.fetchall()
    vectors = {int(row[0]): str(row[1]) for row in rows}
    missing = [query_id for query_id in query_ids if query_id not in vectors]
    if missing:
        raise RuntimeError(f"missing {len(missing)} query vectors from {table}")
    return vectors


def run_query(
    cur: psycopg.Cursor,
    table: str,
    query_vector: str,
    mode: str,
    ef_search: int,
    iterative_scan: str,
    max_scan_tuples: int,
    page_window: int,
    k: int,
) -> tuple[list[int], float, dict[str, object]]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(max_scan_tuples)}")
    cur.execute(f"SET hnsw.page_window = {int(page_window)}")
    cur.execute(f"SET hnsw.page_access = {mode}")

    start = time.perf_counter()
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
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile = json.loads(cur.fetchone()[0])
    return ids, elapsed_ms, profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark C-level pgvector HNSW page-aware access modes.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-id-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="relaxed_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--page-window", type=int, default=1000)
    parser.add_argument("--modes", nargs="+", default=["off", "prefetch", "reorder"], choices=["off", "prefetch", "reorder"])
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_hnsw_page_access_benchmark.csv"))
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    query_ids = load_query_ids(args.query_id_csv, args.queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SELECT vector_dims('[1,2]'::vector)")
        vectors = load_query_vectors(cur, args.table, query_ids)

        for query_no, query_id in enumerate(query_ids):
            baseline_ids: list[int] | None = None
            for mode in args.modes:
                ids, elapsed_ms, profile = run_query(
                    cur,
                    args.table,
                    vectors[query_id],
                    mode,
                    args.ef_search,
                    args.iterative_scan,
                    args.max_scan_tuples,
                    args.page_window,
                    args.k,
                )
                if baseline_ids is None:
                    baseline_ids = ids

                row = {
                    "table": args.table,
                    "query_no": query_no,
                    "query_id": query_id,
                    "mode": mode,
                    "k": args.k,
                    "ef_search": args.ef_search,
                    "page_window": args.page_window,
                    "elapsed_ms": elapsed_ms,
                    "same_as_first_mode": ids == baseline_ids,
                    "returned": len(ids),
                    "ids": ",".join(str(x) for x in ids),
                    **profile,
                }
                rows.append(row)

                if args.stream:
                    print(
                        f"q={query_no} id={query_id} mode={mode} "
                        f"ms={elapsed_ms:.2f} same={row['same_as_first_mode']} "
                        f"runs={profile.get('page_access_distance_runs')} "
                        f"pages={profile.get('page_access_distinct_pages')}",
                        flush=True,
                    )

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
