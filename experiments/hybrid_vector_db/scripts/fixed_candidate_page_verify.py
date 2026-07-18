from __future__ import annotations

import argparse
import csv
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from common_pg import pg_config_from_env, require_psycopg
from pgvector_prefilter_10m import TABLE


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000


def read_fbin_memmap(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr, n, d


def load_queries(path: Path, queries: int, query_offset: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["query_no"]), int(row["query_id"])))
    rows = sorted(rows)
    return rows[query_offset : query_offset + queries]


def faiss_search(index, query: np.ndarray, topk: int, ef_search: int) -> list[int]:
    import faiss

    params = faiss.SearchParametersHNSW()
    params.efSearch = int(ef_search)
    _, ids = index.search(query.reshape(1, -1), int(topk), params=params)
    return [int(x) for x in ids[0] if x >= 0]


def create_temp_tables(cur) -> None:
    cur.execute("DROP TABLE IF EXISTS fixed_candidate_ids")
    cur.execute(
        """
        CREATE TEMP TABLE fixed_candidate_ids (
            ord integer NOT NULL,
            id bigint NOT NULL
        ) ON COMMIT PRESERVE ROWS
        """
    )
    cur.execute("DROP TABLE IF EXISTS fixed_candidate_pages")
    cur.execute(
        """
        CREATE TEMP TABLE fixed_candidate_pages (
            ord integer NOT NULL,
            id bigint NOT NULL,
            heap_tid tid NOT NULL,
            heap_block bigint NOT NULL,
            heap_off integer NOT NULL
        ) ON COMMIT PRESERVE ROWS
        """
    )


def load_candidates(cur, table: str, ids: list[int]) -> tuple[int, float]:
    def run() -> int:
        cur.execute("TRUNCATE fixed_candidate_ids")
        cur.execute("TRUNCATE fixed_candidate_pages")
        with cur.copy("COPY fixed_candidate_ids (ord, id) FROM STDIN") as copy:
            for ord_no, row_id in enumerate(ids):
                copy.write(f"{ord_no}\t{row_id}\n".encode("utf-8"))
        cur.execute(
            f"""
            INSERT INTO fixed_candidate_pages (ord, id, heap_tid, heap_block, heap_off)
            SELECT
                c.ord,
                t.id,
                t.ctid AS heap_tid,
                split_part(split_part(t.ctid::text, ',', 1), '(', 2)::bigint AS heap_block,
                rtrim(split_part(t.ctid::text, ',', 2), ')')::integer AS heap_off
            FROM fixed_candidate_ids c
            JOIN {table} t ON t.id = c.id
            ORDER BY c.ord
            """
        )
        return int(cur.rowcount)

    return timed(run)


def candidate_locality(cur) -> dict[str, float]:
    cur.execute(
        """
        WITH ordered AS (
            SELECT ord, heap_block, lag(heap_block) OVER (ORDER BY ord) AS prev_block
            FROM fixed_candidate_pages
        )
        SELECT
            count(*)::float8,
            count(DISTINCT heap_block)::float8,
            coalesce(sum(CASE WHEN prev_block IS NULL OR heap_block <> prev_block THEN 1 ELSE 0 END), 0)::float8
        FROM ordered
        """
    )
    candidates, distinct_pages, page_runs = cur.fetchone()
    return {
        "candidates": float(candidates),
        "distinct_heap_pages": float(distinct_pages),
        "distance_order_page_runs": float(page_runs),
        "page_run_reduction": float(page_runs) / max(float(distinct_pages), 1.0),
    }


def verify_sql(table: str, predicate: str, order: str) -> str:
    if order == "distance":
        order_clause = "ord"
    elif order == "page":
        order_clause = "heap_block, heap_off"
    else:
        raise ValueError(order)
    return f"""
        WITH ordered AS MATERIALIZED (
            SELECT *
            FROM fixed_candidate_pages
            ORDER BY {order_clause}
        ),
        verified AS MATERIALIZED (
            SELECT c.id, c.ord
            FROM ordered c
            JOIN LATERAL (
                SELECT 1
                FROM {table} t
                WHERE t.ctid = c.heap_tid AND {predicate}
            ) ok ON true
        )
        SELECT id
        FROM verified
        ORDER BY ord
    """


def run_verify(cur, table: str, predicate: str, order: str, k: int, repeats: int) -> tuple[list[int], float]:
    sql = f"SELECT id FROM ({verify_sql(table, predicate, order)}) v LIMIT {int(k)}"
    ids: list[int] = []
    latencies: list[float] = []
    for _ in range(repeats):
        result, elapsed = timed(lambda: fetch_ids(cur, sql))
        ids = result
        latencies.append(elapsed)
    return ids, statistics.mean(latencies)


def fetch_ids(cur, sql: str) -> list[int]:
    cur.execute(sql)
    return [int(row[0]) for row in cur.fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Page-aware verification with fixed candidate IDs.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-id-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fbin",
        type=Path,
        default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"),
    )
    parser.add_argument("--faiss-index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--faiss-ef-search", type=int, default=1000)
    parser.add_argument("--predicate", default="has_price AND price > 10 AND price <= 20")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--locality-only", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    require_psycopg()
    import faiss
    import psycopg

    queries = load_queries(args.query_id_csv, args.queries, args.query_offset)
    xb, _, _ = read_fbin_memmap(args.fbin)
    index = faiss.read_index(str(args.faiss_index))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, object]] = []
    stream_file = args.out.open("w", newline="") if args.stream else None
    writer: csv.DictWriter | None = None
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            create_temp_tables(cur)
            for qno, qid in queries:
                query = np.asarray(xb[qid], dtype=np.float32)
                ids, faiss_ms = timed(lambda: faiss_search(index, query, args.candidate_limit, args.faiss_ef_search))
                candidate_count, load_ms = load_candidates(cur, args.table, ids)
                locality = candidate_locality(cur)
                if args.locality_only:
                    distance_ids: list[int] = []
                    page_ids: list[int] = []
                    distance_ms = 0.0
                    page_ms = 0.0
                else:
                    distance_ids, distance_ms = run_verify(cur, args.table, args.predicate, "distance", args.k, args.repeats)
                    page_ids, page_ms = run_verify(cur, args.table, args.predicate, "page", args.k, args.repeats)
                row = {
                    "table": args.table,
                    "query_no": qno,
                    "query_id": qid,
                    "candidate_limit": args.candidate_limit,
                    "candidate_count": candidate_count,
                    "faiss_ms": faiss_ms,
                    "load_candidate_pages_ms": load_ms,
                    "same_results": "" if args.locality_only else distance_ids == page_ids,
                    "distance_ids": ",".join(str(x) for x in distance_ids),
                    "page_ids": ",".join(str(x) for x in page_ids),
                    "distance_verify_ms": distance_ms,
                    "page_verify_ms": page_ms,
                    **locality,
                }
                rows_out.append(row)
                if stream_file is not None:
                    if writer is None:
                        writer = csv.DictWriter(stream_file, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    stream_file.flush()
                print(
                    f"table={args.table} q={qno} candidates={candidate_count} "
                    f"runs={row['distance_order_page_runs']:.0f}->{row['distinct_heap_pages']:.0f} "
                    f"distance_ms={distance_ms:.2f} page_ms={page_ms:.2f}",
                    flush=True,
                )
    if stream_file is not None:
        stream_file.close()
    if not rows_out:
        raise RuntimeError("no rows produced")
    if not args.stream:
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    summary = args.out.with_name(args.out.stem + "_summary.csv")
    distance_mean = statistics.mean(float(r["distance_verify_ms"]) for r in rows_out)
    page_mean = statistics.mean(float(r["page_verify_ms"]) for r in rows_out)
    summary_row = {
        "table": args.table,
        "queries": len(rows_out),
        "candidate_limit": args.candidate_limit,
        "candidate_count_mean": statistics.mean(float(r["candidate_count"]) for r in rows_out),
        "faiss_ms_mean": statistics.mean(float(r["faiss_ms"]) for r in rows_out),
        "load_candidate_pages_ms_mean": statistics.mean(float(r["load_candidate_pages_ms"]) for r in rows_out),
        "distinct_heap_pages_mean": statistics.mean(float(r["distinct_heap_pages"]) for r in rows_out),
        "distance_order_page_runs_mean": statistics.mean(float(r["distance_order_page_runs"]) for r in rows_out),
        "page_run_reduction": statistics.mean(float(r["distance_order_page_runs"]) for r in rows_out)
        / max(statistics.mean(float(r["distinct_heap_pages"]) for r in rows_out), 1.0),
        "distance_verify_ms_mean": distance_mean,
        "page_verify_ms_mean": page_mean,
        "verify_speedup": "" if args.locality_only else distance_mean / max(page_mean, 1e-9),
        "same_results": "" if args.locality_only else all(str(r["same_results"]) == "True" for r in rows_out),
    }
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
