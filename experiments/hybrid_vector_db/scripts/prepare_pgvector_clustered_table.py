from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np

from common_pg import pg_config_from_env, require_psycopg
from pgvector_prefilter_10m import TABLE


def load_query_id(truth_csv: Path, query_no: int, truth_method: str) -> int:
    with truth_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == truth_method and int(row["query_no"]) == query_no:
                return int(row["query_id"])
    raise RuntimeError(f"query_no={query_no} method={truth_method} not found in {truth_csv}")


def load_query_id_csv(path: Path, query_no: int) -> int:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["query_no"]) == query_no:
                return int(row["query_id"])
    raise RuntimeError(f"query_no={query_no} not found in {path}")


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def exact_top_ids(path: Path, query_id: int, rows: int, source_rows: int | None, chunk_size: int) -> list[int]:
    xb, n, _ = read_fbin_memmap(path, source_rows)
    if query_id >= n:
        raise RuntimeError(f"query_id={query_id} outside fbin rows={n}")
    query = np.asarray(xb[query_id], dtype=np.float32)
    query_norm = float(np.dot(query, query))
    best_dist = np.empty(0, dtype=np.float32)
    best_ids = np.empty(0, dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        vecs = np.asarray(xb[start:stop], dtype=np.float32)
        ids = np.arange(start, stop, dtype=np.int64)
        dists = np.einsum("ij,ij->i", vecs, vecs) + query_norm - 2.0 * (vecs @ query)
        if best_ids.size:
            ids = np.concatenate([best_ids, ids])
            dists = np.concatenate([best_dist, dists])
        take = min(rows, len(dists))
        pos = np.argpartition(dists, take - 1)[:take]
        order = np.argsort(dists[pos])
        best_dist = dists[pos][order]
        best_ids = ids[pos][order]
        if start and start % (chunk_size * 10) == 0:
            print(f"exact_scan rows={stop}/{n}", flush=True)
    return [int(x) for x in best_ids[:rows]]


def create_table_from_ordered_ids(cur, source_table: str, target_table: str, ids: list[int]) -> None:
    cur.execute("DROP TABLE IF EXISTS pgvector_cluster_ids")
    cur.execute("CREATE TEMP TABLE pgvector_cluster_ids (ord integer NOT NULL, id bigint NOT NULL) ON COMMIT PRESERVE ROWS")
    with cur.copy("COPY pgvector_cluster_ids (ord, id) FROM STDIN") as copy:
        for ord_no, row_id in enumerate(ids):
            copy.write(f"{ord_no}\t{row_id}\n".encode("utf-8"))
    cur.execute(
        f"""
        CREATE TABLE {target_table} AS
        SELECT s.*
        FROM pgvector_cluster_ids c
        JOIN {source_table} s ON s.id = c.id
        ORDER BY c.ord
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a pgvector table physically ordered by distance to an anchor query vector."
    )
    parser.add_argument("--source-table", default=TABLE)
    parser.add_argument("--target-table", default=f"{TABLE}_anchor_clustered")
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"),
    )
    parser.add_argument("--truth-method", default="pre_filter_exact")
    parser.add_argument("--query-id-csv", type=Path)
    parser.add_argument("--query-no", type=int, default=0)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument(
        "--exact-fbin",
        type=Path,
        help="Use exact fbin scan to choose top-N ids before materializing the clustered table",
    )
    parser.add_argument("--source-rows", type=int, default=10_000_000)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--max-scan-tuples", type=int, default=1_000_000)
    parser.add_argument("--maintenance-work-mem", default="4GB")
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    if args.query_id_csv is not None:
        query_id = load_query_id_csv(args.query_id_csv, args.query_no)
    else:
        query_id = load_query_id(args.truth_csv, args.query_no, args.truth_method)
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT embedding FROM {args.source_table} WHERE id = %s", (query_id,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"query id {query_id} not found in {args.source_table}")
            query = row[0]

            if args.drop:
                cur.execute(f"DROP TABLE IF EXISTS {args.target_table}")

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
            cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
            cur.execute("SET hnsw.iterative_scan = 'relaxed_order'")
            cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
            cur.execute("SET enable_seqscan = off")
            if args.exact_fbin is not None:
                if args.drop:
                    # DROP already happened above; keep the branch explicit for readability.
                    pass
                cur.execute(f"SELECT to_regclass(%s)", (args.target_table,))
                if cur.fetchone()[0] is None:
                    ids = exact_top_ids(args.exact_fbin, query_id, args.rows, args.source_rows, args.chunk_size)
                    create_table_from_ordered_ids(cur, args.source_table, args.target_table, ids)
            else:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {args.target_table} AS
                    SELECT *
                    FROM {args.source_table}
                    ORDER BY embedding <-> %s::vector
                    LIMIT {int(args.rows)}
                    """,
                    (query,),
                )
            cur.execute(f"ALTER TABLE {args.target_table} ADD PRIMARY KEY (id)")
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {args.target_table}_embedding_hnsw_idx
                ON {args.target_table}
                USING hnsw (embedding vector_l2_ops)
                WITH (m = {int(args.hnsw_m)}, ef_construction = {int(args.ef_construction)})
                """
            )
            cur.execute(f"ANALYZE {args.target_table}")
            cur.execute(f"SELECT count(*) FROM {args.target_table}")
            count = int(cur.fetchone()[0])
            print(
                f"created table={args.target_table} rows={count} anchor_query_no={args.query_no} "
                f"anchor_query_id={query_id}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
