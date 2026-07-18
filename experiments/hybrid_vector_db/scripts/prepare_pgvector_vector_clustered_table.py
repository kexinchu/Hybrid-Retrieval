from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

from common_pg import pg_config_from_env, require_psycopg
from pgvector_prefilter_10m import TABLE


def timed(label: str, fn):
    t0 = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - t0
    print(f"{label} elapsed_s={elapsed:.2f}", flush=True)
    return value


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def train_and_assign(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import faiss

    xb, rows, dim = read_fbin_memmap(args.fbin, args.rows)
    rng = np.random.default_rng(args.seed)
    sample_size = min(args.train_sample, rows)
    sample_ids = rng.choice(rows, size=sample_size, replace=False)
    train = np.ascontiguousarray(xb[sample_ids], dtype=np.float32)

    kmeans = faiss.Kmeans(
        dim,
        args.clusters,
        niter=args.kmeans_iter,
        nredo=1,
        verbose=True,
        seed=args.seed,
        spherical=False,
        gpu=False,
    )
    timed(f"train kmeans rows={sample_size} clusters={args.clusters}", lambda: kmeans.train(train))

    index = faiss.IndexFlatL2(dim)
    index.add(kmeans.centroids)
    cluster_ids = np.empty(rows, dtype=np.int32)
    distances = np.empty(rows, dtype=np.float32)

    def assign_all() -> None:
        for start in range(0, rows, args.chunk_size):
            stop = min(start + args.chunk_size, rows)
            x = np.ascontiguousarray(xb[start:stop], dtype=np.float32)
            d, c = index.search(x, 1)
            cluster_ids[start:stop] = c[:, 0].astype(np.int32)
            distances[start:stop] = d[:, 0].astype(np.float32)
            print(f"assigned rows={stop}/{rows}", flush=True)

    timed("assign clusters", assign_all)
    ids = np.arange(rows, dtype=np.int64)
    if args.order_within_cluster == "distance":
        order = np.lexsort((distances, cluster_ids))
    else:
        order = np.lexsort((ids, cluster_ids))
    return ids[order], cluster_ids[order], distances[order]


def create_id_order_table(
    cur, source_table: str, target_table: str, rows: int, hnsw_m: int, ef_construction: int, skip_hnsw: bool
) -> None:
    cur.execute(f"DROP TABLE IF EXISTS {target_table}")
    cur.execute(
        f"""
        CREATE TABLE {target_table} AS
        SELECT *, NULL::integer AS vector_cluster_id
        FROM {source_table}
        WHERE id < {int(rows)}
        ORDER BY id
        """
    )
    cur.execute(f"ALTER TABLE {target_table} ADD PRIMARY KEY (id)")
    if not skip_hnsw:
        create_hnsw(cur, target_table, hnsw_m, ef_construction)
    else:
        cur.execute(f"ANALYZE {target_table}")


def create_vector_clustered_table(
    cur,
    source_table: str,
    target_table: str,
    ordered_ids: np.ndarray,
    ordered_clusters: np.ndarray,
    hnsw_m: int,
    ef_construction: int,
    skip_hnsw: bool,
) -> None:
    cur.execute(f"DROP TABLE IF EXISTS {target_table}")
    cur.execute("DROP TABLE IF EXISTS pgvector_cluster_order")
    cur.execute(
        """
        CREATE TEMP TABLE pgvector_cluster_order (
            ord integer NOT NULL,
            id bigint NOT NULL,
            vector_cluster_id integer NOT NULL
        ) ON COMMIT PRESERVE ROWS
        """
    )
    with cur.copy("COPY pgvector_cluster_order (ord, id, vector_cluster_id) FROM STDIN") as copy:
        for ord_no, (row_id, cluster_id) in enumerate(zip(ordered_ids, ordered_clusters, strict=True)):
            copy.write(f"{ord_no}\t{int(row_id)}\t{int(cluster_id)}\n".encode("utf-8"))
    cur.execute("ANALYZE pgvector_cluster_order")

    cur.execute(
        f"""
        CREATE TABLE {target_table} AS
        SELECT s.*, c.vector_cluster_id
        FROM pgvector_cluster_order c
        JOIN {source_table} s ON s.id = c.id
        ORDER BY c.ord
        """
    )
    cur.execute(f"ALTER TABLE {target_table} ADD PRIMARY KEY (id)")
    if not skip_hnsw:
        create_hnsw(cur, target_table, hnsw_m, ef_construction)
    else:
        cur.execute(f"ANALYZE {target_table}")


def create_hnsw(cur, table: str, hnsw_m: int, ef_construction: int) -> None:
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx
        ON {table}
        USING hnsw (embedding vector_l2_ops)
        WITH (m = {int(hnsw_m)}, ef_construction = {int(ef_construction)})
        """
    )
    cur.execute(f"ANALYZE {table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create id-order and vector-cluster-order pgvector tables.")
    parser.add_argument("--source-table", default=TABLE)
    parser.add_argument("--id-order-table", default=f"{TABLE}_id_order_200k")
    parser.add_argument("--clustered-table", default=f"{TABLE}_vector_clustered_200k")
    parser.add_argument(
        "--fbin",
        type=Path,
        default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"),
    )
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--train-sample", type=int, default=50_000)
    parser.add_argument("--kmeans-iter", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--order-within-cluster", choices=["id", "distance"], default="distance")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--maintenance-work-mem", default="2GB")
    parser.add_argument("--work-mem", default="2GB")
    parser.add_argument("--skip-id-order", action="store_true")
    parser.add_argument("--skip-hnsw", action="store_true")
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    ordered_ids, ordered_clusters, _ = train_and_assign(args)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
            cur.execute(f"SET work_mem = '{args.work_mem}'")
            if not args.skip_id_order:
                timed(
                    f"create id-order table={args.id_order_table}",
                    lambda: create_id_order_table(
                        cur,
                        args.source_table,
                        args.id_order_table,
                        args.rows,
                        args.hnsw_m,
                        args.ef_construction,
                        args.skip_hnsw,
                    ),
                )
            timed(
                f"create vector-clustered table={args.clustered_table}",
                lambda: create_vector_clustered_table(
                    cur,
                    args.source_table,
                    args.clustered_table,
                    ordered_ids,
                    ordered_clusters,
                    args.hnsw_m,
                    args.ef_construction,
                    args.skip_hnsw,
                ),
            )
            tables = []
            if not args.skip_id_order:
                tables.append(args.id_order_table)
            tables.append(args.clustered_table)
            for table in tables:
                cur.execute(f"SELECT count(*) FROM {table}")
                print(f"table={table} rows={int(cur.fetchone()[0])}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
