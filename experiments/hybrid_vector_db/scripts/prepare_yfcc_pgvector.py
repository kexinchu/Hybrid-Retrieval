from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import psycopg

from common_pg import pg_config_from_env


BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M"
FILES = {
    "base": "base.10M.u8bin",
    "query": "query.public.100K.u8bin",
    "base_metadata": "base.metadata.10M.spmat",
    "query_metadata": "query.metadata.public.100K.spmat",
    "gt": "GT.public.ibin",
}
TABLE = "yfcc10m_pgvector"
QUERY_TABLE = "yfcc10m_queries"
INDEX = f"{TABLE}_embedding_hnsw"
DIM = 192


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"exists {dst} ({dst.stat().st_size} bytes)", flush=True)
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    done = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    print(f"download {url} -> {dst}", flush=True)
    req = __import__("urllib.request").request.Request(url, headers=headers)
    with urlopen(req) as response, tmp.open("ab" if done else "wb") as out:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) + done if total_header else 0
        copied = done
        last = time.perf_counter()
        while True:
            chunk = response.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            now = time.perf_counter()
            if now - last >= 5:
                if total:
                    print(f"  {copied / 2**20:.1f}/{total / 2**20:.1f} MiB", flush=True)
                else:
                    print(f"  {copied / 2**20:.1f} MiB", flush=True)
                last = now
    tmp.rename(dst)


def xbin_mmap(path: Path, dtype: str) -> np.memmap:
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    expected = 8 + int(n) * int(d) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path} size mismatch: got {path.stat().st_size}, expected {expected}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(int(n), int(d)))


def ibin_ids_mmap(path: Path) -> np.memmap:
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    ids_only_size = 8 + int(n) * int(d) * np.dtype("int32").itemsize
    ids_and_dist_size = 8 + int(n) * int(d) * (np.dtype("int32").itemsize + np.dtype("float32").itemsize)
    actual = path.stat().st_size
    if actual not in {ids_only_size, ids_and_dist_size}:
        raise ValueError(f"{path} size mismatch: got {actual}, expected {ids_only_size} or {ids_and_dist_size}")
    return np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(int(n), int(d)))


def spmat_fields(path: Path):
    with path.open("rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
    offset = 3 * np.dtype("int64").itemsize
    indptr = np.memmap(path, dtype=np.int64, mode="r", offset=offset, shape=int(nrow) + 1)
    offset += indptr.nbytes
    indices = np.memmap(path, dtype=np.int32, mode="r", offset=offset, shape=int(nnz))
    offset += indices.nbytes
    data = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=int(nnz))
    return int(nrow), int(ncol), int(nnz), indptr, indices, data


def vector_literal(row: np.ndarray) -> str:
    return "[" + ",".join(str(int(x)) for x in row) + "]"


def int_array_literal(values: np.ndarray | list[int]) -> str:
    if len(values) == 0:
        return "{}"
    return "{" + ",".join(str(int(x)) for x in values) + "}"


def ensure_schema(cur: psycopg.Cursor, table: str, query_table: str, drop: bool) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if drop:
        cur.execute(f"DROP TABLE IF EXISTS {table}_guidance_meta")
        cur.execute(f"DROP TABLE IF EXISTS {query_table}")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
          id bigint PRIMARY KEY,
          embedding vector({DIM}) NOT NULL,
          tags int[] NOT NULL,
          tag_count int NOT NULL
        )
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {query_table} (
          qid int PRIMARY KEY,
          embedding vector({DIM}) NOT NULL,
          tags int[] NOT NULL,
          tag_count int NOT NULL,
          gt int[] NOT NULL
        )
        """
    )


def table_count(cur: psycopg.Cursor, table: str) -> int:
    cur.execute("SELECT to_regclass(%s)", (table,))
    if cur.fetchone()[0] is None:
        return 0
    cur.execute(f"SELECT count(*) FROM {table}")
    return int(cur.fetchone()[0])


def load_base(cur: psycopg.Cursor, args: argparse.Namespace, paths: dict[str, Path]) -> None:
    xb = xbin_mmap(paths["base"], "uint8")
    nrow, _, _, indptr, indices, _ = spmat_fields(paths["base_metadata"])
    if xb.shape != (nrow, DIM):
        raise ValueError(f"base vector shape {xb.shape} does not match metadata rows {nrow}")
    existing = table_count(cur, args.table)
    if existing == xb.shape[0] and not args.reload:
        print(f"{args.table} already has {existing} rows; skip base load", flush=True)
        return
    if existing:
        raise SystemExit(f"{args.table} has {existing} rows; pass --reload to rebuild")
    print(f"loading {xb.shape[0]} YFCC base rows into {args.table}", flush=True)
    loaded = 0
    start = time.perf_counter()
    with cur.copy(f"COPY {args.table} (id, embedding, tags, tag_count) FROM STDIN") as copy:
        for i in range(xb.shape[0]):
            lo = int(indptr[i])
            hi = int(indptr[i + 1])
            tags = indices[lo:hi]
            copy.write_row((i, vector_literal(xb[i]), int_array_literal(tags), hi - lo))
            loaded += 1
            if loaded % args.progress_rows == 0:
                elapsed = time.perf_counter() - start
                print(f"  loaded base {loaded}/{xb.shape[0]} rows at {loaded / max(elapsed, 1):.0f} rows/s", flush=True)
    print(f"loaded base rows in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


def load_queries(cur: psycopg.Cursor, args: argparse.Namespace, paths: dict[str, Path]) -> None:
    xq = xbin_mmap(paths["query"], "uint8")
    gt = ibin_ids_mmap(paths["gt"])
    nrow, _, _, indptr, indices, _ = spmat_fields(paths["query_metadata"])
    if xq.shape[0] != nrow or xq.shape[1] != DIM:
        raise ValueError(f"query shape {xq.shape} does not match metadata rows {nrow}")
    if gt.shape[0] != xq.shape[0]:
        raise ValueError(f"GT shape {gt.shape} does not match query rows {xq.shape[0]}")
    existing = table_count(cur, args.query_table)
    if existing == xq.shape[0] and not args.reload:
        print(f"{args.query_table} already has {existing} rows; skip query load", flush=True)
        return
    if existing:
        raise SystemExit(f"{args.query_table} has {existing} rows; pass --reload to rebuild")
    print(f"loading {xq.shape[0]} YFCC query rows into {args.query_table}", flush=True)
    with cur.copy(f"COPY {args.query_table} (qid, embedding, tags, tag_count, gt) FROM STDIN") as copy:
        for i in range(xq.shape[0]):
            lo = int(indptr[i])
            hi = int(indptr[i + 1])
            tags = indices[lo:hi]
            copy.write_row((i, vector_literal(xq[i]), int_array_literal(tags), hi - lo, int_array_literal(gt[i])))
            if (i + 1) % args.progress_rows == 0:
                print(f"  loaded queries {i + 1}/{xq.shape[0]}", flush=True)


def create_indexes(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    print("creating scalar/query indexes", flush=True)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {args.table}_tags_gin ON {args.table} USING gin (tags)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {args.table}_tag_count_idx ON {args.table} (tag_count)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {args.query_table}_tag_count_idx ON {args.query_table} (tag_count)")
    cur.execute(f"ANALYZE {args.table}")
    cur.execute(f"ANALYZE {args.query_table}")
    print("creating HNSW index", flush=True)
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {args.index}
        ON {args.table}
        USING hnsw (embedding vector_l2_ops)
        WITH (m = {int(args.hnsw_m)}, ef_construction = {int(args.ef_construction)})
        """
    )
    cur.execute(f"ANALYZE {args.table}")


def create_guidance_meta(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    meta = f"{args.table}_guidance_meta"
    print(f"creating {meta}", flush=True)
    cur.execute(f"DROP TABLE IF EXISTS {meta}")
    cur.execute(
        f"""
        CREATE TABLE {meta} AS
        SELECT ctid AS heap_tid, id, tags, tag_count
        FROM {args.table}
        """
    )
    cur.execute(f"CREATE INDEX {meta}_tags_gin ON {meta} USING gin (tags)")
    cur.execute(f"CREATE INDEX {meta}_id_idx ON {meta} (id)")
    cur.execute(f"ANALYZE {meta}")


def write_manifest(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset,path,size_bytes",
        *[f"{name},{path},{path.stat().st_size if path.exists() else 0}" for name, path in paths.items()],
    ]
    args.manifest.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and load BigANN YFCC10M filtered dataset into PostgreSQL + pgvector.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("YFCC10M_DATA_DIR", Path(os.environ.get("OOD_ANNS_DATA", "data/ood_anns")) / "YFCC10M")),
    )
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-guidance-meta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--maintenance-work-mem", default="4GB")
    parser.add_argument("--parallel-maintenance-workers", type=int, default=4)
    parser.add_argument("--progress-rows", type=int, default=100000)
    parser.add_argument("--manifest", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_pgvector_manifest_20260713.csv"))
    args = parser.parse_args()

    paths = {name: args.data_dir / filename for name, filename in FILES.items()}
    if args.download:
        for name, filename in FILES.items():
            download_file(f"{BASE_URL}/{filename}", paths[name])
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")
    write_manifest(args, paths)

    if not args.load and not args.create_indexes and not args.create_guidance_meta:
        return

    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SET client_min_messages = warning")
        cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
        cur.execute(f"SET max_parallel_maintenance_workers = {int(args.parallel_maintenance_workers)}")
        ensure_schema(cur, args.table, args.query_table, args.reload)
        if args.load:
            load_base(cur, args, paths)
            load_queries(cur, args, paths)
        if args.create_indexes:
            create_indexes(cur, args)
        if args.create_guidance_meta:
            create_guidance_meta(cur, args)


if __name__ == "__main__":
    main()
