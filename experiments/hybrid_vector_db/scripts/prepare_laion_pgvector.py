from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from common_pg import pg_config_from_env


DATA_DIR = Path(os.environ.get("LAION10M_DATA_DIR", Path(os.environ.get("OOD_ANNS_DATA", "data/ood_anns")) / "LAION10M"))
BASE_FBIN = DATA_DIR / "base.10M.fbin"
QUERY_FBIN = DATA_DIR / "query.10k.fbin"
GT_IBIN = DATA_DIR / "gt.10k.ibin"
TABLE = "laion10m_pgvector"
QUERY_TABLE = "laion10m_queries"
INDEX = f"{TABLE}_embedding_hnsw"
DIM = 512
TARGET_PCTS = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def xbin_mmap(path: Path, dtype: str = "float32") -> np.memmap:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    expected = 8 + int(n) * int(d) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path} size mismatch: got {path.stat().st_size}, expected {expected}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(int(n), int(d)))


def ibin_mmap(path: Path) -> np.memmap:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    expected = 8 + int(n) * int(d) * np.dtype("int32").itemsize
    expected_with_dist = 8 + int(n) * int(d) * (np.dtype("int32").itemsize + np.dtype("float32").itemsize)
    if path.stat().st_size not in {expected, expected_with_dist}:
        raise ValueError(
            f"{path} size mismatch: got {path.stat().st_size}, expected {expected} or {expected_with_dist}"
        )
    return np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(int(n), int(d)))


def direction(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def ensure_scores(base: np.memmap, scores_path: Path, seed: int, chunk_rows: int) -> np.memmap:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    n, d = base.shape
    scores = np.memmap(scores_path, dtype=np.float32, mode="w+" if not scores_path.exists() else "r+", shape=(n,))
    marker = scores_path.with_suffix(scores_path.suffix + ".done")
    if marker.exists():
        return scores
    w = direction(d, seed)
    start_time = time.perf_counter()
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        scores[start:end] = np.asarray(base[start:end], dtype=np.float32) @ w
        if end % (chunk_rows * 5) == 0 or end == n:
            elapsed = time.perf_counter() - start_time
            print(f"  scored {end}/{n} rows at {end / max(elapsed, 1):.0f} rows/s", flush=True)
    scores.flush()
    marker.write_text(json.dumps({"rows": int(n), "dim": int(d), "seed": seed}) + "\n")
    return scores


def write_thresholds(scores: np.ndarray, out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(scores.shape[0])
    values = np.asarray(scores)
    for pct in TARGET_PCTS:
        threshold = float(np.quantile(values, pct / 100.0, method="nearest"))
        count = int(np.count_nonzero(values <= threshold))
        rows.append(
            {
                "filter_name": f"topic_le_{str(pct).replace('.', 'p')}",
                "target_pct": pct,
                "actual_pct": 100.0 * count / max(n, 1),
                "threshold": threshold,
                "rows": count,
                "predicate": f"topic_score <= {threshold:.9g}::real",
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote thresholds {out}", flush=True)
    return rows


def vector_text(row: np.ndarray) -> str:
    return "[" + ",".join(format(float(x), ".8g") for x in row) + "]"


def int_array_text(row: np.ndarray) -> str:
    return "{" + ",".join(str(int(x)) for x in row) + "}"


def binary_copy_header() -> bytes:
    return b"PGCOPY\n\xff\r\n\0" + struct.pack("!ii", 0, 0)


def binary_copy_trailer() -> bytes:
    return struct.pack("!h", -1)


def vector_binary(row_be: np.ndarray) -> bytes:
    return struct.pack("!hh", row_be.shape[0], 0) + row_be.tobytes()


def append_field(buf: bytearray, payload: bytes) -> None:
    buf.extend(struct.pack("!i", len(payload)))
    buf.extend(payload)


def append_base_row(buf: bytearray, row_id: int, vector_payload: bytes, topic_score: float) -> None:
    buf.extend(struct.pack("!h", 3))
    append_field(buf, struct.pack("!i", row_id))
    append_field(buf, vector_payload)
    append_field(buf, struct.pack("!f", float(topic_score)))


def table_count(cur: psycopg.Cursor, table: str) -> int:
    cur.execute("SELECT to_regclass(%s)", (table,))
    if cur.fetchone()[0] is None:
        return 0
    cur.execute(f"SELECT count(*) FROM {table}")
    return int(cur.fetchone()[0])


def ensure_schema(cur: psycopg.Cursor, table: str, query_table: str, reload: bool) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if reload:
        cur.execute(f"DROP TABLE IF EXISTS {table}_guidance_meta")
        cur.execute(f"DROP TABLE IF EXISTS {query_table}")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
          id int PRIMARY KEY,
          embedding vector({DIM}) NOT NULL,
          topic_score real NOT NULL
        )
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {query_table} (
          qid int PRIMARY KEY,
          embedding vector({DIM}) NOT NULL,
          gt int[] NOT NULL
        )
        """
    )


def load_base(cur: psycopg.Cursor, table: str, base: np.memmap, scores: np.memmap, progress_rows: int, flush_rows: int) -> None:
    existing = table_count(cur, table)
    if existing == base.shape[0]:
        print(f"{table} already has {existing} rows; skip base load", flush=True)
        return
    if existing:
        raise SystemExit(f"{table} has {existing} rows; pass --reload to rebuild")
    n, d = base.shape
    print(f"loading {n} LAION base rows into {table} with binary COPY", flush=True)
    loaded = 0
    start_time = time.perf_counter()
    with cur.copy(f"COPY {table} (id, embedding, topic_score) FROM STDIN WITH (FORMAT binary)") as copy:
        copy.write(binary_copy_header())
        for start in range(0, n, flush_rows):
            end = min(start + flush_rows, n)
            chunk_be = np.asarray(base[start:end], dtype=">f4")
            buf = bytearray()
            for offset in range(end - start):
                row_id = start + offset
                append_base_row(buf, row_id, vector_binary(chunk_be[offset]), float(scores[row_id]))
            copy.write(bytes(buf))
            loaded = end
            if loaded % progress_rows == 0 or loaded == n:
                elapsed = time.perf_counter() - start_time
                print(f"  loaded base {loaded}/{n} at {loaded / max(elapsed, 1):.0f} rows/s", flush=True)
        copy.write(binary_copy_trailer())
    print(f"loaded base rows in {(time.perf_counter() - start_time) / 60:.1f} min", flush=True)


def load_queries(cur: psycopg.Cursor, query_table: str, query_fbin: Path, gt_ibin: Path) -> None:
    xq = xbin_mmap(query_fbin)
    gt = ibin_mmap(gt_ibin)
    existing = table_count(cur, query_table)
    if existing == xq.shape[0]:
        print(f"{query_table} already has {existing} rows; skip query load", flush=True)
        return
    if existing:
        raise SystemExit(f"{query_table} has {existing} rows; pass --reload to rebuild")
    print(f"loading {xq.shape[0]} LAION query rows into {query_table}", flush=True)
    with cur.copy(f"COPY {query_table} (qid, embedding, gt) FROM STDIN") as copy:
        for i in range(xq.shape[0]):
            copy.write_row((i, vector_text(xq[i]), int_array_text(gt[i])))


def create_indexes(cur: psycopg.Cursor, table: str, index: str, hnsw_m: int, ef_construction: int) -> None:
    print("creating scalar index", flush=True)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_topic_score_idx ON {table} (topic_score)")
    cur.execute(f"ANALYZE {table}")
    print("creating HNSW index", flush=True)
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {index}
        ON {table}
        USING hnsw (embedding vector_l2_ops)
        WITH (m = {int(hnsw_m)}, ef_construction = {int(ef_construction)})
        """
    )
    cur.execute(f"ANALYZE {table}")


def create_guidance_meta(cur: psycopg.Cursor, table: str) -> None:
    meta = f"{table}_guidance_meta"
    print(f"creating {meta}", flush=True)
    cur.execute(f"DROP TABLE IF EXISTS {meta}")
    cur.execute(
        f"""
        CREATE TABLE {meta} AS
        SELECT ctid AS heap_tid, id, topic_score
        FROM {table}
        """
    )
    cur.execute(f"CREATE INDEX {meta}_topic_score_idx ON {meta} (topic_score)")
    cur.execute(f"CREATE INDEX {meta}_id_idx ON {meta} (id)")
    cur.execute(f"ANALYZE {meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load LAION10M into PostgreSQL + pgvector and build controlled metadata.")
    parser.add_argument("--base-fbin", type=Path, default=BASE_FBIN)
    parser.add_argument("--query-fbin", type=Path, default=QUERY_FBIN)
    parser.add_argument("--gt-ibin", type=Path, default=GT_IBIN)
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--scores", type=Path, default=Path("results/hybrid_vector_db/laion10m_topic_score_seed13.float32"))
    parser.add_argument("--thresholds", type=Path, default=Path("results/hybrid_vector_db/laion10m_controlled_filters_20260713.csv"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--flush-rows", type=int, default=5000)
    parser.add_argument("--progress-rows", type=int, default=100000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-guidance-meta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--maintenance-work-mem", default="8GB")
    parser.add_argument("--parallel-maintenance-workers", type=int, default=4)
    args = parser.parse_args()

    base = xbin_mmap(args.base_fbin)
    if base.shape[1] != DIM:
        raise ValueError(f"expected dim {DIM}, got {base.shape[1]}")
    scores = ensure_scores(base, args.scores, args.seed, args.chunk_rows)
    write_thresholds(scores, args.thresholds)
    if args.scores_only:
        return

    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SET client_min_messages = warning")
        cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
        cur.execute(f"SET max_parallel_maintenance_workers = {int(args.parallel_maintenance_workers)}")
        ensure_schema(cur, args.table, args.query_table, args.reload)
        if args.load:
            load_base(cur, args.table, base, scores, args.progress_rows, args.flush_rows)
            load_queries(cur, args.query_table, args.query_fbin, args.gt_ibin)
        if args.create_indexes:
            create_indexes(cur, args.table, args.index, args.hnsw_m, args.ef_construction)
        if args.create_guidance_meta:
            create_guidance_meta(cur, args.table)


if __name__ == "__main__":
    main()
