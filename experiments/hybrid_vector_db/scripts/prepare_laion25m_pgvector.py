from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psycopg
import pyarrow.parquet as pq
import spacy

from common_pg import pg_config_from_env


DATA_DIR = Path(os.environ.get("LAION25M_DATA_DIR", Path(os.environ.get("OOD_ANNS_DATA", "data/ood_anns")) / "LAION25M"))
PROCESSED = DATA_DIR / "processed"
TABLE = "laion25m_pgvector"
QUERY_TABLE = "laion25m_queries"
INDEX = f"{TABLE}_embedding_hnsw"
DIM = 512
BASE_ROW_TARGET = 25_000_000
MAX_BASE_SHARDS = 26
QUERY_SHARD = 26
QUERY_ROWS = 10000
# LAION captions are often title/product/brand/location strings. spaCy tags many
# useful content words in this style as proper nouns, so excluding PROPN drops a
# large fraction of usable caption labels.
POS_KEEP = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
PG_INT4_OID = 23


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def metadata_path(shard: int) -> Path:
    return DATA_DIR / "metadata" / f"metadata_{shard}.parquet"


def img_path(shard: int) -> Path:
    return DATA_DIR / "img_emb" / f"img_emb_{shard}.npy"


def text_path(shard: int) -> Path:
    return DATA_DIR / "text_emb" / f"text_emb_{shard}.npy"


def base_plan() -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    remaining = BASE_ROW_TARGET
    for shard in range(MAX_BASE_SHARDS):
        n = int(npy(img_path(shard)).shape[0])
        take = min(n, remaining)
        rows.append((shard, take))
        remaining -= take
        if remaining == 0:
            return rows
    raise SystemExit(f"not enough LAION shards to reach {BASE_ROW_TARGET} rows")


def require_files() -> None:
    missing: list[Path] = []
    for shard in range(MAX_BASE_SHARDS):
        for path in (img_path(shard), metadata_path(shard)):
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path)
    for path in (metadata_path(QUERY_SHARD), text_path(QUERY_SHARD)):
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
    if missing:
        raise SystemExit("missing LAION25M source files:\n" + "\n".join(str(p) for p in missing))


def total_base_rows() -> int:
    return BASE_ROW_TARGET


def read_metadata_columns(path: Path, row_limit: int | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    schema_names = set(pq.ParquetFile(path).schema.names)
    text_col = "TEXT" if "TEXT" in schema_names else "caption"
    width_col = "original_width" if "original_width" in schema_names else ("WIDTH" if "WIDTH" in schema_names else "width")
    height_col = "original_height" if "original_height" in schema_names else ("HEIGHT" if "HEIGHT" in schema_names else "height")
    table = pq.read_table(path, columns=[text_col, width_col, height_col])
    if row_limit is not None:
        table = table.slice(0, row_limit)
    text = table.column(text_col).to_pylist()
    width = np.asarray(table.column(width_col).to_numpy(zero_copy_only=False), dtype=np.int32)
    height = np.asarray(table.column(height_col).to_numpy(zero_copy_only=False), dtype=np.int32)
    return [str(x) if x is not None else "" for x in text], width, height


def normalize_token(token) -> str | None:
    if token.pos_ not in POS_KEEP:
        return None
    if token.is_stop or not token.is_alpha:
        return None
    text = token.lemma_.lower().strip() if token.lemma_ else token.text.lower().strip()
    if text in {"", "-pron-"} or len(text) < 2 or len(text) > 64:
        return None
    return text


def iter_doc_labels(nlp, texts: Iterable[str], batch_size: int, n_process: int):
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        labels = sorted({tok for token in doc if (tok := normalize_token(token)) is not None})
        yield labels


def write_vocab(path: Path, vocab: dict[str, int], counts: list[int]) -> None:
    rows = sorted(vocab.items(), key=lambda kv: kv[1])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label_id", "token", "base_count"])
        writer.writeheader()
        for token, label_id in rows:
            writer.writerow({"label_id": label_id, "token": token, "base_count": counts[label_id]})


def load_vocab(path: Path) -> tuple[dict[str, int], list[int]]:
    vocab: dict[str, int] = {}
    counts: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_id = int(row["label_id"])
            token = row["token"]
            while len(counts) <= label_id:
                counts.append(0)
            vocab[token] = label_id
            counts[label_id] = int(row["base_count"])
    return vocab, counts


def build_labels(args: argparse.Namespace) -> None:
    require_files()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    marker = PROCESSED / "labels.done"
    if marker.exists() and not args.rebuild_labels:
        print(f"labels already built: {marker}", flush=True)
        return

    n_total = total_base_rows()
    width_path = PROCESSED / "base_width.int32"
    offsets_path = PROCESSED / "base_label_offsets.int64"
    labels_path = PROCESSED / "base_labels.int32"
    vocab_path = PROCESSED / "label_vocab.csv"
    query_width_path = PROCESSED / "query_width.int32"
    query_offsets_path = PROCESSED / "query_label_offsets.int64"
    query_labels_path = PROCESSED / "query_labels.int32"
    query_fbin_path = PROCESSED / "query.text_emb_10k.fbin"

    nlp = spacy.load(args.spacy_model, disable=["parser", "ner"])
    vocab: dict[str, int] = {}
    counts: list[int] = []
    base_width = np.memmap(width_path, dtype=np.int32, mode="w+", shape=(n_total,))
    offsets = np.memmap(offsets_path, dtype=np.int64, mode="w+", shape=(n_total + 1,))
    offsets[0] = 0

    print(f"building LAION25M labels for {n_total} base rows", flush=True)
    started = time.perf_counter()
    label_chunks: list[np.ndarray] = []
    row_base = 0
    label_total = 0
    for shard, take_rows in base_plan():
        texts, width, _ = read_metadata_columns(metadata_path(shard), take_rows)
        base_width[row_base : row_base + len(width)] = width
        shard_labels: list[int] = []
        for local, tokens in enumerate(iter_doc_labels(nlp, texts, args.spacy_batch_size, args.spacy_processes)):
            ids: list[int] = []
            for token in tokens:
                label_id = vocab.get(token)
                if label_id is None:
                    label_id = len(vocab)
                    vocab[token] = label_id
                    counts.append(0)
                ids.append(label_id)
                counts[label_id] += 1
            shard_labels.extend(ids)
            offsets[row_base + local + 1] = label_total + len(shard_labels)
        label_chunks.append(np.asarray(shard_labels, dtype=np.int32))
        label_total += len(shard_labels)
        row_base += len(width)
        elapsed = time.perf_counter() - started
        print(
            f"  shard {shard} rows_used={take_rows} done rows={row_base}/{n_total} "
            f"labels={label_total} vocab={len(vocab)} elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

    labels = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=(label_total,))
    pos = 0
    for chunk in label_chunks:
        labels[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
    labels.flush()
    offsets.flush()
    base_width.flush()
    write_vocab(vocab_path, vocab, counts)

    print(f"building query labels and query fbin from text_emb_{QUERY_SHARD} first 10K", flush=True)
    q_texts, q_width, _ = read_metadata_columns(metadata_path(QUERY_SHARD), QUERY_ROWS)
    q_width_mm = np.memmap(query_width_path, dtype=np.int32, mode="w+", shape=(QUERY_ROWS,))
    q_offsets = np.memmap(query_offsets_path, dtype=np.int64, mode="w+", shape=(QUERY_ROWS + 1,))
    q_width_mm[:] = q_width[:QUERY_ROWS]
    q_offsets[0] = 0
    q_label_list: list[int] = []
    for i, tokens in enumerate(iter_doc_labels(nlp, q_texts, args.spacy_batch_size, args.spacy_processes)):
        ids = sorted({vocab[token] for token in tokens if token in vocab})
        q_label_list.extend(ids)
        q_offsets[i + 1] = len(q_label_list)
    q_labels = np.memmap(query_labels_path, dtype=np.int32, mode="w+", shape=(len(q_label_list),))
    q_labels[:] = np.asarray(q_label_list, dtype=np.int32)
    q_labels.flush()
    q_offsets.flush()
    q_width_mm.flush()

    xq = np.asarray(npy(text_path(QUERY_SHARD))[:QUERY_ROWS], dtype=np.float32)
    with query_fbin_path.open("wb") as f:
        f.write(struct.pack("<ii", QUERY_ROWS, DIM))
        f.write(xq.astype("<f4", copy=False).tobytes())

    marker.write_text(
        json.dumps(
            {
                "base_rows": n_total,
                "query_rows": QUERY_ROWS,
                "base_labels": int(label_total),
                "query_labels": int(len(q_label_list)),
                "vocab_size": int(len(vocab)),
                "spacy_model": args.spacy_model,
                "pos_keep": sorted(POS_KEEP),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {marker}", flush=True)


def rebuild_widths(args: argparse.Namespace) -> None:
    require_files()
    n_total = total_base_rows()
    base_width = np.memmap(PROCESSED / "base_width.int32", dtype=np.int32, mode="r+", shape=(n_total,))
    row_base = 0
    for shard, take_rows in base_plan():
        _, width, _ = read_metadata_columns(metadata_path(shard), take_rows)
        base_width[row_base : row_base + take_rows] = width
        row_base += take_rows
        print(f"  rebuilt base width shard {shard} rows={row_base}/{n_total}", flush=True)
    base_width.flush()

    q_width = np.memmap(PROCESSED / "query_width.int32", dtype=np.int32, mode="r+", shape=(QUERY_ROWS,))
    _, width, _ = read_metadata_columns(metadata_path(QUERY_SHARD), QUERY_ROWS)
    q_width[:] = width[:QUERY_ROWS]
    q_width.flush()
    marker = PROCESSED / "widths.done"
    marker.write_text(json.dumps({"base_rows": n_total, "query_rows": QUERY_ROWS, "source": "original_width"}, indent=2) + "\n")
    print(f"wrote {marker}", flush=True)


def binary_copy_header() -> bytes:
    return b"PGCOPY\n\xff\r\n\0" + struct.pack("!ii", 0, 0)


def binary_copy_trailer() -> bytes:
    return struct.pack("!h", -1)


def append_field(buf: bytearray, payload: bytes) -> None:
    buf.extend(struct.pack("!i", len(payload)))
    buf.extend(payload)


def vector_binary(row_be: np.ndarray) -> bytes:
    return struct.pack("!hh", row_be.shape[0], 0) + row_be.tobytes()


def int4_array_binary(values: np.ndarray) -> bytes:
    if values.size == 0:
        return struct.pack("!iiI", 0, 0, PG_INT4_OID)
    buf = bytearray()
    buf.extend(struct.pack("!iiI", 1, 0, PG_INT4_OID))
    buf.extend(struct.pack("!ii", int(values.size), 1))
    for value in values:
        buf.extend(struct.pack("!ii", 4, int(value)))
    return bytes(buf)


def append_base_row(
    buf: bytearray,
    row_id: int,
    vector_payload: bytes,
    width: int,
    labels_payload: bytes,
    label_count: int,
) -> None:
    buf.extend(struct.pack("!h", 5))
    append_field(buf, struct.pack("!i", int(row_id)))
    append_field(buf, vector_payload)
    append_field(buf, struct.pack("!i", int(width)))
    append_field(buf, labels_payload)
    append_field(buf, struct.pack("!i", int(label_count)))


def table_count(cur: psycopg.Cursor, table: str) -> int:
    cur.execute("SELECT to_regclass(%s)", (table,))
    if cur.fetchone()[0] is None:
        return 0
    cur.execute(f"SELECT count(*) FROM {table}")
    return int(cur.fetchone()[0])


def ensure_schema(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if args.reload:
        cur.execute(f"DROP TABLE IF EXISTS {args.table}_guidance_meta")
        cur.execute(f"DROP TABLE IF EXISTS {args.query_table}")
        cur.execute(f"DROP TABLE IF EXISTS {args.table}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {args.table} (
          id int PRIMARY KEY,
          embedding vector({DIM}) NOT NULL,
          width int NOT NULL,
          labels int[] NOT NULL,
          label_count int NOT NULL
        )
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {args.query_table} (
          qid int PRIMARY KEY,
          embedding vector({DIM}) NOT NULL,
          width int NOT NULL,
          labels int[] NOT NULL,
          label_count int NOT NULL
        )
        """
    )


def load_base(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    n_total = total_base_rows()
    existing = table_count(cur, args.table)
    if existing == n_total and not args.reload:
        print(f"{args.table} already has {existing} rows; skip base load", flush=True)
        return
    if existing:
        raise SystemExit(f"{args.table} has {existing} rows; pass --reload to rebuild")

    width = np.memmap(PROCESSED / "base_width.int32", dtype=np.int32, mode="r", shape=(n_total,))
    offsets = np.memmap(PROCESSED / "base_label_offsets.int64", dtype=np.int64, mode="r", shape=(n_total + 1,))
    labels = np.memmap(PROCESSED / "base_labels.int32", dtype=np.int32, mode="r", shape=(int(offsets[-1]),))
    print(f"loading {n_total} LAION25M base rows into {args.table}", flush=True)
    loaded = 0
    started = time.perf_counter()
    with cur.copy(f"COPY {args.table} (id, embedding, width, labels, label_count) FROM STDIN WITH (FORMAT binary)") as copy:
        copy.write(binary_copy_header())
        row_base = 0
        for shard, take_rows in base_plan():
            xb = np.asarray(npy(img_path(shard))[:take_rows], dtype=np.float32)
            for start in range(0, take_rows, args.copy_chunk_rows):
                end = min(start + args.copy_chunk_rows, take_rows)
                chunk_be = np.asarray(xb[start:end], dtype=">f4")
                buf = bytearray()
                for local in range(end - start):
                    row_id = row_base + start + local
                    lo = int(offsets[row_id])
                    hi = int(offsets[row_id + 1])
                    label_values = labels[lo:hi]
                    append_base_row(
                        buf,
                        row_id,
                        vector_binary(chunk_be[local]),
                        int(width[row_id]),
                        int4_array_binary(label_values),
                        hi - lo,
                    )
                copy.write(bytes(buf))
                loaded = row_base + end
                if loaded % args.progress_rows < args.copy_chunk_rows or loaded == n_total:
                    elapsed = time.perf_counter() - started
                    print(f"  loaded base {loaded}/{n_total} at {loaded / max(elapsed, 1):.0f} rows/s", flush=True)
            row_base += take_rows
        copy.write(binary_copy_trailer())
    print(f"loaded base in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def vector_text(row: np.ndarray) -> str:
    return "[" + ",".join(format(float(x), ".8g") for x in row) + "]"


def int_array_text(values: np.ndarray) -> str:
    return "{" + ",".join(str(int(x)) for x in values) + "}"


def load_queries(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    existing = table_count(cur, args.query_table)
    if existing == QUERY_ROWS and not args.reload:
        print(f"{args.query_table} already has {existing} rows; skip query load", flush=True)
        return
    if existing:
        raise SystemExit(f"{args.query_table} has {existing} rows; pass --reload to rebuild")
    q_width = np.memmap(PROCESSED / "query_width.int32", dtype=np.int32, mode="r", shape=(QUERY_ROWS,))
    q_offsets = np.memmap(PROCESSED / "query_label_offsets.int64", dtype=np.int64, mode="r", shape=(QUERY_ROWS + 1,))
    q_labels = np.memmap(PROCESSED / "query_labels.int32", dtype=np.int32, mode="r", shape=(int(q_offsets[-1]),))
    xq = np.asarray(npy(text_path(QUERY_SHARD))[:QUERY_ROWS], dtype=np.float32)
    print(f"loading {QUERY_ROWS} LAION25M query rows into {args.query_table}", flush=True)
    with cur.copy(f"COPY {args.query_table} (qid, embedding, width, labels, label_count) FROM STDIN") as copy:
        for qid in range(QUERY_ROWS):
            lo = int(q_offsets[qid])
            hi = int(q_offsets[qid + 1])
            copy.write_row((qid, vector_text(xq[qid]), int(q_width[qid]), int_array_text(q_labels[lo:hi]), hi - lo))
            if (qid + 1) % 1000 == 0:
                print(f"  loaded queries {qid + 1}/{QUERY_ROWS}", flush=True)


def create_scalar_indexes(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    print("creating scalar indexes", flush=True)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {args.table}_width_idx ON {args.table} (width)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {args.table}_labels_gin ON {args.table} USING gin (labels)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {args.table}_label_count_idx ON {args.table} (label_count)")
    cur.execute(f"ANALYZE {args.table}")
    cur.execute(f"ANALYZE {args.query_table}")


def create_hnsw_index(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
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
        SELECT ctid AS heap_tid, id, width, labels, label_count
        FROM {args.table}
        """
    )
    cur.execute(f"CREATE INDEX {meta}_width_idx ON {meta} (width)")
    cur.execute(f"CREATE INDEX {meta}_labels_gin ON {meta} USING gin (labels)")
    cur.execute(f"CREATE INDEX {meta}_id_idx ON {meta} (id)")
    cur.execute(f"ANALYZE {meta}")


def summarize_processed(args: argparse.Namespace) -> None:
    marker = PROCESSED / "labels.done"
    if not marker.exists():
        print("labels not built yet", flush=True)
        return
    print(marker.read_text(encoding="utf-8"), flush=True)
    with (PROCESSED / "label_vocab.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        top = sorted(reader, key=lambda r: int(r["base_count"]), reverse=True)[:20]
    print("top labels:", top, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LAION25M image vectors + caption-derived filters for pgvector.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--build-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--rebuild-widths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--create-scalar-indexes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--create-hnsw-index", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--create-guidance-meta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--spacy-batch-size", type=int, default=1000)
    parser.add_argument("--spacy-processes", type=int, default=8)
    parser.add_argument("--copy-chunk-rows", type=int, default=2000)
    parser.add_argument("--progress-rows", type=int, default=100000)
    parser.add_argument("--maintenance-work-mem", default="16GB")
    parser.add_argument("--parallel-maintenance-workers", type=int, default=8)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    args = parser.parse_args()

    if args.build_labels:
        build_labels(args)
    if args.rebuild_widths:
        rebuild_widths(args)
    if args.summarize:
        summarize_processed(args)
    if args.create_indexes:
        args.create_scalar_indexes = True
        args.create_hnsw_index = True
    if not (args.load or args.create_scalar_indexes or args.create_hnsw_index or args.create_guidance_meta):
        return
    require_files()
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SET client_min_messages = warning")
        cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
        cur.execute(f"SET max_parallel_maintenance_workers = {int(args.parallel_maintenance_workers)}")
        ensure_schema(cur, args)
        if args.load:
            load_base(cur, args)
            load_queries(cur, args)
        if args.create_scalar_indexes:
            create_scalar_indexes(cur, args)
        if args.create_hnsw_index:
            create_hnsw_index(cur, args)
        if args.create_guidance_meta:
            create_guidance_meta(cur, args)


if __name__ == "__main__":
    main()
