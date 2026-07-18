from __future__ import annotations

import argparse
import csv
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FILTERS: list[tuple[str, str, str]] = [
    ("popular_ge1000", "attribute_filters", "item_rating_number >= 1000"),
    ("price_10_to_20", "attribute_filters", "has_price AND price > 10 AND price <= 20"),
    ("rating5_price_le10", "attribute_filters", "has_price AND price <= 10 AND rating = 5"),
    ("long_review_ge500", "attribute_filters", "review_text_len >= 500"),
    ("grocery_rating5", "attribute_filters", "main_category = 'Grocery' AND rating = 5"),
    ("grocery_helpful", "attribute_filters", "main_category = 'Grocery' AND helpful_vote >= 1"),
    ("helpful_ge20", "attribute_filters", "helpful_vote >= 20"),
    ("grocery_long500", "attribute_filters", "main_category = 'Grocery' AND review_text_len >= 500"),
]


@dataclass
class Materialized:
    ids: np.ndarray | None
    bitmap: np.ndarray | None
    backing: Any
    metrics: dict[str, object]


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def pack_bitmap_from_ids(ids: np.ndarray, total_rows: int) -> tuple[np.ndarray, float]:
    def run() -> np.ndarray:
        mask = np.zeros(total_rows, dtype=np.bool_)
        mask[ids] = True
        return np.packbits(mask, bitorder="little")

    return timed(run)


def materialize_numpy_ids(conn, table: str, predicate: str) -> Materialized:
    cursor, execute_ms = timed(lambda: conn.execute(f"SELECT id FROM {table} WHERE {predicate}"))
    result, fetch_numpy_ms = timed(lambda: cursor.fetchnumpy())
    ids, asarray_ms = timed(lambda: np.asarray(result["id"], dtype=np.int64))
    return Materialized(
        ids=ids,
        bitmap=None,
        backing=result,
        metrics={
            "source": "fetchnumpy_ids",
            "execute_ms": execute_ms,
            "fetch_ms": fetch_numpy_ms,
            "convert_ms": asarray_ms,
            "bitmap_pack_ms": 0.0,
            "total_materialize_ms": execute_ms + fetch_numpy_ms + asarray_ms,
            "arrow_chunks": "",
            "zero_copy": "",
        },
    )


def materialize_arrow_ids(conn, table: str, predicate: str) -> Materialized:
    cursor, execute_ms = timed(lambda: conn.execute(f"SELECT id FROM {table} WHERE {predicate}"))
    arrow_table, fetch_arrow_ms = timed(lambda: cursor.to_arrow_table())
    col = arrow_table.column("id")
    chunk_count = col.num_chunks
    zero_copy = False

    def to_numpy() -> tuple[np.ndarray, Any]:
        nonlocal zero_copy
        if col.num_chunks == 1:
            chunk = col.chunk(0)
            try:
                arr = chunk.to_numpy(zero_copy_only=True)
                zero_copy = True
                return np.asarray(arr, dtype=np.int64), chunk
            except Exception:
                arr = chunk.to_numpy(zero_copy_only=False)
                return np.asarray(arr, dtype=np.int64), chunk
        combined = col.combine_chunks()
        try:
            arr = combined.to_numpy(zero_copy_only=True)
            zero_copy = True
            return np.asarray(arr, dtype=np.int64), combined
        except Exception:
            arr = combined.to_numpy(zero_copy_only=False)
            return np.asarray(arr, dtype=np.int64), combined

    (ids, array_backing), arrow_to_numpy_ms = timed(to_numpy)
    return Materialized(
        ids=ids,
        bitmap=None,
        backing=(arrow_table, array_backing),
        metrics={
            "source": "arrow_ids",
            "execute_ms": execute_ms,
            "fetch_ms": fetch_arrow_ms,
            "convert_ms": arrow_to_numpy_ms,
            "bitmap_pack_ms": 0.0,
            "total_materialize_ms": execute_ms + fetch_arrow_ms + arrow_to_numpy_ms,
            "arrow_chunks": chunk_count,
            "zero_copy": zero_copy,
        },
    )


def materialize_numpy_bitmap(conn, table: str, predicate: str, total_rows: int) -> Materialized:
    mat = materialize_numpy_ids(conn, table, predicate)
    assert mat.ids is not None
    bitmap, pack_ms = pack_bitmap_from_ids(mat.ids, total_rows)
    metrics = dict(mat.metrics)
    metrics["source"] = "fetchnumpy_ids_to_bitmap"
    metrics["bitmap_pack_ms"] = pack_ms
    metrics["total_materialize_ms"] = float(metrics["total_materialize_ms"]) + pack_ms
    return Materialized(ids=mat.ids, bitmap=bitmap, backing=(mat.backing, bitmap), metrics=metrics)


def materialize_arrow_bitmap(conn, table: str, predicate: str, total_rows: int) -> Materialized:
    mat = materialize_arrow_ids(conn, table, predicate)
    assert mat.ids is not None
    bitmap, pack_ms = pack_bitmap_from_ids(mat.ids, total_rows)
    metrics = dict(mat.metrics)
    metrics["source"] = "arrow_ids_to_bitmap"
    metrics["bitmap_pack_ms"] = pack_ms
    metrics["total_materialize_ms"] = float(metrics["total_materialize_ms"]) + pack_ms
    return Materialized(ids=mat.ids, bitmap=bitmap, backing=(mat.backing, bitmap), metrics=metrics)


def materialize_duckdb_bitstring_bitmap(conn, table: str, predicate: str, total_rows: int) -> Materialized:
    query = f"""
        SELECT count(*)::BIGINT, bitstring_agg(id, 0, {total_rows - 1})
        FROM {table}
        WHERE {predicate}
    """
    cursor, execute_ms = timed(lambda: conn.execute(query))
    row, fetch_ms = timed(lambda: cursor.fetchone())

    def convert() -> np.ndarray:
        bits = str(row[1])
        values = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48
        return np.packbits(values.astype(np.bool_), bitorder="little")

    bitmap, convert_ms = timed(convert)
    return Materialized(
        ids=None,
        bitmap=bitmap,
        backing=bitmap,
        metrics={
            "source": "duckdb_bitstring_to_bitmap",
            "execute_ms": execute_ms,
            "fetch_ms": fetch_ms,
            "convert_ms": convert_ms,
            "bitmap_pack_ms": 0.0,
            "total_materialize_ms": execute_ms + fetch_ms + convert_ms,
            "arrow_chunks": "",
            "zero_copy": "",
            "bitstring_count": int(row[0]),
        },
    )


def build_selector(mat: Materialized, selector_kind: str):
    import faiss

    def run():
        if selector_kind == "batch":
            if mat.ids is None:
                return None
            return faiss.IDSelectorBatch(mat.ids.size, faiss.swig_ptr(mat.ids))
        if selector_kind == "bitmap":
            if mat.bitmap is None:
                return None
            return faiss.IDSelectorBitmap(mat.bitmap.size, faiss.swig_ptr(mat.bitmap))
        raise ValueError(selector_kind)

    return timed(run)


def hnsw_search(index, query: np.ndarray, k: int, ef_search: int, selector=None) -> tuple[list[int], float]:
    import faiss

    def run() -> list[int]:
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(ef_search)
        if selector is not None:
            params.sel = selector
        _, ids = index.search(query.reshape(1, -1), k, params=params)
        return [int(x) for x in ids[0] if x >= 0]

    return timed(run)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} rows={len(rows)}", flush=True)


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["source"]), str(row["selector_kind"])), []).append(row)
    summary: list[dict[str, object]] = []
    order = {name: i for i, (name, _, _) in enumerate(FILTERS)}
    for (filter_name, source, selector_kind), items in sorted(groups.items(), key=lambda x: (order[x[0][0]], x[0][1], x[0][2])):
        search_items = [row for row in items if row["phase"] == "search"]
        mat_items = [row for row in items if row["phase"] == "materialize"]
        first = items[0]
        summary.append(
            {
                "suite": first["suite"],
                "filter_name": filter_name,
                "predicate": first["predicate"],
                "source": source,
                "selector_kind": selector_kind,
                "rows": first["rows"],
                "actual_rate": first["actual_rate"],
                "execute_ms_avg": statistics.mean(float(row["execute_ms"]) for row in mat_items),
                "fetch_ms_avg": statistics.mean(float(row["fetch_ms"]) for row in mat_items),
                "convert_ms_avg": statistics.mean(float(row["convert_ms"]) for row in mat_items),
                "bitmap_pack_ms_avg": statistics.mean(float(row["bitmap_pack_ms"]) for row in mat_items),
                "materialize_ms_avg": statistics.mean(float(row["total_materialize_ms"]) for row in mat_items),
                "selector_build_ms_avg": statistics.mean(float(row["selector_build_ms"]) for row in mat_items),
                "search_ms_avg": statistics.mean(float(row["search_ms"]) for row in search_items) if search_items else "",
                "returned_avg": statistics.mean(float(row["returned"]) for row in search_items) if search_items else "",
                "arrow_chunks": first["arrow_chunks"],
                "zero_copy": first["zero_copy"],
            }
        )
    summary_path = out.with_name(out.stem + "_summary.csv")
    write_csv(summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/duckdb_arrow_materialize_benchmark.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--filter-suite", choices=["all", "attribute_filters"], default="all")
    args = parser.parse_args()

    import duckdb
    import faiss

    conn = duckdb.connect(str(args.duckdb))
    conn.execute(f"PRAGMA threads={int(args.threads)}")
    total_rows = int(conn.execute(f"SELECT count(*) FROM {args.table}").fetchone()[0])
    xb, rows, _ = read_fbin_memmap(args.fbin, args.rows)
    rows = min(rows, total_rows)
    rng = np.random.default_rng(args.seed)
    query_ids = rng.choice(rows, size=args.queries, replace=False)
    queries = np.ascontiguousarray(xb[query_ids], dtype=np.float32)
    index = faiss.read_index(str(args.index))

    materializers = [
        ("fetchnumpy_ids", "batch", lambda pred: materialize_numpy_ids(conn, args.table, pred)),
        ("arrow_ids", "batch", lambda pred: materialize_arrow_ids(conn, args.table, pred)),
        ("fetchnumpy_ids_to_bitmap", "bitmap", lambda pred: materialize_numpy_bitmap(conn, args.table, pred, total_rows)),
        ("arrow_ids_to_bitmap", "bitmap", lambda pred: materialize_arrow_bitmap(conn, args.table, pred, total_rows)),
        ("duckdb_bitstring_to_bitmap", "bitmap", lambda pred: materialize_duckdb_bitstring_bitmap(conn, args.table, pred, total_rows)),
    ]

    out_rows: list[dict[str, object]] = []
    for filter_name, suite, predicate in FILTERS:
        if args.filter_suite != "all" and suite != args.filter_suite:
            continue
        count = int(conn.execute(f"SELECT count(*) FROM {args.table} WHERE {predicate}").fetchone()[0])
        actual_rate = count / total_rows
        print(f"filter={filter_name} suite={suite} rows={count} rate={actual_rate:.4f}", flush=True)
        for source_name, selector_kind, materializer in materializers:
            for repeat in range(args.repeats):
                mat = materializer(predicate)
                if mat.ids is not None and len(mat.ids) != count:
                    raise RuntimeError(f"{source_name} produced {len(mat.ids)} ids, expected {count}")
                if "bitstring_count" in mat.metrics and int(mat.metrics["bitstring_count"]) != count:
                    raise RuntimeError(f"{source_name} bitstring count mismatch")
                selector, selector_build_ms = build_selector(mat, selector_kind)
                if selector is None:
                    continue
                common = {
                    "phase": "materialize",
                    "suite": suite,
                    "filter_name": filter_name,
                    "predicate": predicate,
                    "source": source_name,
                    "selector_kind": selector_kind,
                    "repeat": repeat,
                    "rows": count,
                    "actual_rate": actual_rate,
                    "ef_search": args.ef_search,
                    "execute_ms": mat.metrics["execute_ms"],
                    "fetch_ms": mat.metrics["fetch_ms"],
                    "convert_ms": mat.metrics["convert_ms"],
                    "bitmap_pack_ms": mat.metrics["bitmap_pack_ms"],
                    "total_materialize_ms": mat.metrics["total_materialize_ms"],
                    "selector_build_ms": selector_build_ms,
                    "search_ms": "",
                    "returned": "",
                    "arrow_chunks": mat.metrics["arrow_chunks"],
                    "zero_copy": mat.metrics["zero_copy"],
                }
                out_rows.append(common)
                for query_no, query in enumerate(queries):
                    ids, search_ms = hnsw_search(index, query, args.k, args.ef_search, selector=selector)
                    out_rows.append(
                        {
                            **common,
                            "phase": "search",
                            "query_no": query_no,
                            "query_id": int(query_ids[query_no]),
                            "search_ms": search_ms,
                            "returned": len(ids),
                        }
                    )
            print(f"  source={source_name} selector={selector_kind} done", flush=True)

    write_csv(args.out, out_rows)
    summarize(out_rows, args.out)


if __name__ == "__main__":
    main()
