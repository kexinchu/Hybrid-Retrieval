from __future__ import annotations

import argparse
import csv
import statistics
import struct
import time
from pathlib import Path

import numpy as np


RATE_FILTERS: list[tuple[str, float, str]] = [
    ("id_range_50", 0.5032, "id >= 0 AND id < 5032000"),
    ("id_range_20", 0.2189, "id >= 0 AND id < 2189000"),
    ("id_range_10", 0.0959, "id >= 0 AND id < 959000"),
    ("id_range_5", 0.0588, "id >= 0 AND id < 588000"),
    ("id_range_2", 0.0234, "id >= 0 AND id < 234000"),
    ("id_range_1", 0.0101, "id >= 0 AND id < 101000"),
    ("id_range_0_5", 0.0050, "id >= 0 AND id < 50000"),
    ("id_range_0_2", 0.0020, "id >= 0 AND id < 20000"),
]


SQL_COMPLEXITY_FILTERS: list[tuple[str, str]] = [
    ("id_range_0_5", "id >= 0 AND id < 50000"),
    ("helpful_ge25", "helpful_vote >= 25"),
    ("review_len_ge1300", "review_text_len >= 1300"),
    ("grocery_helpful_ge2", "main_category = 'Grocery' AND helpful_vote >= 2"),
    ("price_30_to_31", "has_price AND price > 30 AND price <= 31"),
    ("price_1_to_2", "has_price AND price > 1 AND price <= 2"),
]


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


def count_predicate(conn, table: str, predicate: str) -> tuple[int, float]:
    return timed(lambda: int(conn.execute(f"SELECT count(*) FROM {table} WHERE {predicate}").fetchone()[0]))


def materialize_ids(conn, table: str, predicate: str) -> tuple[np.ndarray, dict[str, float]]:
    cursor, execute_ms = timed(lambda: conn.execute(f"SELECT id FROM {table} WHERE {predicate}"))
    result, fetch_numpy_ms = timed(lambda: cursor.fetchnumpy())
    ids, asarray_ms = timed(lambda: np.asarray(result["id"], dtype=np.int64))
    return ids, {
        "select_execute_ms": execute_ms,
        "fetch_numpy_ms": fetch_numpy_ms,
        "asarray_ms": asarray_ms,
        "select_total_ms": execute_ms + fetch_numpy_ms + asarray_ms,
    }


def build_selector(ids: np.ndarray, selector_type: str, total_rows: int):
    import faiss

    def run():
        if selector_type == "batch":
            return faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids)), ids
        if selector_type == "bitmap":
            mask = np.zeros(total_rows, dtype=np.bool_)
            mask[ids] = True
            bitmap = np.packbits(mask, bitorder="little")
            return faiss.IDSelectorBitmap(bitmap.size, faiss.swig_ptr(bitmap)), bitmap
        raise ValueError(selector_type)

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


def exact_topk(xb: np.memmap, query: np.ndarray, ids: np.ndarray, k: int, chunk_size: int) -> list[int]:
    best_dist = np.empty(0, dtype=np.float32)
    best_ids = np.empty(0, dtype=np.int64)
    query_norm = float(np.dot(query, query))
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start : start + chunk_size]
        vecs = np.asarray(xb[chunk_ids], dtype=np.float32)
        dists = np.einsum("ij,ij->i", vecs, vecs) + query_norm - 2.0 * (vecs @ query)
        if best_ids.size:
            dists = np.concatenate([best_dist, dists])
            chunk_ids = np.concatenate([best_ids, chunk_ids])
        take = min(k, len(dists))
        pos = np.argpartition(dists, take - 1)[:take]
        order = np.argsort(dists[pos])
        best_dist = dists[pos][order]
        best_ids = chunk_ids[pos][order]
    return [int(x) for x in best_ids[:k]]


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ids[:k]) & set(truth[:k])) / min(k, len(truth))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} rows={len(rows)}", flush=True)


def summarize_sql(rows: list[dict[str, object]], group_keys: list[str], out: Path) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summary: list[dict[str, object]] = []
    for key, items in groups.items():
        base = {group_keys[i]: key[i] for i in range(len(group_keys))}
        count_ms = [float(row["count_ms"]) for row in items]
        select_total_ms = [float(row["select_total_ms"]) for row in items]
        fetch_ms = [float(row["fetch_numpy_ms"]) + float(row["asarray_ms"]) for row in items]
        extra_ms = [max(0.0, float(row["select_total_ms"]) - float(row["count_ms"])) for row in items]
        summary.append(
            {
                **base,
                "predicate": items[0]["predicate"],
                "actual_rate": items[0]["actual_rate"],
                "rows": items[0]["rows"],
                "count_ms_avg": mean(count_ms),
                "select_total_ms_avg": mean(select_total_ms),
                "fetch_numpy_asarray_ms_avg": mean(fetch_ms),
                "materialize_extra_vs_count_ms_avg": mean(extra_ms),
            }
        )
    write_csv(out, summary)


def run_sql_breakdowns(conn, table: str, total_rows: int, repeats: int, out_prefix: Path) -> None:
    rows: list[dict[str, object]] = []
    for name, target_rate, predicate in RATE_FILTERS:
        for repeat in range(repeats):
            count, count_ms = count_predicate(conn, table, predicate)
            ids, timings = materialize_ids(conn, table, predicate)
            rows.append(
                {
                    "suite": "rate_controlled_id_range",
                    "filter_name": name,
                    "target_rate": target_rate,
                    "predicate": predicate,
                    "repeat": repeat,
                    "rows": count,
                    "actual_rate": count / total_rows,
                    "count_ms": count_ms,
                    **timings,
                }
            )
            if len(ids) != count:
                raise RuntimeError(f"materialized {len(ids)} ids but count is {count} for {name}")
    write_csv(out_prefix.with_name(out_prefix.name + "_rate_sql_detail.csv"), rows)
    summarize_sql(rows, ["suite", "filter_name", "target_rate"], out_prefix.with_name(out_prefix.name + "_rate_sql_summary.csv"))

    rows = []
    for name, predicate in SQL_COMPLEXITY_FILTERS:
        for repeat in range(repeats):
            count, count_ms = count_predicate(conn, table, predicate)
            ids, timings = materialize_ids(conn, table, predicate)
            rows.append(
                {
                    "suite": "sql_complexity_around_0_5pct",
                    "filter_name": name,
                    "predicate": predicate,
                    "repeat": repeat,
                    "rows": count,
                    "actual_rate": count / total_rows,
                    "count_ms": count_ms,
                    **timings,
                }
            )
            if len(ids) != count:
                raise RuntimeError(f"materialized {len(ids)} ids but count is {count} for {name}")
    write_csv(out_prefix.with_name(out_prefix.name + "_complexity_sql_detail.csv"), rows)
    summarize_sql(rows, ["suite", "filter_name"], out_prefix.with_name(out_prefix.name + "_complexity_sql_summary.csv"))


def run_vector_fixed_recall(
    conn,
    table: str,
    xb: np.memmap,
    index,
    total_rows: int,
    query_ids: np.ndarray,
    ef_values: list[int],
    repeats: int,
    k: int,
    selector_type: str,
    target_recall: float,
    chunk_size: int,
    out_prefix: Path,
) -> None:
    queries = np.ascontiguousarray(xb[query_ids], dtype=np.float32)
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for name, target_rate, predicate in RATE_FILTERS:
        ids, materialize_timing = materialize_ids(conn, table, predicate)
        (selector, selector_backing), selector_build_ms = build_selector(ids, selector_type, total_rows)
        truth_by_query = [exact_topk(xb, query, ids, k, chunk_size) for query in queries]
        ef_stats: list[dict[str, float]] = []
        for ef in ef_values:
            latencies: list[float] = []
            recalls: list[float] = []
            for repeat in range(repeats):
                for query_no, query in enumerate(queries):
                    found, search_ms = hnsw_search(index, query, k, ef, selector=selector)
                    rec = recall_at_k(found, truth_by_query[query_no], k)
                    latencies.append(search_ms)
                    recalls.append(rec)
                    detail.append(
                        {
                            "filter_name": name,
                            "target_rate": target_rate,
                            "predicate": predicate,
                            "actual_rate": len(ids) / total_rows,
                            "rows": len(ids),
                            "selector_type": selector_type,
                            "ef_search": ef,
                            "repeat": repeat,
                            "query_id": int(query_ids[query_no]),
                            "recall": rec,
                            "vector_search_ms": search_ms,
                            "selector_build_ms_once": selector_build_ms,
                            "sql_select_total_ms_once": materialize_timing["select_total_ms"],
                        }
                    )
            ef_stats.append({"ef_search": ef, "recall": mean(recalls), "vector_search_ms": mean(latencies)})
        reached = [row for row in ef_stats if row["recall"] >= target_recall]
        chosen = reached[0] if reached else ef_stats[-1]
        summary.append(
            {
                "filter_name": name,
                "target_rate": target_rate,
                "predicate": predicate,
                "actual_rate": len(ids) / total_rows,
                "rows": len(ids),
                "selector_type": selector_type,
                "target_recall": target_recall,
                "target_reached": bool(reached),
                "chosen_ef_search": int(chosen["ef_search"]),
                "chosen_recall": chosen["recall"],
                "chosen_vector_search_ms": chosen["vector_search_ms"],
                "all_ef_recall_ms": "; ".join(
                    f"ef={int(row['ef_search'])}:recall={row['recall']:.3f},ms={row['vector_search_ms']:.2f}"
                    for row in ef_stats
                ),
                "selector_build_ms_once": selector_build_ms,
                "sql_select_total_ms_once": materialize_timing["select_total_ms"],
            }
        )
        print(f"vector fixed-recall suite finished {name}", flush=True)
    write_csv(out_prefix.with_name(out_prefix.name + "_vector_fixed_recall_detail.csv"), detail)
    write_csv(out_prefix.with_name(out_prefix.name + "_vector_fixed_recall_summary.csv"), summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--out-prefix", type=Path, default=Path("results/hybrid_vector_db/faiss_duckdb_breakdown_20260606"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ef-search-values", type=int, nargs="+", default=[100, 300, 1000, 3000])
    parser.add_argument("--selector-type", choices=["bitmap", "batch"], default="bitmap")
    parser.add_argument("--target-recall", type=float, default=0.6)
    parser.add_argument("--exact-chunk-size", type=int, default=200_000)
    parser.add_argument("--skip-vector", action="store_true")
    args = parser.parse_args()

    import duckdb
    import faiss

    conn = duckdb.connect(str(args.duckdb))
    conn.execute(f"PRAGMA threads={int(args.threads)}")
    total_rows = int(conn.execute(f"SELECT count(*) FROM {args.table}").fetchone()[0])
    run_sql_breakdowns(conn, args.table, total_rows, args.repeats, args.out_prefix)
    if args.skip_vector:
        return

    xb, rows, _ = read_fbin_memmap(args.fbin, args.rows)
    rows = min(rows, total_rows)
    rng = np.random.default_rng(args.seed)
    query_ids = rng.choice(rows, size=args.queries, replace=False)
    index = faiss.read_index(str(args.index))
    run_vector_fixed_recall(
        conn,
        args.table,
        xb,
        index,
        total_rows,
        query_ids,
        args.ef_search_values,
        args.repeats,
        args.k,
        args.selector_type,
        args.target_recall,
        args.exact_chunk_size,
        args.out_prefix,
    )


if __name__ == "__main__":
    main()
