from __future__ import annotations

import argparse
import csv
import statistics
import struct
import time
from pathlib import Path

import numpy as np

from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000.0


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def load_truth_rows(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    truth: dict[tuple[str, int], dict[str, object]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "pre_filter_exact":
                truth[(row["filter_name"], int(row["query_no"]))] = row
    return truth


def sql_count(conn, table: str, predicate: str) -> tuple[int, float]:
    def run() -> int:
        return int(conn.execute(f"SELECT count(*) FROM {table} WHERE {predicate}").fetchone()[0])

    return timed(run)


def sql_id_materialize(conn, table: str, predicate: str) -> tuple[np.ndarray, float, float, float]:
    t0 = time.perf_counter()
    cursor = conn.execute(f"SELECT id FROM {table} WHERE {predicate}")
    execute_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    result = cursor.fetchnumpy()
    fetch_ms = (time.perf_counter() - t1) * 1000.0
    ids = np.asarray(result["id"], dtype=np.int64)
    return ids, execute_ms, fetch_ms, execute_ms + fetch_ms


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def id_range_filters(total_rows: int) -> list[tuple[str, str, str]]:
    filters = []
    for label, rate in [
        ("50.00%", 0.50),
        ("20.00%", 0.20),
        ("10.00%", 0.10),
        ("5.00%", 0.05),
        ("2.00%", 0.02),
        ("1.00%", 0.01),
        ("0.50%", 0.005),
        ("0.20%", 0.002),
    ]:
        hi = max(1, int(total_rows * rate))
        filters.append((f"id_range_{label}", label, f"id >= 0 AND id < {hi}"))
    return filters


def difficulty_filters(total_rows: int) -> list[tuple[str, str, str]]:
    hi = max(1, int(total_rows * 0.005))
    return [
        ("simple_range_0_5", "0.50%", f"id >= 0 AND id < {hi}"),
        ("range_extra_columns_0_5", "0.50%", f"id >= 0 AND id < {hi} AND rating >= 0 AND review_text_len >= 0 AND item_rating_number >= 0"),
        ("non_sarg_arithmetic_0_5", "0.50%", f"id + 1 >= 1 AND id + 1 < {hi + 1}"),
        ("scattered_mod_0_5", "0.50%", "id % 200 = 0"),
    ]


def run_sql_decompose(args) -> None:
    import duckdb

    conn = duckdb.connect(str(args.duckdb), read_only=True)
    conn.execute(f"PRAGMA threads={int(args.threads)}")
    total_rows = int(conn.execute(f"SELECT count(*) FROM {args.table}").fetchone()[0])

    predicates = id_range_filters(total_rows) + difficulty_filters(total_rows)
    out_rows = []
    for name, target_rate, predicate in predicates:
        count_values = []
        execute_values = []
        fetch_values = []
        total_values = []
        rows_count = 0
        for _ in range(args.repeats):
            rows_count, count_ms = sql_count(conn, args.table, predicate)
            ids, execute_ms, fetch_ms, total_ms = sql_id_materialize(conn, args.table, predicate)
            if len(ids) != rows_count:
                raise RuntimeError(f"row mismatch for {name}: count={rows_count} ids={len(ids)}")
            count_values.append(count_ms)
            execute_values.append(execute_ms)
            fetch_values.append(fetch_ms)
            total_values.append(total_ms)
        out_rows.append(
            {
                "group": "id_rate" if name.startswith("id_range_") else "sql_difficulty_0_5",
                "filter_name": name,
                "target_rate": target_rate,
                "predicate": predicate,
                "rows": rows_count,
                "actual_rate": rows_count / total_rows,
                "sql_count_ms_mean": statistics.mean(count_values),
                "sql_execute_ms_mean": statistics.mean(execute_values),
                "materialize_fetch_ms_mean": statistics.mean(fetch_values),
                "select_id_total_ms_mean": statistics.mean(total_values),
                "select_id_total_ms_p50": statistics.median(total_values),
                "select_id_total_ms_p95": percentile(total_values, 95),
            }
        )

    args.sql_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sql_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.sql_out}", flush=True)


def run_ef_sweep(args) -> None:
    import duckdb
    import faiss

    conn = duckdb.connect(str(args.duckdb), read_only=True)
    conn.execute(f"PRAGMA threads={int(args.threads)}")
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    index = faiss.read_index(str(args.index))
    truth_rows = load_truth_rows(args.truth_csv)
    query_by_no = {int(row["query_no"]): int(row["query_id"]) for row in truth_rows.values()}
    query_nos = sorted(query_by_no)[: args.queries]
    query_ids = np.asarray([query_by_no[q] for q in query_nos], dtype=np.int64)
    queries = np.ascontiguousarray(xb[query_ids], dtype=np.float32)

    out_rows = []
    for filter_name, target_rate, predicate in ATTR_FILTERS:
        ids, _, _, id_total_ms = sql_id_materialize(conn, args.table, predicate)
        selector, selector_ms = timed(lambda: faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids)))
        for ef_search in args.ef_search:
            recalls = []
            vector_ms = []
            returned = []
            for local_no, query in enumerate(queries):
                query_no = query_nos[local_no]
                truth = truth_rows[(filter_name, query_no)]
                truth_ids = [int(x) for x in str(truth["exact_filtered_topk_ids"]).split(",") if x]
                pre_ids, pre_ms = hnsw_search(index, query, args.k, ef_search, selector=selector)
                recalls.append(recall_at_k(pre_ids, truth_ids, args.k))
                vector_ms.append(pre_ms)
                returned.append(len(pre_ids))
            out_rows.append(
                {
                    "filter_name": filter_name,
                    "target_rate": target_rate,
                    "predicate": predicate,
                    "sql_rows": len(ids),
                    "actual_rate": len(ids) / args.rows,
                    "ef_search": ef_search,
                    "sql_select_id_total_ms": id_total_ms,
                    "selector_build_ms": selector_ms,
                    "vector_search_ms_mean": statistics.mean(vector_ms),
                    "vector_search_ms_p50": statistics.median(vector_ms),
                    "vector_search_ms_p95": percentile(vector_ms, 95),
                    "recall_mean": statistics.mean(recalls),
                    "returned_mean": statistics.mean(returned),
                }
            )
        print(f"swept {filter_name}", flush=True)

    args.ef_out.parent.mkdir(parents=True, exist_ok=True)
    with args.ef_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.ef_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000])
    parser.add_argument("--sql-out", type=Path, default=Path("results/hybrid_vector_db/duckdb_sql_decompose_20260606.csv"))
    parser.add_argument("--ef-out", type=Path, default=Path("results/hybrid_vector_db/faiss_allow_list_ef_sweep_20260606.csv"))
    parser.add_argument("--skip-sql", action="store_true")
    parser.add_argument("--skip-ef", action="store_true")
    args = parser.parse_args()

    if not args.skip_sql:
        run_sql_decompose(args)
    if not args.skip_ef:
        run_ef_sweep(args)


if __name__ == "__main__":
    main()
