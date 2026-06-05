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
    return value, (time.perf_counter() - t0) * 1000


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


def ensure_duckdb_table(conn, db_path: Path, csv_path: Path, table: str) -> None:
    conn.execute(
        """
        SELECT count(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table],
    )
    exists = int(conn.fetchone()[0]) > 0
    if exists:
        conn.execute(f"SELECT count(*) FROM {table}")
        print(f"duckdb table exists rows={int(conn.fetchone()[0])}", flush=True)
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"loading CSV into DuckDB table={table} csv={csv_path}", flush=True)
    _, elapsed_ms = timed(
        lambda: conn.execute(
            f"""
            CREATE TABLE {table} AS
            SELECT
                id::BIGINT AS id,
                rating::DOUBLE AS rating,
                verified_purchase::BOOLEAN AS verified_purchase,
                helpful_vote::INTEGER AS helpful_vote,
                review_text_len::INTEGER AS review_text_len,
                main_category::VARCHAR AS main_category,
                price::DOUBLE AS price,
                has_price::BOOLEAN AS has_price,
                item_rating_number::INTEGER AS item_rating_number
            FROM read_csv_auto(?, header = true)
            """,
            [str(csv_path)],
        )
    )
    print(f"loaded DuckDB table in {elapsed_ms:.2f} ms", flush=True)


def duckdb_ids(conn, table: str, predicate: str) -> tuple[np.ndarray, float]:
    def run() -> np.ndarray:
        result = conn.execute(f"SELECT id FROM {table} WHERE {predicate}").fetchnumpy()
        return np.asarray(result["id"], dtype=np.int64)

    return timed(run)


def hnsw_search(index, query: np.ndarray, topn: int, ef_search: int, selector=None) -> tuple[list[int], float]:
    import faiss

    def run() -> list[int]:
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(ef_search)
        if selector is not None:
            params.sel = selector
        _, ids = index.search(query.reshape(1, -1), topn, params=params)
        return [int(x) for x in ids[0] if x >= 0]

    return timed(run)


def exact_rerank(xb: np.memmap, query: np.ndarray, candidate_ids: list[int], k: int) -> tuple[list[int], float]:
    def run() -> list[int]:
        if not candidate_ids:
            return []
        ids = np.asarray(candidate_ids, dtype=np.int64)
        vecs = np.asarray(xb[ids], dtype=np.float32)
        query_norm = float(np.dot(query, query))
        dists = np.einsum("ij,ij->i", vecs, vecs) + query_norm - 2.0 * (vecs @ query)
        take = min(k, len(ids))
        pos = np.argpartition(dists, take - 1)[:take]
        order = np.argsort(dists[pos])
        return [int(x) for x in ids[pos][order]]

    return timed(run)


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["method"])), []).append(row)
    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    summary_rows: list[dict[str, object]] = []
    for (name, method), items in sorted(groups.items(), key=lambda item: (order[item[0][0]], item[0][1])):
        summary_rows.append(
            {
                "filter_name": name,
                "target_rate": items[0]["target_rate"],
                "predicate": items[0]["predicate"],
                "method": method,
                "actual_predicate_selectivity": items[0]["actual_predicate_selectivity"],
                "sql_rows": items[0]["sql_rows"],
                "sql_latency_ms": statistics.mean(float(row["sql_latency_ms"]) for row in items),
                "vector_latency_ms": statistics.mean(float(row["vector_latency_ms"]) for row in items),
                "join_rerank_latency_ms": statistics.mean(float(row["join_rerank_latency_ms"]) for row in items),
                "latency_mean_ms": statistics.mean(float(row["latency_ms"]) for row in items),
                "recall_mean": statistics.mean(float(row["recall_at_10_exact_filtered"]) for row in items),
                "intersection_mean": statistics.mean(float(row["intersection"]) for row in items),
                "returned_mean": statistics.mean(float(row["returned"]) for row in items),
            }
        )
    summary_out = out.with_name(out.stem + "_summary.csv")
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"))
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/faiss_duckdb_attribute_filter_10m.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--vector-topn", type=int, default=50_000)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import duckdb
    import faiss

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.duckdb.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.duckdb))
    conn.execute(f"PRAGMA threads={int(args.threads)}")
    ensure_duckdb_table(conn, args.duckdb, args.csv, args.table)
    conn.execute(f"SELECT count(*) FROM {args.table}")
    total_rows = int(conn.fetchone()[0])

    xb, rows, _ = read_fbin_memmap(args.fbin, args.rows)
    index = faiss.read_index(str(args.index))
    truth_rows = load_truth_rows(args.truth_csv)
    query_by_no = {int(row["query_no"]): int(row["query_id"]) for row in truth_rows.values()}
    query_nos = sorted(query_by_no)[: args.queries]
    query_ids = np.asarray([query_by_no[q] for q in query_nos], dtype=np.int64)
    queries = np.ascontiguousarray(xb[query_ids], dtype=np.float32)

    out_rows: list[dict[str, object]] = []
    for filter_name, target_rate, predicate in ATTR_FILTERS:
        ids, sql_ms = duckdb_ids(conn, args.table, predicate)
        id_set = set(int(x) for x in ids)
        actual_selectivity = len(ids) / total_rows
        selector = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))
        print(
            f"filter={filter_name} target={target_rate} actual={actual_selectivity:.4f} "
            f"sql_rows={len(ids)} duckdb_sql_ms={sql_ms:.2f}",
            flush=True,
        )
        for local_no, query in enumerate(queries):
            query_no = query_nos[local_no]
            truth = truth_rows[(filter_name, query_no)]
            truth_ids = [int(x) for x in str(truth["exact_filtered_topk_ids"]).split(",") if x]

            vec_ids, vec_ms = hnsw_search(index, query, args.vector_topn, args.ef_search)
            intersection = [row_id for row_id in vec_ids if row_id in id_set]
            parallel_ids, rerank_ms = exact_rerank(xb, query, intersection, args.k)
            pre_ids, pre_vec_ms = hnsw_search(index, query, args.k, args.ef_search, selector=selector)

            rows_common = {
                "query_no": query_no,
                "query_id": int(query_ids[local_no]),
                "filter_name": filter_name,
                "target_rate": target_rate,
                "predicate": predicate,
                "actual_predicate_selectivity": actual_selectivity,
                "sql_rows": len(ids),
                "sql_latency_ms": sql_ms,
                "vector_topn": args.vector_topn,
                "ef_search": args.ef_search,
            }
            out_rows.append(
                {
                    **rows_common,
                    "method": "parallel_join",
                    "vector_latency_ms": vec_ms,
                    "join_rerank_latency_ms": rerank_ms,
                    "latency_ms": max(sql_ms, vec_ms) + rerank_ms,
                    "intersection": len(intersection),
                    "returned": len(parallel_ids),
                    "recall_at_10_exact_filtered": recall_at_k(parallel_ids, truth_ids, args.k),
                }
            )
            out_rows.append(
                {
                    **rows_common,
                    "method": "pre_filter_allow_list",
                    "vector_latency_ms": pre_vec_ms,
                    "join_rerank_latency_ms": 0.0,
                    "latency_ms": sql_ms + pre_vec_ms,
                    "intersection": len(pre_ids),
                    "returned": len(pre_ids),
                    "recall_at_10_exact_filtered": recall_at_k(pre_ids, truth_ids, args.k),
                }
            )
        print(f"  searched {len(queries)} queries", flush=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.out} rows={len(out_rows)}", flush=True)
    summarize(out_rows, args.out)


if __name__ == "__main__":
    main()
