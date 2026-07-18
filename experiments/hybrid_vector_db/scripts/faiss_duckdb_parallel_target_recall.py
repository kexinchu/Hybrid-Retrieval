from __future__ import annotations

import argparse
import csv
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k  # noqa: E402


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


def fetch_duckdb_ids(conn, table: str, predicate: str) -> dict[str, object]:
    cursor, execute_ms = timed(lambda: conn.execute(f"SELECT id FROM {table} WHERE {predicate}"))
    result, fetch_numpy_ms = timed(lambda: cursor.fetchnumpy())
    ids, asarray_ms = timed(lambda: np.asarray(result["id"], dtype=np.int64))
    return {
        "ids": ids,
        "duckdb_execute_ms": execute_ms,
        "duckdb_fetch_numpy_ms": fetch_numpy_ms,
        "duckdb_asarray_ms": asarray_ms,
        "duckdb_export_ms": execute_ms + fetch_numpy_ms + asarray_ms,
    }


def build_membership_mask(ids: np.ndarray, total_rows: int) -> tuple[np.ndarray, float]:
    def run() -> np.ndarray:
        mask = np.zeros(total_rows, dtype=np.bool_)
        mask[ids] = True
        return mask

    return timed(run)


def hnsw_search(index, query: np.ndarray, topn: int, ef_search: int) -> tuple[np.ndarray, float]:
    import faiss

    def run() -> np.ndarray:
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(ef_search)
        _, ids = index.search(query.reshape(1, -1), topn, params=params)
        return np.asarray([int(x) for x in ids[0] if x >= 0], dtype=np.int64)

    return timed(run)


def exact_rerank(xb: np.memmap, query: np.ndarray, candidate_ids: np.ndarray, k: int) -> tuple[list[int], float]:
    def run() -> list[int]:
        if candidate_ids.size == 0:
            return []
        vecs = np.asarray(xb[candidate_ids], dtype=np.float32)
        query_norm = float(np.dot(query, query))
        dists = np.einsum("ij,ij->i", vecs, vecs) + query_norm - 2.0 * (vecs @ query)
        take = min(k, len(candidate_ids))
        pos = np.argpartition(dists, take - 1)[:take]
        order = np.argsort(dists[pos])
        return [int(x) for x in candidate_ids[pos][order]]

    return timed(run)


def summarize(rows: list[dict[str, object]], out: Path, target_recall: float) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), int(row["ef_search"]), int(row["vector_topn"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    summary_rows: list[dict[str, object]] = []
    for (filter_name, ef_search, vector_topn), items in sorted(
        groups.items(), key=lambda item: (order[item[0][0]], item[0][1], item[0][2])
    ):
        summary_rows.append(
            {
                "filter_name": filter_name,
                "target_rate": items[0]["target_rate"],
                "actual_selectivity": items[0]["actual_selectivity"],
                "sql_rows": items[0]["sql_rows"],
                "ef_search": ef_search,
                "vector_topn": vector_topn,
                "duckdb_export_ms": items[0]["duckdb_export_ms"],
                "membership_mask_build_ms": items[0]["membership_mask_build_ms"],
                "sql_full_export_plus_mask_ms": items[0]["sql_full_export_plus_mask_ms"],
                "vector_latency_ms": statistics.mean(float(row["vector_latency_ms"]) for row in items),
                "membership_filter_ms": statistics.mean(float(row["membership_filter_ms"]) for row in items),
                "rerank_latency_ms": statistics.mean(float(row["rerank_latency_ms"]) for row in items),
                "latency_end_to_end_ms": statistics.mean(float(row["latency_end_to_end_ms"]) for row in items),
                "recall_mean": statistics.mean(float(row["recall_at_10_exact_filtered"]) for row in items),
                "recall_abs_error_to_target": abs(
                    statistics.mean(float(row["recall_at_10_exact_filtered"]) for row in items) - target_recall
                ),
                "intersection_mean": statistics.mean(float(row["intersection"]) for row in items),
                "returned_mean": statistics.mean(float(row["returned"]) for row in items),
            }
        )

    summary_out = out.with_name(out.stem + "_summary.csv")
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    best_rows = []
    for filter_name in sorted({r["filter_name"] for r in summary_rows}, key=lambda name: order[name]):
        candidates = [r for r in summary_rows if r["filter_name"] == filter_name]
        best_rows.append(
            min(
                candidates,
                key=lambda r: (
                    float(r["recall_abs_error_to_target"]),
                    float(r["latency_end_to_end_ms"]),
                ),
            )
        )
    best_out = out.with_name(out.stem + "_target_best.csv")
    with best_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)
    print(f"wrote {summary_out}", flush=True)
    print(f"wrote {best_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/faiss_duckdb_parallel_target_recall.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.5)
    parser.add_argument("--vector-topn-values", type=int, nargs="+", default=[5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000])
    parser.add_argument("--ef-search-values", type=int, nargs="+", default=[1000])
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import duckdb
    import faiss

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.duckdb))
    conn.execute(f"PRAGMA threads={int(args.threads)}")
    conn.execute(f"SELECT count(*) FROM {args.table}")
    total_rows = int(conn.fetchone()[0])

    xb, rows, _ = read_fbin_memmap(args.fbin, args.rows)
    rows = min(rows, total_rows)
    index = faiss.read_index(str(args.index))
    truth_rows = load_truth_rows(args.truth_csv)
    query_by_no = {int(row["query_no"]): int(row["query_id"]) for row in truth_rows.values()}
    query_nos = sorted(query_by_no)[: args.queries]
    query_ids = np.asarray([query_by_no[q] for q in query_nos], dtype=np.int64)
    queries = np.ascontiguousarray(xb[query_ids], dtype=np.float32)

    out_rows: list[dict[str, object]] = []
    selected_filters = set(args.filter_names or [])
    for filter_name, target_rate, predicate in ATTR_FILTERS:
        if selected_filters and filter_name not in selected_filters:
            continue
        sql_parts = fetch_duckdb_ids(conn, args.table, predicate)
        ids = sql_parts["ids"]
        mask, mask_build_ms = build_membership_mask(ids, rows)
        actual_selectivity = len(ids) / total_rows
        sql_full_ms = float(sql_parts["duckdb_export_ms"]) + mask_build_ms
        print(
            f"filter={filter_name} actual={actual_selectivity:.4f} rows={len(ids)} "
            f"duckdb_export_ms={float(sql_parts['duckdb_export_ms']):.2f} mask_build_ms={mask_build_ms:.2f}",
            flush=True,
        )

        for ef_search in args.ef_search_values:
            for vector_topn in args.vector_topn_values:
                recalls = []
                for local_no, query in enumerate(queries):
                    query_no = query_nos[local_no]
                    truth = truth_rows[(filter_name, query_no)]
                    truth_ids = [int(x) for x in str(truth["exact_filtered_topk_ids"]).split(",") if x]

                    vec_ids, vec_ms = hnsw_search(index, query, vector_topn, ef_search)
                    intersection, membership_filter_ms = timed(lambda: vec_ids[mask[vec_ids]])
                    reranked_ids, rerank_ms = exact_rerank(xb, query, intersection, args.k)
                    recall = recall_at_k(reranked_ids, truth_ids, args.k)
                    recalls.append(recall)
                    out_rows.append(
                        {
                            "query_no": query_no,
                            "query_id": int(query_ids[local_no]),
                            "filter_name": filter_name,
                            "target_rate": target_rate,
                            "predicate": predicate,
                            "actual_selectivity": actual_selectivity,
                            "sql_rows": len(ids),
                            "ef_search": ef_search,
                            "vector_topn": vector_topn,
                            "duckdb_execute_ms": sql_parts["duckdb_execute_ms"],
                            "duckdb_fetch_numpy_ms": sql_parts["duckdb_fetch_numpy_ms"],
                            "duckdb_asarray_ms": sql_parts["duckdb_asarray_ms"],
                            "duckdb_export_ms": sql_parts["duckdb_export_ms"],
                            "membership_mask_build_ms": mask_build_ms,
                            "sql_full_export_plus_mask_ms": sql_full_ms,
                            "vector_latency_ms": vec_ms,
                            "membership_filter_ms": membership_filter_ms,
                            "rerank_latency_ms": rerank_ms,
                            "latency_end_to_end_ms": max(sql_full_ms, vec_ms) + membership_filter_ms + rerank_ms,
                            "intersection": len(intersection),
                            "returned": len(reranked_ids),
                            "recall_at_10_exact_filtered": recall,
                        }
                    )
                print(
                    f"  ef={ef_search} topn={vector_topn} recall={statistics.mean(recalls):.3f}",
                    flush=True,
                )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.out} rows={len(out_rows)}", flush=True)
    summarize(out_rows, args.out, args.target_recall)


if __name__ == "__main__":
    main()
