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


def fetch_duckdb_ids_arrow(conn, table: str, predicate: str) -> dict[str, object]:
    query = f"SELECT id FROM {table} WHERE {predicate}"
    cursor, execute_ms = timed(lambda: conn.execute(query))
    arrow_table, fetch_arrow_ms = timed(lambda: cursor.to_arrow_table())

    def to_numpy() -> np.ndarray:
        ids = arrow_table.column("id").combine_chunks().to_numpy(zero_copy_only=False)
        return np.asarray(ids, dtype=np.int64)

    ids, arrow_to_numpy_ms = timed(to_numpy)
    return {
        "ids": ids,
        "execute_ms": execute_ms,
        "fetch_arrow_ms": fetch_arrow_ms,
        "read_ms": execute_ms + fetch_arrow_ms,
        "arrow_to_numpy_ms": arrow_to_numpy_ms,
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


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((int(row["ef_search"]), str(row["filter_name"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    summary_rows: list[dict[str, object]] = []
    for (ef_search, filter_name), items in sorted(groups.items(), key=lambda item: (item[0][0], order[item[0][1]])):
        summary_rows.append(
            {
                "ef_search": ef_search,
                "filter_name": filter_name,
                "target_rate": items[0]["target_rate"],
                "predicate": items[0]["predicate"],
                "actual_selectivity": items[0]["actual_selectivity"],
                "sql_rows": items[0]["sql_rows"],
                "duckdb_execute_ms": items[0]["duckdb_execute_ms"],
                "duckdb_fetch_arrow_ms": items[0]["duckdb_fetch_arrow_ms"],
                "duckdb_read_ms": items[0]["duckdb_read_ms"],
                "arrow_to_numpy_ms": items[0]["arrow_to_numpy_ms"],
                "membership_mask_build_ms": items[0]["membership_mask_build_ms"],
                "vector_latency_ms": statistics.mean(float(row["vector_latency_ms"]) for row in items),
                "membership_filter_ms": statistics.mean(float(row["membership_filter_ms"]) for row in items),
                "rerank_latency_ms": statistics.mean(float(row["rerank_latency_ms"]) for row in items),
                "latency_read_only_ms": statistics.mean(float(row["latency_read_only_ms"]) for row in items),
                "latency_end_to_end_ms": statistics.mean(float(row["latency_end_to_end_ms"]) for row in items),
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
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/faiss_duckdb_parallel_breakdown.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--vector-topn", type=int, default=50_000)
    parser.add_argument("--ef-search-values", type=int, nargs="+", default=[500, 1000, 1500])
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
    for filter_name, target_rate, predicate in ATTR_FILTERS:
        sql_parts = fetch_duckdb_ids_arrow(conn, args.table, predicate)
        ids = sql_parts["ids"]
        mask, mask_build_ms = build_membership_mask(ids, rows)
        actual_selectivity = len(ids) / total_rows
        sql_prep_ms = float(sql_parts["read_ms"]) + float(sql_parts["arrow_to_numpy_ms"]) + mask_build_ms
        print(
            f"filter={filter_name} actual={actual_selectivity:.4f} rows={len(ids)} "
            f"duckdb_read_ms={float(sql_parts['read_ms']):.2f} "
            f"arrow_to_numpy_ms={float(sql_parts['arrow_to_numpy_ms']):.2f} "
            f"mask_build_ms={mask_build_ms:.2f}",
            flush=True,
        )

        for ef_search in args.ef_search_values:
            for local_no, query in enumerate(queries):
                query_no = query_nos[local_no]
                truth = truth_rows[(filter_name, query_no)]
                truth_ids = [int(x) for x in str(truth["exact_filtered_topk_ids"]).split(",") if x]

                vec_ids, vec_ms = hnsw_search(index, query, args.vector_topn, ef_search)
                intersection, membership_filter_ms = timed(lambda: vec_ids[mask[vec_ids]])
                reranked_ids, rerank_ms = exact_rerank(xb, query, intersection, args.k)

                common = {
                    "query_no": query_no,
                    "query_id": int(query_ids[local_no]),
                    "filter_name": filter_name,
                    "target_rate": target_rate,
                    "predicate": predicate,
                    "actual_selectivity": actual_selectivity,
                    "sql_rows": len(ids),
                    "ef_search": ef_search,
                    "vector_topn": args.vector_topn,
                    "duckdb_execute_ms": sql_parts["execute_ms"],
                    "duckdb_fetch_arrow_ms": sql_parts["fetch_arrow_ms"],
                    "duckdb_read_ms": sql_parts["read_ms"],
                    "arrow_to_numpy_ms": sql_parts["arrow_to_numpy_ms"],
                    "membership_mask_build_ms": mask_build_ms,
                }
                out_rows.append(
                    {
                        **common,
                        "method": "parallel_join",
                        "vector_latency_ms": vec_ms,
                        "membership_filter_ms": membership_filter_ms,
                        "rerank_latency_ms": rerank_ms,
                        "latency_read_only_ms": max(float(sql_parts["read_ms"]), vec_ms) + membership_filter_ms + rerank_ms,
                        "latency_end_to_end_ms": max(sql_prep_ms, vec_ms) + membership_filter_ms + rerank_ms,
                        "intersection": len(intersection),
                        "returned": len(reranked_ids),
                        "recall_at_10_exact_filtered": recall_at_k(reranked_ids, truth_ids, args.k),
                    }
                )
            print(f"  ef_search={ef_search} searched {len(queries)} queries", flush=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.out} rows={len(out_rows)}", flush=True)
    summarize(out_rows, args.out)


if __name__ == "__main__":
    main()
