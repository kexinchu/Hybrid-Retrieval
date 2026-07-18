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


def fetch_duckdb_ids(conn, table: str, predicate: str) -> tuple[np.ndarray, dict[str, float]]:
    cursor, execute_ms = timed(lambda: conn.execute(f"SELECT id FROM {table} WHERE {predicate}"))
    result, fetch_numpy_ms = timed(lambda: cursor.fetchnumpy())
    ids, asarray_ms = timed(lambda: np.asarray(result["id"], dtype=np.int64))
    return ids, {
        "duckdb_execute_ms": execute_ms,
        "duckdb_fetch_numpy_ms": fetch_numpy_ms,
        "duckdb_asarray_ms": asarray_ms,
        "duckdb_export_ms": execute_ms + fetch_numpy_ms + asarray_ms,
    }


def build_mask(ids: np.ndarray, total_rows: int) -> tuple[np.ndarray, float]:
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


def evaluate_config(
    index,
    xb: np.memmap,
    queries: np.ndarray,
    query_nos: list[int],
    truth_rows: dict[tuple[str, int], dict[str, object]],
    filter_name: str,
    mask: np.ndarray,
    topn: int,
    ef_search: int,
    k: int,
) -> dict[str, float]:
    recalls: list[float] = []
    vector_ms: list[float] = []
    membership_ms: list[float] = []
    rerank_ms: list[float] = []
    intersections: list[int] = []
    returned: list[int] = []
    for local_no, query in enumerate(queries):
        query_no = query_nos[local_no]
        truth = truth_rows[(filter_name, query_no)]
        truth_ids = [int(x) for x in str(truth["exact_filtered_topk_ids"]).split(",") if x]
        vec_ids, vec_time = hnsw_search(index, query, topn, ef_search)
        intersection, member_time = timed(lambda: vec_ids[mask[vec_ids]])
        reranked_ids, rerank_time = exact_rerank(xb, query, intersection, k)
        recalls.append(recall_at_k(reranked_ids, truth_ids, k))
        vector_ms.append(vec_time)
        membership_ms.append(member_time)
        rerank_ms.append(rerank_time)
        intersections.append(len(intersection))
        returned.append(len(reranked_ids))
    return {
        "recall": statistics.mean(recalls),
        "vector_ms": statistics.mean(vector_ms),
        "membership_ms": statistics.mean(membership_ms),
        "rerank_ms": statistics.mean(rerank_ms),
        "intersection": statistics.mean(intersections),
        "returned": statistics.mean(returned),
    }


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["filter_name"]), int(row["topn"]), int(row["ef_search"]), int(row["repeat"])),
            [],
        ).append(row)

    summary_rows: list[dict[str, object]] = []
    for (filter_name, topn, ef_search, repeat), items in groups.items():
        sql_export_ms = float(items[0]["duckdb_export_ms"])
        mask_build_ms = float(items[0]["mask_build_ms"])
        vector_ms = statistics.mean(float(r["vector_ms"]) for r in items)
        membership_ms = statistics.mean(float(r["membership_ms"]) for r in items)
        rerank_ms = statistics.mean(float(r["rerank_ms"]) for r in items)
        summary_rows.append(
            {
                "filter": items[0]["filter"],
                "filter_name": filter_name,
                "actual_selectivity": items[0]["actual_selectivity"],
                "sql_rows": items[0]["sql_rows"],
                "topn": topn,
                "ef_search": ef_search,
                "repeat": repeat,
                "duckdb_export_ms": sql_export_ms,
                "mask_build_ms": mask_build_ms,
                "vector_ms": vector_ms,
                "membership_ms": membership_ms,
                "rerank_ms": rerank_ms,
                "latency_end_to_end_ms": max(sql_export_ms + mask_build_ms, vector_ms) + membership_ms + rerank_ms,
                "recall": statistics.mean(float(r["recall"]) for r in items),
                "intersection": statistics.mean(float(r["intersection"]) for r in items),
                "returned": statistics.mean(float(r["returned"]) for r in items),
            }
        )

    repeat_groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in summary_rows:
        repeat_groups.setdefault((str(row["filter_name"]), int(row["topn"]), int(row["ef_search"])), []).append(row)
    final_rows: list[dict[str, object]] = []
    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    for (filter_name, topn, ef_search), items in sorted(
        repeat_groups.items(), key=lambda item: (order[item[0][0]], item[0][1], item[0][2])
    ):
        final_rows.append(
            {
                "filter": items[0]["filter"],
                "filter_name": filter_name,
                "actual_selectivity": items[0]["actual_selectivity"],
                "sql_rows": items[0]["sql_rows"],
                "topn": topn,
                "ef_search": ef_search,
                "repeats": len(items),
                "duckdb_export_ms": statistics.mean(float(r["duckdb_export_ms"]) for r in items),
                "mask_build_ms": statistics.mean(float(r["mask_build_ms"]) for r in items),
                "vector_ms": statistics.mean(float(r["vector_ms"]) for r in items),
                "membership_ms": statistics.mean(float(r["membership_ms"]) for r in items),
                "rerank_ms": statistics.mean(float(r["rerank_ms"]) for r in items),
                "latency_end_to_end_ms": statistics.mean(float(r["latency_end_to_end_ms"]) for r in items),
                "recall": statistics.mean(float(r["recall"]) for r in items),
                "intersection": statistics.mean(float(r["intersection"]) for r in items),
                "returned": statistics.mean(float(r["returned"]) for r in items),
            }
        )

    summary_out = out.with_name(out.stem + "_summary.csv")
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"wrote {summary_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/faiss_duckdb_target_recall_sweep.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.6)
    parser.add_argument("--topn-values", type=int, nargs="+", default=[5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 500_000])
    parser.add_argument("--ef-search-values", type=int, nargs="+", default=[200, 500, 1000, 1500, 2000])
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import duckdb
    import faiss

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.duckdb), read_only=True)
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

    selected: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    selected_filters = set(args.filter_names or [])
    for filter_name, target_rate, predicate in ATTR_FILTERS:
        if selected_filters and filter_name not in selected_filters:
            continue
        ids, sql_timings = fetch_duckdb_ids(conn, args.table, predicate)
        mask, mask_build_ms = build_mask(ids, rows)
        actual_selectivity = len(ids) / total_rows
        print(
            f"filter={filter_name} rate={target_rate} rows={len(ids)} "
            f"sql_export_ms={sql_timings['duckdb_export_ms']:.2f} mask_ms={mask_build_ms:.2f}",
            flush=True,
        )
        candidates: list[dict[str, object]] = []
        for topn in args.topn_values:
            for ef_search in args.ef_search_values:
                result = evaluate_config(index, xb, queries, query_nos, truth_rows, filter_name, mask, topn, ef_search, args.k)
                row = {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "predicate": predicate,
                    "actual_selectivity": actual_selectivity,
                    "sql_rows": len(ids),
                    "topn": topn,
                    "ef_search": ef_search,
                    "duckdb_export_ms": sql_timings["duckdb_export_ms"],
                    "mask_build_ms": mask_build_ms,
                    **result,
                }
                candidates.append(row)
                print(
                    f"  topn={topn} ef={ef_search} recall={result['recall']:.3f} "
                    f"vec_ms={result['vector_ms']:.2f}",
                    flush=True,
                )
        above = [r for r in candidates if float(r["recall"]) >= args.target_recall]
        if above:
            chosen = min(
                above,
                key=lambda r: (
                    float(r["vector_ms"]) + float(r["membership_ms"]) + float(r["rerank_ms"]),
                    abs(float(r["recall"]) - args.target_recall),
                ),
            )
        else:
            chosen = max(candidates, key=lambda r: float(r["recall"]))
        selected.append(chosen)
        print(
            f"selected filter={filter_name} topn={chosen['topn']} ef={chosen['ef_search']} "
            f"recall={float(chosen['recall']):.3f}",
            flush=True,
        )

        for repeat in range(args.repeats):
            repeat_ids, repeat_sql_timings = fetch_duckdb_ids(conn, args.table, predicate)
            repeat_mask, repeat_mask_build_ms = build_mask(repeat_ids, rows)
            result = evaluate_config(
                index,
                xb,
                queries,
                query_nos,
                truth_rows,
                filter_name,
                repeat_mask,
                int(chosen["topn"]),
                int(chosen["ef_search"]),
                args.k,
            )
            all_rows.append(
                {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "predicate": predicate,
                    "actual_selectivity": len(repeat_ids) / total_rows,
                    "sql_rows": len(repeat_ids),
                    "topn": int(chosen["topn"]),
                    "ef_search": int(chosen["ef_search"]),
                    "repeat": repeat,
                    "duckdb_export_ms": repeat_sql_timings["duckdb_export_ms"],
                    "mask_build_ms": repeat_mask_build_ms,
                    "vector_ms": result["vector_ms"],
                    "membership_ms": result["membership_ms"],
                    "rerank_ms": result["rerank_ms"],
                    "recall": result["recall"],
                    "intersection": result["intersection"],
                    "returned": result["returned"],
                }
            )
            print(
                f"  repeat={repeat} recall={result['recall']:.3f} "
                f"sql_ms={repeat_sql_timings['duckdb_export_ms']:.2f} "
                f"mask_ms={repeat_mask_build_ms:.2f} vector_ms={result['vector_ms']:.2f}",
                flush=True,
            )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {args.out} rows={len(all_rows)}", flush=True)
    summarize(all_rows, args.out)


if __name__ == "__main__":
    main()
