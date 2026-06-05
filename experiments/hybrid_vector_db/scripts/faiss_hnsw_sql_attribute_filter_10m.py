from __future__ import annotations

import argparse
import csv
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common_pg import pg_config_from_env, require_psycopg


ATTR_FILTERS: list[tuple[str, str, str]] = [
    ("popular_ge1000", "50%", "item_rating_number >= 1000"),
    ("price_10_to_20", "20%", "has_price AND price > 10 AND price <= 20"),
    ("rating5_price_le10", "10%", "has_price AND price <= 10 AND rating = 5"),
    ("long_review_ge500", "5%", "review_text_len >= 500"),
    ("grocery_rating5", "2%", "main_category = 'Grocery' AND rating = 5"),
    ("grocery_helpful", "1%", "main_category = 'Grocery' AND helpful_vote >= 1"),
    ("helpful_ge20", "0.5%", "helpful_vote >= 20"),
    ("grocery_long500", "0.2%", "main_category = 'Grocery' AND review_text_len >= 500"),
]


@dataclass
class QueryResult:
    ids: list[int]
    latency_ms: float
    candidates: int = 0


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


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ids[:k]) & set(truth[:k])) / min(k, len(truth))


def sql_filter_ids(cur, table: str, predicate: str) -> tuple[np.ndarray, float]:
    def run() -> list[int]:
        cur.execute(f"SELECT id FROM {table} WHERE {predicate} ORDER BY id")
        return [int(row[0]) for row in cur.fetchall()]

    ids, elapsed_ms = timed(run)
    return np.asarray(ids, dtype=np.int64), elapsed_ms


def exact_topk(xb: np.memmap, query: np.ndarray, ids: np.ndarray, k: int, chunk_size: int) -> QueryResult:
    def run() -> list[int]:
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

    ids_out, elapsed = timed(run)
    return QueryResult(ids_out, elapsed, len(ids_out))


def hnsw_search(index, query: np.ndarray, k: int, ef_search: int, selector=None) -> QueryResult:
    import faiss

    def run() -> list[int]:
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(ef_search)
        if selector is not None:
            params.sel = selector
        _, ids = index.search(query.reshape(1, -1), k, params=params)
        return [int(x) for x in ids[0] if x >= 0]

    ids_out, elapsed = timed(run)
    return QueryResult(ids_out, elapsed, len(ids_out))


def validate_sql(cur, table: str, candidates: list[int], predicate: str, k: int) -> QueryResult:
    def run() -> list[int]:
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            WHERE id = ANY(%s) AND {predicate}
            """,
            (candidates,),
        )
        valid = {int(row[0]) for row in cur.fetchall()}
        return [row_id for row_id in candidates if row_id in valid][:k]

    ids_out, elapsed = timed(run)
    return QueryResult(ids_out, elapsed, len(candidates))


def load_truth_rows(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    truth: dict[tuple[str, int], dict[str, object]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "pre_filter_exact":
                continue
            truth[(row["filter_name"], int(row["query_no"]))] = row
    return truth


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["method"])), []).append(row)
    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    summary_rows = []
    for (name, method), items in sorted(groups.items(), key=lambda x: (order[x[0][0]], x[0][1])):
        latencies = [float(row["latency_ms"]) for row in items]
        recalls = [float(row["recall_at_10_exact_filtered"]) for row in items]
        returned = [float(row["returned"]) for row in items]
        candidates = [float(row["candidates"]) for row in items]
        summary_rows.append(
            {
                "filter_name": name,
                "target_rate": items[0]["target_rate"],
                "predicate": items[0]["predicate"],
                "actual_selectivity": statistics.median(float(row["actual_selectivity"]) for row in items),
                "method": method,
                "queries": len(items),
                "latency_mean_ms": statistics.mean(latencies),
                "recall_at_10_exact_filtered_mean": statistics.mean(recalls),
                "returned_mean": statistics.mean(returned),
                "candidates_mean": statistics.mean(candidates),
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
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--post-overfetch", type=int, default=1000)
    parser.add_argument("--post-ef-search", type=int, default=1000)
    parser.add_argument("--in-ef-search", type=int, default=1000)
    parser.add_argument("--exact-chunk-size", type=int, default=200_000)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--post-in-only", action="store_true")
    parser.add_argument("--filter-names", nargs="+")
    args = parser.parse_args()

    require_psycopg()
    import faiss
    import psycopg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    xb, rows, _ = read_fbin_memmap(args.fbin, args.rows)
    print(f"loading HNSW index {args.index}", flush=True)
    index = faiss.read_index(str(args.index))
    print(f"loaded index d={index.d} ntotal={index.ntotal}", flush=True)
    rows = min(rows, index.ntotal)

    truth_rows = load_truth_rows(args.truth_csv) if args.truth_csv else {}
    if truth_rows:
        query_by_no = {int(row["query_no"]): int(row["query_id"]) for row in truth_rows.values()}
        query_nos = sorted(query_by_no)[: args.queries]
        query_ids = np.asarray([query_by_no[q] for q in query_nos], dtype=np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        query_ids = rng.choice(rows, size=args.queries, replace=False)
        query_nos = list(range(args.queries))
    queries = np.ascontiguousarray(xb[query_ids], dtype=np.float32)

    rows_out: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {args.table}")
            sql_rows = int(cur.fetchone()[0])
            rows = min(rows, sql_rows)

            selected = set(args.filter_names or [])
            for filter_name, target_rate, predicate in ATTR_FILTERS:
                if selected and filter_name not in selected:
                    continue
                filter_ids, sql_ms = sql_filter_ids(cur, args.table, predicate)
                actual_selectivity = len(filter_ids) / rows
                selector = faiss.IDSelectorBatch(filter_ids.size, faiss.swig_ptr(filter_ids))
                print(
                    f"filter={filter_name} target={target_rate} actual_selectivity={actual_selectivity:.4f} "
                    f"matches={len(filter_ids)} sql_ids_ms={sql_ms:.2f}",
                    flush=True,
                )
                for local_query_no, query in enumerate(queries):
                    query_no = query_nos[local_query_no]
                    truth_row = truth_rows.get((filter_name, query_no))
                    if truth_row:
                        exact = QueryResult(
                            ids=[int(x) for x in str(truth_row["exact_filtered_topk_ids"]).split(",") if x],
                            latency_ms=float(truth_row["latency_ms"]),
                            candidates=int(float(truth_row["candidates"])),
                        )
                    else:
                        exact = exact_topk(xb, query, filter_ids, args.k, args.exact_chunk_size)
                    post_vec = hnsw_search(index, query, args.post_overfetch, args.post_ef_search)
                    post = validate_sql(cur, args.table, post_vec.ids, predicate, args.k)
                    post.latency_ms += post_vec.latency_ms
                    in_filter = hnsw_search(index, query, args.k, args.in_ef_search, selector)
                    method_results = [
                        ("pre_filter_exact", exact),
                        ("post_filtering", post),
                        ("in_filtering", in_filter),
                    ]
                    if args.post_in_only:
                        method_results = method_results[1:]
                    for method, result in method_results:
                        rows_out.append(
                            {
                                "query_no": query_no,
                                "query_id": int(query_ids[local_query_no]),
                                "filter_name": filter_name,
                                "target_rate": target_rate,
                                "predicate": predicate,
                                "actual_selectivity": actual_selectivity,
                                "method": method,
                                "k": args.k,
                                "post_overfetch": args.post_overfetch,
                                "post_ef_search": args.post_ef_search,
                                "in_ef_search": args.in_ef_search,
                                "latency_ms": result.latency_ms,
                                "recall_at_10_exact_filtered": recall_at_k(result.ids, exact.ids, args.k),
                                "returned": len(result.ids),
                                "candidates": result.candidates,
                                "result_ids": ",".join(str(x) for x in result.ids),
                                "exact_filtered_topk_ids": ",".join(str(x) for x in exact.ids),
                            }
                        )
                    if (local_query_no + 1) % 20 == 0:
                        print(f"  finished {local_query_no + 1}/{len(queries)} queries", flush=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {args.out} rows={len(rows_out)}", flush=True)
    summarize(rows_out, args.out)


if __name__ == "__main__":
    main()
