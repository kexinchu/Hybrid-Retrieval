from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


@dataclass(frozen=True)
class Workload:
    name: str
    vectors: np.ndarray
    attrs: np.ndarray
    queries: np.ndarray
    ranges: list[tuple[float, float]]


@dataclass
class SearchResult:
    ids: np.ndarray
    latency_ms: float
    predicate_evals: int
    exact_distance_evals: int
    ann_returned: int
    failed_to_fill: bool
    extra: str


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def make_workload(
    *,
    n: int,
    dim: int,
    queries: int,
    selectivity: float,
    correlation: str,
    seed: int,
) -> Workload:
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.normal(size=(32, dim)).astype("float32"))
    cluster_ids = rng.integers(0, len(centers), size=n)
    vectors = centers[cluster_ids] + 0.10 * rng.normal(size=(n, dim)).astype("float32")
    vectors = l2_normalize(vectors).astype("float32")

    if correlation == "random":
        attrs = rng.random(n).astype("float32")
    elif correlation == "correlated":
        attrs = (cluster_ids + 0.25 * rng.random(n)) / len(centers)
        attrs = np.clip(attrs, 0.0, 1.0).astype("float32")
    elif correlation == "anti_correlated":
        attrs = (1.0 - cluster_ids / len(centers)) + 0.25 * rng.random(n) / len(centers)
        attrs = np.clip(attrs, 0.0, 1.0).astype("float32")
    else:
        raise ValueError(f"unknown correlation: {correlation}")

    q_clusters = rng.integers(0, len(centers), size=queries)
    q = centers[q_clusters] + 0.10 * rng.normal(size=(queries, dim)).astype("float32")
    q = l2_normalize(q).astype("float32")

    width = selectivity
    ranges: list[tuple[float, float]] = []
    for qc in q_clusters:
        if correlation == "correlated":
            center = float(qc / len(centers))
        elif correlation == "anti_correlated":
            center = float(1.0 - qc / len(centers))
        else:
            center = float(rng.random())
        lo = min(max(center - width / 2, 0.0), 1.0 - width)
        ranges.append((lo, lo + width))

    return Workload(
        name=f"{correlation}_sel{selectivity:g}",
        vectors=vectors,
        attrs=attrs,
        queries=q,
        ranges=ranges,
    )


def build_hnsw(vectors: np.ndarray, m: int, ef_construction: int) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def filtered_ground_truth(
    vectors: np.ndarray,
    attrs: np.ndarray,
    q: np.ndarray,
    pred_range: tuple[float, float],
    k: int,
) -> np.ndarray:
    lo, hi = pred_range
    mask = (attrs >= lo) & (attrs <= hi)
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return np.empty(0, dtype=np.int64)
    scores = vectors[ids] @ q
    take = min(k, len(ids))
    local = np.argpartition(-scores, take - 1)[:take]
    local = local[np.argsort(-scores[local])]
    return ids[local].astype(np.int64)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    if len(truth) == 0:
        return 1.0
    return len(set(found[:k]).intersection(truth[:k])) / min(k, len(truth))


def pre_exact(
    workload: Workload,
    q: np.ndarray,
    pred_range: tuple[float, float],
    k: int,
) -> SearchResult:
    t0 = time.perf_counter()
    lo, hi = pred_range
    mask = (workload.attrs >= lo) & (workload.attrs <= hi)
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        ranked = np.empty(0, dtype=np.int64)
    else:
        scores = workload.vectors[ids] @ q
        take = min(k, len(ids))
        local = np.argpartition(-scores, take - 1)[:take]
        local = local[np.argsort(-scores[local])]
        ranked = ids[local].astype(np.int64)
    return SearchResult(
        ids=ranked,
        latency_ms=(time.perf_counter() - t0) * 1000,
        predicate_evals=len(workload.attrs),
        exact_distance_evals=len(ids),
        ann_returned=0,
        failed_to_fill=len(ranked) < min(k, len(ids)),
        extra="",
    )


def ann_then_filter(
    index: faiss.IndexHNSWFlat,
    attrs: np.ndarray,
    q: np.ndarray,
    pred_range: tuple[float, float],
    k: int,
    ann_k: int,
    ef_search: int,
) -> SearchResult:
    index.hnsw.efSearch = ef_search
    t0 = time.perf_counter()
    _, ids = index.search(q.reshape(1, -1), ann_k)
    ids = ids[0]
    ids = ids[ids >= 0]
    lo, hi = pred_range
    keep = ids[(attrs[ids] >= lo) & (attrs[ids] <= hi)]
    ranked = keep[:k].astype(np.int64)
    return SearchResult(
        ids=ranked,
        latency_ms=(time.perf_counter() - t0) * 1000,
        predicate_evals=len(ids),
        exact_distance_evals=0,
        ann_returned=len(ids),
        failed_to_fill=len(ranked) < k,
        extra=f"ann_k={ann_k};ef={ef_search}",
    )


def iterative_ann(
    index: faiss.IndexHNSWFlat,
    attrs: np.ndarray,
    q: np.ndarray,
    pred_range: tuple[float, float],
    k: int,
    start_k: int,
    max_k: int,
    ef_search: int,
) -> SearchResult:
    total_latency = 0.0
    total_predicates = 0
    last_ids = np.empty(0, dtype=np.int64)
    ann_k = start_k
    rounds = 0
    while True:
        rounds += 1
        res = ann_then_filter(index, attrs, q, pred_range, k, ann_k, ef_search)
        total_latency += res.latency_ms
        total_predicates += res.predicate_evals
        last_ids = res.ids
        if len(last_ids) >= k or ann_k >= max_k:
            return SearchResult(
                ids=last_ids,
                latency_ms=total_latency,
                predicate_evals=total_predicates,
                exact_distance_evals=0,
                ann_returned=ann_k,
                failed_to_fill=len(last_ids) < k,
                extra=f"rounds={rounds};final_ann_k={ann_k};ef={ef_search}",
            )
        ann_k = min(max_k, ann_k * 2)


def adaptive(
    workload: Workload,
    index: faiss.IndexHNSWFlat,
    q: np.ndarray,
    pred_range: tuple[float, float],
    k: int,
    start_k: int,
    max_k: int,
    ef_search: int,
    pre_threshold: float,
) -> SearchResult:
    lo, hi = pred_range
    estimated_selectivity = max(0.0, hi - lo)
    if estimated_selectivity <= pre_threshold:
        res = pre_exact(workload, q, pred_range, k)
        res.extra = f"choice=pre_exact;est_sel={estimated_selectivity:g}"
        return res
    res = iterative_ann(index, workload.attrs, q, pred_range, k, start_k, max_k, ef_search)
    res.extra = f"choice=iterative_ann;est_sel={estimated_selectivity:g};{res.extra}"
    return res


def run_one_workload(args: argparse.Namespace, workload: Workload) -> list[dict[str, object]]:
    index = build_hnsw(workload.vectors, args.hnsw_m, args.ef_construction)
    rows: list[dict[str, object]] = []
    for qi, (q, pred_range) in enumerate(zip(workload.queries, workload.ranges)):
        truth_t0 = time.perf_counter()
        truth = filtered_ground_truth(workload.vectors, workload.attrs, q, pred_range, args.k)
        truth_ms = (time.perf_counter() - truth_t0) * 1000
        actual_sel = int(((workload.attrs >= pred_range[0]) & (workload.attrs <= pred_range[1])).sum())

        strategies = {
            "pre_exact": pre_exact(workload, q, pred_range, args.k),
            "post_ann_10x": ann_then_filter(
                index, workload.attrs, q, pred_range, args.k, args.k * 10, args.ef_search
            ),
            "post_ann_100x": ann_then_filter(
                index, workload.attrs, q, pred_range, args.k, args.k * 100, args.ef_search
            ),
            "iterative_ann": iterative_ann(
                index,
                workload.attrs,
                q,
                pred_range,
                args.k,
                args.k * 10,
                args.max_ann_k,
                args.ef_search,
            ),
            "adaptive": adaptive(
                workload,
                index,
                q,
                pred_range,
                args.k,
                args.k * 10,
                args.max_ann_k,
                args.ef_search,
                args.pre_threshold,
            ),
        }
        for name, res in strategies.items():
            rows.append(
                {
                    "workload": workload.name,
                    "correlation": workload.name.split("_sel")[0],
                    "target_selectivity": pred_range[1] - pred_range[0],
                    "actual_selectivity": actual_sel / len(workload.attrs),
                    "query_id": qi,
                    "strategy": name,
                    "recall_at_k": recall_at_k(res.ids, truth, args.k),
                    "returned": len(res.ids),
                    "truth_size": len(truth),
                    "latency_ms": res.latency_ms,
                    "truth_latency_ms": truth_ms,
                    "predicate_evals": res.predicate_evals,
                    "exact_distance_evals": res.exact_distance_evals,
                    "ann_returned": res.ann_returned,
                    "failed_to_fill": int(res.failed_to_fill),
                    "extra": res.extra,
                }
            )
    return rows


def write_rows(rows: Iterable[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selectivities", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.2])
    parser.add_argument("--correlations", nargs="+", default=["random", "correlated", "anti_correlated"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--max-ann-k", type=int, default=10000)
    parser.add_argument("--pre-threshold", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=Path("results/filtered_ann_scheduler.csv"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = 20000
        args.queries = 24
        args.selectivities = [0.001, 0.01, 0.05, 0.2]
        args.correlations = ["random", "correlated"]
        args.max_ann_k = 5000
        args.out = Path("results/filtered_ann_scheduler_quick.csv")

    all_rows: list[dict[str, object]] = []
    for wi, (sel, corr) in enumerate((s, c) for s in args.selectivities for c in args.correlations):
        workload = make_workload(
            n=args.n,
            dim=args.dim,
            queries=args.queries,
            selectivity=sel,
            correlation=corr,
            seed=args.seed + wi,
        )
        print(f"running {workload.name} n={args.n} q={args.queries}", flush=True)
        all_rows.extend(run_one_workload(args, workload))
    write_rows(all_rows, args.out)
    print(f"wrote {args.out} rows={len(all_rows)}")


if __name__ == "__main__":
    main()

