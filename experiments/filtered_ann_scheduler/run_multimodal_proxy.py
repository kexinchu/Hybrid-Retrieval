from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class MMWorkload:
    name: str
    text_vecs: np.ndarray
    image_vecs: np.ndarray
    attrs: np.ndarray
    text_queries: np.ndarray
    image_queries: np.ndarray
    query_weights: np.ndarray
    ranges: list[tuple[float, float]]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_hnsw(vectors: np.ndarray, m: int, ef_construction: int) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def make_workload(
    *,
    n: int,
    dim: int,
    queries: int,
    selectivity: float,
    scenario: str,
    seed: int,
) -> MMWorkload:
    rng = np.random.default_rng(seed)
    clusters = 32
    text_centers = l2_normalize(rng.normal(size=(clusters, dim)).astype("float32"))
    image_centers = l2_normalize(rng.normal(size=(clusters, dim)).astype("float32"))
    text_cluster = rng.integers(0, clusters, size=n)
    image_cluster = rng.integers(0, clusters, size=n)

    text_vecs = text_centers[text_cluster] + 0.10 * rng.normal(size=(n, dim)).astype("float32")
    image_vecs = image_centers[image_cluster] + 0.10 * rng.normal(size=(n, dim)).astype("float32")
    text_vecs = l2_normalize(text_vecs).astype("float32")
    image_vecs = l2_normalize(image_vecs).astype("float32")

    if scenario == "text_filter_aligned":
        attrs = (text_cluster + 0.25 * rng.random(n)) / clusters
    elif scenario == "image_filter_aligned":
        attrs = (image_cluster + 0.25 * rng.random(n)) / clusters
    elif scenario == "cross_modal_conflict":
        attrs = ((text_cluster + clusters - image_cluster) % clusters + 0.25 * rng.random(n)) / clusters
    elif scenario == "random_filter":
        attrs = rng.random(n)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    attrs = np.clip(attrs, 0.0, 1.0).astype("float32")

    seed_ids = rng.integers(0, n, size=queries)
    text_queries = text_vecs[seed_ids] + 0.08 * rng.normal(size=(queries, dim)).astype("float32")
    image_queries = image_vecs[seed_ids] + 0.08 * rng.normal(size=(queries, dim)).astype("float32")
    text_queries = l2_normalize(text_queries).astype("float32")
    image_queries = l2_normalize(image_queries).astype("float32")

    modes = rng.choice(["text", "image", "both"], size=queries, p=[0.35, 0.35, 0.30])
    weights = np.zeros((queries, 2), dtype="float32")
    weights[modes == "text"] = [0.85, 0.15]
    weights[modes == "image"] = [0.15, 0.85]
    weights[modes == "both"] = [0.50, 0.50]

    ranges: list[tuple[float, float]] = []
    for sid in seed_ids:
        center = float(attrs[sid])
        lo = min(max(center - selectivity / 2, 0.0), 1.0 - selectivity)
        ranges.append((lo, lo + selectivity))

    return MMWorkload(
        name=f"{scenario}_sel{selectivity:g}",
        text_vecs=text_vecs,
        image_vecs=image_vecs,
        attrs=attrs,
        text_queries=text_queries,
        image_queries=image_queries,
        query_weights=weights,
        ranges=ranges,
    )


def exact_rank(
    workload: MMWorkload,
    qi: int,
    candidate_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    if len(candidate_ids) == 0:
        return candidate_ids.astype(np.int64)
    wt, wi = workload.query_weights[qi]
    text_scores = workload.text_vecs[candidate_ids] @ workload.text_queries[qi]
    image_scores = workload.image_vecs[candidate_ids] @ workload.image_queries[qi]
    scores = wt * text_scores + wi * image_scores
    take = min(k, len(candidate_ids))
    local = np.argpartition(-scores, take - 1)[:take]
    local = local[np.argsort(-scores[local])]
    return candidate_ids[local].astype(np.int64)


def ground_truth(workload: MMWorkload, qi: int, k: int) -> np.ndarray:
    lo, hi = workload.ranges[qi]
    ids = np.flatnonzero((workload.attrs >= lo) & (workload.attrs <= hi))
    return exact_rank(workload, qi, ids, k)


def search_modality(
    index: faiss.IndexHNSWFlat,
    query: np.ndarray,
    budget: int,
    ef_search: int,
) -> np.ndarray:
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    index.hnsw.efSearch = ef_search
    _, ids = index.search(query.reshape(1, -1), budget)
    ids = ids[0]
    return ids[ids >= 0].astype(np.int64)


def run_strategy(
    workload: MMWorkload,
    text_index: faiss.IndexHNSWFlat,
    image_index: faiss.IndexHNSWFlat,
    qi: int,
    k: int,
    strategy: str,
    total_budget: int,
    probe_budget: int,
    ef_search: int,
) -> dict[str, object]:
    wt, wi = workload.query_weights[qi]
    t0 = time.perf_counter()

    if strategy == "text_only":
        text_budget, image_budget = total_budget, 0
    elif strategy == "image_only":
        text_budget, image_budget = 0, total_budget
    elif strategy == "balanced_union":
        text_budget, image_budget = total_budget // 2, total_budget - total_budget // 2
    elif strategy == "oracle_route":
        if wt > 0.7:
            text_budget, image_budget = total_budget, 0
        elif wi > 0.7:
            text_budget, image_budget = 0, total_budget
        else:
            text_budget, image_budget = total_budget // 2, total_budget - total_budget // 2
    elif strategy == "adaptive_probe":
        text_probe = search_modality(text_index, workload.text_queries[qi], probe_budget, ef_search)
        image_probe = search_modality(image_index, workload.image_queries[qi], probe_budget, ef_search)
        lo, hi = workload.ranges[qi]
        text_hits = np.mean((workload.attrs[text_probe] >= lo) & (workload.attrs[text_probe] <= hi)) if len(text_probe) else 0
        image_hits = np.mean((workload.attrs[image_probe] >= lo) & (workload.attrs[image_probe] <= hi)) if len(image_probe) else 0
        text_utility = float(text_hits * (wt + 0.10))
        image_utility = float(image_hits * (wi + 0.10))
        remaining = max(total_budget - 2 * probe_budget, 0)
        if abs(text_utility - image_utility) < 0.02:
            text_budget = probe_budget + remaining // 2
            image_budget = probe_budget + remaining - remaining // 2
        elif text_utility > image_utility:
            text_budget = probe_budget + remaining
            image_budget = probe_budget
        else:
            text_budget = probe_budget
            image_budget = probe_budget + remaining
    else:
        raise ValueError(strategy)

    text_ids = search_modality(text_index, workload.text_queries[qi], text_budget, ef_search)
    image_ids = search_modality(image_index, workload.image_queries[qi], image_budget, ef_search)
    candidates = np.unique(np.concatenate([text_ids, image_ids]))
    lo, hi = workload.ranges[qi]
    filtered = candidates[(workload.attrs[candidates] >= lo) & (workload.attrs[candidates] <= hi)]
    ranked = exact_rank(workload, qi, filtered, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    truth = ground_truth(workload, qi, k)
    recall = 1.0 if len(truth) == 0 else len(set(ranked).intersection(truth)) / min(k, len(truth))
    candidate_recall = 1.0 if len(truth) == 0 else len(set(filtered).intersection(truth)) / min(k, len(truth))
    return {
        "strategy": strategy,
        "recall_at_k": recall,
        "candidate_recall_at_k": candidate_recall,
        "returned": len(ranked),
        "failed_to_fill": int(len(ranked) < k),
        "latency_ms": latency_ms,
        "text_budget": text_budget,
        "image_budget": image_budget,
        "candidate_count": len(candidates),
        "filtered_candidate_count": len(filtered),
        "rerank_distance_evals": 2 * len(filtered),
        "query_text_weight": float(wt),
        "query_image_weight": float(wi),
    }


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--selectivities", type=float, nargs="+", default=[0.002, 0.01, 0.05])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["text_filter_aligned", "image_filter_aligned", "cross_modal_conflict", "random_filter"],
    )
    parser.add_argument("--total-budget", type=int, default=1000)
    parser.add_argument("--probe-budget", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--out", type=Path, default=Path("results/multimodal_proxy.csv"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = 20000
        args.queries = 24
        args.selectivities = [0.002, 0.01, 0.05]
        args.scenarios = ["text_filter_aligned", "image_filter_aligned", "random_filter"]
        args.total_budget = 800
        args.out = Path("results/multimodal_proxy_quick.csv")

    rows: list[dict[str, object]] = []
    wid = 0
    for sel in args.selectivities:
        for scenario in args.scenarios:
            workload = make_workload(
                n=args.n,
                dim=args.dim,
                queries=args.queries,
                selectivity=sel,
                scenario=scenario,
                seed=args.seed + wid,
            )
            wid += 1
            print(f"running {workload.name} n={args.n} q={args.queries}", flush=True)
            text_index = build_hnsw(workload.text_vecs, args.hnsw_m, args.ef_construction)
            image_index = build_hnsw(workload.image_vecs, args.hnsw_m, args.ef_construction)
            for qi in range(args.queries):
                lo, hi = workload.ranges[qi]
                actual_sel = float(np.mean((workload.attrs >= lo) & (workload.attrs <= hi)))
                for strategy in ["text_only", "image_only", "balanced_union", "oracle_route", "adaptive_probe"]:
                    row = run_strategy(
                        workload,
                        text_index,
                        image_index,
                        qi,
                        args.k,
                        strategy,
                        args.total_budget,
                        args.probe_budget,
                        args.ef_search,
                    )
                    row.update(
                        {
                            "workload": workload.name,
                            "scenario": scenario,
                            "target_selectivity": hi - lo,
                            "actual_selectivity": actual_sel,
                            "query_id": qi,
                        }
                    )
                    rows.append(row)

    write_csv(rows, args.out)
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()

