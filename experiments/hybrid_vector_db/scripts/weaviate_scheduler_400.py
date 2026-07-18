from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from weaviate_prefilter_10m import CLASS_NAME, FILTERS, json_to_graphql, load_truth


# Measured from flatSearchCutoff=0, ef=-1 dynamic run.
DEFAULT_STAGE_COSTS_MS = {
    "popular_ge1000": (6.90, 5.49),
    "price_10_to_20": (20.29, 8.71),
    "rating5_price_le10": (24.47, 8.49),
    "long_review_ge500": (7.53, 12.59),
    "grocery_rating5": (2.12, 12.32),
    "grocery_helpful": (8.06, 17.61),
    "helpful_ge20": (7.79, 13.80),
    "grocery_long500": (8.42, 20.36),
}


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def build_graphql(vector: np.ndarray, where: dict[str, object], k: int) -> str:
    return f"""
    {{
      Get {{
        {CLASS_NAME}(
          nearVector:{{vector:{json_to_graphql(np.asarray(vector, dtype=np.float32).tolist())}}}
          where:{json_to_graphql(where)}
          limit:{k}
        ) {{
          row_id
          _additional {{ distance }}
        }}
      }}
    }}
    """


def post_graphql(base_url: str, query: str, timeout: int) -> int:
    data = json.dumps({"query": query}, separators=(",", ":")).encode("utf-8")
    req = Request(
        base_url + "/v1/graphql",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if "errors" in body:
        raise RuntimeError(body["errors"])
    return len(body["data"]["Get"][CLASS_NAME])


def make_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[: args.queries_per_filter]
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    jobs: list[dict[str, object]] = []
    filter_defs = {name: (target, where) for target, name, where in FILTERS}
    for _, filter_name, _ in FILTERS:
        target, where = filter_defs[filter_name]
        for qno in query_nos:
            qid = query_by_no[qno]
            jobs.append(
                {
                    "filter": target,
                    "filter_name": filter_name,
                    "query_no": qno,
                    "query_id": qid,
                    "payload": build_graphql(xb[qid], where, args.k),
                }
            )
    return jobs


def random_order(jobs: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    ordered = list(jobs)
    random.Random(seed).shuffle(ordered)
    return ordered


def scheduled_order(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    by_filter: dict[str, list[dict[str, object]]] = {}
    for job in jobs:
        by_filter.setdefault(str(job["filter_name"]), []).append(job)

    # Pair high filter/vector ratios with low ratios. This is the original
    # coarse scheduler used for request-level ordering experiments.
    names = sorted(
        by_filter,
        key=lambda name: DEFAULT_STAGE_COSTS_MS[name][0] / DEFAULT_STAGE_COSTS_MS[name][1],
        reverse=True,
    )
    paired_names: list[str] = []
    left, right = 0, len(names) - 1
    while left <= right:
        paired_names.append(names[left])
        if left != right:
            paired_names.append(names[right])
        left += 1
        right -= 1

    ordered: list[dict[str, object]] = []
    remaining = True
    while remaining:
        remaining = False
        for name in paired_names:
            bucket = by_filter[name]
            if bucket:
                ordered.append(bucket.pop(0))
                remaining = True
    return ordered


def scheduled_alternating_stage_heavy_order(jobs: list[dict[str, object]], seed: int = 0) -> list[dict[str, object]]:
    by_filter: dict[str, list[dict[str, object]]] = {}
    for job in jobs:
        by_filter.setdefault(str(job["filter_name"]), []).append(job)

    rng = random.Random(seed)
    for bucket in by_filter.values():
        rng.shuffle(bucket)

    filter_heavy = sorted(
        [name for name, (filter_ms, vector_ms) in DEFAULT_STAGE_COSTS_MS.items() if filter_ms >= vector_ms],
        key=lambda name: DEFAULT_STAGE_COSTS_MS[name][0] - DEFAULT_STAGE_COSTS_MS[name][1],
        reverse=True,
    )
    vector_heavy = sorted(
        [name for name, (filter_ms, vector_ms) in DEFAULT_STAGE_COSTS_MS.items() if vector_ms > filter_ms],
        key=lambda name: DEFAULT_STAGE_COSTS_MS[name][1] - DEFAULT_STAGE_COSTS_MS[name][0],
        reverse=True,
    )

    heavy_cycle: list[str] = []
    max_len = max(len(filter_heavy), len(vector_heavy))
    for i in range(max_len):
        if i < len(filter_heavy):
            heavy_cycle.append(filter_heavy[i])
        if i < len(vector_heavy):
            heavy_cycle.append(vector_heavy[i])

    ordered: list[dict[str, object]] = []
    while any(by_filter[name] for name in heavy_cycle):
        for name in heavy_cycle:
            bucket = by_filter[name]
            if bucket:
                ordered.append(bucket.pop())
    return ordered


def run_once(args: argparse.Namespace, order_name: str, jobs: list[dict[str, object]], seed: int | None = None) -> dict[str, object]:
    latencies: list[float] = []
    counts_by_filter: dict[str, int] = {}
    start = time.perf_counter()

    def one(job: dict[str, object]) -> tuple[str, float, int]:
        t0 = time.perf_counter()
        returned = post_graphql(args.base_url, str(job["payload"]), args.timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        return str(job["filter_name"]), elapsed, returned

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for fut in as_completed(futures):
            filter_name, elapsed, returned = fut.result()
            latencies.append(elapsed)
            counts_by_filter[filter_name] = counts_by_filter.get(filter_name, 0) + 1
            if returned != args.k:
                raise RuntimeError(f"query returned {returned}, expected {args.k}")

    wall_ms = (time.perf_counter() - start) * 1000
    return {
        "order": order_name,
        "seed": "" if seed is None else seed,
        "concurrency": args.concurrency,
        "queries": len(jobs),
        "wall_ms": wall_ms,
        "throughput_qps": len(jobs) / (wall_ms / 1000),
        "request_latency_mean_ms": statistics.mean(latencies),
        "request_latency_p50_ms": statistics.median(latencies),
        "request_latency_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "counts_by_filter": json.dumps(counts_by_filter, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/weaviate_scheduler_400_flat0_dynamic_20260606.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--random-trials", type=int, default=5)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"

    jobs = make_jobs(args)
    if len(jobs) != len(FILTERS) * args.queries_per_filter:
        raise RuntimeError(f"unexpected job count={len(jobs)}")

    rows: list[dict[str, object]] = []
    scheduled = scheduled_order(jobs)
    rows.append(run_once(args, "scheduled_pair_extremes", scheduled))
    alternating = scheduled_alternating_stage_heavy_order(jobs)
    rows.append(run_once(args, "scheduled_alternating_stage_heavy", alternating))
    for seed in range(args.random_trials):
        rows.append(run_once(args, "random", random_order(jobs, seed), seed=seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}", flush=True)
    for row in rows:
        print(
            f"{row['order']} seed={row['seed']} wall_ms={float(row['wall_ms']):.2f} "
            f"qps={float(row['throughput_qps']):.2f} mean_ms={float(row['request_latency_mean_ms']):.2f} "
            f"p95_ms={float(row['request_latency_p95_ms']):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
