from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from weaviate_prefilter_10m import CLASS_NAME, FILTERS, json_to_graphql, load_truth


PROM_SUM_RE = re.compile(
    r'^queries_filtered_vector_durations_ms_sum\{class_name="(?P<class>[^"]+)",operation="(?P<op>[^"]+)",shard_name="(?P<shard>[^"]+)"\} (?P<value>[-+0-9.eE]+)$'
)
PROM_COUNT_RE = re.compile(
    r'^queries_filtered_vector_durations_ms_count\{class_name="(?P<class>[^"]+)",operation="(?P<op>[^"]+)",shard_name="(?P<shard>[^"]+)"\} (?P<value>[-+0-9.eE]+)$'
)


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
        }}
      }}
    }}
    """


def post_graphql(base_url: str, query: str, timeout: int) -> int:
    data = json.dumps({"query": query}, separators=(",", ":")).encode("utf-8")
    req = Request(base_url + "/v1/graphql", data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if "errors" in body:
        raise RuntimeError(body["errors"])
    return len(body["data"]["Get"][CLASS_NAME])


def scrape_metrics(metrics_url: str) -> dict[str, tuple[float, float]]:
    with urlopen(metrics_url, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for line in text.splitlines():
        m = PROM_SUM_RE.match(line)
        if m and m.group("class") == CLASS_NAME:
            sums[m.group("op")] = sums.get(m.group("op"), 0.0) + float(m.group("value"))
        m = PROM_COUNT_RE.match(line)
        if m and m.group("class") == CLASS_NAME:
            counts[m.group("op")] = counts.get(m.group("op"), 0.0) + float(m.group("value"))
    return {op: (sums.get(op, 0.0), counts.get(op, 0.0)) for op in ("filter", "vector", "objects", "sort")}


def metric_delta(before: dict[str, tuple[float, float]], after: dict[str, tuple[float, float]], op: str) -> tuple[float, float, float]:
    sum_delta = after.get(op, (0.0, 0.0))[0] - before.get(op, (0.0, 0.0))[0]
    count_delta = after.get(op, (0.0, 0.0))[1] - before.get(op, (0.0, 0.0))[1]
    mean = sum_delta / count_delta if count_delta else float("nan")
    return sum_delta, count_delta, mean


def make_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    _, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[: args.queries_per_filter]
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    jobs: list[dict[str, object]] = []
    for target, filter_name, where in FILTERS:
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


def run_trial(args: argparse.Namespace, jobs: list[dict[str, object]], seed: int) -> dict[str, object]:
    ordered = list(jobs)
    random.Random(seed).shuffle(ordered)
    before = scrape_metrics(args.metrics_url)
    latencies: list[float] = []
    returned_counts: list[int] = []
    counts_by_filter: dict[str, int] = {}
    start = time.perf_counter()

    def one(job: dict[str, object]) -> tuple[str, float, int]:
        t0 = time.perf_counter()
        returned = post_graphql(args.base_url, str(job["payload"]), args.timeout)
        return str(job["filter_name"]), (time.perf_counter() - t0) * 1000, returned

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, job) for job in ordered]
        for fut in as_completed(futures):
            filter_name, elapsed, returned = fut.result()
            latencies.append(elapsed)
            returned_counts.append(returned)
            counts_by_filter[filter_name] = counts_by_filter.get(filter_name, 0) + 1

    wall_ms = (time.perf_counter() - start) * 1000
    after = scrape_metrics(args.metrics_url)
    filter_sum, filter_count, filter_mean = metric_delta(before, after, "filter")
    vector_sum, vector_count, vector_mean = metric_delta(before, after, "vector")
    return {
        "version": args.version,
        "seed": seed,
        "concurrency": args.concurrency,
        "queries": len(ordered),
        "wall_ms": wall_ms,
        "throughput_qps": len(ordered) / (wall_ms / 1000),
        "request_latency_mean_ms": statistics.mean(latencies),
        "request_latency_p50_ms": statistics.median(latencies),
        "request_latency_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "returned_mean": statistics.mean(returned_counts),
        "filter_stage_sum_ms": filter_sum,
        "filter_stage_count": filter_count,
        "filter_stage_mean_ms": filter_mean,
        "vector_stage_sum_ms": vector_sum,
        "vector_stage_count": vector_count,
        "vector_stage_mean_ms": vector_mean,
        "counts_by_filter": json.dumps(counts_by_filter, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--metrics-port", type=int, default=2112)
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/weaviate_allowlist_cache_400.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--version", default="unknown")
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    args.metrics_url = f"http://{args.host}:{args.metrics_port}/metrics"

    jobs = make_jobs(args)
    rows = []
    for seed in range(args.trials):
        row = run_trial(args, jobs, seed)
        rows.append(row)
        print(
            f"{args.version} seed={seed} c={args.concurrency} wall={float(row['wall_ms']):.2f} "
            f"qps={float(row['throughput_qps']):.2f} filter_mean={float(row['filter_stage_mean_ms']):.3f} "
            f"vector_mean={float(row['vector_stage_mean_ms']):.3f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
