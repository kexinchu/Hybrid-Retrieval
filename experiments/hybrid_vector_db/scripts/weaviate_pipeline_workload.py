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


def post_graphql(base_url: str, query: str, timeout: int) -> dict[str, object]:
    body = json.dumps({"query": query}, separators=(",", ":")).encode("utf-8")
    req = Request(base_url + "/v1/graphql", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def scrape_metrics(metrics_url: str) -> dict[str, tuple[float, float]]:
    with urlopen(metrics_url, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for line in text.splitlines():
        m = PROM_SUM_RE.match(line)
        if m and m.group("class") == CLASS_NAME:
            op = m.group("op")
            sums[op] = sums.get(op, 0.0) + float(m.group("value"))
        m = PROM_COUNT_RE.match(line)
        if m and m.group("class") == CLASS_NAME:
            op = m.group("op")
            counts[op] = counts.get(op, 0.0) + float(m.group("value"))
    return {op: (sums.get(op, 0.0), counts.get(op, 0.0)) for op in ("filter", "vector", "objects", "sort")}


def metric_delta(before: dict[str, tuple[float, float]], after: dict[str, tuple[float, float]], op: str) -> tuple[float, float, float]:
    sum_delta = after.get(op, (0.0, 0.0))[0] - before.get(op, (0.0, 0.0))[0]
    count_delta = after.get(op, (0.0, 0.0))[1] - before.get(op, (0.0, 0.0))[1]
    mean = sum_delta / count_delta if count_delta > 0 else float("nan")
    return sum_delta, count_delta, mean


def build_hybrid(vector: np.ndarray, where: dict[str, object], k: int) -> str:
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


def build_vector_only(vector: np.ndarray, k: int) -> str:
    return f"""
    {{
      Get {{
        {CLASS_NAME}(
          nearVector:{{vector:{json_to_graphql(np.asarray(vector, dtype=np.float32).tolist())}}}
          limit:{k}
        ) {{
          row_id
        }}
      }}
    }}
    """


def build_filter_only(where: dict[str, object], k: int) -> str:
    return f"""
    {{
      Get {{
        {CLASS_NAME}(
          where:{json_to_graphql(where)}
          limit:{k}
        ) {{
          row_id
        }}
      }}
    }}
    """


def make_tasks(args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    _, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[: args.queries_per_filter]
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    filters = list(FILTERS)

    hybrid: list[dict[str, object]] = []
    filter_only: list[dict[str, object]] = []
    vector_only: list[dict[str, object]] = []
    pairwise_pipeline: list[dict[str, object]] = []

    for target, filter_name, where in filters:
        for qno in query_nos:
            qid = query_by_no[qno]
            vector = xb[qid]
            ftask = {
                "kind": "filter_only",
                "filter": target,
                "filter_name": filter_name,
                "query_no": qno,
                "payload": build_filter_only(where, args.k),
            }
            vtask = {
                "kind": "vector_only",
                "filter": "",
                "filter_name": "",
                "query_no": qno,
                "payload": build_vector_only(vector, args.k),
            }
            htask = {
                "kind": "hybrid",
                "filter": target,
                "filter_name": filter_name,
                "query_no": qno,
                "payload": build_hybrid(vector, where, args.k),
            }
            filter_only.append(ftask)
            vector_only.append(vtask)
            hybrid.append(htask)
            pairwise_pipeline.extend([ftask, vtask])

    return {
        "hybrid": hybrid,
        "filter_only": filter_only,
        "vector_only": vector_only,
        "pipeline_pairwise": pairwise_pipeline,
    }


def order_tasks(name: str, base: dict[str, list[dict[str, object]]], seed: int) -> list[dict[str, object]]:
    if name == "hybrid_only":
        return list(base["hybrid"])
    if name == "filter_then_vector":
        return list(base["filter_only"]) + list(base["vector_only"])
    if name == "vector_then_filter":
        return list(base["vector_only"]) + list(base["filter_only"])
    if name == "interleaved":
        return list(base["pipeline_pairwise"])
    if name == "random_mixed":
        tasks = list(base["filter_only"]) + list(base["vector_only"])
        random.Random(seed).shuffle(tasks)
        return tasks
    raise ValueError(f"unknown schedule={name}")


def run_schedule(args: argparse.Namespace, schedule: str, tasks: list[dict[str, object]], seed: int) -> dict[str, object]:
    latencies: list[float] = []
    counts: dict[str, int] = {}
    before = scrape_metrics(args.metrics_url)
    start = time.perf_counter()

    def one(task: dict[str, object]) -> tuple[str, float]:
        t0 = time.perf_counter()
        data = post_graphql(args.base_url, str(task["payload"]), args.timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        if "errors" in data:
            raise RuntimeError(data["errors"])
        return str(task["kind"]), elapsed

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for fut in as_completed(futures):
            kind, elapsed = fut.result()
            latencies.append(elapsed)
            counts[kind] = counts.get(kind, 0) + 1

    wall_ms = (time.perf_counter() - start) * 1000
    after = scrape_metrics(args.metrics_url)
    filter_sum, filter_count, filter_mean = metric_delta(before, after, "filter")
    vector_sum, vector_count, vector_mean = metric_delta(before, after, "vector")
    return {
        "schedule": schedule,
        "seed": seed,
        "concurrency": args.concurrency,
        "queries": len(tasks),
        "wall_ms": wall_ms,
        "throughput_qps": len(tasks) / (wall_ms / 1000),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "filter_stage_sum_ms": filter_sum,
        "filter_stage_count": filter_count,
        "filter_stage_mean_ms": filter_mean,
        "vector_stage_sum_ms": vector_sum,
        "vector_stage_count": vector_count,
        "vector_stage_mean_ms": vector_mean,
        "counts_by_kind": json.dumps(counts, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--metrics-port", type=int, default=2112)
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/weaviate_pipeline_workload_20260606.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--random-trials", type=int, default=3)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    args.metrics_url = f"http://{args.host}:{args.metrics_port}/metrics"

    base = make_tasks(args)
    schedules = ["hybrid_only", "filter_then_vector", "vector_then_filter", "interleaved"]
    rows: list[dict[str, object]] = []
    for schedule in schedules:
        tasks = order_tasks(schedule, base, seed=0)
        row = run_schedule(args, schedule, tasks, seed=0)
        rows.append(row)
        print(
            f"{schedule} q={row['queries']} wall={float(row['wall_ms']):.2f} "
            f"qps={float(row['throughput_qps']):.2f} p95={float(row['latency_p95_ms']):.2f} "
            f"filter_sum={float(row['filter_stage_sum_ms']):.2f} vector_sum={float(row['vector_stage_sum_ms']):.2f}",
            flush=True,
        )
    for seed in range(args.random_trials):
        tasks = order_tasks("random_mixed", base, seed=seed)
        row = run_schedule(args, "random_mixed", tasks, seed=seed)
        rows.append(row)
        print(
            f"random_mixed seed={seed} q={row['queries']} wall={float(row['wall_ms']):.2f} "
            f"qps={float(row['throughput_qps']):.2f} p95={float(row['latency_p95_ms']):.2f} "
            f"filter_sum={float(row['filter_stage_sum_ms']):.2f} vector_sum={float(row['vector_stage_sum_ms']):.2f}",
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
