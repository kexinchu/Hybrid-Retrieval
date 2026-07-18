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

from weaviate_prefilter_10m import CLASS_NAME, FILTERS, json_to_graphql, load_truth, read_fbin_memmap


def post_graphql(base_url: str, query: str) -> dict[str, object]:
    body = json.dumps({"query": query}, separators=(",", ":")).encode("utf-8")
    req = Request(base_url + "/v1/graphql", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    _, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[: args.queries_per_filter]

    estimates = {}
    with args.estimate_summary.open(newline="") as f:
        for row in csv.DictReader(f):
            estimates[row["filter_name"]] = float(row["pre_latency_ms"])

    tasks: list[dict[str, object]] = []
    for target_rate, filter_name, where in FILTERS:
        estimate = estimates[filter_name]
        for qno in query_nos:
            query_id = query_by_no[qno]
            vector = np.asarray(xb[query_id], dtype=np.float32).tolist()
            gq = f"""
            {{
              Get {{
                {CLASS_NAME}(
                  nearVector:{{vector:{json_to_graphql(vector)}}}
                  where:{json_to_graphql(where)}
                  limit:{args.k}
                ) {{
                  row_id
                }}
              }}
            }}
            """
            tasks.append(
                {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "query_no": qno,
                    "query_id": query_id,
                    "estimate_ms": estimate,
                    "graphql": gq,
                }
            )
    return tasks


def run_one(base_url: str, task: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    data = post_graphql(base_url, str(task["graphql"]))
    latency_ms = (time.perf_counter() - start) * 1000
    if "errors" in data:
        raise RuntimeError(data["errors"])
    returned = len(data["data"]["Get"][CLASS_NAME])
    return {
        "filter": task["filter"],
        "filter_name": task["filter_name"],
        "query_no": task["query_no"],
        "query_id": task["query_id"],
        "estimate_ms": task["estimate_ms"],
        "latency_ms": latency_ms,
        "returned": returned,
    }


def run_order(args: argparse.Namespace, name: str, ordered_tasks: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    base_url = f"http://{args.host}:{args.port}"
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, base_url, task) for task in ordered_tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())
    wall_ms = (time.perf_counter() - t0) * 1000
    latencies = [float(row["latency_ms"]) for row in rows]
    summary = {
        "schedule": name,
        "queries": len(rows),
        "concurrency": args.concurrency,
        "wall_ms": wall_ms,
        "mean_latency_ms": statistics.mean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "max_latency_ms": max(latencies),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--estimate-summary", type=Path, default=Path("results/hybrid_vector_db/weaviate_prefilter_10m_q100_flat0_dynamic_20260606_summary.csv"))
    parser.add_argument("--out-prefix", type=Path, default=Path("results/hybrid_vector_db/weaviate_mixed400_schedule_20260606"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=57)
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    tasks = build_tasks(args)
    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []

    for trial in range(args.random_trials):
        rng = random.Random(args.random_seed + trial)
        ordered = tasks[:]
        rng.shuffle(ordered)
        summary, rows = run_order(args, f"random_{trial}", ordered)
        summaries.append(summary)
        for row in rows:
            all_rows.append({"schedule": summary["schedule"], **row})
        print(
            f"{summary['schedule']} wall_ms={summary['wall_ms']:.2f} "
            f"mean={summary['mean_latency_ms']:.2f} p95={summary['p95_latency_ms']:.2f}",
            flush=True,
        )

    lpt = sorted(tasks, key=lambda item: float(item["estimate_ms"]), reverse=True)
    summary, rows = run_order(args, "scheduled_lpt", lpt)
    summaries.append(summary)
    for row in rows:
        all_rows.append({"schedule": summary["schedule"], **row})
    print(
        f"{summary['schedule']} wall_ms={summary['wall_ms']:.2f} "
        f"mean={summary['mean_latency_ms']:.2f} p95={summary['p95_latency_ms']:.2f}",
        flush=True,
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_out = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    detail_out = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with detail_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {summary_out}", flush=True)
    print(f"wrote {detail_out}", flush=True)


if __name__ == "__main__":
    main()
