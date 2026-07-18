from __future__ import annotations

import argparse
import csv
import json
import queue
import random
import statistics
import threading
import time
from pathlib import Path

from pgvector_prefilter_10m import TABLE
from pgvector_scheduler_400 import get_conn, make_jobs, random_order, run_once


def execute_pgvector_query(args: argparse.Namespace, job: dict[str, object]) -> tuple[str, float, int]:
    conn = get_conn(args)
    sql = f"""
        SELECT id
        FROM {TABLE}
        WHERE {job["predicate"]}
        ORDER BY embedding <-> %s::vector
        LIMIT {int(args.k)}
        """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (job["query"],))
        rows = cur.fetchall()
    return str(job["filter_name"]), (time.perf_counter() - t0) * 1000, len(rows)


def run_pipeline_control(
    args: argparse.Namespace,
    jobs: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    # This is intentionally a control, not a new pgvector algorithm:
    # stage 1 can only prepare request metadata because stock pgvector does not
    # expose a separate allow-list HNSW search API. Stage 2 still executes the
    # same atomic PostgreSQL query as the baseline.
    in_q: queue.Queue[dict[str, object] | None] = queue.Queue()
    vector_q: queue.Queue[dict[str, object] | None] = queue.Queue()
    latencies: list[float] = []
    returned_counts: list[int] = []
    counts_by_filter: dict[str, int] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def filter_stage() -> None:
        while True:
            job = in_q.get()
            try:
                if job is None:
                    vector_q.put(None)
                    return
                # Metadata preparation placeholder. The predicate and vector
                # literal are already prebuilt in make_jobs().
                vector_q.put(job)
            finally:
                in_q.task_done()

    def vector_stage() -> None:
        while True:
            job = vector_q.get()
            try:
                if job is None:
                    return
                try:
                    filter_name, elapsed, returned = execute_pgvector_query(args, job)
                except Exception as exc:
                    with lock:
                        errors.append(repr(exc))
                    continue
                with lock:
                    latencies.append(elapsed)
                    returned_counts.append(returned)
                    counts_by_filter[filter_name] = counts_by_filter.get(filter_name, 0) + 1
            finally:
                vector_q.task_done()

    start = time.perf_counter()
    filter_threads = [threading.Thread(target=filter_stage) for _ in range(args.filter_workers)]
    vector_threads = [threading.Thread(target=vector_stage) for _ in range(args.vector_workers)]
    for thread in filter_threads + vector_threads:
        thread.start()
    for job in jobs:
        in_q.put(job)
    for _ in filter_threads:
        in_q.put(None)
    in_q.join()
    vector_q.join()
    for thread in filter_threads + vector_threads:
        thread.join()
    wall_ms = (time.perf_counter() - start) * 1000
    if errors:
        raise RuntimeError(f"{len(errors)} query errors, first={errors[0]}")
    return {
        "version": "pipeline_control_atomic_pgvector",
        "seed": seed,
        "concurrency": args.concurrency,
        "filter_workers": args.filter_workers,
        "vector_workers": args.vector_workers,
        "queries": len(jobs),
        "wall_ms": wall_ms,
        "throughput_qps": len(jobs) / (wall_ms / 1000),
        "request_latency_mean_ms": statistics.mean(latencies),
        "request_latency_p50_ms": statistics.median(latencies),
        "request_latency_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "returned_mean": statistics.mean(returned_counts),
        "counts_by_filter": json.dumps(counts_by_filter, sort_keys=True),
        "ef_search": args.ef_search,
        "iterative_scan": args.iterative_scan or "",
        "max_scan_tuples": args.max_scan_tuples,
        "scan_mem_multiplier": args.scan_mem_multiplier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_pipeline_control_400.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--filter-workers", type=int, default=32)
    parser.add_argument("--vector-workers", type=int, default=32)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order")
    parser.add_argument("--max-scan-tuples", type=int, default=200_000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=1.0)
    parser.add_argument("--disable-seqscan", action="store_true")
    parser.add_argument("--random-trials", type=int, default=5)
    args = parser.parse_args()

    base_jobs = make_jobs(args)
    rows: list[dict[str, object]] = []
    for seed in range(args.random_trials):
        jobs = random_order(base_jobs, seed)
        baseline = run_once(args, "random", jobs, seed=seed)
        baseline = {
            "version": "original_atomic_pgvector",
            "filter_workers": "",
            "vector_workers": "",
            **baseline,
        }
        rows.append(baseline)
        print(
            f"original seed={seed} c={args.concurrency} wall={float(baseline['wall_ms']):.2f} "
            f"qps={float(baseline['throughput_qps']):.2f}",
            flush=True,
        )
    for seed in range(args.random_trials):
        jobs = random_order(base_jobs, seed)
        row = run_pipeline_control(args, jobs, seed=seed)
        rows.append(row)
        print(
            f"pipeline_control seed={seed} c={args.concurrency} wall={float(row['wall_ms']):.2f} "
            f"qps={float(row['throughput_qps']):.2f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version",
        "order",
        "seed",
        "concurrency",
        "filter_workers",
        "vector_workers",
        "queries",
        "wall_ms",
        "throughput_qps",
        "request_latency_mean_ms",
        "request_latency_p50_ms",
        "request_latency_p95_ms",
        "returned_mean",
        "counts_by_filter",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "scan_mem_multiplier",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
