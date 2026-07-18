from __future__ import annotations

import argparse
import csv
import json
import queue
import random
import statistics
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k


STAGE_COSTS_MS = {
    "popular_ge1000": (6.90, 5.49),
    "price_10_to_20": (20.29, 8.71),
    "rating5_price_le10": (24.47, 8.49),
    "long_review_ge500": (7.53, 12.59),
    "grocery_rating5": (2.12, 12.32),
    "grocery_helpful": (8.06, 17.61),
    "helpful_ge20": (7.79, 13.80),
    "grocery_long500": (8.42, 20.36),
}


@dataclass(frozen=True)
class Job:
    seq: int
    filter_name: str
    target_rate: str
    predicate: str
    query_no: int
    query_id: int


@dataclass
class FilteredJob:
    job: Job
    selector: object
    backing: np.ndarray
    sql_rows: int
    filter_ms: float
    build_ms: float


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


def load_truth(path: Path) -> tuple[dict[tuple[str, int], list[int]], dict[int, int]]:
    truth: dict[tuple[str, int], list[int]] = {}
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "pre_filter_exact":
                continue
            qno = int(row["query_no"])
            fname = row["filter_name"]
            truth[(fname, qno)] = [int(x) for x in row["exact_filtered_topk_ids"].split(",") if x]
            query_by_no[qno] = int(row["query_id"])
    return truth, query_by_no


def make_jobs(args: argparse.Namespace) -> tuple[list[Job], dict[tuple[str, int], list[int]]]:
    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[: args.queries_per_filter]
    jobs: list[Job] = []
    seq = 0
    for filter_name, target_rate, predicate in ATTR_FILTERS:
        for qno in query_nos:
            jobs.append(Job(seq, filter_name, target_rate, predicate, qno, query_by_no[qno]))
            seq += 1
    return jobs, truth


def order_random(jobs: list[Job], seed: int) -> list[Job]:
    ordered = list(jobs)
    random.Random(seed).shuffle(ordered)
    return ordered


def order_scheduled(jobs: list[Job]) -> list[Job]:
    by_filter: dict[str, list[Job]] = {}
    for job in jobs:
        by_filter.setdefault(job.filter_name, []).append(job)
    names = sorted(
        by_filter,
        key=lambda name: STAGE_COSTS_MS[name][0] / STAGE_COSTS_MS[name][1],
        reverse=True,
    )
    mixed: list[str] = []
    left, right = 0, len(names) - 1
    while left <= right:
        mixed.append(names[left])
        if left != right:
            mixed.append(names[right])
        left += 1
        right -= 1
    ordered: list[Job] = []
    again = True
    while again:
        again = False
        for name in mixed:
            if by_filter[name]:
                ordered.append(by_filter[name].pop(0))
                again = True
    return ordered


def build_selector(conn, table: str, predicate: str, total_rows: int):
    import faiss

    def sql() -> np.ndarray:
        result = conn.execute(f"SELECT id FROM {table} WHERE {predicate}").fetchnumpy()
        return np.asarray(result["id"], dtype=np.int64)

    ids, filter_ms = timed(sql)

    def build():
        mask = np.zeros(total_rows, dtype=np.bool_)
        mask[ids] = True
        bitmap = np.packbits(mask, bitorder="little")
        selector = faiss.IDSelectorBitmap(bitmap.size, faiss.swig_ptr(bitmap))
        return selector, bitmap

    (selector, backing), build_ms = timed(build)
    return FilteredJob(None, selector, backing, int(ids.size), filter_ms, build_ms)  # type: ignore[arg-type]


def hnsw_search(index, query: np.ndarray, k: int, ef_search: int, selector):
    import faiss

    def run() -> list[int]:
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(ef_search)
        params.sel = selector
        _, ids = index.search(query.reshape(1, -1), k, params=params)
        return [int(x) for x in ids[0] if x >= 0]

    return timed(run)


def run_serial(args, jobs: list[Job], xb, index, truth) -> dict[str, object]:
    import duckdb

    conn = duckdb.connect(str(args.duckdb), read_only=True)
    conn.execute(f"PRAGMA threads={args.duckdb_threads_per_conn}")
    rows = []
    start = time.perf_counter()
    for job in jobs:
        fj = build_selector(conn, args.table, job.predicate, args.rows)
        got, vector_ms = hnsw_search(index, np.asarray(xb[job.query_id], dtype=np.float32), args.k, args.ef_search, fj.selector)
        rows.append((fj.filter_ms, fj.build_ms, vector_ms, recall_at_k(got, truth[(job.filter_name, job.query_no)], args.k)))
    wall_ms = (time.perf_counter() - start) * 1000
    conn.close()
    return summarize_run("serial", args, jobs, rows, wall_ms)


def run_request_concurrent(args, jobs: list[Job], xb, index, truth, workers: int) -> dict[str, object]:
    import duckdb

    local = threading.local()

    def get_conn():
        if not hasattr(local, "conn"):
            local.conn = duckdb.connect(str(args.duckdb), read_only=True)
            local.conn.execute(f"PRAGMA threads={args.duckdb_threads_per_conn}")
        return local.conn

    def one(job: Job):
        fj = build_selector(get_conn(), args.table, job.predicate, args.rows)
        got, vector_ms = hnsw_search(index, np.asarray(xb[job.query_id], dtype=np.float32), args.k, args.ef_search, fj.selector)
        return fj.filter_ms, fj.build_ms, vector_ms, recall_at_k(got, truth[(job.filter_name, job.query_no)], args.k)

    rows = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, job) for job in jobs]):
            rows.append(fut.result())
    wall_ms = (time.perf_counter() - start) * 1000
    return summarize_run(f"request_concurrent_{workers}", args, jobs, rows, wall_ms)


def run_pipeline(args, jobs: list[Job], xb, index, truth, filter_workers: int, vector_workers: int) -> dict[str, object]:
    import duckdb

    in_q: queue.Queue[Job | None] = queue.Queue(maxsize=args.pipeline_queue_size)
    mid_q: queue.Queue[FilteredJob | None] = queue.Queue(maxsize=args.pipeline_queue_size)
    rows = []
    rows_lock = threading.Lock()

    def filter_loop():
        conn = duckdb.connect(str(args.duckdb), read_only=True)
        conn.execute(f"PRAGMA threads={args.duckdb_threads_per_conn}")
        try:
            while True:
                job = in_q.get()
                if job is None:
                    in_q.task_done()
                    break
                fj = build_selector(conn, args.table, job.predicate, args.rows)
                fj.job = job
                mid_q.put(fj)
                in_q.task_done()
        finally:
            conn.close()

    def vector_loop():
        while True:
            fj = mid_q.get()
            if fj is None:
                mid_q.task_done()
                break
            job = fj.job
            got, vector_ms = hnsw_search(index, np.asarray(xb[job.query_id], dtype=np.float32), args.k, args.ef_search, fj.selector)
            rec = recall_at_k(got, truth[(job.filter_name, job.query_no)], args.k)
            with rows_lock:
                rows.append((fj.filter_ms, fj.build_ms, vector_ms, rec))
            mid_q.task_done()

    start = time.perf_counter()
    filter_threads = [threading.Thread(target=filter_loop) for _ in range(filter_workers)]
    vector_threads = [threading.Thread(target=vector_loop) for _ in range(vector_workers)]
    for t in filter_threads + vector_threads:
        t.start()
    for job in jobs:
        in_q.put(job)
    for _ in filter_threads:
        in_q.put(None)
    in_q.join()
    for _ in vector_threads:
        mid_q.put(None)
    mid_q.join()
    for t in filter_threads + vector_threads:
        t.join()
    wall_ms = (time.perf_counter() - start) * 1000
    return summarize_run(f"pipeline_f{filter_workers}_v{vector_workers}", args, jobs, rows, wall_ms)


def summarize_run(name: str, args, jobs: list[Job], rows: list[tuple[float, float, float, float]], wall_ms: float) -> dict[str, object]:
    filter_ms = [x[0] for x in rows]
    build_ms = [x[1] for x in rows]
    vector_ms = [x[2] for x in rows]
    recalls = [x[3] for x in rows]
    return {
        "method": name,
        "order": args.order,
        "queries": len(jobs),
        "wall_ms": wall_ms,
        "throughput_qps": len(jobs) / (wall_ms / 1000),
        "filter_ms_mean": statistics.mean(filter_ms),
        "filter_ms_sum": sum(filter_ms),
        "build_ms_mean": statistics.mean(build_ms),
        "build_ms_sum": sum(build_ms),
        "vector_ms_mean": statistics.mean(vector_ms),
        "vector_ms_sum": sum(vector_ms),
        "recall_mean": statistics.mean(recalls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=Path("data/duckdb/amazon_grocery_10m.duckdb"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--index", type=Path, default=Path("data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/faiss_duckdb_pipeline_400_20260606.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--faiss-omp-threads", type=int, default=1)
    parser.add_argument("--duckdb-threads-per-conn", type=int, default=1)
    parser.add_argument("--request-workers", type=int, default=4)
    parser.add_argument("--filter-workers", type=int, default=4)
    parser.add_argument("--vector-workers", type=int, default=4)
    parser.add_argument("--pipeline-queue-size", type=int, default=32)
    parser.add_argument("--order", choices=["random", "scheduled"], default="scheduled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-serial", action="store_true")
    args = parser.parse_args()

    import faiss

    faiss.omp_set_num_threads(args.faiss_omp_threads)
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    index = faiss.read_index(str(args.index))
    jobs, truth = make_jobs(args)
    jobs = order_random(jobs, args.seed) if args.order == "random" else order_scheduled(jobs)
    print(f"jobs={len(jobs)} order={args.order}", flush=True)

    results = []
    if not args.skip_serial:
        result = run_serial(args, jobs, xb, index, truth)
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
    result = run_request_concurrent(args, jobs, xb, index, truth, args.request_workers)
    print(json.dumps(result, sort_keys=True), flush=True)
    results.append(result)
    result = run_pipeline(args, jobs, xb, index, truth, args.filter_workers, args.vector_workers)
    print(json.dumps(result, sort_keys=True), flush=True)
    results.append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
