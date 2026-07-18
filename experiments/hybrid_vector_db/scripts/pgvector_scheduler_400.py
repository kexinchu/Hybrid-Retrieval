from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common_pg import pg_config_from_env, require_psycopg
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS
from pgvector_prefilter_10m import TABLE, load_truth, vector_literal


DEFAULT_STAGE_COSTS_MS = {
    # filter predicate evaluation inside pgvector's HNSW scan, vector HNSW scan.
    # These are measured from pgvector_native_postfilter_10m_q100_r5_ef1000_iter_strict_200k_profile_20260609_summary.csv.
    "popular_ge1000": (0.0053, 16.9002),
    "price_10_to_20": (0.0097, 15.9541),
    "rating5_price_le10": (0.0263, 17.5559),
    "long_review_ge500": (0.3307, 45.9478),
    "grocery_rating5": (0.0568, 14.8950),
    "grocery_helpful": (0.0788, 15.3206),
    "helpful_ge20": (0.3118, 50.8627),
    "grocery_long500": (0.8244, 59.1332),
}

_TLS = threading.local()


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def load_stage_costs(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None or not path.exists():
        return dict(DEFAULT_STAGE_COSTS_MS)
    costs: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["filter_name"]
            filter_ms = float(row.get("query_sql_filter_ms") or 0.0)
            vector_ms = float(row.get("query_vector_ms") or row.get("post_vector_search_ms") or 0.0)
            costs[name] = (filter_ms, vector_ms)
    return costs or dict(DEFAULT_STAGE_COSTS_MS)


def make_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    _, query_by_no = load_truth(args.truth_csv, "pre_filter_exact")
    query_nos = sorted(query_by_no)[: args.queries_per_filter]
    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    jobs: list[dict[str, object]] = []
    for filter_name, target, predicate in ATTR_FILTERS:
        for qno in query_nos:
            qid = query_by_no[qno]
            jobs.append(
                {
                    "filter": target,
                    "filter_name": filter_name,
                    "query_no": qno,
                    "query_id": qid,
                    "predicate": predicate,
                    "query": vector_literal(np.asarray(xb[qid], dtype=np.float32)),
                }
            )
    return jobs


def random_order(jobs: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    ordered = list(jobs)
    random.Random(seed).shuffle(ordered)
    return ordered


def pair_extremes_order(jobs: list[dict[str, object]], stage_costs: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    by_filter: dict[str, list[dict[str, object]]] = {}
    for job in jobs:
        by_filter.setdefault(str(job["filter_name"]), []).append(job)

    names = sorted(
        by_filter,
        key=lambda name: stage_costs[name][0] / max(stage_costs[name][1], 1e-9),
        reverse=True,
    )
    cycle: list[str] = []
    left, right = 0, len(names) - 1
    while left <= right:
        cycle.append(names[left])
        if left != right:
            cycle.append(names[right])
        left += 1
        right -= 1

    ordered: list[dict[str, object]] = []
    while any(by_filter[name] for name in cycle):
        for name in cycle:
            if by_filter[name]:
                ordered.append(by_filter[name].pop(0))
    return ordered


def scheduled_alternating_stage_heavy_order(
    jobs: list[dict[str, object]], stage_costs: dict[str, tuple[float, float]], seed: int = 0
) -> list[dict[str, object]]:
    by_filter: dict[str, list[dict[str, object]]] = {}
    for job in jobs:
        by_filter.setdefault(str(job["filter_name"]), []).append(job)

    rng = random.Random(seed)
    for bucket in by_filter.values():
        rng.shuffle(bucket)

    filter_heavy = sorted(
        [name for name, (filter_ms, vector_ms) in stage_costs.items() if filter_ms >= vector_ms and name in by_filter],
        key=lambda name: stage_costs[name][0] - stage_costs[name][1],
        reverse=True,
    )
    vector_heavy = sorted(
        [name for name, (filter_ms, vector_ms) in stage_costs.items() if vector_ms > filter_ms and name in by_filter],
        key=lambda name: stage_costs[name][1] - stage_costs[name][0],
        reverse=True,
    )
    if not filter_heavy or not vector_heavy:
        return pair_extremes_order(jobs, stage_costs)

    cycle: list[str] = []
    for i in range(max(len(filter_heavy), len(vector_heavy))):
        if i < len(filter_heavy):
            cycle.append(filter_heavy[i])
        if i < len(vector_heavy):
            cycle.append(vector_heavy[i])

    ordered: list[dict[str, object]] = []
    while any(by_filter[name] for name in cycle):
        for name in cycle:
            if by_filter[name]:
                ordered.append(by_filter[name].pop())
    return ordered


def get_conn(args: argparse.Namespace):
    require_psycopg()
    import psycopg

    conn = getattr(_TLS, "conn", None)
    if conn is not None and not conn.closed:
        return conn
    cfg = pg_config_from_env()
    conn = psycopg.connect(cfg.conninfo, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
        if args.iterative_scan:
            cur.execute(f"SET hnsw.iterative_scan = '{args.iterative_scan}'")
        cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
        cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
        cur.execute(f"SET enable_seqscan = {'off' if args.disable_seqscan else 'on'}")
    _TLS.conn = conn
    return conn


def close_thread_conn() -> None:
    conn = getattr(_TLS, "conn", None)
    if conn is not None and not conn.closed:
        conn.close()
    _TLS.conn = None


def run_once(
    args: argparse.Namespace,
    order_name: str,
    jobs: list[dict[str, object]],
    seed: int | None = None,
) -> dict[str, object]:
    latencies: list[float] = []
    returned_counts: list[int] = []
    counts_by_filter: dict[str, int] = {}
    errors: list[str] = []
    start = time.perf_counter()

    def one(job: dict[str, object]) -> tuple[str, float, int]:
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
        elapsed = (time.perf_counter() - t0) * 1000
        return str(job["filter_name"]), elapsed, len(rows)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                filter_name, elapsed, returned = fut.result()
            except Exception as exc:  # keep enough context before failing the whole run
                errors.append(repr(exc))
                continue
            latencies.append(elapsed)
            returned_counts.append(returned)
            counts_by_filter[filter_name] = counts_by_filter.get(filter_name, 0) + 1

    wall_ms = (time.perf_counter() - start) * 1000
    if errors:
        raise RuntimeError(f"{len(errors)} query errors, first={errors[0]}")
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
    parser.add_argument("--stage-cost-summary", type=Path, default=Path("results/hybrid_vector_db/pgvector_native_postfilter_10m_q100_r5_ef1000_iter_strict_200k_profile_20260609_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_scheduler_400_ef1000_iter_strict_200k.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries-per-filter", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order")
    parser.add_argument("--max-scan-tuples", type=int, default=200_000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=1.0)
    parser.add_argument("--disable-seqscan", action="store_true")
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--scheduled-trials", type=int, default=1)
    args = parser.parse_args()

    jobs = make_jobs(args)
    expected = len(ATTR_FILTERS) * args.queries_per_filter
    if len(jobs) != expected:
        raise RuntimeError(f"unexpected job count={len(jobs)} expected={expected}")

    stage_costs = load_stage_costs(args.stage_cost_summary)
    rows: list[dict[str, object]] = []
    for seed in range(args.scheduled_trials):
        scheduled = scheduled_alternating_stage_heavy_order(jobs, stage_costs, seed=seed)
        rows.append(run_once(args, "scheduled_stage_alternating", scheduled, seed=seed))
    for seed in range(args.random_trials):
        rows.append(run_once(args, "random", random_order(jobs, seed), seed=seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}", flush=True)
    print(
        "stage_costs="
        + json.dumps({k: {"filter_ms": v[0], "vector_ms": v[1]} for k, v in stage_costs.items()}, sort_keys=True),
        flush=True,
    )
    for row in rows:
        print(
            f"{row['order']} seed={row['seed']} concurrency={row['concurrency']} "
            f"wall_ms={float(row['wall_ms']):.2f} qps={float(row['throughput_qps']):.1f} "
            f"mean_req_ms={float(row['request_latency_mean_ms']):.2f} returned={float(row['returned_mean']):.2f}",
            flush=True,
        )
    close_thread_conn()


if __name__ == "__main__":
    main()
