from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from pgvector_c4_guidance_memory_benchmark import (
    INDEX,
    TABLE,
    configure,
    load_all_memory,
    prebuild_store,
    reset_store,
    run_query,
    store_summary,
)
from pgvector_c4_query_filter_cache_benchmark import load_workload


_TLS = threading.local()


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[int(0.95 * (len(vals) - 1))]


def configure_worker(cur: psycopg.Cursor, args: argparse.Namespace, mode: str) -> None:
    memory = args.all_memory_mb if mode == "all_memory" else args.managed_memory_mb
    filter_strategy = "off" if mode != "native" else "off"
    configure(cur, args, memory if mode != "native" else None, filter_strategy)


def get_conn(args: argparse.Namespace, mode: str, cache_keys: list[str]):
    key = f"conn_{mode}"
    conn = getattr(_TLS, key, None)
    if conn is not None and not conn.closed:
        return conn
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    with conn.cursor() as cur:
        configure_worker(cur, args, mode)
        if mode == "all_memory":
            load_all_memory(cur, args.index, cache_keys, args.cache_kind)
    setattr(_TLS, key, conn)
    return conn


def run_one(
    args: argparse.Namespace,
    mode: str,
    cache_keys: list[str],
    row: dict[str, object],
    baseline_ids: dict[int, str],
) -> dict[str, object]:
    conn = get_conn(args, mode, cache_keys)
    predicate = str(row["predicate"])
    cache_key = str(row["cache_key"])
    query_no = int(row["query_no"])
    query_id = int(row["query_id_int"])
    activation_ms = 0.0
    activation_profile: dict[str, object] = {}
    resident_profile: dict[str, object] = {}
    error = ""
    try:
        with conn.cursor() as cur:
            if mode != "native":
                t0 = time.perf_counter()
                cur.execute(
                    "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
                    (args.index, [cache_key], args.cache_kind),
                )
                cur.execute("SELECT vector_hnsw_guidance_profile()")
                activation_profile = json.loads(cur.fetchone()[0])
                cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
                resident_profile = json.loads(cur.fetchone()[0])
                activation_ms = (time.perf_counter() - t0) * 1000.0
            (ids, profile, error), latency_ms = timed_ms(lambda: run_query(cur, args.table, predicate, query_id, args.k))
    except errors.QueryCanceled as exc:
        ids = []
        profile = {}
        latency_ms = 0.0
        error = exc.__class__.__name__
    ids_text = ",".join(str(x) for x in ids)
    return {
        "mode": mode,
        "concurrency": args.concurrency,
        "query_no": query_no,
        "c4_query_no": row["query_no"],
        "query_id": query_id,
        "cache_key": cache_key,
        "latency_ms": latency_ms,
        "activation_ms": activation_ms,
        "end_to_end_ms": latency_ms + activation_ms,
        "returned": len(ids),
        "ids": ids_text,
        "baseline_ordered_match": ids_text == baseline_ids.get(query_no, ""),
        "error": error,
        "vector_search_ms": profile.get("vector_search_ms", 0.0),
        "visited_tuples": profile.get("visited_tuples", 0),
        "returned_tuples": profile.get("returned_tuples", 0),
        "guidance_checks": profile.get("guidance_checks", 0),
        "guidance_skips": profile.get("guidance_skips", 0),
        "fragment_cache_hits": activation_profile.get("fragment_cache_hits", 0),
        "fragment_store_hits": activation_profile.get("fragment_store_hits", 0),
        "fragment_builds": activation_profile.get("fragment_builds", 0),
        "cache_resident_entries": resident_profile.get("resident_entries", 0),
        "cache_resident_bytes": resident_profile.get("resident_bytes", 0),
        "cache_evictions": resident_profile.get("evictions", 0),
    }


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def summarize(mode: str, concurrency: int, rows: list[dict[str, object]], wall_ms: float) -> dict[str, object]:
    ok = [r for r in rows if not r["error"]]

    def mean(key: str) -> float:
        return statistics.fmean(float(r[key]) for r in ok) if ok else 0.0

    checks = mean("guidance_checks")
    skips = mean("guidance_skips")
    return {
        "mode": mode,
        "concurrency": concurrency,
        "queries": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "wall_ms": wall_ms,
        "throughput_qps": len(ok) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        "e2e_mean_ms": mean("end_to_end_ms"),
        "e2e_p50_ms": statistics.median(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0,
        "e2e_p95_ms": p95([float(r["end_to_end_ms"]) for r in ok]),
        "activation_mean_ms": mean("activation_ms"),
        "returned_tuples_mean": mean("returned_tuples"),
        "guidance_skip_rate": skips / checks if checks else 0.0,
        "ordered_match": sum(1 for r in ok if str(r["baseline_ordered_match"]).lower() == "true"),
        "cache_resident_entries_mean": mean("cache_resident_entries"),
        "cache_evictions_max": max((int(r["cache_evictions"]) for r in ok), default=0),
    }


def build_baseline_ids(args: argparse.Namespace, workload: list[dict[str, object]]) -> dict[int, str]:
    baseline: dict[int, str] = {}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, None, "off")
        for row in workload:
            ids, _, error = run_query(cur, args.table, str(row["predicate"]), int(row["query_id_int"]), args.k)
            if error:
                raise RuntimeError(f"baseline query failed: {error}")
            baseline[int(row["query_no"])] = ",".join(str(x) for x in ids)
    return baseline


def run_concurrent(
    args: argparse.Namespace,
    mode: str,
    workload: list[dict[str, object]],
    cache_keys: list[str],
    baseline_ids: dict[int, str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, args, mode, cache_keys, row, baseline_ids) for row in workload]
        for fut in as_completed(futures):
            rows.append(fut.result())
    wall_ms = (time.perf_counter() - start) * 1000.0
    return rows, summarize(mode, args.concurrency, rows, wall_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent C4 D3 cache-control benchmark.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--methods", default="native,managed_cache")
    parser.add_argument("--cache-kind", default="bloom", choices=["page", "bloom"])
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--managed-memory-mb", type=int, default=1)
    parser.add_argument("--all-memory-mb", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--filter-strategy", default="off", choices=["off", "acorn1"])
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    workload = load_workload(args.query_csv, args.queries, args.mode)
    cache_keys = sorted({str(row["cache_key"]) for row in workload})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb, "off")
        reset_store(cur)
        prebuild_wall_ms, _ = prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        store = store_summary(cur)

    baseline_ids = build_baseline_ids(args, workload)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        rows, summary = run_concurrent(args, method, workload, cache_keys, baseline_ids)
        summary["prebuild_wall_ms"] = prebuild_wall_ms
        summary["ssd_payload_bytes"] = store["ssd_payload_bytes"]
        summary["ssd_fragments"] = store["ssd_fragments"]
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"{method} c={args.concurrency} qps={summary['throughput_qps']:.1f} "
            f"mean={summary['e2e_mean_ms']:.2f} p95={summary['e2e_p95_ms']:.2f} "
            f"match={summary['ordered_match']}/{summary['ok']}",
            flush=True,
        )

    raw_fields = list(all_rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {args.out}")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
