from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg

from common_pg import pg_config_from_env
from pgvector_c4_guidance_memory_benchmark import (
    INDEX,
    TABLE,
    activate,
    cache_profile,
    compare_same_ids,
    configure,
    load_all_memory,
    prebuild_store,
    reset_store,
    run_query,
    store_summary,
)
from pgvector_c4_query_filter_cache_benchmark import load_workload


MODES = ["native", "cold_rebuild", "store_reuse", "memory_reuse", "managed_reuse"]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row.get(key, 0) or 0) for row in rows) if rows else 0.0


def p95(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    vals = sorted(float(row.get(key, 0) or 0) for row in rows)
    return vals[int(0.95 * (len(vals) - 1))]


def run_mode(cur: psycopg.Cursor, args: argparse.Namespace, workload: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    memory_mb = args.managed_memory_mb if mode == "managed_reuse" else args.all_memory_mb
    configure(cur, args, memory_mb, "off")
    rows: list[dict[str, Any]] = []
    if mode == "cold_rebuild":
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    elif mode in {"store_reuse", "managed_reuse"}:
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    elif mode == "memory_reuse":
        cache_keys = sorted({str(row["cache_key"]) for row in workload})
        load_all_memory(cur, args.index, cache_keys, args.cache_kind)

    for query_no, row in enumerate(workload):
        predicate = str(row["predicate"])
        cache_key = str(row["cache_key"])
        query_id = int(row["query_id_int"])
        activation_profile: dict[str, Any] = {}
        resident_profile: dict[str, Any] = {}
        activation_ms = 0.0

        if mode == "cold_rebuild":
            reset_store(cur)
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
        elif mode == "store_reuse":
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")

        if mode != "native":
            activation_profile, activation_ms = timed_ms(lambda: activate(cur, args.index, cache_key, args.cache_kind))
            resident_profile = cache_profile(cur)

        (ids, profile, error), latency_ms = timed_ms(lambda: run_query(cur, args.table, predicate, query_id, args.k))
        rows.append(
            {
                "mode": mode,
                "query_no": query_no,
                "c4_query_no": row["query_no"],
                "query_id": query_id,
                "predicate": predicate,
                "cache_key": cache_key,
                "latency_ms": latency_ms,
                "activation_ms": activation_ms,
                "end_to_end_ms": latency_ms + activation_ms,
                "returned": len(ids),
                "ids": ",".join(str(x) for x in ids),
                "error": error,
                "vector_search_ms": profile.get("vector_search_ms", 0.0),
                "visited_tuples": profile.get("visited_tuples", 0),
                "returned_tuples": profile.get("returned_tuples", 0),
                "guidance_checks": profile.get("guidance_checks", 0),
                "guidance_matches": profile.get("guidance_matches", 0),
                "guidance_skips": profile.get("guidance_skips", 0),
                "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
                "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
                "fragment_cache_hits": activation_profile.get("fragment_cache_hits", 0),
                "fragment_cache_misses": activation_profile.get("fragment_cache_misses", 0),
                "fragment_store_hits": activation_profile.get("fragment_store_hits", 0),
                "fragment_builds": activation_profile.get("fragment_builds", 0),
                "cache_entries": resident_profile.get("entries", 0),
                "cache_resident_entries": resident_profile.get("resident_entries", 0),
                "cache_resident_bytes": resident_profile.get("resident_bytes", 0),
                "cache_evictions": resident_profile.get("evictions", 0),
            }
        )

        if args.stream and (query_no + 1) % args.progress_queries == 0:
            ok = [r for r in rows if not r["error"]]
            print(
                f"{mode} progress {query_no + 1}/{len(workload)} "
                f"e2e={mean(ok, 'end_to_end_ms'):.2f} "
                f"activation={mean(ok, 'activation_ms'):.2f}",
                flush=True,
            )

    cur.execute("SELECT vector_hnsw_guidance_reset()")
    return rows


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if not row["error"]]
    checks = mean(ok, "guidance_checks")
    skips = mean(ok, "guidance_skips")
    return {
        "rows": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "latency_mean_ms": mean(ok, "latency_ms"),
        "latency_p95_ms": p95(ok, "latency_ms"),
        "activation_mean_ms": mean(ok, "activation_ms"),
        "activation_p95_ms": p95(ok, "activation_ms"),
        "activation_build_mean_ms": mean(ok, "activation_build_ms"),
        "activation_build_p95_ms": p95(ok, "activation_build_ms"),
        "end_to_end_mean_ms": mean(ok, "end_to_end_ms"),
        "end_to_end_p95_ms": p95(ok, "end_to_end_ms"),
        "vector_search_mean_ms": mean(ok, "vector_search_ms"),
        "visited_tuples_mean": mean(ok, "visited_tuples"),
        "returned_tuples_mean": mean(ok, "returned_tuples"),
        "guidance_skip_rate": skips / checks if checks else 0.0,
        "fragment_cache_hits_mean": mean(ok, "fragment_cache_hits"),
        "fragment_store_hits_mean": mean(ok, "fragment_store_hits"),
        "fragment_builds_mean": mean(ok, "fragment_builds"),
        "cache_resident_entries_mean": mean(ok, "cache_resident_entries"),
        "cache_resident_bytes_mean": mean(ok, "cache_resident_bytes"),
        "cache_evictions_max": max((int(row.get("cache_evictions", 0) or 0) for row in ok), default=0),
    }


def write_rows(path: Path, rows_by_mode: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(next(iter(rows_by_mode.values()))[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in MODES:
            if mode in rows_by_mode:
                writer.writerows(rows_by_mode[mode])


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate SQLens D3 predicate-state reuse on the Amazon-C4 workload.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--cache-kind", default="bloom", choices=["page", "bloom"])
    parser.add_argument("--all-memory-mb", type=int, default=1024)
    parser.add_argument("--managed-memory-mb", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--progress-queries", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    workload = load_workload(args.query_csv, args.queries, args.mode)
    cache_keys = sorted({str(row["cache_key"]) for row in workload})
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb, "off")
        reset_store(cur)
        prebuild_wall_ms, _ = prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        store = store_summary(cur)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        rows_by_mode["native"] = run_mode(conn.cursor(), args, workload, "native")

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        rows_by_mode["cold_rebuild"] = run_mode(conn.cursor(), args, workload, "cold_rebuild")

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb, "off")
        reset_store(cur)
        prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        rows_by_mode["store_reuse"] = run_mode(cur, args, workload, "store_reuse")

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb, "off")
        reset_store(cur)
        prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        rows_by_mode["memory_reuse"] = run_mode(cur, args, workload, "memory_reuse")

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.managed_memory_mb, "off")
        reset_store(cur)
        prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        rows_by_mode["managed_reuse"] = run_mode(cur, args, workload, "managed_reuse")

    write_rows(args.out, rows_by_mode)
    summary = {
        "table": args.table,
        "index": args.index,
        "queries": len(workload),
        "mode": args.mode,
        "cache_kind": args.cache_kind,
        "distinct_fragments": len(cache_keys),
        "prebuild_wall_ms": prebuild_wall_ms,
        "ssd_payload_bytes": store["ssd_payload_bytes"],
        "ssd_fragments": store["ssd_fragments"],
        "all_memory_budget_mb": args.all_memory_mb,
        "managed_memory_budget_mb": args.managed_memory_mb,
        "modes": {mode: summarize_mode(rows) for mode, rows in rows_by_mode.items()},
        **compare_same_ids(rows_by_mode),
    }
    native_mean = summary["modes"]["native"]["end_to_end_mean_ms"]
    cold_mean = summary["modes"]["cold_rebuild"]["end_to_end_mean_ms"]
    for mode, mode_summary in summary["modes"].items():
        e2e = mode_summary["end_to_end_mean_ms"]
        mode_summary["speedup_vs_native_e2e"] = native_mean / e2e if e2e else 0.0
        mode_summary["speedup_vs_cold_rebuild_e2e"] = cold_mean / e2e if e2e else 0.0

    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
