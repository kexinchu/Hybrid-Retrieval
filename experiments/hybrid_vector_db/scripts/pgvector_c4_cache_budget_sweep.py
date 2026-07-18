from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import psycopg

from common_pg import pg_config_from_env
from pgvector_c4_guidance_memory_benchmark import (
    INDEX,
    TABLE,
    compare_same_ids,
    configure,
    load_all_memory,
    prebuild_store,
    reset_store,
    run_mode,
    store_summary,
    summarize_rows,
)
from pgvector_c4_query_filter_cache_benchmark import load_workload


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[int(0.95 * (len(vals) - 1))]


def summarize_budget(
    budget_mb: int | str,
    native_rows: list[dict[str, object]],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    summary = summarize_rows(rows)
    matches = compare_same_ids({"native": native_rows, "managed_cache": rows})
    ok = [r for r in rows if not r["error"]]
    native = summarize_rows(native_rows)
    activation = [float(r["activation_ms"]) for r in ok]
    latency = [float(r["latency_ms"]) for r in ok]
    e2e = [float(r["end_to_end_ms"]) for r in ok]
    return {
        "budget_mb": budget_mb,
        "queries": summary["rows"],
        "ok": summary["ok"],
        "ordered_match": matches["managed_cache_same_ordered_ids"],
        "set_match": matches["managed_cache_same_set_ids"],
        "native_e2e_mean_ms": native["end_to_end_mean_ms"],
        "e2e_mean_ms": summary["end_to_end_mean_ms"],
        "e2e_p95_ms": summary["end_to_end_p95_ms"],
        "latency_mean_ms": summary["latency_mean_ms"],
        "activation_mean_ms": summary["activation_mean_ms"],
        "activation_p95_ms": p95(activation),
        "server_latency_p95_ms": p95(latency),
        "client_e2e_p95_ms": p95(e2e),
        "returned_tuples_mean": summary["returned_tuples_mean"],
        "guidance_checks_mean": summary["guidance_checks_mean"],
        "guidance_skips_mean": summary["guidance_skips_mean"],
        "guidance_skip_rate": summary["guidance_skip_rate"],
        "resident_entries_mean": summary["cache_resident_entries_mean"],
        "resident_entries_max": summary["cache_resident_entries_max"],
        "resident_bytes_mean": summary["cache_resident_bytes_mean"],
        "resident_bytes_max": summary["cache_resident_bytes_max"],
        "evictions_max": max((int(r.get("cache_evictions", 0) or 0) for r in ok), default=0),
        "speedup_vs_native_e2e": (
            native["end_to_end_mean_ms"] / summary["end_to_end_mean_ms"]
            if summary["end_to_end_mean_ms"]
            else 0.0
        ),
    }


def add_cache_evictions(cur: psycopg.Cursor, row: dict[str, object]) -> None:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    profile = json.loads(cur.fetchone()[0])
    row["cache_evictions"] = int(profile.get("evictions", 0) or 0)


def run_managed_with_evictions(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    workload: list[dict[str, object]],
    budget_mb: int,
) -> list[dict[str, object]]:
    rows = run_mode(cur, args, workload, "managed_cache", budget_mb)
    # run_mode records resident size but not cumulative evictions. Fetch the
    # final value once and attach it to rows so the summary has an audit trail.
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    profile = json.loads(cur.fetchone()[0])
    evictions = int(profile.get("evictions", 0) or 0)
    for row in rows:
        row["cache_evictions"] = evictions
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep C4 D3 managed-cache memory budgets with filter_strategy=off.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--cache-kind", default="bloom", choices=["page", "bloom"])
    parser.add_argument("--budgets-mb", default="1,2,4,8,16")
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
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--progress-queries", type=int, default=100)
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets_mb.split(",") if x.strip()]
    workload = load_workload(args.query_csv, args.queries, args.mode)
    cache_keys = sorted({str(row["cache_key"]) for row in workload})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb, args.filter_strategy)
        reset_store(cur)
        prebuild_wall_ms, _ = prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        store = store_summary(cur)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        native_rows = run_mode(conn.cursor(), args, workload, "native", None)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb, args.filter_strategy)
        load_all_memory(cur, args.index, cache_keys, args.cache_kind)
        all_memory_rows = run_mode(cur, args, workload, "all_memory", args.all_memory_mb)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    summaries.append(summarize_budget("all", native_rows, all_memory_rows))

    for row in native_rows:
        row = dict(row)
        row["budget_mb"] = "native"
        row["cache_evictions"] = 0
        all_rows.append(row)

    for row in all_memory_rows:
        row = dict(row)
        row["budget_mb"] = "all"
        row["cache_evictions"] = 0
        all_rows.append(row)

    for budget in budgets:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            managed_rows = run_managed_with_evictions(conn.cursor(), args, workload, budget)
        summaries.append(summarize_budget(budget, native_rows, managed_rows))
        for row in managed_rows:
            row = dict(row)
            row["budget_mb"] = budget
            all_rows.append(row)
        print(
            f"budget={budget}MB e2e={summaries[-1]['e2e_mean_ms']:.2f} "
            f"p95={summaries[-1]['e2e_p95_ms']:.2f} "
            f"resident={summaries[-1]['resident_entries_mean']:.2f} "
            f"match={summaries[-1]['ordered_match']}/{summaries[-1]['queries']}",
            flush=True,
        )

    raw_fields = ["budget_mb", *list(native_rows[0].keys()), "cache_evictions"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    summary_fields = list(summaries[0].keys())
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    meta = {
        "prebuild_wall_ms": prebuild_wall_ms,
        "ssd_payload_bytes": store["ssd_payload_bytes"],
        "ssd_fragments": store["ssd_fragments"],
        "budgets_mb": budgets,
        "cache_keys": len(cache_keys),
        "native": summarize_rows(native_rows),
    }
    meta_out = args.out.with_name(args.out.stem + "_meta.json")
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {summary_out}")
    print(f"wrote {meta_out}")


if __name__ == "__main__":
    main()
