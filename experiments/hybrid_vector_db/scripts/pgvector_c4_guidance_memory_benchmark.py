from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from pgvector_c4_query_filter_cache_benchmark import load_workload


TABLE = "amazon_grocery_reviews_10m_pgvector"
INDEX = f"{TABLE}_embedding_hnsw_idx"


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def configure(cur: psycopg.Cursor, args: argparse.Namespace, memory_mb: int | None = None, filter_strategy: str | None = None) -> None:
    # Force-load the vector extension library before setting hnsw.* GUCs.
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    if filter_strategy is not None:
        cur.execute(f"SET hnsw.filter_strategy = {filter_strategy}")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")
    if memory_mb is not None:
        cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(memory_mb)}")


def run_query(cur: psycopg.Cursor, table: str, predicate: str, query_id: int, k: int):
    try:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            WHERE {predicate}
            ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
            LIMIT {int(k)}
            """,
            (int(query_id),),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
        error = ""
    except errors.QueryCanceled as exc:
        ids = []
        error = exc.__class__.__name__
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile = json.loads(cur.fetchone()[0])
    return ids, profile, error


def activate(cur: psycopg.Cursor, index: str, cache_key: str, kind: str) -> dict[str, object]:
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (index, [cache_key], kind),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    return json.loads(cur.fetchone()[0])


def cache_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    return json.loads(cur.fetchone()[0])


def reset_store(cur: psycopg.Cursor) -> None:
    cur.execute("DROP TABLE IF EXISTS public.pgvector_hnsw_fragment_store")


def store_summary(cur: psycopg.Cursor) -> dict[str, int]:
    cur.execute(
        """
        SELECT coalesce(sum(octet_length(payload)), 0)::bigint,
               count(*)::bigint
        FROM public.pgvector_hnsw_fragment_store
        """
    )
    payload_bytes, fragments = cur.fetchone()
    return {"ssd_payload_bytes": int(payload_bytes), "ssd_fragments": int(fragments)}


def prebuild_store(cur: psycopg.Cursor, index: str, cache_keys: list[str], kind: str) -> tuple[float, dict[str, dict[str, object]]]:
    profiles: dict[str, dict[str, object]] = {}
    build_total = 0.0
    for cache_key in cache_keys:
        _, wall_ms = timed_ms(lambda ck=cache_key: activate(cur, index, ck, kind))
        cur.execute("SELECT vector_hnsw_guidance_profile()")
        profile = json.loads(cur.fetchone()[0])
        profiles[cache_key] = profile
        build_total += wall_ms
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    return build_total, profiles


def load_all_memory(cur: psycopg.Cursor, index: str, cache_keys: list[str], kind: str) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for cache_key in cache_keys:
        profiles[cache_key] = activate(cur, index, cache_key, kind)
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    return profiles


def run_mode(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    workload: list[dict[str, object]],
    mode: str,
    memory_mb: int | None,
) -> list[dict[str, object]]:
    filter_strategy = "off" if mode == "native" else args.filter_strategy
    configure(cur, args, memory_mb, filter_strategy)
    rows: list[dict[str, object]] = []
    for query_no, row in enumerate(workload):
        predicate = str(row["predicate"])
        cache_key = str(row["cache_key"])
        query_id = int(row["query_id_int"])
        activation_profile: dict[str, object] = {}
        resident_profile: dict[str, object] = {}
        activation_ms = 0.0
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
                "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
                "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
                "cache_entries": resident_profile.get("entries", 0),
                "cache_resident_entries": resident_profile.get("resident_entries", 0),
                "cache_resident_bytes": resident_profile.get("resident_bytes", 0),
                "cache_largest_entry_bytes": resident_profile.get("largest_entry_bytes", 0),
                "cache_budget_bytes": resident_profile.get("budget_bytes", 0),
            }
        )
        if args.stream and (query_no + 1) % args.progress_queries == 0:
            ok = [r for r in rows if not r["error"]]
            print(
                f"{mode} progress {query_no + 1}/{len(workload)} "
                f"lat={statistics.fmean(float(r['latency_ms']) for r in ok):.2f} "
                f"e2e={statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}",
                flush=True,
            )
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    ok = [r for r in rows if not r["error"]]

    def mean(key: str) -> float:
        return statistics.fmean(float(r[key]) for r in ok) if ok else 0.0

    def median(key: str) -> float:
        return statistics.median(float(r[key]) for r in ok) if ok else 0.0

    def p95(key: str) -> float:
        if not ok:
            return 0.0
        vals = sorted(float(r[key]) for r in ok)
        return vals[int(0.95 * (len(vals) - 1))]

    checks = mean("guidance_checks")
    skips = mean("guidance_skips")
    return {
        "rows": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "latency_mean_ms": mean("latency_ms"),
        "latency_p50_ms": median("latency_ms"),
        "latency_p95_ms": p95("latency_ms"),
        "activation_mean_ms": mean("activation_ms"),
        "end_to_end_mean_ms": mean("end_to_end_ms"),
        "end_to_end_p95_ms": p95("end_to_end_ms"),
        "returned_mean": mean("returned"),
        "visited_tuples_mean": mean("visited_tuples"),
        "returned_tuples_mean": mean("returned_tuples"),
        "guidance_checks_mean": checks,
        "guidance_skips_mean": skips,
        "guidance_skip_rate": skips / checks if checks else 0.0,
        "activation_build_ms_max": max((float(r["activation_build_ms"]) for r in ok), default=0.0),
        "cache_resident_bytes_mean": mean("cache_resident_bytes"),
        "cache_resident_bytes_max": max((int(r["cache_resident_bytes"]) for r in ok), default=0),
        "cache_resident_entries_mean": mean("cache_resident_entries"),
        "cache_resident_entries_max": max((int(r["cache_resident_entries"]) for r in ok), default=0),
    }


def compare_same_ids(rows_by_mode: dict[str, list[dict[str, object]]]) -> dict[str, int]:
    native = rows_by_mode["native"]
    out: dict[str, int] = {}
    for mode, rows in rows_by_mode.items():
        if mode == "native":
            continue
        same_ordered = 0
        same_set = 0
        for base, other in zip(native, rows, strict=True):
            if base["ids"] == other["ids"]:
                same_ordered += 1
            if set(str(base["ids"]).split(",")) == set(str(other["ids"]).split(",")):
                same_set += 1
        out[f"{mode}_same_ordered_ids"] = same_ordered
        out[f"{mode}_same_set_ids"] = same_set
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native pgvector, all-memory guidance, and LRU-managed guidance on Amazon-C4 filters.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--cache-kind", default="bloom", choices=["page", "bloom"])
    parser.add_argument("--all-memory-mb", type=int, default=1024)
    parser.add_argument("--managed-memory-mb", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--filter-strategy", default="acorn1", choices=["off", "acorn1"])
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--fragments-out", type=Path)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--progress-queries", type=int, default=50)
    args = parser.parse_args()

    workload = load_workload(args.query_csv, args.queries, args.mode)
    cache_keys = sorted({str(row["cache_key"]) for row in workload})
    predicate_counts = collections.Counter(str(row["predicate"]) for row in workload)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows_by_mode: dict[str, list[dict[str, object]]] = {}
    fragment_profiles: dict[str, dict[str, object]] = {}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb)
        reset_store(cur)
        prebuild_wall_ms, fragment_profiles = prebuild_store(cur, args.index, cache_keys, args.cache_kind)
        store = store_summary(cur)
        all_memory_profiles = load_all_memory(cur, args.index, cache_keys, args.cache_kind)
        all_memory_bytes = sum(int(p.get("last_cache_memory_bytes", 0)) for p in all_memory_profiles.values())

    # Use separate connections to make native/all-memory/managed memory states clean.
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        rows_by_mode["native"] = run_mode(conn.cursor(), args, workload, "native", None)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        configure(cur, args, args.all_memory_mb)
        load_all_memory(cur, args.index, cache_keys, args.cache_kind)
        rows_by_mode["all_memory"] = run_mode(cur, args, workload, "all_memory", args.all_memory_mb)

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        rows_by_mode["managed_cache"] = run_mode(conn.cursor(), args, workload, "managed_cache", args.managed_memory_mb)

    fieldnames = list(next(iter(rows_by_mode.values()))[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in ["native", "all_memory", "managed_cache"]:
            writer.writerows(rows_by_mode[mode])

    fragments_out = args.fragments_out or args.out.with_name(args.out.stem + "_fragments.csv")
    with fragments_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cache_key",
                "frequency",
                "rows",
                "pages",
                "memory_bytes",
                "build_ms",
            ],
        )
        writer.writeheader()
        for cache_key in cache_keys:
            p = fragment_profiles.get(cache_key, {})
            writer.writerow(
                {
                    "cache_key": cache_key,
                    "frequency": sum(1 for row in workload if str(row["cache_key"]) == cache_key),
                    "rows": p.get("last_cache_rows", 0),
                    "pages": p.get("last_cache_pages", 0),
                    "memory_bytes": p.get("last_cache_memory_bytes", 0),
                    "build_ms": p.get("last_cache_build_ms", 0.0),
                }
            )

    summary = {
        "table": args.table,
        "index": args.index,
        "queries": len(workload),
        "mode": args.mode,
        "cache_kind": args.cache_kind,
        "distinct_predicates": len(predicate_counts),
        "distinct_fragments": len(cache_keys),
        "prebuild_wall_ms": prebuild_wall_ms,
        "ssd_payload_bytes": store["ssd_payload_bytes"],
        "ssd_fragments": store["ssd_fragments"],
        "all_memory_budget_mb": args.all_memory_mb,
        "all_memory_loaded_bytes": all_memory_bytes,
        "managed_memory_budget_mb": args.managed_memory_mb,
        "native": summarize_rows(rows_by_mode["native"]),
        "all_memory": summarize_rows(rows_by_mode["all_memory"]),
        "managed_cache": summarize_rows(rows_by_mode["managed_cache"]),
        **compare_same_ids(rows_by_mode),
    }
    native_mean = summary["native"]["end_to_end_mean_ms"]
    summary["all_memory"]["speedup_vs_native_e2e"] = native_mean / summary["all_memory"]["end_to_end_mean_ms"] if summary["all_memory"]["end_to_end_mean_ms"] else 0.0
    summary["managed_cache"]["speedup_vs_native_e2e"] = native_mean / summary["managed_cache"]["end_to_end_mean_ms"] if summary["managed_cache"]["end_to_end_mean_ms"] else 0.0

    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {fragments_out}")
    print(f"wrote {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
