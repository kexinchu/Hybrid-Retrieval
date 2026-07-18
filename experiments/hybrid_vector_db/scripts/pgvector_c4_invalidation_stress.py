from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import psycopg

from common_pg import pg_config_from_env
from pgvector_c4_guidance_memory_benchmark import INDEX, TABLE, activate, configure, run_query
from pgvector_c4_query_filter_cache_benchmark import load_workload


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def mutation_values(cache_key: str) -> dict[str, object]:
    values: dict[str, object] = {"rating": 5.0}
    if "_p_le10_" in cache_key:
        values.update({"has_price": True, "price": 5.0})
    elif "_p_10_20_" in cache_key:
        values.update({"has_price": True, "price": 15.0})
    elif "_p_20_50_" in cache_key:
        values.update({"has_price": True, "price": 30.0})
    elif "_p_gt50_" in cache_key:
        values.update({"has_price": True, "price": 60.0})
    elif "_p_missing_" in cache_key:
        values.update({"has_price": False, "price": 0.0})
    else:
        raise ValueError(f"unsupported C4 cache key for mutation: {cache_key}")

    if cache_key.endswith("_pop_high"):
        values["item_rating_number"] = 1000
    elif cache_key.endswith("_pop_mid"):
        values["item_rating_number"] = 100
    elif cache_key.endswith("_pop_low"):
        values["item_rating_number"] = 0
    else:
        raise ValueError(f"unsupported C4 cache key for mutation: {cache_key}")
    return values


def update_ids(cur: psycopg.Cursor, table: str, ids: list[int], values: dict[str, object]) -> float:
    if not ids:
        return 0.0
    assignments = ", ".join(f"{k} = %s" for k in values)
    params = [values[k] for k in values]
    params.append(ids)
    _, wall_ms = timed_ms(
        lambda: cur.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ANY(%s)",
            params,
        )
    )
    return wall_ms


def refresh_guidance_meta(cur: psycopg.Cursor, table: str, ids: list[int], values: dict[str, object]) -> float:
    if not ids:
        return 0.0
    meta = f"{table}_guidance_meta"
    assignments = ["heap_tid = base.ctid"]
    assignments.extend(f"{k} = base.{k}" for k in values)
    _, wall_ms = timed_ms(
        lambda: cur.execute(
            f"""
            UPDATE {meta} AS meta
            SET {", ".join(assignments)}
            FROM {table} AS base
            WHERE meta.id = base.id
              AND meta.id = ANY(%s)
            """,
            (ids,),
        )
    )
    return wall_ms


def invalidate_fragment(cur: psycopg.Cursor, index: str, cache_key: str, kind: str) -> float:
    def do_invalidate() -> None:
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
        cur.execute(
            """
            DELETE FROM public.pgvector_hnsw_fragment_store
            WHERE heap_oid = (SELECT indrelid FROM pg_index WHERE indexrelid = %s::regclass)
              AND filter_name = %s
              AND kind = %s
            """,
            (index, cache_key, kind),
        )

    _, wall_ms = timed_ms(do_invalidate)
    return wall_ms


def profile(cur: psycopg.Cursor, sql: str) -> dict[str, object]:
    cur.execute(sql)
    return json.loads(cur.fetchone()[0])


def run_case(cur: psycopg.Cursor, args: argparse.Namespace, row: dict[str, object]) -> dict[str, object]:
    predicate = str(row["predicate"])
    cache_key = str(row["cache_key"])
    query_id = int(row["query_id_int"])
    query_no = int(row["query_no"])
    values = mutation_values(cache_key)

    cur.execute("BEGIN")
    try:
        configure(cur, args, args.managed_memory_mb, "off")
        invalidate_fragment(cur, args.index, cache_key, args.cache_kind)

        _, stale_build_ms = timed_ms(lambda: activate(cur, args.index, cache_key, args.cache_kind))
        stale_build_profile = profile(cur, "SELECT vector_hnsw_guidance_profile()")
        cur.execute("SELECT vector_hnsw_guidance_reset()")

        cur.execute(
            f"""
            SELECT id
            FROM {args.table}
            WHERE NOT ({predicate})
            ORDER BY embedding <-> (SELECT embedding FROM {args.table} WHERE id = %s)
            LIMIT %s
            """,
            (query_id, int(args.mutations_per_query)),
        )
        mutated_ids = [int(r[0]) for r in cur.fetchall()]
        base_update_ms = update_ids(cur, args.table, mutated_ids, values)
        meta_update_ms = refresh_guidance_meta(cur, args.table, mutated_ids, values)

        cur.execute("SELECT vector_hnsw_guidance_reset()")
        native_ids, native_profile, native_error = run_query(cur, args.table, predicate, query_id, args.k)

        # Re-activate without invalidating the backend cache or fragment store.
        _, stale_activate_ms = timed_ms(lambda: activate(cur, args.index, cache_key, args.cache_kind))
        stale_ids, stale_profile, stale_error = run_query(cur, args.table, predicate, query_id, args.k)

        invalidation_ms = invalidate_fragment(cur, args.index, cache_key, args.cache_kind)
        _, rebuild_activate_ms = timed_ms(lambda: activate(cur, args.index, cache_key, args.cache_kind))
        rebuild_profile = profile(cur, "SELECT vector_hnsw_guidance_profile()")
        rebuild_ids, rebuild_scan_profile, rebuild_error = run_query(cur, args.table, predicate, query_id, args.k)

        native_text = ",".join(str(x) for x in native_ids)
        stale_text = ",".join(str(x) for x in stale_ids)
        rebuild_text = ",".join(str(x) for x in rebuild_ids)
        return {
            "query_no": query_no,
            "query_id": query_id,
            "cache_key": cache_key,
            "predicate": predicate,
            "mutated_rows": len(mutated_ids),
            "mutated_ids": ",".join(str(x) for x in mutated_ids),
            "base_update_ms": base_update_ms,
            "meta_update_ms": meta_update_ms,
            "pre_update_build_ms": stale_build_ms,
            "pre_update_build_profile_ms": stale_build_profile.get("last_cache_build_ms", 0.0),
            "stale_activate_ms": stale_activate_ms,
            "invalidation_ms": invalidation_ms,
            "rebuild_activate_ms": rebuild_activate_ms,
            "rebuild_profile_build_ms": rebuild_profile.get("last_cache_build_ms", 0.0),
            "native_ids": native_text,
            "stale_ids": stale_text,
            "rebuild_ids": rebuild_text,
            "stale_ordered_match": stale_text == native_text,
            "rebuild_ordered_match": rebuild_text == native_text,
            "native_error": native_error,
            "stale_error": stale_error,
            "rebuild_error": rebuild_error,
            "native_returned_tuples": native_profile.get("returned_tuples", 0),
            "stale_returned_tuples": stale_profile.get("returned_tuples", 0),
            "rebuild_returned_tuples": rebuild_scan_profile.get("returned_tuples", 0),
            "stale_guidance_skips": stale_profile.get("guidance_skips", 0),
            "rebuild_guidance_skips": rebuild_scan_profile.get("guidance_skips", 0),
        }
    finally:
        cur.execute("ROLLBACK")
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "cases": len(rows),
        "mutated_rows_mean": mean("mutated_rows"),
        "stale_ordered_match": sum(1 for r in rows if str(r["stale_ordered_match"]).lower() == "true"),
        "rebuild_ordered_match": sum(1 for r in rows if str(r["rebuild_ordered_match"]).lower() == "true"),
        "base_update_mean_ms": mean("base_update_ms"),
        "meta_update_mean_ms": mean("meta_update_ms"),
        "pre_update_build_mean_ms": mean("pre_update_build_ms"),
        "stale_activate_mean_ms": mean("stale_activate_ms"),
        "invalidation_mean_ms": mean("invalidation_ms"),
        "rebuild_activate_mean_ms": mean("rebuild_activate_ms"),
        "rebuild_profile_build_mean_ms": mean("rebuild_profile_build_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative invalidation stress for C4 D3 fragments.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--cache-kind", default="bloom", choices=["page", "bloom"])
    parser.add_argument("--managed-memory-mb", type=int, default=16)
    parser.add_argument("--mutations-per-query", type=int, default=20)
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
    rows: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        for i, row in enumerate(workload, start=1):
            result = run_case(cur, args, row)
            rows.append(result)
            print(
                f"{i}/{len(workload)} {result['cache_key']} stale_match={result['stale_ordered_match']} "
                f"rebuild_match={result['rebuild_ordered_match']} rebuild_ms={float(result['rebuild_activate_ms']):.2f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summarize(rows), indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {summary_out}")
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
