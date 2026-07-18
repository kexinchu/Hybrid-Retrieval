from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from yfcc_pgvector_filtered_benchmark import parse_int_array, recall_at_k


METHODS = ["stock", "d1", "d1_d2", "d1_d2_d3"]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_functions(cur: psycopg.Cursor) -> None:
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in functions:
        try:
            cur.execute(sql)
        except Exception as exc:  # noqa: BLE001 - parallel workers may race on pg_proc
            if "tuple concurrently updated" not in str(exc):
                raise
            cur.connection.rollback()
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def pin_backend(cur: psycopg.Cursor, cpu_list: str) -> None:
    if not cpu_list:
        return
    cur.execute("SELECT pg_backend_pid()")
    pid = int(cur.fetchone()[0])
    result = subprocess.run(["taskset", "-pc", cpu_list, str(pid)], check=False, capture_output=True, text=True)
    print(f"pinned PostgreSQL backend pid={pid} cpus={cpu_list} rc={result.returncode}", flush=True)


def parse_selected(path: Path, targets: list[float], limit_per_group: int) -> list[dict[str, Any]]:
    target_set = {float(x) for x in targets} if targets else set()
    rows: list[dict[str, Any]] = []
    per_group: dict[float, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = float(row["target_band_pct"])
            if target_set and target not in target_set:
                continue
            if limit_per_group > 0 and per_group.get(target, 0) >= limit_per_group:
                continue
            per_group[target] = per_group.get(target, 0) + 1
            row["target_band_pct"] = target
            row["filter_pct"] = float(row["filter_pct"])
            row["filter_rows"] = int(row["filter_rows"])
            row["qid"] = int(row["qid"])
            row["tags_list"] = parse_int_array(row["tags"])
            row["gt_list"] = parse_int_array(row["gt"])
            rows.append(row)
    if not rows:
        raise SystemExit("no selected YFCC overlap rows after filtering")
    return rows


def tag_atoms(tags: list[int]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        if out:
            out.append("|")
        out.append(f"sql:tags @> ARRAY[{int(tag)}]")
    return out


def should_enable_guidance(args: argparse.Namespace, row: dict[str, Any], method: str) -> tuple[bool, str]:
    filter_pct = float(row.get("filter_pct", 100.0) or 100.0)
    atom_count = len(row.get("tags_list") or [])
    max_atoms = int(args.d3_guidance_max_atoms if method == "d1_d2_d3" else args.guidance_max_atoms)
    if filter_pct > float(args.guidance_selectivity_max_pct):
        return False, f"selectivity>{args.guidance_selectivity_max_pct:g}%"
    if atom_count > max_atoms:
        return False, f"atoms>{max_atoms}"
    return True, "enabled"


def method_table_index(args: argparse.Namespace, method: str) -> tuple[str, str]:
    if method in {"d1_d2", "d1_d2_d3"}:
        return args.bfs_table, args.bfs_index
    return args.stock_table, args.stock_index


def method_uses_d2(method: str) -> bool:
    return method in {"d1_d2", "d1_d2_d3"}


def configure(cur: psycopg.Cursor, args: argparse.Namespace, method: str) -> None:
    cur.execute("SET jit = off")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(args.guided_collect_target)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(args.d3_cache_mb if method == 'd1_d2_d3' else args.d1_cache_mb)}")
    cur.execute(f"SET hnsw.page_access = {args.d2_page_access if method_uses_d2(method) else 'off'}")
    cur.execute(f"SET hnsw.index_page_access = {args.d2_index_page_access if method_uses_d2(method) else 'off'}")
    cur.execute(f"SET hnsw.page_window = {int(args.d2_page_window)}")
    cur.execute(f"SET hnsw.page_prefetch_min_items = {int(args.d2_page_prefetch_min_items)}")
    cur.execute(f"SET hnsw.page_disable_after_no_merge = {int(args.d2_page_disable_after_no_merge)}")
    cur.execute(f"SET hnsw.filter_strategy = {'off' if method == 'stock' else 'guided_collect'}")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def fetch_json(cur: psycopg.Cursor, sql: str) -> dict[str, Any]:
    cur.execute(sql)
    value = cur.fetchone()[0]
    return json.loads(value) if isinstance(value, str) else dict(value)


def activate_guidance(cur: psycopg.Cursor, args: argparse.Namespace, method: str, row: dict[str, Any]) -> tuple[dict[str, Any], float]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    table, index = method_table_index(args, method)
    if method == "stock":
        cur.execute("SET hnsw.filter_strategy = off")
        return {"guidance_enabled": False, "guidance_route": "stock", "table": table, "index": index}, 0.0
    enabled, route = should_enable_guidance(args, row, method)
    if not enabled:
        cur.execute("SET hnsw.filter_strategy = off")
        return {"guidance_enabled": False, "guidance_route": route, "table": table, "index": index}, 0.0
    cur.execute("SET hnsw.filter_strategy = guided_collect")
    atoms = tag_atoms(row["tags_list"])

    def run():
        cur.execute(
            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
            (index, atoms, args.guidance_kind),
        )
        return fetch_json(cur, "SELECT vector_hnsw_guidance_profile()")

    profile, elapsed_ms = timed_ms(run)
    profile["guidance_enabled"] = True
    profile["guidance_route"] = route
    profile["table"] = table
    profile["index"] = index
    return profile, elapsed_ms


def d3_guidance_signature(args: argparse.Namespace, method: str, row: dict[str, Any]) -> tuple[str, str, tuple[str, ...]] | None:
    if method != "d1_d2_d3" or not args.d3_reuse_active_guidance:
        return None
    enabled, _ = should_enable_guidance(args, row, method)
    if not enabled:
        return None
    table, index = method_table_index(args, method)
    return table, index, tuple(tag_atoms(row["tags_list"]))


def reuse_activation_profile(args: argparse.Namespace, method: str, profile: dict[str, Any]) -> dict[str, Any]:
    table, index = method_table_index(args, method)
    reused = dict(profile)
    reused["table"] = table
    reused["index"] = index
    reused["guidance_enabled"] = True
    reused["guidance_route"] = "d3_reuse_active_guidance"
    reused["d3_active_guidance_reused"] = True
    return reused


def prewarm_d3(cur: psycopg.Cursor, args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    if not args.prewarm_d3:
        return
    seen: set[tuple[int, ...]] = set()
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    configure(cur, args, "d1_d2_d3")
    for row in selected:
        if not should_enable_guidance(args, row, "d1_d2_d3")[0]:
            continue
        key = tuple(row["tags_list"])
        if key in seen:
            continue
        seen.add(key)
        _, _ = activate_guidance(cur, args, "d1_d2_d3", row)
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    print(f"prewarmed D3 guidance for {len(seen)} unique filters", flush=True)


def run_query(cur: psycopg.Cursor, args: argparse.Namespace, method: str, row: dict[str, Any]) -> tuple[list[int], float, dict[str, Any], str]:
    table, _ = method_table_index(args, method)
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")

    def execute():
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            WHERE tags && %s::int[]
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (row["tags_list"], int(row["qid"])),
        )
        return [int(x[0]) for x in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        return ids, latency_ms, fetch_json(cur, "SELECT vector_hnsw_last_scan_profile()"), ""
    except errors.QueryCanceled as exc:
        cur.connection.rollback()
        configure(cur, args, method)
        return [], float(args.statement_timeout_ms), {}, exc.__class__.__name__


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((float(row["target_band_pct"]), str(row["method"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (target, method), items in sorted(groups.items()):
        ok = [r for r in items if not r.get("error")]
        if not ok:
            continue

        def vals(key: str) -> list[float]:
            return [float(row.get(key, 0) or 0) for row in ok]

        checks = statistics.fmean(vals("guidance_checks")) if ok else 0.0
        skips = statistics.fmean(vals("guidance_skips")) if ok else 0.0
        out.append(
            {
                "target_band_pct": target,
                "method": method,
                "ef_search": items[0].get("ef_search", ""),
                "max_scan_tuples": items[0].get("max_scan_tuples", ""),
                "guided_collect_target": items[0].get("guided_collect_target", ""),
                "scan_mem_multiplier": items[0].get("scan_mem_multiplier", ""),
                "queries": len(ok),
                "filter_pct_mean": statistics.fmean(vals("filter_pct")),
                "filter_rows_mean": statistics.fmean(vals("filter_rows")),
                "recall_mean": statistics.fmean(vals("recall")),
                "latency_ms_mean": statistics.fmean(vals("latency_ms")),
                "end_to_end_ms_mean": statistics.fmean(vals("end_to_end_ms")),
                "activation_ms_mean": statistics.fmean(vals("activation_ms")),
                "guidance_enabled_rate": statistics.fmean(vals("guidance_enabled")),
                "composed_exact_active_rate": statistics.fmean(vals("composed_exact_active")),
                "composed_exact_hit_rate": statistics.fmean(vals("composed_exact_hit")),
                "composed_exact_build_ms_mean": statistics.fmean(vals("composed_exact_build_ms")),
                "composed_exact_rows_mean": statistics.fmean(vals("composed_exact_rows")),
                "cache_composed_exact_bytes_mean": statistics.fmean(vals("cache_composed_exact_bytes")),
                "vector_search_ms_mean": statistics.fmean(vals("vector_search_ms")),
                "visited_tuples_mean": statistics.fmean(vals("visited_tuples")),
                "returned_tuples_mean": statistics.fmean(vals("returned_tuples")),
                "distance_compute_count_mean": statistics.fmean(vals("distance_compute_count")),
                "idx_blks_hit_mean": statistics.fmean(vals("idx_blks_hit")),
                "idx_blks_read_mean": statistics.fmean(vals("idx_blks_read")),
                "heap_blks_hit_mean": statistics.fmean(vals("heap_blks_hit")),
                "heap_blks_read_mean": statistics.fmean(vals("heap_blks_read")),
                "index_page_neighbor_distinct_pages_mean": statistics.fmean(vals("index_page_neighbor_distinct_pages")),
                "index_page_element_distinct_pages_mean": statistics.fmean(vals("index_page_element_distinct_pages")),
                "index_page_prefetches_mean": statistics.fmean(vals("index_page_prefetches")),
                "page_access_batches_mean": statistics.fmean(vals("page_access_batches")),
                "page_access_prefetches_mean": statistics.fmean(vals("page_access_prefetches")),
                "page_access_distinct_pages_mean": statistics.fmean(vals("page_access_distinct_pages")),
                "guidance_skip_rate_mean": skips / checks if checks else 0.0,
                "errors": len(items) - len(ok),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YFCC overlap Stock/D1/D1+D2/D1+D2+D3 variants.")
    parser.add_argument("--stock-table", default="yfcc10m_pgvector")
    parser.add_argument("--stock-index", default="yfcc10m_pgvector_embedding_hnsw")
    parser.add_argument("--bfs-table", default="yfcc10m_pgvector_bfs")
    parser.add_argument("--bfs-index", default="yfcc10m_pgvector_bfs_embedding_hnsw")
    parser.add_argument("--query-table", default="yfcc10m_queries")
    parser.add_argument("--selected-queries-in", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_overlap_selectivity14_selected_q100_20260716.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_overlap_sqlens_variants_20260716.csv"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--target-bands", type=float, nargs="+", default=[])
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--guided-collect-target", type=int, default=100)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--d1-cache-mb", type=int, default=4096)
    parser.add_argument("--d3-cache-mb", type=int, default=4096)
    parser.add_argument("--guidance-kind", default="exact", choices=["exact", "page", "bloom"])
    parser.add_argument("--d3-reuse-active-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=float,
        default=10.0,
        help="Disable predicate guidance above this actual filter percentage; D1+D2 then runs as D2-only.",
    )
    parser.add_argument(
        "--guidance-max-atoms",
        type=int,
        default=1,
        help="Disable D1/D2 predicate guidance when a query decomposes into more atoms than this.",
    )
    parser.add_argument(
        "--d3-guidance-max-atoms",
        type=int,
        default=64,
        help="Disable D3 predicate guidance when a query decomposes into more atoms than this.",
    )
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prewarm-d3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--warmup-all-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one unmeasured pass over every selected query for each method before recording latency.",
    )
    parser.add_argument("--progress-queries", type=int, default=50)
    parser.add_argument("--backend-cpu-list", default="")
    args = parser.parse_args()

    selected = parse_selected(args.selected_queries_in, args.target_bands, args.limit_per_group)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        pin_backend(cur, args.backend_cpu_list)
        if "d1_d2_d3" in args.methods:
            prewarm_d3(cur, args, selected)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "method",
                "target_band_pct",
                "filter_pct",
                "filter_rows",
                "filter_name",
                "qid",
                "tags",
                "ef_search",
                "max_scan_tuples",
                "guided_collect_target",
                "scan_mem_multiplier",
                "repeat",
                "recall",
                "latency_ms",
                "activation_ms",
                "end_to_end_ms",
                "activation_build_ms",
                "guidance_enabled",
                "guidance_route",
                "d3_active_guidance_reused",
                "fragment_cache_hits",
                "fragment_cache_misses",
                "fragment_store_hits",
                "fragment_builds",
                "composed_guide_hit",
                "composed_exact_active",
                "composed_exact_hit",
                "composed_exact_rows",
                "composed_exact_memory_bytes",
                "composed_exact_build_ms",
                "cache_composed_exact_entries",
                "cache_composed_exact_rows",
                "cache_composed_exact_bytes",
                "cache_composed_exact_hits",
                "cache_resident_bytes",
                "cache_resident_entries",
                "cache_evictions",
                "vector_search_ms",
                "visited_tuples",
                "returned_tuples",
                "distance_compute_count",
                "guidance_checks",
                "guidance_skips",
                "guidance_skip_rate",
                "index_page_neighbor_distinct_pages",
                "index_page_element_distinct_pages",
                "index_page_prefetches",
                "page_access_batches",
                "page_access_candidates",
                "page_access_prefetches",
                "page_access_distinct_pages",
                "idx_blks_hit",
                "idx_blks_read",
                "heap_blks_hit",
                "heap_blks_read",
                "returned",
                "ids",
                "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for method in args.methods:
                configure(cur, args, method)
                active_signature: tuple[str, str, tuple[str, ...]] | None = None
                active_profile: dict[str, Any] = {}
                if args.warmup_all_queries:
                    for qno, query in enumerate(selected, start=1):
                        try:
                            activate_guidance(cur, args, method, query)
                            run_query(cur, args, method, query)
                        except Exception:
                            try:
                                cur.execute("ROLLBACK")
                            except Exception:
                                pass
                            configure(cur, args, method)
                        if args.progress_queries and qno % args.progress_queries == 0:
                            print(f"{method} warmup {qno}/{len(selected)}", flush=True)
                for qno, query in enumerate(selected, start=1):
                    signature = d3_guidance_signature(args, method, query)
                    if signature is not None and signature == active_signature:
                        activation_profile = reuse_activation_profile(args, method, active_profile)
                        activation_ms = 0.0
                    else:
                        activation_profile, activation_ms = activate_guidance(cur, args, method, query)
                        active_signature = signature
                        active_profile = dict(activation_profile) if signature is not None else {}
                    for repeat in range(args.repeats):
                        ids, latency_ms, scan_profile, error = run_query(cur, args, method, query)
                        try:
                            cache_profile = fetch_json(cur, "SELECT vector_hnsw_metadata_cache_profile()")
                        except Exception:
                            cache_profile = {}
                        checks = float(scan_profile.get("guidance_checks", 0) or 0)
                        skips = float(scan_profile.get("guidance_skips", 0) or 0)
                        row = {
                            "method": method,
                            "target_band_pct": float(query["target_band_pct"]),
                            "filter_pct": float(query["filter_pct"]),
                            "filter_rows": int(query["filter_rows"]),
                            "filter_name": query["filter_name"],
                            "qid": int(query["qid"]),
                            "tags": query["tags"],
                            "ef_search": int(args.ef_search),
                            "max_scan_tuples": int(args.max_scan_tuples),
                            "guided_collect_target": int(args.guided_collect_target),
                            "scan_mem_multiplier": float(args.scan_mem_multiplier),
                            "repeat": repeat,
                            "recall": recall_at_k(ids, query["gt_list"], args.k) if not error else 0.0,
                            "latency_ms": latency_ms,
                            "activation_ms": activation_ms,
                            "end_to_end_ms": activation_ms + latency_ms,
                            "activation_build_ms": float(activation_profile.get("last_cache_build_ms", 0) or 0),
                            "guidance_enabled": bool(activation_profile.get("guidance_enabled", method != "stock")),
                            "guidance_route": str(activation_profile.get("guidance_route", "")),
                            "d3_active_guidance_reused": bool(activation_profile.get("d3_active_guidance_reused", False)),
                            "fragment_cache_hits": int(activation_profile.get("fragment_cache_hits", 0) or 0),
                            "fragment_cache_misses": int(activation_profile.get("fragment_cache_misses", 0) or 0),
                            "fragment_store_hits": int(activation_profile.get("fragment_store_hits", 0) or 0),
                            "fragment_builds": int(activation_profile.get("fragment_builds", 0) or 0),
                            "composed_guide_hit": bool(activation_profile.get("composed_guide_hit", False)),
                            "composed_exact_active": bool(activation_profile.get("composed_exact_active", False)),
                            "composed_exact_hit": bool(activation_profile.get("composed_exact_hit", False)),
                            "composed_exact_rows": int(activation_profile.get("composed_exact_rows", 0) or 0),
                            "composed_exact_memory_bytes": int(activation_profile.get("composed_exact_memory_bytes", 0) or 0),
                            "composed_exact_build_ms": float(activation_profile.get("composed_exact_build_ms", 0) or 0),
                            "cache_composed_exact_entries": int(cache_profile.get("composed_exact_entries", 0) or 0),
                            "cache_composed_exact_rows": int(cache_profile.get("composed_exact_rows", 0) or 0),
                            "cache_composed_exact_bytes": int(cache_profile.get("composed_exact_bytes", 0) or 0),
                            "cache_composed_exact_hits": int(cache_profile.get("composed_exact_hits", 0) or 0),
                            "cache_resident_bytes": int(cache_profile.get("resident_bytes", 0) or 0),
                            "cache_resident_entries": int(cache_profile.get("resident_entries", 0) or 0),
                            "cache_evictions": int(cache_profile.get("evictions", 0) or 0),
                            "vector_search_ms": float(scan_profile.get("vector_search_ms", 0) or 0),
                            "visited_tuples": float(scan_profile.get("visited_tuples", 0) or 0),
                            "returned_tuples": float(scan_profile.get("returned_tuples", 0) or 0),
                            "distance_compute_count": float(scan_profile.get("distance_compute_count", 0) or 0),
                            "guidance_checks": checks,
                            "guidance_skips": skips,
                            "guidance_skip_rate": skips / checks if checks else 0.0,
                            "index_page_neighbor_distinct_pages": float(scan_profile.get("index_page_neighbor_distinct_pages", 0) or 0),
                            "index_page_element_distinct_pages": float(scan_profile.get("index_page_element_distinct_pages", 0) or 0),
                            "index_page_prefetches": float(scan_profile.get("index_page_prefetches", 0) or 0),
                            "page_access_batches": float(scan_profile.get("page_access_batches", 0) or 0),
                            "page_access_candidates": float(scan_profile.get("page_access_candidates", 0) or 0),
                            "page_access_prefetches": float(scan_profile.get("page_access_prefetches", 0) or 0),
                            "page_access_distinct_pages": float(scan_profile.get("page_access_distinct_pages", 0) or 0),
                            "idx_blks_hit": float(scan_profile.get("idx_blks_hit", 0) or 0),
                            "idx_blks_read": float(scan_profile.get("idx_blks_read", 0) or 0),
                            "heap_blks_hit": float(scan_profile.get("heap_blks_hit", 0) or 0),
                            "heap_blks_read": float(scan_profile.get("heap_blks_read", 0) or 0),
                            "returned": len(ids),
                            "ids": ",".join(str(x) for x in ids),
                            "error": error,
                        }
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                    if args.progress_queries and qno % args.progress_queries == 0:
                        latest = [r for r in rows if r["method"] == method and not r["error"]]
                        print(
                            f"{method} progress {qno}/{len(selected)} "
                            f"e2e={statistics.fmean(float(r['end_to_end_ms']) for r in latest):.2f} "
                            f"recall={statistics.fmean(float(r['recall']) for r in latest):.3f}",
                            flush=True,
                        )
                cur.execute("SELECT vector_hnsw_guidance_reset()")

    summary = summarize(rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summary)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.out.with_name(args.out.stem + '_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
