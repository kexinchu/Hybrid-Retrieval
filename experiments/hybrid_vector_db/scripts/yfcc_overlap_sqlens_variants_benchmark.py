from __future__ import annotations

import argparse
import csv
import json
import math
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
SQLENS_V11_BUILD_PREFIX = "sqlens-v11-"
SQLENS_MIN_PROFILE_SEMANTICS = 4.0
SQLENS_PROFILE_FIELDS = (
    "graph_elements_visited",
    "raw_index_tids_returned",
    "hnsw_am_callback_ms",
    "executor_residual_ms",
)


class SqlensProvenanceGateError(RuntimeError):
    """Raised when the formal runner is not connected to the required SQLens ABI."""


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


def require_sqlens_provenance(cur: psycopg.Cursor) -> tuple[str, dict[str, Any]]:
    """Verify the loaded SQLens ABI before installing any C-backed SQL wrappers."""
    try:
        cur.execute("SELECT vector_sqlens_build_id()")
        row = cur.fetchone()
        build_id = str(row[0]) if row and row[0] is not None else ""
    except Exception as exc:  # noqa: BLE001 - the gate must turn missing SQL into an actionable failure
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_sqlens_build_id() is unavailable. "
            "Install/reload the SQLens v11 extension (and reconnect) before running this formal benchmark."
        ) from exc
    if not build_id.startswith(SQLENS_V11_BUILD_PREFIX):
        raise SqlensProvenanceGateError(
            f"SQLens v11 provenance gate failed: vector_sqlens_build_id() returned {build_id!r}; "
            f"expected the {SQLENS_V11_BUILD_PREFIX!r} prefix. "
            "Rebuild/reload the SQLens v11 extension and reconnect before running this formal benchmark."
        )

    try:
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        row = cur.fetchone()
        raw_profile = row[0] if row else None
        profile = json.loads(raw_profile) if isinstance(raw_profile, str) else dict(raw_profile)
    except Exception as exc:  # noqa: BLE001 - profile absence/ABI errors must fail closed
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is unavailable or is not valid JSON. "
            "Load the SQLens v11 extension and reconnect before running this formal benchmark."
        ) from exc
    if not isinstance(profile, dict):
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() did not return a JSON object. "
            "Load the SQLens v11 extension and reconnect before running this formal benchmark."
        )
    try:
        profile_version = float(profile["profile_semantics_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is missing a numeric "
            "profile_semantics_version. Load the SQLens v11 extension and reconnect."
        ) from exc
    missing = [field for field in SQLENS_PROFILE_FIELDS if field not in profile]
    if not math.isfinite(profile_version) or profile_version < SQLENS_MIN_PROFILE_SEMANTICS or missing:
        details = []
        if not math.isfinite(profile_version) or profile_version < SQLENS_MIN_PROFILE_SEMANTICS:
            details.append(
                f"profile_semantics_version={profile.get('profile_semantics_version')!r} "
                f"(need >= {SQLENS_MIN_PROFILE_SEMANTICS:g})"
            )
        if missing:
            details.append(f"missing fields={missing!r}")
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is incompatible: "
            + "; ".join(details)
            + ". Load the SQLens v11 extension and reconnect before running this formal benchmark."
        )
    return build_id, profile


def ensure_functions(cur: psycopg.Cursor) -> None:
    require_sqlens_provenance(cur)
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_bind(regclass, text[], text) "
        "RETURNS boolean AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_epoch_bump_trigger() "
        "RETURNS trigger AS 'vector' LANGUAGE C",
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_tracking_enable(regclass) "
        "RETURNS int8 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
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


def ensure_tracking(cur: psycopg.Cursor, *tables: str) -> None:
    for table in dict.fromkeys(tables):
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (table,))


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


def guidance_kind(args: argparse.Namespace, method: str) -> str:
    return "adaptive" if method == "d1_d2_d3" else args.guidance_kind


def warmup_enabled(args: argparse.Namespace, method: str) -> bool:
    return bool(args.warmup_all_queries) and method != "d1_d2_d3"


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
    cur.execute(f"SET hnsw.filter_strategy = {'off' if method == 'stock' else args.guidance_filter_strategy}")
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
    enabled, route = (True, "enabled") if method == "d1_d2_d3" else should_enable_guidance(args, row, method)
    if not enabled:
        cur.execute("SET hnsw.filter_strategy = off")
        return {"guidance_enabled": False, "guidance_route": route, "table": table, "index": index}, 0.0
    cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
    atoms = tag_atoms(row["tags_list"])

    def run():
        cur.execute(
            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
            (index, atoms, guidance_kind(args, method)),
        )
        activation_row = cur.fetchone()
        activated_atoms = int(activation_row[0]) if activation_row and activation_row[0] is not None else 0
        return fetch_json(cur, "SELECT vector_hnsw_guidance_profile()"), activated_atoms

    (profile, activated_atoms), elapsed_ms = timed_ms(run)
    profile["activation_atom_count"] = activated_atoms
    if method == "d1_d2_d3" and (activated_atoms <= 0 or not bool(profile.get("active", False))):
        cur.execute("SET hnsw.filter_strategy = off")
        profile["guidance_enabled"] = False
        profile["guidance_route"] = "d3_probe"
    else:
        profile["guidance_enabled"] = True
        profile["guidance_route"] = route
    profile["table"] = table
    profile["index"] = index
    return profile, elapsed_ms


def reset_after_plan_gate(cur: psycopg.Cursor, args: argparse.Namespace, method: str) -> None:
    """Discard gate state so adaptive admission starts with the measured workload."""
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    configure(cur, args, method)


def plan_index_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "Index Name" in value:
            nodes.append(value)
        for child in value.values():
            nodes.extend(plan_index_nodes(child))
    elif isinstance(value, list):
        for child in value:
            nodes.extend(plan_index_nodes(child))
    return nodes


def gate_method_plan(cur: psycopg.Cursor, args: argparse.Namespace, method: str, row: dict[str, Any]) -> bool:
    """Prove the expected HNSW index once, then reset before workload execution."""
    try:
        activation_profile, _ = activate_guidance(cur, args, method, row)
        sql, params = build_hybrid_query(args, method, row, activation_profile)
        cur.execute("EXPLAIN (FORMAT JSON, VERBOSE) " + sql, params)
        plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        _, expected_index = method_table_index(args, method)
        expected_name = expected_index.rsplit(".", 1)[-1]
        return any(
            node.get("Node Type") in {"Index Scan", "Index Only Scan"}
            and node.get("Index Name") == expected_name
            for node in plan_index_nodes(plan)
        )
    finally:
        reset_after_plan_gate(cur, args, method)


def build_hybrid_query(
    args: argparse.Namespace,
    method: str,
    row: dict[str, Any],
    activation_profile: dict[str, Any],
) -> tuple[str, tuple[Any, ...]]:
    table, _ = method_table_index(args, method)
    binding = ""
    params: tuple[Any, ...] = ()
    if activation_profile.get("guidance_enabled") is True:
        index = str(activation_profile.get("index") or method_table_index(args, method)[1])
        params = (index, tag_atoms(row["tags_list"]), guidance_kind(args, method))
        binding = "(SELECT vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) OFFSET 0) AND "
    params += (row["tags_list"], int(row["qid"]))
    return (
        f"""
            SELECT id
            FROM {table}
            WHERE {binding}tags && %s::int[]
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
        params,
    )


def guidance_scan_contract_satisfied(profile: dict[str, Any], strategy: str) -> bool:
    if int(profile.get("guidance_checks", 0) or 0) <= 0:
        return False
    return strategy != "guided_collect" or int(
        profile.get("traversal_guidance_checks", 0) or 0
    ) > 0


def run_query(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    method: str,
    row: dict[str, Any],
    activation_profile: dict[str, Any] | None = None,
) -> tuple[list[int], float, dict[str, Any], str]:
    activation_profile = activation_profile or {}
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")

    def execute():
        sql, params = build_hybrid_query(args, method, row, activation_profile)
        cur.execute(sql, params)
        return [int(x[0]) for x in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        profile = fetch_json(cur, "SELECT vector_hnsw_last_scan_profile()")
        error = ""
        if activation_profile.get("guidance_enabled") is True and not guidance_scan_contract_satisfied(
            profile, args.guidance_filter_strategy
        ):
            error = "GuidanceBindingInactive"
        return ids, latency_ms, profile, error
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
                "traversal_expanded_nodes_mean": statistics.fmean(vals("traversal_expanded_nodes")),
                "traversal_neighbors_examined_mean": statistics.fmean(vals("traversal_neighbors_examined")),
                "traversal_guidance_checks_mean": statistics.fmean(vals("traversal_guidance_checks")),
                "traversal_guidance_matches_mean": statistics.fmean(vals("traversal_guidance_matches")),
                "traversal_matching_expanded_mean": statistics.fmean(vals("traversal_matching_expanded")),
                "traversal_bridge_expanded_mean": statistics.fmean(vals("traversal_bridge_expanded")),
                "traversal_candidate_admissions_mean": statistics.fmean(vals("traversal_candidate_admissions")),
                "traversal_result_admissions_mean": statistics.fmean(vals("traversal_result_admissions")),
                "traversal_guided_admissions_mean": statistics.fmean(vals("traversal_guided_admissions")),
                "traversal_guided_suppressions_mean": statistics.fmean(vals("traversal_guided_suppressions")),
                "traversal_heap_tids_suppressed_mean": statistics.fmean(vals("traversal_heap_tids_suppressed")),
                "traversal_stop_deferrals_mean": statistics.fmean(vals("traversal_stop_deferrals")),
                "traversal_discarded_pushes_mean": statistics.fmean(vals("traversal_discarded_pushes")),
                "traversal_discarded_pops_mean": statistics.fmean(vals("traversal_discarded_pops")),
                "traversal_initial_batches_mean": statistics.fmean(vals("traversal_initial_batches")),
                "traversal_resume_batches_mean": statistics.fmean(vals("traversal_resume_batches")),
                "traversal_strict_order_drops_mean": statistics.fmean(vals("traversal_strict_order_drops")),
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
    parser.add_argument("--guidance-filter-strategy", default="guided_collect", choices=["safe_guided", "guided_collect", "acorn1"])
    parser.add_argument(
        "--d3-reuse-active-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated no-op: D3 activates workload-driven adaptive guidance for every request.",
    )
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=float,
        default=50.0,
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
    parser.add_argument(
        "--prewarm-d3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated no-op: D3 never prebuilds guidance fragments.",
    )
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
        ensure_tracking(cur, args.stock_table, args.bfs_table)
        pin_backend(cur, args.backend_cpu_list)
        if "d1_d2_d3" in args.methods:
            # Tracking creates only descriptors; adaptive fragments are workload-driven.
            cur.execute("SELECT vector_hnsw_guidance_reset()")
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
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
                "d3_initialization",
                "prebuilt_fragments",
                "planner_proof_verified",
                "guidance_scan_verified",
                "activation_atom_count",
                "adaptive_active",
                "adaptive_probes",
                "adaptive_admissions",
                "adaptive_page_builds",
                "adaptive_bloom_builds",
                "adaptive_refinements",
                "adaptive_rejections",
                "adaptive_bytes",
                "adaptive_score",
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
                "traversal_expanded_nodes",
                "traversal_neighbors_examined",
                "traversal_guidance_checks",
                "traversal_guidance_matches",
                "traversal_guidance_misses",
                "traversal_matching_expanded",
                "traversal_bridge_expanded",
                "traversal_candidate_admissions",
                "traversal_result_admissions",
                "traversal_guided_admissions",
                "traversal_guided_suppressions",
                "traversal_heap_tids_suppressed",
                "traversal_stop_deferrals",
                "traversal_discarded_pushes",
                "traversal_discarded_pops",
                "traversal_initial_batches",
                "traversal_resume_batches",
                "traversal_strict_order_drops",
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
                planner_proof_verified = gate_method_plan(cur, args, method, selected[0])
                if not planner_proof_verified:
                    raise RuntimeError(f"HNSW plan gate failed for method={method}")
                if warmup_enabled(args, method):
                    for qno, query in enumerate(selected, start=1):
                        try:
                            warmup_profile, _ = activate_guidance(cur, args, method, query)
                            run_query(cur, args, method, query, warmup_profile)
                        except Exception:
                            try:
                                cur.execute("ROLLBACK")
                            except Exception:
                                pass
                            configure(cur, args, method)
                        if args.progress_queries and qno % args.progress_queries == 0:
                            print(f"{method} warmup {qno}/{len(selected)}", flush=True)
                for qno, query in enumerate(selected, start=1):
                    for repeat in range(args.repeats):
                        activation_profile, activation_ms = activate_guidance(cur, args, method, query)
                        ids, latency_ms, scan_profile, error = run_query(
                            cur, args, method, query, activation_profile
                        )
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
                            "d3_active_guidance_reused": False,
                            "d3_initialization": "workload_driven_adaptive",
                            "prebuilt_fragments": 0,
                            "planner_proof_verified": planner_proof_verified,
                            "guidance_scan_verified": (
                                not bool(activation_profile.get("guidance_enabled", False))
                                or guidance_scan_contract_satisfied(scan_profile, args.guidance_filter_strategy)
                            ),
                            "activation_atom_count": int(activation_profile.get("activation_atom_count", 0) or 0),
                            "adaptive_active": bool(activation_profile.get("active", False)),
                            "adaptive_probes": int(activation_profile.get("adaptive_probes", 0) or 0),
                            "adaptive_admissions": int(activation_profile.get("adaptive_admissions", 0) or 0),
                            "adaptive_page_builds": int(activation_profile.get("adaptive_page_builds", 0) or 0),
                            "adaptive_bloom_builds": int(activation_profile.get("adaptive_bloom_builds", 0) or 0),
                            "adaptive_refinements": int(activation_profile.get("adaptive_refinements", 0) or 0),
                            "adaptive_rejections": int(activation_profile.get("adaptive_rejections", 0) or 0),
                            "adaptive_bytes": int(activation_profile.get("adaptive_bytes", cache_profile.get("adaptive_bytes", 0)) or 0),
                            "adaptive_score": float(activation_profile.get("adaptive_score", cache_profile.get("adaptive_score", 0)) or 0),
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
                            "traversal_expanded_nodes": float(scan_profile.get("traversal_expanded_nodes", 0) or 0),
                            "traversal_neighbors_examined": float(scan_profile.get("traversal_neighbors_examined", 0) or 0),
                            "traversal_guidance_checks": float(scan_profile.get("traversal_guidance_checks", 0) or 0),
                            "traversal_guidance_matches": float(scan_profile.get("traversal_guidance_matches", 0) or 0),
                            "traversal_guidance_misses": float(scan_profile.get("traversal_guidance_misses", 0) or 0),
                            "traversal_matching_expanded": float(scan_profile.get("traversal_matching_expanded", 0) or 0),
                            "traversal_bridge_expanded": float(scan_profile.get("traversal_bridge_expanded", 0) or 0),
                            "traversal_candidate_admissions": float(scan_profile.get("traversal_candidate_admissions", 0) or 0),
                            "traversal_result_admissions": float(scan_profile.get("traversal_result_admissions", 0) or 0),
                            "traversal_guided_admissions": float(scan_profile.get("traversal_guided_admissions", 0) or 0),
                            "traversal_guided_suppressions": float(scan_profile.get("traversal_guided_suppressions", 0) or 0),
                            "traversal_heap_tids_suppressed": float(scan_profile.get("traversal_heap_tids_suppressed", 0) or 0),
                            "traversal_stop_deferrals": float(scan_profile.get("traversal_stop_deferrals", 0) or 0),
                            "traversal_discarded_pushes": float(scan_profile.get("traversal_discarded_pushes", 0) or 0),
                            "traversal_discarded_pops": float(scan_profile.get("traversal_discarded_pops", 0) or 0),
                            "traversal_initial_batches": float(scan_profile.get("traversal_initial_batches", 0) or 0),
                            "traversal_resume_batches": float(scan_profile.get("traversal_resume_batches", 0) or 0),
                            "traversal_strict_order_drops": float(scan_profile.get("traversal_strict_order_drops", 0) or 0),
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
