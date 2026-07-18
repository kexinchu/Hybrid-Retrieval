from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
from pgvector_predicate_guidance_benchmark import FILTER_ATOMS, load_truth


INSERTION_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_insert"
INSERTION_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_insert_hnsw"
BFS_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs"
BFS_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs_hnsw"

MODES = [
    "original",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
]

MODE_LABELS = {
    "original": "Original pgvector",
    "design1_bloom": "Design 1",
    "design1_bloom_bfs_layout": "Design 1 + Design 2",
    "design1_bloom_bfs_layout_d3": "Design 1 + Design 2 + Design 3",
}


def parse_atoms(text: str) -> list[str]:
    atoms = [part.strip() for part in str(text or "").split("||") if part.strip()]
    if not atoms:
        raise ValueError("empty atoms field")
    return atoms


def load_filter_specs(path: Path | None) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    if path is None:
        return ATTR_FILTERS, dict(FILTER_ATOMS)
    filters: list[tuple[str, str, str]] = []
    atoms_by_filter: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["filter_name"]
            filters.append((name, row.get("actual_pct") or row["target_rate"], row["predicate"]))
            atoms_by_filter[name] = parse_atoms(row["atoms"])
    if not filters:
        raise SystemExit(f"no filters loaded from {path}")
    return filters, atoms_by_filter


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def parse_pct(value: object) -> float:
    text = str(value).strip().replace("%", "")
    return float(text)


def ensure_functions(cur: psycopg.Cursor) -> None:
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
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
        except Exception as exc:  # noqa: BLE001 - parallel runners may race on pg_proc updates
            if "tuple concurrently updated" not in str(exc):
                raise
            cur.connection.rollback()
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def ensure_tracking(cur: psycopg.Cursor, *tables: str) -> None:
    for table in dict.fromkeys(tables):
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (table,))


def mode_uses_d2(mode: str) -> bool:
    return mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}


def configure(cur: psycopg.Cursor, args: argparse.Namespace, cache_mb: int, mode: str = "original") -> None:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(args.guided_collect_target)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(cache_mb)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute(f"SET hnsw.page_access = {args.d2_page_access if mode_uses_d2(mode) else 'off'}")
    cur.execute(f"SET hnsw.index_page_access = {args.d2_index_page_access if mode_uses_d2(mode) else 'off'}")
    cur.execute(f"SET hnsw.page_window = {int(args.d2_page_window)}")
    cur.execute(f"SET hnsw.page_prefetch_min_items = {int(args.d2_page_prefetch_min_items)}")
    cur.execute(f"SET hnsw.page_disable_after_no_merge = {int(args.d2_page_disable_after_no_merge)}")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def mode_table_index(args: argparse.Namespace, mode: str) -> tuple[str, str]:
    if mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}:
        return args.bfs_table, args.bfs_index
    return args.insertion_table, args.insertion_index


def should_enable_guidance(args: argparse.Namespace, filter_name: str) -> tuple[bool, str]:
    selectivity = float(args.filter_selectivity_by_name.get(filter_name, 100.0))
    atom_count = len(args.filter_atoms.get(filter_name, []))
    if selectivity > float(args.guidance_selectivity_max_pct):
        return False, f"selectivity>{args.guidance_selectivity_max_pct:g}%"
    if atom_count > int(args.guidance_max_atoms):
        return False, f"atoms>{args.guidance_max_atoms}"
    return True, "enabled"


def activate(cur: psycopg.Cursor, args: argparse.Namespace, mode: str, filter_name: str) -> dict[str, object]:
    table, index = mode_table_index(args, mode)
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    if mode == "original":
        cur.execute("SET hnsw.filter_strategy = off")
        return {"table": table, "index": index, "guidance_enabled": False, "guidance_route": "stock"}
    enabled, route = should_enable_guidance(args, filter_name)
    if not enabled:
        cur.execute("SET hnsw.filter_strategy = off")
        return {"table": table, "index": index, "guidance_enabled": False, "guidance_route": route}
    cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
    if args.reset_cache_per_query and mode in {"design1_bloom", "design1_bloom_bfs_layout"}:
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'bloom')",
        (index, args.filter_atoms[filter_name]),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    profile = json.loads(cur.fetchone()[0])
    profile["table"] = table
    profile["index"] = index
    profile["guidance_enabled"] = True
    profile["guidance_route"] = route
    return profile


def d3_guidance_signature(args: argparse.Namespace, mode: str, filter_name: str) -> tuple[str, str, tuple[str, ...]] | None:
    if mode != "design1_bloom_bfs_layout_d3" or not args.d3_reuse_active_guidance:
        return None
    enabled, _ = should_enable_guidance(args, filter_name)
    if not enabled:
        return None
    table, index = mode_table_index(args, mode)
    return table, index, tuple(args.filter_atoms[filter_name])


def reuse_activation_profile(args: argparse.Namespace, mode: str, filter_name: str, profile: dict[str, object]) -> dict[str, object]:
    table, index = mode_table_index(args, mode)
    reused = dict(profile)
    reused["table"] = table
    reused["index"] = index
    reused["guidance_enabled"] = True
    reused["guidance_route"] = "d3_reuse_active_guidance"
    reused["d3_active_guidance_reused"] = True
    return reused


def run_query(cur: psycopg.Cursor, table: str, predicate: str, query_id: int, k: int) -> tuple[list[int], dict[str, object]]:
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
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def prewarm_d3(cur: psycopg.Cursor, args: argparse.Namespace, filters: list[tuple[str, float, str]]) -> None:
    configure(cur, args, args.d3_cache_mb, "design1_bloom_bfs_layout_d3")
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    for filter_name, _, _ in filters:
        if not should_enable_guidance(args, filter_name)[0]:
            continue
        cur.execute(
            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'bloom')",
            (args.bfs_index, args.filter_atoms[filter_name]),
        )
    cur.execute("SELECT vector_hnsw_guidance_reset()")


def run_mode(
    args: argparse.Namespace,
    mode: str,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], list[int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache_mb = args.d3_cache_mb if mode == "design1_bloom_bfs_layout_d3" else args.d1_cache_mb
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        ensure_tracking(cur, args.insertion_table, args.bfs_table)
        configure(cur, args, cache_mb, mode)
        if mode == "design1_bloom_bfs_layout_d3":
            prewarm_d3(cur, args, filters)
        else:
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
        active_signature: tuple[str, str, tuple[str, ...]] | None = None
        active_profile: dict[str, object] = {}

        warm_nos = query_nos if args.warmup_all_queries else query_nos[: args.warmup_queries]
        for filter_name, _, predicate in filters:
            for qno in warm_nos:
                try:
                    activation_profile = activate(cur, args, mode, filter_name)
                    run_query(cur, str(activation_profile["table"]), predicate, query_by_no[qno], args.k)
                except Exception:
                    try:
                        cur.execute("ROLLBACK")
                    except Exception:
                        pass
                    configure(cur, args, cache_mb, mode)

        for filter_name, selectivity, predicate in filters:
            for idx, qno in enumerate(query_nos, start=1):
                qid = query_by_no[qno]
                for repeat in range(args.repeats):
                    error = ""
                    ids: list[int] = []
                    activation_profile: dict[str, object] = {}
                    scan_profile: dict[str, object] = {}
                    cache_profile: dict[str, object] = {}
                    activation_ms = 0.0
                    query_ms = 0.0
                    table, index = mode_table_index(args, mode)
                    try:
                        signature = d3_guidance_signature(args, mode, filter_name)
                        if signature is not None and signature == active_signature:
                            activation_profile = reuse_activation_profile(args, mode, filter_name, active_profile)
                            activation_ms = 0.0
                        else:
                            activation_profile, activation_ms = timed_ms(lambda: activate(cur, args, mode, filter_name))
                            active_signature = signature
                            active_profile = dict(activation_profile) if signature is not None else {}
                        table = str(activation_profile["table"])
                        index = str(activation_profile["index"])
                        (ids, scan_profile), query_ms = timed_ms(lambda: run_query(cur, table, predicate, qid, args.k))
                        cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
                        cache_profile = json.loads(cur.fetchone()[0])
                    except errors.QueryCanceled as exc:
                        error = exc.__class__.__name__
                        cur.execute("SET statement_timeout = 0")
                    except Exception as exc:  # noqa: BLE001
                        error = exc.__class__.__name__
                        try:
                            cur.execute("ROLLBACK")
                        except Exception:
                            pass
                        configure(cur, args, cache_mb, mode)

                    rows.append(
                        {
                            "selectivity": selectivity,
                            "filter_name": filter_name,
                            "mode": mode,
                            "mode_label": MODE_LABELS[mode],
                            "table": table,
                            "index": index,
                            "d2_page_access": args.d2_page_access if mode_uses_d2(mode) else "off",
                            "d2_index_page_access": args.d2_index_page_access if mode_uses_d2(mode) else "off",
                            "query_no": qno,
                            "query_id": qid,
                            "repeat": repeat,
                            "recall": recall_at_k(ids, truth[(filter_name, qno)], args.k) if not error else 0.0,
                            "activation_ms": activation_ms,
                            "query_latency_ms": query_ms,
                            "end_to_end_ms": activation_ms + query_ms,
                            "guidance_enabled": bool(activation_profile.get("guidance_enabled", mode != "original")),
                            "guidance_route": str(activation_profile.get("guidance_route", "")),
                            "d3_active_guidance_reused": bool(activation_profile.get("d3_active_guidance_reused", False)),
                            "vector_search_ms": scan_profile.get("vector_search_ms", 0.0),
                            "visited_tuples": scan_profile.get("visited_tuples", 0),
                            "returned_tuples": scan_profile.get("returned_tuples", 0),
                            "distance_compute_count": scan_profile.get("distance_compute_count", 0),
                            "page_access_batches": scan_profile.get("page_access_batches", 0),
                            "page_access_candidates": scan_profile.get("page_access_candidates", 0),
                            "page_access_prefetches": scan_profile.get("page_access_prefetches", 0),
                            "page_access_distinct_pages": scan_profile.get("page_access_distinct_pages", 0),
                            "index_page_prefetches": scan_profile.get("index_page_prefetches", 0),
                            "guidance_checks": scan_profile.get("guidance_checks", 0),
                            "guidance_skips": scan_profile.get("guidance_skips", 0),
                            "fragment_cache_hits": activation_profile.get("fragment_cache_hits", 0),
                            "fragment_cache_misses": activation_profile.get("fragment_cache_misses", 0),
                            "fragment_store_hits": activation_profile.get("fragment_store_hits", 0),
                            "fragment_builds": activation_profile.get("fragment_builds", 0),
                            "composed_guide_hit": activation_profile.get("composed_guide_hit", False),
                            "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
                            "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
                            "cache_resident_bytes": cache_profile.get("resident_bytes", 0),
                            "cache_resident_entries": cache_profile.get("resident_entries", 0),
                            "cache_evictions": cache_profile.get("evictions", 0),
                            "composed_guide_entries": cache_profile.get("composed_guide_entries", 0),
                            "composed_guide_hits_total": cache_profile.get("composed_guide_hits", 0),
                            "returned": len(ids),
                            "ids": ",".join(str(x) for x in ids),
                            "error": error,
                        }
                    )
                if args.progress_queries and idx % args.progress_queries == 0:
                    ok = [r for r in rows if r["filter_name"] == filter_name and not r["error"]]
                    if ok:
                        print(
                            f"progress mode={mode} filter={filter_name} "
                            f"queries={idx}/{len(query_nos)} "
                            f"e2e={statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}ms",
                            flush=True,
                        )
        cur.execute("SELECT vector_hnsw_guidance_reset()")
    return rows


def write_summary(rows: list[dict[str, object]], out: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filter_name"]), str(row["mode"])), []).append(row)

    mode_mean: dict[tuple[str, str], float] = {}
    for key, items in grouped.items():
        ok = [r for r in items if not r["error"]]
        mode_mean[key] = statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0

    table_out = out.with_name(out.stem + "_table.csv")
    fields = [
        "Selectivity",
        "Filter",
        "Original pgvector",
        "Design 1",
        "Design 1 + Design 2",
        "Design 1 + Design 2 + Design 3",
        "D1 speedup",
        "D1+D2 speedup",
        "D1+D2 + D3 speedup",
    ]
    with table_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        seen_filters = []
        for row in rows:
            key = (str(row["filter_name"]), str(row["selectivity"]))
            if key not in seen_filters:
                seen_filters.append(key)
        for filter_name, selectivity in seen_filters:
            if (filter_name, "original") not in mode_mean:
                continue
            original = mode_mean[(filter_name, "original")]
            d1 = mode_mean.get((filter_name, "design1_bloom"), 0.0)
            d12 = mode_mean.get((filter_name, "design1_bloom_bfs_layout"), 0.0)
            d123 = mode_mean.get((filter_name, "design1_bloom_bfs_layout_d3"), 0.0)
            writer.writerow(
                {
					"Selectivity": str(selectivity),
                    "Filter": filter_name,
                    "Original pgvector": f"{original:.4f}",
                    "Design 1": f"{d1:.4f}",
                    "Design 1 + Design 2": f"{d12:.4f}",
                    "Design 1 + Design 2 + Design 3": f"{d123:.4f}",
                    "D1 speedup": f"{(original / d1):.4f}" if d1 else "0.0000",
                    "D1+D2 speedup": f"{(original / d12):.4f}" if d12 else "0.0000",
                    "D1+D2 + D3 speedup": f"{(original / d123):.4f}" if d123 else "0.0000",
                }
            )

    profile_out = out.with_name(out.stem + "_profile_summary.csv")
    profile_fields = [
        "filter_name",
        "mode",
        "ok",
        "errors",
        "recall_mean",
        "end_to_end_mean_ms",
        "activation_mean_ms",
        "query_latency_mean_ms",
        "guidance_enabled_rate",
        "cache_resident_bytes_max",
        "fragment_cache_hits_mean",
        "fragment_store_hits_mean",
        "fragment_builds_mean",
        "composed_guide_hit_rate",
        "guidance_skip_rate",
    ]
    with profile_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fields)
        writer.writeheader()
        for (filter_name, mode), items in sorted(grouped.items()):
            ok = [r for r in items if not r["error"]]
            checks = statistics.fmean(float(r["guidance_checks"]) for r in ok) if ok else 0.0
            skips = statistics.fmean(float(r["guidance_skips"]) for r in ok) if ok else 0.0
            writer.writerow(
                {
                    "filter_name": filter_name,
                    "mode": mode,
                    "ok": len(ok),
                    "errors": len(items) - len(ok),
                    "recall_mean": statistics.fmean(float(r["recall"]) for r in ok) if ok else 0.0,
                    "end_to_end_mean_ms": statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0,
                    "activation_mean_ms": statistics.fmean(float(r["activation_ms"]) for r in ok) if ok else 0.0,
                    "query_latency_mean_ms": statistics.fmean(float(r["query_latency_ms"]) for r in ok) if ok else 0.0,
                    "guidance_enabled_rate": statistics.fmean(float(r["guidance_enabled"]) for r in ok) if ok else 0.0,
                    "cache_resident_bytes_max": max((int(r["cache_resident_bytes"]) for r in ok), default=0),
                    "fragment_cache_hits_mean": statistics.fmean(float(r["fragment_cache_hits"]) for r in ok) if ok else 0.0,
                    "fragment_store_hits_mean": statistics.fmean(float(r["fragment_store_hits"]) for r in ok) if ok else 0.0,
                    "fragment_builds_mean": statistics.fmean(float(r["fragment_builds"]) for r in ok) if ok else 0.0,
                    "composed_guide_hit_rate": statistics.fmean(1.0 if r["composed_guide_hit"] else 0.0 for r in ok) if ok else 0.0,
                    "guidance_skip_rate": skips / checks if checks else 0.0,
                }
            )
    print(f"wrote {table_out}", flush=True)
    print(f"wrote {profile_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Original, D1, D1+D2, and D1+D2+D3 pgvector variants.")
    parser.add_argument("--insertion-table", default=INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path)
    parser.add_argument("--modes", nargs="*", choices=MODES, default=MODES)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument(
        "--warmup-all-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one unmeasured pass over every measured query for each filter before recording latency.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--guided-collect-target", type=int, default=1000)
    parser.add_argument("--guidance-filter-strategy", default="guided_collect", choices=["guided_collect", "acorn1"])
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-reuse-active-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=float,
        default=10.0,
        help="Disable predicate guidance above this filter percentage; D1+D2 then runs as D2-only.",
    )
    parser.add_argument(
        "--guidance-max-atoms",
        type=int,
        default=64,
        help="Disable predicate guidance when a query decomposes into more atoms than this.",
    )
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--reset-cache-per-query", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    all_filters, args.filter_atoms = load_filter_specs(args.filters_csv)
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in all_filters if not selected or name in selected]
    args.filter_selectivity_by_name = {name: parse_pct(target) for name, target, _ in filters}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for mode in args.modes:
        print(f"running mode={mode}", flush=True)
        rows.extend(run_mode(args, mode, filters, query_nos, query_by_no, truth))

    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)
    write_summary(rows, args.out)


if __name__ == "__main__":
    main()
