from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import recall_at_k


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
RESULTS = ROOT / "results" / "hybrid_vector_db"
sys.path.insert(0, str(SCRIPT_DIR))

import pgvector_design1_design2_design3_selectivity_benchmark as bench  # noqa: E402


DEFAULT_FILTERS = [
    "popular_ge1000",
    "price_10_to_20",
    "rating5_price_le10",
    "long_review_ge500",
    "grocery_rating5",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
]

MODE_LABELS = {
    "original": "Stock pgvector",
    "design1_bloom": "D1",
    "design1_bloom_bfs_layout": "D1+D2",
    "design1_bloom_bfs_layout_d3": "SQLens",
}


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_filters(args: argparse.Namespace) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    filters, atoms = bench.load_filter_specs(args.filters_csv)
    selected = set(args.filter_names)
    chosen = [(name, sel, pred) for name, sel, pred in filters if name in selected]
    order = {name: i for i, name in enumerate(args.filter_names)}
    chosen.sort(key=lambda item: order[item[0]])
    missing = selected - {name for name, _, _ in chosen}
    if missing:
        raise SystemExit(f"missing filter specs for: {sorted(missing)}")
    return chosen, atoms


def build_workload(
    filters: list[tuple[str, str, str]],
    query_nos: list[int],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], list[int]],
    queries_per_filter: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    entries: list[dict[str, object]] = []
    for filter_name, selectivity, predicate in filters:
        valid_qnos = [qno for qno in query_nos if (filter_name, qno) in truth]
        if not valid_qnos:
            raise SystemExit(f"no exact GT for filter={filter_name}")
        if queries_per_filter <= len(valid_qnos):
            picked = rng.sample(valid_qnos, queries_per_filter)
        else:
            picked = [rng.choice(valid_qnos) for _ in range(queries_per_filter)]
        for qno in picked:
            entries.append(
                {
                    "filter_name": filter_name,
                    "selectivity": selectivity,
                    "predicate": predicate,
                    "query_no": qno,
                    "query_id": query_by_no[qno],
                }
            )
    rng.shuffle(entries)
    for idx, entry in enumerate(entries):
        entry["workload_pos"] = idx
    return entries


def run_config(
    args: argparse.Namespace,
    mode: str,
    workload: list[dict[str, object]],
    filters: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache_mb = args.d3_cache_mb if mode == "design1_bloom_bfs_layout_d3" else args.d1_cache_mb

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        bench.ensure_functions(cur)
        bench.configure(cur, args, cache_mb, mode)
        if mode == "design1_bloom_bfs_layout_d3":
            bench.prewarm_d3(cur, args, filters)
            bench.configure(cur, args, cache_mb, mode)
        else:
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")

        if args.warmup_all_queries:
            for entry in workload:
                try:
                    activation = bench.activate(cur, args, mode, str(entry["filter_name"]))
                    bench.run_query(
                        cur,
                        str(activation["table"]),
                        str(entry["predicate"]),
                        int(entry["query_id"]),
                        args.k,
                    )
                except Exception:
                    try:
                        cur.execute("ROLLBACK")
                    except Exception:
                        pass
                    bench.configure(cur, args, cache_mb, mode)

        active_signature: tuple[str, str, tuple[str, ...]] | None = None
        active_profile: dict[str, object] = {}
        for idx, entry in enumerate(workload, start=1):
            filter_name = str(entry["filter_name"])
            predicate = str(entry["predicate"])
            query_no = int(entry["query_no"])
            query_id = int(entry["query_id"])
            for repeat in range(args.repeats):
                ids: list[int] = []
                activation_profile: dict[str, object] = {}
                scan_profile: dict[str, object] = {}
                cache_profile: dict[str, object] = {}
                activation_ms = 0.0
                query_ms = 0.0
                error = ""
                table, index = bench.mode_table_index(args, mode)
                try:
                    signature = bench.d3_guidance_signature(args, mode, filter_name)
                    if signature is not None and signature == active_signature:
                        activation_profile = bench.reuse_activation_profile(args, mode, filter_name, active_profile)
                    else:
                        activation_profile, activation_ms = timed_ms(lambda: bench.activate(cur, args, mode, filter_name))
                        active_signature = signature
                        active_profile = dict(activation_profile) if signature is not None else {}
                    table = str(activation_profile["table"])
                    index = str(activation_profile["index"])
                    (ids, scan_profile), query_ms = timed_ms(
                        lambda: bench.run_query(cur, table, predicate, query_id, args.k)
                    )
                    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
                    cache_profile = json.loads(cur.fetchone()[0])
                except errors.QueryCanceled as exc:
                    error = exc.__class__.__name__
                    try:
                        cur.execute("ROLLBACK")
                    except Exception:
                        pass
                    bench.configure(cur, args, cache_mb, mode)
                    active_signature = None
                    active_profile = {}
                except Exception as exc:  # noqa: BLE001
                    error = exc.__class__.__name__
                    try:
                        cur.execute("ROLLBACK")
                    except Exception:
                        pass
                    bench.configure(cur, args, cache_mb, mode)
                    active_signature = None
                    active_profile = {}

                truth_ids = args.truth[(filter_name, query_no)]
                rows.append(
                    {
                        "workload": args.workload_name,
                        "workload_pos": entry["workload_pos"],
                        "selectivity": entry["selectivity"],
                        "filter_name": filter_name,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "table": table,
                        "index": index,
                        "query_no": query_no,
                        "query_id": query_id,
                        "repeat": repeat,
                        "ef_search": args.ef_search,
                        "max_scan_tuples": args.max_scan_tuples,
                        "guided_collect_target": args.guided_collect_target,
                        "recall": recall_at_k(ids, truth_ids, args.k) if not error else 0.0,
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
                        "guidance_checks": scan_profile.get("guidance_checks", 0),
                        "guidance_skips": scan_profile.get("guidance_skips", 0),
                        "page_access_batches": scan_profile.get("page_access_batches", 0),
                        "page_access_candidates": scan_profile.get("page_access_candidates", 0),
                        "page_access_distinct_pages": scan_profile.get("page_access_distinct_pages", 0),
                        "fragment_cache_hits": activation_profile.get("fragment_cache_hits", 0),
                        "fragment_cache_misses": activation_profile.get("fragment_cache_misses", 0),
                        "fragment_store_hits": activation_profile.get("fragment_store_hits", 0),
                        "fragment_builds": activation_profile.get("fragment_builds", 0),
                        "composed_guide_hit": activation_profile.get("composed_guide_hit", False),
                        "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
                        "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
                        "cache_resident_bytes": cache_profile.get("resident_bytes", 0),
                        "cache_resident_entries": cache_profile.get("resident_entries", 0),
                        "composed_guide_entries": cache_profile.get("composed_guide_entries", 0),
                        "composed_guide_hits_total": cache_profile.get("composed_guide_hits", 0),
                        "returned": len(ids),
                        "ids": ",".join(str(x) for x in ids),
                        "error": error,
                    }
                )
            if args.progress_queries and idx % args.progress_queries == 0:
                ok = [r for r in rows if not r["error"]]
                if ok:
                    print(
                        f"progress budget={args.max_scan_tuples} mode={mode} "
                        f"queries={idx}/{len(workload)} e2e="
                        f"{statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}ms",
                        flush=True,
                    )
        cur.execute("SELECT vector_hnsw_guidance_reset()")
    return rows


def summarize(rows: list[dict[str, object]], workload_name: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), int(row["ef_search"]), int(row["max_scan_tuples"])), []).append(row)

    out: list[dict[str, object]] = []
    for (mode, ef_search, max_scan_tuples), items in sorted(groups.items()):
        ok = [r for r in items if not r["error"]]
        lat = [float(r["end_to_end_ms"]) for r in ok]
        recall = [float(r["recall"]) for r in ok]
        visited = [float(r["visited_tuples"]) for r in ok]
        returned_tuples = [float(r["returned_tuples"]) for r in ok]
        checks = [float(r["guidance_checks"]) for r in ok]
        skips = [float(r["guidance_skips"]) for r in ok]
        total_ms = sum(lat)
        out.append(
            {
                "workload": workload_name,
                "filter_name": workload_name,
                "selectivity": "random",
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "ef_search": ef_search,
                "max_scan_tuples": max_scan_tuples,
                "samples": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": statistics.fmean(recall) if recall else 0.0,
                "recall_p50": statistics.median(recall) if recall else 0.0,
                "recall_min": min(recall) if recall else 0.0,
                "latency_mean_ms": statistics.fmean(lat) if lat else 0.0,
                "latency_p50_ms": statistics.median(lat) if lat else 0.0,
                "latency_p95_ms": p95(lat),
                "single_client_throughput_qps": (1000.0 * len(ok) / total_ms) if total_ms > 0 else 0.0,
                "visited_tuples_mean": statistics.fmean(visited) if visited else 0.0,
                "returned_tuples_mean": statistics.fmean(returned_tuples) if returned_tuples else 0.0,
                "guidance_skip_rate": (
                    statistics.fmean(skips) / statistics.fmean(checks)
                    if checks and statistics.fmean(checks) > 0
                    else 0.0
                ),
            }
        )
    return out


def summarize_by_filter(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["mode"]), int(row["max_scan_tuples"])), []).append(row)

    out: list[dict[str, object]] = []
    for (filter_name, mode, max_scan_tuples), items in sorted(groups.items()):
        ok = [r for r in items if not r["error"]]
        lat = [float(r["end_to_end_ms"]) for r in ok]
        recall = [float(r["recall"]) for r in ok]
        out.append(
            {
                "filter_name": filter_name,
                "mode": mode,
                "max_scan_tuples": max_scan_tuples,
                "samples": len(items),
                "ok": len(ok),
                "recall_mean": statistics.fmean(recall) if recall else 0.0,
                "latency_mean_ms": statistics.fmean(lat) if lat else 0.0,
                "latency_p95_ms": p95(lat),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon random-selectivity recall/latency/throughput sweep.")
    parser.add_argument("--out-prefix", default="amazon_random_selectivity_recall_tradeoff")
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--filters-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_selectivity14_diverse_filters_20260715.csv"))
    parser.add_argument("--filter-names", nargs="*", default=DEFAULT_FILTERS)
    parser.add_argument("--modes", nargs="*", choices=bench.MODES, default=["original", "design1_bloom_bfs_layout_d3"])
    parser.add_argument("--queries-per-filter", type=int, default=12)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument(
        "--ef-search-values",
        default="",
        help="Comma-separated ef_search sweep. Overrides --ef-search when set.",
    )
    parser.add_argument("--guided-collect-target", type=int, default=1000)
    parser.add_argument(
        "--guided-collect-target-from-ef",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set hnsw.guided_collect_target to the current ef_search value for each sweep point.",
    )
    parser.add_argument("--max-scan-tuples-values", default="2000,5000,10000,20000,50000,200000")
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--guidance-filter-strategy", default="guided_collect", choices=["guided_collect", "acorn1"])
    parser.add_argument("--guidance-selectivity-max-pct", type=float, default=10.0)
    parser.add_argument("--guidance-max-atoms", type=int, default=64)
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-reuse-active-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--statement-timeout-ms", type=int, default=180000)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-queries", type=int, default=24)
    parser.add_argument("--reset-cache-per-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--insertion-table", default=bench.INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=bench.INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=bench.BFS_TABLE)
    parser.add_argument("--bfs-index", default=bench.BFS_INDEX)
    args = parser.parse_args()

    truth, query_by_no = bench.load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + 100]
    filters, args.filter_atoms = load_filters(args)
    args.filter_selectivity_by_name = {name: bench.parse_pct(sel) for name, sel, _ in filters}
    args.truth = truth
    args.workload_name = "amazon_random_selectivity"

    workload = build_workload(filters, query_nos, query_by_no, truth, args.queries_per_filter, args.seed)
    ef_values = parse_ints(args.ef_search_values) if args.ef_search_values else [args.ef_search]
    max_scan_values = parse_ints(args.max_scan_tuples_values)

    rows: list[dict[str, object]] = []
    for ef_search in ef_values:
        for max_scan_tuples in max_scan_values:
            for mode in args.modes:
                run_args = SimpleNamespace(**vars(args))
                run_args.ef_search = ef_search
                run_args.max_scan_tuples = max_scan_tuples
                if args.guided_collect_target_from_ef:
                    run_args.guided_collect_target = ef_search
                print(
                    f"running workload={args.workload_name} ef={run_args.ef_search} "
                    f"target={run_args.guided_collect_target} max_scan={max_scan_tuples} "
                    f"mode={mode} queries={len(workload)} repeats={args.repeats}",
                    flush=True,
                )
                rows.extend(run_config(run_args, mode, workload, filters))

    suffix = f"q{len(workload)}r{args.repeats}_{args.out_prefix}_seed{args.seed}"
    raw_out = RESULTS / f"{suffix}.csv"
    summary_out = RESULTS / f"{suffix}_summary.csv"
    by_filter_out = RESULTS / f"{suffix}_by_filter_summary.csv"
    workload_out = RESULTS / f"{suffix}_workload.csv"
    write_csv(raw_out, rows)
    write_csv(summary_out, summarize(rows, args.workload_name))
    write_csv(by_filter_out, summarize_by_filter(rows))
    write_csv(workload_out, workload)
    print(f"wrote {raw_out}", flush=True)
    print(f"wrote {summary_out}", flush=True)
    print(f"wrote {by_filter_out}", flush=True)
    print(f"wrote {workload_out}", flush=True)


if __name__ == "__main__":
    main()
