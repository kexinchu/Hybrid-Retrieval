from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import psycopg

import laion25m_sqlens_variants_benchmark as bench
from common_pg import pg_config_from_env
from prepare_laion25m_pgvector import INDEX, QUERY_TABLE, TABLE


METHODS = ["d1_d2", "d1_d2_d3"]


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


def row_from_profiles(
    args: argparse.Namespace,
    method: str,
    query: dict[str, Any],
    repeat: int,
    ids: list[int],
    latency_ms: float,
    activation_ms: float,
    activation_profile: dict[str, Any],
    scan_profile: dict[str, Any],
    cache_profile: dict[str, Any],
    expected: list[int],
    error: str,
) -> dict[str, Any]:
    checks = float(scan_profile.get("guidance_checks", 0) or 0)
    skips = float(scan_profile.get("guidance_skips", 0) or 0)
    return {
        "method": method,
        "target_band_pct": float(query["target_band_pct"]),
        "actual_pct": float(query["actual_pct"]),
        "filter_rows": int(query["filter_rows"]),
        "filter_name": query["filter_name"],
        "qid": int(query["qid"]),
        "labels": query.get("labels", ""),
        "repeat": repeat,
        "recall": bench.recall_at_k(ids, expected or [], args.k) if not error else 0.0,
        "latency_ms": latency_ms,
        "activation_ms": activation_ms,
        "end_to_end_ms": activation_ms + latency_ms,
        "activation_build_ms": float(activation_profile.get("last_cache_build_ms", 0) or 0),
        "guidance_enabled": bool(activation_profile.get("guidance_enabled", method != "stock")),
        "guidance_route": str(activation_profile.get("guidance_route", "")),
        "predicted_skip_rate": float(activation_profile.get("predicted_skip_rate", bench.predicted_skip_rate(query)) or 0),
        "d3_guard_can_compose_exact_or": bool(activation_profile.get("d3_guard_can_compose_exact_or", False)),
        "d3_guard_disabled_after_activation": bool(activation_profile.get("d3_guard_disabled_after_activation", False)),
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
        "cache_evictions": int(cache_profile.get("evictions", 0) or 0),
        "vector_search_ms": float(scan_profile.get("vector_search_ms", 0) or 0),
        "visited_tuples": float(scan_profile.get("visited_tuples", 0) or 0),
        "returned_tuples": float(scan_profile.get("returned_tuples", 0) or 0),
        "distance_compute_count": float(scan_profile.get("distance_compute_count", 0) or 0),
        "guidance_checks": checks,
        "guidance_skips": skips,
        "guidance_skip_rate": skips / checks if checks else 0.0,
        "idx_blks_hit": float(scan_profile.get("idx_blks_hit", 0) or 0),
        "idx_blks_read": float(scan_profile.get("idx_blks_read", 0) or 0),
        "heap_blks_hit": float(scan_profile.get("heap_blks_hit", 0) or 0),
        "heap_blks_read": float(scan_profile.get("heap_blks_read", 0) or 0),
        "ids": ",".join(str(x) for x in ids),
        "error": error,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((float(row["target_band_pct"]), str(row["method"])), []).append(row)
    for (target, method), items in sorted(groups.items()):
        ok = [r for r in items if not r.get("error")]
        if not ok:
            continue

        def mean(key: str) -> float:
            return statistics.fmean(float(r.get(key, 0) or 0) for r in ok)

        checks = mean("guidance_checks")
        skips = mean("guidance_skips")
        out.append(
            {
                "target_band_pct": target,
                "method": method,
                "filter_name": ok[0]["filter_name"],
                "queries": len({int(r["qid"]) for r in ok}),
                "rows": len(ok),
                "actual_pct_mean": mean("actual_pct"),
                "recall_mean": mean("recall"),
                "end_to_end_ms_mean": mean("end_to_end_ms"),
                "activation_ms_mean": mean("activation_ms"),
                "vector_search_ms_mean": mean("vector_search_ms"),
                "visited_tuples_mean": mean("visited_tuples"),
                "idx_blks_read_mean": mean("idx_blks_read"),
                "heap_blks_read_mean": mean("heap_blks_read"),
                "guidance_enabled_rate": mean("guidance_enabled"),
                "predicted_skip_rate_mean": mean("predicted_skip_rate"),
                "d3_guard_can_compose_exact_or_rate": mean("d3_guard_can_compose_exact_or"),
                "d3_guard_disabled_after_activation_rate": mean("d3_guard_disabled_after_activation"),
                "composed_exact_active_rate": mean("composed_exact_active"),
                "composed_exact_hit_rate": mean("composed_exact_hit"),
                "composed_exact_build_ms_mean": mean("composed_exact_build_ms"),
                "composed_exact_rows_mean": mean("composed_exact_rows"),
                "guidance_skip_rate_mean": skips / checks if checks else 0.0,
                "errors": len(items) - len(ok),
            }
        )
    return out


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[float, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (float(row["target_band_pct"]), int(row["qid"]), int(row["repeat"]))
        by_key.setdefault(key, {})[str(row["method"])] = row
    groups: dict[float, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for (target, _, _), pair in by_key.items():
        if all(method in pair for method in METHODS):
            groups.setdefault(target, []).append((pair["d1_d2"], pair["d1_d2_d3"]))
    out: list[dict[str, Any]] = []
    for target, pairs in sorted(groups.items()):
        if not pairs:
            continue
        d2 = [p[0] for p in pairs]
        d3 = [p[1] for p in pairs]
        deltas = [float(b["end_to_end_ms"]) - float(a["end_to_end_ms"]) for a, b in pairs]
        speedups = [float(a["end_to_end_ms"]) / float(b["end_to_end_ms"]) for a, b in pairs if float(b["end_to_end_ms"]) > 0]
        out.append(
            {
                "target_band_pct": target,
                "filter_name": d2[0]["filter_name"],
                "pairs": len(pairs),
                "actual_pct_mean": statistics.fmean(float(r["actual_pct"]) for r in d2),
                "d1_d2_ms_mean": statistics.fmean(float(r["end_to_end_ms"]) for r in d2),
                "d1_d2_d3_ms_mean": statistics.fmean(float(r["end_to_end_ms"]) for r in d3),
                "paired_delta_ms_mean": statistics.fmean(deltas),
                "paired_speedup_mean": statistics.fmean(speedups) if speedups else 0.0,
                "d1_d2_recall_mean": statistics.fmean(float(r["recall"]) for r in d2),
                "d1_d2_d3_recall_mean": statistics.fmean(float(r["recall"]) for r in d3),
                "d3_composed_exact_active_rate": statistics.fmean(float(r["composed_exact_active"]) for r in d3),
                "d3_guard_disabled_rate": statistics.fmean(float(r["d3_guard_disabled_after_activation"]) for r in d3),
                "d3_guidance_skip_rate": statistics.fmean(float(r["guidance_skip_rate"]) for r in d3),
            }
        )
    return out


def run_once(cur: psycopg.Cursor, args: argparse.Namespace, method: str, query: dict[str, Any], repeat: int, expected: list[int]) -> dict[str, Any]:
    bench.configure(cur, args, method)
    activation_profile, activation_ms = bench.activate_guidance(cur, args, method, query)
    ids, latency_ms, scan_profile, error = bench.run_query(cur, args, method, query)
    try:
        cache_profile = bench.fetch_json(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    except Exception:
        cache_profile = {}
    return row_from_profiles(args, method, query, repeat, ids, latency_ms, activation_ms, activation_profile, scan_profile, cache_profile, expected, error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired/interleaved LAION25M D1+D2 vs D1+D2+D3 benchmark.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--bfs-table", default="laion25m_pgvector")
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--stock-index", default=INDEX)
    parser.add_argument("--bfs-index", default="laion25m_pgvector_embedding_hnsw_bfs")
    parser.add_argument("--selected-queries-in", type=Path, default=Path("results/hybrid_vector_db/laion25m_label_or_global_selected_q100_20260716.csv"))
    parser.add_argument("--truth", type=Path, default=Path("results/hybrid_vector_db/laion25m_label_or_global_truth_q100_20260716.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/laion25m_d2_d3_paired.csv"))
    parser.add_argument("--target-bands", type=float, nargs="+", default=[])
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--guided-collect-target", type=int, default=1000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--d1-cache-mb", type=int, default=4096)
    parser.add_argument("--d3-cache-mb", type=int, default=4096)
    parser.add_argument("--guidance-kind", default="exact", choices=["exact", "page", "bloom"])
    parser.add_argument("--guidance-compose-exact-or", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-compose-exact-guc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d3-enable-policy", default="compose_or_skip", choices=["legacy", "compose_or_skip"])
    parser.add_argument("--d3-min-predicted-skip-rate", type=float, default=0.5)
    parser.add_argument("--guidance-selectivity-max-pct", type=float, default=10.0)
    parser.add_argument("--guidance-max-atoms", type=int, default=1)
    parser.add_argument("--d3-guidance-max-atoms", type=int, default=64)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-queries", type=int, default=25)
    args = parser.parse_args()

    selected = bench.parse_selected(args.selected_queries_in, args.target_bands, args.limit_per_group)
    truth = bench.load_truth(args.truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        bench.ensure_functions(cur)
        if args.warmup_all_queries:
            for qno, query in enumerate(selected, start=1):
                expected = truth.get(bench.truth_key(query), [])
                for method in METHODS:
                    try:
                        run_once(cur, args, method, query, -1, expected)
                    except Exception:
                        try:
                            cur.execute("ROLLBACK")
                        except Exception:
                            pass
                    finally:
                        bench.configure(cur, args, method)
                if args.progress_queries and qno % args.progress_queries == 0:
                    print(f"warmup {qno}/{len(selected)}", flush=True)
        for qno, query in enumerate(selected, start=1):
            expected = truth.get(bench.truth_key(query), [])
            for repeat in range(args.repeats):
                order = METHODS if (qno + repeat) % 2 == 0 else list(reversed(METHODS))
                for method in order:
                    rows.append(run_once(cur, args, method, query, repeat, expected))
            if args.progress_queries and qno % args.progress_queries == 0:
                ok = [r for r in rows if not r.get("error")]
                latest = summarize(ok)
                print(f"progress {qno}/{len(selected)} rows={len(rows)} summaries={len(latest)}", flush=True)
        cur.execute("SELECT vector_hnsw_guidance_reset()")

    write_csv(args.out, rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summarize(rows))
    write_csv(args.out.with_name(args.out.stem + "_paired_summary.csv"), paired_summary(rows))
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.out.with_name(args.out.stem + '_summary.csv')}", flush=True)
    print(f"wrote {args.out.with_name(args.out.stem + '_paired_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
