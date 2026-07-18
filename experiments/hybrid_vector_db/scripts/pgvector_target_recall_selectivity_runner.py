from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
FILTER_ORDER = [
    "popular_ge1000",
    "price_10_to_20",
    "rating5_price_le10",
    "long_review_ge500",
    "grocery_rating5",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
]
DEFAULT_MODES = [
    "original",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
]


@dataclass(frozen=True)
class Config:
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str
    guided_collect_target: int

    @property
    def label(self) -> str:
        mem = str(self.scan_mem_multiplier).replace(".", "p")
        return (
            f"ef{self.ef_search}_target{self.guided_collect_target}_"
            f"max{self.max_scan_tuples}_mem{mem}_{self.iterative_scan}"
        )


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_targets(value: str) -> list[float]:
    targets = sorted(set(parse_floats(value)))
    if not targets or any(target <= 0 or target > 1 for target in targets):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return targets


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def query_means(rows: list[dict[str, str]], field: str) -> list[float]:
    by_query: dict[str, list[float]] = {}
    for row in rows:
        by_query.setdefault(row["query_no"], []).append(float(row[field]))
    return [statistics.fmean(values) for _, values in sorted(by_query.items())]


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    means = [statistics.fmean(rng.choices(values, k=size)) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def run_command(cmd: list[str], log: Path | None = None) -> float:
    print(shlex.join(cmd), flush=True)
    start = time.perf_counter()
    if log is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            f.write("$ " + shlex.join(cmd) + "\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
    return (time.perf_counter() - start) * 1000.0


def summarize_raw(path: Path, bootstrap_samples: int = 2000, seed: int = 20260718) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["filter_name"], row["mode"]), []).append(row)

    out: list[dict[str, object]] = []
    for group_no, ((filter_name, mode), items) in enumerate(sorted(groups.items())):
        ok = [row for row in items if not row.get("error")]
        recalls = [float(row["recall"]) for row in ok]
        latencies = [float(row["end_to_end_ms"]) for row in ok]
        latency_query_means = query_means(ok, "end_to_end_ms")
        recall_query_means = query_means(ok, "recall")
        ci_low, ci_high = bootstrap_mean_ci(
            latency_query_means,
            bootstrap_samples,
            seed + group_no,
        )
        out.append(
            {
                "filter_name": filter_name,
                "mode": mode,
                "queries": len(latency_query_means),
                "samples": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": statistics.fmean(recalls) if recalls else 0.0,
                "recall_min_query_mean": min(recall_query_means) if recall_query_means else 0.0,
                "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
                "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
                "latency_p95_ms": percentile(latencies, 0.95),
                "latency_p99_ms": percentile(latencies, 0.99),
                "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "latency_query_mean_ci95_low_ms": ci_low,
                "latency_query_mean_ci95_high_ms": ci_high,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reusable_summary(
    path: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        summary = summarize_raw(path, bootstrap_samples, bootstrap_seed)
    except (KeyError, TypeError, ValueError, csv.Error):
        return None
    return summary if len(summary) == 1 else None


def append_option(cmd: list[str], name: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def run_d123(
    out: Path,
    filter_name: str,
    mode: str,
    query_offset: int,
    queries: int,
    repeats: int,
    config: Config,
    args: argparse.Namespace,
    log: Path | None,
) -> float:
    cmd = [
        sys.executable,
        "experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py",
        "--out",
        str(out),
        "--queries",
        str(queries),
        "--query-offset",
        str(query_offset),
        "--repeats",
        str(repeats),
        "--ef-search",
        str(config.ef_search),
        "--guided-collect-target",
        str(config.guided_collect_target),
        "--max-scan-tuples",
        str(config.max_scan_tuples),
        "--scan-mem-multiplier",
        str(config.scan_mem_multiplier),
        "--iterative-scan",
        config.iterative_scan,
        "--progress-queries",
        str(args.progress_queries),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--filter-names",
        filter_name,
        "--modes",
        mode,
        "--guidance-filter-strategy",
        args.guidance_filter_strategy,
        "--guidance-selectivity-max-pct",
        str(args.guidance_selectivity_max_pct),
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--d2-page-access",
        args.d2_page_access,
        "--d2-index-page-access",
        args.d2_index_page_access,
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--d3-cache-mb",
        str(args.d3_cache_mb),
    ]
    append_option(cmd, "--filters-csv", args.filters_csv)
    append_option(cmd, "--truth-csv", args.truth_csv)
    append_option(cmd, "--insertion-table", args.insertion_table)
    append_option(cmd, "--insertion-index", args.insertion_index)
    append_option(cmd, "--bfs-table", args.bfs_table)
    append_option(cmd, "--bfs-index", args.bfs_index)
    if args.warmup_all_queries:
        cmd.append("--warmup-all-queries")
    if not args.force_hnsw:
        cmd.append("--no-force-hnsw")
    if not args.d3_reuse_active_guidance:
        cmd.append("--no-d3-reuse-active-guidance")
    return run_command(cmd, log)


def configs_for_mode(configs: list[Config], mode: str) -> list[Config]:
    if mode != "original":
        return configs
    # guided_collect_target has no effect on stock pgvector. Do not rerun duplicates.
    unique: dict[tuple[int, int, float, str], Config] = {}
    for config in configs:
        key = (
            config.ef_search,
            config.max_scan_tuples,
            config.scan_mem_multiplier,
            config.iterative_scan,
        )
        unique.setdefault(key, config)
    return list(unique.values())


def calibrate_mode_filter(
    filter_name: str,
    mode: str,
    configs: list[Config],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config in configs_for_mode(configs, mode):
        stem = f"sigmod_matched_calib_{filter_name}_{mode}_{config.label}_{args.tag}"
        out = RESULTS / f"{stem}.csv"
        log = RESULTS / "logs" / f"sigmod_matched_{args.tag}" / f"{stem}.log"
        summary = reusable_summary(out, args.bootstrap_samples, args.bootstrap_seed) if args.resume else None
        if summary is None:
            elapsed_ms = run_d123(
                out,
                filter_name,
                mode,
                args.calibration_query_offset,
                args.calibration_queries,
                args.calibration_repeats,
                config,
                args,
                log,
            )
            summary = summarize_raw(out, args.bootstrap_samples, args.bootstrap_seed)
            if len(summary) != 1:
                raise RuntimeError(f"expected one calibration summary in {out}, got {len(summary)}")
        else:
            elapsed_ms = 0.0
            print(f"reusing {out}", flush=True)
        if int(summary[0]["queries"]) != args.calibration_queries:
            raise RuntimeError(
                f"calibration query split is incomplete in {out}: "
                f"expected {args.calibration_queries}, got {summary[0]['queries']}"
            )
        row = {
            "filter_name": filter_name,
            "mode": mode,
            "config": config.label,
            **asdict(config),
            **summary[0],
            "elapsed_ms": elapsed_ms,
            "raw": str(out),
            "log": str(log),
        }
        rows.append(row)
    return rows


def select_row(rows: list[dict[str, object]], target: float) -> tuple[dict[str, object], bool]:
    valid = [row for row in rows if int(row["ok"]) > 0 and int(row["errors"]) == 0]
    if not valid:
        raise RuntimeError("all calibration configurations failed")
    reached = [row for row in valid if float(row["recall_mean"]) >= target]
    if reached:
        return min(reached, key=lambda row: (float(row["latency_mean_ms"]), float(row["recall_mean"]))), True
    return max(valid, key=lambda row: (float(row["recall_mean"]), -float(row["latency_mean_ms"]))), False


def config_from_row(row: dict[str, object]) -> Config:
    return Config(
        ef_search=int(row["ef_search"]),
        max_scan_tuples=int(row["max_scan_tuples"]),
        scan_mem_multiplier=float(row["scan_mem_multiplier"]),
        iterative_scan=str(row["iterative_scan"]),
        guided_collect_target=int(row["guided_collect_target"]),
    )


def build_configs(args: argparse.Namespace) -> list[Config]:
    ef_values = parse_ints(args.ef_search_values)
    target_tokens = [token.strip() for token in args.guided_collect_target_values.split(",") if token.strip()]
    configs: list[Config] = []
    for iterative in [x for x in args.iterative_scan_values.split(",") if x]:
        for ef in ef_values:
            targets = [ef if token == "ef" else int(token) for token in target_tokens]
            for target in sorted(set(targets)):
                for max_scan in parse_ints(args.max_scan_tuples_values):
                    for mem in parse_floats(args.scan_mem_multiplier_values):
                        configs.append(Config(ef, max_scan, mem, iterative, target))
    if not configs:
        raise SystemExit("empty calibration configuration space")
    return configs


def selected_rows(
    calibration_rows: list[dict[str, object]],
    filters: list[str],
    modes: list[str],
    targets: list[float],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for filter_name in filters:
        for mode in modes:
            candidates = [
                row
                for row in calibration_rows
                if row["filter_name"] == filter_name and row["mode"] == mode
            ]
            for target in targets:
                selected, met = select_row(candidates, target)
                out.append(
                    {
                        "target_recall": target,
                        "target_met_in_calibration": met,
                        **selected,
                    }
                )
    return out


def run_final_unique(
    selected: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], dict[str, object]]:
    results: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in selected:
        key = (str(row["filter_name"]), str(row["mode"]), str(row["config"]))
        if key in results:
            continue
        filter_name, mode, _ = key
        config = config_from_row(row)
        stem = (
            f"sigmod_matched_final_{filter_name}_{mode}_{config.label}_"
            f"q{args.final_queries}r{args.final_repeats}_{args.tag}"
        )
        raw = RESULTS / f"{stem}.csv"
        log = RESULTS / "logs" / f"sigmod_matched_{args.tag}" / f"{stem}.log"
        summary = reusable_summary(raw, args.bootstrap_samples, args.bootstrap_seed) if args.resume else None
        if summary is None:
            elapsed_ms = run_d123(
                raw,
            filter_name,
            mode,
            args.final_query_offset,
            args.final_queries,
                args.final_repeats,
                config,
                args,
                log,
            )
            summary = summarize_raw(raw, args.bootstrap_samples, args.bootstrap_seed)
            if len(summary) != 1:
                raise RuntimeError(f"expected one final summary in {raw}, got {len(summary)}")
        else:
            elapsed_ms = 0.0
            print(f"reusing {raw}", flush=True)
        if int(summary[0]["queries"]) != args.final_queries:
            raise RuntimeError(
                f"final query split is incomplete in {raw}: "
                f"expected {args.final_queries}, got {summary[0]['queries']}"
            )
        results[key] = {
            **summary[0],
            "final_elapsed_ms": elapsed_ms,
            "final_raw": str(raw),
            "final_log": str(log),
        }
    return results


def consolidate_final(
    selected: list[dict[str, object]],
    final_results: dict[tuple[str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selected_row in selected:
        key = (
            str(selected_row["filter_name"]),
            str(selected_row["mode"]),
            str(selected_row["config"]),
        )
        final = final_results[key]
        rows.append(
            {
                "target_recall": selected_row["target_recall"],
                "target_met_in_calibration": selected_row["target_met_in_calibration"],
                "target_met_in_final": float(final["recall_mean"]) >= float(selected_row["target_recall"]),
                "filter_name": selected_row["filter_name"],
                "mode": selected_row["mode"],
                "config": selected_row["config"],
                "ef_search": selected_row["ef_search"],
                "guided_collect_target": selected_row["guided_collect_target"],
                "max_scan_tuples": selected_row["max_scan_tuples"],
                "scan_mem_multiplier": selected_row["scan_mem_multiplier"],
                "iterative_scan": selected_row["iterative_scan"],
                **final,
            }
        )

    stock_latency = {
        (float(row["target_recall"]), str(row["filter_name"])): float(row["latency_mean_ms"])
        for row in rows
        if row["mode"] == "original"
    }
    for row in rows:
        baseline = stock_latency.get((float(row["target_recall"]), str(row["filter_name"])))
        latency = float(row["latency_mean_ms"])
        row["speedup_vs_stock"] = baseline / latency if baseline and latency > 0 else 0.0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently tune stock pgvector and every SQLens variant, then compare "
            "their fastest configurations at matched recall targets."
        )
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-recalls", default="0.90,0.95,0.99")
    parser.add_argument("--target-recall", type=float, help="Backward-compatible single recall target")
    parser.add_argument("--filters", nargs="*", default=FILTER_ORDER)
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--calibration-queries", type=int, default=30)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    parser.add_argument("--calibration-query-offset", type=int, default=0)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument("--final-query-offset", type=int)
    parser.add_argument("--allow-overlapping-query-splits", action="store_true")
    parser.add_argument("--ef-search-values", default="250,500,1000,2000,5000,10000")
    parser.add_argument("--guided-collect-target-values", default="ef")
    parser.add_argument("--max-scan-tuples-values", default="200000,1000000,5000000")
    parser.add_argument("--scan-mem-multiplier-values", default="8,32")
    parser.add_argument("--iterative-scan-values", default="strict_order")
    parser.add_argument("--filters-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--insertion-table")
    parser.add_argument("--insertion-index")
    parser.add_argument("--bfs-table")
    parser.add_argument("--bfs-index")
    parser.add_argument("--guidance-filter-strategy", default="guided_collect", choices=["guided_collect", "acorn1"])
    parser.add_argument("--guidance-selectivity-max-pct", type=float, default=10.0)
    parser.add_argument("--guidance-max-atoms", type=int, default=64)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-reuse-active-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()

    if args.final_query_offset is None:
        args.final_query_offset = args.calibration_query_offset + args.calibration_queries
    calibration_range = range(
        args.calibration_query_offset,
        args.calibration_query_offset + args.calibration_queries,
    )
    final_range = range(args.final_query_offset, args.final_query_offset + args.final_queries)
    if not args.allow_overlapping_query_splits and set(calibration_range).intersection(final_range):
        raise SystemExit("calibration and final query splits overlap")

    targets = [args.target_recall] if args.target_recall is not None else parse_targets(args.target_recalls)
    configs = build_configs(args)
    calibration_rows: list[dict[str, object]] = []
    for filter_name in args.filters:
        for mode in args.modes:
            print(f"calibrating filter={filter_name} mode={mode}", flush=True)
            calibration_rows.extend(calibrate_mode_filter(filter_name, mode, configs, args))

    calibration_out = RESULTS / f"sigmod_matched_recall_calibration_{args.tag}.csv"
    write_csv(calibration_out, calibration_rows)
    selected = selected_rows(calibration_rows, args.filters, args.modes, targets)
    selected_out = RESULTS / f"sigmod_matched_recall_selected_{args.tag}.csv"
    write_csv(selected_out, selected)

    final_rows: list[dict[str, object]] = []
    if not args.skip_final:
        final_results = run_final_unique(selected, args)
        final_rows = consolidate_final(selected, final_results)
        write_csv(RESULTS / f"sigmod_matched_recall_final_{args.tag}.csv", final_rows)

    manifest = {
        "tag": args.tag,
        "targets": targets,
        "filters": args.filters,
        "modes": args.modes,
        "calibration_queries": args.calibration_queries,
        "calibration_repeats": args.calibration_repeats,
        "calibration_query_offset": args.calibration_query_offset,
        "final_queries": args.final_queries,
        "final_repeats": args.final_repeats,
        "final_query_offset": args.final_query_offset,
        "config_count": len(configs),
        "configs": [asdict(config) for config in configs],
        "calibration": str(calibration_out),
        "selected": str(selected_out),
        "final": str(RESULTS / f"sigmod_matched_recall_final_{args.tag}.csv") if final_rows else "",
    }
    manifest_out = RESULTS / f"sigmod_matched_recall_manifest_{args.tag}.json"
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {calibration_out}", flush=True)
    print(f"wrote {selected_out}", flush=True)
    print(f"wrote {manifest_out}", flush=True)


if __name__ == "__main__":
    main()
