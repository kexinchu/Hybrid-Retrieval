from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Config:
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str

    @property
    def label(self) -> str:
        mem = str(self.scan_mem_multiplier).replace(".", "p")
        return f"ef{self.ef_search}_max{self.max_scan_tuples}_mem{mem}_{self.iterative_scan}"


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def run_command(cmd: list[str], log: Path | None = None) -> float:
    print(shlex.join(cmd), flush=True)
    start = time.perf_counter()
    if log is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
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


def summarize_raw(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["filter_name"], row["mode"]), []).append(row)

    out: list[dict[str, object]] = []
    for (filter_name, mode), items in sorted(groups.items()):
        ok = [row for row in items if not row.get("error")]
        recalls = [float(row["recall"]) for row in ok]
        latencies = [float(row["end_to_end_ms"]) for row in ok]
        out.append(
            {
                "filter_name": filter_name,
                "mode": mode,
                "samples": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": statistics.fmean(recalls) if recalls else 0.0,
                "recall_min_query_mean": min_query_mean(ok),
                "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
            }
        )
    return out


def min_query_mean(rows: list[dict[str, str]]) -> float:
    by_query: dict[str, list[float]] = {}
    for row in rows:
        by_query.setdefault(row["query_no"], []).append(float(row["recall"]))
    means = [statistics.fmean(vals) for vals in by_query.values()]
    return min(means) if means else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_d123(
    out: Path,
    filters: list[str],
    modes: list[str],
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
        "--repeats",
        str(repeats),
        "--ef-search",
        str(config.ef_search),
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
    ]
    if args.warmup_all_queries:
        cmd.append("--warmup-all-queries")
    if filters:
        cmd += ["--filter-names", *filters]
    if modes:
        cmd += ["--modes", *modes]
    return run_command(cmd, log)


def calibrate_filter(filter_name: str, configs: list[Config], args: argparse.Namespace) -> tuple[Config, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    selected_above: tuple[float, Config] | None = None
    selected_fallback: tuple[float, float, Config] | None = None
    for config in configs:
        out = RESULTS / f"sigmod_target_recall_calib_{filter_name}_{config.label}_{args.tag}.csv"
        log = RESULTS / "logs" / f"sigmod_target_{args.tag}" / f"calib_{filter_name}_{config.label}.log"
        elapsed_ms = run_d123(
            out,
            [filter_name],
            args.calibration_modes,
            args.calibration_queries,
            args.calibration_repeats,
            config,
            args,
            log,
        )
        summary = summarize_raw(out)
        recalls = [float(row["recall_mean"]) for row in summary if row["mode"] in args.calibration_modes]
        latency = statistics.fmean(float(row["latency_mean_ms"]) for row in summary) if summary else float("inf")
        recall_mean = min(recalls) if recalls else 0.0
        row = {
            "filter_name": filter_name,
            "config": config.label,
            "ef_search": config.ef_search,
            "max_scan_tuples": config.max_scan_tuples,
            "scan_mem_multiplier": config.scan_mem_multiplier,
            "iterative_scan": config.iterative_scan,
            "calibration_modes": ",".join(args.calibration_modes),
            "recall_mean_min_mode": recall_mean,
            "latency_mean_ms": latency,
            "elapsed_ms": elapsed_ms,
            "raw": str(out),
            "log": str(log),
        }
        rows.append(row)
        if recall_mean >= args.target_recall:
            if selected_above is None or latency < selected_above[0]:
                selected_above = (latency, config)
            if not args.sweep_all_configs:
                break
        if selected_fallback is None or recall_mean > selected_fallback[0] or (
            recall_mean == selected_fallback[0] and latency < selected_fallback[1]
        ):
            selected_fallback = (recall_mean, latency, config)
    selected = selected_above[1] if selected_above is not None else selected_fallback[2]
    return selected, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate pgvector D1-D3 to a target recall, then run final q/r table per filter.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--filters", nargs="*", default=FILTER_ORDER)
    parser.add_argument("--calibration-queries", type=int, default=20)
    parser.add_argument("--calibration-repeats", type=int, default=1)
    parser.add_argument("--calibration-modes", nargs="*", default=["original", "design1_bloom_bfs_layout_d3"])
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=10)
    parser.add_argument("--final-modes", nargs="*", default=["original", "design1_bloom", "design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"])
    parser.add_argument("--ef-search-values", default="1000")
    parser.add_argument("--max-scan-tuples-values", default="200000,500000,1000000,2000000")
    parser.add_argument("--scan-mem-multiplier-values", default="8,32")
    parser.add_argument("--iterative-scan-values", default="strict_order")
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-all-configs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=10)
    args = parser.parse_args()

    configs = [
        Config(ef, max_scan, mem, iterative)
        for iterative in [x for x in args.iterative_scan_values.split(",") if x]
        for ef in parse_ints(args.ef_search_values)
        for max_scan in parse_ints(args.max_scan_tuples_values)
        for mem in parse_floats(args.scan_mem_multiplier_values)
    ]

    calibration_rows: list[dict[str, object]] = []
    final_manifest: dict[str, object] = {
        "tag": args.tag,
        "target_recall": args.target_recall,
        "configs": [config.__dict__ for config in configs],
        "filters": {},
    }

    for filter_name in args.filters:
        selected, rows = calibrate_filter(filter_name, configs, args)
        calibration_rows.extend(rows)
        if args.skip_final:
            final_manifest["filters"][filter_name] = {
                "selected_config": selected.__dict__,
                "final_raw": "",
                "final_summary": "",
                "final_log": "",
                "final_elapsed_ms": 0.0,
                "final_recall_min_mode": 0.0,
                "skipped_final": True,
            }
            continue
        final_out = RESULTS / f"sigmod_target_recall_final_{filter_name}_{selected.label}_q{args.final_queries}r{args.final_repeats}_{args.tag}.csv"
        final_log = RESULTS / "logs" / f"sigmod_target_{args.tag}" / f"final_{filter_name}_{selected.label}.log"
        final_elapsed_ms = run_d123(
            final_out,
            [filter_name],
            args.final_modes,
            args.final_queries,
            args.final_repeats,
            selected,
            args,
            final_log,
        )
        final_summary = summarize_raw(final_out)
        summary_out = final_out.with_name(final_out.stem + "_target_summary.csv")
        write_csv(summary_out, final_summary)
        final_manifest["filters"][filter_name] = {
            "selected_config": selected.__dict__,
            "final_raw": str(final_out),
            "final_summary": str(summary_out),
            "final_log": str(final_log),
            "final_elapsed_ms": final_elapsed_ms,
            "final_recall_min_mode": min((float(row["recall_mean"]) for row in final_summary), default=0.0),
        }

    calibration_out = RESULTS / f"sigmod_target_recall_calibration_{args.tag}.csv"
    write_csv(calibration_out, calibration_rows)
    manifest_out = RESULTS / f"sigmod_target_recall_manifest_{args.tag}.json"
    manifest_out.write_text(json.dumps(final_manifest, indent=2) + "\n")
    print(f"wrote {calibration_out}")
    print(f"wrote {manifest_out}")


if __name__ == "__main__":
    main()
