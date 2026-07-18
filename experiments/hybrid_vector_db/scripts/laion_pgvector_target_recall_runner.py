from __future__ import annotations

import argparse
import csv
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
BENCH = "experiments/hybrid_vector_db/scripts/laion_pgvector_filtered_benchmark.py"
FILTERS = [
    "topic_le_50p0",
    "topic_le_20p0",
    "topic_le_10p0",
    "topic_le_5p0",
    "topic_le_2p0",
    "topic_le_1p0",
    "topic_le_0p5",
    "topic_le_0p2",
]


@dataclass(frozen=True)
class Config:
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float

    @property
    def label(self) -> str:
        mem = str(self.scan_mem_multiplier).replace(".", "p")
        return f"ef{self.ef_search}_max{self.max_scan_tuples}_mem{mem}"


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def parse_config_specs(value: str) -> list[Config]:
    configs: list[Config] = []
    for item in value.split(","):
        if not item:
            continue
        ef, max_scan, mem = item.split(":")
        configs.append(Config(int(ef), int(max_scan), float(mem)))
    return configs


def run_command(cmd: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(cmd), flush=True)
    start = time.perf_counter()
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_benchmark(
    out: Path,
    filters: list[str],
    methods: list[str],
    queries: int,
    repeats: int,
    config: Config,
    args: argparse.Namespace,
    log: Path,
) -> float:
    cmd = [
        sys.executable,
        BENCH,
        "--benchmark-only",
        "--out",
        str(out),
        "--methods",
        *methods,
        "--filter-names",
        *filters,
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
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--progress-queries",
        str(args.progress_queries),
    ]
    return run_command(cmd, log)


def summary_path(raw: Path) -> Path:
    return raw.with_name(raw.stem + "_summary.csv")


def mean_latency(rows: list[dict[str, str]]) -> float:
    vals = [float(row["latency_ms_mean"]) for row in rows]
    return statistics.fmean(vals) if vals else float("inf")


def choose_config(candidates: list[dict[str, object]], target_recall: float) -> dict[str, object]:
    above = [row for row in candidates if float(row["recall_mean"]) >= target_recall]
    if above:
        return min(above, key=lambda row: float(row["latency_ms_mean"]))
    return max(candidates, key=lambda row: (float(row["recall_mean"]), -float(row["latency_ms_mean"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and run LAION10M pgvector fixed-recall tables.")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--filters", nargs="*", default=FILTERS)
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--calibration-queries", type=int, default=20)
    parser.add_argument("--calibration-repeats", type=int, default=1)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument("--final-methods", nargs="*", default=["stock", "bloom", "page"])
    parser.add_argument("--ef-search-values", default="1000,2000,4000,8000")
    parser.add_argument("--max-scan-tuples-values", default="500000,1000000,2000000,5000000")
    parser.add_argument("--scan-mem-multiplier-values", default="8,16,32,64")
    parser.add_argument(
        "--config-specs",
        default="",
        help="Optional comma-separated ef:max_scan:mem configs. Overrides cartesian config values.",
    )
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--skip-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sweep-all-configs", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.config_specs:
        configs = parse_config_specs(args.config_specs)
    else:
        configs = [
            Config(ef, max_scan, mem)
            for ef in parse_ints(args.ef_search_values)
            for max_scan in parse_ints(args.max_scan_tuples_values)
            for mem in parse_floats(args.scan_mem_multiplier_values)
        ]

    calibration_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    final_summary_rows: list[dict[str, object]] = []
    logs = RESULTS / "logs" / f"laion_fixed_recall_{args.tag}"

    for filter_name in args.filters:
        candidates: list[dict[str, object]] = []
        for config in configs:
            raw = RESULTS / f"laion10m_fixed_calib_{filter_name}_{config.label}_{args.tag}.csv"
            elapsed = run_benchmark(
                raw,
                [filter_name],
                ["stock"],
                args.calibration_queries,
                args.calibration_repeats,
                config,
                args,
                logs / f"calib_{filter_name}_{config.label}.log",
            )
            summary = read_summary(summary_path(raw))
            if not summary:
                continue
            row = summary[0]
            candidate = {
                "filter_name": filter_name,
                "config": config.label,
                "ef_search": config.ef_search,
                "max_scan_tuples": config.max_scan_tuples,
                "scan_mem_multiplier": config.scan_mem_multiplier,
                "recall_mean": float(row["recall_mean"]),
                "latency_ms_mean": float(row["latency_ms_mean"]),
                "latency_ms_p95": float(row["latency_ms_p95"]),
                "elapsed_ms": elapsed,
                "raw": str(raw),
                "summary": str(summary_path(raw)),
            }
            candidates.append(candidate)
            calibration_rows.append(candidate)
            if candidate["recall_mean"] >= args.target_recall and not args.sweep_all_configs:
                break
        if not candidates:
            continue
        selected = choose_config(candidates, args.target_recall)
        selected_rows.append(selected)
        if args.skip_final:
            continue

        config = Config(
            int(selected["ef_search"]),
            int(selected["max_scan_tuples"]),
            float(selected["scan_mem_multiplier"]),
        )
        final_raw = RESULTS / f"laion10m_fixed_final_{filter_name}_{config.label}_{args.tag}.csv"
        run_benchmark(
            final_raw,
            [filter_name],
            args.final_methods,
            args.final_queries,
            args.final_repeats,
            config,
            args,
            logs / f"final_{filter_name}_{config.label}.log",
        )
        for row in read_summary(summary_path(final_raw)):
            enriched: dict[str, object] = dict(row)
            enriched["selected_config"] = config.label
            enriched["ef_search"] = config.ef_search
            enriched["max_scan_tuples"] = config.max_scan_tuples
            enriched["scan_mem_multiplier"] = config.scan_mem_multiplier
            enriched["raw"] = str(final_raw)
            final_summary_rows.append(enriched)

    write_csv(RESULTS / f"laion10m_fixed_recall_calibration_{args.tag}.csv", calibration_rows)
    write_csv(RESULTS / f"laion10m_fixed_recall_selected_{args.tag}.csv", selected_rows)
    write_csv(RESULTS / f"laion10m_fixed_recall_final_summary_{args.tag}.csv", final_summary_rows)
    print(f"wrote calibration/selection/final summaries for tag {args.tag}", flush=True)


if __name__ == "__main__":
    main()
