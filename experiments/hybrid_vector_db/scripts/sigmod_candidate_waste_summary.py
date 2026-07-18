from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(row.get(key, 0) or 0) for row in rows]
    return statistics.fmean(vals) if vals else 0.0


def stdev(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(row.get(key, 0) or 0) for row in rows]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def summarize(raw: Path) -> list[dict[str, object]]:
    with raw.open(newline="") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (
                row.get("selectivity", ""),
                row.get("filter_name", ""),
                row.get("mode", ""),
                row.get("mode_label", row.get("mode", "")),
            ),
            [],
        ).append(row)

    out: list[dict[str, object]] = []
    for (selectivity, filter_name, mode, mode_label), items in sorted(groups.items()):
        ok = [row for row in items if not row.get("error")]
        returned = mean(ok, "returned")
        returned_tuples = mean(ok, "returned_tuples")
        visited = mean(ok, "visited_tuples")
        checks = mean(ok, "guidance_checks")
        skips = mean(ok, "guidance_skips")
        candidate_per_valid = returned_tuples / returned if returned else 0.0
        visited_per_valid = visited / returned if returned else 0.0
        sql_reject_rate = (returned_tuples - returned) / returned_tuples if returned_tuples else 0.0
        guidance_skip_rate = skips / checks if checks else 0.0
        latencies = [float(row.get("end_to_end_ms", 0) or 0) for row in ok]
        out.append(
            {
                "selectivity": selectivity,
                "filter_name": filter_name,
                "mode": mode,
                "mode_label": mode_label,
                "samples": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": mean(ok, "recall"),
                "latency_mean_ms": mean(ok, "end_to_end_ms"),
                "latency_p50_ms": percentile(latencies, 0.50),
                "latency_p95_ms": percentile(latencies, 0.95),
                "latency_std_ms": stdev(ok, "end_to_end_ms"),
                "visited_tuples_mean": visited,
                "returned_tuples_mean": returned_tuples,
                "final_returned_mean": returned,
                "candidate_per_valid_result": candidate_per_valid,
                "visited_per_valid_result": visited_per_valid,
                "sql_reject_rate_est": sql_reject_rate,
                "guidance_checks_mean": checks,
                "guidance_skips_mean": skips,
                "guidance_skip_rate": guidance_skip_rate,
            }
        )
    return out


def add_speedups(rows: list[dict[str, object]]) -> None:
    by_filter: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_filter.setdefault(str(row["filter_name"]), {})[str(row["mode"])] = row

    for filter_rows in by_filter.values():
        base = filter_rows.get("original")
        if not base:
            continue
        base_latency = float(base["latency_mean_ms"])
        base_candidates = float(base["candidate_per_valid_result"])
        base_returned = float(base["returned_tuples_mean"])
        for row in filter_rows.values():
            latency = float(row["latency_mean_ms"])
            candidates = float(row["candidate_per_valid_result"])
            returned = float(row["returned_tuples_mean"])
            row["latency_speedup_vs_original"] = base_latency / latency if latency else 0.0
            row["candidate_reduction_vs_original"] = base_candidates / candidates if candidates else 0.0
            row["returned_tuple_reduction_vs_original"] = base_returned / returned if returned else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SIGMOD candidate waste from D1-D3 raw pgvector output.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = summarize(args.raw)
    add_speedups(rows)
    write_csv(args.out, rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
