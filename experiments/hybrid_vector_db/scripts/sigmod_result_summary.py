from __future__ import annotations

import argparse
import csv
import json
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


def fmean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(row.get(key, 0) or 0) for row in rows]
    return statistics.fmean(vals) if vals else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_csv(path: Path, group_keys: list[str], metric_keys: list[str]) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    out: list[dict[str, object]] = []
    for group, items in sorted(groups.items()):
        ok = [row for row in items if not row.get("error")]
        result: dict[str, object] = {key: value for key, value in zip(group_keys, group, strict=True)}
        result["samples"] = len(items)
        result["ok"] = len(ok)
        result["errors"] = len(items) - len(ok)
        for key in metric_keys:
            vals = [float(row.get(key, 0) or 0) for row in ok]
            result[f"{key}_mean"] = statistics.fmean(vals) if vals else 0.0
            result[f"{key}_p50"] = percentile(vals, 0.50)
            result[f"{key}_p95"] = percentile(vals, 0.95)
            result[f"{key}_std"] = stdev(vals)
        if ok and "guidance_checks" in ok[0] and "guidance_skips" in ok[0]:
            checks = fmean(ok, "guidance_checks")
            skips = fmean(ok, "guidance_skips")
            result["guidance_skip_rate"] = skips / checks if checks else 0.0
        out.append(result)
    return out


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


def summarize_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SIGMOD raw experiment outputs with mean/p50/p95/std.")
    parser.add_argument("--d123-raw", type=Path)
    parser.add_argument("--d4-raw", type=Path)
    parser.add_argument("--c4-memory-summary", type=Path)
    parser.add_argument("--c4-route-summary", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest: dict[str, object] = {}

    if args.d123_raw and args.d123_raw.exists():
        rows = summarize_csv(
            args.d123_raw,
            ["selectivity", "filter_name", "mode", "mode_label"],
            [
                "end_to_end_ms",
                "activation_ms",
                "query_latency_ms",
                "recall",
                "visited_tuples",
                "returned_tuples",
                "guidance_checks",
                "guidance_skips",
            ],
        )
        out = args.out_prefix.with_name(args.out_prefix.name + "_d123_summary.csv")
        write_csv(out, rows)
        manifest["d123_summary"] = str(out)

    if args.d4_raw and args.d4_raw.exists():
        rows = summarize_csv(
            args.d4_raw,
            ["selectivity", "filter_name", "chosen_route"],
            [
                "end_to_end_ms",
                "recall",
                "visited_tuples",
                "guidance_checks",
                "guidance_skips",
            ],
        )
        out = args.out_prefix.with_name(args.out_prefix.name + "_d4_summary.csv")
        write_csv(out, rows)
        manifest["d4_summary"] = str(out)

    if args.c4_memory_summary and args.c4_memory_summary.exists():
        manifest["c4_memory"] = summarize_json(args.c4_memory_summary)

    if args.c4_route_summary and args.c4_route_summary.exists():
        manifest["c4_route"] = summarize_json(args.c4_route_summary)

    manifest_out = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
