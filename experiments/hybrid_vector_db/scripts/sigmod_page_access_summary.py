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


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def summarize(path: Path) -> dict[str, object]:
    with path.open(newline="") as infile:
        rows = [row for row in csv.DictReader(infile) if not row.get("error")]

    lat = [f(row, "elapsed_ms") for row in rows]
    visited = [f(row, "visited_tuples") for row in rows]
    runs = [f(row, "page_access_distance_runs") for row in rows]
    pages = [f(row, "page_access_distinct_pages") for row in rows]
    elem_runs = [f(row, "index_page_element_runs") for row in rows]
    elem_pages = [f(row, "index_page_element_distinct_pages") for row in rows]

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    return {
        "file": str(path),
        "rows": len(rows),
        "table": rows[0].get("table", "") if rows else "",
        "mode": rows[0].get("mode", "") if rows else "",
        "index_page_access": rows[0].get("index_page_access", "") if rows else "",
        "k": rows[0].get("k", "") if rows else "",
        "ef_search": rows[0].get("ef_search", "") if rows else "",
        "iterative_scan": rows[0].get("iterative_scan", "") if rows else "",
        "max_scan_tuples": rows[0].get("max_scan_tuples", "") if rows else "",
        "page_window": rows[0].get("page_window", "") if rows else "",
        "latency_mean_ms": mean(lat),
        "latency_p50_ms": percentile(lat, 0.50),
        "latency_p95_ms": percentile(lat, 0.95),
        "latency_std_ms": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "visited_tuples_mean": mean(visited),
        "heap_distance_runs_mean": mean(runs),
        "heap_distinct_pages_mean": mean(pages),
        "heap_run_per_page": mean(runs) / max(mean(pages), 1.0),
        "index_element_runs_mean": mean(elem_runs),
        "index_element_distinct_pages_mean": mean(elem_pages),
        "index_element_run_per_page": mean(elem_runs) / max(mean(elem_pages), 1.0),
        "index_prefetches_mean": mean([f(row, "index_page_prefetches") for row in rows]),
        "same_ordered_ids": sum(1 for row in rows if str(row.get("same_as_first_mode", "")).lower() == "true"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pgvector page-access group benchmark CSV files.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize(path) for path in args.inputs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
