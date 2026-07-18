from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any


AMAZON_SQLENS_MODES = {
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
}
VARIANT_SQLENS_METHODS = {"d1", "d1_d2", "d1_d2_d3", "bloom", "page"}
METHOD_LABELS = {
    "stock": "Stock pgvector",
    "sqlens": "SQLens",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]


def method_from_amazon(row: dict[str, str]) -> str | None:
    mode = row.get("mode", "")
    if mode == "original":
        return "stock"
    if mode in AMAZON_SQLENS_MODES:
        return "sqlens"
    return None


def method_from_variant(row: dict[str, str], sqlens_method: str) -> str | None:
    method = row.get("method", "")
    if method == "stock":
        return "stock"
    if method == sqlens_method:
        return "sqlens"
    if sqlens_method == "any" and method in VARIANT_SQLENS_METHODS:
        return "sqlens"
    return None


def normalize_amazon_summary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        method = method_from_amazon(row)
        if method is None:
            continue
        rows.append(
            {
                "dataset": "Amazon-10M",
                "workload": row.get("workload", "amazon_random_selectivity"),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "source_method": row.get("mode", ""),
                "ef_search": as_int(row, "ef_search"),
                "max_scan_tuples": as_int(row, "max_scan_tuples"),
                "guided_collect_target": as_int(row, "guided_collect_target"),
                "samples": as_int(row, "samples"),
                "ok": as_int(row, "ok"),
                "errors": as_int(row, "errors"),
                "recall_mean": as_float(row, "recall_mean"),
                "recall_p50": as_float(row, "recall_p50"),
                "recall_min": as_float(row, "recall_min"),
                "latency_mean_ms": as_float(row, "latency_mean_ms"),
                "latency_p50_ms": as_float(row, "latency_p50_ms"),
                "latency_p95_ms": as_float(row, "latency_p95_ms"),
                "single_client_throughput_qps": as_float(row, "single_client_throughput_qps"),
                "visited_tuples_mean": as_float(row, "visited_tuples_mean"),
                "returned_tuples_mean": as_float(row, "returned_tuples_mean"),
                "guidance_skip_rate": as_float(row, "guidance_skip_rate"),
                "source_file": path.name,
            }
        )
    return rows


def normalize_variant_raw_paths(dataset: str, paths: list[Path], sqlens_method: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, str]]] = {}
    source_files: dict[tuple[str, int, int, int], set[str]] = {}
    for path in paths:
        for row in read_csv(path):
            if row.get("error"):
                continue
            method = method_from_variant(row, sqlens_method)
            if method is None:
                continue
            key = (
                method,
                as_int(row, "ef_search"),
                as_int(row, "max_scan_tuples"),
                as_int(row, "guided_collect_target"),
            )
            grouped.setdefault(key, []).append(row)
            source_files.setdefault(key, set()).add(path.name)

    out: list[dict[str, Any]] = []
    for (method, ef_search, max_scan, guided_target), items in sorted(grouped.items()):
        latencies = [
            as_float(row, "end_to_end_ms", as_float(row, "latency_ms") + as_float(row, "activation_ms"))
            for row in items
        ]
        recalls = [as_float(row, "recall") for row in items]
        visited = [as_float(row, "visited_tuples") for row in items]
        returned = [as_float(row, "returned_tuples") for row in items]
        checks = [as_float(row, "guidance_checks") for row in items]
        skips = [as_float(row, "guidance_skips") for row in items]
        total_ms = sum(latencies)
        out.append(
            {
                "dataset": dataset,
                "workload": "random_selectivity",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "source_method": sqlens_method if method == "sqlens" else "stock",
                "ef_search": ef_search,
                "max_scan_tuples": max_scan,
                "guided_collect_target": guided_target,
                "samples": len(items),
                "ok": len(items),
                "errors": 0,
                "recall_mean": statistics.fmean(recalls) if recalls else 0.0,
                "recall_p50": statistics.median(recalls) if recalls else 0.0,
                "recall_min": min(recalls) if recalls else 0.0,
                "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
                "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
                "latency_p95_ms": p95(latencies),
                "single_client_throughput_qps": (1000.0 * len(items) / total_ms) if total_ms > 0 else 0.0,
                "visited_tuples_mean": statistics.fmean(visited) if visited else 0.0,
                "returned_tuples_mean": statistics.fmean(returned) if returned else 0.0,
                "guidance_skip_rate": (
                    statistics.fmean(skips) / statistics.fmean(checks)
                    if checks and statistics.fmean(checks) > 0
                    else 0.0
                ),
                "source_file": ";".join(sorted(source_files.get((method, ef_search, max_scan, guided_target), set()))),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate mixed random-selectivity frontier summaries.")
    parser.add_argument("--amazon-summary", type=Path, nargs="*", default=[])
    parser.add_argument("--yfcc-raw", type=Path, nargs="*", default=[])
    parser.add_argument("--laion-raw", type=Path, nargs="*", default=[])
    parser.add_argument("--yfcc-sqlens-method", default="d1_d2_d3")
    parser.add_argument("--laion-sqlens-method", default="d1_d2_d3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.amazon_summary:
        rows.extend(normalize_amazon_summary(path))
    rows.extend(normalize_variant_raw_paths("YFCC-10M", args.yfcc_raw, args.yfcc_sqlens_method))
    rows.extend(normalize_variant_raw_paths("LAION-25M", args.laion_raw, args.laion_sqlens_method))

    rows = sorted(rows, key=lambda r: (str(r["dataset"]), str(r["method"]), int(r["ef_search"]), int(r["max_scan_tuples"])))
    write_csv(args.out, rows)
    print(f"wrote {args.out}", flush=True)
    for row in rows:
        print(
            f"{row['dataset']}\t{row['method_label']}\tef={row['ef_search']}\t"
            f"recall={float(row['recall_mean']):.3f}\tlat={float(row['latency_mean_ms']):.2f}ms\t"
            f"qps={float(row['single_client_throughput_qps']):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
