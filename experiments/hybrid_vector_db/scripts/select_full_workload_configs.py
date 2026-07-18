from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_targets(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return (row["method"], int(row["ef_search"]), int(row["max_scan_tuples"]))


def score(row: dict[str, str], target: float) -> tuple[float, float]:
    recall = float(row["recall_mean"])
    latency = float(row["latency_mean_ms"])
    return (abs(recall - target), latency)


def select_configs(rows: list[dict[str, str]], method: str, targets: list[float], include_exact: bool) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["method"] == method
        and int(row["ef_search"]) > 0
        and int(row["max_scan_tuples"]) > 0
        and int(row.get("ok", row.get("requests", 0)) or 0) > 0
    ]
    exact = [
        row
        for row in rows
        if row["method"] == "exact"
        and int(row.get("ok", row.get("requests", 0)) or 0) > 0
    ]
    selected: dict[tuple[str, int, int], dict[str, Any]] = {}
    for target in targets:
        if target >= 0.999 and include_exact and exact:
            chosen = min(exact, key=lambda row: float(row["latency_mean_ms"]))
        else:
            chosen = min(candidates, key=lambda row, t=target: score(row, t))
        out = {
            "target_recall": target,
            "method": chosen["method"],
            "ef_search": int(chosen["ef_search"]),
            "max_scan_tuples": int(chosen["max_scan_tuples"]),
            "observed_recall": float(chosen["recall_mean"]),
            "latency_mean_ms": float(chosen["latency_mean_ms"]),
            "throughput_qps": float(chosen["single_client_throughput_qps"]),
        }
        selected[key(chosen)] = out
    return sorted(selected.values(), key=lambda row: (float(row["observed_recall"]), int(row["ef_search"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select full-workload sweep configs near target recall levels.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", default="bloom")
    parser.add_argument("--targets", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--include-exact", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = read_csv(args.summary)
    selected = select_configs(rows, args.method, parse_targets(args.targets), args.include_exact)
    write_csv(args.out, selected)

    ef_values = sorted({int(row["ef_search"]) for row in selected if int(row["ef_search"]) > 0})
    max_values = sorted({int(row["max_scan_tuples"]) for row in selected if int(row["max_scan_tuples"]) > 0})
    methods = sorted({row["method"] for row in selected})
    print(f"wrote {args.out}")
    print("selected methods:", ",".join(methods))
    print("ef_search_values:", ",".join(str(x) for x in ef_values))
    print("max_scan_tuples_values:", ",".join(str(x) for x in max_values))


if __name__ == "__main__":
    main()
