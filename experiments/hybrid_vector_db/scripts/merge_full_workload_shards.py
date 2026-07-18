from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from yfcc_full_workload_recall_sweep import summarize, write_csv  # noqa: E402


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (str(row["method"]), int(row["ef_search"]), int(row["max_scan_tuples"]), int(row["request_no"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge full-workload recall sweep shard CSVs.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for path in args.inputs:
        for row in read_csv(path):
            merged[row_key(row)] = row

    rows = sorted(merged.values(), key=lambda row: (row["method"], int(row["ef_search"]), int(row["max_scan_tuples"]), int(row["request_no"])))
    write_csv(args.out, rows)
    write_csv(args.summary_out, summarize(rows))
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    print(f"merged_rows={len(rows)} input_files={len(args.inputs)}")


if __name__ == "__main__":
    main()
