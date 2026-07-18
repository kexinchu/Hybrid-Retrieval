from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from yfcc_full_workload_recall_sweep import summarize, write_csv  # noqa: E402


def parse_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(x) for x in value.replace(" ", ",").split(",") if x]


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ids[:k]) & set(truth[:k])) / min(k, len(truth))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute full-workload recall against SQL-first exact endpoint rows.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    rows = read_csv(args.raw)
    exact: dict[int, list[int]] = {}
    for row in rows:
        if row["method"] == "exact" and not row.get("error"):
            exact[int(row["request_no"])] = parse_ids(str(row["ids"]))
    if not exact:
        raise SystemExit("no exact endpoint rows found")

    rewritten: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        out = dict(row)
        truth = exact.get(int(row["request_no"]))
        if truth is None:
            missing += 1
        elif not row.get("error"):
            out["recall"] = recall_at_k(parse_ids(str(row["ids"])), truth, args.k)
        else:
            out["recall"] = 0.0
        out["truth_source"] = "sql_first_exact"
        rewritten.append(out)

    write_csv(args.out, rewritten)
    write_csv(args.summary_out, summarize(rewritten))
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_out}")
    print(f"exact_truth_requests={len(exact)} missing_truth_rows={missing}")


if __name__ == "__main__":
    main()
