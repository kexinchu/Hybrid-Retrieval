from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    summary = (
        df.groupby(["predicate", "strategy"])
        .agg(
            recall=("recall_at_k", "mean"),
            fail_rate=("failed_to_fill", "mean"),
            p50_ms=("latency_ms", "median"),
            p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
            sqlite_ms=("sqlite_ms", "mean"),
            exact_dists=("exact_distance_evals", "mean"),
            ann_returned=("ann_returned", "mean"),
            pred_evals=("predicate_evals", "mean"),
            selectivity=("actual_selectivity", "mean"),
            matches=("match_count", "mean"),
        )
        .reset_index()
        .sort_values(["predicate", "strategy"])
    )
    pd.set_option("display.max_rows", 240)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFailure slices:")
    failures = summary[(summary["recall"] < 0.9) | (summary["fail_rate"] > 0.2)].sort_values(
        ["fail_rate", "recall"], ascending=[False, True]
    )
    if failures.empty:
        print("_None_")
    else:
        print(
            failures[
                ["predicate", "strategy", "recall", "fail_rate", "p95_ms", "ann_returned", "selectivity", "matches"]
            ].to_string(index=False)
        )

    print("\nHighest p95 latency:")
    print(
        summary.sort_values("p95_ms", ascending=False)
        .head(12)[
            ["predicate", "strategy", "p95_ms", "recall", "exact_dists", "ann_returned", "selectivity"]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

