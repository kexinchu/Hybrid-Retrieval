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
        df.groupby(["workload", "strategy"])
        .agg(
            recall=("recall_at_k", "mean"),
            fail_rate=("failed_to_fill", "mean"),
            p50_ms=("latency_ms", "median"),
            p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
            pred_evals=("predicate_evals", "mean"),
            exact_dists=("exact_distance_evals", "mean"),
            ann_returned=("ann_returned", "mean"),
            actual_sel=("actual_selectivity", "mean"),
        )
        .reset_index()
        .sort_values(["workload", "strategy"])
    )

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBottleneck slices:")
    low_recall = summary[summary["recall"] < 0.9]
    if not low_recall.empty:
        print("\nStrategies with mean recall < 0.9")
        print(low_recall[["workload", "strategy", "recall", "fail_rate", "ann_returned"]].to_string(index=False))

    high_tail = summary.sort_values("p95_ms", ascending=False).head(10)
    print("\nHighest p95 latency")
    print(high_tail[["workload", "strategy", "p95_ms", "recall", "pred_evals", "exact_dists", "ann_returned"]].to_string(index=False))

    fails = summary[summary["fail_rate"] > 0.05]
    if not fails.empty:
        print("\nStrategies that often fail to return k filtered hits")
        print(fails[["workload", "strategy", "fail_rate", "recall", "ann_returned"]].to_string(index=False))


if __name__ == "__main__":
    main()

