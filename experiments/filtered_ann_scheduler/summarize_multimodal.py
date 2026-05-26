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
            candidate_recall=("candidate_recall_at_k", "mean"),
            fail_rate=("failed_to_fill", "mean"),
            p50_ms=("latency_ms", "median"),
            p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
            text_budget=("text_budget", "mean"),
            image_budget=("image_budget", "mean"),
            filtered_candidates=("filtered_candidate_count", "mean"),
            rerank_dists=("rerank_distance_evals", "mean"),
            actual_sel=("actual_selectivity", "mean"),
        )
        .reset_index()
        .sort_values(["workload", "strategy"])
    )

    pd.set_option("display.max_rows", 240)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBottleneck slices:")
    weak = summary[(summary["candidate_recall"] < 0.8) | (summary["fail_rate"] > 0.2)]
    if not weak.empty:
        print("\nCandidate-source failures")
        print(
            weak[
                [
                    "workload",
                    "strategy",
                    "recall",
                    "candidate_recall",
                    "fail_rate",
                    "text_budget",
                    "image_budget",
                    "filtered_candidates",
                ]
            ].to_string(index=False)
        )

    tail = summary.sort_values("p95_ms", ascending=False).head(12)
    print("\nHighest p95 latency")
    print(
        tail[
            [
                "workload",
                "strategy",
                "p95_ms",
                "recall",
                "candidate_recall",
                "rerank_dists",
                "filtered_candidates",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

