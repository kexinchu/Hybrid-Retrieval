from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def scalar_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    summary = (
        df.groupby(["workload", "strategy"])
        .agg(
            recall=("recall_at_k", "mean"),
            fail_rate=("failed_to_fill", "mean"),
            p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
            pred_evals=("predicate_evals", "mean"),
            exact_dists=("exact_distance_evals", "mean"),
            ann_returned=("ann_returned", "mean"),
            actual_sel=("actual_selectivity", "mean"),
        )
        .reset_index()
    )
    failures = summary[(summary["recall"] < 0.9) | (summary["fail_rate"] > 0.2)].sort_values(
        ["fail_rate", "recall"], ascending=[False, True]
    )
    tail = summary.sort_values("p95_ms", ascending=False).head(8)
    return summary, failures, tail


def multimodal_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    summary = (
        df.groupby(["workload", "strategy"])
        .agg(
            recall=("recall_at_k", "mean"),
            candidate_recall=("candidate_recall_at_k", "mean"),
            fail_rate=("failed_to_fill", "mean"),
            p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
            text_budget=("text_budget", "mean"),
            image_budget=("image_budget", "mean"),
            filtered_candidates=("filtered_candidate_count", "mean"),
            rerank_dists=("rerank_distance_evals", "mean"),
            actual_sel=("actual_selectivity", "mean"),
        )
        .reset_index()
    )
    failures = summary[
        (summary["candidate_recall"] < 0.8) | (summary["fail_rate"] > 0.2)
    ].sort_values(["fail_rate", "candidate_recall"], ascending=[False, True])
    tail = summary.sort_values("p95_ms", ascending=False).head(8)
    return summary, failures, tail


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_None._"
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--multimodal", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/motivation_report.md"))
    args = parser.parse_args()

    scalar_summary, scalar_failures, scalar_tail = scalar_tables(args.scalar)
    mm_summary, mm_failures, mm_tail = multimodal_tables(args.multimodal)

    lines = [
        "# Motivation Test Report: Filtered ANN Scheduling",
        "",
        "## Current Takeaways",
        "",
        "1. ANN-first filtering is brittle when the structured predicate is selective and weakly correlated with the vector neighborhood. It can spend thousands of ANN candidates and still fail to fill top-k.",
        "2. Pre-filter exact search is robust for very selective filters, but it becomes expensive as selectivity grows because it evaluates the predicate over the full table and computes exact distances for all matches.",
        "3. A scheduler needs runtime evidence about selectivity and vector/filter correlation. Selectivity alone is not enough: the random-filter and aligned-filter cases behave very differently at similar pass rates.",
        "4. In the multimodal proxy, candidate-source scheduling matters. When the filter aligns with one modality, searching the wrong modality wastes budget and returns too few filtered candidates.",
        "5. Random low-selectivity filters create the hardest case for both scalar and multimodal settings: fixed overfetch and fixed modality splits fail for the same reason, candidate admission is not filter-aware.",
        "",
        "## Scalar Filtered ANN Bottlenecks",
        "",
        md_table(
            scalar_failures,
            ["workload", "strategy", "recall", "fail_rate", "p95_ms", "ann_returned", "actual_sel"],
        ),
        "",
        "### Scalar Tail Latency",
        "",
        md_table(
            scalar_tail,
            ["workload", "strategy", "p95_ms", "recall", "pred_evals", "exact_dists", "ann_returned"],
            max_rows=8,
        ),
        "",
        "## Multimodal Candidate-Source Bottlenecks",
        "",
        md_table(
            mm_failures,
            [
                "workload",
                "strategy",
                "recall",
                "candidate_recall",
                "fail_rate",
                "text_budget",
                "image_budget",
                "filtered_candidates",
            ],
        ),
        "",
        "### Multimodal Tail Latency",
        "",
        md_table(
            mm_tail,
            ["workload", "strategy", "p95_ms", "recall", "candidate_recall", "rerank_dists"],
            max_rows=8,
        ),
        "",
        "## Scheduler Hypotheses To Test Next",
        "",
        "- Runtime plan choice: choose between filter-first, ANN-first, and hybrid expansion using estimated selectivity plus a small neighborhood pass-rate probe.",
        "- Adaptive overfetch: grow ANN budget based on observed filtered-hit yield, but switch to filter-first when marginal yield collapses.",
        "- Candidate-source scheduling: allocate text/image/vector-field budgets using filtered-hit yield from cheap probes, not fixed 50/50 splits.",
        "- Correlation-aware stopping: stop expanding a candidate source when its filtered-hit rate is below the estimated global pass rate.",
        "- Tail-latency control: optimize P95/P99 by capping repeated ANN expansions and falling back to structured-index injection.",
        "",
        "## Files",
        "",
        f"- Scalar CSV: `{args.scalar}`",
        f"- Multimodal CSV: `{args.multimodal}`",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

