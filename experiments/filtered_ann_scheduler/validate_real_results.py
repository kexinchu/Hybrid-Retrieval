from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    errors: list[str] = []

    required = {
        "dataset",
        "n",
        "predicate",
        "strategy",
        "actual_selectivity",
        "match_count",
        "recall_at_k",
        "latency_ms",
        "sqlite_ms",
        "exact_distance_evals",
        "ann_returned",
        "predicate_evals",
        "failed_to_fill",
    }
    missing = required - set(df.columns)
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    else:
        if not df["recall_at_k"].between(0, 1).all():
            errors.append("recall_at_k outside [0, 1]")
        if not df["actual_selectivity"].between(0, 1).all():
            errors.append("actual_selectivity outside [0, 1]")
        for col in ["match_count", "latency_ms", "sqlite_ms", "exact_distance_evals", "ann_returned", "predicate_evals"]:
            if not (df[col] >= 0).all():
                errors.append(f"{col} has negative values")
        if set(df["failed_to_fill"].unique()) - {0, 1}:
            errors.append("failed_to_fill is not binary")
        exact = df[df["strategy"] == "sqlite_prefilter_exact"]
        if not (exact["recall_at_k"] >= 0.999).all():
            errors.append("sqlite_prefilter_exact is not exact")
        if not (exact["ann_returned"] == 0).all():
            errors.append("sqlite_prefilter_exact should not return ANN candidates")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALIDATION PASSED")
    print(f"rows={len(df)}")
    print(f"dataset={df['dataset'].iloc[0]}")
    print(f"n={int(df['n'].iloc[0])}")


if __name__ == "__main__":
    main()

