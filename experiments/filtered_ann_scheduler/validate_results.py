from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_scalar(path: Path) -> list[str]:
    errors: list[str] = []
    df = pd.read_csv(path)
    required = {
        "workload",
        "strategy",
        "recall_at_k",
        "returned",
        "truth_size",
        "latency_ms",
        "predicate_evals",
        "exact_distance_evals",
        "ann_returned",
        "failed_to_fill",
        "actual_selectivity",
    }
    require(required.issubset(df.columns), f"scalar missing columns: {required - set(df.columns)}", errors)
    if errors:
        return errors
    require(df["recall_at_k"].between(0, 1).all(), "scalar recall outside [0, 1]", errors)
    require((df["latency_ms"] >= 0).all(), "scalar negative latency", errors)
    require((df["predicate_evals"] >= 0).all(), "scalar negative predicate evals", errors)
    require((df["exact_distance_evals"] >= 0).all(), "scalar negative exact distance evals", errors)
    require((df["ann_returned"] >= 0).all(), "scalar negative ann_returned", errors)
    require(df["actual_selectivity"].between(0, 1).all(), "scalar selectivity outside [0, 1]", errors)
    require(set(df["failed_to_fill"].unique()).issubset({0, 1}), "scalar failed_to_fill is not binary", errors)
    require((df[df["strategy"] == "pre_exact"]["recall_at_k"] >= 0.999).all(), "pre_exact is not exact", errors)
    return errors


def validate_multimodal(path: Path) -> list[str]:
    errors: list[str] = []
    df = pd.read_csv(path)
    required = {
        "workload",
        "strategy",
        "recall_at_k",
        "candidate_recall_at_k",
        "returned",
        "failed_to_fill",
        "latency_ms",
        "text_budget",
        "image_budget",
        "candidate_count",
        "filtered_candidate_count",
        "rerank_distance_evals",
        "actual_selectivity",
    }
    require(required.issubset(df.columns), f"multimodal missing columns: {required - set(df.columns)}", errors)
    if errors:
        return errors
    require(df["recall_at_k"].between(0, 1).all(), "multimodal recall outside [0, 1]", errors)
    require(df["candidate_recall_at_k"].between(0, 1).all(), "multimodal candidate recall outside [0, 1]", errors)
    require((df["candidate_recall_at_k"] + 1e-9 >= df["recall_at_k"]).all(), "candidate recall below final recall", errors)
    require((df["latency_ms"] >= 0).all(), "multimodal negative latency", errors)
    require((df["text_budget"] >= 0).all(), "multimodal negative text budget", errors)
    require((df["image_budget"] >= 0).all(), "multimodal negative image budget", errors)
    require((df["filtered_candidate_count"] <= df["candidate_count"]).all(), "filtered candidates exceed candidates", errors)
    require((df["rerank_distance_evals"] == 2 * df["filtered_candidate_count"]).all(), "rerank eval accounting mismatch", errors)
    require(df["actual_selectivity"].between(0, 1).all(), "multimodal selectivity outside [0, 1]", errors)
    require(set(df["failed_to_fill"].unique()).issubset({0, 1}), "multimodal failed_to_fill is not binary", errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--multimodal", type=Path, required=True)
    args = parser.parse_args()

    errors = []
    errors.extend(validate_scalar(args.scalar))
    errors.extend(validate_multimodal(args.multimodal))
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"scalar_rows={len(pd.read_csv(args.scalar))}")
    print(f"multimodal_rows={len(pd.read_csv(args.multimodal))}")


if __name__ == "__main__":
    main()

