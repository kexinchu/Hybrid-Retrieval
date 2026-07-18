from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if not row.get("standard_error") and not row.get("cache_error")]


def online_profile_simulation(rows: list[dict[str, str]], min_samples: int) -> dict[str, object]:
    history: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: {"standard": [], "cache": []})
    choices = collections.Counter()
    total_ms = 0.0
    correct = 0

    for row in rows:
        key = row["predicate"]
        standard_ms = float(row["standard_ms"])
        cache_ms = float(row["cache_ms"])
        hist = history[key]
        if len(hist["standard"]) < min_samples:
            choice = "standard"
        else:
            standard_mean = sum(hist["standard"]) / len(hist["standard"])
            cache_mean = sum(hist["cache"]) / len(hist["cache"])
            choice = "cache" if cache_mean < standard_mean else "standard"

        total_ms += cache_ms if choice == "cache" else standard_ms
        choices[choice] += 1
        if (choice == "cache") == (cache_ms < standard_ms):
            correct += 1

        hist["standard"].append(standard_ms)
        hist["cache"].append(cache_ms)

    return {
        "min_samples": min_samples,
        "total_ms": total_ms,
        "choices": dict(choices),
        "decision_accuracy": correct / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Design 3 route/cache opportunity frequency on real C4-derived pgvector workload.")
    parser.add_argument("--input", type=Path, default=Path("results/hybrid_vector_db/c4_query_filter_bloom_cache_bfs_mixed_q400_admit1m_match100.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("results/hybrid_vector_db/design3_c4_route_frequency_summary_20260705.json"))
    parser.add_argument("--out-predicate-csv", type=Path, default=Path("results/hybrid_vector_db/design3_c4_route_frequency_by_predicate_20260705.csv"))
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("no usable rows")

    standard_total = sum(float(row["standard_ms"]) for row in rows)
    cache_total = sum(float(row["cache_ms"]) for row in rows)
    oracle_total = sum(min(float(row["standard_ms"]), float(row["cache_ms"])) for row in rows)
    cache_wins = [row for row in rows if float(row["cache_ms"]) < float(row["standard_ms"])]

    thresholds = {}
    for threshold in [1.0, 1.05, 1.1, 1.2, 1.5, 2.0]:
        count = sum(float(row["speedup"]) >= threshold for row in rows)
        thresholds[f"speedup_ge_{str(threshold).replace('.', 'p')}"] = {"count": count, "ratio": count / len(rows)}
    for threshold in [0.95, 0.9, 0.8, 0.5]:
        count = sum(float(row["speedup"]) <= threshold for row in rows)
        thresholds[f"slowdown_le_{str(threshold).replace('.', 'p')}"] = {"count": count, "ratio": count / len(rows)}

    by_predicate_rows = []
    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["predicate"]].append(row)
    for predicate, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        standard = sum(float(row["standard_ms"]) for row in items)
        cache = sum(float(row["cache_ms"]) for row in items)
        wins = sum(float(row["cache_ms"]) < float(row["standard_ms"]) for row in items)
        by_predicate_rows.append(
            {
                "queries": len(items),
                "cache_wins": wins,
                "cache_win_ratio": wins / len(items),
                "standard_total_ms": standard,
                "cache_total_ms": cache,
                "cache_total_speedup": standard / cache if cache else 0.0,
                "standard_mean_ms": standard / len(items),
                "cache_mean_ms": cache / len(items),
                "predicate": predicate,
            }
        )

    online = [online_profile_simulation(rows, min_samples) for min_samples in [1, 2, 3, 5, 10]]
    for item in online:
        item["speedup_vs_standard"] = standard_total / float(item["total_ms"]) if item["total_ms"] else 0.0
        item["captured_oracle_gain_ratio"] = (
            (standard_total - float(item["total_ms"])) / (standard_total - oracle_total)
            if standard_total > oracle_total
            else 0.0
        )

    summary = {
        "input": str(args.input),
        "queries": len(rows),
        "distinct_predicates": len(grouped),
        "cache_win_count": len(cache_wins),
        "cache_win_ratio": len(cache_wins) / len(rows),
        "standard_total_ms": standard_total,
        "always_cache_total_ms": cache_total,
        "oracle_total_ms": oracle_total,
        "always_cache_speedup_vs_standard": standard_total / cache_total if cache_total else 0.0,
        "oracle_speedup_vs_standard": standard_total / oracle_total if oracle_total else 0.0,
        "thresholds": thresholds,
        "online_profile_cache": online,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.out_predicate_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_predicate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_predicate_rows)

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_predicate_csv}")


if __name__ == "__main__":
    main()
