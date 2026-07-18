from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
RESULTS = ROOT / "results" / "hybrid_vector_db"
sys.path.insert(0, str(SCRIPT_DIR))

import pgvector_design1_design2_design3_selectivity_benchmark as bench  # noqa: E402


MODE_LABELS = {
    "original": "Stock pgvector",
    "design1_bloom": "D1 guidance",
    "design1_bloom_bfs_layout": "D1+D2",
    "design1_bloom_bfs_layout_d3": "SQLens (D1-D3)",
}


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(0.95 * (len(values) - 1)))
    return values[idx]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["filter_name"]),
                str(row["mode"]),
                int(row["ef_search"]),
                int(row["max_scan_tuples"]),
            ),
            [],
        ).append(row)

    out: list[dict[str, object]] = []
    selectivity_by_filter = {name: target for name, target, _ in bench.ATTR_FILTERS}
    order = {name: i for i, (name, _, _) in enumerate(bench.ATTR_FILTERS)}
    mode_order = {mode: i for i, mode in enumerate(bench.MODES)}
    for (filter_name, mode, ef_search, max_scan_tuples), items in sorted(
        groups.items(), key=lambda kv: (order.get(kv[0][0], 999), mode_order.get(kv[0][1], 999), kv[0][2], kv[0][3])
    ):
        ok = [r for r in items if not r.get("error")]
        lat = [float(r["end_to_end_ms"]) for r in ok]
        recall = [float(r["recall"]) for r in ok]
        returned = [float(r["returned"]) for r in ok]
        visited = [float(r["visited_tuples"]) for r in ok]
        returned_tuples = [float(r["returned_tuples"]) for r in ok]
        guidance_checks = [float(r["guidance_checks"]) for r in ok]
        guidance_skips = [float(r["guidance_skips"]) for r in ok]
        total_ms = sum(lat)
        out.append(
            {
                "filter_name": filter_name,
                "selectivity": selectivity_by_filter.get(filter_name, ""),
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "ef_search": ef_search,
                "max_scan_tuples": max_scan_tuples,
                "samples": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": statistics.fmean(recall) if recall else 0.0,
                "recall_p50": statistics.median(recall) if recall else 0.0,
                "recall_min": min(recall) if recall else 0.0,
                "latency_mean_ms": statistics.fmean(lat) if lat else 0.0,
                "latency_p50_ms": statistics.median(lat) if lat else 0.0,
                "latency_p95_ms": p95(lat),
                "single_client_throughput_qps": (1000.0 * len(ok) / total_ms) if total_ms > 0 else 0.0,
                "returned_mean": statistics.fmean(returned) if returned else 0.0,
                "visited_tuples_mean": statistics.fmean(visited) if visited else 0.0,
                "returned_tuples_mean": statistics.fmean(returned_tuples) if returned_tuples else 0.0,
                "guidance_skip_rate": (
                    statistics.fmean(guidance_skips) / statistics.fmean(guidance_checks)
                    if guidance_checks and statistics.fmean(guidance_checks) > 0
                    else 0.0
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon-10M pgvector/SQLens recall-latency-throughput sweep.")
    parser.add_argument("--out-prefix", default="amazon_recall_tradeoff")
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--filters", nargs="*", default=["long_review_ge500", "helpful_ge20"])
    parser.add_argument("--modes", nargs="*", choices=bench.MODES, default=["original", "design1_bloom", "design1_bloom_bfs_layout_d3"])
    parser.add_argument("--ef-search-values", default="40,80,160,320,640,1000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument(
        "--max-scan-tuples-values",
        default="",
        help="Comma-separated scan-budget sweep. Overrides --max-scan-tuples when set.",
    )
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument("--statement-timeout-ms", type=int, default=180000)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-cache-per-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--insertion-table", default=bench.INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=bench.INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=bench.BFS_TABLE)
    parser.add_argument("--bfs-index", default=bench.BFS_INDEX)
    args = parser.parse_args()

    truth, query_by_no = bench.load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    selected = set(args.filters or [])
    filters = [(name, target, pred) for name, target, pred in bench.ATTR_FILTERS if not selected or name in selected]

    rows: list[dict[str, object]] = []
    max_scan_values = parse_ints(args.max_scan_tuples_values) if args.max_scan_tuples_values else [args.max_scan_tuples]
    for ef_search in parse_ints(args.ef_search_values):
        for max_scan_tuples in max_scan_values:
            run_args = SimpleNamespace(**vars(args))
            run_args.ef_search = ef_search
            run_args.max_scan_tuples = max_scan_tuples
            for mode in args.modes:
                print(
                    f"running ef={ef_search} max_scan={max_scan_tuples} mode={mode} "
                    f"filters={','.join(name for name, _, _ in filters)} q={len(query_nos)} r={args.repeats}",
                    flush=True,
                )
                part = bench.run_mode(run_args, mode, filters, query_nos, query_by_no, truth)
                for row in part:
                    row["ef_search"] = ef_search
                    row["max_scan_tuples"] = max_scan_tuples
                    row["scan_mem_multiplier"] = args.scan_mem_multiplier
                    row["iterative_scan"] = args.iterative_scan
                rows.extend(part)

    suffix = f"q{args.queries}r{args.repeats}_{args.out_prefix}"
    raw_out = RESULTS / f"{suffix}.csv"
    summary_out = RESULTS / f"{suffix}_summary.csv"
    write_csv(raw_out, rows)
    write_csv(summary_out, summarize(rows))
    print(f"wrote {raw_out}", flush=True)
    print(f"wrote {summary_out}", flush=True)


if __name__ == "__main__":
    main()
