from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
RESULTS = ROOT / "results" / "hybrid_vector_db"
sys.path.insert(0, str(SCRIPT_DIR))

import psycopg  # noqa: E402

import yfcc_pgvector_filtered_benchmark as bench  # noqa: E402
from common_pg import pg_config_from_env  # noqa: E402


METHOD_LABELS = {
    "stock": "Stock pgvector",
    "bloom": "SQLens guidance",
    "page": "Page guidance",
}


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_loaded_queries(rows: list[dict[str, Any]], bands: list[float], per_band: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for band in bands:
        matching = [row for row in rows if float(row["target_band_pct"]) == float(band)]
        selected.extend(matching[:per_band])
    return selected


def summarize(rows: list[dict[str, Any]], warm_repeats_only: bool) -> list[dict[str, Any]]:
    filtered = [row for row in rows if not row.get("error")]
    if warm_repeats_only:
        filtered = [row for row in filtered if int(row["repeat"]) > 0]
    groups: dict[tuple[float, str, int, int], list[dict[str, Any]]] = {}
    for row in filtered:
        groups.setdefault(
            (
                float(row["target_band_pct"]),
                str(row["method"]),
                int(row["ef_search"]),
                int(row["max_scan_tuples"]),
            ),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (band, method, ef_search, max_scan), items in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
    ):
        lat = [float(row["latency_ms"]) + float(row.get("activation_ms", 0) or 0) for row in items]
        recall = [float(row["recall"]) for row in items]
        total_ms = sum(lat)
        checks_mean = statistics.fmean(float(row.get("guidance_checks", 0) or 0) for row in items)
        skips_mean = statistics.fmean(float(row.get("guidance_skips", 0) or 0) for row in items)
        out.append(
            {
                "target_band_pct": band,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "ef_search": ef_search,
                "max_scan_tuples": max_scan,
                "samples": len(items),
                "recall_mean": statistics.fmean(recall),
                "recall_p50": statistics.median(recall),
                "recall_min": min(recall),
                "latency_mean_ms": statistics.fmean(lat),
                "latency_p50_ms": statistics.median(lat),
                "latency_p95_ms": p95(lat),
                "single_client_throughput_qps": 1000.0 * len(items) / total_ms if total_ms > 0 else 0.0,
                "filter_pct_mean": statistics.fmean(float(row["filter_pct"]) for row in items),
                "filter_rows_mean": statistics.fmean(float(row["filter_rows"]) for row in items),
                "visited_tuples_mean": statistics.fmean(float(row.get("visited_tuples", 0) or 0) for row in items),
                "returned_tuples_mean": statistics.fmean(float(row.get("returned_tuples", 0) or 0) for row in items),
                "guidance_skip_rate": skips_mean / checks_mean if checks_mean else 0.0,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="YFCC-10M recall/latency/throughput scan-budget sweep.")
    parser.add_argument("--out-prefix", default="yfcc_recall_tradeoff")
    parser.add_argument("--table", default=bench.TABLE)
    parser.add_argument("--query-table", default=bench.QUERY_TABLE)
    parser.add_argument("--index", default=bench.INDEX)
    parser.add_argument("--selected-queries-in", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_selected_queries_20260713.csv"))
    parser.add_argument("--target-bands", type=float, nargs="+", default=[5.0, 0.5])
    parser.add_argument("--queries-per-band", type=int, default=20)
    parser.add_argument("--methods", nargs="+", choices=["stock", "bloom", "page"], default=["stock", "bloom"])
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument(
        "--ef-search-values",
        default="",
        help="Comma-separated ef_search sweep. Overrides --ef-search when set.",
    )
    parser.add_argument("--max-scan-tuples-values", default="2000,5000,10000,20000,50000,200000")
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--statement-timeout-ms", type=int, default=180000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--progress-queries", type=int, default=10)
    args = parser.parse_args()

    cfg = pg_config_from_env()
    rows: list[dict[str, Any]] = []
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        bench.vector_functions(cur)
        loaded = bench.load_selected_queries(cur, args)
        selected = select_loaded_queries(loaded, args.target_bands, args.queries_per_band)
        if not selected:
            raise SystemExit("no selected YFCC queries matched requested target bands")
        ef_values = parse_ints(args.ef_search_values) if args.ef_search_values else [args.ef_search]
        for ef_search in ef_values:
            for max_scan in parse_ints(args.max_scan_tuples_values):
                run_args = SimpleNamespace(**vars(args))
                run_args.ef_search = ef_search
                run_args.max_scan_tuples = max_scan
                for method in args.methods:
                    print(
                        f"running YFCC method={method} ef={ef_search} max_scan={max_scan} "
                        f"bands={','.join(str(x) for x in args.target_bands)} q={len(selected)} r={args.repeats}",
                        flush=True,
                    )
                    for qno, query in enumerate(selected, start=1):
                        bench.configure(cur, run_args, True)
                        activation_profile, activation_ms = bench.activate_guidance(cur, run_args, method, query["tags"])
                        for repeat in range(args.repeats):
                            ids, latency_ms, profile, error = bench.run_hnsw_query(cur, run_args, query)
                            checks = float(profile.get("guidance_checks", 0) or 0)
                            skips = float(profile.get("guidance_skips", 0) or 0)
                            rows.append(
                                {
                                    "method": method,
                                    "target_band_pct": float(query["target_band_pct"]),
                                    "filter_pct": float(query["filter_pct"]),
                                    "filter_rows": int(query["filter_rows"]),
                                    "qid": int(query["qid"]),
                                    "tags": " ".join(str(x) for x in query["tags"]),
                                    "repeat": repeat,
                                    "recall": bench.recall_at_k(ids, query["gt"], args.k) if not error else 0.0,
                                    "latency_ms": latency_ms,
                                    "activation_ms": activation_ms,
                                    "vector_search_ms": float(profile.get("vector_search_ms", 0) or 0),
                                    "visited_tuples": float(profile.get("visited_tuples", 0) or 0),
                                    "returned_tuples": float(profile.get("returned_tuples", 0) or 0),
                                    "guidance_checks": checks,
                                    "guidance_skips": skips,
                                    "guidance_skip_rate": skips / checks if checks else 0.0,
                                    "returned": len(ids),
                                    "ids": ",".join(str(x) for x in ids),
                                    "error": error,
                                    "ef_search": ef_search,
                                    "max_scan_tuples": max_scan,
                                }
                            )
                        if args.progress_queries and qno % args.progress_queries == 0:
                            latest = [
                                row
                                for row in rows
                                if row["method"] == method
                                and int(row["ef_search"]) == ef_search
                                and int(row["max_scan_tuples"]) == max_scan
                                and not row["error"]
                            ]
                            print(
                                f"progress method={method} ef={ef_search} max_scan={max_scan} q={qno}/{len(selected)} "
                                f"lat={statistics.fmean(float(row['latency_ms']) for row in latest):.2f} "
                                f"recall={statistics.fmean(float(row['recall']) for row in latest):.3f}",
                                flush=True,
                            )
                    cur.execute("SELECT vector_hnsw_guidance_reset()")

    suffix = f"q{args.queries_per_band}x{len(args.target_bands)}r{args.repeats}_{args.out_prefix}"
    raw_out = RESULTS / f"{suffix}.csv"
    summary_out = RESULTS / f"{suffix}_summary.csv"
    warm_summary_out = RESULTS / f"{suffix}_warm_summary.csv"
    write_csv(raw_out, rows)
    write_csv(summary_out, summarize(rows, warm_repeats_only=False))
    write_csv(warm_summary_out, summarize(rows, warm_repeats_only=True))
    print(f"wrote {raw_out}", flush=True)
    print(f"wrote {summary_out}", flush=True)
    print(f"wrote {warm_summary_out}", flush=True)


if __name__ == "__main__":
    main()
