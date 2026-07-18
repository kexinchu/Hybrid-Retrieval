from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
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
    "bloom": "SQLens-D1",
    "page": "Page guidance",
    "exact": "SQL-first exact",
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


def load_queries(cur: psycopg.Cursor, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.selected_queries_in and args.selected_queries_in.exists():
        rows: list[dict[str, Any]] = []
        with args.selected_queries_in.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "request_no": int(row["request_no"]),
                        "qid": int(row["qid"]),
                        "tags": bench.parse_int_array(row["tags"]),
                        "tag_count": int(row["tag_count"]),
                        "gt": bench.parse_int_array(row["gt"]),
                    }
                )
                if len(rows) >= args.requests:
                    break
        return rows

    cur.execute(
        f"""
        SELECT qid, tags, tag_count, gt
        FROM {args.query_table}
        WHERE tag_count > 0
        ORDER BY md5(qid::text)
        OFFSET %s
        LIMIT %s
        """,
        (int(args.query_offset), int(args.requests)),
    )
    rows = []
    for request_no, (qid, tags_value, tag_count, gt_value) in enumerate(cur.fetchall()):
        rows.append(
            {
                "request_no": request_no,
                "qid": int(qid),
                "tags": bench.parse_int_array(tags_value),
                "tag_count": int(tag_count),
                "gt": bench.parse_int_array(gt_value),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["method"]), int(row["ef_search"]), int(row["max_scan_tuples"])),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (method, ef_search, max_scan), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ok = [row for row in items if not row.get("error")]
        lat = [float(row["total_latency_ms"]) for row in ok]
        search_lat = [float(row["latency_ms"]) for row in ok]
        activation = [float(row.get("activation_ms", 0) or 0) for row in ok]
        recall = [float(row["recall"]) for row in ok]
        returned = [float(row["returned"]) for row in ok]
        visited = [float(row.get("visited_tuples", 0) or 0) for row in ok]
        returned_tuples = [float(row.get("returned_tuples", 0) or 0) for row in ok]
        checks = [float(row.get("guidance_checks", 0) or 0) for row in ok]
        skips = [float(row.get("guidance_skips", 0) or 0) for row in ok]
        total_ms = sum(lat)
        out.append(
            {
                "dataset": "YFCC-10M",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "ef_search": ef_search,
                "max_scan_tuples": max_scan,
                "requests": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": statistics.fmean(recall) if recall else 0.0,
                "recall_p50": statistics.median(recall) if recall else 0.0,
                "recall_min": min(recall) if recall else 0.0,
                "latency_mean_ms": statistics.fmean(lat) if lat else 0.0,
                "latency_p50_ms": statistics.median(lat) if lat else 0.0,
                "latency_p95_ms": p95(lat),
                "search_latency_mean_ms": statistics.fmean(search_lat) if search_lat else 0.0,
                "activation_mean_ms": statistics.fmean(activation) if activation else 0.0,
                "single_client_throughput_qps": 1000.0 * len(ok) / total_ms if total_ms > 0 else 0.0,
                "returned_mean": statistics.fmean(returned) if returned else 0.0,
                "visited_tuples_mean": statistics.fmean(visited) if visited else 0.0,
                "returned_tuples_mean": statistics.fmean(returned_tuples) if returned_tuples else 0.0,
                "guidance_skip_rate": (
                    statistics.fmean(skips) / statistics.fmean(checks)
                    if checks and statistics.fmean(checks) > 0
                    else 0.0
                ),
            }
        )
    return out


def write_selected(path: Path, queries: list[dict[str, Any]]) -> None:
    serializable = []
    for row in queries:
        serializable.append(
            {
                "request_no": row["request_no"],
                "qid": row["qid"],
                "tags": " ".join(str(x) for x in row["tags"]),
                "tag_count": row["tag_count"],
                "gt": " ".join(str(x) for x in row["gt"]),
            }
        )
    write_csv(path, serializable)


def run_hnsw_query_light(
    cur: psycopg.Cursor, args: argparse.Namespace, query: dict[str, Any]
) -> tuple[list[int], float, dict[str, Any], str]:
    def execute() -> list[int]:
        cur.execute(
            f"""
            SELECT id
            FROM {args.table}
            WHERE tags @> %s::int[]
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (query["tags"], int(query["qid"])),
        )
        return [int(row[0]) for row in cur.fetchall()]

    start = time.perf_counter()
    try:
        ids = execute()
        return ids, (time.perf_counter() - start) * 1000.0, {}, ""
    except psycopg.errors.QueryCanceled as exc:
        cur.connection.rollback()
        bench.configure(cur, args, True)
        return [], float(args.statement_timeout_ms), {}, exc.__class__.__name__


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YFCC-10M full-workload recall/latency/throughput sweep without selectivity grouping."
    )
    parser.add_argument("--out-prefix", default="yfcc_full_workload_recall_sweep")
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--table", default=bench.TABLE)
    parser.add_argument("--query-table", default=bench.QUERY_TABLE)
    parser.add_argument("--index", default=bench.INDEX)
    parser.add_argument("--selected-queries-in", type=Path)
    parser.add_argument("--selected-queries-out", type=Path)
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=["stock", "bloom", "page", "exact"], default=["stock", "bloom"])
    parser.add_argument("--ef-search-values", default="4,8,16,32,64,128,256,512,1000,2000")
    parser.add_argument("--max-scan-tuples-values", default="200000")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--statement-timeout-ms", type=int, default=180000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=25)
    parser.add_argument("--progress-requests", type=int, default=500)
    parser.add_argument("--collect-profile", action="store_true")
    parser.add_argument("--skip-function-ddl", action="store_true")
    args = parser.parse_args()

    cfg = pg_config_from_env()
    suffix = f"q{args.requests}_{args.out_prefix}"
    raw_out = args.raw_out or RESULTS / f"{suffix}.csv"
    summary_out = args.summary_out or RESULTS / f"{suffix}_summary.csv"
    raw_fields = [
        "dataset",
        "method",
        "request_no",
        "qid",
        "tag_count",
        "ef_search",
        "max_scan_tuples",
        "recall",
        "latency_ms",
        "activation_ms",
        "total_latency_ms",
        "vector_search_ms",
        "visited_tuples",
        "returned_tuples",
        "guidance_checks",
        "guidance_skips",
        "guidance_skip_rate",
        "returned",
        "ids",
        "error",
    ]
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, int, int, int]] = set()
    if args.resume and raw_out.exists():
        with raw_out.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add((str(row["method"]), int(row["ef_search"]), int(row["max_scan_tuples"]), int(row["request_no"])))

    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        if not args.skip_function_ddl:
            bench.vector_functions(cur)
        selected = load_queries(cur, args)
        if not selected:
            raise SystemExit("no YFCC requests selected")
        if len(selected) < args.requests:
            raise SystemExit(f"selected only {len(selected)} requests, expected {args.requests}")
        if args.selected_queries_out:
            write_selected(args.selected_queries_out, selected)
        if args.num_workers < 1:
            raise SystemExit("--num-workers must be >= 1")
        if not 0 <= args.worker_id < args.num_workers:
            raise SystemExit("--worker-id must be in [0, --num-workers)")
        if args.num_workers > 1:
            selected = [query for query in selected if int(query["request_no"]) % args.num_workers == args.worker_id]
            if not selected:
                raise SystemExit("worker shard has no selected YFCC requests")

        config_pairs = [
            (ef_search, max_scan)
            for ef_search in parse_ints(args.ef_search_values)
            for max_scan in parse_ints(args.max_scan_tuples_values)
        ]
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume and raw_out.exists() else "w"
        with raw_out.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=raw_fields)
            if mode == "w":
                writer.writeheader()
            for method in args.methods:
                method_pairs = [(0, 0)] if method == "exact" else config_pairs
                for ef_search, max_scan in method_pairs:
                    run_args = SimpleNamespace(**vars(args))
                    run_args.ef_search = max(1, ef_search)
                    run_args.max_scan_tuples = max(1, max_scan)
                    expected = {(method, ef_search, max_scan, int(query["request_no"])) for query in selected}
                    remaining = len(expected - done)
                    print(
                        f"running YFCC-full method={method} ef={ef_search} max_scan={max_scan} "
                        f"requests={len(selected)} worker={args.worker_id}/{args.num_workers} remaining={remaining}",
                        flush=True,
                    )
                    if remaining == 0:
                        continue
                    bench.configure(cur, run_args, method != "exact")
                    for query in selected[: max(0, args.warmup_requests)]:
                        if method == "exact":
                            bench.run_sql_first(cur, run_args, query)
                        elif method == "stock":
                            cur.execute("SELECT vector_hnsw_guidance_reset()")
                        else:
                            bench.activate_guidance(cur, run_args, method, query["tags"])
                        if method != "exact":
                            if args.collect_profile:
                                bench.run_hnsw_query(cur, run_args, query)
                            else:
                                run_hnsw_query_light(cur, run_args, query)
                    cur.execute("SELECT vector_hnsw_guidance_reset()")
                    bench.configure(cur, run_args, method != "exact")

                    for pos, query in enumerate(selected, start=1):
                        key = (method, ef_search, max_scan, int(query["request_no"]))
                        if key in done:
                            continue
                        if method == "exact":
                            activation_ms = 0.0
                            ids, latency_ms, error = bench.run_sql_first(cur, run_args, query)
                            profile = {}
                        elif method == "stock":
                            activation_ms = 0.0
                            if args.collect_profile:
                                ids, latency_ms, profile, error = bench.run_hnsw_query(cur, run_args, query)
                            else:
                                ids, latency_ms, profile, error = run_hnsw_query_light(cur, run_args, query)
                        else:
                            _, activation_ms = bench.activate_guidance(cur, run_args, method, query["tags"])
                            if args.collect_profile:
                                ids, latency_ms, profile, error = bench.run_hnsw_query(cur, run_args, query)
                            else:
                                ids, latency_ms, profile, error = run_hnsw_query_light(cur, run_args, query)
                        checks = float(profile.get("guidance_checks", 0) or 0)
                        skips = float(profile.get("guidance_skips", 0) or 0)
                        row = {
                            "dataset": "YFCC-10M",
                            "method": method,
                            "request_no": int(query["request_no"]),
                            "qid": int(query["qid"]),
                            "tag_count": int(query["tag_count"]),
                            "ef_search": ef_search,
                            "max_scan_tuples": max_scan,
                            "recall": bench.recall_at_k(ids, query["gt"], args.k) if not error else 0.0,
                            "latency_ms": latency_ms,
                            "activation_ms": activation_ms,
                            "total_latency_ms": latency_ms + activation_ms,
                            "vector_search_ms": float(profile.get("vector_search_ms", 0) or 0),
                            "visited_tuples": float(profile.get("visited_tuples", 0) or 0),
                            "returned_tuples": float(profile.get("returned_tuples", 0) or 0),
                            "guidance_checks": checks,
                            "guidance_skips": skips,
                            "guidance_skip_rate": skips / checks if checks else 0.0,
                            "returned": len(ids),
                            "ids": ",".join(str(x) for x in ids),
                            "error": error,
                        }
                        rows.append(row)
                        done.add(key)
                        writer.writerow(row)
                        f.flush()
                        if args.progress_requests and pos % args.progress_requests == 0:
                            latest = [
                                row
                                for row in rows
                                if row["method"] == method
                                and int(row["ef_search"]) == ef_search
                                and int(row["max_scan_tuples"]) == max_scan
                                and not row["error"]
                            ]
                            print(
                                f"progress method={method} ef={ef_search} max_scan={max_scan} "
                                f"{pos}/{len(selected)} "
                                f"lat={statistics.fmean(float(row['total_latency_ms']) for row in latest):.2f} "
                                f"recall={statistics.fmean(float(row['recall']) for row in latest):.3f}",
                                flush=True,
                            )
                    cur.execute("SELECT vector_hnsw_guidance_reset()")

    write_csv(summary_out, summarize(rows))
    print(f"wrote {raw_out}", flush=True)
    print(f"wrote {summary_out}", flush=True)


if __name__ == "__main__":
    main()
