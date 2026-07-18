from __future__ import annotations

import argparse
import csv
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psycopg

from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
BENCH = "experiments/hybrid_vector_db/scripts/laion25m_sqlens_variants_benchmark.py"
SELECTED = RESULTS / "laion25m_label_or_global_selected_q100_20260716.csv"
TRUTH = RESULTS / "laion25m_label_or_global_truth_q100_20260716.csv"
STOCK_INDEX = "laion25m_pgvector_embedding_hnsw"
BFS_TABLE = "laion25m_pgvector_bfs"
BFS_INDEX = "laion25m_pgvector_bfs_embedding_hnsw"
METHOD_LABEL = {
    "stock": "Stock",
    "d1": "D1",
    "d1_d2": "D1+D2",
    "d1_d2_d3": "D1+D2+D3",
}


def fmt_target(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value).rstrip("0").rstrip(".")


def read_targets(path: Path) -> list[float]:
    seen: set[float] = set()
    targets: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = float(row["target_band_pct"])
            if target not in seen:
                seen.add(target)
                targets.append(target)
    return sorted(targets, reverse=True)


def read_filter_meta(path: Path) -> dict[float, dict[str, float]]:
    meta: dict[float, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = float(row["target_band_pct"])
            if target in meta:
                continue
            labels = [part for part in str(row.get("labels", "")).split() if part]
            atom_count = len(labels)
            if atom_count == 0 and str(row.get("predicate", "")).strip():
                atom_count = 1
            meta[target] = {
                "actual_pct": float(row.get("actual_pct", 0) or 0),
                "atom_count": float(atom_count),
            }
    return meta


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_index_visibility(args: argparse.Namespace, stock_valid: bool, bfs_valid: bool) -> None:
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SET allow_system_table_mods = on")
        cur.execute("UPDATE pg_index SET indisvalid = %s WHERE indexrelid = %s::regclass", (stock_valid, STOCK_INDEX))
        cur.execute("UPDATE pg_index SET indisvalid = %s WHERE indexrelid = %s::regclass", (bfs_valid, args.bfs_index))
        cur.execute(
            """
            SELECT c.relname, i.indisvalid, i.indisready
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname IN (%s, %s)
            ORDER BY c.relname
            """,
            (STOCK_INDEX, args.bfs_index),
        )
        print("index visibility:", cur.fetchall(), flush=True)


def run_command(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(cmd), flush=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + shlex.join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        return proc.wait()


def make_cmd(args: argparse.Namespace, method: str, target: float, out: Path) -> list[str]:
    cmd = [
        sys.executable,
        BENCH,
        "--selected-queries-in",
        str(args.selected_queries_in),
        "--truth",
        str(args.truth),
        "--bfs-table",
        args.bfs_table,
        "--bfs-index",
        args.bfs_index,
        "--target-bands",
        fmt_target(target),
        "--methods",
        method,
        "--repeats",
        str(args.repeats),
        "--ef-search",
        str(args.ef_search),
        "--iterative-scan",
        args.iterative_scan,
        "--max-scan-tuples",
        str(args.max_scan_tuples),
        "--guided-collect-target",
        str(args.guided_collect_target),
        "--guidance-kind",
        args.guidance_kind,
        "--guidance-selectivity-max-pct",
        str(args.guidance_selectivity_max_pct),
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--d3-guidance-max-atoms",
        str(args.d3_guidance_max_atoms),
        "--d2-page-access",
        args.d2_page_access,
        "--d2-index-page-access",
        args.d2_index_page_access,
        "--d2-page-window",
        str(args.d2_page_window),
        "--d2-page-prefetch-min-items",
        str(args.d2_page_prefetch_min_items),
        "--d2-page-disable-after-no-merge",
        str(args.d2_page_disable_after_no_merge),
        "--scan-mem-multiplier",
        str(args.scan_mem_multiplier),
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--d3-cache-mb",
        str(args.d3_cache_mb),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--progress-queries",
        str(args.progress_queries),
        "--out",
        str(out),
    ]
    if args.limit_per_group:
        cmd += ["--limit-per-group", str(args.limit_per_group)]
    if not args.prewarm_d3:
        cmd += ["--no-prewarm-d3"]
    if not args.force_hnsw:
        cmd += ["--no-force-hnsw"]
    if args.warmup_all_queries:
        cmd += ["--warmup-all-queries"]
    return cmd


def run_phase(args: argparse.Namespace, phase: str, methods: list[str], targets: list[float], shard_dir: Path) -> None:
    print(f"phase {phase}: methods={methods}", flush=True)
    if args.toggle_index_visibility:
        if phase == "stock_index":
            set_index_visibility(args, stock_valid=True, bfs_valid=False)
        elif phase == "bfs_index":
            set_index_visibility(args, stock_valid=False, bfs_valid=True)
        else:
            raise ValueError(phase)

    tasks: list[tuple[list[str], Path]] = []
    for method in methods:
        for target in targets:
            label = fmt_target(target)
            out = shard_dir / f"laion25m_label_or_global_{label}pct_{method}_q100_r{args.repeats}_ef{args.ef_search}_t{args.guided_collect_target}_{args.guidance_kind}.csv"
            log = shard_dir / f"laion25m_label_or_global_{label}pct_{method}.log"
            if args.skip_existing and out.exists() and out.with_name(out.stem + "_summary.csv").exists():
                print(f"skip existing {out}", flush=True)
                continue
            tasks.append((make_cmd(args, method, target, out), log))

    running: list[tuple[subprocess.Popen, Path, Path]] = []
    failures: list[tuple[Path, int]] = []
    task_iter = iter(tasks)

    def start_one(cmd: list[str], log: Path) -> None:
        print(shlex.join(cmd), flush=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        f = log.open("w", encoding="utf-8")
        f.write("$ " + shlex.join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        running.append((proc, log, f))

    while True:
        while len(running) < args.jobs:
            try:
                cmd, log = next(task_iter)
            except StopIteration:
                break
            start_one(cmd, log)
        if not running:
            break
        time.sleep(args.poll_seconds)
        still: list[tuple[subprocess.Popen, Path, Path]] = []
        for proc, log, f in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, log, f))
            else:
                f.close()
                print(f"done rc={rc} log={log}", flush=True)
                if rc != 0:
                    failures.append((log, rc))
        running = still

    if failures:
        raise SystemExit("failed shards: " + ", ".join(f"{log}:{rc}" for log, rc in failures))


def read_summary_rows(shard_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(shard_dir.glob("*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def combine(args: argparse.Namespace, shard_dir: Path, out_prefix: Path) -> None:
    rows = read_summary_rows(shard_dir)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary_combined.csv"), rows)
    filter_meta = read_filter_meta(args.selected_queries_in)

    by_key: dict[tuple[float, str], dict[str, str]] = {}
    for row in rows:
        by_key[(float(row["target_band_pct"]), row["method"])] = row

    wide: list[dict[str, object]] = []
    profile: list[dict[str, object]] = []
    for target in read_targets(args.selected_queries_in):
        present = {method: by_key.get((target, method)) for method in METHOD_LABEL}
        if not present.get("stock"):
            continue
        stock = present["stock"]
        assert stock is not None
        sqlens = [row for method, row in present.items() if method != "stock" and row is not None]
        best = min(sqlens, key=lambda r: float(r["end_to_end_ms_mean"])) if sqlens else None
        stock_ms = float(stock["end_to_end_ms_mean"])
        best_ms = float(best["end_to_end_ms_mean"]) if best else 0.0
        meta = filter_meta.get(target, {})
        atom_count = int(meta.get("atom_count", 0))
        route_to_stock = float(stock["actual_pct_mean"]) >= float(args.adaptive_selectivity_threshold) or atom_count > int(args.adaptive_max_atoms)
        adaptive = stock if route_to_stock else (present.get("d1_d2_d3") or best or stock)
        adaptive_ms = float(adaptive["end_to_end_ms_mean"]) if adaptive else 0.0
        adaptive_method = "Stock" if adaptive is stock else METHOD_LABEL.get(adaptive["method"], adaptive["method"])
        wide.append(
            {
                "Sel.": f"{fmt_target(target)}%",
                "Actual": f"{float(stock['actual_pct_mean']):.3f}%",
                "Filter": stock.get("filter_name", ""),
                "Atoms": atom_count,
                "Stock": f"{stock_ms:.2f}",
                "D1": f"{float(present['d1']['end_to_end_ms_mean']):.2f}" if present.get("d1") else "",
                "D1+D2": f"{float(present['d1_d2']['end_to_end_ms_mean']):.2f}" if present.get("d1_d2") else "",
                "D1+D2+D3": f"{float(present['d1_d2_d3']['end_to_end_ms_mean']):.2f}" if present.get("d1_d2_d3") else "",
                "Best": f"{best_ms:.2f}" if best else "",
                "Best method": METHOD_LABEL.get(best["method"], best["method"]) if best else "",
                "Speedup": f"{stock_ms / best_ms:.2f}x" if best_ms else "",
                "Recall": f"{float(best['recall_mean']):.3f}" if best else "",
                "Adaptive": f"{adaptive_ms:.2f}" if adaptive_ms else "",
                "Adaptive method": adaptive_method,
                "Adaptive speedup": f"{stock_ms / adaptive_ms:.2f}x" if adaptive_ms else "",
                "Adaptive recall": f"{float(adaptive['recall_mean']):.3f}" if adaptive else "",
            }
        )
        for method, row in present.items():
            if row is None:
                continue
            profile.append(
                {
                    "Sel.": f"{fmt_target(target)}%",
                    "method": METHOD_LABEL[method],
                    "actual_pct": float(row["actual_pct_mean"]),
                    "e2e_ms": float(row["end_to_end_ms_mean"]),
                    "recall": float(row["recall_mean"]),
                    "activation_ms": float(row["activation_ms_mean"]),
                    "composed_exact_active_rate": float(row.get("composed_exact_active_rate", 0) or 0),
                    "composed_exact_hit_rate": float(row.get("composed_exact_hit_rate", 0) or 0),
                    "composed_exact_build_ms": float(row.get("composed_exact_build_ms_mean", 0) or 0),
                    "composed_exact_rows": float(row.get("composed_exact_rows_mean", 0) or 0),
                    "cache_composed_exact_bytes": float(row.get("cache_composed_exact_bytes_mean", 0) or 0),
                    "vector_search_ms": float(row["vector_search_ms_mean"]),
                    "visited_tuples": float(row["visited_tuples_mean"]),
                    "returned_tuples": float(row["returned_tuples_mean"]),
                    "guidance_skip_rate": float(row["guidance_skip_rate_mean"]),
                    "errors": int(float(row["errors"])),
                }
            )

    write_csv(out_prefix.with_name(out_prefix.name + "_wide.csv"), wide)
    write_csv(out_prefix.with_name(out_prefix.name + "_profile_wide.csv"), profile)

    print(
        "\nSel.\tActual\tFilter\tAtoms\tStock\tD1\tD1+D2\tD1+D2+D3\tBest\tBest method\tSpeedup\tRecall\tAdaptive\tAdaptive method\tAdaptive speedup\tAdaptive recall",
        flush=True,
    )
    for row in wide:
        print(
            "\t".join(
                str(row[key])
                for key in [
                    "Sel.",
                    "Actual",
                    "Filter",
                    "Atoms",
                    "Stock",
                    "D1",
                    "D1+D2",
                    "D1+D2+D3",
                    "Best",
                    "Best method",
                    "Speedup",
                    "Recall",
                    "Adaptive",
                    "Adaptive method",
                    "Adaptive speedup",
                    "Adaptive recall",
                ]
            ),
            flush=True,
        )

    if profile:
        print("\nProfile means by method:", flush=True)
        by_method: dict[str, list[dict[str, object]]] = {}
        for row in profile:
            by_method.setdefault(str(row["method"]), []).append(row)
        for method, items in by_method.items():
            print(
                method,
                "e2e=",
                f"{statistics.fmean(float(x['e2e_ms']) for x in items):.2f}",
                "visited=",
                f"{statistics.fmean(float(x['visited_tuples']) for x in items):.0f}",
                "skip=",
                f"{statistics.fmean(float(x['guidance_skip_rate']) for x in items):.3f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAION-25M SQLens same-setting benchmark.")
    parser.add_argument("--selected-queries-in", type=Path, default=SELECTED)
    parser.add_argument("--truth", type=Path, default=TRUTH)
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument("--shard-dir", type=Path, default=RESULTS / "laion25m_sqlens_guided_t100_exact_shards_20260716")
    parser.add_argument("--out-prefix", type=Path, default=RESULTS / "laion25m_label_or_global_sqlens_guided_t100_exact_q100_r5_ef1000_20260716")
    parser.add_argument("--targets", default="")
    parser.add_argument("--phase", choices=["all", "stock_index", "bfs_index", "combine"], default="all")
    parser.add_argument("--stock-methods", nargs="*", default=["stock", "d1"])
    parser.add_argument("--bfs-methods", nargs="*", default=["d1_d2", "d1_d2_d3"])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order")
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--guided-collect-target", type=int, default=100)
    parser.add_argument("--guidance-kind", default="exact")
    parser.add_argument("--guidance-selectivity-max-pct", type=float, default=10.0)
    parser.add_argument("--guidance-max-atoms", type=int, default=1)
    parser.add_argument("--d3-guidance-max-atoms", type=int, default=64)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d1-cache-mb", type=int, default=4096)
    parser.add_argument("--d3-cache-mb", type=int, default=4096)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=50)
    parser.add_argument("--prewarm-d3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--toggle-index-visibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-selectivity-threshold", type=float, default=5.0)
    parser.add_argument("--adaptive-max-atoms", type=int, default=1)
    args = parser.parse_args()

    targets = [float(x) for x in args.targets.split(",") if x] if args.targets else read_targets(args.selected_queries_in)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase in {"all", "stock_index"}:
            run_phase(args, "stock_index", args.stock_methods, targets, args.shard_dir)
        if args.phase in {"all", "bfs_index"}:
            run_phase(args, "bfs_index", args.bfs_methods, targets, args.shard_dir)
        if args.phase in {"all", "combine"}:
            combine(args, args.shard_dir, args.out_prefix)
    finally:
        if args.toggle_index_visibility and args.phase != "combine":
            set_index_visibility(args, stock_valid=True, bfs_valid=False)


if __name__ == "__main__":
    main()
