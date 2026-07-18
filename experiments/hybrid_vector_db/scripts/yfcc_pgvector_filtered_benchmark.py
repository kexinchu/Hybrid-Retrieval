from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env


TABLE = "yfcc10m_pgvector"
QUERY_TABLE = "yfcc10m_queries"
INDEX = f"{TABLE}_embedding_hnsw"
TARGET_BANDS = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def parse_int_array(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value)
    if text in {"{}", "[]", ""}:
        return []
    text = text.strip("{}[]")
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(x.strip()) for x in text.split() if x.strip()]


def vector_functions(cur: psycopg.Cursor) -> None:
    function_sql = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in function_sql:
        try:
            cur.execute(sql)
        except Exception as exc:  # noqa: BLE001 - parallel runners may race on pg_proc updates
            if "tuple concurrently updated" not in str(exc):
                raise
            cur.connection.rollback()
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def pin_backend(cur: psycopg.Cursor, cpu_list: str) -> None:
    if not cpu_list:
        return
    cur.execute("SELECT pg_backend_pid()")
    pid = int(cur.fetchone()[0])
    result = subprocess.run(["taskset", "-pc", cpu_list, str(pid)], check=False, capture_output=True, text=True)
    print(f"pinned PostgreSQL backend pid={pid} cpus={cpu_list} rc={result.returncode}", flush=True)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)


def configure(cur: psycopg.Cursor, args: argparse.Namespace, force_hnsw: bool) -> None:
    cur.execute("SET jit = off")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(args.metadata_cache_mb)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    if force_hnsw:
        cur.execute("SET enable_sort = off")
    else:
        cur.execute("SET enable_sort = on")


def fetch_profile(cur: psycopg.Cursor) -> dict[str, Any]:
    try:
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        value = cur.fetchone()[0]
        return json.loads(value) if isinstance(value, str) else dict(value)
    except Exception:
        cur.connection.rollback()
        return {}


def fetch_cache_profile(cur: psycopg.Cursor) -> dict[str, Any]:
    try:
        cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
        value = cur.fetchone()[0]
        return json.loads(value) if isinstance(value, str) else dict(value)
    except Exception:
        cur.connection.rollback()
        return {}


def reset_profile(cur: psycopg.Cursor) -> None:
    try:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    except Exception:
        cur.connection.rollback()


def table_rows(cur: psycopg.Cursor, table: str) -> int:
    cur.execute(f"SELECT count(*) FROM {table}")
    return int(cur.fetchone()[0])


def tag_operator(args: argparse.Namespace) -> str:
    if args.tag_predicate_mode == "contains_all":
        return "@>"
    if args.tag_predicate_mode == "overlap":
        return "&&"
    raise ValueError(args.tag_predicate_mode)


def select_queries(cur: psycopg.Cursor, args: argparse.Namespace) -> list[dict[str, Any]]:
    total = table_rows(cur, args.table)
    op = tag_operator(args)
    cur.execute(
        f"""
        SELECT qid, tags, gt
        FROM {args.query_table}
        WHERE tag_count >= %s
          AND (%s = 0 OR tag_count <= %s)
        ORDER BY md5(qid::text)
        LIMIT %s
        """,
        (args.min_tag_count, args.max_tag_count, args.max_tag_count, args.query_sample),
    )
    candidates = []
    for qid, tags_value, gt_value in cur.fetchall():
        tags = parse_int_array(tags_value)
        gt = parse_int_array(gt_value)
        (count,), count_ms = timed_ms(
            lambda t=tags: (
                cur.execute(f"SELECT count(*) FROM {args.table} WHERE tags {op} %s::int[]", (t,)),
                cur.fetchone(),
            )[1]
        )
        pct = 100.0 * int(count) / max(total, 1)
        candidates.append({"qid": int(qid), "tags": tags, "gt": gt, "filter_rows": int(count), "filter_pct": pct, "count_ms": count_ms})
        if len(candidates) % 100 == 0:
            print(f"counted query filters {len(candidates)}/{args.query_sample}", flush=True)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for target in args.target_bands:
        choices = sorted(
            (row for row in candidates if row["qid"] not in used and row["filter_rows"] >= args.k),
            key=lambda row, t=target: (abs(row["filter_pct"] - t), row["qid"]),
        )
        for row in choices[: args.queries_per_band]:
            out = dict(row)
            out["target_band_pct"] = target
            selected.append(out)
            used.add(int(row["qid"]))
    if not selected:
        raise SystemExit("no YFCC queries selected")
    serializable = []
    for row in selected:
        out = dict(row)
        out["tags"] = " ".join(str(x) for x in out["tags"])
        out["gt"] = " ".join(str(x) for x in out["gt"])
        serializable.append(out)
    write_csv(args.selected_queries_out, serializable)
    print(f"selected {len(selected)} queries -> {args.selected_queries_out}", flush=True)
    return selected


def load_selected_queries(cur: psycopg.Cursor, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.selected_queries_in and args.selected_queries_in.exists():
        rows = []
        targets = {float(x) for x in args.target_bands} if args.target_bands else set()
        with args.selected_queries_in.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["qid"] = int(row["qid"])
                row["tags"] = parse_int_array(row["tags"])
                row["gt"] = parse_int_array(row["gt"])
                row["filter_rows"] = int(row["filter_rows"])
                row["filter_pct"] = float(row["filter_pct"])
                row["target_band_pct"] = float(row["target_band_pct"])
                if targets and float(row["target_band_pct"]) not in targets:
                    continue
                rows.append(row)
        return rows
    return select_queries(cur, args)


def tag_predicate(tags: list[int], args: argparse.Namespace) -> str:
    return "tags " + tag_operator(args) + " ARRAY[" + ",".join(str(int(x)) for x in tags) + "]"


def guidance_atoms(tags: list[int], args: argparse.Namespace) -> list[str]:
    atoms = [f"sql:tags @> ARRAY[{int(tag)}]" for tag in tags]
    if args.tag_predicate_mode == "contains_all":
        return atoms
    out: list[str] = []
    for atom in atoms:
        if out:
            out.append("|")
        out.append(atom)
    return out


def unique_guidance_tag_sets(selected: list[dict[str, Any]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for row in selected:
        tags = [int(x) for x in row["tags"]]
        key = tuple(tags)
        if key in seen:
            continue
        seen.add(key)
        out.append(tags)
    return out


def activate_guidance(cur: psycopg.Cursor, args: argparse.Namespace, method: str, tags: list[int]) -> tuple[dict[str, Any], float]:
    if method == "stock":
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        return {}, 0.0
    atoms = guidance_atoms(tags, args)

    def run():
        cur.execute("SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)", (args.index, atoms, method))
        cur.execute("SELECT vector_hnsw_guidance_profile()")
        value = cur.fetchone()[0]
        return json.loads(value) if isinstance(value, str) else dict(value)

    return timed_ms(run)


def prewarm_guidance_cache(cur: psycopg.Cursor, args: argparse.Namespace, method: str, selected: list[dict[str, Any]]) -> None:
    if method == "stock" or not args.prewarm_guidance_cache:
        return
    tag_sets = unique_guidance_tag_sets(selected)
    print(f"prewarming {method} guidance cache for {len(tag_sets)} unique YFCC tag predicates", flush=True)
    for idx, tags in enumerate(tag_sets, start=1):
        activate_guidance(cur, args, method, tags)
        if args.progress_queries and idx % max(1, args.progress_queries) == 0:
            print(f"prewarm {method} {idx}/{len(tag_sets)}", flush=True)
    cur.execute("SELECT vector_hnsw_guidance_reset()")


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    truth_k = [x for x in truth[:k] if x >= 0]
    if not truth_k:
        return 0.0
    return len(set(ids[:k]) & set(truth_k)) / min(k, len(truth_k))


def run_hnsw_query(cur: psycopg.Cursor, args: argparse.Namespace, query: dict[str, Any]) -> tuple[list[int], float, dict[str, Any], str]:
    reset_profile(cur)
    op = tag_operator(args)

    def execute():
        cur.execute(
            f"""
            SELECT id
            FROM {args.table}
            WHERE tags {op} %s::int[]
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (query["tags"], int(query["qid"])),
        )
        return [int(row[0]) for row in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        return ids, latency_ms, fetch_profile(cur), ""
    except errors.QueryCanceled as exc:
        cur.connection.rollback()
        configure(cur, args, True)
        return [], float(args.statement_timeout_ms), {}, exc.__class__.__name__


def run_sql_first(cur: psycopg.Cursor, args: argparse.Namespace, query: dict[str, Any]) -> tuple[list[int], float, str]:
    op = tag_operator(args)

    def execute():
        cur.execute(
            f"""
            WITH valid AS MATERIALIZED (
              SELECT id, embedding
              FROM {args.table}
              WHERE tags {op} %s::int[]
            )
            SELECT id
            FROM valid
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (query["tags"], int(query["qid"])),
        )
        return [int(row[0]) for row in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        return ids, latency_ms, ""
    except errors.QueryCanceled as exc:
        cur.connection.rollback()
        configure(cur, args, False)
        return [], float(args.statement_timeout_ms), exc.__class__.__name__


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[max(0, int(0.95 * len(vals)) - 1)]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((float(row["target_band_pct"]), str(row["method"])), []).append(row)
    out = []
    for (band, method), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ok = [row for row in items if not row.get("error")]
        if not ok:
            continue

        def vals(key: str) -> list[float]:
            return [float(row.get(key, 0) or 0) for row in ok]

        out.append(
            {
                "target_band_pct": band,
                "method": method,
                "queries": len(ok),
                "filter_pct_mean": statistics.fmean(vals("filter_pct")),
                "filter_rows_mean": statistics.fmean(vals("filter_rows")),
                "recall_mean": statistics.fmean(vals("recall")),
                "latency_ms_mean": statistics.fmean(vals("latency_ms")),
                "latency_ms_p50": statistics.median(vals("latency_ms")),
                "latency_ms_p95": p95(vals("latency_ms")),
                "activation_ms_mean": statistics.fmean(vals("activation_ms")),
                "activation_build_ms_mean": statistics.fmean(vals("activation_build_ms")),
                "fragment_cache_hits_mean": statistics.fmean(vals("fragment_cache_hits")),
                "fragment_cache_misses_mean": statistics.fmean(vals("fragment_cache_misses")),
                "fragment_store_hits_mean": statistics.fmean(vals("fragment_store_hits")),
                "fragment_builds_mean": statistics.fmean(vals("fragment_builds")),
                "composed_guide_hit_rate": statistics.fmean(vals("composed_guide_hit")),
                "cache_resident_bytes_max": max(vals("cache_resident_bytes")) if ok else 0.0,
                "vector_search_ms_mean": statistics.fmean(vals("vector_search_ms")),
                "visited_tuples_mean": statistics.fmean(vals("visited_tuples")),
                "returned_tuples_mean": statistics.fmean(vals("returned_tuples")),
                "guidance_skip_rate_mean": statistics.fmean(vals("guidance_skip_rate")),
                "errors": len(items) - len(ok),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-db filtered pgvector benchmark on BigANN YFCC10M.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_pgvector_filtered_20260713.csv"))
    parser.add_argument("--selected-queries-out", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_selected_queries_20260713.csv"))
    parser.add_argument("--selected-queries-in", type=Path)
    parser.add_argument("--query-sample", type=int, default=2000)
    parser.add_argument("--min-tag-count", type=int, default=1)
    parser.add_argument("--max-tag-count", type=int, default=0, help="0 means no upper bound")
    parser.add_argument("--tag-predicate-mode", choices=["contains_all", "overlap"], default="contains_all")
    parser.add_argument("--queries-per-band", type=int, default=20)
    parser.add_argument("--target-bands", type=float, nargs="+", default=TARGET_BANDS)
    parser.add_argument("--methods", nargs="+", default=["stock", "bloom", "page", "sql_first"], choices=["stock", "bloom", "page", "exact", "sql_first"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--metadata-cache-mb", type=int, default=4096)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--prewarm-guidance-cache", action="store_true")
    parser.add_argument("--reset-metadata-cache-per-method", action="store_true")
    parser.add_argument("--backend-cpu-list", default="")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        pin_backend(cur, args.backend_cpu_list)
        vector_functions(cur)
        configure(cur, args, True)
        selected = load_selected_queries(cur, args)
        if args.select_only:
            print(f"selected {len(selected)} YFCC queries; stopping before benchmark because --select-only was set", flush=True)
            return
        with args.out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "method",
                "target_band_pct",
                "filter_pct",
                "filter_rows",
                "qid",
                "tags",
                "repeat",
                "recall",
                "latency_ms",
                "activation_ms",
                "guidance_atoms",
                "guidance_groups",
                "activation_build_ms",
                "activation_cache_rows",
                "activation_cache_memory_bytes",
                "fragment_cache_hits",
                "fragment_cache_misses",
                "fragment_store_hits",
                "fragment_builds",
                "composed_guide_hit",
                "cache_resident_bytes",
                "cache_resident_entries",
                "cache_evictions",
                "composed_guide_entries",
                "composed_guide_hits_total",
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
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for method in args.methods:
                if args.reset_metadata_cache_per_method:
                    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
                prewarm_guidance_cache(cur, args, method, selected)
                for qno, query in enumerate(selected, start=1):
                    if method == "sql_first":
                        configure(cur, args, False)
                        activation_profile: dict[str, Any] = {}
                        activation_ms = 0.0
                    else:
                        configure(cur, args, True)
                        activation_profile, activation_ms = activate_guidance(cur, args, method, query["tags"])
                    for repeat in range(args.repeats):
                        if method == "sql_first":
                            ids, latency_ms, error = run_sql_first(cur, args, query)
                            profile: dict[str, Any] = {}
                        else:
                            ids, latency_ms, profile, error = run_hnsw_query(cur, args, query)
                        cache_profile = fetch_cache_profile(cur)
                        checks = float(profile.get("guidance_checks", 0) or 0)
                        skips = float(profile.get("guidance_skips", 0) or 0)
                        row = {
                            "method": method,
                            "target_band_pct": float(query["target_band_pct"]),
                            "filter_pct": float(query["filter_pct"]),
                            "filter_rows": int(query["filter_rows"]),
                            "qid": int(query["qid"]),
                            "tags": " ".join(str(x) for x in query["tags"]),
                            "repeat": repeat,
                            "recall": recall_at_k(ids, query["gt"], args.k) if not error else 0.0,
                            "latency_ms": latency_ms,
                            "activation_ms": activation_ms,
                            "guidance_atoms": int(activation_profile.get("atoms", 0) or 0),
                            "guidance_groups": int(activation_profile.get("groups", 0) or 0),
                            "activation_build_ms": float(activation_profile.get("last_cache_build_ms", 0) or 0),
                            "activation_cache_rows": int(activation_profile.get("last_cache_rows", 0) or 0),
                            "activation_cache_memory_bytes": int(activation_profile.get("last_cache_memory_bytes", 0) or 0),
                            "fragment_cache_hits": int(activation_profile.get("fragment_cache_hits", 0) or 0),
                            "fragment_cache_misses": int(activation_profile.get("fragment_cache_misses", 0) or 0),
                            "fragment_store_hits": int(activation_profile.get("fragment_store_hits", 0) or 0),
                            "fragment_builds": int(activation_profile.get("fragment_builds", 0) or 0),
                            "composed_guide_hit": 1.0 if activation_profile.get("composed_guide_hit", False) else 0.0,
                            "cache_resident_bytes": int(cache_profile.get("resident_bytes", 0) or 0),
                            "cache_resident_entries": int(cache_profile.get("resident_entries", 0) or 0),
                            "cache_evictions": int(cache_profile.get("evictions", 0) or 0),
                            "composed_guide_entries": int(cache_profile.get("composed_guide_entries", 0) or 0),
                            "composed_guide_hits_total": int(cache_profile.get("composed_guide_hits", 0) or 0),
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
                        writer.writerow(row)
                        f.flush()
                    if args.progress_queries and qno % args.progress_queries == 0:
                        latest = [r for r in rows if r["method"] == method and not r["error"]]
                        print(
                            f"{method} progress {qno}/{len(selected)} "
                            f"lat={statistics.fmean(float(r['latency_ms']) for r in latest):.2f} "
                            f"recall={statistics.fmean(float(r['recall']) for r in latest):.3f}",
                            flush=True,
                        )
                cur.execute("SELECT vector_hnsw_guidance_reset()")
    summary = summarize(rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summary)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.out.with_name(args.out.stem + '_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
