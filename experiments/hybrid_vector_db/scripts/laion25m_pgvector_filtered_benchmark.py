from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from prepare_laion25m_pgvector import INDEX, QUERY_TABLE, TABLE


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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_array(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if text in {"", "{}", "[]"}:
        return []
    text = text.strip("{}[]")
    if "," in text:
        return [int(x) for x in text.split(",") if x.strip()]
    return [int(x) for x in text.split() if x.strip()]


def parse_ids(text: Any) -> list[int]:
    value = str(text or "").strip()
    if not value:
        return []
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(float(text))


def parse_width_bucket_edges(value: str | None) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return sorted({int(float(x)) for x in text.replace(",", " ").split() if x.strip()})


def truth_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["workload"]),
            str(float(row["target_band_pct"])),
            str(int(row["qid"])),
            str(row["filter_name"]),
        ]
    )


def load_truth(path: Path | None) -> dict[str, list[int]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("truth_key") or truth_key(row)
            out[str(key)] = parse_ids(row.get("gt", ""))
    return out


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    truth_k = [x for x in truth[:k] if x >= 0]
    if not truth_k:
        return 0.0
    return len(set(ids[:k]) & set(truth_k)) / min(k, len(truth_k))


def sql_int_array(values: list[int]) -> str:
    return "ARRAY[" + ",".join(str(int(x)) for x in values) + "]::int[]"


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


def width_ranges(cur: psycopg.Cursor, args: argparse.Namespace) -> dict[float, tuple[int, int]]:
    ranges: dict[float, tuple[int, int]] = {}
    for pct in args.target_bands:
        cur.execute(f"SELECT percentile_disc(%s) WITHIN GROUP (ORDER BY width) FROM {args.table}", (pct / 100.0,))
        upper = int(cur.fetchone()[0])
        ranges[float(pct)] = (-2147483648, upper + 1)
    return ranges


def count_predicate(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str) -> tuple[int, float]:
    def run():
        cur.execute(f"SELECT count(*) FROM {args.table} WHERE {predicate}")
        return int(cur.fetchone()[0])

    return timed_ms(run)


def query_candidates(cur: psycopg.Cursor, args: argparse.Namespace) -> list[dict[str, Any]]:
    cur.execute(
        f"""
        SELECT qid, labels, label_count, width
        FROM {args.query_table}
        WHERE label_count > 0
        ORDER BY md5(qid::text)
        LIMIT %s
        """,
        (args.query_sample,),
    )
    rows = []
    for qid, labels, label_count, width in cur.fetchall():
        rows.append({"qid": int(qid), "labels": parse_int_array(labels), "label_count": int(label_count), "width": int(width)})
    return rows


def make_filter(workload: str, qid: int, labels: list[int], range_l: int | None, range_r: int | None) -> tuple[str, str]:
    arr = sql_int_array(labels)
    if workload == "label":
        label = int(labels[0])
        predicate = f"labels @> ARRAY[{label}]::int[]"
        return predicate, f"label_{label}"
    if workload == "label_or":
        predicate = f"labels && {arr}"
        return predicate, "labelor_" + "_".join(str(x) for x in labels[:8])
    if workload == "range":
        assert range_l is not None and range_r is not None
        predicate = f"width >= {int(range_l)} AND width < {int(range_r)}"
        return predicate, f"width_{int(range_l)}_{int(range_r)}"
    if workload == "hybrid":
        assert range_l is not None and range_r is not None
        predicate = f"(labels && {arr}) OR (width >= {int(range_l)} AND width < {int(range_r)})"
        return predicate, "hybrid_" + "_".join(str(x) for x in labels[:4]) + f"_{int(range_l)}_{int(range_r)}"
    raise ValueError(workload)


def sql_label_atom(label: int) -> str:
    return f"sql:labels @> ARRAY[{int(label)}]::int[]"


def sql_range_atom(lo: int, hi: int) -> str:
    return f"sql:width >= {int(lo)} AND width < {int(hi)}"


def or_atoms(parts: list[str]) -> list[str]:
    atoms: list[str] = []
    for part in parts:
        if atoms:
            atoms.append("|")
        atoms.append(part)
    return atoms


def sql_range_bucket_atoms(lo: int, hi: int, args: argparse.Namespace) -> list[str]:
    edges = list(getattr(args, "width_bucket_edges_list", []))
    if not edges and int(args.width_bucket_size) > 0:
        size = int(args.width_bucket_size)
        first = ((int(lo) // size) + 1) * size
        edges = list(range(first, int(hi), size))
    points = [int(lo)] + [int(edge) for edge in edges if int(lo) < int(edge) < int(hi)] + [int(hi)]
    points = sorted(set(points))
    parts = [sql_range_atom(points[i], points[i + 1]) for i in range(len(points) - 1) if points[i] < points[i + 1]]
    return or_atoms(parts or [sql_range_atom(lo, hi)])


def guidance_atoms(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    mode = str(args.guidance_mode)
    if mode == "full_sql":
        return [f"sql:{row['predicate']}"]
    if mode not in {"fragment_atoms", "width_bucket_atoms"}:
        raise ValueError(mode)

    workload = str(row["workload"])
    labels = parse_int_array(row.get("labels", ""))
    lo = parse_optional_int(row.get("range_l"))
    hi = parse_optional_int(row.get("range_r"))

    if workload == "label":
        if not labels:
            raise ValueError("label workload without labels")
        return [sql_label_atom(labels[0])]
    if workload == "label_or":
        if not labels:
            raise ValueError("label_or workload without labels")
        return or_atoms([sql_label_atom(label) for label in labels])
    if workload == "range":
        if lo is None or hi is None:
            raise ValueError("range workload without bounds")
        if mode == "width_bucket_atoms":
            return sql_range_bucket_atoms(lo, hi, args)
        return [sql_range_atom(lo, hi)]
    if workload == "hybrid":
        if not labels or lo is None or hi is None:
            raise ValueError("hybrid workload without labels or bounds")
        parts = [sql_label_atom(label) for label in labels]
        if mode == "width_bucket_atoms":
            width_atoms = sql_range_bucket_atoms(lo, hi, args)
            parts.extend(atom for atom in width_atoms if atom != "|")
            return or_atoms(parts)
        parts.append(sql_range_atom(lo, hi))
        return or_atoms(parts)
    raise ValueError(workload)


def select_workload(cur: psycopg.Cursor, args: argparse.Namespace) -> list[dict[str, Any]]:
    total = table_rows(cur, args.table)
    candidates = query_candidates(cur, args)
    ranges = width_ranges(cur, args)
    selected: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    range_rows: dict[float, dict[str, Any]] = {}
    if "range" in args.workloads or "hybrid" in args.workloads:
        for target, (lo, hi) in ranges.items():
            predicate, name = make_filter("range", 0, [], lo, hi)
            count, count_ms = count_predicate(cur, args, predicate)
            range_rows[target] = {
                "workload": "range",
                "filter_name": name,
                "target_band_pct": target,
                "actual_pct": 100.0 * count / max(total, 1),
                "filter_rows": count,
                "labels": "",
                "range_l": lo,
                "range_r": hi,
                "predicate": predicate,
                "count_ms": count_ms,
            }
            print(f"range target={target} actual={range_rows[target]['actual_pct']:.3f}% rows={count}", flush=True)

    label_rows: list[dict[str, Any]] = []
    label_or_rows: list[dict[str, Any]] = []
    for query in candidates:
        labels = query["labels"]
        if not labels:
            continue
        label = rng.choice(labels)
        if "label" in args.workloads:
            predicate, name = make_filter("label", query["qid"], [label], None, None)
            count, count_ms = count_predicate(cur, args, predicate)
            label_rows.append(
                {
                    "workload": "label",
                    "filter_name": name,
                    "target_band_pct": 0.0,
                    "actual_pct": 100.0 * count / max(total, 1),
                    "filter_rows": count,
                    "qid": query["qid"],
                    "labels": str(label),
                    "range_l": "",
                    "range_r": "",
                    "predicate": predicate,
                    "count_ms": count_ms,
                }
            )
        if "label_or" in args.workloads or "hybrid" in args.workloads:
            predicate, name = make_filter("label_or", query["qid"], labels, None, None)
            count, count_ms = count_predicate(cur, args, predicate)
            label_or_rows.append(
                {
                    "workload": "label_or",
                    "filter_name": name,
                    "target_band_pct": 0.0,
                    "actual_pct": 100.0 * count / max(total, 1),
                    "filter_rows": count,
                    "qid": query["qid"],
                    "labels": " ".join(str(x) for x in labels),
                    "range_l": "",
                    "range_r": "",
                    "predicate": predicate,
                    "count_ms": count_ms,
                }
            )
        if len(label_or_rows) % 50 == 0 and label_or_rows:
            print(f"measured label candidates {len(label_or_rows)}", flush=True)

    used: set[tuple[str, int, float]] = set()
    for workload in args.workloads:
        if workload == "range":
            qids = [row["qid"] for row in candidates[: args.queries_per_band]]
            for target in args.target_bands:
                base = range_rows[float(target)]
                for qid in qids:
                    out = dict(base)
                    out["qid"] = qid
                    selected.append(out)
        elif workload in {"label", "label_or"}:
            source = label_rows if workload == "label" else label_or_rows
            for target in args.target_bands:
                rows = sorted(
                    [r for r in source if r["filter_rows"] >= args.k],
                    key=lambda r, t=target: (abs(float(r["actual_pct"]) - float(t)), int(r["qid"])),
                )
                picked = 0
                for row in rows:
                    key = (workload, int(row["qid"]), float(target))
                    if key in used:
                        continue
                    out = dict(row)
                    out["target_band_pct"] = target
                    selected.append(out)
                    used.add(key)
                    picked += 1
                    if picked >= args.queries_per_band:
                        break
        elif workload == "hybrid":
            source = sorted([r for r in label_or_rows if r["filter_rows"] >= args.k], key=lambda r: int(r["qid"]))
            for target in args.target_bands:
                lo, hi = ranges[float(target)]
                picked = 0
                for row in source:
                    labels = parse_int_array(row["labels"])
                    predicate, name = make_filter("hybrid", int(row["qid"]), labels, lo, hi)
                    count, count_ms = count_predicate(cur, args, predicate)
                    out = {
                        "workload": "hybrid",
                        "filter_name": name,
                        "target_band_pct": target,
                        "actual_pct": 100.0 * count / max(total, 1),
                        "filter_rows": count,
                        "qid": int(row["qid"]),
                        "labels": row["labels"],
                        "range_l": lo,
                        "range_r": hi,
                        "predicate": predicate,
                        "count_ms": count_ms,
                    }
                    selected.append(out)
                    picked += 1
                    if picked >= args.queries_per_band:
                        break
    if not selected:
        raise SystemExit("no LAION25M workload selected")
    write_csv(args.selected_queries_out, selected)
    print(f"selected {len(selected)} workload rows -> {args.selected_queries_out}", flush=True)
    return selected


def load_workload(cur: psycopg.Cursor, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.selected_queries_in and args.selected_queries_in.exists():
        with args.selected_queries_in.open(newline="", encoding="utf-8") as f:
            rows = []
            for row in csv.DictReader(f):
                row["qid"] = int(row["qid"])
                row["target_band_pct"] = float(row["target_band_pct"])
                row["actual_pct"] = float(row["actual_pct"])
                row["filter_rows"] = int(row["filter_rows"])
                rows.append(row)
    else:
        rows = select_workload(cur, args)

    workloads = set(args.workloads or [])
    targets = {float(x) for x in args.target_bands} if args.target_bands else set()
    filtered: list[dict[str, Any]] = []
    per_group: dict[tuple[str, float], int] = {}
    for row in rows:
        if workloads and str(row["workload"]) not in workloads:
            continue
        if targets and float(row["target_band_pct"]) not in targets:
            continue
        group = (str(row["workload"]), float(row["target_band_pct"]))
        if args.limit_per_group > 0 and per_group.get(group, 0) >= args.limit_per_group:
            continue
        per_group[group] = per_group.get(group, 0) + 1
        filtered.append(row)
    return filtered


def activate_guidance(cur: psycopg.Cursor, args: argparse.Namespace, method: str, row: dict[str, Any]) -> tuple[dict[str, Any], float]:
    if method == "stock":
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        return {}, 0.0
    atoms = guidance_atoms(row, args)

    def run():
        cur.execute("SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)", (args.index, atoms, method))
        cur.execute("SELECT vector_hnsw_guidance_profile()")
        value = cur.fetchone()[0]
        return json.loads(value) if isinstance(value, str) else dict(value)

    return timed_ms(run)


def run_hnsw_query(cur: psycopg.Cursor, args: argparse.Namespace, row: dict[str, Any]) -> tuple[list[int], float, dict[str, Any], str]:
    reset_profile(cur)

    def execute():
        cur.execute(
            f"""
            SELECT id
            FROM {args.table}
            WHERE {row["predicate"]}
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (int(row["qid"]),),
        )
        return [int(x[0]) for x in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        return ids, latency_ms, fetch_profile(cur), ""
    except errors.QueryCanceled as exc:
        cur.connection.rollback()
        configure(cur, args, True)
        return [], float(args.statement_timeout_ms), {}, exc.__class__.__name__


def run_sql_first(cur: psycopg.Cursor, args: argparse.Namespace, row: dict[str, Any]) -> tuple[list[int], float, str]:
    def execute():
        cur.execute(
            f"""
            WITH valid AS MATERIALIZED (
              SELECT id, embedding
              FROM {args.table}
              WHERE {row["predicate"]}
            )
            SELECT id
            FROM valid
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (int(row["qid"]),),
        )
        return [int(x[0]) for x in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        return ids, latency_ms, ""
    except errors.QueryCanceled as exc:
        cur.connection.rollback()
        configure(cur, args, False)
        return [], float(args.statement_timeout_ms), exc.__class__.__name__


def p95(values: list[float]) -> float:
    vals = sorted(values)
    return vals[max(0, int(0.95 * len(vals)) - 1)] if vals else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["workload"]), float(row["target_band_pct"]), str(row["method"])), []).append(row)
    out = []
    for (workload, target, method), items in sorted(groups.items()):
        ok = [row for row in items if not row.get("error")]
        if not ok:
            continue

        def vals(key: str) -> list[float]:
            return [float(row.get(key, 0) or 0) for row in ok]

        out.append(
            {
                "workload": workload,
                "target_band_pct": target,
                "method": method,
                "queries": len(ok),
                "actual_pct_mean": statistics.fmean(vals("actual_pct")),
                "filter_rows_mean": statistics.fmean(vals("filter_rows")),
                "recall_mean": statistics.fmean(
                    float(row["recall"]) for row in ok if str(row.get("recall", "")).strip()
                )
                if any(str(row.get("recall", "")).strip() for row in ok)
                else "",
                "latency_ms_mean": statistics.fmean(vals("latency_ms")),
                "latency_ms_p50": statistics.median(vals("latency_ms")),
                "latency_ms_p95": p95(vals("latency_ms")),
                "activation_ms_mean": statistics.fmean(vals("activation_ms")),
                "activation_build_ms_mean": statistics.fmean(vals("activation_build_ms")),
                "fragment_cache_hits_mean": statistics.fmean(vals("fragment_cache_hits")),
                "fragment_cache_misses_mean": statistics.fmean(vals("fragment_cache_misses")),
                "fragment_store_hits_mean": statistics.fmean(vals("fragment_store_hits")),
                "fragment_builds_mean": statistics.fmean(vals("fragment_builds")),
                "composed_guide_hit_rate": statistics.fmean(1.0 if row.get("composed_guide_hit") in {True, "True", "true", "1", 1} else 0.0 for row in ok),
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
    parser = argparse.ArgumentParser(description="Run LAION25M caption/range filtered pgvector benchmark.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/laion25m_pgvector_filtered_20260714.csv"))
    parser.add_argument("--selected-queries-out", type=Path, default=Path("results/hybrid_vector_db/laion25m_selected_filters_20260714.csv"))
    parser.add_argument("--selected-queries-in", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--require-truth", action="store_true")
    parser.add_argument("--workloads", nargs="+", default=["label_or", "range", "hybrid"], choices=["label", "label_or", "range", "hybrid"])
    parser.add_argument("--target-bands", type=float, nargs="+", default=TARGET_BANDS)
    parser.add_argument("--query-sample", type=int, default=500)
    parser.add_argument("--queries-per-band", type=int, default=20)
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["stock", "bloom", "page"], choices=["stock", "bloom", "page", "sql_first"])
    parser.add_argument("--guidance-mode", default="full_sql", choices=["full_sql", "fragment_atoms", "width_bucket_atoms"])
    parser.add_argument(
        "--width-bucket-edges",
        default="",
        help="Comma/space separated canonical width boundaries used by --guidance-mode width_bucket_atoms.",
    )
    parser.add_argument(
        "--width-bucket-size",
        type=int,
        default=128,
        help="Fallback fixed width bucket size when --width-bucket-edges is empty.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--metadata-cache-mb", type=int, default=4096)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=20)
    parser.add_argument("--backend-cpu-list", default="")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    args.width_bucket_edges_list = parse_width_bucket_edges(args.width_bucket_edges)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = pg_config_from_env()
    rows: list[dict[str, Any]] = []
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        pin_backend(cur, args.backend_cpu_list)
        vector_functions(cur)
        configure(cur, args, True)
        selected = load_workload(cur, args)
        truth = load_truth(args.truth)
        if args.require_truth:
            missing = [truth_key(row) for row in selected if truth_key(row) not in truth]
            if missing:
                raise SystemExit(f"missing truth for {len(missing)} selected rows; first={missing[0]}")
        if args.select_only:
            return
        with args.out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "method",
                "workload",
                "target_band_pct",
                "actual_pct",
                "filter_rows",
                "qid",
                "filter_name",
                "predicate",
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
                for qno, query in enumerate(selected, start=1):
                    activation_profile: dict[str, Any] = {}
                    if method == "sql_first":
                        configure(cur, args, False)
                        activation_ms = 0.0
                    else:
                        configure(cur, args, True)
                        activation_profile, activation_ms = activate_guidance(cur, args, method, query)
                    for repeat in range(args.repeats):
                        if method == "sql_first":
                            ids, latency_ms, error = run_sql_first(cur, args, query)
                            profile: dict[str, Any] = {}
                        else:
                            ids, latency_ms, profile, error = run_hnsw_query(cur, args, query)
                        cache_profile = fetch_cache_profile(cur)
                        checks = float(profile.get("guidance_checks", 0) or 0)
                        skips = float(profile.get("guidance_skips", 0) or 0)
                        expected = truth.get(truth_key(query))
                        row = {
                            "method": method,
                            "workload": query["workload"],
                            "target_band_pct": query["target_band_pct"],
                            "actual_pct": query["actual_pct"],
                            "filter_rows": query["filter_rows"],
                            "qid": query["qid"],
                            "filter_name": query["filter_name"],
                            "predicate": query["predicate"],
                            "repeat": repeat,
                            "recall": recall_at_k(ids, expected, args.k) if expected is not None and not error else "",
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
                            "composed_guide_hit": bool(activation_profile.get("composed_guide_hit", False)),
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
                        recall_values = [float(r["recall"]) for r in latest if str(r.get("recall", "")).strip()]
                        recall_text = f" recall={statistics.fmean(recall_values):.3f}" if recall_values else ""
                        print(
                            f"{method} progress {qno}/{len(selected)} "
                            f"lat={statistics.fmean(float(r['latency_ms']) for r in latest):.2f}"
                            f"{recall_text}",
                            flush=True,
                        )
                cur.execute("SELECT vector_hnsw_guidance_reset()")
    summary = summarize(rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summary)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.out.with_name(args.out.stem + '_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
