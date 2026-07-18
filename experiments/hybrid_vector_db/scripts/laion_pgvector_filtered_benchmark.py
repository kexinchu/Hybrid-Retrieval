from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from prepare_laion_pgvector import BASE_FBIN, DIM, INDEX, QUERY_FBIN, QUERY_TABLE, TABLE, xbin_mmap


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


def parse_ids(text: str) -> list[int]:
    if not text:
        return []
    return [int(x) for x in text.replace(",", " ").split()]


def load_filters(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            row["target_pct"] = float(row["target_pct"])
            row["actual_pct"] = float(row["actual_pct"])
            row["threshold"] = float(row["threshold"])
            row["rows"] = int(row["rows"])
            rows.append(row)
        return rows


def select_query_ids(total_queries: int, queries: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(total_queries, size=queries, replace=False))


def ensure_norms(base: np.memmap, norms_path: Path, chunk_rows: int) -> np.memmap:
    n = base.shape[0]
    norms_path.parent.mkdir(parents=True, exist_ok=True)
    norms = np.memmap(norms_path, dtype=np.float32, mode="w+" if not norms_path.exists() else "r+", shape=(n,))
    marker = norms_path.with_suffix(norms_path.suffix + ".done")
    if marker.exists():
        return norms
    start_time = time.perf_counter()
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = np.asarray(base[start:end], dtype=np.float32)
        norms[start:end] = np.einsum("ij,ij->i", chunk, chunk)
        if end % (chunk_rows * 5) == 0 or end == n:
            elapsed = time.perf_counter() - start_time
            print(f"  normed {end}/{n} rows at {end / max(elapsed, 1):.0f} rows/s", flush=True)
    norms.flush()
    marker.write_text(json.dumps({"rows": int(n), "dim": int(base.shape[1])}) + "\n")
    return norms


def update_topk(
    top_dist: np.ndarray,
    top_ids: np.ndarray,
    query_pos: int,
    candidate_dist: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> None:
    if candidate_dist.size == 0:
        return
    take = min(k, candidate_dist.size)
    local = np.argpartition(candidate_dist, take - 1)[:take]
    merged_dist = np.concatenate([top_dist[query_pos], candidate_dist[local]])
    merged_ids = np.concatenate([top_ids[query_pos], candidate_ids[local]])
    keep = np.argpartition(merged_dist, k - 1)[:k]
    order = np.argsort(merged_dist[keep], kind="stable")
    keep = keep[order]
    top_dist[query_pos] = merged_dist[keep]
    top_ids[query_pos] = merged_ids[keep]


def generate_truth(args: argparse.Namespace, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.truth.exists() and not args.recompute_truth:
        print(f"truth exists {args.truth}; skip exact GT generation", flush=True)
        with args.truth.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    base = xbin_mmap(args.base_fbin)
    queries = xbin_mmap(args.query_fbin)
    scores = np.memmap(args.scores, dtype=np.float32, mode="r", shape=(base.shape[0],))
    norms = ensure_norms(base, args.norms, args.chunk_rows)
    qids = select_query_ids(queries.shape[0], args.queries, args.query_seed)
    filter_masks = [(row, float(row["threshold"])) for row in filters]
    truth_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for batch_start in range(0, len(qids), args.query_batch):
        batch_qids = qids[batch_start : batch_start + args.query_batch]
        q = np.asarray(queries[batch_qids], dtype=np.float32)
        q_t = np.ascontiguousarray(q.T)
        q_norm = np.einsum("ij,ij->i", q, q)
        top_dist: dict[str, np.ndarray] = {}
        top_ids: dict[str, np.ndarray] = {}
        for row, _ in filter_masks:
            top_dist[row["filter_name"]] = np.full((len(batch_qids), args.k), np.inf, dtype=np.float32)
            top_ids[row["filter_name"]] = np.full((len(batch_qids), args.k), -1, dtype=np.int32)

        for start in range(0, base.shape[0], args.chunk_rows):
            end = min(start + args.chunk_rows, base.shape[0])
            xb = np.asarray(base[start:end], dtype=np.float32)
            dots = xb @ q_t
            dist = norms[start:end, None] + q_norm[None, :] - 2.0 * dots
            chunk_scores = scores[start:end]
            chunk_ids = np.arange(start, end, dtype=np.int32)
            for row, threshold in filter_masks:
                mask = chunk_scores <= threshold
                if not np.any(mask):
                    continue
                ids = chunk_ids[mask]
                masked = dist[mask]
                td = top_dist[row["filter_name"]]
                ti = top_ids[row["filter_name"]]
                for query_pos in range(len(batch_qids)):
                    update_topk(td, ti, query_pos, masked[:, query_pos], ids, args.k)
            if args.progress_chunks and (end // args.chunk_rows) % args.progress_chunks == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  truth batch {batch_start // args.query_batch + 1} "
                    f"chunk {end}/{base.shape[0]} elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )

        for row, _ in filter_masks:
            ids_for_filter = top_ids[row["filter_name"]]
            for pos, qid in enumerate(batch_qids):
                truth_rows.append(
                    {
                        "filter_name": row["filter_name"],
                        "target_pct": row["target_pct"],
                        "actual_pct": row["actual_pct"],
                        "threshold": row["threshold"],
                        "qid": int(qid),
                        "gt": " ".join(str(int(x)) for x in ids_for_filter[pos] if int(x) >= 0),
                    }
                )
        print(
            f"finished truth batch {batch_start + len(batch_qids)}/{len(qids)} "
            f"elapsed={(time.perf_counter() - started) / 60:.1f} min",
            flush=True,
        )

    write_csv(args.truth, truth_rows)
    print(f"wrote truth {args.truth}", flush=True)
    return truth_rows


def truth_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[int]]:
    out = {}
    for row in rows:
        out[(str(row["filter_name"]), int(row["qid"]))] = parse_ids(str(row["gt"]))
    return out


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
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in function_sql:
        cur.execute(sql)


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


def reset_profile(cur: psycopg.Cursor) -> None:
    try:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    except Exception:
        cur.connection.rollback()


def activate_guidance(cur: psycopg.Cursor, args: argparse.Namespace, method: str, predicate: str) -> tuple[dict[str, Any], float]:
    if method == "stock":
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        return {}, 0.0

    def run():
        cur.execute("SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)", (args.index, [f"sql:{predicate}"], method))
        cur.execute("SELECT vector_hnsw_guidance_profile()")
        value = cur.fetchone()[0]
        return json.loads(value) if isinstance(value, str) else dict(value)

    return timed_ms(run)


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    truth_k = [x for x in truth[:k] if x >= 0]
    if not truth_k:
        return 0.0
    return len(set(ids[:k]) & set(truth_k)) / min(k, len(truth_k))


def run_hnsw_query(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, qid: int) -> tuple[list[int], float, dict[str, Any], str]:
    reset_profile(cur)

    def execute():
        cur.execute(
            f"""
            SELECT id
            FROM {args.table}
            WHERE {predicate}
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (int(qid),),
        )
        return [int(row[0]) for row in cur.fetchall()]

    try:
        ids, latency_ms = timed_ms(execute)
        return ids, latency_ms, fetch_profile(cur), ""
    except errors.QueryCanceled as exc:
        cur.connection.rollback()
        configure(cur, args, True)
        return [], float(args.statement_timeout_ms), {}, exc.__class__.__name__


def run_sql_first(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, qid: int) -> tuple[list[int], float, str]:
    def execute():
        cur.execute(
            f"""
            WITH valid AS MATERIALIZED (
              SELECT id, embedding
              FROM {args.table}
              WHERE {predicate}
            )
            SELECT id
            FROM valid
            ORDER BY embedding <-> (SELECT embedding FROM {args.query_table} WHERE qid = %s)
            LIMIT {int(args.k)}
            """,
            (int(qid),),
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
        groups.setdefault((float(row["target_pct"]), str(row["method"])), []).append(row)
    out = []
    for (target, method), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ok = [row for row in items if not row.get("error")]
        if not ok:
            continue

        def vals(key: str) -> list[float]:
            return [float(row.get(key, 0) or 0) for row in ok]

        out.append(
            {
                "target_pct": target,
                "method": method,
                "queries": len(ok),
                "actual_pct_mean": statistics.fmean(vals("actual_pct")),
                "filter_rows_mean": statistics.fmean(vals("filter_rows")),
                "recall_mean": statistics.fmean(vals("recall")),
                "latency_ms_mean": statistics.fmean(vals("latency_ms")),
                "latency_ms_p50": statistics.median(vals("latency_ms")),
                "latency_ms_p95": p95(vals("latency_ms")),
                "activation_ms_mean": statistics.fmean(vals("activation_ms")),
                "vector_search_ms_mean": statistics.fmean(vals("vector_search_ms")),
                "visited_tuples_mean": statistics.fmean(vals("visited_tuples")),
                "returned_tuples_mean": statistics.fmean(vals("returned_tuples")),
                "guidance_skip_rate_mean": statistics.fmean(vals("guidance_skip_rate")),
                "errors": len(items) - len(ok),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAION10M controlled filtered pgvector benchmark.")
    parser.add_argument("--base-fbin", type=Path, default=BASE_FBIN)
    parser.add_argument("--query-fbin", type=Path, default=QUERY_FBIN)
    parser.add_argument("--scores", type=Path, default=Path("results/hybrid_vector_db/laion10m_topic_score_seed13.float32"))
    parser.add_argument("--norms", type=Path, default=Path("results/hybrid_vector_db/laion10m_l2_norm.float32"))
    parser.add_argument("--filters", type=Path, default=Path("results/hybrid_vector_db/laion10m_controlled_filters_20260713.csv"))
    parser.add_argument("--truth", type=Path, default=Path("results/hybrid_vector_db/laion10m_controlled_truth_q100_20260713.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/laion10m_pgvector_controlled_20260713.csv"))
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--query-table", default=QUERY_TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--methods", nargs="+", default=["stock", "bloom", "page", "sql_first"], choices=["stock", "bloom", "page", "exact", "sql_first"])
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--query-seed", type=int, default=20260713)
    parser.add_argument("--query-batch", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--progress-chunks", type=int, default=10)
    parser.add_argument("--recompute-truth", action="store_true")
    parser.add_argument("--truth-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--metadata-cache-mb", type=int, default=1024)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--progress-queries", type=int, default=10)
    args = parser.parse_args()

    filters = load_filters(args.filters)
    if args.filter_names:
        wanted = set(args.filter_names)
        filters = [row for row in filters if row["filter_name"] in wanted]
        if not filters:
            raise SystemExit(f"no filters matched --filter-names {sorted(wanted)}")
    if args.benchmark_only:
        with args.truth.open(newline="", encoding="utf-8") as f:
            truth_rows = list(csv.DictReader(f))
    else:
        truth_rows = generate_truth(args, filters)
    if args.truth_only:
        return
    qids = sorted({int(row["qid"]) for row in truth_rows})
    if args.benchmark_only and args.queries and args.queries < len(qids):
        keep_qids = set(qids[: args.queries])
        truth_rows = [row for row in truth_rows if int(row["qid"]) in keep_qids]
        qids = sorted(keep_qids)
    truth = truth_map(truth_rows)

    rows: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        vector_functions(cur)
        configure(cur, args, True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "method",
                "filter_name",
                "target_pct",
                "actual_pct",
                "filter_rows",
                "threshold",
                "predicate",
                "qid",
                "repeat",
                "recall",
                "latency_ms",
                "activation_ms",
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
                for filter_row in filters:
                    predicate = str(filter_row["predicate"])
                    if method == "sql_first":
                        configure(cur, args, False)
                        activation_ms = 0.0
                    else:
                        configure(cur, args, True)
                        _, activation_ms = activate_guidance(cur, args, method, predicate)
                    for qno, qid in enumerate(qids, start=1):
                        for repeat in range(args.repeats):
                            if method == "sql_first":
                                ids, latency_ms, error = run_sql_first(cur, args, predicate, qid)
                                profile: dict[str, Any] = {}
                            else:
                                ids, latency_ms, profile, error = run_hnsw_query(cur, args, predicate, qid)
                            checks = float(profile.get("guidance_checks", 0) or 0)
                            skips = float(profile.get("guidance_skips", 0) or 0)
                            row = {
                                "method": method,
                                "filter_name": filter_row["filter_name"],
                                "target_pct": filter_row["target_pct"],
                                "actual_pct": filter_row["actual_pct"],
                                "filter_rows": filter_row["rows"],
                                "threshold": filter_row["threshold"],
                                "predicate": predicate,
                                "qid": qid,
                                "repeat": repeat,
                                "recall": recall_at_k(ids, truth[(str(filter_row["filter_name"]), qid)], args.k) if not error else 0.0,
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
                            }
                            rows.append(row)
                            writer.writerow(row)
                            f.flush()
                        if args.progress_queries and qno % args.progress_queries == 0:
                            latest = [r for r in rows if r["method"] == method and not r["error"]]
                            if latest:
                                print(
                                    f"{method} {filter_row['filter_name']} progress {qno}/{len(qids)} "
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
