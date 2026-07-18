from __future__ import annotations

import argparse
import csv
import json
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common_pg import pg_config_from_env, require_psycopg
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
from pgvector_prefilter_10m import TABLE, load_truth, vector_literal


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def flatten_plan(node: dict[str, Any], out: list[dict[str, Any]]) -> None:
    out.append(node)
    for child in node.get("Plans", []) or []:
        flatten_plan(child, out)


def plan_buffer_totals(plan: dict[str, Any]) -> dict[str, float]:
    nodes: list[dict[str, Any]] = []
    flatten_plan(plan["Plan"], nodes)
    keys = [
        "Shared Hit Blocks",
        "Shared Read Blocks",
        "Shared Dirtied Blocks",
        "Shared Written Blocks",
        "Local Hit Blocks",
        "Local Read Blocks",
        "Temp Read Blocks",
        "Temp Written Blocks",
    ]
    totals = {key: 0.0 for key in keys}
    for node in nodes:
        for key in keys:
            totals[key] += float(node.get(key, 0.0) or 0.0)
    return totals


def explain_analyze(cur, sql: str, params: tuple[object, ...] = ()) -> tuple[dict[str, Any], float]:
    def run():
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
        value = cur.fetchone()[0]
        if isinstance(value, str):
            return json.loads(value)[0]
        return value[0]

    return timed(run)


def configure_session(cur, args: argparse.Namespace) -> None:
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    if args.iterative_scan:
        cur.execute(f"SET hnsw.iterative_scan = '{args.iterative_scan}'")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute("SET enable_hashjoin = off")
    cur.execute("SET enable_mergejoin = off")
    cur.execute(f"SET enable_seqscan = {'off' if args.disable_seqscan else 'on'}")


def create_candidate_table(cur) -> None:
    cur.execute("DROP TABLE IF EXISTS pgvector_page_candidates")
    cur.execute(
        """
        CREATE TEMP TABLE pgvector_page_candidates (
            ord integer NOT NULL,
            id bigint NOT NULL,
            heap_tid tid NOT NULL,
            heap_block bigint NOT NULL,
            heap_off integer NOT NULL
        ) ON COMMIT PRESERVE ROWS
        """
    )


def parse_embedding(value: object) -> np.ndarray:
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("[") and v.endswith("]"):
            v = v[1:-1]
        return np.fromstring(v, sep=",", dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def load_query_vectors_from_table(cur, table: str, query_ids: list[int]) -> dict[int, np.ndarray]:
    if not query_ids:
        return {}
    cur.execute(
        f"""
        SELECT id, embedding
        FROM {table}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    rows = cur.fetchall()
    vectors = {int(row[0]): parse_embedding(row[1]) for row in rows}
    missing = [qid for qid in query_ids if qid not in vectors]
    if missing:
        raise RuntimeError(f"missing embeddings for {len(missing)} query ids from {table}")
    return vectors


def load_candidates(cur, table: str, query: np.ndarray, candidate_limit: int) -> tuple[int, float]:
    q = vector_literal(query)

    def run() -> int:
        cur.execute("TRUNCATE pgvector_page_candidates")
        cur.execute(
            f"""
            INSERT INTO pgvector_page_candidates (ord, id, heap_tid, heap_block, heap_off)
            SELECT
                row_number() OVER ()::integer AS ord,
                id,
                ctid AS heap_tid,
                split_part(split_part(ctid::text, ',', 1), '(', 2)::bigint AS heap_block,
                rtrim(split_part(ctid::text, ',', 2), ')')::integer AS heap_off
            FROM (
                SELECT id, ctid
                FROM {table}
                ORDER BY embedding <-> %s::vector
                LIMIT {int(candidate_limit)}
            ) AS ann
            """,
            (q,),
        )
        return int(cur.rowcount)

    return timed(run)


def candidate_locality(cur) -> dict[str, float]:
    cur.execute(
        """
        WITH ordered AS (
            SELECT
                ord,
                heap_block,
                lag(heap_block) OVER (ORDER BY ord) AS prev_block
            FROM pgvector_page_candidates
        )
        SELECT
            count(*)::float8 AS candidates,
            count(DISTINCT heap_block)::float8 AS distinct_heap_pages,
            coalesce(sum(CASE WHEN prev_block IS NULL OR heap_block <> prev_block THEN 1 ELSE 0 END), 0)::float8
                AS distance_order_page_runs
        FROM ordered
        """
    )
    candidates, distinct_pages, page_runs = cur.fetchone()
    return {
        "candidates": float(candidates),
        "distinct_heap_pages": float(distinct_pages),
        "distance_order_page_runs": float(page_runs),
        "distance_order_mean_run": float(candidates) / max(float(page_runs), 1.0),
        "ideal_page_order_mean_run": float(candidates) / max(float(distinct_pages), 1.0),
    }


def verify_sql(order: str, predicate: str, table: str, materialized: bool) -> str:
    if order == "distance":
        order_clause = "ord"
    elif order == "page":
        order_clause = "heap_block, heap_off"
    else:
        raise ValueError(order)

    materialized_keyword = "MATERIALIZED" if materialized else ""
    return f"""
        WITH ordered AS {materialized_keyword} (
            SELECT *
            FROM pgvector_page_candidates
            ORDER BY {order_clause}
        ),
        verified AS {materialized_keyword} (
            SELECT c.id, c.ord
            FROM ordered c
            JOIN LATERAL (
                SELECT 1
                FROM {table} t
                WHERE t.ctid = c.heap_tid AND {predicate}
            ) ok ON true
        )
        SELECT id
        FROM verified
        ORDER BY ord
    """


def run_verification(
    cur,
    table: str,
    order: str,
    predicate: str,
    k: int,
    repeats: int,
    materialized: bool,
    explain: bool,
) -> tuple[list[int], dict[str, float]]:
    sql = verify_sql(order, predicate, table, materialized)
    limit_sql = f"SELECT id FROM ({sql}) v LIMIT {int(k)}"
    all_ids: list[int] = []
    latencies: list[float] = []
    buffer_runs: list[dict[str, float]] = []

    for idx in range(repeats):
        if explain and idx == 0:
            plan, _ = explain_analyze(cur, limit_sql)
            buffer_runs.append(plan_buffer_totals(plan))
        ids, elapsed = timed(lambda: fetch_ids(cur, limit_sql))
        all_ids = ids
        latencies.append(elapsed)

    metrics: dict[str, float] = {
        f"{order}_verify_ms": statistics.mean(latencies),
        f"{order}_verify_ms_p50": statistics.median(latencies),
    }
    if buffer_runs:
        for key, value in buffer_runs[0].items():
            metrics[f"{order}_{key.lower().replace(' ', '_')}"] = value
    return all_ids, metrics


def fetch_ids(cur, sql: str) -> list[int]:
    cur.execute(sql)
    return [int(row[0]) for row in cur.fetchall()]


def selected_filters(filter_names: list[str] | None) -> list[tuple[str, str, str]]:
    selected = set(filter_names or [])
    return [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]


def load_query_id_csv(path: Path) -> dict[int, int]:
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            query_by_no[int(row["query_no"])] = int(row["query_id"])
    if not query_by_no:
        raise RuntimeError(f"no query ids loaded from {path}")
    return query_by_no


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["filter_name"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for filter_name, items in groups.items():
        recall_values = [float(r["recall"]) for r in items if str(r["recall"]) != ""]
        same_values = [str(r["same_results"]) for r in items if str(r["same_results"]) != ""]
        distance_verify_ms = statistics.mean(float(r["distance_verify_ms"]) for r in items)
        page_verify_ms = statistics.mean(float(r["page_verify_ms"]) for r in items)
        summary_rows.append(
            {
                "table": items[0].get("table", ""),
                "filter_name": filter_name,
                "queries": len(items),
                "candidate_limit": int(items[0]["candidate_limit"]),
                "recall_mean": statistics.mean(recall_values) if recall_values else "",
                "candidate_ms_mean": statistics.mean(float(r["candidate_ms"]) for r in items),
                "distance_verify_ms_mean": distance_verify_ms,
                "page_verify_ms_mean": page_verify_ms,
                "verify_speedup": "" if page_verify_ms == 0 else distance_verify_ms / max(page_verify_ms, 1e-9),
                "distinct_heap_pages_mean": statistics.mean(float(r["distinct_heap_pages"]) for r in items),
                "distance_order_page_runs_mean": statistics.mean(float(r["distance_order_page_runs"]) for r in items),
                "page_run_reduction": statistics.mean(float(r["distance_order_page_runs"]) for r in items)
                / max(statistics.mean(float(r["distinct_heap_pages"]) for r in items), 1.0),
                "same_results": all(value == "True" for value in same_values) if same_values else "",
            }
        )

    summary = out.with_name(out.stem + "_summary.csv")
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare distance-order vs heap-page-order verification for the same pgvector candidates."
    )
    parser.add_argument(
        "--fbin",
        type=Path,
        default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"),
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/hybrid_vector_db/pgvector_page_cluster_verify.csv"),
    )
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--max-scan-tuples", type=int, default=200_000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=1.0)
    parser.add_argument("--iterative-scan", choices=["strict_order", "relaxed_order"])
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--truth-method", default="pre_filter_exact")
    parser.add_argument(
        "--query-id-csv",
        type=Path,
        help="CSV with query_no and query_id columns, for Amazon-C4 or other custom query selections",
    )
    parser.add_argument("--query-vectors-from-db", action="store_true")
    parser.add_argument("--locality-only", action="store_true")
    parser.add_argument("--stream", action="store_true", help="write each result row immediately instead of only at the end")
    parser.add_argument("--disable-seqscan", action="store_true", default=True)
    parser.add_argument("--no-materialized", action="store_true")
    parser.add_argument("--explain", action="store_true", help="collect EXPLAIN ANALYZE BUFFERS for the first repeat")
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    if args.query_id_csv is not None:
        query_by_no = load_query_id_csv(args.query_id_csv)
        truth: dict[tuple[str, int], list[int]] = {}
    else:
        truth, query_by_no = load_truth(args.truth_csv, args.truth_method)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    if args.query_vectors_from_db:
        xb = None
        query_vector_lookup = None
    else:
        xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
        query_vector_lookup = None

    rows_out: list[dict[str, object]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stream_file = args.out.open("w", newline="") if args.stream else None
    stream_writer: csv.DictWriter | None = None
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            configure_session(cur, args)
            create_candidate_table(cur)
            if args.query_vectors_from_db:
                query_ids = [query_by_no[qno] for qno in query_nos]
                query_vector_lookup = load_query_vectors_from_table(cur, args.table, query_ids)

            for filter_name, target_rate, predicate in selected_filters(args.filter_names):
                for qno in query_nos:
                    qid = query_by_no[qno]
                    if query_vector_lookup is not None:
                        query = query_vector_lookup[qid]
                    else:
                        query = np.asarray(xb[qid], dtype=np.float32)

                    candidate_count, candidate_ms = load_candidates(cur, args.table, query, args.candidate_limit)
                    locality = candidate_locality(cur)
                    if args.locality_only:
                        distance_ids = []
                        page_ids = []
                        distance_metrics = {"distance_verify_ms": 0.0, "distance_verify_ms_p50": 0.0}
                        page_metrics = {"page_verify_ms": 0.0, "page_verify_ms_p50": 0.0}
                    else:
                        distance_ids, distance_metrics = run_verification(
                            cur,
                            args.table,
                            "distance",
                            predicate,
                            args.k,
                            args.repeats,
                            not args.no_materialized,
                            args.explain,
                        )
                        page_ids, page_metrics = run_verification(
                            cur,
                            args.table,
                            "page",
                            predicate,
                            args.k,
                            args.repeats,
                            not args.no_materialized,
                            args.explain,
                        )
                    truth_ids = truth.get((filter_name, qno), [])
                    row = {
                        "filter": target_rate,
                        "filter_name": filter_name,
                        "query_no": qno,
                        "query_id": qid,
                        "candidate_limit": args.candidate_limit,
                        "candidate_count": candidate_count,
                        "candidate_ms": candidate_ms,
                        "table": args.table,
                        "recall": "" if args.locality_only or not truth_ids else recall_at_k(distance_ids, truth_ids, args.k),
                        "same_results": "" if args.locality_only else distance_ids == page_ids,
                        "distance_ids": ",".join(str(x) for x in distance_ids),
                        "page_ids": ",".join(str(x) for x in page_ids),
                        "ef_search": args.ef_search,
                        "iterative_scan": args.iterative_scan or "",
                        "max_scan_tuples": args.max_scan_tuples,
                        "scan_mem_multiplier": args.scan_mem_multiplier,
                        **locality,
                        **distance_metrics,
                        **page_metrics,
                    }
                    rows_out.append(row)
                    if stream_file is not None:
                        if stream_writer is None:
                            stream_writer = csv.DictWriter(stream_file, fieldnames=list(row.keys()))
                            stream_writer.writeheader()
                        stream_writer.writerow(row)
                        stream_file.flush()
                    recall_display = "" if row["recall"] == "" else f"{float(row['recall']):.3f}"
                    print(
                        f"filter={filter_name} q={qno} candidates={candidate_count} "
                        f"recall={recall_display} "
                        f"same={row['same_results']} "
                        f"distance_ms={row['distance_verify_ms']:.2f} page_ms={row['page_verify_ms']:.2f} "
                        f"page_runs={row['distance_order_page_runs']:.0f}->{row['distinct_heap_pages']:.0f}",
                        flush=True,
                    )
    if stream_file is not None:
        stream_file.close()

    if not rows_out:
        raise RuntimeError("no benchmark rows produced")

    if not args.stream:
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    print(f"wrote {args.out}", flush=True)
    summarize(rows_out, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
