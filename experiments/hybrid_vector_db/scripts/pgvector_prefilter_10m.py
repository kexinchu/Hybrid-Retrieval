from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from common_pg import pg_config_from_env, require_psycopg
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k


TABLE = "amazon_grocery_reviews_10m_pgvector"


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


def load_truth(path: Path, method: str) -> tuple[dict[tuple[str, int], list[int]], dict[int, int]]:
    truth: dict[tuple[str, int], list[int]] = {}
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != method:
                continue
            qno = int(row["query_no"])
            truth[(row["filter_name"], qno)] = [int(x) for x in row["exact_filtered_topk_ids"].split(",") if x]
            query_by_no[qno] = int(row["query_id"])
    return truth, query_by_no


def load_query_vectors_from_db(cur, query_ids: list[int]) -> dict[int, np.ndarray]:
    if not query_ids:
        return {}

    def parse_embedding(value: object) -> np.ndarray:
        if isinstance(value, str):
            v = value.strip()
            if v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            return np.fromstring(v, sep=",", dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    cur.execute(
        f"""
        SELECT id, embedding
        FROM {TABLE}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    rows = cur.fetchall()
    vectors = {int(row[0]): parse_embedding(row[1]) for row in rows}
    missing = [qid for qid in query_ids if qid not in vectors]
    if missing:
        raise RuntimeError(f"missing embeddings for {len(missing)} query ids from DB")
    return vectors


def vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


def ensure_schema(cur, dim: int, drop: bool) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if drop:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id bigint PRIMARY KEY,
            rating double precision,
            verified_purchase boolean,
            helpful_vote int,
            review_text_len int,
            main_category text,
            price double precision,
            has_price boolean,
            item_rating_number int,
            embedding vector({dim})
        )
        """
    )


def import_data(args, cur, xb: np.memmap, rows: int) -> None:
    cur.execute(f"SELECT count(*) FROM {TABLE}")
    current = int(cur.fetchone()[0])
    target_rows = min(rows, args.rows)
    if current >= target_rows:
        print(f"import already complete count={current} target={target_rows}", flush=True)
        return
    if current and not args.resume:
        raise RuntimeError(f"table has {current} rows; pass --resume or --drop")

    start = current
    print(f"copy start={start} target={target_rows}", flush=True)
    t0 = time.perf_counter()
    last = t0
    imported = start
    with args.csv.open(newline="") as f:
        reader = csv.DictReader(f)
        with cur.copy(
            f"""
            COPY {TABLE}
            (id, rating, verified_purchase, helpful_vote, review_text_len, main_category,
             price, has_price, item_rating_number, embedding)
            FROM STDIN
            """
        ) as copy:
            for i, row in enumerate(reader):
                if i < start:
                    continue
                if i >= target_rows:
                    break
                price = row["price"] if row["has_price"] == "True" and row["price"] else "0"
                line = "\t".join(
                    [
                        str(i),
                        row["rating"],
                        "t" if row["verified_purchase"] == "True" else "f",
                        row["helpful_vote"],
                        row["review_text_len"],
                        row["main_category"].replace("\\", "\\\\").replace("\t", " "),
                        price,
                        "t" if row["has_price"] == "True" else "f",
                        str(int(float(row["item_rating_number"]))),
                        vector_literal(np.asarray(xb[i], dtype=np.float32)),
                    ]
                )
                copy.write((line + "\n").encode("utf-8"))
                imported = i + 1
                now = time.perf_counter()
                if now - last >= args.progress_seconds:
                    rate = (imported - start) / max(now - t0, 1e-9)
                    print(f"copied={imported}/{target_rows} rate={rate:.1f} rows/s elapsed={(now-t0)/60:.1f}m", flush=True)
                    last = now
    print(f"copy done rows={imported} elapsed_min={(time.perf_counter()-t0)/60:.2f}", flush=True)


def ensure_indexes(cur, args) -> None:
    cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
    cur.execute(f"SET max_parallel_maintenance_workers = {int(args.max_parallel_maintenance_workers)}")
    indexes = [
        ("rating", "rating"),
        ("price_rating", "has_price, price, rating"),
        ("item_rating_number", "item_rating_number"),
        ("helpful_vote", "helpful_vote"),
        ("review_text_len", "review_text_len"),
        ("main_category_rating", "main_category, rating"),
        ("main_category_helpful", "main_category, helpful_vote"),
        ("main_category_review_len", "main_category, review_text_len"),
    ]
    for name, cols in indexes:
        print(f"index metadata {name}", flush=True)
        cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_{name}_idx ON {TABLE} ({cols})")
    print("analyze metadata", flush=True)
    cur.execute(f"ANALYZE {TABLE}")
    if args.skip_hnsw_index:
        return
    print("index pgvector hnsw embedding", flush=True)
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw_idx
        ON {TABLE}
        USING hnsw (embedding vector_l2_ops)
        WITH (m = {int(args.hnsw_m)}, ef_construction = {int(args.ef_construction)})
        """
    )
    print("analyze after hnsw", flush=True)
    cur.execute(f"ANALYZE {TABLE}")


def count_filter(cur, predicate: str) -> tuple[int, float]:
    def run():
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE {predicate}")
        return int(cur.fetchone()[0])

    return timed(run)


def count_filter_only(cur, predicate: str) -> float:
    _, ms = count_filter(cur, predicate)
    return ms


def pgvector_query(cur, predicate: str, query: np.ndarray, k: int) -> tuple[list[int], float]:
    q = vector_literal(query)

    def run():
        cur.execute(
            f"""
            SELECT id
            FROM {TABLE}
            WHERE {predicate}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (q,),
        )
        return [int(row[0]) for row in cur.fetchall()]

    return timed(run)


def pgvector_unfiltered_query(cur, query: np.ndarray, k: int) -> tuple[list[int], float]:
    q = vector_literal(query)

    def run():
        cur.execute(
            f"""
            SELECT id
            FROM {TABLE}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (q,),
        )
        return [int(row[0]) for row in cur.fetchall()]

    return timed(run)


def run_plan(cur, sql: str, params: tuple[object, ...]) -> tuple[dict, float]:
    def run():
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
        plan_text = cur.fetchone()[0]
        if isinstance(plan_text, str):
            return json.loads(plan_text)[0]
        return plan_text[0]

    return timed(run)


def load_profile(cur) -> dict:
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile_text = cur.fetchone()[0]
    if profile_text is None:
        return {"valid": False, "vector_search_ms": 0.0, "visited_tuples": 0.0, "returned_tuples": 0.0}
    if isinstance(profile_text, dict):
        profile = profile_text
    else:
        profile = json.loads(profile_text)
    return {
        "valid": bool(profile.get("valid", False)),
        "vector_search_ms": float(profile.get("vector_search_ms", 0.0)),
        "visited_tuples": float(profile.get("visited_tuples", 0.0)),
        "returned_tuples": float(profile.get("returned_tuples", 0.0)),
    }


def load_qual_profile(cur) -> dict:
    cur.execute("SELECT hybrid_qual_profile_last()")
    profile_text = cur.fetchone()[0]
    if profile_text is None:
        return {"qual_ms": 0.0, "qual_calls": 0.0, "qual_true": 0.0, "qual_false": 0.0}
    if isinstance(profile_text, dict):
        profile = profile_text
    else:
        profile = json.loads(profile_text)
    qual_true = 0.0
    qual_false = 0.0
    for entry in profile.get("entries", []) or []:
        qual_true += float(entry.get("true", 0.0))
        qual_false += float(entry.get("false", 0.0))
    return {
        "qual_ms": float(profile.get("qual_ms", 0.0)),
        "qual_calls": float(profile.get("qual_calls", 0.0)),
        "qual_true": qual_true,
        "qual_false": qual_false,
        "seen_plan_nodes": float(profile.get("seen_plan_nodes", 0.0)),
        "seen_qual_nodes": float(profile.get("seen_qual_nodes", 0.0)),
    }


def ensure_profile_functions(cur) -> bool:
    cur.execute("CREATE EXTENSION IF NOT EXISTS hybrid_qual_profile")
    cur.execute("SELECT to_regprocedure('vector_hnsw_last_scan_profile()') IS NOT NULL")
    has_hnsw_profile = bool(cur.fetchone()[0])
    cur.execute("SELECT to_regprocedure('hybrid_qual_profile_last()') IS NOT NULL")
    has_qual_profile = bool(cur.fetchone()[0])
    return has_hnsw_profile and has_qual_profile


def pgvector_query_with_profile(cur, predicate: str, query: np.ndarray, k: int) -> tuple[list[int], float, dict, dict]:
    q = vector_literal(query)

    def run():
        cur.execute(
            f"""
            SELECT id
            FROM {TABLE}
            WHERE {predicate}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (q,),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
        return ids

    # Reset profile immediately before query to avoid stale values.
    cur.execute("SELECT hybrid_qual_profile_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    ids, query_ms = timed(run)
    hnsw_profile = load_profile(cur)
    qual_profile = load_qual_profile(cur)
    return ids, query_ms, hnsw_profile, qual_profile


def pgvector_post_filter_with_profile(cur, predicate: str, query: np.ndarray, k: int, post_overfetch: int) -> tuple[
    list[int], float, dict, float, float
]:
    q = vector_literal(query)
    k_limit = int(post_overfetch)

    def run_candidates():
        cur.execute(
            f"""
            SELECT id
            FROM {TABLE}
            ORDER BY embedding <-> %s::vector
            LIMIT {k_limit}
            """,
            (q,),
        )
        return [int(row[0]) for row in cur.fetchall()]

    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    candidates, query_ms = timed(run_candidates)
    profile = load_profile(cur)

    if not candidates:
        return [], query_ms, profile, 0.0, 0.0

    def run_filter():
        filter_sql = (
            f"SELECT t.id "
            f"FROM unnest(%s::bigint[]) WITH ORDINALITY AS u(id, ord) "
            f"JOIN {TABLE} t ON t.id = u.id "
            f"WHERE {predicate} "
            f"ORDER BY u.ord "
            f"LIMIT {int(k)}"
        )
        cur.execute(filter_sql, (candidates,))
        return [int(row[0]) for row in cur.fetchall()]

    filtered, filter_ms = timed(run_filter)
    filtered_ids = filtered
    total_ms = query_ms + filter_ms
    vector_ms = float(profile.get("vector_search_ms", 0.0))
    return filtered_ids, total_ms, profile, vector_ms, filter_ms


def flatten_plan(node: dict, flat: list[dict]) -> None:
    flat.append(node)
    for child in node.get("Plans", []) or []:
        flatten_plan(child, flat)


def extract_scan_metrics(plan_root: dict, table_name: str) -> tuple[float, int, str, int, float]:
    flat: list[dict] = []
    flatten_plan(plan_root, flat)

    scan_nodes = [
        node
        for node in flat
        if (
            node.get("Node Type") in {"Index Scan", "Index Only Scan"}
            or "Scan" in node.get("Node Type", "")
            or node.get("Node Type") == "Custom Scan"
        )
    ]

    target = None
    for node in scan_nodes:
        if node.get("Relation Name") == table_name:
            target = node
            break
    if target is None and scan_nodes:
        target = max(scan_nodes, key=lambda n: float(n.get("Actual Total Time", 0.0)))
    if target is None:
        return 0.0, 0, "", 0, 0.0

    node_time = float(target.get("Actual Total Time", 0.0))
    rows_removed = int(target.get("Rows Removed by Filter", 0) or 0)
    filter_clause = target.get("Filter", "") or ""
    actual_rows = int(target.get("Actual Rows", 0))
    plans_time = float(target.get("Actual Total Time", 0.0))
    return node_time, rows_removed, filter_clause, actual_rows, plans_time


def run_queries(args, cur, xb: np.memmap | np.ndarray | None = None, query_vector_lookup: dict[int, np.ndarray] | None = None) -> None:
    truth_method = "pre_filter_exact" if args.search_mode == "pre_filter" else "post_filtering"
    truth, query_by_no = load_truth(args.truth_csv, truth_method)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    rows_out: list[dict[str, object]] = []
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    if args.iterative_scan:
        cur.execute(f"SET hnsw.iterative_scan = '{args.iterative_scan}'")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET enable_seqscan = {'off' if args.disable_seqscan else 'on'}")

    if not ensure_profile_functions(cur):
        raise RuntimeError(
            "Required profile functions are not available. "
            "Install instrumented pgvector and hybrid_qual_profile in PostgreSQL."
        )

    filter_counts: dict[str, tuple[int, float]] = {}
    selected = set(args.filter_names or [])
    selected_filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    method_label = "pgvector_hybrid_post_filter" if args.search_mode == "pre_filter" else "external_post_filter_sim"
    for filter_name, _, predicate in selected_filters:
        filter_counts[filter_name] = count_filter(cur, predicate)
    print(f"filter_counts={filter_counts}", flush=True)

    for filter_name, target_rate, predicate in selected_filters:
        sql_rows, filter_count_ms_once = filter_counts[filter_name]
        recalls: list[float] = []
        latencies: list[float] = []
        returned: list[int] = []
        for idx, qno in enumerate(query_nos, start=1):
            qid = query_by_no[qno]
            if query_vector_lookup is not None:
                qv = query_vector_lookup[qid]
            else:
                qv = np.asarray(xb[qid], dtype=np.float32)
            q_lit = vector_literal(qv)
            ids: list[int] = []
            query_latencies: list[float] = []
            vector_scan_ms: list[float] = []
            filter_scan_ms: list[float] = []
            vector_filter_ms: list[float] = []
            executor_non_hnsw_ms: list[float] = []
            qual_calls: list[float] = []
            qual_true: list[float] = []
            qual_false: list[float] = []
            visited_tuples: list[int] = []
            returned_tuples: list[int] = []
            explain_stats = None

            filtered_scan_ms = 0.0
            filtered_plan: dict | None = None
            explain_stats = None
            if args.explain and args.search_mode == "pre_filter":
                q_plan_query = f"""
                    SELECT id
                    FROM {TABLE}
                    WHERE {predicate}
                    ORDER BY embedding <-> %s::vector
                    LIMIT {int(args.k)}
                    """
                filtered_plan_root, _ = run_plan(cur, q_plan_query, (q_lit,))
                filtered_plan = filtered_plan_root["Plan"]
                filtered_scan_ms, _, _, _, _ = extract_scan_metrics(filtered_plan, TABLE)

            for _ in range(args.repeats):
                if args.search_mode == "pre_filter":
                    ids, query_ms, profile, qual_profile = pgvector_query_with_profile(cur, predicate, qv, args.k)
                    filter_ms = 0.0
                    vector_ms = float(profile.get("vector_search_ms", 0.0))
                    qual_ms = float(qual_profile.get("qual_ms", 0.0))
                    qual_calls.append(float(qual_profile.get("qual_calls", 0.0)))
                    qual_true.append(float(qual_profile.get("qual_true", 0.0)))
                    qual_false.append(float(qual_profile.get("qual_false", 0.0)))
                else:
                    ids, query_ms, profile, vector_ms, filter_ms = pgvector_post_filter_with_profile(
                        cur, predicate, qv, args.k, args.post_overfetch
                    )
                    qual_ms = filter_ms
                query_latencies.append(query_ms)
                vector_candidates = float(profile.get("visited_tuples", 0.0))
                vector_returned = float(profile.get("returned_tuples", 0.0))
                visited_tuples.append(vector_candidates)
                returned_tuples.append(vector_returned)

                vector_scan_ms.append(vector_ms)
                if args.search_mode == "pre_filter":
                    vector_filter_ms.append(qual_ms)
                    executor_non_hnsw_ms.append(max(query_ms - vector_ms, 0.0))
                else:
                    vector_filter_ms.append(filter_ms)
                    executor_non_hnsw_ms.append(filter_ms)

                if args.search_mode == "pre_filter" and args.explain and filtered_plan is not None and profile.get("valid"):
                    filter_scan_ms.append(max(filtered_scan_ms - float(profile.get("vector_search_ms", 0.0)), 0.0))
                else:
                    filter_scan_ms.append(qual_ms)

            if args.explain and args.search_mode == "pre_filter" and explain_stats is None:
                unfiltered_sql = f"""
                    SELECT id
                    FROM {TABLE}
                    ORDER BY embedding <-> %s::vector
                    LIMIT {int(args.k)}
                    """
                unfiltered_plan_root, _ = run_plan(cur, unfiltered_sql, (q_lit,))
                unfiltered_plan = unfiltered_plan_root["Plan"]
                unfiltered_scan_ms, _, _, unfiltered_rows, _ = extract_scan_metrics(unfiltered_plan, TABLE)

                explain_stats = {
                    "filtered_top_ms": float(filtered_plan["Actual Total Time"]),
                    "unfiltered_top_ms": float(unfiltered_plan["Actual Total Time"]),
                    "filtered_scan_ms": filtered_scan_ms,
                    "unfiltered_scan_ms": unfiltered_scan_ms,
                    "filtered_rows_removed_by_filter": filtered_plan.get("Rows Removed by Filter", 0),
                    "filtered_filter_clause": filtered_plan.get("Filter", ""),
                    "filtered_actual_rows": int(filtered_plan.get("Actual Rows", 0)),
                    "unfiltered_actual_rows": int(unfiltered_rows),
                    "filter_extra_top_ms": max(float(filtered_plan["Actual Total Time"]) - float(unfiltered_plan["Actual Total Time"]), 0.0),
                    "filter_extra_scan_ms": max(filtered_scan_ms - unfiltered_scan_ms, 0.0),
                }

            query_ms = statistics.mean(query_latencies)
            truth_ids = truth[(filter_name, qno)]
            recalls.append(recall_at_k(ids, truth_ids, args.k))
            latencies.append(query_ms)
            returned.append(len(ids))
            rows_out.append(
                {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "query_no": qno,
                    "query_id": qid,
                    "sql_rows": sql_rows,
                    "method": method_label,
                    "post_recall": recalls[-1],
                    "post_latency_ms": query_ms,
                    "filter_count_ms": filter_count_ms_once,
                    "filter_count_ms_per_query": filter_count_ms_once / len(query_nos) if len(query_nos) else 0.0,
                    "filter_count_ms_per_execution": filter_count_ms_once / (len(query_nos) * max(1, args.repeats)) if len(query_nos) else 0.0,
                    "post_vector_search_ms": statistics.mean(vector_scan_ms) if vector_scan_ms else query_ms,
                    "returned": returned[-1],
                    "repeats": args.repeats,
                    "ef_search": args.ef_search,
                    "iterative_scan": args.iterative_scan or "",
                    "max_scan_tuples": args.max_scan_tuples,
                    "scan_mem_multiplier": args.scan_mem_multiplier,
                    "query_sql_filter_ms": statistics.mean(vector_filter_ms) if vector_filter_ms else 0.0,
                    "query_sql_filter_calls": statistics.mean(qual_calls) if qual_calls else 0.0,
                    "query_sql_filter_true": statistics.mean(qual_true) if qual_true else 0.0,
                    "query_sql_filter_false": statistics.mean(qual_false) if qual_false else 0.0,
                    "query_executor_non_hnsw_ms": statistics.mean(executor_non_hnsw_ms)
                    if executor_non_hnsw_ms
                    else 0.0,
                    "query_vector_ms": statistics.mean(vector_scan_ms) if vector_scan_ms else query_ms,
                    "query_sql_filter_scan_ms": statistics.mean(filter_scan_ms) if filter_scan_ms else 0.0,
                    "query_vector_scan_ms": statistics.mean(vector_scan_ms) if vector_scan_ms else query_ms,
                    "filtered_top_ms": explain_stats["filtered_top_ms"] if explain_stats is not None else query_ms,
                    "unfiltered_top_ms": explain_stats["unfiltered_top_ms"] if explain_stats is not None else query_ms,
                    "filtered_scan_ms": explain_stats["filtered_scan_ms"] if explain_stats is not None else query_ms,
                    "unfiltered_scan_ms": explain_stats["unfiltered_scan_ms"] if explain_stats is not None else query_ms,
                    "filter_extra_top_ms": explain_stats["filter_extra_top_ms"] if explain_stats is not None else 0.0,
                    "filter_extra_scan_ms": explain_stats["filter_extra_scan_ms"] if explain_stats is not None else 0.0,
                    "filtered_rows_removed_by_filter": explain_stats["filtered_rows_removed_by_filter"]
                    if explain_stats is not None
                    else 0,
                    "filtered_actual_rows": explain_stats["filtered_actual_rows"] if explain_stats is not None else 0,
                    "unfiltered_actual_rows": explain_stats["unfiltered_actual_rows"] if explain_stats is not None else 0,
                    "hnsw_profile_visits": statistics.mean(visited_tuples) if visited_tuples else 0.0,
                    "hnsw_profile_returned": statistics.mean(returned_tuples) if returned_tuples else 0.0,
                }
            )
            if args.progress_queries and idx % args.progress_queries == 0:
                print(
                    f"progress filter={filter_name} queries={idx}/{len(query_nos)} "
                    f"latency_avg={statistics.mean(latencies):.2f}",
                    flush=True,
                )
            print(
                f"filter={target_rate} name={filter_name} rows={sql_rows} "
                f"recall={statistics.mean(recalls):.3f} latency={statistics.mean(latencies):.2f} "
                f"filter_count_ms={filter_count_ms_once:.2f} returned={statistics.mean(returned):.2f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    summary = args.out.with_name(args.out.stem + "_summary.csv")
    with summary.open("w", newline="") as f:
        fieldnames = [
            "method",
            "filter",
            "filter_name",
            "post_recall",
            "post_latency_ms",
            "filter_count_ms",
            "filter_count_ms_per_query",
            "filter_count_ms_per_execution",
            "sql_rows",
            "post_vector_search_ms",
            "query_sql_filter_ms",
            "query_sql_filter_calls",
            "query_sql_filter_true",
            "query_sql_filter_false",
            "query_executor_non_hnsw_ms",
            "query_vector_ms",
            "query_sql_filter_scan_ms",
            "query_vector_scan_ms",
            "hnsw_profile_visits",
            "hnsw_profile_returned",
            "returned_mean",
            "repeats",
            "ef_search",
            "iterative_scan",
            "max_scan_tuples",
            "scan_mem_multiplier",
        ]
        if args.explain:
            fieldnames += [
                "filtered_top_ms",
                "unfiltered_top_ms",
                "filtered_scan_ms",
                "unfiltered_scan_ms",
                "filter_extra_top_ms",
                "filter_extra_scan_ms",
                "filtered_rows_removed_by_filter",
                "filtered_actual_rows",
                "unfiltered_actual_rows",
            ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filter_name, target_rate, _ in selected_filters:
            items = [r for r in rows_out if r["filter_name"] == filter_name]
            writer.writerow(
                {
                    "method": method_label,
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "post_recall": statistics.mean(float(r["post_recall"]) for r in items),
                    "post_latency_ms": statistics.mean(float(r["post_latency_ms"]) for r in items),
                    "filter_count_ms": float(items[0]["filter_count_ms"]),
                    "filter_count_ms_per_query": statistics.mean(float(r["filter_count_ms_per_query"]) for r in items),
                    "filter_count_ms_per_execution": statistics.mean(
                        float(r["filter_count_ms_per_execution"]) for r in items
                    ),
                    "sql_rows": int(items[0]["sql_rows"]),
                    "post_vector_search_ms": statistics.mean(float(r["post_vector_search_ms"]) for r in items),
                    "query_sql_filter_ms": statistics.mean(float(r["query_sql_filter_ms"]) for r in items),
                    "query_sql_filter_calls": statistics.mean(float(r["query_sql_filter_calls"]) for r in items),
                    "query_sql_filter_true": statistics.mean(float(r["query_sql_filter_true"]) for r in items),
                    "query_sql_filter_false": statistics.mean(float(r["query_sql_filter_false"]) for r in items),
                    "query_executor_non_hnsw_ms": statistics.mean(float(r["query_executor_non_hnsw_ms"]) for r in items),
                    "query_vector_ms": statistics.mean(float(r["query_vector_ms"]) for r in items),
                    "query_sql_filter_scan_ms": statistics.mean(float(r["query_sql_filter_scan_ms"]) for r in items),
                    "query_vector_scan_ms": statistics.mean(float(r["query_vector_scan_ms"]) for r in items),
                    "hnsw_profile_visits": statistics.mean(float(r["hnsw_profile_visits"]) for r in items),
                    "hnsw_profile_returned": statistics.mean(float(r["hnsw_profile_returned"]) for r in items),
                    "returned_mean": statistics.mean(float(r["returned"]) for r in items),
                    "repeats": int(items[0]["repeats"]),
                    "ef_search": int(items[0]["ef_search"]),
                    "iterative_scan": items[0]["iterative_scan"],
                    "max_scan_tuples": int(items[0]["max_scan_tuples"]),
                    "scan_mem_multiplier": float(items[0]["scan_mem_multiplier"]),
                }
            )
        print(f"wrote {args.out}", flush=True)
        print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"))
    parser.add_argument(
        "--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin")
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_prefilter_10m_q100.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--max-scan-tuples", type=int, default=20000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=1.0)
    parser.add_argument("--iterative-scan", choices=["strict_order", "relaxed_order"])
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--maintenance-work-mem", default="16GB")
    parser.add_argument("--max-parallel-maintenance-workers", type=int, default=7)
    parser.add_argument("--query-vectors-from-db", action="store_true")
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-indexes", action="store_true")
    parser.add_argument("--skip-hnsw-index", action="store_true")
    parser.add_argument("--query-only", action="store_true")
    parser.add_argument("--disable-seqscan", action="store_true", default=True)
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--progress-seconds", type=int, default=60)
    parser.add_argument("--progress-queries", type=int, default=0)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument(
        "--search-mode",
        choices=["pre_filter", "post_filter"],
        default="pre_filter",
        help="pre_filter: WHERE predicate ORDER BY embedding <-> q LIMIT k; "
        "post_filter: ANN then SQL filter candidates",
    )
    parser.add_argument("--post-overfetch", type=int, default=1000, help="candidate overfetch for post_filter mode")
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    if args.query_vectors_from_db:
        xb = None
        rows = args.rows
        dim = 0
    else:
        xb, rows, dim = read_fbin_memmap(args.fbin, args.rows)
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_schema(cur, dim, args.drop)
            if not args.query_only and not args.skip_import:
                import_data(args, cur, xb, rows)
            if not args.query_only and not args.skip_indexes:
                ensure_indexes(cur, args)
            if args.query_vectors_from_db:
                query_data = []
                with open(args.truth_csv, "r") as f:
                    rows_truth = list(csv.DictReader(f))
                truth_query_method = (
                    "pre_filter_exact" if args.search_mode == "pre_filter" else "post_filtering"
                )
                truth_query_ids = {
                    int(row["query_no"]): int(row["query_id"])
                    for row in rows_truth
                    if row["method"] == truth_query_method
                }
                query_nos = sorted(truth_query_ids)[args.query_offset : args.query_offset + args.queries]
                query_ids = [truth_query_ids[q] for q in query_nos]
                if not query_ids:
                    raise RuntimeError(
                        "query-vectors-from-db requires truth csv with query ids; "
                        "add --query-only and ensure truth csv is present"
                    )
                query_vector_lookup = load_query_vectors_from_db(cur, query_ids)
                run_queries(args, cur, None, query_vector_lookup=query_vector_lookup)
            else:
                run_queries(args, cur, xb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
