from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from common_pg import pg_config_from_env, require_psycopg
from faiss_hnsw_sql_attribute_filter_10m import recall_at_k


TABLE = "amazon_grocery_reviews_10m_pgvector"
INDEX = f"{TABLE}_embedding_hnsw_idx"
SQLENS_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs"
SQLENS_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs_hnsw"
SIDE_TABLE = "amazon_external_payload_sidecar"
MEMBERSHIP_TABLE = "amazon_external_filter_membership"
CATALOG_TABLE = "amazon_external_filter_catalog"

PREDICATE_COLUMNS = [
    "item_rating_number",
    "review_text_len",
    "verified_purchase",
    "main_category",
    "helpful_vote",
    "has_price",
    "rating",
    "price",
]


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000.0


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def vector_literal(value: object) -> str:
    if isinstance(value, str):
        return value
    return "[" + ",".join(f"{float(x):.7g}" for x in value) + "]"


def alias_predicate(predicate: str, alias: str) -> str:
    out = predicate
    for col in sorted(PREDICATE_COLUMNS, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(col)}\b", f"{alias}.{col}", out)
    return out


def load_filters(path: Path, selected_names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            if selected_names and row["filter_name"] not in selected_names:
                continue
            rows.append(
                {
                    "filter_id": idx,
                    "target_rate": row["target_rate"],
                    "filter_name": row["filter_name"],
                    "predicate": row["predicate"],
                    "atoms": [part.strip() for part in row.get("atoms", "").split("||") if part.strip()],
                    "actual_selectivity": float(row["actual_pct"]) / 100.0,
                    "sql_rows": int(row["count"]),
                }
            )
    if not rows:
        raise RuntimeError(f"no filters selected from {path}")
    return rows


def selectivity_band(selectivity: float) -> str:
    if selectivity <= 0.0061:
        return "0.2-0.6%"
    if selectivity <= 0.0235:
        return "1-2%"
    if selectivity <= 0.096:
        return "5-10%"
    return "15-50%"


def load_truth(path: Path) -> tuple[dict[tuple[str, int], list[int]], dict[int, int]]:
    truth: dict[tuple[str, int], list[int]] = {}
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") != "pre_filter_exact":
                continue
            qno = int(row["query_no"])
            query_by_no[qno] = int(row["query_id"])
            truth[(row["filter_name"], qno)] = [
                int(x) for x in row["exact_filtered_topk_ids"].split(",") if x
            ]
    if not truth:
        raise RuntimeError(f"no pre_filter_exact truth rows in {path}")
    return truth, query_by_no


def fetch_json(cur, sql: str) -> dict[str, Any]:
    cur.execute(sql)
    value = cur.fetchone()[0]
    return json.loads(value) if isinstance(value, str) else dict(value)


def load_hnsw_profile(cur) -> dict[str, float]:
    profile = fetch_json(cur, "SELECT vector_hnsw_last_scan_profile()")
    out = {
        "valid": bool(profile.get("valid", False)),
        "vector_ms": float(profile.get("vector_search_ms", 0.0)),
        "visited": float(profile.get("visited_tuples", 0.0)),
        "returned": float(profile.get("returned_tuples", 0.0)),
    }
    for key in [
        "distance_compute_count",
        "guidance_checks",
        "guidance_skips",
        "page_access_batches",
        "page_access_candidates",
        "page_access_prefetches",
        "page_access_distinct_pages",
        "index_page_prefetches",
    ]:
        out[key] = float(profile.get(key, 0.0) or 0.0)
    return out


def load_qual_profile(cur) -> dict[str, float]:
    profile = fetch_json(cur, "SELECT hybrid_qual_profile_last()")
    true_count = 0.0
    false_count = 0.0
    for entry in profile.get("entries", []) or []:
        true_count += float(entry.get("true", 0.0))
        false_count += float(entry.get("false", 0.0))
    return {
        "qual_ms": float(profile.get("qual_ms", 0.0)),
        "qual_calls": float(profile.get("qual_calls", 0.0)),
        "qual_true": true_count,
        "qual_false": false_count,
    }


def explain_plan(cur, sql: str, params: tuple[Any, ...]) -> tuple[str, str]:
    cur.execute("EXPLAIN (FORMAT JSON, COSTS OFF) " + sql, params)
    plan = cur.fetchone()[0]
    if isinstance(plan, str):
        plan = json.loads(plan)
    nodes: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        desc = node.get("Node Type", "")
        if "Index Name" in node:
            desc += f":{node['Index Name']}"
        if "Relation Name" in node:
            desc += f":{node['Relation Name']}"
        if "Join Type" in node:
            desc += f":{node['Join Type']}"
        if "Filter" in node:
            desc += ":Filter"
        if "Order By" in node:
            desc += ":Order"
        if "Sort Key" in node:
            desc += ":Sort"
        nodes.append(desc)
        for child in node.get("Plans", []) or []:
            walk(child)

    walk(plan[0]["Plan"])
    plan_text = " > ".join(nodes)
    if "hnsw" in plan_text and "Nested Loop" in plan_text:
        return "hnsw_nested_loop_boundary", plan_text
    if "hnsw" in plan_text:
        return "hnsw_payload_filter", plan_text
    if "Sort" in plan_text:
        return "filter_then_exact_sort", plan_text
    return "other", plan_text


def query_sql(method: str, predicate: str, k: int) -> str:
    if method == "payload_direct":
        return f"""
            SELECT t.id
            FROM {TABLE} t
            WHERE {alias_predicate(predicate, "t")}
            ORDER BY t.embedding <-> %s::vector
            LIMIT {int(k)}
        """
    if method == "sidecar_exists":
        return f"""
            SELECT t.id
            FROM {TABLE} t
            WHERE EXISTS (
                SELECT 1
                FROM {SIDE_TABLE} s
                WHERE s.id = t.id
                  AND {alias_predicate(predicate, "s")}
            )
            ORDER BY t.embedding <-> %s::vector
            LIMIT {int(k)}
        """
    if method == "membership_exists":
        return f"""
            SELECT t.id
            FROM {TABLE} t
            WHERE EXISTS (
                SELECT 1
                FROM {MEMBERSHIP_TABLE} m
                WHERE m.filter_id = %s
                  AND m.id = t.id
            )
            ORDER BY t.embedding <-> %s::vector
            LIMIT {int(k)}
        """
    if method == "sqlens_sidecar":
        return f"""
            SELECT t.id
            FROM {SQLENS_TABLE} t
            WHERE EXISTS (
                SELECT 1
                FROM {SIDE_TABLE} s
                WHERE s.id = t.id
                  AND {alias_predicate(predicate, "s")}
            )
            ORDER BY t.embedding <-> %s::vector
            LIMIT {int(k)}
        """
    raise ValueError(method)


def query_params(method: str, filter_id: int, query_vector: str) -> tuple[Any, ...]:
    if method == "membership_exists":
        return (filter_id, query_vector)
    return (query_vector,)


def ensure_functions(cur) -> None:
    functions = [
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
    for sql in functions:
        try:
            cur.execute(sql)
        except Exception as exc:  # noqa: BLE001 - parallel scripts can race on pg_proc updates
            if "tuple concurrently updated" not in str(exc):
                raise
            cur.connection.rollback()
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def configure_common(cur, args: argparse.Namespace, method: str) -> None:
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    if args.iterative_scan == "off":
        cur.execute("SET hnsw.iterative_scan = off")
    else:
        cur.execute(f"SET hnsw.iterative_scan = '{args.iterative_scan}'")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(args.guided_collect_target)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(args.metadata_cache_mb)}")
    cur.execute(f"SET jit = {args.jit}")
    cur.execute("SET enable_sort = off" if args.force_hnsw else "SET enable_sort = on")
    if method == "sqlens_sidecar":
        cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
        cur.execute(f"SET hnsw.page_access = {args.d2_page_access}")
        cur.execute(f"SET hnsw.index_page_access = {args.d2_index_page_access}")
        cur.execute(f"SET hnsw.page_window = {int(args.d2_page_window)}")
        cur.execute(f"SET hnsw.page_prefetch_min_items = {int(args.d2_page_prefetch_min_items)}")
        cur.execute(f"SET hnsw.page_disable_after_no_merge = {int(args.d2_page_disable_after_no_merge)}")
    else:
        cur.execute("SET hnsw.filter_strategy = off")
        cur.execute("SET hnsw.page_access = off")
        cur.execute("SET hnsw.index_page_access = off")


def should_enable_sqlens(args: argparse.Namespace, filt: dict[str, Any]) -> tuple[bool, str]:
    if not args.sqlens_adaptive:
        return True, "forced"
    selectivity_pct = 100.0 * float(filt["actual_selectivity"])
    if selectivity_pct > float(args.sqlens_guidance_max_selectivity_pct):
        return False, "broad_d2_only"
    if len(filt["atoms"]) > int(args.sqlens_guidance_max_atoms):
        return False, "too_many_atoms_d2_only"
    return True, "guided"


def activate_sqlens(cur, args: argparse.Namespace, filt: dict[str, Any]) -> tuple[dict[str, Any], float]:
    def do_activate() -> dict[str, Any]:
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        enabled, route = should_enable_sqlens(args, filt)
        if not enabled:
            cur.execute("SET hnsw.filter_strategy = off")
            return {
                "active": False,
                "guidance_enabled": False,
                "guidance_route": route,
                "kind": "off",
                "atoms": 0,
                "last_cache_build_ms": 0.0,
                "last_cache_memory_bytes": 0,
                "fragment_cache_hits": 0,
                "fragment_cache_misses": 0,
                "fragment_store_hits": 0,
                "fragment_builds": 0,
                "composed_guide_hit": False,
            }
        cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
        cur.execute(
            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
            (args.sqlens_index, filt["atoms"], args.guidance_kind),
        )
        return fetch_json(cur, "SELECT vector_hnsw_guidance_profile()") | {
            "guidance_enabled": True,
            "guidance_route": route,
        }

    profile, activation_ms = timed(do_activate)
    return profile, activation_ms


def sqlens_signature(args: argparse.Namespace, filt: dict[str, Any]) -> tuple[str, str, tuple[str, ...]] | None:
    if not args.sqlens_reuse_active_guidance:
        return None
    enabled, _ = should_enable_sqlens(args, filt)
    if not enabled:
        return None
    return args.sqlens_index, args.guidance_kind, tuple(filt["atoms"])


def reuse_sqlens_profile(profile: dict[str, Any]) -> dict[str, Any]:
    reused = dict(profile)
    reused["guidance_route"] = "d3_reuse_active_guidance"
    reused["d3_active_guidance_reused"] = True
    return reused


def deactivate_sqlens(cur) -> None:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SET hnsw.filter_strategy = off")


def prewarm_sqlens(cur, args: argparse.Namespace, filters: list[dict[str, Any]]) -> None:
    if not args.prewarm_sqlens:
        return
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    configure_common(cur, args, "sqlens_sidecar")
    for filt in filters:
        enabled, _ = should_enable_sqlens(args, filt)
        if not enabled:
            continue
        cur.execute(
            "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
            (args.sqlens_index, filt["atoms"], args.guidance_kind),
        )
    deactivate_sqlens(cur)


def run_query(cur, sql: str, params: tuple[Any, ...]) -> tuple[list[int], float, dict[str, float], dict[str, float]]:
    def execute() -> list[int]:
        cur.execute("SELECT hybrid_qual_profile_reset()")
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
        cur.execute(sql, params)
        return [int(row[0]) for row in cur.fetchall()]

    ids, total_ms = timed(execute)
    return ids, total_ms, load_hnsw_profile(cur), load_qual_profile(cur)


def ensure_sidecar(cur, rebuild: bool) -> dict[str, Any]:
    if rebuild:
        cur.execute(f"DROP TABLE IF EXISTS {SIDE_TABLE}")
    cur.execute("SELECT to_regclass(%s)", (SIDE_TABLE,))
    exists = cur.fetchone()[0] is not None
    build_ms = 0.0
    if not exists:
        _, build_ms = timed(
            lambda: cur.execute(
                f"""
                CREATE UNLOGGED TABLE {SIDE_TABLE} AS
                SELECT id, rating, verified_purchase, helpful_vote, review_text_len,
                       main_category, price, has_price, item_rating_number
                FROM {TABLE}
                """
            )
        )
        cur.execute(f"CREATE UNIQUE INDEX {SIDE_TABLE}_id_idx ON {SIDE_TABLE}(id)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_helpful_vote_idx ON {SIDE_TABLE}(helpful_vote)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_item_rating_number_idx ON {SIDE_TABLE}(item_rating_number)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_rating_idx ON {SIDE_TABLE}(rating)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_review_text_len_idx ON {SIDE_TABLE}(review_text_len)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_price_rating_idx ON {SIDE_TABLE}(has_price, price, rating)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_category_rating_idx ON {SIDE_TABLE}(main_category, rating)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_category_helpful_idx ON {SIDE_TABLE}(main_category, helpful_vote)")
        cur.execute(f"CREATE INDEX {SIDE_TABLE}_category_review_len_idx ON {SIDE_TABLE}(main_category, review_text_len)")
        cur.execute(f"ANALYZE {SIDE_TABLE}")
    cur.execute(f"SELECT count(*), pg_total_relation_size(%s::regclass) FROM {SIDE_TABLE}", (SIDE_TABLE,))
    count, bytes_ = cur.fetchone()
    return {"rows": int(count), "bytes": int(bytes_), "build_ms": build_ms, "rebuilt": not exists or rebuild}


def ensure_membership(cur, filters: list[dict[str, Any]], rebuild: bool) -> dict[str, Any]:
    if rebuild:
        cur.execute(f"DROP TABLE IF EXISTS {MEMBERSHIP_TABLE}")
        cur.execute(f"DROP TABLE IF EXISTS {CATALOG_TABLE}")
    cur.execute("SELECT to_regclass(%s)", (MEMBERSHIP_TABLE,))
    exists = cur.fetchone()[0] is not None
    build_ms = 0.0
    if not exists:
        t0 = time.perf_counter()
        cur.execute(
            f"""
            CREATE UNLOGGED TABLE {CATALOG_TABLE} (
                filter_id integer PRIMARY KEY,
                filter_name text NOT NULL UNIQUE,
                target_rate text NOT NULL,
                predicate text NOT NULL,
                actual_selectivity double precision NOT NULL,
                sql_rows bigint NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE UNLOGGED TABLE {MEMBERSHIP_TABLE} (
                filter_id integer NOT NULL,
                id bigint NOT NULL
            )
            """
        )
        for row in filters:
            cur.execute(
                f"""
                INSERT INTO {CATALOG_TABLE}
                (filter_id, filter_name, target_rate, predicate, actual_selectivity, sql_rows)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    row["filter_id"],
                    row["filter_name"],
                    row["target_rate"],
                    row["predicate"],
                    row["actual_selectivity"],
                    row["sql_rows"],
                ),
            )
            cur.execute(
                f"""
                INSERT INTO {MEMBERSHIP_TABLE}
                SELECT %s, id
                FROM {TABLE}
                WHERE {row["predicate"]}
                """,
                (row["filter_id"],),
            )
            print(
                f"prepared membership {row['filter_name']} rows={row['sql_rows']}",
                flush=True,
            )
        cur.execute(f"CREATE INDEX {MEMBERSHIP_TABLE}_filter_id_id_idx ON {MEMBERSHIP_TABLE}(filter_id, id)")
        cur.execute(f"CREATE INDEX {MEMBERSHIP_TABLE}_id_filter_id_idx ON {MEMBERSHIP_TABLE}(id, filter_id)")
        cur.execute(f"ANALYZE {CATALOG_TABLE}")
        cur.execute(f"ANALYZE {MEMBERSHIP_TABLE}")
        build_ms = (time.perf_counter() - t0) * 1000.0
    cur.execute(
        f"""
        SELECT count(*), pg_total_relation_size(%s::regclass)
        FROM {MEMBERSHIP_TABLE}
        """,
        (MEMBERSHIP_TABLE,),
    )
    count, bytes_ = cur.fetchone()
    return {"rows": int(count), "bytes": int(bytes_), "build_ms": build_ms, "rebuilt": not exists or rebuild}


def load_query_vectors(cur, query_ids: list[int]) -> dict[int, str]:
    cur.execute(
        f"""
        SELECT id, embedding
        FROM {TABLE}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    vectors = {int(row[0]): vector_literal(row[1]) for row in cur.fetchall()}
    missing = [qid for qid in query_ids if qid not in vectors]
    if missing:
        raise RuntimeError(f"missing {len(missing)} query vectors")
    return vectors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["method"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (filter_name, method), items in groups.items():
        total = [float(r["total_ms"]) for r in items]
        activation = [float(r.get("activation_ms", 0.0)) for r in items]
        query = [float(r.get("query_ms", r["total_ms"])) for r in items]
        hnsw = [float(r["hnsw_vector_ms"]) for r in items]
        boundary = [float(r["boundary_exec_ms"]) for r in items]
        checks = statistics.fmean(float(r.get("guidance_checks", 0.0)) for r in items)
        skips = statistics.fmean(float(r.get("guidance_skips", 0.0)) for r in items)
        out.append(
            {
                "target_rate": items[0]["target_rate"],
                "filter_name": filter_name,
                "selectivity_band": items[0].get("selectivity_band", selectivity_band(float(items[0]["actual_selectivity"]))),
                "actual_selectivity": items[0]["actual_selectivity"],
                "sql_rows": items[0]["sql_rows"],
                "method": method,
                "plan_class": items[0]["plan_class"],
                "queries": len({int(r["query_no"]) for r in items}),
                "samples": len(items),
                "repeats": items[0]["repeats"],
                "recall_mean": statistics.fmean(float(r["recall"]) for r in items),
                "total_mean_ms": statistics.fmean(total),
                "total_p95_ms": pct(total, 0.95),
                "activation_mean_ms": statistics.fmean(activation),
                "query_mean_ms": statistics.fmean(query),
                "hnsw_vector_mean_ms": statistics.fmean(hnsw),
                "boundary_exec_mean_ms": statistics.fmean(boundary),
                "qual_mean_ms": statistics.fmean(float(r["qual_ms"]) for r in items),
                "qual_calls_mean": statistics.fmean(float(r["qual_calls"]) for r in items),
                "hnsw_visited_mean": statistics.fmean(float(r["hnsw_visited"]) for r in items),
                "hnsw_returned_mean": statistics.fmean(float(r["hnsw_returned"]) for r in items),
                "guidance_enabled_rate": statistics.fmean(float(bool(r.get("guidance_enabled", False))) for r in items),
                "guidance_checks_mean": checks,
                "guidance_skips_mean": skips,
                "guidance_skip_rate": skips / checks if checks else 0.0,
                "fragment_cache_hits_mean": statistics.fmean(float(r.get("fragment_cache_hits", 0.0)) for r in items),
                "fragment_store_hits_mean": statistics.fmean(float(r.get("fragment_store_hits", 0.0)) for r in items),
                "fragment_builds_mean": statistics.fmean(float(r.get("fragment_builds", 0.0)) for r in items),
                "d3_reuse_rate": statistics.fmean(float(bool(r.get("d3_active_guidance_reused", False))) for r in items),
                "cache_resident_bytes_max": max((int(r.get("cache_resident_bytes", 0)) for r in items), default=0),
                "page_access_batches_mean": statistics.fmean(float(r.get("page_access_batches", 0.0)) for r in items),
                "page_access_prefetches_mean": statistics.fmean(float(r.get("page_access_prefetches", 0.0)) for r in items),
                "page_access_distinct_pages_mean": statistics.fmean(float(r.get("page_access_distinct_pages", 0.0)) for r in items),
                "returned_mean": statistics.fmean(float(r["returned"]) for r in items),
            }
        )

    order = {
        str(row["filter_name"]): idx
        for idx, row in enumerate(sorted(rows, key=lambda r: float(r["actual_selectivity"]), reverse=True))
    }
    method_order = {"payload_direct": 0, "sidecar_exists": 1, "membership_exists": 2, "sqlens_sidecar": 3}
    out.sort(key=lambda r: (order.get(str(r["filter_name"]), 999), method_order.get(str(r["method"]), 99)))

    payload_by_filter = {
        str(r["filter_name"]): float(r["total_mean_ms"])
        for r in out
        if r["method"] == "payload_direct"
    }
    for row in out:
        base = payload_by_filter.get(str(row["filter_name"]), 0.0)
        val = float(row["total_mean_ms"])
        row["speedup_vs_payload"] = base / val if base and val else 0.0
        row["slowdown_vs_payload"] = val / base if base and val else 0.0

    sidecar_by_filter = {
        str(r["filter_name"]): float(r["total_mean_ms"])
        for r in out
        if r["method"] == "sidecar_exists"
    }
    for row in out:
        sidecar = sidecar_by_filter.get(str(row["filter_name"]), 0.0)
        payload = payload_by_filter.get(str(row["filter_name"]), 0.0)
        val = float(row["total_mean_ms"])
        row["speedup_vs_sidecar"] = sidecar / val if sidecar and val else 0.0
        row["gap_closed_vs_payload"] = (sidecar - val) / (sidecar - payload) if sidecar > payload else 0.0
    return out


def summarize_bands(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_filter: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        by_filter.setdefault(str(row["filter_name"]), {})[str(row["method"])] = row

    grouped: dict[str, list[dict[str, Any]]] = {}
    for methods in by_filter.values():
        if not {"payload_direct", "sidecar_exists", "sqlens_sidecar"}.issubset(methods):
            continue
        grouped.setdefault(str(methods["payload_direct"]["selectivity_band"]), []).append(methods)

    order = {"15-50%": 0, "5-10%": 1, "1-2%": 2, "0.2-0.6%": 3}
    out: list[dict[str, Any]] = []
    for band, items in sorted(grouped.items(), key=lambda item: order.get(item[0], 99)):
        payload_vals = [float(m["payload_direct"]["total_mean_ms"]) for m in items]
        sidecar_vals = [float(m["sidecar_exists"]["total_mean_ms"]) for m in items]
        sqlens_vals = [float(m["sqlens_sidecar"]["total_mean_ms"]) for m in items]
        recall_vals = [float(m["sqlens_sidecar"]["recall_mean"]) for m in items]
        selectivity_vals = [100.0 * float(m["payload_direct"]["actual_selectivity"]) for m in items]
        payload = statistics.fmean(payload_vals)
        sidecar = statistics.fmean(sidecar_vals)
        sqlens = statistics.fmean(sqlens_vals)
        out.append(
            {
                "selectivity_band": band,
                "filters": len(items),
                "actual_selectivity_min_pct": min(selectivity_vals),
                "actual_selectivity_max_pct": max(selectivity_vals),
                "payload_lower_bound_mean_ms": payload,
                "stock_sql_boundary_mean_ms": sidecar,
                "sqlens_mean_ms": sqlens,
                "speedup_vs_stock_sql": sidecar / sqlens if sqlens else 0.0,
                "payload_to_stock_slowdown": sidecar / payload if payload else 0.0,
                "sqlens_to_payload_slowdown": sqlens / payload if payload else 0.0,
                "gap_closed_vs_payload": (sidecar - sqlens) / (sidecar - payload) if sidecar > payload else 0.0,
                "recall_mean": statistics.fmean(recall_vals),
                "guidance_enabled_rate": statistics.fmean(float(m["sqlens_sidecar"]["guidance_enabled_rate"]) for m in items),
                "guidance_skip_rate": statistics.fmean(float(m["sqlens_sidecar"]["guidance_skip_rate"]) for m in items),
                "hnsw_visited_stock": statistics.fmean(float(m["sidecar_exists"]["hnsw_visited_mean"]) for m in items),
                "hnsw_visited_sqlens": statistics.fmean(float(m["sqlens_sidecar"]["hnsw_visited_mean"]) for m in items),
                "qual_calls_stock": statistics.fmean(float(m["sidecar_exists"]["qual_calls_mean"]) for m in items),
                "qual_calls_sqlens": statistics.fmean(float(m["sqlens_sidecar"]["qual_calls_mean"]) for m in items),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon-10M payload vs SQL-derived filter boundary benchmark.")
    parser.add_argument("--filters-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_selectivity14_diverse_filters_20260715.csv"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_selectivity14_diverse_truth_q100_20260715.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/amazon_external_payload_boundary_q100_r5_20260717.csv"))
    parser.add_argument("--meta-out", type=Path)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["payload_direct", "sidecar_exists", "sqlens_sidecar"],
        choices=["payload_direct", "sidecar_exists", "membership_exists", "sqlens_sidecar"],
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--guided-collect-target", type=int, default=1000)
    parser.add_argument("--metadata-cache-mb", type=int, default=1024)
    parser.add_argument("--guidance-filter-strategy", default="guided_collect", choices=["guided_collect", "acorn1"])
    parser.add_argument("--guidance-kind", default="bloom", choices=["bloom", "page", "exact"])
    parser.add_argument("--sqlens-index", default=SQLENS_INDEX)
    parser.add_argument("--prewarm-sqlens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sqlens-reuse-active-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sqlens-adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sqlens-guidance-max-selectivity-pct", type=float, default=10.0)
    parser.add_argument("--sqlens-guidance-max-atoms", type=int, default=64)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["strict_order", "relaxed_order", "off"])
    parser.add_argument("--jit", choices=["on", "off"], default="off")
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-sidecar", action="store_true")
    parser.add_argument("--rebuild-membership", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--interleave-methods", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    selected = set(args.filter_names or [])
    filters = load_filters(args.filters_csv, selected)
    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    query_ids = [query_by_no[qno] for qno in query_nos]

    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "table": TABLE,
        "sqlens_table": SQLENS_TABLE,
        "sqlens_index": args.sqlens_index,
        "side_table": SIDE_TABLE,
        "membership_table": MEMBERSHIP_TABLE,
        "filters_csv": str(args.filters_csv),
        "truth_csv": str(args.truth_csv),
        "queries": len(query_nos),
        "repeats": args.repeats,
        "ef_search": args.ef_search,
        "max_scan_tuples": args.max_scan_tuples,
        "scan_mem_multiplier": args.scan_mem_multiplier,
        "guided_collect_target": args.guided_collect_target,
        "metadata_cache_mb": args.metadata_cache_mb,
        "guidance_filter_strategy": args.guidance_filter_strategy,
        "guidance_kind": args.guidance_kind,
        "sqlens_adaptive": args.sqlens_adaptive,
        "sqlens_guidance_max_selectivity_pct": args.sqlens_guidance_max_selectivity_pct,
        "sqlens_guidance_max_atoms": args.sqlens_guidance_max_atoms,
        "sqlens_reuse_active_guidance": args.sqlens_reuse_active_guidance,
        "prewarm_sqlens": args.prewarm_sqlens,
        "d2_page_access": args.d2_page_access,
        "d2_index_page_access": args.d2_index_page_access,
        "iterative_scan": args.iterative_scan,
        "methods": args.methods,
        "interleave_methods": args.interleave_methods,
    }

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS hybrid_qual_profile")
            ensure_functions(cur)
            cur.execute("SELECT to_regprocedure('vector_hnsw_last_scan_profile()'), to_regprocedure('hybrid_qual_profile_last()')")
            if not all(cur.fetchone()):
                raise RuntimeError("profile functions are not available")

            if not args.skip_prepare:
                if "sidecar_exists" in args.methods or "sqlens_sidecar" in args.methods:
                    meta["sidecar"] = ensure_sidecar(cur, args.rebuild_sidecar)
                if "membership_exists" in args.methods:
                    meta["membership"] = ensure_membership(cur, filters, args.rebuild_membership)

            query_vectors = load_query_vectors(cur, query_ids)
            if "sqlens_sidecar" in args.methods:
                prewarm_sqlens(cur, args, filters)

            plan_cache: dict[tuple[str, str], tuple[str, str]] = {}
            active_sqlens_signature: tuple[str, str, tuple[str, ...]] | None = None
            active_sqlens_profile: dict[str, Any] = {}

            def execute_one(
                method: str,
                filt: dict[str, Any],
                sql: str,
                qno: int,
                qid: int,
                repeat: int,
                exact: list[int],
            ) -> dict[str, Any]:
                nonlocal active_sqlens_signature, active_sqlens_profile

                configure_common(cur, args, method)
                activation_profile: dict[str, Any] = {}
                activation_ms = 0.0
                if method == "sqlens_sidecar":
                    signature = sqlens_signature(args, filt)
                    if signature is not None and signature == active_sqlens_signature:
                        activation_profile = reuse_sqlens_profile(active_sqlens_profile)
                        cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
                    else:
                        activation_profile, activation_ms = activate_sqlens(cur, args, filt)
                        active_sqlens_signature = signature
                        active_sqlens_profile = dict(activation_profile) if signature is not None else {}
                else:
                    deactivate_sqlens(cur)
                    active_sqlens_signature = None
                    active_sqlens_profile = {}

                ids, query_ms, hnsw, qual = run_query(
                    cur,
                    sql,
                    query_params(method, int(filt["filter_id"]), query_vectors[qid]),
                )
                total_ms = activation_ms + query_ms
                recall = recall_at_k(ids, exact, args.k)
                boundary_ms = max(query_ms - float(hnsw["vector_ms"]), 0.0)
                try:
                    cache_profile = fetch_json(cur, "SELECT vector_hnsw_metadata_cache_profile()")
                except Exception:
                    cache_profile = {}
                checks = float(hnsw.get("guidance_checks", 0.0))
                skips = float(hnsw.get("guidance_skips", 0.0))
                return {
                    "target_rate": filt["target_rate"],
                    "filter_name": filt["filter_name"],
                    "selectivity_band": selectivity_band(float(filt["actual_selectivity"])),
                    "actual_selectivity": filt["actual_selectivity"],
                    "sql_rows": filt["sql_rows"],
                    "method": method,
                    "plan_class": plan_cache[(str(filt["filter_name"]), method)][0],
                    "plan_text": plan_cache[(str(filt["filter_name"]), method)][1],
                    "query_no": qno,
                    "query_id": qid,
                    "repeat": repeat,
                    "recall": recall,
                    "total_ms": total_ms,
                    "activation_ms": activation_ms,
                    "query_ms": query_ms,
                    "hnsw_vector_ms": float(hnsw["vector_ms"]),
                    "boundary_exec_ms": boundary_ms,
                    "qual_ms": float(qual["qual_ms"]),
                    "qual_calls": float(qual["qual_calls"]),
                    "qual_true": float(qual["qual_true"]),
                    "qual_false": float(qual["qual_false"]),
                    "hnsw_visited": float(hnsw["visited"]),
                    "hnsw_returned": float(hnsw["returned"]),
                    "distance_compute_count": float(hnsw.get("distance_compute_count", 0.0)),
                    "guidance_checks": checks,
                    "guidance_skips": skips,
                    "guidance_skip_rate": skips / checks if checks else 0.0,
                    "guidance_enabled": bool(activation_profile.get("guidance_enabled", False)),
                    "guidance_route": str(activation_profile.get("guidance_route", "")),
                    "d3_active_guidance_reused": bool(activation_profile.get("d3_active_guidance_reused", False)),
                    "activation_build_ms": float(activation_profile.get("last_cache_build_ms", 0.0) or 0.0),
                    "activation_memory_bytes": int(activation_profile.get("last_cache_memory_bytes", 0) or 0),
                    "activation_rows": int(activation_profile.get("last_cache_rows", 0) or 0),
                    "fragment_cache_hits": int(activation_profile.get("fragment_cache_hits", 0) or 0),
                    "fragment_cache_misses": int(activation_profile.get("fragment_cache_misses", 0) or 0),
                    "fragment_store_hits": int(activation_profile.get("fragment_store_hits", 0) or 0),
                    "fragment_builds": int(activation_profile.get("fragment_builds", 0) or 0),
                    "composed_guide_hit": bool(activation_profile.get("composed_guide_hit", False)),
                    "cache_resident_bytes": int(cache_profile.get("resident_bytes", 0) or 0),
                    "cache_resident_entries": int(cache_profile.get("resident_entries", 0) or 0),
                    "cache_evictions": int(cache_profile.get("evictions", 0) or 0),
                    "page_access_batches": float(hnsw.get("page_access_batches", 0.0)),
                    "page_access_candidates": float(hnsw.get("page_access_candidates", 0.0)),
                    "page_access_prefetches": float(hnsw.get("page_access_prefetches", 0.0)),
                    "page_access_distinct_pages": float(hnsw.get("page_access_distinct_pages", 0.0)),
                    "index_page_prefetches": float(hnsw.get("index_page_prefetches", 0.0)),
                    "returned": len(ids),
                    "ids": ",".join(str(x) for x in ids),
                    "repeats": args.repeats,
                    "ef_search": args.ef_search,
                    "guided_collect_target": args.guided_collect_target,
                    "iterative_scan": args.iterative_scan,
                    "max_scan_tuples": args.max_scan_tuples,
                    "scan_mem_multiplier": args.scan_mem_multiplier,
                    "metadata_cache_mb": args.metadata_cache_mb,
                }

            for fidx, filt in enumerate(filters, start=1):
                method_sql: dict[str, str] = {}
                for method in args.methods:
                    configure_common(cur, args, method)
                    sql = query_sql(method, str(filt["predicate"]), args.k)
                    method_sql[method] = sql
                    first_params = query_params(method, int(filt["filter_id"]), query_vectors[query_ids[0]])
                    plan_cache[(str(filt["filter_name"]), method)] = explain_plan(cur, sql, first_params)

                case_total_by_method: dict[str, list[float]] = {method: [] for method in args.methods}
                case_recall_by_method: dict[str, list[float]] = {method: [] for method in args.methods}

                if args.interleave_methods:
                    for qidx, qno in enumerate(query_nos, start=1):
                        qid = query_by_no[qno]
                        exact = truth.get((str(filt["filter_name"]), qno))
                        if exact is None:
                            raise RuntimeError(f"missing truth for filter={filt['filter_name']} query_no={qno}")
                        for repeat in range(args.repeats):
                            shift = (qidx + repeat) % len(args.methods)
                            run_methods = args.methods[shift:] + args.methods[:shift]
                            for method in run_methods:
                                row = execute_one(
                                    method,
                                    filt,
                                    method_sql[method],
                                    qno,
                                    qid,
                                    repeat,
                                    exact,
                                )
                                rows.append(row)
                                case_total_by_method[method].append(float(row["total_ms"]))
                                case_recall_by_method[method].append(float(row["recall"]))
                else:
                    for method in args.methods:
                        for qidx, qno in enumerate(query_nos, start=1):
                            qid = query_by_no[qno]
                            exact = truth.get((str(filt["filter_name"]), qno))
                            if exact is None:
                                raise RuntimeError(f"missing truth for filter={filt['filter_name']} query_no={qno}")
                            for repeat in range(args.repeats):
                                row = execute_one(
                                    method,
                                    filt,
                                    method_sql[method],
                                    qno,
                                    qid,
                                    repeat,
                                    exact,
                                )
                                rows.append(row)
                                case_total_by_method[method].append(float(row["total_ms"]))
                                case_recall_by_method[method].append(float(row["recall"]))

                if args.progress_every:
                    parts = []
                    for method in args.methods:
                        parts.append(
                            f"{method}={statistics.fmean(case_total_by_method[method]):.2f}ms/"
                            f"r{statistics.fmean(case_recall_by_method[method]):.3f}"
                        )
                    print(
                        f"progress filter={filt['filter_name']} ({fidx}/{len(filters)}) "
                        f"samples_per_method={len(case_total_by_method[args.methods[0]])} "
                        + " ".join(parts),
                        flush=True,
                    )

    write_csv(args.out, rows)
    summary = summarize(rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_csv(summary_path, summary)
    band_summary = summarize_bands(summary)
    band_summary_path = args.out.with_name(args.out.stem + "_band_summary.csv")
    if band_summary:
        write_csv(band_summary_path, band_summary)
    meta_path = args.meta_out or args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    if band_summary:
        print(f"wrote {band_summary_path}", flush=True)
    print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
