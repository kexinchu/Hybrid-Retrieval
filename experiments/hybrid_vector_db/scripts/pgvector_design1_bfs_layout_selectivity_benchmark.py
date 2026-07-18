from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

try:
    from .common_pg import pg_config_from_env
    from .faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
    from .pgvector_predicate_guidance_benchmark import FILTER_ATOMS, load_truth
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env
    from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
    from pgvector_predicate_guidance_benchmark import FILTER_ATOMS, load_truth


SOURCE_TABLE = "amazon_grocery_reviews_10m_pgvector"
INSERTION_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_insert"
INSERTION_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_insert_hnsw"
BFS_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs"
BFS_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs_hnsw"
MODES = ["original", "design1_bloom", "design1_bloom_bfs_layout"]
SCALAR_INDEXES = [
    ("main_category_rating", "main_category, rating"),
    ("price_rating", "has_price, price, rating"),
    ("rating", "rating"),
    ("item_rating_number", "item_rating_number"),
    ("review_text_len", "review_text_len"),
    ("helpful_vote", "helpful_vote"),
    ("category_helpful", "main_category, helpful_vote"),
    ("category_review_len", "main_category, review_text_len"),
]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_verification_manifest(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> Path:
    cur.execute(
        "SELECT vector_sqlens_build_id(), current_setting('server_version'), "
        "coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), '')"
    )
    build_id, postgres_version, vector_version = cur.fetchone()
    relations: dict[str, dict[str, object]] = {}
    for relation in (
        args.source_table,
        args.insertion_table,
        args.insertion_index,
        args.bfs_table,
        args.bfs_index,
    ):
        cur.execute(
            "SELECT c.oid::bigint, c.relkind, c.reloptions, "
            "pg_relation_size(c.oid)::bigint FROM pg_class AS c "
            "WHERE c.oid = to_regclass(%s)",
            (relation,),
        )
        state = cur.fetchone()
        if state is None:
            raise RuntimeError(f"verification relation disappeared: {relation}")
        relations[relation] = {
            "oid": int(state[0]),
            "relkind": state[1],
            "reloptions": list(state[2] or []),
            "relation_bytes": int(state[3]),
        }
    recalls = [
        float(row["exact_recall_at_k"])
        for row in rows
        if row.get("exact_recall_at_k") not in (None, "")
    ]
    payload = {
        "artifact_valid": bool(rows) and bool(recalls),
        "sqlens_build_id": str(build_id),
        "postgres_version": str(postgres_version),
        "vector_extension_version": str(vector_version),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "truth_csv": sha256_file(args.truth_csv),
        },
        "build": {
            "action": (
                "prepare"
                if args.prepare_same_graph_layouts
                else "rebuild_indexes"
                if args.rebuild_same_graph_indexes_only
                else "verify_existing"
            ),
            "seed": int(args.hnsw_build_seed),
            "m": int(args.hnsw_m),
            "ef_construction": int(args.hnsw_ef_construction),
            "page_orders": {"insertion": "insertion", "bfs": "bfs"},
            "same_logical_graph_required": bool(args.verify_same_graph),
            "require_full_memory_build": bool(args.require_full_memory_build),
        },
        "verification": {
            "query_offset": int(args.query_offset),
            "queries": int(args.queries),
            "exact_queries": len(recalls),
            "k": int(args.verify_k),
            "ef_search": int(args.ef_search),
            "mean_exact_recall": statistics.fmean(recalls) if recalls else None,
            "min_exact_recall": min(recalls) if recalls else None,
            "mean_threshold": float(args.verify_min_exact_recall),
            "per_query_threshold": float(args.verify_min_query_exact_recall),
            "ordered_ids_equal": all(bool(row["ordered_ids_equal"]) for row in rows),
        },
        "relations": relations,
    }
    manifest = args.out.with_suffix(args.out.suffix + ".manifest.json")
    staged = manifest.with_suffix(manifest.suffix + ".tmp")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged.replace(manifest)
    return manifest


def ensure_functions(cur: psycopg.Cursor) -> None:
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
    ]
    for sql in functions:
        cur.execute(sql)
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def prepare_same_graph_layouts(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    """Build twin tables and HNSW indexes where the logical graph is reproducible."""
    source = qident(args.source_table)
    insertion_table = qident(args.insertion_table)
    bfs_table = qident(args.bfs_table)
    insertion_index = qident(args.insertion_index)
    bfs_index = qident(args.bfs_index)
    order_sql = args.copy_order_by.strip()
    order_clause = f" ORDER BY {order_sql}" if order_sql else ""

    print("preparing same-graph twin tables", flush=True)
    cur.execute("SET statement_timeout = 0")
    cur.execute(f"DROP TABLE IF EXISTS {insertion_table} CASCADE")
    cur.execute(f"DROP TABLE IF EXISTS {bfs_table} CASCADE")
    logged = "" if args.logged_tables else "UNLOGGED "
    cur.execute(f"CREATE {logged}TABLE {insertion_table} AS SELECT * FROM {source}{order_clause}")
    cur.execute(f"CREATE {logged}TABLE {bfs_table} AS SELECT * FROM {source}{order_clause}")
    if args.disable_autovacuum_during_build:
        cur.execute(f"ALTER TABLE {insertion_table} SET (autovacuum_enabled = false)")
        cur.execute(f"ALTER TABLE {bfs_table} SET (autovacuum_enabled = false)")
    cur.execute(f"CREATE UNIQUE INDEX {qident(args.insertion_table + '_id_idx')} ON {insertion_table} (id)")
    cur.execute(f"CREATE UNIQUE INDEX {qident(args.bfs_table + '_id_idx')} ON {bfs_table} (id)")
    cur.execute(f"ANALYZE {insertion_table}")
    cur.execute(f"ANALYZE {bfs_table}")
    validate_same_graph_tables(cur, args)
    cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
    cur.execute("SET max_parallel_maintenance_workers = 0")
    cur.execute(f"SET hnsw.build_seed = {int(args.hnsw_build_seed)}")
    cur.execute(f"SET hnsw.require_full_memory_build = {'on' if args.require_full_memory_build else 'off'}")
    cur.execute("SET hnsw.build_page_order = insertion")
    print("building insertion-layout HNSW index", flush=True)
    cur.execute(
        f"CREATE INDEX {insertion_index} ON {insertion_table} USING hnsw (embedding vector_l2_ops) "
        f"WITH (m = {int(args.hnsw_m)}, ef_construction = {int(args.hnsw_ef_construction)})"
    )
    cur.execute(f"SET hnsw.build_seed = {int(args.hnsw_build_seed)}")
    cur.execute("SET hnsw.build_page_order = bfs")
    print("building BFS-layout HNSW index from the same deterministic graph", flush=True)
    cur.execute(
        f"CREATE INDEX {bfs_index} ON {bfs_table} USING hnsw (embedding vector_l2_ops) "
        f"WITH (m = {int(args.hnsw_m)}, ef_construction = {int(args.hnsw_ef_construction)})"
    )
    cur.execute("SET hnsw.build_page_order = insertion")
    cur.execute("SET hnsw.build_seed = -1")
    cur.execute("SET hnsw.require_full_memory_build = off")

    if args.create_scalar_indexes:
        for table_name, table_sql in ((args.insertion_table, insertion_table), (args.bfs_table, bfs_table)):
            for suffix, columns in SCALAR_INDEXES:
                index_name = qident(f"{table_name}_{suffix}_idx")
                cur.execute(f"CREATE INDEX {index_name} ON {table_sql} ({columns})")
    cur.execute(f"ANALYZE {insertion_table}")
    cur.execute(f"ANALYZE {bfs_table}")
    if args.disable_autovacuum_during_build:
        cur.execute(f"ALTER TABLE {insertion_table} RESET (autovacuum_enabled)")
        cur.execute(f"ALTER TABLE {bfs_table} RESET (autovacuum_enabled)")


def validate_same_graph_tables(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    source_table = qident(args.source_table)
    insertion_table = qident(args.insertion_table)
    bfs_table = qident(args.bfs_table)
    summaries = []
    for table in (source_table, insertion_table, bfs_table):
        print(f"validating table ID space: {table}", flush=True)
        cur.execute(f"SELECT count(*), min(id), max(id) FROM {table}")
        summaries.append(tuple(cur.fetchone()))
    if summaries[0][0] <= 0 or summaries[0] != summaries[1] or summaries[0] != summaries[2]:
        raise RuntimeError(
            "source/twin table ID spaces differ: "
            f"source={summaries[0]} insertion={summaries[1]} bfs={summaries[2]}"
        )

    print("validating source/twin logical row sample", flush=True)
    cur.execute(
        f"""
        WITH sampled_ids AS MATERIALIZED (
            SELECT id
            FROM {source_table} TABLESAMPLE SYSTEM (0.1) REPEATABLE (20260718)
            LIMIT 10000
        )
        SELECT count(*)
        FROM sampled_ids
        JOIN {source_table} AS source USING (id)
        LEFT JOIN {insertion_table} AS insertion USING (id)
        LEFT JOIN {bfs_table} AS bfs USING (id)
        WHERE insertion.id IS NULL
           OR bfs.id IS NULL
           OR to_jsonb(source) IS DISTINCT FROM to_jsonb(insertion)
           OR to_jsonb(source) IS DISTINCT FROM to_jsonb(bfs)
        """
    )
    logical_mismatches = int(cur.fetchone()[0])
    if logical_mismatches:
        raise RuntimeError(
            f"source/twin logical sample has {logical_mismatches} mismatches"
        )

    print("validating twin physical/vector row sample", flush=True)
    cur.execute(
        f"""
        WITH sampled AS MATERIALIZED (
            SELECT id, ctid, embedding
            FROM {insertion_table} TABLESAMPLE SYSTEM (0.1) REPEATABLE (20260718)
            LIMIT 10000
        )
        SELECT count(*)
        FROM sampled AS source
        LEFT JOIN {bfs_table} AS target USING (id)
        WHERE target.id IS NULL
           OR source.ctid <> target.ctid
           OR source.embedding <> target.embedding
        """
    )
    mismatches = int(cur.fetchone()[0])
    if mismatches:
        raise RuntimeError(f"twin table physical/vector sample has {mismatches} mismatches")


def rebuild_same_graph_indexes(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    """Rebuild only the two HNSW indexes in one backend on existing twin heaps."""
    insertion_table = qident(args.insertion_table)
    bfs_table = qident(args.bfs_table)
    insertion_index = qident(args.insertion_index)
    bfs_index = qident(args.bfs_index)

    validate_same_graph_tables(cur, args)
    cur.execute("SET statement_timeout = 0")
    cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
    cur.execute("SET max_parallel_maintenance_workers = 0")
    cur.execute(f"SET hnsw.require_full_memory_build = {'on' if args.require_full_memory_build else 'off'}")
    if args.disable_autovacuum_during_build:
        cur.execute(f"ALTER TABLE {insertion_table} SET (autovacuum_enabled = false)")
        cur.execute(f"ALTER TABLE {bfs_table} SET (autovacuum_enabled = false)")

    try:
        cur.execute(f"DROP INDEX IF EXISTS {insertion_index}")
        cur.execute(f"DROP INDEX IF EXISTS {bfs_index}")

        cur.execute(f"SET hnsw.build_seed = {int(args.hnsw_build_seed)}")
        cur.execute("SET hnsw.build_page_order = insertion")
        print("rebuilding insertion-layout HNSW index", flush=True)
        cur.execute(
            f"CREATE INDEX {insertion_index} ON {insertion_table} USING hnsw "
            f"(embedding vector_l2_ops) WITH (m = {int(args.hnsw_m)}, "
            f"ef_construction = {int(args.hnsw_ef_construction)})"
        )

        # BuildIndex reseeds internally, and the explicit reset records the
        # reproducibility contract at the SQL/session boundary as well.
        cur.execute(f"SET hnsw.build_seed = {int(args.hnsw_build_seed)}")
        cur.execute("SET hnsw.build_page_order = bfs")
        print("rebuilding BFS-layout HNSW index from the same deterministic graph", flush=True)
        cur.execute(
            f"CREATE INDEX {bfs_index} ON {bfs_table} USING hnsw "
            f"(embedding vector_l2_ops) WITH (m = {int(args.hnsw_m)}, "
            f"ef_construction = {int(args.hnsw_ef_construction)})"
        )
    finally:
        cur.execute("SET hnsw.build_page_order = insertion")
        cur.execute("SET hnsw.build_seed = -1")
        cur.execute("SET hnsw.require_full_memory_build = off")
        if args.disable_autovacuum_during_build:
            cur.execute(f"ALTER TABLE {insertion_table} RESET (autovacuum_enabled)")
            cur.execute(f"ALTER TABLE {bfs_table} RESET (autovacuum_enabled)")

    for index_name in (args.insertion_index, args.bfs_index):
        cur.execute(
            "SELECT indisvalid, indisready FROM pg_index WHERE indexrelid = %s::regclass",
            (index_name,),
        )
        state = cur.fetchone()
        if state != (True, True):
            raise RuntimeError(f"rebuilt HNSW index is not valid/ready: {index_name} state={state}")
    cur.execute(f"ANALYZE {insertion_table}")
    cur.execute(f"ANALYZE {bfs_table}")


def create_same_graph_scalar_indexes(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    """Create the identical scalar-index set on existing twin heaps."""
    validate_same_graph_tables(cur, args)
    cur.execute("SET statement_timeout = 0")
    cur.execute(f"SET maintenance_work_mem = '{args.scalar_maintenance_work_mem}'")
    cur.execute(f"SET max_parallel_maintenance_workers = {int(args.scalar_parallel_workers)}")

    for table_name in (args.insertion_table, args.bfs_table):
        table = qident(table_name)
        id_index_name = f"{table_name}_id_idx"
        cur.execute(
            "SELECT indisunique FROM pg_index WHERE indexrelid = to_regclass(%s)",
            (id_index_name,),
        )
        state = cur.fetchone()
        if state is not None and not bool(state[0]):
            cur.execute(f"DROP INDEX {qident(id_index_name)}")
            state = None
        if state is None:
            cur.execute(f"CREATE UNIQUE INDEX {qident(id_index_name)} ON {table} (id)")

        for suffix, columns in SCALAR_INDEXES:
            index_name = qident(f"{table_name}_{suffix}_idx")
            print(f"creating scalar index {table_name}_{suffix}_idx", flush=True)
            cur.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({columns})")
        cur.execute(f"ANALYZE {table}")


def configure_base(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(args.metadata_cache_max_mb)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def mode_table_index(args: argparse.Namespace, mode: str) -> tuple[str, str]:
    if mode == "design1_bloom_bfs_layout":
        return args.bfs_table, args.bfs_index
    return args.insertion_table, args.insertion_index


def activate_mode(cur: psycopg.Cursor, args: argparse.Namespace, mode: str, filter_name: str, preload: bool = False) -> dict[str, object]:
    configure_base(cur, args)
    if preload:
        cur.execute("SET statement_timeout = 0")
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    table, index = mode_table_index(args, mode)
    if mode == "original":
        return {"table": table, "index": index}

    cur.execute("SET hnsw.filter_strategy = safe_guided")
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'bloom')",
        (index, FILTER_ATOMS[filter_name]),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    profile = json.loads(cur.fetchone()[0])
    profile["table"] = table
    profile["index"] = index
    return profile


def run_query(cur: psycopg.Cursor, table: str, predicate: str, query_id: int, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {table}
        WHERE ({predicate}) AND id <> %s
        ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
        LIMIT {int(k)}
        """,
        (int(query_id), int(query_id)),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def run_unfiltered_query(
    cur: psycopg.Cursor, table: str, query_id: int, k: int
) -> tuple[list[int], list[float], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id, embedding <-> (SELECT embedding FROM {table} WHERE id = %s) AS distance
        FROM {table}
        WHERE id <> %s
        ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
        LIMIT {int(k)}
        """,
        (int(query_id), int(query_id), int(query_id)),
    )
    rows = cur.fetchall()
    ids = [int(row[0]) for row in rows]
    distances = [float(row[1]) for row in rows]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, distances, json.loads(cur.fetchone()[0])


def run_exact_unfiltered_query(
    cur: psycopg.Cursor, table: str, query_id: int, k: int
) -> list[tuple[int, float]]:
    cur.execute(f"SELECT embedding::text FROM {table} WHERE id = %s", (int(query_id),))
    query_vector = cur.fetchone()[0]
    cur.execute("SHOW enable_indexscan")
    old_indexscan = cur.fetchone()[0]
    cur.execute("SHOW enable_indexonlyscan")
    old_indexonlyscan = cur.fetchone()[0]
    cur.execute("SHOW enable_bitmapscan")
    old_bitmapscan = cur.fetchone()[0]
    cur.execute("SHOW enable_seqscan")
    old_seqscan = cur.fetchone()[0]
    cur.execute("SHOW enable_sort")
    old_sort = cur.fetchone()[0]
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_indexonlyscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_sort = on")
    try:
        cur.execute(
            f"""
            SELECT id, embedding <-> %s::vector AS distance
            FROM {table}
            WHERE id <> %s
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k + 1)}
            """,
            (query_vector, int(query_id), query_vector),
        )
        return [(int(row[0]), float(row[1])) for row in cur.fetchall()]
    finally:
        cur.execute(f"SET enable_indexscan = {old_indexscan}")
        cur.execute(f"SET enable_indexonlyscan = {old_indexonlyscan}")
        cur.execute(f"SET enable_bitmapscan = {old_bitmapscan}")
        cur.execute(f"SET enable_seqscan = {old_seqscan}")
        cur.execute(f"SET enable_sort = {old_sort}")


def verify_same_logical_graph(cur: psycopg.Cursor, args: argparse.Namespace, query_nos, query_by_no) -> list[dict[str, object]]:
    if not args.verify_same_graph:
        return []

    print("verifying same logical HNSW graph before benchmark", flush=True)
    configure_base(cur, args)
    mismatches = []
    exact_recalls: list[float] = []
    verification_rows: list[dict[str, object]] = []
    for qno in query_nos[: args.verify_queries]:
        qid = query_by_no[qno]
        insert_ids, insert_distances, insert_profile = run_unfiltered_query(
            cur, args.insertion_table, qid, args.verify_k
        )
        bfs_ids, bfs_distances, bfs_profile = run_unfiltered_query(
            cur, args.bfs_table, qid, args.verify_k
        )
        insert_visited = int(insert_profile.get("visited_tuples", -1))
        bfs_visited = int(bfs_profile.get("visited_tuples", -1))
        counter_names = [
            "distance_compute_count",
            "traversal_expanded_nodes",
            "traversal_neighbors_examined",
            "traversal_candidate_admissions",
            "traversal_result_admissions",
            "traversal_stock_terminations",
            "traversal_max_scan_terminations",
            "traversal_exhausted_terminations",
        ]
        counter_mismatch = {
            name: (int(insert_profile.get(name, -1)), int(bfs_profile.get(name, -1)))
            for name in counter_names
            if int(insert_profile.get(name, -1)) != int(bfs_profile.get(name, -1))
        }
        exhausted = int(insert_profile.get("traversal_exhausted_terminations", 0))
        max_scan = int(insert_profile.get("traversal_max_scan_terminations", 0))
        expanded = int(insert_profile.get("traversal_expanded_nodes", 0))
        incomplete = len(insert_ids) != args.verify_k or len(bfs_ids) != args.verify_k
        distance_mismatch = len(insert_distances) != len(bfs_distances) or any(
            not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
            for left, right in zip(insert_distances, bfs_distances)
        )
        invalid_profile = not bool(insert_profile.get("valid")) or not bool(bfs_profile.get("valid"))
        if (
            insert_ids != bfs_ids
            or distance_mismatch
            or insert_visited != bfs_visited
            or counter_mismatch
            or max_scan
            or incomplete
            or invalid_profile
        ):
            mismatches.append(
                {
                    "query_no": qno,
                    "query_id": qid,
                    "insert_visited": insert_visited,
                    "bfs_visited": bfs_visited,
                    "insert_ids": insert_ids,
                    "bfs_ids": bfs_ids,
                    "counter_mismatch": counter_mismatch,
                    "distance_mismatch": distance_mismatch,
                    "exhausted_terminations": exhausted,
                    "max_scan_terminations": max_scan,
                    "expanded_nodes": expanded,
                    "invalid_profile": invalid_profile,
                    "incomplete": incomplete,
                }
            )

        exact_recall: float | str = ""
        if len(exact_recalls) < args.verify_exact_queries:
            exact_rows = run_exact_unfiltered_query(
                cur, args.insertion_table, qid, args.verify_k
            )
            if len(exact_rows) < args.verify_k:
                raise RuntimeError(f"exact unfiltered quality GT returned {len(exact_rows)} rows")
            kth_distance = exact_rows[args.verify_k - 1][1]
            tolerance = max(1e-9, abs(kth_distance) * 1e-6)
            exact_recall = min(
                args.verify_k,
                sum(distance <= kth_distance + tolerance for distance in insert_distances),
            ) / args.verify_k
            exact_recalls.append(float(exact_recall))
        verification_rows.append(
            {
                "query_no": qno,
                "query_id": qid,
                "ordered_ids_equal": insert_ids == bfs_ids,
                "topk_count": len(insert_ids),
                "topk_ids": ",".join(str(value) for value in insert_ids),
                "visited_tuples": insert_visited,
                "distance_compute_count": int(insert_profile.get("distance_compute_count", -1)),
                "expanded_nodes": expanded,
                "neighbors_examined": int(insert_profile.get("traversal_neighbors_examined", -1)),
                "candidate_admissions": int(insert_profile.get("traversal_candidate_admissions", -1)),
                "result_admissions": int(insert_profile.get("traversal_result_admissions", -1)),
                "stock_terminations": int(insert_profile.get("traversal_stock_terminations", -1)),
                "max_scan_terminations": max_scan,
                "exhausted_terminations": exhausted,
                "exact_recall_at_k": exact_recall,
            }
        )

    if mismatches:
        sample = json.dumps(mismatches[:3], ensure_ascii=False)
        raise RuntimeError(
            "same-graph verification failed; insertion and BFS indexes differ logically. "
            f"sample={sample}"
        )
    if exact_recalls and (
        statistics.fmean(exact_recalls) < args.verify_min_exact_recall
        or min(exact_recalls) < args.verify_min_query_exact_recall
    ):
        raise RuntimeError(
            "HNSW quality gate failed against exact unfiltered top-k: "
            f"mean={statistics.fmean(exact_recalls):.4f}, "
            f"min={min(exact_recalls):.4f}, recalls={exact_recalls}"
        )
    print(f"same-graph verification passed for {min(len(query_nos), args.verify_queries)} queries", flush=True)
    if exact_recalls:
        print(
            f"exact quality gate passed for {len(exact_recalls)} queries: "
            f"mean_recall={statistics.fmean(exact_recalls):.4f}, min_recall={min(exact_recalls):.4f}",
            flush=True,
        )
    return verification_rows


def warmup(cur: psycopg.Cursor, args: argparse.Namespace, filters, query_nos, query_by_no) -> None:
    if args.warmup_queries <= 0:
        return
    warm_nos = query_nos[: args.warmup_queries]
    warm_filter_names = set(args.warmup_filter_names or [])
    for filter_name, _, predicate in filters:
        if warm_filter_names and filter_name not in warm_filter_names:
            continue
        for mode in MODES:
            profile = activate_mode(cur, args, mode, filter_name)
            table = str(profile["table"])
            for qno in warm_nos:
                try:
                    print(f"warmup mode={mode} filter={filter_name} q={qno}", flush=True)
                    run_query(cur, table, predicate, query_by_no[qno], args.k)
                except Exception:
                    cur.execute("ROLLBACK")
                    configure_base(cur, args)
            cur.execute("SELECT vector_hnsw_guidance_reset()")


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    summary = out.with_name(out.stem + "_summary.csv")
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["mode"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    mode_order = {mode: i for i, mode in enumerate(MODES)}

    def mean(items, key):
        vals = [float(r[key]) for r in items]
        return statistics.fmean(vals) if vals else 0.0

    def p95(items, key):
        vals = sorted(float(r[key]) for r in items)
        return vals[int(0.95 * (len(vals) - 1))] if vals else 0.0

    fields = [
        "filter",
        "filter_name",
        "mode",
        "table",
        "index",
        "ok",
        "errors",
        "recall_mean",
        "end_to_end_mean_ms",
        "end_to_end_p95_ms",
        "query_latency_mean_ms",
        "activation_mean_ms",
        "vector_search_mean_ms",
        "visited_tuples_mean",
        "returned_tuples_mean",
        "guidance_checks_mean",
        "guidance_skips_mean",
        "index_element_runs_mean",
        "index_element_distinct_pages_mean",
        "speedup_vs_original",
        "speedup_vs_design1",
    ]
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    for (filter_name, mode), items in groups.items():
        ok = [r for r in items if not r["error"]]
        first = items[0]
        summaries[(filter_name, mode)] = {
            "filter": first["filter"],
            "filter_name": filter_name,
            "mode": mode,
            "table": first["table"],
            "index": first["index"],
            "ok": len(ok),
            "errors": len(items) - len(ok),
            "recall_mean": mean(ok, "recall"),
            "end_to_end_mean_ms": mean(ok, "end_to_end_ms"),
            "end_to_end_p95_ms": p95(ok, "end_to_end_ms"),
            "query_latency_mean_ms": mean(ok, "query_latency_ms"),
            "activation_mean_ms": mean(ok, "activation_ms"),
            "vector_search_mean_ms": mean(ok, "vector_search_ms"),
            "visited_tuples_mean": mean(ok, "visited_tuples"),
            "returned_tuples_mean": mean(ok, "returned_tuples"),
            "guidance_checks_mean": mean(ok, "guidance_checks"),
            "guidance_skips_mean": mean(ok, "guidance_skips"),
            "index_element_runs_mean": mean(ok, "index_page_element_runs"),
            "index_element_distinct_pages_mean": mean(ok, "index_page_element_distinct_pages"),
            "speedup_vs_original": 0.0,
            "speedup_vs_design1": 0.0,
        }

    for filter_name in {key[0] for key in summaries}:
        original = summaries.get((filter_name, "original"))
        design1 = summaries.get((filter_name, "design1_bloom"))
        if not original:
            continue
        base = float(original["end_to_end_mean_ms"])
        d1 = float(design1["end_to_end_mean_ms"]) if design1 else 0.0
        for mode in MODES:
            item = summaries.get((filter_name, mode))
            if not item:
                continue
            val = float(item["end_to_end_mean_ms"])
            item["speedup_vs_original"] = base / val if val else 0.0
            item["speedup_vs_design1"] = d1 / val if d1 and val else 0.0

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, item in sorted(summaries.items(), key=lambda kv: (order.get(kv[0][0], 999), mode_order.get(kv[0][1], 999))):
            writer.writerow(item)
    print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare insertion-order HNSW with Design 1 and BFS-layout Design 2.")
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--insertion-table", default=INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument("--prepare-same-graph-layouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rebuild-same-graph-indexes-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing twin heaps and rebuild both deterministic HNSW indexes in this backend.",
    )
    parser.add_argument(
        "--create-scalar-indexes-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing twin heaps and create identical scalar predicate indexes.",
    )
    parser.add_argument("--copy-order-by", default="id")
    parser.add_argument("--maintenance-work-mem", default="32GB")
    parser.add_argument("--scalar-maintenance-work-mem", default="2GB")
    parser.add_argument("--scalar-parallel-workers", type=int, default=2)
    parser.add_argument("--hnsw-build-seed", type=int, default=20260718)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=100)
    parser.add_argument("--require-full-memory-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-scalar-indexes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-autovacuum-during-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logged-tables", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-same-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run the same-graph and exact-quality gates without building guidance or benchmarking modes.",
    )
    parser.add_argument("--verify-queries", type=int, default=8)
    parser.add_argument("--verify-k", type=int, default=20)
    parser.add_argument("--verify-min-expanded-nodes", type=int, default=100)
    parser.add_argument("--verify-min-visited-tuples", type=int, default=100)
    parser.add_argument("--verify-exact-queries", type=int, default=8)
    parser.add_argument("--verify-min-exact-recall", type=float, default=0.90)
    parser.add_argument("--verify-min-query-exact-recall", type=float, default=0.70)
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--warmup-filter-names", nargs="*", default=["popular_ge1000"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument(
        "--scan-mem-multiplier",
        type=float,
        default=32.0,
        help="Use a non-binding main-experiment budget; evaluate iterative-scan memory separately.",
    )
    parser.add_argument("--metadata-cache-max-mb", type=int, default=1024)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-queries", type=int, default=10)
    args = parser.parse_args()

    build_actions = sum(
        bool(value)
        for value in (
            args.prepare_same_graph_layouts,
            args.rebuild_same_graph_indexes_only,
            args.create_scalar_indexes_only,
        )
    )
    if build_actions > 1:
        parser.error("same-graph prepare/rebuild/scalar-only actions are mutually exclusive")

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filter",
        "filter_name",
        "mode",
        "table",
        "index",
        "query_no",
        "query_id",
        "repeat",
        "run_order",
        "recall",
        "activation_ms",
        "query_latency_ms",
        "end_to_end_ms",
        "vector_search_ms",
        "visited_tuples",
        "returned_tuples",
        "guidance_checks",
        "guidance_skips",
        "index_page_element_runs",
        "index_page_element_distinct_pages",
        "returned",
        "ids",
        "error",
    ]

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        if args.prepare_same_graph_layouts:
            prepare_same_graph_layouts(cur, args)
        elif args.rebuild_same_graph_indexes_only:
            rebuild_same_graph_indexes(cur, args)
        elif args.create_scalar_indexes_only:
            create_same_graph_scalar_indexes(cur, args)
        configure_base(cur, args)
        verification_rows = verify_same_logical_graph(cur, args, query_nos, query_by_no)
        if args.verify_only:
            if not verification_rows:
                raise RuntimeError("verification-only run produced no checks")
            with args.out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(verification_rows[0].keys()))
                writer.writeheader()
                writer.writerows(verification_rows)
            print(f"wrote verification rows to {args.out}", flush=True)
            manifest = write_verification_manifest(cur, args, verification_rows)
            print(f"wrote verification manifest to {manifest}", flush=True)
            print("verification-only run complete", flush=True)
            return

        # Build/load Design 1 fragments for both indexes before timing.
        for filter_name, _, _ in filters:
            for mode in ["design1_bloom", "design1_bloom_bfs_layout"]:
                activate_mode(cur, args, mode, filter_name, preload=True)
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        configure_base(cur, args)

        print("warming up", flush=True)
        warmup(cur, args, filters, query_nos, query_by_no)

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for filter_name, target_rate, predicate in filters:
                for idx, qno in enumerate(query_nos, start=1):
                    qid = query_by_no[qno]
                    for repeat in range(args.repeats):
                        run_modes = MODES[:]
                        rng.shuffle(run_modes)
                        for run_order, mode in enumerate(run_modes):
                            error = ""
                            activation_profile: dict[str, object] = {}
                            ids: list[int] = []
                            profile: dict[str, object] = {}
                            activation_ms = 0.0
                            query_ms = 0.0
                            try:
                                activation_profile, activation_ms = timed_ms(lambda m=mode: activate_mode(cur, args, m, filter_name))
                                table = str(activation_profile["table"])
                                index = str(activation_profile["index"])
                                (ids, profile), query_ms = timed_ms(lambda: run_query(cur, table, predicate, qid, args.k))
                            except errors.QueryCanceled as exc:
                                error = exc.__class__.__name__
                                cur.execute("SET statement_timeout = 0")
                                table, index = mode_table_index(args, mode)
                            except Exception as exc:  # noqa: BLE001
                                error = exc.__class__.__name__
                                try:
                                    cur.execute("ROLLBACK")
                                except Exception:
                                    pass
                                table, index = mode_table_index(args, mode)
                            row = {
                                "filter": target_rate,
                                "filter_name": filter_name,
                                "mode": mode,
                                "table": table,
                                "index": index,
                                "query_no": qno,
                                "query_id": qid,
                                "repeat": repeat,
                                "run_order": run_order,
                                "recall": recall_at_k(ids, truth[(filter_name, qno)], args.k) if not error else 0.0,
                                "activation_ms": activation_ms,
                                "query_latency_ms": query_ms,
                                "end_to_end_ms": activation_ms + query_ms,
                                "vector_search_ms": profile.get("vector_search_ms", 0.0),
                                "visited_tuples": profile.get("visited_tuples", 0),
                                "returned_tuples": profile.get("returned_tuples", 0),
                                "guidance_checks": profile.get("guidance_checks", 0),
                                "guidance_skips": profile.get("guidance_skips", 0),
                                "index_page_element_runs": profile.get("index_page_element_runs", 0),
                                "index_page_element_distinct_pages": profile.get("index_page_element_distinct_pages", 0),
                                "returned": len(ids),
                                "ids": ",".join(str(x) for x in ids),
                                "error": error,
                            }
                            rows.append(row)
                            writer.writerow(row)
                            f.flush()
                    if args.progress_queries and idx % args.progress_queries == 0:
                        recent = [r for r in rows if r["filter_name"] == filter_name and not r["error"]]
                        parts = []
                        for mode in MODES:
                            vals = [float(r["end_to_end_ms"]) for r in recent if r["mode"] == mode]
                            if vals:
                                parts.append(f"{mode}={statistics.fmean(vals):.2f}ms")
                        print(f"progress filter={filter_name} queries={idx}/{len(query_nos)} " + " ".join(parts), flush=True)

    print(f"wrote {args.out}", flush=True)
    summarize(rows, args.out)


if __name__ == "__main__":
    main()
