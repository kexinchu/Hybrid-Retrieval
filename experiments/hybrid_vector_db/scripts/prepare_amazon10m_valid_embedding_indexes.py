from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import psycopg
from psycopg import sql

try:
    from .common_pg import pg_config_from_env
    from .pgvector_design1_design2_design3_selectivity_benchmark import (
        require_d2_graph_proof,
    )
except ImportError:  # Direct execution places this directory on sys.path.
    from common_pg import pg_config_from_env
    from pgvector_design1_design2_design3_selectivity_benchmark import (
        require_d2_graph_proof,
    )


DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_SOURCE_INDEX = "public.amazon10m_embedding_valid_hnsw_source_idx"
DEFAULT_CLONE_INDEX = "public.amazon10m_embedding_valid_hnsw_bfs_clone_idx"
DEFAULT_CONSTRAINT = "amazon10m_embedding_valid_norm_check"
DEFAULT_PROOF_OUTPUT = Path(
    "results/hybrid_vector_db/amazon10m_valid_embedding_d2_graph_proof.json"
)
LEGACY_INDEXES = {
    "public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx",
    "public.amazon_grocery_reviews_10m_pgvector_hnsw_bfs_clone_idx",
}
EXPECTED_ROWS = 10_000_000
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
INDEX_PROVENANCE_PREFIX = "sqlens-valid-embedding-index-v1:"
INDEX_PROVENANCE_CONTRACT = "sqlens_valid_embedding_same_heap_hnsw_v1"
ARTIFACT_CONTRACT = "sqlens_amazon10m_valid_embedding_indexes_v1"
STAGES = ("all", "column", "source", "clone", "proof", "verify")

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:kB|MB|GB|TB)$")


class PreparationError(RuntimeError):
    """Raised when a resumable preparation invariant fails closed."""


@dataclass(frozen=True)
class RelationState:
    name: str
    oid: int
    relfilenode: int
    physical_relfilenode: int


@dataclass(frozen=True)
class ColumnState:
    data_type: str
    not_null: bool
    default_expression: str | None
    has_missing_value: bool


@dataclass(frozen=True)
class ConstraintState:
    name: str
    validated: bool
    no_inherit: bool
    expression: str


@dataclass(frozen=True)
class IndexState:
    name: str
    oid: int
    relfilenode: int
    physical_relfilenode: int
    heap_oid: int
    heap_relfilenode: int
    valid: bool
    ready: bool
    live: bool
    access_method: str
    unique: bool
    primary: bool
    key_attributes: int
    total_attributes: int
    indexed_column: str | None
    opclass: str | None
    predicate: str | None
    reloptions: tuple[str, ...]
    comment: str | None
    definition: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_qualified_name(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) != 2 or any(not IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise argparse.ArgumentTypeError(
            "relation names must use unquoted schema.relation identifiers"
        )
    return parts[0].lower(), parts[1].lower()


def qualified_name(value: str) -> str:
    schema, relation = parse_qualified_name(value)
    return f"{schema}.{relation}"


def quote_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid PostgreSQL identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def quote_qualified_name(value: str) -> str:
    schema, relation = parse_qualified_name(value)
    return f"{quote_identifier(schema)}.{quote_identifier(relation)}"


def sha256_value(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA256")
    return normalized


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def memory_setting(value: str) -> str:
    if not MEMORY_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "memory setting must be a positive PostgreSQL unit such as 64GB"
        )
    return value


def normalize_default_expression(value: str | None) -> str:
    text = "" if value is None else value.strip().lower()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return re.sub(r"\s+", "", text).replace("::boolean", "")


def normalize_check_expression(value: str) -> str:
    text = value.lower()
    text = re.sub(r"::\s*(?:double precision|float8)", "", text)
    text = re.sub(r"[\s()]", "", text)
    return text


def normalize_predicate(value: str | None) -> str:
    text = "" if value is None else value.lower()
    return re.sub(r"[\s()]", "", text)


def parse_reloptions(values: tuple[str, ...] | list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values or ():
        key, separator, value = str(item).partition("=")
        if not separator or not key or key in parsed:
            raise PreparationError(f"invalid or duplicate index reloption: {item!r}")
        parsed[key] = value
    return parsed


def provenance_comment(contract: dict[str, object]) -> str:
    return INDEX_PROVENANCE_PREFIX + json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def relation_state(cur: psycopg.Cursor, relation: str) -> RelationState:
    cur.execute(
        "SELECT c.oid::bigint, c.relfilenode::bigint, "
        "pg_relation_filenode(c.oid)::bigint "
        "FROM pg_class c WHERE c.oid = to_regclass(%s)",
        (relation,),
    )
    row = cur.fetchone()
    if row is None:
        raise PreparationError(f"required relation does not exist: {relation}")
    state = RelationState(relation, int(row[0]), int(row[1]), int(row[2]))
    if min(state.oid, state.relfilenode, state.physical_relfilenode) <= 0:
        raise PreparationError(f"relation has invalid physical identity: {state}")
    return state


def column_state(cur: psycopg.Cursor, table: str) -> ColumnState | None:
    cur.execute(
        "SELECT format_type(a.atttypid, a.atttypmod), a.attnotnull, "
        "pg_get_expr(d.adbin, d.adrelid, true), a.atthasmissing "
        "FROM pg_attribute a "
        "LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
        "WHERE a.attrelid = %s::regclass AND a.attname = 'embedding_valid' "
        "AND a.attnum > 0 AND NOT a.attisdropped",
        (table,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return ColumnState(str(row[0]), bool(row[1]), row[2], bool(row[3]))


def validate_column_definition(state: ColumnState | None) -> ColumnState:
    if state is None:
        raise PreparationError("embedding_valid column is missing")
    mismatches: list[str] = []
    if state.data_type != "boolean":
        mismatches.append(f"type={state.data_type!r}, expected 'boolean'")
    if not state.not_null:
        mismatches.append("NOT NULL is missing")
    if normalize_default_expression(state.default_expression) != "true":
        mismatches.append(
            f"default={state.default_expression!r}, expected constant true"
        )
    if mismatches:
        raise PreparationError(
            "embedding_valid column definition mismatch: " + "; ".join(mismatches)
        )
    return state


def constraint_state(
    cur: psycopg.Cursor, table: str, constraint_name: str
) -> ConstraintState | None:
    cur.execute(
        "SELECT c.conname, c.convalidated, c.connoinherit, "
        "pg_get_expr(c.conbin, c.conrelid, true) "
        "FROM pg_constraint c "
        "WHERE c.conrelid = %s::regclass AND c.conname = %s AND c.contype = 'c'",
        (table, constraint_name),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return ConstraintState(str(row[0]), bool(row[1]), bool(row[2]), str(row[3]))


def validate_constraint_definition(
    state: ConstraintState | None, *, require_validated: bool
) -> ConstraintState:
    if state is None:
        raise PreparationError("embedding_valid CHECK constraint is missing")
    expected = "embedding_valid=vector_normembedding>0"
    if normalize_check_expression(state.expression) != expected:
        raise PreparationError(
            "embedding_valid CHECK constraint definition mismatch: "
            f"observed={state.expression!r}"
        )
    if state.no_inherit:
        raise PreparationError("embedding_valid CHECK constraint is unexpectedly NO INHERIT")
    if require_validated and not state.validated:
        raise PreparationError("embedding_valid CHECK constraint is not validated")
    return state


def table_counts(cur: psycopg.Cursor, table: str) -> dict[str, int]:
    quoted_table = quote_qualified_name(table)
    cur.execute(
        f"SELECT count(*)::bigint, "
        f"count(*) FILTER (WHERE embedding_valid)::bigint, "
        f"count(*) FILTER (WHERE NOT embedding_valid)::bigint, "
        f"count(*) FILTER (WHERE embedding_valid IS DISTINCT FROM "
        f"(vector_norm(embedding) > 0))::bigint FROM {quoted_table}"
    )
    row = cur.fetchone()
    if row is None:
        raise PreparationError("could not count embedding validity populations")
    return {
        "total_rows": int(row[0]),
        "valid_rows": int(row[1]),
        "invalid_rows": int(row[2]),
        "inconsistent_rows": int(row[3]),
    }


def validate_counts(counts: dict[str, int], expected_rows: int) -> None:
    if counts["total_rows"] != expected_rows:
        raise PreparationError(
            f"table row count mismatch: expected={expected_rows}, "
            f"observed={counts['total_rows']}"
        )
    if counts["valid_rows"] + counts["invalid_rows"] != counts["total_rows"]:
        raise PreparationError("embedding validity populations do not partition the table")
    if counts["inconsistent_rows"] != 0:
        raise PreparationError(
            f"embedding_valid disagrees with vector_norm(embedding)>0 for "
            f"{counts['inconsistent_rows']} rows"
        )


def require_metadata_only_column_add(cur: psycopg.Cursor) -> int:
    cur.execute("SELECT current_setting('server_version_num')::integer")
    row = cur.fetchone()
    if row is None:
        raise PreparationError("could not determine PostgreSQL server_version_num")
    version = int(row[0])
    if version < 110000:
        raise PreparationError(
            "metadata-only ADD COLUMN with a constant default requires PostgreSQL 11+"
        )
    return version


def embedding_valid_statistics(
    cur: psycopg.Cursor, table: str, *, refresh: bool
) -> dict[str, object]:
    schema, relation = parse_qualified_name(table)
    if refresh:
        cur.execute(
            f"ANALYZE {quote_qualified_name(table)} "
            f"({quote_identifier('embedding_valid')})"
        )
    cur.execute(
        "SELECT null_frac::double precision, n_distinct::double precision, "
        "most_common_vals::text, most_common_freqs::text "
        "FROM pg_stats WHERE schemaname = %s AND tablename = %s "
        "AND attname = 'embedding_valid'",
        (schema, relation),
    )
    row = cur.fetchone()
    if row is None:
        raise PreparationError("embedding_valid planner statistics are missing")
    return {
        "refreshed": refresh,
        "null_frac": float(row[0]),
        "n_distinct": float(row[1]),
        "most_common_vals": str(row[2]),
        "most_common_freqs": str(row[3]),
    }


def ensure_embedding_valid_column(
    conn: psycopg.Connection,
    cur: psycopg.Cursor,
    args: argparse.Namespace,
) -> dict[str, object]:
    table = qualified_name(args.table)
    quoted_table = quote_qualified_name(table)
    before = relation_state(cur, table)
    existing = column_state(cur, table)
    column_added = existing is None
    if column_added:
        with conn.transaction():
            cur.execute(
                f"ALTER TABLE {quoted_table} ADD COLUMN embedding_valid "
                "boolean NOT NULL DEFAULT true"
            )
        after_add = relation_state(cur, table)
        if before.oid != after_add.oid or before.relfilenode != after_add.relfilenode:
            raise PreparationError(
                "metadata-only column addition rewrote/replaced the table: "
                f"before={before}, after={after_add}"
            )
    else:
        after_add = before
    state = validate_column_definition(column_state(cur, table))

    with conn.transaction():
        cur.execute(
            f"UPDATE {quoted_table} "
            "SET embedding_valid = (vector_norm(embedding) > 0) "
            "WHERE embedding_valid IS DISTINCT FROM (vector_norm(embedding) > 0)"
        )
        corrected_rows = int(cur.rowcount)

    constraint = constraint_state(cur, table, args.constraint_name)
    constraint_added = constraint is None
    quoted_constraint = quote_identifier(args.constraint_name)
    if constraint_added:
        with conn.transaction():
            cur.execute(
                f"ALTER TABLE {quoted_table} ADD CONSTRAINT {quoted_constraint} "
                "CHECK (embedding_valid = (vector_norm(embedding) > 0)) NOT VALID"
            )
        constraint = constraint_state(cur, table, args.constraint_name)
    validate_constraint_definition(constraint, require_validated=False)
    if not constraint.validated:
        with conn.transaction():
            cur.execute(
                f"ALTER TABLE {quoted_table} VALIDATE CONSTRAINT {quoted_constraint}"
            )
    constraint = validate_constraint_definition(
        constraint_state(cur, table, args.constraint_name), require_validated=True
    )
    counts = table_counts(cur, table)
    validate_counts(counts, args.expected_rows)
    planner_statistics = embedding_valid_statistics(cur, table, refresh=True)
    final_relation = relation_state(cur, table)
    return {
        "column_added": column_added,
        "constraint_added": constraint_added,
        "corrected_inconsistent_rows": corrected_rows,
        "metadata_only_add_verified": (
            before.oid == after_add.oid
            and before.relfilenode == after_add.relfilenode
            if column_added
            else None
        ),
        "relation_before_add": asdict(before),
        "relation_after_add": asdict(after_add),
        "relation_final": asdict(final_relation),
        "column": asdict(state),
        "constraint": asdict(constraint),
        "row_counts": counts,
        "planner_statistics": planner_statistics,
    }


def verify_embedding_valid_column(
    cur: psycopg.Cursor, args: argparse.Namespace
) -> dict[str, object]:
    table = qualified_name(args.table)
    relation = relation_state(cur, table)
    column = validate_column_definition(column_state(cur, table))
    constraint = validate_constraint_definition(
        constraint_state(cur, table, args.constraint_name), require_validated=True
    )
    counts = table_counts(cur, table)
    validate_counts(counts, args.expected_rows)
    planner_statistics = embedding_valid_statistics(cur, table, refresh=False)
    return {
        "column_added": False,
        "constraint_added": False,
        "corrected_inconsistent_rows": 0,
        "metadata_only_add_verified": None,
        "relation_final": asdict(relation),
        "column": asdict(column),
        "constraint": asdict(constraint),
        "row_counts": counts,
        "planner_statistics": planner_statistics,
    }


def index_state(cur: psycopg.Cursor, index_name: str) -> IndexState | None:
    cur.execute(
        "SELECT idx.oid::bigint, idx.relfilenode::bigint, "
        "pg_relation_filenode(idx.oid)::bigint, ix.indrelid::bigint, "
        "heap.relfilenode::bigint, ix.indisvalid, ix.indisready, ix.indislive, "
        "am.amname, ix.indisunique, ix.indisprimary, ix.indnkeyatts, ix.indnatts, "
        "att.attname, opc.opcname, pg_get_expr(ix.indpred, ix.indrelid, true), "
        "idx.reloptions, obj_description(idx.oid, 'pg_class'), pg_get_indexdef(idx.oid) "
        "FROM pg_class idx "
        "JOIN pg_index ix ON ix.indexrelid = idx.oid "
        "JOIN pg_class heap ON heap.oid = ix.indrelid "
        "JOIN pg_am am ON am.oid = idx.relam "
        "LEFT JOIN pg_attribute att ON att.attrelid = ix.indrelid "
        "AND att.attnum = ix.indkey[0] "
        "LEFT JOIN pg_opclass opc ON opc.oid = ix.indclass[0] "
        "WHERE idx.oid = to_regclass(%s)",
        (index_name,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return IndexState(
        name=index_name,
        oid=int(row[0]),
        relfilenode=int(row[1]),
        physical_relfilenode=int(row[2]),
        heap_oid=int(row[3]),
        heap_relfilenode=int(row[4]),
        valid=bool(row[5]),
        ready=bool(row[6]),
        live=bool(row[7]),
        access_method=str(row[8]),
        unique=bool(row[9]),
        primary=bool(row[10]),
        key_attributes=int(row[11]),
        total_attributes=int(row[12]),
        indexed_column=None if row[13] is None else str(row[13]),
        opclass=None if row[14] is None else str(row[14]),
        predicate=None if row[15] is None else str(row[15]),
        reloptions=tuple(row[16] or ()),
        comment=None if row[17] is None else str(row[17]),
        definition=str(row[18]),
    )


def source_build_contract(
    args: argparse.Namespace, table: RelationState
) -> dict[str, object]:
    return {
        "contract": INDEX_PROVENANCE_CONTRACT,
        "role": "source",
        "table": qualified_name(args.table),
        "table_oid": table.oid,
        "column": "embedding",
        "opclass": "vector_l2_ops",
        "predicate": "embedding_valid",
        "m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "build_page_order": "insertion",
        "clone_source": "",
        "require_full_memory_build": False,
        "build_seed": args.build_seed,
        "maintenance_work_mem": args.maintenance_work_mem,
        "max_parallel_maintenance_workers": 0,
    }


def clone_build_contract(
    args: argparse.Namespace, table: RelationState, source: IndexState
) -> dict[str, object]:
    return {
        "contract": INDEX_PROVENANCE_CONTRACT,
        "role": "clone",
        "table": qualified_name(args.table),
        "table_oid": table.oid,
        "column": "embedding",
        "opclass": "vector_l2_ops",
        "predicate": "embedding_valid",
        "m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "build_page_order": "bfs",
        "clone_source": qualified_name(args.source_index),
        "source_oid": source.oid,
        "source_relfilenode": source.relfilenode,
        "require_full_memory_build": True,
        "build_seed": None,
        "maintenance_work_mem": args.maintenance_work_mem,
        "max_parallel_maintenance_workers": 0,
    }


def index_definition_diff(
    state: IndexState,
    *,
    table: RelationState,
    expected_contract: dict[str, object],
) -> dict[str, dict[str, object]]:
    observed_options = parse_reloptions(state.reloptions)
    expected = {
        "heap_oid": table.oid,
        "heap_relfilenode": table.relfilenode,
        "valid": True,
        "ready": True,
        "live": True,
        "access_method": "hnsw",
        "unique": False,
        "primary": False,
        "key_attributes": 1,
        "total_attributes": 1,
        "indexed_column": "embedding",
        "opclass": "vector_l2_ops",
        "predicate": "embedding_valid",
        "reloptions": {"m": str(HNSW_M), "ef_construction": str(HNSW_EF_CONSTRUCTION)},
        "comment": provenance_comment(expected_contract),
    }
    observed = {
        "heap_oid": state.heap_oid,
        "heap_relfilenode": state.heap_relfilenode,
        "valid": state.valid,
        "ready": state.ready,
        "live": state.live,
        "access_method": state.access_method,
        "unique": state.unique,
        "primary": state.primary,
        "key_attributes": state.key_attributes,
        "total_attributes": state.total_attributes,
        "indexed_column": state.indexed_column,
        "opclass": state.opclass,
        "predicate": normalize_predicate(state.predicate),
        "reloptions": observed_options,
        "comment": state.comment,
    }
    return {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }


def validate_index_state(
    state: IndexState | None,
    *,
    table: RelationState,
    expected_contract: dict[str, object],
    role: str,
) -> IndexState:
    if state is None:
        raise PreparationError(f"{role} HNSW index is missing")
    diff = index_definition_diff(
        state, table=table, expected_contract=expected_contract
    )
    if diff:
        raise PreparationError(
            f"existing {role} HNSW index definition/provenance mismatch; "
            "refusing to drop or rebuild it: "
            + json.dumps(diff, sort_keys=True)
        )
    return state


def hnsw_create_sql(index_name: str, table: str) -> str:
    index_schema, index_relation = parse_qualified_name(index_name)
    table_schema, _ = parse_qualified_name(table)
    if index_schema != table_schema:
        raise PreparationError(
            "PostgreSQL creates an index in its table's schema; index and table schemas "
            "must match"
        )
    return (
        f"CREATE INDEX {quote_identifier(index_relation)} "
        f"ON {quote_qualified_name(table)} USING hnsw "
        f"(embedding vector_l2_ops) WITH (m = {HNSW_M}, "
        f"ef_construction = {HNSW_EF_CONSTRUCTION}) WHERE embedding_valid"
    )


def hnsw_comment_sql(index_name: str, comment: str) -> sql.Composed:
    index_schema, index_relation = parse_qualified_name(index_name)
    return sql.SQL("COMMENT ON INDEX {} IS {}").format(
        sql.Identifier(index_schema, index_relation),
        sql.Literal(comment),
    )


def build_source_index(
    conn: psycopg.Connection,
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    table: RelationState,
) -> tuple[IndexState, bool, dict[str, object]]:
    index_name = qualified_name(args.source_index)
    contract = source_build_contract(args, table)
    existing = index_state(cur, index_name)
    if existing is not None:
        return (
            validate_index_state(
                existing, table=table, expected_contract=contract, role="source"
            ),
            False,
            contract,
        )
    with conn.transaction():
        cur.execute("SET LOCAL statement_timeout = 0")
        cur.execute(
            "SELECT set_config('maintenance_work_mem', %s, true)",
            (args.maintenance_work_mem,),
        )
        cur.execute("SET LOCAL max_parallel_maintenance_workers = 0")
        cur.execute("SELECT set_config('hnsw.build_page_order', 'insertion', true)")
        cur.execute("SELECT set_config('hnsw.require_full_memory_build', 'off', true)")
        cur.execute("SELECT set_config('hnsw.clone_source', '', true)")
        cur.execute(
            "SELECT set_config('hnsw.build_seed', %s, true)",
            (str(args.build_seed),),
        )
        cur.execute(hnsw_create_sql(index_name, args.table))
        cur.execute(hnsw_comment_sql(index_name, provenance_comment(contract)))
    created = validate_index_state(
        index_state(cur, index_name),
        table=table,
        expected_contract=contract,
        role="source",
    )
    return created, True, contract


def build_clone_index(
    conn: psycopg.Connection,
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    table: RelationState,
    source: IndexState,
) -> tuple[IndexState, bool, dict[str, object]]:
    index_name = qualified_name(args.clone_index)
    contract = clone_build_contract(args, table, source)
    existing = index_state(cur, index_name)
    if existing is not None:
        return (
            validate_index_state(
                existing, table=table, expected_contract=contract, role="clone"
            ),
            False,
            contract,
        )
    with conn.transaction():
        cur.execute("SET LOCAL statement_timeout = 0")
        cur.execute(
            "SELECT set_config('maintenance_work_mem', %s, true)",
            (args.maintenance_work_mem,),
        )
        cur.execute("SET LOCAL max_parallel_maintenance_workers = 0")
        cur.execute("SELECT set_config('hnsw.require_full_memory_build', 'on', true)")
        cur.execute("SELECT set_config('hnsw.build_page_order', 'bfs', true)")
        cur.execute(
            "SELECT set_config('hnsw.clone_source', %s, true)",
            (qualified_name(args.source_index),),
        )
        cur.execute(hnsw_create_sql(index_name, args.table))
        cur.execute(hnsw_comment_sql(index_name, provenance_comment(contract)))
    created = validate_index_state(
        index_state(cur, index_name),
        table=table,
        expected_contract=contract,
        role="clone",
    )
    return created, True, contract


def exact_sqlens_provenance(
    cur: psycopg.Cursor, expected_build_id: str, expected_sha256: str
) -> dict[str, object]:
    try:
        cur.execute(
            "WITH lib AS (SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name = 'PKGLIBDIR') "
            "SELECT vector_sqlens_build_id(), path, "
            "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - identity gates must fail closed.
        raise PreparationError(
            "SQLens build ID/server vector.so SHA256 gate is unavailable"
        ) from exc
    if row is None:
        raise PreparationError("SQLens binary provenance query returned no row")
    build_id, path, observed_sha = (str(row[0]), str(row[1]), str(row[2]))
    if build_id != expected_build_id:
        raise PreparationError(
            f"SQLens build ID mismatch: expected={expected_build_id!r}, "
            f"observed={build_id!r}"
        )
    if observed_sha != expected_sha256:
        raise PreparationError(
            "server vector.so SHA256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha}"
        )
    if not path.endswith("/vector.so") or not SHA256_RE.fullmatch(observed_sha):
        raise PreparationError(
            f"invalid server vector.so identity: path={path!r}, sha256={observed_sha!r}"
        )
    return {
        "expected_sqlens_build_id": expected_build_id,
        "observed_sqlens_build_id": build_id,
        "expected_vector_so_sha256": expected_sha256,
        "observed_vector_so_sha256": observed_sha,
        "observed_vector_so_path": path,
        "exact_match": True,
        "checked_at": utc_now(),
    }


def hnsw_build_capabilities(cur: psycopg.Cursor) -> dict[str, str]:
    names = (
        "hnsw.build_page_order",
        "hnsw.require_full_memory_build",
        "hnsw.clone_source",
        "hnsw.build_seed",
    )
    observed: dict[str, str] = {}
    for name in names:
        cur.execute("SELECT current_setting(%s, true)", (name,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise PreparationError(f"required SQLens HNSW build GUC is unavailable: {name}")
        observed[name] = str(row[0])
    return observed


def acquire_advisory_lock(cur: psycopg.Cursor, table: str) -> None:
    cur.execute(
        "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
        (f"{ARTIFACT_CONTRACT}:{table}",),
    )
    row = cur.fetchone()
    if row is None or row[0] is not True:
        raise PreparationError("another valid-embedding index preparation owns the DB lock")


def release_advisory_lock(cur: psycopg.Cursor, table: str) -> None:
    cur.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
        (f"{ARTIFACT_CONTRACT}:{table}",),
    )


def verify_source(
    cur: psycopg.Cursor, args: argparse.Namespace, table: RelationState
) -> tuple[IndexState, dict[str, object]]:
    contract = source_build_contract(args, table)
    return (
        validate_index_state(
            index_state(cur, qualified_name(args.source_index)),
            table=table,
            expected_contract=contract,
            role="source",
        ),
        contract,
    )


def verify_clone(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    table: RelationState,
    source: IndexState,
) -> tuple[IndexState, dict[str, object]]:
    contract = clone_build_contract(args, table, source)
    return (
        validate_index_state(
            index_state(cur, qualified_name(args.clone_index)),
            table=table,
            expected_contract=contract,
            role="clone",
        ),
        contract,
    )


def preparation_payload(
    args: argparse.Namespace,
    *,
    binary: dict[str, object],
    capabilities: dict[str, str],
    column_report: dict[str, object],
    table: RelationState,
    source: IndexState | None,
    clone: IndexState | None,
    source_contract: dict[str, object] | None,
    clone_contract: dict[str, object] | None,
    source_created: bool = False,
    clone_created: bool = False,
) -> dict[str, object]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "prepared_at": utc_now(),
        "stage": args.stage,
        "binary_provenance": binary,
        "hnsw_build_capabilities": capabilities,
        "table": asdict(table),
        "column_hygiene": column_report,
        "indexes": {
            "source": None
            if source is None
            else {
                "state": asdict(source),
                "created_in_this_run": source_created,
                "definition_diff": index_definition_diff(
                    source, table=table, expected_contract=source_contract or {}
                ),
                "build_contract": source_contract,
            },
            "clone": None
            if clone is None
            else {
                "state": asdict(clone),
                "created_in_this_run": clone_created,
                "definition_diff": index_definition_diff(
                    clone, table=table, expected_contract=clone_contract or {}
                ),
                "build_contract": clone_contract,
            },
        },
        "legacy_indexes_not_reused_or_modified": sorted(LEGACY_INDEXES),
        "drop_policy": "never_drop_by_default_fail_closed_on_mismatch",
    }


def dry_run_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "dry_run": True,
        "database_connected": False,
        "input_files_read": False,
        "stage": args.stage,
        "table": qualified_name(args.table),
        "constraint": args.constraint_name,
        "source_index": qualified_name(args.source_index),
        "clone_index": qualified_name(args.clone_index),
        "source_create_sql": hnsw_create_sql(args.source_index, args.table),
        "clone_create_sql": hnsw_create_sql(args.clone_index, args.table),
        "hnsw": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
        "source_layout": "insertion",
        "clone_layout": "bfs",
        "clone_source": qualified_name(args.source_index),
        "clone_requires_full_memory_build": True,
        "proof_output": str(args.proof_output),
    }


def run(
    args: argparse.Namespace,
    *,
    connect: Callable[..., psycopg.Connection] = psycopg.connect,
) -> dict[str, object]:
    if args.dry_run:
        return dry_run_plan(args)

    conninfo = pg_config_from_env().conninfo
    with connect(conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        server_version_num = require_metadata_only_column_add(cur)
        binary = exact_sqlens_provenance(
            cur, args.expected_sqlens_build_id, args.expected_vector_so_sha256
        )
        capabilities = hnsw_build_capabilities(cur)
        capabilities["server_version_num"] = str(server_version_num)
        acquire_advisory_lock(cur, args.table)
        try:
            if args.stage in {"all", "column"}:
                column_report = ensure_embedding_valid_column(conn, cur, args)
            else:
                column_report = verify_embedding_valid_column(cur, args)
            table = relation_state(cur, args.table)

            source: IndexState | None = None
            clone: IndexState | None = None
            source_contract: dict[str, object] | None = None
            clone_contract: dict[str, object] | None = None
            source_created = False
            clone_created = False

            if args.stage in {"all", "source"}:
                source, source_created, source_contract = build_source_index(
                    conn, cur, args, table
                )
            elif args.stage in {"clone", "proof", "verify"}:
                source, source_contract = verify_source(cur, args, table)

            if args.stage in {"all", "clone"}:
                if source is None:
                    source, source_contract = verify_source(cur, args, table)
                clone, clone_created, clone_contract = build_clone_index(
                    conn, cur, args, table, source
                )
            elif args.stage in {"proof", "verify"}:
                if source is None:
                    source, source_contract = verify_source(cur, args, table)
                clone, clone_contract = verify_clone(cur, args, table, source)

            preparation = preparation_payload(
                args,
                binary=binary,
                capabilities=capabilities,
                column_report=column_report,
                table=table,
                source=source,
                clone=clone,
                source_contract=source_contract,
                clone_contract=clone_contract,
                source_created=source_created,
                clone_created=clone_created,
            )

            if args.stage in {"all", "clone", "proof", "verify"}:
                if source is None or clone is None:
                    raise PreparationError("proof stage requires both validated indexes")
                proof = require_d2_graph_proof(
                    cur,
                    qualified_name(args.source_index),
                    qualified_name(args.clone_index),
                )
                comparison = proof.get("comparison")
                if not isinstance(comparison, dict):
                    raise PreparationError(
                        "canonical D2 proof is missing its source/clone comparison"
                    )
                preparation["source_clone_graph_diff"] = comparison
                payload = {
                    **proof,
                    "artifact_valid": True,
                    "preparation": preparation,
                }
                if args.stage in {"all", "clone", "proof"}:
                    write_json_atomic(args.proof_output, payload)
                return payload
            return {
                "artifact_valid": True,
                "preparation": preparation,
            }
        finally:
            release_advisory_lock(cur, args.table)
            cur.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare resumable Amazon-10M nonzero-vector partial HNSW source/clone indexes"
        )
    )
    parser.add_argument("--table", type=qualified_name, default=DEFAULT_TABLE)
    parser.add_argument(
        "--source-index", type=qualified_name, default=DEFAULT_SOURCE_INDEX
    )
    parser.add_argument(
        "--clone-index", type=qualified_name, default=DEFAULT_CLONE_INDEX
    )
    parser.add_argument("--constraint-name", default=DEFAULT_CONSTRAINT)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument(
        "--expected-vector-so-sha256", type=sha256_value, required=True
    )
    parser.add_argument("--expected-rows", type=positive_int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--maintenance-work-mem", type=memory_setting, default="64GB"
    )
    parser.add_argument("--build-seed", type=nonnegative_int, default=57)
    parser.add_argument("--proof-output", type=Path, default=DEFAULT_PROOF_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not IDENTIFIER_RE.fullmatch(args.constraint_name):
        raise PreparationError("constraint name must be an unquoted PostgreSQL identifier")
    if qualified_name(args.source_index) == qualified_name(args.clone_index):
        raise PreparationError("source and clone index names must be different")
    table_schema, _ = parse_qualified_name(args.table)
    for role, index_name in (
        ("source", args.source_index),
        ("clone", args.clone_index),
    ):
        index_schema, _ = parse_qualified_name(index_name)
        if index_schema != table_schema:
            raise PreparationError(
                f"{role} index must be in the same schema as the table"
            )
    if DEFAULT_SOURCE_INDEX in LEGACY_INDEXES or DEFAULT_CLONE_INDEX in LEGACY_INDEXES:
        raise AssertionError("new default indexes must not reuse legacy index names")
    if not args.expected_sqlens_build_id.strip():
        raise PreparationError("expected SQLens build ID must not be empty")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        payload = run(args)
    except (PreparationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
