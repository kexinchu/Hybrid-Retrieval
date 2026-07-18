"""Strict, real PostgreSQL/pgvector concurrency correctness harness.

The harness deliberately creates one uniquely named table per invocation.  It
does not use a model, a fake index, or a synthetic in-process result: every
guided result and every truth result comes from PostgreSQL on a backend-local
snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .common_pg import pg_config_from_env
except ImportError:  # pragma: no cover - direct script execution
    from common_pg import pg_config_from_env


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = ROOT / "results" / "hybrid_vector_db"
ISOLATIONS = ("read_committed", "repeatable_read")
GUIDANCE_KINDS = ("page", "bloom", "exact")
OPERATIONS = (
    "committed_insert",
    "rollback_insert",
    "predicate_crossing_update",
    "vector_update",
    "delete",
    "truncate_tid_reuse",
)
DURING_BUILD_OPERATIONS = set(OPERATIONS) - {"truncate_tid_reuse"}
PHASES = ("build_before_write", "write_during_fragment_build", "write_before_load")
QUERY_VECTORS = ("[0,0,0]", "[0.25,0.25,0.25]", "[1,1,1]")


class StrictCorrectnessFailure(RuntimeError):
    """Raised when a guided result is not exactly equal to SQL truth."""


@dataclass(frozen=True)
class CaseSpec:
    operation: str
    isolation: str
    guidance_kind: str
    phase: str
    repeat: int

    @property
    def scenario(self) -> str:
        return f"{self.operation}/{self.phase}"


@dataclass
class HarnessState:
    table: str
    index: str
    predicate: str
    atom: str
    query_k: int
    query_count: int
    rows: int
    epoch_before: int
    epoch_after: int | None = None
    observed_epoch_before: int | None = None
    observed_epoch_after: int | None = None
    guide_before: dict[str, Any] | None = None
    activation_error: str = ""
    build_profile: dict[str, Any] | None = None
    build_error: str = ""


def build_schedule_grid(
    *,
    isolations: Iterable[str] = ISOLATIONS,
    guidance_kinds: Iterable[str] = GUIDANCE_KINDS,
    operations: Iterable[str] = OPERATIONS,
    phases: Iterable[str] = PHASES,
    repeats: int = 1,
) -> list[CaseSpec]:
    """Return the deterministic regression grid used by the CLI.

    TRUNCATE cannot overlap a reader that already holds a relation lock:
    PostgreSQL correctly waits for that transaction.  TID reuse is therefore
    covered only by write-before-load; the remaining mutations exercise all
    three schedules.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    isolation_values = tuple(isolations)
    kind_values = tuple(guidance_kinds)
    operation_values = tuple(operations)
    phase_values = tuple(phases)
    unknown = (set(isolation_values) - set(ISOLATIONS)) | (set(kind_values) - set(GUIDANCE_KINDS))
    unknown |= set(operation_values) - set(OPERATIONS)
    unknown |= set(phase_values) - set(PHASES)
    if unknown:
        raise ValueError(f"unknown schedule value: {sorted(unknown)[0]}")

    grid: list[CaseSpec] = []
    for repeat in range(repeats):
        for isolation in isolation_values:
            for operation in operation_values:
                for phase in phase_values:
                    if operation == "truncate_tid_reuse" and phase != "write_before_load":
                        continue
                    if phase == "write_during_fragment_build" and operation not in DURING_BUILD_OPERATIONS:
                        continue
                    for kind in kind_values:
                        grid.append(CaseSpec(operation, isolation, kind, phase, repeat))
    return grid


def validate_result(result_ids: Iterable[int], truth_ids: Iterable[int]) -> None:
    """Apply the harness's strict set and ordered-top-k contract."""
    result = tuple(int(value) for value in result_ids)
    truth = tuple(int(value) for value in truth_ids)
    missing = tuple(value for value in truth if value not in result)
    if missing:
        raise StrictCorrectnessFailure(
            f"false negative: missing={list(missing)} result={list(result)} truth={list(truth)}"
        )
    if result != truth:
        raise StrictCorrectnessFailure(
            f"ordered mismatch: result={list(result)} truth={list(truth)}"
        )


def strict_failure_status(records: Iterable[dict[str, Any]]) -> int:
    """Return the process status required by strict correctness mode."""
    for row in records:
        if row.get("false_negative") or row.get("ordered_mismatch") or row.get("error"):
            return 1
    return 0


def validate_schedule_completeness(
    records: Iterable[dict[str, Any]], schedule: Iterable[CaseSpec], queries: int
) -> list[str]:
    """Validate exact case coverage before declaring a formal artifact valid."""
    expected = Counter(
        {
            (
                f"{spec.operation}-{spec.phase}-{spec.isolation}-"
                f"{spec.guidance_kind}-r{spec.repeat}"
            ): queries * (2 if spec.phase == "build_before_write" else 1)
            for spec in schedule
        }
    )
    rows = list(records)
    observed = Counter(str(row.get("case_id", "")) for row in rows if row.get("case_id"))
    errors: list[str] = []
    if observed != expected:
        missing = {key: count - observed.get(key, 0) for key, count in expected.items() if observed.get(key, 0) < count}
        extra = {key: count - expected.get(key, 0) for key, count in observed.items() if count > expected.get(key, 0)}
        errors.append(f"schedule coverage mismatch: missing={missing} extra={extra}")

    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not row.get("case_id"):
            errors.append(f"non-case record: scenario={row.get('scenario', '')} error={row.get('error', '')}")
            continue
        key = (
            str(row.get("case_id")),
            str(row.get("phase")),
            str(row.get("query_no")),
            str(row.get("backend_role")),
        )
        if key in keys:
            errors.append(f"duplicate result key: {key}")
        keys.add(key)
    return errors


def source_hash() -> str:
    digest = hashlib.sha256()
    source_root = ROOT / "third_party" / "pgvector-sqlens"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".c", ".h", ".sql"}
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _import_psycopg():
    try:
        import psycopg
        from psycopg import sql
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing psycopg. Install with: .venv/bin/python -m pip install 'psycopg[binary]'"
        ) from exc
    return psycopg, sql


def _json_profile(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def _error_text(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def _wait(barrier: threading.Barrier) -> None:
    try:
        barrier.wait()
    except threading.BrokenBarrierError as exc:
        raise RuntimeError("backend barrier broke after a peer failure") from exc


def _table_ident(sql: Any, table: str) -> Any:
    return sql.Identifier("public", table)


def _set_guided(cur: Any, sql: Any, rows: int) -> None:
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_sort = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute(f"SET hnsw.ef_search = {max(100, rows * 4)}")
    cur.execute(f"SET hnsw.max_scan_tuples = {max(10000, rows * 8)}")
    cur.execute("SET hnsw.iterative_scan = strict_order")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET hnsw.filter_strategy = safe_guided")


def _set_exact(cur: Any) -> None:
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_sort = on")
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_indexonlyscan = off")
    cur.execute("SET enable_bitmapscan = off")


def _begin(cur: Any, isolation: str) -> None:
    level = "READ COMMITTED" if isolation == "read_committed" else "REPEATABLE READ"
    cur.execute(f"BEGIN ISOLATION LEVEL {level}")


def _epoch(cur: Any, table: str) -> int:
    cur.execute(
        "SELECT epoch FROM public.pgvector_hnsw_fragment_epoch "
        "WHERE heap_oid = (%s::regclass)::oid",
        (f"public.{table}",),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _activate(cur: Any, table: str, index: str, atom: str, kind: str) -> dict[str, Any]:
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (f"public.{index}", [atom], kind),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    return _json_profile(cur.fetchone()[0])


def _query_statement(sql: Any, table: str, predicate: str, *, exact: bool) -> Any:
    order = sql.SQL("embedding <-> %s::vector, id") if exact else sql.SQL("embedding <-> %s::vector")
    binding = sql.SQL("") if exact else sql.SQL(
        "(SELECT vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) OFFSET 0) AND "
    )
    return sql.SQL("SELECT id FROM {} WHERE {}{} ORDER BY {} LIMIT %s").format(
        _table_ident(sql, table), binding, sql.SQL(predicate), order
    )


def _plan_index_names(plan: Any) -> list[str]:
    names: list[str] = []
    if isinstance(plan, dict):
        if plan.get("Index Name"):
            names.append(str(plan["Index Name"]))
        for value in plan.values():
            names.extend(_plan_index_names(value))
    elif isinstance(plan, list):
        for value in plan:
            names.extend(_plan_index_names(value))
    return names


def _run_sql_query(
    cur: Any,
    sql: Any,
    table: str,
    index: str,
    vector: str,
    predicate: str,
    k: int,
    *,
    exact: bool,
    atom: str = "",
    guidance_kind: str = "",
) -> list[int]:
    statement = _query_statement(sql, table, predicate, exact=exact)
    params = (vector, k) if exact else (f"public.{index}", [atom], guidance_kind, vector, k)
    if not exact:
        cur.execute(sql.SQL("EXPLAIN (FORMAT JSON) ") + statement, params)
        plan_row = cur.fetchone()
        names = _plan_index_names(plan_row[0] if plan_row else [])
        if index not in names:
            raise RuntimeError(f"guided query did not use {index}: index_names={names}")
    cur.execute(statement, params)
    return [int(row[0]) for row in cur.fetchall()]


def _query_pair(
    cur: Any,
    sql: Any,
    table: str,
    vector: str,
    predicate: str,
    k: int,
    rows: int,
    guide_before: dict[str, Any],
    epoch_before: int,
    epoch_after: int,
    phase: str,
    query_no: int,
    build_error: str,
    atom: str,
    guidance_kind: str,
) -> dict[str, Any]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    _set_guided(cur, sql, rows)
    guided_ids = _run_sql_query(
        cur,
        sql,
        table,
        f"{table}_hnsw",
        vector,
        predicate,
        k,
        exact=False,
        atom=atom,
        guidance_kind=guidance_kind,
    )
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    scan_profile = _json_profile(cur.fetchone()[0])
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    guide_after = _json_profile(cur.fetchone()[0])

    # Force a heap sort for exact SQL truth on the same backend snapshot.  Keep
    # the backend-local guide alive: the next safe-guided statement must be
    # able to observe and fail open on a committed epoch change.
    _set_exact(cur)
    truth_ids = _run_sql_query(
        cur, sql, table, f"{table}_hnsw", vector, predicate, k, exact=True
    )
    false_negative = any(value not in guided_ids for value in truth_ids)
    ordered_mismatch = tuple(guided_ids) != tuple(truth_ids)
    error = ""
    try:
        validate_result(guided_ids, truth_ids)
    except StrictCorrectnessFailure as exc:
        error = _error_text(exc)

    return {
        "phase": phase,
        "query_no": query_no,
        "query_vector": vector,
        "epoch_before": epoch_before,
        "epoch_after": epoch_after,
        "epoch_observed_before": guide_before.get("relation_epoch"),
        "epoch_observed_after": guide_after.get("relation_epoch"),
        "guide_active_before": bool(guide_before.get("active", False)),
        "guide_active_after": bool(guide_after.get("active", False)),
        "stale_bypass": bool(guide_before.get("active", False)) and not bool(guide_after.get("active", False)),
        "fragment_cache_hits": int(guide_before.get("fragment_cache_hits", 0) or 0),
        "fragment_cache_misses": int(guide_before.get("fragment_cache_misses", 0) or 0),
        "fragment_store_hits": int(guide_before.get("fragment_store_hits", 0) or 0),
        "fragment_builds": int(guide_before.get("fragment_builds", 0) or 0),
        "guide_build_ms": float(guide_before.get("last_cache_build_ms", 0.0) or 0.0),
        "scan_profile": scan_profile,
        "result_ids": guided_ids,
        "truth_ids": truth_ids,
        "result_count": len(guided_ids),
        "truth_count": len(truth_ids),
        "false_negative": false_negative,
        "ordered_mismatch": ordered_mismatch,
        "build_error": build_error,
        "error": error,
    }


def _insert_base(cur: Any, sql: Any, table: str, rows: int) -> None:
    table_id = _table_ident(sql, table)
    cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY").format(table_id))
    cur.execute(
        sql.SQL(
            "INSERT INTO {} (id, embedding, tenant_id, generation, note) "
            "SELECT i, CASE i "
            "WHEN 1 THEN '[0,0,0]'::vector "
            "WHEN 2 THEN '[0.0005,0,0]'::vector "
            "WHEN 3 THEN '[0.001,0,0]'::vector "
            "WHEN 4 THEN '[0.9,0.9,0.9]'::vector "
            "ELSE ARRAY[(i %% 97)::float / 97, (i %% 53)::float / 53, (i %% 31)::float / 31]::vector END, "
            "CASE WHEN i = 2 THEN 0 ELSE i %% 2 END, 0, 'base' "
            "FROM generate_series(1, %s) AS i"
        ).format(table_id),
        (rows,),
    )
    cur.execute(sql.SQL("ANALYZE {}").format(table_id))


def _apply_operation(cur: Any, sql: Any, table: str, operation: str, rows: int) -> None:
    table_id = _table_ident(sql, table)
    if operation in {"committed_insert", "rollback_insert"}:
        cur.execute(
            sql.SQL("INSERT INTO {} VALUES (%s, %s::vector, 1, 1, %s)").format(table_id),
            (rows + 1, "[0.0001,0,0]", operation),
        )
    elif operation == "predicate_crossing_update":
        cur.execute(sql.SQL("UPDATE {} SET tenant_id = 1, generation = generation + 1 WHERE id = 2").format(table_id))
    elif operation == "vector_update":
        cur.execute(
            sql.SQL("UPDATE {} SET embedding = %s::vector, generation = generation + 1 WHERE id = 4").format(table_id),
            ("[0.0001,0,0]",),
        )
    elif operation == "delete":
        cur.execute(sql.SQL("DELETE FROM {} WHERE id = 1").format(table_id))
    elif operation == "truncate_tid_reuse":
        cur.execute(sql.SQL("TRUNCATE TABLE {}").format(table_id))
        count = max(2, min(rows, 32))
        cur.execute(
            sql.SQL(
                "INSERT INTO {} (id, embedding, tenant_id, generation, note) "
                "SELECT i, ARRAY[(i - 1)::float / 1000, 0, 0]::vector, 1, 1, 'tid_reuse' "
                "FROM generate_series(1, %s) AS i"
            ).format(table_id),
            (count,),
        )
    else:  # pragma: no cover - grid validation catches this
        raise ValueError(f"unsupported operation: {operation}")


def _base_record(spec: CaseSpec, table: str, index: str, rows: int, predicate: str, atom: str) -> dict[str, Any]:
    return {
        "case_id": f"{spec.operation}-{spec.phase}-{spec.isolation}-{spec.guidance_kind}-r{spec.repeat}",
        "scenario": spec.operation,
        "phase": spec.phase,
        "operation": spec.operation,
        "isolation": spec.isolation,
        "guidance_kind": spec.guidance_kind,
        "filter_strategy": "safe_guided",
        "write_outcome": "rollback" if spec.operation == "rollback_insert" else "commit",
        "table": table,
        "index": index,
        "rows": rows,
        "predicate": predicate,
        "guidance_atom": atom,
        "backend_count": 2,
        "repeat": spec.repeat,
    }


def _reader_build_before_write(
    cfg: Any,
    sql: Any,
    spec: CaseSpec,
    state: HarnessState,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    barrier = threading.Barrier(2, timeout=120)
    shared: dict[str, Any] = {"writer_error": ""}

    def reader() -> None:
        conn = None
        try:
            import psycopg
            conn = psycopg.connect(cfg.conninfo)
            cur = conn.cursor()
            _begin(cur, spec.isolation)
            cur.execute("SELECT vector_hnsw_guidance_reset()")
            try:
                state.guide_before = _activate(cur, state.table, state.index, state.atom, spec.guidance_kind)
                state.observed_epoch_before = int(state.guide_before.get("relation_epoch", 0))
            except Exception as exc:
                state.activation_error = _error_text(exc)
                conn.rollback()
                cur = conn.cursor()
                _begin(cur, spec.isolation)
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(_table_ident(sql, state.table)))
            _wait(barrier)
            _wait(barrier)
            if spec.operation != "truncate_tid_reuse":
                for query_no, vector in enumerate(QUERY_VECTORS[: state.query_count]):
                    records.append(
                        _base_record(spec, state.table, state.index, state.rows, state.predicate, state.atom)
                        | {"backend_role": "reader", "query_no": query_no, "phase": "precommit"}
                        | _query_pair(
                            cur, sql, state.table, vector, state.predicate, state.query_k, state.rows,
                            state.guide_before or {}, state.epoch_before, state.epoch_before,
                            "precommit", query_no, state.activation_error, state.atom, spec.guidance_kind,
                        )
                    )
                _wait(barrier)
                _wait(barrier)
                state.epoch_after = int(shared.get("epoch_after", state.epoch_before))
                for query_no, vector in enumerate(QUERY_VECTORS[: state.query_count]):
                    records.append(
                        _base_record(spec, state.table, state.index, state.rows, state.predicate, state.atom)
                        | {"backend_role": "reader", "query_no": query_no, "phase": "postcommit"}
                        | _query_pair(
                            cur, sql, state.table, vector, state.predicate, state.query_k, state.rows,
                            state.guide_before or {}, state.epoch_before, state.epoch_after,
                            "postcommit", query_no, state.activation_error, state.atom, spec.guidance_kind,
                        )
                    )
            else:
                _wait(barrier)
                _wait(barrier)
                state.epoch_after = int(shared.get("epoch_after", state.epoch_before))
                for query_no, vector in enumerate(QUERY_VECTORS[: state.query_count]):
                    records.append(
                        _base_record(spec, state.table, state.index, state.rows, state.predicate, state.atom)
                        | {"backend_role": "reader", "query_no": query_no, "phase": "postcommit"}
                        | _query_pair(
                            cur, sql, state.table, vector, state.predicate, state.query_k, state.rows,
                            state.guide_before or {}, state.epoch_before, state.epoch_after,
                            "postcommit", query_no, state.activation_error, state.atom, spec.guidance_kind,
                        )
                    )
            conn.rollback()
        except Exception as exc:
            shared["reader_error"] = _error_text(exc)
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            if conn is not None:
                conn.close()

    def writer() -> None:
        conn = None
        try:
            import psycopg
            conn = psycopg.connect(cfg.conninfo)
            cur = conn.cursor()
            _begin(cur, spec.isolation)
            _wait(barrier)
            _apply_operation(cur, sql, state.table, spec.operation, state.rows)
            if spec.operation == "truncate_tid_reuse":
                if spec.operation == "rollback_insert":
                    conn.rollback()
                else:
                    conn.commit()
                cur = conn.cursor()
                shared["epoch_after"] = _epoch(cur, state.table)
                _wait(barrier)
                _wait(barrier)
            else:
                _wait(barrier)
                _wait(barrier)
                if spec.operation == "rollback_insert":
                    conn.rollback()
                else:
                    conn.commit()
                cur = conn.cursor()
                shared["epoch_after"] = _epoch(cur, state.table)
                _wait(barrier)
        except Exception as exc:
            shared["writer_error"] = _error_text(exc)
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            if conn is not None:
                conn.close()

    reader_thread = threading.Thread(target=reader, name="pgvector-reader")
    writer_thread = threading.Thread(target=writer, name="pgvector-writer")
    reader_thread.start()
    writer_thread.start()
    reader_thread.join()
    writer_thread.join()
    return shared


def _run_during_build(cfg: Any, sql: Any, spec: CaseSpec, state: HarnessState) -> list[dict[str, Any]]:
    barrier = threading.Barrier(2, timeout=120)
    shared: dict[str, Any] = {}

    def builder() -> None:
        conn = None
        try:
            import psycopg
            conn = psycopg.connect(cfg.conninfo)
            cur = conn.cursor()
            _begin(cur, spec.isolation)
            _wait(barrier)
            _wait(barrier)
            state.build_profile = _activate(cur, state.table, state.index, state.atom, spec.guidance_kind)
            state.observed_epoch_before = int(state.build_profile.get("relation_epoch", 0))
            conn.commit()
            shared["builder_done"] = True
            _wait(barrier)
        except Exception as exc:
            state.build_error = _error_text(exc)
            shared["builder_error"] = state.build_error
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            if conn is not None:
                conn.close()

    def writer() -> None:
        conn = None
        try:
            import psycopg
            conn = psycopg.connect(cfg.conninfo)
            cur = conn.cursor()
            _begin(cur, spec.isolation)
            _wait(barrier)
            _apply_operation(cur, sql, state.table, spec.operation, state.rows)
            _wait(barrier)
            _wait(barrier)
            if spec.operation == "rollback_insert":
                conn.rollback()
            else:
                conn.commit()
            shared["epoch_after"] = _epoch(conn.cursor(), state.table)
        except Exception as exc:
            shared["writer_error"] = _error_text(exc)
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            if conn is not None:
                conn.close()

    # The builder waits for the writer transaction's DML before activation, so
    # the writer transaction remains open for the complete fragment build.
    writer_thread = threading.Thread(target=writer, name="pgvector-writer")
    builder_thread = threading.Thread(target=builder, name="pgvector-builder")
    writer_thread.start()
    builder_thread.start()
    writer_thread.join()
    builder_thread.join()
    state.epoch_after = int(shared.get("epoch_after", state.epoch_before))
    if shared.get("builder_error") or shared.get("writer_error"):
        state.build_error = state.build_error or shared.get("builder_error", "") or shared.get("writer_error", "")

    import psycopg
    conn = psycopg.connect(cfg.conninfo)
    try:
        cur = conn.cursor()
        _begin(cur, spec.isolation)
        _set_guided(cur, sql, state.rows)
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        try:
            state.guide_before = _activate(cur, state.table, state.index, state.atom, spec.guidance_kind)
        except Exception as exc:
            state.activation_error = _error_text(exc)
            conn.rollback()
            cur = conn.cursor()
            _begin(cur, spec.isolation)
            state.guide_before = {}
        records = []
        for query_no, vector in enumerate(QUERY_VECTORS[: state.query_count]):
            records.append(
                _base_record(spec, state.table, state.index, state.rows, state.predicate, state.atom)
                | {"backend_role": "query", "query_no": query_no, "phase": "postcommit"}
                | _query_pair(
                    cur, sql, state.table, vector, state.predicate, state.query_k, state.rows,
                    state.guide_before, state.epoch_before, int(shared.get("epoch_after", state.epoch_before)),
                    "postcommit", query_no, state.build_error or state.activation_error,
                    state.atom, spec.guidance_kind,
                )
            )
        conn.rollback()
        return records
    finally:
        conn.close()


def _run_before_load(cfg: Any, sql: Any, spec: CaseSpec, state: HarnessState) -> list[dict[str, Any]]:
    import psycopg

    # Backend A builds and commits a fragment-store entry.  Backend B starts
    # only after the write commits and must reject the old epoch on load.
    builder = psycopg.connect(cfg.conninfo)
    try:
        cur = builder.cursor()
        _begin(cur, spec.isolation)
        state.build_profile = _activate(cur, state.table, state.index, state.atom, spec.guidance_kind)
        state.observed_epoch_before = int(state.build_profile.get("relation_epoch", 0))
        builder.commit()
    except Exception as exc:
        state.build_error = _error_text(exc)
        builder.rollback()
    finally:
        builder.close()

    barrier = threading.Barrier(2, timeout=120)
    shared: dict[str, Any] = {"records": []}

    def writer() -> None:
        writer_conn = None
        try:
            writer_conn = psycopg.connect(cfg.conninfo)
            cur = writer_conn.cursor()
            _begin(cur, spec.isolation)
            _apply_operation(cur, sql, state.table, spec.operation, state.rows)
            if spec.operation == "rollback_insert":
                writer_conn.rollback()
            else:
                writer_conn.commit()
            state.epoch_after = _epoch(writer_conn.cursor(), state.table)
            _wait(barrier)
        except Exception as exc:
            shared["writer_error"] = _error_text(exc)
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            if writer_conn is not None:
                writer_conn.close()

    def query_backend() -> None:
        query_conn = None
        try:
            query_conn = psycopg.connect(cfg.conninfo)
            _wait(barrier)
            cur = query_conn.cursor()
            _begin(cur, spec.isolation)
            cur.execute("SELECT vector_hnsw_guidance_reset()")
            try:
                state.guide_before = _activate(cur, state.table, state.index, state.atom, spec.guidance_kind)
            except Exception as exc:
                state.activation_error = _error_text(exc)
                query_conn.rollback()
                cur = query_conn.cursor()
                _begin(cur, spec.isolation)
                state.guide_before = {}
            for query_no, vector in enumerate(QUERY_VECTORS[: state.query_count]):
                row = _query_pair(
                    cur, sql, state.table, vector, state.predicate, state.query_k, state.rows,
                    state.guide_before, state.epoch_before, state.epoch_after or state.epoch_before,
                    "postcommit", query_no, state.build_error or state.activation_error,
                    state.atom, spec.guidance_kind,
                )
                row["stale_store_bypass"] = bool(
                    spec.guidance_kind in {"page", "bloom"}
                    and state.epoch_after != state.epoch_before
                    and not state.guide_before.get("fragment_store_hits", 0)
                )
                shared["records"].append(
                    _base_record(spec, state.table, state.index, state.rows, state.predicate, state.atom)
                    | {"backend_role": "query", "query_no": query_no}
                    | row
                )
            query_conn.rollback()
        except Exception as exc:
            shared["query_error"] = _error_text(exc)
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            if query_conn is not None:
                query_conn.close()

    writer_thread = threading.Thread(target=writer, name="pgvector-writer")
    query_thread = threading.Thread(target=query_backend, name="pgvector-query")
    writer_thread.start()
    query_thread.start()
    writer_thread.join()
    query_thread.join()
    records = list(shared["records"])
    if not records and (shared.get("writer_error") or shared.get("query_error")):
        records.append(
            _base_record(spec, state.table, state.index, state.rows, state.predicate, state.atom)
            | {"backend_role": "barrier", "error": shared.get("writer_error") or shared.get("query_error")}
        )
    return records


def run_case(cfg: Any, spec: CaseSpec, table: str, index: str, rows: int, queries: int, k: int) -> list[dict[str, Any]]:
    _, sql = _import_psycopg()
    predicate = "tenant_id = 1"
    atom = f"{spec.guidance_kind}:sql:{predicate}"
    state = HarnessState(table, index, predicate, atom, k, min(queries, len(QUERY_VECTORS)), rows, 0)

    setup = __import__("psycopg").connect(cfg.conninfo, autocommit=True)
    try:
        cur = setup.cursor()
        _insert_base(cur, sql, table, rows)
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (f"public.{table}",))
        state.epoch_before = _epoch(cur, table)
    finally:
        setup.close()

    if spec.phase == "build_before_write":
        shared = _reader_build_before_write(cfg, sql, spec, state, records := [])
        if shared.get("reader_error"):
            state.activation_error = state.activation_error or shared["reader_error"]
        if shared.get("writer_error"):
            state.activation_error = state.activation_error or shared["writer_error"]
    elif spec.phase == "write_during_fragment_build":
        records = _run_during_build(cfg, sql, spec, state)
    else:
        records = _run_before_load(cfg, sql, spec, state)

    for row in records:
        row["build_error"] = row.get("build_error") or state.build_error or state.activation_error
        row["error"] = row.get("error") or state.build_error or state.activation_error
        row["query_k"] = k
        row["epoch_after"] = state.epoch_after if row.get("phase") != "precommit" else state.epoch_before
        build_profile = state.build_profile or {}
        row["build_active"] = bool(build_profile.get("active", False))
        row["build_relation_epoch"] = build_profile.get("relation_epoch")
        row["build_fragment_store_hits"] = int(build_profile.get("fragment_store_hits", 0) or 0)
        row["build_fragment_builds"] = int(build_profile.get("fragment_builds", 0) or 0)
    return records


def _manifest(cfg: Any, cur: Any) -> dict[str, Any]:
    cur.execute("SHOW server_version")
    pg_version = str(cur.fetchone()[0])
    build_id = "unavailable"
    build_error = ""
    vector_library_path = "unavailable"
    vector_library_sha256 = "unavailable"
    vector_library_error = ""
    try:
        cur.execute("SELECT vector_sqlens_build_id()")
        build_id = str(cur.fetchone()[0])
    except Exception as exc:
        build_error = _error_text(exc)
        cur.connection.rollback()
    try:
        cur.execute(
            "SELECT setting || '/vector.so', "
            "encode(sha256(pg_read_binary_file(setting || '/vector.so')), 'hex') "
            "FROM pg_config WHERE name = 'PKGLIBDIR'"
        )
        library_row = cur.fetchone()
        if not library_row:
            raise RuntimeError("pg_config did not expose PKGLIBDIR")
        vector_library_path = str(library_row[0])
        vector_library_sha256 = str(library_row[1])
    except Exception as exc:
        vector_library_error = _error_text(exc)
        cur.connection.rollback()
    return {
        "build_id": build_id,
        "build_id_error": build_error,
        "vector_library_path": vector_library_path,
        "vector_library_sha256": vector_library_sha256,
        "vector_library_error": vector_library_error,
        "postgres_version": pg_version,
        "postgres_version_num": _server_version_num(cur),
        "source_hash_sha256": source_hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
    }


def validate_runtime_provenance(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    build_id = str(manifest.get("build_id", ""))
    library_sha = str(manifest.get("vector_library_sha256", ""))
    if manifest.get("build_id_error") or not build_id.startswith("sqlens-v11-"):
        errors.append("loaded SQLens build ID is absent or incompatible")
    if manifest.get("vector_library_error") or len(library_sha) != 64 or any(
        char not in "0123456789abcdef" for char in library_sha.lower()
    ):
        errors.append("loaded vector.so SHA-256 is absent or invalid")
    return errors


def _server_version_num(cur: Any) -> str:
    cur.execute("SHOW server_version_num")
    return str(cur.fetchone()[0])


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def _create_isolated_table(cfg: Any, table: str, index: str, rows: int) -> None:
    _, sql = _import_psycopg()
    import psycopg

    conn = psycopg.connect(cfg.conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        table_id = _table_ident(sql, table)
        # CREATE INDEX places the index in the table's schema; PostgreSQL does
        # not accept a schema-qualified index name in this syntax position.
        index_id = sql.Identifier(index)
        cur.execute(
            sql.SQL(
                "CREATE TABLE {} (id bigint PRIMARY KEY, embedding vector(3) NOT NULL, "
                "tenant_id integer NOT NULL, generation integer NOT NULL, note text NOT NULL)"
            ).format(table_id)
        )
        _insert_base(cur, sql, table, rows)
        cur.execute(sql.SQL("CREATE INDEX {} ON {} USING hnsw (embedding vector_l2_ops)").format(index_id, table_id))
        cur.execute(sql.SQL("ANALYZE {}").format(table_id))
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (f"public.{table}",))
    finally:
        conn.close()


def _drop_isolated_table(cfg: Any, table: str) -> None:
    _, sql = _import_psycopg()
    import psycopg

    conn = psycopg.connect(cfg.conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT (%s::regclass)::oid", (f"public.{table}",))
            row = cur.fetchone()
            if row:
                oid = int(row[0])
                cur.execute("DELETE FROM public.pgvector_hnsw_fragment_store WHERE heap_oid = %s", (oid,))
                cur.execute("DELETE FROM public.pgvector_hnsw_fragment_epoch WHERE heap_oid = %s", (oid,))
        except Exception:
            conn.rollback()
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(_table_ident(sql, table)))
    finally:
        conn.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(
    records: list[dict[str, Any]], manifest: dict[str, Any], validation_errors: list[str],
    out_json: Path, out_csv: Path,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    strict_failures = sum(
        1 for row in records
        if row.get("false_negative") or row.get("ordered_mismatch") or row.get("error")
    )
    artifact_valid = strict_failures == 0 and not validation_errors
    csv_records = [
        row
        | {
            "manifest_build_id": manifest.get("build_id", ""),
            "manifest_postgres_version": manifest.get("postgres_version", ""),
            "manifest_source_hash_sha256": manifest.get("source_hash_sha256", ""),
            "manifest_vector_library_sha256": manifest.get("vector_library_sha256", ""),
        }
        for row in records
    ]
    fieldnames = sorted(
        {key for row in csv_records for key in row}
        | {"manifest_build_id", "manifest_postgres_version", "manifest_source_hash_sha256",
           "manifest_vector_library_sha256"}
    )
    csv_tmp = out_csv.with_name(f".{out_csv.name}.{os.getpid()}.tmp")
    json_tmp = out_json.with_name(f".{out_json.name}.{os.getpid()}.tmp")
    try:
        with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in csv_records:
                flat = dict(row)
                for key, value in list(flat.items()):
                    if isinstance(value, (list, tuple, dict)):
                        flat[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
                writer.writerow(flat)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(csv_tmp, out_csv)

        payload = {
            "artifact": "pgvector_update_concurrency_correctness",
            "artifact_valid": artifact_valid,
            "status": "complete" if artifact_valid else "invalid",
            "validation_errors": validation_errors,
            "manifest": manifest,
            "outputs": {
                "csv": str(out_csv),
                "csv_sha256": _sha256_file(out_csv),
                "json": str(out_json),
            },
            "records": records,
            "summary": {
                "records": len(records),
                "strict_failures": strict_failures,
            },
        }
        with json_tmp.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(json_tmp, out_json)
    finally:
        for path in (csv_tmp, json_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--queries", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--guidance-kinds", nargs="+", choices=GUIDANCE_KINDS, default=list(GUIDANCE_KINDS))
    parser.add_argument("--isolations", nargs="+", choices=ISOLATIONS, default=list(ISOLATIONS))
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--dsn", default=None, help="psycopg conninfo; defaults to PG* environment variables")
    parser.add_argument("--dry-run", action="store_true", help="print the real schedule without connecting")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.rows < 8 or args.queries < 1 or args.queries > len(QUERY_VECTORS) or args.repeats < 1 or args.k < 1:
        raise SystemExit("rows >= 8, 1 <= queries <= 3, repeats >= 1, and k >= 1 are required")
    schedule = build_schedule_grid(
        isolations=args.isolations,
        guidance_kinds=args.guidance_kinds,
        operations=args.operations,
        phases=args.phases,
        repeats=args.repeats,
    )
    if args.dry_run:
        print(json.dumps([case.__dict__ for case in schedule], indent=2))
        return 0

    _, sql = _import_psycopg()
    import psycopg

    cfg = pgvector_config_from_env()
    if args.dsn:
        cfg = type("DsnConfig", (), {"conninfo": args.dsn})()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:10]
    table = f"pgv_conc_{os.getpid()}_{suffix}"
    index = f"{table}_hnsw"
    out_json = args.out_json or DEFAULT_RESULT_DIR / f"pgvector_update_concurrency_correctness_{stamp}.json"
    out_csv = args.out_csv or out_json.with_suffix(".csv")
    records: list[dict[str, Any]] = []
    conn = psycopg.connect(cfg.conninfo, autocommit=True)
    try:
        manifest = _manifest(cfg, conn.cursor())
    finally:
        conn.close()

    try:
        _create_isolated_table(cfg, table, index, args.rows)
        for position, spec in enumerate(schedule, start=1):
            print(f"[{position}/{len(schedule)}] {spec.scenario} {spec.isolation} {spec.guidance_kind}", flush=True)
            records.extend(run_case(cfg, spec, table, index, args.rows, args.queries, args.k))
    except Exception as exc:
        records.append({"scenario": "harness", "error": _error_text(exc), "traceback": traceback.format_exc()})
    finally:
        try:
            _drop_isolated_table(cfg, table)
        except Exception as exc:
            records.append({"scenario": "cleanup", "error": _error_text(exc)})

    validation_errors = validate_runtime_provenance(manifest)
    validation_errors.extend(validate_schedule_completeness(records, schedule, args.queries))
    manifest |= {
        "table": table,
        "index": index,
        "schedule_cases": len(schedule),
        "run_contract": {
            "rows": args.rows,
            "queries": args.queries,
            "repeats": args.repeats,
            "k": args.k,
            "guidance_kinds": list(args.guidance_kinds),
            "isolations": list(args.isolations),
            "operations": list(args.operations),
            "phases": list(args.phases),
        },
    }
    _write_outputs(records, manifest, validation_errors, out_json, out_csv)
    failures = strict_failure_status(records)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print(f"strict_failures={sum(1 for row in records if row.get('false_negative') or row.get('ordered_mismatch') or row.get('error'))}")
    return 1 if failures or validation_errors else 0


def pgvector_config_from_env() -> Any:
    """Keep --dsn override typed like the repository's common PgConfig."""
    return pg_config_from_env()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
