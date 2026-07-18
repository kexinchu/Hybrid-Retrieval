"""Independent PostgreSQL exact-truth audit for the Amazon 10M workload.

The module is deliberately import-safe: importing it never opens a database
connection.  The audit consumes a completed truth artifact, then checks a
small, fixed query sample against PostgreSQL itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRUTH_CSV = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv"
DEFAULT_TRUTH_MANIFEST = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal_manifest.json"
DEFAULT_FILTERS_CSV = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_OUT = ROOT / "results/hybrid_vector_db/amazon10m_exact_truth_postgres_audit.csv"
DEFAULT_MANIFEST = ROOT / "results/hybrid_vector_db/amazon10m_exact_truth_postgres_audit_manifest.json"
DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_VALIDITY_PREDICATE = "embedding_valid"
DEFAULT_QUERY_NOS = (0, 50, 99, 100, 150, 199)
AUDIT_VERSION = 1


@dataclass(frozen=True)
class TruthCell:
    query_no: int
    query_id: int
    filter_name: str
    predicate: str
    k: int
    topk_ids: tuple[int, ...]
    topk_distances: tuple[float, ...]
    topk_plus_one_ids: tuple[int, ...]
    topk_plus_one_distances: tuple[float, ...]
    kth_distance: float
    tie_tolerance: float
    strict_closer_count: int
    boundary_tied: bool
    self_excluded: bool


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(payload.encode("utf-8"))


def safe_sql_predicate(value: str) -> str:
    """Accept one SQL expression, never a statement or a multi-statement value."""
    predicate = str(value).strip()
    if not predicate:
        raise ValueError("SQL predicate must not be empty")
    if "\x00" in predicate:
        raise ValueError("SQL predicate contains NUL")
    if any(marker in predicate for marker in (";", "--", "/*", "*/", "\\x00")):
        raise ValueError("SQL predicate contains a statement separator or comment marker")
    if predicate.count("'") % 2:
        raise ValueError("SQL predicate has an unterminated string literal")
    depth = 0
    for char in predicate:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("SQL predicate has unbalanced parentheses")
    if depth:
        raise ValueError("SQL predicate has unbalanced parentheses")
    if re.search(r"\b(?:select|insert|update|delete|merge|drop|alter|create|grant|revoke|copy|call|execute)\b", predicate, re.I):
        raise ValueError("SQL predicate must be an expression, not a SQL statement")
    return predicate


def qualified_identifier(value: str) -> str:
    parts = str(value).split(".")
    if len(parts) not in (1, 2) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", part or "") for part in parts
    ):
        raise ValueError("table must be an unquoted table or schema.table identifier")
    return ".".join(parts)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_filter_specs(path: Path) -> list[dict[str, str]]:
    rows = _csv_rows(path)
    required = {"filter_name", "predicate"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"filter CSV missing columns: {sorted(missing)}")
    seen: set[str] = set()
    specs: list[dict[str, str]] = []
    for row in rows:
        name = str(row["filter_name"]).strip()
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate filter_name: {name!r}")
        predicate = safe_sql_predicate(row["predicate"])
        seen.add(name)
        specs.append({"filter_name": name, "predicate": predicate, "target_rate": str(row.get("target_rate", ""))})
    if not specs:
        raise ValueError(f"no filters loaded from {path}")
    return specs


def _manifest_hash(manifest: Mapping[str, Any], *keys: str) -> str | None:
    for container_key in ("source_hashes", "inputs", "outputs"):
        container = manifest.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, Mapping):
                value = value.get("sha256")
            if value:
                return str(value)
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, Mapping):
            value = value.get("sha256")
        if value:
            return str(value)
    return None


def _csv_ints(value: Any) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return ()
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _csv_floats(value: Any) -> tuple[float, ...]:
    if value is None or str(value).strip() == "":
        return ()
    result = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not all(math.isfinite(item) for item in result):
        raise ValueError("truth distances must be finite")
    return result


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _field(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def parse_truth_cell(row: Mapping[str, Any]) -> TruthCell:
    required = {"query_no", "query_id", "filter_name", "predicate", "k"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"truth row missing columns: {sorted(missing)}")
    k = int(row["k"])
    topk_ids = _csv_ints(_field(row, "exact_topk_ids", "exact_filtered_topk_ids", "result_ids"))
    topk_distances = _csv_floats(_field(row, "exact_topk_distances_sq", "exact_filtered_topk_distances_sq"))
    plus_ids = _csv_ints(_field(row, "exact_topk_plus_one_ids", "exact_filtered_topk_plus_one_ids"))
    plus_distances = _csv_floats(_field(row, "exact_topk_plus_one_distances_sq", "exact_filtered_topk_plus_one_distances_sq"))
    if len(topk_ids) != k or len(topk_distances) not in (0, k):
        raise ValueError(f"truth cell query_no={row['query_no']} has invalid top-k payload")
    kth = float(row.get("kth_distance_sq", topk_distances[-1] if topk_distances else "nan"))
    tolerance = float(row.get("tie_tolerance", "nan"))
    if not math.isfinite(kth) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"truth cell query_no={row['query_no']} has invalid kth/tie tolerance")
    return TruthCell(
        query_no=int(row["query_no"]), query_id=int(row["query_id"]),
        filter_name=str(row["filter_name"]), predicate=safe_sql_predicate(row["predicate"]), k=k,
        topk_ids=topk_ids, topk_distances=topk_distances, topk_plus_one_ids=plus_ids,
        topk_plus_one_distances=plus_distances, kth_distance=kth, tie_tolerance=tolerance,
        strict_closer_count=int(row.get("strict_closer_count", max(0, k - 1))),
        boundary_tied=_bool(row.get("boundary_tied", False)),
        self_excluded=_bool(row.get("self_excluded", False)),
    )


def load_truth_inputs(
    truth_csv: Path,
    truth_manifest: Path,
    filters_csv: Path,
    query_nos: Sequence[int] = DEFAULT_QUERY_NOS,
) -> tuple[dict[tuple[int, str], TruthCell], dict[str, Any]]:
    """Load only a complete, hash-bound truth artifact; reject partial inputs."""
    if not truth_csv.is_file() or not truth_manifest.is_file() or not filters_csv.is_file():
        raise ValueError("truth CSV, truth manifest, and filter CSV must all exist")
    try:
        source_manifest = json.loads(truth_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("truth manifest is unreadable") from exc
    if not isinstance(source_manifest, dict) or source_manifest.get("artifact_valid") is not True:
        raise ValueError("truth manifest must declare artifact_valid=true")
    truth_hash = sha256_file(truth_csv)
    declared_truth_hash = _manifest_hash(source_manifest, "truth_csv_sha256", "truth_csv")
    if declared_truth_hash != truth_hash:
        raise ValueError("truth CSV hash does not match its manifest")
    filters = read_filter_specs(filters_csv)
    filter_hash = sha256_file(filters_csv)
    declared_filter_hash = _manifest_hash(source_manifest, "filters_csv_sha256", "filters_csv")
    if declared_filter_hash is None or declared_filter_hash != filter_hash:
        raise ValueError("filter CSV hash does not match the truth manifest")
    wanted_queries = tuple(int(value) for value in query_nos)
    if len(set(wanted_queries)) != len(wanted_queries):
        raise ValueError("query sample contains duplicates")
    rows = _csv_rows(truth_csv)
    cells: dict[tuple[int, str], TruthCell] = {}
    filter_by_name = {item["filter_name"]: item for item in filters}
    for raw in rows:
        cell = parse_truth_cell(raw)
        if cell.query_no not in wanted_queries:
            continue
        spec = filter_by_name.get(cell.filter_name)
        if spec is None or spec["predicate"] != cell.predicate:
            raise ValueError(f"truth/filter predicate mismatch for {cell.filter_name}")
        key = (cell.query_no, cell.filter_name)
        if key in cells:
            raise ValueError(f"duplicate truth cell: {key}")
        if not cell.self_excluded:
            raise ValueError(f"truth cell is not self-excluded: {key}")
        cells[key] = cell
    expected = {(query_no, spec["filter_name"]) for query_no in wanted_queries for spec in filters}
    if set(cells) != expected:
        raise ValueError("truth CSV does not cover every sampled query/filter cell")
    return cells, {
        "truth_csv_sha256": truth_hash,
        "filters_csv_sha256": filter_hash,
        "truth_manifest_sha256": sha256_file(truth_manifest),
        "truth_manifest_artifact_valid": True,
        "query_nos": list(wanted_queries),
        "filter_names": [spec["filter_name"] for spec in filters],
        "expected_cells": len(expected),
    }


def build_exact_sql(table: str, predicate: str, k: int, candidate_validity_predicate: str = DEFAULT_VALIDITY_PREDICATE) -> str:
    table = qualified_identifier(table)
    predicate = safe_sql_predicate(predicate)
    candidate_validity_predicate = safe_sql_predicate(candidate_validity_predicate)
    if k <= 0:
        raise ValueError("k must be positive")
    return f"""WITH candidate_rows AS MATERIALIZED (
    SELECT v.id,
           vector_l2_squared_distance(v.embedding, %s::vector) AS distance_sq,
           (v.id <> %s) AS self_excluded,
           ({candidate_validity_predicate}) AS candidate_valid
    FROM {table} AS v
    WHERE ({predicate}) AND ({candidate_validity_predicate}) AND v.id <> %s
)
SELECT id, distance_sq, self_excluded, candidate_valid
FROM candidate_rows
WHERE self_excluded AND candidate_valid
ORDER BY distance_sq ASC, id ASC
LIMIT %s""".strip()


def decode_explain(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        value = value[0]
    if isinstance(value, Mapping) and isinstance(value.get("Plan"), Mapping):
        return dict(value["Plan"])
    if isinstance(value, Mapping) and value.get("Node Type"):
        return dict(value)
    raise ValueError("malformed EXPLAIN (FORMAT JSON) result")


def plan_node_types(plan: Any) -> list[str]:
    types: list[str] = []
    if isinstance(plan, Mapping):
        if plan.get("Node Type"):
            types.append(str(plan["Node Type"]))
        for value in plan.values():
            types.extend(plan_node_types(value))
    elif isinstance(plan, list):
        for value in plan:
            types.extend(plan_node_types(value))
    return types


def plan_index_names(plan: Any) -> list[str]:
    names: list[str] = []
    if isinstance(plan, Mapping):
        if plan.get("Index Name"):
            names.append(str(plan["Index Name"]))
        for value in plan.values():
            names.extend(plan_index_names(value))
    elif isinstance(plan, list):
        for value in plan:
            names.extend(plan_index_names(value))
    return names


def validate_plan_gate(explain_value: Any) -> dict[str, Any]:
    """Fail closed on malformed plans and every index/bitmap/HNSW access path."""
    plan = decode_explain(explain_value)
    node_types = plan_node_types(plan)
    index_names = plan_index_names(plan)
    if not node_types:
        raise RuntimeError("EXPLAIN plan has no node types")
    forbidden_nodes = [
        node for node in node_types
        if re.search(r"index|bitmap|hnsw", node, re.I)
    ]
    # Any Index Name field is evidence of an index access path, even when an
    # index was given an innocuous name such as "ann_path".
    forbidden_names = list(index_names)
    if forbidden_nodes or forbidden_names:
        raise RuntimeError(
            "exact plan gate rejected an index access path: "
            + json.dumps({"node_types": forbidden_nodes, "index_names": forbidden_names}, sort_keys=True)
        )
    if not any(re.search(r"seq scan", node, re.I) for node in node_types):
        raise RuntimeError("exact plan gate found no sequential scan")
    return {"passed": True, "node_types": node_types, "index_names": index_names}


def _truth_is_tie(cell: TruthCell) -> bool:
    if len(cell.topk_plus_one_distances) > cell.k:
        return cell.topk_plus_one_distances[cell.k] <= cell.kth_distance + cell.tie_tolerance
    return cell.boundary_tied


def _observed_rows(rows: Iterable[Sequence[Any]]) -> tuple[list[int], list[float], bool, bool]:
    ids: list[int] = []
    distances: list[float] = []
    self_ok = True
    valid_ok = True
    for row in rows:
        if len(row) < 2:
            raise ValueError("exact SQL returned a row shorter than id,distance")
        ids.append(int(row[0]))
        distances.append(float(row[1]))
        if len(row) >= 3:
            self_ok = self_ok and bool(row[2])
        if len(row) >= 4:
            valid_ok = valid_ok and bool(row[3])
    if not all(math.isfinite(distance) for distance in distances):
        raise ValueError("exact SQL returned a non-finite distance")
    return ids, distances, self_ok, valid_ok


def compare_truth_cell(cell: TruthCell | Mapping[str, Any], observed_rows: Iterable[Sequence[Any]], k: int | None = None) -> dict[str, Any]:
    if not isinstance(cell, TruthCell):
        cell = parse_truth_cell(cell)
    k = cell.k if k is None else int(k)
    ids, distances, self_ok, valid_ok = _observed_rows(observed_rows)
    enough = len(ids) >= k + 1
    unique = len(set(ids)) == len(ids)
    observed_ids = ids[:k]
    observed_distances = distances[:k]
    expected_distances = list(cell.topk_distances)
    distance_errors = [abs(left - right) for left, right in zip(observed_distances, expected_distances)]
    max_distance_error = max(distance_errors, default=float("inf")) if expected_distances else 0.0
    distance_limit = max(
        cell.tie_tolerance,
        max((abs(value) for value in expected_distances), default=0.0) * 1e-6,
        1e-7,
    )
    tied = _truth_is_tie(cell)
    if expected_distances:
        distance_passed = len(expected_distances) == k and len(distance_errors) == k and max_distance_error <= distance_limit
    elif tied:
        # A tie cannot be audited without the truth boundary's distance data.
        distance_passed = False
    else:
        distance_passed = len(observed_distances) == k and abs(observed_distances[-1] - cell.kth_distance) <= distance_limit
        max_distance_error = abs(observed_distances[-1] - cell.kth_distance) if observed_distances else float("inf")
    if tied:
        strict_count = min(max(cell.strict_closer_count, 0), k)
        strict_ids_passed = observed_ids[:strict_count] == list(cell.topk_ids[:strict_count])
        threshold = cell.kth_distance + cell.tie_tolerance
        tied_ids_passed = len(observed_distances) == k and all(distance <= threshold for distance in observed_distances)
        ids_passed = strict_ids_passed and tied_ids_passed
    else:
        ids_passed = observed_ids == list(cell.topk_ids)
    passed = enough and unique and self_ok and valid_ok and ids_passed and distance_passed
    return {
        "passed": passed,
        "tie_cell": tied,
        "truth_ids": list(cell.topk_ids),
        "observed_ids": observed_ids,
        "truth_distances_sq": list(cell.topk_distances),
        "observed_distances_sq": observed_distances,
        "max_distance_error": max_distance_error,
        "distance_error_limit": distance_limit,
        "distance_passed": distance_passed,
        "ids_passed": ids_passed,
        "self_excluded_passed": self_ok and cell.self_excluded,
        "candidate_validity_passed": valid_ok,
        "returned_rows": len(ids),
        "required_rows": k + 1,
    }


def set_local_exact_settings(cursor: Any) -> None:
    cursor.execute("SET LOCAL enable_indexscan = off")
    cursor.execute("SET LOCAL enable_bitmapscan = off")
    cursor.execute("SET LOCAL enable_indexonlyscan = off")
    cursor.execute("SET LOCAL enable_seqscan = on")


def fetch_database_metadata(cursor: Any, table: str) -> dict[str, Any]:
    qualified_identifier(table)
    cursor.execute(
        """SELECT current_setting('server_version')::text,
                      coalesce((SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'), '')::text,
                      c.oid::bigint, c.relfilenode::bigint
               FROM pg_catalog.pg_class AS c
              WHERE c.oid = to_regclass(%s)""",
        (table,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"PostgreSQL table does not exist: {table}")
    cursor.execute(
        "WITH lib AS (SELECT setting || '/vector.so' AS path "
        "FROM pg_config WHERE name = 'PKGLIBDIR') "
        "SELECT path, encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
    )
    binary_row = cursor.fetchone()
    if binary_row is None or not re.fullmatch(r"[0-9a-f]{64}", str(binary_row[1])):
        raise RuntimeError("could not bind the server vector.so binary identity")
    cursor.execute("SELECT to_regprocedure('vector_sqlens_build_id()') IS NOT NULL")
    build_function = cursor.fetchone()
    sqlens_build_id: str | None = None
    if build_function and build_function[0] is True:
        cursor.execute("SELECT vector_sqlens_build_id()")
        build_row = cursor.fetchone()
        if build_row is None or not str(build_row[0]):
            raise RuntimeError("SQLens build identity function returned no value")
        sqlens_build_id = str(build_row[0])
    return {
        "server_version": str(row[0]), "vector_version": str(row[1]),
        "table_oid": int(row[2]), "table_relfilenode": int(row[3]), "table": table,
        "vector_binary_path": str(binary_row[0]),
        "vector_binary_sha256": str(binary_row[1]),
        "sqlens_build_id": sqlens_build_id,
    }


def fetch_query_vectors(
    cursor: Any,
    table: str,
    query_ids: Sequence[int],
    candidate_validity_predicate: str,
) -> dict[int, str]:
    table = qualified_identifier(table)
    validity = safe_sql_predicate(candidate_validity_predicate)
    wanted = sorted({int(value) for value in query_ids})
    cursor.execute(
        f"SELECT id, embedding::text FROM {table} "
        f"WHERE id = ANY(%s::bigint[]) AND ({validity})",
        (wanted,),
    )
    observed = {int(row[0]): str(row[1]) for row in cursor.fetchall()}
    missing = sorted(set(wanted) - set(observed))
    if missing:
        raise RuntimeError(f"valid query vectors are missing: {missing}")
    return observed


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    if fields:
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_audit_manifest(
    rows: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    database: Mapping[str, Any],
    expected_cells: int,
    csv_payload: str,
    query_nos: Sequence[int] = DEFAULT_QUERY_NOS,
) -> dict[str, Any]:
    passed_cells = sum(row.get("passed") is True for row in rows)
    return {
        "artifact": "amazon10m_exact_truth_postgres_audit",
        "version": AUDIT_VERSION,
        "artifact_valid": bool(len(rows) == expected_cells and passed_cells == expected_cells),
        "expected_cells": int(expected_cells), "completed_cells": len(rows), "passed_cells": passed_cells,
        "source_hashes": dict(source_hashes), "database": dict(database),
        "query_sample": [int(value) for value in query_nos],
        "plan_contract": {
            "transaction_settings": ["enable_indexscan=off", "enable_bitmapscan=off", "enable_indexonlyscan=off", "enable_seqscan=on"],
            "forbidden_access": ["Index Scan", "Index Only Scan", "Bitmap", "HNSW"],
            "distance_function": "vector_l2_squared_distance",
        },
        "outputs": {"per_cell_csv_sha256": sha256_bytes(csv_payload.encode("utf-8"))},
    }


def validate_audit_manifest(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None = None) -> None:
    """Validate the publish contract; missing proof is invalid, never implicit success."""
    required = ("artifact_valid", "expected_cells", "completed_cells", "passed_cells", "source_hashes", "database", "plan_contract", "outputs")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise RuntimeError(f"audit manifest is missing required proof: {missing}")
    if manifest.get("artifact_valid") is not True:
        raise RuntimeError("audit manifest artifact_valid is not true")
    if int(manifest["expected_cells"]) <= 0 or int(manifest["completed_cells"]) != int(manifest["expected_cells"]):
        raise RuntimeError("audit manifest cell coverage is incomplete")
    if int(manifest["passed_cells"]) != int(manifest["expected_cells"]):
        raise RuntimeError("audit manifest contains failed cells")
    source_hashes = manifest["source_hashes"]
    if not isinstance(source_hashes, Mapping) or not all(
        re.fullmatch(r"[0-9a-f]{64}", str(source_hashes.get(key, "")))
        for key in ("truth_csv_sha256", "filters_csv_sha256")
    ):
        raise RuntimeError("audit manifest lacks truth/filter hash proof")
    database = manifest["database"]
    if not isinstance(database, Mapping) or not all(database.get(key) not in (None, "") for key in ("table_oid", "table_relfilenode", "server_version", "vector_version", "vector_binary_path")):
        raise RuntimeError("audit manifest lacks database identity proof")
    if not re.fullmatch(r"[0-9a-f]{64}", str(database.get("vector_binary_sha256", ""))):
        raise RuntimeError("audit manifest lacks server vector.so SHA256 proof")
    if rows is not None and any(row.get("passed") is not True for row in rows):
        raise RuntimeError("audit manifest cannot be valid with a failed cell")


def publish_audit(out: Path, manifest_path: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    payload = _render_csv(rows)
    expected = ((manifest.get("outputs") or {}).get("per_cell_csv_sha256"))
    if expected != sha256_bytes(payload.encode("utf-8")):
        raise RuntimeError("audit CSV hash does not match manifest")
    if manifest.get("artifact_valid") is True:
        validate_audit_manifest(manifest, rows)
    atomic_write_text(out, payload)
    atomic_write_text(manifest_path, json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")


def run_audit(
    truth_csv: Path = DEFAULT_TRUTH_CSV,
    truth_manifest: Path = DEFAULT_TRUTH_MANIFEST,
    filters_csv: Path = DEFAULT_FILTERS_CSV,
    out: Path = DEFAULT_OUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    table: str = DEFAULT_TABLE,
    candidate_validity_predicate: str = DEFAULT_VALIDITY_PREDICATE,
    query_nos: Sequence[int] = DEFAULT_QUERY_NOS,
) -> dict[str, Any]:
    cells, source_hashes = load_truth_inputs(truth_csv, truth_manifest, filters_csv, query_nos)
    candidate_validity_predicate = safe_sql_predicate(candidate_validity_predicate)
    qualified_identifier(table)
    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise RuntimeError("psycopg is required only when executing the PostgreSQL audit") from exc
    from common_pg import pg_config_from_env

    rows: list[dict[str, Any]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=False) as connection:
        cursor = connection.cursor()
        database = fetch_database_metadata(cursor, table)
        query_vectors = fetch_query_vectors(
            cursor,
            table,
            [cell.query_id for cell in cells.values()],
            candidate_validity_predicate,
        )
        connection.commit()
        for key in sorted(cells):
            cell = cells[key]
            sql = build_exact_sql(table, cell.predicate, cell.k, candidate_validity_predicate)
            started = time.perf_counter()
            try:
                cursor.execute("BEGIN")
                set_local_exact_settings(cursor)
                parameters = (
                    query_vectors[cell.query_id],
                    cell.query_id,
                    cell.query_id,
                    cell.k + 1,
                )
                cursor.execute("EXPLAIN (FORMAT JSON, COSTS OFF) " + sql, parameters)
                explain_row = cursor.fetchone()
                if not explain_row:
                    raise RuntimeError("EXPLAIN returned no row")
                gate = validate_plan_gate(explain_row[0])
                cursor.execute(sql, parameters)
                observed = cursor.fetchall()
                comparison = compare_truth_cell(cell, observed)
                connection.commit()
                rows.append({
                    "query_no": cell.query_no, "query_id": cell.query_id, "filter_name": cell.filter_name,
                    "passed": comparison["passed"], "tie_cell": comparison["tie_cell"],
                    "truth_ids": ",".join(map(str, comparison["truth_ids"])),
                    "observed_ids": ",".join(map(str, comparison["observed_ids"])),
                    "max_distance_error": comparison["max_distance_error"],
                    "distance_error_limit": comparison["distance_error_limit"],
                    "ids_passed": comparison["ids_passed"], "distance_passed": comparison["distance_passed"],
                    "self_excluded_passed": comparison["self_excluded_passed"],
                    "candidate_validity_passed": comparison["candidate_validity_passed"],
                    "plan_node_types": ";".join(gate["node_types"]), "plan_index_names": ";".join(gate["index_names"]),
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                })
            except BaseException as exc:
                connection.rollback()
                rows.append({
                    "query_no": cell.query_no, "query_id": cell.query_id, "filter_name": cell.filter_name,
                    "passed": False, "error": f"{exc.__class__.__name__}: {exc}",
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                })
        cursor.close()
    payload = _render_csv(rows)
    manifest = build_audit_manifest(rows, source_hashes, database, len(cells), payload, query_nos)
    publish_audit(out, manifest_path, rows, manifest)
    return manifest


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--candidate-validity-predicate", default=DEFAULT_VALIDITY_PREDICATE)
    parser.add_argument("--execute", action="store_true", help="connect to PostgreSQL and run the audit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if not args.execute:
        print("dry contract only: pass --execute to connect to PostgreSQL")
        return 0
    try:
        manifest = run_audit(
            truth_csv=args.truth_csv, truth_manifest=args.truth_manifest, filters_csv=args.filters_csv,
            out=args.out, manifest_path=args.manifest_path, table=args.table,
            candidate_validity_predicate=args.candidate_validity_predicate,
        )
    except Exception as exc:
        print(f"audit failed closed: {exc}")
        return 2
    return 0 if manifest.get("artifact_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
