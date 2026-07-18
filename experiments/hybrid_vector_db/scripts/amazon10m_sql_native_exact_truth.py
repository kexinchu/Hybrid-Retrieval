from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .common_pg import pg_config_from_env, require_psycopg
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env, require_psycopg  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_QUERY_IDS = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_valid_embeddings_formal.csv"
DEFAULT_QUERY_COHORT_MANIFEST = (
    ROOT
    / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_valid_embeddings_formal_manifest.json"
)
DEFAULT_ARTIFACT_DIR = (
    ROOT / "results/hybrid_vector_db/amazon10m_sql_native_exact_truth_valid_embeddings"
)
DEFAULT_VECTOR_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_PRINCIPAL = "amazon10m_sql_native_benchmark"
DEFAULT_K = 10
DEFAULT_CALIBRATION_QUERIES = 100
DEFAULT_FINAL_QUERIES = 100
DEFAULT_BASE_TABLE_MAPPING_SAMPLE_SIZE = 1024
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
CHECKPOINT_VERSION = 4
QUERY_COHORT_PARSER_VERSION = 1
QUERY_COHORT_HASH_CONTRACT = (
    "sha256(canonical-json [[query_no,query_id,query_split],...]), ordered by query_no"
)
RELATION_EPOCH_HASH_CONTRACT = (
    "sha256(canonical-json {qualified_relation:data_epoch,...})"
)
FILTER_COLUMNS = (
    "rating",
    "verified_purchase",
    "helpful_vote",
    "review_text_len",
    "store",
    "main_category",
    "category_id",
    "price",
    "has_price",
    "item_avg_rating",
    "item_rating_number",
)
CANDIDATE_VALIDITY_COLUMNS = ("embedding_valid",)


def require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for exact truth computation") from exc
    return np


@dataclass(frozen=True)
class FilterSpec:
    name: str
    target_rate: str
    predicate: str
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    description: str
    bucket_pct: float
    temporal_kind: str


@dataclass(frozen=True)
class QueryCohort:
    query_ids: dict[int, int]
    query_splits: dict[int, str]
    sha256: str
    provenance: dict[str, Any]


WORKLOADS = (
    WorkloadSpec("acl_only", "product dimension plus RLS-derived principal ACL", 50.0, "none"),
    WorkloadSpec(
        "grant_temporal_selectivity",
        "derived benchmark grant validity at a real review-derived as_of",
        20.0,
        "grant",
    ),
    WorkloadSpec(
        "fact_temporal_selectivity",
        "source review timestamp validity at a real as_of",
        5.0,
        "fact",
    ),
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_candidate_validity_predicate(value: str) -> str:
    """Accept only the preregistered formal candidate-universe predicate."""
    predicate = str(value or "").strip()
    if not predicate:
        raise argparse.ArgumentTypeError("candidate validity predicate must not be empty")
    if predicate != DEFAULT_CANDIDATE_VALIDITY_PREDICATE:
        raise argparse.ArgumentTypeError(
            "candidate validity predicate must be exactly "
            f"{DEFAULT_CANDIDATE_VALIDITY_PREDICATE!r}"
        )
    return predicate


def candidate_universe_predicate_sha256(value: str) -> str:
    return sha256_text(validate_candidate_validity_predicate(value))


def workload_scalar_predicate_sha256(value: str) -> str:
    return sha256_text(str(value).strip())


def query_cohort_sha256(
    query_ids: dict[int, int], query_splits: dict[int, str]
) -> str:
    if set(query_ids) != set(query_splits):
        raise ValueError("query cohort IDs and split metadata have different keyspaces")
    records = [
        [int(query_no), int(query_ids[query_no]), str(query_splits[query_no])]
        for query_no in sorted(query_ids)
    ]
    return canonical_sha256(records)


def relation_epoch_contract(relations: dict[str, Any]) -> dict[str, Any]:
    if not relations:
        raise RuntimeError("relation epoch contract requires relation fingerprints")
    epochs: dict[str, int] = {}
    for relation, fingerprint in sorted(relations.items()):
        if not isinstance(fingerprint, dict):
            raise RuntimeError(f"relation epoch fingerprint is malformed: {relation}")
        epoch = fingerprint.get("data_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise RuntimeError(f"relation epoch is missing or invalid: {relation}")
        epochs[str(relation)] = epoch
    return {
        "hash_contract": RELATION_EPOCH_HASH_CONTRACT,
        "relations": epochs,
        "sha256": canonical_sha256(epochs),
    }


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


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


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def render_csv(rows: Sequence[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_text(path, render_csv(rows))


def publish_exact_artifact(
    out: Path,
    manifest_path: Path,
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    payload = render_csv(rows)
    expected_hash = (manifest.get("outputs") or {}).get("truth_csv_sha256")
    if manifest.get("artifact_valid") is not True:
        raise RuntimeError("exact-truth manifest must declare artifact_valid=true")
    if expected_hash != sha256_text(payload):
        raise RuntimeError("exact-truth CSV hash does not match the validated manifest")
    atomic_write_text(out, payload)
    atomic_write_json(manifest_path, manifest)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def role_name(value: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise argparse.ArgumentTypeError("principal must be a lowercase PostgreSQL role identifier")
    return value


def qualified_name(value: str) -> str:
    parts = value.split(".")
    if len(parts) not in (1, 2) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", part or "") for part in parts
    ):
        raise argparse.ArgumentTypeError("table must be an unquoted table or schema.table identifier")
    return ".".join(part.lower() for part in parts)


def read_filters(path: Path, selected: set[str] | None = None) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = str(row["filter_name"])
            if selected and name not in selected:
                continue
            if name in seen:
                raise ValueError(f"duplicate filter_name: {name}")
            specs.append(
                FilterSpec(
                    name=name,
                    target_rate=str(row["target_rate"]),
                    predicate=str(row["predicate"]).strip(),
                    expected_rows=int(str(row["count"])),
                    actual_pct=float(str(row["actual_pct"])),
                )
            )
            seen.add(name)
    if selected and selected - seen:
        raise ValueError(f"missing filters: {sorted(selected - seen)}")
    if not specs:
        raise ValueError(f"no filters loaded from {path}")
    return specs


def _load_query_source_manifest(
    path: Path,
    manifest_path: Path,
    candidate_validity_predicate: str,
    expected_splits: dict[int, str],
    source_rows: int,
    filter_predicates: dict[str, str],
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"query cohort provenance manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("query cohort provenance manifest must be a JSON object")
    outputs = manifest.get("outputs")
    truth_output = outputs.get("truth_csv") if isinstance(outputs, dict) else None
    validity = manifest.get("validity_contract")
    manifest_filters = manifest.get("filters")
    calibration = manifest.get("calibration")
    final = manifest.get("final")
    expected_calibration = sum(
        split == "calibration" for split in expected_splits.values()
    )
    expected_final = sum(split == "final" for split in expected_splits.values())
    if (
        manifest.get("artifact_valid") is not True
        or not isinstance(truth_output, dict)
        or truth_output.get("sha256") != sha256_file(path)
        or not isinstance(validity, dict)
        or validity.get("candidate_validity_predicate") != candidate_validity_predicate
        or validity.get("query_validity_predicate") != candidate_validity_predicate
        or validity.get("query_inherits_candidate") is not True
        or int(manifest.get("truth_rows", -1)) != source_rows
        or isinstance(manifest_filters, bool)
        or not isinstance(manifest_filters, int)
        or manifest_filters != len(filter_predicates)
        or not isinstance(calibration, dict)
        or int(calibration.get("queries", -1)) != expected_calibration
        or not isinstance(final, dict)
        or int(final.get("queries", -1)) != expected_final
        or manifest.get("query_ids_disjoint") is not True
        or manifest.get("self_excluded") is not True
        or manifest.get("method") != "exact_filtered_l2_tie_aware"
    ):
        raise ValueError("query cohort provenance manifest does not bind the CSV/validity universe")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "artifact_valid": True,
        "source_truth_csv_sha256": str(truth_output["sha256"]),
    }


def load_query_cohort(
    path: Path,
    expected_splits: dict[int, str],
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    *,
    source_manifest_path: Path | None = None,
    expected_filters: Sequence[FilterSpec] | None = None,
) -> QueryCohort:
    """Load either a dedicated cohort CSV or a formal truth CSV without lossy extraction."""
    candidate_validity_predicate = validate_candidate_validity_predicate(
        candidate_validity_predicate
    )
    if not expected_splits or any(
        split not in {"calibration", "final"} for split in expected_splits.values()
    ):
        raise ValueError("query cohort requires explicit calibration/final split metadata")
    truth_required = {
        "query_no",
        "query_id",
        "query_split",
        "filter_name",
        "predicate",
        "method",
        "self_excluded",
        "kth_distance_sq",
        "candidate_validity_predicate",
        "query_validity_predicate",
    }
    dedicated_required = {
        "query_no",
        "query_id",
        "query_split",
        "candidate_validity_predicate",
        "query_validity_predicate",
    }
    found: dict[int, int] = {}
    observed_filters: dict[int, set[str]] = {}
    observed_filter_predicates: dict[str, str] = {}
    seen_truth_keys: set[tuple[int, str]] = set()
    seen_dedicated_query_nos: set[int] = set()
    row_count = 0
    source_kind = ""
    source_fields: list[str]
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        source_fields = list(reader.fieldnames or ())
        field_set = set(source_fields)
        if truth_required <= field_set:
            source_kind = "formal_exact_truth_csv"
        elif field_set == dedicated_required:
            source_kind = "dedicated_query_cohort_csv"
        else:
            missing = sorted(truth_required - field_set)
            raise ValueError(
                "query cohort CSV has neither the dedicated schema nor the strict formal "
                f"truth schema: missing_truth_columns={missing}"
            )
        for row in reader:
            row_count += 1
            try:
                query_no = int(str(row["query_no"]))
                query_id = int(str(row["query_id"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"query cohort has a non-integer query mapping at row={row_count}") from exc
            if query_no not in expected_splits:
                raise ValueError(f"query cohort has unexpected query_no={query_no}")
            expected_split = expected_splits[query_no]
            if row.get("query_split") != expected_split:
                raise ValueError(
                    f"query_no={query_no} has query_split={row.get('query_split')!r}; "
                    f"expected={expected_split!r}"
                )
            if (
                row.get("candidate_validity_predicate") != candidate_validity_predicate
                or row.get("query_validity_predicate") != candidate_validity_predicate
            ):
                raise ValueError(
                    f"query_no={query_no} is outside the requested candidate/query validity universe"
                )
            previous = found.setdefault(query_no, query_id)
            if previous != query_id:
                raise ValueError(f"query_no={query_no} maps to multiple query IDs")
            if source_kind == "formal_exact_truth_csv":
                filter_name = str(row.get("filter_name", ""))
                predicate = str(row.get("predicate", "")).strip()
                key = (query_no, filter_name)
                try:
                    distance = float(str(row.get("kth_distance_sq", "")))
                except ValueError as exc:
                    raise ValueError(f"query truth row has invalid kth_distance_sq: {key}") from exc
                if (
                    not filter_name
                    or not predicate
                    or key in seen_truth_keys
                    or row.get("method") != "pre_filter_exact"
                    or str(row.get("self_excluded", "")).strip().lower() != "true"
                    or not (distance >= 0.0 and distance < float("inf"))
                ):
                    raise ValueError(f"query truth row violates the formal extraction contract: {key}")
                seen_truth_keys.add(key)
                observed_filters.setdefault(query_no, set()).add(filter_name)
                previous_predicate = observed_filter_predicates.setdefault(
                    filter_name, predicate
                )
                if previous_predicate != predicate:
                    raise ValueError(
                        f"query truth filter has inconsistent predicates: {filter_name}"
                    )
            elif query_no in seen_dedicated_query_nos:
                raise ValueError(f"dedicated query cohort has duplicate query_no={query_no}")
            else:
                seen_dedicated_query_nos.add(query_no)
    wanted = set(expected_splits)
    if set(found) != wanted:
        raise ValueError(f"query cohort is incomplete: missing={sorted(wanted - set(found))}")
    if len(set(found.values())) != len(found):
        raise ValueError("query IDs must be unique across calibration and final splits")
    if source_kind == "formal_exact_truth_csv":
        if source_manifest_path is None:
            raise ValueError("truth-format query cohort requires an explicit provenance manifest")
        if expected_filters is None:
            raise ValueError("truth-format query cohort requires expected filter predicates")
        expected_filter_predicates = {
            spec.name: spec.predicate for spec in expected_filters
        }
        if len(expected_filter_predicates) != len(expected_filters):
            raise ValueError("expected query cohort filters contain duplicate names")
        filter_sets = {tuple(sorted(values)) for values in observed_filters.values()}
        if len(observed_filters) != len(found) or len(filter_sets) != 1:
            raise ValueError("truth-format query cohort does not have one complete filter keyspace")
        if observed_filter_predicates != expected_filter_predicates:
            raise ValueError(
                "truth-format query cohort filter names/predicates do not match the workload"
            )
        source_manifest = _load_query_source_manifest(
            path,
            source_manifest_path,
            candidate_validity_predicate,
            expected_splits,
            row_count,
            expected_filter_predicates,
        )
    else:
        source_manifest = None
    ordered = dict(sorted(found.items()))
    ordered_splits = {query_no: expected_splits[query_no] for query_no in ordered}
    cohort_hash = query_cohort_sha256(ordered, ordered_splits)
    provenance = {
        "parser": "amazon10m_query_cohort_csv",
        "parser_version": QUERY_COHORT_PARSER_VERSION,
        "source_kind": source_kind,
        "source_csv": str(path.resolve()),
        "source_csv_sha256": sha256_file(path),
        "source_schema_sha256": canonical_sha256(source_fields),
        "source_rows": row_count,
        "source_manifest": source_manifest,
        "query_count": len(ordered),
        "query_cohort_sha256": cohort_hash,
        "query_cohort_hash_contract": QUERY_COHORT_HASH_CONTRACT,
        "candidate_universe_predicate": candidate_validity_predicate,
        "candidate_universe_predicate_sha256": candidate_universe_predicate_sha256(
            candidate_validity_predicate
        ),
        "source_filter_predicates_sha256": (
            canonical_sha256(observed_filter_predicates)
            if source_kind == "formal_exact_truth_csv"
            else None
        ),
    }
    return QueryCohort(ordered, ordered_splits, cohort_hash, provenance)


def load_query_ids(
    path: Path,
    calibration_queries: int,
    final_queries: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    *,
    source_manifest_path: Path | None = None,
) -> dict[int, int]:
    expected_splits = {
        query_no: query_split(query_no, calibration_queries)
        for query_no in range(calibration_queries + final_queries)
    }
    return load_query_cohort(
        path,
        expected_splits,
        candidate_validity_predicate,
        source_manifest_path=source_manifest_path,
    ).query_ids


def query_split(query_no: int, calibration_queries: int) -> str:
    return "calibration" if query_no < calibration_queries else "final"


def qualify_predicate(predicate: str, alias: str = "v") -> str:
    result = predicate
    for column in sorted(FILTER_COLUMNS, key=len, reverse=True):
        result = re.sub(rf"(?<![A-Za-z0-9_$.]){re.escape(column)}\b", f"{alias}.{column}", result)
    return result


def qualify_candidate_validity_predicate(predicate: str, alias: str = "v") -> str:
    result = validate_candidate_validity_predicate(predicate)
    for column in CANDIDATE_VALIDITY_COLUMNS:
        result = re.sub(
            rf"(?<![A-Za-z0-9_$.]){re.escape(column)}\b",
            f"{alias}.{column}",
            result,
        )
    return result


def temporal_predicate(workload: WorkloadSpec) -> str:
    if workload.temporal_kind == "none":
        return ""
    if workload.temporal_kind == "grant":
        return """
  AND grant_row.valid_from <= %(as_of)s
  AND (grant_row.valid_to IS NULL OR grant_row.valid_to > %(as_of)s)"""
    if workload.temporal_kind == "fact":
        return """
  AND fact.valid_from <= %(as_of)s
  AND (fact.valid_to IS NULL OR fact.valid_to > %(as_of)s)"""
    raise ValueError(f"unknown temporal workload kind: {workload.temporal_kind}")


def build_candidate_sql(
    table: str,
    predicate: str,
    workload: WorkloadSpec,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """The one unbounded, vector-free relational candidate export per pair."""
    return f"""
SELECT v.id
FROM {qualified_name(table)} AS v
JOIN public.amazon_review_facts AS fact
  ON fact.review_id = v.id
JOIN public.amazon_product_dim AS product
  ON product.parent_asin = fact.parent_asin
JOIN public.amazon_principal_tenant_grants AS grant_row
  ON grant_row.tenant_id = product.tenant_id
WHERE ({qualify_predicate(predicate)})
  AND ({qualify_candidate_validity_predicate(candidate_validity_predicate)})
  AND grant_row.principal_name = CURRENT_USER::text
  AND grant_row.can_read
{temporal_predicate(workload)}
ORDER BY v.id
""".strip()


def build_spot_check_sql(
    table: str,
    predicate: str,
    workload: WorkloadSpec,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """Exact PostgreSQL reference query. MATERIALIZED prevents an ANN vector scan."""
    return f"""
WITH query_vector AS (
    SELECT id AS query_id, embedding
    FROM {qualified_name(table)} AS query_row
    WHERE query_row.id = %(query_id)s
      AND ({qualify_candidate_validity_predicate(candidate_validity_predicate, 'query_row')})
), valid AS MATERIALIZED (
    SELECT v.id, v.embedding
    FROM {qualified_name(table)} AS v
    JOIN public.amazon_review_facts AS fact
      ON fact.review_id = v.id
    JOIN public.amazon_product_dim AS product
      ON product.parent_asin = fact.parent_asin
    JOIN public.amazon_principal_tenant_grants AS grant_row
      ON grant_row.tenant_id = product.tenant_id
    CROSS JOIN query_vector
    WHERE ({qualify_predicate(predicate)})
      AND ({qualify_candidate_validity_predicate(candidate_validity_predicate)})
      AND v.id <> query_vector.query_id
      AND grant_row.principal_name = CURRENT_USER::text
      AND grant_row.can_read
{temporal_predicate(workload)}
)
SELECT valid.id, valid.embedding <-> query_vector.embedding AS distance
FROM valid
CROSS JOIN query_vector
ORDER BY distance, valid.id
LIMIT %(limit)s
""".strip()


def validate_exact_sql_text(sql_text: str) -> None:
    normalized = " ".join(sql_text.lower().split())
    forbidden = [token for token in ("hnsw", "guidance", "vector_hnsw") if token in normalized]
    if forbidden:
        raise RuntimeError(f"exact SQL contains approximate marker(s): {forbidden}")


def validate_candidate_sql(sql_text: str) -> None:
    validate_exact_sql_text(sql_text)
    normalized = " ".join(sql_text.lower().split())
    if " limit " in f" {normalized} ":
        raise RuntimeError("candidate export SQL must never use LIMIT")
    required = (
        "join public.amazon_review_facts",
        "join public.amazon_product_dim",
        "join public.amazon_principal_tenant_grants",
        "current_user",
        "order by v.id",
    )
    missing = [token for token in required if token not in normalized]
    if missing:
        raise RuntimeError(f"candidate SQL misses relational contract: {missing}")


def read_fbin_memmap(path: Path) -> tuple[np.memmap, int, int]:
    np = require_numpy()
    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid fbin header: {path}")
    rows, dimensions = np.frombuffer(header, dtype="<i4")
    if int(rows) <= 0 or int(dimensions) <= 0:
        raise ValueError(f"invalid fbin dimensions: {path}")
    vectors = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(int(rows), int(dimensions)))
    return vectors, int(rows), int(dimensions)


def parse_vector_text(value: str) -> np.ndarray:
    np = require_numpy()
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("unexpected PostgreSQL vector text")
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def deterministic_base_table_sample_ids(vector_rows: int, sample_size: int) -> list[int]:
    """Evenly span [0, vector_rows) without relying on database row order."""
    if vector_rows <= 0 or sample_size <= 0:
        raise ValueError("vector_rows and sample_size must be positive")
    count = min(vector_rows, sample_size)
    if count == 1:
        return [0]
    return [(position * (vector_rows - 1)) // (count - 1) for position in range(count)]


def base_table_mapping_ids(
    vector_rows: int, sample_size: int, query_ids: Iterable[int]
) -> tuple[list[int], list[int]]:
    base_sample_ids = deterministic_base_table_sample_ids(vector_rows, sample_size)
    query_id_set = {int(query_id) for query_id in query_ids}
    outside_fbin = sorted(query_id for query_id in query_id_set if query_id < 0 or query_id >= vector_rows)
    if outside_fbin:
        raise RuntimeError(f"query ID is outside fbin row space: {outside_fbin[:10]}")
    return base_sample_ids, sorted(set(base_sample_ids) | query_id_set)


def validate_vector_mapping(
    vectors: np.ndarray, checked_ids: Sequence[int], observed: dict[int, np.ndarray]
) -> dict[str, Any]:
    """Fail closed unless every selected PostgreSQL row is its fbin float32 row."""
    np = require_numpy()
    missing = sorted(set(checked_ids) - set(observed))
    if missing:
        raise RuntimeError(f"PostgreSQL is missing base-table mapping IDs: {missing[:10]}")
    maximum_error = 0.0
    for vector_id in checked_ids:
        database_vector = observed[vector_id]
        fbin_vector = np.asarray(vectors[vector_id], dtype=np.float32)
        if database_vector.shape != fbin_vector.shape:
            raise RuntimeError(
                f"PostgreSQL/fbin dimension mismatch at id={vector_id}: "
                f"database={database_vector.shape} fbin={fbin_vector.shape}"
            )
        maximum_error = max(maximum_error, float(np.max(np.abs(database_vector - fbin_vector))))
        if not np.allclose(database_vector, fbin_vector, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"PostgreSQL/fbin vector mismatch at id={vector_id}")
    return {
        "checked_rows": len(checked_ids), "comparison": "float32_allclose",
        "rtol": 1e-6, "atol": 1e-7, "max_abs_error": maximum_error,
    }


def verify_base_table_vector_mapping(
    conn: Any,
    table: str,
    vectors: np.ndarray,
    query_ids: dict[int, int],
    sample_size: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[str, Any]:
    """Audit base-table/fbin mapping before any relational candidate or Faiss work."""
    base_sample_ids, checked_ids = base_table_mapping_ids(len(vectors), sample_size, query_ids.values())
    cur = conn.cursor()
    try:
        cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        cur.execute(
            f"SELECT id, embedding::text FROM {qualified_name(table)} WHERE id = ANY(%s::bigint[])",
            (checked_ids,),
        )
        observed = {int(row[0]): parse_vector_text(str(row[1])) for row in cur.fetchall()}
        query_universe = verify_query_candidate_universe(
            cur,
            table,
            query_ids,
            candidate_validity_predicate,
        )
        cur.execute("COMMIT")
    except BaseException:
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        cur.close()
    return {
        "base_sample_size_requested": sample_size,
        "base_sample_ids": base_sample_ids,
        "base_sample_ids_sha256": canonical_sha256(base_sample_ids),
        "query_ids_included": sorted(int(query_id) for query_id in query_ids.values()),
        "checked_ids": checked_ids,
        "checked_ids_sha256": canonical_sha256(checked_ids),
        "query_candidate_universe": query_universe,
        **validate_vector_mapping(vectors, checked_ids, observed),
    }


def verify_query_candidate_universe(
    cur: Any,
    table: str,
    query_ids: dict[int, int],
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[str, Any]:
    predicate = validate_candidate_validity_predicate(candidate_validity_predicate)
    expected_ids = sorted(int(query_id) for query_id in query_ids.values())
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError("query candidate-universe proof requires distinct query IDs")
    cur.execute(
        f"SELECT query_row.id FROM {qualified_name(table)} AS query_row "
        "WHERE query_row.id = ANY(%s::bigint[]) "
        f"AND ({qualify_candidate_validity_predicate(predicate, 'query_row')}) "
        "ORDER BY query_row.id",
        (expected_ids,),
    )
    observed_ids = [int(row[0]) for row in cur.fetchall()]
    if observed_ids != expected_ids:
        raise RuntimeError(
            "query cohort is not fully contained in the current candidate-validity universe: "
            f"missing={sorted(set(expected_ids) - set(observed_ids))[:10]}"
        )
    return {
        "valid": True,
        "predicate": predicate,
        "predicate_sha256": candidate_universe_predicate_sha256(predicate),
        "query_count": len(expected_ids),
        "query_ids_sha256": canonical_sha256(expected_ids),
    }


def _update_topk(
    top_distances: np.ndarray,
    top_ids: np.ndarray,
    distances: np.ndarray,
    candidate_ids: np.ndarray,
) -> None:
    np = require_numpy()
    keep = top_distances.shape[1]
    for position in range(distances.shape[1]):
        current_distances = np.concatenate((top_distances[position], distances[:, position]))
        current_ids = np.concatenate((top_ids[position], candidate_ids))
        finite = np.isfinite(current_distances) & (current_ids >= 0)
        current_distances = current_distances[finite]
        current_ids = current_ids[finite]
        take = min(keep, len(current_ids))
        if take:
            # argpartition reduces the sorting work; a full lexsort of its boundary
            # candidates preserves deterministic id ordering for float32 ties.
            threshold = np.partition(current_distances, take - 1)[take - 1]
            positions = np.flatnonzero(current_distances <= threshold)
            order = np.lexsort((current_ids[positions], current_distances[positions]))[:take]
            chosen = positions[order]
            top_distances[position, :take] = current_distances[chosen]
            top_ids[position, :take] = current_ids[chosen]
        if take < keep:
            top_distances[position, take:] = np.inf
            top_ids[position, take:] = -1


def exact_topk_batched(
    vectors: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    chunk_rows: int,
    query_batch_size: int,
    *,
    progress_label: str = "",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Direct float32 squared-L2 scan without a candidate_count x q200 x dim tensor."""
    np = require_numpy()
    if k <= 0 or chunk_rows <= 0 or query_batch_size <= 0:
        raise ValueError("k, chunk_rows, and query_batch_size must be positive")
    query_ids = np.asarray(query_ids, dtype=np.int64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    retained = k + 1
    top_distances = np.full((len(query_ids), retained), np.inf, dtype=np.float32)
    top_ids = np.full((len(query_ids), retained), -1, dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, len(candidate_ids), chunk_rows):
        ids = candidate_ids[start : start + chunk_rows]
        chunk_vectors = np.asarray(vectors[ids], dtype=np.float32)
        for query_start in range(0, len(query_ids), query_batch_size):
            query_stop = min(query_start + query_batch_size, len(query_ids))
            queries = np.asarray(vectors[query_ids[query_start:query_stop]], dtype=np.float32)
            differences = chunk_vectors[:, None, :] - queries[None, :, :]
            distances = np.einsum("cqd,cqd->cq", differences, differences, dtype=np.float32)
            for local_position, query_id in enumerate(query_ids[query_start:query_stop]):
                distances[ids == query_id, local_position] = np.inf
            _update_topk(
                top_distances[query_start:query_stop],
                top_ids[query_start:query_stop],
                distances,
                ids,
            )
        if progress_label and (start // chunk_rows + 1) % 25 == 0:
            print(f"{progress_label} scanned={min(start + chunk_rows, len(candidate_ids))}/{len(candidate_ids)}", flush=True)
    return top_ids, top_distances, (time.perf_counter() - started) * 1000.0


def available_cpu_count() -> int:
    """Return the CPUs this process is allowed to use, not host-wide CPU count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def require_faiss() -> Any:
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Faiss is required for formal exact truth; install faiss-cpu before --execute"
        ) from exc
    if not hasattr(faiss, "IndexFlatL2") or not hasattr(faiss, "omp_set_num_threads"):
        raise RuntimeError("installed Faiss does not provide the required IndexFlatL2/OpenMP API")
    return faiss


def configure_exact_backend(args: argparse.Namespace) -> dict[str, Any]:
    if args.faiss_threads > available_cpu_count():
        raise ValueError(
            f"--faiss-threads={args.faiss_threads} exceeds this process's CPU affinity "
            f"({available_cpu_count()})"
        )
    if args.backend == "numpy":
        return {
            "backend": "numpy",
            "class": "numpy_reference_chunked_squared_l2",
            "threads": 1,
            "exact": True,
            "formal_default": False,
        }
    faiss = require_faiss()
    faiss.omp_set_num_threads(args.faiss_threads)
    return {
        "backend": "faiss",
        "class": "IndexFlatL2",
        "faiss_version": str(getattr(faiss, "__version__", "unknown")),
        "threads": args.faiss_threads,
        "exact": True,
        "formal_default": True,
    }


def _faiss_ranked_nonself(
    distances: np.ndarray,
    local_positions: np.ndarray,
    candidate_ids: np.ndarray,
    query_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    np = require_numpy()
    valid = (local_positions >= 0) & np.isfinite(distances)
    ids = candidate_ids[local_positions[valid]]
    squared_distances = distances[valid].astype(np.float32, copy=False)
    nonself = ids != query_id
    ids = ids[nonself]
    squared_distances = squared_distances[nonself]
    order = np.lexsort((ids, squared_distances))
    return ids[order], squared_distances[order]


def exact_topk_faiss(
    vectors: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    faiss_module: Any,
    faiss_threads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exhaustive IndexFlatL2 ranking with deterministic global-ID tie ordering."""
    np = require_numpy()
    if k <= 0 or faiss_threads <= 0:
        raise ValueError("k and faiss_threads must be positive")
    if faiss_threads > available_cpu_count():
        raise ValueError("faiss_threads exceeds this process's CPU affinity")
    query_ids = np.asarray(query_ids, dtype=np.int64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if candidate_ids.ndim != 1 or query_ids.ndim != 1 or candidate_ids.size == 0:
        raise ValueError("query and candidate IDs must be non-empty one-dimensional arrays")
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique for local-to-global mapping")

    candidate_vectors = np.ascontiguousarray(vectors[candidate_ids], dtype=np.float32)
    query_vectors = np.ascontiguousarray(vectors[query_ids], dtype=np.float32)
    dimension = int(candidate_vectors.shape[1])
    if query_vectors.shape[1] != dimension:
        raise ValueError("candidate/query vector dimensions differ")
    faiss_module.omp_set_num_threads(faiss_threads)
    index = faiss_module.IndexFlatL2(dimension)
    add_started = time.perf_counter()
    index.add(candidate_vectors)
    add_ms = (time.perf_counter() - add_started) * 1000.0
    if int(index.ntotal) != len(candidate_ids):
        raise RuntimeError("IndexFlatL2 did not retain every SQL-derived candidate")
    del candidate_vectors

    retained = k + 1
    top_ids = np.full((len(query_ids), retained), -1, dtype=np.int64)
    top_distances = np.full((len(query_ids), retained), np.inf, dtype=np.float32)
    # k+2 covers k+1 non-self neighbors when the query itself is a candidate.
    requested = min(int(index.ntotal), retained + 1)
    pending = np.arange(len(query_ids), dtype=np.int64)
    search_ms = 0.0
    search_calls = 0
    maximum_requested = requested
    while pending.size:
        search_started = time.perf_counter()
        distances, local_positions = index.search(query_vectors[pending], requested)
        search_ms += (time.perf_counter() - search_started) * 1000.0
        search_calls += 1
        need_more: list[int] = []
        for row, query_position in enumerate(pending):
            ids, ranked_distances = _faiss_ranked_nonself(
                distances[row], local_positions[row], candidate_ids, int(query_ids[query_position])
            )
            if len(ids) >= retained:
                top_ids[query_position] = ids[:retained]
                top_distances[query_position] = ranked_distances[:retained]
                boundary_distance = float(ranked_distances[retained - 1])
                # A strictly farther last returned result proves no omitted tie can
                # change our (distance, global ID) order. Otherwise grow the pool.
                if requested == int(index.ntotal) or float(distances[row, -1]) > boundary_distance:
                    continue
            if requested == int(index.ntotal):
                raise ValueError(
                    f"need at least k+1 non-self candidates for query_id={int(query_ids[query_position])} k={k}"
                )
            need_more.append(int(query_position))
        pending = np.asarray(need_more, dtype=np.int64)
        if pending.size:
            requested = min(int(index.ntotal), max(requested + 1, requested * 2))
            maximum_requested = max(maximum_requested, requested)

    return top_ids, top_distances, {
        "backend": "faiss",
        "class": "IndexFlatL2",
        "faiss_version": str(getattr(faiss_module, "__version__", "unknown")),
        "threads": faiss_threads,
        "index_ntotal": int(index.ntotal),
        "index_add_ms": add_ms,
        "search_ms": search_ms,
        "search_calls": search_calls,
        "initial_requested_rows": min(int(index.ntotal), retained + 1),
        "maximum_requested_rows": maximum_requested,
        "local_positions_mapped_to_global_ids": True,
        "order": "squared_l2_then_global_id",
        "exact": True,
        "exactness": "exhaustive IndexFlatL2 over the complete SQL-derived candidate relation",
        "elapsed_ms": add_ms + search_ms,
    }


def exact_topk(
    vectors: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    args: argparse.Namespace,
    backend_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.backend == "faiss":
        return exact_topk_faiss(
            vectors, query_ids, candidate_ids, args.k, require_faiss(), args.faiss_threads
        )
    if len(candidate_ids) > args.numpy_max_candidates:
        raise RuntimeError(
            "the NumPy reference backend is limited to --numpy-max-candidates; "
            "formal execution must use --backend faiss"
        )
    top_ids, top_distances, elapsed_ms = exact_topk_batched(
        vectors, query_ids, candidate_ids, args.k, args.chunk_rows, args.query_batch_size
    )
    return top_ids, top_distances, {
        **backend_config,
        "index_ntotal": len(candidate_ids),
        "index_add_ms": 0.0,
        "search_ms": elapsed_ms,
        "elapsed_ms": elapsed_ms,
        "order": "squared_l2_then_global_id",
        "exactness": "reference direct float32 squared-L2 scan over the complete SQL-derived candidate relation",
    }


def distance_tolerance(distance_sq: float) -> float:
    return max(1e-9, abs(distance_sq) * 1e-6)


def truth_metadata(distances: np.ndarray, k: int) -> dict[str, Any]:
    np = require_numpy()
    if len(distances) < k + 1 or not np.isfinite(distances[k]):
        raise ValueError(f"need at least k+1 non-self candidates for k={k}")
    kth = float(distances[k - 1])
    tolerance = distance_tolerance(kth)
    return {
        "kth_distance_sq": kth,
        "tie_tolerance": tolerance,
        "strict_closer_count": int(np.sum(distances[:k] < kth - tolerance)),
        "boundary_tied": bool(distances[k] <= kth + tolerance),
    }


def validate_spot_check(
    vector_ids: Sequence[int],
    vector_distances_sq: Sequence[float],
    sql_rows: Sequence[tuple[int, float]],
    k: int,
) -> dict[str, Any]:
    """Compare each exact SQL rank, allowing only numerically tied substitutions."""
    limit = k + 1
    if len(vector_ids) < limit or len(vector_distances_sq) < limit:
        raise ValueError("vectorized result is shorter than k+1")
    if len(sql_rows) != limit:
        raise RuntimeError(f"spot check returned {len(sql_rows)} rows, expected {limit}")
    observed_ids = [int(row[0]) for row in sql_rows]
    observed_distances = [float(row[1]) for row in sql_rows]
    if len(set(observed_ids)) != len(observed_ids):
        raise RuntimeError("spot check returned duplicate IDs")
    tie_positions: list[int] = []
    for position, (expected_id, expected_distance, observed_id, observed_distance) in enumerate(
        zip(vector_ids[:limit], vector_distances_sq[:limit], observed_ids, observed_distances)
    ):
        tolerance = distance_tolerance(float(expected_distance))
        tied = (
            position > 0 and abs(float(expected_distance) - float(vector_distances_sq[position - 1])) <= tolerance
        ) or (
            position + 1 < limit
            and abs(float(expected_distance) - float(vector_distances_sq[position + 1])) <= tolerance
        )
        if tied:
            tie_positions.append(position)
            if abs(observed_distance - float(expected_distance)) > tolerance:
                raise RuntimeError(
                    f"spot check tie rank={position} has the wrong distance: "
                    f"sql={observed_distance} expected={expected_distance}"
                )
        elif observed_id != int(expected_id):
            raise RuntimeError(
                f"spot check rank={position} id mismatch: sql={observed_id} expected={expected_id}"
            )
    return {
        "valid": True,
        "limit": limit,
        "sql_ids": observed_ids,
        "sql_distances": observed_distances,
        "tie_positions": tie_positions,
    }


def plan_index_names(plan: Any) -> list[str]:
    names: list[str] = []
    if isinstance(plan, dict):
        if plan.get("Index Name"):
            names.append(str(plan["Index Name"]))
        for value in plan.values():
            names.extend(plan_index_names(value))
    elif isinstance(plan, list):
        for value in plan:
            names.extend(plan_index_names(value))
    return names


def require_non_hnsw_plan(plan: Any) -> dict[str, Any]:
    names = plan_index_names(plan)
    hnsw = [name for name in names if "hnsw" in name.lower()]
    if hnsw:
        raise RuntimeError(f"exact SQL unexpectedly used an HNSW index: {hnsw}")
    return {"valid": True, "index_names": names}


def relation_fingerprint(cur: Any, relation: str) -> dict[str, Any]:
    cur.execute(
        """
        SELECT c.oid::bigint, c.relfilenode::bigint, c.reltuples::bigint,
               pg_total_relation_size(c.oid)::bigint, c.relkind,
               c.relrowsecurity, c.relforcerowsecurity,
               coalesce(epoch.epoch, 0)::bigint
        FROM pg_catalog.pg_class AS c
        LEFT JOIN public.amazon_sql_native_relation_epoch AS epoch
          ON epoch.relation_name = %s
        WHERE c.oid = to_regclass(%s)
        """,
        (relation, relation),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"required relation does not exist: {relation}")
    cur.execute(
        """
        SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = to_regclass(%s)
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        (relation,),
    )
    columns = [list(value) for value in cur.fetchall()]
    cur.execute(
        """
        SELECT policyname, roles::text, cmd, qual, with_check
        FROM pg_catalog.pg_policies
        WHERE schemaname || '.' || tablename = %s
        ORDER BY policyname
        """,
        (relation,),
    )
    policies = [list(value) for value in cur.fetchall()]
    cur.execute(
        """
        SELECT trigger_row.tgname,
               trigger_row.tgenabled,
               format('%%I.%%I(%%s)', function_namespace.nspname, function_row.proname,
                      pg_get_function_identity_arguments(function_row.oid)),
               pg_get_functiondef(function_row.oid),
               pg_get_triggerdef(trigger_row.oid, true)
        FROM pg_catalog.pg_trigger AS trigger_row
        JOIN pg_catalog.pg_proc AS function_row ON function_row.oid = trigger_row.tgfoid
        JOIN pg_catalog.pg_namespace AS function_namespace
          ON function_namespace.oid = function_row.pronamespace
        WHERE trigger_row.tgrelid = to_regclass(%s)
          AND NOT trigger_row.tgisinternal
        ORDER BY trigger_row.tgname
        """,
        (relation,),
    )
    return {
        "oid": int(row[0]), "relfilenode": int(row[1]), "reltuples": int(row[2]),
        "bytes": int(row[3]), "relkind": str(row[4]), "rls": bool(row[5]),
        "force_rls": bool(row[6]), "data_epoch": int(row[7]),
        "columns": columns, "policies": policies,
        "triggers": [list(value) for value in cur.fetchall()],
    }


def session_context(cur: Any) -> dict[str, str]:
    cur.execute(
        "SELECT current_user::text, session_user::text, txid_current_snapshot()::text, "
        "current_setting('app.as_of', true), current_database()::text"
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("could not capture role and snapshot")
    return {
        "current_user": str(row[0]), "session_user": str(row[1]), "snapshot": str(row[2]),
        "app_as_of": str(row[3] or ""), "database": str(row[4]),
    }


def assert_principal_and_rls(cur: Any, principal: str) -> None:
    validate_rls_security_metadata(collect_rls_security_metadata(cur), principal)


def validate_rls_security_metadata(
    proof: dict[str, Any], principal: str
) -> dict[str, Any]:
    valid = (
        proof.get("current_user") == principal
        and proof.get("is_superuser") is False
        and proof.get("bypass_rls") is False
        and proof.get("owns_facts") is False
        and proof.get("reader_membership") is True
        and proof.get("rls_enabled") is True
        and isinstance(proof.get("policy_hash"), str)
        and len(str(proof.get("policy_hash"))) == 64
    )
    if not valid:
        raise RuntimeError(
            "RLS security proof session metadata failed: "
            + json.dumps(proof, sort_keys=True, default=str)
        )
    return {**proof, "metadata_valid": True}


def validate_rls_security_proof(
    proof: dict[str, Any], principal: str
) -> dict[str, Any]:
    metadata = validate_rls_security_metadata(proof, principal)
    valid = (
        proof.get("positive_probe_visible") is True
        and proof.get("negative_probe_hidden") is True
    )
    if not valid:
        raise RuntimeError(
            "RLS security proof failed: "
            + json.dumps(proof, sort_keys=True, default=str)
        )
    return {**metadata, "valid": True}


def collect_rls_security_metadata(cur: Any) -> dict[str, Any]:
    cur.execute(
        """
        SELECT current_user::text,
               role_row.rolsuper,
               role_row.rolbypassrls,
               owner_row.rolname = current_user,
               pg_has_role(current_user, 'amazon10m_sql_native_reader', 'MEMBER'),
               fact.relrowsecurity
        FROM pg_catalog.pg_roles AS role_row
        CROSS JOIN pg_catalog.pg_class AS fact
        JOIN pg_catalog.pg_roles AS owner_row ON owner_row.oid = fact.relowner
        WHERE role_row.rolname = current_user
          AND fact.oid = 'public.amazon_review_facts'::regclass
        """
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("RLS security metadata query returned no row")
    policies = relation_fingerprint(cur, "public.amazon_review_facts")["policies"]
    return {
        "current_user": str(row[0]),
        "is_superuser": bool(row[1]),
        "bypass_rls": bool(row[2]),
        "owns_facts": bool(row[3]),
        "reader_membership": bool(row[4]),
        "rls_enabled": bool(row[5]),
        "policy_hash": canonical_sha256(policies),
        "policies": policies,
    }


def select_rls_probe_ids(cur: Any, principal: str) -> dict[str, int]:
    cur.execute(
        """
        SELECT min(fact.review_id) FILTER (WHERE EXISTS (
                   SELECT 1
                   FROM public.amazon_product_dim AS product
                   JOIN public.amazon_principal_tenant_grants AS grant_row
                     ON grant_row.tenant_id = product.tenant_id
                  WHERE product.parent_asin = fact.parent_asin
                    AND grant_row.principal_name = %s
                    AND grant_row.can_read
               )),
               min(fact.review_id) FILTER (WHERE NOT EXISTS (
                   SELECT 1
                   FROM public.amazon_product_dim AS product
                   JOIN public.amazon_principal_tenant_grants AS grant_row
                     ON grant_row.tenant_id = product.tenant_id
                  WHERE product.parent_asin = fact.parent_asin
                    AND grant_row.principal_name = %s
                    AND grant_row.can_read
               ))
        FROM public.amazon_review_facts AS fact
        """,
        (principal, principal),
    )
    row = cur.fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise RuntimeError("RLS controlled probes require both visible and hidden facts")
    return {"positive_review_id": int(row[0]), "negative_review_id": int(row[1])}


def run_rls_visibility_probes(cur: Any, probe_ids: dict[str, int]) -> dict[str, bool]:
    positive = int(probe_ids["positive_review_id"])
    negative = int(probe_ids["negative_review_id"])
    cur.execute(
        """
        SELECT count(*) FILTER (WHERE review_id = %s)::bigint,
               count(*) FILTER (WHERE review_id = %s)::bigint
        FROM public.amazon_review_facts
        WHERE review_id = ANY(%s::bigint[])
        """,
        (positive, negative, [positive, negative]),
    )
    row = cur.fetchone()
    positive_count, negative_count = (int(value) for value in row)
    return {
        "positive_probe_visible": positive_count == 1,
        "negative_probe_hidden": negative_count == 0,
    }


def fingerprint_relations(cur: Any, vector_table: str) -> dict[str, Any]:
    relations = (
        vector_table,
        "public.amazon_review_facts",
        "public.amazon_product_dim",
        "public.amazon_principal_tenant_grants",
        "public.amazon_sql_native_buckets",
    )
    fingerprints = {relation: relation_fingerprint(cur, relation) for relation in relations}
    if not fingerprints["public.amazon_review_facts"]["rls"]:
        raise RuntimeError("amazon_review_facts must have RLS enabled")
    missing_epoch_triggers = [
        relation
        for relation, fingerprint in fingerprints.items()
        if not any(
            valid_epoch_trigger(trigger)
            for trigger in fingerprint.get("triggers", [])
        )
    ]
    if missing_epoch_triggers:
        raise RuntimeError(
            "formal data-version epoch trigger is missing: "
            + ",".join(missing_epoch_triggers)
        )
    return fingerprints


def valid_epoch_trigger(trigger: Sequence[Any]) -> bool:
    if len(trigger) != 5:
        return False
    name, enabled, function_identity, function_definition, trigger_definition = (
        str(value) for value in trigger
    )
    normalized_function = " ".join(function_definition.lower().split())
    normalized_trigger = " ".join(trigger_definition.lower().split())
    return (
        name == "amazon_sql_native_epoch_bump"
        and enabled in {"O", "A"}
        and function_identity == "public.amazon_sql_native_bump_relation_epoch()"
        and "insert into public.amazon_sql_native_relation_epoch" in normalized_function
        and "do update" in normalized_function
        and "epoch + 1" in normalized_function
        and "after insert or delete or update or truncate" in normalized_trigger
        and "for each statement" in normalized_trigger
        and re.search(
            r"execute function (?:public\.)?amazon_sql_native_bump_relation_epoch\(\)",
            normalized_trigger,
        )
        is not None
    )


def formal_data_relations(vector_table: str) -> tuple[str, ...]:
    return (
        qualified_name(vector_table),
        "public.amazon_review_facts",
        "public.amazon_product_dim",
        "public.amazon_principal_tenant_grants",
        "public.amazon_sql_native_buckets",
    )


def acquire_formal_data_guard(cur: Any, vector_table: str) -> dict[str, Any]:
    relations = formal_data_relations(vector_table)
    cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
    try:
        cur.execute("LOCK TABLE " + ", ".join(relations) + " IN SHARE MODE")
        fingerprints = fingerprint_relations(cur, vector_table)
        context = session_context(cur)
    except BaseException:
        cur.execute("ROLLBACK")
        raise
    return {
        "lock_mode": "SHARE",
        "relations": list(relations),
        "start_relations": fingerprints,
        "start_hash": canonical_sha256(fingerprints),
        "start_snapshot": context.get("snapshot", ""),
    }


def release_formal_data_guard(
    cur: Any, vector_table: str, guard: dict[str, Any]
) -> dict[str, Any]:
    try:
        end_relations = fingerprint_relations(cur, vector_table)
        end_hash = canonical_sha256(end_relations)
        valid = (
            end_relations == guard.get("start_relations")
            and end_hash == guard.get("start_hash")
        )
        if not valid:
            raise RuntimeError(
                "formal data version changed while the experiment guard was held"
            )
    except BaseException:
        cur.execute("ROLLBACK")
        raise
    cur.execute("COMMIT")
    return {
        **guard,
        "end_relations": end_relations,
        "end_hash": end_hash,
        "valid": True,
    }


def fetch_as_of(cur: Any, principal: str, workload: WorkloadSpec) -> int:
    cur.execute(
        "SELECT as_of FROM public.amazon_sql_native_buckets "
        "WHERE principal_name = %s AND target_pct = %s::numeric",
        (principal, str(workload.bucket_pct)),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"missing as_of bucket for workload={workload.name}")
    return int(row[0])


def _candidate_path(artifact_dir: Path, workload: str, filter_name: str) -> Path:
    return artifact_dir / "candidates" / f"{workload}__{filter_name}.ids"


def _checkpoint_path(checkpoint_dir: Path, workload: str, filter_name: str) -> Path:
    return checkpoint_dir / f"{workload}__{filter_name}.json"


def stream_candidate_ids(
    conn: Any,
    sql_text: str,
    params: dict[str, Any],
    destination: Path,
    fetch_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute exactly once and atomically export the entire ordered candidate ID stream."""
    np = require_numpy()
    validate_candidate_sql(sql_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    values: list[int] = []
    digest = hashlib.sha256()
    minimum: int | None = None
    maximum: int | None = None
    started = time.perf_counter()
    named = conn.cursor(name="sql_native_candidate_export")
    try:
        named.itersize = fetch_rows
        named.execute(sql_text, params)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as target:
            while batch := named.fetchmany(fetch_rows):
                ids = [int(row[0]) for row in batch]
                if ids != sorted(ids) or len(ids) != len(set(ids)):
                    raise RuntimeError("candidate SQL must emit strictly increasing unique IDs")
                if values and ids and ids[0] <= values[-1]:
                    raise RuntimeError("candidate SQL stream is not globally ordered")
                text = "".join(f"{value}\n" for value in ids)
                target.write(text)
                digest.update(text.encode("ascii"))
                values.extend(ids)
                if ids:
                    minimum = ids[0] if minimum is None else min(minimum, ids[0])
                    maximum = ids[-1] if maximum is None else max(maximum, ids[-1])
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        named.close()
    return np.asarray(values, dtype=np.int64), {
        "count": len(values), "min_id": minimum, "max_id": maximum,
        "sha256": digest.hexdigest(), "path": str(destination.resolve()),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def build_run_spec(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    source_hashes: dict[str, str],
    backend_config: dict[str, Any],
    base_table_mapping: dict[str, Any],
    query_cohort_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_splits = {
        query_no: query_split(query_no, args.calibration_queries)
        for query_no in query_ids
    }
    cohort_hash = query_cohort_sha256(query_ids, query_splits)
    validity_predicate = validate_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    return {
        "version": CHECKPOINT_VERSION,
        "vector_table": args.vector_table,
        "principal": args.principal,
        "k": args.k,
        "calibration_queries": args.calibration_queries,
        "final_queries": args.final_queries,
        "chunk_rows": args.chunk_rows,
        "query_batch_size": args.query_batch_size,
        "backend": backend_config,
        "faiss_threads": args.faiss_threads,
        "numpy_max_candidates": args.numpy_max_candidates,
        "base_table_mapping": base_table_mapping,
        "spot_check_queries": args.spot_check_queries,
        "filters": [asdict(spec) for spec in filters],
        "workload_scalar_predicates": [
            {
                "filter_name": spec.name,
                "predicate": spec.predicate,
                "predicate_sha256": workload_scalar_predicate_sha256(spec.predicate),
            }
            for spec in filters
        ],
        "candidate_universe": {
            "predicate": validity_predicate,
            "predicate_sha256": candidate_universe_predicate_sha256(validity_predicate),
            "sql_role": "candidate_relation_only; separate from workload scalar predicate",
        },
        "workloads": [asdict(workload) for workload in WORKLOADS],
        "query_ids": query_ids,
        "query_splits": query_splits,
        "query_cohort_sha256": cohort_hash,
        "query_cohort_hash_contract": QUERY_COHORT_HASH_CONTRACT,
        "query_cohort": query_cohort_provenance
        or {
            "query_count": len(query_ids),
            "query_cohort_sha256": cohort_hash,
            "query_cohort_hash_contract": QUERY_COHORT_HASH_CONTRACT,
        },
        "source_hashes": source_hashes,
    }


def write_pair_checkpoint(path: Path, run_spec_hash: str, source_hashes: dict[str, str], payload: dict[str, Any]) -> None:
    atomic_write_json(path, {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_spec_hash": run_spec_hash,
        "source_hashes": source_hashes,
        "complete": True,
        **payload,
    })


def load_pair_checkpoint(path: Path, run_spec_hash: str, source_hashes: dict[str, str]) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid checkpoint: {path}") from exc
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION or not checkpoint.get("complete"):
        raise RuntimeError(f"incomplete or incompatible checkpoint: {path}")
    if checkpoint.get("run_spec_hash") != run_spec_hash:
        raise RuntimeError(f"stale checkpoint run-spec mismatch: {path}")
    if checkpoint.get("source_hashes") != source_hashes:
        raise RuntimeError(f"stale checkpoint source-hash mismatch: {path}")
    candidate = checkpoint.get("candidate")
    if isinstance(candidate, dict) and candidate.get("path") and candidate.get("sha256"):
        candidate_path = Path(str(candidate["path"]))
        if not candidate_path.is_file() or sha256_file(candidate_path) != str(candidate["sha256"]):
            raise RuntimeError(f"checkpoint candidate export is missing or stale: {path}")
    return checkpoint


def select_spot_query_nos(query_ids: dict[int, int], count: int) -> list[int]:
    if count <= 0:
        return []
    ordered = sorted(query_ids)
    if count >= len(ordered):
        return ordered
    # Deterministically cover the calibration and final range instead of sampling
    # only the beginning of the q200 workload.
    return sorted({ordered[round(index * (len(ordered) - 1) / (count - 1))] if count > 1 else ordered[0]
                   for index in range(count)})


def truth_rows_for_pair(
    workload: WorkloadSpec,
    filter_spec: FilterSpec,
    query_ids: dict[int, int],
    top_ids: np.ndarray,
    top_distances: np.ndarray,
    candidate: dict[str, Any],
    k: int,
    calibration_queries: int,
    exact_ms: float,
    as_of: int,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> list[dict[str, Any]]:
    validity_hash = candidate_universe_predicate_sha256(candidate_validity_predicate)
    scalar_hash = workload_scalar_predicate_sha256(filter_spec.predicate)
    rows: list[dict[str, Any]] = []
    for position, (query_no, query_id) in enumerate(sorted(query_ids.items())):
        metadata = truth_metadata(top_distances[position], k)
        rows.append({
            "workload": workload.name, "filter_name": filter_spec.name,
            "target_rate": filter_spec.target_rate, "predicate": filter_spec.predicate,
            "workload_scalar_predicate_sha256": scalar_hash,
            "candidate_universe_predicate": candidate_validity_predicate,
            "candidate_universe_predicate_sha256": validity_hash,
            "query_no": query_no, "query_id": query_id,
            "query_split": query_split(query_no, calibration_queries), "k": k,
            "as_of": as_of, "self_excluded": True,
            "candidate_count": candidate["count"], "candidate_min_id": candidate["min_id"],
            "candidate_max_id": candidate["max_id"], "candidate_ids_sha256": candidate["sha256"],
            "exact_topk_ids": ",".join(str(int(value)) for value in top_ids[position, :k]),
            "exact_topk_distances_sq": ",".join(f"{float(value):.9g}" for value in top_distances[position, :k]),
            "exact_topk_plus_one_ids": ",".join(str(int(value)) for value in top_ids[position, : k + 1]),
            "exact_topk_plus_one_distances_sq": ",".join(f"{float(value):.9g}" for value in top_distances[position, : k + 1]),
            "exact_scan_amortized_ms": exact_ms / len(query_ids),
            **metadata,
        })
    return rows


def run_spot_checks(
    cur: Any,
    vectors: np.ndarray,
    query_ids: dict[int, int],
    top_ids: np.ndarray,
    top_distances: np.ndarray,
    sql_text: str,
    as_of: int,
    k: int,
    count: int,
) -> list[dict[str, Any]]:
    np = require_numpy()
    checks: list[dict[str, Any]] = []
    position_by_no = {query_no: pos for pos, query_no in enumerate(sorted(query_ids))}
    for query_no in select_spot_query_nos(query_ids, count):
        query_id = query_ids[query_no]
        cur.execute(sql_text, {"query_id": query_id, "as_of": as_of, "limit": k + 1})
        # pgvector <-> reports L2, while this artifact records squared L2.
        sql_rows = [(int(row[0]), float(row[1]) * float(row[1])) for row in cur.fetchall()]
        position = position_by_no[query_no]
        validation = validate_spot_check(top_ids[position], top_distances[position], sql_rows, k)
        sql_ids = np.asarray([row[0] for row in sql_rows], dtype=np.int64)
        direct = np.asarray(vectors[sql_ids], dtype=np.float32) - np.asarray(vectors[query_id], dtype=np.float32)
        direct_sq = np.einsum("ij,ij->i", direct, direct, dtype=np.float32)
        for sql_distance, vector_distance in zip(validation["sql_distances"], direct_sq):
            if abs(sql_distance - float(vector_distance)) > max(1e-7, abs(float(vector_distance)) * 5e-5):
                raise RuntimeError("spot check PostgreSQL/fbin distance mismatch")
        checks.append({"query_no": query_no, "query_id": query_id, **validation})
    return checks


def execute_pair(
    conn: Any,
    args: argparse.Namespace,
    vectors: np.ndarray,
    vector_rows: int,
    workload: WorkloadSpec,
    filter_spec: FilterSpec,
    query_ids: dict[int, int],
    run_spec_hash: str,
    source_hashes: dict[str, str],
    checkpoint_dir: Path,
    backend_config: dict[str, Any],
) -> dict[str, Any]:
    np = require_numpy()
    cur = conn.cursor()
    checkpoint_path = _checkpoint_path(checkpoint_dir, workload.name, filter_spec.name)
    try:
        cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        as_of = fetch_as_of(cur, args.principal, workload)
        cur.execute("SELECT set_config('app.as_of', %s, true)", (str(as_of),))
        assert_principal_and_rls(cur, args.principal)
        context = session_context(cur)
        relations = fingerprint_relations(cur, args.vector_table)
        candidate_sql = build_candidate_sql(
            args.vector_table,
            filter_spec.predicate,
            workload,
            args.candidate_validity_predicate,
        )
        spot_sql = build_spot_check_sql(
            args.vector_table,
            filter_spec.predicate,
            workload,
            args.candidate_validity_predicate,
        )
        validate_candidate_sql(candidate_sql)
        validate_exact_sql_text(spot_sql)
        cur.execute("EXPLAIN (FORMAT JSON, VERBOSE, SETTINGS) " + candidate_sql, {"as_of": as_of})
        candidate_plan = cur.fetchone()[0]
        candidate_plan_gate = require_non_hnsw_plan(candidate_plan)
        cur.execute("EXPLAIN (FORMAT JSON, VERBOSE, SETTINGS) " + spot_sql, {"query_id": query_ids[min(query_ids)], "as_of": as_of, "limit": args.k + 1})
        spot_plan = cur.fetchone()[0]
        spot_plan_gate = require_non_hnsw_plan(spot_plan)
        candidate_ids, candidate = stream_candidate_ids(
            conn, candidate_sql, {"as_of": as_of},
            _candidate_path(args.artifact_dir, workload.name, filter_spec.name), args.candidate_fetch_rows,
        )
        if candidate_ids.size == 0:
            raise RuntimeError(f"workload={workload.name} filter={filter_spec.name} has no SQL candidates")
        if int(candidate_ids[0]) < 0 or int(candidate_ids[-1]) >= vector_rows:
            raise RuntimeError("SQL candidate ID is outside fbin row space")
        if any(query_id < 0 or query_id >= vector_rows for query_id in query_ids.values()):
            raise RuntimeError("query ID is outside fbin row space")
        top_ids, top_distances, exact_backend = exact_topk(
            vectors, np.asarray(list(query_ids.values()), dtype=np.int64), candidate_ids, args, backend_config,
        )
        exact_ms = float(exact_backend["elapsed_ms"])
        rows = truth_rows_for_pair(workload, filter_spec, query_ids, top_ids, top_distances, candidate,
                                   args.k, args.calibration_queries, exact_ms, as_of,
                                   args.candidate_validity_predicate)
        spot_checks = run_spot_checks(cur, vectors, query_ids, top_ids, top_distances, spot_sql, as_of,
                                      args.k, args.spot_check_queries)
        cur.execute("COMMIT")
        payload = {
            "workload": workload.name, "filter": asdict(filter_spec), "as_of": as_of,
            "workload_scalar_predicate": filter_spec.predicate,
            "workload_scalar_predicate_sha256": workload_scalar_predicate_sha256(
                filter_spec.predicate
            ),
            "candidate_universe_predicate": args.candidate_validity_predicate,
            "candidate_universe_predicate_sha256": candidate_universe_predicate_sha256(
                args.candidate_validity_predicate
            ),
            "session": context, "relations": relations,
            "candidate_sql": candidate_sql, "candidate_sql_sha256": hashlib.sha256(candidate_sql.encode()).hexdigest(),
            "candidate_explain": candidate_plan, "candidate_explain_gate": candidate_plan_gate,
            "spot_check_sql": spot_sql, "spot_check_sql_sha256": hashlib.sha256(spot_sql.encode()).hexdigest(),
            "spot_check_explain": spot_plan, "spot_check_explain_gate": spot_plan_gate,
            "candidate": candidate, "exact_scan_ms": exact_ms, "exact_backend": exact_backend,
            "spot_checks": spot_checks, "rows": rows,
        }
        write_pair_checkpoint(checkpoint_path, run_spec_hash, source_hashes, payload)
        return payload
    except BaseException:
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        cur.close()


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute SQL-native exact ground truth for the Amazon-10M workload.")
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--query-ids-csv", type=Path, default=DEFAULT_QUERY_IDS)
    parser.add_argument(
        "--query-cohort-manifest",
        type=Path,
        default=DEFAULT_QUERY_COHORT_MANIFEST,
        help="provenance manifest for a truth-format query cohort CSV",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--vector-table", type=qualified_name, default=DEFAULT_VECTOR_TABLE)
    parser.add_argument("--principal", type=role_name, default=DEFAULT_PRINCIPAL)
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
        help="global candidate-universe SQL predicate; formal value is embedding_valid",
    )
    parser.add_argument("--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("--calibration-queries", type=positive_int, default=DEFAULT_CALIBRATION_QUERIES)
    parser.add_argument("--final-queries", type=positive_int, default=DEFAULT_FINAL_QUERIES)
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--chunk-rows", type=positive_int, default=20_000)
    parser.add_argument("--query-batch-size", type=positive_int, default=8)
    parser.add_argument(
        "--backend", choices=("faiss", "numpy"), default="faiss",
        help="formal default is exhaustive Faiss IndexFlatL2; NumPy is a size-capped reference path",
    )
    parser.add_argument(
        "--faiss-threads", type=positive_int, default=1,
        help="Faiss OpenMP threads; defaults to one so concurrent formal jobs do not steal reserved cores",
    )
    parser.add_argument(
        "--numpy-max-candidates", type=positive_int, default=100_000,
        help="maximum candidate rows allowed with explicit --backend numpy",
    )
    parser.add_argument(
        "--base-table-mapping-sample-size", type=positive_int,
        default=DEFAULT_BASE_TABLE_MAPPING_SAMPLE_SIZE,
        help="deterministic evenly spaced base-table IDs audited against fbin before GT work",
    )
    parser.add_argument("--candidate-fetch-rows", type=positive_int, default=10_000)
    parser.add_argument("--spot-check-queries", type=nonnegative_int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="print contract only; never read input files or PostgreSQL")
    parser.add_argument("--execute", action="store_true", help="perform the PostgreSQL and fbin exact computation")
    parser.add_argument("--resume", action="store_true", help="resume only complete checkpoints with identical source and run-spec hashes")
    return parser


def validate_formal_dimensions(
    args: argparse.Namespace, filters: Sequence[FilterSpec]
) -> None:
    problems: list[str] = []
    if len(filters) != 14 or len({spec.name for spec in filters}) != 14:
        problems.append("exactly 14 distinct registered filters are required")
    if args.calibration_queries != 100 or args.final_queries != 100:
        problems.append("exact GT must contain disjoint calibration q100 and final q100")
    if args.backend != "faiss":
        problems.append("formal exact GT requires exhaustive Faiss IndexFlatL2")
    if problems:
        raise RuntimeError("formal exact-truth dimensions are invalid: " + "; ".join(problems))


def print_dry_run(args: argparse.Namespace) -> None:
    print("mode=dry-run")
    print("database=not_opened")
    print("inputs=not_read")
    print("backend_imports=not_loaded")
    print(
        "execution=one unbounded pure relational ID export per workload/filter; "
        "exhaustive Faiss IndexFlatL2 squared-L2"
    )
    print("workloads=" + ",".join(workload.name for workload in WORKLOADS))
    print(f"queries=q{args.calibration_queries + args.final_queries}; calibration={args.calibration_queries}; final={args.final_queries}")
    print(f"k={args.k}; retained=k+1; self_excluded=true; spot_checks_per_pair={args.spot_check_queries}")
    print(f"backend={args.backend}; faiss_threads={args.faiss_threads}; cpu_affinity={available_cpu_count()}")
    print(f"base_table_mapping_sample_size={args.base_table_mapping_sample_size}; query_ids_included=true")
    print(f"query_ids_csv={args.query_ids_csv}")
    print(f"query_cohort_manifest={args.query_cohort_manifest}")
    print(f"candidate_validity_predicate={args.candidate_validity_predicate}")
    print(
        "candidate_universe_predicate_sha256="
        + candidate_universe_predicate_sha256(args.candidate_validity_predicate)
    )
    print(f"artifact_dir={args.artifact_dir}")


def build_artifact_manifest(
    *,
    run_spec: dict[str, Any],
    source_hashes: dict[str, str],
    fbin: dict[str, Any],
    base_table_mapping: dict[str, Any],
    outputs: dict[str, Any],
    backend: dict[str, Any],
    pairs: Sequence[dict[str, Any]],
    data_version_proof: dict[str, Any],
    rls_security_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        data_version_proof.get("valid") is not True
        or data_version_proof.get("start_hash") != data_version_proof.get("end_hash")
        or data_version_proof.get("start_relations")
        != data_version_proof.get("end_relations")
        or data_version_proof.get("start_hash")
        != canonical_sha256(data_version_proof.get("start_relations"))
    ):
        raise RuntimeError("exact-truth artifact requires a valid data-version proof")
    if rls_security_proof is None:
        raise RuntimeError("exact-truth artifact requires a controlled RLS security proof")
    validated_security = validate_rls_security_proof(
        rls_security_proof, str(run_spec.get("principal", ""))
    )
    facts = data_version_proof.get("start_relations", {}).get(
        "public.amazon_review_facts", {}
    )
    if validated_security.get("policy_hash") != canonical_sha256(
        facts.get("policies", [])
    ):
        raise RuntimeError("exact-truth RLS policy hash does not match data-version proof")
    try:
        run_query_ids = {
            int(query_no): int(query_id)
            for query_no, query_id in run_spec["query_ids"].items()
        }
        run_query_splits = {
            int(query_no): str(split)
            for query_no, split in run_spec["query_splits"].items()
        }
        expected_cohort_hash = query_cohort_sha256(run_query_ids, run_query_splits)
        cohort = run_spec["query_cohort"]
        candidate_universe = run_spec["candidate_universe"]
        validity_predicate = validate_candidate_validity_predicate(
            candidate_universe["predicate"]
        )
        validity_hash = candidate_universe_predicate_sha256(validity_predicate)
    except (KeyError, TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise RuntimeError("exact-truth artifact has malformed cohort/universe provenance") from exc
    if (
        run_spec.get("query_cohort_sha256") != expected_cohort_hash
        or not isinstance(cohort, dict)
        or cohort.get("query_cohort_sha256") != expected_cohort_hash
        or candidate_universe.get("predicate_sha256") != validity_hash
    ):
        raise RuntimeError("exact-truth artifact cohort/universe hashes do not match the run spec")
    relation_epoch = relation_epoch_contract(data_version_proof["start_relations"])
    manifest = {
        "artifact_valid": True,
        "artifact": "amazon10m_sql_native_exact_truth",
        "version": CHECKPOINT_VERSION,
        "generated_at_unix": time.time(),
        "git_revision": git_revision(),
        "run_spec": run_spec,
        "run_spec_hash": canonical_sha256(run_spec),
        "query_cohort": cohort,
        "query_cohort_sha256": expected_cohort_hash,
        "candidate_universe": candidate_universe,
        "candidate_universe_predicate_sha256": validity_hash,
        "relation_epoch": relation_epoch,
        "source_hashes": source_hashes,
        "fbin": fbin,
        "base_table_mapping": base_table_mapping,
        "outputs": outputs,
        "backend": backend,
        "pairs": list(pairs),
        "data_version_proof": data_version_proof,
    }
    manifest["rls_security_proof"] = validated_security
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.dry_run:
        print_dry_run(args)
        return 0
    if not args.execute:
        raise SystemExit("refusing to execute without --execute (use --dry-run to inspect the contract)")
    backend_config = configure_exact_backend(args)
    require_psycopg()
    filters = read_filters(args.filters_csv, set(args.filter_names))
    validate_formal_dimensions(args, filters)
    expected_splits = {
        query_no: query_split(query_no, args.calibration_queries)
        for query_no in range(args.calibration_queries + args.final_queries)
    }
    query_cohort = load_query_cohort(
        args.query_ids_csv,
        expected_splits,
        args.candidate_validity_predicate,
        source_manifest_path=args.query_cohort_manifest,
        expected_filters=filters,
    )
    query_ids = query_cohort.query_ids
    vectors, vector_rows, dimensions = read_fbin_memmap(args.fbin)
    source_hashes = {
        "script": sha256_file(Path(__file__)), "filters_csv": sha256_file(args.filters_csv),
        "query_ids_csv": sha256_file(args.query_ids_csv), "fbin": sha256_file(args.fbin),
        "query_cohort_manifest": sha256_file(args.query_cohort_manifest),
    }
    import psycopg
    conninfo = pg_config_from_env().conninfo
    data_version_proof: dict[str, Any]
    rls_security_proof: dict[str, Any]
    with (
        psycopg.connect(conninfo, autocommit=True) as guard_conn,
        psycopg.connect(conninfo, autocommit=False) as conn,
    ):
        guard_cur = guard_conn.cursor()
        guard = acquire_formal_data_guard(guard_cur, args.vector_table)
        try:
            probe_ids = select_rls_probe_ids(guard_cur, args.principal)
            role_cur = conn.cursor()
            try:
                role_cur.execute(f'SET ROLE "{args.principal}"')
            finally:
                role_cur.close()
            conn.commit()
            security_cur = conn.cursor()
            try:
                security = collect_rls_security_metadata(security_cur)
                security.update(run_rls_visibility_probes(security_cur, probe_ids))
                security["controlled_probe_ids"] = probe_ids
                rls_security_proof = validate_rls_security_proof(
                    security, args.principal
                )
            finally:
                security_cur.close()
                conn.commit()
            base_table_mapping = verify_base_table_vector_mapping(
                conn,
                args.vector_table,
                vectors,
                query_ids,
                args.base_table_mapping_sample_size,
                args.candidate_validity_predicate,
            )
            run_spec = build_run_spec(
                args,
                filters,
                query_ids,
                source_hashes,
                backend_config,
                base_table_mapping,
                query_cohort.provenance,
            )
            run_spec_hash = canonical_sha256(run_spec)
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            out = args.out or args.artifact_dir / "amazon10m_sql_native_exact_truth_q200.csv"
            manifest_path = args.manifest or args.artifact_dir / "amazon10m_sql_native_exact_truth_manifest.json"
            checkpoint_dir = args.checkpoint_dir or args.artifact_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            completed: list[dict[str, Any]] = []
            for workload in WORKLOADS:
                for filter_spec in filters:
                    checkpoint_path = _checkpoint_path(checkpoint_dir, workload.name, filter_spec.name)
                    if checkpoint_path.exists():
                        if not args.resume:
                            raise RuntimeError(f"checkpoint exists for {workload.name}/{filter_spec.name}; pass --resume")
                        payload = load_pair_checkpoint(checkpoint_path, run_spec_hash, source_hashes)
                        if payload.get("relations") != guard["start_relations"]:
                            raise RuntimeError(
                                "resumed exact-truth pair has a stale data version"
                            )
                        completed.append(payload)
                        print(f"resume workload={workload.name} filter={filter_spec.name}", flush=True)
                        continue
                    payload = execute_pair(conn, args, vectors, vector_rows, workload, filter_spec, query_ids,
                                           run_spec_hash, source_hashes, checkpoint_dir, backend_config)
                    if payload["relations"] != guard["start_relations"]:
                        raise RuntimeError("exact-truth pair escaped the guarded data version")
                    completed.append(payload)
                    print(f"checkpointed workload={workload.name} filter={filter_spec.name}", flush=True)
            data_version_proof = release_formal_data_guard(
                guard_cur, args.vector_table, guard
            )
        except BaseException:
            try:
                guard_cur.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            guard_cur.close()
    rows = sorted((row for payload in completed for row in payload["rows"]),
                  key=lambda row: (str(row["workload"]), str(row["filter_name"]), int(row["query_no"])))
    expected_rows = len(WORKLOADS) * len(filters) * len(query_ids)
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete GT artifact: rows={len(rows)} expected={expected_rows}")
    csv_payload = render_csv(rows)
    manifest = build_artifact_manifest(
        run_spec=run_spec,
        source_hashes=source_hashes,
        fbin={"path": str(args.fbin.resolve()), "rows": vector_rows, "dimensions": dimensions},
        base_table_mapping=base_table_mapping,
        outputs={
            "truth_csv": str(out.resolve()),
            "truth_csv_sha256": sha256_text(csv_payload),
        },
        backend=backend_config,
        pairs=[{key: payload[key] for key in (
            "workload", "filter", "as_of", "workload_scalar_predicate",
            "workload_scalar_predicate_sha256", "candidate_universe_predicate",
            "candidate_universe_predicate_sha256", "session", "relations",
            "candidate_sql", "candidate_sql_sha256", "candidate_explain",
            "candidate_explain_gate", "spot_check_sql", "spot_check_sql_sha256",
            "spot_check_explain", "spot_check_explain_gate", "candidate",
            "exact_scan_ms", "exact_backend", "spot_checks",
        )} for payload in completed],
        data_version_proof=data_version_proof,
        rls_security_proof=rls_security_proof,
    )
    publish_exact_artifact(out, manifest_path, rows, manifest)
    print(f"wrote {out} rows={len(rows)} manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
