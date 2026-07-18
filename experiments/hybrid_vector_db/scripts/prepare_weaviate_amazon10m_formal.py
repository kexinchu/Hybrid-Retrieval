#!/usr/bin/env python3
"""Prepare the formal Amazon-10M corpus for Weaviate matched-recall runs.

This importer never infers a source offset from the live object count.  Progress
is an explicit, contiguous row interval whose batches were acknowledged by
Weaviate and atomically checkpointed.  A resume verifies that interval against
the live service before continuing; the one batch that may have committed before
its checkpoint is reconciled by deterministic UUID and row_id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

import weaviate_matched_recall_baseline as baseline


ROOT = Path(__file__).resolve().parents[3]
CLASS_NAME = baseline.CLASS_NAME
EXPECTED_ROWS = baseline.EXPECTED_ROWS
EXPECTED_VALID_ROWS = baseline.EXPECTED_VALID_ROWS
DEFAULT_CSV = (
    ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"
)
DEFAULT_FBIN = baseline.DEFAULT_FBIN
DEFAULT_FILTERS = baseline.DEFAULT_FILTERS
DEFAULT_CHECKPOINT = (
    ROOT / "results/hybrid_vector_db/weaviate_amazon10m_formal_import_checkpoint.json"
)
DEFAULT_MANIFEST = (
    ROOT / "results/hybrid_vector_db/weaviate_amazon10m_formal_import_manifest.json"
)
CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
UUID_NAMESPACE = uuid.UUID("adf13b90-d38f-5f4b-bd33-5a194d8767a1")
UUID_ALGORITHM = "uuid5(adf13b90-d38f-5f4b-bd33-5a194d8767a1,amazon10m:<row_id>)"
IMAGE_DIGEST_RE = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")

CSV_COLUMNS = (
    "id",
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
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

PROPERTY_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"name": "row_id", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
    {"name": "rating", "dataType": ["number"], "indexFilterable": True, "indexRangeFilters": True},
    {"name": "verified_purchase", "dataType": ["boolean"], "indexFilterable": True},
    {"name": "helpful_vote", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
    {"name": "review_text_len", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
    {
        "name": "main_category",
        "dataType": ["text"],
        "tokenization": "field",
        "indexFilterable": True,
    },
    {"name": "price", "dataType": ["number"], "indexFilterable": True, "indexRangeFilters": True},
    {"name": "has_price", "dataType": ["boolean"], "indexFilterable": True},
    {"name": "item_rating_number", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
    {"name": "embedding_valid", "dataType": ["boolean"], "indexFilterable": True},
)


class PreparationError(RuntimeError):
    """A fail-closed corpus preparation gate failed."""


@dataclass(frozen=True)
class BatchResult:
    attempts: int
    retry_count: int
    recovered_objects: int
    response_sha256: str
    acknowledged_ids_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise PreparationError(f"input is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(payload, target, sort_keys=True, indent=2, ensure_ascii=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def object_uuid(row_id: int) -> str:
    if row_id < 0 or row_id >= EXPECTED_ROWS:
        raise ValueError(f"row_id outside formal corpus: {row_id}")
    return str(uuid.uuid5(UUID_NAMESPACE, f"amazon10m:{row_id}"))


def parse_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise PreparationError(f"invalid {field} boolean: {value!r}")


def expected_schema(
    *,
    ef_construction: int = 128,
    max_connections: int = 32,
    flat_search_cutoff: int = 40_000,
    filter_strategy: str = "acorn",
) -> dict[str, Any]:
    return {
        "class": CLASS_NAME,
        "description": "Formal SQLens Amazon-10M matched-recall corpus (128d)",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "l2-squared",
            "efConstruction": int(ef_construction),
            "maxConnections": int(max_connections),
            "flatSearchCutoff": int(flat_search_cutoff),
            "filterStrategy": filter_strategy,
        },
        "properties": [dict(item) for item in PROPERTY_DEFINITIONS],
    }


def _property_map(schema: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    properties = schema.get("properties")
    if not isinstance(properties, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for value in properties:
        if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
            raise PreparationError("schema contains a malformed property definition")
        name = str(value["name"])
        if name in result:
            raise PreparationError(f"schema contains duplicate property {name!r}")
        result[name] = value
    return result


def verify_formal_schema(
    actual: Mapping[str, Any], requested: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        baseline.verify_schema(dict(actual))
    except RuntimeError as exc:
        errors.append(str(exc))
    if actual.get("class") != CLASS_NAME:
        errors.append(f"class must be {CLASS_NAME}")
    if str(actual.get("vectorizer", "")).lower() != "none":
        errors.append("vectorizer must be none")
    if str(actual.get("vectorIndexType", "")).lower() != "hnsw":
        errors.append("vectorIndexType must be hnsw")

    actual_config = actual.get("vectorIndexConfig")
    requested_config = requested.get("vectorIndexConfig")
    if not isinstance(actual_config, Mapping) or not isinstance(requested_config, Mapping):
        errors.append("vectorIndexConfig must be an object")
    else:
        for key in (
            "distance",
            "efConstruction",
            "maxConnections",
            "flatSearchCutoff",
            "filterStrategy",
        ):
            actual_value = actual_config.get(key)
            expected_value = requested_config.get(key)
            if key in {"distance", "filterStrategy"}:
                equal = str(actual_value).lower() == str(expected_value).lower()
            else:
                try:
                    equal = int(actual_value) == int(expected_value)
                except (TypeError, ValueError):
                    equal = False
            if not equal:
                errors.append(
                    f"vectorIndexConfig.{key} mismatch: "
                    f"expected={expected_value!r} actual={actual_value!r}"
                )

    try:
        actual_properties = _property_map(actual)
        requested_properties = _property_map(requested)
    except PreparationError as exc:
        errors.append(str(exc))
        actual_properties = {}
        requested_properties = {}
    if set(actual_properties) != set(requested_properties):
        errors.append(
            "property set mismatch: "
            f"expected={sorted(requested_properties)} actual={sorted(actual_properties)}"
        )
    for name, expected_property in requested_properties.items():
        actual_property = actual_properties.get(name)
        if actual_property is None:
            continue
        expected_types = [str(item).lower() for item in expected_property["dataType"]]
        actual_types = actual_property.get("dataType", [])
        if isinstance(actual_types, str):
            actual_types = [actual_types]
        if [str(item).lower() for item in actual_types] != expected_types:
            errors.append(f"property {name} dataType mismatch")
        for key in ("indexFilterable", "indexRangeFilters", "tokenization"):
            if key in expected_property and actual_property.get(key) != expected_property[key]:
                errors.append(
                    f"property {name}.{key} mismatch: "
                    f"expected={expected_property[key]!r} actual={actual_property.get(key)!r}"
                )
    if "embedding_valid" not in actual_properties:
        errors.append("embedding_valid property is mandatory")
    if errors:
        raise PreparationError("formal schema gate failed: " + "; ".join(errors))
    return dict(actual)


def _is_not_found(error: RuntimeError) -> bool:
    return "HTTP 404" in str(error)


def ensure_schema(
    base_url: str,
    requested: dict[str, Any],
    *,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], bool, int]:
    retry_count = 0
    try:
        actual, used = baseline.request_json(
            base_url,
            f"/v1/schema/{CLASS_NAME}",
            timeout=timeout,
            retries=retries,
        )
        retry_count += used
        return verify_formal_schema(actual, requested), False, retry_count
    except RuntimeError as exc:
        if not _is_not_found(exc):
            raise
    _, used = baseline.request_json(
        base_url,
        "/v1/schema",
        requested,
        method="POST",
        timeout=timeout,
        retries=retries,
    )
    retry_count += used
    actual, used = baseline.request_json(
        base_url,
        f"/v1/schema/{CLASS_NAME}",
        timeout=timeout,
        retries=retries,
    )
    retry_count += used
    return verify_formal_schema(actual, requested), True, retry_count


def validate_image_digest(value: str) -> str:
    normalized = value.strip().lower()
    if not IMAGE_DIGEST_RE.fullmatch(normalized):
        raise PreparationError(
            "--service-image-digest must be an immutable sha256 digest, for "
            "example semitechnologies/weaviate@sha256:<64 hex>"
        )
    return normalized


def validate_service_identity(
    meta: Mapping[str, Any], expected_version: str, image_digest: str
) -> dict[str, str]:
    actual_version = str(meta.get("version", ""))
    if actual_version != expected_version:
        raise PreparationError(
            f"Weaviate version mismatch: expected={expected_version!r} "
            f"actual={actual_version!r}"
        )
    return {
        "expected_version": expected_version,
        "actual_version": actual_version,
        "service_image_digest": validate_image_digest(image_digest),
    }


def _where_for_range(start: int, end: int) -> dict[str, Any]:
    if start < 0 or end <= start or end > EXPECTED_ROWS:
        raise ValueError(f"invalid row range [{start}, {end})")
    return {
        "operator": "And",
        "operands": [
            {"path": ["row_id"], "operator": "GreaterThanEqual", "valueInt": start},
            {"path": ["row_id"], "operator": "LessThan", "valueInt": end},
        ],
    }


def _count_query(where: Mapping[str, Any] | None = None) -> str:
    arguments = "" if where is None else f"(where:{baseline.json_to_graphql(where)})"
    return (
        "{ Aggregate { "
        f"{CLASS_NAME}{arguments} {{ meta {{ count }} }}"
        " } }"
    )


def query_count(
    base_url: str,
    where: Mapping[str, Any] | None,
    *,
    timeout: float,
    retries: int,
) -> tuple[int, int]:
    payload, used = baseline.graphql(
        base_url, _count_query(where), timeout=timeout, retries=retries
    )
    if payload.get("errors"):
        raise PreparationError(
            "Weaviate aggregate failed: " + json.dumps(payload["errors"], sort_keys=True)
        )
    try:
        value = int(payload["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PreparationError("malformed Weaviate aggregate response") from exc
    return value, used


def inspect_range(
    base_url: str,
    start: int,
    end: int,
    *,
    timeout: float,
    retries: int,
) -> tuple[dict[int, dict[str, Any]], int]:
    limit = end - start + 1
    query = (
        "{ Get { "
        f"{CLASS_NAME}(where:{baseline.json_to_graphql(_where_for_range(start, end))} "
        f"limit:{limit}) {{ row_id embedding_valid _additional {{ id }} }}"
        " } }"
    )
    payload, used = baseline.graphql(base_url, query, timeout=timeout, retries=retries)
    if payload.get("errors"):
        raise PreparationError(
            "Weaviate range inspection failed: "
            + json.dumps(payload["errors"], sort_keys=True)
        )
    try:
        objects = payload["data"]["Get"][CLASS_NAME]
    except (KeyError, TypeError) as exc:
        raise PreparationError("malformed Weaviate range inspection response") from exc
    if not isinstance(objects, list) or len(objects) > end - start:
        raise PreparationError(
            f"range inspection returned an invalid number of objects: {len(objects)}"
        )
    result: dict[int, dict[str, Any]] = {}
    for item in objects:
        try:
            row_id = int(item["row_id"])
            actual_uuid = str(item["_additional"]["id"])
            embedding_valid = item["embedding_valid"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError("malformed object in range inspection") from exc
        if row_id < start or row_id >= end or row_id in result:
            raise PreparationError(f"duplicate or out-of-range row_id in service: {row_id}")
        if actual_uuid != object_uuid(row_id):
            raise PreparationError(
                f"deterministic UUID mismatch for row_id={row_id}: {actual_uuid}"
            )
        if not isinstance(embedding_valid, bool):
            raise PreparationError(f"embedding_valid is not boolean for row_id={row_id}")
        result[row_id] = {
            "id": actual_uuid,
            "embedding_valid": embedding_valid,
        }
    return result, used


def verify_completed_prefix(
    base_url: str,
    start: int,
    end: int,
    *,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    if end == start:
        return {"range": [start, end], "expected_count": 0, "actual_count": 0}
    actual, count_retries = query_count(
        base_url, _where_for_range(start, end), timeout=timeout, retries=retries
    )
    expected = end - start
    if actual != expected:
        raise PreparationError(
            f"completed checkpoint interval is absent or changed: "
            f"range=[{start},{end}) expected={expected} actual={actual}"
        )
    boundary_rows: dict[str, Any] = {}
    inspection_retries = 0
    for row_id in sorted({start, end - 1}):
        objects, used = inspect_range(
            base_url, row_id, row_id + 1, timeout=timeout, retries=retries
        )
        inspection_retries += used
        if set(objects) != {row_id}:
            raise PreparationError(f"checkpoint boundary row is missing: {row_id}")
        boundary_rows[str(row_id)] = objects[row_id]
    return {
        "range": [start, end],
        "expected_count": expected,
        "actual_count": actual,
        "boundary_rows": boundary_rows,
        "http_retries": count_retries + inspection_retries,
        "proof": "aggregate cardinality plus deterministic boundary UUIDs; per-batch ACKs are in checkpoint",
    }


def validate_batch_response(payload: Any, expected_ids: Sequence[str]) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise PreparationError("batch response must be a list")
    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise PreparationError("batch request contains duplicate UUIDs")
    observed: set[str] = set()
    failures: list[str] = []
    for item in payload:
        if not isinstance(item, Mapping):
            failures.append("non-object response item")
            continue
        object_id = str(item.get("id", ""))
        if not object_id or object_id in observed:
            failures.append(f"missing or duplicate response id {object_id!r}")
            continue
        observed.add(object_id)
        result = item.get("result")
        if not isinstance(result, Mapping):
            failures.append(f"{object_id}: missing result")
            continue
        status = str(result.get("status", "")).upper()
        errors = result.get("errors")
        if status != "SUCCESS" or errors not in (None, {}, [], ""):
            failures.append(
                f"{object_id}: status={status!r} errors={json.dumps(errors, sort_keys=True)}"
            )
    if observed != expected:
        failures.append(
            f"response ID set mismatch missing={sorted(expected - observed)[:5]} "
            f"extra={sorted(observed - expected)[:5]}"
        )
    if failures:
        raise PreparationError("batch object failures: " + "; ".join(failures[:10]))
    return {"object_count": len(observed), "response_sha256": sha256_value(payload)}


def _send_batch_once(
    base_url: str,
    objects: Sequence[dict[str, Any]],
    *,
    timeout: float,
) -> Any:
    payload, _ = baseline.request_json(
        base_url,
        "/v1/batch/objects",
        {"objects": list(objects)},
        method="POST",
        timeout=timeout,
        retries=0,
    )
    return payload


def import_batch_with_retry(
    objects: Sequence[dict[str, Any]],
    *,
    send: Callable[[Sequence[dict[str, Any]]], Any],
    inspect: Callable[[int, int], Mapping[int, Mapping[str, Any]]],
    max_retries: int,
    backoff_seconds: float,
    reconcile_before_first_attempt: bool = False,
) -> BatchResult:
    if not objects:
        raise ValueError("cannot import an empty batch")
    by_row = {int(item["properties"]["row_id"]): item for item in objects}
    rows = sorted(by_row)
    if rows != list(range(rows[0], rows[-1] + 1)) or len(by_row) != len(objects):
        raise PreparationError("batch row_ids must be unique and contiguous")
    all_ids = [str(by_row[row_id]["id"]) for row_id in rows]
    pending = list(objects)
    recovered: set[int] = set()
    last_error: BaseException | None = None

    def reconcile() -> None:
        nonlocal pending
        live = inspect(rows[0], rows[-1] + 1)
        unexpected = set(live) - set(rows)
        if unexpected:
            raise PreparationError(
                f"reconciliation observed unexpected row_ids: {sorted(unexpected)[:5]}"
            )
        for row_id, evidence in live.items():
            expected_valid = bool(by_row[row_id]["properties"]["embedding_valid"])
            if evidence.get("id") != by_row[row_id]["id"]:
                raise PreparationError(f"reconciliation UUID mismatch for row_id={row_id}")
            if evidence.get("embedding_valid") is not expected_valid:
                raise PreparationError(
                    f"reconciliation embedding_valid mismatch for row_id={row_id}"
                )
        recovered.update(live)
        pending = [by_row[row_id] for row_id in rows if row_id not in live]

    if reconcile_before_first_attempt:
        reconcile()
        if not pending:
            return BatchResult(
                attempts=0,
                retry_count=0,
                recovered_objects=len(recovered),
                response_sha256=sha256_value(
                    {"recovered_uncheckpointed_batch": rows}
                ),
                acknowledged_ids_sha256=sha256_value(all_ids),
            )

    for attempt in range(max_retries + 1):
        try:
            response = send(pending)
            result = validate_batch_response(
                response, [str(item["id"]) for item in pending]
            )
            return BatchResult(
                attempts=attempt + 1,
                retry_count=attempt,
                recovered_objects=len(recovered),
                response_sha256=result["response_sha256"],
                acknowledged_ids_sha256=sha256_value(all_ids),
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            last_error = exc
            if attempt >= max_retries:
                break
            reconcile()
            if not pending:
                return BatchResult(
                    attempts=attempt + 1,
                    retry_count=attempt,
                    recovered_objects=len(recovered),
                    response_sha256=sha256_value(
                        {"recovered_after_ambiguous_batch": sorted(recovered)}
                    ),
                    acknowledged_ids_sha256=sha256_value(all_ids),
                )
            if backoff_seconds:
                time.sleep(backoff_seconds * (2**attempt))
    raise PreparationError(
        f"batch failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def validate_csv_header(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8") as source:
        header = tuple(next(csv.reader(source), ()))
    if header != CSV_COLUMNS:
        raise PreparationError(
            f"CSV header mismatch: expected={CSV_COLUMNS!r} actual={header!r}"
        )


def iter_csv_batches(
    path: Path, *, start: int, end: int, batch_size: int
) -> Iterator[list[tuple[int, dict[str, str]]]]:
    if start < 0 or end <= start or end > EXPECTED_ROWS:
        raise ValueError(f"invalid CSV interval [{start}, {end})")
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise PreparationError("CSV header changed after input fingerprinting")
        batch: list[tuple[int, dict[str, str]]] = []
        seen_rows = 0
        for row_index, row in enumerate(reader):
            seen_rows = row_index + 1
            if row_index < start:
                continue
            if row_index >= end:
                raise PreparationError(
                    f"CSV has more than the formal {end} rows; first extra row={row_index}"
                )
            try:
                source_id = int(row["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError(f"invalid CSV id at row {row_index}") from exc
            if source_id != row_index:
                raise PreparationError(
                    f"CSV row/id alignment mismatch at row={row_index}: id={source_id}"
                )
            batch.append((row_index, row))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if seen_rows != end:
            raise PreparationError(f"CSV row count mismatch: expected={end} actual={seen_rows}")
        if batch:
            yield batch


def row_to_object(
    row_id: int, row: Mapping[str, str], vector: np.ndarray
) -> dict[str, Any]:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise PreparationError(f"row_id={row_id} has an invalid vector")
    has_price = parse_bool(row["has_price"], "has_price")
    embedding_valid = bool(np.linalg.norm(array.astype(np.float64)) > 0.0)
    price_text = row["price"].strip()
    if has_price and not price_text:
        raise PreparationError(f"row_id={row_id} has_price=true but price is empty")
    properties = {
        "row_id": row_id,
        "rating": float(row["rating"]),
        "verified_purchase": parse_bool(
            row["verified_purchase"], "verified_purchase"
        ),
        "helpful_vote": int(row["helpful_vote"]),
        "review_text_len": int(row["review_text_len"]),
        "main_category": row["main_category"],
        "price": float(price_text) if has_price else 0.0,
        "has_price": has_price,
        "item_rating_number": int(float(row["item_rating_number"])),
        "embedding_valid": embedding_valid,
    }
    for name, value in properties.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise PreparationError(f"row_id={row_id} property {name} is non-finite")
    return {
        "class": CLASS_NAME,
        "id": object_uuid(row_id),
        "properties": properties,
        "vector": array.tolist(),
    }


def completed_end(
    completed_batches: Sequence[Mapping[str, Any]], start: int, end: int
) -> int:
    position = start
    for index, record in enumerate(completed_batches):
        try:
            batch_no = int(record["batch_no"])
            batch_start, batch_end = (int(value) for value in record["row_range"])
            object_count = int(record["object_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError(f"malformed checkpoint batch {index}") from exc
        if batch_no != index:
            raise PreparationError(f"checkpoint batch_no mismatch at batch {index}")
        if batch_start != position or batch_end <= batch_start or batch_end > end:
            raise PreparationError(
                f"checkpoint batches are not a contiguous prefix at batch {index}"
            )
        if object_count != batch_end - batch_start:
            raise PreparationError(f"checkpoint batch {index} object count mismatch")
        if record.get("status") != "acknowledged":
            raise PreparationError(f"checkpoint batch {index} status is not acknowledged")
        for field in (
            "payload_sha256",
            "response_sha256",
            "acknowledged_ids_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))):
                raise PreparationError(
                    f"checkpoint batch {index} has invalid {field}"
                )
        position = batch_end
    return position


def checkpoint_specification(
    *,
    inputs: Mapping[str, Any],
    implementation: Mapping[str, Any],
    schema: Mapping[str, Any],
    service: Mapping[str, Any],
    start: int,
    end: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "artifact": "weaviate_amazon10m_formal_import",
        "class": CLASS_NAME,
        "row_range": [start, end],
        "expected_rows": EXPECTED_ROWS,
        "expected_embedding_valid_rows": EXPECTED_VALID_ROWS,
        "batch_size": batch_size,
        "uuid_algorithm": UUID_ALGORITHM,
        "inputs": inputs,
        "implementation": implementation,
        "requested_schema_sha256": sha256_value(schema),
        "service": dict(service),
    }


def initialize_or_resume_checkpoint(
    path: Path,
    specification: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    specification_hash = sha256_value(specification)
    if resume:
        if not path.is_file():
            raise PreparationError(f"resume checkpoint does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"cannot read checkpoint: {path}") from exc
        if not isinstance(payload, dict):
            raise PreparationError("checkpoint root must be a JSON object")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise PreparationError("checkpoint schema version mismatch")
        if payload.get("specification_sha256") != specification_hash:
            raise PreparationError("checkpoint specification/input provenance mismatch")
        if payload.get("specification") != specification:
            raise PreparationError("checkpoint specification payload mismatch")
        if payload.get("status") not in {"in_progress", "complete"}:
            raise PreparationError("checkpoint status is invalid")
        if not isinstance(payload.get("completed_batches"), list):
            raise PreparationError("checkpoint completed_batches is invalid")
        return payload
    if path.exists():
        raise PreparationError(
            f"checkpoint already exists; use --resume or choose a new path: {path}"
        )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "in_progress",
        "specification_sha256": specification_hash,
        "specification": dict(specification),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_batches": [],
        "resume_proofs": [],
        "counters": {
            "batch_attempts": 0,
            "batch_retries": 0,
            "recovered_objects": 0,
        },
    }


def wait_for_ready(
    base_url: str,
    *,
    timeout: float,
    retries: int,
    ready_timeout: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + ready_timeout
    attempts = 0
    last_error: BaseException | None = None
    while True:
        attempts += 1
        try:
            nodes, used = baseline.get_ready_nodes(base_url, timeout, retries)
            return {"attempts": attempts, "http_retries": used, "nodes": nodes}
        except RuntimeError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise PreparationError(
                f"Weaviate HNSW did not become ready within {ready_timeout}s: {last_error}"
            ) from last_error
        time.sleep(poll_seconds)


def completion_gates(
    base_url: str,
    filters: Sequence[baseline.FilterSpec],
    *,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    total, total_retries = query_count(
        base_url, None, timeout=timeout, retries=retries
    )
    valid_where = {
        "path": ["embedding_valid"],
        "operator": "Equal",
        "valueBoolean": True,
    }
    valid, valid_retries = query_count(
        base_url, valid_where, timeout=timeout, retries=retries
    )
    errors: list[str] = []
    if total != EXPECTED_ROWS:
        errors.append(f"total expected={EXPECTED_ROWS} actual={total}")
    if valid != EXPECTED_VALID_ROWS:
        errors.append(
            f"embedding_valid expected={EXPECTED_VALID_ROWS} actual={valid}"
        )
    filter_counts: dict[str, dict[str, Any]] = {}
    filter_retries = 0
    for spec in filters:
        actual, used = query_count(
            base_url, spec.where, timeout=timeout, retries=retries
        )
        filter_retries += used
        filter_counts[spec.name] = {
            "target_rate": spec.target_rate,
            "expected": spec.expected_rows,
            "actual": actual,
            "passed": actual == spec.expected_rows,
        }
        if actual != spec.expected_rows:
            errors.append(
                f"{spec.name} expected={spec.expected_rows} actual={actual}"
            )
    if errors:
        raise PreparationError("completion count gates failed: " + "; ".join(errors))
    return {
        "passed": True,
        "total_rows": {"expected": EXPECTED_ROWS, "actual": total},
        "embedding_valid_rows": {
            "expected": EXPECTED_VALID_ROWS,
            "actual": valid,
        },
        "filter_counts": filter_counts,
        "http_retries": total_retries + valid_retries + filter_retries,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.start_row != 0 or args.end_row != EXPECTED_ROWS:
        raise PreparationError(
            "formal corpus preparation requires the explicit full range "
            f"[0,{EXPECTED_ROWS}); partial imports cannot satisfy completion gates"
        )
    if args.batch_size <= 0 or args.batch_size > 5_000:
        raise PreparationError("--batch-size must be in [1, 5000]")
    if args.batch_retries < 0:
        raise PreparationError("--batch-retries must be non-negative")
    if args.http_retries < 0 or args.batch_backoff_seconds < 0:
        raise PreparationError("retry counts/backoff must be non-negative")
    if args.timeout <= 0 or args.ready_timeout <= 0 or args.ready_poll_seconds <= 0:
        raise PreparationError("timeout values must be positive")
    if args.progress_batches <= 0:
        raise PreparationError("--progress-batches must be positive")
    if args.checkpoint.resolve() == args.manifest.resolve():
        raise PreparationError("checkpoint and manifest paths must be different")
    validate_image_digest(args.service_image_digest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    validate_csv_header(args.csv)
    inputs = {
        "attributes_csv": file_identity(args.csv),
        "vectors_fbin": file_identity(args.fbin),
        "filters_csv": file_identity(args.filters_csv),
    }
    vectors, rows, dimensions = baseline.read_fbin_memmap(args.fbin, EXPECTED_ROWS)
    if rows != EXPECTED_ROWS or dimensions != 128:
        raise PreparationError(
            f"formal fbin shape mismatch: expected=({EXPECTED_ROWS},128) "
            f"actual=({rows},{dimensions})"
        )
    expected_fbin_size = 8 + rows * dimensions * np.dtype("<f4").itemsize
    if inputs["vectors_fbin"]["size_bytes"] != expected_fbin_size:
        raise PreparationError(
            f"formal fbin size mismatch: expected={expected_fbin_size} "
            f"actual={inputs['vectors_fbin']['size_bytes']}"
        )
    inputs["attributes_csv"]["columns"] = list(CSV_COLUMNS)
    inputs["vectors_fbin"].update(
        {"header_rows": rows, "dimensions": dimensions, "dtype": "little-endian float32"}
    )
    implementation = {
        "script": file_identity(Path(__file__)),
    }
    filters = baseline.load_filter_specs(args.filters_csv)
    if len(filters) != 14:
        raise PreparationError("formal filter suite must contain exactly 14 predicates")

    requested_schema = expected_schema(
        ef_construction=args.ef_construction,
        max_connections=args.max_connections,
        flat_search_cutoff=args.flat_search_cutoff,
        filter_strategy=args.filter_strategy,
    )
    meta, meta_retries = baseline.request_json(
        args.base_url, "/v1/meta", timeout=args.timeout, retries=args.http_retries
    )
    service = validate_service_identity(
        meta, args.expected_weaviate_version, args.service_image_digest
    )
    live_schema, schema_created, schema_retries = ensure_schema(
        args.base_url,
        requested_schema,
        timeout=args.timeout,
        retries=args.http_retries,
    )
    specification = checkpoint_specification(
        inputs=inputs,
        implementation=implementation,
        schema=requested_schema,
        service=service,
        start=args.start_row,
        end=args.end_row,
        batch_size=args.batch_size,
    )
    checkpoint = initialize_or_resume_checkpoint(
        args.checkpoint, specification, resume=args.resume
    )
    next_row = completed_end(
        checkpoint["completed_batches"], args.start_row, args.end_row
    )

    total_before, total_before_retries = query_count(
        args.base_url, None, timeout=args.timeout, retries=args.http_retries
    )
    if not args.resume and total_before != 0:
        raise PreparationError(
            "new formal import requires an empty class; live count cannot be used "
            f"as a CSV offset (actual={total_before})"
        )
    if args.resume:
        proof = verify_completed_prefix(
            args.base_url,
            args.start_row,
            next_row,
            timeout=args.timeout,
            retries=args.http_retries,
        )
        next_batch_end = min(next_row + args.batch_size, args.end_row)
        in_flight_count = 0
        in_flight_retries = 0
        if next_row < args.end_row:
            in_flight_count, in_flight_retries = query_count(
                args.base_url,
                _where_for_range(next_row, next_batch_end),
                timeout=args.timeout,
                retries=args.http_retries,
            )
        if total_before != (next_row - args.start_row) + in_flight_count:
            raise PreparationError(
                "resume found objects outside the checkpointed prefix and its "
                "single possible in-flight batch"
            )
        proof.update(
            {
                "total_count_at_resume": total_before,
                "possible_in_flight_range": [next_row, next_batch_end],
                "possible_in_flight_count": in_flight_count,
                "http_retries": proof.get("http_retries", 0) + in_flight_retries,
                "verified_at": utc_now(),
            }
        )
        checkpoint["resume_proofs"].append(proof)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(args.checkpoint, checkpoint)

    started = time.perf_counter()
    resumed_first_pending = args.resume and next_row < args.end_row
    for csv_batch in iter_csv_batches(
        args.csv, start=next_row, end=args.end_row, batch_size=args.batch_size
    ):
        objects = [
            row_to_object(row_id, row, vectors[row_id])
            for row_id, row in csv_batch
        ]
        batch_start = csv_batch[0][0]
        batch_end = csv_batch[-1][0] + 1

        result = import_batch_with_retry(
            objects,
            send=lambda values: _send_batch_once(
                args.base_url, values, timeout=args.timeout
            ),
            inspect=lambda start, end: inspect_range(
                args.base_url,
                start,
                end,
                timeout=args.timeout,
                retries=args.http_retries,
            )[0],
            max_retries=args.batch_retries,
            backoff_seconds=args.batch_backoff_seconds,
            reconcile_before_first_attempt=resumed_first_pending,
        )
        resumed_first_pending = False

        record = {
            "batch_no": len(checkpoint["completed_batches"]),
            "status": "acknowledged",
            "row_range": [batch_start, batch_end],
            "object_count": batch_end - batch_start,
            "payload_sha256": sha256_value(objects),
            "response_sha256": result.response_sha256,
            "acknowledged_ids_sha256": sha256_value(
                [str(item["id"]) for item in objects]
            ),
            "attempts": result.attempts,
            "retry_count": result.retry_count,
            "recovered_objects": result.recovered_objects,
            "completed_at": utc_now(),
        }
        checkpoint["completed_batches"].append(record)
        checkpoint["counters"]["batch_attempts"] += result.attempts
        checkpoint["counters"]["batch_retries"] += result.retry_count
        checkpoint["counters"]["recovered_objects"] += record["recovered_objects"]
        checkpoint["updated_at"] = utc_now()
        atomic_write_json(args.checkpoint, checkpoint)
        next_row = batch_end
        if len(checkpoint["completed_batches"]) % args.progress_batches == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "import_progress",
                        "completed_end": next_row,
                        "target_end": args.end_row,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_end = completed_end(
        checkpoint["completed_batches"], args.start_row, args.end_row
    )
    if final_end != args.end_row:
        raise PreparationError(
            f"import checkpoint is incomplete: expected_end={args.end_row} actual={final_end}"
        )
    final_prefix_proof = verify_completed_prefix(
        args.base_url,
        args.start_row,
        args.end_row,
        timeout=args.timeout,
        retries=args.http_retries,
    )
    ready = wait_for_ready(
        args.base_url,
        timeout=args.timeout,
        retries=args.http_retries,
        ready_timeout=args.ready_timeout,
        poll_seconds=args.ready_poll_seconds,
    )
    gates = completion_gates(
        args.base_url, filters, timeout=args.timeout, retries=args.http_retries
    )
    final_schema, final_schema_retries = baseline.request_json(
        args.base_url,
        f"/v1/schema/{CLASS_NAME}",
        timeout=args.timeout,
        retries=args.http_retries,
    )
    verify_formal_schema(final_schema, requested_schema)
    final_meta, final_meta_retries = baseline.request_json(
        args.base_url, "/v1/meta", timeout=args.timeout, retries=args.http_retries
    )
    final_service = validate_service_identity(
        final_meta, args.expected_weaviate_version, args.service_image_digest
    )
    if final_service != service:
        raise PreparationError("Weaviate service identity changed during import")

    checkpoint["status"] = "complete"
    checkpoint["updated_at"] = utc_now()
    checkpoint["completion_gates"] = gates
    checkpoint["final_prefix_proof"] = final_prefix_proof
    atomic_write_json(args.checkpoint, checkpoint)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact": "weaviate_amazon10m_formal_corpus",
        "status": "complete",
        "created_at": utc_now(),
        "candidate_universe": {
            "predicate": "embedding_valid",
            "expected_rows": EXPECTED_VALID_ROWS,
        },
        "input_files": inputs,
        "implementation": implementation,
        "git_revision_at_finalization": baseline.git_revision(),
        "service": {
            **service,
            "meta_initial": meta,
            "meta_final": final_meta,
            "meta_http_retries": meta_retries + final_meta_retries,
        },
        "schema": {
            "created_by_this_run": schema_created,
            "requested": requested_schema,
            "requested_sha256": sha256_value(requested_schema),
            "readback_initial": live_schema,
            "readback_final": final_schema,
            "readback_final_sha256": sha256_value(final_schema),
            "http_retries": schema_retries + final_schema_retries,
        },
        "import": {
            "row_range": [args.start_row, args.end_row],
            "batch_size": args.batch_size,
            "batch_count": len(checkpoint["completed_batches"]),
            "completed_batches": checkpoint["completed_batches"],
            "counters": checkpoint["counters"],
            "resume_proofs": checkpoint["resume_proofs"],
            "final_prefix_proof": final_prefix_proof,
            "uuid_algorithm": UUID_ALGORITHM,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_specification_sha256": checkpoint[
                "specification_sha256"
            ],
            "total_before_import": total_before,
            "total_before_http_retries": total_before_retries,
        },
        "index_ready_gate": ready,
        "completion_gates": gates,
        "artifact_valid": True,
    }
    atomic_write_json(args.manifest, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Amazon-10M Weaviate corpus preparation/import"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--batch-retries", type=int, default=5)
    parser.add_argument("--batch-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--http-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--ready-timeout", type=float, default=14_400.0)
    parser.add_argument("--ready-poll-seconds", type=float, default=10.0)
    parser.add_argument("--progress-batches", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-weaviate-version", default="1.38.0")
    parser.add_argument(
        "--service-image-digest",
        required=True,
        help="immutable image digest; automatic container-runtime discovery is not assumed",
    )
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--flat-search-cutoff", type=int, default=40_000)
    parser.add_argument("--filter-strategy", default="acorn")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except (PreparationError, ValueError, OSError) as exc:
        print(f"preparation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifact_valid": manifest["artifact_valid"],
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
