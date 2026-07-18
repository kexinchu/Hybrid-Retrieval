"""Fail-closed finalizer for the Amazon-10M five-way recall comparison.

The runners in this repository intentionally do not share one CSV schema.  This
sidecar only adapts their already-written raw/final or summary rows into one
small comparison schema.  It never runs a database, a vector service, or Faiss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FINALIZER_VERSION = "amazon10m-five-way-matched-recall-finalizer-v1"
DATASET = "amazon10m"
CANDIDATE_UNIVERSE = "embedding_valid"
EXPECTED_CANDIDATE_ROWS = 9_979_556
METHODS = (
    "official_pgvector",
    "sqlens_disabled",
    "sqlens_enabled",
    "faiss_allowlist",
    "weaviate_production",
)
QUERY_NOS = tuple(range(100, 200))
UNAVAILABLE_STATUSES = {"unavailable", "not_available", "unattainable_on_grid"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalizationFailure(ValueError):
    """Raised when an artifact cannot prove that it belongs in the matrix."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationFailure(f"JSON artifact is not an object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise FinalizationFailure(f"CSV has no header: {path}")
            return list(reader), tuple(reader.fieldnames)
    except (OSError, csv.Error) as exc:
        raise FinalizationFailure(f"cannot read CSV {path}: {exc}") from exc


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _path(value: Any, base: Path, label: str) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not value:
        raise FinalizationFailure(f"missing {label} path")
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _sha(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if not SHA256_RE.fullmatch(text):
        raise FinalizationFailure(f"{label} is not a SHA-256 value")
    return text


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationFailure(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise FinalizationFailure(f"{label} is not finite")
    return result


def _int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationFailure(f"{label} is not an integer") from exc


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok", "complete", "valid"}


def _canonical_method(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "official": "official_pgvector",
        "pgvector": "official_pgvector",
        "stock": "official_pgvector",
        "original": "official_pgvector",
        "official_pgvector": "official_pgvector",
        "sqlens_off": "sqlens_disabled",
        "sqlens_disabled": "sqlens_disabled",
        "disabled": "sqlens_disabled",
        "sqlens_off_original": "sqlens_disabled",
        "sqlens_on": "sqlens_enabled",
        "sqlens_enabled": "sqlens_enabled",
        "enabled": "sqlens_enabled",
        "design1_bloom": "sqlens_enabled",
        "faiss": "faiss_allowlist",
        "faiss_allowlist": "faiss_allowlist",
        "weaviate": "weaviate_production",
        "weaviate_production": "weaviate_production",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise FinalizationFailure(f"unknown comparison method: {value!r}") from exc


def _number_list(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise FinalizationFailure(f"{label} is absent or not a list")
    result = tuple(sorted({_float(item, label) for item in values}))
    if not result or any(item <= 0.0 or item > 1.0 for item in result):
        raise FinalizationFailure(f"{label} must contain recalls in (0, 1]")
    return result


def _names(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = tuple(str(item) for item in value)
    else:
        raise FinalizationFailure(f"{label} is absent or not a list")
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise FinalizationFailure(f"{label} is empty or duplicated")
    return result


def _metadata_value(manifest: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _nested(manifest, *path)
        if value is not None:
            return value
    return None


def _dataset(manifest: Mapping[str, Any]) -> str:
    value = _metadata_value(
        manifest,
        ("dataset",), ("run_spec", "dataset"), ("run_contract", "dataset"),
    )
    if value is None:
        artifact = " ".join(
            str(manifest.get(key, "")).lower() for key in ("artifact", "benchmark", "class")
        )
        value = DATASET if "amazon" in artifact or "grocery" in artifact else None
    if value is None or "amazon" not in str(value).lower():
        raise FinalizationFailure("artifact does not identify the Amazon-10M dataset")
    return DATASET


def _filter_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    value = _metadata_value(
        manifest, ("filter_names",), ("run_spec", "filter_names"), ("config", "filter_names"),
        ("args", "filter_names")
    )
    if value is None:
        filters = manifest.get("filters", _nested(manifest, "run_spec", "filters"))
        if isinstance(filters, Sequence) and not isinstance(filters, (str, bytes, bytearray)):
            value = [item.get("name", item.get("filter_name")) for item in filters if isinstance(item, Mapping)]
    return _names(value, "filter_names")


def _targets(manifest: Mapping[str, Any]) -> tuple[float, ...]:
    value = _metadata_value(
        manifest, ("target_recalls",), ("run_spec", "target_recalls"), ("config", "targets"),
        ("args", "target_recalls"), ("targets",),
    )
    return _number_list(value, "target_recalls")


def _query_nos(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    candidates = (
        _nested(manifest, "query_cohort", "query_nos"),
        _nested(manifest, "query_cohort", "final_query_nos"),
        _nested(manifest, "query_splits", "final_query_nos"),
        _nested(manifest, "run_spec", "query_splits", "final_query_nos"),
        _nested(manifest, "final", "query_nos"),
        _nested(manifest, "run_spec", "final", "query_nos"),
    )
    value = next((item for item in candidates if item is not None), None)
    if value is None:
        offset = _metadata_value(
            manifest, ("args", "final_query_offset"), ("run_contract", "final_query_offset"),
        )
        count = _metadata_value(
            manifest, ("args", "final_queries"), ("run_contract", "final_queries"),
        )
        if offset is not None and count is not None:
            value = list(range(_int(offset, "final_query_offset"), _int(offset, "final_query_offset") + _int(count, "final_queries")))
    if value is None:
        raise FinalizationFailure("manifest does not declare the held-out query cohort")
    try:
        result = tuple(sorted({_int(item, "query_no") for item in value}))
    except TypeError as exc:
        raise FinalizationFailure("manifest query cohort is not a list") from exc
    return result


def _gt_hash(manifest: Mapping[str, Any]) -> str:
    value = _metadata_value(
        manifest,
        ("query_cohort_sha256",), ("query_cohort", "gt_hash"),
        ("query_cohort", "query_cohort_sha256"), ("truth_sha256",),
        ("source_hashes", "truth"), ("source_hashes", "exact_truth"),
        ("source_hashes", "truth_csv"), ("source_hashes", "exact_truth_csv"),
    )
    if value is None:
        truth = _nested(manifest, "inputs", "truth")
        if isinstance(truth, Mapping):
            value = truth.get("sha256")
    return _sha(value, "held-out GT/cohort hash")


def _latency_scope(manifest: Mapping[str, Any]) -> str:
    value = _metadata_value(
        manifest, ("latency_scope",), ("latency_definition",),
        ("query_latency_definition",), ("execution", "latency"),
        ("timing_definition",), ("run_spec", "latency_scope"),
    )
    return _scope_from_value(value)


def _candidate_universe(manifest: Mapping[str, Any]) -> None:
    value = _metadata_value(
        manifest,
        ("candidate_validity_predicate",),
        ("args", "candidate_validity_predicate"),
        ("run_spec", "candidate_validity_predicate"),
        ("config", "candidate_universe", "predicate"),
        ("run_spec", "candidate_universe", "predicate"),
        ("postgres", "candidate_universe", "predicate"),
        ("service", "candidate_universe", "predicate"),
    )
    if " ".join(str(value or "").strip().split()) != CANDIDATE_UNIVERSE:
        raise FinalizationFailure("candidate universe is not explicitly embedding_valid")
    rows = _metadata_value(
        manifest,
        ("candidate_universe", "observed_rows"),
        ("candidate_universe", "expected_rows"),
        ("config", "candidate_universe", "observed_rows"),
        ("config", "candidate_universe", "expected_rows"),
        ("run_spec", "candidate_universe", "expected_rows"),
        ("postgres", "candidate_universe", "rows"),
        ("service", "candidate_universe", "count"),
    )
    if rows is not None and _int(rows, "candidate universe rows") != EXPECTED_CANDIDATE_ROWS:
        raise FinalizationFailure("candidate universe row count mismatch")


def _scope_from_value(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        raise FinalizationFailure("latency scope is not explicitly declared")
    if "search_only" in text or "search only" in text:
        return "search_only"
    if "end_to_end" in text or "e2e" in text:
        return "end_to_end"
    raise FinalizationFailure(f"unsupported latency scope: {value!r}")


def _status_complete(manifest: Mapping[str, Any]) -> None:
    status = manifest.get("status")
    if manifest.get("artifact_valid") is not True or (status is not None and str(status).lower() not in {"complete", "valid"}):
        raise FinalizationFailure("source manifest is not complete and artifact_valid=true")
    if manifest.get("fatal_error") or manifest.get("validation_errors") or manifest.get("artifact_errors"):
        raise FinalizationFailure("source manifest contains a fatal/validation error")


def _runner_sha(manifest: Mapping[str, Any]) -> str:
    candidates = (
        _nested(manifest, "software_versions", "measurement_runner_sha256"),
        _nested(manifest, "inputs", "runner", "sha256"),
        _nested(manifest, "source_hashes", "runner"),
        _nested(manifest, "source_hashes", "baseline_runner"),
        manifest.get("measurement_runner_sha256"),
        manifest.get("runner_sha256"),
        _nested(manifest, "provenance", "runner_sha256"),
    )
    value = next((item for item in candidates if item), None)
    return _sha(value, "measurement runner SHA")


def _provenance(manifest: Mapping[str, Any], method: str) -> dict[str, Any]:
    runner_sha = _runner_sha(manifest)
    sqlens = _metadata_value(
        manifest, ("sqlens_runtime_provenance",), ("sqlens_provenance",),
        ("provenance", "sqlens"),
    )
    service = _metadata_value(
        manifest, ("service_provenance",), ("service_identity",), ("service",), ("weaviate",),
        ("provenance", "service"),
    )
    binary = _metadata_value(
        manifest, ("binary_provenance",), ("backend",), ("postgres",),
        ("source_db",), ("database",), ("provenance", "binary"),
    )
    if method == "sqlens_enabled":
        if not isinstance(sqlens, Mapping):
            raise FinalizationFailure("SQLens-enabled artifact lacks runtime provenance")
        build_id = sqlens.get("loaded_sqlens_build_id", sqlens.get("build_id", sqlens.get("sqlens_build_id")))
        vector_sha = sqlens.get("loaded_vector_so_sha256", sqlens.get("vector_so_sha256"))
        if not build_id or not SHA256_RE.fullmatch(str(vector_sha or "")):
            raise FinalizationFailure("SQLens-enabled runtime provenance is incomplete")
        if sqlens.get("runtime_sqlens_identity_complete") is False:
            raise FinalizationFailure("SQLens-enabled runtime identity is incomplete")
    elif method == "weaviate_production":
        if not isinstance(service, Mapping):
            raise FinalizationFailure("Weaviate artifact lacks service provenance")
        digest = service.get("service_image_digest", service.get("image_digest", service.get("image_sha256")))
        version = service.get("version", service.get("service_version"))
        if not digest or not version:
            raise FinalizationFailure("Weaviate service provenance lacks version/image digest")
    elif not isinstance(binary, Mapping):
        raise FinalizationFailure(f"{method} artifact lacks binary/database provenance")
    value = {
        "runner_sha256": runner_sha,
        "binary": binary,
        "service": service,
        "sqlens": sqlens,
    }
    # Hash the exact retained provenance, so the final manifest binds it.
    value["provenance_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return value


def _declared_hash(manifest: Mapping[str, Any], path: Path, kind: str) -> str | None:
    output_names = (kind, "final", "final_raw", "raw", "summary") if kind else ("final", "raw", "summary")
    for root in ("output_sha256", "outputs", "hashes", "source_hashes"):
        container = manifest.get(root)
        if not isinstance(container, Mapping):
            continue
        for name in output_names:
            item = container.get(name)
            value = item.get("sha256") if isinstance(item, Mapping) else item
            if value and SHA256_RE.fullmatch(str(value).lower()):
                return str(value).lower()
    return None


def _manifest_artifact_entry(entry: Mapping[str, Any], base: Path) -> tuple[str, Path, dict[str, Any], Path, str, str]:
    method = _canonical_method(entry.get("method", entry.get("name")))
    manifest_path = _path(entry.get("manifest", entry.get("manifest_path")), base, f"{method} manifest")
    if not manifest_path.is_file():
        raise FinalizationFailure(f"missing {method} manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    _status_complete(manifest)
    csv_value = entry.get("csv", entry.get("csv_path"))
    kind = str(entry.get("kind", "")).strip().lower()
    if csv_value is None:
        outputs = manifest.get("outputs")
        if not isinstance(outputs, Mapping):
            raise FinalizationFailure(f"{method} manifest has no output CSV")
        for name in (kind, "final_raw", "final", "raw", "summary"):
            if outputs.get(name) is not None:
                csv_value, kind = outputs[name], name
                break
    csv_path = _path(csv_value, manifest_path.parent, f"{method} CSV")
    if not csv_path.is_file():
        raise FinalizationFailure(f"missing {method} CSV: {csv_path}")
    actual_manifest_sha = sha256_file(manifest_path)
    actual_csv_sha = sha256_file(csv_path)
    declared_manifest_sha = entry.get("manifest_sha256", entry.get("manifest_sha"))
    if declared_manifest_sha is None:
        raise FinalizationFailure(f"{method} manifest has no declared SHA-256")
    if _sha(declared_manifest_sha, f"{method} manifest SHA") != actual_manifest_sha:
        raise FinalizationFailure(f"{method} manifest SHA changed")
    declared_csv_sha = entry.get("csv_sha256", entry.get("csv_sha")) or _declared_hash(manifest, csv_path, kind)
    if declared_csv_sha is None:
        raise FinalizationFailure(f"{method} CSV has no declared SHA-256")
    if _sha(declared_csv_sha, f"{method} CSV SHA") != actual_csv_sha:
        raise FinalizationFailure(f"{method} CSV SHA changed")
    return method, csv_path, manifest, manifest_path, actual_manifest_sha, actual_csv_sha


def _target_from_row(row: Mapping[str, str], targets: Sequence[float]) -> tuple[float, ...]:
    for key in ("target_recall", "target"):
        if row.get(key, "").strip() not in {"", "NA", "n/a", "None"}:
            return (_float(row[key], key),)
    matched = row.get("matched_target_recalls", "")
    if matched.strip():
        values = []
        for item in matched.split(","):
            if item.strip():
                values.append(_float(item, "matched_target_recalls"))
        return tuple(values)
    # Some formal final CSVs measure one selected config shared by all targets.
    if row.get("phase", "").strip().lower() in {"final", "held_out", "final_raw"}:
        return tuple(targets)
    return ()


def _row_recall(row: Mapping[str, str]) -> float:
    for key in ("recall_at_10", "recall", "recall_mean", "query_recall"):
        if row.get(key, "").strip() != "":
            value = _float(row[key], key)
            if 0.0 <= value <= 1.0:
                return value
            raise FinalizationFailure(f"recall outside [0,1]: {value}")
    raise FinalizationFailure("CSV row has no recall metric")


def _row_latency(row: Mapping[str, str], scope: str) -> float:
    if scope == "end_to_end":
        keys = ("end_to_end_ms", "e2e_latency_ms", "latency_ms")
    else:
        keys = ("search_latency_ms", "query_ms", "latency_ms")
    for key in keys:
        if row.get(key, "").strip() not in {"", "NA", "n/a", "None"}:
            value = _float(row[key], key)
            if value > 0.0:
                return value
            raise FinalizationFailure(f"latency must be positive: {value}")
    raise FinalizationFailure(f"CSV row has no {scope} latency metric")


def _row_is_valid(row: Mapping[str, str]) -> bool:
    if "valid" not in row and "error" not in row:
        return not row.get("order_error", "").strip()
    return _bool(row.get("valid", "true")) and not row.get("error", "").strip() and not row.get("order_error", "").strip()


@dataclass(frozen=True)
class Cell:
    method: str
    filter_name: str
    target_recall: float
    status: str
    recall_mean: float | str
    latency_mean_ms: float | str
    latency_p50_ms: float | str
    latency_p95_ms: float | str
    latency_p99_ms: float | str
    queries: int
    samples: int
    unavailable_reason: str


def _summary_cell(rows: Sequence[Mapping[str, str]], method: str, filter_name: str, target: float, scope: str) -> Cell:
    matches = [row for row in rows if row.get("filter_name") == filter_name and _target_matches(_target_from_row(row, (target,)), target)]
    if len(matches) != 1:
        raise FinalizationFailure(f"{method} summary missing or duplicated cell: {filter_name}|{target:g}")
    row = matches[0]
    declared_row_scope = row.get("latency_scope", row.get("latency_definition", row.get("query_latency_definition", "")))
    if declared_row_scope and _scope_from_value(declared_row_scope) != scope:
        raise FinalizationFailure(f"{method} summary latency scope conflicts with manifest: {filter_name}|{target:g}")
    status = row.get("status", row.get("comparison_status", "valid")).strip().lower()
    outcome = row.get("target_outcome", "").strip().lower()
    if status in UNAVAILABLE_STATUSES or outcome in UNAVAILABLE_STATUSES or outcome in {"selected_but_final_unconfirmed", "unconfirmed"} or _bool(row.get("available", "true")) is False:
        return Cell(method, filter_name, target, "unavailable", "NA", "NA", "NA", "NA", "NA", 0, 0, row.get("unavailable_reason", row.get("reason", status)))
    if status not in {"valid", "complete", "attained", "selected_and_confirmed", "confirmed", "ok"}:
        raise FinalizationFailure(f"{method} cell is not valid: {filter_name}|{target:g}")
    aggregation = row.get("recall_aggregation", row.get("aggregation", "")).strip().lower()
    # Existing formal runners encode this contract through complete, queries=100,
    # and a query-bootstrap recall bound rather than a named aggregation column.
    legacy_query_mean = (
        not aggregation
        and str(row.get("queries", "")) == str(len(QUERY_NOS))
        and any(key in row for key in ("recall_lcb95", "recall_ci95_low", "recall_ci95_high"))
        and (
            _bool(row.get("complete", "false"))
            or row.get("status", "").strip().lower() in {"valid", "complete", "ok"}
        )
    )
    if aggregation not in {"query_level_mean", "query_mean", "mean_of_query_means"} and not legacy_query_mean:
        raise FinalizationFailure(f"{method} cell does not declare query-level mean recall: {filter_name}|{target:g}")
    recall = _float(row.get("recall_mean"), "recall_mean")
    queries = _int(row.get("queries", row.get("expected_queries", "-1")), "queries")
    if queries != len(QUERY_NOS):
        raise FinalizationFailure(f"{method} cell does not cover q100..199: {filter_name}|{target:g}")
    latency_key = "latency_mean_ms" if row.get("latency_mean_ms", "").strip() else "search_latency_mean_ms"
    latency = _float(row.get(latency_key), latency_key)
    p50 = _float(row.get("latency_p50_ms", row.get("search_latency_p50_ms")), "latency_p50_ms")
    p95 = _float(row.get("latency_p95_ms", row.get("search_latency_p95_ms")), "latency_p95_ms")
    p99 = _float(row.get("latency_p99_ms", row.get("search_latency_p99_ms")), "latency_p99_ms")
    if recall < target:
        return Cell(method, filter_name, target, "unavailable", recall, latency, p50, p95, p99, queries, _int(row.get("samples", queries), "samples"), "query_level_mean_below_target")
    return Cell(method, filter_name, target, "valid", recall, latency, p50, p95, p99, queries, _int(row.get("samples", queries), "samples"), "")


def _target_matches(values: Sequence[float], target: float) -> bool:
    return any(math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-9) for value in values)


def _raw_cells(rows: Sequence[Mapping[str, str]], method: str, filters: Sequence[str], targets: Sequence[float], scope: str) -> dict[tuple[str, float], Cell]:
    grouped: dict[tuple[str, float, int], list[tuple[float, float]]] = {}
    observed_repeats: set[tuple[str, float, int, int]] = set()
    for row in rows:
        phase = row.get("phase", "").strip().lower()
        if phase and phase not in {"final", "held_out", "final_raw"}:
            continue
        filter_name = row.get("filter_name", "").strip()
        if filter_name not in filters:
            raise FinalizationFailure(f"{method} raw CSV has unexpected filter: {filter_name!r}")
        if not _row_is_valid(row):
            raise FinalizationFailure(f"{method} raw CSV contains invalid measurement")
        query_no = _int(row.get("query_no"), "query_no")
        if query_no not in QUERY_NOS:
            raise FinalizationFailure(f"{method} raw CSV is outside held-out q100..199: {query_no}")
        if row.get("latency_definition") or row.get("query_latency_definition"):
            declared = row.get("latency_definition") or row.get("query_latency_definition")
            row_scope = "search_only" if "search" in declared.lower() else "end_to_end" if "end" in declared.lower() or "e2e" in declared.lower() else ""
            if row_scope != scope:
                raise FinalizationFailure(f"{method} row latency scope conflicts with manifest")
        repeat = _int(row.get("repeat", "0"), "repeat")
        recall = _row_recall(row)
        latency = _row_latency(row, scope)
        for target in _target_from_row(row, targets):
            if not _target_matches(targets, target):
                raise FinalizationFailure(f"{method} raw CSV has unexpected target: {target}")
            key = (filter_name, min(targets, key=lambda item: abs(item - target)), query_no)
            repeat_key = (key[0], key[1], key[2], repeat)
            if repeat_key in observed_repeats:
                raise FinalizationFailure(f"{method} raw CSV has duplicate query/repeat cell: {repeat_key}")
            observed_repeats.add(repeat_key)
            grouped.setdefault(key, []).append((recall, latency))
            # Repeat is encoded by list order; duplicate query rows are allowed only as repeats.
            _ = repeat
    result: dict[tuple[str, float], Cell] = {}
    for filter_name in filters:
        for target in targets:
            query_values: dict[int, list[tuple[float, float]]] = {
                query_no: values for (name, item, query_no), values in grouped.items()
                if name == filter_name and math.isclose(item, target)
            }
            if set(query_values) != set(QUERY_NOS):
                raise FinalizationFailure(f"{method} raw CSV missing held-out query cells: {filter_name}|{target:g}")
            query_recalls = [statistics.fmean(item[0] for item in query_values[query_no]) for query_no in QUERY_NOS]
            query_latencies = [statistics.fmean(item[1] for item in query_values[query_no]) for query_no in QUERY_NOS]
            samples = sum(len(item) for item in query_values.values())
            recall_mean = statistics.fmean(query_recalls)
            cell_status = "valid" if recall_mean >= target else "unavailable"
            result[(filter_name, target)] = Cell(
                method, filter_name, target, cell_status, recall_mean,
                statistics.fmean(query_latencies), _percentile(query_latencies, .50),
                _percentile(query_latencies, .95), _percentile(query_latencies, .99),
                len(QUERY_NOS), samples, "" if cell_status == "valid" else "query_level_mean_below_target",
            )
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def _is_summary(fields: Sequence[str], kind: str) -> bool:
    return kind in {"summary", "final_summary"} or "recall_mean" in fields and "query_no" not in fields


def _select_rows(rows: Sequence[dict[str, str]], entry: Mapping[str, Any], method: str) -> list[dict[str, str]]:
    selector = entry.get("row_selector", entry.get("selector"))
    if selector is None:
        selector = {key: entry[key] for key in ("mode", "workload") if entry.get(key) is not None}
    if not selector:
        return list(rows)
    if not isinstance(selector, Mapping) or not selector:
        raise FinalizationFailure(f"{method} row_selector is invalid")
    selected = [
        row for row in rows
        if all(str(row.get(str(key), "")) == str(value) for key, value in selector.items())
    ]
    if not selected:
        raise FinalizationFailure(f"{method} row_selector selected no rows")
    return selected


def _validate_metadata(manifest: Mapping[str, Any], expected_filters: Sequence[str], expected_targets: Sequence[float], expected_gt_hash: str, expected_scope: str) -> None:
    if _dataset(manifest) != DATASET:
        raise FinalizationFailure("dataset mismatch")
    if _filter_names(manifest) != tuple(expected_filters):
        raise FinalizationFailure("filter set/order mismatch")
    if _targets(manifest) != tuple(expected_targets):
        raise FinalizationFailure("target set mismatch")
    query_nos = _query_nos(manifest)
    if query_nos != QUERY_NOS:
        raise FinalizationFailure("held-out cohort is not exactly q100..199")
    if _gt_hash(manifest) != expected_gt_hash:
        raise FinalizationFailure("held-out GT/cohort hash mismatch")
    _candidate_universe(manifest)
    manifest_scope = _latency_scope(manifest)
    if manifest_scope != expected_scope:
        raise FinalizationFailure("latency scope mismatch across methods")


SUMMARY_FIELDS = (
    "dataset", "filter_name", "target_recall", "method", "cell_status", "available",
    "recall_mean", "target_met", "queries", "samples", "latency_scope",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "unavailable_reason", "artifact_manifest_sha256", "csv_sha256", "provenance_sha256",
)


def _csv_rows(cells: Sequence[Cell], scope: str, artifact_info: Mapping[str, tuple[str, str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cell in cells:
        manifest_sha, csv_sha, provenance_sha = artifact_info[cell.method]
        available = cell.status == "valid"
        output.append({
            "dataset": DATASET,
            "filter_name": cell.filter_name,
            "target_recall": cell.target_recall,
            "method": cell.method,
            "cell_status": cell.status,
            "available": available,
            "recall_mean": cell.recall_mean,
            "target_met": available and float(cell.recall_mean) >= cell.target_recall,
            "queries": cell.queries,
            "samples": cell.samples,
            "latency_scope": scope,
            "latency_mean_ms": cell.latency_mean_ms,
            "latency_p50_ms": cell.latency_p50_ms,
            "latency_p95_ms": cell.latency_p95_ms,
            "latency_p99_ms": cell.latency_p99_ms,
            "unavailable_reason": cell.unavailable_reason,
            "artifact_manifest_sha256": manifest_sha,
            "csv_sha256": csv_sha,
            "provenance_sha256": provenance_sha,
        })
    return output


def _atomic_publish(summary_path: Path, manifest_path: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_paths: list[tuple[Path, Path]] = []
    try:
        for destination, kind, value in (
            (summary_path, "csv", rows), (manifest_path, "json", manifest)
        ):
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            temporary = Path(name)
            temp_paths.append((temporary, destination))
            with os.fdopen(fd, "w", encoding="utf-8", newline="" if kind == "csv" else None) as target:
                if kind == "csv":
                    writer = csv.DictWriter(target, fieldnames=list(SUMMARY_FIELDS), extrasaction="raise")
                    writer.writeheader()
                    writer.writerows(rows)
                else:
                    target.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                target.flush()
                os.fsync(target.fileno())
        for temporary, destination in temp_paths:
            os.replace(temporary, destination)
        directory_fd = os.open(summary_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for temporary, _ in temp_paths:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def finalize_artifacts(input_manifest_path: Path, out_prefix: Path) -> dict[str, Path]:
    """Validate and publish one five-method comparison.

    The input manifest is an index with ``artifacts`` entries.  Each entry has
    ``method``, ``manifest``, ``csv`` and their SHA values.  ``kind`` may be
    ``raw``/``final_raw`` or ``summary``.  Source manifests remain the source
    of dataset, cohort, latency, and runtime provenance assertions.
    """
    input_manifest_path = input_manifest_path.resolve()
    input_manifest = _read_json(input_manifest_path)
    if str(input_manifest.get("artifact", "")) not in {
        "amazon10m_sqlens_matched_recall_comparison_inputs",
        "amazon10m_five_way_matched_recall_inputs",
    }:
        raise FinalizationFailure("input manifest has an incompatible artifact type")
    if str(input_manifest.get("dataset", "")).lower() not in {"amazon10m", "amazon-10m"}:
        raise FinalizationFailure("input manifest is not for Amazon-10M")
    filters = _names(input_manifest.get("filter_names", input_manifest.get("filters")), "input filter_names")
    if len(filters) != 14:
        raise FinalizationFailure(f"formal comparison requires 14 filters, got {len(filters)}")
    targets = _number_list(input_manifest.get("target_recalls", input_manifest.get("targets")), "input target_recalls")
    if len(targets) != 3:
        raise FinalizationFailure(f"formal comparison requires 3 targets, got {len(targets)}")
    cohort = input_manifest.get("query_cohort")
    if not isinstance(cohort, Mapping):
        raise FinalizationFailure("input manifest lacks query_cohort")
    if tuple(sorted({_int(item, "query_cohort.query_nos") for item in cohort.get("query_nos", ())})) != QUERY_NOS:
        raise FinalizationFailure("input cohort is not exactly q100..199")
    gt_hash = _sha(cohort.get("gt_hash", cohort.get("query_cohort_sha256")), "input GT/cohort hash")
    scope = str(input_manifest.get("latency_scope", "")).strip().lower().replace("-", "_")
    if scope not in {"search_only", "end_to_end"}:
        raise FinalizationFailure("input manifest latency_scope must be search_only or end_to_end")
    entries = input_manifest.get("artifacts")
    if isinstance(entries, Mapping):
        entries = [dict(value, method=key) if isinstance(value, Mapping) else {"method": key, "manifest": value} for key, value in entries.items()]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise FinalizationFailure("input manifest artifacts is absent or invalid")
    normalized: dict[str, tuple[Path, dict[str, Any], Path, str, str, dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise FinalizationFailure("input artifact entry is not an object")
        method, csv_path, source_manifest, manifest_path, manifest_sha, csv_sha = _manifest_artifact_entry(entry, input_manifest_path.parent)
        if method in normalized:
            raise FinalizationFailure(f"duplicate method artifact: {method}")
        _validate_metadata(source_manifest, filters, targets, gt_hash, scope)
        provenance = _provenance(source_manifest, method)
        normalized[method] = (csv_path, source_manifest, manifest_path, manifest_sha, csv_sha, provenance)
    missing_methods = set(METHODS) - set(normalized)
    if missing_methods:
        raise FinalizationFailure(f"missing method artifacts: {sorted(missing_methods)}")
    if set(normalized) != set(METHODS):
        raise FinalizationFailure("input includes an unexpected method")

    cells: list[Cell] = []
    for method in METHODS:
        csv_path, source_manifest, _, _, _, _ = normalized[method]
        rows, fields = _read_csv(csv_path)
        kind = ""
        entry_for_method: Mapping[str, Any] | None = None
        for entry in entries:
            if isinstance(entry, Mapping) and _canonical_method(entry.get("method", entry.get("name"))) == method:
                kind = str(entry.get("kind", "")).lower()
                entry_for_method = entry
                break
        if entry_for_method is None:
            raise FinalizationFailure(f"missing input entry for {method}")
        rows = _select_rows(rows, entry_for_method, method)
        if _is_summary(fields, kind):
            method_cells: dict[tuple[str, float], Cell] = {}
            for filter_name in filters:
                for target in targets:
                    method_cells[(filter_name, target)] = _summary_cell(rows, method, filter_name, target, scope)
        else:
            method_cells = _raw_cells(rows, method, filters, targets, scope)
        cells.extend(method_cells[(filter_name, target)] for filter_name in filters for target in targets)
    if len(cells) != 14 * 3 * 5 or len({(cell.method, cell.filter_name, cell.target_recall) for cell in cells}) != len(cells):
        raise FinalizationFailure("comparison matrix is not exactly 14*3*5 cells")
    artifact_info = {
        method: (info[3], info[4], info[5]["provenance_sha256"])
        for method, info in normalized.items()
    }
    summary_rows = _csv_rows(cells, scope, artifact_info)
    output_summary = out_prefix.with_name(out_prefix.name + "_summary.csv").resolve()
    output_manifest = out_prefix.with_name(out_prefix.name + "_manifest.json").resolve()
    output_manifest_value: dict[str, Any] = {
        "artifact": "amazon10m_sqlens_matched_recall_comparison_finalized",
        "version": FINALIZER_VERSION,
        "artifact_valid": True,
        "status": "complete",
        "dataset": DATASET,
        "filter_names": list(filters),
        "target_recalls": list(targets),
        "query_cohort": {"query_nos": list(QUERY_NOS), "gt_hash": gt_hash},
        "latency_scope": scope,
        "matrix": {"filters": 14, "targets": 3, "methods": list(METHODS), "cells": len(cells)},
        "cell_counts": {
            "valid": sum(cell.status == "valid" for cell in cells),
            "unavailable": sum(cell.status == "unavailable" for cell in cells),
        },
        "input_manifest": {"path": str(input_manifest_path), "sha256": sha256_file(input_manifest_path)},
        "inputs": {
            method: {
                "manifest": {"path": str(info[2].resolve()), "sha256": info[3]},
                "csv": {"path": str(info[0].resolve()), "sha256": info[4]},
                "provenance_sha256": info[5]["provenance_sha256"],
            }
            for method, info in normalized.items()
        },
        "outputs": {"summary": str(output_summary), "manifest": str(output_manifest)},
        "summary_schema": list(SUMMARY_FIELDS),
    }
    _atomic_publish(output_summary, output_manifest, summary_rows, output_manifest_value)
    return {"summary": output_summary, "manifest": output_manifest}


# Keep the naming used by the existing Amazon finalizers available to callers
# that load finalizers through a small common adapter.
finalize_existing = finalize_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the Amazon-10M five-way matched-recall comparison.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = finalize_artifacts(args.input_manifest, args.out_prefix)
    except (FinalizationFailure, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
