"""Fail-closed merger for Amazon matched-recall result shards.

The merger intentionally treats a shard manifest as an attestation.  It does
not repair, skip, or downgrade an invalid shard.  All validation is completed
before any destination is changed, and the final files are committed as one
best-effort transaction.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACTS = ("raw", "calibration", "final", "summary")
OUTPUT_SUFFIXES = {name: f"_{name}.csv" for name in ARTIFACTS}
DATASET = "amazon10m"
CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
EXPECTED_CANDIDATE_ROWS = 9_979_556
EXPECTED_TARGETS = (0.90, 0.95, 0.99)
EXPECTED_CALIBRATION_QUERY_NOS = tuple(range(20, 100))
EXPECTED_FINAL_QUERY_NOS = tuple(range(100, 200))
SHA256_LENGTH = 64
SHARD_LOCAL_RUN_FIELDS = {
    "filter_names",
    "out_dir",
    "overwrite",
    "progress_queries",
    "tag",
}


class ValidationFailure(ValueError):
    """Raised when a formal merge cannot be proven valid."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


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
        raise ValidationFailure(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationFailure(f"manifest is not an object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            if not fields:
                raise ValidationFailure(f"CSV has no header: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ValidationFailure(f"cannot read CSV {path}: {exc}") from exc
    if any(field is None or field == "" for field in fields):
        raise ValidationFailure(f"CSV has an invalid header: {path}")
    return fields, rows


def _path_value(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path)


def _output_path(manifest_path: Path, manifest: Mapping[str, Any], name: str) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or name not in outputs:
        raise ValidationFailure(f"{manifest_path}: missing outputs.{name}")
    value = outputs[name]
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, (str, os.PathLike)):
        raise ValidationFailure(f"{manifest_path}: outputs.{name} is not a path")
    return _path_value(value, manifest_path.parent)


def _filters_from_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> list[str]:
    value = manifest.get("filter_names")
    if value is None and isinstance(manifest.get("args"), Mapping):
        value = manifest["args"].get("filter_names")
    if value is None:
        value = manifest.get("filters")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValidationFailure(f"{manifest_path}: missing filter_names")
    filters = [str(item) for item in value]
    if not filters or any(not item for item in filters):
        raise ValidationFailure(f"{manifest_path}: filter_names is empty")
    if len(filters) != len(set(filters)):
        raise ValidationFailure(f"{manifest_path}: duplicate declared filter")
    return filters


def _expected_filters(path: Path) -> list[str]:
    _, rows = _read_csv(path)
    names: list[str] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        name = row.get("filter_name", "")
        if not name:
            raise ValidationFailure(f"{path}:{number}: missing filter_name")
        if name in seen:
            raise ValidationFailure(f"{path}:{number}: duplicate expected filter {name}")
        seen.add(name)
        names.append(name)
    if not names:
        raise ValidationFailure(f"expected filter CSV is empty: {path}")
    return names


def _nested(manifest: Mapping[str, Any], *path: str) -> Any:
    value: Any = manifest
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first(manifest: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _nested(manifest, *path)
        if value is not None:
            return value
    return None


def _required(label: str, value: Any, path: Path) -> Any:
    if value is None or value == "":
        raise ValidationFailure(f"{path}: missing provenance field {label}")
    return value


def _hash_values(manifest: Mapping[str, Any], *names: str) -> list[Any]:
    values: list[Any] = []
    for root_name in ("source_hashes", "input_hashes", "hashes"):
        root = manifest.get(root_name)
        if isinstance(root, Mapping):
            for name in names:
                if name in root:
                    value = root[name]
                    if isinstance(value, Mapping):
                        value = value.get("sha256")
                    if value:
                        values.append(value)
    for name in names:
        for root_name in ("faiss", "faiss_index", "fbin", "truth", "ground_truth"):
            root = manifest.get(root_name)
            if root_name == name and isinstance(root, Mapping) and root.get("sha256"):
                values.append(root["sha256"])
        for root_name in ("inputs", "input_artifacts"):
            root = manifest.get(root_name)
            if isinstance(root, Mapping) and name in root:
                value = root[name]
                if isinstance(value, Mapping):
                    value = value.get("sha256")
                if value:
                    values.append(value)
    return values


def _args_value(manifest: Mapping[str, Any], *names: str) -> Any:
    args = manifest.get("args")
    if not isinstance(args, Mapping):
        return None
    for name in names:
        if name in args:
            return args[name]
    return None


def _list_contract_value(value: Any, label: str, path: Path) -> list[Any]:
    if isinstance(value, str):
        try:
            values = [item.strip() for item in value.split(",") if item.strip()]
        except AttributeError as exc:
            raise ValidationFailure(f"{path}: invalid {label}") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        raise ValidationFailure(f"{path}: invalid {label}")
    if not values:
        raise ValidationFailure(f"{path}: empty {label}")
    return values


def _normal_targets(value: Any, path: Path) -> list[float]:
    result = sorted({float(item) for item in _list_contract_value(value, "target recalls", path)})
    if any(item <= 0 or item > 1 for item in result):
        raise ValidationFailure(f"{path}: target recalls outside (0, 1]")
    return result


def _normal_query_nos(value: Any, label: str, path: Path) -> list[int]:
    try:
        result = [int(item) for item in _list_contract_value(value, label, path)]
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"{path}: invalid {label}") from exc
    if result != sorted(set(result)):
        raise ValidationFailure(f"{path}: {label} must be sorted and unique")
    return result


def _sha256(value: Any, label: str, path: Path) -> str:
    digest = str(_required(label, value, path)).lower()
    if len(digest) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in digest):
        raise ValidationFailure(f"{path}: {label} is not a SHA256 digest")
    return digest


def _consistent_hash(
    manifest_path: Path, manifest: Mapping[str, Any], label: str, *names: str
) -> str:
    values = [_sha256(value, label, manifest_path) for value in _hash_values(manifest, *names)]
    if not values:
        raise ValidationFailure(f"{manifest_path}: missing provenance field {label}")
    if len(set(values)) != 1:
        raise ValidationFailure(f"{manifest_path}: conflicting {label} attestations")
    return values[0]


def _candidate_universe(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    predicate_values = [
        value
        for value in (
            _first(manifest, ("candidate_validity_predicate",)),
            _first(manifest, ("candidate_universe", "predicate")),
            _first(manifest, ("run_contract", "candidate_universe", "predicate")),
            _first(manifest, ("args", "candidate_validity_predicate")),
            _first(manifest, ("postgres", "candidate_universe", "predicate")),
            _first(manifest, ("service", "candidate_universe", "predicate")),
        )
        if value is not None
    ]
    if not predicate_values:
        raise ValidationFailure(f"{manifest_path}: missing candidate universe predicate")
    normalized = {" ".join(str(value).strip().split()) for value in predicate_values}
    if normalized != {CANDIDATE_VALIDITY_PREDICATE}:
        raise ValidationFailure(
            f"{manifest_path}: candidate universe must be {CANDIDATE_VALIDITY_PREDICATE!r}"
        )

    expected_values = [
        value
        for value in (
            _first(manifest, ("candidate_universe", "expected_rows")),
            _first(manifest, ("run_contract", "candidate_universe", "expected_rows")),
            _first(manifest, ("config", "candidate_universe", "expected_rows")),
        )
        if value is not None
    ]
    observed_values = [
        value
        for value in (
            _first(manifest, ("candidate_universe", "observed_rows")),
            _first(manifest, ("config", "candidate_universe", "observed_rows")),
            _first(manifest, ("postgres", "candidate_universe", "rows")),
            _first(manifest, ("service", "candidate_universe", "count")),
        )
        if value is not None
    ]
    for label, values in (("expected", expected_values), ("observed", observed_values)):
        for value in values:
            try:
                rows = int(value)
            except (TypeError, ValueError) as exc:
                raise ValidationFailure(
                    f"{manifest_path}: candidate universe {label} rows is not an integer"
                ) from exc
            if rows != EXPECTED_CANDIDATE_ROWS:
                raise ValidationFailure(
                    f"{manifest_path}: candidate universe {label} rows must be "
                    f"{EXPECTED_CANDIDATE_ROWS}, got {rows}"
                )
    return {
        "predicate": CANDIDATE_VALIDITY_PREDICATE,
        "expected_rows": EXPECTED_CANDIDATE_ROWS,
        "observed_rows": EXPECTED_CANDIDATE_ROWS if observed_values else None,
    }


def _query_splits(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    split_values = [
        value
        for value in (
            _first(manifest, ("query_splits",)),
            _first(manifest, ("run_contract", "query_splits")),
        )
        if value is not None
    ]
    if not split_values or any(not isinstance(value, Mapping) for value in split_values):
        raise ValidationFailure(f"{manifest_path}: missing query_splits")
    normalized_values: list[dict[str, Any]] = []
    for splits in split_values:
        calibration = _normal_query_nos(
            _first(splits, ("calibration_query_nos",), ("calibration", "query_nos")),
            "calibration query cohort",
            manifest_path,
        )
        final = _normal_query_nos(
            _first(splits, ("final_query_nos",), ("final", "query_nos")),
            "final query cohort",
            manifest_path,
        )
        if calibration != list(EXPECTED_CALIBRATION_QUERY_NOS):
            raise ValidationFailure(f"{manifest_path}: calibration cohort is not exactly q20..99")
        if final != list(EXPECTED_FINAL_QUERY_NOS):
            raise ValidationFailure(f"{manifest_path}: final cohort is not exactly q100..199")
        if set(calibration) & set(final):
            raise ValidationFailure(f"{manifest_path}: calibration and final cohorts overlap")
        normalized_values.append({
            "reserved_query_nos": list(range(20)),
            "calibration_query_nos": calibration,
            "final_query_nos": final,
            "query_no_overlap": False,
        })
    if any(_json(value) != _json(normalized_values[0]) for value in normalized_values[1:]):
        raise ValidationFailure(f"{manifest_path}: conflicting query split attestations")
    return normalized_values[0]


def _latency_scope(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    values = [
        value
        for value in (
            _first(manifest, ("latency_scope",)),
            _first(manifest, ("latency_definition",)),
            _first(manifest, ("execution", "latency")),
            _first(manifest, ("run_contract", "latency_scope")),
        )
        if value is not None
    ]
    if not values:
        raise ValidationFailure(f"{manifest_path}: missing provenance field latency scope")
    scopes: list[str] = []
    for value in values:
        text = str(value).strip().lower().replace("-", "_")
        if "search_only" in text or "search only" in text:
            scopes.append("search_only")
        elif "end_to_end" in text or "end to end" in text or "e2e" in text:
            scopes.append("end_to_end")
        else:
            raise ValidationFailure(f"{manifest_path}: unsupported latency scope {value!r}")
    if len(set(scopes)) != 1:
        raise ValidationFailure(f"{manifest_path}: conflicting latency scope attestations")
    return {"scope": scopes[0], "definition": str(values[0])}


def _normal_ef(value: Any, path: Path) -> list[int]:
    result = sorted({int(item) for item in _list_contract_value(value, "ef ladder", path)})
    if any(item <= 0 for item in result):
        raise ValidationFailure(f"{path}: ef ladder values must be positive")
    return result


def _contract(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    args = manifest.get("args") if isinstance(manifest.get("args"), Mapping) else {}
    run_contract = manifest.get("run_contract", manifest.get("contract"))
    if run_contract is None:
        # These are the runner's shard-independent parameters.  Output paths,
        # tag, and filter selection are deliberately excluded.
        run_contract = args
    if isinstance(run_contract, Mapping):
        run_contract = {
            key: value
            for key, value in run_contract.items()
            if key not in SHARD_LOCAL_RUN_FIELDS
        }
    _required("run_contract", run_contract, manifest_path)

    source_db = _first(
        manifest,
        ("postgres",),
        ("source_db",),
        ("provenance", "source_db"),
        ("inputs", "source_db"),
    )
    if source_db is None:
        source_db = {
            "table": _first(manifest, ("inputs", "postgres_table"), ("args", "table")),
            "postgres": manifest.get("postgres"),
        }
    _required("source_db", source_db, manifest_path)
    if not isinstance(source_db, Mapping) or not source_db.get("server_version"):
        raise ValidationFailure(
            f"{manifest_path}: PostgreSQL binary provenance lacks server_version"
        )

    faiss_hash = _consistent_hash(
        manifest_path, manifest, "faiss sha256", "faiss", "faiss_index"
    )
    fbin_hash = _consistent_hash(
        manifest_path, manifest, "fbin sha256", "fbin", "vectors"
    )
    truth_hash = _consistent_hash(
        manifest_path, manifest, "truth sha256", "truth", "ground_truth"
    )
    runner_hash = _consistent_hash(
        manifest_path,
        manifest,
        "measurement runner sha256",
        "runner",
        "measurement_runner",
    )

    splits = _query_splits(manifest_path, manifest)
    calibration_repeats = _first(
        manifest,
        ("repeats", "calibration"),
        ("query_splits", "calibration_repeats"),
        ("args", "calibration_repeats"),
    )
    final_repeats = _first(
        manifest,
        ("repeats", "final"),
        ("query_splits", "final_repeats"),
        ("args", "final_repeats"),
    )
    _required("calibration repeats", calibration_repeats, manifest_path)
    _required("final repeats", final_repeats, manifest_path)
    target_values = [
        value
        for value in (
            _first(manifest, ("target_recalls",)),
            _first(manifest, ("run_contract", "target_recalls")),
            _first(manifest, ("args", "target_recalls")),
        )
        if value is not None
    ]
    ef_ladder = _first(
        manifest,
        ("ef_ladder",),
        ("run_contract", "ef_ladder"),
        ("args", "ef_search_values"),
    )
    if not target_values:
        raise ValidationFailure(f"{manifest_path}: missing provenance field target_recalls")
    _required("ef ladder", ef_ladder, manifest_path)
    environment = _first(manifest, ("software_versions",), ("environment",))
    _required("software versions", environment, manifest_path)
    target_evidence = [_normal_targets(value, manifest_path) for value in target_values]
    if any(value != list(EXPECTED_TARGETS) for value in target_evidence):
        raise ValidationFailure(
            f"{manifest_path}: target recalls must be exactly {list(EXPECTED_TARGETS)}"
        )
    normalized_targets = target_evidence[0]
    candidate_universe = _candidate_universe(manifest_path, manifest)
    latency = _latency_scope(manifest_path, manifest)
    faiss_index = _first(manifest, ("faiss_index",), ("inputs", "faiss_index"))
    _required("faiss index provenance", faiss_index, manifest_path)

    return {
        "dataset": DATASET,
        "run_contract": run_contract,
        "source_db": source_db,
        "binary_provenance": {
            "source_db": source_db,
            "faiss_index": faiss_index,
            "faiss_sha256": faiss_hash,
        },
        "runner_provenance": {
            "measurement_runner_sha256": runner_hash,
            "software_versions": environment,
        },
        "source_hashes": {
            "runner": runner_hash,
            "faiss": faiss_hash,
            "fbin": fbin_hash,
            "truth": truth_hash,
        },
        "candidate_universe": candidate_universe,
        "latency": latency,
        "faiss_sha256": faiss_hash,
        "fbin_sha256": fbin_hash,
        "truth_sha256": truth_hash,
        "query_splits": splits,
        "calibration_repeats": int(calibration_repeats),
        "final_repeats": int(final_repeats),
        "target_recalls": normalized_targets,
        "ef_ladder": _normal_ef(ef_ladder, manifest_path),
        "software_versions": environment,
    }


def _key_fields(kind: str, row: Mapping[str, str]) -> tuple[str, ...]:
    if kind == "raw":
        if row.get("phase") == "setup":
            return ("phase", "filter_name", "method")
        else:
            return ("phase", "filter_name", "method", "query_no", "query_id", "repeat", "ef_search")
    elif kind == "calibration":
        return ("filter_name", "method", "target_recall", "ef_search")
    elif kind == "final":
        return ("phase", "filter_name", "method", "query_no", "query_id", "repeat", "ef_search")
    else:
        return ("filter_name", "method", "target_recall", "selected_faiss_ef_search")


def _row_key(kind: str, row: Mapping[str, str]) -> tuple[str, ...]:
    fields = _key_fields(kind, row)
    values = tuple(row.get(field, "") for field in fields)
    return values


def _sort_value(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        try:
            return (1, float(value))
        except (TypeError, ValueError):
            return (2, str(value))


def _row_sort_key(kind: str, row: Mapping[str, str]) -> tuple[Any, ...]:
    priority = {
        "raw": ("filter_name", "phase", "method", "query_no", "repeat", "ef_search", "pair_key"),
        "calibration": ("filter_name", "target_recall", "ef_search", "method"),
        "final": ("filter_name", "phase", "method", "query_no", "repeat", "ef_search"),
        "summary": ("filter_name", "target_recall", "method", "selected_faiss_ef_search"),
    }[kind]
    return tuple(_sort_value(row.get(field, "")) for field in priority) + (_json(dict(row)),)


def _declared_row_count(manifest: Mapping[str, Any], name: str) -> int | None:
    for root_name in ("row_counts", "csv_row_counts", "output_row_counts"):
        root = manifest.get(root_name)
        if isinstance(root, Mapping) and name in root:
            try:
                return int(root[name])
            except (TypeError, ValueError):
                return -1
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping) and isinstance(outputs.get(name), Mapping):
        value = outputs[name].get("rows")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1
    return None


def _validate_shard(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_contract: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], dict[str, tuple[list[str], list[dict[str, str]]]]]:
    errors: list[str] = []
    if manifest.get("artifact_valid") is not True:
        errors.append("artifact_valid is not true")
    if manifest.get("status") != "complete":
        errors.append(f"status is {manifest.get('status')!r}, expected 'complete'")
    filters = _filters_from_manifest(manifest_path, manifest)
    try:
        contract = _contract(manifest_path, manifest)
    except ValidationFailure as exc:
        errors.append(str(exc))
        contract = {}
    if expected_contract is not None and contract and _json(contract) != _json(expected_contract):
        errors.append("provenance/run contract mismatch")

    tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for kind in ARTIFACTS:
        try:
            artifact_path = _output_path(manifest_path, manifest, kind)
            if not artifact_path.is_file():
                raise ValidationFailure(f"missing artifact: {artifact_path}")
            fields, rows = _read_csv(artifact_path)
            tables[kind] = (fields, rows)
            declared_count = _declared_row_count(manifest, kind)
            if declared_count is not None and declared_count != len(rows):
                errors.append(
                    f"{manifest_path}: {kind} row count {len(rows)} != declared {declared_count}"
                )
            if not rows:
                errors.append(f"{manifest_path}: {kind} CSV is empty")
            keys: set[tuple[str, ...]] = set()
            for number, row in enumerate(rows, start=2):
                filter_name = row.get("filter_name", "")
                if filter_name not in filters:
                    errors.append(f"{manifest_path}:{kind}:{number}: foreign filter {filter_name!r}")
                key_fields = _key_fields(kind, row)
                missing_key_fields = [field for field in key_fields if not row.get(field, "")]
                if missing_key_fields:
                    errors.append(
                        f"{manifest_path}:{kind}:{number}: missing key field(s) {missing_key_fields}"
                    )
                key = _row_key(kind, row)
                if key in keys:
                    errors.append(f"{manifest_path}:{kind}:{number}: duplicate key {key!r}")
                keys.add(key)
        except ValidationFailure as exc:
            errors.append(str(exc))
    return contract, errors, tables


def resolve_manifests(values: Iterable[str | os.PathLike[str]]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        text = os.fspath(value)
        matches = glob.glob(text, recursive=True)
        if not matches and Path(text).is_file():
            matches = [text]
        paths.update(Path(match).resolve() for match in matches if Path(match).is_file())
    result = sorted(paths)
    if not result:
        raise ValidationFailure("--input-manifests matched no files")
    return result


def _commit_staged(staged: Mapping[str, Path], destinations: Mapping[str, Path]) -> None:
    """Replace a set of files and restore old files if a replacement fails."""
    parent = next(iter(destinations.values())).parent
    backups: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for name, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = parent / f".{destination.name}.backup"
                if backup.exists():
                    backup.unlink()
                shutil.copy2(destination, backup)
                backups[name] = backup
        for name, destination in destinations.items():
            os.replace(staged[name], destination)
            replaced.append(name)
    except Exception:
        for name in reversed(replaced):
            destination = destinations[name]
            backup = backups.get(name)
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                elif destination.exists():
                    destination.unlink()
            except OSError:
                pass
        raise
    finally:
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass


def combine(input_manifests: Sequence[Path], expected_filters_csv: Path, out_prefix: Path) -> dict[str, Path]:
    expected = _expected_filters(expected_filters_csv)
    expected_set = set(expected)
    all_filters: list[str] = []
    seen_filters: set[str] = set()
    loaded: list[tuple[Path, dict[str, Any], dict[str, tuple[list[str], list[dict[str, str]]]], str, list[str]]] = []
    errors: list[str] = []
    reference_contract: dict[str, Any] | None = None

    for manifest_path in sorted(Path(path).resolve() for path in input_manifests):
        manifest = _read_json(manifest_path)
        filters = _filters_from_manifest(manifest_path, manifest)
        contract, shard_errors, tables = _validate_shard(manifest_path, manifest, reference_contract)
        if reference_contract is None and contract:
            reference_contract = contract
        if set(filters) & seen_filters:
            errors.append(f"duplicate shard filter(s): {sorted(set(filters) & seen_filters)}")
        seen_filters.update(filters)
        all_filters.extend(filters)
        errors.extend(shard_errors)
        loaded.append((manifest_path, manifest, tables, sha256_file(manifest_path), filters))

    if set(all_filters) != expected_set:
        errors.append(
            f"filter coverage mismatch: missing={sorted(expected_set - set(all_filters))} "
            f"extra={sorted(set(all_filters) - expected_set)}"
        )
    if len(all_filters) != len(expected) or len(set(all_filters)) != len(all_filters):
        errors.append("filter coverage is not exactly once")
    if errors:
        raise ValidationFailure("merge refused:\n" + "\n".join(f"- {error}" for error in errors))

    merged: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    input_artifacts: list[dict[str, Any]] = []
    for kind in ARTIFACTS:
        schemas = [tables[kind][0] for _, _, tables, _, _ in loaded]
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValidationFailure(f"{kind} CSV headers differ between shards")
        rows = [row for _, _, tables, _, _ in loaded for row in tables[kind][1]]
        rows.sort(key=lambda row: _row_sort_key(kind, row))
        merged[kind] = (schemas[0], rows)
    for manifest_path, manifest, tables, manifest_sha, filters in loaded:
        input_artifacts.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "filters": filters,
                "artifacts": {
                    kind: {
                        "path": str(_output_path(manifest_path, manifest, kind)),
                        "sha256": sha256_file(_output_path(manifest_path, manifest, kind)),
                        "rows": len(tables[kind][1]),
                    }
                    for kind in ARTIFACTS
                },
            }
        )

    destinations = {kind: out_prefix.with_name(out_prefix.name + suffix) for kind, suffix in OUTPUT_SUFFIXES.items()}
    manifest_destination = out_prefix.with_name(out_prefix.name + "_manifest.json")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{out_prefix.name}.", dir=str(out_prefix.parent)))
    try:
        staged: dict[str, Path] = {}
        for kind, (fields, rows) in merged.items():
            path = stage_dir / f"{kind}.csv"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields, extrasaction="raise")
                writer.writeheader()
                writer.writerows(rows)
            staged[kind] = path

        output_hashes = {kind: sha256_file(path) for kind, path in staged.items()}
        combined_manifest: dict[str, Any] = {
            "artifact": "amazon10m_matched_recall_baselines_combined",
            "artifact_valid": True,
            "status": "complete",
            "validation_errors": [],
            "dataset": DATASET,
            "filters": expected,
            "filter_names": expected,
            "target_recalls": reference_contract["target_recalls"],
            "query_splits": reference_contract["query_splits"],
            "query_cohort": {
                "calibration_query_nos": reference_contract["query_splits"]["calibration_query_nos"],
                "query_nos": reference_contract["query_splits"]["final_query_nos"],
                "gt_hash": reference_contract["truth_sha256"],
            },
            "candidate_validity_predicate": CANDIDATE_VALIDITY_PREDICATE,
            "candidate_universe": reference_contract["candidate_universe"],
            "latency_scope": reference_contract["latency"]["scope"],
            "latency_definition": reference_contract["latency"]["definition"],
            "source_hashes": reference_contract["source_hashes"],
            "measurement_runner_sha256": reference_contract["source_hashes"]["runner"],
            "software_versions": reference_contract["software_versions"],
            "source_db": reference_contract["source_db"],
            "binary_provenance": reference_contract["binary_provenance"],
            "runner_provenance": reference_contract["runner_provenance"],
            "input_manifests": [
                {
                    "path": item["manifest"],
                    "sha256": item["manifest_sha256"],
                    "filters": item["filters"],
                }
                for item in input_artifacts
            ],
            "input_sha256": {
                item["manifest"]: item["manifest_sha256"] for item in input_artifacts
            },
            "input_artifacts": input_artifacts,
            "contract": reference_contract,
            "row_counts": {kind: len(merged[kind][1]) for kind in ARTIFACTS},
            "outputs": {
                **{kind: str(path) for kind, path in destinations.items()},
                "manifest": str(manifest_destination),
            },
            "output_sha256": output_hashes,
            "merged_file_sha256": output_hashes,
        }
        manifest_stage = stage_dir / "manifest.json"
        manifest_stage.write_text(json.dumps(combined_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged["manifest"] = manifest_stage
        _commit_staged(
            staged,
            {**destinations, "manifest": manifest_destination},
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return {**destinations, "manifest": manifest_destination}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed merger for matched-recall result shards.")
    parser.add_argument("--input-manifests", nargs="+", action="append", required=True)
    parser.add_argument("--expected-filters-csv", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_values = [item for group in args.input_manifests for item in group]
    try:
        paths = resolve_manifests(manifest_values)
        outputs = combine(paths, args.expected_filters_csv, args.out_prefix)
    except (OSError, ValidationFailure, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
