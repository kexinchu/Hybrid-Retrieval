"""Fail-closed finalizer for completed Amazon-10M matched-recall measurements.

This program never opens PostgreSQL or Faiss.  It derives a new formal summary
from immutable measurement CSVs and writes only a new summary and manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .amazon10m_matched_recall_baselines import (
        FINALIZER_VERSION,
        NA,
        artifact_validation_errors,
        calibration_outcomes_from_rows,
        calibration_table,
        file_identity,
        final_summary_table,
        load_filter_specs,
        load_truth,
        parse_int_csv,
        parse_targets,
        sha256_file,
        _atomic_write_outputs,
    )
except ImportError:  # Direct script execution puts this directory on sys.path.
    from amazon10m_matched_recall_baselines import (  # type: ignore[no-redef]
        FINALIZER_VERSION,
        NA,
        artifact_validation_errors,
        calibration_outcomes_from_rows,
        calibration_table,
        file_identity,
        final_summary_table,
        load_filter_specs,
        load_truth,
        parse_int_csv,
        parse_targets,
        sha256_file,
        _atomic_write_outputs,
    )


class FinalizationFailure(ValueError):
    pass


SHARD_LOCAL_RUN_FIELDS = {
    "filter_names",
    "out_dir",
    "overwrite",
    "progress_queries",
    "tag",
}


def _global_run_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return only settings that must agree across independently run shards."""
    value = manifest.get("run_contract", manifest.get("args"))
    if not isinstance(value, Mapping):
        raise FinalizationFailure("manifest missing run_contract/args")
    return {
        str(key): item
        for key, item in value.items()
        if key not in SHARD_LOCAL_RUN_FIELDS
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationFailure(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationFailure("legacy manifest is not an object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise FinalizationFailure(f"CSV has no header: {path}")
            return list(reader)
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
    return path if path.is_absolute() else (base / path)


def _manifest_output_path(manifest: Mapping[str, Any], base: Path, name: str) -> Path | None:
    value = _nested(manifest, "outputs", name)
    return None if value is None else _path(value, base, f"outputs.{name}")


def _assert_manifest_path(manifest: Mapping[str, Any], base: Path, name: str, supplied: Path) -> None:
    declared = _manifest_output_path(manifest, base, name)
    if declared is not None and declared.resolve() != supplied.resolve():
        raise FinalizationFailure(f"legacy {name} path does not match manifest outputs.{name}")


def _assert_declared_artifact_hash(manifest: Mapping[str, Any], name: str, path: Path) -> None:
    declared: Any = _nested(manifest, "output_sha256", name)
    if declared is None:
        declared = _nested(manifest, "merged_file_sha256", name)
    output = _nested(manifest, "outputs", name)
    if declared is None and isinstance(output, Mapping):
        declared = output.get("sha256")
    if declared is not None and str(declared) != sha256_file(path):
        raise FinalizationFailure(f"legacy {name} CSV hash changed from manifest provenance")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _metric_error(row: Mapping[str, str]) -> bool:
    if row.get("phase") == "setup":
        return not _bool(row.get("valid", "")) or bool(row.get("error"))
    if not _bool(row.get("valid", "")) or bool(row.get("error")):
        return True
    try:
        latency = float(row["search_latency_ms"])
        recall = float(row["recall_at_10"])
    except (KeyError, TypeError, ValueError):
        return True
    return latency <= 0 or recall < 0 or recall > 1


def _assert_no_measurement_errors(rows: Sequence[Mapping[str, str]], label: str) -> None:
    errors = [number for number, row in enumerate(rows, start=2) if _metric_error(row)]
    if errors:
        raise FinalizationFailure(f"{label} contains measured error/invalid metrics at row(s) {errors[:5]}")


def _args(manifest: Mapping[str, Any], name: str) -> Any:
    value = _nested(manifest, "args", name)
    if value is None:
        value = manifest.get(name)
    if value is None:
        raise FinalizationFailure(f"manifest missing args.{name}")
    return value


def _declared_hash(manifest: Mapping[str, Any], name: str) -> str | None:
    for root in ("source_hashes", "input_hashes", "hashes"):
        value = manifest.get(root)
        if isinstance(value, Mapping) and name in value:
            item = value[name]
            return str(item.get("sha256")) if isinstance(item, Mapping) else str(item)
    aliases = {"faiss": "faiss_index", "truth": "truth", "fbin": "fbin"}
    item = _nested(manifest, "inputs", aliases[name])
    return str(item.get("sha256")) if isinstance(item, Mapping) and item.get("sha256") else None


def _source_path(manifest: Mapping[str, Any], base: Path, name: str) -> Path:
    aliases = {"faiss": "faiss_index", "truth": "truth", "fbin": "fbin", "filters": "filters"}
    value = _nested(manifest, "inputs", aliases[name])
    if value is None:
        raise FinalizationFailure(f"manifest missing inputs.{aliases[name]}")
    path = _path(value, base, f"inputs.{aliases[name]}")
    if not path.is_file():
        raise FinalizationFailure(f"source file is absent: {path}")
    return path


def _validate_provenance(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    if manifest.get("status") == "running" or not manifest.get("finished_at_utc"):
        raise FinalizationFailure("legacy process is incomplete")
    if manifest.get("fatal_error"):
        raise FinalizationFailure("legacy process has a fatal error")
    runner = _nested(manifest, "inputs", "runner")
    if not isinstance(runner, Mapping) or not runner.get("sha256"):
        raise FinalizationFailure("missing measurement runner SHA")
    runner_sha = str(runner["sha256"])
    if len(runner_sha) != 64 or any(char not in "0123456789abcdef" for char in runner_sha.lower()):
        raise FinalizationFailure("measurement runner SHA is not a SHA-256 value")
    if not manifest.get("postgres") and not manifest.get("source_db"):
        raise FinalizationFailure("missing source database provenance")
    result: dict[str, Any] = {}
    for name in ("faiss", "fbin", "truth"):
        path = _source_path(manifest, manifest_path.parent, name)
        actual = sha256_file(path)
        declared = _declared_hash(manifest, name)
        if declared and declared != actual:
            raise FinalizationFailure(f"{name} hash changed from manifest provenance")
        result[name] = {**file_identity(path), "sha256": actual}
    result["measurement_runner_sha256"] = runner_sha
    return result


def _selected_final_efs(
    final_rows: Sequence[Mapping[str, str]], filter_name: str,
) -> set[int]:
    values: set[int] = set()
    for row in final_rows:
        if row.get("phase") == "final" and row.get("method") == "faiss_allowlist" and row.get("filter_name") == filter_name:
            try:
                values.add(int(row["ef_search"]))
            except (KeyError, ValueError) as exc:
                raise FinalizationFailure(f"invalid Faiss final ef_search for {filter_name}") from exc
    return values


def _measurement_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return (
        row.get("phase", ""), row.get("filter_name", ""), row.get("method", ""),
        row.get("query_no", ""), row.get("query_id", ""), row.get("repeat", ""),
        row.get("ef_search", ""),
    )


def finalize_existing(
    manifest_path: Path,
    raw_path: Path,
    calibration_path: Path,
    final_path: Path,
    out_prefix: Path,
) -> dict[str, Path]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    for name, path in (("raw", raw_path), ("calibration", calibration_path), ("final", final_path)):
        if not path.is_file():
            raise FinalizationFailure(f"missing legacy {name} CSV: {path}")
        _assert_manifest_path(manifest, manifest_path.parent, name, path)
        _assert_declared_artifact_hash(manifest, name, path)
    sources = _validate_provenance(manifest, manifest_path)
    raw_rows = _read_csv(raw_path)
    old_calibration_rows = _read_csv(calibration_path)
    final_rows = _read_csv(final_path)
    _assert_no_measurement_errors(raw_rows, "raw CSV")
    _assert_no_measurement_errors(final_rows, "final CSV")

    filters_path = _source_path(manifest, manifest_path.parent, "filters")
    filter_names = manifest.get("filter_names") or _args(manifest, "filter_names")
    if isinstance(filter_names, str):
        filter_names = [filter_names]
    if not isinstance(filter_names, Sequence):
        raise FinalizationFailure("manifest filter_names is absent or invalid")
    if not filter_names:
        filter_names = sorted({row.get("filter_name", "") for row in raw_rows if row.get("phase") != "setup"})
    if not filter_names or any(not name for name in filter_names):
        raise FinalizationFailure("manifest filter_names is absent or invalid")
    specs = load_filter_specs(filters_path, set(filter_names) if filter_names else None)
    calibration_offset = int(_args(manifest, "calibration_query_offset"))
    calibration_count = int(_args(manifest, "calibration_queries"))
    final_offset = int(_args(manifest, "final_query_offset"))
    final_count = int(_args(manifest, "final_queries"))
    calibration_queries = list(range(calibration_offset, calibration_offset + calibration_count))
    final_queries = list(range(final_offset, final_offset + final_count))
    calibration_repeats = int(_args(manifest, "calibration_repeats"))
    final_repeats = int(_args(manifest, "final_repeats"))
    targets = parse_targets(str(_args(manifest, "target_recalls")))
    ef_values = parse_int_csv(str(_args(manifest, "ef_search_values")))
    k = int(_args(manifest, "k"))
    declared_splits = manifest.get("query_splits")
    if not isinstance(declared_splits, Mapping) or (
        declared_splits.get("calibration_query_nos") != calibration_queries
        or declared_splits.get("final_query_nos") != final_queries
    ):
        raise FinalizationFailure("manifest query split coverage is absent or inconsistent")
    load_truth(_source_path(manifest, manifest_path.parent, "truth"), specs, calibration_queries, final_queries, k)

    calibration_rows, selected = calibration_table(
        raw_rows, specs, ef_values, targets, calibration_queries, calibration_repeats,
        int(_args(manifest, "bootstrap_samples")), int(_args(manifest, "bootstrap_seed")),
    )
    expected_calibration_keys = {
        (row["filter_name"], str(row["target_recall"]), str(row["ef_search"]))
        for row in calibration_rows
    }
    observed_calibration_keys = {
        (row.get("filter_name", ""), row.get("target_recall", ""), row.get("ef_search", ""))
        for row in old_calibration_rows
    }
    if len(old_calibration_rows) != len(observed_calibration_keys) or observed_calibration_keys != expected_calibration_keys:
        raise FinalizationFailure("legacy calibration key coverage is incomplete, duplicated, or inconsistent")
    expected_by_key = {
        (row["filter_name"], str(row["target_recall"]), str(row["ef_search"])): row
        for row in calibration_rows
    }
    for row in old_calibration_rows:
        key = (row.get("filter_name", ""), row.get("target_recall", ""), row.get("ef_search", ""))
        expected = expected_by_key[key]
        if row.get("status") != "valid":
            raise FinalizationFailure(f"legacy calibration contains invalid cell: {key}")
        for metric in ("recall_mean", "recall_lcb95", "samples", "expected_samples"):
            try:
                if float(row[metric]) != float(expected[metric]):
                    raise FinalizationFailure(f"legacy calibration metric changed: {key} {metric}")
            except KeyError as exc:
                raise FinalizationFailure(f"legacy calibration lacks {metric}: {key}") from exc
    if any(row.get("status") != "valid" for row in calibration_rows):
        raise FinalizationFailure("calibration is incomplete, duplicated, or has invalid metrics")
    outcomes = calibration_outcomes_from_rows(calibration_rows)
    known_filters = {spec.name for spec in specs}
    for row in final_rows:
        if row.get("phase") != "final" or row.get("filter_name") not in known_filters:
            raise FinalizationFailure("final CSV contains an unexpected phase or filter")
        if row.get("method") not in {"sql_first_exact", "faiss_allowlist"}:
            raise FinalizationFailure("final CSV contains an unexpected method")
    for spec in specs:
        expected_efs = {selected[(spec.name, target)] for target in targets if (spec.name, target) in selected}
        observed_efs = _selected_final_efs(final_rows, spec.name)
        if observed_efs != expected_efs:
            raise FinalizationFailure(
                f"final Faiss config coverage for {spec.name} is {sorted(observed_efs)}, "
                f"expected selected {sorted(expected_efs)}"
            )
    final_keys = [_measurement_key(row) for row in final_rows]
    raw_final_keys = [_measurement_key(row) for row in raw_rows if row.get("phase") == "final"]
    if len(final_keys) != len(set(final_keys)) or set(final_keys) != set(raw_final_keys):
        raise FinalizationFailure("raw/final measurement key coverage is incomplete, duplicated, or inconsistent")
    summary_rows = final_summary_table(
        final_rows, specs, targets, selected, final_queries, final_repeats,
        int(_args(manifest, "bootstrap_samples")), int(_args(manifest, "bootstrap_seed")),
        calibration_outcomes=outcomes,
    )
    validation_errors = artifact_validation_errors(
        calibration_rows, summary_rows, specs, ef_values, targets
    )
    if validation_errors:
        raise FinalizationFailure("finalization refused:\n" + "\n".join(validation_errors))

    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv").resolve()
    output_manifest_path = out_prefix.with_name(out_prefix.name + "_manifest.json").resolve()
    output_hashes = {
        "raw": sha256_file(raw_path),
        "calibration": sha256_file(calibration_path),
        "final": sha256_file(final_path),
    }
    finalizer_path = Path(__file__).resolve()
    output_manifest: dict[str, Any] = {
        "artifact": "amazon10m_matched_recall_baselines_finalized",
        "artifact_valid": True,
        "status": "complete",
        "validation_errors": [],
        "filter_names": [spec.name for spec in specs],
        "run_contract": _global_run_contract(manifest),
        "source_db": manifest.get("source_db", manifest.get("postgres")),
        "source_hashes": {name: sources[name]["sha256"] for name in ("faiss", "fbin", "truth")},
        "query_splits": {
            "calibration_query_nos": calibration_queries,
            "final_query_nos": final_queries,
            "query_no_overlap": False,
        },
        "repeats": {"calibration": calibration_repeats, "final": final_repeats},
        "target_recalls": targets,
        "ef_ladder": ef_values,
        "software_versions": {
            "measurement_runner_sha256": sources["measurement_runner_sha256"],
            "finalizer_sha256": sha256_file(finalizer_path),
            "finalizer_version": FINALIZER_VERSION,
        },
        "legacy_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "measured_artifacts": {
            name: {"path": str(path.resolve()), "sha256": output_hashes[name], "rows": len(rows)}
            for name, path, rows in (
                ("raw", raw_path, raw_rows),
                ("calibration", calibration_path, old_calibration_rows),
                ("final", final_path, final_rows),
            )
        },
        "outputs": {
            "raw": {"path": str(raw_path.resolve()), "rows": len(raw_rows)},
            "calibration": {"path": str(calibration_path.resolve()), "rows": len(old_calibration_rows)},
            "final": {"path": str(final_path.resolve()), "rows": len(final_rows)},
            "summary": {"path": str(summary_path), "rows": len(summary_rows)},
            "manifest": str(output_manifest_path),
        },
        "row_counts": {
            "raw": len(raw_rows), "calibration": len(old_calibration_rows),
            "final": len(final_rows), "summary": len(summary_rows),
        },
        "selected_faiss_ef_search": {
            f"{filter_name}|{target:.2f}": ef
            for (filter_name, target), ef in selected.items()
        },
    }
    _atomic_write_outputs({summary_path: ("csv", summary_rows), output_manifest_path: ("json", output_manifest)})
    return {"summary": summary_path, "manifest": output_manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed finalizer for Amazon-10M matched-recall CSVs.")
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-raw", type=Path, required=True)
    parser.add_argument("--legacy-calibration", type=Path, required=True)
    parser.add_argument("--legacy-final", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = finalize_existing(
            args.legacy_manifest, args.legacy_raw, args.legacy_calibration,
            args.legacy_final, args.out_prefix,
        )
    except (FinalizationFailure, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
