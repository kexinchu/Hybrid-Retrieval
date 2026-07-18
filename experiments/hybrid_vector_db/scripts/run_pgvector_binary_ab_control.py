"""Failure-recoverable A/B controller for two pgvector vector.so binaries.

The controller owns the binary transition and delegates all measurements to
pgvector_upstream_overhead_control.py. It deliberately does not import the
runner: a dry run must not read the runner, experiment inputs, Docker, or
PostgreSQL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


OFFICIAL_VECTOR_SO_SHA256 = (
    "812292e3e7553c3dbe6a4187b528430a7f9c25693f4876b8d22f88829592a778"
)
DEFAULT_SQLENS_BUILD_PREFIX = "sqlens-v11-"
DEFAULT_SQLENS_PROFILE_SEMANTICS = 4.0
RUNNER_PATH = Path(__file__).with_name("pgvector_upstream_overhead_control.py")
ACTIVE_SESSION_QUERY = """
SELECT count(*)
FROM pg_stat_activity
WHERE datname = current_database()
  AND pid <> pg_backend_pid()
  AND state = 'active'
  AND backend_type = 'client backend'
""".strip()


class ControllerError(RuntimeError):
    """A Docker, PostgreSQL, binary, or runner contract failed."""


class ActiveSessionsError(ControllerError):
    """The database has active sessions other than this gate connection."""


class DigestMismatchError(ControllerError):
    """A host or server binary did not match the requested digest."""


class RunnerFailedError(ControllerError):
    """The external runner failed, while retaining its manifest record."""

    def __init__(self, message: str, record: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.record = dict(record)


class FinalizationError(ControllerError):
    """The two staging arms do not form one publishable paired experiment."""


class RecoveryFailedError(ControllerError):
    """Restoring the original server binary failed and takes error priority."""

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class TerminationRequested(BaseException):
    """SIGTERM requested controlled restoration before process termination."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 digest")
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


def validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", value):
        raise argparse.ArgumentTypeError("expected an unquoted identifier or schema.identifier")
    return value


def validate_container(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("invalid Docker container name or ID")
    return value


def parse_target_recalls(value: str) -> list[float]:
    try:
        recalls = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated recall targets") from exc
    if not recalls or any(not 0 < recall <= 1 for recall in recalls):
        raise argparse.ArgumentTypeError("target recalls must be in (0, 1]")
    return sorted(set(recalls))


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def claim_controller_manifest(
    path: Path, value: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing controller manifest {path}")
        if existing.get("run_uuid") != value.get("run_uuid"):
            raise ControllerError("resume manifest belongs to a different run UUID")
        return existing
    if resume:
        raise FileNotFoundError(f"resume requested but controller manifest does not exist: {path}")
    atomic_write_json(path, value)
    return dict(value)


def counterbalanced_final_schedule(
    run_uuid: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    material = f"{run_uuid}|{seed}|pgvector-final-abba"
    parity = int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16) & 1
    a, b = ("official", "sqlens_disabled")
    if parity:
        a, b = b, a
    pair_zero = [a, b]
    pair_one = [b, a]
    implementations = pair_zero + pair_one
    schedule = [
        {
            "sequence": position + 1,
            "final_block": 0 if position < 2 else 1,
            "position_in_pair": position % 2,
            "implementation": implementation,
        }
        for position, implementation in enumerate(implementations)
    ]
    arm_counts = {
        implementation: implementations.count(implementation)
        for implementation in ("official", "sqlens_disabled")
    }
    first_counts = {
        implementation: [pair_zero[0], pair_one[0]].count(implementation)
        for implementation in ("official", "sqlens_disabled")
    }
    audit = {
        "seed": seed,
        "seed_material_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "pair_orders": ["AB", "BA"],
        "concrete_pair_orders": [pair_zero, pair_one],
        "arm_counts": arm_counts,
        "first_in_pair_counts": first_counts,
        "seeded_balance_verified": (
            set(arm_counts.values()) == {2} and set(first_counts.values()) == {1}
        ),
    }
    if not audit["seeded_balance_verified"]:
        raise ControllerError("internal AB/BA schedule balance audit failed")
    return schedule, audit


def audit_controller_execution_journal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    calibration_order = list(manifest.get("calibration_order", []))
    final_schedule = list(manifest.get("final_schedule", []))
    expected = [
        ("calibration", implementation, None)
        for implementation in calibration_order
    ] + [
        ("final", str(item["implementation"]), int(item["final_block"]))
        for item in final_schedule
    ]
    successful = [
        record
        for record in manifest.get("runner_runs", [])
        if record.get("exit_code") == 0
    ]
    actual = [
        (
            str(record.get("execution_stage")),
            str(record.get("implementation")),
            (
                None
                if record.get("final_block") is None
                else int(record.get("final_block"))
            ),
        )
        for record in successful
    ]
    run_uuid = str(manifest.get("run_uuid", ""))
    same_run = all(str(record.get("run_uuid")) == run_uuid for record in successful)
    append_only = all(
        record.get("staging_manifest", {})
        .get("resume_append_only_audit", {})
        .get("passed")
        is True
        for record in successful
    )
    binary_bound = all(
        bool(record.get("staging_manifest", {}).get("server_vector_so_sha256"))
        for record in successful
    )
    balance = manifest.get("seeded_balance_audit", {})
    passed = (
        actual == expected
        and same_run
        and append_only
        and binary_bound
        and len(expected) == 6
        and balance.get("seeded_balance_verified") is True
        and balance.get("pair_orders") == ["AB", "BA"]
    )
    audit = {
        "expected": expected,
        "actual": actual,
        "successful_runner_runs": len(successful),
        "same_run_uuid": same_run,
        "all_stage_resumes_append_only": append_only,
        "all_stages_bound_to_server_binary": binary_bound,
        "passed": passed,
    }
    if not passed:
        raise ControllerError(f"controller execution journal failed AB/BA audit: {audit}")
    return audit


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise FinalizationError("cannot compute a percentile from an empty sample")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _target_key(target: float) -> str:
    return format(float(target), "g")


def _selection_entry(
    manifest: Mapping[str, Any], filter_name: str, target: float
) -> dict[str, str]:
    raw = manifest.get("target_selection", {}).get(filter_name, {}).get(_target_key(target))
    if isinstance(raw, Mapping):
        return {
            "status": str(raw.get("status", "")),
            "config_label": str(raw.get("config_label", "")),
        }
    if isinstance(raw, str):
        if raw.startswith("ef"):
            return {"status": "selected", "config_label": raw}
        return {"status": raw, "config_label": ""}
    raise FinalizationError(f"missing selection for {filter_name}/{target:g}")


def _paired_bootstrap(
    official: Sequence[float],
    sqlens: Sequence[float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not official or len(official) != len(sqlens):
        raise FinalizationError("paired bootstrap requires nonempty equal-length arms")
    pairs = list(zip(map(float, official), map(float, sqlens)))
    if any(left <= 0 or right <= 0 for left, right in pairs):
        raise FinalizationError("paired bootstrap requires positive latency values")
    observed_delta = statistics.fmean(right - left for left, right in pairs)
    observed_speedup = (
        statistics.fmean(left for left, _right in pairs)
        / statistics.fmean(right for _left, right in pairs)
    )
    rng = random.Random(seed)
    deltas: list[float] = []
    speedups: list[float] = []
    for _ in range(samples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(statistics.fmean(right - left for left, right in draw))
        speedups.append(
            statistics.fmean(left for left, _right in draw)
            / statistics.fmean(right for _left, right in draw)
        )
    return {
        "latency_delta_direction": "sqlens_disabled_minus_official",
        "latency_delta_mean_ms": observed_delta,
        "latency_delta_ci_low_ms": _percentile(deltas, 0.025),
        "latency_delta_ci_high_ms": _percentile(deltas, 0.975),
        "speedup_mean": observed_speedup,
        "speedup_ci_low": _percentile(speedups, 0.025),
        "speedup_ci_high": _percentile(speedups, 0.975),
        "speedup_lcb95": _percentile(speedups, 0.05),
    }


def _recall_lcb95(
    rows_by_query: Mapping[int, Sequence[float]], samples: int, seed: int
) -> float:
    query_means = [
        statistics.fmean(float(value) for value in rows_by_query[query_no])
        for query_no in sorted(rows_by_query)
    ]
    if not query_means:
        raise FinalizationError("cannot verify recall from an empty final sample")
    rng = random.Random(seed)
    means = [
        statistics.fmean(
            query_means[rng.randrange(len(query_means))] for _ in query_means
        )
        for _ in range(samples)
    ]
    return _percentile(means, 0.05)


def finalize_ab_artifacts(
    arm_manifest_paths: Sequence[Path],
    publish_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    controller_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise FinalizationError("bootstrap_samples must be positive")
    if publish_path.exists():
        raise FinalizationError(f"refusing to overwrite published artifact {publish_path}")
    if len(arm_manifest_paths) != 2:
        raise FinalizationError("finalizer requires exactly two arm manifests")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in arm_manifest_paths]
    by_arm = {str(item.get("implementation")): item for item in manifests}
    if set(by_arm) != {"official", "sqlens_disabled"} or len(by_arm) != 2:
        raise FinalizationError("finalizer requires official and sqlens_disabled exactly once")
    official = by_arm["official"]
    sqlens = by_arm["sqlens_disabled"]
    for arm, manifest in by_arm.items():
        if manifest.get("status") != "arm_ready" or manifest.get("artifact_valid") is not True:
            raise FinalizationError(f"arm {arm} is not a valid arm_ready staging artifact")
    if official.get("run_uuid") != sqlens.get("run_uuid"):
        raise FinalizationError("arm run UUIDs differ; cross-run pairing is forbidden")
    if controller_manifest is None:
        raise FinalizationError("finalizer requires the controller execution journal")
    if controller_manifest.get("run_uuid") != official.get("run_uuid"):
        raise FinalizationError("controller and arm run UUIDs differ")
    try:
        controller_audit = audit_controller_execution_journal(controller_manifest)
    except ControllerError as exc:
        raise FinalizationError(str(exc)) from exc

    shared_fields = (
        "formal_design",
        "source_hashes",
        "database_fingerprint",
        "query_splits",
        "schedule_contract",
    )
    for field in shared_fields:
        if official.get(field) != sqlens.get(field):
            raise FinalizationError(f"cross-arm {field} mismatch")
    design = official.get("formal_design", {})
    filters = list(design.get("filters", []))
    targets = [float(value) for value in design.get("target_recalls", [])]
    if len(filters) != 14 or targets != [0.90, 0.95, 0.99] or design.get("cell_count") != 42:
        raise FinalizationError("formal design is not the fixed 14-filter/3-target/42-cell design")
    if design.get("formal_family") not in {"off", "strict_order"}:
        raise FinalizationError("formal family must be predeclared as off or strict_order")
    expected_splits = {
        "screen": {"first": 0, "last": 19, "queries": 20},
        "verification": {"first": 20, "last": 99, "queries": 80},
        "final": {"first": 100, "last": 199, "queries": 100},
    }
    if official.get("query_splits") != expected_splits:
        raise FinalizationError("query split contract is not the fixed q0..q199 formal split")
    shared_hashes = official.get("source_hashes", {})
    for key in (
        "runner_sha256",
        "filters_sha256",
        "truth_sha256",
        "graph_identity_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(shared_hashes.get(key, ""))):
            raise FinalizationError(f"shared source hash {key} is missing or invalid")
    sql_hashes = shared_hashes.get("hybrid_sql_sha256_by_filter", {})
    if set(sql_hashes) != set(filters) or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in sql_hashes.values()
    ):
        raise FinalizationError("hybrid SQL hash key space is incomplete or invalid")
    database = official.get("database_fingerprint", {})
    required_database_fields = {
        "system_identifier", "database_oid", "table_oid", "table_relfilenode",
        "index_oid", "index_relfilenode", "indexdef_sha256", "data_epoch",
    }
    if not required_database_fields.issubset(database) or not re.fullmatch(
        r"[0-9a-f]{64}", str(database.get("indexdef_sha256", ""))
    ):
        raise FinalizationError("database/table/index/data-epoch fingerprint is incomplete")
    graph_binding = database.get("source_clone_graph_identity", {})
    graph_proof = graph_binding.get("proof", {})
    if (
        graph_proof.get("same_heap") is not True
        or graph_proof.get("logical_equal") is not True
        or not graph_binding.get("source_index")
        or not graph_binding.get("clone_index")
    ):
        raise FinalizationError("source/clone logical graph identity proof is incomplete")

    for arm, manifest in by_arm.items():
        binary = manifest.get("server_binary_provenance", {})
        source = manifest.get("source_provenance", {})
        required_source = (
            "source_tag", "source_commit", "build_recipe", "compiler_flags",
            "dirty_diff_sha256", "source_tree",
        )
        if binary.get("binary_hash_matches_expected") is not True:
            raise FinalizationError(f"arm {arm} binary digest was not verified")
        image_id = str(binary.get("server_image_id", ""))
        actual_digest = str(binary.get("vector_so_sha256", ""))
        expected_digest = str(binary.get("expected_vector_so_sha256", ""))
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or not re.fullmatch(r"[0-9a-f]{64}", actual_digest)
            or actual_digest != expected_digest
            or any(not source.get(key) for key in required_source)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(source.get("dirty_diff_sha256", ""))
            )
        ):
            raise FinalizationError(f"arm {arm} binary/source provenance is incomplete")
        guc_audit = manifest.get("settings_audit", {}).get("hnsw_guc_audit", {})
        block_audits = list(manifest.get("guc_block_audits", []))
        phase_counts = {
            phase: sum(item.get("phase") == phase for item in block_audits)
            for phase in ("screen", "verification", "final")
        }
        if (
            guc_audit.get("all_nonstock_forced_safe") is not True
            or guc_audit.get("unhandled_nonstock_gucs")
            or phase_counts != {"screen": 14, "verification": 14, "final": 28}
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("after_sha256", "")))
                for item in block_audits
            )
        ):
            raise FinalizationError(
                f"arm {arm} does not prove an inert SQLens GUC inventory at every block"
            )
        warmups = list(manifest.get("warmup_invocations", []))
        required_warmups = {("calibration", None), ("final", 0), ("final", 1)}
        observed_warmups = {
            (
                str(item.get("execution_stage")),
                None if item.get("final_block") is None else int(item.get("final_block")),
            )
            for item in warmups
        }
        expected_warmup_hash = manifest.get("schedule_contract", {}).get(
            "warmup_spec_sha256"
        )
        if not required_warmups.issubset(observed_warmups) or any(
            item.get("warmup_spec_sha256") != expected_warmup_hash for item in warmups
        ):
            raise FinalizationError(f"arm {arm} deterministic warmup evidence is incomplete")
    if official["server_binary_provenance"].get("vector_so_sha256") != OFFICIAL_VECTOR_SO_SHA256:
        raise FinalizationError("official arm does not use the pinned upstream binary")
    if official["server_binary_provenance"].get("vector_so_sha256") == sqlens["server_binary_provenance"].get("vector_so_sha256"):
        raise FinalizationError("the two arms resolve to the same binary digest")

    raw_by_arm: dict[str, list[dict[str, str]]] = {}
    allowed_labels: dict[str, dict[str, set[str]]] = {
        arm: {name: set() for name in filters} for arm in by_arm
    }
    for arm, manifest in by_arm.items():
        for filter_name in filters:
            for target in targets:
                selection = _selection_entry(manifest, filter_name, target)
                if selection["status"] == "selected" and selection["config_label"]:
                    allowed_labels[arm][filter_name].add(selection["config_label"])
    for arm, manifest in by_arm.items():
        raw_path = Path(str(manifest.get("outputs", {}).get("raw", "")))
        if not raw_path.is_file():
            raise FinalizationError(f"arm {arm} raw output is missing")
        recorded_hash = manifest.get("output_hashes", {}).get("raw", {}).get("sha256")
        if not recorded_hash or sha256_file(raw_path) != recorded_hash:
            raise FinalizationError(f"arm {arm} raw output hash mismatch")
        rows = _read_csv(raw_path)
        seen_measurements: set[str] = set()
        half = int(manifest["schedule_contract"]["final_repeats"]) // 2
        for row in rows:
            if row.get("phase") != "final":
                continue
            try:
                query_no = int(row["query_no"])
                repeat = int(row["repeat"])
            except (KeyError, ValueError) as exc:
                raise FinalizationError(f"arm {arm} has invalid final key fields") from exc
            filter_name = str(row.get("filter_name", ""))
            config_label = str(row.get("config_label", ""))
            expected_measurement = (
                f"{arm}|final|{filter_name}|q{query_no}|r{repeat}|{config_label}"
            )
            expected_block = "0" if repeat < half else "1"
            measurement = str(row.get("measurement_key", ""))
            if (
                row.get("run_uuid") != official.get("run_uuid")
                or row.get("implementation") != arm
                or row.get("execution_stage") != "final"
                or row.get("query_split") != "final"
                or row.get("final_block") != expected_block
                or filter_name not in allowed_labels[arm]
                or config_label not in allowed_labels[arm][filter_name]
                or measurement != expected_measurement
                or measurement in seen_measurements
            ):
                raise FinalizationError(
                    f"arm {arm} contains a foreign/duplicate final measurement row"
                )
            seen_measurements.add(measurement)
        raw_by_arm[arm] = rows

    final_query_nos = [int(value) for value in official["schedule_contract"]["final_query_nos"]]
    final_repeats = int(official["schedule_contract"]["final_repeats"])
    if final_query_nos != list(range(100, 200)) or final_repeats <= 0 or final_repeats % 2:
        raise FinalizationError("final schedule must cover q100..q199 with positive even repeats")
    expected_pair_keys = {
        (query_no, repeat)
        for query_no in final_query_nos
        for repeat in range(final_repeats)
    }
    cells: list[dict[str, Any]] = []
    expected_cell_keys = {
        f"{name}|{_target_key(target)}" for name in filters for target in targets
    }
    for filter_name in filters:
        for target in targets:
            cell_key = f"{filter_name}|{_target_key(target)}"
            selections = {
                arm: _selection_entry(manifest, filter_name, target)
                for arm, manifest in by_arm.items()
            }
            if any(item["status"] == "no_verified_config_meets_target" for item in selections.values()):
                raise FinalizationError(
                    f"cell {cell_key} has no verified matched-recall config; "
                    "the run remains staging-only"
                )
                continue
            if any(item["status"] != "selected" or not item["config_label"] for item in selections.values()):
                raise FinalizationError(f"cell {cell_key} has incomplete arm selection")
            latency_maps: dict[str, dict[tuple[int, int], float]] = {}
            recall_lcbs: dict[str, float] = {}
            for arm, rows in raw_by_arm.items():
                selected_label = selections[arm]["config_label"]
                selected_rows = [
                    row for row in rows
                    if row.get("phase") == "final"
                    and row.get("filter_name") == filter_name
                    and row.get("config_label") == selected_label
                ]
                latency_map: dict[tuple[int, int], float] = {}
                recall_by_query: dict[int, list[float]] = {}
                for row in selected_rows:
                    key = (int(row["query_no"]), int(row["repeat"]))
                    expected_pair_key = f"final|{filter_name}|q{key[0]}|r{key[1]}"
                    if key in latency_map or row.get("pair_key") != expected_pair_key:
                        raise FinalizationError(f"cell {cell_key}/{arm} has duplicate or invalid pair keys")
                    if str(row.get("valid", "")).lower() != "true" or row.get("error"):
                        raise FinalizationError(f"cell {cell_key}/{arm} contains invalid final rows")
                    latency_map[key] = float(row["latency_ms"])
                    try:
                        recall = float(row["recall_at_10"])
                    except (KeyError, ValueError) as exc:
                        raise FinalizationError(
                            f"cell {cell_key}/{arm} has invalid recall evidence"
                        ) from exc
                    if not 0.0 <= recall <= 1.0 or str(
                        row.get("truth_self_excluded", "")
                    ).lower() != "true":
                        raise FinalizationError(
                            f"cell {cell_key}/{arm} has invalid/self-included recall evidence"
                        )
                    recall_by_query.setdefault(key[0], []).append(recall)
                if set(latency_map) != expected_pair_keys:
                    raise FinalizationError(f"cell {cell_key}/{arm} final key set is incomplete or foreign")
                latency_maps[arm] = latency_map
                recall_seed = int(
                    hashlib.sha256(
                        f"{bootstrap_seed}|recall|{cell_key}|{arm}".encode("utf-8")
                    ).hexdigest()[:16],
                    16,
                )
                recall_lcbs[arm] = _recall_lcb95(
                    recall_by_query, bootstrap_samples, recall_seed
                )
                if recall_lcbs[arm] < target:
                    raise FinalizationError(
                        f"cell {cell_key}/{arm} held-out recall LCB "
                        f"{recall_lcbs[arm]:.6f} misses target {target:.6f}"
                    )
            ordered_keys = sorted(expected_pair_keys)
            official_query_means = [
                statistics.fmean(
                    latency_maps["official"][(query_no, repeat)]
                    for repeat in range(final_repeats)
                )
                for query_no in final_query_nos
            ]
            sqlens_query_means = [
                statistics.fmean(
                    latency_maps["sqlens_disabled"][(query_no, repeat)]
                    for repeat in range(final_repeats)
                )
                for query_no in final_query_nos
            ]
            seed_material = f"{bootstrap_seed}|{cell_key}"
            cell_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
            paired = _paired_bootstrap(
                official_query_means,
                sqlens_query_means,
                bootstrap_samples,
                cell_seed,
            )
            cells.append(
                {
                    "cell_key": cell_key,
                    "filter_name": filter_name,
                    "target_recall": target,
                    "status": "paired",
                    "pairs": len(ordered_keys),
                    "bootstrap_clusters": len(final_query_nos),
                    "arm_selection": selections,
                    "official_recall_lcb95": recall_lcbs["official"],
                    "sqlens_disabled_recall_lcb95": recall_lcbs["sqlens_disabled"],
                    **paired,
                }
            )
    actual_cell_keys = {str(cell["cell_key"]) for cell in cells}
    if actual_cell_keys != expected_cell_keys or len(cells) != 42:
        raise FinalizationError("finalizer did not produce the exact formal 42-cell key space")
    report = {
        "schema_version": 1,
        "run_uuid": official["run_uuid"],
        "status": "published",
        "published_at_utc": utc_now(),
        "formal_design": design,
        "arm_manifest_paths": [str(path) for path in arm_manifest_paths],
        "paired_gate": {
            "passed": len(cells) == 42 and all(cell["status"] == "paired" for cell in cells),
            "cell_count": 42,
            "paired_cells": sum(cell["status"] == "paired" for cell in cells),
            "incomparable_cells": 0,
            "pair_key_fields": ["filter_name", "target_recall", "query_no", "repeat"],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_unit": "query_no after averaging paired repeats",
            "latency_delta_definition": "SQLens-disabled minus official milliseconds",
            "controller_execution_audit": controller_audit,
        },
        "cells": cells,
    }
    atomic_write_json(publish_path, report)
    return report


def command_text(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "").strip()


def run_command(argv: Sequence[str], *, cwd: Path | None = None) -> Any:
    result = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        stderr = str(getattr(result, "stderr", "") or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise ControllerError(f"command failed ({result.returncode}): {shlex.join(argv)}{detail}")
    return result


def parse_digest_output(output: str, what: str) -> str:
    match = re.search(r"\b([0-9a-fA-F]{64})\b", output)
    if not match:
        raise ControllerError(f"{what} did not return a SHA-256 digest")
    return match.group(1).lower()


def docker_exec(container: str, *command: str) -> Any:
    return run_command(["docker", "exec", container, *command])


def discover_vector_so(container: str) -> str:
    pkglibdir = command_text(docker_exec(container, "pg_config", "--pkglibdir"))
    if not pkglibdir:
        raise ControllerError("pg_config --pkglibdir returned an empty path")
    return f"{pkglibdir.rstrip('/')}/vector.so"


def docker_copy(source: str, destination: str) -> Any:
    return run_command(["docker", "cp", source, destination])


def server_binary_digest(container: str, binary_path: str) -> str:
    output = command_text(docker_exec(container, "sha256sum", binary_path))
    return parse_digest_output(output, "server vector.so digest")


def wait_for_postgres(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.pg_isready_timeout_seconds
    command = [
        "docker", "exec", args.server_container, "pg_isready",
        "-U", args.pg_user,
        "-d", args.pg_database,
    ]
    while True:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return
        if time.monotonic() >= deadline:
            detail = str(getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
            raise ControllerError(f"pg_isready timed out{': ' + detail if detail else ''}")
        time.sleep(args.pg_isready_poll_seconds)


def restart_and_wait(args: argparse.Namespace) -> None:
    run_command(["docker", "restart", args.server_container])
    wait_for_postgres(args)


def database_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "host": args.pg_host,
        "port": args.pg_port,
        "database": args.pg_database,
        "user": args.pg_user,
        "server_container": args.server_container,
    }


def active_session_count(args: argparse.Namespace) -> int:
    command = [
        "docker", "exec", "--env", "PGAPPNAME=pgvector-binary-ab-control-gate",
        args.server_container,
        "psql", "--no-psqlrc", "--quiet", "--no-align", "--tuples-only",
        "--username", args.pg_user, "--dbname", args.pg_database,
        "--command", ACTIVE_SESSION_QUERY,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = str(getattr(result, "stderr", "") or "").strip()
        raise ControllerError(f"active-session gate failed{': ' + stderr if stderr else ''}")
    try:
        return int(command_text(result))
    except ValueError as exc:
        raise ControllerError("active-session gate returned a non-integer count") from exc


def enforce_active_session_gate(args: argparse.Namespace) -> dict[str, Any]:
    count = active_session_count(args)
    evidence = {
        "active_sessions_excluding_gate": count,
        "allow_active_sessions": bool(args.allow_active_sessions),
        "checked_at_utc": utc_now(),
    }
    if count and not args.allow_active_sessions:
        raise ActiveSessionsError(
            f"refusing binary switch with {count} active non-controller sessions; "
            "pass --allow-active-sessions to override"
        )
    return evidence


def source_spec(args: argparse.Namespace, implementation: str) -> dict[str, str]:
    if implementation == "official":
        return {
            "implementation": implementation,
            "source_tag": args.official_vector_source_tag,
            "source_commit": args.official_vector_source_commit,
            "expected_digest": OFFICIAL_VECTOR_SO_SHA256,
            "host_path": str(args.official_vector_so),
            "build_recipe": args.official_vector_build_recipe,
            "compiler_flags": args.official_vector_compiler_flags,
            "source_repo": str(args.official_vector_source_repo),
        }
    return {
        "implementation": implementation,
        "source_tag": args.sqlens_vector_source_tag,
        "source_commit": args.sqlens_vector_source_commit,
        "expected_digest": args.sqlens_vector_so_sha256,
        "host_path": str(args.sqlens_vector_so),
        "build_recipe": args.sqlens_vector_build_recipe,
        "compiler_flags": args.sqlens_vector_compiler_flags,
        "source_repo": str(args.sqlens_vector_source_repo),
        "required_sqlens_build_prefix": args.required_sqlens_build_prefix,
        "minimum_sqlens_profile_semantics": args.minimum_sqlens_profile_semantics,
    }


def validate_host_binary(source: Mapping[str, str]) -> str:
    path = Path(source["host_path"])
    actual = sha256_file(path)
    expected = source["expected_digest"]
    if actual != expected:
        raise DigestMismatchError(
            f"host vector.so digest mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def install_binary(
    args: argparse.Namespace,
    binary_path: str,
    source: Mapping[str, str],
) -> str:
    actual = validate_host_binary(source)
    remote_temp = f"{binary_path}.controller-{uuid.uuid4().hex}.tmp"
    docker_copy(source["host_path"], f"{args.server_container}:{remote_temp}")
    docker_exec(args.server_container, "chmod", "0755", remote_temp)
    # mv is atomic because the temporary file and vector.so share the directory.
    docker_exec(args.server_container, "mv", "-f", remote_temp, binary_path)
    return actual


def shared_runner_args(args: argparse.Namespace, *, resume: bool | None = None) -> list[str]:
    resume_enabled = args.resume if resume is None else resume
    argv = [
        "--filters-csv", str(args.filters_csv),
        "--truth-csv", str(args.truth_csv),
        "--table", args.table,
        "--index", args.index,
        "--source-index", args.source_index,
        "--clone-index", args.clone_index,
        "--graph-identity-json", str(args.graph_identity_json),
        "--out-dir", str(args.out_dir),
        "--tag", args.tag,
        "--run-uuid", args.run_uuid,
        "--formal-family", args.formal_family,
        "--data-epoch", args.data_epoch,
        "--k", str(args.k),
        "--target-recalls", ",".join(format(item, "g") for item in args.target_recalls),
        "--promotion-margin", format(args.promotion_margin, "g"),
        "--screen-repeats", str(args.screen_repeats),
        "--verification-repeats", str(args.verification_repeats),
        "--final-repeats", str(args.final_repeats),
        "--warmup-queries", str(args.warmup_queries),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--bootstrap-seed", str(args.bootstrap_seed),
        "--schedule-seed", str(args.schedule_seed),
        "--statement-timeout-ms", str(args.statement_timeout_ms),
        "--server-container", args.server_container,
        "--resume" if resume_enabled else "--no-resume",
    ]
    if args.config_ladder:
        argv.extend(["--config-ladder", str(args.config_ladder)])
    if args.filter_names:
        argv.append("--filter-names")
        argv.extend(args.filter_names)
    return argv


def build_runner_argv(
    args: argparse.Namespace,
    implementation: str,
    execution_stage: str = "calibration",
    final_block: int | None = None,
) -> list[str]:
    source = source_spec(args, implementation)
    argv = [
        sys.executable,
        str(RUNNER_PATH),
        "--implementation", implementation,
        "--execution-stage", execution_stage,
        *shared_runner_args(
            args, resume=args.resume or execution_stage == "final"
        ),
        "--expected-vector-so-sha256", source["expected_digest"],
        "--vector-source-tag", source["source_tag"],
        "--vector-source-commit", source["source_commit"],
        "--vector-build-recipe", source["build_recipe"],
        "--vector-compiler-flags", source["compiler_flags"],
        "--vector-source-repo", source["source_repo"],
    ]
    if final_block is not None:
        argv.extend(["--final-block", str(final_block)])
    if implementation == "sqlens_disabled":
        argv.extend([
            "--required-sqlens-build-prefix", args.required_sqlens_build_prefix,
            "--minimum-sqlens-profile-semantics",
            format(args.minimum_sqlens_profile_semantics, "g"),
        ])
    return argv


def run_external_runner(
    args: argparse.Namespace,
    implementation: str,
    execution_stage: str = "calibration",
    final_block: int | None = None,
) -> dict[str, Any]:
    argv = build_runner_argv(args, implementation, execution_stage, final_block)
    environment = os.environ.copy()
    environment.update({
        "PGHOST": args.pg_host,
        "PGPORT": str(args.pg_port),
        "PGDATABASE": args.pg_database,
        "PGUSER": args.pg_user,
        "PGAPPNAME": f"pgvector-binary-ab-{implementation}",
    })
    started = utc_now()
    try:
        result = subprocess.run(
            argv,
            cwd=str(args.repo_root),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        record = {
            "implementation": implementation,
            "execution_stage": execution_stage,
            "final_block": final_block,
            "argv": argv,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "exit_code": result.returncode,
        }
        log_dir = args.out_dir / "staging" / args.run_uuid / "controller_logs"
        log_stem = (
            f"{execution_stage}_{'none' if final_block is None else final_block}_"
            f"{implementation}_{started.replace(':', '').replace('-', '')}"
        )
        stdout_path = log_dir / f"{log_stem}.stdout.log"
        stderr_path = log_dir / f"{log_stem}.stderr.log"
        atomic_write_text(stdout_path, result.stdout or "")
        atomic_write_text(stderr_path, result.stderr or "")
        record["child_logs"] = {
            "stdout": {
                "path": str(stdout_path),
                "sha256": sha256_file(stdout_path),
                "bytes": stdout_path.stat().st_size,
            },
            "stderr": {
                "path": str(stderr_path),
                "sha256": sha256_file(stderr_path),
                "bytes": stderr_path.stat().st_size,
                "tail": (result.stderr or "")[-4096:],
            },
        }
    except OSError as exc:
        record = {
            "implementation": implementation,
            "execution_stage": execution_stage,
            "final_block": final_block,
            "argv": argv,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "exit_code": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        raise RunnerFailedError("could not start external runner", record) from exc
    if result.returncode != 0:
        raise RunnerFailedError(
            f"{implementation} runner failed with exit code {result.returncode}",
            record,
        )
    staging_manifest_path = arm_manifest_path(args, implementation)
    if not staging_manifest_path.is_file():
        raise RunnerFailedError(
            f"{implementation} runner returned success without its staging manifest",
            record,
        )
    staging_manifest = json.loads(staging_manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = (
        {"calibration_complete"}
        if execution_stage == "calibration"
        else (
            {"final_in_progress"}
            if final_block == 0
            else {"arm_ready", "staging_unconfirmed"}
        )
    )
    if staging_manifest.get("status") not in allowed_statuses:
        raise RunnerFailedError(
            f"{implementation} runner staging status is invalid for {execution_stage}/{final_block}",
            record,
        )
    record["staging_manifest"] = {
        "path": str(staging_manifest_path),
        "sha256_after_stage": sha256_file(staging_manifest_path),
        "status": staging_manifest.get("status"),
        "checkpoint_spec_sha256": staging_manifest.get("checkpoint_spec_sha256"),
        "resume_append_only_audit": staging_manifest.get("resume_append_only_audit"),
        "server_vector_so_sha256": staging_manifest.get(
            "server_binary_provenance", {}
        ).get("vector_so_sha256"),
        "target_selection_sha256": staging_manifest.get("target_selection_sha256"),
    }
    return record


def _record_switch(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    binary_path: str,
    source: Mapping[str, Any],
    *,
    recovery: bool = False,
) -> dict[str, Any]:
    event = {
        "sequence": len(manifest["switches"]) + 1,
        "implementation": source["implementation"],
        "recovery": recovery,
        "binary_path": binary_path,
        "source": dict(source),
        "started_at_utc": utc_now(),
        "status": "starting",
    }
    manifest["switches"].append(event)
    persist_controller_state(args, manifest)
    return event


def persist_controller_state(
    args: argparse.Namespace, manifest: Mapping[str, Any]
) -> None:
    atomic_write_json(args.manifest, manifest)
    journal_path = getattr(args, "recovery_journal", None)
    if journal_path:
        journal = {
            "schema_version": 1,
            "run_uuid": manifest.get("run_uuid"),
            "status": manifest.get("status"),
            "binary_path": manifest.get("binary_path"),
            "initial_binary": manifest.get("initial_binary"),
            "switches": manifest.get("switches", []),
            "restoration": manifest.get("restoration"),
            "updated_at_utc": utc_now(),
        }
        atomic_write_json(Path(journal_path), journal)


def fsync_existing_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def arm_manifest_path(args: argparse.Namespace, implementation: str) -> Path:
    prefix = f"pgvector_upstream_overhead_control_{implementation}_{args.tag}"
    return args.out_dir / "staging" / args.run_uuid / f"{prefix}_manifest.json"


def switch_binary(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    binary_path: str,
    source: Mapping[str, Any],
    *,
    recovery: bool = False,
) -> dict[str, Any]:
    event = _record_switch(manifest, args, binary_path, source, recovery=recovery)
    try:
        event["active_session_gate"] = (
            {"skipped": True, "reason": "unconditional recovery"}
            if recovery
            else enforce_active_session_gate(args)
        )
        event["replacement_attempted"] = True
        event["host_digest"] = install_binary(args, binary_path, source)
        event["binary_replaced"] = True
        restart_and_wait(args)
        event["server_digest"] = server_binary_digest(args.server_container, binary_path)
        if event["server_digest"] != source["expected_digest"]:
            raise DigestMismatchError(
                f"server vector.so digest mismatch: expected {source['expected_digest']}, "
                f"got {event['server_digest']}"
            )
        event["status"] = "installed_and_verified"
        event["finished_at_utc"] = utc_now()
        persist_controller_state(args, manifest)
        return event
    except Exception as exc:
        event["status"] = "failed"
        event["finished_at_utc"] = utc_now()
        event["error"] = f"{exc.__class__.__name__}: {exc}"
        persist_controller_state(args, manifest)
        raise


def validate_runtime_args(args: argparse.Namespace) -> None:
    required = {
        "server container": args.server_container,
        "official binary": args.official_vector_so,
        "SQLens-disabled binary": args.sqlens_vector_so,
        "SQLens digest": args.sqlens_vector_so_sha256,
        "official source tag": args.official_vector_source_tag,
        "official source commit": args.official_vector_source_commit,
        "SQLens source tag": args.sqlens_vector_source_tag,
        "SQLens source commit": args.sqlens_vector_source_commit,
        "run UUID": args.run_uuid,
        "data epoch": args.data_epoch,
        "official build recipe": args.official_vector_build_recipe,
        "official compiler flags": args.official_vector_compiler_flags,
        "SQLens build recipe": args.sqlens_vector_build_recipe,
        "SQLens compiler flags": args.sqlens_vector_compiler_flags,
        "official source repository": args.official_vector_source_repo,
        "SQLens source repository": args.sqlens_vector_source_repo,
        "graph identity JSON": args.graph_identity_json,
        "source graph index": args.source_index,
        "clone graph index": args.clone_index,
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise ControllerError("missing required runtime option(s): " + ", ".join(missing))
    if tuple(args.target_recalls) != (0.90, 0.95, 0.99):
        raise ControllerError("formal target recalls must be exactly 0.90,0.95,0.99")
    if args.formal_family not in {"off", "strict_order"}:
        raise ControllerError("formal family must be off or strict_order")
    if args.filter_names:
        raise ControllerError("formal controller runs always use the fixed 14-filter CSV")
    if args.final_repeats % 2:
        raise ControllerError("formal final repeats must be even for AB/BA blocks")
    for label, path in (
        ("official source repository", args.official_vector_source_repo),
        ("SQLens source repository", args.sqlens_vector_source_repo),
    ):
        if not path.is_dir():
            raise ControllerError(f"{label} does not exist: {path}")
    if not args.graph_identity_json.is_file():
        raise ControllerError(
            f"graph identity JSON does not exist: {args.graph_identity_json}"
        )


def _legacy_ephemeral_run_controller(args: argparse.Namespace) -> dict[str, Any]:
    validate_runtime_args(args)
    sources = {
        implementation: source_spec(args, implementation)
        for implementation in ("official", "sqlens_disabled")
    }
    prevalidated_host_digests = {
        implementation: validate_host_binary(source)
        for implementation, source in sources.items()
    }
    # Refuse before entering the recovery region. A failed busy gate must not
    # trigger a no-op "restore" restart that kills the sessions it protects.
    preflight_active_session_gate = enforce_active_session_gate(args)
    binary_path = discover_vector_so(args.server_container)
    with tempfile.TemporaryDirectory(prefix="pgvector-binary-ab-") as temporary:
        original_host_path = Path(temporary) / "vector.so.initial"
        docker_copy(
            f"{args.server_container}:{binary_path}",
            str(original_host_path),
        )
        original_digest = sha256_file(original_host_path)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "controller": "run_pgvector_binary_ab_control",
            "status": "running",
            "started_at_utc": utc_now(),
            "database": database_identity(args),
            "preflight_active_session_gate": preflight_active_session_gate,
            "prevalidated_host_digests": prevalidated_host_digests,
            "binary_path": binary_path,
            "initial_binary": {
                "backup_scope": "ephemeral_controller_lifetime",
                "sha256": original_digest,
            },
            "binary_sources": sources,
            "switches": [],
            "runner_runs": [],
        }
        atomic_write_json(args.manifest, manifest)
        failure: BaseException | None = None
        try:
            for implementation in ("official", "sqlens_disabled"):
                source = sources[implementation]
                switch_binary(args, manifest, binary_path, source)
                try:
                    runner_record = run_external_runner(args, implementation)
                except RunnerFailedError as exc:
                    manifest["runner_runs"].append(exc.record)
                    atomic_write_json(args.manifest, manifest)
                    raise
                manifest["runner_runs"].append(runner_record)
                atomic_write_json(args.manifest, manifest)
        except BaseException as exc:
            failure = exc
            manifest["status"] = "failed"
            manifest["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
            atomic_write_json(args.manifest, manifest)
        finally:
            restore_source = {
                "implementation": "restore_initial",
                "source_tag": "initial-container-binary",
                "source_commit": "",
                "expected_digest": original_digest,
                "host_path": str(original_host_path),
            }
            restore_required = any(
                bool(event.get("replacement_attempted"))
                for event in manifest["switches"]
                if not event.get("recovery")
            )
            if restore_required:
                try:
                    switch_binary(
                        args, manifest, binary_path, restore_source, recovery=True
                    )
                    manifest["restoration"] = {
                        "status": "verified",
                        "sha256": original_digest,
                    }
                except BaseException as restore_error:
                    manifest["restoration"] = {
                        "status": "failed",
                        "error": f"{restore_error.__class__.__name__}: {restore_error}",
                    }
                    if failure is None:
                        failure = restore_error
            else:
                manifest["restoration"] = {
                    "status": "not_required",
                    "sha256": original_digest,
                }
            manifest["finished_at_utc"] = utc_now()
            if failure is None:
                manifest["status"] = "completed"
            atomic_write_json(args.manifest, manifest)
        if failure is not None:
            raise failure
        return manifest


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    validate_runtime_args(args)
    sources = {
        implementation: source_spec(args, implementation)
        for implementation in ("official", "sqlens_disabled")
    }
    prevalidated_host_digests = {
        implementation: validate_host_binary(source)
        for implementation, source in sources.items()
    }
    # A busy preflight must not create recovery state or restart PostgreSQL.
    preflight_active_session_gate = enforce_active_session_gate(args)
    binary_path = discover_vector_so(args.server_container)
    schedule, balance_audit = counterbalanced_final_schedule(
        args.run_uuid, args.schedule_seed
    )
    calibration_order = list(balance_audit["concrete_pair_orders"][0])
    controller_spec = {
        "run_uuid": args.run_uuid,
        "implementations": ["official", "sqlens_disabled"],
        "binary_sources": sources,
        "database": database_identity(args),
        "formal_family": args.formal_family,
        "data_epoch": args.data_epoch,
        "filters_csv": str(args.filters_csv),
        "truth_csv": str(args.truth_csv),
        "table": args.table,
        "index": args.index,
        "source_index": args.source_index,
        "clone_index": args.clone_index,
        "graph_identity_json": str(args.graph_identity_json),
        "target_recalls": args.target_recalls,
        "repeats": {
            "screen": args.screen_repeats,
            "verification": args.verification_repeats,
            "final": args.final_repeats,
        },
        "schedule_seed": args.schedule_seed,
        "calibration_order": calibration_order,
        "final_schedule": schedule,
    }
    controller_spec_hash = hashlib.sha256(
        json.dumps(controller_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    recovery_dir = args.manifest.parent / "recovery"
    backup_path = recovery_dir / "vector.so.initial"
    args.recovery_journal = recovery_dir / "journal.json"
    initial_manifest: dict[str, Any] = {
        "schema_version": 2,
        "controller": "run_pgvector_binary_ab_control",
        "run_uuid": args.run_uuid,
        "status": "backing_up_initial_binary",
        "started_at_utc": utc_now(),
        "controller_run_spec": controller_spec,
        "controller_run_spec_sha256": controller_spec_hash,
        "database": database_identity(args),
        "preflight_active_session_gate": preflight_active_session_gate,
        "prevalidated_host_digests": prevalidated_host_digests,
        "binary_path": binary_path,
        "binary_sources": sources,
        "calibration_order": calibration_order,
        "final_schedule": schedule,
        "seeded_balance_audit": balance_audit,
        "switches": [],
        "runner_runs": [],
    }
    manifest = claim_controller_manifest(
        args.manifest, initial_manifest, resume=args.resume
    )
    if manifest.get("controller_run_spec_sha256") != controller_spec_hash:
        raise ControllerError("resume controller run spec does not match the existing run")
    if manifest.get("binary_path") != binary_path:
        raise ControllerError("resume discovered a different server vector.so path")

    if args.resume:
        initial = manifest.get("initial_binary", {})
        if initial.get("host_path") != str(backup_path) or not backup_path.is_file():
            raise ControllerError("persistent initial-binary backup is missing on resume")
        original_digest = sha256_file(backup_path)
        if original_digest != initial.get("sha256"):
            raise ControllerError("persistent initial-binary backup digest mismatch on resume")
    else:
        if backup_path.exists():
            raise FileExistsError(f"recovery backup already exists: {backup_path}")
        recovery_dir.mkdir(parents=True, exist_ok=True)
        docker_copy(f"{args.server_container}:{binary_path}", str(backup_path))
        fsync_existing_file(backup_path)
        original_digest = sha256_file(backup_path)
        manifest["initial_binary"] = {
            "backup_scope": "persistent_run_uuid_recovery_journal",
            "host_path": str(backup_path),
            "sha256": original_digest,
            "fsync_verified": True,
        }
        manifest["status"] = "running"
        persist_controller_state(args, manifest)

    restore_source = {
        "implementation": "restore_initial",
        "source_tag": "initial-container-binary",
        "source_commit": "",
        "expected_digest": original_digest,
        "host_path": str(backup_path),
    }
    previous_sigterm: Any = None
    signal_installed = False

    def request_termination(_signum: int, _frame: Any) -> None:
        raise TerminationRequested("SIGTERM requested controlled binary restoration")

    try:
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_termination)
        signal_installed = True
    except ValueError:
        # Unit-test workers may call the controller outside the main thread.
        signal_installed = False

    failure: BaseException | None = None
    try:
        execution_steps = [
            {
                "implementation": implementation,
                "execution_stage": "calibration",
                "final_block": None,
            }
            for implementation in calibration_order
        ]
        execution_steps.extend(
            {
                "implementation": item["implementation"],
                "execution_stage": "final",
                "final_block": item["final_block"],
            }
            for item in schedule
        )
        completed_keys = {
            str(record.get("controller_step_key"))
            for record in manifest.get("runner_runs", [])
            if record.get("exit_code") == 0
        }
        for step in execution_steps:
            key = (
                f"{step['execution_stage']}|{step['implementation']}|"
                f"{step['final_block'] if step['final_block'] is not None else '-'}"
            )
            if key in completed_keys:
                continue
            implementation = str(step["implementation"])
            switch_binary(args, manifest, binary_path, sources[implementation])
            try:
                runner_record = run_external_runner(
                    args,
                    implementation,
                    str(step["execution_stage"]),
                    step["final_block"],
                )
            except RunnerFailedError as exc:
                record = dict(exc.record)
                record["controller_step_key"] = key
                record["run_uuid"] = args.run_uuid
                manifest["runner_runs"].append(record)
                persist_controller_state(args, manifest)
                raise
            runner_record["controller_step_key"] = key
            runner_record["run_uuid"] = args.run_uuid
            manifest["runner_runs"].append(runner_record)
            completed_keys.add(key)
            persist_controller_state(args, manifest)
        manifest["execution_journal_audit"] = audit_controller_execution_journal(manifest)
        manifest["status"] = "measurements_complete_restoration_pending"
        persist_controller_state(args, manifest)
    except BaseException as exc:
        failure = exc
        manifest["status"] = "failed_restoration_pending"
        manifest["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
        persist_controller_state(args, manifest)
    finally:
        restore_required = any(
            bool(event.get("replacement_attempted"))
            for event in manifest.get("switches", [])
            if not event.get("recovery")
        )
        if restore_required:
            try:
                switch_binary(
                    args, manifest, binary_path, restore_source, recovery=True
                )
                manifest["restoration"] = {
                    "status": "verified",
                    "sha256": original_digest,
                    "finished_at_utc": utc_now(),
                }
            except BaseException as restore_error:
                manifest["restoration"] = {
                    "status": "failed",
                    "error": f"{restore_error.__class__.__name__}: {restore_error}",
                    "finished_at_utc": utc_now(),
                }
                failure = RecoveryFailedError(
                    f"initial binary restoration failed: {restore_error}", failure
                )
        else:
            manifest["restoration"] = {
                "status": "not_required",
                "sha256": original_digest,
            }
        persist_controller_state(args, manifest)
        if signal_installed:
            signal.signal(signal.SIGTERM, previous_sigterm)

    if failure is not None:
        manifest["status"] = "recovery_failed" if isinstance(failure, RecoveryFailedError) else "failed"
        manifest["finished_at_utc"] = utc_now()
        persist_controller_state(args, manifest)
        raise failure

    publish_path = (
        args.publish_path
        or args.out_dir
        / "published"
        / f"pgvector_binary_ab_{args.formal_family}_{args.tag}_{args.run_uuid}.json"
    )
    try:
        report = finalize_ab_artifacts(
            [
                arm_manifest_path(args, "official"),
                arm_manifest_path(args, "sqlens_disabled"),
            ],
            publish_path,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            controller_manifest=manifest,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "staging_incomplete",
                "finished_at_utc": utc_now(),
                "fatal_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        persist_controller_state(args, manifest)
        raise
    manifest.update(
        {
            "status": "completed",
            "finished_at_utc": utc_now(),
            "published_artifact": {
                "path": str(publish_path),
                "sha256": sha256_file(publish_path),
                "paired_gate": report["paired_gate"],
            },
        }
    )
    persist_controller_state(args, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recoverable pgvector vector.so binary A/B controller")
    parser.add_argument("--server-container", type=validate_container)
    parser.add_argument("--official-vector-so", "--official-binary", dest="official_vector_so", type=Path)
    parser.add_argument("--sqlens-vector-so", "--sqlens-binary", dest="sqlens_vector_so", type=Path)
    parser.add_argument("--sqlens-vector-so-sha256", "--sqlens-digest", dest="sqlens_vector_so_sha256", type=validate_sha256)
    parser.add_argument("--official-vector-source-tag", "--official-source-tag", dest="official_vector_source_tag", default="")
    parser.add_argument("--official-vector-source-commit", "--official-source-commit", dest="official_vector_source_commit", default="")
    parser.add_argument("--sqlens-vector-source-tag", "--sqlens-source-tag", dest="sqlens_vector_source_tag", default="")
    parser.add_argument("--sqlens-vector-source-commit", "--sqlens-source-commit", dest="sqlens_vector_source_commit", default="")
    parser.add_argument("--official-vector-build-recipe", default="")
    parser.add_argument("--official-vector-compiler-flags", default="")
    parser.add_argument("--official-vector-source-repo", type=Path)
    parser.add_argument("--sqlens-vector-build-recipe", default="")
    parser.add_argument("--sqlens-vector-compiler-flags", default="")
    parser.add_argument("--sqlens-vector-source-repo", type=Path)
    parser.add_argument("--required-sqlens-build-prefix", "--sqlens-build-prefix", default=DEFAULT_SQLENS_BUILD_PREFIX)
    parser.add_argument("--minimum-sqlens-profile-semantics", "--sqlens-profile-semantics", type=float, default=DEFAULT_SQLENS_PROFILE_SEMANTICS)
    parser.add_argument("--filters-csv", type=Path, default=Path("experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv"))
    parser.add_argument("--config-ladder", type=Path)
    parser.add_argument("--table", type=validate_identifier, default="public.amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--index", type=validate_identifier, default="public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx")
    parser.add_argument("--source-index", type=validate_identifier, default="public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx")
    parser.add_argument("--clone-index", type=validate_identifier, default="public.amazon_grocery_reviews_10m_pgvector_hnsw_bfs_clone_idx")
    parser.add_argument("--graph-identity-json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/hybrid_vector_db"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--publish-path", type=Path)
    parser.add_argument("--tag", default="20260718")
    parser.add_argument("--run-uuid", default="")
    parser.add_argument("--formal-family", choices=("off", "strict_order"), default="off")
    parser.add_argument("--data-epoch", default="")
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--target-recalls", type=parse_target_recalls, default=[0.90, 0.95, 0.99])
    parser.add_argument("--promotion-margin", type=float, default=0.02)
    parser.add_argument("--screen-repeats", type=positive_int, default=1)
    parser.add_argument("--verification-repeats", type=positive_int, default=2)
    parser.add_argument("--final-repeats", type=positive_int, default=6)
    parser.add_argument("--warmup-queries", type=positive_int, default=5)
    parser.add_argument("--bootstrap-samples", type=positive_int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--statement-timeout-ms", type=nonnegative_int, default=300_000)
    parser.add_argument("--pg-host", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument("--pg-port", type=positive_int, default=int(os.environ.get("PGPORT", "55432")))
    parser.add_argument("--pg-database", default=os.environ.get("PGDATABASE", "hybrid_vector"))
    parser.add_argument("--pg-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--pg-isready-timeout-seconds", type=positive_int, default=60)
    parser.add_argument("--pg-isready-poll-seconds", type=float, default=1.0)
    parser.add_argument("--allow-active-sessions", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    dry_uuid = args.run_uuid or "<generated-at-runtime>"
    manifest = args.manifest or args.out_dir / "staging" / dry_uuid / "controller.json"
    schedule, audit = counterbalanced_final_schedule(dry_uuid, args.schedule_seed)
    return {
        "controller": "run_pgvector_binary_ab_control",
        "runner": str(RUNNER_PATH),
        "implementations": ["official", "sqlens_disabled"],
        "calibration_arms": ["official", "sqlens_disabled"],
        "final_schedule": schedule,
        "seeded_balance_audit": audit,
        "formal_family": args.formal_family,
        "formal_target_recalls": args.target_recalls,
        "formal_cell_count": 42,
        "run_uuid": dry_uuid,
        "server_container": args.server_container,
        "manifest": str(manifest),
        "official_pinned_vector_so_sha256": OFFICIAL_VECTOR_SO_SHA256,
        "sqlens_vector_so_sha256_supplied": bool(args.sqlens_vector_so_sha256),
        "sqlens_build_prefix": args.required_sqlens_build_prefix,
        "sqlens_profile_semantics": args.minimum_sqlens_profile_semantics,
        "resume": args.resume,
        "file_access": False,
        "docker_access": False,
        "database_access": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    if not args.run_uuid:
        if args.resume:
            print("controller failed: --resume requires an explicit --run-uuid", file=sys.stderr)
            return 1
        args.run_uuid = str(uuid.uuid4())
    if args.manifest is None:
        args.manifest = args.out_dir / "staging" / args.run_uuid / "controller.json"
    try:
        run_controller(args)
    except TerminationRequested as exc:
        print(f"controller terminated after restoration: {exc}", file=sys.stderr)
        return 128
    except Exception as exc:
        print(f"controller failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote controller manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
