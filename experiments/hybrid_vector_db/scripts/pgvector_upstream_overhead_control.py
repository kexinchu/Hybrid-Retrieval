"""Formal stock-pgvector overhead control with held-out matched-recall tuning.

The caller switches ``vector.so`` before invoking this runner.  The runner
never copies libraries, restarts services, creates indexes, or uses a SQLens
query marker.  Both implementations execute the same stock filtered HNSW SQL.
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
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_TRUTH = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv"
DEFAULT_OUT_DIR = ROOT / "results/hybrid_vector_db"
DEFAULT_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_INDEX = "public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx"
UPSTREAM_MAX_EF_SEARCH = 1000
EVALUATION_EF_PATCH_SHA256 = {
    10_000: "d63b8d75015cffb90d9bd7f04d0c8f572502f0b84f77f59f581d224db7601bcf",
    100_000: "2393fff3ac210d9fd19478ed7552db559359d19cdae693de51e42c71d04cf225",
}
DEFAULT_EF_VALUES = (250, 500, 750, UPSTREAM_MAX_EF_SEARCH)
DEFAULT_BUDGET_RUNGS = (
    (1, 100_000, 1.0),
    (2, 1_000_000, 8.0),
    (3, 5_000_000, 32.0),
)
OFF_REPRESENTATIVE_MAX_SCAN = 100_000
OFF_REPRESENTATIVE_SCAN_MEM = 1.0
OFFICIAL_UPSTREAM_VECTOR_SO_SHA256 = "812292e3e7553c3dbe6a4187b528430a7f9c25693f4876b8d22f88829592a778"
DEFAULT_SQLENS_BUILD_PREFIX = "sqlens-v11-"
DEFAULT_SQLENS_PROFILE_SEMANTICS = 7.0
FORMAL_TARGET_RECALLS = (0.90, 0.95, 0.99)
FORMAL_FAMILIES = ("off", "strict_order")
STOCK_HNSW_GUCS = frozenset(
    {
        "hnsw.ef_search",
        "hnsw.iterative_scan",
        "hnsw.max_scan_tuples",
        "hnsw.scan_mem_multiplier",
    }
)
SQLENS_PROFILE_FIELDS = (
    "graph_elements_visited",
    "raw_index_tids_returned",
    "hnsw_am_callback_ms",
    "executor_residual_ms",
    "index_page_loads",
    "index_page_runs",
    "index_page_distinct_pages",
    "index_page_distinct_pages_exact",
    "index_page_profile_scope",
    "heap_tid_returns",
    "heap_tid_page_runs",
    "heap_tid_distinct_pages",
    "heap_tid_distinct_pages_exact",
    "heap_tid_sequence_scope",
    "heap_blks_are_exact_heap_io",
)
SQLENS_OFF_GUCS = (
    "hnsw.filter_strategy",
    "hnsw.page_access",
    "hnsw.index_page_access",
    "hnsw.guidance_compose_exact_or",
    "hnsw.guidance_require_epoch",
    "hnsw.require_full_memory_build",
)
SQLENS_RESET_GUCS = (
    "hnsw.guided_collect_target",
    "hnsw.page_window",
    "hnsw.page_prefetch_min_items",
    "hnsw.page_disable_after_no_merge",
    "hnsw.d3_probe_requests",
    "hnsw.d3_min_benefit_per_byte",
    "hnsw.d3_max_fragment_mb",
    "hnsw.d3_page_min_skip_rate",
    "hnsw.metadata_cache_max_mb",
    "hnsw.build_page_order",
    "hnsw.build_seed",
)
SQLENS_EMPTY_GUCS = (
    "hnsw.clone_source",
    "hnsw.preferred_index",
)

SCREEN_QUERY_NOS = tuple(range(0, 20))
VERIFICATION_QUERY_NOS = tuple(range(20, 100))
FINAL_QUERY_NOS = tuple(range(100, 200))
STAGE_QUERY_NOS = {
    "screen": SCREEN_QUERY_NOS,
    "verification": VERIFICATION_QUERY_NOS,
    "final": FINAL_QUERY_NOS,
}

RAW_FIELDS = (
    "run_uuid",
    "implementation",
    "execution_stage",
    "final_block",
    "phase",
    "query_split",
    "filter_name",
    "query_no",
    "query_id",
    "repeat",
    "config_label",
    "config_family",
    "budget_rank",
    "ef_search",
    "iterative_scan",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "schedule_position",
    "pair_key",
    "measurement_key",
    "latency_ms",
    "returned",
    "result_ids",
    "recall_at_10",
    "truth_self_excluded",
    "valid",
    "error",
)
# Compatibility alias for callers/tests that consumed the first revision.
CSV_FIELDS = RAW_FIELDS


class ProvenanceGateError(RuntimeError):
    """The loaded server binary/runtime is not the requested implementation."""


class CheckpointContractError(RuntimeError):
    """A checkpoint is partial, stale, duplicated, or otherwise ambiguous."""


class FormalResultInvalid(RuntimeError):
    """The run completed but held-out or completeness validation failed."""


@dataclass(frozen=True)
class Config:
    ef_search: int
    iterative_scan: str
    max_scan_tuples: int
    scan_mem_multiplier: float
    budget_rank: int = 0

    @property
    def family(self) -> str:
        return self.iterative_scan

    @property
    def label(self) -> str:
        mem = format(self.scan_mem_multiplier, "g").replace(".", "p")
        return (
            f"ef{self.ef_search}_iter{self.iterative_scan}_"
            f"max{self.max_scan_tuples}_mem{mem}_b{self.budget_rank}"
        )


@dataclass(frozen=True)
class TruthEntry:
    query_id: int
    kth_distance_sq: float
    tie_tolerance: float
    exact_ids: tuple[int, ...]
    self_excluded: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def target_recalls(value: str) -> list[float]:
    try:
        parsed = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated recall targets") from exc
    if not parsed or any(not math.isfinite(item) or item <= 0 or item > 1 for item in parsed):
        raise argparse.ArgumentTypeError("target recalls must be in (0, 1]")
    return parsed


def validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", value):
        raise argparse.ArgumentTypeError("expected an unquoted identifier or schema.identifier")
    return value


def validate_container_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("invalid Docker container name or ID")
    return value


def validate_sha256(value: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 digest")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_formal_design(
    filters: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    formal_family: str,
) -> dict[str, Any]:
    names = [str(row["filter_name"]) for row in filters]
    if len(names) != 14 or len(set(names)) != 14:
        raise ValueError("formal design requires exactly 14 unique filters")
    normalized = tuple(float(target) for target in targets)
    if normalized != FORMAL_TARGET_RECALLS:
        raise ValueError("formal design requires target recalls exactly 0.90,0.95,0.99")
    if formal_family not in FORMAL_FAMILIES:
        raise ValueError(
            "relaxed_order is exploratory-only; formal_family must be off or strict_order"
        )
    return {
        "formal_family": formal_family,
        "filters": names,
        "target_recalls": list(FORMAL_TARGET_RECALLS),
        "target_metric": "query_level_mean_recall_at_10",
        "uncertainty": "query_bootstrap_95pct_ci_reported_not_used_as_target",
        "cell_keys": [
            f"{name}|{format(target, 'g')}" for name in names for target in normalized
        ],
        "cell_count": len(names) * len(normalized),
    }


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def git_dirty_diff_provenance(root: Path = ROOT) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        raise ProvenanceGateError("could not capture git dirty-diff provenance")
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff.stdout)
    untracked_paths = [item for item in untracked.stdout.split(b"\0") if item]
    for encoded in sorted(untracked_paths):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = root / relative
        digest.update(b"untracked\0" + encoded + b"\0")
        if path.is_file():
            digest.update(sha256_file(path).encode("ascii"))
        else:
            digest.update(b"non-file")
    return {
        "dirty": bool(diff.stdout or untracked_paths),
        "dirty_diff_sha256": digest.hexdigest(),
        "tracked_diff_bytes": len(diff.stdout),
        "untracked_file_count": len(untracked_paths),
        "method": "sha256(git diff --binary HEAD plus untracked path/content hashes)",
    }


def source_tree_provenance(path: Path, expected_commit: str) -> dict[str, Any]:
    if not path.is_dir():
        raise ProvenanceGateError(f"vector source tree does not exist: {path}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0:
        raise ProvenanceGateError(f"vector source tree is not a Git worktree: {path}")
    actual_commit = revision.stdout.strip()
    if actual_commit != expected_commit:
        raise ProvenanceGateError(
            f"vector source commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    dirty = git_dirty_diff_provenance(path)
    return {
        "source_tree": str(path.resolve()),
        "source_commit": actual_commit,
        **dirty,
    }


def upstream_parameter_ceiling_provenance(
    source_repo: Path,
    patch_path: Path | None,
    max_ef_search: int,
) -> dict[str, Any]:
    if max_ef_search == UPSTREAM_MAX_EF_SEARCH:
        if patch_path is not None:
            raise ProvenanceGateError(
                "the release-limit official arm must not declare an evaluation patch"
            )
        return {
            "mode": "unmodified_upstream_release_binary",
            "max_ef_search": UPSTREAM_MAX_EF_SEARCH,
            "patch_applied": False,
        }
    expected_patch_sha256 = EVALUATION_EF_PATCH_SHA256.get(max_ef_search)
    if expected_patch_sha256 is None or patch_path is None:
        raise ProvenanceGateError(
            "an extended evaluation ceiling requires a supported max_ef_search "
            "and its explicit canonical patch"
        )
    if not patch_path.is_file():
        raise ProvenanceGateError(f"upstream evaluation patch is missing: {patch_path}")
    patch_bytes = patch_path.read_bytes()
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    if patch_sha256 != expected_patch_sha256:
        raise ProvenanceGateError(
            "upstream evaluation patch does not match the canonical ef_search-only patch"
        )
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "HEAD", "--", "src/hnsw.c", "src/hnsw.h"],
        cwd=str(source_repo),
        capture_output=True,
        check=False,
    )
    names = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--"],
        cwd=str(source_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(source_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if diff.returncode or names.returncode or untracked.returncode:
        raise ProvenanceGateError("could not audit the upstream evaluation source diff")
    changed_files = sorted(item for item in names.stdout.splitlines() if item)
    untracked_files = sorted(item for item in untracked.stdout.splitlines() if item)
    if diff.stdout != patch_bytes or changed_files != ["src/hnsw.c", "src/hnsw.h"]:
        raise ProvenanceGateError(
            "upstream evaluation source differs from the canonical two-file ceiling patch"
        )
    if untracked_files:
        raise ProvenanceGateError(
            "upstream evaluation source contains untracked files: "
            + ", ".join(untracked_files)
        )
    return {
        "mode": "upstream_algorithm_evaluation_only_guc_ceiling_extension",
        "max_ef_search": max_ef_search,
        "patch_applied": True,
        "patch_path": str(patch_path.resolve()),
        "patch_sha256": patch_sha256,
        "changed_files": changed_files,
        "semantic_change": "HNSW_MAX_EF_SEARCH and its GUC help text only",
        "algorithm_change": False,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
            target.write(text)
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
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def ordered_fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> list[str]:
    present = {key for row in rows for key in row}
    fields = [field for field in preferred if field in present]
    fields.extend(sorted(present - set(fields)))
    return fields


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    selected_fields = list(fieldnames or ordered_fields(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=selected_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_filters(path: Path, selected: set[str] | None = None) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {"filter_name", "predicate", "target_rate"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"filter CSV must contain {sorted(required)}")
    if selected:
        rows = [row for row in rows if row["filter_name"] in selected]
        if selected != {row["filter_name"] for row in rows}:
            raise ValueError("requested filter is missing from filters CSV")
    if len({row["filter_name"] for row in rows}) != len(rows):
        raise ValueError("filters CSV contains duplicate filter_name")
    for row in rows:
        if any(marker in row["predicate"].lower() for marker in (";", "--", "/*", "*/")):
            raise ValueError(f"unsafe predicate for filter {row['filter_name']}")
    return rows


def parse_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def load_truth(
    path: Path,
    query_nos: Iterable[int],
    filter_names: set[str],
    k: int,
    candidate_validity_predicate: str = "",
) -> dict[tuple[str, int], TruthEntry]:
    wanted = set(query_nos)
    rows = read_csv(path)
    required = {
        "filter_name",
        "query_no",
        "query_id",
        "exact_filtered_topk_ids",
        "kth_distance_sq",
        "tie_tolerance",
        "self_excluded",
    }
    if not rows or not required.issubset(rows[0]):
        missing = required - set(rows[0]) if rows else required
        raise ValueError(f"tie-aware exact truth is missing {sorted(missing)}")
    if candidate_validity_predicate:
        if "candidate_validity_predicate" not in rows[0]:
            raise ValueError(
                "tie-aware exact truth is missing candidate_validity_predicate"
            )
        observed_validity = {
            row.get("candidate_validity_predicate", "").strip() for row in rows
        }
        if observed_validity != {candidate_validity_predicate.strip()}:
            raise ValueError(
                "tie-aware exact truth candidate-validity contract does not match the run"
            )
    result: dict[tuple[str, int], TruthEntry] = {}
    ids_by_query: dict[int, int] = {}
    for row in rows:
        name = row["filter_name"]
        query_no = int(row["query_no"])
        if name not in filter_names or query_no not in wanted:
            continue
        key = (name, query_no)
        if key in result:
            raise ValueError(f"duplicate exact truth key {key}")
        if row["self_excluded"].lower() != "true":
            raise ValueError("exact truth must explicitly be self-excluded")
        query_id = int(row["query_id"])
        prior = ids_by_query.setdefault(query_no, query_id)
        if prior != query_id:
            raise ValueError(f"query_no={query_no} maps to multiple query IDs")
        exact_ids = parse_ids(row["exact_filtered_topk_ids"])
        if len(exact_ids) < k:
            raise ValueError(f"truth row {key} has fewer than k={k} IDs")
        result[key] = TruthEntry(
            query_id=query_id,
            kth_distance_sq=float(row["kth_distance_sq"]),
            tie_tolerance=float(row["tie_tolerance"]),
            exact_ids=exact_ids[:k],
            self_excluded=True,
        )
    expected = len(filter_names) * len(wanted)
    if len(result) != expected:
        raise ValueError(f"exact truth is incomplete: rows={len(result)}, expected={expected}")
    return result


def validate_split_contract() -> dict[str, Any]:
    sets = {name: set(values) for name, values in STAGE_QUERY_NOS.items()}
    if sets["screen"] & sets["verification"] or sets["screen"] & sets["final"] or sets["verification"] & sets["final"]:
        raise ValueError("screen, verification, and final query splits overlap")
    if set().union(*sets.values()) != set(range(200)):
        raise ValueError("query splits must cover q0..q199 exactly")
    return {
        name: {"first": values[0], "last": values[-1], "queries": len(values)}
        for name, values in STAGE_QUERY_NOS.items()
    }


def default_config_ladder() -> list[Config]:
    configs: list[Config] = []
    for ef_search in DEFAULT_EF_VALUES:
        configs.append(
            Config(
                ef_search,
                "off",
                OFF_REPRESENTATIVE_MAX_SCAN,
                OFF_REPRESENTATIVE_SCAN_MEM,
                0,
            )
        )
        for iterative_scan in ("strict_order", "relaxed_order"):
            for budget_rank, max_scan_tuples, scan_mem_multiplier in DEFAULT_BUDGET_RUNGS:
                configs.append(
                    Config(
                        ef_search,
                        iterative_scan,
                        max_scan_tuples,
                        scan_mem_multiplier,
                        budget_rank,
                    )
                )
    return configs


def config_from_mapping(
    row: Mapping[str, Any], max_ef_search: int = UPSTREAM_MAX_EF_SEARCH
) -> Config:
    iterative = str(row["iterative_scan"])
    if iterative not in {"off", "strict_order", "relaxed_order"}:
        raise ValueError(f"invalid iterative_scan {iterative!r}")
    config = Config(
        ef_search=int(row["ef_search"]),
        iterative_scan=iterative,
        max_scan_tuples=int(row.get("max_scan_tuples", OFF_REPRESENTATIVE_MAX_SCAN)),
        scan_mem_multiplier=float(row.get("scan_mem_multiplier", OFF_REPRESENTATIVE_SCAN_MEM)),
        budget_rank=int(row.get("budget_rank", 0)),
    )
    if config.ef_search <= 0 or config.max_scan_tuples <= 0:
        raise ValueError("ef_search and max_scan_tuples must be positive")
    if config.ef_search > max_ef_search:
        raise ValueError(
            "binary overhead control must stay inside the provenance-gated official pgvector "
            f"hnsw.ef_search range (<= {max_ef_search})"
        )
    if not math.isfinite(config.scan_mem_multiplier) or config.scan_mem_multiplier <= 0:
        raise ValueError("scan_mem_multiplier must be finite and positive")
    if config.budget_rank < 0:
        raise ValueError("budget_rank must be nonnegative")
    return config


def load_config_ladder(
    path: Path | None, max_ef_search: int = UPSTREAM_MAX_EF_SEARCH
) -> list[Config]:
    if path is None:
        return default_config_ladder()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("configs") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("config ladder JSON must be a list or {'configs': [...]} object")
        return [config_from_mapping(row, max_ef_search) for row in rows]
    if path.suffix.lower() == ".csv":
        return [config_from_mapping(row, max_ef_search) for row in read_csv(path)]
    raise ValueError("--config-ladder must be a .json or .csv file")


def effective_config_grid(configs: Sequence[Config]) -> tuple[list[Config], dict[str, Any]]:
    representatives: dict[tuple[Any, ...], Config] = {}
    sources: dict[tuple[Any, ...], list[str]] = {}
    for config in configs:
        if config.iterative_scan == "off":
            key = (config.ef_search, "off")
            representative = Config(
                config.ef_search,
                "off",
                OFF_REPRESENTATIVE_MAX_SCAN,
                OFF_REPRESENTATIVE_SCAN_MEM,
                0,
            )
        else:
            key = (
                config.ef_search,
                config.iterative_scan,
                config.max_scan_tuples,
                config.scan_mem_multiplier,
            )
            representative = config
            if key in representatives and representatives[key].budget_rank != config.budget_rank:
                raise ValueError(
                    "semantically identical iterative configs have conflicting budget_rank values"
                )
        representatives.setdefault(key, representative)
        sources.setdefault(key, []).append(config.label)
    family_order = {"off": 0, "strict_order": 1, "relaxed_order": 2}
    effective = sorted(
        representatives.values(),
        key=lambda item: (
            item.ef_search,
            family_order[item.family],
            item.budget_rank,
            item.max_scan_tuples,
            item.scan_mem_multiplier,
        ),
    )
    if not effective:
        raise ValueError("config ladder is empty")
    families = {config.family for config in effective}
    required_families = {"off", "strict_order"}
    if not required_families.issubset(families):
        raise ValueError(
            "config ladder must cover the formal off and strict_order families, "
            f"got {sorted(families)}"
        )
    proof_groups = [
        {
            "semantic_key": list(key),
            "representative": representatives[key].label,
            "source_count": len(source_labels),
            "dropped_as_equivalent": max(0, len(source_labels) - 1),
            "source_labels": source_labels,
        }
        for key, source_labels in sorted(sources.items(), key=lambda item: str(item[0]))
    ]
    proof = {
        "input_configs": len(configs),
        "effective_configs": len(effective),
        "dropped_equivalent_configs": len(configs) - len(effective),
        "off_equivalence": (
            "for iterative_scan=off, max_scan_tuples and scan_mem_multiplier do not affect execution; "
            "one canonical representative is retained per ef_search"
        ),
        "iterative_budget_semantics": (
            "present iterative families use explicit correlated budget rungs; "
            "max_scan_tuples and scan_mem_multiplier are never Cartesian-expanded"
        ),
        "families": sorted(families, key=family_order.__getitem__),
        "required_formal_families": sorted(
            required_families, key=family_order.__getitem__
        ),
        "groups": proof_groups,
    }
    return effective, proof


def family_max_budget_configs(configs: Sequence[Config]) -> dict[str, Config]:
    result: dict[str, Config] = {}
    family_order = {"off": 0, "strict_order": 1, "relaxed_order": 2}
    present_families = sorted(
        {config.family for config in configs}, key=family_order.__getitem__
    )
    for family in present_families:
        family_configs = [config for config in configs if config.family == family]
        result[family] = max(
            family_configs,
            key=lambda item: (
                item.budget_rank,
                item.max_scan_tuples if family != "off" else 0,
                item.scan_mem_multiplier if family != "off" else 0,
                item.ef_search,
                item.label,
            ),
        )
    return result


def default_query_count_bounds(
    effective_config_count: int,
    filters: int = 14,
    targets: int = 3,
    screen_repeats: int = 1,
    verification_repeats: int = 2,
    final_repeats: int = 6,
    families: int = 3,
) -> dict[str, int]:
    # Global max recall is necessarily one of the family max-recall configs.
    max_promoted_per_filter = min(effective_config_count, targets + families + families)
    screen = effective_config_count * filters * len(SCREEN_QUERY_NOS) * screen_repeats
    verification = max_promoted_per_filter * filters * len(VERIFICATION_QUERY_NOS) * verification_repeats
    final = targets * filters * len(FINAL_QUERY_NOS) * final_repeats
    return {
        "effective_configs": effective_config_count,
        "screen_queries": screen,
        "max_promoted_configs_per_filter": max_promoted_per_filter,
        "verification_query_upper_bound": verification,
        "final_query_upper_bound": final,
        "total_query_upper_bound": screen + verification + final,
    }


def safe_predicate(value: str, name: str) -> str:
    normalized = value.strip()
    if any(marker in normalized.lower() for marker in (";", "--", "/*", "*/")):
        raise ValueError(f"invalid {name}")
    return normalized


def build_hybrid_sql(
    table: str,
    predicate: str,
    k: int,
    candidate_validity_predicate: str = "",
) -> str:
    validate_identifier(table)
    predicate = safe_predicate(predicate, "hybrid predicate")
    validity = safe_predicate(
        candidate_validity_predicate, "candidate-validity predicate"
    )
    if k <= 0:
        raise ValueError("invalid hybrid SQL inputs")
    combined = f"({predicate})"
    if validity:
        combined += f" AND ({validity})"
    sql = (
        f"SELECT id, embedding <-> %s::vector AS distance "
        f"FROM {table} "
        f"WHERE {combined} AND id <> %s "
        f"ORDER BY embedding <-> %s::vector LIMIT {int(k)}"
    )
    lowered = sql.lower()
    if any(marker in lowered for marker in ("vector_sqlens", "guidance", "filter_strategy")):
        raise ValueError("hybrid SQL contains a SQLens marker")
    return sql


def configure_stock(cur: Any, config: Config) -> None:
    cur.execute(f"SET hnsw.ef_search = {int(config.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {config.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(config.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(config.scan_mem_multiplier)}")


def disable_sqlens_gucs(cur: Any) -> list[str]:
    statements = [f"SET {guc} = off" for guc in SQLENS_OFF_GUCS]
    statements.extend(f"RESET {guc}" for guc in SQLENS_RESET_GUCS)
    statements.extend(f"SET {guc} = ''" for guc in SQLENS_EMPTY_GUCS)
    for statement in statements:
        cur.execute(statement)
    return statements


def _hnsw_guc_inventory(cur: Any) -> list[dict[str, str]]:
    cur.execute(
        "SELECT name, vartype, setting, reset_val FROM pg_settings "
        "WHERE name LIKE 'hnsw.%' ORDER BY name"
    )
    inventory: list[dict[str, str]] = []
    for name, vartype, setting, reset_val in cur.fetchall():
        normalized = str(name)
        if not re.fullmatch(r"hnsw\.[a-z0-9_]+", normalized):
            raise ProvenanceGateError(f"unsafe hnsw GUC name returned by pg_settings: {normalized!r}")
        inventory.append(
            {
                "name": normalized,
                "vartype": str(vartype),
                "setting": str(setting),
                "reset_val": str(reset_val),
            }
        )
    return inventory


def enforce_hnsw_guc_allowlist(cur: Any) -> dict[str, Any]:
    """Inventory every runtime HNSW GUC and neutralize non-stock controls."""
    before_rows = _hnsw_guc_inventory(cur)
    actions: list[dict[str, str]] = []
    expected: dict[str, str] = {}
    for row in before_rows:
        name = row["name"]
        if name in STOCK_HNSW_GUCS:
            continue
        if name in SQLENS_EMPTY_GUCS:
            statement = f"SET {name} = ''"
            expected[name] = ""
        elif name in SQLENS_OFF_GUCS or row["vartype"] == "bool":
            statement = f"SET {name} = off"
            expected[name] = "off"
        else:
            statement = f"RESET {name}"
            expected[name] = row["reset_val"]
        cur.execute(statement)
        actions.append(
            {
                "name": name,
                "action": statement,
                "before": row["setting"],
                "expected_after": expected[name],
            }
        )
    after_rows = _hnsw_guc_inventory(cur)
    after = {row["name"]: row["setting"] for row in after_rows}
    before_names = {row["name"] for row in before_rows}
    if before_names != set(after):
        raise ProvenanceGateError("hnsw GUC inventory changed while enforcing the allowlist")
    unhandled = sorted(
        name for name, value in expected.items() if after.get(name) != value
    )
    if unhandled:
        raise ProvenanceGateError(
            f"non-stock hnsw GUCs did not reach their disabled/reset values: {unhandled}"
        )
    return {
        "stock_allowlist": sorted(STOCK_HNSW_GUCS),
        "before": {row["name"]: row["setting"] for row in before_rows},
        "reset_values": {row["name"]: row["reset_val"] for row in before_rows},
        "actions": actions,
        "after": after,
        "unhandled_nonstock_gucs": unhandled,
        "all_nonstock_forced_safe": not unhandled,
    }


def _profile_object(raw: Any) -> dict[str, Any]:
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise ProvenanceGateError("SQLens profile is not a JSON object")
    return profile


def require_sqlens_profile(
    cur: Any,
    required_prefix: str = DEFAULT_SQLENS_BUILD_PREFIX,
    minimum_semantics: float = DEFAULT_SQLENS_PROFILE_SEMANTICS,
) -> tuple[str, dict[str, Any]]:
    try:
        cur.execute("SELECT vector_sqlens_build_id()")
        row = cur.fetchone()
        build_id = str(row[0]) if row and row[0] is not None else ""
    except Exception as exc:
        raise ProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_sqlens_build_id() is unavailable"
        ) from exc
    if not build_id.startswith(required_prefix):
        raise ProvenanceGateError(
            "SQLens v11 provenance gate failed: "
            f"vector_sqlens_build_id() returned {build_id!r}; expected prefix {required_prefix!r}"
        )
    try:
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        row = cur.fetchone()
        profile = _profile_object(row[0] if row else None)
        semantics = float(profile["profile_semantics_version"])
    except Exception as exc:
        raise ProvenanceGateError(
            "SQLens v11 provenance gate failed: "
            "vector_hnsw_last_scan_profile() is unavailable or invalid"
        ) from exc
    missing = [field for field in SQLENS_PROFILE_FIELDS if field not in profile]
    if not math.isfinite(semantics) or semantics < minimum_semantics or missing:
        raise ProvenanceGateError(
            "SQLens v11 profile gate failed: "
            f"semantics={profile.get('profile_semantics_version')!r}, "
            f"minimum={minimum_semantics}, missing={missing!r}"
        )
    return build_id, profile


def gate_implementation(
    cur: Any,
    implementation: str,
    required_sqlens_prefix: str = DEFAULT_SQLENS_BUILD_PREFIX,
    minimum_profile_semantics: float = DEFAULT_SQLENS_PROFILE_SEMANTICS,
) -> dict[str, Any]:
    cur.execute(
        "SELECT COALESCE((SELECT extversion FROM pg_extension WHERE extname = 'vector'), '')"
    )
    row = cur.fetchone()
    extension_version = str(row[0]) if row and row[0] is not None else ""
    if extension_version != "0.8.2":
        raise ProvenanceGateError(
            f"formal gate requires vector extension 0.8.2, got {extension_version!r}"
        )
    if implementation == "official":
        # The binary digest is the authoritative official/upstream gate.  SQL
        # declarations can be stale after the externally managed .so switch.
        return {
            "implementation": "official",
            "vector_extension_version": extension_version,
            "runtime_sql_declarations_used_as_identity": False,
        }
    if implementation == "sqlens_disabled":
        build_id, profile = require_sqlens_profile(
            cur, required_sqlens_prefix, minimum_profile_semantics
        )
        return {
            "implementation": "sqlens_disabled",
            "vector_extension_version": extension_version,
            "loaded_vector_sqlens_build_id": build_id,
            "profile_gate": {
                "required_build_prefix": required_sqlens_prefix,
                "minimum_profile_semantics_version": minimum_profile_semantics,
                "profile_semantics_version": profile["profile_semantics_version"],
                "required_fields": {field: profile[field] for field in SQLENS_PROFILE_FIELDS},
            },
            "sqlens_gucs": "explicit off/reset at connection setup and every atomic phase/filter block",
        }
    raise ProvenanceGateError(f"unsupported implementation {implementation!r}")


def _checked_command(
    command: Sequence[str],
    command_runner: Callable[..., Any],
) -> str:
    result = command_runner(
        list(command), capture_output=True, text=True, check=False
    )
    if int(result.returncode) != 0:
        stderr = str(getattr(result, "stderr", "")).strip()
        raise ProvenanceGateError(f"command failed: {list(command)!r}: {stderr}")
    return str(result.stdout).strip()


def server_vector_binary_provenance(
    container: str,
    expected_sha256: str | None,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    validate_container_name(container)
    expected = validate_sha256(expected_sha256) if expected_sha256 else None
    image = _checked_command(
        ["docker", "inspect", "--format={{.Config.Image}}", container], command_runner
    )
    pkglibdir = _checked_command(
        ["docker", "exec", container, "pg_config", "--pkglibdir"], command_runner
    )
    if not pkglibdir.startswith("/") or "\n" in pkglibdir:
        raise ProvenanceGateError(f"invalid pg_config --pkglibdir output {pkglibdir!r}")
    binary_path = f"{pkglibdir.rstrip('/')}/vector.so"
    digest_output = _checked_command(
        ["docker", "exec", container, "sha256sum", binary_path], command_runner
    )
    fields = digest_output.split()
    if len(fields) < 2:
        raise ProvenanceGateError("docker exec sha256sum returned an invalid result")
    actual = validate_sha256(fields[0])
    if fields[1] != binary_path:
        raise ProvenanceGateError(
            f"sha256sum path mismatch: expected {binary_path!r}, got {fields[1]!r}"
        )
    if expected is not None and actual != expected:
        raise ProvenanceGateError(
            f"server vector.so SHA-256 mismatch: expected {expected}, got {actual}"
        )
    image_id = _checked_command(
        ["docker", "inspect", "--format={{.Image}}", container], command_runner
    )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ProvenanceGateError(f"invalid Docker image ID {image_id!r}")
    host_config_raw = _checked_command(
        ["docker", "inspect", "--format={{json .HostConfig}}", container],
        command_runner,
    )
    try:
        host_config = json.loads(host_config_raw)
    except json.JSONDecodeError as exc:
        raise ProvenanceGateError("invalid Docker HostConfig JSON") from exc
    if not isinstance(host_config, dict):
        raise ProvenanceGateError("Docker HostConfig must be a JSON object")
    resource_limits = {
        "cpuset_cpus": str(host_config.get("CpusetCpus", "")),
        "cpu_period": int(host_config.get("CpuPeriod", 0) or 0),
        "cpu_quota": int(host_config.get("CpuQuota", 0) or 0),
        "nano_cpus": int(host_config.get("NanoCpus", 0) or 0),
        "memory_bytes": int(host_config.get("Memory", 0) or 0),
        "memory_swap_bytes": int(host_config.get("MemorySwap", 0) or 0),
    }
    return {
        "server_container": container,
        "server_image": image,
        "server_image_id": image_id,
        "server_resource_limits": resource_limits,
        "pg_config_pkglibdir": pkglibdir,
        "vector_so_path": binary_path,
        "vector_so_sha256": actual,
        "expected_vector_so_sha256": expected,
        "binary_hash_matches_expected": expected is None or actual == expected,
        "hash_method": "docker exec <container> sha256sum $(pg_config --pkglibdir)/vector.so",
    }


def validate_runtime_args(args: argparse.Namespace) -> None:
    args.max_ef_search = getattr(args, "max_ef_search", UPSTREAM_MAX_EF_SEARCH)
    args.upstream_evaluation_patch = getattr(
        args, "upstream_evaluation_patch", None
    )
    args.candidate_validity_predicate = getattr(
        args, "candidate_validity_predicate", ""
    )
    if not args.server_container:
        raise ProvenanceGateError("formal runs require explicit --server-container")
    if not args.vector_source_tag or not args.vector_source_commit:
        raise ProvenanceGateError(
            "formal runs require --vector-source-tag and --vector-source-commit"
        )
    if not getattr(args, "run_uuid", ""):
        raise ProvenanceGateError("formal runs require an explicit --run-uuid")
    if not getattr(args, "data_epoch", ""):
        raise ProvenanceGateError("formal runs require an explicit --data-epoch")
    if not getattr(args, "vector_build_recipe", ""):
        raise ProvenanceGateError("formal runs require --vector-build-recipe provenance")
    if not getattr(args, "vector_compiler_flags", ""):
        raise ProvenanceGateError("formal runs require --vector-compiler-flags provenance")
    if tuple(getattr(args, "target_recalls", ())) != FORMAL_TARGET_RECALLS:
        raise ProvenanceGateError("formal target recalls must be exactly 0.90,0.95,0.99")
    if getattr(args, "formal_family", "") not in FORMAL_FAMILIES:
        raise ProvenanceGateError("formal family must be off or strict_order")
    safe_predicate(
        getattr(args, "candidate_validity_predicate", ""),
        "candidate-validity predicate",
    )
    if getattr(args, "max_ef_search", 0) not in {
        UPSTREAM_MAX_EF_SEARCH,
        *EVALUATION_EF_PATCH_SHA256,
    }:
        raise ProvenanceGateError(
            "formal max_ef_search must be the release limit 1000 or an audited "
            "evaluation limit (10000 or 100000)"
        )
    if args.max_ef_search > UPSTREAM_MAX_EF_SEARCH and args.config_ladder is None:
        raise ProvenanceGateError(
            "the evaluation ceiling requires an explicit, hashed config ladder"
        )
    if getattr(args, "final_repeats", 0) <= 0 or args.final_repeats % 2:
        raise ProvenanceGateError("formal --final-repeats must be positive and even")
    if getattr(args, "execution_stage", "") == "final" and getattr(args, "final_block", None) not in {0, 1}:
        raise ProvenanceGateError("final execution requires --final-block 0 or 1")
    for label, path in (
        ("filters CSV", args.filters_csv),
        ("truth CSV", args.truth_csv),
        ("graph identity JSON", args.graph_identity_json),
    ):
        if path is None or not path.is_file():
            raise ProvenanceGateError(f"formal {label} does not exist: {path}")
    if args.vector_source_repo is None or not args.vector_source_repo.is_dir():
        raise ProvenanceGateError(
            f"formal vector source tree does not exist: {args.vector_source_repo}"
        )
    if not args.source_index or not args.clone_index:
        raise ProvenanceGateError("formal runs require source and clone index identities")
    if (
        not math.isfinite(args.promotion_margin)
        or not 0 <= args.promotion_margin < 1
    ):
        raise ProvenanceGateError("--promotion-margin must be in [0, 1)")
    if (
        not math.isfinite(args.minimum_sqlens_profile_semantics)
        or args.minimum_sqlens_profile_semantics < DEFAULT_SQLENS_PROFILE_SEMANTICS
    ):
        raise ProvenanceGateError(
            f"--minimum-sqlens-profile-semantics must be >= {DEFAULT_SQLENS_PROFILE_SEMANTICS:g}"
        )
    if args.implementation == "official":
        if not args.expected_vector_so_sha256:
            raise ProvenanceGateError(
                "official formal mode requires --expected-vector-so-sha256"
            )
        ceiling = upstream_parameter_ceiling_provenance(
            args.vector_source_repo,
            args.upstream_evaluation_patch,
            args.max_ef_search,
        )
        if (
            args.max_ef_search == UPSTREAM_MAX_EF_SEARCH
            and args.expected_vector_so_sha256 != OFFICIAL_UPSTREAM_VECTOR_SO_SHA256
        ):
            raise ProvenanceGateError(
                "official expected digest does not equal the pinned upstream vector.so digest"
            )
        if (
            ceiling["patch_applied"] is True
            and args.expected_vector_so_sha256 == OFFICIAL_UPSTREAM_VECTOR_SO_SHA256
        ):
            raise ProvenanceGateError(
                "the evaluation ceiling patch must produce a distinct binary digest"
            )


def plan_index_names(plan: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(plan, dict):
        if plan.get("Index Name"):
            names.add(str(plan["Index Name"]))
        for child in plan.values():
            names.update(plan_index_names(child))
    elif isinstance(plan, list):
        for child in plan:
            names.update(plan_index_names(child))
    return names


def assert_hnsw_explain_gate(plan: Any, expected_index: str) -> set[str]:
    names = plan_index_names(plan)
    expected = expected_index.split(".")[-1]
    if not any(name.split(".")[-1] == expected for name in names):
        raise RuntimeError(
            f"HNSW EXPLAIN gate failed: expected {expected_index!r}, indexes={sorted(names)!r}"
        )
    return names


def explain_hybrid(
    cur: Any,
    sql: str,
    vector_text: str,
    query_id: int,
    expected_index: str,
) -> dict[str, Any]:
    cur.execute(
        "EXPLAIN (FORMAT JSON, COSTS OFF) " + sql,
        (vector_text, query_id, vector_text),
    )
    row = cur.fetchone()
    payload = row[0] if row else None
    if isinstance(payload, str):
        payload = json.loads(payload)
    plan = payload[0] if isinstance(payload, list) and payload else payload
    indexes = assert_hnsw_explain_gate(plan, expected_index)
    return {
        "valid": True,
        "expected_index": expected_index,
        "index_names": sorted(indexes),
    }


def balanced_order(values: Sequence[Any], block_no: int, seed: int) -> list[Any]:
    if not values:
        return []
    rotation = (block_no + seed) % len(values)
    return list(values[rotation:]) + list(values[:rotation])


def pair_key(phase: str, filter_name: str, query_no: int, repeat: int) -> str:
    return f"{phase}|{filter_name}|q{query_no}|r{repeat}"


def measurement_key(
    implementation: str,
    phase: str,
    filter_name: str,
    query_no: int,
    repeat: int,
    config_label: str,
) -> str:
    return (
        f"{implementation}|{phase}|{filter_name}|q{query_no}|r{repeat}|{config_label}"
    )


def validate_raw_schema_and_dedup(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        missing = set(RAW_FIELDS) - set(row)
        if missing:
            raise CheckpointContractError(
                f"checkpoint schema is missing fields {sorted(missing)}"
            )
        key = str(row["measurement_key"])
        if key in seen:
            raise CheckpointContractError(f"duplicate checkpoint measurement_key {key}")
        seen.add(key)
        phase = str(row["phase"])
        if phase not in STAGE_QUERY_NOS:
            raise CheckpointContractError(f"checkpoint contains unknown phase {phase!r}")
        try:
            recomputed = measurement_key(
                str(row["implementation"]),
                phase,
                str(row["filter_name"]),
                int(row["query_no"]),
                int(row["repeat"]),
                str(row["config_label"]),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointContractError("checkpoint key fields are invalid") from exc
        if key != recomputed:
            raise CheckpointContractError(
                f"checkpoint measurement_key is inconsistent with row fields: {key!r}"
            )


def expected_stage_keys(
    implementation: str,
    phase: str,
    filter_name: str,
    configs: Sequence[Config],
    query_nos: Sequence[int],
    repeats: int,
    repeat_values: Sequence[int] | None = None,
) -> set[str]:
    selected_repeats = list(repeat_values) if repeat_values is not None else list(range(repeats))
    return {
        measurement_key(implementation, phase, filter_name, query_no, repeat, config.label)
        for config in configs
        for query_no in query_nos
        for repeat in selected_repeats
    }


def validate_stage_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    implementation: str,
    phase: str,
    configs_by_filter: Mapping[str, Sequence[Config]],
    query_nos: Sequence[int],
    repeats: int,
    repeat_values: Sequence[int] | None = None,
    run_uuid: str | None = None,
) -> set[tuple[str, str]]:
    validate_raw_schema_and_dedup(rows)
    selected_repeats = set(
        int(value) for value in (
            repeat_values if repeat_values is not None else range(repeats)
        )
    )
    if not selected_repeats.issubset(set(range(repeats))):
        raise CheckpointContractError("checkpoint repeat subset is outside the stage contract")
    all_stage_rows = [row for row in rows if str(row["phase"]) == phase]
    if any(not 0 <= int(row["repeat"]) < repeats for row in all_stage_rows):
        raise CheckpointContractError(f"checkpoint phase={phase} contains foreign repeat values")
    stage_rows = [
        row for row in all_stage_rows if int(row["repeat"]) in selected_repeats
    ]
    unknown_filters = {str(row["filter_name"]) for row in stage_rows} - set(configs_by_filter)
    if unknown_filters:
        raise CheckpointContractError(
            f"checkpoint phase={phase} contains unknown filters {sorted(unknown_filters)}"
        )
    completed: set[tuple[str, str]] = set()
    for filter_name, configs in configs_by_filter.items():
        filter_rows = [
            row for row in stage_rows if str(row["filter_name"]) == filter_name
        ]
        config_by_label = {config.label: config for config in configs}
        for row in filter_rows:
            label = str(row["config_label"])
            config = config_by_label.get(label)
            if config is None:
                raise CheckpointContractError(
                    f"checkpoint {phase}/{filter_name} contains unknown config {label!r}"
                )
            try:
                fields_match = (
                    str(row["implementation"]) == implementation
                    and (run_uuid is None or str(row.get("run_uuid", "")) == run_uuid)
                    and str(row["query_split"]) == phase
                    and int(row["query_no"]) in query_nos
                    and int(row["repeat"]) in selected_repeats
                    and str(row["config_family"]) == config.family
                    and int(row["budget_rank"]) == config.budget_rank
                    and int(row["ef_search"]) == config.ef_search
                    and str(row["iterative_scan"]) == config.iterative_scan
                    and int(row["max_scan_tuples"]) == config.max_scan_tuples
                    and math.isclose(
                        float(row["scan_mem_multiplier"]),
                        config.scan_mem_multiplier,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise CheckpointContractError(
                    f"checkpoint {phase}/{filter_name}/{label} has invalid config fields"
                ) from exc
            if not fields_match:
                raise CheckpointContractError(
                    f"checkpoint {phase}/{filter_name}/{label} fields do not match the run spec"
                )
        actual = {str(row["measurement_key"]) for row in filter_rows}
        if not actual:
            continue
        expected = expected_stage_keys(
            implementation,
            phase,
            filter_name,
            configs,
            query_nos,
            repeats,
            sorted(selected_repeats),
        )
        if actual != expected:
            missing = len(expected - actual)
            extra = len(actual - expected)
            raise CheckpointContractError(
                f"checkpoint has partial/foreign {phase}/{filter_name} block: "
                f"actual={len(actual)} expected={len(expected)} missing={missing} extra={extra}"
            )
        completed.add((phase, filter_name))
    return completed


def validate_checkpoint_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_rows_by_block: Mapping[tuple[str, str], int],
) -> set[tuple[str, str]]:
    """Compatibility count-level validator; formal resume uses exact key sets."""
    validate_raw_schema_and_dedup(rows)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        block = (str(row["phase"]), str(row["filter_name"]))
        counts[block] = counts.get(block, 0) + 1
    completed: set[tuple[str, str]] = set()
    for block, count in counts.items():
        expected = expected_rows_by_block.get(block)
        if expected is None or count != expected:
            raise CheckpointContractError(
                f"checkpoint has partial or unexpected block {block}: {count}/{expected}"
            )
        completed.add(block)
    return completed


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def bootstrap_means(values: Sequence[float], samples: int, seed: int) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [float(values[0])]
    try:
        import numpy as np

        observed = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(observed), size=(samples, len(observed)))
        return np.mean(observed[indices], axis=1).tolist()
    except ModuleNotFoundError:
        rng = random.Random(seed)
        return [
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(samples)
        ]


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_queries: int,
    expected_repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    valid = [row for row in rows if str(row.get("valid", "")).lower() == "true"]
    latencies = [float(row["latency_ms"]) for row in valid]
    by_query_recall: dict[str, list[float]] = {}
    by_query_latency: dict[str, list[float]] = {}
    for row in valid:
        key = str(row["query_no"])
        by_query_recall.setdefault(key, []).append(float(row["recall_at_10"]))
        by_query_latency.setdefault(key, []).append(float(row["latency_ms"]))
    query_recalls = [statistics.fmean(values) for values in by_query_recall.values()]
    query_latencies = [statistics.fmean(values) for values in by_query_latency.values()]
    recall_bootstrap = bootstrap_means(query_recalls, bootstrap_samples, bootstrap_seed)
    latency_bootstrap = bootstrap_means(
        query_latencies, bootstrap_samples, bootstrap_seed + 1
    )
    expected_samples = expected_queries * expected_repeats
    complete = (
        len(rows) == expected_samples
        and len(valid) == expected_samples
        and len(by_query_recall) == expected_queries
        and not any(str(row.get("error", "")) for row in rows)
    )
    return {
        "queries": len(by_query_recall),
        "repeats": expected_repeats,
        "samples": len(valid),
        "errors": len(rows) - len(valid),
        "complete": complete,
        "recall_mean": statistics.fmean(query_recalls) if query_recalls else 0.0,
        "recall_lcb95": percentile(recall_bootstrap, 0.05),
        "recall_ci_low": percentile(recall_bootstrap, 0.025),
        "recall_ci_high": percentile(recall_bootstrap, 0.975),
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_ci_low_ms": percentile(latency_bootstrap, 0.025),
        "latency_ci_high_ms": percentile(latency_bootstrap, 0.975),
    }


def summarize_stage(
    raw_rows: Sequence[Mapping[str, Any]],
    implementation: str,
    phase: str,
    filters: Sequence[Mapping[str, str]],
    configs_by_filter: Mapping[str, Sequence[Config]],
    query_nos: Sequence[int],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for filter_position, filter_spec in enumerate(filters):
        name = filter_spec["filter_name"]
        for config_position, config in enumerate(configs_by_filter[name]):
            rows = [
                row
                for row in raw_rows
                if str(row["phase"]) == phase
                and str(row["filter_name"]) == name
                and str(row["config_label"]) == config.label
            ]
            seed_material = (
                f"{bootstrap_seed}|{phase}|{name}|{config.label}|"
                f"{filter_position}|{config_position}"
            )
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:12], 16)
            summaries.append(
                {
                    "implementation": implementation,
                    "phase": phase,
                    "filter_name": name,
                    "config_label": config.label,
                    "config_family": config.family,
                    "budget_rank": config.budget_rank,
                    "ef_search": config.ef_search,
                    "iterative_scan": config.iterative_scan,
                    "max_scan_tuples": config.max_scan_tuples,
                    "scan_mem_multiplier": config.scan_mem_multiplier,
                    **summarize_rows(
                        rows,
                        len(query_nos),
                        repeats,
                        bootstrap_samples,
                        seed,
                    ),
                }
            )
    return summaries


def _complete_summary(row: Mapping[str, Any]) -> bool:
    value = row.get("complete")
    return value is True or (isinstance(value, str) and value.lower() == "true")


def build_promotion_set(
    screen_summaries: Sequence[Mapping[str, Any]],
    configs: Sequence[Config],
    targets: Sequence[float],
    margin: float,
    family: str | None = None,
) -> tuple[list[Config], list[dict[str, Any]]]:
    if family is not None:
        configs = [config for config in configs if config.family == family]
        screen_summaries = [
            row for row in screen_summaries if str(row.get("config_family")) == family
        ]
        if not configs:
            raise RuntimeError(f"declared family {family!r} has no configs")
    by_label = {config.label: config for config in configs}
    summaries = {str(row["config_label"]): row for row in screen_summaries}
    if set(summaries) != set(by_label) or any(
        not _complete_summary(row) for row in summaries.values()
    ):
        raise RuntimeError("screening must complete the entire effective config grid")
    reasons: dict[str, set[str]] = {}

    def promote(label: str, reason: str) -> None:
        reasons.setdefault(label, set()).add(reason)

    for target in targets:
        threshold = max(0.0, target - margin)
        eligible = [
            row
            for row in summaries.values()
            if float(row["recall_mean"]) >= threshold
        ]
        if eligible:
            winner = min(
                eligible,
                key=lambda row: (
                    float(row["latency_mean_ms"]),
                    str(row["config_label"]),
                ),
            )
            promote(
                str(winner["config_label"]),
                f"fastest_screen_target_{target:g}_minus_margin_{margin:g}",
            )

    families = [family] if family is not None else ["off", "strict_order", "relaxed_order"]
    for selected_family in families:
        family_rows = [
            row
            for row in summaries.values()
            if str(row["config_family"]) == selected_family
        ]
        max_recall = max(
            family_rows,
            key=lambda row: (
                float(row["recall_mean"]),
                float(row["recall_lcb95"]),
                -float(row["latency_mean_ms"]),
                str(row["config_label"]),
            ),
        )
        promote(
            str(max_recall["config_label"]),
            f"family_{selected_family}_max_screen_recall",
        )

    global_max = max(
        summaries.values(),
        key=lambda row: (
            float(row["recall_mean"]),
            float(row["recall_lcb95"]),
            -float(row["latency_mean_ms"]),
            str(row["config_label"]),
        ),
    )
    promote(str(global_max["config_label"]), "global_max_screen_recall")

    if family is None:
        max_budget = family_max_budget_configs(configs)
    else:
        family_configs = [config for config in configs if config.family == family]
        max_budget = {
            family: max(
                family_configs,
                key=lambda item: (
                    item.budget_rank,
                    item.max_scan_tuples if family != "off" else 0,
                    item.scan_mem_multiplier if family != "off" else 0,
                    item.ef_search,
                    item.label,
                ),
            )
        }
    for family, config in max_budget.items():
        promote(config.label, f"family_{family}_maximum_budget_verification_boundary")

    promoted = [config for config in configs if config.label in reasons]
    proof = [
        {
            "config_label": config.label,
            "config_family": config.family,
            "promotion_reasons": "|".join(sorted(reasons[config.label])),
            "screen_recall_mean": summaries[config.label]["recall_mean"],
            "screen_recall_lcb95": summaries[config.label]["recall_lcb95"],
            "screen_latency_mean_ms": summaries[config.label]["latency_mean_ms"],
            "is_family_maximum_budget": config.label
            in {item.label for item in max_budget.values()},
        }
        for config in promoted
    ]
    return promoted, proof


def select_verified_config(
    verification_summaries: Sequence[Mapping[str, Any]],
    target: float,
    family_max_budget_labels: Mapping[str, str] | None = None,
    *,
    verified_config_labels: Sequence[str] | None = None,
    family: str | None = None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    if family is not None:
        verification_summaries = [
            row
            for row in verification_summaries
            if str(row.get("config_family", "")) == family
        ]
    required_labels = sorted(
        set(
            verified_config_labels
            or (family_max_budget_labels or {}).values()
            or [str(row.get("config_label", "")) for row in verification_summaries]
        )
    )
    by_label = {
        str(row.get("config_label", "")): row for row in verification_summaries
    }
    missing = [label for label in required_labels if label not in by_label]
    incomplete = [
        str(row.get("config_label", ""))
        for row in verification_summaries
        if str(row.get("config_label", "")) in required_labels
        and not _complete_summary(row)
    ]
    if missing or incomplete:
        return None, "incomplete_verification", {
            "qualified_configs": 0,
            "claims_unattainable": False,
            "verified_configs": required_labels,
            "missing_verified_configs": missing,
            "incomplete_promoted_configs": sorted(incomplete),
        }
    eligible = [
        dict(row)
        for row in verification_summaries
        if str(row.get("config_label", "")) in required_labels
        and _complete_summary(row)
        and float(row.get("recall_mean", 0.0)) >= target
    ]
    if eligible:
        eligible.sort(
            key=lambda row: (
                float(row.get("latency_mean_ms", math.inf)),
                str(row.get("config_label", "")),
            )
        )
        return eligible[0], "selected", {
            "qualified_configs": len(eligible),
            "claims_unattainable": False,
            "verified_configs": required_labels,
            "missing_verified_configs": [],
        }

    return None, "no_verified_config_meets_target", {
        "qualified_configs": 0,
        "claims_unattainable": False,
        "verified_configs": required_labels,
        "missing_verified_configs": [],
        "interpretation": "no completely verified config in the declared grid met the target",
    }


def select_config(
    summaries: Sequence[Mapping[str, Any]], target_recall: float
) -> tuple[dict[str, Any] | None, str]:
    """Compatibility selection helper without formal unattainable proof."""
    eligible = [
        dict(row)
        for row in summaries
        if _complete_summary(row)
        and float(row.get("recall_mean", 0.0)) >= target_recall
    ]
    if not eligible:
        return None, "unattainable_on_grid"
    eligible.sort(
        key=lambda row: (
            float(row.get("latency_mean_ms", math.inf)),
            str(row.get("config_label", "")),
        )
    )
    return eligible[0], "selected"


def heldout_final_status(
    selection_status: str,
    target: float,
    metrics: Mapping[str, Any] | None,
) -> str:
    if metrics is None:
        return (
            "incomparable_no_verified_config"
            if selection_status == "no_verified_config_meets_target"
            else "not_run_incomplete"
        )
    if not _complete_summary(metrics):
        return "incomplete_final"
    if float(metrics["recall_mean"]) < target:
        return "missed_target"
    return "confirmed"


def output_paths(
    out_dir: Path, implementation: str, tag: str, run_uuid: str | None = None
) -> dict[str, Path]:
    root = out_dir / "staging" / run_uuid if run_uuid else out_dir
    prefix = f"pgvector_upstream_overhead_control_{implementation}_{tag}"
    suffixes = {
        "raw": "raw.csv",
        "screen": "screen.csv",
        "promotion": "promotion.csv",
        "verification": "verification.csv",
        "selection": "selection.csv",
        "final": "final.csv",
        "summary": "summary.csv",
        "manifest": "manifest.json",
    }
    return {name: root / f"{prefix}_{suffix}" for name, suffix in suffixes.items()}


def normalized_args(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {
        "dry_run",
        "paths",
        "resume",
        "execution_stage",
        "final_block",
        "guc_block_audits",
    }
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in ignored
    }


def source_hashes(args: argparse.Namespace, filters: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    sql_hashes = {
        row["filter_name"]: hashlib.sha256(
            build_hybrid_sql(
                args.table,
                row["predicate"],
                args.k,
                args.candidate_validity_predicate,
            ).encode("utf-8")
        ).hexdigest()
        for row in filters
    }
    result: dict[str, Any] = {
        "runner_sha256": sha256_file(Path(__file__)),
        "filters_sha256": sha256_file(args.filters_csv),
        "truth_sha256": sha256_file(args.truth_csv),
        "hybrid_sql_sha256_by_filter": sql_hashes,
        "candidate_validity_predicate_sha256": hashlib.sha256(
            args.candidate_validity_predicate.encode("utf-8")
        ).hexdigest(),
    }
    if args.upstream_evaluation_patch:
        result["upstream_evaluation_patch_sha256"] = sha256_file(
            args.upstream_evaluation_patch
        )
    if args.config_ladder:
        result["config_ladder_sha256"] = sha256_file(args.config_ladder)
    else:
        result["config_ladder_sha256"] = sha256_json(
            [asdict(config) for config in default_config_ladder()]
        )
    result["graph_identity_sha256"] = sha256_file(args.graph_identity_json)
    return result


def load_graph_identity(
    path: Path,
    source_index: str,
    clone_index: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProvenanceGateError("graph identity must be a JSON object")
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ProvenanceGateError(
            "graph identity must contain the canonical comparison object"
        )
    required_true = (
        "same_heap",
        "entry_equal",
        "logical_equal",
        "definition_equal",
        "tuple_coverage_equal",
    )
    failed = [name for name in required_true if comparison.get(name) is not True]
    if failed:
        raise ProvenanceGateError(
            "graph identity failed required equivalence checks: " + ", ".join(failed)
        )
    if comparison.get("format") != "sqlens-hnsw-compare-v2":
        raise ProvenanceGateError("graph identity has an unsupported comparison format")
    if comparison.get("physical_equal") is not False:
        raise ProvenanceGateError(
            "graph identity must prove a physically distinct layout"
        )
    declared_source = str(payload.get("source_index", payload.get("source", "")))
    declared_clone = str(payload.get("clone_index", payload.get("clone", "")))
    if declared_source != source_index:
        raise ProvenanceGateError("graph identity source index does not match CLI")
    if declared_clone != clone_index:
        raise ProvenanceGateError("graph identity clone index does not match CLI")
    equal_digest_pairs = (
        ("left_definition_digest", "right_definition_digest"),
        ("left_tuple_coverage_digest", "right_tuple_coverage_digest"),
        ("left_logical_digest", "right_logical_digest"),
    )
    for left, right in equal_digest_pairs:
        left_value = str(comparison.get(left, ""))
        right_value = str(comparison.get(right, ""))
        if not left_value or left_value != right_value:
            raise ProvenanceGateError(
                f"graph identity has unequal or missing {left}/{right}"
            )
    left_physical = str(comparison.get("left_physical_digest", ""))
    right_physical = str(comparison.get("right_physical_digest", ""))
    if not left_physical or not right_physical or left_physical == right_physical:
        raise ProvenanceGateError("graph identity has invalid physical layout digests")
    logical = str(comparison["left_logical_digest"])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_index": source_index,
        "clone_index": clone_index,
        "same_heap": True,
        "entry_equal": True,
        "logical_equal": True,
        "definition_equal": True,
        "tuple_coverage_equal": True,
        "physical_equal": False,
        "logical_digest": logical,
        "stable_fingerprint_sha256": payload.get("stable_fingerprint_sha256"),
        "proof": payload,
    }


def index_catalog_fingerprint(cur: Any, index: str) -> dict[str, Any]:
    cur.execute(
        "SELECT c.oid::bigint, c.relfilenode::bigint, i.indrelid::bigint, "
        "i.indisvalid, i.indisready, pg_get_indexdef(c.oid) "
        "FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid "
        "WHERE c.oid=%s::regclass",
        (index,),
    )
    row = cur.fetchone()
    if not row:
        raise ProvenanceGateError(f"index identity lookup returned no row for {index}")
    oid, relfilenode, heap_oid, valid, ready, definition = row
    if not valid or not ready:
        raise ProvenanceGateError(f"index {index} is not valid and ready")
    return {
        "index": index,
        "index_oid": int(oid),
        "index_relfilenode": int(relfilenode),
        "heap_oid": int(heap_oid),
        "indexdef_sha256": hashlib.sha256(str(definition).encode("utf-8")).hexdigest(),
        "indisvalid": bool(valid),
        "indisready": bool(ready),
    }


def database_fingerprint(
    cur: Any, table: str, index: str, data_epoch: str
) -> dict[str, Any]:
    if not data_epoch:
        raise ValueError("formal database fingerprint requires a nonempty data_epoch")
    cur.execute(
        "SELECT (SELECT system_identifier::text FROM pg_control_system()), "
        "current_database(), "
        "(SELECT oid::bigint FROM pg_database WHERE datname=current_database()), "
        "current_user, "
        "inet_server_addr()::text, inet_server_port(), current_setting('server_version'), "
        "COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), '')"
    )
    (
        system_identifier,
        database,
        database_oid,
        user,
        server_address,
        server_port,
        server_version,
        vector_version,
    ) = cur.fetchone()
    cur.execute(
        f"SELECT count(*), min(id), max(id), %s::regclass::oid::bigint, "
        f"pg_relation_filenode(%s::regclass)::bigint FROM {table}",
        (table, table),
    )
    rows, minimum, maximum, table_oid, table_filenode = cur.fetchone()
    cur.execute(
        "SELECT c.oid::bigint, c.relfilenode::bigint, pg_get_indexdef(c.oid) "
        "FROM pg_class c WHERE c.oid=%s::regclass",
        (index,),
    )
    index_oid, index_filenode, indexdef = cur.fetchone()
    return {
        "system_identifier": str(system_identifier),
        "database": str(database),
        "database_oid": int(database_oid),
        "connection_user": str(user),
        "server_address": "" if server_address is None else str(server_address),
        "server_port": None if server_port is None else int(server_port),
        "server_version": str(server_version),
        "vector_extension_version": str(vector_version),
        "data_epoch": data_epoch,
        "table": table,
        "table_rows": int(rows),
        "table_min_id": int(minimum),
        "table_max_id": int(maximum),
        "table_oid": int(table_oid),
        "table_relfilenode": int(table_filenode),
        "index": index,
        "index_oid": int(index_oid),
        "index_relfilenode": int(index_filenode),
        "indexdef_sha256": hashlib.sha256(str(indexdef).encode("utf-8")).hexdigest(),
    }


def build_checkpoint_spec(
    *,
    run_uuid: str,
    base_spec: Mapping[str, Any],
    database_fingerprint: Mapping[str, Any],
    binary_provenance: Mapping[str, Any],
    settings_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if not run_uuid:
        raise CheckpointContractError("checkpoint spec requires a run UUID")
    spec: dict[str, Any] = {
        "run_uuid": run_uuid,
        "base_spec": dict(base_spec),
        "database_fingerprint": dict(database_fingerprint),
        "binary_provenance": dict(binary_provenance),
        "settings_audit": dict(settings_audit),
        "checkpoint_spec_sha256": None,
    }
    spec["checkpoint_spec_sha256"] = sha256_json(spec)
    return spec


def parse_vector_text(value: Any) -> str:
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("database returned an invalid vector text value")
    return text


def tie_aware_recall(
    result_rows: Sequence[Sequence[Any]], truth: TruthEntry, k: int
) -> float:
    if not truth.self_excluded:
        raise ValueError("truth must be self-excluded")
    accepted: set[int] = set()
    for row in result_rows[:k]:
        row_id = int(row[0])
        distance_sq = float(row[1]) ** 2
        if (
            row_id != truth.query_id
            and distance_sq <= truth.kth_distance_sq + truth.tie_tolerance
        ):
            accepted.add(row_id)
    return min(1.0, len(accepted) / float(k))


def measurement_row(
    implementation: str,
    phase: str,
    filter_name: str,
    query_no: int,
    query_id: int,
    repeat: int,
    config: Config,
    schedule_position: int,
    vector_text: str,
    truth: TruthEntry,
    cur: Any,
    sql: str,
    k: int,
    run_uuid: str = "",
    execution_stage: str = "",
    final_block: int | str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    error = ""
    result_rows: list[Sequence[Any]] = []
    try:
        cur.execute(sql, (vector_text, query_id, vector_text))
        result_rows = cur.fetchall()
        stopped = time.perf_counter()
    except Exception as exc:
        stopped = time.perf_counter()
        error = f"{exc.__class__.__name__}: {exc}"
    latency_ms = (stopped - started) * 1000.0
    recall = 0.0
    if not error:
        try:
            # Recall is deliberately offline and excluded from the latency timer.
            recall = tie_aware_recall(result_rows, truth, k)
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
    return {
        "run_uuid": run_uuid,
        "implementation": implementation,
        "execution_stage": execution_stage,
        "final_block": final_block,
        "phase": phase,
        "query_split": phase,
        "filter_name": filter_name,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "config_label": config.label,
        "config_family": config.family,
        "budget_rank": config.budget_rank,
        "ef_search": config.ef_search,
        "iterative_scan": config.iterative_scan,
        "max_scan_tuples": config.max_scan_tuples,
        "scan_mem_multiplier": config.scan_mem_multiplier,
        "schedule_position": schedule_position,
        "pair_key": pair_key(phase, filter_name, query_no, repeat),
        "measurement_key": measurement_key(
            implementation, phase, filter_name, query_no, repeat, config.label
        ),
        "latency_ms": latency_ms,
        "returned": len(result_rows),
        "result_ids": ",".join(str(int(row[0])) for row in result_rows),
        "recall_at_10": recall,
        "truth_self_excluded": truth.self_excluded,
        "valid": not error,
        "error": error,
    }


def run_stage_blocks(
    cur: Any,
    args: argparse.Namespace,
    phase: str,
    filters: Sequence[Mapping[str, str]],
    configs_by_filter: Mapping[str, Sequence[Config]],
    query_nos: Sequence[int],
    repeats: int,
    truth: Mapping[tuple[str, int], TruthEntry],
    vectors: Mapping[int, str],
    raw_rows: list[dict[str, Any]],
    completed: set[tuple[str, str]],
    repeat_values: Sequence[int] | None = None,
    final_block: int | str = "",
) -> None:
    phase_offset = {"screen": 0, "verification": 1_000_003, "final": 2_000_003}[phase]
    for filter_position, filter_spec in enumerate(filters):
        name = filter_spec["filter_name"]
        configs = list(configs_by_filter[name])
        block = (phase, name)
        if not configs or block in completed:
            continue
        sql = build_hybrid_sql(
            args.table,
            filter_spec["predicate"],
            args.k,
            args.candidate_validity_predicate,
        )
        block_rows: list[dict[str, Any]] = []
        # Re-enumerate at every atomic boundary so newly introduced extension
        # knobs cannot silently escape the stock allowlist.
        block_guc_audit = enforce_hnsw_guc_allowlist(cur)
        args.guc_block_audits.append(
            {
                "phase": phase,
                "filter_name": name,
                "actions": block_guc_audit["actions"],
                "after_sha256": sha256_json(block_guc_audit["after"]),
            }
        )
        block_no = 0
        selected_repeats = (
            list(repeat_values) if repeat_values is not None else list(range(repeats))
        )
        for repeat in selected_repeats:
            query_order = list(query_nos)
            random.Random(
                args.schedule_seed
                + phase_offset
                + filter_position * 1009
                + repeat * 104729
            ).shuffle(query_order)
            for query_no in query_order:
                config_order = balanced_order(
                    configs, block_no, args.schedule_seed + phase_offset
                )
                for position, config in enumerate(config_order, start=1):
                    configure_stock(cur, config)
                    entry = truth[(name, int(query_no))]
                    block_rows.append(
                        measurement_row(
                            args.implementation,
                            phase,
                            name,
                            int(query_no),
                            entry.query_id,
                            repeat,
                            config,
                            position,
                            vectors[entry.query_id],
                            entry,
                            cur,
                            sql,
                            args.k,
                            args.run_uuid,
                            args.execution_stage,
                            final_block,
                        )
                    )
                block_no += 1
        raw_rows.extend(block_rows)
        raw_rows.sort(key=lambda row: str(row["measurement_key"]))
        write_csv_atomic(args.paths["raw"], raw_rows, RAW_FIELDS)
        completed.add(block)


def load_checkpoint(
    paths: Mapping[str, Path], checkpoint_spec_hash: str, resume: bool
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    raw_exists = paths["raw"].exists()
    manifest_exists = paths["manifest"].exists()
    if not raw_exists and not manifest_exists:
        return [], None
    if not resume:
        raise FileExistsError("run artifacts exist; pass --resume only for the identical run spec")
    if raw_exists and not manifest_exists:
        raise CheckpointContractError("raw checkpoint exists without its manifest")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    recorded_hash = manifest.get(
        "checkpoint_spec_sha256", manifest.get("base_run_spec_hash")
    )
    if recorded_hash != checkpoint_spec_hash:
        raise CheckpointContractError("checkpoint checkpoint_spec_sha256 does not match")
    rows = read_csv(paths["raw"]) if raw_exists else []
    validate_raw_schema_and_dedup(rows)
    return rows, manifest


def resume_append_only_audit(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before_by_key = {str(row["measurement_key"]): dict(row) for row in before}
    after_by_key = {str(row["measurement_key"]): dict(row) for row in after}
    preserved = all(
        key in after_by_key and after_by_key[key] == row
        for key, row in before_by_key.items()
    )
    result = {
        "passed": preserved and len(after_by_key) >= len(before_by_key),
        "rows_before": len(before_by_key),
        "rows_after": len(after_by_key),
        "new_measurements": len(after_by_key) - len(before_by_key),
        "before_sha256": sha256_json(before_by_key),
        "preserved_before_sha256": sha256_json(
            {key: after_by_key.get(key) for key in sorted(before_by_key)}
        ),
    }
    if not result["passed"]:
        raise CheckpointContractError("resume changed or removed prior measurements")
    return result


def validate_derived_resume_hash(
    existing_manifest: Mapping[str, Any] | None,
    field: str,
    actual_hash: str,
    later_rows_exist: bool,
) -> None:
    recorded = existing_manifest.get(field) if existing_manifest else None
    if recorded is not None and recorded != actual_hash:
        raise CheckpointContractError(f"checkpoint {field} does not match recomputed evidence")
    if later_rows_exist and recorded != actual_hash:
        raise CheckpointContractError(
            f"checkpoint contains later-stage rows without matching {field}"
        )


def _config_map_for_all_filters(
    filters: Sequence[Mapping[str, str]], configs: Sequence[Config]
) -> dict[str, list[Config]]:
    return {row["filter_name"]: list(configs) for row in filters}


def _rows_for_filter(
    rows: Sequence[Mapping[str, Any]], phase: str, filter_name: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row["phase"]) == phase and str(row["filter_name"]) == filter_name
    ]


def _output_hashes(paths: Mapping[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in paths.items():
        if name != "manifest" and path.exists():
            result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def warmup_spec(
    args: argparse.Namespace,
    filters: Sequence[Mapping[str, str]],
    config: Config,
) -> dict[str, Any]:
    query_nos = list(SCREEN_QUERY_NOS[: args.warmup_queries])
    spec = {
        "query_nos": query_nos,
        "filters": [row["filter_name"] for row in filters],
        "config": asdict(config) | {"label": config.label},
        "sql_sha256_by_filter": {
            row["filter_name"]: hashlib.sha256(
                build_hybrid_sql(
                    args.table,
                    row["predicate"],
                    args.k,
                    args.candidate_validity_predicate,
                ).encode("utf-8")
            ).hexdigest()
            for row in filters
        },
        "order": "filter CSV order, then ascending warmup query_no",
    }
    return spec | {"warmup_spec_sha256": sha256_json(spec)}


def relation_prewarm_spec(args: argparse.Namespace) -> dict[str, Any]:
    spec = {
        "relations": list(getattr(args, "prewarm_relations", []) or []),
        "mode": "read",
        "fork": "main",
        "scope": "synchronous_os_cache_before_each_runner_invocation",
    }
    return spec | {"prewarm_spec_sha256": sha256_json(spec)}


def prewarm_relations(cur: Any, args: argparse.Namespace) -> dict[str, Any]:
    spec = relation_prewarm_spec(args)
    records: list[dict[str, Any]] = []
    for relation in spec["relations"]:
        cur.execute(
            "SELECT c.oid::bigint, c.relfilenode::bigint, pg_relation_size(c.oid), "
            "current_setting('block_size')::bigint "
            "FROM pg_class c WHERE c.oid = %s::regclass",
            (relation,),
        )
        row = cur.fetchone()
        if row is None:
            raise ProvenanceGateError(f"prewarm relation does not exist: {relation}")
        oid, relfilenode, relation_bytes, block_size = map(int, row)
        expected_blocks = (
            (relation_bytes + block_size - 1) // block_size if relation_bytes else 0
        )
        started = time.perf_counter()
        cur.execute(
            "SELECT pg_prewarm(%s::regclass, 'read', 'main')::bigint",
            (relation,),
        )
        warmed = cur.fetchone()
        warmed_blocks = int(warmed[0]) if warmed else -1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if warmed_blocks != expected_blocks:
            raise ProvenanceGateError(
                f"pg_prewarm block count mismatch for {relation}: "
                f"expected {expected_blocks}, got {warmed_blocks}"
            )
        records.append(
            {
                "relation": relation,
                "oid": oid,
                "relfilenode": relfilenode,
                "relation_bytes": relation_bytes,
                "block_size": block_size,
                "expected_blocks": expected_blocks,
                "warmed_blocks": warmed_blocks,
                "elapsed_ms": elapsed_ms,
            }
        )
    return {
        **spec,
        "records": records,
        "complete": len(records) == len(spec["relations"]),
        "completed_at_utc": utc_now(),
    }


def deterministic_warmup(
    cur: Any,
    args: argparse.Namespace,
    filters: Sequence[Mapping[str, str]],
    config: Config,
    truth: Mapping[tuple[str, int], TruthEntry],
    vectors: Mapping[int, str],
) -> dict[str, Any]:
    spec = warmup_spec(args, filters, config)
    rows_fetched = 0
    for filter_spec in filters:
        configure_stock(cur, config)
        sql = build_hybrid_sql(
            args.table,
            filter_spec["predicate"],
            args.k,
            args.candidate_validity_predicate,
        )
        for query_no in spec["query_nos"]:
            entry = truth[(filter_spec["filter_name"], int(query_no))]
            cur.execute(sql, (vectors[entry.query_id], entry.query_id, vectors[entry.query_id]))
            rows_fetched += len(cur.fetchall())
    return {
        **spec,
        "rows_fetched": rows_fetched,
        "completed_at_utc": utc_now(),
    }


def _legacy_monolithic_run(args: argparse.Namespace) -> dict[str, Path]:
    validate_runtime_args(args)
    split_contract = validate_split_contract()
    filters = load_filters(args.filters_csv, set(args.filter_names) or None)
    if not args.filter_names and len(filters) != 14:
        raise ValueError(f"Amazon 10M formal run requires 14 filters, got {len(filters)}")
    input_configs = load_config_ladder(args.config_ladder)
    configs, dedup_proof = effective_config_grid(input_configs)
    filter_names = {row["filter_name"] for row in filters}
    all_query_nos = SCREEN_QUERY_NOS + VERIFICATION_QUERY_NOS + FINAL_QUERY_NOS
    truth = load_truth(args.truth_csv, all_query_nos, filter_names, args.k)
    query_ids: dict[int, int] = {}
    for query_no in all_query_nos:
        ids = {truth[(name, query_no)].query_id for name in filter_names}
        if len(ids) != 1:
            raise ProvenanceGateError(
                f"query_no={query_no} maps to different query IDs across filters"
            )
        query_ids[query_no] = ids.pop()
    paths = output_paths(args.out_dir, args.implementation, args.tag)
    args.paths = paths
    hashes = source_hashes(args, filters)
    base_run_spec = {
        "args": normalized_args(args),
        "source_hashes": hashes,
        "effective_configs": [asdict(config) | {"label": config.label} for config in configs],
        "config_dedup_proof_sha256": sha256_json(dedup_proof),
        "filters": [
            {"filter_name": row["filter_name"], "predicate": row["predicate"]}
            for row in filters
        ],
        "query_ids": query_ids,
        "stages": {
            "screen": {
                "query_nos": list(SCREEN_QUERY_NOS),
                "repeats": args.screen_repeats,
                "grid": "complete effective grid",
            },
            "verification": {
                "query_nos": list(VERIFICATION_QUERY_NOS),
                "repeats": args.verification_repeats,
                "grid": "promoted configs only",
            },
            "final": {
                "query_nos": list(FINAL_QUERY_NOS),
                "repeats": args.final_repeats,
                "grid": "verification-selected configs only",
                "used_for_tuning": False,
            },
        },
    }
    base_run_spec_hash = sha256_json(base_run_spec)
    raw_rows, existing_manifest = load_checkpoint(
        paths, base_run_spec_hash, args.resume
    )
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": (
            existing_manifest.get("started_at_utc", utc_now())
            if existing_manifest
            else utc_now()
        ),
        "implementation": args.implementation,
        "base_run_spec_hash": base_run_spec_hash,
        "args": normalized_args(args),
        "source_hashes": hashes,
        "git_revision": git_revision(),
        "query_splits": split_contract,
        "config_ladder": {
            "source": str(args.config_ladder) if args.config_ladder else "deterministic_default",
            "effective_configs": [asdict(config) | {"label": config.label} for config in configs],
            "equivalence_dedup_proof": dedup_proof,
        },
        "checkpoint": {
            "path": str(paths["raw"]),
            "atomic": True,
            "complete_block": "phase/filter exact measurement-key set",
            "dedup_key": [
                "implementation",
                "phase",
                "filter_name",
                "query_no",
                "repeat",
                "config_label",
            ],
            "resume": args.resume,
        },
        "csv_schema": list(RAW_FIELDS),
    }
    atomic_write_json(paths["manifest"], manifest)

    try:
        binary = server_vector_binary_provenance(
            args.server_container, args.expected_vector_so_sha256
        )
        binary["source_tag"] = args.vector_source_tag
        binary["source_commit"] = args.vector_source_commit
        manifest["server_binary_provenance"] = binary
        atomic_write_json(paths["manifest"], manifest)

        try:
            from common_pg import pg_config_from_env
        except ImportError:
            from experiments.hybrid_vector_db.scripts.common_pg import pg_config_from_env
        import psycopg

        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("SET max_parallel_workers_per_gather = 0")
            cur.execute("SET jit = off")
            # This is the sole planner-path shortcut.  EXPLAIN must still prove
            # use of the same declared HNSW index for every filter.
            cur.execute("SET enable_seqscan = off")
            cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
            runtime = gate_implementation(
                cur,
                args.implementation,
                args.required_sqlens_build_prefix,
                args.minimum_sqlens_profile_semantics,
            )
            if args.implementation == "sqlens_disabled":
                disable_sqlens_gucs(cur)
            fingerprint = database_fingerprint(cur, args.table, args.index)
            if (
                fingerprint["table_rows"],
                fingerprint["table_min_id"],
                fingerprint["table_max_id"],
            ) != (10_000_000, 0, 9_999_999):
                raise RuntimeError(f"Amazon 10M table fingerprint mismatch: {fingerprint}")
            cur.execute(
                f"SELECT id, embedding::text FROM {args.table} "
                "WHERE id = ANY(%s::bigint[])",
                (list(query_ids.values()),),
            )
            vectors = {int(row[0]): parse_vector_text(row[1]) for row in cur.fetchall()}
            if set(vectors) != set(query_ids.values()):
                raise RuntimeError("database query-vector mapping is incomplete")
            explain_audit: dict[str, Any] = {}
            for filter_spec in filters:
                sql = build_hybrid_sql(
                    args.table,
                    filter_spec["predicate"],
                    args.k,
                    args.candidate_validity_predicate,
                )
                explain_audit[filter_spec["filter_name"]] = explain_hybrid(
                    cur, sql, vectors[query_ids[0]], query_ids[0], args.index
                )
            manifest.update(
                {
                    "runtime_provenance": runtime,
                    "database_fingerprint": fingerprint,
                    "index_fingerprint": {
                        "index": args.index,
                        "index_oid": fingerprint["index_oid"],
                        "index_relfilenode": fingerprint["index_relfilenode"],
                        "indexdef_sha256": fingerprint["indexdef_sha256"],
                    },
                    "explain_hnsw_gate": explain_audit,
                }
            )
            atomic_write_json(paths["manifest"], manifest)

            all_configs_by_filter = _config_map_for_all_filters(filters, configs)
            completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "screen",
                all_configs_by_filter,
                SCREEN_QUERY_NOS,
                args.screen_repeats,
            )
            if any(str(row["phase"]) != "screen" for row in raw_rows) and len(completed) != len(filters):
                raise CheckpointContractError(
                    "later-stage rows exist before every screen block is complete"
                )
            run_stage_blocks(
                cur,
                args,
                "screen",
                filters,
                all_configs_by_filter,
                SCREEN_QUERY_NOS,
                args.screen_repeats,
                truth,
                vectors,
                raw_rows,
                completed,
            )
            completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "screen",
                all_configs_by_filter,
                SCREEN_QUERY_NOS,
                args.screen_repeats,
            )
            if len(completed) != len(filters):
                raise CheckpointContractError("screen stage did not complete every filter block")
            screen_summaries = summarize_stage(
                raw_rows,
                args.implementation,
                "screen",
                filters,
                all_configs_by_filter,
                SCREEN_QUERY_NOS,
                args.screen_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            if any(not _complete_summary(row) for row in screen_summaries):
                raise FormalResultInvalid("screening contains failed measurements")
            write_csv_atomic(paths["screen"], screen_summaries)

            promoted_by_filter: dict[str, list[Config]] = {}
            promotion_rows: list[dict[str, Any]] = []
            promotion_spec: dict[str, list[str]] = {}
            for filter_spec in filters:
                name = filter_spec["filter_name"]
                summaries = [
                    row for row in screen_summaries if row["filter_name"] == name
                ]
                promoted, proof = build_promotion_set(
                    summaries, configs, args.target_recalls, args.promotion_margin
                )
                promoted_by_filter[name] = promoted
                promotion_spec[name] = [config.label for config in promoted]
                promotion_rows.extend(
                    {
                        "implementation": args.implementation,
                        "filter_name": name,
                        "promotion_margin": args.promotion_margin,
                        **row,
                    }
                    for row in proof
                )
            promotion_hash = sha256_json(promotion_spec)
            later_rows = any(
                str(row["phase"]) in {"verification", "final"} for row in raw_rows
            )
            validate_derived_resume_hash(
                existing_manifest, "promotion_set_sha256", promotion_hash, later_rows
            )
            manifest["promotion_set"] = promotion_spec
            manifest["promotion_set_sha256"] = promotion_hash
            manifest["tuning_run_spec_hash"] = sha256_json(
                {
                    "base_run_spec_hash": base_run_spec_hash,
                    "promotion_set_sha256": promotion_hash,
                }
            )
            write_csv_atomic(paths["promotion"], promotion_rows)
            atomic_write_json(paths["manifest"], manifest)

            verification_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "verification",
                promoted_by_filter,
                VERIFICATION_QUERY_NOS,
                args.verification_repeats,
            )
            if any(str(row["phase"]) == "final" for row in raw_rows) and len(verification_completed) != len(filters):
                raise CheckpointContractError(
                    "final rows exist before every verification block is complete"
                )
            run_stage_blocks(
                cur,
                args,
                "verification",
                filters,
                promoted_by_filter,
                VERIFICATION_QUERY_NOS,
                args.verification_repeats,
                truth,
                vectors,
                raw_rows,
                verification_completed,
            )
            verification_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "verification",
                promoted_by_filter,
                VERIFICATION_QUERY_NOS,
                args.verification_repeats,
            )
            if len(verification_completed) != len(filters):
                raise CheckpointContractError(
                    "verification stage did not complete every filter block"
                )
            verification_summaries = summarize_stage(
                raw_rows,
                args.implementation,
                "verification",
                filters,
                promoted_by_filter,
                VERIFICATION_QUERY_NOS,
                args.verification_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed + 100_000,
            )
            write_csv_atomic(paths["verification"], verification_summaries)

            family_max = family_max_budget_configs(configs)
            max_budget_labels = {
                family: config.label for family, config in family_max.items()
            }
            selection_rows: list[dict[str, Any]] = []
            selected_by_filter: dict[str, list[Config]] = {}
            selection_spec: dict[str, dict[str, str]] = {}
            config_by_label = {config.label: config for config in configs}
            for filter_spec in filters:
                name = filter_spec["filter_name"]
                summaries = [
                    row
                    for row in verification_summaries
                    if row["filter_name"] == name
                ]
                selected_by_filter[name] = []
                selection_spec[name] = {}
                for target in args.target_recalls:
                    selected, status, proof = select_verified_config(
                        summaries, target, max_budget_labels
                    )
                    label = str(selected["config_label"]) if selected else ""
                    selection_spec[name][format(target, "g")] = label or status
                    if label and config_by_label[label] not in selected_by_filter[name]:
                        selected_by_filter[name].append(config_by_label[label])
                    selection_rows.append(
                        {
                            "implementation": args.implementation,
                            "filter_name": name,
                            "target_recall": target,
                            "selection_status": status,
                            "selected_config_label": label,
                            "verification_recall_mean": (
                                selected["recall_mean"] if selected else ""
                            ),
                            "verification_recall_lcb95": (
                                selected["recall_lcb95"] if selected else ""
                            ),
                            "verification_latency_mean_ms": (
                                selected["latency_mean_ms"] if selected else ""
                            ),
                            "verification_latency_p95_ms": (
                                selected["latency_p95_ms"] if selected else ""
                            ),
                            "verification_latency_p99_ms": (
                                selected["latency_p99_ms"] if selected else ""
                            ),
                            "verification_proof_json": json.dumps(
                                proof, sort_keys=True, separators=(",", ":")
                            ),
                        }
                    )
            selection_hash = sha256_json(selection_spec)
            validate_derived_resume_hash(
                existing_manifest,
                "target_selection_sha256",
                selection_hash,
                any(str(row["phase"]) == "final" for row in raw_rows),
            )
            manifest["target_selection"] = selection_spec
            manifest["target_selection_sha256"] = selection_hash
            write_csv_atomic(paths["selection"], selection_rows)
            atomic_write_json(paths["manifest"], manifest)

            final_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "final",
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
            )
            run_stage_blocks(
                cur,
                args,
                "final",
                filters,
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                truth,
                vectors,
                raw_rows,
                final_completed,
            )
            final_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "final",
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
            )
            expected_final_blocks = {
                ("final", name)
                for name, selected in selected_by_filter.items()
                if selected
            }
            if final_completed != expected_final_blocks:
                raise CheckpointContractError("final selected-config blocks are incomplete")
            final_summaries = summarize_stage(
                raw_rows,
                args.implementation,
                "final",
                filters,
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed + 200_000,
            )
            final_by_key = {
                (row["filter_name"], row["config_label"]): row
                for row in final_summaries
            }
            target_summaries: list[dict[str, Any]] = []
            final_misses: list[dict[str, Any]] = []
            incomplete_selections = 0
            for selection in selection_rows:
                label = str(selection["selected_config_label"])
                target = float(selection["target_recall"])
                metrics = final_by_key.get((selection["filter_name"], label)) if label else None
                if selection["selection_status"] == "incomplete_verification":
                    incomplete_selections += 1
                heldout_status = heldout_final_status(
                    str(selection["selection_status"]), target, metrics
                )
                if heldout_status != "confirmed":
                    final_misses.append(
                        {
                            "filter_name": selection["filter_name"],
                            "target_recall": target,
                            "config_label": label,
                            "heldout_status": heldout_status,
                            "final_recall_lcb95": (
                                metrics["recall_lcb95"] if metrics else None
                            ),
                        }
                    )
                target_summaries.append(
                    {
                        **selection,
                        "heldout_final_status": heldout_status,
                        "final_complete": metrics["complete"] if metrics else False,
                        "final_recall_mean": metrics["recall_mean"] if metrics else "",
                        "final_recall_lcb95": metrics["recall_lcb95"] if metrics else "",
                        "final_recall_ci_low": metrics["recall_ci_low"] if metrics else "",
                        "final_recall_ci_high": metrics["recall_ci_high"] if metrics else "",
                        "final_latency_mean_ms": (
                            metrics["latency_mean_ms"] if metrics else ""
                        ),
                        "final_latency_p95_ms": (
                            metrics["latency_p95_ms"] if metrics else ""
                        ),
                        "final_latency_p99_ms": (
                            metrics["latency_p99_ms"] if metrics else ""
                        ),
                        "final_latency_ci_low_ms": (
                            metrics["latency_ci_low_ms"] if metrics else ""
                        ),
                        "final_latency_ci_high_ms": (
                            metrics["latency_ci_high_ms"] if metrics else ""
                        ),
                    }
                )
            write_csv_atomic(
                paths["final"],
                [row for row in raw_rows if str(row["phase"]) == "final"],
                RAW_FIELDS,
            )
            write_csv_atomic(paths["summary"], target_summaries)

            planned_counts = {
                "screen": len(configs)
                * len(filters)
                * len(SCREEN_QUERY_NOS)
                * args.screen_repeats,
                "verification": sum(len(items) for items in promoted_by_filter.values())
                * len(VERIFICATION_QUERY_NOS)
                * args.verification_repeats,
                "final": sum(len(items) for items in selected_by_filter.values())
                * len(FINAL_QUERY_NOS)
                * args.final_repeats,
            }
            planned_counts["total"] = sum(planned_counts.values())
            artifact_valid = not final_misses and incomplete_selections == 0
            manifest.update(
                {
                    "status": "valid" if artifact_valid else "invalid",
                    "artifact_valid": artifact_valid,
                    "finished_at_utc": utc_now(),
                    "planned_query_counts": planned_counts,
                    "promotion_proof_rows": len(promotion_rows),
                    "target_selection_rows": len(selection_rows),
                    "heldout_final_misses": final_misses,
                    "incomplete_verification_targets": incomplete_selections,
                    "outputs": {name: str(path) for name, path in paths.items()},
                }
            )
            manifest["output_hashes"] = _output_hashes(paths)
            atomic_write_json(paths["manifest"], manifest)
            if not artifact_valid:
                raise FormalResultInvalid(
                    f"formal artifact invalid: final_misses={len(final_misses)}, "
                    f"incomplete_verification_targets={incomplete_selections}"
                )
            return paths
    except Exception as exc:
        manifest.update(
            {
                "status": "invalid",
                "artifact_valid": False,
                "finished_at_utc": utc_now(),
                "fatal_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        atomic_write_json(paths["manifest"], manifest)
        raise


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run one staging arm phase; publication is owned by the controller finalizer."""
    validate_runtime_args(args)
    split_contract = validate_split_contract()
    filters = load_filters(args.filters_csv, set(args.filter_names) or None)
    formal_design = validate_formal_design(filters, args.target_recalls, args.formal_family)
    input_configs = load_config_ladder(args.config_ladder, args.max_ef_search)
    all_configs, dedup_proof = effective_config_grid(input_configs)
    configs = [config for config in all_configs if config.family == args.formal_family]
    if not configs:
        raise ValueError(f"config ladder has no configs for formal family {args.formal_family}")
    if max(config.ef_search for config in configs) != args.max_ef_search:
        raise ProvenanceGateError(
            "formal config ladder does not exercise the declared max_ef_search ceiling"
        )

    filter_names = {row["filter_name"] for row in filters}
    all_query_nos = SCREEN_QUERY_NOS + VERIFICATION_QUERY_NOS + FINAL_QUERY_NOS
    truth = load_truth(
        args.truth_csv,
        all_query_nos,
        filter_names,
        args.k,
        args.candidate_validity_predicate,
    )
    query_ids: dict[int, int] = {}
    for query_no in all_query_nos:
        ids = {truth[(name, query_no)].query_id for name in filter_names}
        if len(ids) != 1:
            raise ProvenanceGateError(
                f"query_no={query_no} maps to different query IDs across filters"
            )
        query_ids[query_no] = ids.pop()
    paths = output_paths(args.out_dir, args.implementation, args.tag, args.run_uuid)
    args.paths = paths
    args.guc_block_audits = []
    hashes = source_hashes(args, filters)
    dirty = source_tree_provenance(
        args.vector_source_repo, args.vector_source_commit
    )
    ceiling_provenance = (
        upstream_parameter_ceiling_provenance(
            args.vector_source_repo,
            args.upstream_evaluation_patch,
            args.max_ef_search,
        )
        if args.implementation == "official"
        else {
            "mode": "sqlens_disabled_binary_native_parameter_range",
            "max_ef_search": args.max_ef_search,
            "patch_applied": False,
        }
    )
    selected_config_records = [
        asdict(config) | {"label": config.label} for config in configs
    ]
    schedule_contract = {
        "schedule_seed": args.schedule_seed,
        "screen_query_nos": list(SCREEN_QUERY_NOS),
        "verification_query_nos": list(VERIFICATION_QUERY_NOS),
        "final_query_nos": list(FINAL_QUERY_NOS),
        "screen_repeats": args.screen_repeats,
        "verification_repeats": args.verification_repeats,
        "final_repeats": args.final_repeats,
        "final_blocks": 2,
        "final_repeat_partition": "contiguous equal halves",
        "balanced_config_order": "seeded cyclic rotation per query/repeat block",
        "warmup_spec_sha256": warmup_spec(args, filters, configs[0])["warmup_spec_sha256"],
        "prewarm_spec_sha256": relation_prewarm_spec(args)["prewarm_spec_sha256"],
        "prewarm_relations": list(getattr(args, "prewarm_relations", []) or []),
        "prewarm_mode": "read",
    }
    base_run_spec = {
        "args": normalized_args(args),
        "source_hashes": hashes,
        "vector_source_tree_provenance": {
            **dirty,
            "source_tag": args.vector_source_tag,
            "build_recipe": args.vector_build_recipe,
            "compiler_flags": args.vector_compiler_flags,
        },
        "formal_design": formal_design,
        "effective_configs": selected_config_records,
        "config_dedup_proof_sha256": sha256_json(dedup_proof),
        "filters": [
            {"filter_name": row["filter_name"], "predicate": row["predicate"]}
            for row in filters
        ],
        "candidate_validity_predicate": args.candidate_validity_predicate,
        "parameter_ceiling_provenance": ceiling_provenance,
        "query_ids": query_ids,
        "query_splits": split_contract,
        "schedule_contract": schedule_contract,
    }
    base_run_spec_hash = sha256_json(base_run_spec)
    manifest: dict[str, Any] | None = None

    binary = server_vector_binary_provenance(
        args.server_container, args.expected_vector_so_sha256
    )
    binary["source_tag"] = args.vector_source_tag
    binary["source_commit"] = args.vector_source_commit
    source_provenance = {
        "source_tag": args.vector_source_tag,
        "source_commit": args.vector_source_commit,
        "build_recipe": args.vector_build_recipe,
        "compiler_flags": args.vector_compiler_flags,
        "dirty_diff_sha256": dirty["dirty_diff_sha256"],
        "dirty": dirty["dirty"],
        "dirty_diff_method": dirty["method"],
        "source_tree": dirty["source_tree"],
        "git_revision": git_revision(),
        "server_image_id": binary["server_image_id"],
        "parameter_ceiling_provenance": ceiling_provenance,
    }

    try:
        try:
            from common_pg import pg_config_from_env
        except ImportError:
            from experiments.hybrid_vector_db.scripts.common_pg import pg_config_from_env
        import psycopg

        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            planner_settings = {
                "max_parallel_workers_per_gather": "0",
                "jit": "off",
                "enable_seqscan": "off",
                "statement_timeout_ms": int(args.statement_timeout_ms),
            }
            cur.execute("SET max_parallel_workers_per_gather = 0")
            cur.execute("SET jit = off")
            cur.execute("SET enable_seqscan = off")
            cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
            runtime = gate_implementation(
                cur,
                args.implementation,
                args.required_sqlens_build_prefix,
                args.minimum_sqlens_profile_semantics,
            )
            guc_audit = enforce_hnsw_guc_allowlist(cur)
            fingerprint = database_fingerprint(
                cur, args.table, args.index, args.data_epoch
            )
            if (
                fingerprint["table_rows"],
                fingerprint["table_min_id"],
                fingerprint["table_max_id"],
            ) != (10_000_000, 0, 9_999_999):
                raise RuntimeError(f"Amazon 10M table fingerprint mismatch: {fingerprint}")
            graph_identity = load_graph_identity(
                args.graph_identity_json, args.source_index, args.clone_index
            )
            source_index_fingerprint = index_catalog_fingerprint(
                cur, args.source_index
            )
            clone_index_fingerprint = index_catalog_fingerprint(cur, args.clone_index)
            if (
                source_index_fingerprint["heap_oid"] != fingerprint["table_oid"]
                or clone_index_fingerprint["heap_oid"] != fingerprint["table_oid"]
            ):
                raise ProvenanceGateError(
                    "source and clone graph indexes must belong to the bound table"
                )
            fingerprint["source_clone_graph_identity"] = {
                "proof": graph_identity,
                "source_index": source_index_fingerprint,
                "clone_index": clone_index_fingerprint,
            }
            settings_audit = {
                "planner_settings": planner_settings,
                "hnsw_guc_audit": guc_audit,
            }
            checkpoint_spec = build_checkpoint_spec(
                run_uuid=args.run_uuid,
                base_spec=base_run_spec,
                database_fingerprint=fingerprint,
                binary_provenance=binary,
                settings_audit=settings_audit,
            )
            checkpoint_hash = str(checkpoint_spec["checkpoint_spec_sha256"])
            raw_rows, existing_manifest = load_checkpoint(
                paths, checkpoint_hash, args.resume
            )
            checkpoint_rows_before = [dict(row) for row in raw_rows]
            if existing_manifest and (
                existing_manifest.get("run_uuid") != args.run_uuid
                or existing_manifest.get("implementation") != args.implementation
            ):
                raise CheckpointContractError(
                    "checkpoint arm identity does not match run UUID/implementation"
                )
            manifest = dict(existing_manifest or {})
            manifest.pop("fatal_error", None)
            manifest.update(
                {
                    "schema_version": 2,
                    "status": "running",
                    "artifact_valid": False,
                    "started_at_utc": manifest.get("started_at_utc", utc_now()),
                    "last_invocation_started_at_utc": utc_now(),
                    "run_uuid": args.run_uuid,
                    "implementation": args.implementation,
                    "execution_stage": args.execution_stage,
                    "checkpoint_spec_sha256": checkpoint_hash,
                    "base_run_spec_hash": base_run_spec_hash,
                    "checkpoint_spec": checkpoint_spec,
                    "args": normalized_args(args),
                    "source_hashes": hashes,
                    "source_provenance": source_provenance,
                    "server_binary_provenance": binary,
                    "runtime_provenance": runtime,
                    "database_fingerprint": fingerprint,
                    "settings_audit": settings_audit,
                    "formal_design": formal_design,
                    "query_splits": split_contract,
                    "schedule_contract": schedule_contract,
                    "config_ladder": {
                        "source": (
                            str(args.config_ladder)
                            if args.config_ladder
                            else "deterministic_default"
                        ),
                        "formal_family": args.formal_family,
                        "effective_configs": selected_config_records,
                        "excluded_exploratory_families": sorted(
                            {config.family for config in all_configs}
                            - {args.formal_family}
                        ),
                        "equivalence_dedup_proof": dedup_proof,
                    },
                    "checkpoint": {
                        "path": str(paths["raw"]),
                        "atomic": True,
                        "constructed_after_database_fingerprint": True,
                        "complete_block": (
                            "phase/filter exact measurement-key set; final additionally "
                            "partitions repeats into two complete blocks"
                        ),
                        "resume_cross_run_pairing": False,
                        "resume": args.resume,
                    },
                    "csv_schema": list(RAW_FIELDS),
                }
            )
            atomic_write_json(paths["manifest"], manifest)

            cur.execute(
                f"SELECT id, embedding::text FROM {args.table} "
                "WHERE id = ANY(%s::bigint[])",
                (list(query_ids.values()),),
            )
            vectors = {
                int(row[0]): parse_vector_text(row[1]) for row in cur.fetchall()
            }
            if set(vectors) != set(query_ids.values()):
                raise RuntimeError("database query-vector mapping is incomplete")
            prewarm_audit = prewarm_relations(cur, args)
            explain_audit: dict[str, Any] = {}
            for filter_spec in filters:
                sql = build_hybrid_sql(
                    args.table,
                    filter_spec["predicate"],
                    args.k,
                    args.candidate_validity_predicate,
                )
                explain_audit[filter_spec["filter_name"]] = explain_hybrid(
                    cur, sql, vectors[query_ids[0]], query_ids[0], args.index
                )
            warmup_audit = deterministic_warmup(
                cur, args, filters, configs[0], truth, vectors
            )
            post_warmup_gucs = enforce_hnsw_guc_allowlist(cur)
            manifest["explain_hnsw_gate"] = explain_audit
            manifest.setdefault("prewarm_invocations", []).append(
                {
                    "execution_stage": args.execution_stage,
                    "final_block": args.final_block,
                    **prewarm_audit,
                }
            )
            manifest.setdefault("warmup_invocations", []).append(
                {
                    "execution_stage": args.execution_stage,
                    "final_block": args.final_block,
                    **warmup_audit,
                }
            )
            manifest["post_warmup_hnsw_guc_audit"] = post_warmup_gucs
            atomic_write_json(paths["manifest"], manifest)

            all_configs_by_filter = _config_map_for_all_filters(filters, configs)
            screen_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "screen",
                all_configs_by_filter,
                SCREEN_QUERY_NOS,
                args.screen_repeats,
                run_uuid=args.run_uuid,
            )
            if args.execution_stage == "calibration":
                if any(str(row["phase"]) == "final" for row in raw_rows):
                    raise CheckpointContractError(
                        "calibration invocation cannot resume after final rows exist"
                    )
                run_stage_blocks(
                    cur,
                    args,
                    "screen",
                    filters,
                    all_configs_by_filter,
                    SCREEN_QUERY_NOS,
                    args.screen_repeats,
                    truth,
                    vectors,
                    raw_rows,
                    screen_completed,
                )
                screen_completed = validate_stage_checkpoint(
                    raw_rows,
                    args.implementation,
                    "screen",
                    all_configs_by_filter,
                    SCREEN_QUERY_NOS,
                    args.screen_repeats,
                    run_uuid=args.run_uuid,
                )
            if len(screen_completed) != len(filters):
                raise CheckpointContractError(
                    "final invocation requires every calibration screen block"
                )
            screen_summaries = summarize_stage(
                raw_rows,
                args.implementation,
                "screen",
                filters,
                all_configs_by_filter,
                SCREEN_QUERY_NOS,
                args.screen_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            if any(not _complete_summary(row) for row in screen_summaries):
                raise FormalResultInvalid("screening contains failed measurements")
            write_csv_atomic(paths["screen"], screen_summaries)

            promoted_by_filter: dict[str, list[Config]] = {}
            promotion_rows: list[dict[str, Any]] = []
            promotion_spec: dict[str, list[str]] = {}
            for filter_spec in filters:
                name = filter_spec["filter_name"]
                summaries = [
                    row for row in screen_summaries if row["filter_name"] == name
                ]
                promoted, proof = build_promotion_set(
                    summaries,
                    configs,
                    args.target_recalls,
                    args.promotion_margin,
                    family=args.formal_family,
                )
                promoted_by_filter[name] = promoted
                promotion_spec[name] = [config.label for config in promoted]
                promotion_rows.extend(
                    {
                        "implementation": args.implementation,
                        "filter_name": name,
                        "formal_family": args.formal_family,
                        "promotion_margin": args.promotion_margin,
                        **row,
                    }
                    for row in proof
                )
            promotion_hash = sha256_json(promotion_spec)
            validate_derived_resume_hash(
                existing_manifest,
                "promotion_set_sha256",
                promotion_hash,
                any(str(row["phase"]) in {"verification", "final"} for row in raw_rows),
            )
            manifest["promotion_set"] = promotion_spec
            manifest["promotion_set_sha256"] = promotion_hash
            write_csv_atomic(paths["promotion"], promotion_rows)

            verification_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "verification",
                promoted_by_filter,
                VERIFICATION_QUERY_NOS,
                args.verification_repeats,
                run_uuid=args.run_uuid,
            )
            if args.execution_stage == "calibration":
                run_stage_blocks(
                    cur,
                    args,
                    "verification",
                    filters,
                    promoted_by_filter,
                    VERIFICATION_QUERY_NOS,
                    args.verification_repeats,
                    truth,
                    vectors,
                    raw_rows,
                    verification_completed,
                )
                verification_completed = validate_stage_checkpoint(
                    raw_rows,
                    args.implementation,
                    "verification",
                    promoted_by_filter,
                    VERIFICATION_QUERY_NOS,
                    args.verification_repeats,
                    run_uuid=args.run_uuid,
                )
            if len(verification_completed) != len(filters):
                raise CheckpointContractError(
                    "final invocation requires every calibration verification block"
                )
            verification_summaries = summarize_stage(
                raw_rows,
                args.implementation,
                "verification",
                filters,
                promoted_by_filter,
                VERIFICATION_QUERY_NOS,
                args.verification_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed + 100_000,
            )
            write_csv_atomic(paths["verification"], verification_summaries)

            selection_rows: list[dict[str, Any]] = []
            selected_by_filter: dict[str, list[Config]] = {}
            selection_spec: dict[str, dict[str, dict[str, str]]] = {}
            config_by_label = {config.label: config for config in configs}
            incomplete_selections = 0
            for filter_spec in filters:
                name = filter_spec["filter_name"]
                summaries = [
                    row
                    for row in verification_summaries
                    if row["filter_name"] == name
                ]
                verified_labels = [config.label for config in promoted_by_filter[name]]
                selected_by_filter[name] = []
                selection_spec[name] = {}
                for target in args.target_recalls:
                    selected, status, proof = select_verified_config(
                        summaries,
                        target,
                        verified_config_labels=verified_labels,
                        family=args.formal_family,
                    )
                    label = str(selected["config_label"]) if selected else ""
                    if status == "incomplete_verification":
                        incomplete_selections += 1
                    selection_spec[name][format(float(target), "g")] = {
                        "status": status,
                        "config_label": label,
                        "formal_family": args.formal_family,
                    }
                    if label and config_by_label[label] not in selected_by_filter[name]:
                        selected_by_filter[name].append(config_by_label[label])
                    selection_rows.append(
                        {
                            "implementation": args.implementation,
                            "filter_name": name,
                            "formal_family": args.formal_family,
                            "target_recall": target,
                            "selection_status": status,
                            "selected_config_label": label,
                            "verification_recall_mean": (
                                selected["recall_mean"] if selected else ""
                            ),
                            "verification_recall_lcb95": (
                                selected["recall_lcb95"] if selected else ""
                            ),
                            "verification_latency_mean_ms": (
                                selected["latency_mean_ms"] if selected else ""
                            ),
                            "verification_proof_json": json.dumps(
                                proof, sort_keys=True, separators=(",", ":")
                            ),
                        }
                    )
            selection_hash = sha256_json(selection_spec)
            validate_derived_resume_hash(
                existing_manifest,
                "target_selection_sha256",
                selection_hash,
                any(str(row["phase"]) == "final" for row in raw_rows),
            )
            manifest["target_selection"] = selection_spec
            manifest["target_selection_sha256"] = selection_hash
            manifest["guc_block_audits"] = [
                *manifest.get("guc_block_audits", []),
                *args.guc_block_audits,
            ]
            write_csv_atomic(paths["selection"], selection_rows)

            if args.execution_stage == "calibration":
                manifest.update(
                    {
                        "status": "calibration_complete",
                        "artifact_valid": False,
                        "last_invocation_finished_at_utc": utc_now(),
                        "incomplete_verification_targets": incomplete_selections,
                        "outputs": {name: str(path) for name, path in paths.items()},
                        "resume_append_only_audit": resume_append_only_audit(
                            checkpoint_rows_before, raw_rows
                        ),
                    }
                )
                manifest["output_hashes"] = _output_hashes(paths)
                atomic_write_json(paths["manifest"], manifest)
                return paths

            if existing_manifest is None:
                raise CheckpointContractError(
                    "final stage requires a completed calibration manifest"
                )
            half = args.final_repeats // 2
            repeat_values = (
                list(range(0, half))
                if args.final_block == 0
                else list(range(half, args.final_repeats))
            )
            final_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "final",
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                repeat_values=repeat_values,
                run_uuid=args.run_uuid,
            )
            run_stage_blocks(
                cur,
                args,
                "final",
                filters,
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                truth,
                vectors,
                raw_rows,
                final_completed,
                repeat_values=repeat_values,
                final_block=args.final_block,
            )
            final_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "final",
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                repeat_values=repeat_values,
                run_uuid=args.run_uuid,
            )
            expected_selected_blocks = {
                ("final", name)
                for name, selected in selected_by_filter.items()
                if selected
            }
            if final_completed != expected_selected_blocks:
                raise CheckpointContractError(
                    f"final block {args.final_block} selected-config rows are incomplete"
                )
            completed_invocations = set(
                int(value) for value in manifest.get("completed_final_blocks", [])
            )
            completed_invocations.add(int(args.final_block))
            manifest["completed_final_blocks"] = sorted(completed_invocations)

            if completed_invocations != {0, 1}:
                manifest.update(
                    {
                        "status": "final_in_progress",
                        "artifact_valid": False,
                        "last_invocation_finished_at_utc": utc_now(),
                        "outputs": {name: str(path) for name, path in paths.items()},
                        "resume_append_only_audit": resume_append_only_audit(
                            checkpoint_rows_before, raw_rows
                        ),
                    }
                )
                manifest["output_hashes"] = _output_hashes(paths)
                atomic_write_json(paths["manifest"], manifest)
                return paths

            full_final_completed = validate_stage_checkpoint(
                raw_rows,
                args.implementation,
                "final",
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                run_uuid=args.run_uuid,
            )
            if full_final_completed != expected_selected_blocks:
                raise CheckpointContractError("combined AB/BA final blocks are incomplete")
            final_summaries = summarize_stage(
                raw_rows,
                args.implementation,
                "final",
                filters,
                selected_by_filter,
                FINAL_QUERY_NOS,
                args.final_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed + 200_000,
            )
            final_by_key = {
                (row["filter_name"], row["config_label"]): row
                for row in final_summaries
            }
            target_summaries: list[dict[str, Any]] = []
            final_misses: list[dict[str, Any]] = []
            for selection in selection_rows:
                label = str(selection["selected_config_label"])
                target = float(selection["target_recall"])
                metrics = (
                    final_by_key.get((selection["filter_name"], label))
                    if label
                    else None
                )
                heldout_status = heldout_final_status(
                    str(selection["selection_status"]), target, metrics
                )
                if label and heldout_status != "confirmed":
                    final_misses.append(
                        {
                            "filter_name": selection["filter_name"],
                            "target_recall": target,
                            "config_label": label,
                            "heldout_status": heldout_status,
                            "final_recall_lcb95": (
                                metrics["recall_lcb95"] if metrics else None
                            ),
                        }
                    )
                target_summaries.append(
                    {
                        **selection,
                        "heldout_final_status": heldout_status,
                        "final_complete": metrics["complete"] if metrics else False,
                        "final_recall_mean": metrics["recall_mean"] if metrics else "",
                        "final_recall_lcb95": metrics["recall_lcb95"] if metrics else "",
                        "final_latency_mean_ms": (
                            metrics["latency_mean_ms"] if metrics else ""
                        ),
                        "final_latency_p95_ms": (
                            metrics["latency_p95_ms"] if metrics else ""
                        ),
                        "final_latency_p99_ms": (
                            metrics["latency_p99_ms"] if metrics else ""
                        ),
                    }
                )
            write_csv_atomic(
                paths["final"],
                [row for row in raw_rows if str(row["phase"]) == "final"],
                RAW_FIELDS,
            )
            write_csv_atomic(paths["summary"], target_summaries)
            all_targets_selected = all(
                str(row["selection_status"]) == "selected" and bool(row["selected_config_label"])
                for row in selection_rows
            )
            artifact_valid = (
                len(selection_rows) == 42
                and len(target_summaries) == 42
                and all_targets_selected
                and not final_misses
                and incomplete_selections == 0
            )
            manifest.update(
                {
                    "status": "arm_ready" if artifact_valid else "staging_unconfirmed",
                    "artifact_valid": artifact_valid,
                    "finished_at_utc": utc_now(),
                    "last_invocation_finished_at_utc": utc_now(),
                    "heldout_final_misses": final_misses,
                    "incomplete_verification_targets": incomplete_selections,
                    "target_result_rows": len(target_summaries),
                    "resume_append_only_audit": resume_append_only_audit(
                        checkpoint_rows_before, raw_rows
                    ),
                    "outputs": {name: str(path) for name, path in paths.items()},
                }
            )
            manifest["output_hashes"] = _output_hashes(paths)
            atomic_write_json(paths["manifest"], manifest)
            return paths
    except Exception as exc:
        if manifest is not None:
            manifest.update(
                {
                    "status": "invalid",
                    "artifact_valid": False,
                    "last_invocation_finished_at_utc": utc_now(),
                    "fatal_error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            atomic_write_json(paths["manifest"], manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage matched-recall stock HNSW control for official pgvector "
            "and SQLens-disabled binaries."
        )
    )
    parser.add_argument(
        "--implementation", required=True, choices=("official", "sqlens_disabled")
    )
    parser.add_argument("--run-uuid", default="")
    parser.add_argument(
        "--execution-stage", choices=("calibration", "final"), default="calibration"
    )
    parser.add_argument("--final-block", type=int, choices=(0, 1))
    parser.add_argument("--formal-family", choices=FORMAL_FAMILIES, default="off")
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--config-ladder", type=Path)
    parser.add_argument(
        "--max-ef-search",
        type=positive_int,
        default=UPSTREAM_MAX_EF_SEARCH,
        help="Provenance-gated binary ceiling; formal values are 1000, 10000, or 100000.",
    )
    parser.add_argument(
        "--upstream-evaluation-patch",
        type=Path,
        help="Canonical two-line upstream patch required for an extended official ceiling.",
    )
    parser.add_argument(
        "--candidate-validity-predicate",
        default="",
        help="Global SQL predicate implied by the partial HNSW index and exact truth.",
    )
    parser.add_argument("--table", type=validate_identifier, default=DEFAULT_TABLE)
    parser.add_argument("--index", type=validate_identifier, default=DEFAULT_INDEX)
    parser.add_argument("--source-index", type=validate_identifier, default=DEFAULT_INDEX)
    parser.add_argument("--clone-index", type=validate_identifier)
    parser.add_argument("--graph-identity-json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="20260718")
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--target-recalls", type=target_recalls, default=[0.90, 0.95, 0.99])
    parser.add_argument("--promotion-margin", type=float, default=0.02)
    parser.add_argument("--screen-repeats", type=positive_int, default=1)
    parser.add_argument("--verification-repeats", type=positive_int, default=2)
    parser.add_argument("--final-repeats", type=positive_int, default=6)
    parser.add_argument("--warmup-queries", type=positive_int, default=5)
    parser.add_argument(
        "--prewarm-relation",
        dest="prewarm_relations",
        action="append",
        type=validate_identifier,
        default=[],
        help="Relation synchronously read with pg_prewarm before each invocation; repeatable.",
    )
    parser.add_argument("--bootstrap-samples", type=positive_int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--statement-timeout-ms", type=nonnegative_int, default=300_000)
    parser.add_argument("--server-container", type=validate_container_name)
    parser.add_argument("--expected-vector-so-sha256", type=validate_sha256)
    parser.add_argument("--vector-source-tag", default="")
    parser.add_argument("--vector-source-commit", default="")
    parser.add_argument("--vector-source-repo", type=Path)
    parser.add_argument("--vector-build-recipe", default="")
    parser.add_argument("--vector-compiler-flags", default="")
    parser.add_argument("--data-epoch", default="")
    parser.add_argument(
        "--required-sqlens-build-prefix", default=DEFAULT_SQLENS_BUILD_PREFIX
    )
    parser.add_argument(
        "--minimum-sqlens-profile-semantics",
        type=float,
        default=DEFAULT_SQLENS_PROFILE_SEMANTICS,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic contract without touching files, Docker, or PostgreSQL.",
    )
    return parser


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    default_configs, proof = effective_config_grid(default_config_ladder())
    formal_configs = [
        config for config in default_configs if config.family == args.formal_family
    ]
    counts = default_query_count_bounds(
        len(formal_configs),
        targets=len(args.target_recalls),
        screen_repeats=args.screen_repeats,
        verification_repeats=args.verification_repeats,
        final_repeats=args.final_repeats,
        families=1,
    )
    dry_uuid = args.run_uuid or "<required-at-runtime>"
    return {
        "implementation": args.implementation,
        "execution_stage": args.execution_stage,
        "final_block": args.final_block,
        "run_uuid": dry_uuid,
        "staging_dir": str(args.out_dir / "staging" / dry_uuid),
        "formal_family": args.formal_family,
        "formal_target_recalls": list(FORMAL_TARGET_RECALLS),
        "formal_filter_count": 14,
        "formal_cell_count": 42,
        "calibration_final_separated": True,
        "final_blocks": 2,
        "warmup_queries_per_filter_per_invocation": args.warmup_queries,
        "relation_prewarm": relation_prewarm_spec(args),
        "resume": args.resume,
        "config_ladder": str(args.config_ladder) if args.config_ladder else "deterministic_default",
        "declared_max_ef_search": args.max_ef_search,
        "upstream_evaluation_patch": (
            str(args.upstream_evaluation_patch)
            if args.upstream_evaluation_patch
            else None
        ),
        "candidate_validity_predicate": args.candidate_validity_predicate,
        "default_effective_config_count": len(default_configs),
        "formal_family_effective_config_count": len(formal_configs),
        "default_dropped_equivalent_configs": proof["dropped_equivalent_configs"],
        "query_splits": validate_split_contract(),
        "default_query_count_bounds_per_implementation": counts,
        "promotion_margin": args.promotion_margin,
        "official_pinned_vector_so_sha256": OFFICIAL_UPSTREAM_VECTOR_SO_SHA256,
        "formal_cli_complete": bool(
            args.server_container
            and args.run_uuid
            and args.data_epoch
            and args.vector_build_recipe
            and args.vector_compiler_flags
            and args.vector_source_repo
            and args.graph_identity_json
            and args.clone_index
            and (
                args.implementation != "official"
                or (
                    args.expected_vector_so_sha256
                    and args.vector_source_tag
                    and args.vector_source_commit
                )
            )
        ),
        "file_access": False,
        "docker_access": False,
        "database_access": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    if not math.isfinite(args.promotion_margin) or not 0 <= args.promotion_margin < 1:
        raise SystemExit("--promotion-margin must be in [0, 1)")
    if args.dry_run:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return
    paths = run(args)
    for name, path in paths.items():
        print(f"wrote {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
