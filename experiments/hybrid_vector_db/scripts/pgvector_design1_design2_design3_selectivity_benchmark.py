from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg import errors

try:
    from .common_pg import pg_config_from_env
    from .faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS
    from .pgvector_predicate_guidance_benchmark import FILTER_ATOMS
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env
    from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS
    from pgvector_predicate_guidance_benchmark import FILTER_ATOMS


INSERTION_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
INSERTION_INDEX = "public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx"
BFS_TABLE = INSERTION_TABLE
BFS_INDEX = "public.amazon_grocery_reviews_10m_pgvector_hnsw_bfs_clone_idx"

MODES = [
    "original",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
]

MODE_LABELS = {
    "original": "Original pgvector",
    "design1_bloom": "Design 1",
    "design1_bloom_bfs_layout": "Design 1 + Design 2",
    "design1_bloom_bfs_layout_d3": "Design 1 + Design 2 + Design 3",
}

MODE_CONFIG_FIELDS = (
    "ef_search",
    "max_scan_tuples",
    "scan_mem_multiplier",
    "iterative_scan",
    "guided_collect_target",
)
ITERATIVE_SCAN_VALUES = {"off", "strict_order", "relaxed_order"}
SQLENS_V11_BUILD_PREFIX = "sqlens-v11-"
SQLENS_MIN_PROFILE_SEMANTICS = 7.0
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
SQLENS_TRAVERSAL_PROFILE_FIELDS = (
    "final_path",
    "planner_proof_attempted",
    "planner_proof_succeeded",
    "planner_proof_bypass_reason",
    "traversal_guidance_scope",
    "graph_expansion_pruned",
    "distance_computations_pruned",
    "pre_distance_membership_checks",
    "pre_distance_membership_matches",
    "pre_distance_membership_misses",
    "distance_computations_avoided_attempted",
    "distance_computations_avoided",
    "neighbor_expansion_guidance_checks",
    "neighbor_expansion_guidance_matches",
    "neighbor_expansion_guidance_misses",
    "traversal_guided_admissions",
    "traversal_guided_suppressions",
    "traversal_heap_tids_suppressed",
    "guided_expanded_nodes",
    "guided_phase_distance_computations",
    "stock_phase_expanded_nodes",
    "stock_phase_distance_computations",
    "stock_bypass_requests",
    "stock_bypass_reason",
    "fallback_requests",
    "fallback_reason",
    "fallback_stock_expanded_nodes",
    "fallback_stock_distance_computations",
    "traversal_estimated_skip_rate_valid",
    "traversal_estimated_skip_rate",
)
D2_GRAPH_PROOF_FIELDS = (
    "same_heap",
    "logical_equal",
    "entry_equal",
    "tuple_coverage_equal",
    "physical_equal",
)
D2_STABLE_COMPARISON_FIELDS = (
    "format",
    "same_heap",
    "logical_equal",
    "physical_equal",
    "entry_equal",
    "definition_equal",
    "tuple_coverage_equal",
    "left_definition_digest",
    "right_definition_digest",
    "left_tuple_coverage_digest",
    "right_tuple_coverage_digest",
    "left_logical_digest",
    "right_logical_digest",
    "left_physical_digest",
    "right_physical_digest",
)
D2_RELATION_IDENTITY_FIELDS = ("name", "oid", "relfilenode", "heap_oid")
D2_BFS_LOCALITY_COMPARISON_FIELDS = (
    "left_bfs_locality",
    "right_bfs_locality",
)


class SqlensProvenanceGateError(RuntimeError):
    """Raised when the formal runner is not connected to the required SQLens ABI."""


class D2GraphProofGateError(RuntimeError):
    """Raised when D2 is not a same-heap, same-logical-graph layout comparison."""


class BackendAffinityGateError(RuntimeError):
    """Raised when a production PostgreSQL backend is not on the requested CPUs."""


@dataclass(frozen=True)
class TruthEntry:
    query_id: int
    filtered_rows: int
    kth_distance_sq: float | None
    tie_tolerance: float
    self_excluded: bool
    strict_closer_count: int | None = None
    boundary_tied: bool | None = None


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_candidate_validity_predicate(value: str) -> str:
    """Accept one SQL expression, not a statement or comment-bearing fragment."""
    predicate = str(value or "").strip()
    forbidden = (";", "--", "/*", "*/", "\x00")
    token = next((token for token in forbidden if token in predicate), None)
    if token is not None:
        raise argparse.ArgumentTypeError(
            "candidate validity predicate must be a single comment-free SQL expression; "
            f"found forbidden token {token!r}"
        )
    return predicate


def effective_candidate_validity_predicate(value: object = "") -> str:
    predicate = validate_candidate_validity_predicate(str(value or ""))
    return predicate or "TRUE"


def candidate_validity_sha256(value: object = "") -> str:
    return hashlib.sha256(
        effective_candidate_validity_predicate(value).encode("utf-8")
    ).hexdigest()


def normalized_sql_predicate(value: object = "") -> str:
    """Normalize catalog-rendered SQL enough for a fail-closed predicate bind."""
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for position, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0 or (depth == 0 and position != len(text) - 1):
                    balanced = False
                    break
        if balanced and depth == 0:
            text = text[1:-1].strip()
        else:
            break
    return text


def candidate_validity_index_predicate_matches(
    catalog_predicate: object,
    candidate_validity_predicate: object = "",
) -> bool:
    """Bind TRUE to a full index and every other value to its exact partial qual."""
    expected = normalized_sql_predicate(
        effective_candidate_validity_predicate(candidate_validity_predicate)
    )
    observed = normalized_sql_predicate(catalog_predicate)
    if expected == "true":
        return not observed
    return bool(observed) and observed == expected


def validate_guidance_atoms(
    atoms: list[str],
    candidate_validity_predicate: object = "",
) -> list[str]:
    """Reject global candidate validity quals from the D1 atom channel."""
    validity = normalized_sql_predicate(
        effective_candidate_validity_predicate(candidate_validity_predicate)
    )
    if not validity:
        return atoms
    pattern = re.compile(
        r"(?<![a-z0-9_$])" + re.escape(validity) + r"(?![a-z0-9_$])",
        re.IGNORECASE,
    )
    invalid = [atom for atom in atoms if pattern.search(normalized_sql_predicate(atom))]
    if invalid:
        raise ValueError(
            "D1 guidance atoms must not contain the global candidate validity predicate; "
            f"invalid atoms={invalid!r}, predicate={validity!r}"
        )
    return atoms


def _cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for token in str(value).strip().split(","):
        token = token.strip()
        if not token:
            raise ValueError("CPU list contains an empty range")
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first = int(first_text)
            last = int(last_text)
        else:
            first = last = int(token)
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU range: {token!r}")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def normalize_cpu_list(value: str) -> str:
    try:
        cpus = sorted(_cpu_set(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid CPU list {value!r}: {exc}") from exc
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def backend_cpu_provenance(
    cur: psycopg.Cursor,
    requested_cpu_list: str | None,
) -> dict[str, object]:
    cur.execute("SELECT pg_backend_pid(), pg_read_file('/proc/self/status')")
    row = cur.fetchone()
    if not row or row[0] is None or not isinstance(row[1], str):
        raise BackendAffinityGateError(
            "could not read pg_backend_pid()/DB-side /proc/self/status affinity"
        )
    observed_raw = ""
    for line in row[1].splitlines():
        if line.startswith("Cpus_allowed_list:"):
            observed_raw = line.split(":", 1)[1].strip()
            break
    if not observed_raw:
        raise BackendAffinityGateError(
            "DB-side /proc/self/status is missing Cpus_allowed_list"
        )
    try:
        observed = normalize_cpu_list(observed_raw)
    except argparse.ArgumentTypeError as exc:
        raise BackendAffinityGateError(
            f"DB-side Cpus_allowed_list is invalid: {observed_raw!r}"
        ) from exc
    requested = normalize_cpu_list(requested_cpu_list) if requested_cpu_list else ""
    exact_match = _cpu_set(observed) == _cpu_set(requested) if requested else None
    return {
        "backend_pid": int(row[0]),
        "pid_namespace": "postgresql_container_namespace",
        "requested_cpu_list": requested,
        "observed_cpu_list": observed,
        "exact_match": exact_match,
        "pinning_attempted_by_runner": False,
        "mapping_trust": "db_side_proc_self_status",
        "checked_at": utc_now(),
    }


def enforce_backend_cpu_provenance(provenance: dict[str, object]) -> None:
    if provenance.get("requested_cpu_list") and provenance.get("exact_match") is not True:
        raise BackendAffinityGateError(
            "PostgreSQL backend CPU affinity mismatch: "
            f"backend_pid={provenance.get('backend_pid')}, "
            f"requested={provenance.get('requested_cpu_list')!r}, "
            f"observed={provenance.get('observed_cpu_list')!r}. "
            "Pin the trustworthy host PostgreSQL PID in orchestration; the runner will not "
            "apply taskset to a Docker namespace PID."
        )


def load_tie_aware_truth(
    path: Path,
    method: str = "pre_filter_exact",
    expected_self_excluded: bool = True,
    expected_candidate_validity_predicate: str | None = None,
) -> tuple[dict[tuple[str, int], TruthEntry], dict[int, int]]:
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_by_no: dict[int, int] = {}
    required = {
        "filtered_rows",
        "kth_distance_sq",
        "tie_tolerance",
        "self_excluded",
    }
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"truth CSV is missing tie-aware fields: {sorted(missing)}")
        if (
            expected_candidate_validity_predicate is not None
            and "candidate_validity_predicate" not in set(reader.fieldnames or ())
        ):
            raise ValueError(
                "truth CSV is missing candidate_validity_predicate required by the "
                "explicit candidate-validity contract"
            )
        for row in reader:
            if row.get("method") != method:
                continue
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            previous = query_by_no.setdefault(query_no, query_id)
            if previous != query_id:
                raise ValueError(f"query_no={query_no} maps to multiple query IDs")
            self_excluded = parse_bool(row["self_excluded"])
            if self_excluded != expected_self_excluded:
                raise ValueError(
                    f"truth row {(row['filter_name'], query_no)} self_excluded="
                    f"{self_excluded!r} does not match expected {expected_self_excluded!r}"
                )
            if expected_candidate_validity_predicate is not None:
                expected_validity = effective_candidate_validity_predicate(
                    expected_candidate_validity_predicate
                )
                observed_validity = effective_candidate_validity_predicate(
                    row.get("candidate_validity_predicate", "")
                )
                if observed_validity != expected_validity:
                    raise ValueError(
                        f"truth row {(row['filter_name'], query_no)} candidate_validity_predicate="
                        f"{observed_validity!r} does not match expected {expected_validity!r}"
                    )
            filtered_rows = int(row["filtered_rows"])
            kth_distance_sq = (
                float(row["kth_distance_sq"]) if row["kth_distance_sq"].strip() else None
            )
            tie_tolerance = float(row["tie_tolerance"])
            strict_closer_count = (
                int(row["strict_closer_count"])
                if row.get("strict_closer_count", "").strip()
                else None
            )
            boundary_tied = (
                parse_bool(row["boundary_tied"])
                if row.get("boundary_tied", "").strip()
                else None
            )
            if filtered_rows < 0 or (strict_closer_count is not None and strict_closer_count < 0):
                raise ValueError("tie-aware truth counts must be non-negative")
            if tie_tolerance < 0:
                raise ValueError("tie_tolerance must be non-negative")
            if filtered_rows and kth_distance_sq is None:
                raise ValueError("non-empty formal truth requires kth_distance_sq")
            key = (row["filter_name"], query_no)
            if key in truth:
                raise ValueError(f"duplicate truth row: {key}")
            truth[key] = TruthEntry(
                query_id=query_id,
                filtered_rows=filtered_rows,
                kth_distance_sq=kth_distance_sq,
                tie_tolerance=tie_tolerance,
                self_excluded=self_excluded,
                strict_closer_count=strict_closer_count,
                boundary_tied=boundary_tied,
            )
    return truth, query_by_no


def tie_aware_recall(result_distances: list[float], truth: TruthEntry, k: int) -> float:
    denominator = min(k, truth.filtered_rows)
    if denominator == 0:
        return 0.0
    if truth.kth_distance_sq is None:
        raise ValueError("formal truth is missing kth_distance_sq")
    if truth.strict_closer_count is not None and truth.strict_closer_count > denominator:
        raise ValueError("strict_closer_count exceeds the recall denominator")
    credit = min(
        denominator,
        k,
        sum(
            distance * distance <= truth.kth_distance_sq + truth.tie_tolerance
            for distance in result_distances[:k]
        ),
    )
    return credit / denominator


def parse_mode_configs_json(value: str) -> dict[str, dict[str, object]]:
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(f"cannot read mode config JSON from {value}: {exc}") from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid mode config JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("mode config JSON must be an object")

    configs: dict[str, dict[str, object]] = {}
    for mode, overrides in parsed.items():
        if mode not in MODES:
            raise argparse.ArgumentTypeError(f"unknown mode in mode config JSON: {mode}")
        if not isinstance(overrides, dict):
            raise argparse.ArgumentTypeError(f"mode config for {mode} must be an object")
        unknown = sorted(set(overrides) - set(MODE_CONFIG_FIELDS))
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown config field for {mode}: {unknown[0]}")

        normalized: dict[str, object] = {}
        for field, field_value in overrides.items():
            if field in {"ef_search", "max_scan_tuples", "guided_collect_target"}:
                if isinstance(field_value, bool) or not isinstance(field_value, int):
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be an integer")
            elif field == "scan_mem_multiplier":
                if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be a number")
                field_value = float(field_value)
            elif field == "iterative_scan":
                if field_value not in ITERATIVE_SCAN_VALUES:
                    choices = ", ".join(sorted(ITERATIVE_SCAN_VALUES))
                    raise argparse.ArgumentTypeError(f"{mode}.{field} must be one of: {choices}")
            normalized[field] = field_value
        configs[mode] = normalized
    return configs


def effective_mode_config(args: argparse.Namespace, mode: str) -> dict[str, object]:
    config = {field: getattr(args, field) for field in MODE_CONFIG_FIELDS}
    config.update(getattr(args, "mode_configs_json", {}).get(mode, {}))
    return config


def shuffled_modes(modes: list[str], rng: random.Random) -> list[str]:
    scheduled = list(modes)
    rng.shuffle(scheduled)
    return scheduled


def balanced_mode_order(modes: list[str], block_no: int, seed: int) -> list[str]:
    base = list(modes)
    random.Random(seed).shuffle(base)
    if not base:
        return base
    offset = block_no % len(base)
    return base[offset:] + base[:offset]


def parse_atoms(text: str) -> list[str]:
    atoms = [part.strip() for part in str(text or "").split("||") if part.strip()]
    if not atoms:
        raise ValueError("empty atoms field")
    return atoms


def load_filter_specs(path: Path | None) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    if path is None:
        return ATTR_FILTERS, dict(FILTER_ATOMS)
    filters: list[tuple[str, str, str]] = []
    atoms_by_filter: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["filter_name"]
            filters.append((name, row.get("actual_pct") or row["target_rate"], row["predicate"]))
            atoms_by_filter[name] = parse_atoms(row["atoms"])
    if not filters:
        raise SystemExit(f"no filters loaded from {path}")
    return filters, atoms_by_filter


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def parse_pct(value: object) -> float:
    text = str(value).strip().replace("%", "")
    return float(text)


def require_sqlens_provenance(cur: psycopg.Cursor) -> tuple[str, dict[str, Any]]:
    """Verify the SQLens v11 ABI before installing any C-backed SQL wrappers."""
    try:
        cur.execute("SELECT vector_sqlens_build_id()")
        row = cur.fetchone()
        build_id = str(row[0]) if row and row[0] is not None else ""
    except Exception as exc:  # noqa: BLE001 - missing SQL must fail closed
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_sqlens_build_id() is unavailable. "
            "Install/reload the SQLens v11 extension (and reconnect) before running this formal benchmark."
        ) from exc
    if not build_id.startswith(SQLENS_V11_BUILD_PREFIX):
        raise SqlensProvenanceGateError(
            f"SQLens v11 provenance gate failed: vector_sqlens_build_id() returned {build_id!r}; "
            f"expected the {SQLENS_V11_BUILD_PREFIX!r} prefix. "
            "Rebuild/reload the SQLens v11 extension and reconnect before running this formal benchmark."
        )

    try:
        cur.execute("SELECT vector_hnsw_last_scan_profile()")
        row = cur.fetchone()
        raw_profile = row[0] if row else None
        profile = json.loads(raw_profile) if isinstance(raw_profile, str) else raw_profile
    except Exception as exc:  # noqa: BLE001 - missing SQL or invalid JSON must fail closed
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is unavailable or is not valid JSON. "
            "Load the SQLens v11 extension and reconnect before running this formal benchmark."
        ) from exc
    if not isinstance(profile, dict):
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() did not return a JSON object. "
            "Load the SQLens v11 extension and reconnect before running this formal benchmark."
        )

    try:
        profile_version = float(profile["profile_semantics_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is missing a numeric "
            "profile_semantics_version. Load the SQLens v11 extension and reconnect."
        ) from exc

    required_fields = SQLENS_PROFILE_FIELDS + SQLENS_TRAVERSAL_PROFILE_FIELDS
    missing = [field for field in required_fields if field not in profile]
    if not math.isfinite(profile_version) or profile_version < SQLENS_MIN_PROFILE_SEMANTICS or missing:
        details = []
        if not math.isfinite(profile_version) or profile_version < SQLENS_MIN_PROFILE_SEMANTICS:
            details.append(
                f"profile_semantics_version={profile.get('profile_semantics_version')!r} "
                f"(need >= {SQLENS_MIN_PROFILE_SEMANTICS:g})"
            )
        if missing:
            details.append(f"missing fields={missing!r}")
        raise SqlensProvenanceGateError(
            "SQLens v11 provenance gate failed: vector_hnsw_last_scan_profile() is incompatible: "
            + "; ".join(details)
            + ". Load the SQLens v11 extension and reconnect before running this formal benchmark."
        )
    return build_id, profile


def require_exact_sqlens_identity(
    cur: psycopg.Cursor,
    expected_build_id: str,
    expected_vector_so_sha256: str,
) -> dict[str, object]:
    if not expected_build_id or len(expected_vector_so_sha256) != 64:
        raise SqlensProvenanceGateError(
            "exact SQLens identity gate requires a parent-provided build ID and vector.so SHA256"
        )
    try:
        cur.execute(
            "WITH lib AS ("
            "SELECT setting || '/vector.so' AS path "
            "FROM pg_config WHERE name = 'PKGLIBDIR'"
            ") SELECT vector_sqlens_build_id(), path, "
            "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - binary identity must fail closed
        raise SqlensProvenanceGateError(
            "exact SQLens identity gate could not read the server-side vector.so"
        ) from exc
    observed_build_id = str(row[0]) if row and row[0] is not None else ""
    observed_path = str(row[1]) if row and row[1] is not None else ""
    observed_sha = str(row[2]) if row and row[2] is not None else ""
    if observed_build_id != expected_build_id:
        raise SqlensProvenanceGateError(
            "SQLens build ID mismatch: "
            f"expected {expected_build_id!r}, observed {observed_build_id!r}"
        )
    if observed_sha != expected_vector_so_sha256:
        raise SqlensProvenanceGateError(
            "server-side vector.so SHA256 mismatch: "
            f"expected {expected_vector_so_sha256!r}, observed {observed_sha!r}"
        )
    if not observed_path.endswith("/vector.so"):
        raise SqlensProvenanceGateError(
            f"server-side vector.so path is invalid: {observed_path!r}"
        )
    return {
        "expected_build_id": expected_build_id,
        "expected_vector_so_sha256": expected_vector_so_sha256,
        "observed_build_id": observed_build_id,
        "observed_vector_so_path": observed_path,
        "observed_vector_so_sha256": observed_sha,
        "exact_match": True,
        "checked_at": utc_now(),
    }


def parse_json_object(value: str) -> dict[str, object]:
    source = value
    if not value.lstrip().startswith("{"):
        try:
            source = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise argparse.ArgumentTypeError(f"cannot read JSON object from {value}: {exc}") from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def stable_d2_graph_proof(proof: dict[str, object]) -> dict[str, object]:
    comparison = proof.get("comparison")
    relations = proof.get("relations")
    if not isinstance(comparison, dict) or not isinstance(relations, dict):
        raise D2GraphProofGateError("D2 stable proof is missing comparison/relation identity")
    stable_relations: dict[str, dict[str, object]] = {}
    for role in ("source", "clone"):
        relation = relations.get(role)
        if not isinstance(relation, dict):
            raise D2GraphProofGateError(f"D2 proof is missing {role} relation identity")
        missing = [field for field in D2_RELATION_IDENTITY_FIELDS if field not in relation]
        if missing:
            raise D2GraphProofGateError(
                f"D2 {role} relation identity is missing fields: {missing}"
            )
        stable_relations[role] = {
            field: relation[field] for field in D2_RELATION_IDENTITY_FIELDS
        }
    missing_comparison = [
        field for field in D2_STABLE_COMPARISON_FIELDS if field not in comparison
    ]
    if missing_comparison:
        raise D2GraphProofGateError(
            f"D2 graph proof is missing stable comparison fields: {missing_comparison}"
        )
    stable_comparison = {
        field: comparison[field] for field in D2_STABLE_COMPARISON_FIELDS
    }
    locality_fields_present = [
        field for field in D2_BFS_LOCALITY_COMPARISON_FIELDS if field in comparison
    ]
    if locality_fields_present and len(locality_fields_present) != len(
        D2_BFS_LOCALITY_COMPARISON_FIELDS
    ):
        raise D2GraphProofGateError(
            "D2 graph proof has only one side of the BFS locality comparison"
        )
    for field in D2_BFS_LOCALITY_COMPARISON_FIELDS:
        locality = comparison.get(field)
        if locality is not None:
            validate_d2_bfs_locality(locality, field)
            stable_comparison[field] = locality
    return {
        "proof_contract": "sqlens_same_heap_same_logical_graph_physical_layout_v2",
        "source_index": proof.get("source_index"),
        "clone_index": proof.get("clone_index"),
        "relations": stable_relations,
        "comparison": stable_comparison,
    }


def validate_d2_bfs_locality(value: object, field: str) -> None:
    """Validate the C proof's complete counters and bounded rank evidence."""
    if not isinstance(value, dict):
        raise D2GraphProofGateError(f"D2 {field} locality is not a JSON object")
    required = (
        "format",
        "rank_base",
        "graph_nodes",
        "reachable_nodes",
        "fallback_nodes",
        "sequence_nodes",
        "adjacent_pairs",
        "same_block_pairs",
        "next_block_pairs",
        "same_or_next_page_pairs",
        "nondecreasing_pairs",
        "backward_pairs",
        "total_abs_block_delta",
        "max_abs_block_delta",
        "page_runs",
        "same_block_ratio",
        "same_or_next_page_ratio",
        "nondecreasing_ratio",
        "full_statistics",
        "sample_limit",
        "sample_count",
        "sample_truncated",
        "sample_strategy",
        "rank_samples",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise D2GraphProofGateError(
            f"D2 {field} locality is missing fields: {missing}"
        )
    if value["format"] != "sqlens-hnsw-bfs-locality-v1":
        raise D2GraphProofGateError(f"D2 {field} locality has an unsupported format")
    if value["rank_base"] != 0 or value["full_statistics"] is not True:
        raise D2GraphProofGateError(
            f"D2 {field} locality must use zero-based complete statistics"
        )

    def nonnegative_int(name: str) -> int:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise D2GraphProofGateError(f"D2 {field} locality has invalid {name}")
        return item

    graph_nodes = nonnegative_int("graph_nodes")
    reachable_nodes = nonnegative_int("reachable_nodes")
    fallback_nodes = nonnegative_int("fallback_nodes")
    sequence_nodes = nonnegative_int("sequence_nodes")
    adjacent_pairs = nonnegative_int("adjacent_pairs")
    same_block_pairs = nonnegative_int("same_block_pairs")
    next_block_pairs = nonnegative_int("next_block_pairs")
    same_or_next_pairs = nonnegative_int("same_or_next_page_pairs")
    nondecreasing_pairs = nonnegative_int("nondecreasing_pairs")
    backward_pairs = nonnegative_int("backward_pairs")
    total_abs_block_delta = nonnegative_int("total_abs_block_delta")
    max_abs_block_delta = nonnegative_int("max_abs_block_delta")
    page_runs = nonnegative_int("page_runs")
    sample_limit = nonnegative_int("sample_limit")
    sample_count = nonnegative_int("sample_count")
    if graph_nodes != sequence_nodes or reachable_nodes + fallback_nodes != sequence_nodes:
        raise D2GraphProofGateError(f"D2 {field} locality sequence coverage is incomplete")
    if adjacent_pairs != max(sequence_nodes - 1, 0):
        raise D2GraphProofGateError(f"D2 {field} locality adjacent-pair count is invalid")
    if same_block_pairs + next_block_pairs != same_or_next_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality same/next counters disagree")
    if same_or_next_pairs > adjacent_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality has too many same/next pairs")
    if nondecreasing_pairs + backward_pairs != adjacent_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality monotonicity counters disagree")
    if page_runs != sequence_nodes - same_block_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality page-run count is invalid")
    if next_block_pairs > nondecreasing_pairs:
        raise D2GraphProofGateError(f"D2 {field} locality forward counters disagree")
    if max_abs_block_delta > total_abs_block_delta:
        raise D2GraphProofGateError(f"D2 {field} locality block-delta counters disagree")
    if adjacent_pairs == same_block_pairs and (
        total_abs_block_delta != 0 or max_abs_block_delta != 0
    ):
        raise D2GraphProofGateError(f"D2 {field} locality zero-delta counters disagree")
    if sample_limit != 256 or sample_count != min(sample_limit, sequence_nodes):
        raise D2GraphProofGateError(f"D2 {field} locality sample bound is invalid")
    if value["sample_truncated"] is not (sample_count < sequence_nodes):
        raise D2GraphProofGateError(f"D2 {field} locality sample truncation is invalid")
    if value["sample_strategy"] != "evenly_spaced_inclusive":
        raise D2GraphProofGateError(f"D2 {field} locality sample strategy is invalid")
    samples = value["rank_samples"]
    if not isinstance(samples, list) or len(samples) != sample_count:
        raise D2GraphProofGateError(f"D2 {field} locality rank samples are incomplete")
    previous_rank = -1
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise D2GraphProofGateError(f"D2 {field} locality has an invalid rank sample")
        rank = sample.get("rank")
        block = sample.get("block")
        offset = sample.get("offset")
        expected_rank = (
            0
            if sample_count == 1
            else sample_index * (sequence_nodes - 1) // (sample_count - 1)
        )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= previous_rank
            or rank < 0
            or rank >= sequence_nodes
            or rank != expected_rank
            or isinstance(block, bool)
            or not isinstance(block, int)
            or block < 0
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset <= 0
        ):
            raise D2GraphProofGateError(f"D2 {field} locality has invalid rank samples")
        previous_rank = rank
    if sample_count and (samples[0]["rank"] != 0 or samples[-1]["rank"] != sequence_nodes - 1):
        raise D2GraphProofGateError(f"D2 {field} locality samples do not cover sequence ends")
    ratio_counters = {
        "same_block_ratio": same_block_pairs,
        "same_or_next_page_ratio": same_or_next_pairs,
        "nondecreasing_ratio": nondecreasing_pairs,
    }
    denominator = adjacent_pairs if adjacent_pairs else 1
    for ratio_name, numerator in ratio_counters.items():
        ratio = value[ratio_name]
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
            or not math.isclose(
                float(ratio), numerator / denominator, rel_tol=1e-15, abs_tol=1e-15
            )
        ):
            raise D2GraphProofGateError(f"D2 {field} locality has invalid {ratio_name}")


def d2_stable_fingerprint(proof: dict[str, object]) -> str:
    encoded = json.dumps(
        stable_d2_graph_proof(proof), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_d2_graph_proof(
    proof: dict[str, object],
    source_index: str,
    clone_index: str,
) -> dict[str, object]:
    if source_index == clone_index:
        raise D2GraphProofGateError("D2 source and clone index must be different relations")
    comparison_value = proof.get("comparison", proof)
    if not isinstance(comparison_value, dict):
        raise D2GraphProofGateError("D2 graph proof comparison is not a JSON object")
    missing = [field for field in D2_GRAPH_PROOF_FIELDS if field not in comparison_value]
    if missing:
        raise D2GraphProofGateError(f"D2 graph proof is missing fields: {missing}")
    required_true = (
        "same_heap",
        "logical_equal",
        "entry_equal",
        "definition_equal",
        "tuple_coverage_equal",
    )
    failed = [field for field in required_true if comparison_value.get(field) is not True]
    if failed:
        raise D2GraphProofGateError(
            "D2 graph proof failed required equivalence checks: " + ", ".join(failed)
        )
    if comparison_value.get("physical_equal") is not False:
        raise D2GraphProofGateError(
            "D2 graph proof is not a meaningful layout experiment: physical_equal must be false"
        )
    expected_source = proof.get("source_index")
    expected_clone = proof.get("clone_index")
    if expected_source is not None and str(expected_source) != source_index:
        raise D2GraphProofGateError(
            f"D2 proof source index mismatch: proof={expected_source!r}, requested={source_index!r}"
        )
    if expected_clone is not None and str(expected_clone) != clone_index:
        raise D2GraphProofGateError(
            f"D2 proof clone index mismatch: proof={expected_clone!r}, requested={clone_index!r}"
        )
    stable = stable_d2_graph_proof(
        {
            **proof,
            "source_index": source_index,
            "clone_index": clone_index,
            "comparison": comparison_value,
        }
    )
    stable_comparison = stable["comparison"]
    if stable_comparison["format"] != "sqlens-hnsw-compare-v2":
        raise D2GraphProofGateError("D2 graph proof has an unsupported comparison format")
    for field in D2_STABLE_COMPARISON_FIELDS[7:]:
        digest = str(stable_comparison[field])
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(char not in "0123456789abcdef" for char in digest[7:])
        ):
            raise D2GraphProofGateError(f"D2 graph proof has an invalid digest in {field}")
    for left, right in (
        ("left_definition_digest", "right_definition_digest"),
        ("left_tuple_coverage_digest", "right_tuple_coverage_digest"),
        ("left_logical_digest", "right_logical_digest"),
    ):
        if stable_comparison[left] != stable_comparison[right]:
            raise D2GraphProofGateError(f"D2 equal graph proof has mismatched {left}/{right}")
    if stable_comparison["left_physical_digest"] == stable_comparison["right_physical_digest"]:
        raise D2GraphProofGateError("D2 physical digests are equal despite physical_equal=false")
    source_relation = stable["relations"]["source"]
    clone_relation = stable["relations"]["clone"]
    if source_relation["name"] != source_index or clone_relation["name"] != clone_index:
        raise D2GraphProofGateError("D2 relation identity names do not match requested indexes")
    for role, relation in (("source", source_relation), ("clone", clone_relation)):
        for field in ("oid", "relfilenode", "heap_oid"):
            if int(relation[field]) <= 0:
                raise D2GraphProofGateError(
                    f"D2 {role} relation identity has invalid {field}={relation[field]!r}"
                )
    if int(source_relation["heap_oid"]) != int(clone_relation["heap_oid"]):
        raise D2GraphProofGateError("D2 source and clone relation identities do not share a heap")
    stable_fingerprint = d2_stable_fingerprint(stable)
    delegated_fingerprint = proof.get("stable_fingerprint_sha256")
    if delegated_fingerprint is not None and str(delegated_fingerprint) != stable_fingerprint:
        raise D2GraphProofGateError("D2 delegated stable fingerprint does not match proof fields")
    return {
        **stable,
        "checked_at": proof.get("checked_at") or utc_now(),
        "stable_fingerprint_sha256": stable_fingerprint,
    }


def require_d2_graph_proof(
    cur: psycopg.Cursor,
    source_index: str,
    clone_index: str,
) -> dict[str, object]:
    try:
        cur.execute(
            "SELECT vector_hnsw_graph_compare(%s::regclass, %s::regclass), "
            "source.oid::bigint, source.relfilenode::bigint, source_index.indrelid::bigint, "
            "clone.oid::bigint, clone.relfilenode::bigint, clone_index.indrelid::bigint "
            "FROM pg_class source "
            "JOIN pg_index source_index ON source_index.indexrelid = source.oid "
            "JOIN pg_class clone ON clone.oid = %s::regclass "
            "JOIN pg_index clone_index ON clone_index.indexrelid = clone.oid "
            "WHERE source.oid = %s::regclass",
            (source_index, clone_index, clone_index, source_index),
        )
        row = cur.fetchone()
        raw = row[0] if row else None
        comparison = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:  # noqa: BLE001 - a formal D2 gate must fail closed
        raise D2GraphProofGateError(
            "D2 graph proof gate failed: vector_hnsw_graph_compare(source, clone) is unavailable "
            "or could not fingerprint both indexes"
        ) from exc
    if not isinstance(comparison, dict):
        raise D2GraphProofGateError(
            "D2 graph proof gate failed: vector_hnsw_graph_compare() did not return a JSON object"
        )
    return validate_d2_graph_proof(
        {
            "checked_at": utc_now(),
            "source_index": source_index,
            "clone_index": clone_index,
            "relations": {
                "source": {
                    "name": source_index,
                    "oid": row[1],
                    "relfilenode": row[2],
                    "heap_oid": row[3],
                },
                "clone": {
                    "name": clone_index,
                    "oid": row[4],
                    "relfilenode": row[5],
                    "heap_oid": row[6],
                },
            },
            "comparison": comparison,
        },
        source_index,
        clone_index,
    )


def require_d2_relation_identity(
    cur: psycopg.Cursor,
    source_index: str,
    clone_index: str,
) -> dict[str, dict[str, object]]:
    cur.execute(
        "SELECT source.oid::bigint, source.relfilenode::bigint, source_index.indrelid::bigint, "
        "source_index.indisvalid, source_index.indisready, source_index.indislive, "
        "clone.oid::bigint, clone.relfilenode::bigint, clone_index.indrelid::bigint, "
        "clone_index.indisvalid, clone_index.indisready, clone_index.indislive "
        "FROM pg_class source "
        "JOIN pg_index source_index ON source_index.indexrelid = source.oid "
        "JOIN pg_class clone ON clone.oid = %s::regclass "
        "JOIN pg_index clone_index ON clone_index.indexrelid = clone.oid "
        "WHERE source.oid = %s::regclass",
        (clone_index, source_index),
    )
    row = cur.fetchone()
    if row is None:
        raise D2GraphProofGateError("D2 source or clone index identity is unavailable")
    relations = {
        "source": {
            "name": source_index,
            "oid": int(row[0]),
            "relfilenode": int(row[1]),
            "heap_oid": int(row[2]),
            "indisvalid": bool(row[3]),
            "indisready": bool(row[4]),
            "indislive": bool(row[5]),
        },
        "clone": {
            "name": clone_index,
            "oid": int(row[6]),
            "relfilenode": int(row[7]),
            "heap_oid": int(row[8]),
            "indisvalid": bool(row[9]),
            "indisready": bool(row[10]),
            "indislive": bool(row[11]),
        },
    }
    if relations["source"]["heap_oid"] != relations["clone"]["heap_oid"]:
        raise D2GraphProofGateError("D2 source and clone no longer share the same heap")
    for role, relation in relations.items():
        if not all(bool(relation[field]) for field in ("indisvalid", "indisready", "indislive")):
            raise D2GraphProofGateError(f"D2 {role} index is no longer valid, ready, and live")
    return relations


def require_d2_graph_proof_from_env(
    args: argparse.Namespace,
    delegated_proof: dict[str, object] | None = None,
) -> dict[str, object]:
    delegated = (
        validate_d2_graph_proof(
            delegated_proof,
            args.insertion_index,
            args.bfs_index,
        )
        if delegated_proof is not None
        else None
    )
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            ensure_functions(cur)
            if delegated is None:
                live = require_d2_graph_proof(cur, args.insertion_index, args.bfs_index)
                live["live_revalidated"] = True
                live["full_graph_fingerprint_recomputed"] = True
                return live
            live_relations = require_d2_relation_identity(
                cur, args.insertion_index, args.bfs_index
            )
            delegated_relations = delegated["relations"]
            for role in ("source", "clone"):
                for field in D2_RELATION_IDENTITY_FIELDS:
                    if live_relations[role][field] != delegated_relations[role][field]:
                        raise D2GraphProofGateError(
                            "D2 live revalidation changed delegated relation identity: "
                            f"{role}.{field}"
                        )
            return {
                **delegated,
                "delegated_checked_at": delegated.get("checked_at"),
                "live_identity_checked_at": utc_now(),
                "live_relation_identity": live_relations,
                "live_revalidated": True,
                "full_graph_fingerprint_recomputed": False,
            }
        finally:
            cur.close()
    finally:
        conn.close()


def require_sqlens_provenance_from_env() -> tuple[str, dict[str, Any]]:
    """Run the formal-entry gate on a short-lived connection before any wrapper DDL."""
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            return require_sqlens_provenance(cur)
        finally:
            cur.close()
    finally:
        conn.close()


def require_exact_sqlens_identity_from_env(
    expected_build_id: str,
    expected_vector_so_sha256: str,
) -> dict[str, object]:
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        try:
            return require_exact_sqlens_identity(
                cur,
                expected_build_id,
                expected_vector_so_sha256,
            )
        finally:
            cur.close()
    finally:
        conn.close()


def ensure_functions(cur: psycopg.Cursor) -> None:
    require_sqlens_provenance(cur)
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_bind(regclass, text[], text) "
        "RETURNS boolean AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_sqlens_build_id() "
        "RETURNS text AS 'vector' LANGUAGE C IMMUTABLE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_epoch_bump_trigger() "
        "RETURNS trigger AS 'vector' LANGUAGE C",
        "CREATE OR REPLACE FUNCTION vector_hnsw_fragment_tracking_enable(regclass) "
        "RETURNS int8 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_graph_compare(regclass, regclass) "
        "RETURNS jsonb AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
    ]
    for sql in functions:
        try:
            cur.execute(sql)
        except Exception as exc:  # noqa: BLE001 - parallel runners may race on pg_proc updates
            if "tuple concurrently updated" not in str(exc):
                raise
            cur.connection.rollback()
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def ensure_tracking(cur: psycopg.Cursor, *tables: str) -> None:
    for table in dict.fromkeys(tables):
        cur.execute("SELECT vector_hnsw_fragment_tracking_enable(%s::regclass)", (table,))


def mode_uses_d2(mode: str) -> bool:
    return mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}


def mode_uses_guidance(mode: str) -> bool:
    return mode != "original"


def configure(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    cache_mb: int,
    mode: str = "original",
    mode_config: dict[str, object] | None = None,
) -> None:
    config = mode_config or effective_mode_config(args, mode)
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(config['ef_search'])}")
    cur.execute(f"SET hnsw.iterative_scan = {config['iterative_scan']}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(config['max_scan_tuples'])}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(config['scan_mem_multiplier'])}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(config['guided_collect_target'])}")
    cur.execute(f"SET hnsw.traversal_guided_target = {int(config['guided_collect_target'])}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(cache_mb)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute(f"SET hnsw.page_access = {args.d2_page_access if mode_uses_d2(mode) else 'off'}")
    cur.execute(f"SET hnsw.index_page_access = {args.d2_index_page_access if mode_uses_d2(mode) else 'off'}")
    cur.execute(f"SET hnsw.page_window = {int(args.d2_page_window)}")
    cur.execute(f"SET hnsw.page_prefetch_min_items = {int(args.d2_page_prefetch_min_items)}")
    cur.execute(f"SET hnsw.page_disable_after_no_merge = {int(args.d2_page_disable_after_no_merge)}")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def mode_table_index(args: argparse.Namespace, mode: str) -> tuple[str, str]:
    if mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}:
        return args.bfs_table, args.bfs_index
    return args.insertion_table, args.insertion_index


def set_preferred_index_if_supported(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    expected_index: str,
) -> str | None:
    guc = str(getattr(args, "preferred_index_guc", "hnsw.preferred_index"))
    cur.execute("SELECT current_setting(%s, true)", (guc,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    cur.execute("SELECT set_config(%s, %s, false)", (guc, expected_index))
    cur.execute(
        "SELECT current_setting(%s), current_setting(%s)::regclass = %s::regclass",
        (guc, guc, expected_index),
    )
    observed = cur.fetchone()
    if not observed or observed[0] is None or observed[1] is not True:
        raise RuntimeError(
            f"{guc} did not resolve to expected index {expected_index!r}: {observed!r}"
        )
    return str(observed[0])


def uses_exact_predicate_scan_contract(filter_strategy: str) -> bool:
    return filter_strategy == "traversal_guided"


def query_table_for_candidate(args: argparse.Namespace, candidate_table: str) -> str:
    """Use the candidate heap as the query source unless an external query heap is supplied."""
    return str(getattr(args, "query_table", None) or candidate_table)


def candidate_self_exclusion(args: argparse.Namespace, candidate_table: str) -> bool:
    return query_table_for_candidate(args, candidate_table) == candidate_table


def validate_query_source_contract(args: argparse.Namespace) -> None:
    observed = {
        candidate_self_exclusion(args, mode_table_index(args, mode)[0])
        for mode in args.modes
    }
    if len(observed) != 1:
        raise RuntimeError(
            "query source has inconsistent self-exclusion semantics across candidate tables; "
            "supply a query table that is either external to every mode or the candidate table"
        )
    actual = observed.pop()
    if actual != args.expected_truth_self_excluded:
        raise RuntimeError(
            "query/truth self-exclusion contract mismatch: "
            f"candidate_self_excluded={actual!r}, "
            f"expected_truth_self_excluded={args.expected_truth_self_excluded!r}"
        )


def quoted_column(identifier: str) -> str:
    if not identifier or "." in identifier or "\x00" in identifier:
        raise ValueError(f"invalid column identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'


def search_query_sql(
    table: str,
    predicate: str,
    k: int,
    bind_guidance: bool = False,
    client_self_exclusion: bool = False,
    *,
    candidate_validity_predicate: str = "",
    query_table: str | None = None,
    query_id_column: str = "id",
    query_vector_column: str = "embedding",
    self_exclusion: bool = True,
) -> str:
    binding = (
        "(SELECT vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) OFFSET 0) AND "
        if bind_guidance
        else ""
    )
    source_table = query_table or table
    query_id = quoted_column(query_id_column)
    query_vector = quoted_column(query_vector_column)
    validity_predicate = effective_candidate_validity_predicate(
        candidate_validity_predicate
    )
    self_qual = "" if client_self_exclusion or not self_exclusion else " AND id <> %s"
    scan_limit = int(k) + 1 if client_self_exclusion else int(k)
    return f"""
        SELECT id,
               embedding <-> (
                   SELECT q.{query_vector}
                   FROM {source_table} AS q
                   WHERE q.{query_id} = %s
               ) AS distance
        FROM {table}
        WHERE {binding}({predicate}) AND ({validity_predicate}){self_qual}
        ORDER BY distance
        LIMIT {scan_limit}
    """


def plan_index_nodes(value: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "Index Name" in value:
            found.append(
                {
                    key: value.get(key)
                    for key in ("Node Type", "Index Name", "Schema", "Relation Name", "Alias")
                }
            )
        for child in value.values():
            found.extend(plan_index_nodes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(plan_index_nodes(child))
    return found


def explain_hnsw_plan(
    cur: psycopg.Cursor,
    table: str,
    expected_index: str,
    predicate: str,
    query_id: int,
    k: int,
    binding: tuple[str, list[str], str] | None = None,
    client_self_exclusion: bool = False,
    *,
    candidate_validity_predicate: str = "",
    query_table: str | None = None,
    query_id_column: str = "id",
    query_vector_column: str = "embedding",
    self_exclusion: bool = True,
) -> dict[str, object]:
    cur.execute(
        "SELECT idx.oid::bigint, idx.relname, idx_ns.nspname, am.amname, "
        "tbl.oid::bigint, tbl.relname, tbl_ns.nspname "
        ", pg_get_expr(ix.indpred, ix.indrelid) "
        "FROM pg_class idx "
        "JOIN pg_namespace idx_ns ON idx_ns.oid = idx.relnamespace "
        "JOIN pg_am am ON am.oid = idx.relam "
        "JOIN pg_index ix ON ix.indexrelid = idx.oid "
        "JOIN pg_class tbl ON tbl.oid = ix.indrelid "
        "JOIN pg_namespace tbl_ns ON tbl_ns.oid = tbl.relnamespace "
        "WHERE idx.oid = to_regclass(%s) AND tbl.oid = to_regclass(%s)",
        (expected_index, table),
    )
    metadata = cur.fetchone()
    if metadata is None:
        return {
            "passed": False,
            "expected_index": expected_index,
            "expected_table": table,
            "failure": "expected index/table metadata not found",
            "observed_index_nodes": [],
        }

    if len(metadata) < 8:
        raise RuntimeError("index catalog metadata is missing pg_index.indpred")
    (
        index_oid,
        index_name,
        index_schema,
        access_method,
        table_oid,
        table_name,
        table_schema,
        catalog_predicate,
    ) = metadata
    sql = search_query_sql(
        table,
        predicate,
        k,
        binding is not None,
        client_self_exclusion,
        candidate_validity_predicate=candidate_validity_predicate,
        query_table=query_table,
        query_id_column=query_id_column,
        query_vector_column=query_vector_column,
        self_exclusion=self_exclusion,
    )
    params: tuple[object, ...] = (int(query_id),)
    if binding is not None:
        params += binding
    if self_exclusion and not client_self_exclusion:
        params += (int(query_id),)
    cur.execute("EXPLAIN (FORMAT JSON, VERBOSE) " + sql, params)
    explain_value: Any = cur.fetchone()[0]
    if isinstance(explain_value, str):
        explain_value = json.loads(explain_value)
    observed = plan_index_nodes(explain_value)
    matched = [
        node
        for node in observed
        if node.get("Node Type") in {"Index Scan", "Index Only Scan"}
        and node.get("Index Name") == index_name
        and node.get("Relation Name") == table_name
        and node.get("Schema") == table_schema
    ]
    expected_predicate = effective_candidate_validity_predicate(
        candidate_validity_predicate
    )
    expected_is_partial = not candidate_validity_index_predicate_matches(
        None, expected_predicate
    )
    predicate_matches = candidate_validity_index_predicate_matches(
        catalog_predicate, expected_predicate
    )
    passed = access_method == "hnsw" and bool(matched) and predicate_matches
    if access_method != "hnsw" or not matched:
        failure = "EXPLAIN did not use the expected HNSW index"
    elif not predicate_matches:
        failure = (
            "expected index pg_index.indpred does not match candidate validity predicate: "
            f"catalog={catalog_predicate!r}, expected={expected_predicate!r}"
        )
    else:
        failure = ""
    return {
        "passed": passed,
        "expected_index": expected_index,
        "expected_index_oid": index_oid,
        "expected_index_identity": f"{index_schema}.{index_name}",
        "expected_index_access_method": access_method,
        "expected_index_predicate": expected_predicate,
        "expected_index_predicate_sha256": candidate_validity_sha256(expected_predicate),
        "expected_index_is_partial": expected_is_partial,
        "catalog_index_oid": index_oid,
        "catalog_index_predicate": catalog_predicate,
        "catalog_index_predicate_sha256": (
            candidate_validity_sha256(catalog_predicate)
            if catalog_predicate is not None
            else candidate_validity_sha256("TRUE")
        ),
        "catalog_index_is_partial": catalog_predicate is not None,
        "catalog_index_predicate_matches": predicate_matches,
        "expected_table": table,
        "expected_table_oid": table_oid,
        "expected_table_identity": f"{table_schema}.{table_name}",
        "query_id": query_id,
        "query_table": query_table or table,
        "query_id_column": query_id_column,
        "query_vector_column": query_vector_column,
        "self_excluded": self_exclusion,
        "self_exclusion_contract": (
            "limit_k_plus_1_client_remove_query_id"
            if client_self_exclusion
            else "sql_residual_id_not_equal" if self_exclusion else "none_external_query_source"
        ),
        "scan_limit": int(k) + 1 if client_self_exclusion else int(k),
        "residual_self_qual_present": self_exclusion and not client_self_exclusion,
        "candidate_validity_predicate": effective_candidate_validity_predicate(
            candidate_validity_predicate
        ),
        "candidate_validity_predicate_sha256": candidate_validity_sha256(
            candidate_validity_predicate
        ),
        "statement_binding_present": binding is not None,
        "observed_index_nodes": observed,
        "matched_index_nodes": matched,
        "plan": explain_value,
        "failure": failure,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_plan_evidence(
    args: argparse.Namespace,
    status: str,
    error: BaseException | None = None,
) -> None:
    path = args.plan_evidence_out
    payload = {
        "status": status,
        "started_at": args.plan_started_at,
        "completed_at": utc_now() if status in {"complete", "failed"} else None,
        "output": str(args.out),
        "output_rows": getattr(args, "output_rows", 0),
        "output_sha256": sha256_file(args.out),
        "d3_initialization": "workload_driven_adaptive",
        "prebuilt_fragments": 0,
        "warmup_all_queries": bool(getattr(args, "warmup_all_queries", False)),
        "warmup_evidence": getattr(args, "warmup_evidence", []),
        "execution_lifecycle": getattr(args, "execution_lifecycle", None),
        "guidance_filter_strategy": args.guidance_filter_strategy,
        "query_contract": {
            "query_table": args.query_table or "candidate_table_per_mode",
            "query_id_column": args.query_id_column,
            "query_vector_column": args.query_vector_column,
            "self_excluded": args.expected_truth_self_excluded,
            "candidate_validity_predicate": effective_candidate_validity_predicate(
                getattr(args, "candidate_validity_predicate", "")
            ),
            "candidate_validity_predicate_explicit": bool(
                getattr(args, "candidate_validity_predicate_explicit", False)
            ),
            "candidate_validity_predicate_sha256": candidate_validity_sha256(
                getattr(args, "candidate_validity_predicate", "")
            ),
            "candidate_validity_contract": (
                "planner_partial_index_predicate_and_sql_candidate_qual_not_guidance_atom"
            ),
            "predicate_contract": (
                "exact_activated_workload_predicate_plus_candidate_validity_sql_qual"
                if uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                else "diagnostic_workload_plus_candidate_validity_sql_quals"
            ),
            "self_exclusion": (
                "limit_k_plus_1_client_remove_query_id"
                if (
                    uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                    and args.expected_truth_self_excluded
                )
                else (
                    "sql_residual_id_not_equal"
                    if args.expected_truth_self_excluded
                    else "none_external_query_source"
                )
            ),
            "measured_latency_includes_client_self_exclusion": bool(
                uses_exact_predicate_scan_contract(args.guidance_filter_strategy)
                and args.expected_truth_self_excluded
            ),
        },
        "d2_graph_proof": getattr(args, "d2_graph_proof", {"required": False}),
        "d2_graph_proof_final": getattr(
            args, "d2_graph_proof_final", {"required": False}
        ),
        "sqlens_runtime_identity_startup": getattr(
            args, "sqlens_runtime_identity", None
        ),
        "sqlens_runtime_identity_final": getattr(
            args, "sqlens_runtime_identity_final", None
        ),
        "backend_cpu_evidence": getattr(args, "backend_cpu_evidence", []),
        "runtime_sqlens_identity_evidence": getattr(
            args, "runtime_sqlens_identity_evidence", []
        ),
        "checks": args.plan_evidence,
        "error": (
            {"type": error.__class__.__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def should_enable_guidance(args: argparse.Namespace, filter_name: str) -> tuple[bool, str]:
    selectivity = float(args.filter_selectivity_by_name.get(filter_name, 100.0))
    atom_count = len(args.filter_atoms.get(filter_name, []))
    if selectivity > float(args.guidance_selectivity_max_pct):
        return False, f"selectivity>{args.guidance_selectivity_max_pct:g}%"
    if atom_count > int(args.guidance_max_atoms):
        return False, f"atoms>{args.guidance_max_atoms}"
    return True, "enabled"


def read_guidance_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    raw = cur.fetchone()[0]
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise RuntimeError("guidance profile is not a JSON object")
    return profile


def read_scan_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    raw = cur.fetchone()[0]
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise RuntimeError("scan profile is not a JSON object")
    return profile


def read_cache_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")
    raw = cur.fetchone()[0]
    profile = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(profile, dict):
        raise RuntimeError("metadata cache profile is not a JSON object")
    return profile


def activate(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    mode: str,
    filter_name: str,
    *,
    read_profile: bool = True,
) -> dict[str, object]:
    table, index = mode_table_index(args, mode)
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    if mode == "original":
        cur.execute("SET hnsw.filter_strategy = off")
        return {"table": table, "index": index, "guidance_enabled": False, "guidance_route": "stock"}
    enabled, route = should_enable_guidance(args, filter_name)
    if not enabled:
        cur.execute("SET hnsw.filter_strategy = off")
        return {"table": table, "index": index, "guidance_enabled": False, "guidance_route": route}
    atoms = args.filter_atoms[filter_name]
    validate_guidance_atoms(
        atoms, getattr(args, "candidate_validity_predicate", "")
    )
    cur.execute(f"SET hnsw.filter_strategy = {args.guidance_filter_strategy}")
    if args.reset_cache_per_query and mode in {"design1_bloom", "design1_bloom_bfs_layout"}:
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    kind = "adaptive" if mode == "design1_bloom_bfs_layout_d3" else "bloom"
    # These atoms are only the workload predicate from filters CSV. A broad
    # partial-index predicate (for example embedding_valid) is enforced by the
    # planner/index and SQL candidate qual, and must never become D1 guidance.
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (index, atoms, kind),
    )
    activation_row = cur.fetchone()
    activated_atoms = int(activation_row[0]) if activation_row and activation_row[0] is not None else 0
    profile = read_guidance_profile(cur) if read_profile else {}
    profile["table"] = table
    profile["index"] = index
    profile["activation_atom_count"] = activated_atoms
    active = bool(profile.get("active", activated_atoms > 0))
    if mode == "design1_bloom_bfs_layout_d3" and (activated_atoms <= 0 or not active):
        cur.execute("SET hnsw.filter_strategy = off")
        profile["guidance_enabled"] = False
        profile["guidance_route"] = "d3_probe"
    else:
        profile["guidance_enabled"] = True
        profile["guidance_route"] = route
    return profile


def activation_binding(
    args: argparse.Namespace,
    mode: str,
    filter_name: str,
    activation_profile: dict[str, object],
) -> tuple[str, list[str], str] | None:
    if not activation_profile.get("guidance_enabled"):
        return None
    kind = "adaptive" if mode == "design1_bloom_bfs_layout_d3" else "bloom"
    return str(activation_profile["index"]), args.filter_atoms[filter_name], kind


def run_query(
    cur: psycopg.Cursor,
    table: str,
    predicate: str,
    query_id: int,
    k: int,
    binding: tuple[str, list[str], str] | None = None,
    client_self_exclusion: bool = False,
    *,
    candidate_validity_predicate: str = "",
    query_table: str | None = None,
    query_id_column: str = "id",
    query_vector_column: str = "embedding",
    self_exclusion: bool = True,
    reset_profile: bool = True,
    read_profile: bool = True,
) -> tuple[list[int], list[float], dict[str, object]]:
    if reset_profile:
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    params: tuple[object, ...] = (int(query_id),)
    if binding is not None:
        params += binding
    if self_exclusion and not client_self_exclusion:
        params += (int(query_id),)
    cur.execute(
        search_query_sql(
            table,
            predicate,
            k,
            binding is not None,
            client_self_exclusion,
            candidate_validity_predicate=candidate_validity_predicate,
            query_table=query_table,
            query_id_column=query_id_column,
            query_vector_column=query_vector_column,
            self_exclusion=self_exclusion,
        ),
        params,
    )
    result_rows = cur.fetchall()
    raw_returned = len(result_rows)
    if client_self_exclusion:
        result_rows = [row for row in result_rows if int(row[0]) != int(query_id)][:k]
    ids = [int(row[0]) for row in result_rows[:k]]
    distances = [float(row[1]) for row in result_rows[:k]]
    profile = read_scan_profile(cur) if read_profile else {}
    profile["sqlens_raw_returned_before_self_exclusion"] = raw_returned
    return ids, distances, profile


@dataclass
class ModeRuntime:
    mode: str
    config: dict[str, object]
    cache_mb: int
    conn: psycopg.Connection
    cur: psycopg.Cursor
    planner_proof_verified: bool = False
    preferred_index_current_setting: str | None = None
    backend_cpu_provenance: dict[str, object] | None = None
    sqlens_runtime_identity: dict[str, object] | None = None


def gate_runtime_plans(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filters: list[tuple[str, float, str]],
    query_id: int,
) -> None:
    table, expected_index = mode_table_index(args, runtime.mode)
    query_table = query_table_for_candidate(args, table)
    self_exclusion = candidate_self_exclusion(args, table)
    client_self_exclusion = (
        uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion
    )
    try:
        for filter_name, _, predicate in filters:
            try:
                activation = activate(runtime.cur, args, runtime.mode, filter_name)
                binding = activation_binding(args, runtime.mode, filter_name, activation)
                evidence = explain_hnsw_plan(
                    runtime.cur,
                    table,
                    expected_index,
                    predicate,
                    query_id,
                    args.k,
                    binding,
                    client_self_exclusion,
                    candidate_validity_predicate=getattr(
                        args, "candidate_validity_predicate", ""
                    ),
                    query_table=query_table,
                    query_id_column=getattr(args, "query_id_column", "id"),
                    query_vector_column=getattr(args, "query_vector_column", "embedding"),
                    self_exclusion=self_exclusion,
                )
            except Exception as exc:
                evidence = {
                    "passed": False,
                    "mode": runtime.mode,
                    "filter_name": filter_name,
                    "expected_index": expected_index,
                    "expected_table": table,
                    "query_id": query_id,
                    "query_table": query_table,
                    "self_excluded": self_exclusion,
                    "candidate_validity_predicate": effective_candidate_validity_predicate(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                    "candidate_validity_predicate_sha256": candidate_validity_sha256(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                    "failure": f"{exc.__class__.__name__}: {exc}",
                }
                args.plan_evidence.append(evidence)
                raise RuntimeError(
                    f"HNSW plan gate failed for mode={runtime.mode} filter={filter_name}: {exc}"
                ) from exc
            evidence.update(
                {
                    "mode": runtime.mode,
                    "filter_name": filter_name,
                    "config": runtime.config,
                    "planner_proof_verified": bool(evidence["passed"]),
                    "d3_initialization": "workload_driven_adaptive",
                    "prebuilt_fragments": 0,
                    "preferred_index_guc": getattr(
                        args, "preferred_index_guc", "hnsw.preferred_index"
                    ),
                    "preferred_index_guc_available": (
                        runtime.preferred_index_current_setting is not None
                    ),
                    "preferred_index_current_setting": runtime.preferred_index_current_setting,
                    "backend_cpu_provenance": runtime.backend_cpu_provenance,
                    "sqlens_runtime_identity": runtime.sqlens_runtime_identity,
                    "candidate_validity_predicate": effective_candidate_validity_predicate(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                    "candidate_validity_predicate_sha256": candidate_validity_sha256(
                        getattr(args, "candidate_validity_predicate", "")
                    ),
                }
            )
            args.plan_evidence.append(evidence)
            if not evidence["passed"]:
                raise RuntimeError(
                    f"HNSW plan gate failed for mode={runtime.mode} filter={filter_name}: "
                    f"{evidence['failure']}"
                )
    finally:
        runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")


def open_mode_runtime(
    args: argparse.Namespace,
    mode: str,
    filters: list[tuple[str, float, str]],
) -> ModeRuntime:
    cache_mb = args.d3_cache_mb if mode == "design1_bloom_bfs_layout_d3" else args.d1_cache_mb
    config = effective_mode_config(args, mode)
    conn = psycopg.connect(pg_config_from_env().conninfo, autocommit=True)
    try:
        cur = conn.cursor()
        cpu_provenance = backend_cpu_provenance(
            cur,
            getattr(args, "backend_cpu_list", None),
        )
        if not hasattr(args, "backend_cpu_evidence"):
            args.backend_cpu_evidence = []
        args.backend_cpu_evidence.append({"mode": mode, **cpu_provenance})
        enforce_backend_cpu_provenance(cpu_provenance)
        runtime_identity = require_exact_sqlens_identity(
            cur,
            args.expected_sqlens_build_id,
            args.expected_vector_so_sha256,
        )
        if not hasattr(args, "runtime_sqlens_identity_evidence"):
            args.runtime_sqlens_identity_evidence = []
        args.runtime_sqlens_identity_evidence.append(
            {"mode": mode, "backend_pid": cpu_provenance["backend_pid"], **runtime_identity}
        )
        ensure_functions(cur)
        if mode_uses_guidance(mode) and not bool(
            getattr(args, "fragment_tracking_prepared", False)
        ):
            ensure_tracking(cur, args.insertion_table, args.bfs_table)
        configure(cur, args, cache_mb, mode, config)
        _, expected_index = mode_table_index(args, mode)
        preferred_index_current_setting = set_preferred_index_if_supported(
            cur,
            args,
            expected_index,
        )
        if (
            preferred_index_current_setting is None
            and bool(getattr(args, "require_preferred_index_guc", True))
        ):
            raise RuntimeError(
                f"formal index-selection gate requires {args.preferred_index_guc}; "
                "load the SQLens build that exposes the session preferred-index GUC"
            )
        cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
        runtime = ModeRuntime(
            mode=mode,
            config=config,
            cache_mb=cache_mb,
            conn=conn,
            cur=cur,
            preferred_index_current_setting=preferred_index_current_setting,
            backend_cpu_provenance=cpu_provenance,
            sqlens_runtime_identity=runtime_identity,
        )
        plan_query_id = getattr(args, "plan_query_id", None)
        if plan_query_id is not None:
            gate_runtime_plans(args, runtime, filters, int(plan_query_id))
            runtime.planner_proof_verified = True
            # Gate activations must not seed the workload-driven adaptive state.
            cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
            configure(cur, args, cache_mb, mode, config)
            runtime.preferred_index_current_setting = set_preferred_index_if_supported(
                cur,
                args,
                expected_index,
            )
        return runtime
    except Exception:
        conn.close()
        raise


def close_mode_runtime(runtime: ModeRuntime) -> None:
    try:
        runtime.cur.execute("SELECT vector_hnsw_guidance_reset()")
    finally:
        runtime.cur.close()
        runtime.conn.close()


def recover_runtime(args: argparse.Namespace, runtime: ModeRuntime) -> None:
    try:
        runtime.cur.execute("ROLLBACK")
    except Exception:
        pass
    configure(runtime.cur, args, runtime.cache_mb, runtime.mode, runtime.config)
    _, expected_index = mode_table_index(args, runtime.mode)
    runtime.preferred_index_current_setting = set_preferred_index_if_supported(
        runtime.cur,
        args,
        expected_index,
    )


def run_warmup(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
    predicate: str,
    query_id: int,
) -> None:
    evidence: dict[str, object] = {
        "mode": runtime.mode,
        "filter_name": filter_name,
        "query_id": query_id,
        "status": "running",
        "error": "",
    }
    try:
        evidence["guidance_before"] = read_guidance_profile(runtime.cur)
        evidence["cache_before"] = read_cache_profile(runtime.cur)
        activation_profile = activate(runtime.cur, args, runtime.mode, filter_name)
        binding = activation_binding(args, runtime.mode, filter_name, activation_profile)
        candidate_table = str(activation_profile["table"])
        self_exclusion = candidate_self_exclusion(args, candidate_table)
        run_query(
            runtime.cur,
            candidate_table,
            predicate,
            query_id,
            args.k,
            binding,
            uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion,
            candidate_validity_predicate=getattr(
                args, "candidate_validity_predicate", ""
            ),
            query_table=query_table_for_candidate(args, candidate_table),
            query_id_column=getattr(args, "query_id_column", "id"),
            query_vector_column=getattr(args, "query_vector_column", "embedding"),
            self_exclusion=self_exclusion,
        )
        evidence["guidance_after"] = read_guidance_profile(runtime.cur)
        evidence["cache_after"] = read_cache_profile(runtime.cur)
        evidence["status"] = "complete"
        getattr(args, "warmup_evidence").append(evidence)
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = f"{exc.__class__.__name__}: {exc}"
        getattr(args, "warmup_evidence").append(evidence)
        recover_runtime(args, runtime)
        raise RuntimeError(
            f"warmup failed for mode={runtime.mode} filter={filter_name} query_id={query_id}: {exc}"
        ) from exc


def pair_key(filter_name: str, query_no: int, repeat: int) -> str:
    return f"{filter_name}|q{query_no}|r{repeat}"


def guidance_scan_contract_satisfied(
    scan_profile: dict[str, object], filter_strategy: str
) -> bool:
    return guidance_scan_contract_failure(scan_profile, filter_strategy) == ""


def guidance_scan_contract_failure(
    scan_profile: dict[str, object], filter_strategy: str
) -> str:
    if int(scan_profile.get("guidance_checks", 0) or 0) <= 0:
        return "heap validation did not consult the active guide"
    if filter_strategy == "traversal_guided":
        missing = [field for field in SQLENS_TRAVERSAL_PROFILE_FIELDS if field not in scan_profile]
        if missing:
            return f"traversal profile is missing fields: {missing}"
        if scan_profile.get("final_path") != "guided":
            return f"traversal final_path={scan_profile.get('final_path')!r}, expected 'guided'"
        if scan_profile.get("planner_proof_attempted") is not True:
            return "planner proof was not attempted"
        if scan_profile.get("planner_proof_succeeded") is not True:
            return (
                "planner proof failed: "
                f"{scan_profile.get('planner_proof_bypass_reason', 'unknown')}"
            )
        for field in ("stock_bypass_requests", "fallback_requests"):
            if int(scan_profile.get(field, 0) or 0) != 0:
                return f"traversal used a stock bypass/fallback ({field}={scan_profile.get(field)})"
        for field in (
            "stock_phase_expanded_nodes",
            "stock_phase_distance_computations",
            "fallback_stock_expanded_nodes",
            "fallback_stock_distance_computations",
        ):
            if int(scan_profile.get(field, 0) or 0) != 0:
                return f"guided final path contains stock work ({field}={scan_profile.get(field)})"
        pre_distance_checks = int(scan_profile.get("pre_distance_membership_checks", 0) or 0)
        attempted_avoided = int(scan_profile.get("distance_computations_avoided_attempted", 0) or 0)
        avoided = int(scan_profile.get("distance_computations_avoided", 0) or 0)
        neighbor_checks = int(scan_profile.get("neighbor_expansion_guidance_checks", 0) or 0)
        neighbor_matches = int(scan_profile.get("neighbor_expansion_guidance_matches", 0) or 0)
        neighbor_misses = int(scan_profile.get("neighbor_expansion_guidance_misses", 0) or 0)
        guided_admissions = int(scan_profile.get("traversal_guided_admissions", 0) or 0)
        guided_suppressions = int(scan_profile.get("traversal_guided_suppressions", 0) or 0)
        heap_tids_suppressed = int(scan_profile.get("traversal_heap_tids_suppressed", 0) or 0)
        expanded = int(scan_profile.get("guided_expanded_nodes", 0) or 0)
        distance_calls = int(scan_profile.get("guided_phase_distance_computations", 0) or 0)
        total_distance_calls = int(scan_profile.get("distance_compute_count", 0) or 0)
        total_expanded = int(scan_profile.get("traversal_expanded_nodes", 0) or 0)
        try:
            estimated_skip_rate = float(scan_profile["traversal_estimated_skip_rate"])
        except (KeyError, TypeError, ValueError):
            return "traversal skip-rate estimate is missing or invalid"
        if scan_profile.get("traversal_estimated_skip_rate_valid") is not True:
            return "traversal skip-rate estimate was not valid for formal admission"
        if not math.isfinite(estimated_skip_rate) or not 0.0 <= estimated_skip_rate <= 1.0:
            return "traversal skip-rate estimate is outside [0, 1]"
        if scan_profile.get("traversal_guidance_scope") != "candidate_admission_and_validation":
            return "traversal guidance scope is not candidate admission/validation"
        if scan_profile.get("graph_expansion_pruned") is not False:
            return "formal candidate admission must not claim graph-expansion pruning"
        if scan_profile.get("distance_computations_pruned") is not False:
            return "formal candidate admission must not claim distance pruning"
        if pre_distance_checks != 0 or attempted_avoided != 0 or avoided != 0:
            return "candidate admission unexpectedly recorded pre-distance pruning"
        if neighbor_checks <= 0 or neighbor_checks != neighbor_matches + neighbor_misses:
            return "invalid or empty neighbor-expansion membership accounting"
        if guided_admissions <= 0 or guided_suppressions <= 0:
            return "candidate admission recorded no guided admissions or suppressions"
        if heap_tids_suppressed < guided_suppressions:
            return "heap-TID suppression count is smaller than suppressed HNSW elements"
        if expanded <= 0 or distance_calls <= 0:
            return "guided path recorded no expansions or no distance calls"
        if total_expanded < expanded or total_distance_calls < distance_calls:
            return "guided expansion/distance counters exceed total scan counters"
        return ""
    if filter_strategy in {"guided_collect", "acorn1"}:
        if int(scan_profile.get("traversal_guidance_checks", 0) or 0) <= 0:
            return "legacy diagnostic strategy recorded no traversal guidance checks"
        if scan_profile.get("final_path") not in {None, "legacy_guided"}:
            return f"unexpected legacy diagnostic final_path={scan_profile.get('final_path')!r}"
    elif filter_strategy == "safe_guided" and scan_profile.get("final_path") not in {
        None,
        "validation_only",
    }:
        return f"unexpected safe_guided final_path={scan_profile.get('final_path')!r}"
    return ""


def _counter(profile: dict[str, object], field: str) -> int:
    return int(profile.get(field, 0) or 0)


def d3_phase_evidence(
    guidance_before: dict[str, object],
    guidance_after: dict[str, object],
    cache_before: dict[str, object],
    cache_after: dict[str, object],
    activation_profile: dict[str, object],
) -> dict[str, object]:
    state_before = str(guidance_before.get("adaptive_state", "stock"))
    state_after = str(guidance_after.get("adaptive_state", "stock"))
    active_before = bool(guidance_before.get("active", False))
    active_after = bool(guidance_after.get("active", False))
    admissions_before = _counter(guidance_before, "adaptive_admissions")
    admissions_after = _counter(guidance_after, "adaptive_admissions")
    admitted_before = active_before and admissions_before > 0 and state_before not in {
        "stock",
        "probing",
        "rejected",
    }
    admitted_after = active_after and admissions_after > 0 and state_after not in {
        "stock",
        "probing",
        "rejected",
    }
    if active_before and not admitted_before:
        raise RuntimeError(
            "D3 active state before request lacks a prior adaptive admission proof"
        )
    if active_after and not admitted_after:
        raise RuntimeError(
            "D3 active state after request lacks an adaptive admission proof"
        )
    if admitted_before:
        phase = "warm"
    elif admitted_after:
        if admissions_after <= admissions_before:
            raise RuntimeError("D3 admission phase did not advance adaptive_admissions")
        phase = "admission"
    else:
        phase = "cold"
    activation_atoms = _counter(activation_profile, "activation_atom_count")
    reused = bool(
        phase == "warm"
        and admitted_before
        and admitted_after
        and activation_atoms > 0
        and activation_profile.get("guidance_enabled") is True
    )
    if phase == "warm" and not reused:
        raise RuntimeError("D3 warm request did not reuse active admitted guidance")

    fields: dict[str, object] = {
        "d3_phase": phase,
        "d3_state_before": state_before,
        "d3_state_after": state_after,
        "d3_active_before": active_before,
        "d3_active_after": active_after,
        "d3_admitted_before": admitted_before,
        "d3_admitted_after": admitted_after,
        "d3_active_guidance_reused": reused,
    }
    for field in (
        "adaptive_requests",
        "adaptive_probes",
        "adaptive_admissions",
        "adaptive_page_builds",
        "adaptive_bloom_builds",
        "adaptive_refinements",
        "adaptive_rejections",
        "fragment_cache_hits",
        "fragment_store_hits",
        "fragment_builds",
    ):
        before = _counter(guidance_before, field)
        after = _counter(guidance_after, field)
        output = f"d3_{field}"
        fields[f"{output}_before"] = before
        fields[f"{output}_after"] = after
        fields[f"{output}_delta"] = after - before
    for field in ("resident_entries", "resident_bytes", "composed_guide_hits", "evictions"):
        before = _counter(cache_before, field)
        after = _counter(cache_after, field)
        output = f"d3_cache_{field}" if not field.startswith("composed") else "d3_composed_guide_hits"
        fields[f"{output}_before"] = before
        fields[f"{output}_after"] = after
        fields[f"{output}_delta"] = after - before
    fields["d3_cache_reuse_observed"] = bool(
        int(fields["d3_fragment_cache_hits_delta"]) > 0
        or int(fields["d3_fragment_store_hits_delta"]) > 0
        or int(fields["d3_composed_guide_hits_delta"]) > 0
    )
    return fields


def run_measured_query(
    args: argparse.Namespace,
    runtime: ModeRuntime,
    filter_name: str,
    selectivity: float,
    predicate: str,
    query_no: int,
    query_id: int,
    repeat: int,
    truth: dict[tuple[str, int], TruthEntry],
    schedule_position: int,
    block_no: int = 0,
    query_order_position: int = 0,
) -> dict[str, object]:
    mode = runtime.mode
    error = ""
    ids: list[int] = []
    distances: list[float] = []
    activation_profile: dict[str, object] = {}
    scan_profile: dict[str, object] = {}
    cache_profile: dict[str, object] = {}
    guidance_before: dict[str, object] = {}
    guidance_after: dict[str, object] = {}
    cache_before: dict[str, object] = {}
    d3_evidence: dict[str, object] = {}
    activation_ms = 0.0
    query_ms = 0.0
    end_to_end_ms = 0.0
    error_detail = ""
    table, index = mode_table_index(args, mode)
    query_table = query_table_for_candidate(args, table)
    self_exclusion = candidate_self_exclusion(args, table)
    client_self_exclusion = (
        uses_exact_predicate_scan_contract(args.guidance_filter_strategy) and self_exclusion
    )
    try:
        runtime.cur.execute("SELECT vector_hnsw_reset_scan_profile()")
        guidance_before = read_guidance_profile(runtime.cur)
        cache_before = read_cache_profile(runtime.cur)
        e2e_started = time.perf_counter()
        activation_profile = activate(
            runtime.cur,
            args,
            mode,
            filter_name,
            read_profile=False,
        )
        activation_completed = time.perf_counter()
        table = str(activation_profile["table"])
        index = str(activation_profile["index"])
        binding = activation_binding(args, mode, filter_name, activation_profile)
        ids, distances, query_metadata = run_query(
            runtime.cur,
            table,
            predicate,
            query_id,
            args.k,
            binding,
            client_self_exclusion,
            candidate_validity_predicate=getattr(
                args, "candidate_validity_predicate", ""
            ),
            query_table=query_table,
            query_id_column=getattr(args, "query_id_column", "id"),
            query_vector_column=getattr(args, "query_vector_column", "embedding"),
            self_exclusion=self_exclusion,
            reset_profile=False,
            read_profile=False,
        )
        query_completed = time.perf_counter()
        activation_ms = (activation_completed - e2e_started) * 1000.0
        query_ms = (query_completed - activation_completed) * 1000.0
        end_to_end_ms = (query_completed - e2e_started) * 1000.0
        scan_profile = read_scan_profile(runtime.cur)
        scan_profile.update(query_metadata)
        guidance_after = read_guidance_profile(runtime.cur)
        cache_profile = read_cache_profile(runtime.cur)
        activation_profile = {**guidance_after, **activation_profile}
        if (
            args.guidance_filter_strategy == "traversal_guided"
            and mode in {"design1_bloom", "design1_bloom_bfs_layout"}
            and not activation_profile.get("guidance_enabled")
        ):
            raise RuntimeError(
                f"formal {mode} measurement disabled traversal guidance: "
                f"{activation_profile.get('guidance_route', 'unknown')}"
            )
        if mode == "design1_bloom_bfs_layout_d3":
            d3_evidence = d3_phase_evidence(
                guidance_before,
                guidance_after,
                cache_before,
                cache_profile,
                activation_profile,
            )
            getattr(args, "d3_phase_evidence", []).append(
                {
                    "filter_name": filter_name,
                    "query_no": query_no,
                    "repeat": repeat,
                    **d3_evidence,
                }
            )
        if activation_profile.get("guidance_enabled"):
            contract_failure = guidance_scan_contract_failure(
                scan_profile, args.guidance_filter_strategy
            )
            if contract_failure:
                raise RuntimeError(
                    "active guidance did not execute the required measured HNSW path: "
                    + contract_failure
                )
    except errors.QueryCanceled as exc:
        error = exc.__class__.__name__
        error_detail = str(exc)
        recover_runtime(args, runtime)
    except Exception as exc:  # noqa: BLE001
        error = exc.__class__.__name__
        error_detail = str(exc)
        recover_runtime(args, runtime)

    truth_entry = truth[(filter_name, query_no)]
    return {
        "selectivity": selectivity,
        "filter_name": filter_name,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "table": table,
        "index": index,
        "query_table": query_table,
        "query_id_column": getattr(args, "query_id_column", "id"),
        "query_vector_column": getattr(args, "query_vector_column", "embedding"),
        "candidate_validity_predicate": effective_candidate_validity_predicate(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "candidate_validity_predicate_sha256": candidate_validity_sha256(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "d2_page_access": args.d2_page_access if mode_uses_d2(mode) else "off",
        "d2_index_page_access": args.d2_index_page_access if mode_uses_d2(mode) else "off",
        "preferred_index_guc": getattr(args, "preferred_index_guc", "hnsw.preferred_index"),
        "preferred_index_current_setting": runtime.preferred_index_current_setting or "",
        "backend_pid": int((runtime.backend_cpu_provenance or {}).get("backend_pid", 0)),
        "backend_cpu_requested": str(
            (runtime.backend_cpu_provenance or {}).get("requested_cpu_list", "")
        ),
        "backend_cpu_observed": str(
            (runtime.backend_cpu_provenance or {}).get("observed_cpu_list", "")
        ),
        "backend_cpu_exact_match": (
            (runtime.backend_cpu_provenance or {}).get("exact_match")
        ),
        "backend_cpu_pinning_attempted_by_runner": False,
        "ef_search": runtime.config["ef_search"],
        "max_scan_tuples": runtime.config["max_scan_tuples"],
        "scan_mem_multiplier": runtime.config["scan_mem_multiplier"],
        "iterative_scan": runtime.config["iterative_scan"],
        "guided_collect_target": runtime.config["guided_collect_target"],
        "pair_key": pair_key(filter_name, query_no, repeat),
        "block_no": block_no,
        "query_order_position": query_order_position,
        "execution_order": getattr(args, "execution_order", "mode_major"),
        "schedule_seed": getattr(args, "schedule_seed", 20260718),
        "schedule_position": schedule_position,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "k": args.k,
        "scan_limit": (
            args.k + 1
            if client_self_exclusion
            else args.k
        ),
        "self_exclusion_contract": (
            "limit_k_plus_1_client_remove_query_id"
            if client_self_exclusion
            else "sql_residual_id_not_equal" if self_exclusion else "none_external_query_source"
        ),
        "recall": tie_aware_recall(distances, truth_entry, args.k) if not error else 0.0,
        "recall_contract": "distance_squared_threshold_tie_aware_v1",
        "truth_filtered_rows": truth_entry.filtered_rows,
        "truth_kth_distance_sq": truth_entry.kth_distance_sq,
        "truth_tie_tolerance": truth_entry.tie_tolerance,
        "truth_strict_closer_count": truth_entry.strict_closer_count,
        "truth_boundary_tied": truth_entry.boundary_tied,
        "truth_self_excluded": truth_entry.self_excluded,
        "truth_candidate_validity_predicate": effective_candidate_validity_predicate(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "truth_candidate_validity_predicate_sha256": candidate_validity_sha256(
            getattr(args, "candidate_validity_predicate", "")
        ),
        "activation_ms": activation_ms,
        "query_latency_ms": query_ms,
        "end_to_end_ms": end_to_end_ms,
        "guidance_enabled": bool(activation_profile.get("guidance_enabled", mode != "original")),
        "guidance_scan_verified": (
            not bool(activation_profile.get("guidance_enabled", mode != "original"))
            or guidance_scan_contract_satisfied(scan_profile, args.guidance_filter_strategy)
        ),
        # Kept for CSV consumers that predate the clearer scan-specific name.
        "guidance_binding_verified": (
            not bool(activation_profile.get("guidance_enabled", mode != "original"))
            or guidance_scan_contract_satisfied(scan_profile, args.guidance_filter_strategy)
        ),
        "planner_proof_verified": runtime.planner_proof_verified,
        "guidance_route": str(activation_profile.get("guidance_route", "")),
        "activation_atom_count": activation_profile.get("activation_atom_count", 0),
        "d3_active_guidance_reused": bool(
            d3_evidence.get("d3_active_guidance_reused", False)
        ),
        **d3_evidence,
        "d3_initialization": "workload_driven_adaptive",
        "prebuilt_fragments": 0,
        "warmup_all_queries": bool(getattr(args, "warmup_all_queries", False)),
        "guidance_filter_strategy": args.guidance_filter_strategy,
        "vector_search_ms": scan_profile.get("vector_search_ms", 0.0),
        "visited_tuples": scan_profile.get("visited_tuples", 0),
        "returned_tuples": scan_profile.get("returned_tuples", 0),
        "distance_compute_count": scan_profile.get("distance_compute_count", 0),
        "page_access_batches": scan_profile.get("page_access_batches", 0),
        "page_access_candidates": scan_profile.get("page_access_candidates", 0),
        "page_access_prefetches": scan_profile.get("page_access_prefetches", 0),
        "page_access_distinct_pages": scan_profile.get("page_access_distinct_pages", 0),
        "index_page_prefetches": scan_profile.get("index_page_prefetches", 0),
        "index_page_loads": scan_profile.get("index_page_loads", 0),
        "index_page_runs": scan_profile.get("index_page_runs", 0),
        "index_page_distinct_pages": scan_profile.get("index_page_distinct_pages", 0),
        "index_page_distinct_pages_exact": scan_profile.get("index_page_distinct_pages_exact", False),
        "index_page_profile_scope": scan_profile.get("index_page_profile_scope", ""),
        "heap_tid_returns": scan_profile.get("heap_tid_returns", 0),
        "heap_tid_page_runs": scan_profile.get("heap_tid_page_runs", 0),
        "heap_tid_distinct_pages": scan_profile.get("heap_tid_distinct_pages", 0),
        "heap_tid_distinct_pages_exact": scan_profile.get("heap_tid_distinct_pages_exact", False),
        "heap_tid_sequence_scope": scan_profile.get("heap_tid_sequence_scope", ""),
        "idx_blks_hit": scan_profile.get("idx_blks_hit", 0),
        "idx_blks_read": scan_profile.get("idx_blks_read", 0),
        "heap_blks_hit": scan_profile.get("heap_blks_hit", 0),
        "heap_blks_read": scan_profile.get("heap_blks_read", 0),
        "heap_blks_are_exact_heap_io": scan_profile.get("heap_blks_are_exact_heap_io", True),
        "guidance_checks": scan_profile.get("guidance_checks", 0),
        "guidance_skips": scan_profile.get("guidance_skips", 0),
        "traversal_expanded_nodes": scan_profile.get("traversal_expanded_nodes", 0),
        "traversal_neighbors_examined": scan_profile.get("traversal_neighbors_examined", 0),
        "traversal_guidance_checks": scan_profile.get("traversal_guidance_checks", 0),
        "traversal_guidance_matches": scan_profile.get("traversal_guidance_matches", 0),
        "traversal_guidance_misses": scan_profile.get("traversal_guidance_misses", 0),
        "traversal_matching_expanded": scan_profile.get("traversal_matching_expanded", 0),
        "traversal_bridge_expanded": scan_profile.get("traversal_bridge_expanded", 0),
        "traversal_candidate_admissions": scan_profile.get("traversal_candidate_admissions", 0),
        "traversal_result_admissions": scan_profile.get("traversal_result_admissions", 0),
        "traversal_guided_admissions": scan_profile.get("traversal_guided_admissions", 0),
        "traversal_guided_suppressions": scan_profile.get("traversal_guided_suppressions", 0),
        "traversal_heap_tids_suppressed": scan_profile.get("traversal_heap_tids_suppressed", 0),
        "traversal_stop_deferrals": scan_profile.get("traversal_stop_deferrals", 0),
        "traversal_discarded_pushes": scan_profile.get("traversal_discarded_pushes", 0),
        "traversal_discarded_pops": scan_profile.get("traversal_discarded_pops", 0),
        "traversal_initial_batches": scan_profile.get("traversal_initial_batches", 0),
        "traversal_resume_batches": scan_profile.get("traversal_resume_batches", 0),
        "traversal_strict_order_drops": scan_profile.get("traversal_strict_order_drops", 0),
        "traversal_stock_terminations": scan_profile.get("traversal_stock_terminations", 0),
        "traversal_max_scan_terminations": scan_profile.get("traversal_max_scan_terminations", 0),
        "traversal_exhausted_terminations": scan_profile.get("traversal_exhausted_terminations", 0),
        "neighbor_expansion_guidance_checks": scan_profile.get("neighbor_expansion_guidance_checks", 0),
        "neighbor_expansion_guidance_matches": scan_profile.get("neighbor_expansion_guidance_matches", 0),
        "neighbor_expansion_guidance_misses": scan_profile.get("neighbor_expansion_guidance_misses", 0),
        "pre_distance_membership_checks": scan_profile.get("pre_distance_membership_checks", 0),
        "pre_distance_membership_matches": scan_profile.get("pre_distance_membership_matches", 0),
        "pre_distance_membership_misses": scan_profile.get("pre_distance_membership_misses", 0),
        "distance_computations_avoided_attempted": scan_profile.get("distance_computations_avoided_attempted", 0),
        "distance_computations_avoided": scan_profile.get("distance_computations_avoided", 0),
        "guided_expanded_nodes": scan_profile.get("guided_expanded_nodes", 0),
        "guided_phase_distance_computations": scan_profile.get("guided_phase_distance_computations", 0),
        "stock_phase_expanded_nodes": scan_profile.get("stock_phase_expanded_nodes", 0),
        "stock_phase_distance_computations": scan_profile.get("stock_phase_distance_computations", 0),
        "stock_bypass_requests": scan_profile.get("stock_bypass_requests", 0),
        "stock_bypass_reason": scan_profile.get("stock_bypass_reason", ""),
        "fallback_requests": scan_profile.get("fallback_requests", 0),
        "fallback_reason": scan_profile.get("fallback_reason", ""),
        "fallback_stock_expanded_nodes": scan_profile.get("fallback_stock_expanded_nodes", 0),
        "fallback_stock_distance_computations": scan_profile.get("fallback_stock_distance_computations", 0),
        "traversal_estimated_skip_rate_valid": scan_profile.get("traversal_estimated_skip_rate_valid", False),
        "traversal_estimated_skip_rate": scan_profile.get("traversal_estimated_skip_rate", 0.0),
        "traversal_guidance_scope": scan_profile.get("traversal_guidance_scope", ""),
        "graph_expansion_pruned": scan_profile.get("graph_expansion_pruned", False),
        "distance_computations_pruned": scan_profile.get("distance_computations_pruned", False),
        "final_path": scan_profile.get("final_path", ""),
        "planner_proof_attempted": scan_profile.get("planner_proof_attempted", False),
        "planner_proof_succeeded": scan_profile.get("planner_proof_succeeded", False),
        "planner_proof_bypass_reason": scan_profile.get("planner_proof_bypass_reason", ""),
        "fragment_cache_hits": activation_profile.get("fragment_cache_hits", 0),
        "fragment_cache_misses": activation_profile.get("fragment_cache_misses", 0),
        "fragment_store_hits": activation_profile.get("fragment_store_hits", 0),
        "fragment_builds": activation_profile.get("fragment_builds", 0),
        "composed_guide_hit": activation_profile.get("composed_guide_hit", False),
        "activation_build_ms": activation_profile.get("last_cache_build_ms", 0.0),
        "activation_memory_bytes": activation_profile.get("last_cache_memory_bytes", 0),
        "cache_resident_bytes": cache_profile.get("resident_bytes", 0),
        "cache_resident_entries": cache_profile.get("resident_entries", 0),
        "cache_evictions": cache_profile.get("evictions", 0),
        "composed_guide_entries": cache_profile.get("composed_guide_entries", 0),
        "composed_guide_hits_total": cache_profile.get("composed_guide_hits", 0),
        "adaptive_state": activation_profile.get("adaptive_state", "stock"),
        "adaptive_requests": activation_profile.get("adaptive_requests", 0),
        "adaptive_probes": activation_profile.get("adaptive_probes", 0),
        "adaptive_admissions": activation_profile.get("adaptive_admissions", 0),
        "adaptive_page_builds": activation_profile.get("adaptive_page_builds", 0),
        "adaptive_bloom_builds": activation_profile.get("adaptive_bloom_builds", 0),
        "adaptive_refinements": activation_profile.get("adaptive_refinements", 0),
        "adaptive_rejections": activation_profile.get("adaptive_rejections", 0),
        "adaptive_bytes": activation_profile.get("adaptive_bytes", cache_profile.get("adaptive_bytes", 0)),
        "adaptive_score": activation_profile.get("adaptive_score", cache_profile.get("adaptive_score", 0.0)),
        "sqlens_build_id": str(
            (runtime.sqlens_runtime_identity or {}).get("observed_build_id", "")
        ),
        "vector_so_sha256": str(
            (runtime.sqlens_runtime_identity or {}).get("observed_vector_so_sha256", "")
        ),
        "returned": len(ids),
        "raw_returned_before_self_exclusion": scan_profile.get(
            "sqlens_raw_returned_before_self_exclusion", len(ids)
        ),
        "ids": ",".join(str(x) for x in ids),
        "result_distances": json.dumps(distances, separators=(",", ":")),
        "error": error,
        "error_detail": error_detail,
    }


def print_progress(
    rows: list[dict[str, object]],
    mode: str,
    filter_name: str,
    query_index: int,
    query_count: int,
) -> None:
    ok = [r for r in rows if r["mode"] == mode and r["filter_name"] == filter_name and not r["error"]]
    if ok:
        print(
            f"progress mode={mode} filter={filter_name} queries={query_index}/{query_count} "
            f"e2e={statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}ms",
            flush=True,
        )


def run_mode(
    args: argparse.Namespace,
    mode: str,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], TruthEntry],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    runtime = open_mode_runtime(args, mode, filters)
    try:
        warm_nos = query_nos if args.warmup_all_queries else query_nos[: args.warmup_queries]
        if mode != "design1_bloom_bfs_layout_d3":
            for filter_name, _, predicate in filters:
                for qno in warm_nos:
                    run_warmup(args, runtime, filter_name, predicate, query_by_no[qno])

        schedule_position = getattr(args, "modes", [mode]).index(mode) + 1
        for filter_name, selectivity, predicate in filters:
            for idx, qno in enumerate(query_nos, start=1):
                for repeat in range(args.repeats):
                    rows.append(
                        run_measured_query(
                            args,
                            runtime,
                            filter_name,
                            selectivity,
                            predicate,
                            qno,
                            query_by_no[qno],
                            repeat,
                            truth,
                            schedule_position,
                        )
                    )
                if args.progress_queries and idx % args.progress_queries == 0:
                    print_progress(rows, mode, filter_name, idx, len(query_nos))
    finally:
        close_mode_runtime(runtime)
    return rows


def run_interleaved(
    args: argparse.Namespace,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
    query_by_no: dict[int, int],
    truth: dict[tuple[str, int], TruthEntry],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    runtimes: dict[str, ModeRuntime] = {}
    warmup_block = 0
    measured_block = 0
    try:
        for mode in args.modes:
            runtimes[mode] = open_mode_runtime(args, mode, filters)

        warm_nos = query_nos if args.warmup_all_queries else query_nos[: args.warmup_queries]
        for filter_name, _, predicate in filters:
            for qno in warm_nos:
                warmup_modes = [
                    mode for mode in args.modes if mode != "design1_bloom_bfs_layout_d3"
                ]
                for mode in balanced_mode_order(warmup_modes, warmup_block, args.schedule_seed):
                    run_warmup(args, runtimes[mode], filter_name, predicate, query_by_no[qno])
                warmup_block += 1

        for filter_no, (filter_name, selectivity, predicate) in enumerate(filters):
            completed_queries = 0
            for repeat in range(args.repeats):
                repeat_query_nos = list(query_nos)
                random.Random(args.schedule_seed + 1009 * filter_no + 104729 * repeat).shuffle(repeat_query_nos)
                for query_position, qno in enumerate(repeat_query_nos, start=1):
                    mode_order = balanced_mode_order(args.modes, measured_block, args.schedule_seed)
                    for position, mode in enumerate(mode_order, start=1):
                        rows.append(
                            run_measured_query(
                                args,
                                runtimes[mode],
                                filter_name,
                                selectivity,
                                predicate,
                                qno,
                                query_by_no[qno],
                                repeat,
                                truth,
                                position,
                                measured_block,
                                query_position,
                            )
                        )
                    measured_block += 1
                    completed_queries += 1
                if args.progress_queries and completed_queries % args.progress_queries == 0:
                    for mode in args.modes:
                        print_progress(rows, mode, filter_name, completed_queries, len(query_nos) * args.repeats)
    finally:
        for runtime in reversed(list(runtimes.values())):
            close_mode_runtime(runtime)
    return rows


def validate_execution_lifecycle(
    args: argparse.Namespace,
    filters: list[tuple[str, float, str]],
    query_nos: list[int],
) -> dict[str, object]:
    backend_evidence = getattr(args, "backend_cpu_evidence", [])
    if len(backend_evidence) != len(args.modes) or any(
        int(item.get("backend_pid") or 0) <= 0
        or not item.get("observed_cpu_list")
        or item.get("pinning_attempted_by_runner") is not False
        for item in backend_evidence
    ):
        raise RuntimeError(
            "production backend CPU provenance is incomplete: "
            f"expected {len(args.modes)}, observed {len(backend_evidence)}"
        )
    requested_cpu_list = str(getattr(args, "backend_cpu_list", None) or "")
    if requested_cpu_list and any(
        item.get("exact_match") is not True for item in backend_evidence
    ):
        raise RuntimeError("one or more production backends failed the requested CPU affinity gate")
    runtime_identities = getattr(args, "runtime_sqlens_identity_evidence", [])
    if len(runtime_identities) != len(args.modes) or any(
        item.get("exact_match") is not True
        or item.get("expected_build_id") != args.expected_sqlens_build_id
        or item.get("expected_vector_so_sha256") != args.expected_vector_so_sha256
        for item in runtime_identities
    ):
        raise RuntimeError(
            "production backend SQLens identity evidence is incomplete or mismatched"
        )
    warm_query_count = len(query_nos) if args.warmup_all_queries else min(
        len(query_nos), args.warmup_queries
    )
    warm_modes = [
        mode for mode in args.modes if mode != "design1_bloom_bfs_layout_d3"
    ]
    expected_warmups = len(filters) * warm_query_count * len(warm_modes)
    warmup_evidence = getattr(args, "warmup_evidence", [])
    if len(warmup_evidence) != expected_warmups or any(
        item.get("status") != "complete" for item in warmup_evidence
    ):
        raise RuntimeError(
            "warmup evidence is incomplete or failed: "
            f"expected {expected_warmups}, observed {len(warmup_evidence)}"
        )

    phase_counts: dict[str, dict[str, int]] = {}
    d3_evidence = getattr(args, "d3_phase_evidence", [])
    if "design1_bloom_bfs_layout_d3" in args.modes:
        expected_d3_requests = len(filters) * len(query_nos) * args.repeats
        if len(d3_evidence) != expected_d3_requests:
            raise RuntimeError(
                "D3 phase evidence is incomplete: "
                f"expected {expected_d3_requests}, observed {len(d3_evidence)}"
            )
        for filter_name, _, _ in filters:
            counts = {phase: 0 for phase in ("cold", "admission", "warm")}
            for item in d3_evidence:
                if item.get("filter_name") == filter_name:
                    phase = str(item.get("d3_phase") or "")
                    if phase in counts:
                        counts[phase] += 1
            if any(counts[phase] <= 0 for phase in ("cold", "admission", "warm")):
                raise RuntimeError(
                    f"D3 lifecycle is incomplete for filter={filter_name}: {counts}"
                )
            phase_counts[filter_name] = counts
    return {
        "warmup_policy": "d3_admission_preserved_in_measured_requests",
        "backend_cpu_requested": requested_cpu_list,
        "backend_cpu_evidence_count": len(backend_evidence),
        "backend_cpu_provenance_complete": True,
        "runtime_sqlens_identity_evidence_count": len(runtime_identities),
        "runtime_sqlens_identity_complete": True,
        "warmup_expected": expected_warmups,
        "warmup_observed": len(warmup_evidence),
        "warmup_complete": True,
        "d3_expected_measured_requests": (
            len(filters) * len(query_nos) * args.repeats
            if "design1_bloom_bfs_layout_d3" in args.modes
            else 0
        ),
        "d3_phase_counts": phase_counts,
        "d3_lifecycle_complete": (
            "design1_bloom_bfs_layout_d3" not in args.modes or bool(phase_counts)
        ),
    }


def write_summary(rows: list[dict[str, object]], out: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filter_name"]), str(row["mode"])), []).append(row)

    mode_mean: dict[tuple[str, str], float] = {}
    for key, items in grouped.items():
        ok = [r for r in items if not r["error"]]
        mode_mean[key] = statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0

    table_out = out.with_name(out.stem + "_table.csv")
    fields = [
        "Selectivity",
        "Filter",
        "Original pgvector",
        "Design 1",
        "Design 1 + Design 2",
        "Design 1 + Design 2 + Design 3",
        "D1 speedup",
        "D1+D2 speedup",
        "D1+D2 + D3 speedup",
    ]
    with table_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        seen_filters = []
        for row in rows:
            key = (str(row["filter_name"]), str(row["selectivity"]))
            if key not in seen_filters:
                seen_filters.append(key)
        for filter_name, selectivity in seen_filters:
            if (filter_name, "original") not in mode_mean:
                continue
            original = mode_mean[(filter_name, "original")]
            d1 = mode_mean.get((filter_name, "design1_bloom"), 0.0)
            d12 = mode_mean.get((filter_name, "design1_bloom_bfs_layout"), 0.0)
            d123 = mode_mean.get((filter_name, "design1_bloom_bfs_layout_d3"), 0.0)
            writer.writerow(
                {
					"Selectivity": str(selectivity),
                    "Filter": filter_name,
                    "Original pgvector": f"{original:.4f}",
                    "Design 1": f"{d1:.4f}",
                    "Design 1 + Design 2": f"{d12:.4f}",
                    "Design 1 + Design 2 + Design 3": f"{d123:.4f}",
                    "D1 speedup": f"{(original / d1):.4f}" if d1 else "0.0000",
                    "D1+D2 speedup": f"{(original / d12):.4f}" if d12 else "0.0000",
                    "D1+D2 + D3 speedup": f"{(original / d123):.4f}" if d123 else "0.0000",
                }
            )

    profile_out = out.with_name(out.stem + "_profile_summary.csv")
    profile_fields = [
        "filter_name",
        "mode",
        "ok",
        "errors",
        "recall_mean",
        "end_to_end_mean_ms",
        "activation_mean_ms",
        "query_latency_mean_ms",
        "guidance_enabled_rate",
        "cache_resident_bytes_max",
        "fragment_cache_hits_mean",
        "fragment_store_hits_mean",
        "fragment_builds_mean",
        "composed_guide_hit_rate",
        "guidance_skip_rate",
        "traversal_expanded_nodes_mean",
        "traversal_neighbors_examined_mean",
        "traversal_guidance_checks_mean",
        "traversal_guidance_match_rate",
        "traversal_matching_expanded_mean",
        "traversal_bridge_expanded_mean",
        "traversal_candidate_admissions_mean",
        "traversal_result_admissions_mean",
        "traversal_guided_admissions_mean",
        "traversal_guided_suppressions_mean",
        "traversal_heap_tids_suppressed_mean",
        "traversal_stop_deferrals_mean",
        "traversal_discarded_pushes_mean",
        "traversal_discarded_pops_mean",
        "traversal_initial_batches_mean",
        "traversal_resume_batches_mean",
        "traversal_strict_order_drops_mean",
        "guided_final_path_rate",
        "planner_proof_success_rate",
        "pre_distance_membership_checks_mean",
        "pre_distance_membership_misses_mean",
        "distance_computations_avoided_mean",
        "guided_expanded_nodes_mean",
        "guided_phase_distance_computations_mean",
        "stock_bypass_requests_mean",
        "fallback_requests_mean",
        "index_page_loads_mean",
        "index_page_runs_mean",
        "index_page_distinct_pages_mean",
        "index_page_distinct_pages_exact_rate",
        "heap_tid_returns_mean",
        "heap_tid_page_runs_mean",
        "heap_tid_distinct_pages_mean",
        "heap_tid_distinct_pages_exact_rate",
        "idx_blks_hit_mean",
        "idx_blks_read_mean",
        "heap_blks_hit_mean",
        "heap_blks_read_mean",
        "heap_blks_exact_io_claim_rate",
    ]
    with profile_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fields)
        writer.writeheader()
        for (filter_name, mode), items in sorted(grouped.items()):
            ok = [r for r in items if not r["error"]]
            checks = statistics.fmean(float(r["guidance_checks"]) for r in ok) if ok else 0.0
            skips = statistics.fmean(float(r["guidance_skips"]) for r in ok) if ok else 0.0
            traversal_checks = statistics.fmean(float(r["traversal_guidance_checks"]) for r in ok) if ok else 0.0
            traversal_matches = statistics.fmean(float(r["traversal_guidance_matches"]) for r in ok) if ok else 0.0
            writer.writerow(
                {
                    "filter_name": filter_name,
                    "mode": mode,
                    "ok": len(ok),
                    "errors": len(items) - len(ok),
                    "recall_mean": statistics.fmean(float(r["recall"]) for r in ok) if ok else 0.0,
                    "end_to_end_mean_ms": statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0,
                    "activation_mean_ms": statistics.fmean(float(r["activation_ms"]) for r in ok) if ok else 0.0,
                    "query_latency_mean_ms": statistics.fmean(float(r["query_latency_ms"]) for r in ok) if ok else 0.0,
                    "guidance_enabled_rate": statistics.fmean(float(r["guidance_enabled"]) for r in ok) if ok else 0.0,
                    "cache_resident_bytes_max": max((int(r["cache_resident_bytes"]) for r in ok), default=0),
                    "fragment_cache_hits_mean": statistics.fmean(float(r["fragment_cache_hits"]) for r in ok) if ok else 0.0,
                    "fragment_store_hits_mean": statistics.fmean(float(r["fragment_store_hits"]) for r in ok) if ok else 0.0,
                    "fragment_builds_mean": statistics.fmean(float(r["fragment_builds"]) for r in ok) if ok else 0.0,
                    "composed_guide_hit_rate": statistics.fmean(1.0 if r["composed_guide_hit"] else 0.0 for r in ok) if ok else 0.0,
                    "guidance_skip_rate": skips / checks if checks else 0.0,
                    "traversal_expanded_nodes_mean": statistics.fmean(float(r["traversal_expanded_nodes"]) for r in ok) if ok else 0.0,
                    "traversal_neighbors_examined_mean": statistics.fmean(float(r["traversal_neighbors_examined"]) for r in ok) if ok else 0.0,
                    "traversal_guidance_checks_mean": traversal_checks,
                    "traversal_guidance_match_rate": traversal_matches / traversal_checks if traversal_checks else 0.0,
                    "traversal_matching_expanded_mean": statistics.fmean(float(r["traversal_matching_expanded"]) for r in ok) if ok else 0.0,
                    "traversal_bridge_expanded_mean": statistics.fmean(float(r["traversal_bridge_expanded"]) for r in ok) if ok else 0.0,
                    "traversal_candidate_admissions_mean": statistics.fmean(float(r["traversal_candidate_admissions"]) for r in ok) if ok else 0.0,
                    "traversal_result_admissions_mean": statistics.fmean(float(r["traversal_result_admissions"]) for r in ok) if ok else 0.0,
                    "traversal_guided_admissions_mean": statistics.fmean(float(r["traversal_guided_admissions"]) for r in ok) if ok else 0.0,
                    "traversal_guided_suppressions_mean": statistics.fmean(float(r["traversal_guided_suppressions"]) for r in ok) if ok else 0.0,
                    "traversal_heap_tids_suppressed_mean": statistics.fmean(float(r["traversal_heap_tids_suppressed"]) for r in ok) if ok else 0.0,
                    "traversal_stop_deferrals_mean": statistics.fmean(float(r["traversal_stop_deferrals"]) for r in ok) if ok else 0.0,
                    "traversal_discarded_pushes_mean": statistics.fmean(float(r["traversal_discarded_pushes"]) for r in ok) if ok else 0.0,
                    "traversal_discarded_pops_mean": statistics.fmean(float(r["traversal_discarded_pops"]) for r in ok) if ok else 0.0,
                    "traversal_initial_batches_mean": statistics.fmean(float(r["traversal_initial_batches"]) for r in ok) if ok else 0.0,
                    "traversal_resume_batches_mean": statistics.fmean(float(r["traversal_resume_batches"]) for r in ok) if ok else 0.0,
                    "traversal_strict_order_drops_mean": statistics.fmean(float(r["traversal_strict_order_drops"]) for r in ok) if ok else 0.0,
                    "guided_final_path_rate": statistics.fmean(1.0 if r["final_path"] == "guided" else 0.0 for r in ok) if ok else 0.0,
                    "planner_proof_success_rate": statistics.fmean(1.0 if r["planner_proof_succeeded"] else 0.0 for r in ok) if ok else 0.0,
                    "pre_distance_membership_checks_mean": statistics.fmean(float(r["pre_distance_membership_checks"]) for r in ok) if ok else 0.0,
                    "pre_distance_membership_misses_mean": statistics.fmean(float(r["pre_distance_membership_misses"]) for r in ok) if ok else 0.0,
                    "distance_computations_avoided_mean": statistics.fmean(float(r["distance_computations_avoided"]) for r in ok) if ok else 0.0,
                    "guided_expanded_nodes_mean": statistics.fmean(float(r["guided_expanded_nodes"]) for r in ok) if ok else 0.0,
                    "guided_phase_distance_computations_mean": statistics.fmean(float(r["guided_phase_distance_computations"]) for r in ok) if ok else 0.0,
                    "stock_bypass_requests_mean": statistics.fmean(float(r["stock_bypass_requests"]) for r in ok) if ok else 0.0,
                    "fallback_requests_mean": statistics.fmean(float(r["fallback_requests"]) for r in ok) if ok else 0.0,
                    "index_page_loads_mean": statistics.fmean(float(r["index_page_loads"]) for r in ok) if ok else 0.0,
                    "index_page_runs_mean": statistics.fmean(float(r["index_page_runs"]) for r in ok) if ok else 0.0,
                    "index_page_distinct_pages_mean": statistics.fmean(float(r["index_page_distinct_pages"]) for r in ok) if ok else 0.0,
                    "index_page_distinct_pages_exact_rate": statistics.fmean(1.0 if r["index_page_distinct_pages_exact"] else 0.0 for r in ok) if ok else 0.0,
                    "heap_tid_returns_mean": statistics.fmean(float(r["heap_tid_returns"]) for r in ok) if ok else 0.0,
                    "heap_tid_page_runs_mean": statistics.fmean(float(r["heap_tid_page_runs"]) for r in ok) if ok else 0.0,
                    "heap_tid_distinct_pages_mean": statistics.fmean(float(r["heap_tid_distinct_pages"]) for r in ok) if ok else 0.0,
                    "heap_tid_distinct_pages_exact_rate": statistics.fmean(1.0 if r["heap_tid_distinct_pages_exact"] else 0.0 for r in ok) if ok else 0.0,
                    "idx_blks_hit_mean": statistics.fmean(float(r["idx_blks_hit"]) for r in ok) if ok else 0.0,
                    "idx_blks_read_mean": statistics.fmean(float(r["idx_blks_read"]) for r in ok) if ok else 0.0,
                    "heap_blks_hit_mean": statistics.fmean(float(r["heap_blks_hit"]) for r in ok) if ok else 0.0,
                    "heap_blks_read_mean": statistics.fmean(float(r["heap_blks_read"]) for r in ok) if ok else 0.0,
                    "heap_blks_exact_io_claim_rate": statistics.fmean(1.0 if r["heap_blks_are_exact_heap_io"] else 0.0 for r in ok) if ok else 0.0,
                }
            )
    print(f"wrote {table_out}", flush=True)
    print(f"wrote {profile_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Original, D1, D1+D2, and D1+D2+D3 pgvector variants.")
    parser.add_argument("--insertion-table", default=INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument(
        "--query-table",
        help="External query relation. Omit to read each mode's query vector from its candidate table.",
    )
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default="",
        help=(
            "Global candidate validity expression implied by a partial HNSW index, such as "
            "embedding_valid. It is a SQL/planner qual and is never added to guidance atoms."
        ),
    )
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require this exact self_excluded value in every formal truth row.",
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--filters-csv", type=Path)
    parser.add_argument("--modes", nargs="*", choices=MODES, default=MODES)
    parser.add_argument("--execution-order", choices=["mode_major", "interleaved"], default="mode_major")
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument(
        "--mode-configs-json",
        type=parse_mode_configs_json,
        default={},
        help="JSON object or JSON file mapping modes to per-mode search-setting overrides.",
    )
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument(
        "--warmup-all-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one unmeasured pass over every measured query for each filter before recording latency.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--guided-collect-target", type=int, default=1000)
    parser.add_argument(
        "--guidance-filter-strategy",
        default="traversal_guided",
        choices=["traversal_guided", "safe_guided", "guided_collect", "acorn1"],
        help=(
            "Formal D1 uses planner-proven traversal_guided. safe_guided, guided_collect, "
            "and acorn1 are diagnostic strategies and are not traversal-safe D1 results."
        ),
    )
    parser.add_argument("--iterative-scan", default="off", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument(
        "--preferred-index-guc",
        default="hnsw.preferred_index",
        help=(
            "Optional SQLens planner preference GUC. If the loaded C build exposes it, the runner "
            "sets it to the mode's expected index; EXPLAIN remains the fail-closed source of truth."
        ),
    )
    parser.add_argument(
        "--require-preferred-index-guc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail formal runs when the preferred-index GUC is unavailable. Disabling this is "
            "diagnostic only; the exact EXPLAIN Index Name gate still applies."
        ),
    )
    parser.add_argument("--d2-page-window", type=int, default=128)
    parser.add_argument("--d2-page-prefetch-min-items", type=int, default=2)
    parser.add_argument("--d2-page-disable-after-no-merge", type=int, default=2)
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument(
        "--fragment-tracking-prepared",
        action="store_true",
        help=(
            "Assert that the parent prepared fragment epoch tracking before acquiring "
            "its long-lived data guard; child mode sessions must not run tracking DDL."
        ),
    )
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=float,
        default=100.0,
        help="Disable predicate guidance above this filter percentage; D1+D2 then runs as D2-only.",
    )
    parser.add_argument(
        "--guidance-max-atoms",
        type=int,
        default=64,
        help="Disable predicate guidance when a query decomposes into more atoms than this.",
    )
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--reset-cache-per-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--d2-graph-proof-json",
        type=parse_json_object,
        help=(
            "Delegated proof produced by vector_hnsw_graph_compare at the parent formal-runner "
            "startup. Standalone runs compute the proof directly."
        ),
    )
    parser.add_argument("--expected-sqlens-build-id", required=True)
    parser.add_argument("--expected-vector-so-sha256", required=True)
    parser.add_argument(
        "--backend-cpu-list",
        type=normalize_cpu_list,
        help=(
            "Required DB-side Cpus_allowed_list for every production backend. The runner records "
            "pg_backend_pid() but never tasksets a Docker namespace PID."
        ),
    )
    args = parser.parse_args()
    args.candidate_validity_predicate_explicit = (
        "--candidate-validity-predicate" in sys.argv
    )
    args.candidate_validity_predicate = effective_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    args.plan_started_at = utc_now()
    args.plan_evidence_out = args.out.with_suffix(args.out.suffix + ".plan.json")
    args.plan_evidence = []
    args.output_rows = 0
    args.warmup_evidence = []
    args.d3_phase_evidence = []
    args.backend_cpu_evidence = []
    args.runtime_sqlens_identity_evidence = []
    try:
        quoted_column(args.query_id_column)
        quoted_column(args.query_vector_column)
        validate_query_source_contract(args)
        require_sqlens_provenance_from_env()
        args.sqlens_runtime_identity = require_exact_sqlens_identity_from_env(
            args.expected_sqlens_build_id,
            args.expected_vector_so_sha256,
        )
        if args.guidance_filter_strategy == "traversal_guided":
            invalid_modes = [
                mode
                for mode in args.modes
                if mode != "original"
                and effective_mode_config(args, mode)["iterative_scan"] != "off"
            ]
            if invalid_modes:
                raise RuntimeError(
                    "formal traversal_guided measurements require iterative_scan=off; "
                    f"invalid modes: {invalid_modes}"
                )
        if any(mode_uses_d2(mode) for mode in args.modes):
            args.d2_graph_proof = require_d2_graph_proof_from_env(
                args,
                args.d2_graph_proof_json,
            )
        else:
            args.d2_graph_proof = {"required": False}
        truth, query_by_no = load_tie_aware_truth(
            args.truth_csv,
            expected_self_excluded=args.expected_truth_self_excluded,
            expected_candidate_validity_predicate=(
                args.candidate_validity_predicate
                if args.candidate_validity_predicate_explicit
                else None
            ),
        )
        query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
        if len(query_nos) != args.queries:
            raise RuntimeError(f"requested {args.queries} queries, found {len(query_nos)}")
        args.plan_query_id = query_by_no[query_nos[0]]
        all_filters, args.filter_atoms = load_filter_specs(args.filters_csv)
        selected = set(args.filter_names or [])
        filters = [(name, target, pred) for name, target, pred in all_filters if not selected or name in selected]
        if not filters:
            raise RuntimeError("no benchmark filters selected")
        args.filter_selectivity_by_name = {name: parse_pct(target) for name, target, _ in filters}
        args.out.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        if args.execution_order == "interleaved":
            print(f"running interleaved modes={','.join(args.modes)} seed={args.schedule_seed}", flush=True)
            rows = run_interleaved(args, filters, query_nos, query_by_no, truth)
        else:
            for mode in args.modes:
                print(f"running mode={mode}", flush=True)
                rows.extend(run_mode(args, mode, filters, query_nos, query_by_no, truth))
        expected_plan_checks = len(args.modes) * len(filters)
        if len(args.plan_evidence) != expected_plan_checks or not all(
            bool(item.get("passed")) for item in args.plan_evidence
        ):
            raise RuntimeError(
                f"HNSW plan evidence incomplete: expected {expected_plan_checks}, got {len(args.plan_evidence)}"
            )
        args.execution_lifecycle = validate_execution_lifecycle(args, filters, query_nos)
        args.sqlens_runtime_identity_final = require_exact_sqlens_identity_from_env(
            args.expected_sqlens_build_id,
            args.expected_vector_so_sha256,
        )
        if any(mode_uses_d2(mode) for mode in args.modes):
            args.d2_graph_proof_final = require_d2_graph_proof_from_env(
                args,
                args.d2_graph_proof,
            )
        else:
            args.d2_graph_proof_final = {"required": False}

        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        args.output_rows = len(rows)
        print(f"wrote {args.out}", flush=True)
        write_summary(rows, args.out)
        write_plan_evidence(args, "complete")
    except BaseException as exc:
        write_plan_evidence(args, "failed", exc)
        raise


if __name__ == "__main__":
    main()
