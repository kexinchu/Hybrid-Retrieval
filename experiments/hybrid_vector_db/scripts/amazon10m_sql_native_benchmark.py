from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .common_pg import pg_config_from_env, require_psycopg
    from . import amazon10m_sql_native_exact_truth as exact_truth_contract
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env, require_psycopg  # type: ignore[no-redef]
    import amazon10m_sql_native_exact_truth as exact_truth_contract  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_SCHEMA = ROOT / "experiments/hybrid_vector_db/sql/amazon10m_sql_native_schema.sql"
DEFAULT_QUERY_IDS = exact_truth_contract.DEFAULT_QUERY_IDS
DEFAULT_QUERY_COHORT_MANIFEST = exact_truth_contract.DEFAULT_QUERY_COHORT_MANIFEST
DEFAULT_RESULTS = ROOT / "results/hybrid_vector_db"
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_EXACT_TRUTH_DIR = (
    DEFAULT_RESULTS / "amazon10m_sql_native_exact_truth_valid_embeddings"
)
DEFAULT_EXACT_TRUTH_CSV = DEFAULT_EXACT_TRUTH_DIR / "amazon10m_sql_native_exact_truth_q200.csv"
DEFAULT_EXACT_TRUTH_MANIFEST = DEFAULT_EXACT_TRUTH_DIR / "amazon10m_sql_native_exact_truth_manifest.json"
DEFAULT_VECTOR_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_PRINCIPAL = "amazon10m_sql_native_benchmark"
DEFAULT_SOURCE_INDEX = "public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx"
DEFAULT_CLONE_INDEX = (
    "public.amazon_grocery_reviews_10m_pgvector_hnsw_bfs_clone_idx"
)
# Compatibility for callers that imported the old constant. Formal artifacts use
# source_index and clone_index explicitly.
DEFAULT_VECTOR_INDEX = DEFAULT_SOURCE_INDEX
DEFAULT_K = 10
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = (
    exact_truth_contract.DEFAULT_CANDIDATE_VALIDITY_PREDICATE
)
DEFAULT_CALIBRATION_QUERIES = 100
DEFAULT_FINAL_QUERIES = 100
DEFAULT_CALIBRATION_REPEATS = 2
DEFAULT_FINAL_REPEATS = 5
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_D3_PROBE_REQUESTS = 2
DEFAULT_D3_MIN_BENEFIT_PER_BYTE = 0.0
DEFAULT_D3_MAX_FRAGMENT_MB = 16
DEFAULT_D3_PAGE_MIN_SKIP_RATE = 0.05
TARGET_RECALLS = (0.90, 0.95, 0.99)
MODES = ("stock", "d1", "d1_d2", "d1_d2_d3")
SQLENS_MODES = MODES[1:]
NA = "N/A"
SQLENS_BUILD_PREFIX = "sqlens-v11-"
SQLENS_PROFILE_SEMANTICS = 7.0
CHECKPOINT_VERSION = 5
EXACT_TRUTH_ARTIFACT_VERSION = 4
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
SQLENS_PROFILE_EXPORT_FIELDS = SQLENS_PROFILE_FIELDS + (
    "visited_tuples",
    "returned_tuples",
    "distance_compute_count",
    "idx_blks_hit",
    "idx_blks_read",
    "heap_blks_hit",
    "heap_blks_read",
)
TIMING_DEFINITION = (
    "activation_ms and query_ms are diagnostic sub-intervals. e2e_ms is one continuous "
    "client wall-clock interval from per-request as_of/guidance setup through the single "
    "PostgreSQL hybrid SELECT and result transfer; it is not reconstructed by addition. "
    "The SELECT returns an in-executor guidance proof column in every mode. Connection "
    "setup, EXPLAIN and exact-GT generation are outside e2e_ms. Primary selection and "
    "summaries use e2e_ms."
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


@dataclass(frozen=True)
class FilterSpec:
    name: str
    target_rate: str
    predicate: str
    atoms: tuple[str, ...]
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    description: str
    bucket_pct: float
    temporal: bool


@dataclass(frozen=True)
class Config:
    ef_search: int
    max_scan_tuples: int
    scan_mem_multiplier: float
    iterative_scan: str
    guided_collect_target: int

    @property
    def label(self) -> str:
        mem = str(self.scan_mem_multiplier).replace(".", "p")
        return (
            f"ef{self.ef_search}_max{self.max_scan_tuples}_mem{mem}_"
            f"{self.iterative_scan}_target{self.guided_collect_target}"
        )


@dataclass(frozen=True)
class ExactTruth:
    ids: tuple[int, ...]
    kth_distance: float
    tie_tolerance: float
    boundary_tied: bool


@dataclass(frozen=True)
class ModeSpec:
    index_role: str
    filter_strategy: str
    guidance_kind: str | None
    adaptive: bool
    guidance_semantics: str


MODE_SPECS = {
    "stock": ModeSpec("source", "off", None, False, "stock_pgvector"),
    "d1": ModeSpec(
        "source",
        "safe_guided",
        "bloom",
        False,
        "candidate_admission_and_validation_guidance",
    ),
    "d1_d2": ModeSpec(
        "clone",
        "safe_guided",
        "bloom",
        False,
        "candidate_admission_and_validation_guidance_on_same_graph_bfs_clone",
    ),
    "d1_d2_d3": ModeSpec(
        "clone",
        "safe_guided",
        "adaptive",
        True,
        "workload_driven_adaptive_candidate_admission_and_validation_guidance",
    ),
}


def mode_index(mode: str, source_index: str, clone_index: str) -> str:
    try:
        role = MODE_SPECS[mode].index_role
    except KeyError as exc:
        raise ValueError(f"unknown benchmark mode: {mode}") from exc
    return source_index if role == "source" else clone_index


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(text)
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
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


WORKLOADS = (
    WorkloadSpec(
        "acl_only",
        "vector retrieval joined to product dimension and RLS-derived principal ACL",
        50.0,
        False,
    ),
    WorkloadSpec(
        "grant_temporal_selectivity",
        "derived benchmark grant validity at a real review-derived as_of",
        20.0,
        True,
    ),
    WorkloadSpec(
        "fact_temporal_selectivity",
        "source review timestamp validity with RLS and tie-aware real timestamp as_of",
        5.0,
        True,
    ),
)


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


def parse_float_list(value: str, *, minimum: float = 0.0, maximum: float | None = None) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated number list") from exc
    if not parsed or any(item <= minimum or (maximum is not None and item > maximum) for item in parsed):
        raise argparse.ArgumentTypeError("number list contains an out-of-range value")
    return list(dict.fromkeys(parsed))


def parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer list values must be greater than zero")
    return list(dict.fromkeys(parsed))


def parse_word_list(value: str, allowed: set[str] | None = None) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed or (allowed is not None and any(item not in allowed for item in parsed)):
        allowed_text = f"; allowed={sorted(allowed)}" if allowed is not None else ""
        raise argparse.ArgumentTypeError(f"expected a non-empty word list{allowed_text}")
    return list(dict.fromkeys(parsed))


def parse_guided_targets(value: str) -> list[str]:
    parsed = parse_word_list(value)
    if any(item != "ef" and (not item.isdigit() or int(item) <= 0) for item in parsed):
        raise argparse.ArgumentTypeError("guided collect targets must be positive integers or ef")
    return parsed


def parse_qualified_name(value: str) -> tuple[str, ...]:
    parts = tuple(value.split("."))
    if len(parts) not in (1, 2) or any(
        not part or not (part[0].isalpha() or part[0] == "_")
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$" for char in part)
        for part in parts
    ):
        raise argparse.ArgumentTypeError("table names must be unquoted table or schema.table identifiers")
    return tuple(part.lower() for part in parts)


def qualified_name_arg(value: str) -> str:
    return ".".join(parse_qualified_name(value))


def parse_role_name(value: str) -> str:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise argparse.ArgumentTypeError("principal must be a lowercase PostgreSQL role identifier")
    return value


validate_candidate_validity_predicate = (
    exact_truth_contract.validate_candidate_validity_predicate
)
candidate_universe_predicate_sha256 = (
    exact_truth_contract.candidate_universe_predicate_sha256
)
workload_scalar_predicate_sha256 = (
    exact_truth_contract.workload_scalar_predicate_sha256
)
query_cohort_sha256 = exact_truth_contract.query_cohort_sha256
relation_epoch_contract = exact_truth_contract.relation_epoch_contract


def parse_targets(value: str) -> list[float]:
    parsed = sorted(set(parse_float_list(value, minimum=0.0, maximum=1.0)))
    if any(target <= 0.0 or target > 1.0 for target in parsed):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return parsed


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def bootstrap_bounds(values: Sequence[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(list(values), k=len(values))) for _ in range(samples)]
    return percentile(means, 0.05), percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_ratio_bounds(
    stock: dict[int, float], method: dict[int, float], samples: int, seed: int
) -> tuple[float, float, float]:
    keys = sorted(set(stock) & set(method))
    if not keys:
        return 0.0, 0.0, 0.0
    if len(keys) == 1 or samples <= 0:
        ratio = stock[keys[0]] / method[keys[0]] if method[keys[0]] > 0 else 0.0
        return ratio, ratio, ratio
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        sampled = rng.choices(keys, k=len(keys))
        stock_mean = statistics.fmean(stock[key] for key in sampled)
        method_mean = statistics.fmean(method[key] for key in sampled)
        ratios.append(stock_mean / method_mean if method_mean > 0 else 0.0)
    return percentile(ratios, 0.05), percentile(ratios, 0.025), percentile(ratios, 0.975)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise RuntimeError(f"exact-truth artifact has invalid {label} SHA256")
    return normalized


def _csv_ints(value: Any, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(f"exact-truth CSV has invalid {label}") from exc
    if not parsed:
        raise RuntimeError(f"exact-truth CSV has empty {label}")
    return parsed


def _csv_floats(value: Any, label: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(f"exact-truth CSV has invalid {label}") from exc
    if not parsed or any(not math.isfinite(item) or item < 0.0 for item in parsed):
        raise RuntimeError(f"exact-truth CSV has invalid {label}")
    return parsed


def _artifact_error(message: str) -> RuntimeError:
    return RuntimeError(f"exact-truth artifact rejected: {message}")


def _artifact_float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9)


def _artifact_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise _artifact_error("truth CSV has invalid boolean metadata")


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def read_filters(path: Path, selected: set[str] | None = None) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["filter_name"]
            if selected and name not in selected:
                continue
            if name in seen:
                raise ValueError(f"duplicate filter_name: {name}")
            atoms = tuple(part.strip() for part in row["atoms"].split("||") if part.strip())
            if not atoms:
                raise ValueError(f"filter has no SQL atoms: {name}")
            specs.append(
                FilterSpec(
                    name=name,
                    target_rate=row["target_rate"],
                    predicate=row["predicate"].strip(),
                    atoms=atoms,
                    expected_rows=int(row["count"]),
                    actual_pct=float(row["actual_pct"]),
                )
            )
            seen.add(name)
    if selected and selected - seen:
        raise ValueError(f"missing filters: {sorted(selected - seen)}")
    if not specs:
        raise ValueError(f"no filters loaded from {path}")
    return specs


def load_query_ids(
    path: Path,
    offset: int,
    count: int,
    *,
    expected_split: str | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[int, int]:
    candidate_validity_predicate = validate_candidate_validity_predicate(
        candidate_validity_predicate
    )
    wanted = set(range(offset, offset + count))
    found: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "query_no", "query_id", "query_split", "self_excluded", "kth_distance_sq",
            "candidate_validity_predicate", "query_validity_predicate",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"query source uses the retired non-formal truth schema: missing={sorted(missing)}"
            )
        for row in reader:
            if row.get("method") not in (None, "", "pre_filter_exact"):
                continue
            if str(row["self_excluded"]).strip().lower() != "true":
                raise ValueError("query source did not exclude query rows")
            if (
                row.get("candidate_validity_predicate") != candidate_validity_predicate
                or row.get("query_validity_predicate") != candidate_validity_predicate
            ):
                raise ValueError("query source validity universe does not match the benchmark")
            query_no = int(row["query_no"])
            if query_no not in wanted:
                continue
            if expected_split is not None and row.get("query_split") != expected_split:
                raise ValueError(
                    f"query_no={query_no} has query_split={row.get('query_split')!r}; "
                    f"expected {expected_split!r}"
                )
            query_id = int(row["query_id"])
            previous = found.setdefault(query_no, query_id)
            if previous != query_id:
                raise ValueError(f"query_no={query_no} maps to multiple query IDs")
    if set(found) != wanted:
        raise ValueError(f"query split is incomplete: missing={sorted(wanted - set(found))}")
    if len(set(found.values())) != count:
        raise ValueError("query IDs must be unique within a split")
    return dict(sorted(found.items()))


def validate_query_splits(calibration: dict[int, int], final: dict[int, int]) -> None:
    if set(calibration) & set(final):
        raise ValueError("calibration and final query_no sets overlap")
    if set(calibration.values()) & set(final.values()):
        raise ValueError("calibration and final query IDs overlap")


def qualify_predicate(predicate: str, alias: str = "v") -> str:
    result = predicate
    for column in sorted(FILTER_COLUMNS, key=len, reverse=True):
        result = re.sub(rf"(?<![A-Za-z0-9_$.]){re.escape(column)}\b", f"{alias}.{column}", result)
    return result


def build_hybrid_sql(
    table: str,
    predicate: str,
    *,
    workload: WorkloadSpec | str | None = None,
    exact: bool = False,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> str:
    """One SQL statement: vector ORDER BY plus real dimension, ACL and fact joins."""
    table = ".".join(parse_qualified_name(table))
    workload_name = workload.name if isinstance(workload, WorkloadSpec) else workload
    if workload_name == "acl_only":
        temporal_predicate = ""
    elif workload_name == "grant_temporal_selectivity":
        temporal_predicate = """
  AND grant_row.valid_from <= %(as_of)s
  AND (grant_row.valid_to IS NULL OR grant_row.valid_to > %(as_of)s)"""
    else:
        temporal_predicate = """
  AND fact.valid_from <= %(as_of)s
  AND (fact.valid_to IS NULL OR fact.valid_to > %(as_of)s)"""
    binding_predicate = "" if exact else """
  AND (SELECT vector_hnsw_guidance_bind(
           %(vector_index)s::regclass,
           %(binding_atoms)s::text[],
           %(binding_kind)s
       ) OFFSET 0)"""
    candidate_validity = exact_truth_contract.qualify_candidate_validity_predicate(
        candidate_validity_predicate
    )
    validity = f"""
  WHERE ({qualify_predicate(predicate)})
  AND ({candidate_validity})
  AND v.id <> query_vector.query_id
{binding_predicate}
  AND grant_row.principal_name = CURRENT_USER::text
  AND grant_row.can_read
{temporal_predicate}"""
    if not exact:
        return f"""
WITH query_vector AS (
    SELECT id AS query_id, embedding
    FROM {table} AS query_row
    WHERE query_row.id = %(query_id)s
      AND ({exact_truth_contract.qualify_candidate_validity_predicate(candidate_validity_predicate, 'query_row')})
)
SELECT v.id,
       v.embedding <-> query_vector.embedding AS distance,
       vector_hnsw_guidance_profile() AS execution_guidance_profile
FROM {table} AS v
JOIN public.amazon_review_facts AS fact
  ON fact.review_id = v.id
JOIN public.amazon_product_dim AS product
  ON product.parent_asin = fact.parent_asin
JOIN public.amazon_principal_tenant_grants AS grant_row
  ON grant_row.tenant_id = product.tenant_id
CROSS JOIN query_vector
{validity}
ORDER BY v.embedding <-> query_vector.embedding
LIMIT %(k)s
""".strip()
    return f"""
WITH query_vector AS (
    SELECT id AS query_id, embedding
    FROM {table} AS query_row
    WHERE query_row.id = %(query_id)s
      AND ({exact_truth_contract.qualify_candidate_validity_predicate(candidate_validity_predicate, 'query_row')})
), valid AS MATERIALIZED (
    SELECT v.id, v.embedding
    FROM {table} AS v
    JOIN public.amazon_review_facts AS fact
      ON fact.review_id = v.id
    JOIN public.amazon_product_dim AS product
      ON product.parent_asin = fact.parent_asin
    JOIN public.amazon_principal_tenant_grants AS grant_row
      ON grant_row.tenant_id = product.tenant_id
    CROSS JOIN query_vector
    {validity}
)
SELECT valid.id,
       valid.embedding <-> query_vector.embedding AS distance
FROM valid
CROSS JOIN query_vector
ORDER BY distance, valid.id
LIMIT %(k)s
""".strip()


def validate_exact_sql_text(sql_text: str) -> None:
    normalized = sql_text.lower()
    forbidden = [token for token in ("vector_hnsw", "guidance_bind", "hnsw.") if token in normalized]
    if forbidden:
        raise RuntimeError(f"exact SQL contains approximate guidance/HNSW marker(s): {forbidden}")


def build_config_grid(args: argparse.Namespace, mode: str) -> list[Config]:
    configs: list[Config] = []
    targets = args.guided_collect_target_values
    for ef in args.ef_search_values:
        for max_scan in args.max_scan_tuples_values:
            for multiplier in args.scan_mem_multiplier_values:
                for iterative in args.iterative_scan_values:
                    for guided in targets:
                        target = ef if guided == "ef" else int(guided)
                        configs.append(Config(ef, max_scan, multiplier, iterative, target))
    if mode == "stock":
        unique: dict[tuple[int, int, float, str], Config] = {}
        for config in configs:
            unique.setdefault(
                (config.ef_search, config.max_scan_tuples, config.scan_mem_multiplier, config.iterative_scan),
                config,
            )
        configs = list(unique.values())
    unique_labels = {config.label: config for config in configs}
    return sorted(unique_labels.values(), key=lambda config: (config.ef_search, config.label))


def config_groups(configs: Sequence[Config]) -> list[tuple[int, list[Config]]]:
    grouped: dict[int, list[Config]] = {}
    for config in sorted(configs, key=lambda item: (item.ef_search, item.label)):
        grouped.setdefault(config.ef_search, []).append(config)
    return list(grouped.items())


def interleaved_schedule(
    keys: Sequence[tuple[Any, ...]], modes: Sequence[str], seed: int
) -> list[tuple[tuple[Any, ...], str]]:
    rng = random.Random(seed)
    schedule: list[tuple[tuple[Any, ...], str]] = []
    for key in keys:
        order = list(modes)
        rng.shuffle(order)
        schedule.extend((key, mode) for mode in order)
    return schedule


def recall_at_k(ids: Sequence[int], truth: Sequence[int], k: int) -> float:
    expected = set(truth[:k])
    return len(expected & set(ids[:k])) / len(expected) if expected else 0.0


def distance_tolerance(distance: float) -> float:
    return max(1e-9, abs(distance) * 1e-6)


def tie_aware_recall_at_k(
    results: Sequence[tuple[int, float]], truth: ExactTruth, query_id: int, k: int
) -> float:
    seen: set[int] = set()
    qualifying = 0
    threshold = truth.kth_distance + truth.tie_tolerance
    for row_id, distance in results:
        row_id = int(row_id)
        if row_id == query_id or row_id in seen:
            continue
        seen.add(row_id)
        if float(distance) <= threshold:
            qualifying += 1
        if len(seen) == k:
            break
    return min(k, qualifying) / k


def _expected_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str, int, int]]:
    return {
        (str(row["workload"]), str(row["filter_name"]), int(row["query_no"]), int(row["repeat"]))
        for row in rows
    }


def adaptive_transition_evidence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    adaptive_rows = [row for row in rows if row.get("mode") == "d1_d2_d3"]
    probes = sum(bool(row.get("adaptive_probe_observed")) for row in adaptive_rows)
    materializations = sum(
        bool(row.get("adaptive_materialized")) for row in adaptive_rows
    )
    active = sum(bool(row.get("adaptive_active")) for row in adaptive_rows)
    admissions = sum(bool(row.get("adaptive_admission_observed")) for row in adaptive_rows)
    hidden_reuse = sum(
        bool(row.get("hidden_prebuilt_fragment_reused")) for row in adaptive_rows
    )
    return {
        "required": bool(adaptive_rows),
        "rows": len(adaptive_rows),
        "probe_transitions": probes,
        "materialize_transitions": materializations,
        "admission_transitions": admissions,
        "active_requests": active,
        "hidden_prebuilt_fragment_reuse_requests": hidden_reuse,
        "valid": bool(
            adaptive_rows
            and probes > 0
            and materializations > 0
            and admissions > 0
            and active > 0
            and hidden_reuse == 0
        ),
    }


def grouped_adaptive_transition_evidence(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("mode") != "d1_d2_d3":
            continue
        key = (
            row.get("phase"),
            row.get("workload"),
            row.get("filter_name"),
            row.get("config"),
            row.get("target_recall"),
        )
        groups.setdefault(key, []).append(row)
    return [
        {
            "phase": key[0],
            "workload": key[1],
            "filter_name": key[2],
            "config": key[3],
            "target_recall": key[4],
            **adaptive_transition_evidence(group),
        }
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_keys: set[tuple[str, str, int, int]],
    target_recall: float | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 20260718,
    timing_field: str = "e2e_ms",
) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    observed = _expected_keys(ok)
    complete = observed == expected_keys and len(ok) == len(rows) == len(expected_keys)
    adaptive_evidence = adaptive_transition_evidence(rows)
    if adaptive_evidence["required"]:
        complete = complete and bool(adaptive_evidence["valid"])
    group = {(str(row["workload"]), str(row["filter_name"])) for row in rows}
    if len(group) != 1:
        raise ValueError("summarize_rows expects one workload/filter group")
    workload, filter_name = next(iter(group))
    by_query: dict[int, list[dict[str, Any]]] = {}
    for row in ok:
        by_query.setdefault(int(row["query_no"]), []).append(row)
    def timing_value(row: dict[str, Any]) -> float:
        value = row.get(timing_field, row.get("query_ms"))
        return float(value)

    latency_query = {
        query_no: statistics.fmean(timing_value(row) for row in items)
        for query_no, items in by_query.items()
    }
    recall_query = {
        query_no: statistics.fmean(float(row["recall"]) for row in items)
        for query_no, items in by_query.items()
    }
    recalls = list(recall_query.values())
    latencies = [timing_value(row) for row in ok]
    activation_values = [float(row.get("activation_ms", 0.0)) for row in ok]
    query_values = [float(row.get("query_ms", 0.0)) for row in ok]
    recall_lcb, recall_ci_low, recall_ci_high = bootstrap_bounds(
        recalls, bootstrap_samples, seed + 1
    )
    latency_ci_low, latency_ci_high = (0.0, 0.0)
    if latency_query:
        _, latency_ci_low, latency_ci_high = bootstrap_bounds(
            list(latency_query.values()), bootstrap_samples, seed + 2
        )
    recall_mean = statistics.fmean(recalls) if recalls else 0.0
    target_met = bool(
        complete and (target_recall is None or recall_mean >= target_recall)
    )
    numeric = complete and (target_recall is None or target_met)
    result: dict[str, Any] = {
        "workload": workload,
        "filter_name": filter_name,
        "rows": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "queries": len(latency_query),
        "complete": complete,
        "target_recall": target_recall if target_recall is not None else "",
        "target_met": target_met,
        "recall_mean": recall_mean if recalls else NA,
        "recall_lcb95": recall_lcb if recalls else NA,
        "recall_ci95_low": recall_ci_low if recalls else NA,
        "recall_ci95_high": recall_ci_high if recalls else NA,
        "activation_mean_ms": statistics.fmean(activation_values) if numeric and activation_values else NA,
        "query_mean_ms": statistics.fmean(query_values) if numeric and query_values else NA,
        "primary_timing_field": timing_field,
        "latency_mean_ms": statistics.fmean(latencies) if numeric and latencies else NA,
        "latency_p50_ms": statistics.median(latencies) if numeric and latencies else NA,
        "latency_p95_ms": percentile(latencies, 0.95) if numeric and latencies else NA,
        "latency_p99_ms": percentile(latencies, 0.99) if numeric and latencies else NA,
        "latency_ci95_low_ms": latency_ci_low if numeric and latencies else NA,
        "latency_ci95_high_ms": latency_ci_high if numeric and latencies else NA,
        "status": "complete" if numeric else NA,
        "query_latency_definition": TIMING_DEFINITION,
        "adaptive_transition_evidence": adaptive_evidence,
        "adaptive_mode_active": (
            bool(adaptive_evidence["valid"]) if adaptive_evidence["required"] else NA
        ),
    }
    return result


def select_config(summaries: Sequence[dict[str, Any]], target: float) -> dict[str, Any] | None:
    eligible = [
        row
        for row in summaries
        if bool(row.get("complete"))
        and bool(row.get("target_met"))
        and row.get("latency_mean_ms") not in (None, NA)
        and float(row.get("recall_mean", 0.0)) >= target
    ]
    return min(eligible, key=lambda row: (float(row["latency_mean_ms"]), str(row["config"]))) if eligible else None


def calibration_outcome(
    summaries: Sequence[dict[str, Any]],
    configs: Sequence[Config],
    executed_labels: Sequence[str],
    targets: Sequence[float],
) -> dict[str, Any]:
    planned_labels = [config.label for config in configs]
    if list(executed_labels) != planned_labels[: len(executed_labels)]:
        raise ValueError("calibration blocks are not a prefix of the ef-ordered grid")
    by_target = {
        float(target): select_config(
            [row for row in summaries if float(row["target_recall"]) == float(target)],
            float(target),
        )
        for target in targets
    }
    highest_target = max(float(target) for target in targets)
    stopped = by_target[highest_target] is not None
    grid_exhausted = len(executed_labels) == len(planned_labels)
    error_free = all(bool(row.get("complete")) and int(row.get("errors", 0)) == 0 for row in summaries)
    unattainable = [
        target
        for target, choice in by_target.items()
        if choice is None and grid_exhausted and error_free
    ]
    return {
        "planned_blocks": len(planned_labels),
        "executed_blocks": len(executed_labels),
        "executed_labels": list(executed_labels),
        "stopped": stopped,
        "stop_reason": "highest_target_attained" if stopped else "grid_exhausted" if grid_exhausted else "in_progress",
        "grid_exhausted": grid_exhausted,
        "error_free_grid_exhaustion": bool(grid_exhausted and error_free),
        "attainable_targets": [target for target, choice in by_target.items() if choice is not None],
        "unattainable_on_grid": unattainable,
        "indeterminate_targets": [
            target
            for target, choice in by_target.items()
            if choice is None and target not in unattainable
        ],
        "selected": by_target,
    }


def common_attainable_targets(
    outcomes: Sequence[dict[str, Any]], targets: Sequence[float]
) -> list[float]:
    return [
        float(target)
        for target in targets
        if outcomes
        and all(outcome["selected"].get(float(target)) is not None for outcome in outcomes)
    ]


def preregister_formal_matrix(
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    targets: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "workload": workload.name,
            "filter_name": spec.name,
            "target_recall": float(target),
            "status": "preregistered",
            "reason": "awaiting_independent_calibration",
            "selected_configs": {mode: None for mode in MODES},
        }
        for workload in workloads
        for spec in filters
        for target in targets
    ]


def finalize_formal_matrix(
    preregistered: Sequence[dict[str, Any]],
    outcomes: dict[tuple[str, str, str], dict[str, Any]],
    completed_final: set[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in preregistered:
        cell = dict(source)
        workload = str(cell["workload"])
        filter_name = str(cell["filter_name"])
        target = float(cell["target_recall"])
        mode_outcomes = [outcomes.get((workload, filter_name, mode)) for mode in MODES]
        if any(outcome is None for outcome in mode_outcomes):
            cell.update(status="invalid", reason="missing_calibration_outcome")
            result.append(cell)
            continue
        selected = {
            mode: outcome["selected"].get(target)  # type: ignore[index]
            for mode, outcome in zip(MODES, mode_outcomes)
        }
        cell["selected_configs"] = {
            mode: choice.get("config") if isinstance(choice, dict) else None
            for mode, choice in selected.items()
        }
        if all(choice is not None for choice in selected.values()):
            if (workload, filter_name, target) in completed_final:
                cell.update(status="complete", reason="held_out_final_complete")
            else:
                cell.update(status="invalid", reason="attainable_final_missing")
        elif all(
            target in outcome.get("unattainable_on_grid", [])  # type: ignore[union-attr]
            for outcome in mode_outcomes
        ):
            cell.update(status=NA, reason="unattainable_on_grid")
        elif any(
            target in outcome.get("indeterminate_targets", [])  # type: ignore[union-attr]
            for outcome in mode_outcomes
        ):
            cell.update(status="invalid", reason="calibration_indeterminate")
        else:
            cell.update(status=NA, reason="not_jointly_attainable")
        result.append(cell)
    return result


def artifact_validation_errors(
    expected_final_blocks: int,
    summaries: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]] = (),
    plans: Sequence[dict[str, Any]] = (),
    formal_matrix: Sequence[dict[str, Any]] = (),
) -> list[str]:
    errors: list[str] = []
    if formal_matrix and any(
        cell.get("status") not in {"complete", NA} for cell in formal_matrix
    ):
        errors.append("pre-registered workload/filter/target matrix is unresolved")
    invalid_final = [
        f"{row['workload']}|{row['filter_name']}|target={row['target_recall']}"
        for row in summaries
        if row.get("phase") == "final"
        and str(row.get("mode", "")).startswith("paired_")
        and row.get("status") != "complete"
    ]
    if invalid_final:
        errors.append(
            "held-out matched-recall validation failed: " + ",".join(invalid_final)
        )
    paired_final = [
        row
        for row in summaries
        if row.get("phase") == "final"
        and str(row.get("mode", "")).startswith("paired_")
    ]
    paired_groups: dict[tuple[Any, Any, Any], set[str]] = {}
    for row in paired_final:
        key = (row.get("workload"), row.get("filter_name"), row.get("target_recall"))
        paired_groups.setdefault(key, set()).add(str(row.get("mode"))[len("paired_") :])
    if (
        len(paired_final) != expected_final_blocks * len(SQLENS_MODES)
        or len(paired_groups) != expected_final_blocks
        or any(modes != set(SQLENS_MODES) for modes in paired_groups.values())
    ):
        errors.append("held-out comparison matrix is incomplete for D1/D2/D3")
    unsafe_rows = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if row.get("mode") in SQLENS_MODES
        and (
            row.get("filter_strategy") != "safe_guided"
            or row.get("guidance_semantics")
            != MODE_SPECS[str(row.get("mode"))].guidance_semantics
            or bool(row.get("hard_traversal_used"))
        )
    ]
    if unsafe_rows:
        errors.append(
            "join/RLS workload used unsafe or mislabeled guidance: "
            + ",".join(unsafe_rows[:10])
        )
    unproven_rows = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if not recorded_guidance_proof_is_valid(row)
    ]
    if unproven_rows:
        errors.append(
            "per-row guidance binding/effective/scan/final-path proof failed: "
            + ",".join(unproven_rows[:10])
        )
    d3_store_mismatches = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if row.get("mode") == "d1_d2_d3"
        and (
            row.get("persistent_fragment_reset_proof", {}).get("valid") is not True
            or int(row.get("prebuilt_fragments", -1)) != 0
        )
    ]
    if d3_store_mismatches:
        errors.append(
            "D3 block did not start from an audited empty persistent store: "
            + ",".join(d3_store_mismatches[:10])
        )
    context_mismatches = [
        str(row.get("pair_key", "unknown"))
        for row in rows
        if not row.get("principal")
        or str(row.get("snapshot_as_of", "")) != str(row.get("as_of", ""))
        or row.get("preferred_index_current_setting")
        != row.get("selected_vector_index")
        or row.get("page_access_current_setting") != "off"
        or row.get("index_page_access_current_setting") != "off"
    ]
    principals = {str(row.get("principal")) for row in rows if row.get("principal")}
    if context_mismatches or len(principals) > 1:
        errors.append(
            "principal/snapshot/preferred-index/prefetch runtime context mismatch: "
            + ",".join(context_mismatches[:10])
        )
    invalid_plans = [
        f"{plan.get('phase')}|{plan.get('workload')}|{plan.get('filter_name')}|{plan.get('mode')}"
        for plan in plans
        if plan.get("mode") in MODES
        and (
            not bool(plan.get("explain_gate", {}).get("valid"))
            or plan.get("selected_vector_index")
            != plan.get("explain_gate", {}).get("expected_index_qualified")
            or plan.get("preferred_index_current_setting")
            != plan.get("selected_vector_index")
            or plan.get("page_access_current_setting") != "off"
            or plan.get("index_page_access_current_setting") != "off"
            or plan.get("explain_order") != "after_all_timed_requests_in_block"
        )
    ]
    if invalid_plans:
        errors.append("wrong or unproven vector index: " + ",".join(invalid_plans[:10]))
    return errors


def database_contract_errors(database: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    proof = database.get("d2_graph_proof")
    comparison = proof.get("comparison") if isinstance(proof, dict) else None
    if (
        not isinstance(proof, dict)
        or proof.get("valid") is not True
        or not isinstance(comparison, dict)
        or any(
            comparison.get(field) is not True
            for field in (
                "same_heap",
                "logical_equal",
                "entry_equal",
                "tuple_coverage_equal",
                "definition_equal",
            )
        )
        or comparison.get("physical_equal") is not False
    ):
        errors.append("D2 same-heap/same-logical-graph BFS proof is missing or invalid")
    end_proof = database.get("d2_graph_proof_end")
    if end_proof != proof:
        errors.append("D2 graph proof changed or was not revalidated at formal-run end")
    start_relations = database.get("relations", {})
    end_indexes = database.get("d2_index_fingerprints_end", {})
    start_indexes = {
        index: start_relations.get(index)
        for index in database.get("d2_index_names", [])
    }
    if not start_indexes or end_indexes != start_indexes:
        errors.append("source/clone index fingerprints changed during the formal run")
    settings = database.get("preferred_index_current_settings")
    mode_indexes = database.get("mode_indexes")
    if (
        not isinstance(settings, dict)
        or not isinstance(mode_indexes, dict)
        or set(settings) != set(MODES)
        or set(mode_indexes) != set(MODES)
        or any(settings.get(mode) != mode_indexes.get(mode) for mode in MODES)
    ):
        errors.append("per-mode hnsw.preferred_index current_setting proof is invalid")
    reset = database.get("d3_startup_reset_evidence")
    try:
        prebuilt_fragments = int(reset.get("prebuilt_fragments", -1))
    except (AttributeError, TypeError, ValueError):
        prebuilt_fragments = -1
    if (
        not isinstance(reset, dict)
        or reset.get("after_reset_empty") is not True
        or prebuilt_fragments != 0
    ):
        errors.append("D3 did not start from an audited empty workload-driven cache")
    persistent_reset = database.get("d3_persistent_fragment_reset")
    if (
        not isinstance(persistent_reset, dict)
        or persistent_reset.get("valid") is not True
        or int(persistent_reset.get("prebuilt_fragments", -1)) != 0
        or not isinstance(database.get("d3_fragment_store_end"), dict)
    ):
        errors.append("D3 persistent fragment store reset/count/hash audit is invalid")
    data_guard = database.get("formal_data_version_proof")
    if (
        not isinstance(data_guard, dict)
        or data_guard.get("valid") is not True
        or data_guard.get("start_hash") != data_guard.get("end_hash")
    ):
        errors.append("formal GT/mode execution data-version guard is missing or invalid")
    security = database.get("rls_security_proofs")
    if not isinstance(security, dict) or set(security) != set(MODES):
        errors.append("per-session RLS principal/policy/probe proof is incomplete")
    else:
        for mode in MODES:
            try:
                validate_rls_security_proof(security[mode], str(database.get("principal", "")))
            except RuntimeError:
                errors.append(f"RLS security proof is invalid for mode={mode}")
    return errors


def validate_paired_query_contract(
    stock_rows: Sequence[dict[str, Any]], method_rows: Sequence[dict[str, Any]]
) -> None:
    def contract(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
        return {
            str(row["pair_key"]): (
                int(row["query_id"]),
                str(row["predicate"]),
                str(row["query_sql_sha256"]),
                str(row["exact_gt_ids"]),
                str(row.get("exact_gt_kth_distance", "")),
                str(row.get("exact_gt_tie_tolerance", "")),
                str(row.get("exact_gt_boundary_tied", "")),
                str(row.get("principal", "")),
                str(row.get("as_of", "")),
                str(row.get("snapshot_as_of", "")),
            )
            for row in rows
        }

    stock_contract = contract(stock_rows)
    method_contract = contract(method_rows)
    for pair_key in set(stock_contract) & set(method_contract):
        if stock_contract[pair_key] != method_contract[pair_key]:
            raise RuntimeError(f"paired SQL/GT contract mismatch for {pair_key}")


def paired_summary(
    stock_rows: Sequence[dict[str, Any]],
    method_rows: Sequence[dict[str, Any]],
    *,
    expected_keys: set[tuple[str, str, int, int]],
    target_recall: float,
    bootstrap_samples: int,
    seed: int,
    method_mode: str = "d1",
) -> dict[str, Any]:
    validate_paired_query_contract(stock_rows, method_rows)
    stock = summarize_rows(
        stock_rows, expected_keys=expected_keys, target_recall=target_recall,
        bootstrap_samples=bootstrap_samples, seed=seed,
    )
    method = summarize_rows(
        method_rows, expected_keys=expected_keys, target_recall=target_recall,
        bootstrap_samples=bootstrap_samples, seed=seed + 10,
    )
    stock_by_query: dict[int, list[float]] = {}
    method_by_query: dict[int, list[float]] = {}
    for row in stock_rows:
        if not row.get("error"):
            stock_by_query.setdefault(int(row["query_no"]), []).append(float(row.get("e2e_ms", row["query_ms"])))
    for row in method_rows:
        if not row.get("error"):
            method_by_query.setdefault(int(row["query_no"]), []).append(float(row.get("e2e_ms", row["query_ms"])))
    stock_q = {query: statistics.fmean(values) for query, values in stock_by_query.items()}
    method_q = {query: statistics.fmean(values) for query, values in method_by_query.items()}
    paired = set(stock_q) & set(method_q)
    valid = (
        stock["status"] != NA
        and method["status"] != NA
        and paired == {key[2] for key in expected_keys}
    )
    if valid:
        speed_lcb, speed_low, speed_high = bootstrap_ratio_bounds(
            stock_q, method_q, bootstrap_samples, seed + 20
        )
        deltas = [stock_q[query] - method_q[query] for query in sorted(paired)]
        _, delta_low, delta_high = bootstrap_bounds(deltas, bootstrap_samples, seed + 21)
        paired_values: dict[str, Any] = {
            "paired_queries": len(paired),
            "speedup_vs_stock": statistics.fmean(stock_q.values()) / statistics.fmean(method_q.values()),
            "speedup_lcb95": speed_lcb,
            "speedup_ci95_low": speed_low,
            "speedup_ci95_high": speed_high,
            "paired_latency_saving_mean_ms": statistics.fmean(deltas),
            "paired_latency_saving_ci95_low_ms": delta_low,
            "paired_latency_saving_ci95_high_ms": delta_high,
        }
    else:
        paired_values = {
            "paired_queries": len(paired),
            "speedup_vs_stock": NA,
            "speedup_lcb95": NA,
            "speedup_ci95_low": NA,
            "speedup_ci95_high": NA,
            "paired_latency_saving_mean_ms": NA,
            "paired_latency_saving_ci95_low_ms": NA,
            "paired_latency_saving_ci95_high_ms": NA,
        }
    return {
        "stock": stock,
        method_mode: method,
        **paired_values,
        "status": "complete" if valid else NA,
    }


def set_preferred_index(cur: Any, vector_index: str) -> str:
    cur.execute("SELECT set_config('hnsw.preferred_index', %s, false)", (vector_index,))
    cur.execute("SELECT current_setting('hnsw.preferred_index')")
    row = cur.fetchone()
    current = str(row[0]) if row else ""
    if current != vector_index:
        raise RuntimeError(
            f"preferred-index gate failed: requested={vector_index!r} current={current!r}"
        )
    return current


def set_mode(
    cur: Any,
    mode: str,
    config: Config,
    vector_index: str,
    d3_settings: dict[str, Any] | None = None,
) -> dict[str, str]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unknown benchmark mode: {mode}")
    spec = MODE_SPECS[mode]
    cur.execute(f"SET hnsw.ef_search = {int(config.ef_search)}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(config.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(config.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.iterative_scan = {config.iterative_scan}")
    cur.execute(f"SET hnsw.guided_collect_target = {int(config.guided_collect_target)}")
    # D2 is the physical BFS clone only. Keep heap/index prefetch out of this
    # factorial comparison and prove the settings again immediately per query.
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    settings = d3_settings or {
        "probe_requests": DEFAULT_D3_PROBE_REQUESTS,
        "min_benefit_per_byte": DEFAULT_D3_MIN_BENEFIT_PER_BYTE,
        "max_fragment_mb": DEFAULT_D3_MAX_FRAGMENT_MB,
        "page_min_skip_rate": DEFAULT_D3_PAGE_MIN_SKIP_RATE,
    }
    cur.execute("SET hnsw.guidance_require_epoch = on")
    cur.execute(f"SET hnsw.d3_probe_requests = {int(settings['probe_requests'])}")
    cur.execute(
        f"SET hnsw.d3_min_benefit_per_byte = {float(settings['min_benefit_per_byte'])}"
    )
    cur.execute(f"SET hnsw.d3_max_fragment_mb = {int(settings['max_fragment_mb'])}")
    cur.execute(
        f"SET hnsw.d3_page_min_skip_rate = {float(settings['page_min_skip_rate'])}"
    )
    cur.execute(f"SET hnsw.filter_strategy = {spec.filter_strategy}")
    preferred = set_preferred_index(cur, vector_index)
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    return {
        "filter_strategy": spec.filter_strategy,
        "preferred_index": preferred,
    }


def set_as_of(cur: Any, as_of: int) -> None:
    # SET does not accept a bind parameter; set_config keeps this value safely
    # parameterized and session-scoped for the RLS policy.
    cur.execute("SELECT set_config('app.as_of', %s, false)", (str(int(as_of)),))


def fetch_json_object(cur: Any, sql_text: str) -> dict[str, Any]:
    cur.execute(sql_text)
    row = cur.fetchone()
    value = row[0] if row else None
    try:
        parsed = json.loads(value) if isinstance(value, str) else dict(value or {})
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SQLens profile is not a JSON object: {value!r}") from exc
    return parsed


def _profile_counter(profile: dict[str, Any], name: str) -> int:
    try:
        return int(profile.get(name, 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"SQLens profile counter {name!r} is invalid") from exc


def scan_profile_export(profile: dict[str, Any]) -> dict[str, Any]:
    return {field: profile.get(field, NA) for field in SQLENS_PROFILE_EXPORT_FIELDS}


def configure_guidance(
    cur: Any, mode: str, vector_index: str, atoms: Sequence[str]
) -> dict[str, Any]:
    mode_spec = MODE_SPECS[mode]
    before = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    started = time.perf_counter()
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    if mode_spec.guidance_kind is None:
        activation_ms = (time.perf_counter() - started) * 1000.0
        return {
            "guidance_enabled": False,
            "guidance_route": "stock",
            "activation_atom_count": 0,
            "before": before,
            "after_activation": fetch_json_object(
                cur, "SELECT vector_hnsw_guidance_profile()"
            ),
            "activation_ms": activation_ms,
        }
    cur.execute(f"SET hnsw.filter_strategy = {mode_spec.filter_strategy}")
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)",
        (vector_index, list(atoms), mode_spec.guidance_kind),
    )
    row = cur.fetchone()
    activated_atoms = int(row[0]) if row and row[0] is not None else 0
    activation_ms = (time.perf_counter() - started) * 1000.0
    after = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    enabled = bool(after.get("active")) and activated_atoms > 0
    if mode_spec.adaptive and not enabled:
        # Probe/rejection requests execute stock HNSW while the extension records
        # workload evidence. The mode is not reported as active on these requests.
        route_started = time.perf_counter()
        cur.execute("SET hnsw.filter_strategy = off")
        activation_ms += (time.perf_counter() - route_started) * 1000.0
    elif not mode_spec.adaptive and not enabled:
        raise RuntimeError(f"{mode} guidance activation did not become active")
    return {
        "guidance_enabled": enabled,
        "guidance_route": (
            f"d3_{after.get('adaptive_state', 'unknown')}" if mode_spec.adaptive
            else "safe_guided_candidate_validation"
        ),
        "activation_atom_count": activated_atoms,
        "before": before,
        "after_activation": after,
        "activation_ms": activation_ms,
    }


def adaptive_transition_for_request(
    activation: dict[str, Any], post_query: dict[str, Any]
) -> dict[str, Any]:
    before = dict(activation.get("before") or {})
    after = dict(activation.get("after_activation") or {})
    probe = _profile_counter(post_query, "adaptive_probes") > _profile_counter(
        before, "adaptive_probes"
    )
    admissions = _profile_counter(post_query, "adaptive_admissions") - _profile_counter(
        before, "adaptive_admissions"
    )
    builds = sum(
        _profile_counter(post_query, field) - _profile_counter(before, field)
        for field in ("adaptive_page_builds", "adaptive_bloom_builds")
    )
    fragment_store_hits = _profile_counter(
        post_query, "fragment_store_hits"
    ) - _profile_counter(before, "fragment_store_hits")
    active = bool(activation.get("guidance_enabled")) and bool(after.get("active"))
    return {
        "adaptive_state_before": str(before.get("adaptive_state", "missing")),
        "adaptive_state_after_activation": str(after.get("adaptive_state", "missing")),
        "adaptive_state_after_query": str(post_query.get("adaptive_state", "missing")),
        "adaptive_probe_observed": probe,
        "adaptive_admission_observed": admissions > 0,
        "adaptive_materialized": builds > 0,
        "adaptive_active": active,
        "hidden_prebuilt_fragment_reused": fragment_store_hits > 0,
        "fragment_store_hit_delta": fragment_store_hits,
        "adaptive_transition": (
            f"{before.get('adaptive_state', 'missing')}->"
            f"{after.get('adaptive_state', 'missing')}->"
            f"{post_query.get('adaptive_state', 'missing')}"
        ),
    }


def guidance_execution_proof(
    mode: str,
    activation: dict[str, Any],
    execution_profile: dict[str, Any],
    scan_profile: dict[str, Any],
) -> dict[str, Any]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unknown benchmark mode: {mode}")
    required_execution_fields = {
        "active",
        "effective_active",
        "statement_bound",
        "binding_attempts",
        "binding_matches",
        "binding_scan_checks",
        "binding_scan_matches",
        "binding_scan_bypasses",
    }
    execution_profile_complete = required_execution_fields.issubset(execution_profile)
    before = dict(activation.get("before") or {})
    attempts_delta = _profile_counter(
        execution_profile, "binding_attempts"
    ) - _profile_counter(before, "binding_attempts")
    matches_delta = _profile_counter(
        execution_profile, "binding_matches"
    ) - _profile_counter(before, "binding_matches")
    scan_checks_delta = _profile_counter(
        execution_profile, "binding_scan_checks"
    ) - _profile_counter(before, "binding_scan_checks")
    scan_matches_delta = _profile_counter(
        execution_profile, "binding_scan_matches"
    ) - _profile_counter(before, "binding_scan_matches")
    scan_bypasses_delta = _profile_counter(
        execution_profile, "binding_scan_bypasses"
    ) - _profile_counter(before, "binding_scan_bypasses")
    guidance_checks = _profile_counter(scan_profile, "guidance_checks")
    final_path = str(scan_profile.get("final_path", "missing"))
    effective_active = bool(execution_profile.get("effective_active"))
    statement_bound = bool(execution_profile.get("statement_bound"))
    scan_valid = bool(scan_profile.get("valid"))

    if mode == "stock":
        valid = (
            execution_profile_complete
            and scan_valid
            and attempts_delta > 0
            and matches_delta == 0
            and not effective_active
            and guidance_checks == 0
            and final_path in {"stock", "stock_bypass"}
        )
        return {
            "valid": valid,
            "execution_profile_complete": execution_profile_complete,
            "binding_attempted": attempts_delta > 0,
            "binding_matched": matches_delta > 0,
            "effective_active": False,
            "statement_bound": statement_bound,
            "binding_attempts_delta": attempts_delta,
            "binding_matches_delta": matches_delta,
            "binding_scan_checks_delta": scan_checks_delta,
            "binding_scan_matches_delta": scan_matches_delta,
            "binding_scan_bypasses_delta": scan_bypasses_delta,
            "guidance_checks": guidance_checks,
            "final_path": final_path,
            "reported_active": False,
            "d3_probe_exception": False,
            "exception_reason": "",
        }

    d3_probe = (
        mode == "d1_d2_d3"
        and execution_profile_complete
        and activation.get("guidance_route") == "d3_probing"
        and not bool(activation.get("guidance_enabled"))
        and not effective_active
        and attempts_delta > 0
        and matches_delta == 0
        and str(execution_profile.get("adaptive_state", "")) == "probing"
        and _profile_counter(execution_profile, "adaptive_probes")
        > _profile_counter(before, "adaptive_probes")
        and scan_valid
        and guidance_checks == 0
        and final_path in {"stock", "stock_bypass"}
    )
    active_valid = (
        execution_profile_complete
        and bool(activation.get("guidance_enabled"))
        and bool(execution_profile.get("active"))
        and effective_active
        and statement_bound
        and attempts_delta > 0
        and matches_delta > 0
        and scan_checks_delta > 0
        and scan_matches_delta > 0
        and scan_bypasses_delta == 0
        and scan_valid
        and guidance_checks > 0
        and final_path == "validation_only"
    )
    return {
        "valid": bool(active_valid or d3_probe),
        "execution_profile_complete": execution_profile_complete,
        "binding_attempted": attempts_delta > 0,
        "binding_matched": matches_delta > 0,
        "effective_active": effective_active,
        "statement_bound": statement_bound,
        "binding_attempts_delta": attempts_delta,
        "binding_matches_delta": matches_delta,
        "binding_scan_checks_delta": scan_checks_delta,
        "binding_scan_matches_delta": scan_matches_delta,
        "binding_scan_bypasses_delta": scan_bypasses_delta,
        "guidance_checks": guidance_checks,
        "final_path": final_path,
        "reported_active": bool(active_valid),
        "d3_probe_exception": bool(d3_probe),
        "exception_reason": "workload_driven_probe_stock_route" if d3_probe else "",
    }


def recorded_guidance_proof_is_valid(row: dict[str, Any]) -> bool:
    activation = row.get("guidance_activation_profile")
    execution = row.get("execution_guidance_profile")
    scan = row.get("scan_profile")
    stored = row.get("guidance_execution_proof")
    if not all(isinstance(value, dict) for value in (activation, execution, scan, stored)):
        return False
    try:
        recomputed = guidance_execution_proof(
            str(row.get("mode")), activation, execution, scan
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    return bool(
        recomputed.get("valid") is True
        and stored == recomputed
        and row.get("guidance_binding_matched")
        is recomputed.get("binding_matched")
        and row.get("guidance_effective_active")
        is recomputed.get("effective_active")
        and row.get("guidance_checks") == recomputed.get("guidance_checks")
        and row.get("guidance_final_path") == recomputed.get("final_path")
        and (
            recomputed.get("d3_probe_exception") is not True
            or recomputed.get("reported_active") is False
        )
    )


def validate_fragment_store_reset(
    before: dict[str, Any], deleted_count: int, after: dict[str, Any]
) -> dict[str, Any]:
    before_count = int(before.get("count", -1))
    after_count = int(after.get("count", -1))
    if deleted_count != before_count or after_count != 0:
        raise RuntimeError(
            "persistent fragment store reset failed: "
            f"before={before_count} deleted={deleted_count} after={after_count}"
        )
    return {
        "valid": True,
        "before": before,
        "deleted_count": deleted_count,
        "after": after,
        "prebuilt_fragments": after_count,
    }


def audit_fragment_store(cur: Any, vector_table: str) -> dict[str, Any]:
    cur.execute("SELECT to_regclass('public.pgvector_hnsw_fragment_store')")
    row = cur.fetchone()
    if row is None or row[0] is None:
        records: list[str] = []
        return {
            "exists": False,
            "count": 0,
            "content_sha256": canonical_sha256(records),
        }
    cur.execute(
        """
        SELECT row_to_json(store_row)::text
        FROM public.pgvector_hnsw_fragment_store AS store_row
        WHERE store_row.heap_oid = to_regclass(%s)
        ORDER BY row_to_json(store_row)::text
        """,
        (vector_table,),
    )
    records = [str(value[0]) for value in cur.fetchall()]
    return {
        "exists": True,
        "count": len(records),
        "content_sha256": canonical_sha256(records),
    }


def clear_fragment_store(cur: Any, vector_table: str) -> dict[str, Any]:
    before = audit_fragment_store(cur, vector_table)
    deleted = 0
    if before["exists"]:
        cur.execute(
            "DELETE FROM public.pgvector_hnsw_fragment_store "
            "WHERE heap_oid = to_regclass(%s)",
            (vector_table,),
        )
        deleted = int(cur.rowcount)
    after = audit_fragment_store(cur, vector_table)
    return validate_fragment_store_reset(before, deleted, after)


def cache_profile_is_empty(profile: dict[str, Any]) -> bool:
    fields = (
        "entries",
        "resident_entries",
        "resident_bytes",
        "composed_guide_entries",
        "composed_exact_entries",
        "adaptive_cache_entries",
        "adaptive_bytes",
    )
    return all(_profile_counter(profile, field) == 0 for field in fields)


def reset_adaptive_state(
    cur: Any, persistent_reset: dict[str, Any] | None = None
) -> dict[str, Any]:
    before = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    after = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    if not cache_profile_is_empty(after):
        raise RuntimeError("D3 cold-start gate failed: metadata cache is not empty after reset")
    prebuilt_fragments: Any = NA
    if persistent_reset is not None:
        if persistent_reset.get("valid") is not True:
            raise RuntimeError("D3 persistent fragment reset proof is invalid")
        prebuilt_fragments = int(persistent_reset.get("prebuilt_fragments", -1))
    return {
        "prebuilt_fragments": prebuilt_fragments,
        "before_reset": before,
        "after_reset": after,
        "after_reset_empty": True,
        "persistent_reset": persistent_reset or {"status": "not_supplied"},
    }


def plan_index_names(plan: Any) -> list[str]:
    names: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            value = node.get("Index Name")
            if value:
                names.append(str(value))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(plan)
    return names


def validate_explain_gate(plan: Any, vector_index: str, *, require_hnsw: bool) -> dict[str, Any]:
    expected_index = parse_qualified_name(vector_index)[-1].lower()
    index_names = plan_index_names(plan)
    hnsw_names = [name for name in index_names if "hnsw" in name.lower()]
    uses_expected = any(name.lower() == expected_index for name in index_names)
    vector_hnsw_names = list(hnsw_names)
    valid = (
        uses_expected
        and all(name.lower() == expected_index for name in vector_hnsw_names)
        if require_hnsw
        else not hnsw_names
    )
    if not valid:
        mode = "approximate HNSW" if require_hnsw else "exact non-HNSW"
        raise RuntimeError(
            f"EXPLAIN gate failed for {mode}: expected_index={expected_index!r} "
            f"index_names={index_names!r}"
        )
    return {
        "valid": True,
        "require_hnsw": require_hnsw,
        "expected_index": expected_index,
        "expected_index_qualified": vector_index,
        "index_names": index_names,
        "vector_hnsw_index_names": vector_hnsw_names,
    }


def validate_graph_compare(
    proof: Any, source_index: str, clone_index: str
) -> dict[str, Any]:
    try:
        normalized = json.loads(proof) if isinstance(proof, str) else dict(proof)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("same-graph BFS proof is not a JSON object") from exc
    required_true = (
        "same_heap",
        "logical_equal",
        "entry_equal",
        "tuple_coverage_equal",
        "definition_equal",
    )
    valid = (
        source_index != clone_index
        and all(normalized.get(field) is True for field in required_true)
        and normalized.get("physical_equal") is False
    )
    if not valid:
        raise RuntimeError(
            "same-heap/same-logical-graph BFS proof failed: "
            + json.dumps(
                {
                    "source_index": source_index,
                    "clone_index": clone_index,
                    "comparison": normalized,
                },
                sort_keys=True,
            )
        )
    return {
        "valid": True,
        "source_index": source_index,
        "clone_index": clone_index,
        "required": {
            "same_heap": True,
            "logical_equal": True,
            "entry_equal": True,
            "tuple_coverage_equal": True,
            "definition_equal": True,
            "physical_equal": False,
        },
        "comparison": normalized,
    }


def graph_clone_proof(cur: Any, source_index: str, clone_index: str) -> dict[str, Any]:
    cur.execute(
        "SELECT vector_hnsw_graph_compare(%s::regclass, %s::regclass)",
        (source_index, clone_index),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("same-graph BFS proof query returned no row")
    return validate_graph_compare(row[0], source_index, clone_index)


def runtime_sql_context(cur: Any, principal: str, as_of: int) -> dict[str, Any]:
    cur.execute(
        "SELECT current_user::text, current_setting('app.as_of', true), "
        "current_setting('hnsw.preferred_index', true), "
        "current_setting('hnsw.filter_strategy', true), "
        "current_setting('hnsw.page_access', true), "
        "current_setting('hnsw.index_page_access', true)"
    )
    row = cur.fetchone()
    context = {
        "current_user": str(row[0]) if row else "",
        "app_as_of": str(row[1]) if row and row[1] is not None else "",
        "preferred_index": str(row[2]) if row and row[2] is not None else "",
        "filter_strategy": str(row[3]) if row and row[3] is not None else "",
        "page_access": str(row[4]) if row and row[4] is not None else "",
        "index_page_access": str(row[5]) if row and row[5] is not None else "",
    }
    if (
        context["current_user"] != principal
        or context["app_as_of"] != str(int(as_of))
        or context["page_access"] != "off"
        or context["index_page_access"] != "off"
    ):
        raise RuntimeError(
            "principal/snapshot/prefetch gate failed: "
            f"expected=({principal!r},{int(as_of)!r}) observed={context!r}"
        )
    return context


def query_results(
    cur: Any, sql_text: str, params: dict[str, Any], *, exact: bool = False
) -> list[tuple[int, float]]:
    if exact:
        cur.execute("BEGIN")
        try:
            cur.execute("SET LOCAL enable_indexscan = on")
            cur.execute("SET LOCAL enable_bitmapscan = on")
            cur.execute("SET LOCAL enable_seqscan = on")
            cur.execute(sql_text, params)
            rows = [(int(row[0]), float(row[1])) for row in cur.fetchall()]
            cur.execute("COMMIT")
            return rows
        except Exception:
            cur.execute("ROLLBACK")
            raise
    cur.execute(sql_text, params)
    return [(int(row[0]), float(row[1])) for row in cur.fetchall()]


def query_rows(cur: Any, sql_text: str, params: dict[str, Any], *, exact: bool = False) -> list[int]:
    return [row_id for row_id, _ in query_results(cur, sql_text, params, exact=exact)]


def explain(
    cur: Any,
    sql_text: str,
    params: dict[str, Any],
    *,
    vector_index: str,
    require_hnsw: bool,
) -> tuple[Any, dict[str, Any]]:
    cur.execute("EXPLAIN (FORMAT JSON) " + sql_text, params)
    row = cur.fetchone()
    plan = row[0] if row else []
    return plan, validate_explain_gate(plan, vector_index, require_hnsw=require_hnsw)


def prepare_explain_without_runtime_state(cur: Any) -> dict[str, Any]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    guidance = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    cache = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    if bool(guidance.get("active")) or not cache_profile_is_empty(cache):
        raise RuntimeError("EXPLAIN pre-state is not cold and inactive")
    return {"guidance": guidance, "cache": cache}


def finish_explain_without_runtime_state(
    cur: Any, before: dict[str, Any]
) -> dict[str, Any]:
    guidance = fetch_json_object(cur, "SELECT vector_hnsw_guidance_profile()")
    cache = fetch_json_object(cur, "SELECT vector_hnsw_metadata_cache_profile()")
    counter_fields = (
        "binding_attempts",
        "binding_matches",
        "binding_scan_checks",
        "adaptive_probes",
        "adaptive_admissions",
        "fragment_builds",
    )
    unchanged = all(
        _profile_counter(guidance, field)
        == _profile_counter(before["guidance"], field)
        for field in counter_fields
    )
    valid = (
        not bool(guidance.get("active"))
        and cache_profile_is_empty(cache)
        and unchanged
    )
    if not valid:
        raise RuntimeError("EXPLAIN mutated measured guidance/adaptive state")
    return {
        "valid": True,
        "execution": "EXPLAIN_without_ANALYZE",
        "guidance_before": before["guidance"],
        "guidance_after": guidance,
        "cache_before": before["cache"],
        "cache_after": cache,
        "counters_unchanged": True,
    }


relation_fingerprint = exact_truth_contract.relation_fingerprint
validate_rls_security_proof = exact_truth_contract.validate_rls_security_proof


def validate_sqlens_provenance(build_id: Any, profile: Any) -> tuple[str, dict[str, Any]]:
    normalized_build_id = str(build_id or "")
    if not normalized_build_id.startswith(SQLENS_BUILD_PREFIX):
        raise RuntimeError(
            "SQLens build gate failed: "
            f"vector_sqlens_build_id() returned {normalized_build_id!r}; "
            f"expected prefix {SQLENS_BUILD_PREFIX!r}. Rebuild/reload vector.so and reconnect."
        )
    try:
        normalized_profile = json.loads(profile) if isinstance(profile, str) else dict(profile)
        semantics = float(normalized_profile["profile_semantics_version"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "SQLens build gate failed: vector_hnsw_last_scan_profile() has no valid "
            "profile_semantics_version. Rebuild/reload vector.so and reconnect."
        ) from exc
    missing = [field for field in SQLENS_PROFILE_FIELDS if field not in normalized_profile]
    if not math.isfinite(semantics) or semantics < SQLENS_PROFILE_SEMANTICS or missing:
        raise RuntimeError(
            "SQLens build gate failed: incompatible scan profile "
            f"semantics={normalized_profile.get('profile_semantics_version')!r} "
            f"minimum={SQLENS_PROFILE_SEMANTICS:g} "
            f"missing={missing!r}. Rebuild/reload vector.so and reconnect."
        )
    return normalized_build_id, normalized_profile


def database_fingerprint(cur: Any, relations: Sequence[str]) -> dict[str, Any]:
    cur.execute(
        "SELECT current_database(), current_setting('server_version'), "
        "coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), ''), "
        "vector_sqlens_build_id()"
    )
    database, postgres, vector_version, build_id = cur.fetchone()
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile_row = cur.fetchone()
    build_id, scan_profile = validate_sqlens_provenance(
        build_id, profile_row[0] if profile_row else None
    )
    relation_data = {relation: relation_fingerprint(cur, relation) for relation in dict.fromkeys(relations)}
    fact_relation = relation_data.get("public.amazon_review_facts")
    if fact_relation is None or not fact_relation["rls"]:
        raise RuntimeError("artifact invalid: amazon_review_facts must have RLS enabled")
    return {
        "database": database,
        "postgres_version": postgres,
        "vector_extension_version": vector_version,
        "sqlens_build_id": build_id,
        "profile_semantics_version": scan_profile["profile_semantics_version"],
        "required_profile_fields": list(SQLENS_PROFILE_FIELDS),
        "loaded_profile": scan_profile,
        "relations": relation_data,
    }


def loaded_session_context(cur: Any) -> dict[str, str]:
    cur.execute(
        "SELECT current_user::text, session_user::text, "
        "txid_current_snapshot()::text, current_database()::text, "
        "current_setting('hnsw.preferred_index', true), "
        "current_setting('hnsw.filter_strategy', true), "
        "current_setting('hnsw.guidance_require_epoch', true)"
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("could not capture loaded role/snapshot context")
    return {
        "current_user": str(row[0]),
        "session_user": str(row[1]),
        "transaction_snapshot": str(row[2]),
        "database": str(row[3]),
        "preferred_index_current_setting": str(row[4] or ""),
        "filter_strategy_current_setting": str(row[5] or ""),
        "guidance_require_epoch_current_setting": str(row[6] or ""),
    }


def expected_keys_for(
    workloads: Sequence[WorkloadSpec], filters: Sequence[FilterSpec], query_ids: dict[int, int], repeats: int
) -> set[tuple[str, str, int, int]]:
    return {
        (workload.name, spec.name, query_no, repeat)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
        for repeat in range(repeats)
    }


def sql_contract_hashes(
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    table: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[str, dict[str, dict[str, str]]]:
    validity_hash = candidate_universe_predicate_sha256(
        candidate_validity_predicate
    )
    result: dict[str, dict[str, dict[str, str]]] = {}
    for workload in workloads:
        result[workload.name] = {}
        for spec in filters:
            exact_sql = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                exact=True,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            validate_exact_sql_text(exact_sql)
            approx_sql = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            result[workload.name][spec.name] = {
                "exact_sha256": hashlib.sha256(exact_sql.encode()).hexdigest(),
                "approx_sha256": hashlib.sha256(approx_sql.encode()).hexdigest(),
                "workload_scalar_predicate": spec.predicate,
                "workload_scalar_predicate_sha256": workload_scalar_predicate_sha256(
                    spec.predicate
                ),
                "candidate_universe_predicate": candidate_validity_predicate,
                "candidate_universe_predicate_sha256": validity_hash,
            }
    return result


def _artifact_plan_is_non_hnsw(gate: Any, label: str) -> None:
    if not isinstance(gate, dict) or gate.get("valid") is not True:
        raise _artifact_error(f"{label} is missing a successful non-HNSW EXPLAIN gate")
    names = gate.get("index_names")
    if not isinstance(names, list) or any("hnsw" in str(name).lower() for name in names):
        raise _artifact_error(f"{label} EXPLAIN gate used HNSW or has invalid index provenance")


def _artifact_filters_match(source: Any, filters: Sequence[FilterSpec]) -> bool:
    if not isinstance(source, list) or len(source) != len(filters):
        return False
    try:
        return all(
            isinstance(row, dict)
            and row.get("name") == spec.name
            and row.get("target_rate") == spec.target_rate
            and row.get("predicate") == spec.predicate
            and int(row.get("expected_rows", -1)) == spec.expected_rows
            and float(row.get("actual_pct", math.nan)) == spec.actual_pct
            for row, spec in zip(source, filters)
        )
    except (TypeError, ValueError):
        return False


def _artifact_workloads_match(source: Any, workloads: Sequence[WorkloadSpec]) -> bool:
    if not isinstance(source, list) or len(source) != len(workloads):
        return False
    try:
        for row, workload in zip(source, workloads):
            if not isinstance(row, dict):
                return False
            temporal = row.get("temporal")
            if temporal is None:
                temporal = row.get("temporal_kind") != "none"
            if (
                row.get("name") != workload.name
                or float(row.get("bucket_pct", math.nan)) != workload.bucket_pct
                or bool(temporal) != workload.temporal
            ):
                return False
    except (TypeError, ValueError):
        return False
    return True


def _artifact_query_ids(source: Any) -> dict[int, int]:
    if not isinstance(source, dict):
        raise _artifact_error("run-spec query_ids is malformed")
    try:
        parsed = {int(query_no): int(query_id) for query_no, query_id in source.items()}
    except (TypeError, ValueError) as exc:
        raise _artifact_error("run-spec query_ids is malformed") from exc
    if len(parsed) != len(source) or len(set(parsed.values())) != len(parsed):
        raise _artifact_error("run-spec query IDs are duplicate or malformed")
    return parsed


def _artifact_pair_map(
    manifest: dict[str, Any],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    table: str,
    candidate_validity_predicate: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {(workload.name, spec.name) for workload in workloads for spec in filters}
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise _artifact_error("manifest pairs are missing")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise _artifact_error("manifest pair is malformed")
        filter_data = pair.get("filter")
        key = (str(pair.get("workload")), str(filter_data.get("name") if isinstance(filter_data, dict) else ""))
        if key not in expected or key in result:
            raise _artifact_error(f"manifest has unexpected/duplicate pair: {key}")
        spec = next(item for item in filters if item.name == key[1])
        exact_workload = next(
            item for item in exact_truth_contract.WORKLOADS if item.name == key[0]
        )
        validity_hash = candidate_universe_predicate_sha256(
            candidate_validity_predicate
        )
        scalar_hash = workload_scalar_predicate_sha256(spec.predicate)
        if (
            pair.get("workload_scalar_predicate") != spec.predicate
            or pair.get("workload_scalar_predicate_sha256") != scalar_hash
            or pair.get("candidate_universe_predicate")
            != candidate_validity_predicate
            or pair.get("candidate_universe_predicate_sha256") != validity_hash
        ):
            raise _artifact_error(f"pair scalar/candidate-universe provenance is stale: {key}")
        candidate = pair.get("candidate")
        if not isinstance(candidate, dict):
            raise _artifact_error(f"candidate provenance is missing: {key}")
        candidate_path = Path(str(candidate.get("path", "")))
        candidate_hash = _require_sha256(candidate.get("sha256"), f"candidate {key}")
        if (
            int(candidate.get("count", 0)) <= 0
            or int(candidate.get("min_id", -1)) < 0
            or int(candidate.get("max_id", -1)) < int(candidate.get("min_id", -1))
            or not candidate_path.is_file()
            or sha256_file(candidate_path) != candidate_hash
        ):
            raise _artifact_error(f"candidate provenance is stale or incomplete: {key}")
        candidate_sql = pair.get("candidate_sql")
        if not isinstance(candidate_sql, str) or pair.get("candidate_sql_sha256") != _sha256_text(candidate_sql):
            raise _artifact_error(f"candidate SQL provenance is stale: {key}")
        expected_candidate_sql = exact_truth_contract.build_candidate_sql(
            table, spec.predicate, exact_workload, candidate_validity_predicate
        )
        if candidate_sql != expected_candidate_sql:
            raise _artifact_error(f"candidate SQL scalar/validity contract is stale: {key}")
        validate_exact_sql_text(candidate_sql)
        normalized_candidate = " ".join(candidate_sql.lower().split())
        required_candidate_tokens = (
            "join public.amazon_review_facts",
            "join public.amazon_product_dim",
            "join public.amazon_principal_tenant_grants",
            "current_user",
            "order by v.id",
        )
        if " limit " in f" {normalized_candidate} " or any(token not in normalized_candidate for token in required_candidate_tokens):
            raise _artifact_error(f"candidate SQL is not the required unbounded relational export: {key}")
        _artifact_plan_is_non_hnsw(pair.get("candidate_explain_gate"), f"candidate {key}")
        spot_sql = pair.get("spot_check_sql")
        if not isinstance(spot_sql, str) or pair.get("spot_check_sql_sha256") != _sha256_text(spot_sql):
            raise _artifact_error(f"spot-check SQL provenance is stale: {key}")
        expected_spot_sql = exact_truth_contract.build_spot_check_sql(
            table, spec.predicate, exact_workload, candidate_validity_predicate
        )
        if spot_sql != expected_spot_sql:
            raise _artifact_error(f"spot-check SQL scalar/validity contract is stale: {key}")
        validate_exact_sql_text(spot_sql)
        _artifact_plan_is_non_hnsw(pair.get("spot_check_explain_gate"), f"spot check {key}")
        checks = pair.get("spot_checks")
        if not isinstance(checks, list) or not checks:
            raise _artifact_error(f"spot checks are missing: {key}")
        seen_checks: set[int] = set()
        for check in checks:
            if not isinstance(check, dict) or check.get("valid") is not True:
                raise _artifact_error(f"spot check is invalid: {key}")
            query_no = int(check.get("query_no", -1))
            sql_ids = check.get("sql_ids")
            sql_distances = check.get("sql_distances")
            if (
                query_no in seen_checks
                or query_no < 0
                or int(check.get("limit", -1)) <= 0
                or not isinstance(sql_ids, list)
                or not isinstance(sql_distances, list)
                or len(sql_ids) != int(check["limit"])
                or len(sql_distances) != int(check["limit"])
                or len({int(value) for value in sql_ids}) != len(sql_ids)
                or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in sql_distances)
            ):
                raise _artifact_error(f"spot check is malformed: {key}")
            seen_checks.add(query_no)
        result[key] = pair
    if set(result) != expected:
        raise _artifact_error("manifest pair keyspace is incomplete")
    return result


def _validate_artifact_spot_check(
    check: dict[str, Any],
    query_id: int,
    expected_ids: Sequence[int],
    expected_distances_sq: Sequence[float],
) -> None:
    observed_ids = [int(value) for value in check["sql_ids"]]
    observed_distances = [float(value) for value in check["sql_distances"]]
    if query_id in observed_ids:
        raise _artifact_error("spot check includes its query row")
    for position, (expected_id, expected_distance, observed_id, observed_distance) in enumerate(
        zip(expected_ids, expected_distances_sq, observed_ids, observed_distances)
    ):
        tolerance = distance_tolerance(expected_distance)
        tied = (
            position > 0 and abs(expected_distance - expected_distances_sq[position - 1]) <= tolerance
        ) or (
            position + 1 < len(expected_distances_sq)
            and abs(expected_distance - expected_distances_sq[position + 1]) <= tolerance
        )
        if tied:
            if abs(observed_distance - expected_distance) > tolerance:
                raise _artifact_error("spot check tied result has the wrong distance")
        elif (
            observed_id != expected_id
            or abs(observed_distance - expected_distance) > max(1e-7, abs(expected_distance) * 5e-5)
        ):
            raise _artifact_error("spot check result does not match the truth record")


def load_external_exact_truth(
    truth_csv: Path,
    manifest_path: Path,
    fbin: Path,
    filters_csv: Path,
    query_ids_csv: Path,
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    query_splits: dict[int, str],
    as_of_by_workload: dict[str, int],
    table: str,
    principal: str,
    k: int,
    database_relations: dict[str, Any],
    *,
    require_formal_keyspace: bool = True,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
    query_cohort_manifest: Path | None = None,
) -> tuple[dict[tuple[str, str, int], ExactTruth], dict[str, str]]:
    """Load a producer artifact only after its immutable provenance is fully verified."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _artifact_error(f"manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("artifact_valid") is not True:
        raise _artifact_error("manifest does not declare artifact_valid=true")
    if manifest.get("artifact") != "amazon10m_sql_native_exact_truth":
        raise _artifact_error("manifest has an incompatible artifact type")
    if manifest.get("version") != EXACT_TRUTH_ARTIFACT_VERSION:
        raise _artifact_error(
            "legacy/incompatible exact-truth artifact version rejected: "
            f"observed={manifest.get('version')!r} expected={EXACT_TRUTH_ARTIFACT_VERSION}"
        )
    candidate_validity_predicate = validate_candidate_validity_predicate(
        candidate_validity_predicate
    )
    expected_validity_hash = candidate_universe_predicate_sha256(
        candidate_validity_predicate
    )
    expected_cohort_hash = query_cohort_sha256(query_ids, query_splits)
    data_version = manifest.get("data_version_proof")
    expected_data_relations = {
        relation: database_relations.get(relation)
        for relation in exact_truth_contract.formal_data_relations(table)
    }
    if (
        not isinstance(data_version, dict)
        or data_version.get("valid") is not True
        or data_version.get("start_hash") != data_version.get("end_hash")
        or data_version.get("start_relations") != expected_data_relations
        or data_version.get("end_relations") != expected_data_relations
        or data_version.get("start_hash") != canonical_sha256(expected_data_relations)
    ):
        raise _artifact_error("GT data-version/epoch proof does not match the formal run")
    if any(
        not isinstance(fingerprint, dict)
        or not any(
            exact_truth_contract.valid_epoch_trigger(trigger)
            for trigger in fingerprint.get("triggers", [])
        )
        for fingerprint in expected_data_relations.values()
    ):
        raise _artifact_error("formal data-version epoch trigger proof is incomplete")
    try:
        expected_relation_epoch = relation_epoch_contract(expected_data_relations)
    except RuntimeError as exc:
        raise _artifact_error("current formal relation epoch contract is incomplete") from exc
    if manifest.get("relation_epoch") != expected_relation_epoch:
        raise _artifact_error("GT relation epoch contract does not match the formal run")
    try:
        gt_security = exact_truth_contract.validate_rls_security_proof(
            dict(manifest.get("rls_security_proof") or {}), principal
        )
    except (TypeError, RuntimeError) as exc:
        raise _artifact_error("GT RLS principal/policy/probe proof is invalid") from exc
    expected_policy_hash = canonical_sha256(
        database_relations.get("public.amazon_review_facts", {}).get("policies", [])
    )
    if gt_security.get("policy_hash") != expected_policy_hash:
        raise _artifact_error("GT RLS policy hash does not match the formal run")
    run_spec = manifest.get("run_spec")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(run_spec, dict) or not isinstance(source_hashes, dict):
        raise _artifact_error("manifest run-spec/source hashes are missing")
    if manifest.get("run_spec_hash") != canonical_sha256(run_spec) or run_spec.get("source_hashes") != source_hashes:
        raise _artifact_error("manifest run-spec/source hash mismatch")
    candidate_universe = run_spec.get("candidate_universe")
    cohort = run_spec.get("query_cohort")
    try:
        run_splits = {
            int(query_no): str(split)
            for query_no, split in run_spec.get("query_splits", {}).items()
        }
    except (TypeError, ValueError) as exc:
        raise _artifact_error("run-spec query split contract is malformed") from exc
    if (
        not isinstance(candidate_universe, dict)
        or candidate_universe.get("predicate") != candidate_validity_predicate
        or candidate_universe.get("predicate_sha256") != expected_validity_hash
        or manifest.get("candidate_universe") != candidate_universe
        or manifest.get("candidate_universe_predicate_sha256")
        != expected_validity_hash
        or not isinstance(cohort, dict)
        or cohort.get("query_cohort_sha256") != expected_cohort_hash
        or cohort.get("source_csv_sha256") != source_hashes.get("query_ids_csv")
        or not isinstance(cohort.get("source_manifest"), dict)
        or cohort["source_manifest"].get("sha256")
        != source_hashes.get("query_cohort_manifest")
        or run_spec.get("query_cohort_sha256") != expected_cohort_hash
        or manifest.get("query_cohort") != cohort
        or manifest.get("query_cohort_sha256") != expected_cohort_hash
        or run_splits != query_splits
    ):
        raise _artifact_error("manifest query cohort/candidate universe contract mismatch")
    for name in (
        "script", "filters_csv", "query_ids_csv", "query_cohort_manifest", "fbin"
    ):
        _require_sha256(source_hashes.get(name), f"source {name}")
    if query_cohort_manifest is None:
        raise _artifact_error("v4 artifact loading requires the query cohort provenance manifest")
    local_sources = {
        "filters_csv": sha256_file(filters_csv),
        "query_ids_csv": sha256_file(query_ids_csv),
        "query_cohort_manifest": sha256_file(query_cohort_manifest),
        "fbin": sha256_file(fbin),
    }
    if any(source_hashes[name] != digest for name, digest in local_sources.items()):
        raise _artifact_error("source hashes do not match the current fbin/query/filter inputs")
    fbin_info = manifest.get("fbin")
    if not isinstance(fbin_info, dict) or Path(str(fbin_info.get("path", ""))).resolve() != fbin.resolve():
        raise _artifact_error("manifest fbin path is incompatible")
    try:
        compatible_run = (
            run_spec.get("vector_table") == table
            and run_spec.get("principal") == principal
            and int(run_spec.get("k", -1)) == k
            and int(run_spec.get("calibration_queries", -1)) == sum(split == "calibration" for split in query_splits.values())
            and int(run_spec.get("final_queries", -1)) == sum(split == "final" for split in query_splits.values())
        )
    except (TypeError, ValueError):
        compatible_run = False
    if not compatible_run:
        raise _artifact_error("table/principal/k/calibration/final compatibility check failed")
    backend = manifest.get("backend")
    run_backend = run_spec.get("backend")
    if (
        not isinstance(backend, dict)
        or backend != run_backend
        or backend.get("backend") != "faiss"
        or backend.get("class") != "IndexFlatL2"
        or backend.get("exact") is not True
        or backend.get("formal_default") is not True
        or int(backend.get("threads", 0)) <= 0
    ):
        raise _artifact_error("formal exact backend provenance is incompatible")
    mapping = manifest.get("base_table_mapping")
    if not isinstance(mapping, dict) or mapping != run_spec.get("base_table_mapping"):
        raise _artifact_error("base-table/fbin mapping provenance is missing or stale")
    try:
        base_sample_ids = [int(value) for value in mapping["base_sample_ids"]]
        checked_ids = [int(value) for value in mapping["checked_ids"]]
        included_query_ids = [int(value) for value in mapping["query_ids_included"]]
        mapping_valid = (
            base_sample_ids == sorted(set(base_sample_ids))
            and checked_ids == sorted(set(checked_ids))
            and included_query_ids == sorted(query_ids.values())
            and set(included_query_ids) <= set(checked_ids)
            and int(mapping["checked_rows"]) == len(checked_ids)
            and int(mapping["base_sample_size_requested"]) > 0
            and mapping["base_sample_ids_sha256"] == canonical_sha256(base_sample_ids)
            and mapping["checked_ids_sha256"] == canonical_sha256(checked_ids)
            and mapping["comparison"] == "float32_allclose"
            and math.isfinite(float(mapping["max_abs_error"]))
            and float(mapping["max_abs_error"]) >= 0.0
        )
    except (KeyError, TypeError, ValueError):
        mapping_valid = False
    if not mapping_valid:
        raise _artifact_error("base-table/fbin mapping audit is malformed")
    if _artifact_query_ids(run_spec.get("query_ids")) != query_ids:
        raise _artifact_error("query cohort mismatch: query ID compatibility check failed")
    if not _artifact_filters_match(run_spec.get("filters"), filters) or not _artifact_workloads_match(run_spec.get("workloads"), workloads):
        raise _artifact_error("filter/workload compatibility check failed")
    expected_rows = len(workloads) * len(filters) * len(query_ids)
    if require_formal_keyspace and expected_rows != 3 * 14 * 200:
        raise _artifact_error("formal execution requires the complete 3*14*q200 keyspace")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise _artifact_error("manifest outputs are missing")
    expected_truth_hash = _require_sha256(outputs.get("truth_csv_sha256"), "truth CSV")
    if not truth_csv.is_file() or sha256_file(truth_csv) != expected_truth_hash:
        raise _artifact_error("truth CSV SHA256 does not match the manifest")
    try:
        pair_map = _artifact_pair_map(
            manifest, workloads, filters, table, candidate_validity_predicate
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _artifact_error("manifest candidate/plan provenance is malformed") from exc
    expected_keys = {(workload.name, spec.name, query_no) for workload in workloads for spec in filters for query_no in query_ids}
    truth: dict[tuple[str, str, int], ExactTruth] = {}
    required_columns = {
        "workload", "filter_name", "predicate", "workload_scalar_predicate_sha256",
        "candidate_universe_predicate", "candidate_universe_predicate_sha256",
        "query_no", "query_id", "query_split", "k", "as_of", "self_excluded",
        "candidate_count", "candidate_min_id", "candidate_max_id", "candidate_ids_sha256", "exact_topk_ids", "exact_topk_distances_sq",
        "exact_topk_plus_one_ids", "exact_topk_plus_one_distances_sq", "kth_distance_sq", "tie_tolerance", "strict_closer_count", "boundary_tied",
    }
    try:
        with truth_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = required_columns - set(reader.fieldnames or ())
            if missing:
                raise _artifact_error(f"truth CSV has wrong schema: missing={sorted(missing)}")
            for row in reader:
                key = (str(row.get("workload")), str(row.get("filter_name")), int(row.get("query_no", -1)))
                if key not in expected_keys or key in truth:
                    raise _artifact_error(f"truth CSV has unexpected/duplicate key: {key}")
                workload_name, filter_name, query_no = key
                pair = pair_map[(workload_name, filter_name)]
                candidate = pair["candidate"]
                exact_backend = pair.get("exact_backend")
                try:
                    backend_valid = (
                        isinstance(exact_backend, dict)
                        and exact_backend.get("backend") == "faiss"
                        and exact_backend.get("class") == "IndexFlatL2"
                        and exact_backend.get("exact") is True
                        and exact_backend.get("local_positions_mapped_to_global_ids") is True
                        and exact_backend.get("order") == "squared_l2_then_global_id"
                        and int(exact_backend.get("index_ntotal", -1)) == int(candidate["count"])
                        and int(exact_backend.get("threads", 0)) == int(backend["threads"])
                        and int(exact_backend.get("search_calls", 0)) > 0
                        and all(
                            math.isfinite(float(exact_backend.get(field, math.nan)))
                            and float(exact_backend[field]) >= 0.0
                            for field in ("index_add_ms", "search_ms", "elapsed_ms")
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    backend_valid = False
                if not backend_valid:
                    raise _artifact_error(f"pair exact backend provenance is malformed: {key}")
                query_id = query_ids[query_no]
                ids = _csv_ints(row.get("exact_topk_ids"), "exact_topk_ids")
                distances_sq = _csv_floats(row.get("exact_topk_distances_sq"), "exact_topk_distances_sq")
                plus_one_ids = _csv_ints(row.get("exact_topk_plus_one_ids"), "exact_topk_plus_one_ids")
                plus_one_distances_sq = _csv_floats(row.get("exact_topk_plus_one_distances_sq"), "exact_topk_plus_one_distances_sq")
                kth_sq = float(row.get("kth_distance_sq", math.nan))
                source_tolerance = float(row.get("tie_tolerance", math.nan))
                if (
                    row.get("predicate") != next(spec.predicate for spec in filters if spec.name == filter_name)
                    or row.get("workload_scalar_predicate_sha256")
                    != workload_scalar_predicate_sha256(
                        next(spec.predicate for spec in filters if spec.name == filter_name)
                    )
                    or row.get("candidate_universe_predicate")
                    != candidate_validity_predicate
                    or row.get("candidate_universe_predicate_sha256")
                    != expected_validity_hash
                    or int(row.get("query_id", -1)) != query_id
                    or row.get("query_split") != query_splits[query_no]
                    or int(row.get("k", -1)) != k
                    or int(row.get("as_of", -1)) != as_of_by_workload[workload_name]
                    or str(row.get("self_excluded")).lower() != "true"
                    or len(ids) != k or len(distances_sq) != k
                    or len(plus_one_ids) != k + 1 or len(plus_one_distances_sq) != k + 1
                    or ids != plus_one_ids[:k] or distances_sq != plus_one_distances_sq[:k]
                    or len(set(plus_one_ids)) != len(plus_one_ids)
                    or query_id in plus_one_ids or any(value < 0 for value in plus_one_ids)
                    or any(right < left for left, right in zip(plus_one_distances_sq, plus_one_distances_sq[1:]))
                    or not math.isfinite(kth_sq) or kth_sq < 0.0
                    or not _artifact_float_equal(kth_sq, distances_sq[-1])
                    or not math.isfinite(source_tolerance) or not _artifact_float_equal(source_tolerance, distance_tolerance(kth_sq))
                    or int(row.get("candidate_count", -1)) != int(candidate["count"])
                    or int(row.get("candidate_min_id", -1)) != int(candidate["min_id"])
                    or int(row.get("candidate_max_id", -1)) != int(candidate["max_id"])
                    or row.get("candidate_ids_sha256") != candidate["sha256"]
                ):
                    raise _artifact_error(f"truth CSV record is stale or malformed: {key}")
                boundary_tied = plus_one_distances_sq[k] <= kth_sq + source_tolerance
                if _artifact_bool(row.get("boundary_tied")) != boundary_tied or int(row.get("strict_closer_count", -1)) != sum(value < kth_sq - source_tolerance for value in plus_one_distances_sq[:k]):
                    raise _artifact_error(f"truth CSV tie metadata is invalid: {key}")
                for check in pair["spot_checks"]:
                    if int(check["query_no"]) == query_no:
                        _validate_artifact_spot_check(check, query_id, plus_one_ids, plus_one_distances_sq)
                kth_l2 = math.sqrt(kth_sq)
                truth[key] = ExactTruth(ids, kth_l2, distance_tolerance(kth_l2), boundary_tied)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _artifact_error(f"truth CSV is unreadable: {truth_csv}") from exc
    if set(truth) != expected_keys or len(truth) != expected_rows:
        raise _artifact_error(f"truth CSV keyspace is incomplete: rows={len(truth)} expected={expected_rows}")
    for pair_key, pair in pair_map.items():
        if int(pair.get("as_of", -1)) != as_of_by_workload[pair_key[0]]:
            raise _artifact_error(f"manifest as_of is incompatible: {pair_key}")
        session = pair.get("session")
        if not isinstance(session, dict) or session.get("current_user") != principal:
            raise _artifact_error(f"manifest principal/session is incompatible: {pair_key}")
        relations = pair.get("relations")
        expected_relations = {relation: database_relations.get(relation) for relation in relations} if isinstance(relations, dict) else {}
        if (
            not isinstance(relations, dict)
            or set(relations) != {
                table,
                "public.amazon_review_facts",
                "public.amazon_product_dim",
                "public.amazon_principal_tenant_grants",
                "public.amazon_sql_native_buckets",
            }
            or relations != expected_relations
        ):
            raise _artifact_error(f"manifest relation/table fingerprint is incompatible: {pair_key}")
        for check in pair["spot_checks"]:
            query_no = int(check["query_no"])
            if query_no not in query_ids or int(check.get("query_id", -1)) != query_ids[query_no] or int(check["limit"]) != k + 1:
                raise _artifact_error(f"spot check is incompatible: {pair_key}")
    records_sha256 = canonical_sha256([
        [workload, filter_name, query_no, list(entry.ids), entry.kth_distance, entry.tie_tolerance, entry.boundary_tied]
        for (workload, filter_name, query_no), entry in sorted(truth.items())
    ])
    return truth, {
        "truth_csv_sha256": expected_truth_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "records_sha256": records_sha256,
        "run_spec_hash": str(manifest["run_spec_hash"]),
        "query_cohort_sha256": expected_cohort_hash,
        "candidate_universe_predicate_sha256": expected_validity_hash,
        "relation_epoch_sha256": str(expected_relation_epoch["sha256"]),
    }


def build_run_spec(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    workloads: Sequence[WorkloadSpec],
    calibration: dict[int, int],
    final: dict[int, int],
    database: dict[str, Any],
    as_of_by_workload: dict[str, int],
    external_truth_provenance: dict[str, str] | None = None,
    query_cohort_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stable_database = {
        key: database[key]
        for key in (
            "database",
            "postgres_version",
            "vector_extension_version",
            "sqlens_build_id",
            "profile_semantics_version",
            "required_profile_fields",
            "relations",
            "d2_graph_proof",
            "d2_index_names",
            "preferred_index_current_settings",
            "mode_indexes",
            "principal",
            "rls_security_proofs",
            "query_candidate_universe_proof",
        )
        if key in database
    }
    reset = database.get("d3_persistent_fragment_reset")
    if isinstance(reset, dict):
        reset_after = reset.get("after") if isinstance(reset.get("after"), dict) else {}
        stable_database["d3_persistent_fragment_empty_start"] = {
            "valid": reset.get("valid"),
            "count": reset_after.get("count"),
            "content_sha256": reset_after.get("content_sha256"),
            "prebuilt_fragments": reset.get("prebuilt_fragments"),
        }
    guard = database.get("formal_data_guard_start")
    if isinstance(guard, dict):
        stable_database["formal_data_version"] = {
            "lock_mode": guard.get("lock_mode"),
            "relations": guard.get("relations"),
            "start_relations": guard.get("start_relations"),
            "start_hash": guard.get("start_hash"),
        }
    combined_queries = {**calibration, **final}
    query_splits = {
        **{query_no: "calibration" for query_no in calibration},
        **{query_no: "final" for query_no in final},
    }
    cohort_hash = query_cohort_sha256(combined_queries, query_splits)
    validity_predicate = validate_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    data_relations = {
        relation: database.get("relations", {}).get(relation)
        for relation in exact_truth_contract.formal_data_relations(args.vector_table)
        if database.get("relations", {}).get(relation) is not None
    }
    result = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "runner_sha256": sha256_file(Path(__file__)),
        "schema_sql_sha256": sha256_file(args.schema_sql),
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "query_ids_csv_sha256": sha256_file(args.query_ids_csv),
        "query_cohort_manifest_sha256": sha256_file(args.query_cohort_manifest),
        "database": stable_database,
        "vector_table": args.vector_table,
        "source_index": args.source_index,
        "clone_index": args.clone_index,
        "mode_semantics": {
            mode: asdict(spec) for mode, spec in MODE_SPECS.items()
        },
        "principal": args.principal,
        "k": args.k,
        "targets": list(args.targets),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "schedule_seed": args.schedule_seed,
        "workloads": [asdict(item) for item in workloads],
        "filters": [asdict(item) for item in filters],
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
            "predicate_sha256": candidate_universe_predicate_sha256(
                validity_predicate
            ),
            "sql_role": "candidate_relation_only; separate from workload scalar predicate",
        },
        "query_cohort_sha256": cohort_hash,
        "query_cohort_hash_contract": exact_truth_contract.QUERY_COHORT_HASH_CONTRACT,
        "query_cohort": query_cohort_provenance
        or {
            "query_count": len(combined_queries),
            "query_cohort_sha256": cohort_hash,
            "query_cohort_hash_contract": exact_truth_contract.QUERY_COHORT_HASH_CONTRACT,
        },
        "relation_epoch": (
            relation_epoch_contract(data_relations) if data_relations else None
        ),
        "calibration": {
            "query_ids": [[query_no, query_id] for query_no, query_id in calibration.items()],
            "repeats": args.calibration_repeats,
        },
        "final": {
            "query_ids": [[query_no, query_id] for query_no, query_id in final.items()],
            "repeats": args.final_repeats,
        },
        "config_grids": {
            mode: [asdict(config) for config in build_config_grid(args, mode)] for mode in MODES
        },
        "as_of_by_workload": as_of_by_workload,
        "d3": {
            "initialization": "workload_driven_empty_cache_no_prebuilt_fragments",
            "probe_requests": args.d3_probe_requests,
            "min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "max_fragment_mb": args.d3_max_fragment_mb,
            "page_min_skip_rate": args.d3_page_min_skip_rate,
        },
        "sql_hashes": sql_contract_hashes(
            workloads,
            filters,
            args.vector_table,
            args.candidate_validity_predicate,
        ),
    }
    if external_truth_provenance is not None:
        result["external_exact_truth"] = external_truth_provenance
    return result


def validate_checkpoint_run_spec(checkpoint: dict[str, Any], run_spec: dict[str, Any]) -> None:
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise RuntimeError("checkpoint version mismatch")
    expected_hash = canonical_sha256(run_spec)
    if checkpoint.get("run_spec_sha256") != expected_hash or checkpoint.get("run_spec") != run_spec:
        raise RuntimeError("checkpoint run-spec mismatch; refusing stale resume")


def checkpoint_entry_path(path: Path, section: str, key: str) -> Path:
    return path / section / f"{canonical_sha256(key)}.json"


def initialize_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        atomic_write_json(temporary / "run_spec.json", checkpoint["run_spec"])
        persist_checkpoint_meta(temporary, checkpoint)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def persist_checkpoint_meta(path: Path, checkpoint: dict[str, Any]) -> None:
    atomic_write_json(
        path / "meta.json",
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "run_spec_sha256": canonical_sha256(checkpoint["run_spec"]),
            "loaded_sessions": checkpoint["loaded_sessions"],
        },
    )


def persist_checkpoint_entry(path: Path, section: str, key: str, value: dict[str, Any]) -> None:
    entry_path = checkpoint_entry_path(path, section, key)
    if entry_path.exists():
        try:
            existing = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"checkpoint shard is unreadable/incomplete: {entry_path}") from exc
        if existing != value:
            raise RuntimeError(f"checkpoint shard changed for immutable key: {key}")
        return
    atomic_write_json(entry_path, value)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        run_spec = json.loads((path / "run_spec.json").read_text(encoding="utf-8"))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"checkpoint is unreadable/incomplete: {path}") from exc

    def load_section(section: str) -> list[dict[str, Any]]:
        directory = path / section
        if not directory.exists():
            return []
        entries: list[dict[str, Any]] = []
        for entry_path in sorted(directory.glob("*.json")):
            try:
                entry = json.loads(entry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"checkpoint shard is unreadable/incomplete: {entry_path}"
                ) from exc
            if not isinstance(entry, dict):
                raise RuntimeError(f"checkpoint shard is not an object: {entry_path}")
            entries.append(entry)
        return entries

    return {
        "checkpoint_version": meta.get("checkpoint_version"),
        "run_spec": run_spec,
        "run_spec_sha256": meta.get("run_spec_sha256"),
        "loaded_sessions": meta.get("loaded_sessions"),
        "exact_truth": load_section("exact_truth"),
        "exact_plans": load_section("exact_plans"),
        "calibration_blocks": load_section("calibration_blocks"),
        "final_blocks": load_section("final_blocks"),
        "invalid_blocks": load_section("invalid_blocks"),
    }


def remove_checkpoint(path: Path) -> None:
    tombstone = path.with_name(f".{path.name}.completed-{os.getpid()}")
    os.replace(path, tombstone)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    shutil.rmtree(tombstone)


def exact_truth_record(
    workload: WorkloadSpec,
    spec: FilterSpec,
    query_no: int,
    query_id: int,
    query_split: str,
    as_of: int,
    table_fingerprint_sha256: str,
    sql_text: str,
    truth: ExactTruth,
) -> dict[str, Any]:
    return {
        "workload": workload.name,
        "filter_name": spec.name,
        "query_no": query_no,
        "query_id": query_id,
        "query_split": query_split,
        "as_of": as_of,
        "vector_table_fingerprint_sha256": table_fingerprint_sha256,
        "exact_sql": sql_text,
        "exact_sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
        "ids": list(truth.ids),
        "kth_distance": truth.kth_distance,
        "tie_tolerance": truth.tie_tolerance,
        "boundary_tied": truth.boundary_tied,
    }


def restore_exact_truth(
    records: Sequence[dict[str, Any]],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    query_splits: dict[int, str],
    as_of_by_workload: dict[str, int],
    table: str,
    table_fingerprint_sha256: str,
    k: int,
    *,
    require_complete: bool = False,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[tuple[str, str, int], ExactTruth]:
    workload_by_name = {item.name: item for item in workloads}
    filter_by_name = {item.name: item for item in filters}
    expected_keys = {
        (workload.name, spec.name, query_no)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
    }
    restored: dict[tuple[str, str, int], ExactTruth] = {}
    for record in records:
        key = (str(record.get("workload")), str(record.get("filter_name")), int(record.get("query_no", -1)))
        if key not in expected_keys or key in restored:
            raise RuntimeError(f"checkpoint exact GT has unexpected/duplicate key: {key}")
        workload = workload_by_name[key[0]]
        spec = filter_by_name[key[1]]
        query_no = key[2]
        sql_text = build_hybrid_sql(
            table,
            spec.predicate,
            workload=workload,
            exact=True,
            candidate_validity_predicate=candidate_validity_predicate,
        )
        validate_exact_sql_text(sql_text)
        expected_sql_hash = hashlib.sha256(sql_text.encode()).hexdigest()
        ids = tuple(int(value) for value in record.get("ids", []))
        kth_distance = float(record.get("kth_distance", math.nan))
        tie_tolerance = float(record.get("tie_tolerance", math.nan))
        valid = (
            int(record.get("query_id", -1)) == query_ids[query_no]
            and record.get("query_split") == query_splits[query_no]
            and int(record.get("as_of", -1)) == as_of_by_workload[workload.name]
            and record.get("vector_table_fingerprint_sha256") == table_fingerprint_sha256
            and record.get("exact_sql") == sql_text
            and record.get("exact_sql_sha256") == expected_sql_hash
            and len(ids) == k
            and len(set(ids)) == k
            and all(value != query_ids[query_no] for value in ids)
            and math.isfinite(kth_distance)
            and kth_distance >= 0.0
            and math.isfinite(tie_tolerance)
            and tie_tolerance == distance_tolerance(kth_distance)
            and isinstance(record.get("boundary_tied"), bool)
        )
        if not valid:
            raise RuntimeError(f"checkpoint exact GT is incomplete or stale: {key}")
        restored[key] = ExactTruth(ids, kth_distance, tie_tolerance, record["boundary_tied"])
    if require_complete and set(restored) != expected_keys:
        missing = sorted(expected_keys - set(restored))
        raise RuntimeError(f"checkpoint exact GT is incomplete: missing={missing[:5]} count={len(missing)}")
    return restored


def calibration_block_id(workload: str, filter_name: str, mode: str, config: Config) -> str:
    return f"calibration|{workload}|{filter_name}|{mode}|{config.label}"


def final_block_id(workload: str, filter_name: str, target: float) -> str:
    return f"final|{workload}|{filter_name}|target{target:.12g}"


def validate_measurement_block(
    block: dict[str, Any],
    *,
    phase: str,
    workload: WorkloadSpec,
    spec: FilterSpec,
    query_ids: dict[int, int],
    repeats: int,
    modes: Sequence[str],
    configs: dict[str, Config],
    target_recall: float | None,
    truth: dict[tuple[str, str, int], ExactTruth],
    table: str,
    principal: str,
    source_index: str,
    clone_index: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> None:
    expected = {
        (mode, query_no, repeat)
        for mode in modes
        for query_no in query_ids
        for repeat in range(repeats)
    }
    rows = block.get("rows")
    plans = block.get("plans")
    if not isinstance(rows, list) or not isinstance(plans, list):
        raise RuntimeError("checkpoint measurement block is incomplete")
    observed: set[tuple[str, int, int]] = set()
    sql_text = build_hybrid_sql(
        table,
        spec.predicate,
        workload=workload,
        candidate_validity_predicate=candidate_validity_predicate,
    )
    sql_hash = hashlib.sha256(sql_text.encode()).hexdigest()
    for row in rows:
        key = (str(row.get("mode")), int(row.get("query_no", -1)), int(row.get("repeat", -1)))
        if key in observed:
            raise RuntimeError(f"checkpoint measurement block has duplicate row: {key}")
        observed.add(key)
        mode, query_no, repeat = key
        truth_entry = truth.get((workload.name, spec.name, query_no))
        if mode not in configs or truth_entry is None:
            raise RuntimeError(f"checkpoint measurement block has unexpected row: {key}")
        expected_target: Any = "" if target_recall is None else target_recall
        expected_index = mode_index(mode, source_index, clone_index)
        expected_strategy = MODE_SPECS[mode].filter_strategy
        valid = (
            row.get("phase") == phase
            and row.get("workload") == workload.name
            and row.get("filter_name") == spec.name
            and row.get("predicate") == spec.predicate
            and row.get("workload_scalar_predicate_sha256")
            == workload_scalar_predicate_sha256(spec.predicate)
            and row.get("candidate_universe_predicate")
            == candidate_validity_predicate
            and row.get("candidate_universe_predicate_sha256")
            == candidate_universe_predicate_sha256(candidate_validity_predicate)
            and row.get("config") == configs[mode].label
            and int(row.get("query_id", -1)) == query_ids[query_no]
            and row.get("target_recall") == expected_target
            and row.get("pair_key") == f"{workload.name}|{spec.name}|q{query_no}|r{repeat}"
            and row.get("query_sql") == sql_text
            and row.get("query_sql_sha256") == sql_hash
            and row.get("exact_gt_ids") == ",".join(str(value) for value in truth_entry.ids)
            and float(row.get("exact_gt_kth_distance", math.nan)) == truth_entry.kth_distance
            and float(row.get("exact_gt_tie_tolerance", math.nan)) == truth_entry.tie_tolerance
            and row.get("exact_gt_boundary_tied") is truth_entry.boundary_tied
            and row.get("selected_vector_index") == expected_index
            and row.get("preferred_index_current_setting") == expected_index
            and row.get("principal") == principal
            and row.get("snapshot_as_of") == as_of_value(workload.name, row)
            and row.get("filter_strategy") == expected_strategy
            and row.get("guidance_semantics") == MODE_SPECS[mode].guidance_semantics
            and not bool(row.get("hard_traversal_used"))
            and recorded_guidance_proof_is_valid(row)
        )
        if not valid:
            raise RuntimeError(f"checkpoint measurement row is incomplete or stale: {key}")
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("checkpoint measurement block has wrong row count/query IDs/repeats")
    if len(plans) != len(modes):
        raise RuntimeError("checkpoint measurement block has wrong EXPLAIN plan count")
    for plan in plans:
        mode = str(plan.get("mode"))
        gate = plan.get("explain_gate", {})
        expected_index = mode_index(mode, source_index, clone_index) if mode in MODE_SPECS else ""
        if (
            mode not in configs
            or plan.get("phase") != phase
            or plan.get("workload") != workload.name
            or plan.get("filter_name") != spec.name
            or plan.get("config") != configs[mode].label
            or plan.get("sql_sha256") != sql_hash
            or not gate.get("valid")
            or gate.get("require_hnsw") is not True
            or gate.get("expected_index_qualified") != expected_index
            or plan.get("selected_vector_index") != expected_index
            or plan.get("preferred_index_current_setting") != expected_index
            or plan.get("principal") != principal
            or plan.get("snapshot_as_of") != int(plan.get("as_of", -1))
            or plan.get("filter_strategy") != MODE_SPECS[mode].filter_strategy
            or bool(plan.get("hard_traversal_used"))
            or plan.get("plan_state_proof", {}).get("valid") is not True
            or plan.get("explain_order") != "after_all_timed_requests_in_block"
        ):
            raise RuntimeError("checkpoint measurement EXPLAIN plan is incomplete or stale")


def as_of_value(workload_name: str, row: dict[str, Any]) -> int:
    try:
        return int(row["as_of"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"measurement row has invalid as_of for {workload_name}") from exc


def validate_exact_plans(
    plans: Sequence[dict[str, Any]],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    table: str,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> None:
    expected = {(workload.name, spec.name) for workload in workloads for spec in filters}
    observed: set[tuple[str, str]] = set()
    workload_by_name = {item.name: item for item in workloads}
    filter_by_name = {item.name: item for item in filters}
    for plan in plans:
        key = (str(plan.get("workload")), str(plan.get("filter_name")))
        if key not in expected or key in observed:
            raise RuntimeError(f"checkpoint exact EXPLAIN plan has unexpected/duplicate key: {key}")
        observed.add(key)
        sql_text = build_hybrid_sql(
            table,
            filter_by_name[key[1]].predicate,
            workload=workload_by_name[key[0]],
            exact=True,
            candidate_validity_predicate=candidate_validity_predicate,
        )
        validate_exact_sql_text(sql_text)
        gate = plan.get("explain_gate", {})
        if (
            plan.get("phase") != "exact_gt"
            or plan.get("mode") != "exact_gt"
            or plan.get("sql_sha256") != hashlib.sha256(sql_text.encode()).hexdigest()
            or not gate.get("valid")
            or gate.get("require_hnsw") is not False
        ):
            raise RuntimeError(f"checkpoint exact EXPLAIN plan is stale: {key}")


def run_exact_truth(
    cur: Any,
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    as_of_by_workload: dict[str, int],
    table: str,
    vector_index: str,
    k: int,
    plans: list[dict[str, Any]],
    existing: dict[tuple[str, str, int], ExactTruth] | None = None,
    on_truth: Any | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> dict[tuple[str, str, int], ExactTruth]:
    truth: dict[tuple[str, str, int], ExactTruth] = dict(existing or {})
    for workload in workloads:
        for spec in filters:
            sql_text = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                exact=True,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            validate_exact_sql_text(sql_text)
            first_query_id = next(iter(query_ids.values()))
            set_as_of(cur, as_of_by_workload[workload.name])
            exact_plan, gate = explain(
                cur,
                sql_text,
                {
                    "query_id": first_query_id,
                    "as_of": as_of_by_workload[workload.name],
                    "k": k,
                },
                vector_index=vector_index,
                require_hnsw=False,
            )
            prior_plan = next(
                (
                    plan
                    for plan in plans
                    if plan.get("phase") == "exact_gt"
                    and plan.get("workload") == workload.name
                    and plan.get("filter_name") == spec.name
                ),
                None,
            )
            if prior_plan is None:
                plans.append(
                    {
                        "phase": "exact_gt",
                        "workload": workload.name,
                        "filter_name": spec.name,
                        "mode": "exact_gt",
                        "sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                        "plan": exact_plan,
                        "explain_gate": gate,
                    }
                )
            for query_no, query_id in query_ids.items():
                if (workload.name, spec.name, query_no) in truth:
                    continue
                set_as_of(cur, as_of_by_workload[workload.name])
                results = query_results(
                    cur,
                    sql_text,
                    {
                        "query_id": query_id,
                        "as_of": as_of_by_workload[workload.name],
                        "k": k + 1,
                    },
                    exact=True,
                )
                ids = [row_id for row_id, _ in results[:k]]
                if len(ids) != k or len(set(ids)) != k:
                    raise RuntimeError(
                        f"exact SQL GT incomplete for {workload.name}/{spec.name}/q{query_no}: {len(ids)} rows"
                    )
                kth_distance = float(results[k - 1][1])
                tolerance = distance_tolerance(kth_distance)
                truth_entry = ExactTruth(
                    ids=tuple(ids),
                    kth_distance=kth_distance,
                    tie_tolerance=tolerance,
                    boundary_tied=(
                        len(results) > k and float(results[k][1]) <= kth_distance + tolerance
                    ),
                )
                truth[(workload.name, spec.name, query_no)] = truth_entry
                if on_truth is not None:
                    on_truth(workload, spec, query_no, query_id, sql_text, truth_entry)
    return truth


def run_measurements(
    connections: dict[str, Any],
    configs: dict[str, Config],
    workloads: Sequence[WorkloadSpec],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    truth: dict[tuple[str, str, int], ExactTruth | tuple[int, ...]],
    as_of_by_workload: dict[str, int],
    table: str,
    source_index: str,
    clone_index: str,
    principal: str,
    k: int,
    repeats: int,
    phase: str,
    target_recall: float | None,
    schedule_seed: int,
    selected_modes: Sequence[str] | None = None,
    d3_settings: dict[str, Any] | None = None,
    fragment_store_reset: dict[str, Any] | None = None,
    candidate_validity_predicate: str = DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_modes = tuple(selected_modes or MODES)
    if not active_modes or any(mode not in MODES for mode in active_modes):
        raise ValueError(f"selected_modes must be a non-empty subset of {MODES}")
    if any(mode not in configs for mode in active_modes):
        raise ValueError(f"missing config for selected mode: {active_modes}")
    keys = [
        (workload.name, spec.name, query_no, repeat)
        for workload in workloads
        for spec in filters
        for query_no in query_ids
        for repeat in range(repeats)
    ]
    rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    cursors = {mode: connections[mode].cursor() for mode in active_modes}
    selected_indexes = {
        mode: mode_index(mode, source_index, clone_index) for mode in active_modes
    }
    try:
        for mode in active_modes:
            set_mode(
                cursors[mode], mode, configs[mode], selected_indexes[mode], d3_settings
            )
        schedule = interleaved_schedule(keys, active_modes, schedule_seed)
        for position, (key, mode) in enumerate(schedule):
            workload_name, filter_name, query_no, repeat = key
            workload = next(item for item in workloads if item.name == workload_name)
            spec = next(item for item in filters if item.name == filter_name)
            query_id = query_ids[query_no]
            sql_text = build_hybrid_sql(
                table,
                spec.predicate,
                workload=workload,
                candidate_validity_predicate=candidate_validity_predicate,
            )
            params = {
                "query_id": query_id,
                "as_of": as_of_by_workload[workload.name],
                "k": k,
                "vector_index": selected_indexes[mode],
                "binding_atoms": list(spec.atoms),
                "binding_kind": MODE_SPECS[mode].guidance_kind or "bloom",
            }
            cur = cursors[mode]
            error = ""
            ids: list[int] = []
            result_pairs: list[tuple[int, float]] = []
            activation_ms = 0.0
            query_ms = 0.0
            elapsed_ms = 0.0
            activation: dict[str, Any] = {}
            post_guidance: dict[str, Any] = {}
            execution_guidance: dict[str, Any] = {}
            scan_profile: dict[str, Any] = {}
            context: dict[str, Any] = {}
            try:
                set_as_of(cur, as_of_by_workload[workload.name])
                activation = configure_guidance(
                    cur, mode, selected_indexes[mode], spec.atoms
                )
                activation_ms = float(activation["activation_ms"])
                context = runtime_sql_context(
                    cur, principal, as_of_by_workload[workload.name]
                )
                if context["preferred_index"] != selected_indexes[mode]:
                    raise RuntimeError(
                        f"preferred index changed before query for {mode}: {context}"
                    )
                cur.execute("SELECT vector_hnsw_reset_scan_profile()")
                query_started = time.perf_counter()
                cur.execute(sql_text, params)
                query_rows = cur.fetchall()
                result_pairs = [(int(row[0]), float(row[1])) for row in query_rows]
                ids = [row_id for row_id, _ in result_pairs]
                query_ms = (time.perf_counter() - query_started) * 1000.0
                # Profile/context probes sit outside both component timers.
                # SQLens request latency is the required guidance activation
                # followed by the real hybrid SQL execute+fetch path.
                elapsed_ms = activation_ms + query_ms
                if query_rows and len(query_rows[-1]) > 2:
                    value = query_rows[-1][2]
                    execution_guidance = (
                        json.loads(value) if isinstance(value, str) else dict(value or {})
                    )
                post_guidance = fetch_json_object(
                    cur, "SELECT vector_hnsw_guidance_profile()"
                )
                scan_profile = fetch_json_object(
                    cur, "SELECT vector_hnsw_last_scan_profile()"
                )
            except Exception as exc:  # noqa: BLE001 - retain failed pair in the artifact.
                error = f"{exc.__class__.__name__}: {exc}"
                try:
                    cur.execute("ROLLBACK")
                    set_mode(
                        cur,
                        mode,
                        configs[mode],
                        selected_indexes[mode],
                        d3_settings,
                    )
                except Exception:
                    pass
            if error:
                activation_ms = 0.0
                query_ms = 0.0
                elapsed_ms = 0.0
            truth_entry = truth[(workload.name, spec.name, query_no)]
            truth_ids = truth_entry.ids if isinstance(truth_entry, ExactTruth) else truth_entry
            recall = (
                tie_aware_recall_at_k(result_pairs, truth_entry, query_id, k)
                if isinstance(truth_entry, ExactTruth)
                else recall_at_k(ids, truth_ids, k)
            )
            adaptive = (
                adaptive_transition_for_request(activation, post_guidance)
                if MODE_SPECS[mode].adaptive and not error
                else {
                    "adaptive_state_before": "not_adaptive",
                    "adaptive_state_after_activation": "not_adaptive",
                    "adaptive_state_after_query": "not_adaptive",
                    "adaptive_probe_observed": False,
                    "adaptive_admission_observed": False,
                    "adaptive_materialized": False,
                    "adaptive_active": False,
                    "hidden_prebuilt_fragment_reused": False,
                    "fragment_store_hit_delta": 0,
                    "adaptive_transition": "not_adaptive",
                }
            )
            final_path = str(scan_profile.get("final_path", ""))
            execution_proof = guidance_execution_proof(
                mode, activation, execution_guidance, scan_profile
            )
            if not error and not execution_proof["valid"]:
                error = (
                    "RuntimeError: per-row guidance execution proof failed: "
                    + json.dumps(execution_proof, sort_keys=True)
                )
            hard_traversal_used = bool(
                mode in SQLENS_MODES
                and (
                    not bool(execution_proof.get("valid"))
                    or context.get("filter_strategy") not in {"off", "safe_guided"}
                    or final_path in {"guided", "legacy_guided", "unknown", ""}
                )
            )
            rows.append(
                {
                    "phase": phase,
                    "target_recall": target_recall if target_recall is not None else "",
                    "workload": workload.name,
                    "filter_name": filter_name,
                    "predicate": spec.predicate,
                    "workload_scalar_predicate_sha256": workload_scalar_predicate_sha256(
                        spec.predicate
                    ),
                    "candidate_universe_predicate": candidate_validity_predicate,
                    "candidate_universe_predicate_sha256": candidate_universe_predicate_sha256(
                        candidate_validity_predicate
                    ),
                    "as_of": as_of_by_workload[workload.name],
                    "principal": context.get("current_user", ""),
                    "snapshot_as_of": (
                        int(context["app_as_of"]) if context.get("app_as_of") else -1
                    ),
                    "mode": mode,
                    "selected_vector_index": selected_indexes[mode],
                    "preferred_index_current_setting": context.get(
                        "preferred_index", ""
                    ),
                    "filter_strategy": MODE_SPECS[mode].filter_strategy,
                    "filter_strategy_current_setting": context.get(
                        "filter_strategy", ""
                    ),
                    "page_access_current_setting": context.get("page_access", ""),
                    "index_page_access_current_setting": context.get(
                        "index_page_access", ""
                    ),
                    "guidance_kind": MODE_SPECS[mode].guidance_kind or "none",
                    "guidance_semantics": MODE_SPECS[mode].guidance_semantics,
                    "hard_traversal_used": hard_traversal_used,
                    "traversal_final_path": final_path,
                    "config": configs[mode].label,
                    "query_no": query_no,
                    "query_id": query_id,
                    "repeat": repeat,
                    "pair_key": f"{workload.name}|{filter_name}|q{query_no}|r{repeat}",
                    "schedule_position": position,
                    "execution_order": "interleaved",
                    "query_sql": sql_text,
                    "query_sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                    "exact_gt_ids": ",".join(str(value) for value in truth_ids),
                    "exact_gt_kth_distance": (
                        truth_entry.kth_distance if isinstance(truth_entry, ExactTruth) else NA
                    ),
                    "exact_gt_tie_tolerance": (
                        truth_entry.tie_tolerance if isinstance(truth_entry, ExactTruth) else NA
                    ),
                    "exact_gt_boundary_tied": (
                        truth_entry.boundary_tied if isinstance(truth_entry, ExactTruth) else NA
                    ),
                    "returned_ids": ",".join(str(value) for value in ids),
                    "returned_distances": ",".join(f"{distance:.17g}" for _, distance in result_pairs),
                    "returned": len(ids),
                    "recall": recall if not error else NA,
                    "activation_ms": activation_ms if not error else NA,
                    "query_ms": query_ms if not error else NA,
                    "e2e_ms": elapsed_ms if not error else NA,
                    "guidance_enabled": bool(
                        activation.get("guidance_enabled", False)
                    ),
                    "guidance_route": activation.get("guidance_route", ""),
                    "activation_atom_count": activation.get(
                        "activation_atom_count", 0
                    ),
                    "adaptive_initialization": (
                        "workload_driven_empty_cache_no_prebuilt_fragments"
                        if MODE_SPECS[mode].adaptive
                        else "not_adaptive"
                    ),
                    "prebuilt_fragments": (
                        int(fragment_store_reset.get("prebuilt_fragments", -1))
                        if fragment_store_reset is not None
                        else NA
                    ),
                    "persistent_fragment_reset_proof": (
                        fragment_store_reset or {"status": "not_supplied"}
                    ),
                    "guidance_activation_profile": activation,
                    "execution_guidance_profile": execution_guidance,
                    "post_query_guidance_profile": post_guidance,
                    "guidance_execution_proof": execution_proof,
                    "guidance_binding_matched": execution_proof["binding_matched"],
                    "guidance_effective_active": execution_proof["effective_active"],
                    "guidance_checks": execution_proof["guidance_checks"],
                    "guidance_final_path": execution_proof["final_path"],
                    **adaptive,
                    **scan_profile_export(scan_profile),
                    "scan_profile": scan_profile,
                    "error": error,
                }
            )
        for workload in workloads:
            for spec in filters:
                sql_text = build_hybrid_sql(
                    table,
                    spec.predicate,
                    workload=workload,
                    candidate_validity_predicate=candidate_validity_predicate,
                )
                params = {
                    "query_id": next(iter(query_ids.values())),
                    "as_of": as_of_by_workload[workload.name],
                    "k": k,
                    "vector_index": "",
                    "binding_atoms": list(spec.atoms),
                    "binding_kind": "bloom",
                }
                for mode in active_modes:
                    vector_index = selected_indexes[mode]
                    params["vector_index"] = vector_index
                    params["binding_kind"] = MODE_SPECS[mode].guidance_kind or "bloom"
                    set_mode(
                        cursors[mode], mode, configs[mode], vector_index, d3_settings
                    )
                    set_as_of(cursors[mode], as_of_by_workload[workload.name])
                    plan_state_before = prepare_explain_without_runtime_state(
                        cursors[mode]
                    )
                    context = runtime_sql_context(
                        cursors[mode], principal, as_of_by_workload[workload.name]
                    )
                    if context["preferred_index"] != vector_index:
                        raise RuntimeError(
                            f"preferred index changed before EXPLAIN for {mode}: {context}"
                        )
                    plan, gate = explain(
                        cursors[mode],
                        sql_text,
                        params,
                        vector_index=vector_index,
                        require_hnsw=True,
                    )
                    plan_state_proof = finish_explain_without_runtime_state(
                        cursors[mode], plan_state_before
                    )
                    plans.append(
                        {
                            "phase": phase,
                            "target_recall": target_recall,
                            "workload": workload.name,
                            "filter_name": spec.name,
                            "mode": mode,
                            "config": configs[mode].label,
                            "sql_sha256": hashlib.sha256(sql_text.encode()).hexdigest(),
                            "as_of": as_of_by_workload[workload.name],
                            "principal": context["current_user"],
                            "snapshot_as_of": int(context["app_as_of"]),
                            "selected_vector_index": vector_index,
                            "preferred_index_current_setting": context["preferred_index"],
                            "filter_strategy": MODE_SPECS[mode].filter_strategy,
                            "filter_strategy_current_setting": context["filter_strategy"],
                            "page_access_current_setting": context["page_access"],
                            "index_page_access_current_setting": context[
                                "index_page_access"
                            ],
                            "guidance_semantics": MODE_SPECS[mode].guidance_semantics,
                            "hard_traversal_used": False,
                            "explain_order": "after_all_timed_requests_in_block",
                            "plan_state_proof": plan_state_proof,
                            "plan": plan,
                            "explain_gate": gate,
                        }
                    )
    finally:
        for cur in cursors.values():
            cur.close()
    return rows, plans


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    target = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, target.getvalue())


def publish_benchmark_artifacts(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    plans: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    checkpoint_path: Path,
) -> int:
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    plans_path = args.plans or args.out.with_suffix(".plans.json")
    valid = manifest.get("artifact_valid") is True
    manifest["formal_outputs_published"] = valid
    if not valid:
        manifest["checkpoint_preserved"] = str(checkpoint_path)
        atomic_write_json(plans_path, plans)
        atomic_write_json(manifest_path, manifest)
        return 2
    write_csv(args.out, rows)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summaries)
    atomic_write_json(plans_path, plans)
    atomic_write_json(manifest_path, manifest)
    remove_checkpoint(checkpoint_path)
    return 0


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Amazon-10M SQL-native PostgreSQL+pgvector hybrid benchmark."
    )
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--schema-sql", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--query-ids-csv", type=Path, default=DEFAULT_QUERY_IDS)
    parser.add_argument(
        "--query-cohort-manifest",
        type=Path,
        default=DEFAULT_QUERY_COHORT_MANIFEST,
        help="provenance manifest for the truth-format formal q200 cohort CSV",
    )
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--exact-truth-csv", type=Path, default=DEFAULT_EXACT_TRUTH_CSV)
    parser.add_argument("--exact-truth-manifest", type=Path, default=DEFAULT_EXACT_TRUTH_MANIFEST)
    parser.add_argument("--vector-table", type=qualified_name_arg, default=DEFAULT_VECTOR_TABLE)
    parser.add_argument(
        "--source-index",
        "--vector-index",
        dest="source_index",
        type=qualified_name_arg,
        default=DEFAULT_SOURCE_INDEX,
    )
    parser.add_argument("--clone-index", type=qualified_name_arg, default=DEFAULT_CLONE_INDEX)
    parser.add_argument("--principal", type=parse_role_name, default=DEFAULT_PRINCIPAL)
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE,
        help="global candidate-universe SQL predicate; formal value is embedding_valid",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS / "amazon10m_sql_native_benchmark.csv")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--plans", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("--calibration-query-offset", type=nonnegative_int, default=0)
    parser.add_argument("--calibration-queries", type=positive_int, default=DEFAULT_CALIBRATION_QUERIES)
    parser.add_argument("--calibration-repeats", type=positive_int, default=DEFAULT_CALIBRATION_REPEATS)
    parser.add_argument("--final-query-offset", type=nonnegative_int, default=100)
    parser.add_argument("--final-queries", type=positive_int, default=DEFAULT_FINAL_QUERIES)
    parser.add_argument("--final-repeats", type=positive_int, default=DEFAULT_FINAL_REPEATS)
    parser.add_argument("--targets", type=parse_targets, default=list(TARGET_RECALLS))
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--ef-search-values", type=parse_int_list, default=[250, 500, 1000, 2000, 5000, 10000])
    parser.add_argument("--max-scan-tuples-values", type=parse_int_list, default=[5_000_000])
    parser.add_argument("--scan-mem-multiplier-values", type=lambda value: parse_float_list(value), default=[32.0])
    parser.add_argument(
        "--iterative-scan-values",
        type=lambda value: parse_word_list(value, {"off", "strict_order", "relaxed_order"}),
        default=["strict_order", "relaxed_order"],
    )
    parser.add_argument("--guided-collect-target-values", type=parse_guided_targets, default=["ef"])
    parser.add_argument(
        "--d3-probe-requests", type=positive_int, default=DEFAULT_D3_PROBE_REQUESTS
    )
    parser.add_argument(
        "--d3-min-benefit-per-byte",
        type=float,
        default=DEFAULT_D3_MIN_BENEFIT_PER_BYTE,
    )
    parser.add_argument(
        "--d3-max-fragment-mb", type=positive_int, default=DEFAULT_D3_MAX_FRAGMENT_MB
    )
    parser.add_argument(
        "--d3-page-min-skip-rate",
        type=float,
        default=DEFAULT_D3_PAGE_MIN_SKIP_RATE,
    )
    parser.add_argument("--bootstrap-samples", type=positive_int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--dry-run", action="store_true", help="print the run contract without reading files or opening PostgreSQL")
    parser.add_argument("--execute", action="store_true", help="open PostgreSQL and run the benchmark")
    parser.add_argument(
        "--debug-compute-exact-truth",
        action="store_true",
        help="DEBUG ONLY: compute PostgreSQL exact GT instead of requiring the precomputed artifact",
    )
    parser.add_argument("--resume", action="store_true", help="strictly resume a matching atomic checkpoint")
    return parser


def validate_formal_dimensions(
    args: argparse.Namespace, filters: Sequence[FilterSpec]
) -> None:
    problems: list[str] = []
    if len(filters) != 14 or len({spec.name for spec in filters}) != 14:
        problems.append("exactly 14 distinct registered filters are required")
    if args.calibration_query_offset != 0 or args.calibration_queries != 100:
        problems.append("calibration must be q100 at offset 0")
    if args.final_query_offset != 100 or args.final_queries != 100:
        problems.append("final must be disjoint q100 at offset 100")
    if args.calibration_repeats != 2 or args.final_repeats != 5:
        problems.append("repeats must be calibration r2 and final r5")
    if [float(value) for value in args.targets] != list(TARGET_RECALLS):
        problems.append("matched-recall targets must be 0.90,0.95,0.99")
    if problems:
        raise RuntimeError("formal experiment dimensions are invalid: " + "; ".join(problems))


def print_dry_run(args: argparse.Namespace) -> None:
    print("mode=dry-run")
    print("database=not_opened")
    print("execution=single PostgreSQL SELECT with pgvector ORDER BY plus JOIN/ACL/RLS/temporal predicates")
    print("modes=" + ",".join(MODES))
    print("guidance_claim=candidate-admission/validation; hard-traversal-pruning=false")
    print("workloads=" + ",".join(item.name for item in WORKLOADS))
    print("calibration=q100/r2; final=q100/r5; final_queries_disjoint=true")
    print("targets=" + ",".join(f"{target:.2f}" for target in args.targets))
    print("bootstrap_samples=" + str(args.bootstrap_samples))
    print(
        "config_grid="
        f"ef{args.ef_search_values};max_scan{args.max_scan_tuples_values};"
        f"mem{args.scan_mem_multiplier_values}"
    )
    print("timing=" + TIMING_DEFINITION)
    print(f"filters_csv={args.filters_csv}")
    print(f"schema_sql={args.schema_sql}")
    print(f"query_ids_csv={args.query_ids_csv}")
    print(f"query_cohort_manifest={args.query_cohort_manifest}")
    print(f"candidate_validity_predicate={args.candidate_validity_predicate}")
    print(
        "candidate_universe_predicate_sha256="
        + candidate_universe_predicate_sha256(args.candidate_validity_predicate)
    )
    print(f"exact_truth_csv={args.exact_truth_csv}")
    print(f"exact_truth_manifest={args.exact_truth_manifest}")
    print(f"vector_table={args.vector_table}")
    print(f"source_index={args.source_index}")
    print(f"clone_index={args.clone_index}")
    print(f"out={args.out}")


def _manifest(
    args: argparse.Namespace,
    filters: Sequence[FilterSpec],
    calibration: dict[int, int],
    final: dict[int, int],
    database: dict[str, Any],
    as_of: dict[str, int],
    query_cohort_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    combined_queries = {**calibration, **final}
    query_splits = {
        **{query_no: "calibration" for query_no in calibration},
        **{query_no: "final" for query_no in final},
    }
    cohort_hash = query_cohort_sha256(combined_queries, query_splits)
    validity_predicate = validate_candidate_validity_predicate(
        args.candidate_validity_predicate
    )
    data_version = database.get("formal_data_version_proof", {})
    epoch = relation_epoch_contract(data_version.get("start_relations", {}))
    return {
        "artifact_valid": True,
        "benchmark": "amazon10m_sql_native_hybrid",
        "dataset": "Amazon-10M real SQL-derived Grocery workload",
        "git_revision": git_revision(),
        "runner_sha256": sha256_file(Path(__file__)),
        "schema_sql_sha256": sha256_file(args.schema_sql),
        "filters_csv_sha256": sha256_file(args.filters_csv),
        "query_ids_csv_sha256": sha256_file(args.query_ids_csv),
        "query_cohort_manifest_sha256": sha256_file(args.query_cohort_manifest),
        "database": database,
        "vector_table": args.vector_table,
        "source_index": args.source_index,
        "clone_index": args.clone_index,
        "principal": args.principal,
        "modes": list(MODES),
        "mode_semantics": {mode: asdict(spec) for mode, spec in MODE_SPECS.items()},
        "sqlens_filter_strategy": "safe_guided candidate-admission/validation only",
        "workloads": [asdict(item) for item in WORKLOADS],
        "filters": [asdict(item) for item in filters],
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
            "predicate_sha256": candidate_universe_predicate_sha256(
                validity_predicate
            ),
            "sql_role": "candidate_relation_only; separate from workload scalar predicate",
        },
        "candidate_universe_predicate_sha256": candidate_universe_predicate_sha256(
            validity_predicate
        ),
        "query_cohort": query_cohort_provenance
        or {
            "query_count": len(combined_queries),
            "query_cohort_sha256": cohort_hash,
            "query_cohort_hash_contract": exact_truth_contract.QUERY_COHORT_HASH_CONTRACT,
        },
        "query_cohort_sha256": cohort_hash,
        "relation_epoch": epoch,
        "calibration": {"queries": len(calibration), "repeats": args.calibration_repeats, "query_nos": list(calibration), "query_ids": list(calibration.values())},
        "final": {"queries": len(final), "repeats": args.final_repeats, "query_nos": list(final), "query_ids": list(final.values())},
        "as_of_by_workload": as_of,
        "sql_hashes": sql_contract_hashes(
            WORKLOADS,
            filters,
            args.vector_table,
            args.candidate_validity_predicate,
        ),
        "target_recalls": args.targets,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed, "recall_lcb": "5th percentile of query-level bootstrap means"},
        "execution_order": "interleaved paired by workload/filter/query/repeat",
        "timing_definition": TIMING_DEFINITION,
        "sql_contract": {
            "single_select": True,
            "all_modes_same_sql_text_and_relational_semantics": True,
            "statement_binding": "uncorrelated InitPlan; mismatch fails open to stock HNSW",
            "marker_semantics": "the approximate SQL binding marker is always true; per-row executor binding and scan proof controls guidance claims",
            "approx_order_by": "vector distance only; no secondary v.id key so HNSW can satisfy the order",
            "exact_gt": "valid AS MATERIALIZED isolates exact vector sorting; B-tree JOIN indexes remain available; ORDER BY distance, id",
            "recall": "distance-threshold tie-aware using PostgreSQL exact kth distance; query row excluded",
        },
        "rls_and_guidance_contract": {
            "facts_policy_always_enforced": True,
            "rls_table": "public.amazon_review_facts",
            "rls_policy": "amazon_review_facts_acl_select",
            "rls_scope": "ACL only; grant/fact temporal predicates remain explicit workload SQL",
            "guidance_scope": "safe candidate-admission/validation superset for row-local predicates on the non-RLS vector heap only",
            "hard_traversal_pruning": False,
            "hard_traversal_equivalence": (
                "intentionally ineligible: normalized JOIN, RLS-derived ACL, and temporal "
                "residuals are executor semantics that the row-local guide cannot prove equivalent"
            ),
            "executor_recheck": ["JOIN", "ACL", "temporal", "RLS"],
            "guidance_never_replaces": ["JOIN", "ACL", "temporal", "RLS"],
            "rls_relation_fingerprints": {
                relation: fingerprint
                for relation, fingerprint in database.get("relations", {}).items()
                if fingerprint.get("rls")
            },
        },
        "d2_graph_contract": database.get("d2_graph_proof", {}),
        "preferred_index_current_settings": database.get(
            "preferred_index_current_settings", {}
        ),
        "d3_contract": {
            "initialization": "workload_driven_empty_cache_no_prebuilt_fragments",
            "prebuilt_fragments": database.get(
                "d3_persistent_fragment_reset", {}
            ).get("prebuilt_fragments", NA),
            "probe_requests": args.d3_probe_requests,
            "min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "max_fragment_mb": args.d3_max_fragment_mb,
            "page_min_skip_rate": args.d3_page_min_skip_rate,
            "startup_reset_evidence": database.get("d3_startup_reset_evidence", {}),
            "persistent_fragment_reset": database.get(
                "d3_persistent_fragment_reset", {}
            ),
            "persistent_fragment_store_end": database.get(
                "d3_fragment_store_end", {}
            ),
            "formal_active_requires": ["probe", "materialize", "admission", "active"],
        },
        "checkpoint_contract": {
            "version": CHECKPOINT_VERSION,
            "exact_granularity": "workload/filter/query",
            "measurement_granularity": "complete q/repeat block",
            "atomic_replace": True,
            "strict_run_spec": True,
        },
    }


def run_benchmark(args: argparse.Namespace) -> int:
    if not args.execute:
        raise RuntimeError("refusing to open PostgreSQL without --execute")
    require_psycopg()
    import psycopg

    if args.d3_min_benefit_per_byte < 0:
        raise ValueError("--d3-min-benefit-per-byte must be nonnegative")
    if not 0.0 <= args.d3_page_min_skip_rate <= 1.0:
        raise ValueError("--d3-page-min-skip-rate must be in [0, 1]")

    filters = read_filters(args.filters_csv, set(args.filter_names) or None)
    validate_formal_dimensions(args, filters)
    expected_query_splits = {
        **{
            query_no: "calibration"
            for query_no in range(
                args.calibration_query_offset,
                args.calibration_query_offset + args.calibration_queries,
            )
        },
        **{
            query_no: "final"
            for query_no in range(
                args.final_query_offset,
                args.final_query_offset + args.final_queries,
            )
        },
    }
    query_cohort = exact_truth_contract.load_query_cohort(
        args.query_ids_csv,
        expected_query_splits,
        args.candidate_validity_predicate,
        source_manifest_path=args.query_cohort_manifest,
        expected_filters=filters,
    )
    calibration = {
        query_no: query_id
        for query_no, query_id in query_cohort.query_ids.items()
        if expected_query_splits[query_no] == "calibration"
    }
    final = {
        query_no: query_id
        for query_no, query_id in query_cohort.query_ids.items()
        if expected_query_splits[query_no] == "final"
    }
    validate_query_splits(calibration, final)
    workloads = list(WORKLOADS)
    preregistered_matrix = preregister_formal_matrix(
        workloads, filters, args.targets
    )
    conninfo = pg_config_from_env().conninfo
    checkpoint_path = args.checkpoint or args.out.with_suffix(".checkpoint")
    connections: dict[str, Any] = {}
    database: dict[str, Any] = {}
    as_of_by_workload: dict[str, int] = {}
    truth: dict[tuple[str, str, int], ExactTruth] = {}
    summaries: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    fingerprint_cur: Any | None = None
    guard_conn: Any | None = None
    guard_cur: Any | None = None
    formal_guard: dict[str, Any] | None = None
    fragment_conn: Any | None = None
    fragment_cur: Any | None = None
    new_checkpoint = False
    d3_settings = {
        "probe_requests": args.d3_probe_requests,
        "min_benefit_per_byte": args.d3_min_benefit_per_byte,
        "max_fragment_mb": args.d3_max_fragment_mb,
        "page_min_skip_rate": args.d3_page_min_skip_rate,
    }
    try:
        fragment_conn = psycopg.connect(conninfo, autocommit=True)
        fragment_cur = fragment_conn.cursor()
        persistent_fragment_reset = clear_fragment_store(
            fragment_cur, args.vector_table
        )
        guard_conn = psycopg.connect(conninfo, autocommit=True)
        guard_cur = guard_conn.cursor()
        formal_guard = exact_truth_contract.acquire_formal_data_guard(
            guard_cur, args.vector_table
        )
        probe_ids = exact_truth_contract.select_rls_probe_ids(
            guard_cur, args.principal
        )
        for mode in MODES:
            connections[mode] = psycopg.connect(conninfo, autocommit=True)
        session_contexts: dict[str, dict[str, str]] = {}
        security_proofs: dict[str, dict[str, Any]] = {}
        for mode, conn in connections.items():
            cur = conn.cursor()
            cur.execute(f'SET ROLE "{args.principal}"')
            cur.execute("SET hnsw.guidance_require_epoch = on")
            set_preferred_index(
                cur, mode_index(mode, args.source_index, args.clone_index)
            )
            session_contexts[mode] = loaded_session_context(cur)
            if session_contexts[mode]["current_user"] != args.principal:
                raise RuntimeError(f"loaded role mismatch for {mode}: {session_contexts[mode]}")
            security = exact_truth_contract.collect_rls_security_metadata(cur)
            security.update(
                exact_truth_contract.run_rls_visibility_probes(cur, probe_ids)
            )
            security["controlled_probe_ids"] = probe_ids
            security_proofs[mode] = validate_rls_security_proof(
                security, args.principal
            )
            cur.close()
        fingerprint_cur = connections["stock"].cursor()
        relations = [
            args.vector_table,
            args.source_index,
            args.clone_index,
            "public.amazon_review_facts",
            "public.amazon_product_dim",
            "public.amazon_principal_tenant_grants",
            "public.amazon_sql_native_buckets",
        ]
        database = database_fingerprint(fingerprint_cur, relations)
        database["d2_graph_proof"] = graph_clone_proof(
            guard_cur, args.source_index, args.clone_index
        )
        database["d2_index_names"] = [args.source_index, args.clone_index]
        database["principal"] = args.principal
        database["rls_security_proofs"] = security_proofs
        database["formal_data_guard_start"] = formal_guard
        database["query_candidate_universe_proof"] = (
            exact_truth_contract.verify_query_candidate_universe(
                guard_cur,
                args.vector_table,
                {**calibration, **final},
                args.candidate_validity_predicate,
            )
        )
        database["d3_persistent_fragment_reset"] = persistent_fragment_reset
        database["preferred_index_current_settings"] = {
            mode: context["preferred_index_current_setting"]
            for mode, context in session_contexts.items()
        }
        database["mode_indexes"] = {
            mode: mode_index(mode, args.source_index, args.clone_index)
            for mode in MODES
        }
        d3_startup_cur = connections["d1_d2_d3"].cursor()
        try:
            database["d3_startup_reset_evidence"] = reset_adaptive_state(
                d3_startup_cur, persistent_fragment_reset
            )
        finally:
            d3_startup_cur.close()
        fingerprint_cur.execute(
            "SELECT target_pct, as_of FROM public.amazon_sql_native_buckets WHERE principal_name = %s AND target_pct = ANY(%s::numeric[])",
            (args.principal, [item.bucket_pct for item in workloads]),
        )
        as_of = {str(float(row[0])): int(row[1]) for row in fingerprint_cur.fetchall()}
        for workload in workloads:
            key = str(float(workload.bucket_pct))
            if key not in as_of:
                raise RuntimeError(f"missing prepared as_of bucket for {workload.name}: {workload.bucket_pct}")
            as_of_by_workload[workload.name] = as_of[key]

        combined_queries = {**calibration, **final}
        query_splits = {
            **{query_no: "calibration" for query_no in calibration},
            **{query_no: "final" for query_no in final},
        }
        external_truth_provenance: dict[str, str] | None = None
        if not args.debug_compute_exact_truth:
            truth, external_truth_provenance = load_external_exact_truth(
                args.exact_truth_csv,
                args.exact_truth_manifest,
                args.fbin,
                args.filters_csv,
                args.query_ids_csv,
                workloads,
                filters,
                combined_queries,
                query_splits,
                as_of_by_workload,
                args.vector_table,
                args.principal,
                args.k,
                database["relations"],
                candidate_validity_predicate=args.candidate_validity_predicate,
                query_cohort_manifest=args.query_cohort_manifest,
            )
        run_spec = build_run_spec(
            args, filters, workloads, calibration, final, database, as_of_by_workload,
            external_truth_provenance,
            query_cohort.provenance,
        )
        if args.resume:
            if not checkpoint_path.is_dir():
                raise RuntimeError(f"resume checkpoint does not exist: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path)
            validate_checkpoint_run_spec(checkpoint, run_spec)
        else:
            if checkpoint_path.exists():
                raise RuntimeError(
                    f"checkpoint already exists: {checkpoint_path}; use --resume or move it aside"
                )
            checkpoint = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "run_spec": run_spec,
                "run_spec_sha256": canonical_sha256(run_spec),
                "loaded_sessions": [],
                "exact_truth": [],
                "exact_plans": [],
                "calibration_blocks": [],
                "final_blocks": [],
                "invalid_blocks": [],
            }
            new_checkpoint = True
        for field in ("loaded_sessions", "exact_truth", "exact_plans", "calibration_blocks", "final_blocks", "invalid_blocks"):
            if not isinstance(checkpoint.get(field), list):
                raise RuntimeError(f"checkpoint field is incomplete: {field}")
        if checkpoint["invalid_blocks"]:
            raise RuntimeError(
                "checkpoint contains a failed formal block; inspect invalid_blocks and restart from a clean checkpoint"
            )
        checkpoint["loaded_sessions"].append(
            {"resume": bool(args.resume), "connections": session_contexts}
        )
        if new_checkpoint:
            initialize_checkpoint(checkpoint_path, checkpoint)
        else:
            persist_checkpoint_meta(checkpoint_path, checkpoint)

        table_fingerprint_sha256 = canonical_sha256(database["relations"][args.vector_table])
        if args.debug_compute_exact_truth:
            validate_exact_plans(
                checkpoint["exact_plans"],
                workloads,
                filters,
                args.vector_table,
                args.candidate_validity_predicate,
            )
            truth = restore_exact_truth(
                checkpoint["exact_truth"],
                workloads,
                filters,
                combined_queries,
                query_splits,
                as_of_by_workload,
                args.vector_table,
                table_fingerprint_sha256,
                args.k,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
        elif checkpoint["exact_truth"] or checkpoint["exact_plans"]:
            raise RuntimeError("checkpoint exact-GT records are incompatible with external exact truth")
        exact_plan_pairs = {
            (plan["workload"], plan["filter_name"]) for plan in checkpoint["exact_plans"]
        }
        if args.debug_compute_exact_truth and any((key[0], key[1]) not in exact_plan_pairs for key in truth):
            raise RuntimeError("checkpoint exact GT is missing its successful non-HNSW EXPLAIN gate")
        if (checkpoint["calibration_blocks"] or checkpoint["final_blocks"]) and len(truth) != len(
            workloads
        ) * len(filters) * len(combined_queries):
            raise RuntimeError("checkpoint has measurement blocks before exact GT is complete")

        exact_plans = checkpoint["exact_plans"] if args.debug_compute_exact_truth else []

        def checkpoint_truth(
            workload: WorkloadSpec,
            spec: FilterSpec,
            query_no: int,
            query_id: int,
            sql_text: str,
            truth_entry: ExactTruth,
        ) -> None:
            record = exact_truth_record(
                workload,
                spec,
                query_no,
                query_id,
                query_splits[query_no],
                as_of_by_workload[workload.name],
                table_fingerprint_sha256,
                sql_text,
                truth_entry,
            )
            plan = next(
                plan
                for plan in exact_plans
                if plan["workload"] == workload.name and plan["filter_name"] == spec.name
            )
            persist_checkpoint_entry(
                checkpoint_path,
                "exact_plans",
                f"{workload.name}|{spec.name}",
                plan,
            )
            persist_checkpoint_entry(
                checkpoint_path,
                "exact_truth",
                f"{workload.name}|{spec.name}|q{query_no}",
                record,
            )
            checkpoint["exact_truth"].append(record)
            checkpoint["exact_plans"] = exact_plans
            print(
                json.dumps(
                    {
                        "progress": "exact_gt",
                        "completed": len(checkpoint["exact_truth"]),
                        "planned": len(workloads) * len(filters) * len(combined_queries),
                        "block": f"{workload.name}/{spec.name}/q{query_no}",
                    }
                ),
                flush=True,
            )

        if args.debug_compute_exact_truth:
            truth = run_exact_truth(
                fingerprint_cur,
                workloads,
                filters,
                combined_queries,
                as_of_by_workload,
                args.vector_table,
                args.source_index,
                args.k,
                exact_plans,
                existing=truth,
                on_truth=checkpoint_truth,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
            checkpoint["exact_plans"] = exact_plans
            for plan in exact_plans:
                persist_checkpoint_entry(
                    checkpoint_path,
                    "exact_plans",
                    f"{plan['workload']}|{plan['filter_name']}",
                    plan,
                )
            restore_exact_truth(
                checkpoint["exact_truth"],
                workloads,
                filters,
                combined_queries,
                query_splits,
                as_of_by_workload,
                args.vector_table,
                table_fingerprint_sha256,
                args.k,
                require_complete=True,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )

        workload_by_name = {item.name: item for item in workloads}
        filter_by_name = {item.name: item for item in filters}
        grids = {mode: build_config_grid(args, mode) for mode in MODES}
        config_by_label = {
            mode: {config.label: config for config in configs} for mode, configs in grids.items()
        }
        calibration_blocks: dict[str, dict[str, Any]] = {}
        for block in checkpoint["calibration_blocks"]:
            workload = workload_by_name.get(str(block.get("workload")))
            spec = filter_by_name.get(str(block.get("filter_name")))
            mode = str(block.get("mode"))
            config = config_by_label.get(mode, {}).get(str(block.get("config")))
            if workload is None or spec is None or config is None:
                raise RuntimeError("checkpoint calibration block is stale")
            block_id = calibration_block_id(workload.name, spec.name, mode, config)
            if block.get("block_id") != block_id or block_id in calibration_blocks:
                raise RuntimeError("checkpoint calibration block ID is unexpected/duplicate")
            validate_measurement_block(
                block,
                phase="calibration",
                workload=workload,
                spec=spec,
                query_ids=calibration,
                repeats=args.calibration_repeats,
                modes=(mode,),
                configs={mode: config},
                target_recall=None,
                truth=truth,
                table=args.vector_table,
                principal=args.principal,
                source_index=args.source_index,
                clone_index=args.clone_index,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
            calibration_blocks[block_id] = block

        def calibration_summaries(
            workload: WorkloadSpec, spec: FilterSpec, mode: str, configs: Sequence[Config]
        ) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            expected = expected_keys_for(
                [workload], [spec], calibration, args.calibration_repeats
            )
            for config in configs:
                block = calibration_blocks.get(
                    calibration_block_id(workload.name, spec.name, mode, config)
                )
                if block is None:
                    continue
                for target in args.targets:
                    summary = summarize_rows(
                        block["rows"],
                        expected_keys=expected,
                        target_recall=target,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed,
                    )
                    summary.update(
                        {
                            "phase": "calibration",
                            "mode": mode,
                            "config": config.label,
                            **asdict(config),
                        }
                    )
                    result.append(summary)
            return result

        for workload in workloads:
            for spec in filters:
                for mode in MODES:
                    labels = [config.label for config in grids[mode]]
                    present = [
                        calibration_block_id(workload.name, spec.name, mode, config)
                        in calibration_blocks
                        for config in grids[mode]
                    ]
                    seen_gap = False
                    for exists in present:
                        seen_gap = seen_gap or not exists
                        if exists and seen_gap:
                            raise RuntimeError(
                                "checkpoint calibration blocks are not an ef-ordered prefix"
                            )
                    executed_count = sum(present)
                    for _, group in config_groups(grids[mode]):
                        group_end = labels.index(group[-1].label) + 1
                        if group_end > executed_count:
                            break
                        prefix = grids[mode][:group_end]
                        outcome = calibration_outcome(
                            calibration_summaries(workload, spec, mode, prefix),
                            grids[mode],
                            [config.label for config in prefix],
                            args.targets,
                        )
                        if outcome["stopped"] and executed_count > group_end:
                            raise RuntimeError(
                                "checkpoint calibration continued after highest target was attained"
                            )

        outcomes: dict[tuple[str, str, str], dict[str, Any]] = {}
        for workload in workloads:
            for spec in filters:
                for mode in MODES:
                    executed_configs: list[Config] = []
                    for ef_search, group in config_groups(grids[mode]):
                        for config in group:
                            block_id = calibration_block_id(
                                workload.name, spec.name, mode, config
                            )
                            if block_id not in calibration_blocks:
                                block_fragment_reset = (
                                    clear_fragment_store(fragment_cur, args.vector_table)
                                    if mode == "d1_d2_d3"
                                    else persistent_fragment_reset
                                )
                                try:
                                    rows, plans = run_measurements(
                                        connections,
                                        {mode: config},
                                        [workload],
                                        [spec],
                                        calibration,
                                        truth,
                                        as_of_by_workload,
                                        args.vector_table,
                                        args.source_index,
                                        args.clone_index,
                                        args.principal,
                                        args.k,
                                        args.calibration_repeats,
                                        "calibration",
                                        None,
                                        args.schedule_seed,
                                        selected_modes=(mode,),
                                        d3_settings=d3_settings,
                                        fragment_store_reset=block_fragment_reset,
                                        candidate_validity_predicate=args.candidate_validity_predicate,
                                    )
                                except BaseException as exc:
                                    persist_checkpoint_entry(
                                        checkpoint_path,
                                        "invalid_blocks",
                                        block_id,
                                        {
                                            "block_id": block_id,
                                            "execution_error": f"{exc.__class__.__name__}: {exc}",
                                        },
                                    )
                                    raise
                                block = {
                                    "block_id": block_id,
                                    "phase": "calibration",
                                    "workload": workload.name,
                                    "filter_name": spec.name,
                                    "mode": mode,
                                    "config": config.label,
                                    "rows": rows,
                                    "plans": plans,
                                }
                                try:
                                    validate_measurement_block(
                                        block,
                                        phase="calibration",
                                        workload=workload,
                                        spec=spec,
                                        query_ids=calibration,
                                        repeats=args.calibration_repeats,
                                        modes=(mode,),
                                        configs={mode: config},
                                        target_recall=None,
                                        truth=truth,
                                        table=args.vector_table,
                                        principal=args.principal,
                                        source_index=args.source_index,
                                        clone_index=args.clone_index,
                                        candidate_validity_predicate=args.candidate_validity_predicate,
                                    )
                                except BaseException as exc:
                                    persist_checkpoint_entry(
                                        checkpoint_path,
                                        "invalid_blocks",
                                        block_id,
                                        {
                                            "block_id": block_id,
                                            "validation_error": f"{exc.__class__.__name__}: {exc}",
                                            "block": block,
                                        },
                                    )
                                    raise
                                persist_checkpoint_entry(
                                    checkpoint_path,
                                    "calibration_blocks",
                                    block_id,
                                    block,
                                )
                                checkpoint["calibration_blocks"].append(block)
                                calibration_blocks[block_id] = block
                                highest_summary = calibration_summaries(
                                    workload, spec, mode, [config]
                                )[-1]
                                print(
                                    json.dumps(
                                        {
                                            "progress": "calibration",
                                            "block": block_id,
                                            "ef_search": ef_search,
                                            "rows": len(rows),
                                            "errors": highest_summary["errors"],
                                            "highest_target_lcb": highest_summary["recall_lcb95"],
                                        }
                                    ),
                                    flush=True,
                                )
                            executed_configs.append(config)
                        pair_summaries = calibration_summaries(
                            workload, spec, mode, executed_configs
                        )
                        outcome = calibration_outcome(
                            pair_summaries,
                            grids[mode],
                            [config.label for config in executed_configs],
                            args.targets,
                        )
                        if outcome["stopped"]:
                            break
                    pair_summaries = calibration_summaries(
                        workload, spec, mode, executed_configs
                    )
                    summaries.extend(pair_summaries)
                    outcomes[(workload.name, spec.name, mode)] = calibration_outcome(
                        pair_summaries,
                        grids[mode],
                        [config.label for config in executed_configs],
                        args.targets,
                    )

        common_by_pair: dict[tuple[str, str], list[float]] = {}
        expected_final: dict[str, tuple[WorkloadSpec, FilterSpec, float, dict[str, Config]]] = {}
        for workload in workloads:
            for spec in filters:
                mode_outcomes = [
                    outcomes[(workload.name, spec.name, mode)] for mode in MODES
                ]
                common = common_attainable_targets(mode_outcomes, args.targets)
                common_by_pair[(workload.name, spec.name)] = common
                for target in common:
                    config_map = {
                        mode: config_by_label[mode][outcomes[(workload.name, spec.name, mode)]["selected"][target]["config"]]
                        for mode in MODES
                    }
                    expected_final[final_block_id(workload.name, spec.name, target)] = (
                        workload,
                        spec,
                        target,
                        config_map,
                    )

        final_blocks: dict[str, dict[str, Any]] = {}
        for block in checkpoint["final_blocks"]:
            block_id = str(block.get("block_id"))
            expected_block = expected_final.get(block_id)
            if expected_block is None or block_id in final_blocks:
                raise RuntimeError("checkpoint final block is stale/unexpected/duplicate")
            workload, spec, target, config_map = expected_block
            if block.get("configs") != {
                mode: config.label for mode, config in config_map.items()
            }:
                raise RuntimeError("checkpoint final block selected configs changed")
            validate_measurement_block(
                block,
                phase="final",
                workload=workload,
                spec=spec,
                query_ids=final,
                repeats=args.final_repeats,
                modes=MODES,
                configs=config_map,
                target_recall=target,
                truth=truth,
                table=args.vector_table,
                principal=args.principal,
                source_index=args.source_index,
                clone_index=args.clone_index,
                candidate_validity_predicate=args.candidate_validity_predicate,
            )
            final_blocks[block_id] = block

        for block_id, (workload, spec, target, config_map) in expected_final.items():
            if block_id not in final_blocks:
                block_fragment_reset = clear_fragment_store(
                    fragment_cur, args.vector_table
                )
                try:
                    rows, plans = run_measurements(
                        connections,
                        config_map,
                        [workload],
                        [spec],
                        final,
                        truth,
                        as_of_by_workload,
                        args.vector_table,
                        args.source_index,
                        args.clone_index,
                        args.principal,
                        args.k,
                        args.final_repeats,
                        "final",
                        target,
                        args.schedule_seed + int(target * 1000),
                        d3_settings=d3_settings,
                        fragment_store_reset=block_fragment_reset,
                        candidate_validity_predicate=args.candidate_validity_predicate,
                    )
                except BaseException as exc:
                    persist_checkpoint_entry(
                        checkpoint_path,
                        "invalid_blocks",
                        block_id,
                        {
                            "block_id": block_id,
                            "execution_error": f"{exc.__class__.__name__}: {exc}",
                        },
                    )
                    raise
                block = {
                    "block_id": block_id,
                    "phase": "final",
                    "workload": workload.name,
                    "filter_name": spec.name,
                    "target_recall": target,
                    "configs": {mode: config.label for mode, config in config_map.items()},
                    "rows": rows,
                    "plans": plans,
                }
                try:
                    validate_measurement_block(
                        block,
                        phase="final",
                        workload=workload,
                        spec=spec,
                        query_ids=final,
                        repeats=args.final_repeats,
                        modes=MODES,
                        configs=config_map,
                        target_recall=target,
                        truth=truth,
                        table=args.vector_table,
                        principal=args.principal,
                        source_index=args.source_index,
                        clone_index=args.clone_index,
                        candidate_validity_predicate=args.candidate_validity_predicate,
                    )
                except BaseException as exc:
                    persist_checkpoint_entry(
                        checkpoint_path,
                        "invalid_blocks",
                        block_id,
                        {
                            "block_id": block_id,
                            "validation_error": f"{exc.__class__.__name__}: {exc}",
                            "block": block,
                        },
                    )
                    raise
                persist_checkpoint_entry(
                    checkpoint_path,
                    "final_blocks",
                    block_id,
                    block,
                )
                checkpoint["final_blocks"].append(block)
                final_blocks[block_id] = block
                print(
                    json.dumps(
                        {
                            "progress": "final",
                            "block": block_id,
                            "rows": len(rows),
                            "execution_order": "query/repeat interleaved across stock and SQLens",
                        }
                    ),
                    flush=True,
                )

        for block_id, (workload, spec, target, config_map) in expected_final.items():
            rows = final_blocks[block_id]["rows"]
            expected = expected_keys_for([workload], [spec], final, args.final_repeats)
            for mode in MODES:
                mode_rows = [row for row in rows if row["mode"] == mode]
                summary = summarize_rows(
                    mode_rows,
                    expected_keys=expected,
                    target_recall=target,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + int(target * 1000),
                )
                summary.update(
                    {
                        "phase": "final",
                        "mode": mode,
                        "config": config_map[mode].label,
                        **asdict(config_map[mode]),
                    }
                )
                summaries.append(summary)
            for method_mode in SQLENS_MODES:
                paired = paired_summary(
                    [row for row in rows if row["mode"] == "stock"],
                    [row for row in rows if row["mode"] == method_mode],
                    expected_keys=expected,
                    target_recall=target,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + int(target * 1000),
                    method_mode=method_mode,
                )
                summaries.append(
                    {
                        "phase": "final",
                        "mode": f"paired_{method_mode}",
                        "workload": workload.name,
                        "filter_name": spec.name,
                        "target_recall": target,
                        "config": (
                            f"stock={config_map['stock'].label};"
                            f"{method_mode}={config_map[method_mode].label}"
                        ),
                        "status": paired["status"],
                        "paired_queries": paired["paired_queries"],
                        "speedup_vs_stock": paired["speedup_vs_stock"],
                        "speedup_lcb95": paired["speedup_lcb95"],
                        "speedup_ci95_low": paired["speedup_ci95_low"],
                        "speedup_ci95_high": paired["speedup_ci95_high"],
                        "paired_latency_saving_mean_ms": paired[
                            "paired_latency_saving_mean_ms"
                        ],
                        "paired_latency_saving_ci95_low_ms": paired[
                            "paired_latency_saving_ci95_low_ms"
                        ],
                        "paired_latency_saving_ci95_high_ms": paired[
                            "paired_latency_saving_ci95_high_ms"
                        ],
                        "stock_recall_lcb95": paired["stock"]["recall_lcb95"],
                        "method_recall_lcb95": paired[method_mode]["recall_lcb95"],
                        "query_latency_definition": TIMING_DEFINITION,
                    }
                )

        pair_execution: list[dict[str, Any]] = []
        for workload in workloads:
            for spec in filters:
                for mode in MODES:
                    outcome = outcomes[(workload.name, spec.name, mode)]
                    pair_execution.append(
                        {
                            "workload": workload.name,
                            "filter_name": spec.name,
                            "mode": mode,
                            **{key: value for key, value in outcome.items() if key != "selected"},
                            "selected_configs": {
                                str(target): choice["config"] if choice else None
                                for target, choice in outcome["selected"].items()
                            },
                        }
                    )
        completed_final = {
            (workload.name, spec.name, float(target))
            for workload, spec, target, _ in expected_final.values()
        }
        formal_matrix = finalize_formal_matrix(
            preregistered_matrix, outcomes, completed_final
        )
        database["d2_graph_proof_end"] = graph_clone_proof(
            guard_cur, args.source_index, args.clone_index
        )
        database["d2_index_fingerprints_end"] = {
            index: relation_fingerprint(guard_cur, index)
            for index in (args.source_index, args.clone_index)
        }
        if database["d2_graph_proof_end"] != database["d2_graph_proof"]:
            raise RuntimeError("D2 graph proof changed during the formal run")
        database["formal_data_version_proof"] = (
            exact_truth_contract.release_formal_data_guard(
                guard_cur, args.vector_table, formal_guard
            )
        )
        formal_guard = None
        database["d3_fragment_store_end"] = audit_fragment_store(
            fragment_cur, args.vector_table
        )
        database["loaded_sessions"] = checkpoint["loaded_sessions"]
        all_rows = [
            row
            for block in checkpoint["calibration_blocks"] + checkpoint["final_blocks"]
            for row in block["rows"]
        ]
        all_plans = list(exact_plans) + [
            plan
            for block in checkpoint["calibration_blocks"] + checkpoint["final_blocks"]
            for plan in block["plans"]
        ]
        manifest = _manifest(
            args,
            filters,
            calibration,
            final,
            database,
            as_of_by_workload,
            query_cohort.provenance,
        )
        artifact_errors = artifact_validation_errors(
            len(expected_final), summaries, all_rows, all_plans, formal_matrix
        )
        artifact_errors.extend(database_contract_errors(database))
        manifest["artifact_valid"] = not artifact_errors
        manifest["artifact_errors"] = artifact_errors
        manifest.update(
            {
                "external_exact_truth": external_truth_provenance,
                "exact_truth_mode": "debug_in_database" if args.debug_compute_exact_truth else "external_precomputed",
                "selected_calibration_and_final_summaries": summaries,
                "d3_measured_transition_evidence": grouped_adaptive_transition_evidence(
                    all_rows
                ),
                "d3_block_fragment_reset_proofs": {
                    canonical_sha256(row["persistent_fragment_reset_proof"]): row[
                        "persistent_fragment_reset_proof"
                    ]
                    for row in all_rows
                    if row.get("mode") == "d1_d2_d3"
                    and isinstance(row.get("persistent_fragment_reset_proof"), dict)
                },
                "truth_pairs": len(truth),
                "calibration_execution": {
                    "planned_blocks": sum(item["planned_blocks"] for item in pair_execution),
                    "executed_blocks": len(checkpoint["calibration_blocks"]),
                    "stopped_pairs": sum(bool(item["stopped"]) for item in pair_execution),
                    "grid_exhausted_pairs": sum(
                        bool(item["grid_exhausted"]) for item in pair_execution
                    ),
                    "pairs": pair_execution,
                },
                "final_execution": {
                    "planned_blocks": len(expected_final),
                    "executed_blocks": len(checkpoint["final_blocks"]),
                    "query_repeat_interleaved": True,
                    "paired_ci": True,
                },
                "common_attainable": [
                    {
                        "workload": workload,
                        "filter_name": filter_name,
                        "targets": targets,
                    }
                    for (workload, filter_name), targets in common_by_pair.items()
                ],
                "pre_registered_formal_matrix": formal_matrix,
            }
        )
        manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
        plans_path = args.plans or args.out.with_suffix(".plans.json")
        status = publish_benchmark_artifacts(
            args, all_rows, summaries, all_plans, manifest, checkpoint_path
        )
        print(
            json.dumps(
                {
                    "rows": len(all_rows),
                    "summaries": len(summaries),
                    "manifest": str(manifest_path),
                    "plans": str(plans_path),
                    "checkpoint_removed": status == 0,
                    "artifact_valid": manifest["artifact_valid"],
                },
                indent=2,
            )
        )
        return status
    finally:
        if formal_guard is not None and guard_cur is not None:
            try:
                guard_cur.execute("ROLLBACK")
            except Exception:
                pass
        if fingerprint_cur is not None:
            fingerprint_cur.close()
        for conn in connections.values():
            conn.close()
        if guard_cur is not None:
            guard_cur.close()
        if guard_conn is not None:
            guard_conn.close()
        if fragment_cur is not None:
            fragment_cur.close()
        if fragment_conn is not None:
            fragment_conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.dry_run or not args.execute:
        print_dry_run(args)
        return 0
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
