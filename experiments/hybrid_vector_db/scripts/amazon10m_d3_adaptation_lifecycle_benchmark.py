#!/usr/bin/env python3
"""Formal Amazon-10M online D3 adaptation lifecycle benchmark.

This runner deliberately distinguishes request-driven D3 admission from an
eagerly materialized control.  It runs the modes on three independent,
persistent PostgreSQL sessions in deterministic paired windows.  It never
invents predicates: every request uses one of the fourteen observed Amazon
predicates in amazon10m_selectivity14_filters.csv and one of the fixed q200
exact-truth query IDs.  Cross-process resume fails closed because the D3/cache
lifecycle is backend-local.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_TRUTH = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_valid_embeddings_formal.csv"
DEFAULT_TRUTH_MANIFEST = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_valid_embeddings_formal_manifest.json"
DEFAULT_TABLE = "amazon_grocery_reviews_10m_pgvector"
DEFAULT_INDEX = "amazon10m_embedding_valid_hnsw_source_idx"
DEFAULT_CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"
MODES = ("stock", "adaptive", "eager_prebuilt")
FORMAL_REQUESTS = 10_000
FORMAL_WINDOW = 100
FORMAL_Q200 = 200
CHECKPOINT_SCHEMA_VERSION = 3
PAIRING_SCHEDULE = "deterministic_request_interleaved_round_robin"
FRAGMENT_STORE_RELATION = "public.pgvector_hnsw_fragment_store"
FRAGMENT_EPOCH_RELATION = "public.pgvector_hnsw_fragment_epoch"
FORMAL_WORKLOAD_MANIFEST_NAME = "amazon10m_d3_online_adaptation_10000_request_trace_q200_reused_vectors"


@dataclass(frozen=True)
class FilterSpec:
    name: str
    predicate: str
    atoms: tuple[str, ...]
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class TruthEntry:
    filter_name: str
    query_no: int
    query_id: int
    ids: tuple[int, ...]
    kth_distance_sq: float
    tie_tolerance: float


@dataclass(frozen=True)
class Request:
    request_no: int
    phase: str
    window: int
    filter_name: str
    query_no: int
    query_id: int
    reuse_distance: int | None


class Session(Protocol):
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None: ...

    def one(self) -> Any: ...

    def row(self) -> Any: ...

    def all(self) -> Sequence[Any]: ...


class CursorSession:
    """Small adapter so the lifecycle code is both fakeable and psycopg-neutral."""

    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self.cursor.execute(sql, params)

    def one(self) -> Any:
        return self.cursor.fetchone()[0]

    def row(self) -> Any:
        return self.cursor.fetchone()

    def all(self) -> Sequence[Any]:
        return self.cursor.fetchall()


@dataclass
class ModeBackend:
    """One long-lived PostgreSQL backend, dedicated to one experimental mode."""

    mode: str
    connection: Any
    session: Session
    backend_pid: int
    database: dict[str, Any]


class BenchmarkContractError(RuntimeError):
    """A run or checkpoint no longer satisfies the formal experiment contract."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_resume_contract() -> dict[str, Any]:
    """Describe the deliberately conservative cross-process recovery policy."""
    return {
        "checkpoint_unit": "complete_cross_mode_paired_window",
        "cross_process_resume": "forbidden",
        "policy": "fail_closed",
        "reason": "D3 lifecycle, metadata cache, and persistent fragment-store state are backend-local and have no portable restore API",
        "cache_lifecycle_fingerprints": "audit_only_not_replayable",
        "timed_replay": "not_implemented",
    }


def reject_cross_process_resume(resume_requested: bool) -> None:
    if resume_requested:
        raise BenchmarkContractError(
            "cross-process --resume is disabled: a checkpoint cannot restore backend-local D3/cache state; "
            "start a fresh run after preserving or removing the checkpoint"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def parse_atoms(value: str) -> tuple[str, ...]:
    atoms = tuple(part.strip() for part in value.split("||") if part.strip())
    if not atoms or any(not atom.startswith("sql:") for atom in atoms):
        raise BenchmarkContractError("filter atoms must be nonempty sql: atoms")
    return atoms


def load_filters(path: Path) -> list[FilterSpec]:
    rows = read_csv(path)
    required = {"filter_name", "predicate", "atoms", "count", "actual_pct", "source"}
    if len(rows) != 14 or not rows or not required <= set(rows[0]):
        raise BenchmarkContractError("formal benchmark requires the real fourteen-filter Amazon CSV")
    result = [
        FilterSpec(
            name=row["filter_name"],
            predicate=row["predicate"],
            atoms=parse_atoms(row["atoms"]),
            expected_rows=int(row["count"]),
            actual_pct=float(row["actual_pct"]),
        )
        for row in rows
    ]
    if len({item.name for item in result}) != 14 or any("%" in item.predicate for item in result):
        raise BenchmarkContractError("filters must be the fourteen distinct real predicates, without modulo synthesis")
    return result


def parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part) for part in value.split(",") if part.strip())
    if len(ids) != 10 or len(set(ids)) != 10:
        raise BenchmarkContractError("exact truth must provide ten distinct IDs")
    return ids


def load_truth(path: Path, filters: Sequence[FilterSpec]) -> dict[tuple[str, int], TruthEntry]:
    rows = read_csv(path)
    required = {"filter_name", "query_no", "query_id", "exact_filtered_topk_ids", "kth_distance_sq", "tie_tolerance"}
    if not rows or not required <= set(rows[0]):
        raise BenchmarkContractError("exact truth is missing the fixed q200 tie-aware schema")
    wanted = {item.name for item in filters}
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_ids: dict[int, int] = {}
    for row in rows:
        if row.get("method") not in (None, "", "pre_filter_exact") or row["filter_name"] not in wanted:
            continue
        query_no = int(row["query_no"])
        if not 0 <= query_no < FORMAL_Q200:
            continue
        key = (row["filter_name"], query_no)
        if key in truth:
            raise BenchmarkContractError(f"duplicate exact truth pair: {key}")
        query_id = int(row["query_id"])
        old = query_ids.setdefault(query_no, query_id)
        if old != query_id:
            raise BenchmarkContractError(f"query_no={query_no} maps to multiple IDs")
        if str(row.get("self_excluded", "true")).lower() != "true":
            raise BenchmarkContractError("exact truth must exclude each query row")
        truth[key] = TruthEntry(row["filter_name"], query_no, query_id, parse_ids(row["exact_filtered_topk_ids"]),
                                float(row["kth_distance_sq"]), float(row["tie_tolerance"]))
    expected = {(item.name, query_no) for item in filters for query_no in range(FORMAL_Q200)}
    missing = expected - set(truth)
    if missing or set(query_ids) != set(range(FORMAL_Q200)) or len(set(query_ids.values())) != FORMAL_Q200:
        raise BenchmarkContractError(f"fixed q200 truth grid is incomplete; missing={len(missing)}")
    return truth


def _weighted_pick(rng: random.Random, names: Sequence[str], weights: Sequence[float]) -> str:
    return names[rng.choices(range(len(names)), weights=weights, k=1)[0]]


def build_trace(filters: Sequence[FilterSpec], truth: Mapping[tuple[str, int], TruthEntry], *,
                requests: int = FORMAL_REQUESTS, window_size: int = FORMAL_WINDOW, seed: int = 20260718,
                hot_reuse_probability: float = 0.78) -> list[Request]:
    """Create a deterministic hot/cold Zipf trace with a disjoint-hot phase shift."""
    if requests <= 0 or window_size <= 0 or requests % window_size:
        raise ValueError("requests must be positive and divisible by window_size")
    if len(filters) != 14:
        raise BenchmarkContractError("trace needs exactly fourteen real filters")
    filter_names = [item.name for item in filters]
    if any((name, q) not in truth for name in filter_names for q in range(FORMAL_Q200)):
        raise BenchmarkContractError("trace cannot use query IDs outside the fixed q200 truth grid")
    rng = random.Random(seed)
    ranked = filter_names[:]
    rng.shuffle(ranked)
    phase_hot = (set(ranked[:4]), set(ranked[4:8]))
    previous: dict[str, int] = {}
    trace: list[Request] = []
    half = requests // 2
    for request_no in range(requests):
        phase_index = 0 if request_no < half else 1
        phase = "steady_hot" if phase_index == 0 else "phase_shift_hot"
        hot = phase_hot[phase_index]
        cold = [name for name in ranked if name not in hot]
        recent = trace[-1].filter_name if trace else None
        if recent in hot and rng.random() < hot_reuse_probability:
            filter_name = recent
        elif rng.random() < 0.88:
            ordered_hot = [name for name in ranked if name in hot]
            filter_name = _weighted_pick(rng, ordered_hot, [1.0 / (rank + 1) for rank in range(len(ordered_hot))])
        else:
            filter_name = _weighted_pick(rng, cold, [1.0 / (rank + 1) for rank in range(len(cold))])
        query_no = rng.randrange(FORMAL_Q200)
        query_id = truth[(filter_name, query_no)].query_id
        old = previous.get(filter_name)
        trace.append(Request(request_no, phase, request_no // window_size, filter_name, query_no, query_id,
                             None if old is None else request_no - old))
        previous[filter_name] = request_no
    return trace


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples))
    return percentile(means, 0.025), percentile(means, 0.975)


def cache_is_empty(profile: Mapping[str, Any]) -> bool:
    return all(int(profile.get(key, 0) or 0) == 0 for key in (
        "entries", "resident_entries", "resident_bytes", "composed_guide_entries",
        "composed_exact_entries", "adaptive_cache_entries", "adaptive_bytes",
    ))


def _counter(profile: Mapping[str, Any], key: str) -> int:
    return int(profile.get(key, 0) or 0)


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int:
    return max(0, _counter(after, key) - _counter(before, key))


def audit_fragment_store(session: Session, table: str) -> dict[str, Any]:
    """Audit only fragments bound to ``table`` and prove their heap version."""
    session.execute(
        f"SELECT to_regclass(%s), %s::regclass::oid::bigint, "
        f"pg_relation_filenode(%s::regclass)::bigint, "
        f"coalesce((SELECT epoch FROM {FRAGMENT_EPOCH_RELATION} "
        "WHERE heap_oid = %s::regclass::oid), 0)::bigint, "
        f"EXISTS (SELECT 1 FROM {FRAGMENT_EPOCH_RELATION} "
        "WHERE heap_oid = %s::regclass::oid)",
        (FRAGMENT_STORE_RELATION, table, table, table, table),
    )
    store_name, heap_oid, relfilenode, epoch, epoch_present = session.row()
    store_exists = store_name is not None
    records: list[dict[str, Any]] = []
    if store_exists:
        session.execute(
            f"SELECT row_to_json(store_row)::text FROM {FRAGMENT_STORE_RELATION} AS store_row "
            "WHERE store_row.heap_oid = %s::regclass::oid ORDER BY row_to_json(store_row)::text",
            (table,),
        )
        records = [json.loads(str(row[0])) for row in session.all()]
    epoch_value = int(epoch or 0)
    relfilenode_value = int(relfilenode)
    epoch_matches = sum(int(record.get("build_epoch", -1)) == epoch_value for record in records)
    relfilenode_matches = sum(int(record.get("relfilenode", -1)) == relfilenode_value for record in records)
    proof = {
        "valid": all(
            int(record.get("heap_oid", -1)) == int(heap_oid)
            and int(record.get("build_epoch", -1)) == epoch_value
            and int(record.get("relfilenode", -1)) == relfilenode_value
            for record in records
        ) and bool(epoch_present),
        "heap_oid": int(heap_oid),
        "epoch": epoch_value,
        "epoch_present": bool(epoch_present),
        "relfilenode": relfilenode_value,
        "rows_checked": len(records),
        "rows_epoch_match": epoch_matches,
        "rows_relfilenode_match": relfilenode_matches,
    }
    return {
        "exists": store_exists,
        "count": len(records),
        "content_sha256": canonical_sha256(records),
        "heap_oid": int(heap_oid),
        "epoch": epoch_value,
        "relfilenode": relfilenode_value,
        "epoch_proof": proof,
    }


def validate_fragment_store_reset(before: Mapping[str, Any], deleted_count: int,
                                  after: Mapping[str, Any]) -> dict[str, Any]:
    before_count = int(before.get("count", -1))
    after_count = int(after.get("count", -1))
    if before_count < 0 or after_count < 0 or int(deleted_count) != before_count or after_count != 0:
        raise BenchmarkContractError(
            "persistent fragment store reset failed: "
            f"before={before_count} deleted={deleted_count} after={after_count}"
        )
    if int(before.get("heap_oid", -1)) != int(after.get("heap_oid", -2)):
        raise BenchmarkContractError("persistent fragment store reset changed target heap identity")
    epoch_proof = dict(after.get("epoch_proof") or {})
    if epoch_proof.get("valid") is not True or int(epoch_proof.get("rows_checked", -1)) != 0:
        raise BenchmarkContractError("persistent fragment store reset has incomplete epoch proof")
    return {
        "valid": True,
        "before": dict(before),
        "deleted": int(deleted_count),
        "after": dict(after),
        "heap_oid": int(after["heap_oid"]),
        "epoch_proof": epoch_proof,
        "prebuilt_fragments": after_count,
    }


def clear_fragment_store(session: Session, table: str) -> dict[str, Any]:
    """Clear persistent fragments for one heap, retaining an epoch proof."""
    before = audit_fragment_store(session, table)
    if before["exists"]:
        session.execute(
            f"DELETE FROM {FRAGMENT_STORE_RELATION} "
            "WHERE heap_oid = %s::regclass::oid RETURNING heap_oid",
            (table,),
        )
        deleted = len(session.all())
    else:
        deleted = 0
    after = audit_fragment_store(session, table)
    return validate_fragment_store_reset(before, deleted, after)


def lifecycle_classification(before: Mapping[str, Any], after: Mapping[str, Any], guidance: Mapping[str, Any], *,
                             admitted: bool, reason: str) -> dict[str, Any]:
    builds = int(guidance.get("fragment_builds", 0) or 0)
    store_hits = int(guidance.get("fragment_store_hits", 0) or 0)
    cache_hits = int(guidance.get("fragment_cache_hits", 0) or 0)
    evicted = int(after.get("evictions", 0) or 0) > int(before.get("evictions", 0) or 0)
    created = admitted and (builds > 0 or int(after.get("entries", 0) or 0) > int(before.get("entries", 0) or 0))
    reused = admitted and not created and (store_hits > 0 or cache_hits > 0 or bool(guidance.get("active", False)))
    return {"fragment_created": created, "fragment_reused": reused, "fragment_evicted": evicted,
            "admission_reason": reason, "fragment_builds": builds, "fragment_store_hits": store_hits,
            "fragment_cache_hits": cache_hits}


def recall_at_10(returned: Iterable[int], truth: TruthEntry) -> float:
    returned_set = set(int(value) for value in returned)
    return len(returned_set & set(truth.ids)) / len(truth.ids)


def summary_for_window(rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    e2e = [float(row["e2e_ms"]) for row in ok]
    query = [float(row["query_ms"]) for row in ok]
    recalls = [float(row["recall_at_10"]) for row in ok]
    low, high = bootstrap_ci(e2e, samples=bootstrap_samples, seed=bootstrap_seed)
    checks = sum(float(row.get("guidance_checks", 0) or 0) for row in ok)
    skips = sum(float(row.get("guidance_skips", 0) or 0) for row in ok)
    hits = sum(1 for row in ok if row.get("fragment_reused"))
    lifecycle_fields = {
        "probe": "probe_observed",
        "materialize": "materialization_observed",
        "reuse": "reuse_observed",
        "refine": "refine_observed",
        "evict": "evict_observed",
    }
    return {
        "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
        "e2e_mean_ms": statistics.fmean(e2e) if e2e else 0.0, "e2e_p50_ms": percentile(e2e, .50),
        "e2e_p95_ms": percentile(e2e, .95), "e2e_p99_ms": percentile(e2e, .99),
        "e2e_mean_ci95_low_ms": low, "e2e_mean_ci95_high_ms": high,
        "query_mean_ms": statistics.fmean(query) if query else 0.0, "query_p50_ms": percentile(query, .50),
        "query_p95_ms": percentile(query, .95), "query_p99_ms": percentile(query, .99),
        "recall_mean": statistics.fmean(recalls) if recalls else 0.0,
        "cache_hit_rate": hits / len(ok) if ok else 0.0,
        "memory_bytes_end": int(ok[-1].get("cache_resident_bytes_after", 0) or 0) if ok else 0,
        "guidance_skip_rate": skips / checks if checks else 0.0,
        "lifecycle_event_counts": {
            event: sum(1 for row in ok if row.get(field)) for event, field in lifecycle_fields.items()
        },
        "probe_count": sum(1 for row in ok if row.get("probe_observed")),
        "materialization_count": sum(1 for row in ok if row.get("materialization_observed")),
        "reuse_count": sum(1 for row in ok if row.get("reuse_observed")),
        "refine_count": sum(1 for row in ok if row.get("refine_observed")),
        "evict_count": sum(1 for row in ok if row.get("evict_observed")),
        "fragment_store_hit_delta": sum(_counter(row, "fragment_store_hit_delta") for row in ok),
        "hidden_prebuilt_reuse_count": sum(1 for row in ok if row.get("hidden_prebuilt_fragment_reused")),
        "lifecycle_paths": dict(statistics.Counter(str(row.get("lifecycle_path", "unknown")) for row in ok)),
    }


def break_even_request(rows: Sequence[Mapping[str, Any]], stock_by_request: Mapping[int, Mapping[str, Any]]) -> int | None:
    cumulative_by_request: list[tuple[int, float]] = []
    cumulative = 0.0
    for row in sorted(rows, key=lambda item: int(item["request_no"])):
        stock = stock_by_request.get(int(row["request_no"]))
        if not stock or row.get("error") or stock.get("error"):
            continue
        cumulative += float(stock["e2e_ms"]) - float(row["e2e_ms"])
        cumulative_by_request.append((int(row["request_no"]), cumulative))
    # A transient crossing caused by request-level noise is not amortization.
    # Report the first point after which cumulative savings stay nonnegative.
    suffix_minimum = math.inf
    stable_request: int | None = None
    for request_no, value in reversed(cumulative_by_request):
        suffix_minimum = min(suffix_minimum, value)
        if value >= 0.0 and suffix_minimum >= 0.0:
            stable_request = request_no
    if stable_request is not None:
        return stable_request
    return None


def validate_artifact(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], trace: Sequence[Request], *,
                      recall_delta: float, provenance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {request.request_no for request in trace}
    stock = {int(row["request_no"]): row for row in rows_by_mode.get("stock", [])}
    for mode in MODES:
        rows = list(rows_by_mode.get(mode, []))
        observed = {int(row.get("request_no", -1)) for row in rows}
        if observed != expected or len(rows) != len(trace):
            errors.append(f"missing_or_duplicate_windows:{mode}")
        if any(row.get("error") for row in rows):
            errors.append(f"request_errors:{mode}")
        if any(str(row.get("database_build_id", "")) != str(provenance.get("database_build_id", ""))
               or str(row.get("profile_build_id", "")) != str(provenance.get("database_build_id", "")) for row in rows):
            errors.append(f"profile_build_mismatch:{mode}")
        if mode != "stock":
            planner_failed = False
            for row in rows:
                if row.get("planner_proof_required") and not row.get("planner_proof_verified"):
                    planner_failed = True
                stock_row = stock.get(int(row.get("request_no", -1)))
                if stock_row and float(row.get("recall_at_10", 0.0)) + recall_delta < float(stock_row.get("recall_at_10", 0.0)):
                    errors.append(f"recall_regression:{mode}")
                    break
                if int(row.get("fragment_store_hit_delta", 0) or 0) < 0:
                    errors.append(f"negative_fragment_store_hit_delta:{mode}")
                    break
            if planner_failed:
                errors.append(f"planner_proof_failure:{mode}")
    adaptive_rows = rows_by_mode.get("adaptive", [])
    if adaptive_rows and not bool(adaptive_rows[0].get("adaptive_cache_started_empty")):
        errors.append("preexisting_adaptive_cache")
    if adaptive_rows:
        reset_proof = adaptive_rows[0].get("persistent_fragment_reset_proof")
        if not isinstance(reset_proof, Mapping) or reset_proof.get("valid") is not True:
            errors.append("missing_adaptive_fragment_store_reset_proof")
        elif int(reset_proof.get("prebuilt_fragments", -1)) != 0:
            errors.append("prebuilt_adaptive_fragments")
        if any(row.get("hidden_prebuilt_fragment_reused") for row in adaptive_rows):
            errors.append("hidden_prebuilt_fragment_reuse:adaptive")
        if any(not row.get("online_arm") for row in adaptive_rows):
            errors.append("adaptive_arm_contract_missing")
    eager_rows = rows_by_mode.get("eager_prebuilt", [])
    if eager_rows and any(not row.get("explicit_eager_control") for row in eager_rows):
        errors.append("eager_control_contract_missing")
    return errors


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_path(out: Path) -> Path:
    return out.with_name(out.stem + "_checkpoint.json")


def paired_request_mode_order(request_no: int) -> tuple[str, ...]:
    """Rotate first position for every request so no mode owns a fixed time slot."""
    if request_no < 0:
        raise ValueError("request_no must be nonnegative")
    offset = request_no % len(MODES)
    return MODES[offset:] + MODES[:offset]


def validate_independent_mode_sessions(backends: Mapping[str, ModeBackend]) -> None:
    if set(backends) != set(MODES):
        raise BenchmarkContractError("paired execution requires exactly stock, adaptive, and eager backends")
    if len({id(backends[mode].session) for mode in MODES}) != len(MODES):
        raise BenchmarkContractError("each mode requires an independent persistent session/cache")
    pids = [int(backends[mode].backend_pid) for mode in MODES]
    if len(set(pids)) != len(MODES):
        raise BenchmarkContractError("each mode requires a distinct PostgreSQL backend PID")


def _completed_paired_windows(value: Any) -> list[int]:
    if not isinstance(value, list) or any(isinstance(window, bool) for window in value):
        raise BenchmarkContractError("checkpoint completed-paired-window schema is invalid")
    try:
        completed = [int(window) for window in value]
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("checkpoint completed-paired-window schema is invalid") from exc
    if completed != list(range(len(completed))):
        raise BenchmarkContractError("checkpoint paired windows are not a complete prefix")
    return completed


def validate_checkpoint_rows(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], completed_windows: Sequence[int],
                             window_size: int) -> None:
    if window_size <= 0:
        raise BenchmarkContractError("checkpoint window size is invalid")
    completed = list(completed_windows)
    if completed != list(range(len(completed))):
        raise BenchmarkContractError("checkpoint paired windows are not a complete prefix")
    for mode in MODES:
        rows = list(rows_by_mode.get(mode, []))
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        try:
            for row in rows:
                grouped.setdefault(int(row["window"]), []).append(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError(f"checkpoint rows are invalid for {mode}") from exc
        if set(grouped) != set(completed):
            raise BenchmarkContractError(f"checkpoint is not a complete paired window set: {mode}")
        for window in completed:
            block = grouped[window]
            try:
                request_numbers = {int(row["request_no"]) for row in block}
            except (KeyError, TypeError, ValueError) as exc:
                raise BenchmarkContractError(f"checkpoint rows are invalid for {mode}/{window}") from exc
            if len(block) != window_size or len(request_numbers) != window_size:
                raise BenchmarkContractError(f"checkpoint has partial paired window: {mode}/{window}")


def paired_window_fingerprints(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
                               completed_windows: Sequence[int]) -> dict[str, str]:
    return {
        str(window): canonical_sha256({
            mode: sorted((row for row in rows_by_mode.get(mode, []) if int(row["window"]) == window),
                         key=lambda row: int(row["request_no"]))
            for mode in MODES
        })
        for window in completed_windows
    }


def backend_lifecycle_fingerprints(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for mode in MODES:
        rows = sorted(rows_by_mode.get(mode, []), key=lambda row: int(row["request_no"]))
        last = rows[-1] if rows else {}
        fingerprints[mode] = canonical_sha256({
            "request_no": last.get("request_no"),
            "cache_profile_after": last.get("cache_profile_after"),
            "guidance_profile": last.get("guidance_profile"),
            "adaptive_state": last.get("adaptive_state"),
            "fragment_created": last.get("fragment_created"),
            "fragment_reused": last.get("fragment_reused"),
            "fragment_store_hit_delta": last.get("fragment_store_hit_delta"),
            "lifecycle_path": last.get("lifecycle_path"),
        })
    return fingerprints


def load_checkpoint(path: Path, run_spec_hash: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BenchmarkContractError(f"cannot read checkpoint: {exc}") from exc
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BenchmarkContractError("checkpoint schema does not support paired-window recovery evidence")
    if payload.get("run_spec_hash") != run_spec_hash:
        raise BenchmarkContractError("checkpoint run-spec/source/database/index mismatch")
    if payload.get("resume_contract") != checkpoint_resume_contract():
        raise BenchmarkContractError("checkpoint resume contract is incompatible")
    completed = _completed_paired_windows(payload.get("completed_paired_windows", []))
    rows = payload.get("rows_by_mode", {})
    if not isinstance(rows, Mapping):
        raise BenchmarkContractError("checkpoint rows-by-mode schema is invalid")
    try:
        window_size = int(payload["window_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError("checkpoint window size is invalid") from exc
    validate_checkpoint_rows(rows, completed, window_size)
    if payload.get("paired_window_fingerprints") != paired_window_fingerprints(rows, completed):
        raise BenchmarkContractError("checkpoint paired-window fingerprint mismatch")
    if payload.get("backend_lifecycle_fingerprints") != backend_lifecycle_fingerprints(rows):
        raise BenchmarkContractError("checkpoint backend lifecycle fingerprint mismatch")
    return payload


def write_checkpoint(path: Path, run_spec_hash: str, rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
                     completed_paired_windows: Sequence[int], window_size: int) -> None:
    completed = [int(window) for window in completed_paired_windows]
    validate_checkpoint_rows(rows_by_mode, completed, window_size)
    atomic_json(path, {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_spec_hash": run_spec_hash,
        "window_size": window_size,
        "rows_by_mode": rows_by_mode,
        "completed_paired_windows": completed,
        "paired_window_fingerprints": paired_window_fingerprints(rows_by_mode, completed),
        "backend_lifecycle_fingerprints": backend_lifecycle_fingerprints(rows_by_mode),
        "resume_contract": checkpoint_resume_contract(),
    })


def json_profile(session: Session, sql: str) -> dict[str, Any]:
    session.execute(sql)
    value = session.one()
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def configure(session: Session, args: argparse.Namespace, mode: str) -> None:
    json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    settings = [
        "SET jit = off", f"SET statement_timeout = {int(args.statement_timeout_ms)}",
        f"SET hnsw.ef_search = {int(args.ef_search)}", f"SET hnsw.iterative_scan = {args.iterative_scan}",
        f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}", f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}",
        f"SET hnsw.metadata_cache_max_mb = {int(args.cache_mb)}", "SET hnsw.page_access = off", "SET hnsw.index_page_access = off",
        f"SET hnsw.d3_probe_requests = {int(args.d3_probe_requests)}",
        f"SET hnsw.d3_min_benefit_per_byte = {float(args.d3_min_benefit_per_byte)}",
        f"SET hnsw.d3_max_fragment_mb = {int(args.d3_max_fragment_mb)}",
        f"SET hnsw.d3_page_min_skip_rate = {float(args.d3_page_min_skip_rate)}",
        f"SET hnsw.filter_strategy = {'off' if mode == 'stock' else args.guidance_filter_strategy}",
    ]
    if args.force_hnsw:
        settings.append("SET enable_sort = off")
    for statement in settings:
        session.execute(statement)


def reset_guidance(session: Session) -> None:
    session.execute("SELECT vector_hnsw_guidance_reset()")


def adaptive_cache_empty_gate(session: Session) -> tuple[bool, dict[str, Any]]:
    reset_guidance(session)
    before = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    session.execute("SELECT vector_hnsw_metadata_cache_reset()")
    after = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    evidence = {
        "before_reset": before,
        "after_reset": after,
        "before_reset_empty": cache_is_empty(before),
        "after_reset_empty": cache_is_empty(after),
    }
    return bool(evidence["after_reset_empty"]), evidence


def activate(session: Session, index: str, atoms: Sequence[str], kind: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    session.execute("SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)", (index, list(atoms), kind))
    session.one()
    activation_ms = (time.perf_counter() - started) * 1000.0
    return json_profile(session, "SELECT vector_hnsw_guidance_profile()"), activation_ms


def run_search(
    session: Session, table: str, predicate: str, candidate_validity_predicate: str,
    query_id: int, k: int
) -> tuple[list[int], dict[str, Any], str, float]:
    session.execute("SELECT vector_hnsw_reset_scan_profile()")
    started = time.perf_counter()
    try:
        session.execute(
            f"SELECT id FROM {table} WHERE ({predicate}) AND ({candidate_validity_predicate}) AND id <> %s "
            f"ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s) LIMIT {int(k)}",
            (query_id, query_id),
        )
        ids = [int(row[0]) for row in session.all()]
        error = ""
    except Exception as exc:  # The row stays in the artifact and invalidates it later.
        ids, error = [], exc.__class__.__name__
    query_ms = (time.perf_counter() - started) * 1000.0
    profile = json_profile(session, "SELECT vector_hnsw_last_scan_profile()")
    return ids, profile, error, query_ms


def reported_profile_build_id(profiles: Sequence[Mapping[str, Any]]) -> str:
    for profile in profiles:
        for key in ("profile_build_id", "build_id", "guide_generation", "fragment_epoch"):
            if profile.get(key) not in (None, ""):
                return str(profile[key])
    return "unreported"


def run_request(session: Session, args: argparse.Namespace, mode: str, request: Request, filter_spec: FilterSpec,
                truth: TruthEntry, provenance: Mapping[str, Any], *, adaptive_started_empty: bool,
                online_materializations_before: int = 0) -> dict[str, Any]:
    cache_before = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    guidance_before = (
        json_profile(session, "SELECT vector_hnsw_guidance_profile()")
        if mode != "stock" else {}
    )
    reset_guidance(session)
    activation_ms = 0.0
    guidance: dict[str, Any] = {}
    activation_attempted = False
    guidance_active = False
    reason = "stock_no_fragment_cache" if mode == "stock" else "eager_prebuilt_request_activation"
    if mode == "adaptive":
        # Every request enters the extension's D3 state machine.  The extension,
        # not this runner, decides whether to probe, admit page guidance, refine
        # to Bloom, reject, or reuse a resident fragment.
        activation_attempted = True
        activation_kind = "adaptive"
        reason = "extension_adaptive_state_machine"
    elif mode == "eager_prebuilt":
        activation_attempted = True
        activation_kind = args.eager_kind
    if activation_attempted:
        guidance, activation_ms = activate(session, args.index, filter_spec.atoms, activation_kind)
        guidance_active = bool(guidance.get("active", False))
        if mode == "adaptive":
            reason = f"extension_adaptive_{guidance.get('adaptive_state', 'unknown')}"
    ids, scan, error, query_ms = run_search(
        session, args.table, filter_spec.predicate, args.candidate_validity_predicate,
        request.query_id, args.k
    )
    cache_after = json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")
    guidance_after = (
        json_profile(session, "SELECT vector_hnsw_guidance_profile()")
        if mode != "stock" else guidance
    )
    e2e_ms = activation_ms + query_ms
    lifecycle_guidance = dict(guidance)
    lifecycle_guidance.update(guidance_after)
    lifecycle = lifecycle_classification(
        cache_before, cache_after, lifecycle_guidance, admitted=guidance_active, reason=reason
    )
    counter_fields = (
        "adaptive_probes", "adaptive_admissions", "adaptive_page_builds", "adaptive_bloom_builds",
        "adaptive_refinements", "adaptive_rejections", "adaptive_evictions", "fragment_cache_hits",
        "fragment_store_hits", "fragment_builds",
    )
    deltas = {f"{field}_delta": _counter_delta(guidance_before, guidance_after, field) for field in counter_fields}
    materialization_observed = bool(
        deltas["fragment_builds_delta"]
        or deltas["adaptive_page_builds_delta"]
        or deltas["adaptive_bloom_builds_delta"]
    )
    probe_observed = bool(deltas["adaptive_probes_delta"])
    refine_observed = bool(deltas["adaptive_refinements_delta"])
    evict_observed = bool(deltas["adaptive_evictions_delta"] or _counter_delta(cache_before, cache_after, "evictions"))
    reuse_observed = bool(
        deltas["fragment_cache_hits_delta"]
        or deltas["fragment_store_hits_delta"]
        or guidance_after.get("composed_guide_hit", False)
        or (guidance_active and not materialization_observed)
    )
    hidden_prebuilt_reused = bool(
        mode == "adaptive"
        and deltas["fragment_store_hits_delta"] > 0
        and online_materializations_before == 0
    )
    lifecycle_events = [
        name for name, observed in (
            ("probe", probe_observed), ("materialize", materialization_observed),
            ("reuse", reuse_observed), ("refine", refine_observed), ("evict", evict_observed),
        ) if observed
    ]
    lifecycle_path = "->".join(lifecycle_events) if lifecycle_events else "none"
    proof = bool(scan.get("planner_proof_succeeded", False)) if guidance_active else True
    return {
        "mode": mode, "request_no": request.request_no, "phase": request.phase, "window": request.window,
        "filter_name": request.filter_name, "predicate": filter_spec.predicate, "atoms": list(filter_spec.atoms),
        "query_no": request.query_no, "query_id": request.query_id, "reuse_distance": request.reuse_distance,
        "e2e_ms": e2e_ms, "query_ms": query_ms, "activation_ms": activation_ms,
        "materialization_ms": float(guidance.get("last_cache_build_ms", 0.0) or 0.0), "returned": len(ids),
        "returned_ids": ids, "recall_at_10": recall_at_10(ids, truth), "error": error,
        "activation_attempted": activation_attempted, "guidance_active": guidance_active,
        "planner_proof_required": guidance_active, "planner_proof_verified": proof,
        "planner_proof_attempted": scan.get("planner_proof_attempted", False),
        "planner_proof_bypass_reason": scan.get("planner_proof_bypass_reason", ""),
        "visited": scan.get("visited_tuples", 0), "returned_profile": scan.get("returned_tuples", 0),
        "guidance_checks": scan.get("guidance_checks", 0), "guidance_skips": scan.get("guidance_skips", 0),
        "cache_entries_before": cache_before.get("entries", 0), "cache_entries_after": cache_after.get("entries", 0),
        "cache_fragments_before": cache_before.get("composed_guide_entries", 0), "cache_fragments_after": cache_after.get("composed_guide_entries", 0),
        "cache_resident_bytes_before": cache_before.get("resident_bytes", 0), "cache_resident_bytes_after": cache_after.get("resident_bytes", 0),
        "cache_profile_before": cache_before, "cache_profile_after": cache_after, "guidance_profile": guidance,
        "adaptive_state": guidance.get("adaptive_state", "not_adaptive"),
        "adaptive_requests": guidance.get("adaptive_requests", 0),
        "adaptive_probes": guidance.get("adaptive_probes", 0),
        "adaptive_admissions": guidance.get("adaptive_admissions", 0),
        "adaptive_refinements": guidance.get("adaptive_refinements", 0),
        "adaptive_rejections": guidance.get("adaptive_rejections", 0),
        "adaptive_score": guidance.get("adaptive_score", 0.0),
        # The patched profile has no immutable build ID in older installations.
        # Bind every request to the sampled index build ID and retain any profile ID
        # separately, so either future profile values or relation rebuilds are auditable.
        "profile_build_id": provenance["database_build_id"],
        "profile_reported_build_id": reported_profile_build_id((scan, guidance, cache_after)),
        "database_build_id": provenance["database_build_id"], "adaptive_cache_started_empty": adaptive_started_empty,
        "online_arm": mode == "adaptive", "explicit_eager_control": mode == "eager_prebuilt",
        "guidance_profile_before": guidance_before, "guidance_profile_after": guidance_after,
        **deltas,
        "probe_observed": probe_observed, "materialization_observed": materialization_observed,
        "reuse_observed": reuse_observed, "refine_observed": refine_observed,
        "evict_observed": evict_observed, "lifecycle_events": lifecycle_events,
        "lifecycle_path": lifecycle_path,
        "fragment_store_hit_delta": deltas["fragment_store_hits_delta"],
        "online_materializations_before": online_materializations_before,
        "hidden_prebuilt_fragment_reused": hidden_prebuilt_reused,
        **lifecycle,
    }


def run_paired_window(backends: Mapping[str, ModeBackend], args: argparse.Namespace, trace: Sequence[Request],
                      filters_by_name: Mapping[str, FilterSpec], truth: Mapping[tuple[str, int], TruthEntry],
                      provenance: Mapping[str, Any], *, window: int,
                      adaptive_started_empty: bool,
                      adaptive_lifecycle_state: dict[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Run one complete trace window once per mode on isolated persistent backends."""
    validate_independent_mode_sessions(backends)
    window_trace = [request for request in trace if request.window == window]
    if len(window_trace) != args.window_size:
        raise BenchmarkContractError(f"trace does not contain one full paired window: {window}")
    if adaptive_lifecycle_state is None:
        adaptive_lifecycle_state = {"online_materializations": 0}
    blocks: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for request in window_trace:
        mode_order = paired_request_mode_order(request.request_no)
        for rank, mode in enumerate(mode_order):
            backend = backends[mode]
            row = run_request(
                backend.session, args, mode, request, filters_by_name[request.filter_name],
                truth[(request.filter_name, request.query_no)], provenance,
                adaptive_started_empty=adaptive_started_empty,
                online_materializations_before=(
                    int(adaptive_lifecycle_state.get("online_materializations", 0))
                    if mode == "adaptive" else 0
                ),
            )
            if mode == "adaptive":
                builds = int(row.get("fragment_builds_delta", 0) or 0)
                if builds <= 0 and row.get("materialization_observed"):
                    builds = 1
                adaptive_lifecycle_state["online_materializations"] = (
                    int(adaptive_lifecycle_state.get("online_materializations", 0)) + builds
                )
                row["online_materializations_after"] = adaptive_lifecycle_state["online_materializations"]
            row["backend_mode"] = mode
            row["backend_pid"] = backend.backend_pid
            row["paired_request_mode_order"] = list(mode_order)
            row["paired_request_mode_rank"] = rank
            row["measurement_schedule"] = PAIRING_SCHEDULE
            blocks[mode].append(row)
    if set(blocks) != set(MODES):
        raise BenchmarkContractError(f"window did not execute every mode: {window}")
    return blocks


def eager_prebuild(session: Session, args: argparse.Namespace, filters: Sequence[FilterSpec]) -> dict[str, Any]:
    """This is intentionally outside timed requests and only used for the eager control."""
    reset_guidance(session)
    session.execute("SELECT vector_hnsw_metadata_cache_reset()")
    total_ms = 0.0
    for item in filters:
        reset_guidance(session)
        profile, activation_ms = activate(session, args.index, item.atoms, args.eager_kind)
        if not bool(profile.get("active", False)):
            raise BenchmarkContractError(
                f"eager {args.eager_kind} prebuild did not activate filter {item.name}"
            )
        total_ms += activation_ms
    reset_guidance(session)
    return {"eager_prebuild_ms": total_ms, "cache_profile": json_profile(session, "SELECT vector_hnsw_metadata_cache_profile()")}


def database_provenance(session: Session, table: str, index: str) -> dict[str, Any]:
    session.execute(
        "SELECT current_setting('server_version'), %s::regclass::oid::bigint, pg_relation_filenode(%s::regclass)::bigint, "
        "%s::regclass::oid::bigint, pg_relation_filenode(%s::regclass)::bigint, pg_get_indexdef(%s::regclass), "
        "(SELECT pg_get_expr(i.indpred, i.indrelid) FROM pg_index AS i WHERE i.indexrelid = %s::regclass)",
        (table, table, index, index, index, index),
    )
    server, table_oid, table_node, index_oid, index_node, indexdef, index_predicate = session.row()
    extension = json_profile(session, "SELECT json_build_object('vector_extension', coalesce((SELECT extversion FROM pg_extension WHERE extname = 'vector'), 'missing'))::text")
    value = {"server_version": server, "table": table, "table_oid": int(table_oid), "table_relfilenode": int(table_node),
             "index": index, "index_oid": int(index_oid), "index_relfilenode": int(index_node),
             "indexdef": indexdef, "index_predicate": index_predicate, **extension}
    value["database_build_id"] = canonical_sha256(value)
    return value


def open_mode_backends(psycopg: Any, conninfo: str, *, table: str, index: str) -> dict[str, ModeBackend]:
    """Open three distinct, long-lived sessions before any timed request begins."""
    backends: dict[str, ModeBackend] = {}
    try:
        for mode in MODES:
            connection = psycopg.connect(conninfo, autocommit=True)
            try:
                session: Session = CursorSession(connection.cursor())
                database = database_provenance(session, table, index)
                session.execute("SELECT pg_backend_pid()")
                backends[mode] = ModeBackend(mode, connection, session, int(session.one()), database)
            except Exception:
                connection.close()
                raise
        validate_independent_mode_sessions(backends)
        database_build_ids = {backend.database["database_build_id"] for backend in backends.values()}
        if len(database_build_ids) != 1:
            raise BenchmarkContractError("mode backends do not observe the same database/index build")
        return backends
    except Exception:
        close_mode_backends(backends)
        raise


def close_mode_backends(backends: Mapping[str, ModeBackend]) -> None:
    for mode in reversed(MODES):
        backend = backends.get(mode)
        if backend is None:
            continue
        try:
            backend.connection.close()
        except Exception:
            pass


def initialize_mode_backends(backends: Mapping[str, ModeBackend], args: argparse.Namespace,
                             filters: Sequence[FilterSpec]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Configure modes, retain eager's local prebuild, then cold-start online persistently."""
    validate_independent_mode_sessions(backends)
    for mode in MODES:
        configure(backends[mode].session, args, mode)
    reset_guidance(backends["stock"].session)
    backends["stock"].session.execute("SELECT vector_hnsw_metadata_cache_reset()")
    adaptive_started_empty, adaptive_reset_evidence = adaptive_cache_empty_gate(backends["adaptive"].session)
    if not adaptive_started_empty:
        raise BenchmarkContractError("adaptive cold-start reset did not leave an empty metadata cache")
    eager_prebuild_evidence = eager_prebuild(backends["eager_prebuilt"].session, args, filters)
    # Eager intentionally materializes its control cache.  Its writes are shared
    # through the persistent store, so the final online gate must happen after
    # that control setup and before the first timed request.
    persistent_reset = clear_fragment_store(backends["adaptive"].session, args.table)
    if int(persistent_reset["prebuilt_fragments"]) != 0:
        raise BenchmarkContractError("online arm persistent fragment store is not empty before run")
    adaptive_reset_evidence["persistent_fragment_store_reset"] = persistent_reset
    adaptive_reset_evidence["prebuilt_fragments"] = persistent_reset["prebuilt_fragments"]
    eager_prebuild_evidence["persistent_store_cleared_after_prebuild"] = True
    return adaptive_started_empty, adaptive_reset_evidence, eager_prebuild_evidence


def source_provenance(args: argparse.Namespace) -> dict[str, Any]:
    truth_manifest = args.truth_manifest
    if not truth_manifest.exists():
        raise BenchmarkContractError("fixed exact GT manifest is required for strict source provenance")
    expected_manifest_name = args.truth.with_name(args.truth.stem + "_manifest.json").name
    if truth_manifest.name != expected_manifest_name:
        raise BenchmarkContractError(
            f"truth manifest name must match its truth CSV: expected {expected_manifest_name}"
        )
    try:
        manifest = json.loads(truth_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"cannot read exact GT manifest: {exc}") from exc
    truth_sha256 = sha256_file(args.truth)
    manifest_truth = ((manifest.get("outputs") or {}).get("truth_csv") or {})
    manifest_predicate = ((manifest.get("validity_contract") or {}).get("candidate_validity_predicate"))
    if manifest.get("artifact_valid") is not True:
        raise BenchmarkContractError("exact GT manifest is not valid")
    if manifest_truth.get("sha256") != truth_sha256:
        raise BenchmarkContractError("exact GT manifest does not bind the selected truth CSV")
    if manifest_predicate != args.candidate_validity_predicate:
        raise BenchmarkContractError(
            "exact GT candidate universe does not match --candidate-validity-predicate"
        )
    return {"script_sha256": sha256_file(Path(__file__)), "filters_sha256": sha256_file(args.filters_csv),
            "truth_sha256": truth_sha256, "truth_manifest_sha256": sha256_file(truth_manifest),
            "truth_manifest_name": truth_manifest.name,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            "truth_manifest_artifact_valid": True}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def make_run_spec(args: argparse.Namespace, source: Mapping[str, Any], database: Mapping[str, Any], trace: Sequence[Request]) -> dict[str, Any]:
    workload_name = FORMAL_WORKLOAD_MANIFEST_NAME if args.requests == FORMAL_REQUESTS and args.window_size == FORMAL_WINDOW else "amazon10m_d3_request_trace_q200_reused_vectors"
    return {"formal": args.requests == FORMAL_REQUESTS and args.window_size == FORMAL_WINDOW, "requests": args.requests,
            "window_size": args.window_size, "seed": args.seed, "phase_boundary": args.requests // 2,
            "workload_manifest_name": workload_name,
            "trace_contract": "10,000-request trace over q200 fixed truth query vectors; vectors are reused",
            "q200_fixed_exact_truth": True, "effective_unique_queries": FORMAL_Q200,
            "unique_query_vectors": FORMAL_Q200, "database_cracking": False,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            "modes": list(MODES),
            "single_client_sequential": True, "measurement_schedule": PAIRING_SCHEDULE,
            "mode_backend_topology": "three_independent_persistent_postgresql_backends",
            "paired_request_mode_order": "round_robin rotation by request number",
            "checkpoint_resume_contract": checkpoint_resume_contract(),
            "adaptive_admission_owner": "pgvector_extension",
            "adaptive_kind": "adaptive", "eager_kind": args.eager_kind,
            "d3_probe_requests": args.d3_probe_requests,
            "d3_min_benefit_per_byte": args.d3_min_benefit_per_byte,
            "d3_max_fragment_mb": args.d3_max_fragment_mb,
            "d3_page_min_skip_rate": args.d3_page_min_skip_rate,
            "cache_mb": args.cache_mb, "guidance_filter_strategy": args.guidance_filter_strategy,
            "source": source, "database": database, "trace_sha256": canonical_sha256([asdict(item) for item in trace])}


def execute_experiment(args: argparse.Namespace) -> int:
    reject_cross_process_resume(args.resume)
    try:
        import psycopg
        from common_pg import pg_config_from_env
    except ImportError as exc:
        raise BenchmarkContractError("execution needs psycopg and common_pg") from exc
    filters = load_filters(args.filters_csv)
    truth = load_truth(args.truth, filters)
    trace = build_trace(filters, truth, requests=args.requests, window_size=args.window_size, seed=args.seed)
    source = source_provenance(args)
    filters_by_name = {item.name: item for item in filters}
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    completed_paired_windows: list[int] = []
    adaptive_lifecycle_state = {"online_materializations": 0}
    adaptive_reset_evidence: dict[str, Any] | None = None
    eager_prebuild_evidence: dict[str, Any] | None = None
    backends: dict[str, ModeBackend] = {}
    database: dict[str, Any] = {}
    backend_sessions: dict[str, dict[str, Any]] = {}
    checkpoint = checkpoint_path(args.out)
    try:
        backends = open_mode_backends(psycopg, pg_config_from_env().conninfo, table=args.table, index=args.index)
        database = dict(backends["stock"].database)
        normalized_index_predicate = "".join(str(database.get("index_predicate") or "").split()).strip("()")
        normalized_candidate_predicate = "".join(args.candidate_validity_predicate.split()).strip("()")
        if normalized_index_predicate != normalized_candidate_predicate:
            raise BenchmarkContractError(
                "target HNSW partial-index predicate does not match the exact GT candidate universe: "
                f"index={database.get('index_predicate')!r} expected={args.candidate_validity_predicate!r}"
            )
        backend_sessions = {
            mode: {"backend_pid": backends[mode].backend_pid, "database_build_id": backends[mode].database["database_build_id"]}
            for mode in MODES
        }
        run_spec = make_run_spec(args, source, database, trace)
        spec_hash = canonical_sha256(run_spec)
        if checkpoint.exists():
            load_checkpoint(checkpoint, spec_hash)
            raise BenchmarkContractError(
                "checkpoint exists after a prior interrupted run; it is complete paired-window evidence only and cannot be resumed "
                "across newly opened mode backends"
            )
        adaptive_started_empty, adaptive_reset_evidence, eager_prebuild_evidence = initialize_mode_backends(
            backends, args, filters
        )
        persistent_reset = adaptive_reset_evidence.get("persistent_fragment_store_reset", {})
        if int(persistent_reset.get("heap_oid", -1)) != int(database.get("table_oid", -2)):
            raise BenchmarkContractError("online fragment reset heap_oid does not match target table")
        for window in range(args.requests // args.window_size):
            blocks = run_paired_window(
                backends, args, trace, filters_by_name, truth, database, window=window,
                adaptive_started_empty=adaptive_started_empty,
                adaptive_lifecycle_state=adaptive_lifecycle_state,
            )
            if not rows_by_mode["adaptive"] and blocks["adaptive"]:
                blocks["adaptive"][0]["adaptive_reset_evidence"] = adaptive_reset_evidence
                blocks["adaptive"][0]["persistent_fragment_reset_proof"] = persistent_reset
            for mode in MODES:
                rows_by_mode[mode].extend(blocks[mode])
            completed_paired_windows.append(window)
            # The only durable state transition occurs after all three isolated caches completed this trace window.
            write_checkpoint(checkpoint, spec_hash, rows_by_mode, completed_paired_windows, args.window_size)
    finally:
        close_mode_backends(backends)
    errors = validate_artifact(rows_by_mode, trace, recall_delta=args.recall_delta, provenance=database)
    all_rows = [row for mode in MODES for row in rows_by_mode[mode]]
    stock_by_request = {int(row["request_no"]): row for row in rows_by_mode["stock"]}
    windows = [{"mode": mode, "window": window, "phase": next(request.phase for request in trace if request.window == window),
                **summary_for_window([row for row in rows_by_mode[mode] if row["window"] == window], bootstrap_samples=args.bootstrap_samples,
                                     bootstrap_seed=args.bootstrap_seed + window)}
               for mode in MODES for window in range(args.requests // args.window_size)]
    for item in windows:
        if item["mode"] != "stock":
            mode_rows = [row for row in rows_by_mode[item["mode"]] if row["window"] == item["window"]]
            item["cumulative_savings_vs_stock_ms"] = sum(
                float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
                for row in mode_rows if int(row["request_no"]) in stock_by_request and not row.get("error")
            )
            item["cumulative_break_even_request"] = break_even_request(mode_rows, stock_by_request)
        else:
            item["cumulative_savings_vs_stock_ms"] = 0.0
            item["cumulative_break_even_request"] = None
    timeline = []
    for mode in MODES:
        cumulative_build = 0.0
        cumulative_savings = 0.0
        for row in sorted(rows_by_mode[mode], key=lambda item: item["request_no"]):
            cumulative_build += float(row["materialization_ms"])
            if mode != "stock" and int(row["request_no"]) in stock_by_request:
                cumulative_savings += float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
            timeline.append({"mode": mode, "request_no": row["request_no"], "phase": row["phase"],
                             "cumulative_build_ms": cumulative_build, "cache_resident_bytes": row["cache_resident_bytes_after"],
                             "fragment_created": row["fragment_created"], "fragment_reused": row["fragment_reused"],
                             "fragment_store_hit_delta": row.get("fragment_store_hit_delta", 0),
                             "lifecycle_path": row.get("lifecycle_path", "none"),
                             "cumulative_savings_vs_stock_ms": cumulative_savings})
    for item in windows:
        stock_window = next((candidate for candidate in windows if candidate["mode"] == "stock" and candidate["window"] == item["window"]), None)
        item["benefit_vs_stock_mean_ms"] = (float(stock_window["e2e_mean_ms"]) - float(item["e2e_mean_ms"])) if stock_window else 0.0
    cumulative_build_cost = {
        mode: sum(float(row["materialization_ms"]) for row in rows_by_mode[mode]) for mode in MODES
    }
    phase_shift_recovery = {
        mode: {
            "first_shift_window": next((item for item in windows if item["mode"] == mode and item["phase"] == "phase_shift_hot"), None),
            "shift_windows": [item for item in windows if item["mode"] == mode and item["phase"] == "phase_shift_hot"],
        }
        for mode in MODES
    }
    phase_summaries = []
    for mode in MODES:
        for phase in ("steady_hot", "phase_shift_hot"):
            phase_rows = [row for row in rows_by_mode[mode] if row["phase"] == phase]
            phase_summary = summary_for_window(
                phase_rows, bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + (0 if phase == "steady_hot" else 1),
            )
            if mode != "stock":
                phase_summary["cumulative_savings_vs_stock_ms"] = sum(
                    float(stock_by_request[int(row["request_no"])] ["e2e_ms"]) - float(row["e2e_ms"])
                    for row in phase_rows if int(row["request_no"]) in stock_by_request and not row.get("error")
                )
                phase_summary["cumulative_break_even_request"] = break_even_request(phase_rows, stock_by_request)
            else:
                phase_summary["cumulative_savings_vs_stock_ms"] = 0.0
                phase_summary["cumulative_break_even_request"] = None
            phase_summaries.append({"mode": mode, "phase": phase, **phase_summary})
    summary = {"artifact_valid": not errors, "validation_errors": errors, "run_spec": run_spec, "run_spec_hash": spec_hash,
               "effective_unique_queries": FORMAL_Q200, "unique_query_vectors": FORMAL_Q200,
               "database_cracking": False,
               "non_formal_debug_override": args.requests != FORMAL_REQUESTS or args.window_size != FORMAL_WINDOW,
               "window_summaries": windows, "phase_summaries": phase_summaries,
               "cumulative_build_cost_ms": cumulative_build_cost,
               "break_even_request": {mode: break_even_request(rows_by_mode[mode], stock_by_request) for mode in MODES if mode != "stock"},
               "cumulative_break_even_request": {mode: break_even_request(rows_by_mode[mode], stock_by_request) for mode in MODES if mode != "stock"},
               "phase_shift_recovery": phase_shift_recovery,
               "adaptive_reset_evidence": adaptive_reset_evidence,
               "eager_prebuild_evidence": eager_prebuild_evidence,
               "backend_sessions": backend_sessions,
               "checkpoint_resume_contract": checkpoint_resume_contract(),
               "measurement_mode": PAIRING_SCHEDULE}
    write_csv(args.out, all_rows)
    write_csv(args.out.with_name(args.out.stem + "_windows.csv"), windows)
    write_csv(args.out.with_name(args.out.stem + "_timeline.csv"), timeline)
    atomic_json(args.out.with_name(args.out.stem + "_summary.json"), summary)
    if errors:
        return 2
    checkpoint_path(args.out).unlink(missing_ok=True)
    return 0


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal Amazon-10M workload-driven D3 adaptation lifecycle benchmark")
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--candidate-validity-predicate", default=DEFAULT_CANDIDATE_VALIDITY_PREDICATE)
    parser.add_argument("--out", type=Path, default=ROOT / "results/hybrid_vector_db/amazon10m_d3_adaptation_lifecycle.csv")
    parser.add_argument("--execute", action="store_true", help="run the database experiment; dry-run is the default")
    parser.add_argument("--dry-run", action="store_true", help="print the formal contract without reading inputs or connecting")
    parser.add_argument("--resume", action="store_true",
                        help="rejected: backend-local D3/cache state cannot be restored across processes")
    parser.add_argument("--requests", type=int, default=FORMAL_REQUESTS, help="debug only when not 10000; labels output non-formal")
    parser.add_argument("--window-size", type=int, default=FORMAL_WINDOW, help="debug only when not 100; labels output non-formal")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--d3-probe-requests", "--admission-reuse-threshold", type=int, default=2,
                        help="stock probes observed by the extension before adaptive admission")
    parser.add_argument("--eager-kind", choices=("bloom", "page"), default="bloom")
    parser.add_argument("--d3-min-benefit-per-byte", type=float, default=0.0)
    parser.add_argument("--d3-max-fragment-mb", type=int, default=256)
    parser.add_argument(
        "--d3-page-min-skip-rate", "--d3-refine-skip-rate",
        dest="d3_page_min_skip_rate", type=float, default=0.80,
        help="refine page guidance to Bloom when its measured skip rate is below this value",
    )
    parser.add_argument("--cache-mb", type=int, default=1024)
    parser.add_argument("--guidance-filter-strategy", choices=("guided_collect", "safe_guided"), default="safe_guided")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=10000)
    parser.add_argument("--iterative-scan", choices=("off", "relaxed_order", "strict_order"), default="strict_order")
    parser.add_argument("--max-scan-tuples", type=int, default=500000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recall-delta", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {"dry_run": True, "database_connected": False, "inputs_read": False, "files_written": False,
            "modes": list(MODES), "requests": args.requests, "window_size": args.window_size,
            "formal": args.requests == FORMAL_REQUESTS and args.window_size == FORMAL_WINDOW,
            "debug_override_labeled_non_formal": args.requests != FORMAL_REQUESTS or args.window_size != FORMAL_WINDOW,
            "fixed_exact_gt_q200": True, "unique_query_vectors": FORMAL_Q200,
            "candidate_validity_predicate": args.candidate_validity_predicate,
            "trace_contract": "10,000-request trace over q200 fixed truth query vectors; vectors are reused",
            "database_cracking": False, "single_client_sequential": True,
            "workload_manifest_name": FORMAL_WORKLOAD_MANIFEST_NAME,
            "measurement_schedule": PAIRING_SCHEDULE,
            "mode_backend_topology": "three_independent_persistent_postgresql_backends",
            "checkpoint_resume_contract": checkpoint_resume_contract(),
            "adaptive_contract": "reset empty metadata cache and target-heap persistent fragment store; no activate/prewarm outside timed requests"}


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if args.requests <= 0 or args.window_size <= 0 or args.requests % args.window_size:
        raise SystemExit("--requests must be positive and divisible by --window-size")
    if args.d3_probe_requests < 1:
        raise SystemExit("--d3-probe-requests must be at least one")
    if args.dry_run or not args.execute:
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    return execute_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
