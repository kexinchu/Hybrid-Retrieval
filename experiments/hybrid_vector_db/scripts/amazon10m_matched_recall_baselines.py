from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .pgvector_target_recall_selectivity_runner import (
        bootstrap_mean_bounds,
        bootstrap_mean_ci,
        percentile,
    )
except ImportError:  # Direct script execution puts this directory on sys.path.
    from pgvector_target_recall_selectivity_runner import (  # type: ignore[no-redef]
        bootstrap_mean_bounds,
        bootstrap_mean_ci,
        percentile,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_TRUTH = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv"
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_FAISS_INDEX = ROOT / "data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index"
DEFAULT_RESULTS = ROOT / "results/hybrid_vector_db"
DEFAULT_TABLE = "amazon_grocery_reviews_10m_pgvector"
DEFAULT_EF_SEARCH = (250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
DEFAULT_TARGETS = (0.90, 0.95, 0.99)
DEFAULT_CALIBRATION_QUERY_OFFSET = 20
DEFAULT_CALIBRATION_QUERIES = 80
DEFAULT_FINAL_QUERY_OFFSET = 100
DEFAULT_FINAL_QUERIES = 100
TARGET_SELECTION_RULE = "query-level mean recall@10 >= target; bootstrap CI/LCB reporting only"
NA = "N/A"
FINALIZER_VERSION = "amazon10m-matched-recall-finalizer-v1"


@dataclass(frozen=True)
class FilterSpec:
    name: str
    target_rate: str
    predicate: str
    expected_rows: int
    actual_pct: float


@dataclass(frozen=True)
class TruthEntry:
    query_no: int
    query_id: int
    filter_name: str
    split: str
    ids: tuple[int, ...]
    candidate_rows: int
    kth_distance_sq: float
    tie_tolerance: float
    self_excluded: bool


@dataclass
class AllowList:
    selector: Any | None
    bitmap: Any | None
    rows: int
    build_ms: float
    bitmap_bytes: int
    valid: bool
    error: str = ""


def parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("integer list values must be greater than zero")
    return list(dict.fromkeys(parsed))


def parse_targets(value: str) -> list[float]:
    try:
        parsed = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated recall list") from exc
    if not parsed or any(target <= 0.0 or target > 1.0 for target in parsed):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return parsed


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


def validate_table_name(value: str) -> str:
    parts = value.split(".")
    if len(parts) not in (1, 2):
        raise argparse.ArgumentTypeError("table must be table or schema.table")
    for part in parts:
        if not part or not (part[0].isalpha() or part[0] == "_"):
            raise argparse.ArgumentTypeError("table must contain unquoted SQL identifiers")
        if any(not (char.isalnum() or char in "_$") for char in part):
            raise argparse.ArgumentTypeError("table must contain unquoted SQL identifiers")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as target:
        if not fields:
            return
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_write_outputs(
    outputs: dict[Path, tuple[str, Any]],
) -> None:
    """Write a small derived artifact set without exposing a partial result."""
    destinations = list(outputs)
    if not destinations:
        return
    parent = destinations[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".amazon10m-finalize-", dir=str(parent)))
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for destination, (kind, value) in outputs.items():
            staged_path = stage / destination.name
            if kind == "json":
                write_json(staged_path, value)
            else:
                write_csv(staged_path, value)
            staged[destination] = staged_path
        for destination in destinations:
            if destination.exists():
                backup = stage / f"{destination.name}.backup"
                shutil.copy2(destination, backup)
                backups[destination] = backup
        for destination in destinations:
            os.replace(staged[destination], destination)
            replaced.append(destination)
    except Exception:
        for destination in reversed(replaced):
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            elif destination.exists():
                destination.unlink()
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_filter_specs(path: Path, selected: set[str] | None = None) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    seen: set[str] = set()
    for row in read_csv(path):
        name = row["filter_name"]
        if selected and name not in selected:
            continue
        if name in seen:
            raise ValueError(f"duplicate filter_name in {path}: {name}")
        seen.add(name)
        specs.append(
            FilterSpec(
                name=name,
                target_rate=row["target_rate"],
                predicate=row["predicate"],
                expected_rows=int(row["count"]),
                actual_pct=float(row["actual_pct"]),
            )
        )
    if selected:
        missing = selected - seen
        if missing:
            raise ValueError(f"missing filter specs: {sorted(missing)}")
    if not specs:
        raise ValueError(f"no filter specs in {path}")
    return specs


def _parse_ids(value: str, k: int) -> tuple[int, ...]:
    ids = tuple(int(part) for part in value.split(",") if part.strip())
    if len(ids) != k or len(set(ids)) != k:
        raise ValueError(f"truth top-k must contain {k} distinct IDs, got {len(ids)}")
    return ids


def load_truth(
    path: Path,
    filter_specs: Sequence[FilterSpec],
    calibration_query_nos: Sequence[int],
    final_query_nos: Sequence[int],
    k: int,
) -> tuple[dict[tuple[str, int], TruthEntry], dict[int, int]]:
    calibration_set = set(calibration_query_nos)
    final_set = set(final_query_nos)
    if calibration_set & final_set:
        raise ValueError("calibration and final query_no sets overlap")
    expected_query_nos = calibration_set | final_set
    filter_names = {spec.name for spec in filter_specs}
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_ids: dict[int, int] = {}

    rows = read_csv(path)
    required_fields = {
        "filtered_rows",
        "kth_distance_sq",
        "tie_tolerance",
        "self_excluded",
        "query_split",
    }
    if not rows or not required_fields.issubset(rows[0]):
        missing = sorted(required_fields - (set(rows[0]) if rows else set()))
        raise ValueError(f"truth artifact uses the retired schema; missing fields: {missing}")

    for row in rows:
        if row.get("method") != "pre_filter_exact":
            continue
        filter_name = row["filter_name"]
        query_no = int(row["query_no"])
        if filter_name not in filter_names or query_no not in expected_query_nos:
            continue
        query_id = int(row["query_id"])
        previous_query_id = query_ids.setdefault(query_no, query_id)
        if previous_query_id != query_id:
            raise ValueError(f"query_no={query_no} maps to multiple query IDs")
        expected_split = "calibration" if query_no in calibration_set else "final"
        split = row.get("query_split", expected_split)
        if split != expected_split:
            raise ValueError(
                f"query_no={query_no} has split={split!r}, expected {expected_split!r}"
            )
        key = (filter_name, query_no)
        if key in truth:
            raise ValueError(f"duplicate truth pair: {key}")
        ids = _parse_ids(row["exact_filtered_topk_ids"], k)
        self_excluded = str(row["self_excluded"]).strip().lower() == "true"
        if not self_excluded:
            raise ValueError(f"truth pair {key} did not exclude the query row")
        if query_id in ids:
            raise ValueError(f"truth pair {key} contains its own query ID")
        if row.get("recall_at_10_exact_filtered") not in (None, "", "1", "1.0"):
            raise ValueError(f"truth pair {key} is not marked exact")
        truth[key] = TruthEntry(
            query_no=query_no,
            query_id=query_id,
            filter_name=filter_name,
            split=split,
            ids=ids,
            candidate_rows=int(float(row["filtered_rows"])),
            kth_distance_sq=float(row["kth_distance_sq"]),
            tie_tolerance=float(row["tie_tolerance"]),
            self_excluded=self_excluded,
        )

    expected_pairs = {
        (filter_name, query_no)
        for filter_name in filter_names
        for query_no in expected_query_nos
    }
    missing = expected_pairs - set(truth)
    extra_query_ids = expected_query_nos - set(query_ids)
    if missing or extra_query_ids:
        preview = sorted(missing)[:5]
        raise ValueError(
            f"truth grid incomplete: missing_pairs={len(missing)} preview={preview} "
            f"missing_query_ids={sorted(extra_query_ids)}"
        )
    calibration_ids = {query_ids[query_no] for query_no in calibration_set}
    final_ids = {query_ids[query_no] for query_no in final_set}
    if len(calibration_ids) != len(calibration_set) or len(final_ids) != len(final_set):
        raise ValueError("query IDs must be unique within each query split")
    if calibration_ids & final_ids:
        raise ValueError("calibration and final query IDs overlap")
    for spec in filter_specs:
        candidate_counts = {
            truth[(spec.name, query_no)].candidate_rows for query_no in expected_query_nos
        }
        if candidate_counts != {spec.expected_rows}:
            raise ValueError(
                f"filter={spec.name} candidate count mismatch: "
                f"config={spec.expected_rows} truth={sorted(candidate_counts)}"
            )
    return truth, query_ids


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[Any, int, int]:
    import numpy as np

    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid fbin header: {path}")
    stored_rows, dimensions = struct.unpack("ii", header)
    rows = min(stored_rows, limit) if limit is not None else stored_rows
    vectors = np.memmap(
        path,
        dtype="float32",
        mode="r",
        offset=8,
        shape=(stored_rows, dimensions),
    )
    return vectors[:rows], rows, dimensions


def exact_sql(table: str, predicate: str, k: int) -> str:
    return f"""
WITH filtered AS MATERIALIZED (
    SELECT id, embedding
    FROM {table}
    WHERE ({predicate}) AND id <> %s
)
SELECT id
FROM filtered
ORDER BY embedding <-> %s::vector, id
LIMIT {int(k)}
""".strip()


def plan_index_names(plan: Any) -> set[str]:
    names: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("Index Name"):
                names.add(str(node["Index Name"]))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(plan)
    return names


def assert_no_hnsw_index(plan: Any, hnsw_indexes: Iterable[str]) -> set[str]:
    used = plan_index_names(plan)
    forbidden = {name for item in hnsw_indexes for name in (item, item.rsplit(".", 1)[-1])}
    offenders = sorted(
        name for name in used if name in forbidden or "hnsw" in name.lower()
    )
    if offenders:
        raise RuntimeError(f"sql_first_exact EXPLAIN used HNSW index(es): {offenders}")
    return used


def decode_explain(value: Any) -> Any:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, list) and value and isinstance(value[0], dict) and "Plan" in value[0]:
        return value[0]["Plan"]
    if isinstance(value, dict) and "Plan" in value:
        return value["Plan"]
    raise ValueError("unexpected EXPLAIN (FORMAT JSON) result")


def balanced_order(values: Sequence[Any], block_no: int, seed: int) -> list[Any]:
    base = list(values)
    random.Random(seed).shuffle(base)
    if not base:
        return []
    offset = block_no % len(base)
    return base[offset:] + base[:offset]


def set_bitmap_ids(bitmap: Any, ids: Sequence[int] | Any, total_rows: int) -> int:
    import numpy as np

    values = np.asarray(ids, dtype=np.int64)
    if values.size == 0:
        return 0
    if int(values.min()) < 0 or int(values.max()) >= total_rows:
        raise ValueError("allow-list contains an ID outside the Faiss row range")
    byte_positions = values >> 3
    masks = np.left_shift(np.uint8(1), (values & 7).astype(np.uint8))
    np.bitwise_or.at(bitmap, byte_positions, masks)
    return int(values.size)


def bitmap_contains(bitmap: Any, row_id: int) -> bool:
    return bool(int(bitmap[row_id >> 3]) & (1 << (row_id & 7)))


def build_allow_list(
    conn: Any,
    faiss_module: Any,
    table: str,
    spec: FilterSpec,
    total_rows: int,
    fetch_rows: int,
) -> AllowList:
    import numpy as np

    started = time.perf_counter()
    bitmap = np.zeros((total_rows + 7) // 8, dtype=np.uint8)
    streamed_rows = 0
    try:
        cursor_name = f"allowlist_{hashlib.sha1(spec.name.encode()).hexdigest()[:12]}"
        with conn.transaction():
            with conn.cursor(name=cursor_name) as cursor:
                cursor.execute(f"SELECT id FROM {table} WHERE {spec.predicate}")
                while True:
                    batch = cursor.fetchmany(fetch_rows)
                    if not batch:
                        break
                    streamed_rows += set_bitmap_ids(
                        bitmap,
                        np.fromiter((int(row[0]) for row in batch), dtype=np.int64),
                        total_rows,
                    )
        # Faiss expects the number of addressable IDs (bits), not the backing
        # array's byte count. Passing bitmap.size would silently exclude IDs
        # above total_rows / 8 and collapse recall on a 10M collection.
        selector = faiss_module.IDSelectorBitmap(total_rows, faiss_module.swig_ptr(bitmap))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if streamed_rows != spec.expected_rows:
            return AllowList(
                selector=selector,
                bitmap=bitmap,
                rows=streamed_rows,
                build_ms=elapsed_ms,
                bitmap_bytes=int(bitmap.nbytes),
                valid=False,
                error=f"row_count_mismatch: expected={spec.expected_rows} actual={streamed_rows}",
            )
        return AllowList(
            selector=selector,
            bitmap=bitmap,
            rows=streamed_rows,
            build_ms=elapsed_ms,
            bitmap_bytes=int(bitmap.nbytes),
            valid=True,
        )
    except Exception as exc:  # Keep other filters measurable and make this one explicitly invalid.
        return AllowList(
            selector=None,
            bitmap=bitmap,
            rows=streamed_rows,
            build_ms=(time.perf_counter() - started) * 1000.0,
            bitmap_bytes=int(bitmap.nbytes),
            valid=False,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def recall_at_k(result_ids: Sequence[int], truth_ids: Sequence[int], k: int) -> float:
    denominator = min(k, len(truth_ids))
    if denominator == 0:
        return 0.0
    return len(set(result_ids[:k]) & set(truth_ids[:k])) / denominator


def tie_aware_recall_at_k(
    result_ids: Sequence[int],
    query_id: int,
    vectors: Any,
    truth: TruthEntry,
    k: int,
) -> float:
    import numpy as np

    unique_ids: list[int] = []
    seen: set[int] = set()
    for value in result_ids:
        row_id = int(value)
        if row_id == query_id or row_id in seen:
            continue
        if row_id < 0 or row_id >= len(vectors):
            raise ValueError(f"result ID outside vector row range: {row_id}")
        seen.add(row_id)
        unique_ids.append(row_id)
        if len(unique_ids) == k:
            break
    if not unique_ids:
        return 0.0
    query = np.asarray(vectors[query_id], dtype=np.float32)
    candidates = np.asarray(vectors[np.asarray(unique_ids, dtype=np.int64)], dtype=np.float32)
    distances = np.einsum("ij,ij->i", candidates - query, candidates - query)
    threshold = truth.kth_distance_sq + truth.tie_tolerance
    qualifying = int(np.count_nonzero(distances <= threshold))
    return min(k, qualifying) / k


def search_faiss(
    index: Any,
    faiss_module: Any,
    query: Any,
    selector: Any,
    ef_search: int,
    k: int,
    query_id: int | None = None,
) -> tuple[list[int], float]:
    import numpy as np

    query_batch = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    params = faiss_module.SearchParametersHNSW()
    params.efSearch = int(ef_search)
    params.sel = selector
    started = time.perf_counter()
    request_k = k + 1 if query_id is not None else k
    _, labels = index.search(query_batch, request_k, params=params)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids = [
        int(value)
        for value in labels[0]
        if int(value) >= 0 and (query_id is None or int(value) != query_id)
    ]
    return ids[:k], elapsed_ms


def search_sql_exact(
    cursor: Any,
    sql_text: str,
    query_id: int,
    query_vector: str,
) -> tuple[list[int], float]:
    started = time.perf_counter()
    cursor.execute(sql_text, (query_id, query_vector))
    rows = cursor.fetchall()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return [int(row[0]) for row in rows], elapsed_ms


def pair_key(filter_name: str, query_no: int, repeat: int) -> str:
    return f"{filter_name}|q{query_no}|r{repeat}"


def measurement_row(
    *,
    phase: str,
    method: str,
    spec: FilterSpec,
    query_no: int,
    query_id: int,
    repeat: int,
    schedule_position: int,
    block_no: int,
    ef_search: int | str,
    result_ids: Sequence[int] | None,
    truth_ids: Sequence[int],
    latency_ms: float | str,
    truth_entry: TruthEntry | None = None,
    vectors: Any | None = None,
    error: str = "",
    matched_target_recalls: Sequence[float] = (),
) -> dict[str, Any]:
    valid = not error
    return {
        "phase": phase,
        "method": method,
        "filter_name": spec.name,
        "target_rate": spec.target_rate,
        "predicate": spec.predicate,
        "actual_selectivity": spec.actual_pct / 100.0,
        "ef_search": ef_search,
        "pair_key": pair_key(spec.name, query_no, repeat),
        "block_no": block_no,
        "schedule_position": schedule_position,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "matched_target_recalls": ",".join(f"{target:.2f}" for target in matched_target_recalls),
        "latency_definition": "search_only",
        "search_latency_ms": latency_ms if valid else NA,
        "recall_at_10": (
            tie_aware_recall_at_k(
                result_ids or [], query_id, vectors, truth_entry, len(truth_ids)
            )
            if valid and truth_entry is not None and vectors is not None
            else recall_at_k(result_ids or [], truth_ids, len(truth_ids))
            if valid
            else NA
        ),
        "recall_contract": (
            "distance_threshold_tie_aware" if truth_entry is not None else "id_intersection_test_only"
        ),
        "returned": len(result_ids or []) if valid else NA,
        "result_ids": ",".join(str(value) for value in (result_ids or [])) if valid else NA,
        "valid": valid,
        "error": error,
    }


def setup_row(spec: FilterSpec, allow_list: AllowList) -> dict[str, Any]:
    return {
        "phase": "setup",
        "method": "faiss_allowlist",
        "filter_name": spec.name,
        "target_rate": spec.target_rate,
        "predicate": spec.predicate,
        "actual_selectivity": spec.actual_pct / 100.0,
        "ef_search": NA,
        "pair_key": NA,
        "query_no": NA,
        "query_id": NA,
        "repeat": NA,
        "latency_definition": "one_time_allowlist_build",
        "search_latency_ms": NA,
        "recall_at_10": NA,
        "returned": NA,
        "result_ids": NA,
        "allowlist_build_rows": allow_list.rows,
        "allowlist_build_ms": allow_list.build_ms,
        "allowlist_bitmap_bytes": allow_list.bitmap_bytes,
        "valid": allow_list.valid,
        "error": allow_list.error,
    }


def _row_ok(row: dict[str, Any]) -> bool:
    value = row.get("valid", False)
    if isinstance(value, str):
        value = value.lower() in {"1", "true", "yes"}
    return bool(value) and not row.get("error")


def aggregate_measurements(
    rows: Sequence[dict[str, Any]],
    *,
    phase: str,
    method: str,
    filter_name: str,
    ef_search: int | None,
    query_nos: Sequence[int],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    items = [
        row
        for row in rows
        if row.get("phase") == phase
        and row.get("method") == method
        and row.get("filter_name") == filter_name
        and (ef_search is None or int(row.get("ef_search", -1)) == ef_search)
    ]
    expected_pairs = {(int(query_no), repeat) for query_no in query_nos for repeat in range(repeats)}
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in items:
        key = (int(row["query_no"]), int(row["repeat"]))
        if key in by_pair:
            duplicates += 1
        by_pair[key] = row
    observed_pairs = set(by_pair)
    def metrics_ok(row: dict[str, Any]) -> bool:
        try:
            latency = float(row["search_latency_ms"])
            recall = float(row["recall_at_10"])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(latency) and latency > 0.0 and 0.0 <= recall <= 1.0

    error_rows = sum(not _row_ok(row) for row in items)
    invalid_metric_rows = sum(_row_ok(row) and not metrics_ok(row) for row in items)
    complete = observed_pairs == expected_pairs and duplicates == 0 and error_rows == 0
    complete = complete and invalid_metric_rows == 0
    base = {
        "queries": len({query_no for query_no, _ in observed_pairs}),
        "samples": len(items),
        "expected_queries": len(query_nos),
        "expected_repeats": repeats,
        "expected_samples": len(expected_pairs),
        "missing_pairs": len(expected_pairs - observed_pairs),
        "extra_pairs": len(observed_pairs - expected_pairs),
        "duplicate_pairs": duplicates,
        "errors": error_rows,
        "invalid_metric_rows": invalid_metric_rows,
        "rows_complete": complete,
        "status": "valid" if complete else "invalid",
    }
    if not complete:
        return {
            **base,
            "recall_mean": NA,
            "recall_lcb95": NA,
            "recall_ci95_low": NA,
            "recall_ci95_high": NA,
            "latency_mean_ms": NA,
            "latency_p50_ms": NA,
            "latency_p95_ms": NA,
            "latency_p99_ms": NA,
            "latency_query_mean_ci95_low_ms": NA,
            "latency_query_mean_ci95_high_ms": NA,
        }

    query_recalls: list[float] = []
    query_latencies: list[float] = []
    sample_latencies: list[float] = []
    for query_no in query_nos:
        query_items = [by_pair[(int(query_no), repeat)] for repeat in range(repeats)]
        query_recalls.append(statistics.fmean(float(row["recall_at_10"]) for row in query_items))
        query_latencies.append(
            statistics.fmean(float(row["search_latency_ms"]) for row in query_items)
        )
        sample_latencies.extend(float(row["search_latency_ms"]) for row in query_items)
    recall_lcb, recall_ci_low, recall_ci_high = bootstrap_mean_bounds(
        query_recalls, bootstrap_samples, bootstrap_seed
    )
    latency_ci_low, latency_ci_high = bootstrap_mean_ci(
        query_latencies, bootstrap_samples, bootstrap_seed + 1
    )
    return {
        **base,
        "recall_mean": statistics.fmean(query_recalls),
        "recall_lcb95": recall_lcb,
        "recall_ci95_low": recall_ci_low,
        "recall_ci95_high": recall_ci_high,
        "latency_mean_ms": statistics.fmean(query_latencies),
        "latency_p50_ms": statistics.median(sample_latencies),
        "latency_p95_ms": percentile(sample_latencies, 0.95),
        "latency_p99_ms": percentile(sample_latencies, 0.99),
        "latency_query_mean_ci95_low_ms": latency_ci_low,
        "latency_query_mean_ci95_high_ms": latency_ci_high,
    }


def calibration_table(
    raw_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    ef_values: Sequence[int],
    targets: Sequence[float],
    query_nos: Sequence[int],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    allow_lists: dict[str, AllowList] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], int]]:
    rows: list[dict[str, Any]] = []
    for filter_no, spec in enumerate(filter_specs):
        for ef_search in ef_values:
            stats = aggregate_measurements(
                raw_rows,
                phase="calibration",
                method="faiss_allowlist",
                filter_name=spec.name,
                ef_search=ef_search,
                query_nos=query_nos,
                repeats=repeats,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + filter_no * 1009 + ef_search,
            )
            allow_list = (allow_lists or {}).get(spec.name)
            for target in targets:
                eligible = bool(
                    stats["status"] == "valid"
                    and float(stats["recall_mean"]) >= target
                )
                rows.append(
                    {
                        "filter_name": spec.name,
                        "target_rate": spec.target_rate,
                        "predicate": spec.predicate,
                        "actual_selectivity": spec.actual_pct / 100.0,
                        "method": "faiss_allowlist",
                        "target_recall": target,
                        "ef_search": ef_search,
                        **stats,
                        "eligible": eligible,
                        "selected": False,
                        "allowlist_build_rows": allow_list.rows if allow_list else NA,
                        "allowlist_build_ms": allow_list.build_ms if allow_list else NA,
                        "allowlist_bitmap_bytes": allow_list.bitmap_bytes if allow_list else NA,
                    }
                )

    selected: dict[tuple[str, float], int] = {}
    for spec in filter_specs:
        for target in targets:
            ladder_rows = [
                row
                for row in rows
                if row["filter_name"] == spec.name
                and float(row["target_recall"]) == target
            ]
            ladder_complete = len(ladder_rows) == len(ef_values) and all(
                row["status"] == "valid" for row in ladder_rows
            )
            max_row = max(ladder_rows, key=lambda row: int(row["ef_search"])) if ladder_rows else None
            observed_metrics = [row for row in ladder_rows if row["status"] == "valid"]
            ladder_proof = {
                "configured_ef_search": list(ef_values),
                "observed_ef_search": [int(row["ef_search"]) for row in ladder_rows],
                "all_configs_complete": ladder_complete,
                "all_mean_below_target": bool(
                    ladder_complete
                    and all(float(row["recall_mean"]) < target for row in ladder_rows)
                ),
            }
            candidates = [row for row in ladder_rows if row["eligible"]]
            if candidates:
                winner = min(
                    candidates,
                    key=lambda row: (float(row["latency_mean_ms"]), int(row["ef_search"])),
                )
                winner["selected"] = True
                selected[(spec.name, target)] = int(winner["ef_search"])
            for row in ladder_rows:
                if not candidates and ladder_complete:
                    row["outcome"] = "unattainable_on_grid"
                    row["selection_status"] = "unattainable_on_grid"
                elif not candidates:
                    row["outcome"] = "calibration_invalid"
                    row["selection_status"] = "no_config_meets_mean"
                elif row["selected"]:
                    row["outcome"] = "selected_pending_final"
                    row["selection_status"] = "selected"
                elif row["eligible"]:
                    row["outcome"] = "selected_pending_final"
                    row["selection_status"] = "eligible_not_selected"
                else:
                    row["outcome"] = "selected_pending_final"
                    row["selection_status"] = "ineligible"
                row["selected_ef_search"] = selected.get((spec.name, target), NA)
                row["calibration_ladder_complete"] = ladder_complete
                row["max_ef_search"] = max_row["ef_search"] if max_row else NA
                row["max_observed_recall_mean"] = (
                    max(float(item["recall_mean"]) for item in observed_metrics)
                    if observed_metrics else NA
                )
                row["max_observed_recall_lcb95"] = (
                    max(float(item["recall_lcb95"]) for item in observed_metrics)
                    if observed_metrics else NA
                )
                row["full_ladder_proof"] = json.dumps(ladder_proof, sort_keys=True)
    return rows, selected


def paired_speedup_bounds(
    sql_rows: Sequence[dict[str, Any]],
    faiss_rows: Sequence[dict[str, Any]],
    query_nos: Sequence[int],
    repeats: int,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    sql_by_pair = {
        (int(row["query_no"]), int(row["repeat"])): float(row["search_latency_ms"])
        for row in sql_rows
    }
    faiss_by_pair = {
        (int(row["query_no"]), int(row["repeat"])): float(row["search_latency_ms"])
        for row in faiss_rows
    }
    sql_query = {
        int(query_no): statistics.fmean(sql_by_pair[(int(query_no), repeat)] for repeat in range(repeats))
        for query_no in query_nos
    }
    faiss_query = {
        int(query_no): statistics.fmean(
            faiss_by_pair[(int(query_no), repeat)] for repeat in range(repeats)
        )
        for query_no in query_nos
    }
    point = statistics.fmean(sql_query.values()) / statistics.fmean(faiss_query.values())
    rng = random.Random(seed)
    query_list = [int(query_no) for query_no in query_nos]
    values: list[float] = []
    for _ in range(max(1, samples)):
        chosen = rng.choices(query_list, k=len(query_list)) if len(query_list) > 1 else query_list
        sql_mean = statistics.fmean(sql_query[query_no] for query_no in chosen)
        faiss_mean = statistics.fmean(faiss_query[query_no] for query_no in chosen)
        values.append(sql_mean / faiss_mean)
    return point, percentile(values, 0.025), percentile(values, 0.975)


def final_summary_table(
    final_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    targets: Sequence[float],
    selected: dict[tuple[str, float], int],
    query_nos: Sequence[int],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    allow_lists: dict[str, AllowList] | None = None,
    calibration_outcomes: dict[tuple[str, float], str] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for filter_no, spec in enumerate(filter_specs):
        sql_stats = aggregate_measurements(
            final_rows,
            phase="final",
            method="sql_first_exact",
            filter_name=spec.name,
            ef_search=None,
            query_nos=query_nos,
            repeats=repeats,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + filter_no * 1009,
        )
        for target_no, target in enumerate(targets):
            ef_search = selected.get((spec.name, target))
            calibration_outcome = (calibration_outcomes or {}).get(
                (spec.name, target),
                "selected_pending_final" if ef_search is not None else "calibration_invalid",
            )
            if ef_search is None:
                faiss_stats = {
                    "status": "valid" if calibration_outcome == "unattainable_on_grid" else "invalid",
                    "rows_complete": calibration_outcome == "unattainable_on_grid",
                    "samples": 0,
                    "expected_samples": 0,
                    "missing_pairs": 0 if calibration_outcome == "unattainable_on_grid" else len(query_nos) * repeats,
                    "errors": 0,
                    "recall_mean": NA,
                    "recall_lcb95": NA,
                    "latency_mean_ms": NA,
                    "latency_p50_ms": NA,
                    "latency_p95_ms": NA,
                    "latency_p99_ms": NA,
                    "latency_query_mean_ci95_low_ms": NA,
                    "latency_query_mean_ci95_high_ms": NA,
                }
            else:
                faiss_stats = aggregate_measurements(
                    final_rows,
                    phase="final",
                    method="faiss_allowlist",
                    filter_name=spec.name,
                    ef_search=ef_search,
                    query_nos=query_nos,
                    repeats=repeats,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + filter_no * 1009 + target_no * 104729,
                )
            matched_pairs = bool(
                ef_search is not None
                and sql_stats["status"] == "valid"
                and faiss_stats["status"] == "valid"
            )
            target_confirmed = bool(
                matched_pairs
                and float(sql_stats["recall_mean"]) >= target
                and float(faiss_stats["recall_mean"]) >= target
            )
            outcome = (
                "unattainable_on_grid"
                if calibration_outcome == "unattainable_on_grid"
                else "selected_and_confirmed"
                if target_confirmed
                else "selected_but_final_unconfirmed"
                if matched_pairs
                else "calibration_invalid"
                if ef_search is None
                else "invalid"
            )
            speedup: float | str = NA
            speedup_low: float | str = NA
            speedup_high: float | str = NA
            if target_confirmed and ef_search is not None:
                sql_items = [
                    row
                    for row in final_rows
                    if row.get("phase") == "final"
                    and row.get("method") == "sql_first_exact"
                    and row.get("filter_name") == spec.name
                ]
                faiss_items = [
                    row
                    for row in final_rows
                    if row.get("phase") == "final"
                    and row.get("method") == "faiss_allowlist"
                    and row.get("filter_name") == spec.name
                    and int(row["ef_search"]) == ef_search
                ]
                speedup, speedup_low, speedup_high = paired_speedup_bounds(
                    sql_items,
                    faiss_items,
                    query_nos,
                    repeats,
                    bootstrap_samples,
                    bootstrap_seed + filter_no * 1009 + target_no,
                )
            allow_list = (allow_lists or {}).get(spec.name)
            common = {
                "filter_name": spec.name,
                "target_rate": spec.target_rate,
                "predicate": spec.predicate,
                "actual_selectivity": spec.actual_pct / 100.0,
                "target_recall": target,
                "selected_faiss_ef_search": ef_search if ef_search is not None else NA,
                "matched_pairs_valid": matched_pairs,
                "matched_recall_comparison_valid": target_confirmed,
                "outcome": outcome,
                "comparison_status": (
                    "valid"
                    if target_confirmed
                    else "unattainable_on_grid"
                    if outcome == "unattainable_on_grid"
                    else "target_unconfirmed"
                    if matched_pairs
                    else "invalid"
                ),
            }
            for method, stats in (("sql_first_exact", sql_stats), ("faiss_allowlist", faiss_stats)):
                artifact_row_valid = bool(
                    outcome == "unattainable_on_grid"
                    or matched_pairs
                )
                metrics_valid = artifact_row_valid and not (
                    method == "faiss_allowlist" and outcome == "unattainable_on_grid"
                )
                method_target_confirmed = bool(
                    metrics_valid and float(stats["recall_mean"]) >= target
                )
                output.append(
                    {
                        **common,
                        "method": method,
                        "status": "valid" if artifact_row_valid else "invalid",
                        "queries": (
                            0 if method == "faiss_allowlist" and outcome == "unattainable_on_grid"
                            else len(query_nos) if artifact_row_valid else NA
                        ),
                        "samples": (
                            0 if method == "faiss_allowlist" and outcome == "unattainable_on_grid"
                            else len(query_nos) * repeats if artifact_row_valid else NA
                        ),
                        "expected_samples": (
                            0 if method == "faiss_allowlist" and outcome == "unattainable_on_grid"
                            else len(query_nos) * repeats if artifact_row_valid else NA
                        ),
                        "recall_mean": stats["recall_mean"] if metrics_valid else NA,
                        "recall_lcb95": stats["recall_lcb95"] if metrics_valid else NA,
                        "search_latency_mean_ms": stats["latency_mean_ms"] if metrics_valid else NA,
                        "search_latency_p50_ms": stats.get("latency_p50_ms", NA) if metrics_valid else NA,
                        "search_latency_p95_ms": stats.get("latency_p95_ms", NA) if metrics_valid else NA,
                        "search_latency_p99_ms": stats.get("latency_p99_ms", NA) if metrics_valid else NA,
                        "search_latency_mean_ci95_low_ms": (
                            stats.get("latency_query_mean_ci95_low_ms", NA) if metrics_valid else NA
                        ),
                        "search_latency_mean_ci95_high_ms": (
                            stats.get("latency_query_mean_ci95_high_ms", NA) if metrics_valid else NA
                        ),
                        "target_confirmed_in_final": method_target_confirmed if metrics_valid else False if outcome == "unattainable_on_grid" else NA,
                        "speedup_vs_sql_first_exact": (
                            1.0
                            if target_confirmed and method == "sql_first_exact"
                            else speedup
                        ),
                        "speedup_ci95_low": (
                            1.0
                            if target_confirmed and method == "sql_first_exact"
                            else speedup_low
                        ),
                        "speedup_ci95_high": (
                            1.0
                            if target_confirmed and method == "sql_first_exact"
                            else speedup_high
                        ),
                        "allowlist_build_rows": (
                            allow_list.rows if method == "faiss_allowlist" and allow_list else NA
                        ),
                        "allowlist_build_ms_one_time": (
                            allow_list.build_ms if method == "faiss_allowlist" and allow_list else NA
                        ),
                        "allowlist_bitmap_bytes": (
                            allow_list.bitmap_bytes if method == "faiss_allowlist" and allow_list else NA
                        ),
                        "missing_pairs": stats.get("missing_pairs", NA),
                        "errors": stats.get("errors", NA),
                    }
                )
    return output


def calibration_outcomes_from_rows(
    calibration_rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, float], str]:
    outcomes: dict[tuple[str, float], str] = {}
    for row in calibration_rows:
        key = (str(row["filter_name"]), float(row["target_recall"]))
        outcome = str(row.get("outcome", ""))
        if not outcome:
            raise ValueError(f"calibration row has no outcome: {key}")
        previous = outcomes.setdefault(key, outcome)
        if previous != outcome:
            raise ValueError(f"inconsistent calibration outcome for {key}")
    return outcomes


def artifact_validation_errors(
    calibration_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    filter_specs: Sequence[FilterSpec],
    ef_values: Sequence[int],
    targets: Sequence[float],
) -> list[str]:
    """Validate only integrity. A target outcome is deliberately not an error."""
    errors: list[str] = []
    expected_calibration = {
        (spec.name, float(target), int(ef))
        for spec in filter_specs
        for target in targets
        for ef in ef_values
    }
    observed_calibration: set[tuple[str, float, int]] = set()
    for row in calibration_rows:
        try:
            key = (str(row["filter_name"]), float(row["target_recall"]), int(row["ef_search"]))
        except (KeyError, TypeError, ValueError):
            errors.append("calibration row has an invalid key")
            continue
        if key in observed_calibration:
            errors.append(f"duplicate calibration key: {key}")
        observed_calibration.add(key)
        if row.get("status") != "valid":
            errors.append(f"invalid calibration grid cell: {key}")
    if observed_calibration != expected_calibration:
        errors.append("calibration key coverage is incomplete or contains extras")

    expected_summary = {
        (spec.name, float(target), method)
        for spec in filter_specs
        for target in targets
        for method in ("sql_first_exact", "faiss_allowlist")
    }
    observed_summary: set[tuple[str, float, str]] = set()
    accepted_outcomes = {
        "selected_and_confirmed",
        "selected_but_final_unconfirmed",
        "unattainable_on_grid",
    }
    for row in summary_rows:
        try:
            key = (str(row["filter_name"]), float(row["target_recall"]), str(row["method"]))
        except (KeyError, TypeError, ValueError):
            errors.append("summary row has an invalid key")
            continue
        if key in observed_summary:
            errors.append(f"duplicate summary key: {key}")
        observed_summary.add(key)
        if row.get("status") != "valid":
            errors.append(f"invalid selected-config final: {key}")
        if row.get("outcome") not in accepted_outcomes:
            errors.append(f"invalid outcome: {key}")
    if observed_summary != expected_summary:
        errors.append("summary key coverage is incomplete or contains extras")
    return errors


def prefetch_sql_query_vectors(cursor: Any, table: str, query_ids: Iterable[int]) -> dict[int, str]:
    wanted = sorted(set(int(query_id) for query_id in query_ids))
    cursor.execute(
        f"SELECT id, embedding::text FROM {table} WHERE id = ANY(%s::bigint[])",
        (wanted,),
    )
    vectors = {int(row[0]): str(row[1]) for row in cursor.fetchall()}
    missing = set(wanted) - set(vectors)
    if missing:
        raise ValueError(f"query vector prefetch missing IDs: {sorted(missing)}")
    return vectors


def explain_exact_plan(
    cursor: Any,
    table: str,
    spec: FilterSpec,
    query_vector: str,
    k: int,
    hnsw_indexes: Sequence[str],
) -> dict[str, Any]:
    cursor.execute(
        "EXPLAIN (FORMAT JSON, COSTS OFF) " + exact_sql(table, spec.predicate, k),
        (-1, query_vector),
    )
    plan = decode_explain(cursor.fetchone()[0])
    used_indexes = assert_no_hnsw_index(plan, hnsw_indexes)
    return {"used_indexes": sorted(used_indexes), "hnsw_indexes": sorted(hnsw_indexes)}


def faiss_index_metadata(index: Any, faiss_module: Any, rows: int, dimensions: int) -> dict[str, Any]:
    if not hasattr(index, "hnsw"):
        raise ValueError(f"Faiss index is not an ordinary HNSW index: {type(index).__name__}")
    if int(index.ntotal) != rows or int(index.d) != dimensions:
        raise ValueError(
            f"Faiss/fbin mismatch: index=({index.ntotal}, {index.d}) fbin=({rows}, {dimensions})"
        )
    if int(index.metric_type) != int(faiss_module.METRIC_L2):
        raise ValueError("Faiss HNSW index must use L2 distance")
    level0_neighbors = int(index.hnsw.nb_neighbors(0))
    level1_neighbors = int(index.hnsw.nb_neighbors(1))
    if (level0_neighbors, level1_neighbors) != (32, 16):
        raise ValueError(
            "Faiss HNSW index is not M16: "
            f"level0_neighbors={level0_neighbors} level1_neighbors={level1_neighbors}"
        )
    return {
        "type": type(index).__name__,
        "ntotal": int(index.ntotal),
        "dimensions": int(index.d),
        "metric": "L2",
        "m": 16,
        "level0_neighbors": level0_neighbors,
        "level1_neighbors": level1_neighbors,
    }


def _run_faiss_measurement(
    raw_rows: list[dict[str, Any]],
    *,
    phase: str,
    spec: FilterSpec,
    query_no: int,
    query_id: int,
    query: Any,
    repeat: int,
    schedule_position: int,
    block_no: int,
    ef_search: int,
    truth_entry: TruthEntry,
    vectors: Any,
    allow_list: AllowList,
    index: Any,
    faiss_module: Any,
    k: int,
    matched_target_recalls: Sequence[float] = (),
) -> None:
    try:
        if not allow_list.valid or allow_list.selector is None:
            raise RuntimeError(allow_list.error or "allow-list setup is invalid")
        ids, latency_ms = search_faiss(
            index,
            faiss_module,
            query,
            allow_list.selector,
            ef_search,
            k,
            query_id=query_id,
        )
        raw_rows.append(
            measurement_row(
                phase=phase,
                method="faiss_allowlist",
                spec=spec,
                query_no=query_no,
                query_id=query_id,
                repeat=repeat,
                schedule_position=schedule_position,
                block_no=block_no,
                ef_search=ef_search,
                result_ids=ids,
                truth_ids=truth_entry.ids,
                truth_entry=truth_entry,
                vectors=vectors,
                latency_ms=latency_ms,
                matched_target_recalls=matched_target_recalls,
            )
        )
    except Exception as exc:
        raw_rows.append(
            measurement_row(
                phase=phase,
                method="faiss_allowlist",
                spec=spec,
                query_no=query_no,
                query_id=query_id,
                repeat=repeat,
                schedule_position=schedule_position,
                block_no=block_no,
                ef_search=ef_search,
                result_ids=None,
                truth_ids=truth_entry.ids,
                latency_ms=NA,
                error=f"{exc.__class__.__name__}: {exc}",
                matched_target_recalls=matched_target_recalls,
            )
        )


def run_calibration(
    raw_rows: list[dict[str, Any]],
    *,
    filter_specs: Sequence[FilterSpec],
    query_nos: Sequence[int],
    repeats: int,
    ef_values: Sequence[int],
    query_ids: dict[int, int],
    query_vectors: dict[int, Any],
    truth: dict[tuple[str, int], TruthEntry],
    allow_lists: dict[str, AllowList],
    index: Any,
    faiss_module: Any,
    vectors: Any,
    k: int,
    schedule_seed: int,
    progress_queries: int,
    checkpoint_path: Path | None = None,
) -> None:
    block_no = 0
    for filter_no, spec in enumerate(filter_specs):
        allow_list = allow_lists[spec.name]
        completed = 0
        for repeat in range(repeats):
            ordered_queries = list(query_nos)
            random.Random(schedule_seed + filter_no * 1009 + repeat * 104729).shuffle(ordered_queries)
            for query_no in ordered_queries:
                order = balanced_order(ef_values, block_no, schedule_seed)
                query_id = query_ids[int(query_no)]
                for position, ef_search in enumerate(order, start=1):
                    _run_faiss_measurement(
                        raw_rows,
                        phase="calibration",
                        spec=spec,
                        query_no=int(query_no),
                        query_id=query_id,
                        query=query_vectors[int(query_no)],
                        repeat=repeat,
                        schedule_position=position,
                        block_no=block_no,
                        ef_search=int(ef_search),
                        truth_entry=truth[(spec.name, int(query_no))],
                        vectors=vectors,
                        allow_list=allow_list,
                        index=index,
                        faiss_module=faiss_module,
                        k=k,
                    )
                block_no += 1
                completed += 1
                if progress_queries and completed % progress_queries == 0:
                    print(
                        f"calibration filter={spec.name} queries={completed}/{len(query_nos) * repeats}",
                        flush=True,
                    )
        if checkpoint_path is not None:
            write_csv(checkpoint_path, raw_rows)


def run_final(
    raw_rows: list[dict[str, Any]],
    *,
    table: str,
    filter_specs: Sequence[FilterSpec],
    query_nos: Sequence[int],
    repeats: int,
    selected: dict[tuple[str, float], int],
    targets: Sequence[float],
    query_ids: dict[int, int],
    faiss_query_vectors: dict[int, Any],
    sql_query_vectors: dict[int, str],
    truth: dict[tuple[str, int], TruthEntry],
    allow_lists: dict[str, AllowList],
    exact_plan_valid: dict[str, bool],
    cursor: Any,
    index: Any,
    faiss_module: Any,
    vectors: Any,
    k: int,
    schedule_seed: int,
    progress_queries: int,
    checkpoint_path: Path | None = None,
) -> None:
    block_no = 0
    for filter_no, spec in enumerate(filter_specs):
        selected_efs = sorted(
            {
                selected[(spec.name, target)]
                for target in targets
                if (spec.name, target) in selected
            }
        )
        tasks: list[tuple[str, int | None]] = [("sql_first_exact", None)]
        tasks.extend(("faiss_allowlist", ef_search) for ef_search in selected_efs)
        completed = 0
        sql_text = exact_sql(table, spec.predicate, k)
        for repeat in range(repeats):
            ordered_queries = list(query_nos)
            random.Random(schedule_seed + 1_000_003 + filter_no * 1009 + repeat * 104729).shuffle(
                ordered_queries
            )
            for query_no in ordered_queries:
                query_no = int(query_no)
                query_id = query_ids[query_no]
                truth_entry = truth[(spec.name, query_no)]
                truth_ids = truth_entry.ids
                for position, (method, ef_search) in enumerate(
                    balanced_order(tasks, block_no, schedule_seed + 1), start=1
                ):
                    if method == "faiss_allowlist":
                        if ef_search is None:
                            raise RuntimeError("faiss_allowlist final task is missing ef_search")
                        matched_targets = [
                            target
                            for target in targets
                            if selected.get((spec.name, target)) == ef_search
                        ]
                        _run_faiss_measurement(
                            raw_rows,
                            phase="final",
                            spec=spec,
                            query_no=query_no,
                            query_id=query_id,
                            query=faiss_query_vectors[query_no],
                            repeat=repeat,
                            schedule_position=position,
                            block_no=block_no,
                            ef_search=ef_search,
                            truth_entry=truth_entry,
                            vectors=vectors,
                            allow_list=allow_lists[spec.name],
                            index=index,
                            faiss_module=faiss_module,
                            k=k,
                            matched_target_recalls=matched_targets,
                        )
                        continue
                    try:
                        if not exact_plan_valid.get(spec.name, False):
                            raise RuntimeError("EXPLAIN validation failed or used an HNSW index")
                        ids, latency_ms = search_sql_exact(
                            cursor, sql_text, query_id, sql_query_vectors[query_id]
                        )
                        recall = tie_aware_recall_at_k(ids, query_id, vectors, truth_entry, k)
                        error = "" if recall == 1.0 and len(ids) == k else (
                            f"exact_result_mismatch: recall={recall} returned={len(ids)}"
                        )
                        raw_rows.append(
                            measurement_row(
                                phase="final",
                                method="sql_first_exact",
                                spec=spec,
                                query_no=query_no,
                                query_id=query_id,
                                repeat=repeat,
                                schedule_position=position,
                                block_no=block_no,
                                ef_search=NA,
                                result_ids=ids,
                                truth_ids=truth_ids,
                                truth_entry=truth_entry,
                                vectors=vectors,
                                latency_ms=latency_ms,
                                error=error,
                                matched_target_recalls=targets,
                            )
                        )
                    except Exception as exc:
                        raw_rows.append(
                            measurement_row(
                                phase="final",
                                method="sql_first_exact",
                                spec=spec,
                                query_no=query_no,
                                query_id=query_id,
                                repeat=repeat,
                                schedule_position=position,
                                block_no=block_no,
                                ef_search=NA,
                                result_ids=None,
                                truth_ids=truth_ids,
                                latency_ms=NA,
                                error=f"{exc.__class__.__name__}: {exc}",
                                matched_target_recalls=targets,
                            )
                        )
                block_no += 1
                completed += 1
                if progress_queries and completed % progress_queries == 0:
                    print(
                        f"final filter={spec.name} queries={completed}/{len(query_nos) * repeats}",
                        flush=True,
                    )
        if checkpoint_path is not None:
            write_csv(checkpoint_path, raw_rows)


def output_paths(out_dir: Path, tag: str) -> dict[str, Path]:
    prefix = f"amazon10m_matched_recall_baselines_{tag}"
    return {
        "raw": out_dir / f"{prefix}_raw.csv",
        "calibration": out_dir / f"{prefix}_calibration.csv",
        "final": out_dir / f"{prefix}_final.csv",
        "summary": out_dir / f"{prefix}_summary.csv",
        "manifest": out_dir / f"{prefix}_manifest.json",
    }


def normalized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    import faiss
    import numpy as np
    import psycopg

    try:
        from .common_pg import pg_config_from_env
    except ImportError:
        from common_pg import pg_config_from_env

    paths = output_paths(args.out_dir, args.tag)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {existing[0]}")
    if args.overwrite:
        for path in existing:
            path.unlink()

    calibration_query_nos = list(
        range(args.calibration_query_offset, args.calibration_query_offset + args.calibration_queries)
    )
    final_query_nos = list(range(args.final_query_offset, args.final_query_offset + args.final_queries))
    expected_calibration = list(range(
        DEFAULT_CALIBRATION_QUERY_OFFSET,
        DEFAULT_CALIBRATION_QUERY_OFFSET + DEFAULT_CALIBRATION_QUERIES,
    ))
    expected_final = list(range(
        DEFAULT_FINAL_QUERY_OFFSET,
        DEFAULT_FINAL_QUERY_OFFSET + DEFAULT_FINAL_QUERIES,
    ))
    if calibration_query_nos != expected_calibration or final_query_nos != expected_final:
        raise ValueError("formal matched-recall split requires calibration q20..q99 and held-out final q100..q199")
    targets = parse_targets(args.target_recalls)
    ef_values = parse_int_csv(args.ef_search_values)
    specs = load_filter_specs(args.filters_csv, set(args.filter_names) or None)
    truth, query_ids = load_truth(
        args.truth_csv, specs, calibration_query_nos, final_query_nos, args.k
    )
    vectors, vector_rows, dimensions = read_fbin_memmap(args.fbin, args.rows)
    if vector_rows != args.rows:
        raise ValueError(f"fbin rows={vector_rows}, expected --rows={args.rows}")
    if any(query_id < 0 or query_id >= vector_rows for query_id in query_ids.values()):
        raise ValueError("truth query ID is outside the fbin row range")
    index = faiss.read_index(str(args.faiss_index))
    index_meta = faiss_index_metadata(index, faiss, vector_rows, dimensions)
    faiss.omp_set_num_threads(args.faiss_threads)
    faiss_query_vectors = {
        query_no: np.ascontiguousarray(vectors[query_id], dtype=np.float32)
        for query_no, query_id in query_ids.items()
    }
    filters_identity = file_identity(args.filters_csv, hash_contents=True)
    truth_identity = file_identity(args.truth_csv, hash_contents=True)
    fbin_identity = file_identity(args.fbin, hash_contents=True)
    faiss_identity = file_identity(args.faiss_index, hash_contents=True)
    runner_identity = file_identity(Path(__file__), hash_contents=True)

    manifest: dict[str, Any] = {
        "artifact": "amazon10m_matched_recall_baselines",
        "artifact_valid": False,
        "status": "running",
        "validation_errors": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": normalized_args(args),
        "inputs": {
            "filters": filters_identity,
            "truth": truth_identity,
            "fbin": fbin_identity,
            "faiss_index": faiss_identity,
            "runner": runner_identity,
            "postgres_table": args.table,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "filter_names": [spec.name for spec in specs],
        "run_contract": {
            key: value
            for key, value in normalized_args(args).items()
            if key not in {"filter_names", "tag", "out_dir", "overwrite", "progress_queries"}
        },
        "source_db": {"table": args.table},
        "source_hashes": {
            "faiss": faiss_identity["sha256"],
            "fbin": fbin_identity["sha256"],
            "truth": truth_identity["sha256"],
        },
        "query_splits": {
            "calibration_query_nos": calibration_query_nos,
            "final_query_nos": final_query_nos,
            "reserved_query_nos": list(range(20)),
            "query_no_overlap": False,
            "query_id_overlap": False,
        },
        "repeats": {"calibration": args.calibration_repeats, "final": args.final_repeats},
        "target_recalls": targets,
        "ef_ladder": ef_values,
        "faiss_index": index_meta,
        "environment": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "psycopg": getattr(psycopg, "__version__", "unknown"),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "psycopg": getattr(psycopg, "__version__", "unknown"),
            "measurement_runner_sha256": runner_identity["sha256"],
        },
        "execution": {
            "parallel_claim": False,
            "faiss_openmp_threads": args.faiss_threads,
            "postgres_max_parallel_workers_per_gather": 0,
            "config_order": "deterministic_balanced_interleaved_rotation",
            "schedule_seed": args.schedule_seed,
            "latency": "search-only; query-vector prefetch, allow-list build, EXPLAIN, and output I/O excluded",
            "ground_truth_latency_used_as_baseline": False,
            "allowlist_cost": "one real SQL stream and one bitmap construction per predicate",
        },
        "bootstrap": {
            "unit": "query cluster after averaging repeats",
            "target_selection": TARGET_SELECTION_RULE,
            "ci_lcb": "reported_only",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
    }
    write_json(paths["manifest"], manifest)

    raw_rows: list[dict[str, Any]] = []
    allow_lists: dict[str, AllowList] = {}
    exact_plan_valid: dict[str, bool] = {}
    explain_audit: dict[str, Any] = {}
    warmup_errors: list[str] = []
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SET max_parallel_workers_per_gather = 0")
            cursor.execute("SET jit = off")
            cursor.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
            cursor.execute(f"SELECT count(*), min(id), max(id) FROM {args.table}")
            table_rows, min_id, max_id = (int(value) for value in cursor.fetchone())
            if (table_rows, min_id, max_id) != (vector_rows, 0, vector_rows - 1):
                raise ValueError(
                    "PostgreSQL/Faiss ID-space mismatch: "
                    f"table=({table_rows}, {min_id}, {max_id}) faiss=(0, {vector_rows - 1})"
                )
            cursor.execute(
                "SELECT indexrelid::regclass::text "
                "FROM pg_index JOIN pg_class ON pg_class.oid=indexrelid "
                "JOIN pg_am ON pg_am.oid=pg_class.relam "
                "WHERE indrelid=%s::regclass AND pg_am.amname='hnsw'",
                (args.table,),
            )
            hnsw_indexes = [str(row[0]) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT current_setting('server_version'), "
                "COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), ''), "
                "c.oid::bigint, c.relfilenode::bigint "
                "FROM pg_class AS c WHERE c.oid=%s::regclass",
                (args.table,),
            )
            postgres_version, vector_version, table_oid, table_relfilenode = cursor.fetchone()
            sql_query_vectors = prefetch_sql_query_vectors(cursor, args.table, query_ids.values())
            manifest["postgres"] = {
                "server_version": postgres_version,
                "vector_extension_version": vector_version,
                "table_oid": int(table_oid),
                "table_relfilenode": int(table_relfilenode),
                "rows": table_rows,
                "min_id": min_id,
                "max_id": max_id,
                "hnsw_indexes": hnsw_indexes,
            }

            explain_query_id = query_ids[calibration_query_nos[0]]
            for spec in specs:
                try:
                    explain_audit[spec.name] = explain_exact_plan(
                        cursor,
                        args.table,
                        spec,
                        sql_query_vectors[explain_query_id],
                        args.k,
                        hnsw_indexes,
                    )
                    exact_plan_valid[spec.name] = True
                except Exception as exc:
                    exact_plan_valid[spec.name] = False
                    explain_audit[spec.name] = {
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "hnsw_indexes": hnsw_indexes,
                    }

            for position, spec in enumerate(specs, start=1):
                allow_list = build_allow_list(
                    conn,
                    faiss,
                    args.table,
                    spec,
                    vector_rows,
                    args.allowlist_fetch_rows,
                )
                allow_lists[spec.name] = allow_list
                raw_rows.append(setup_row(spec, allow_list))
                print(
                    f"allow-list {position}/{len(specs)} filter={spec.name} rows={allow_list.rows} "
                    f"ms={allow_list.build_ms:.2f} bytes={allow_list.bitmap_bytes} "
                    f"valid={allow_list.valid}",
                    flush=True,
                )
                write_csv(paths["raw"], raw_rows)

            if args.warmup_queries:
                warm_query_nos = calibration_query_nos[: args.warmup_queries]
                for spec in specs:
                    allow_list = allow_lists[spec.name]
                    if not allow_list.valid:
                        continue
                    for query_no in warm_query_nos:
                        for ef_search in ef_values:
                            try:
                                search_faiss(
                                    index,
                                    faiss,
                                    faiss_query_vectors[query_no],
                                    allow_list.selector,
                                    ef_search,
                                    args.k,
                                    query_id=query_ids[query_no],
                                )
                            except Exception as exc:
                                warmup_errors.append(
                                    f"calibration|{spec.name}|q{query_no}|ef{ef_search}|"
                                    f"{exc.__class__.__name__}: {exc}"
                                )

            run_calibration(
                raw_rows,
                filter_specs=specs,
                query_nos=calibration_query_nos,
                repeats=args.calibration_repeats,
                ef_values=ef_values,
                query_ids=query_ids,
                query_vectors=faiss_query_vectors,
                truth=truth,
                allow_lists=allow_lists,
                index=index,
                faiss_module=faiss,
                vectors=vectors,
                k=args.k,
                schedule_seed=args.schedule_seed,
                progress_queries=args.progress_queries,
                checkpoint_path=paths["raw"],
            )
            write_csv(paths["raw"], raw_rows)
            calibration_rows, selected = calibration_table(
                raw_rows,
                specs,
                ef_values,
                targets,
                calibration_query_nos,
                args.calibration_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
                allow_lists,
            )
            write_csv(paths["calibration"], calibration_rows)
            calibration_outcomes = calibration_outcomes_from_rows(calibration_rows)

            if args.warmup_queries:
                for spec in specs:
                    for query_no in final_query_nos[: args.warmup_queries]:
                        if exact_plan_valid[spec.name]:
                            try:
                                search_sql_exact(
                                    cursor,
                                    exact_sql(args.table, spec.predicate, args.k),
                                    query_ids[query_no],
                                    sql_query_vectors[query_ids[query_no]],
                                )
                            except Exception as exc:
                                warmup_errors.append(
                                    f"final|sql_first_exact|{spec.name}|q{query_no}|"
                                    f"{exc.__class__.__name__}: {exc}"
                                )
                        for ef_search in sorted(
                            {
                                selected[(spec.name, target)]
                                for target in targets
                                if (spec.name, target) in selected
                            }
                        ):
                            try:
                                search_faiss(
                                    index,
                                    faiss,
                                    faiss_query_vectors[query_no],
                                    allow_lists[spec.name].selector,
                                    ef_search,
                                    args.k,
                                    query_id=query_ids[query_no],
                                )
                            except Exception as exc:
                                warmup_errors.append(
                                    f"final|faiss_allowlist|{spec.name}|q{query_no}|ef{ef_search}|"
                                    f"{exc.__class__.__name__}: {exc}"
                                )

            run_final(
                raw_rows,
                table=args.table,
                filter_specs=specs,
                query_nos=final_query_nos,
                repeats=args.final_repeats,
                selected=selected,
                targets=targets,
                query_ids=query_ids,
                faiss_query_vectors=faiss_query_vectors,
                sql_query_vectors=sql_query_vectors,
                truth=truth,
                allow_lists=allow_lists,
                exact_plan_valid=exact_plan_valid,
                cursor=cursor,
                index=index,
                faiss_module=faiss,
                vectors=vectors,
                k=args.k,
                schedule_seed=args.schedule_seed,
                progress_queries=args.progress_queries,
                checkpoint_path=paths["raw"],
            )
            write_csv(paths["raw"], raw_rows)
            final_rows = [row for row in raw_rows if row.get("phase") == "final"]
            write_csv(paths["final"], final_rows)
            summary_rows = final_summary_table(
                final_rows,
                specs,
                targets,
                selected,
                final_query_nos,
                args.final_repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
                allow_lists,
                calibration_outcomes,
            )
            write_csv(paths["summary"], summary_rows)

        validation_errors = artifact_validation_errors(
            calibration_rows, summary_rows, specs, ef_values, targets
        )
        manifest["artifact_valid"] = not validation_errors
        manifest["status"] = "complete" if manifest["artifact_valid"] else "invalid"
        manifest["validation_errors"] = validation_errors
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["sql_first_exact_explain"] = explain_audit
        manifest["allowlists"] = {
            name: {
                "rows": value.rows,
                "build_ms": value.build_ms,
                "bitmap_bytes": value.bitmap_bytes,
                "valid": value.valid,
                "error": value.error,
            }
            for name, value in allow_lists.items()
        }
        manifest["selected_faiss_ef_search"] = {
            f"{filter_name}|{target:.2f}": ef_search
            for (filter_name, target), ef_search in selected.items()
        }
        manifest["summary_valid_rows"] = sum(row["status"] == "valid" for row in summary_rows)
        manifest["matched_recall_valid_rows"] = sum(
            bool(row["matched_recall_comparison_valid"]) for row in summary_rows
        )
        manifest["summary_rows"] = len(summary_rows)
        manifest["row_counts"] = {
            "raw": len(raw_rows),
            "calibration": len(calibration_rows),
            "final": len(final_rows),
            "summary": len(summary_rows),
        }
        manifest["outputs"] = {
            **{name: str(path) for name, path in paths.items()},
            "raw": {"path": str(paths["raw"]), "rows": len(raw_rows)},
            "calibration": {"path": str(paths["calibration"]), "rows": len(calibration_rows)},
            "final": {"path": str(paths["final"]), "rows": len(final_rows)},
            "summary": {"path": str(paths["summary"]), "rows": len(summary_rows)},
            "manifest": str(paths["manifest"]),
        }
        manifest["warmup_errors"] = warmup_errors
        write_json(paths["manifest"], manifest)
        return paths
    except Exception as exc:
        manifest["artifact_valid"] = False
        manifest["status"] = "invalid"
        manifest["validation_errors"] = [f"fatal_error: {exc.__class__.__name__}: {exc}"]
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
        manifest["sql_first_exact_explain"] = explain_audit
        manifest["warmup_errors"] = warmup_errors
        if raw_rows:
            write_csv(paths["raw"], raw_rows)
        write_json(paths["manifest"], manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched-recall Amazon-10M sql-first exact and Faiss HNSW allow-list baselines."
        )
    )
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--faiss-index", type=Path, default=DEFAULT_FAISS_INDEX)
    parser.add_argument("--table", type=validate_table_name, default=DEFAULT_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tag", default="20260718")
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--rows", type=positive_int, default=10_000_000)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--ef-search-values", default=",".join(str(value) for value in DEFAULT_EF_SEARCH))
    parser.add_argument("--target-recalls", default=",".join(str(value) for value in DEFAULT_TARGETS))
    parser.add_argument("--calibration-query-offset", type=nonnegative_int, default=DEFAULT_CALIBRATION_QUERY_OFFSET)
    parser.add_argument("--calibration-queries", type=positive_int, default=DEFAULT_CALIBRATION_QUERIES)
    parser.add_argument("--calibration-repeats", type=positive_int, default=2)
    parser.add_argument("--final-query-offset", type=nonnegative_int, default=DEFAULT_FINAL_QUERY_OFFSET)
    parser.add_argument("--final-queries", type=positive_int, default=DEFAULT_FINAL_QUERIES)
    parser.add_argument("--final-repeats", type=positive_int, default=5)
    parser.add_argument("--bootstrap-samples", type=positive_int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--faiss-threads", type=positive_int, default=1)
    parser.add_argument("--allowlist-fetch-rows", type=positive_int, default=100_000)
    parser.add_argument("--warmup-queries", type=nonnegative_int, default=1)
    parser.add_argument("--progress-queries", type=nonnegative_int, default=25)
    parser.add_argument("--statement-timeout-ms", type=nonnegative_int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run(args)
    for name, path in paths.items():
        print(f"wrote {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
