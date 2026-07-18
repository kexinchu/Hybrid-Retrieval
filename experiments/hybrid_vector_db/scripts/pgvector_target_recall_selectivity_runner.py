from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import psycopg

try:
    from .common_pg import pg_config_from_env
    from .pgvector_design1_design2_design3_selectivity_benchmark import (
        candidate_validity_index_predicate_matches,
        candidate_validity_sha256,
        effective_candidate_validity_predicate,
        ensure_functions,
        ensure_tracking,
        normalize_cpu_list,
        require_d2_graph_proof,
        stable_d2_graph_proof,
        validate_candidate_validity_predicate,
        validate_d2_graph_proof,
    )
except ImportError:
    from common_pg import pg_config_from_env
    from pgvector_design1_design2_design3_selectivity_benchmark import (
        candidate_validity_index_predicate_matches,
        candidate_validity_sha256,
        effective_candidate_validity_predicate,
        ensure_functions,
        ensure_tracking,
        normalize_cpu_list,
        require_d2_graph_proof,
        stable_d2_graph_proof,
        validate_candidate_validity_predicate,
        validate_d2_graph_proof,
    )


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
FILTER_ORDER = [
    "popular_ge1000",
    "popular_ge1340",
    "popular_ge1780",
    "popular_ge2428",
    "popular_ge3284",
    "popular_ge4559",
    "price_10_to_20",
    "popular_ge10066",
    "rating5_price_le10",
    "long_review_ge500",
    "grocery_rating5",
    "grocery_helpful",
    "helpful_ge20",
    "grocery_long500",
]
DEFAULT_MODES = [
    "original",
    "design1_bloom",
    "design1_bloom_bfs_layout",
    "design1_bloom_bfs_layout_d3",
]
DEFAULT_INSERTION_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
DEFAULT_INSERTION_INDEX = "public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx"
DEFAULT_BFS_TABLE = DEFAULT_INSERTION_TABLE
DEFAULT_BFS_INDEX = "public.amazon_grocery_reviews_10m_pgvector_hnsw_bfs_clone_idx"
DEFAULT_FILTERS_CSV = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv"
DEFAULT_TRUTH_CSV = RESULTS / "amazon_selectivity14_exact_truth_q200_formal.csv"
DENSE_12_EF_SEARCH = "250,500,750,1000,1500,2000,3000,4000,5000,7000,8500,10000"
FORMAL_TARGETS = (0.90, 0.95, 0.99)
TIE_AWARE_RECALL_CONTRACT = "distance_squared_threshold_tie_aware_v1"
SQLENS_V11_BUILD_PREFIX = "sqlens-v11-"
SQLENS_MIN_PROFILE_SEMANTICS = 7.0
SQLENS_PROFILE_REQUIRED_FIELDS = (
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
SQLENS_TRAVERSAL_PROFILE_REQUIRED_FIELDS = (
    "final_path",
    "planner_proof_attempted",
    "planner_proof_succeeded",
    "traversal_guidance_scope",
    "graph_expansion_pruned",
    "distance_computations_pruned",
    "pre_distance_membership_checks",
    "pre_distance_membership_matches",
    "pre_distance_membership_misses",
    "distance_computations_avoided",
    "neighbor_expansion_guidance_checks",
    "traversal_guided_admissions",
    "traversal_guided_suppressions",
    "traversal_heap_tids_suppressed",
    "guided_expanded_nodes",
    "guided_phase_distance_computations",
    "stock_bypass_requests",
    "fallback_requests",
    "traversal_estimated_skip_rate_valid",
    "traversal_estimated_skip_rate",
)


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
            f"ef{self.ef_search}_target{self.guided_collect_target}_"
            f"max{self.max_scan_tuples}_mem{mem}_{self.iterative_scan}"
        )


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_targets(value: str) -> list[float]:
    targets = sorted(set(parse_floats(value)))
    if not targets or any(target <= 0 or target > 1 for target in targets):
        raise argparse.ArgumentTypeError("recall targets must be in (0, 1]")
    return targets


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def query_means(rows: list[dict[str, str]], field: str) -> list[float]:
    by_query: dict[str, list[float]] = {}
    for row in rows:
        by_query.setdefault(row["query_no"], []).append(float(row[field]))
    return [statistics.fmean(values) for _, values in sorted(by_query.items())]


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    means = [statistics.fmean(rng.choices(values, k=size)) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_mean_bounds(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    size = len(values)
    means = [statistics.fmean(rng.choices(values, k=size)) for _ in range(samples)]
    return percentile(means, 0.05), percentile(means, 0.025), percentile(means, 0.975)


def paired_query_latency_means(
    path: Path,
    filter_name: str,
    mode: str,
) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row.get("filter_name") == filter_name
            and row.get("mode") == mode
            and not row.get("error")
        ]
    by_query: dict[str, list[float]] = {}
    for row in rows:
        by_query.setdefault(row["query_no"], []).append(float(row["end_to_end_ms"]))
    return {query_no: statistics.fmean(values) for query_no, values in by_query.items()}


def paired_speedup_ci(
    stock_path: Path,
    stock_mode: str,
    method_path: Path,
    method_mode: str,
    filter_name: str,
    samples: int,
    seed: int,
) -> tuple[int, float, float]:
    stats = paired_comparison_stats(
        stock_path,
        stock_mode,
        method_path,
        method_mode,
        filter_name,
        samples,
        seed,
    )
    return (
        int(stats.get("paired_queries", 0)),
        float(stats.get("speedup_ci95_low", 0.0)),
        float(stats.get("speedup_ci95_high", 0.0)),
    )


def paired_comparison_stats(
    stock_path: Path,
    stock_mode: str,
    method_path: Path,
    method_mode: str,
    filter_name: str,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    def blocks(path: Path, mode: str) -> dict[tuple[str, str], tuple[float, float]] | None:
        with path.open(newline="", encoding="utf-8") as f:
            rows = [
                row
                for row in csv.DictReader(f)
                if row.get("filter_name") == filter_name
                and row.get("mode") == mode
                and not row.get("error")
            ]
        values: dict[tuple[str, str], tuple[float, float]] = {}
        for row in rows:
            key = (row["query_no"], row["repeat"])
            if key in values:
                return None
            values[key] = float(row["end_to_end_ms"]), float(row["recall"])
        return values

    stock_blocks = blocks(stock_path, stock_mode)
    method_blocks = blocks(method_path, method_mode)
    if not stock_blocks or not method_blocks or set(stock_blocks) != set(method_blocks):
        return {}
    query_nos = sorted({query_no for query_no, _ in stock_blocks})
    repeats_by_query = {
        query_no: sorted(repeat for candidate, repeat in stock_blocks if candidate == query_no)
        for query_no in query_nos
    }
    if len({tuple(repeats) for repeats in repeats_by_query.values()}) != 1:
        return {}
    paired_repeats = len(next(iter(repeats_by_query.values())))
    stock_latency = {
        query_no: statistics.fmean(stock_blocks[(query_no, repeat)][0] for repeat in repeats)
        for query_no, repeats in repeats_by_query.items()
    }
    method_latency = {
        query_no: statistics.fmean(method_blocks[(query_no, repeat)][0] for repeat in repeats)
        for query_no, repeats in repeats_by_query.items()
    }
    stock_recall = {
        query_no: statistics.fmean(stock_blocks[(query_no, repeat)][1] for repeat in repeats)
        for query_no, repeats in repeats_by_query.items()
    }
    method_recall = {
        query_no: statistics.fmean(method_blocks[(query_no, repeat)][1] for repeat in repeats)
        for query_no, repeats in repeats_by_query.items()
    }
    ratios: list[float] = []
    savings: list[float] = []
    stock_recalls: list[float] = []
    method_recalls: list[float] = []
    recall_deltas: list[float] = []
    rng = random.Random(seed)
    iterations = max(1, samples)
    for _ in range(iterations):
        sampled = rng.choices(query_nos, k=len(query_nos)) if len(query_nos) > 1 else query_nos
        stock_mean = statistics.fmean(stock_latency[query_no] for query_no in sampled)
        method_mean = statistics.fmean(method_latency[query_no] for query_no in sampled)
        stock_recall_mean = statistics.fmean(stock_recall[query_no] for query_no in sampled)
        method_recall_mean = statistics.fmean(method_recall[query_no] for query_no in sampled)
        ratios.append(stock_mean / method_mean if method_mean > 0 else 0.0)
        savings.append(stock_mean - method_mean)
        stock_recalls.append(stock_recall_mean)
        method_recalls.append(method_recall_mean)
        recall_deltas.append(method_recall_mean - stock_recall_mean)
    return {
        "paired_queries": len(query_nos),
        "paired_repeats": paired_repeats,
        "paired_samples": len(stock_blocks),
        "paired_latency_saving_mean_ms": statistics.fmean(
            stock_latency[query_no] - method_latency[query_no] for query_no in query_nos
        ),
        "paired_latency_saving_ci95_low_ms": percentile(savings, 0.025),
        "paired_latency_saving_ci95_high_ms": percentile(savings, 0.975),
        "speedup_ci95_low": percentile(ratios, 0.025),
        "speedup_ci95_high": percentile(ratios, 0.975),
        "stock_recall_paired_mean": statistics.fmean(stock_recall.values()),
        "stock_recall_paired_ci95_low": percentile(stock_recalls, 0.025),
        "stock_recall_paired_ci95_high": percentile(stock_recalls, 0.975),
        "method_recall_paired_mean": statistics.fmean(method_recall.values()),
        "method_recall_paired_ci95_low": percentile(method_recalls, 0.025),
        "method_recall_paired_ci95_high": percentile(method_recalls, 0.975),
        "recall_delta_paired_mean": statistics.fmean(
            method_recall[query_no] - stock_recall[query_no] for query_no in query_nos
        ),
        "recall_delta_paired_ci95_low": percentile(recall_deltas, 0.025),
        "recall_delta_paired_ci95_high": percentile(recall_deltas, 0.975),
    }


def run_command(cmd: list[str], log: Path | None = None) -> float:
    print(shlex.join(cmd), flush=True)
    start = time.perf_counter()
    if log is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            f.write("$ " + shlex.join(cmd) + "\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
    return (time.perf_counter() - start) * 1000.0


def csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def required_csv_bool(value: object, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"{field} must be an explicit boolean")


def validate_tie_aware_raw_row(
    row: dict[str, str],
    expected_truth_self_excluded: bool = True,
) -> float:
    if row.get("recall_contract") != TIE_AWARE_RECALL_CONTRACT:
        raise ValueError("raw row does not use the tie-aware recall contract")
    if required_csv_bool(row.get("truth_self_excluded", ""), "truth_self_excluded") != expected_truth_self_excluded:
        raise ValueError(
            "raw row truth self_excluded does not match the expected query contract"
        )
    if not row.get("sqlens_build_id", "").startswith(SQLENS_V11_BUILD_PREFIX):
        raise ValueError("raw row is missing the exact SQLens build ID")
    vector_sha = row.get("vector_so_sha256", "")
    if len(vector_sha) != 64 or any(char not in "0123456789abcdef" for char in vector_sha):
        raise ValueError("raw row is missing the server-side vector.so SHA256")
    if int(row.get("backend_pid") or 0) <= 0 or not row.get("backend_cpu_observed"):
        raise ValueError("raw row is missing PostgreSQL backend CPU provenance")
    if row.get("backend_cpu_requested") and not csv_bool(
        row.get("backend_cpu_exact_match", "")
    ):
        raise ValueError("raw row failed the requested backend CPU affinity gate")
    if csv_bool(row.get("backend_cpu_pinning_attempted_by_runner", "")):
        raise ValueError("raw row claims unsafe in-runner backend pinning")
    if row.get("mode") == "design1_bloom_bfs_layout_d3" and not row.get("error"):
        phase = row.get("d3_phase")
        if phase not in {"cold", "admission", "warm"}:
            raise ValueError("D3 raw row is missing an explicit lifecycle phase")
        if phase == "admission" and not csv_bool(row.get("d3_admitted_after", "")):
            raise ValueError("D3 admission row lacks post-request admission proof")
        if phase == "warm" and (
            not csv_bool(row.get("d3_admitted_before", ""))
            or not csv_bool(row.get("d3_active_guidance_reused", ""))
        ):
            raise ValueError("D3 warm row lacks pre-request admission/reuse proof")
    if row.get("guidance_filter_strategy") == "traversal_guided" and not row.get("error"):
        k = int(row.get("k") or 0)
        expected_contract = (
            "limit_k_plus_1_client_remove_query_id"
            if expected_truth_self_excluded
            else "none_external_query_source"
        )
        if row.get("self_exclusion_contract") != expected_contract:
            raise ValueError("formal traversal raw row has the wrong self-exclusion contract")
        expected_limit = k + 1 if expected_truth_self_excluded else k
        if int(row.get("scan_limit") or 0) != expected_limit:
            raise ValueError("formal traversal raw row has the wrong measured LIMIT")
        if int(row.get("returned") or 0) > k:
            raise ValueError("formal traversal raw row returned more than k rows after self removal")
        if int(row.get("raw_returned_before_self_exclusion") or 0) > expected_limit:
            raise ValueError("formal traversal raw row fetched more than its measured LIMIT")
        ids = [value for value in row.get("ids", "").split(",") if value]
        if expected_truth_self_excluded and str(row.get("query_id", "")) in ids:
            raise ValueError("formal traversal raw row still contains the query row")
        mode = row.get("mode")
        guidance_enabled = csv_bool(row.get("guidance_enabled", ""))
        if mode in {"design1_bloom", "design1_bloom_bfs_layout"} and not guidance_enabled:
            raise ValueError("formal D1 raw row did not enable traversal guidance")
        if guidance_enabled:
            if not csv_bool(row.get("guidance_scan_verified", "")):
                raise ValueError("formal traversal raw row was not scan-verified")
            if row.get("final_path") != "guided":
                raise ValueError("formal traversal raw row did not finish on the guided path")
            if not csv_bool(row.get("planner_proof_succeeded", "")):
                raise ValueError("formal traversal raw row lacks a successful planner proof")
            if int(row.get("stock_bypass_requests") or 0) != 0:
                raise ValueError("formal traversal raw row used stock bypass")
            if int(row.get("fallback_requests") or 0) != 0:
                raise ValueError("formal traversal raw row used fresh-stock fallback")
            if row.get("traversal_guidance_scope") != "candidate_admission_and_validation":
                raise ValueError("formal traversal raw row has the wrong guidance scope")
            if csv_bool(row.get("graph_expansion_pruned", "")):
                raise ValueError("formal candidate admission claimed graph-expansion pruning")
            if csv_bool(row.get("distance_computations_pruned", "")):
                raise ValueError("formal candidate admission claimed distance pruning")
            if int(row.get("pre_distance_membership_checks") or 0) != 0:
                raise ValueError("formal candidate admission recorded pre-distance checks")
            if int(row.get("distance_computations_avoided") or 0) != 0:
                raise ValueError("formal candidate admission recorded avoided distance work")
            neighbor_checks = int(row.get("neighbor_expansion_guidance_checks") or 0)
            guided_admissions = int(row.get("traversal_guided_admissions") or 0)
            guided_suppressions = int(row.get("traversal_guided_suppressions") or 0)
            heap_suppressions = int(row.get("traversal_heap_tids_suppressed") or 0)
            if neighbor_checks <= 0 or guided_admissions <= 0 or guided_suppressions <= 0:
                raise ValueError("formal candidate admission has empty traversal evidence")
            if heap_suppressions < guided_suppressions:
                raise ValueError(
                    "formal heap-TID suppression count is smaller than suppressed HNSW elements"
                )
            if not csv_bool(row.get("traversal_estimated_skip_rate_valid", "")):
                raise ValueError("formal traversal raw row lacks a valid skip-rate estimate")
            estimated_skip_rate = float(row.get("traversal_estimated_skip_rate") or -1)
            if not math.isfinite(estimated_skip_rate) or not 0.0 <= estimated_skip_rate <= 1.0:
                raise ValueError("formal traversal raw row has an invalid skip-rate estimate")
    for field in (
        "truth_filtered_rows",
        "truth_kth_distance_sq",
        "truth_tie_tolerance",
        "result_distances",
        "k",
    ):
        if field not in row:
            raise ValueError(f"raw row is missing {field}")
    distances = json.loads(row["result_distances"])
    if not isinstance(distances, list) or any(not isinstance(value, (int, float)) for value in distances):
        raise ValueError("result_distances must be a JSON number array")
    if len(distances) != int(row.get("returned", len(distances))):
        raise ValueError("result_distances length does not match returned")
    k = int(row["k"])
    filtered_rows = int(row["truth_filtered_rows"])
    denominator = min(k, filtered_rows)
    if denominator == 0:
        computed_recall = 0.0
    else:
        kth_distance_sq = float(row["truth_kth_distance_sq"])
        tie_tolerance = float(row["truth_tie_tolerance"])
        if tie_tolerance < 0:
            raise ValueError("raw tie_tolerance must be non-negative")
        credit = min(
            denominator,
            k,
            sum(
                float(distance) * float(distance) <= kth_distance_sq + tie_tolerance
                for distance in distances[:k]
            ),
        )
        computed_recall = credit / denominator
    recorded_recall = float(row["recall"])
    if not row.get("error") and abs(recorded_recall - computed_recall) > 1e-12:
        raise ValueError(
            f"raw recall {recorded_recall} does not match tie-aware recall {computed_recall}"
        )
    return computed_recall if not row.get("error") else 0.0


def summarize_raw(
    path: Path,
    bootstrap_samples: int = 2000,
    seed: int = 20260718,
    expected_queries: int | None = None,
    expected_repeats: int | None = None,
    expected_truth_self_excluded: bool = True,
) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["recall"] = str(
            validate_tie_aware_raw_row(row, expected_truth_self_excluded)
        )
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["filter_name"], row["mode"]), []).append(row)

    out: list[dict[str, object]] = []
    for group_no, ((filter_name, mode), items) in enumerate(sorted(groups.items())):
        ok = [row for row in items if not row.get("error")]
        recalls = [float(row["recall"]) for row in ok]
        latencies = [float(row["end_to_end_ms"]) for row in ok]
        def mean_field(name: str) -> float:
            values = [float(row.get(name, 0) or 0) for row in ok]
            return statistics.fmean(values) if values else 0.0

        latency_query_means = query_means(ok, "end_to_end_ms")
        recall_query_means = query_means(ok, "recall")
        recall_lcb, recall_ci_low, recall_ci_high = bootstrap_mean_bounds(
            recall_query_means,
            bootstrap_samples,
            seed + 100000 + group_no,
        )
        repeats_by_query: dict[str, set[str]] = {}
        for row in ok:
            repeats_by_query.setdefault(row["query_no"], set()).add(row["repeat"])
        observed_repeats = max((len(values) for values in repeats_by_query.values()), default=0)
        required_repeats = expected_repeats if expected_repeats is not None else observed_repeats
        complete_queries = sum(len(values) == required_repeats for values in repeats_by_query.values())
        required_queries = expected_queries if expected_queries is not None else len(repeats_by_query)
        rows_complete = (
            len(items) == required_queries * required_repeats
            and len(ok) == len(items)
            and complete_queries == required_queries
        )
        ci_low, ci_high = bootstrap_mean_ci(
            latency_query_means,
            bootstrap_samples,
            seed + group_no,
        )
        out.append(
            {
                "filter_name": filter_name,
                "mode": mode,
                "queries": len(latency_query_means),
                "samples": len(items),
                "ok": len(ok),
                "errors": len(items) - len(ok),
                "recall_mean": statistics.fmean(recalls) if recalls else 0.0,
                "recall_min_query_mean": min(recall_query_means) if recall_query_means else 0.0,
                "recall_lcb95": recall_lcb,
                "recall_ci95_low": recall_ci_low,
                "recall_ci95_high": recall_ci_high,
                "expected_queries": required_queries,
                "expected_repeats": required_repeats,
                "complete_queries": complete_queries,
                "rows_complete": rows_complete,
                "recall_contract": TIE_AWARE_RECALL_CONTRACT,
                "truth_self_excluded": all(
                    required_csv_bool(row.get("truth_self_excluded", ""), "truth_self_excluded")
                    == expected_truth_self_excluded
                    for row in items
                ),
                "tie_aware_rows": len(items),
                "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
                "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
                "latency_p95_ms": percentile(latencies, 0.95),
                "latency_p99_ms": percentile(latencies, 0.99),
                "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "latency_query_mean_ci95_low_ms": ci_low,
                "latency_query_mean_ci95_high_ms": ci_high,
                "vector_search_mean_ms": mean_field("vector_search_ms"),
                "visited_tuples_mean": mean_field("visited_tuples"),
                "returned_tuples_mean": mean_field("returned_tuples"),
                "distance_compute_count_mean": mean_field("distance_compute_count"),
                "traversal_expanded_nodes_mean": mean_field("traversal_expanded_nodes"),
                "traversal_neighbors_examined_mean": mean_field("traversal_neighbors_examined"),
                "traversal_guidance_checks_mean": mean_field("traversal_guidance_checks"),
                "traversal_guidance_matches_mean": mean_field("traversal_guidance_matches"),
                "traversal_matching_expanded_mean": mean_field("traversal_matching_expanded"),
                "traversal_bridge_expanded_mean": mean_field("traversal_bridge_expanded"),
                "traversal_candidate_admissions_mean": mean_field("traversal_candidate_admissions"),
                "traversal_result_admissions_mean": mean_field("traversal_result_admissions"),
                "traversal_guided_admissions_mean": mean_field("traversal_guided_admissions"),
                "traversal_guided_suppressions_mean": mean_field("traversal_guided_suppressions"),
                "traversal_heap_tids_suppressed_mean": mean_field("traversal_heap_tids_suppressed"),
                "traversal_stop_deferrals_mean": mean_field("traversal_stop_deferrals"),
                "traversal_discarded_pushes_mean": mean_field("traversal_discarded_pushes"),
                "traversal_discarded_pops_mean": mean_field("traversal_discarded_pops"),
                "traversal_initial_batches_mean": mean_field("traversal_initial_batches"),
                "traversal_resume_batches_mean": mean_field("traversal_resume_batches"),
                "traversal_strict_order_drops_mean": mean_field("traversal_strict_order_drops"),
                "pre_distance_membership_checks_mean": mean_field("pre_distance_membership_checks"),
                "pre_distance_membership_misses_mean": mean_field("pre_distance_membership_misses"),
                "distance_computations_avoided_mean": mean_field("distance_computations_avoided"),
                "guided_expanded_nodes_mean": mean_field("guided_expanded_nodes"),
                "guided_phase_distance_computations_mean": mean_field("guided_phase_distance_computations"),
                "stock_bypass_requests_mean": mean_field("stock_bypass_requests"),
                "fallback_requests_mean": mean_field("fallback_requests"),
                "guided_final_path_rate": (
                    statistics.fmean(1.0 if row.get("final_path") == "guided" else 0.0 for row in ok)
                    if ok
                    else 0.0
                ),
                "planner_proof_success_rate": (
                    statistics.fmean(
                        1.0 if csv_bool(row.get("planner_proof_succeeded", "")) else 0.0
                        for row in ok
                    )
                    if ok
                    else 0.0
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def output_artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "row_count": csv_row_count(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path),
    }


def plan_evidence_path(raw: Path) -> Path:
    return raw.with_suffix(raw.suffix + ".plan.json")


def require_plan_evidence(
    raw: Path,
    expected_candidate_validity_predicate: str | None = None,
    expected_database_fingerprint: dict[str, object] | None = None,
) -> dict[str, object]:
    path = plan_evidence_path(raw)
    if not path.is_file():
        raise RuntimeError(f"missing HNSW plan evidence: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid HNSW plan evidence {path}: {exc}") from exc
    checks = payload.get("checks")
    if payload.get("status") != "complete" or not isinstance(checks, list) or not checks:
        raise RuntimeError(f"incomplete HNSW plan evidence: {path}")
    if not all(isinstance(check, dict) and check.get("passed") is True for check in checks):
        raise RuntimeError(f"failed HNSW plan check in {path}")
    query_contract = payload.get("query_contract")
    contract_predicate = (
        query_contract.get("candidate_validity_predicate")
        if isinstance(query_contract, dict)
        else None
    )
    if expected_candidate_validity_predicate is not None or contract_predicate is not None:
        expected_validity = effective_candidate_validity_predicate(
            expected_candidate_validity_predicate
            if expected_candidate_validity_predicate is not None
            else contract_predicate
        )
        expected_validity_sha = candidate_validity_sha256(expected_validity)
        expected_is_partial = not candidate_validity_index_predicate_matches(
            None, expected_validity
        )
        if (
            not isinstance(query_contract, dict)
            or query_contract.get("candidate_validity_predicate") != expected_validity
            or query_contract.get("candidate_validity_predicate_sha256")
            != expected_validity_sha
        ):
            raise RuntimeError(f"candidate validity query contract mismatch in {path}")
        if any(
            check.get("candidate_validity_predicate") != expected_validity
            or check.get("candidate_validity_predicate_sha256") != expected_validity_sha
            or check.get("expected_index_predicate") != expected_validity
            or check.get("expected_index_predicate_sha256") != expected_validity_sha
            or check.get("expected_index_is_partial") is not expected_is_partial
            or check.get("catalog_index_predicate_matches") is not True
            or check.get("catalog_index_is_partial") is not expected_is_partial
            or check.get("catalog_index_predicate_sha256")
            != candidate_validity_sha256(
                check.get("catalog_index_predicate")
                if check.get("catalog_index_predicate") is not None
                else "TRUE"
            )
            for check in checks
        ):
            raise RuntimeError(f"candidate validity plan evidence mismatch in {path}")
        if expected_database_fingerprint is not None:
            relations = expected_database_fingerprint.get("relations")
            if not isinstance(relations, dict):
                raise RuntimeError(f"missing database relation fingerprint in {path}")
            for check in checks:
                relation_name = str(check.get("expected_index") or "")
                fingerprint = relations.get(relation_name)
                if not isinstance(fingerprint, dict):
                    raise RuntimeError(
                        f"plan evidence index is absent from parent database fingerprint: {relation_name}"
                    )
                if (
                    check.get("expected_index_oid") != fingerprint.get("oid")
                    or check.get("catalog_index_predicate")
                    != fingerprint.get("indpred")
                    or check.get("catalog_index_is_partial")
                    is not fingerprint.get("is_partial")
                ):
                    raise RuntimeError(
                        f"plan evidence does not match parent database fingerprint for {relation_name}"
                    )
    identities = [
        payload.get("sqlens_runtime_identity_startup"),
        payload.get("sqlens_runtime_identity_final"),
    ]
    if any(
        not isinstance(identity, dict) or identity.get("exact_match") is not True
        for identity in identities
    ):
        raise RuntimeError(f"missing or mismatched exact SQLens runtime identity in {path}")
    if any(
        identities[0].get(field) != identities[1].get(field)
        for field in (
            "expected_build_id",
            "expected_vector_so_sha256",
            "observed_build_id",
            "observed_vector_so_sha256",
        )
    ):
        raise RuntimeError(f"SQLens runtime identity changed during child run in {path}")
    lifecycle = payload.get("execution_lifecycle")
    if (
        not isinstance(lifecycle, dict)
        or lifecycle.get("warmup_complete") is not True
        or lifecycle.get("d3_lifecycle_complete") is not True
        or lifecycle.get("backend_cpu_provenance_complete") is not True
        or lifecycle.get("runtime_sqlens_identity_complete") is not True
    ):
        raise RuntimeError(f"incomplete warmup/D3 lifecycle evidence in {path}")
    backend_cpu_evidence = payload.get("backend_cpu_evidence")
    if not isinstance(backend_cpu_evidence, list) or not backend_cpu_evidence:
        raise RuntimeError(f"missing production backend CPU provenance in {path}")
    if any(
        int(item.get("backend_pid") or 0) <= 0
        or not item.get("observed_cpu_list")
        or item.get("pinning_attempted_by_runner") is not False
        or (
            item.get("requested_cpu_list")
            and item.get("exact_match") is not True
        )
        for item in backend_cpu_evidence
        if isinstance(item, dict)
    ) or any(not isinstance(item, dict) for item in backend_cpu_evidence):
        raise RuntimeError(f"invalid production backend CPU provenance in {path}")
    runtime_identities = payload.get("runtime_sqlens_identity_evidence")
    if not isinstance(runtime_identities, list) or not runtime_identities or any(
        not isinstance(identity, dict) or identity.get("exact_match") is not True
        for identity in runtime_identities
    ):
        raise RuntimeError(f"missing production backend SQLens identity in {path}")
    if payload.get("guidance_filter_strategy") == "traversal_guided":
        query_contract = payload.get("query_contract")
        expected_self_excluded = (
            bool(query_contract.get("self_excluded", True))
            if isinstance(query_contract, dict)
            else True
        )
        expected_contract = (
            "limit_k_plus_1_client_remove_query_id"
            if expected_self_excluded
            else "none_external_query_source"
        )
        if not isinstance(query_contract, dict) or query_contract.get("self_exclusion") != expected_contract:
            raise RuntimeError(f"invalid traversal-guided query contract in {path}")
        if any(
            check.get("self_exclusion_contract")
            != expected_contract
            or check.get("residual_self_qual_present") is not False
            or int(check.get("scan_limit") or 0) <= 0
            for check in checks
        ):
            raise RuntimeError(f"residual or incomplete traversal-guided plan contract in {path}")
    d2_modes = {
        "design1_bloom_bfs_layout",
        "design1_bloom_bfs_layout_d3",
    }
    if any(check.get("mode") in d2_modes for check in checks):
        proof = payload.get("d2_graph_proof")
        if not isinstance(proof, dict):
            raise RuntimeError(f"missing D2 same-graph proof in {path}")
        validate_d2_graph_proof(
            proof,
            str(proof.get("source_index") or ""),
            str(proof.get("clone_index") or ""),
        )
        final_proof = payload.get("d2_graph_proof_final")
        if not isinstance(final_proof, dict):
            raise RuntimeError(f"missing final D2 live revalidation in {path}")
        startup_validated = validate_d2_graph_proof(
            proof,
            str(proof.get("source_index") or ""),
            str(proof.get("clone_index") or ""),
        )
        final_validated = validate_d2_graph_proof(
            final_proof,
            str(final_proof.get("source_index") or ""),
            str(final_proof.get("clone_index") or ""),
        )
        if (
            startup_validated.get("stable_fingerprint_sha256")
            != final_validated.get("stable_fingerprint_sha256")
        ):
            raise RuntimeError(f"D2 proof changed before child finalization in {path}")
    expected_sha = str(payload.get("output_sha256") or "")
    actual_sha = sha256_file(raw)
    if not expected_sha or expected_sha != actual_sha:
        raise RuntimeError(f"plan evidence output SHA256 mismatch for {raw}")
    if int(payload.get("output_rows") or -1) != csv_row_count(raw):
        raise RuntimeError(f"plan evidence row count mismatch for {raw}")
    return payload


def plan_evidence_manifest_entry(
    raw: Path,
    expected_candidate_validity_predicate: str | None = None,
    expected_database_fingerprint: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = require_plan_evidence(
        raw,
        expected_candidate_validity_predicate,
        expected_database_fingerprint,
    )
    path = plan_evidence_path(raw)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload["status"],
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "raw_output": output_artifact(raw),
        "query_contract": payload.get("query_contract"),
        "guidance_filter_strategy": payload.get("guidance_filter_strategy"),
        "sqlens_runtime_identity_startup": payload.get("sqlens_runtime_identity_startup"),
        "sqlens_runtime_identity_final": payload.get("sqlens_runtime_identity_final"),
        "execution_lifecycle": payload.get("execution_lifecycle"),
        "backend_cpu_evidence": payload.get("backend_cpu_evidence"),
        "runtime_sqlens_identity_evidence": payload.get(
            "runtime_sqlens_identity_evidence"
        ),
        "d2_graph_proof": payload.get("d2_graph_proof"),
        "d2_graph_proof_final": payload.get("d2_graph_proof_final"),
        "checks": [
            {
                key: check.get(key)
                for key in (
                    "passed",
                    "mode",
                    "filter_name",
                    "query_id",
                    "query_table",
                    "query_id_column",
                    "query_vector_column",
                    "candidate_validity_predicate",
                    "candidate_validity_predicate_sha256",
                    "self_excluded",
                    "self_exclusion_contract",
                    "scan_limit",
                    "residual_self_qual_present",
                    "expected_index_identity",
                    "expected_index_oid",
                    "expected_index_access_method",
                    "expected_index_predicate",
                    "expected_index_predicate_sha256",
                    "expected_index_is_partial",
                    "catalog_index_oid",
                    "catalog_index_predicate",
                    "catalog_index_predicate_sha256",
                    "catalog_index_is_partial",
                    "catalog_index_predicate_matches",
                    "expected_table_identity",
                    "observed_index_nodes",
                    "matched_index_nodes",
                    "preferred_index_guc",
                    "preferred_index_guc_available",
                    "preferred_index_current_setting",
                    "backend_cpu_provenance",
                    "sqlens_runtime_identity",
                    "failure",
                )
            }
            for check in payload["checks"]
        ],
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def sqlens_runtime_provenance() -> dict[str, object]:
    """Read and validate the loaded SQLens ABI/profile contract without mutating the DB."""
    try:
        with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute(
                "WITH lib AS ("
                "SELECT setting || '/vector.so' AS path "
                "FROM pg_config WHERE name = 'PKGLIBDIR'"
                ") SELECT vector_sqlens_build_id(), path, "
                "encode(sha256(pg_read_binary_file(path)), 'hex') FROM lib"
            )
            row = cur.fetchone()
            build_id = str(row[0]) if row and row[0] is not None else ""
            binary_path = str(row[1]) if row and row[1] is not None else ""
            binary_sha256 = str(row[2]) if row and row[2] is not None else ""
            cur.execute("SELECT vector_hnsw_last_scan_profile()")
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - formal provenance must fail closed
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: required SQLens functions are unavailable"
        ) from exc

    if not build_id.startswith(SQLENS_V11_BUILD_PREFIX):
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: "
            f"loaded vector_sqlens_build_id is {build_id!r}, expected {SQLENS_V11_BUILD_PREFIX!r}"
        )
    if (
        not binary_path.endswith("/vector.so")
        or len(binary_sha256) != 64
        or any(char not in "0123456789abcdef" for char in binary_sha256)
    ):
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: loaded vector.so path/SHA256 is invalid"
        )
    raw_profile = row[0] if row else None
    try:
        profile = json.loads(raw_profile) if isinstance(raw_profile, str) else raw_profile
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: last scan profile is not valid JSON"
        ) from exc
    if not isinstance(profile, dict):
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: last scan profile is not a JSON object"
        )
    try:
        profile_version = float(profile["profile_semantics_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: profile_semantics_version is missing or invalid"
        ) from exc
    required_fields = SQLENS_PROFILE_REQUIRED_FIELDS + SQLENS_TRAVERSAL_PROFILE_REQUIRED_FIELDS
    missing = [field for field in required_fields if field not in profile]
    if not math.isfinite(profile_version) or profile_version < SQLENS_MIN_PROFILE_SEMANTICS or missing:
        raise RuntimeError(
            "SQLens v11 runtime provenance gate failed: incompatible profile "
            f"version={profile.get('profile_semantics_version')!r}, "
            f"minimum_version={SQLENS_MIN_PROFILE_SEMANTICS:g}, missing_fields={missing!r}"
        )
    return {
        "loaded_vector_sqlens_build_id": build_id,
        "loaded_vector_so_path": binary_path,
        "loaded_vector_so_sha256": binary_sha256,
        "required_build_prefix": SQLENS_V11_BUILD_PREFIX,
        "minimum_profile_semantics_version": SQLENS_MIN_PROFILE_SEMANTICS,
        "profile_semantics_version": profile["profile_semantics_version"],
        "required_profile_fields": {
            field: profile[field] for field in required_fields
        },
    }


def database_fingerprint(args: argparse.Namespace, sqlens_build_id: str) -> dict[str, object]:
    relations = [
        args.insertion_table,
        args.insertion_index,
        args.bfs_table,
        args.bfs_index,
    ]
    validity = effective_candidate_validity_predicate(
        getattr(args, "candidate_validity_predicate", "")
    )
    out: dict[str, object] = {
        "relations": {},
        "candidate_validity_predicate": validity,
        "candidate_validity_predicate_sha256": candidate_validity_sha256(validity),
        "candidate_validity_predicate_explicit": bool(
            getattr(args, "candidate_validity_predicate_explicit", False)
        ),
    }
    index_relations = {str(args.insertion_index), str(args.bfs_index)}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT current_setting('server_version'), "
            "COALESCE((SELECT extversion FROM pg_extension WHERE extname='vector'), '')"
        )
        out["postgres_version"], out["vector_extension_version"] = cur.fetchone()
        out["sqlens_build_id"] = sqlens_build_id
        for relation in dict.fromkeys(relations):
            cur.execute(
                "SELECT c.oid::bigint, c.relfilenode::bigint, c.reltuples::bigint, "
                "pg_relation_size(c.oid), COALESCE(i.indisvalid, true), "
                "COALESCE(i.indisready, true), pg_get_expr(i.indpred, i.indrelid) "
                "FROM pg_class c LEFT JOIN pg_index i ON i.indexrelid=c.oid "
                "WHERE c.oid=to_regclass(%s)",
                (relation,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"benchmark relation does not exist: {relation}")
            indpred = row[6] if len(row) > 6 else None
            is_index = relation in index_relations
            predicate_matches = (
                candidate_validity_index_predicate_matches(indpred, validity)
                if is_index
                else None
            )
            out["relations"][relation] = {
                "oid": row[0],
                "relfilenode": row[1],
                "reltuples": row[2],
                "bytes": row[3],
                "valid": row[4],
                "ready": row[5],
                "indpred": indpred,
                "is_partial": indpred is not None,
                "candidate_validity_predicate": validity if is_index else None,
                "candidate_validity_predicate_sha256": (
                    candidate_validity_sha256(validity) if is_index else None
                ),
                "candidate_validity_predicate_matches": predicate_matches,
            }
            if is_index and predicate_matches is not True:
                raise RuntimeError(
                    "candidate validity database fingerprint mismatch for "
                    f"{relation}: catalog indpred={indpred!r}, expected={validity!r}"
                )
        out["query_table"] = query_relation_provenance(
            cur, str(getattr(args, "query_table", None) or args.insertion_table)
        )
    return out


def relation_identifier(relation: str) -> psycopg.sql.Identifier:
    parts = relation.split(".")
    if len(parts) not in {1, 2} or any(not part for part in parts):
        raise RuntimeError(f"invalid relation name: {relation!r}")
    return psycopg.sql.Identifier(*parts)


def query_relation_provenance(cur: psycopg.Cursor, table: str) -> dict[str, object]:
    cur.execute(
        "SELECT %s::regclass::text, %s::regclass::oid::bigint, "
        "pg_relation_filenode(%s::regclass)::bigint",
        (table, table, table),
    )
    identity = cur.fetchone()
    if identity is None or identity[2] is None:
        raise RuntimeError(f"could not fingerprint query table {table}")
    cur.execute(psycopg.sql.SQL("SELECT count(*) FROM {}").format(relation_identifier(table)))
    count_row = cur.fetchone()
    cur.execute(
        "SELECT COALESCE(array_agg(attname || ':' || format_type(atttypid, atttypmod) "
        "ORDER BY attnum), ARRAY[]::text[]) "
        "FROM pg_attribute WHERE attrelid=%s::regclass "
        "AND attnum > 0 AND NOT attisdropped",
        (table,),
    )
    columns_row = cur.fetchone()
    if count_row is None or columns_row is None:
        raise RuntimeError(f"could not read query table provenance for {table}")
    return {
        "name": str(identity[0]),
        "oid": int(identity[1]),
        "relfilenode": int(identity[2]),
        "row_count": int(count_row[0]),
        "columns": list(columns_row[0] or []),
    }


def truth_query_ids(
    path: Path,
    expected_self_excluded: bool = True,
    expected_candidate_validity_predicate: str | None = None,
) -> dict[int, int]:
    query_ids: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "filtered_rows",
            "kth_distance_sq",
            "tie_tolerance",
            "self_excluded",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"truth CSV is missing tie-aware fields: {sorted(missing)}")
        if (
            expected_candidate_validity_predicate is not None
            and "candidate_validity_predicate" not in set(reader.fieldnames or ())
        ):
            raise RuntimeError(
                "truth CSV is missing candidate_validity_predicate required by the "
                "explicit candidate-validity contract"
            )
        for row in reader:
            if row.get("method") != "pre_filter_exact":
                continue
            if required_csv_bool(row.get("self_excluded", ""), "self_excluded") != expected_self_excluded:
                raise RuntimeError(
                    "truth CSV self_excluded value does not match the expected query contract"
                )
            if expected_candidate_validity_predicate is not None:
                expected_validity = effective_candidate_validity_predicate(
                    expected_candidate_validity_predicate
                )
                observed_validity = effective_candidate_validity_predicate(
                    row.get("candidate_validity_predicate", "")
                )
                if observed_validity != expected_validity:
                    raise RuntimeError(
                        "truth CSV candidate_validity_predicate does not match the expected "
                        f"candidate contract: observed={observed_validity!r}, "
                        f"expected={expected_validity!r}"
                    )
            filtered_rows = int(row["filtered_rows"])
            if filtered_rows < 0:
                raise RuntimeError("truth CSV contains a negative filtered_rows value")
            if filtered_rows and not row["kth_distance_sq"].strip():
                raise RuntimeError("non-empty formal truth row is missing kth_distance_sq")
            if row["kth_distance_sq"].strip():
                float(row["kth_distance_sq"])
            if float(row["tie_tolerance"]) < 0:
                raise RuntimeError("truth CSV contains a negative tie_tolerance")
            query_no = int(row["query_no"])
            query_id = int(row["query_id"])
            previous = query_ids.setdefault(query_no, query_id)
            if previous != query_id:
                raise RuntimeError(f"truth file maps query_no={query_no} to multiple IDs")
    return query_ids


def git_revision() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def normalized_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key
        not in {
            "resume",
            "skip_final",
            "run_spec_hash",
            "d2_graph_proof",
            "fragment_tracking_evidence",
        }
    }


def stable_fragment_tracking_evidence(args: argparse.Namespace) -> dict[str, object]:
    evidence = getattr(args, "fragment_tracking_evidence", None)
    if not isinstance(evidence, dict):
        return {"required": False, "prepared": False, "tables": []}
    return {key: value for key, value in evidence.items() if key != "prepared_at"}


def explicit_candidate_validity_predicate(
    args: argparse.Namespace,
) -> str | None:
    predicate = str(getattr(args, "candidate_validity_predicate", "") or "").strip()
    explicit = getattr(args, "candidate_validity_predicate_explicit", None)
    if explicit is None:
        # Namespaces made by older callers have no marker; preserve their non-empty
        # predicate behavior while treating an omitted value as legacy truth.
        explicit = bool(predicate)
    if not explicit:
        return None
    return effective_candidate_validity_predicate(predicate)


def formal_run_uses_d2(args: argparse.Namespace) -> bool:
    return any(
        mode in {"design1_bloom_bfs_layout", "design1_bloom_bfs_layout_d3"}
        for mode in args.modes
    )


def prepare_fragment_tracking(args: argparse.Namespace) -> dict[str, object]:
    required = any(mode != "original" for mode in args.modes)
    tables = list(dict.fromkeys((str(args.insertion_table), str(args.bfs_table))))
    if not required:
        return {"required": False, "prepared": False, "tables": []}

    rows: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        ensure_tracking(cur, *tables)
        for table in tables:
            cur.execute(
                "SELECT c.oid::bigint, c.relfilenode::bigint, e.epoch::bigint, "
                "EXISTS (SELECT 1 FROM pg_trigger t WHERE t.tgrelid = c.oid "
                "AND t.tgname = 'pgvector_hnsw_fragment_epoch' "
                "AND NOT t.tgisinternal AND t.tgenabled <> 'D') "
                "FROM pg_class c "
                "JOIN public.pgvector_hnsw_fragment_epoch e ON e.heap_oid = c.oid "
                "WHERE c.oid = %s::regclass",
                (table,),
            )
            row = cur.fetchone()
            if row is None or row[1] is None or row[2] is None or row[3] is not True:
                raise RuntimeError(
                    f"fragment tracking preparation is incomplete for {table}"
                )
            rows.append(
                {
                    "table": table,
                    "oid": int(row[0]),
                    "relfilenode": int(row[1]),
                    "epoch": int(row[2]),
                    "enabled_trigger": True,
                }
            )
    return {
        "required": True,
        "prepared": True,
        "lock_order": "tracking_ddl_committed_before_share_data_guard",
        "tables": rows,
        "prepared_at": utc_now(),
    }


def d2_graph_proof_from_env(args: argparse.Namespace) -> dict[str, object]:
    if not formal_run_uses_d2(args):
        return {"required": False}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        return require_d2_graph_proof(
            cur,
            args.insertion_index,
            args.bfs_index,
        )


def acquire_formal_data_guard(
    args: argparse.Namespace,
) -> tuple[psycopg.Connection, dict[str, object]]:
    query_table = str(getattr(args, "query_table", None) or args.insertion_table)
    tables = sorted({str(args.insertion_table), str(args.bfs_table), query_table})
    connection = psycopg.connect(pg_config_from_env().conninfo, autocommit=False)
    try:
        cur = connection.cursor()
        identifiers = []
        for table in tables:
            identifiers.append(relation_identifier(table))
        cur.execute(
            psycopg.sql.SQL("LOCK TABLE {} IN SHARE MODE").format(
                psycopg.sql.SQL(", ").join(identifiers)
            )
        )
        relation_identity: dict[str, dict[str, object]] = {}
        for table in tables:
            cur.execute(
                "SELECT %s::regclass::text, %s::regclass::oid::bigint, "
                "pg_relation_filenode(%s::regclass)::bigint",
                (table, table, table),
            )
            row = cur.fetchone()
            if row is None or row[2] is None:
                raise RuntimeError(f"could not fingerprint guarded table {table}")
            relation_identity[table] = {
                "name": str(row[0]),
                "oid": int(row[1]),
                "relfilenode": int(row[2]),
            }
        cur.execute("SELECT pg_backend_pid(), txid_current_snapshot()::text")
        context = cur.fetchone()
        query_provenance = query_relation_provenance(cur, query_table)
        return connection, {
            "lock_mode": "SHARE",
            "tables": tables,
            "relations": relation_identity,
            "query_table": query_provenance,
            "backend_pid": int(context[0]),
            "transaction_snapshot": str(context[1]),
            "acquired_at": utc_now(),
            "blocks_dml_and_relation_replacement": True,
        }
    except BaseException:
        connection.rollback()
        connection.close()
        raise


def build_run_spec(args: argparse.Namespace) -> dict[str, object]:
    source_files = list((ROOT / "third_party/pgvector-sqlens/src").glob("*.c"))
    source_files.extend((ROOT / "third_party/pgvector-sqlens/src").glob("*.h"))
    query_ids = truth_query_ids(
        args.truth_csv,
        expected_self_excluded=args.expected_truth_self_excluded,
        expected_candidate_validity_predicate=explicit_candidate_validity_predicate(args),
    )
    sqlens_runtime = sqlens_runtime_provenance()
    d2_proof = d2_graph_proof_from_env(args)
    spec = {
        "args": normalized_args(args),
        "truth_sha256": sha256_file(args.truth_csv),
        "filters_sha256": sha256_file(args.filters_csv),
        "git_revision": git_revision(),
        "runner_sha256": sha256_file(Path(__file__)),
        "benchmark_sha256": sha256_file(
            ROOT / "experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py"
        ),
        "sqlens_source_sha256": sha256_tree(source_files),
        "sqlens_runtime_provenance": sqlens_runtime,
        "d2_graph_proof": d2_proof,
        "fragment_tracking_preparation": stable_fragment_tracking_evidence(args),
        "query_contract": {
            "query_table": args.query_table or args.insertion_table,
            "query_id_column": args.query_id_column,
            "query_vector_column": args.query_vector_column,
            "self_excluded": args.expected_truth_self_excluded,
            "candidate_validity_predicate": effective_candidate_validity_predicate(
                getattr(args, "candidate_validity_predicate", "")
            ),
            "candidate_validity_predicate_sha256": candidate_validity_sha256(
                getattr(args, "candidate_validity_predicate", "")
            ),
            "candidate_validity_predicate_explicit": bool(
                getattr(args, "candidate_validity_predicate_explicit", False)
            ),
            "candidate_validity_contract": (
                "planner_partial_index_predicate_and_sql_candidate_qual_not_guidance_atom"
            ),
            "predicate_contract": (
                "exact_activated_workload_predicate_plus_candidate_validity_sql_qual"
                if args.guidance_filter_strategy == "traversal_guided"
                else "diagnostic_workload_plus_candidate_validity_sql_quals"
            ),
            "self_exclusion": (
                "limit_k_plus_1_client_remove_query_id"
                if (
                    args.guidance_filter_strategy == "traversal_guided"
                    and args.expected_truth_self_excluded
                )
                else (
                    "sql_residual_id_not_equal"
                    if args.expected_truth_self_excluded
                    else "none_external_query_source"
                )
            ),
            "all_modes_share_identical_sql_shape": True,
            "measured_latency_includes_limit_k_plus_1_and_client_self_exclusion": (
                args.guidance_filter_strategy == "traversal_guided"
                and args.expected_truth_self_excluded
            ),
        },
        "database": database_fingerprint(args, str(sqlens_runtime["loaded_vector_sqlens_build_id"])),
        "calibration_query_ids": [
            query_ids[query_no]
            for query_no in range(
                args.calibration_query_offset,
                args.calibration_query_offset + args.calibration_queries,
            )
        ],
        "final_query_ids": [
            query_ids[query_no]
            for query_no in range(args.final_query_offset, args.final_query_offset + args.final_queries)
        ],
    }
    hash_payload = dict(spec)
    if formal_run_uses_d2(args):
        hash_payload["d2_graph_proof"] = stable_d2_graph_proof(d2_proof)
    encoded = json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    spec["run_spec_hash"] = hashlib.sha256(encoded).hexdigest()
    return spec


def reusable_summary(
    path: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    expected_queries: int,
    expected_repeats: int,
    expected_truth_self_excluded: bool = True,
) -> list[dict[str, object]] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        summary = summarize_raw(
            path,
            bootstrap_samples,
            bootstrap_seed,
            expected_queries,
            expected_repeats,
            expected_truth_self_excluded,
        )
    except (KeyError, TypeError, ValueError, csv.Error):
        return None
    return summary if len(summary) == 1 else None


def append_option(cmd: list[str], name: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def append_expected_runtime_identity(cmd: list[str], args: argparse.Namespace) -> None:
    provenance = getattr(args, "sqlens_runtime_provenance", None)
    if not isinstance(provenance, dict):
        raise RuntimeError("parent SQLens runtime provenance is unavailable")
    build_id = str(provenance.get("loaded_vector_sqlens_build_id") or "")
    vector_sha = str(provenance.get("loaded_vector_so_sha256") or "")
    if not build_id or len(vector_sha) != 64:
        raise RuntimeError("parent SQLens exact build ID/vector.so SHA256 is incomplete")
    cmd.extend(
        [
            "--expected-sqlens-build-id",
            build_id,
            "--expected-vector-so-sha256",
            vector_sha,
        ]
    )


def run_d123(
    out: Path,
    filter_name: str,
    mode: str,
    query_offset: int,
    queries: int,
    repeats: int,
    config: Config,
    args: argparse.Namespace,
    log: Path | None,
) -> float:
    cmd = [
        sys.executable,
        "experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py",
        "--out",
        str(out),
        "--queries",
        str(queries),
        "--query-offset",
        str(query_offset),
        "--repeats",
        str(repeats),
        "--ef-search",
        str(config.ef_search),
        "--guided-collect-target",
        str(config.guided_collect_target),
        "--max-scan-tuples",
        str(config.max_scan_tuples),
        "--scan-mem-multiplier",
        str(config.scan_mem_multiplier),
        "--iterative-scan",
        config.iterative_scan,
        "--progress-queries",
        str(args.progress_queries),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--filter-names",
        filter_name,
        "--modes",
        mode,
        "--guidance-filter-strategy",
        args.guidance_filter_strategy,
        "--guidance-selectivity-max-pct",
        str(args.guidance_selectivity_max_pct),
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--d2-page-access",
        args.d2_page_access,
        "--d2-index-page-access",
        args.d2_index_page_access,
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--d3-cache-mb",
        str(args.d3_cache_mb),
        "--preferred-index-guc",
        args.preferred_index_guc,
        "--d2-graph-proof-json",
        json.dumps(args.d2_graph_proof, sort_keys=True, separators=(",", ":")),
    ]
    append_option(cmd, "--filters-csv", args.filters_csv)
    append_option(cmd, "--truth-csv", args.truth_csv)
    append_option(cmd, "--insertion-table", args.insertion_table)
    append_option(cmd, "--insertion-index", args.insertion_index)
    append_option(cmd, "--bfs-table", args.bfs_table)
    append_option(cmd, "--bfs-index", args.bfs_index)
    append_option(cmd, "--query-table", getattr(args, "query_table", None))
    append_option(cmd, "--query-id-column", getattr(args, "query_id_column", "id"))
    append_option(cmd, "--query-vector-column", getattr(args, "query_vector_column", "embedding"))
    if getattr(
        args,
        "candidate_validity_predicate_explicit",
        bool(getattr(args, "candidate_validity_predicate", "").strip()),
    ):
        append_option(
            cmd,
            "--candidate-validity-predicate",
            args.candidate_validity_predicate,
        )
    if not getattr(args, "expected_truth_self_excluded", True):
        cmd.append("--no-expected-truth-self-excluded")
    append_option(cmd, "--backend-cpu-list", getattr(args, "backend_cpu_list", None))
    append_expected_runtime_identity(cmd, args)
    if getattr(args, "fragment_tracking_prepared", False):
        cmd.append("--fragment-tracking-prepared")
    if not args.require_preferred_index_guc:
        cmd.append("--no-require-preferred-index-guc")
    if args.warmup_all_queries:
        cmd.append("--warmup-all-queries")
    if not args.force_hnsw:
        cmd.append("--no-force-hnsw")
    return run_command(cmd, log)


def run_d123_interleaved(
    out: Path,
    filter_name: str,
    modes: list[str],
    mode_configs: dict[str, Config],
    args: argparse.Namespace,
    log: Path | None,
) -> float:
    config_path = out.with_suffix(".configs.json")
    config_path.write_text(
        json.dumps({mode: asdict(mode_configs[mode]) for mode in modes}, indent=2) + "\n",
        encoding="utf-8",
    )
    first = mode_configs[modes[0]]
    cmd = [
        sys.executable,
        "experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py",
        "--out",
        str(out),
        "--queries",
        str(args.final_queries),
        "--query-offset",
        str(args.final_query_offset),
        "--repeats",
        str(args.final_repeats),
        "--ef-search",
        str(first.ef_search),
        "--guided-collect-target",
        str(first.guided_collect_target),
        "--max-scan-tuples",
        str(first.max_scan_tuples),
        "--scan-mem-multiplier",
        str(first.scan_mem_multiplier),
        "--iterative-scan",
        first.iterative_scan,
        "--execution-order",
        "interleaved",
        "--schedule-seed",
        str(args.schedule_seed),
        "--mode-configs-json",
        str(config_path),
        "--progress-queries",
        str(args.progress_queries),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--filter-names",
        filter_name,
        "--modes",
        *modes,
        "--guidance-filter-strategy",
        args.guidance_filter_strategy,
        "--guidance-selectivity-max-pct",
        str(args.guidance_selectivity_max_pct),
        "--guidance-max-atoms",
        str(args.guidance_max_atoms),
        "--d2-page-access",
        args.d2_page_access,
        "--d2-index-page-access",
        args.d2_index_page_access,
        "--d1-cache-mb",
        str(args.d1_cache_mb),
        "--d3-cache-mb",
        str(args.d3_cache_mb),
        "--preferred-index-guc",
        args.preferred_index_guc,
        "--d2-graph-proof-json",
        json.dumps(args.d2_graph_proof, sort_keys=True, separators=(",", ":")),
    ]
    append_option(cmd, "--filters-csv", args.filters_csv)
    append_option(cmd, "--truth-csv", args.truth_csv)
    append_option(cmd, "--insertion-table", args.insertion_table)
    append_option(cmd, "--insertion-index", args.insertion_index)
    append_option(cmd, "--bfs-table", args.bfs_table)
    append_option(cmd, "--bfs-index", args.bfs_index)
    append_option(cmd, "--query-table", getattr(args, "query_table", None))
    append_option(cmd, "--query-id-column", getattr(args, "query_id_column", "id"))
    append_option(cmd, "--query-vector-column", getattr(args, "query_vector_column", "embedding"))
    if getattr(
        args,
        "candidate_validity_predicate_explicit",
        bool(getattr(args, "candidate_validity_predicate", "").strip()),
    ):
        append_option(
            cmd,
            "--candidate-validity-predicate",
            args.candidate_validity_predicate,
        )
    if not getattr(args, "expected_truth_self_excluded", True):
        cmd.append("--no-expected-truth-self-excluded")
    append_option(cmd, "--backend-cpu-list", getattr(args, "backend_cpu_list", None))
    append_expected_runtime_identity(cmd, args)
    if getattr(args, "fragment_tracking_prepared", False):
        cmd.append("--fragment-tracking-prepared")
    if not args.require_preferred_index_guc:
        cmd.append("--no-require-preferred-index-guc")
    if args.warmup_all_queries:
        cmd.append("--warmup-all-queries")
    if not args.force_hnsw:
        cmd.append("--no-force-hnsw")
    return run_command(cmd, log)


def configs_for_mode(
    configs: list[Config],
    mode: str,
    stock_iterative_scan_values: str | None = None,
) -> list[Config]:
    if mode != "original":
        return configs
    # Stock pgvector must be tuned with its production iterative-scan options even
    # though traversal-safe SQLens modes require iterative_scan=off.
    iterative_values = (
        [value.strip() for value in stock_iterative_scan_values.split(",") if value.strip()]
        if stock_iterative_scan_values is not None
        else list(dict.fromkeys(config.iterative_scan for config in configs))
    )
    allowed = {"off", "relaxed_order", "strict_order"}
    if not iterative_values or any(value not in allowed for value in iterative_values):
        raise ValueError(
            "stock iterative scan values must be a non-empty subset of "
            f"{sorted(allowed)}"
        )
    # guided_collect_target has no effect on stock pgvector. Do not rerun duplicates.
    unique: dict[tuple[int, int, float, str], Config] = {}
    for config in configs:
        for iterative_scan in iterative_values:
            key = (
                config.ef_search,
                config.max_scan_tuples,
                config.scan_mem_multiplier,
                iterative_scan,
            )
            unique.setdefault(
                key,
                Config(
                    config.ef_search,
                    config.max_scan_tuples,
                    config.scan_mem_multiplier,
                    iterative_scan,
                    config.guided_collect_target,
                ),
            )
    return list(unique.values())


def calibration_row_is_complete(row: dict[str, object]) -> bool:
    return (
        int(row.get("ok", 0)) > 0
        and int(row.get("errors", 0)) == 0
        and bool(row.get("rows_complete", False))
    )


def calibration_stop_reached(rows: list[dict[str, object]], target: float) -> bool:
    return bool(rows) and all(calibration_row_is_complete(row) for row in rows) and any(
        calibration_row_is_complete(row)
        and float(row["recall_mean"]) >= target
        for row in rows
    )


def _stable_seed(seed: int, *parts: object) -> int:
    encoded = "\0".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def seeded_calibration_block(
    configs: list[Config],
    seed: int,
    filter_name: str,
    mode: str,
    family: str,
    block_no: int,
) -> list[Config]:
    scheduled = list(configs)
    random.Random(
        _stable_seed(seed, "calibration-config-block", filter_name, mode, family, block_no)
    ).shuffle(scheduled)
    return scheduled


def _seeded_family_order(
    families: list[str],
    seed: int,
    filter_name: str,
    mode: str,
    round_no: int,
) -> list[str]:
    scheduled = sorted(families)
    random.Random(
        _stable_seed(seed, "calibration-family-round", filter_name, mode, round_no)
    ).shuffle(scheduled)
    return scheduled


def mode_calibration_grids(
    configs: list[Config],
    modes: list[str],
    stock_iterative_scan_values: str | None,
) -> dict[str, list[dict[str, object]]]:
    return {
        mode: [
            asdict(config)
            for config in configs_for_mode(configs, mode, stock_iterative_scan_values)
        ]
        for mode in modes
    }


def calibrate_mode_filter(
    filter_name: str,
    mode: str,
    configs: list[Config],
    args: argparse.Namespace,
    targets: list[float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Interleave monotone family blocks while stopping each family independently."""
    rows: list[dict[str, object]] = []
    ordered_configs = sorted(
        configs_for_mode(
            configs,
            mode,
            getattr(args, "stock_iterative_scan_values", None),
        ),
        key=lambda config: (config.ef_search, config.iterative_scan, config.label),
    )
    max_target = max(targets)
    reused_configs = 0
    run_configs = 0
    schedule_seed = int(getattr(args, "schedule_seed", 20260718))
    family_configs: dict[str, list[Config]] = {}
    for config in ordered_configs:
        family_configs.setdefault(config.iterative_scan, []).append(config)
    family_state: dict[str, dict[str, object]] = {}
    for family, items in family_configs.items():
        family_state[family] = {
            "ef_values": sorted({config.ef_search for config in items}),
            "next_ef": 0,
            "configs_planned": len(items),
            "configs_executed": 0,
            "configs_run": 0,
            "configs_reused": 0,
            "errors": 0,
            "stopped_early": False,
            "grid_exhausted": False,
            "max_ef_evaluated": None,
            "complete": False,
        }

    calibration_block_no = 0
    family_round = 0
    while True:
        active_families = [
            family for family, state in family_state.items() if not bool(state["complete"])
        ]
        if not active_families:
            break
        for family in _seeded_family_order(
            active_families,
            schedule_seed,
            filter_name,
            mode,
            family_round,
        ):
            state = family_state[family]
            ef_values = state["ef_values"]
            assert isinstance(ef_values, list)
            ef_position = int(state["next_ef"])
            ef_search = int(ef_values[ef_position])
            block_configs = [
                config
                for config in family_configs[family]
                if config.ef_search == ef_search
            ]
            block_configs = seeded_calibration_block(
                block_configs,
                schedule_seed,
                filter_name,
                mode,
                family,
                calibration_block_no,
            )
            block_rows: list[dict[str, object]] = []
            for config in block_configs:
                stem = f"sigmod_matched_calib_{filter_name}_{mode}_{config.label}_{args.run_spec_hash[:12]}_{args.tag}"
                out = RESULTS / f"{stem}.csv"
                log = RESULTS / "logs" / f"sigmod_matched_{args.tag}" / f"{stem}.log"
                summary = reusable_summary(
                    out,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                    args.calibration_queries,
                    args.calibration_repeats,
                    getattr(args, "expected_truth_self_excluded", True),
                ) if args.resume else None
                if summary is not None:
                    try:
                        require_plan_evidence(
                            out,
                            explicit_candidate_validity_predicate(args),
                            getattr(args, "database_fingerprint", None),
                        )
                    except RuntimeError:
                        summary = None
                if summary is None:
                    run_configs += 1
                    state["configs_run"] = int(state["configs_run"]) + 1
                    elapsed_ms = run_d123(
                        out,
                        filter_name,
                        mode,
                        args.calibration_query_offset,
                        args.calibration_queries,
                        args.calibration_repeats,
                        config,
                        args,
                        log,
                    )
                    summary = summarize_raw(
                        out,
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                        args.calibration_queries,
                        args.calibration_repeats,
                        getattr(args, "expected_truth_self_excluded", True),
                    )
                    if len(summary) != 1:
                        raise RuntimeError(
                            f"expected one calibration summary in {out}, got {len(summary)}"
                        )
                else:
                    reused_configs += 1
                    state["configs_reused"] = int(state["configs_reused"]) + 1
                    elapsed_ms = 0.0
                    print(f"reusing {out}", flush=True)
                if int(summary[0]["queries"]) != args.calibration_queries:
                    raise RuntimeError(
                        f"calibration query split is incomplete in {out}: "
                        f"expected {args.calibration_queries}, got {summary[0]['queries']}"
                    )
                plan_evidence = plan_evidence_manifest_entry(
                    out,
                    explicit_candidate_validity_predicate(args),
                    getattr(args, "database_fingerprint", None),
                )
                row = {
                    "filter_name": filter_name,
                    "mode": mode,
                    "guidance_filter_strategy": args.guidance_filter_strategy,
                    "config": config.label,
                    **asdict(config),
                    **summary[0],
                    "elapsed_ms": elapsed_ms,
                    "raw": str(out),
                    "raw_rows": csv_row_count(out),
                    "raw_sha256": sha256_file(out),
                    "log": str(log),
                    "plan_evidence_file": plan_evidence["path"],
                    "plan_evidence_sha256": plan_evidence["sha256"],
                    "plan_checks": len(plan_evidence["checks"]),
                    "plan_gate_passed": True,
                    "calibration_family": family,
                    "calibration_block_no": calibration_block_no,
                }
                rows.append(row)
                block_rows.append(row)
                state["configs_executed"] = int(state["configs_executed"]) + 1
                state["errors"] = int(state["errors"]) + int(row.get("errors", 0))

            state["max_ef_evaluated"] = ef_search
            at_last_ef = ef_position == len(ef_values) - 1
            if calibration_stop_reached(block_rows, max_target):
                state["stopped_early"] = not at_last_ef
                state["grid_exhausted"] = at_last_ef
                state["complete"] = True
            elif at_last_ef:
                state["grid_exhausted"] = True
                state["complete"] = True
            else:
                state["next_ef"] = ef_position + 1
            calibration_block_no += 1
        family_round += 1

    max_ef = max(config.ef_search for config in ordered_configs)
    family_evidence = {
        family: {
            key: value
            for key, value in state.items()
            if key not in {"ef_values", "next_ef", "complete"}
        }
        for family, state in sorted(family_state.items())
    }
    calibration_failed = any(int(state["errors"]) > 0 for state in family_state.values())
    evidence = {
        "filter_name": filter_name,
        "mode": mode,
        "configs_planned": len(ordered_configs),
        "configs_executed": len(rows),
        "configs_run": run_configs,
        "configs_reused": reused_configs,
        "stopped_early": any(bool(state["stopped_early"]) for state in family_state.values()),
        "max_ef_evaluated": max((int(row["ef_search"]) for row in rows), default=None),
        "max_ef_on_grid": max_ef,
        "grid_exhausted": all(bool(state["grid_exhausted"]) for state in family_state.values()),
        "calibration_failed": calibration_failed,
        "families": family_evidence,
        "calibration_execution_order": "seeded_interleaved_family_ef_blocks",
        "calibration_schedule_seed": schedule_seed,
        "calibration_stop_target": max_target,
    }
    for row in rows:
        row.update({key: value for key, value in evidence.items() if key != "families"})
        family = str(row["calibration_family"])
        family_values = family_evidence[family]
        row.update(
            {
                "family_configs_planned": family_values["configs_planned"],
                "family_configs_executed": family_values["configs_executed"],
                "family_stopped_early": family_values["stopped_early"],
                "family_grid_exhausted": family_values["grid_exhausted"],
                "family_errors": family_values["errors"],
            }
        )
    return rows, evidence


def select_row(rows: list[dict[str, object]], target: float) -> tuple[dict[str, object] | None, bool]:
    valid = [
        row
        for row in rows
        if int(row["ok"]) > 0
        and int(row["errors"]) == 0
        and bool(row.get("rows_complete", True))
    ]
    reached = [
        row
        for row in valid
        if float(row["recall_mean"]) >= target
    ]
    if reached:
        return min(
            reached,
            key=lambda row: (
                float(row["latency_mean_ms"]),
                -float(row["recall_mean"]),
                str(row["config"]),
            ),
        ), True
    return None, False


def config_from_row(row: dict[str, object]) -> Config:
    return Config(
        ef_search=int(row["ef_search"]),
        max_scan_tuples=int(row["max_scan_tuples"]),
        scan_mem_multiplier=float(row["scan_mem_multiplier"]),
        iterative_scan=str(row["iterative_scan"]),
        guided_collect_target=int(row["guided_collect_target"]),
    )


def build_configs(args: argparse.Namespace) -> list[Config]:
    ef_values = parse_ints(args.ef_search_values)
    target_tokens = [token.strip() for token in args.guided_collect_target_values.split(",") if token.strip()]
    configs: list[Config] = []
    for iterative in [x for x in args.iterative_scan_values.split(",") if x]:
        if args.guidance_filter_strategy == "traversal_guided" and iterative != "off":
            raise SystemExit(
                "formal traversal_guided calibration requires --iterative-scan-values=off; "
                f"got {iterative!r}"
            )
        for ef in ef_values:
            if args.guidance_filter_strategy == "safe_guided":
                targets = [1]
            else:
                targets = [ef if token == "ef" else int(token) for token in target_tokens]
                if args.guidance_filter_strategy == "traversal_guided":
                    targets = [max(ef, target) for target in targets]
            for target in sorted(set(targets)):
                for max_scan in parse_ints(args.max_scan_tuples_values):
                    for mem in parse_floats(args.scan_mem_multiplier_values):
                        configs.append(Config(ef, max_scan, mem, iterative, target))
    if not configs:
        raise SystemExit("empty calibration configuration space")
    return configs


def selected_rows(
    calibration_rows: list[dict[str, object]],
    filters: list[str],
    modes: list[str],
    targets: list[float],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for filter_name in filters:
        for mode in modes:
            candidates = [
                row
                for row in calibration_rows
                if row["filter_name"] == filter_name and row["mode"] == mode
            ]
            for target in targets:
                calibration_failed = any(
                    not calibration_row_is_complete(row) for row in candidates
                )
                selected, met = (
                    (None, False)
                    if calibration_failed
                    else select_row(candidates, target)
                )
                if selected is not None:
                    out.append(
                        {
                            "target_recall": target,
                            "target_met_in_calibration": True,
                            "target_confirmed_in_calibration": True,
                            "target_lcb95_met_in_calibration": (
                                float(selected.get("recall_lcb95", selected["recall_mean"]))
                                >= target
                            ),
                            "selection_status": "selected",
                            "feasibility_status": "selected",
                            **selected,
                        }
                    )
                    continue

                all_complete = bool(candidates) and all(
                    calibration_row_is_complete(row) for row in candidates
                )
                grid_exhausted = bool(candidates) and bool(candidates[0].get("grid_exhausted"))
                status = (
                    "unattainable_on_grid"
                    if grid_exhausted and all_complete
                    else "incomplete_or_failed"
                )
                out.append(
                    {
                        "target_recall": target,
                        "target_met_in_calibration": False,
                        "target_confirmed_in_calibration": False,
                        "target_lcb95_met_in_calibration": False,
                        "selection_status": status,
                        "feasibility_status": status,
                        "filter_name": filter_name,
                        "mode": mode,
                        "config": "",
                        "ef_search": None,
                        "guided_collect_target": None,
                        "max_scan_tuples": None,
                        "scan_mem_multiplier": None,
                        "iterative_scan": "",
                        "configs_planned": candidates[0].get("configs_planned") if candidates else 0,
                        "configs_executed": len(candidates),
                        "stopped_early": candidates[0].get("stopped_early", False) if candidates else False,
                        "max_ef_evaluated": candidates[0].get("max_ef_evaluated") if candidates else None,
                        "grid_exhausted": grid_exhausted,
                    }
                )
    return out


def formal_completion_gate(
    filters: list[str],
    modes: list[str],
    targets: list[float],
    selected: list[dict[str, object]],
    final_rows: list[dict[str, object]],
    skip_final: bool,
) -> dict[str, object]:
    expected_filters = list(filters)
    expected_modes = set(DEFAULT_MODES)
    expected_targets = set(FORMAL_TARGETS)
    requested_formal_matrix = (
        len(expected_filters) == 14
        and len(set(expected_filters)) == 14
        and len(modes) == len(expected_modes)
        and set(modes) == expected_modes
        and len(targets) == len(expected_targets)
        and set(float(target) for target in targets) == expected_targets
    )
    expected_keys = {
        (filter_name, float(target), mode)
        for filter_name in expected_filters
        for target in FORMAL_TARGETS
        for mode in DEFAULT_MODES
    }

    def key_counts(rows: list[dict[str, object]]) -> Counter[tuple[str, float, str]]:
        return Counter(
            (
                str(row.get("filter_name") or ""),
                float(row.get("target_recall") or 0.0),
                str(row.get("mode") or ""),
            )
            for row in rows
        )

    selected_counts = key_counts(selected)
    final_counts = key_counts(final_rows)
    matrix_complete = bool(
        requested_formal_matrix
        and set(selected_counts) == expected_keys
        and all(count == 1 for count in selected_counts.values())
    )
    measurement_complete = bool(
        matrix_complete
        and not skip_final
        and set(final_counts) == expected_keys
        and all(count == 1 for count in final_counts.values())
        and all(
            row.get("final_status") == "complete"
            and row.get("rows_complete") is True
            and int(row.get("errors") or 0) == 0
            and row.get("target_confirmed_in_calibration") is True
            and row.get("target_confirmed_in_final") is True
            for row in final_rows
        )
    )
    comparison_valid = bool(
        measurement_complete
        and all(row.get("matched_recall_comparison_valid") is True for row in final_rows)
    )
    return {
        "expected_cells": len(expected_keys),
        "requested_formal_matrix": requested_formal_matrix,
        "selected_cells": len(selected),
        "selected_unique_cells": len(selected_counts),
        "final_cells": len(final_rows),
        "final_unique_cells": len(final_counts),
        "matrix_complete": matrix_complete,
        "measurement_complete": measurement_complete,
        "comparison_valid": comparison_valid,
        "status": (
            "complete"
            if matrix_complete and measurement_complete and comparison_valid
            else "incomplete"
        ),
    }


def final_eligible_rows(selected: list[dict[str, object]]) -> list[dict[str, object]]:
    stock_selected = {
        (float(row["target_recall"]), str(row["filter_name"])): row
        for row in selected
        if row["mode"] == "original" and row.get("selection_status") == "selected"
    }
    eligible: list[dict[str, object]] = []
    for key, stock in stock_selected.items():
        methods = [
            row
            for row in selected
            if (float(row["target_recall"]), str(row["filter_name"])) == key
            and row["mode"] != "original"
            and row.get("selection_status") == "selected"
        ]
        if methods:
            eligible.append(stock)
            eligible.extend(methods)
    return eligible


FinalResultKey = tuple[float, str, str, str]


def final_result_key(row: dict[str, object]) -> FinalResultKey:
    return (
        float(row["target_recall"]),
        str(row["filter_name"]),
        str(row["mode"]),
        str(row["config"]),
    )


def run_final_interleaved(
    selected: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[FinalResultKey, dict[str, object]]:
    results: dict[FinalResultKey, dict[str, object]] = {}
    by_target_filter: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in selected:
        by_target_filter.setdefault(
            (float(row["target_recall"]), str(row["filter_name"])),
            [],
        ).append(row)

    for schedule_no, ((target_recall, filter_name), target_rows) in enumerate(
        sorted(by_target_filter.items())
    ):
        mode_configs = {
            str(row["mode"]): config_from_row(row)
            for row in target_rows
        }
        modes = [mode for mode in args.modes if mode in mode_configs]
        signature = tuple((mode, mode_configs[mode].label) for mode in modes)
        digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
        target_token = format(target_recall, ".12g").replace(".", "p")
        schedule_id = f"target{target_token}_{filter_name}_{digest}"
        stem = (
            f"sigmod_matched_final_interleaved_{schedule_id}_"
            f"q{args.final_queries}r{args.final_repeats}_{args.run_spec_hash[:12]}_{args.tag}"
        )
        raw = RESULTS / f"{stem}.csv"
        log = RESULTS / "logs" / f"sigmod_matched_{args.tag}" / f"{stem}.log"
        summary = None
        if args.resume and raw.is_file() and raw.stat().st_size > 0:
            try:
                candidate = summarize_raw(
                    raw,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                    args.final_queries,
                    args.final_repeats,
                    getattr(args, "expected_truth_self_excluded", True),
                )
                require_plan_evidence(
                    raw,
                    explicit_candidate_validity_predicate(args),
                    getattr(args, "database_fingerprint", None),
                )
                if {str(row["mode"]) for row in candidate} == set(modes):
                    summary = candidate
            except (KeyError, TypeError, ValueError, csv.Error, RuntimeError):
                summary = None
        if summary is None:
            elapsed_ms = run_d123_interleaved(
                raw,
                filter_name,
                modes,
                mode_configs,
                args,
                log,
            )
            summary = summarize_raw(
                raw,
                args.bootstrap_samples,
                args.bootstrap_seed,
                args.final_queries,
                args.final_repeats,
                getattr(args, "expected_truth_self_excluded", True),
            )
        else:
            elapsed_ms = 0.0
            print(f"reusing {raw}", flush=True)

        plan_evidence = plan_evidence_manifest_entry(
            raw,
            explicit_candidate_validity_predicate(args),
            getattr(args, "database_fingerprint", None),
        )
        summaries = {str(row["mode"]): row for row in summary}
        if set(summaries) != set(modes):
            raise RuntimeError(f"expected modes {modes} in {raw}, got {sorted(summaries)}")
        for mode in modes:
            mode_summary = summaries[mode]
            if int(mode_summary["queries"]) != args.final_queries:
                raise RuntimeError(
                    f"final query split is incomplete for {mode} in {raw}: "
                    f"expected {args.final_queries}, got {mode_summary['queries']}"
                )
            config = mode_configs[mode]
            key = (target_recall, filter_name, mode, config.label)
            results[key] = {
                **mode_summary,
                "target_recall": target_recall,
                "final_elapsed_ms": elapsed_ms,
                "final_raw": str(raw),
                "final_raw_rows": csv_row_count(raw),
                "final_raw_sha256": sha256_file(raw),
                "final_log": str(log),
                "final_schedule": schedule_no,
                "final_schedule_id": schedule_id,
                "final_execution_order": "interleaved",
                "plan_evidence_file": plan_evidence["path"],
                "plan_evidence_sha256": plan_evidence["sha256"],
                "plan_checks": len(plan_evidence["checks"]),
                "plan_gate_passed": True,
            }
    return results


def run_final_unique(
    selected: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[FinalResultKey, dict[str, object]]:
    results: dict[FinalResultKey, dict[str, object]] = {}
    for row in selected:
        key = final_result_key(row)
        if key in results:
            continue
        target_recall, filter_name, mode, _ = key
        config = config_from_row(row)
        target_token = format(target_recall, ".12g").replace(".", "p")
        stem = (
            f"sigmod_matched_final_target{target_token}_{filter_name}_{mode}_{config.label}_"
            f"q{args.final_queries}r{args.final_repeats}_{args.run_spec_hash[:12]}_{args.tag}"
        )
        raw = RESULTS / f"{stem}.csv"
        log = RESULTS / "logs" / f"sigmod_matched_{args.tag}" / f"{stem}.log"
        summary = reusable_summary(
            raw,
            args.bootstrap_samples,
            args.bootstrap_seed,
            args.final_queries,
            args.final_repeats,
            getattr(args, "expected_truth_self_excluded", True),
        ) if args.resume else None
        if summary is not None:
            try:
                require_plan_evidence(
                    raw,
                    explicit_candidate_validity_predicate(args),
                    getattr(args, "database_fingerprint", None),
                )
            except RuntimeError:
                summary = None
        if summary is None:
            elapsed_ms = run_d123(
                raw,
            filter_name,
            mode,
            args.final_query_offset,
            args.final_queries,
                args.final_repeats,
                config,
                args,
                log,
            )
            summary = summarize_raw(
                raw,
                args.bootstrap_samples,
                args.bootstrap_seed,
                args.final_queries,
                args.final_repeats,
                getattr(args, "expected_truth_self_excluded", True),
            )
            if len(summary) != 1:
                raise RuntimeError(f"expected one final summary in {raw}, got {len(summary)}")
        else:
            elapsed_ms = 0.0
            print(f"reusing {raw}", flush=True)
        if int(summary[0]["queries"]) != args.final_queries:
            raise RuntimeError(
                f"final query split is incomplete in {raw}: "
                f"expected {args.final_queries}, got {summary[0]['queries']}"
            )
        plan_evidence = plan_evidence_manifest_entry(
            raw,
            explicit_candidate_validity_predicate(args),
            getattr(args, "database_fingerprint", None),
        )
        results[key] = {
            **summary[0],
            "target_recall": target_recall,
            "final_elapsed_ms": elapsed_ms,
            "final_raw": str(raw),
            "final_raw_rows": csv_row_count(raw),
            "final_raw_sha256": sha256_file(raw),
            "final_log": str(log),
            "final_schedule_id": f"target{target_token}_{filter_name}_{mode}",
            "final_execution_order": "mode_major",
            "plan_evidence_file": plan_evidence["path"],
            "plan_evidence_sha256": plan_evidence["sha256"],
            "plan_checks": len(plan_evidence["checks"]),
            "plan_gate_passed": True,
        }
    return results


def consolidate_final(
    selected: list[dict[str, object]],
    final_results: dict[FinalResultKey, dict[str, object]],
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260718,
    expected_truth_self_excluded: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selected_row in selected:
        key = final_result_key(selected_row)
        final = final_results.get(key)
        if final is None:
            rows.append(
                {
                    "target_recall": selected_row["target_recall"],
                    "target_met_in_calibration": selected_row["target_met_in_calibration"],
                    "target_confirmed_in_calibration": selected_row.get("target_confirmed_in_calibration", False),
                    "target_lcb95_met_in_calibration": selected_row.get("target_lcb95_met_in_calibration", False),
                    "target_met_in_final": False,
                    "target_confirmed_in_final": False,
                    "target_lcb95_met_in_final": False,
                    "filter_name": selected_row["filter_name"],
                    "mode": selected_row["mode"],
                    "guidance_filter_strategy": selected_row.get("guidance_filter_strategy", ""),
                    "config": selected_row["config"],
                    "ef_search": selected_row["ef_search"],
                    "guided_collect_target": selected_row["guided_collect_target"],
                    "max_scan_tuples": selected_row["max_scan_tuples"],
                    "scan_mem_multiplier": selected_row["scan_mem_multiplier"],
                    "iterative_scan": selected_row["iterative_scan"],
                    "feasibility_status": selected_row.get("feasibility_status", "grid_exhausted"),
                    "final_status": "not_run_infeasible",
                    "recall_mean": None,
                    "latency_mean_ms": None,
                    "errors": None,
                    "rows_complete": False,
                }
            )
            continue
        rows.append(
            {
                "target_recall": selected_row["target_recall"],
                "target_met_in_calibration": selected_row["target_met_in_calibration"],
                "target_confirmed_in_calibration": selected_row.get("target_confirmed_in_calibration", False),
                "target_lcb95_met_in_calibration": selected_row.get("target_lcb95_met_in_calibration", False),
                "target_met_in_final": float(final["recall_mean"]) >= float(selected_row["target_recall"]),
                "target_confirmed_in_final": float(final["recall_mean"]) >= float(selected_row["target_recall"]),
                "target_lcb95_met_in_final": float(final.get("recall_lcb95", final["recall_mean"])) >= float(selected_row["target_recall"]),
                "filter_name": selected_row["filter_name"],
                "mode": selected_row["mode"],
                "guidance_filter_strategy": selected_row.get("guidance_filter_strategy", ""),
                "config": selected_row["config"],
                "ef_search": selected_row["ef_search"],
                "guided_collect_target": selected_row["guided_collect_target"],
                "max_scan_tuples": selected_row["max_scan_tuples"],
                "scan_mem_multiplier": selected_row["scan_mem_multiplier"],
                "iterative_scan": selected_row["iterative_scan"],
                "feasibility_status": selected_row.get("feasibility_status", "feasible"),
                "final_status": "complete",
                **final,
            }
        )

    stock_rows = {
        (float(row["target_recall"]), str(row["filter_name"])): row
        for row in rows
        if row["mode"] == "original"
    }
    for row_no, row in enumerate(rows):
        stock = stock_rows.get((float(row["target_recall"]), str(row["filter_name"])))
        baseline = float(stock["latency_mean_ms"]) if stock and stock.get("latency_mean_ms") is not None else 0.0
        latency = float(row["latency_mean_ms"]) if row.get("latency_mean_ms") is not None else 0.0
        row["recall_delta_vs_stock"] = (
            float(row["recall_mean"]) - float(stock["recall_mean"])
            if stock and stock.get("recall_mean") is not None and row.get("recall_mean") is not None
            else None
        )
        valid = bool(
            stock
            and stock.get("target_confirmed_in_calibration")
            and row.get("target_confirmed_in_calibration")
            and stock.get("target_confirmed_in_final")
            and row.get("target_confirmed_in_final")
            and stock.get("rows_complete")
            and row.get("rows_complete")
            and int(stock.get("errors") or 0) == 0
            and int(row.get("errors") or 0) == 0
            and stock.get("recall_contract") == TIE_AWARE_RECALL_CONTRACT
            and row.get("recall_contract") == TIE_AWARE_RECALL_CONTRACT
            and stock.get("truth_self_excluded") is expected_truth_self_excluded
            and row.get("truth_self_excluded") is expected_truth_self_excluded
            and stock.get("plan_gate_passed") is True
            and row.get("plan_gate_passed") is True
        )
        row["paired_queries"] = 0
        row["paired_repeats"] = 0
        row["paired_samples"] = 0
        row["speedup_ci95_low"] = None
        row["speedup_ci95_high"] = None
        invalid_reason = "" if valid else "infeasible_incomplete_unconfirmed_or_unverified"
        if valid and stock:
            stock_raw = str(stock.get("final_raw") or "")
            method_raw = str(row.get("final_raw") or "")
            if not stock_raw or not method_raw:
                valid = False
                invalid_reason = "missing_final_raw"
            elif (
                stock.get("final_execution_order") == "interleaved"
                or row.get("final_execution_order") == "interleaved"
            ) and (
                stock.get("final_execution_order") != "interleaved"
                or row.get("final_execution_order") != "interleaved"
                or stock.get("final_schedule_id") != row.get("final_schedule_id")
                or Path(stock_raw) != Path(method_raw)
            ):
                valid = False
                invalid_reason = "cross_schedule_pairing_rejected"
        if valid and stock:
            paired_stats = paired_comparison_stats(
                Path(str(stock["final_raw"])),
                "original",
                Path(str(row["final_raw"])),
                str(row["mode"]),
                str(row["filter_name"]),
                bootstrap_samples,
                bootstrap_seed + row_no,
            )
            row.update(paired_stats)
            paired_queries = int(paired_stats.get("paired_queries", 0))
            paired_repeats = int(paired_stats.get("paired_repeats", 0))
            paired_samples = int(paired_stats.get("paired_samples", 0))
            expected_queries = int(row.get("expected_queries") or 0)
            expected_repeats = int(row.get("expected_repeats") or 0)
            if (
                expected_queries <= 0
                or expected_repeats <= 0
                or paired_queries != expected_queries
                or paired_repeats != expected_repeats
                or paired_samples != expected_queries * expected_repeats
            ):
                valid = False
                invalid_reason = "incomplete_paired_query_repeat_schedule"
                row["speedup_ci95_low"] = None
                row["speedup_ci95_high"] = None
        row["matched_recall_comparison_valid"] = valid
        row["invalid_reason"] = "" if valid else invalid_reason
        row["speedup_vs_stock"] = baseline / latency if valid and baseline and latency > 0 else None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently tune stock pgvector and every SQLens variant, then compare "
            "their fastest configurations at matched recall targets."
        )
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-recalls", default="0.90,0.95,0.99")
    parser.add_argument("--target-recall", type=float, help="Backward-compatible single recall target")
    parser.add_argument("--filters", nargs="*", default=FILTER_ORDER)
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--calibration-queries", type=int, default=100)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    parser.add_argument("--calibration-query-offset", type=int, default=0)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument("--final-query-offset", type=int)
    parser.add_argument(
        "--final-execution-order",
        choices=["interleaved", "mode_major"],
        default="interleaved",
        help="Interleave methods within each query/repeat to control cache and time drift.",
    )
    parser.add_argument("--schedule-seed", type=int, default=20260718)
    parser.add_argument("--allow-overlapping-query-splits", action="store_true")
    parser.add_argument("--ef-search-values", default=DENSE_12_EF_SEARCH)
    parser.add_argument("--guided-collect-target-values", default="ef")
    parser.add_argument(
        "--max-scan-tuples-values",
        default="5000000",
        help="Use a non-binding main-experiment ceiling; evaluate this knob in a separate sensitivity sweep.",
    )
    parser.add_argument(
        "--scan-mem-multiplier-values",
        default="32",
        help="Use one ample main-experiment budget; evaluate memory sensitivity separately.",
    )
    parser.add_argument("--iterative-scan-values", default="off")
    parser.add_argument(
        "--stock-iterative-scan-values",
        default="off,strict_order",
        help=(
            "Independent stock-pgvector tuning grid. SQLens traversal-safe modes still use "
            "--iterative-scan-values=off."
        ),
    )
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS_CSV)
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=DEFAULT_TRUTH_CSV,
    )
    parser.add_argument("--insertion-table", default=DEFAULT_INSERTION_TABLE)
    parser.add_argument("--insertion-index", default=DEFAULT_INSERTION_INDEX)
    parser.add_argument("--bfs-table", default=DEFAULT_BFS_TABLE)
    parser.add_argument("--bfs-index", default=DEFAULT_BFS_INDEX)
    parser.add_argument(
        "--query-table",
        help="External query relation. Omit to use the candidate table's id and embedding columns.",
    )
    parser.add_argument("--query-id-column", default="id")
    parser.add_argument("--query-vector-column", default="embedding")
    parser.add_argument(
        "--candidate-validity-predicate",
        type=validate_candidate_validity_predicate,
        default="",
        help=(
            "Global partial-index candidate predicate (for example embedding_valid). "
            "It is forwarded as a SQL/planner qual and never as a D1 guidance atom."
        ),
    )
    parser.add_argument(
        "--expected-truth-self-excluded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expected self_excluded value in the supplied exact-truth CSV.",
    )
    parser.add_argument(
        "--guidance-filter-strategy",
        default="traversal_guided",
        choices=["traversal_guided", "safe_guided", "guided_collect", "acorn1"],
        help=(
            "Formal D1 uses planner-proven traversal_guided with iterative_scan=off. "
            "safe_guided, guided_collect, and acorn1 remain diagnostic modes and must not be "
            "reported as traversal-safe D1."
        ),
    )
    parser.add_argument(
        "--guidance-selectivity-max-pct",
        type=float,
        default=100.0,
        help="Enable pure D1 across the full selectivity sweep; adaptive admission is evaluated separately.",
    )
    parser.add_argument("--guidance-max-atoms", type=int, default=64)
    parser.add_argument("--d2-page-access", default="off", choices=["off", "prefetch", "reorder"])
    parser.add_argument("--d2-index-page-access", default="off", choices=["off", "prefetch"])
    parser.add_argument(
        "--preferred-index-guc",
        default="hnsw.preferred_index",
        help=(
            "Set this planner preference GUC when the loaded SQLens build exposes it; "
            "the child runner still requires EXPLAIN to name the exact expected index."
        ),
    )
    parser.add_argument(
        "--require-preferred-index-guc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the SQLens preferred-index GUC for every formal calibration/final child run.",
    )
    parser.add_argument("--d1-cache-mb", type=int, default=1024)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument(
        "--backend-cpu-list",
        type=normalize_cpu_list,
        help=(
            "Expected DB-side PostgreSQL backend CPU affinity (for example 48-51). "
            "Host PID pinning must be performed by trustworthy external orchestration."
        ),
    )
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--statement-timeout-ms", type=int, default=300000)
    parser.add_argument("--progress-queries", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()
    args.candidate_validity_predicate_explicit = (
        "--candidate-validity-predicate" in sys.argv
    )

    if args.guidance_filter_strategy == "safe_guided":
        print("safe_guided preserves stock termination; guided_collect_target is fixed to 1 and ignored", flush=True)

    if args.final_query_offset is None:
        args.final_query_offset = args.calibration_query_offset + args.calibration_queries
    calibration_range = range(
        args.calibration_query_offset,
        args.calibration_query_offset + args.calibration_queries,
    )
    final_range = range(args.final_query_offset, args.final_query_offset + args.final_queries)
    if not args.allow_overlapping_query_splits and set(calibration_range).intersection(final_range):
        raise SystemExit("calibration and final query splits overlap")

    started_at = utc_now()
    manifest_out = RESULTS / f"sigmod_matched_recall_manifest_failed_{args.tag}.json"
    manifest: dict[str, object] = {
        "status": "running",
        "tag": args.tag,
        "argv": sys.argv,
        "started_at": started_at,
        "completed_at": None,
        "timestamps": {"started_at": started_at},
        "outputs": {},
        "plan_evidence": [],
        "matrix_complete": False,
        "measurement_complete": False,
        "comparison_valid": False,
        "error": None,
    }
    calibration_rows: list[dict[str, object]] = []
    calibration_pairs: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    final_results: dict[FinalResultKey, dict[str, object]] = {}
    guard_connection: psycopg.Connection | None = None
    try:
        tracking_evidence = prepare_fragment_tracking(args)
        args.fragment_tracking_prepared = bool(tracking_evidence["prepared"])
        args.fragment_tracking_evidence = tracking_evidence
        manifest["fragment_tracking_preparation"] = tracking_evidence
        guard_connection, guard_evidence = acquire_formal_data_guard(args)
        manifest["formal_data_guard"] = guard_evidence
        run_spec = build_run_spec(args)
        args.d2_graph_proof = run_spec["d2_graph_proof"]
        args.sqlens_runtime_provenance = run_spec["sqlens_runtime_provenance"]
        args.database_fingerprint = run_spec["database"]
        args.run_spec_hash = str(run_spec["run_spec_hash"])
        run_prefix = f"{args.run_spec_hash[:12]}_{args.tag}"
        manifest_out = RESULTS / f"sigmod_matched_recall_manifest_{run_prefix}.json"
        targets = [args.target_recall] if args.target_recall is not None else parse_targets(args.target_recalls)
        configs = build_configs(args)
        mode_grids = mode_calibration_grids(
            configs,
            args.modes,
            args.stock_iterative_scan_values,
        )
        calibration_out = RESULTS / f"sigmod_matched_recall_calibration_{run_prefix}.csv"
        selected_out = RESULTS / f"sigmod_matched_recall_selected_{run_prefix}.csv"
        final_out = RESULTS / f"sigmod_matched_recall_final_{run_prefix}.csv"
        manifest.update(
            {
                "run_spec_hash": args.run_spec_hash,
                "run_spec": run_spec,
                "sqlens_runtime_provenance": run_spec["sqlens_runtime_provenance"],
                "targets": targets,
                "filters": args.filters,
                "modes": args.modes,
                "calibration_queries": args.calibration_queries,
                "calibration_repeats": args.calibration_repeats,
                "calibration_query_offset": args.calibration_query_offset,
                "final_queries": args.final_queries,
                "final_repeats": args.final_repeats,
                "final_query_offset": args.final_query_offset,
                "final_execution_order": args.final_execution_order,
                "schedule_seed": args.schedule_seed,
                "recall_contract": TIE_AWARE_RECALL_CONTRACT,
                "self_excluded": args.expected_truth_self_excluded,
                "config_count": len(configs),
                "configs": [asdict(config) for config in configs],
                "mode_grids": mode_grids,
                "mode_grid_counts": {
                    mode: len(grid) for mode, grid in mode_grids.items()
                },
                "calibration_policy": {
                    "execution_order": "seeded_interleaved_family_ef_blocks",
                    "schedule_seed": args.schedule_seed,
                    "stop_condition": (
                        "independently per iterative-scan family, after every distinct configuration "
                        "in an ef_search block completes without errors, stop that family when any "
                        "mean Recall@10 reaches the highest requested target"
                    ),
                    "selection": "lowest mean latency among complete error-free configurations with mean Recall@10 at or above target; bootstrap CI/LCB is report-only",
                    "unattainable_condition": "full grid evaluated with every block complete and error-free",
                },
                "calibration_pairs": calibration_pairs,
            }
        )
        write_json_atomic(manifest_out, manifest)

        manifest["timestamps"]["calibration_started_at"] = utc_now()  # type: ignore[index]
        write_json_atomic(manifest_out, manifest)
        for filter_name in args.filters:
            for mode in args.modes:
                print(f"calibrating filter={filter_name} mode={mode}", flush=True)
                pair_rows, pair_evidence = calibrate_mode_filter(
                    filter_name,
                    mode,
                    configs,
                    args,
                    targets,
                )
                calibration_rows.extend(pair_rows)
                calibration_pairs.append(pair_evidence)
        write_csv(calibration_out, calibration_rows)
        selected = selected_rows(calibration_rows, args.filters, args.modes, targets)
        write_csv(selected_out, selected)
        manifest["timestamps"]["calibration_completed_at"] = utc_now()  # type: ignore[index]
        manifest["outputs"] = {
            "calibration": output_artifact(calibration_out),
            "selected": output_artifact(selected_out),
        }
        calibration_raws = sorted({Path(str(row["raw"])) for row in calibration_rows})
        manifest["plan_evidence"] = [
            plan_evidence_manifest_entry(
                path,
                explicit_candidate_validity_predicate(args),
                getattr(args, "database_fingerprint", None),
            )
            for path in calibration_raws
        ]
        write_json_atomic(manifest_out, manifest)

        if not args.skip_final:
            manifest["timestamps"]["final_started_at"] = utc_now()  # type: ignore[index]
            write_json_atomic(manifest_out, manifest)
            eligible = final_eligible_rows(selected)
            if args.final_execution_order == "interleaved":
                final_results = run_final_interleaved(eligible, args)
            else:
                final_results = run_final_unique(eligible, args)
            final_rows = consolidate_final(
                selected,
                final_results,
                args.bootstrap_samples,
                args.bootstrap_seed,
                args.expected_truth_self_excluded,
            )
            write_csv(final_out, final_rows)
            manifest["timestamps"]["final_completed_at"] = utc_now()  # type: ignore[index]
            manifest["outputs"]["final"] = output_artifact(final_out)  # type: ignore[index]
            final_raws = sorted(
                {Path(str(result["final_raw"])) for result in final_results.values()}
            )
            manifest["plan_evidence"] = [
                plan_evidence_manifest_entry(
                    path,
                    explicit_candidate_validity_predicate(args),
                    getattr(args, "database_fingerprint", None),
                )
                for path in sorted(set(calibration_raws + final_raws))
            ]

        if formal_run_uses_d2(args):
            final_d2_proof = d2_graph_proof_from_env(args)
            if (
                final_d2_proof.get("stable_fingerprint_sha256")
                != args.d2_graph_proof.get("stable_fingerprint_sha256")
            ):
                raise RuntimeError(
                    "D2 parent final graph proof changed during the guarded formal run"
                )
            manifest["d2_graph_proof_final"] = final_d2_proof

        completion = formal_completion_gate(
            args.filters,
            args.modes,
            targets,
            selected,
            final_rows,
            args.skip_final,
        )
        manifest.update(completion)
        manifest["status"] = completion["status"]
        manifest["completed_at"] = utc_now()
        manifest["timestamps"]["completed_at"] = manifest["completed_at"]  # type: ignore[index]
        write_json_atomic(manifest_out, manifest)
        print(f"wrote {calibration_out}", flush=True)
        print(f"wrote {selected_out}", flush=True)
        print(f"wrote {manifest_out}", flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["timestamps"]["completed_at"] = manifest["completed_at"]  # type: ignore[index]
        manifest["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        write_json_atomic(manifest_out, manifest)
        raise
    finally:
        if guard_connection is not None:
            guard_connection.rollback()
            guard_connection.close()


if __name__ == "__main__":
    main()
