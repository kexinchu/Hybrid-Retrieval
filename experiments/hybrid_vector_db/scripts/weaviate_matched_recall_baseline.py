"""Formal matched-recall baseline for the Amazon-10M Weaviate workload.

The runner deliberately keeps setup, calibration, and final measurements separate.
HNSW ``ef`` is a schema setting: it is changed once per configuration and gated by
a schema read-back before any query is measured.  Query latency includes the full
HTTP request, response JSON parsing, and row_id extraction.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CLASS_NAME = "AmazonGroceryReview"
EXPECTED_ROWS = 10_000_000
DEFAULT_EF_VALUES = (100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
DEFAULT_TARGETS = (0.90, 0.95, 0.99)
CALIBRATION_QUERY_NOS = tuple(range(20, 100))
FINAL_QUERY_NOS = tuple(range(100, 200))
CALIBRATION_REPEATS = 2
FINAL_REPEATS = 5
BOOTSTRAP_SAMPLES = 10_000
K = 10
NA = "N/A"
TARGET_SELECTION_RULE = "query-level mean recall@10 >= target; bootstrap CI/LCB reporting only"
ACORN_REPORTED_STRATEGY = "acorn_configured_auto_fallback"
ACORN_RATIO_ENV = "HNSW_ACORN_FILTER_RATIO"
DEFAULT_FILTERS = ROOT / "experiments/hybrid_vector_db/configs/amazon10m_selectivity14_valid_embeddings_filters.csv"
DEFAULT_TRUTH = ROOT / "results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_valid_embeddings_formal.csv"
DEFAULT_FBIN = ROOT / "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"
DEFAULT_OUT = ROOT / "results/hybrid_vector_db/weaviate_matched_recall_baseline.csv"


@dataclass(frozen=True)
class FilterSpec:
    target_rate: str
    name: str
    predicate: str
    expected_rows: int
    actual_pct: float
    where: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.where:
            object.__setattr__(
                self, "where", with_candidate_universe(predicate_to_where(self.predicate))
            )


@dataclass(frozen=True)
class TruthEntry:
    query_no: int
    query_id: int
    filter_name: str
    split: str
    filtered_rows: int
    k: int
    kth_distance_sq: float
    tie_tolerance: float
    self_excluded: bool


@dataclass(frozen=True)
class QueryResult:
    ids: tuple[int, ...]
    latency_ms: float
    retry_count: int
    order_error: str
    error: str
    returned_count: int = 0
    request_limit: int = K + 1


EXPECTED_FILTERS = {
    "popular_ge1000": ("item_rating_number >= 1000", 5_019_997),
    "popular_ge1340": ("item_rating_number >= 1340", 4_489_429),
    "popular_ge1780": ("item_rating_number >= 1780", 3_990_500),
    "popular_ge2428": ("item_rating_number >= 2428", 3_491_010),
    "popular_ge3284": ("item_rating_number >= 3284", 2_992_481),
    "popular_ge4559": ("item_rating_number >= 4559", 2_493_459),
    "price_10_to_20": ("has_price AND price > 10 AND price <= 20", 2_184_326),
    "popular_ge10066": ("item_rating_number >= 10066", 1_496_628),
    "rating5_price_le10": ("has_price AND price <= 10 AND rating = 5", 955_625),
    "long_review_ge500": ("review_text_len >= 500", 588_018),
    "grocery_rating5": ("main_category = 'Grocery' AND rating = 5", 233_510),
    "grocery_helpful": ("main_category = 'Grocery' AND helpful_vote >= 1", 101_387),
    "helpful_ge20": ("helpful_vote >= 20", 60_683),
    "grocery_long500": ("main_category = 'Grocery' AND review_text_len >= 500", 21_317),
}

PROPERTY_TYPES = {
    "row_id": "int",
    "rating": "number",
    "verified_purchase": "boolean",
    "helpful_vote": "int",
    "review_text_len": "int",
    "main_category": "text",
    "price": "number",
    "has_price": "boolean",
    "item_rating_number": "int",
    "embedding_valid": "boolean",
}

EXPECTED_VALID_ROWS = 9_979_556
CANDIDATE_VALIDITY_PREDICATE = "embedding_valid"


def with_candidate_universe(where: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator": "And",
        "operands": [
            copy.deepcopy(where),
            {"path": ["embedding_valid"], "operator": "Equal", "valueBoolean": True},
        ],
    }


def _tokenize_predicate(value: str) -> list[str]:
    import re

    token_re = re.compile(r"\s*(>=|<=|=|>|<|\(|\)|\bAND\b|'[^']*'|\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)")
    tokens: list[str] = []
    position = 0
    while position < len(value):
        match = token_re.match(value, position)
        if not match:
            raise ValueError(f"unsupported predicate near {value[position:]!r}")
        tokens.append(match.group(1))
        position = match.end()
    return tokens


class _PredicateParser:
    def __init__(self, value: str) -> None:
        self.tokens = _tokenize_predicate(value)
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or expected is not None and token != expected:
            raise ValueError(f"expected {expected or 'token'}, got {token!r}")
        self.position += 1
        return token

    def expression(self) -> tuple[Any, ...]:
        parts = [self.term()]
        while self.peek() == "AND":
            self.take("AND")
            parts.append(self.term())
        if len(parts) == 1:
            return parts[0]
        return ("and", tuple(parts))

    def term(self) -> tuple[Any, ...]:
        if self.peek() == "(":
            self.take("(")
            value = self.expression()
            self.take(")")
            return value
        field = self.take()
        if self.peek() in {"AND", ")", None}:
            if PROPERTY_TYPES.get(field) != "boolean":
                raise ValueError(f"bare predicate is only valid for booleans: {field!r}")
            return ("comparison", field, "=", True)
        operator = self.take()
        if operator not in {">=", "<=", ">", "<", "="}:
            raise ValueError(f"unsupported comparison operator {operator!r}")
        raw = self.take()
        if raw.startswith("'"):
            parsed: Any = raw[1:-1]
        elif PROPERTY_TYPES.get(field) == "boolean":
            if raw not in {"True", "False", "true", "false"}:
                raise ValueError(f"invalid boolean literal {raw!r}")
            parsed = raw.lower() == "true"
        elif PROPERTY_TYPES.get(field) == "number":
            parsed = float(raw)
        else:
            parsed = int(raw)
        return ("comparison", field, operator, parsed)

    def parse(self) -> tuple[Any, ...]:
        result = self.expression()
        if self.peek() is not None:
            raise ValueError(f"unexpected predicate token {self.peek()!r}")
        return result


GRAPHQL_OPERATORS = {
    ">=": "GreaterThanEqual",
    "<=": "LessThanEqual",
    ">": "GreaterThan",
    "<": "LessThan",
    "=": "Equal",
}
GRAPHQL_VALUE_KEYS = {"int": "valueInt", "number": "valueNumber", "boolean": "valueBoolean", "text": "valueText"}


def predicate_ast(predicate: str) -> tuple[Any, ...]:
    """Parse the CSV's small SQL predicate language into a typed AST."""
    return _PredicateParser(predicate).parse()


def _ast_to_where(ast: tuple[Any, ...]) -> dict[str, Any]:
    if ast[0] == "and":
        return {"operator": "And", "operands": [_ast_to_where(item) for item in ast[1]]}
    _, field, operator, value = ast
    if field not in PROPERTY_TYPES:
        raise ValueError(f"predicate references unknown property {field!r}")
    return {
        "path": [field],
        "operator": GRAPHQL_OPERATORS[operator],
        GRAPHQL_VALUE_KEYS[PROPERTY_TYPES[field]]: value,
    }


def predicate_to_where(predicate: str) -> dict[str, Any]:
    return _ast_to_where(predicate_ast(predicate))


def builtin_filter_specs() -> tuple[FilterSpec, ...]:
    specs: list[FilterSpec] = []
    for name, (predicate, expected_rows) in EXPECTED_FILTERS.items():
        specs.append(
            FilterSpec(
                target_rate=NA,
                name=name,
                predicate=predicate,
                expected_rows=expected_rows,
                actual_pct=expected_rows / EXPECTED_ROWS * 100.0,
                where=with_candidate_universe(predicate_to_where(predicate)),
            )
        )
    return tuple(specs)


FILTERS = builtin_filter_specs()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_filter_specs(path: Path = DEFAULT_FILTERS) -> tuple[FilterSpec, ...]:
    rows = read_csv(path)
    if len(rows) != len(EXPECTED_FILTERS):
        raise ValueError(f"expected 14 filter rows, got {len(rows)}")
    expected_names = set(EXPECTED_FILTERS)
    specs: list[FilterSpec] = []
    for row in rows:
        name = row["filter_name"]
        if name not in expected_names or name in {spec.name for spec in specs}:
            raise ValueError(f"unexpected or duplicate filter_name: {name}")
        predicate, expected_rows = EXPECTED_FILTERS[name]
        if row["predicate"] != predicate or int(row["count"]) != expected_rows:
            raise ValueError(f"filter config mismatch for {name}")
        specs.append(
            FilterSpec(
                target_rate=row["target_rate"],
                name=name,
                predicate=predicate,
                expected_rows=expected_rows,
                actual_pct=float(row["actual_pct"]),
                where=with_candidate_universe(predicate_to_where(predicate)),
            )
        )
    if {spec.name for spec in specs} != expected_names:
        raise ValueError("filter config is incomplete")
    return tuple(specs)


TRUTH_REQUIRED_FIELDS = {
    "filtered_rows",
    "k",
    "kth_distance_sq",
    "tie_tolerance",
    "self_excluded",
    "candidate_validity_predicate",
    "query_validity_predicate",
    "candidate_rows",
}


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_truth_csv_row(row: dict[str, str], k: int = K) -> TruthEntry | None:
    """Decode the tie-aware truth contract in one schema-localized function."""
    if row.get("method") != "pre_filter_exact":
        return None
    missing = TRUTH_REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"truth CSV uses retired legacy schema; missing tie-aware fields: {sorted(missing)}")
    truth_k = int(row["k"])
    if truth_k != k:
        raise ValueError(f"truth k mismatch: expected={k} actual={truth_k}")
    self_excluded = parse_bool(row["self_excluded"])
    if not self_excluded:
        raise ValueError("truth rows must have self_excluded=true")
    if (
        row.get("candidate_validity_predicate") != CANDIDATE_VALIDITY_PREDICATE
        or row.get("query_validity_predicate") != CANDIDATE_VALIDITY_PREDICATE
    ):
        raise ValueError("truth row candidate/query universe is not embedding_valid")
    filtered_rows = int(row["filtered_rows"])
    if int(row["candidate_rows"]) != filtered_rows:
        raise ValueError("truth row candidate_rows/filtered_rows mismatch")
    kth_distance_sq = float(row["kth_distance_sq"])
    tie_tolerance = float(row["tie_tolerance"])
    if filtered_rows < k:
        raise ValueError(f"truth filtered_rows must be at least k={k}")
    if not math.isfinite(kth_distance_sq) or kth_distance_sq < 0:
        raise ValueError("truth kth_distance_sq must be finite and non-negative")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("truth tie_tolerance must be finite and non-negative")
    return TruthEntry(
        query_no=int(row["query_no"]),
        query_id=int(row["query_id"]),
        filter_name=row["filter_name"],
        split=row.get("query_split", ""),
        filtered_rows=filtered_rows,
        k=truth_k,
        kth_distance_sq=kth_distance_sq,
        tie_tolerance=tie_tolerance,
        self_excluded=self_excluded,
    )


def load_truth(
    path: Path,
    filters: Sequence[FilterSpec] = FILTERS,
    calibration_query_nos: Sequence[int] = CALIBRATION_QUERY_NOS,
    final_query_nos: Sequence[int] = FINAL_QUERY_NOS,
    k: int = K,
) -> tuple[dict[tuple[str, int], TruthEntry], dict[int, int]]:
    calibration = set(calibration_query_nos)
    final = set(final_query_nos)
    if calibration & final:
        raise ValueError("calibration and final query_no sets overlap")
    expected_query_nos = calibration | final
    names = {spec.name for spec in filters}
    truth: dict[tuple[str, int], TruthEntry] = {}
    query_ids: dict[int, int] = {}
    rows = read_csv(path)
    available_fields = set(rows[0]) if rows else set()
    missing_fields = TRUTH_REQUIRED_FIELDS - available_fields
    if missing_fields:
        raise ValueError(
            "truth CSV uses retired legacy schema; missing tie-aware fields: "
            f"{sorted(missing_fields)}"
        )
    for row in rows:
        entry = parse_truth_csv_row(row, k)
        if entry is None:
            continue
        query_no = entry.query_no
        name = entry.filter_name
        if query_no not in expected_query_nos or name not in names:
            continue
        query_id = entry.query_id
        if query_no in query_ids and query_ids[query_no] != query_id:
            raise ValueError(f"query_no={query_no} maps to multiple query IDs")
        query_ids[query_no] = query_id
        split = entry.split or ("calibration" if query_no in calibration else "final")
        expected_split = "calibration" if query_no in calibration else "final"
        if split != expected_split:
            raise ValueError(f"query_no={query_no} has split={split!r}, expected {expected_split!r}")
        key = (name, query_no)
        if key in truth:
            raise ValueError(f"duplicate truth pair {key}")
        truth[key] = TruthEntry(
            query_no=entry.query_no,
            query_id=entry.query_id,
            filter_name=entry.filter_name,
            split=split,
            filtered_rows=entry.filtered_rows,
            k=entry.k,
            kth_distance_sq=entry.kth_distance_sq,
            tie_tolerance=entry.tie_tolerance,
            self_excluded=entry.self_excluded,
        )
    expected_pairs = {(spec.name, qno) for spec in filters for qno in expected_query_nos}
    missing = expected_pairs - set(truth)
    if missing or set(query_ids) != expected_query_nos:
        raise ValueError(f"truth grid incomplete: missing_pairs={len(missing)} missing_query_ids={sorted(expected_query_nos - set(query_ids))}")
    calibration_ids = {query_ids[qno] for qno in calibration}
    final_ids = {query_ids[qno] for qno in final}
    if len(calibration_ids) != len(calibration) or len(final_ids) != len(final) or calibration_ids & final_ids:
        raise ValueError("query IDs are not unique and disjoint across splits")
    for spec in filters:
        counts = {truth[(spec.name, qno)].filtered_rows for qno in expected_query_nos}
        if counts != {spec.expected_rows}:
            raise ValueError(f"truth candidate count mismatch for {spec.name}: {sorted(counts)}")
    return truth, query_ids


def read_fbin_memmap(path: Path, expected_rows: int = EXPECTED_ROWS) -> tuple[np.memmap, int, int]:
    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid fbin header: {path}")
    rows, dimensions = struct.unpack("<ii", header)
    if rows != expected_rows:
        raise ValueError(f"fbin row count mismatch: expected={expected_rows} actual={rows}")
    mapped = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dimensions))
    return mapped, rows, dimensions


def json_to_graphql(value: Any, key: str | None = None) -> str:
    if isinstance(value, dict):
        return "{" + " ".join(f"{name}:{json_to_graphql(item, name)}" for name, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(json_to_graphql(item) for item in value) + "]"
    if isinstance(value, str):
        return value if key == "operator" else json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def graphql_query(class_name: str, vector: Sequence[float], where: dict[str, Any], limit: int) -> str:
    vector_text = json_to_graphql(np.asarray(vector, dtype=np.float32).tolist())
    return (
        "{ Get { "
        f"{class_name}(nearVector:{{vector:{vector_text}}} "
        f"where:{json_to_graphql(where)} limit:{int(limit)}) "
        "{ row_id _additional { distance id } } } }"
    )


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str = "GET",
    timeout: float = 60.0,
    retries: int = 2,
    backoff_seconds: float = 0.25,
) -> tuple[dict[str, Any], int]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                base_url.rstrip("/") + path,
                data=body,
                headers={"Content-Type": "application/json"} if body is not None else {},
                method=method,
            )
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return (json.loads(raw.decode("utf-8")) if raw else {}), attempt
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if isinstance(exc, HTTPError):
                try:
                    response_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    response_body = ""
                detail = response_body.strip()
                if len(detail) > 4_000:
                    detail = detail[:4_000] + "..."
                last_error = RuntimeError(
                    f"HTTP {exc.code} {exc.reason}; response={detail or '<empty>'}"
                )
            else:
                last_error = exc
            if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code not in {408, 429}:
                break
            if attempt < retries:
                time.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last_error}") from last_error


def graphql(base_url: str, query: str, **kwargs: Any) -> tuple[dict[str, Any], int]:
    return request_json(base_url, "/v1/graphql", {"query": query}, method="POST", **kwargs)


def _schema_property_map(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(prop.get("name")): prop for prop in schema.get("properties", []) if isinstance(prop, dict)}


def schema_gate(schema: dict[str, Any], strategy: str | None = None, ef: int | None = None) -> list[str]:
    errors: list[str] = []
    if schema.get("class") not in (None, CLASS_NAME):
        errors.append(f"class={schema.get('class')!r}")
    if str(schema.get("vectorIndexType", "")).lower() != "hnsw":
        errors.append("vectorIndexType must be hnsw")
    config = schema.get("vectorIndexConfig") or {}
    if str(config.get("distance", "")).lower() != "l2-squared":
        errors.append("distance must be l2-squared")
    if strategy is not None:
        try:
            cutoff = int(config.get("flatSearchCutoff", -1))
        except (TypeError, ValueError):
            cutoff = -1
        if cutoff != 0:
            errors.append("flatSearchCutoff must be 0")
        if str(config.get("filterStrategy", "")).lower() != strategy.lower():
            errors.append(f"filterStrategy must be {strategy}")
    try:
        actual_ef = int(config.get("ef", -1))
    except (TypeError, ValueError):
        actual_ef = -1
    if ef is not None and actual_ef != int(ef):
        errors.append(f"ef read-back mismatch: expected={ef} actual={config.get('ef')}")
    properties = _schema_property_map(schema)
    for name, data_type in PROPERTY_TYPES.items():
        prop = properties.get(name)
        if prop is None:
            errors.append(f"missing property {name}")
            continue
        types = prop.get("dataType", [])
        if isinstance(types, str):
            types = [types]
        if not types or str(types[0]).lower() != data_type:
            errors.append(f"property {name} dataType mismatch")
        if prop.get("indexFilterable") is not True:
            errors.append(f"property {name} must be indexFilterable")
    return errors


def verify_schema(schema: dict[str, Any], strategy: str | None = None, ef: int | None = None) -> dict[str, Any]:
    errors = schema_gate(schema, strategy, ef)
    if errors:
        raise RuntimeError("schema gate failed: " + "; ".join(errors))
    return schema


def verify_node_ready(nodes_payload: dict[str, Any]) -> dict[str, Any]:
    nodes = nodes_payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError("Weaviate node gate failed: no nodes returned")
    relevant_shards: list[dict[str, Any]] = []
    errors: list[str] = []
    for node in nodes:
        name = str(node.get("name", "<unnamed>"))
        if str(node.get("status", "")).upper() != "HEALTHY":
            errors.append(f"node {name} status={node.get('status')!r}")
        if str(node.get("operationalMode", "")).lower() != "readwrite":
            errors.append(f"node {name} operationalMode={node.get('operationalMode')!r}")
        shards = node.get("shards")
        if not isinstance(shards, list):
            errors.append(f"node {name} did not return verbose shard state")
            continue
        for shard in shards:
            if shard.get("class") != CLASS_NAME:
                continue
            relevant_shards.append(shard)
            if shard.get("loaded") is not True:
                errors.append(f"shard {shard.get('name')} is not loaded")
            if str(shard.get("vectorIndexingStatus", "")).upper() != "READY":
                errors.append(
                    f"shard {shard.get('name')} vectorIndexingStatus="
                    f"{shard.get('vectorIndexingStatus')!r}"
                )
            if int(shard.get("vectorQueueLength", -1)) != 0:
                errors.append(f"shard {shard.get('name')} has a nonzero vector queue")
    if not relevant_shards:
        errors.append(f"no shard found for class {CLASS_NAME}")
    if errors:
        raise RuntimeError("Weaviate node gate failed: " + "; ".join(errors))
    return nodes_payload


def get_ready_nodes(base_url: str, timeout: float, retries: int) -> tuple[dict[str, Any], int]:
    payload, retry_count = request_json(
        base_url,
        f"/v1/nodes/{CLASS_NAME}?output=verbose",
        timeout=timeout,
        retries=retries,
    )
    return verify_node_ready(payload), retry_count


def _verify_full_schema_readback(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if actual != expected:
        expected_keys = sorted(expected)
        actual_keys = sorted(actual)
        raise RuntimeError(
            "full schema read-back mismatch: "
            f"expected_keys={expected_keys} actual_keys={actual_keys}"
        )


def put_schema_definition(
    base_url: str,
    definition: dict[str, Any],
    *,
    timeout: float,
    retries: int,
    strategy: str | None = None,
    ef: int | None = None,
) -> tuple[dict[str, Any], int]:
    _, put_retries = request_json(
        base_url,
        f"/v1/schema/{CLASS_NAME}",
        definition,
        method="PUT",
        timeout=timeout,
        retries=retries,
    )
    readback, read_retries = request_json(
        base_url, f"/v1/schema/{CLASS_NAME}", timeout=timeout, retries=retries
    )
    verify_schema(readback, strategy, ef)
    _verify_full_schema_readback(definition, readback)
    return readback, put_retries + read_retries


def put_hnsw_config(base_url: str, strategy: str, ef: int, timeout: float, retries: int) -> tuple[dict[str, Any], float, int]:
    started = time.perf_counter()
    current, get_retries = request_json(base_url, f"/v1/schema/{CLASS_NAME}", timeout=timeout, retries=retries)
    updated = copy.deepcopy(current)
    config = dict(updated.get("vectorIndexConfig") or {})
    config.update({"ef": int(ef), "flatSearchCutoff": 0, "filterStrategy": strategy})
    updated["vectorIndexConfig"] = config
    readback, put_retries = put_schema_definition(
        base_url,
        updated,
        timeout=timeout,
        retries=retries,
        strategy=strategy,
        ef=ef,
    )
    return readback, (time.perf_counter() - started) * 1000.0, get_retries + put_retries


def query_once(
    base_url: str,
    vector: Sequence[float],
    where: dict[str, Any],
    query_id: int,
    k: int = K,
    *,
    timeout: float = 60.0,
    retries: int = 0,
) -> QueryResult:
    started = time.perf_counter()
    try:
        if k != K:
            raise ValueError(f"formal runner requires k={K}, got {k}")
        request_limit = k + 1
        data, retry_count = graphql(base_url, graphql_query(CLASS_NAME, vector, where, request_limit), timeout=timeout, retries=retries)
        if data.get("errors"):
            raise RuntimeError(json.dumps(data["errors"], sort_keys=True))
        objects = data["data"]["Get"][CLASS_NAME]
        raw_ids = tuple(int(obj["row_id"]) for obj in objects)
        service_distances = [float(obj["_additional"]["distance"]) for obj in objects]
        if len(raw_ids) > request_limit:
            raise RuntimeError(
                f"oversized result: limit={request_limit} actual={len(raw_ids)}"
            )
        if len(set(raw_ids)) != len(raw_ids):
            raise RuntimeError("duplicate row_id in result")
        order_error = ""
        if any(not math.isfinite(distance) for distance in service_distances):
            order_error = "non-finite distance"
        elif any(left > right + 1e-6 for left, right in zip(service_distances, service_distances[1:])):
            order_error = "results are not ordered by ascending distance"
        ids = tuple(row_id for row_id in raw_ids if row_id != query_id)[:k]
        # A filtered ANN search may legitimately return fewer than k objects
        # even when the predicate has ample matches.  That is retrieval
        # quality evidence and must lower recall; treating it as a transport
        # error would discard exactly the baseline failure we need to measure.
        return QueryResult(
            ids, (time.perf_counter() - started) * 1000.0, retry_count, order_error, "",
            returned_count=len(raw_ids), request_limit=request_limit,
        )
    except Exception as exc:
        return QueryResult((), (time.perf_counter() - started) * 1000.0, 0, "", f"{exc.__class__.__name__}: {exc}")


def exact_squared_l2(
    vectors: np.ndarray, query_id: int, result_ids: Sequence[int]
) -> tuple[float, ...]:
    if query_id < 0 or query_id >= len(vectors):
        raise ValueError(f"query_id outside fbin: {query_id}")
    ids = np.asarray(result_ids, dtype=np.int64)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= len(vectors)):
        raise ValueError("result row_id outside fbin")
    query = np.asarray(vectors[query_id], dtype=np.float32)
    candidates = np.asarray(vectors[ids], dtype=np.float32)
    differences = candidates - query
    distances = np.einsum("ij,ij->i", differences, differences)
    return tuple(float(value) for value in distances)


def tie_aware_recall(
    result_distances_sq: Sequence[float], truth: TruthEntry, k: int = K
) -> float:
    if k != K or truth.k != K:
        raise ValueError(f"formal runner requires k={K}")
    denominator = min(k, truth.filtered_rows)
    if denominator <= 0:
        return 0.0
    threshold = truth.kth_distance_sq + truth.tie_tolerance
    valid = sum(
        math.isfinite(float(distance)) and float(distance) <= threshold
        for distance in result_distances_sq[:k]
    )
    return min(k, valid) / denominator


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(list(values), k=len(values))) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_recall_stats(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) == 1 or samples <= 0:
        return mean, mean, mean, mean
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(list(values), k=len(values))) for _ in range(samples)]
    return mean, percentile(means, 0.05), percentile(means, 0.025), percentile(means, 0.975)


def _query_level_means(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["query_no"]), []).append(float(row[field]))
    return [statistics.fmean(grouped[key]) for key in sorted(grouped)]


def reported_strategy(configured_strategy: str) -> str:
    return ACORN_REPORTED_STRATEGY if configured_strategy == "acorn" else configured_strategy


def summarize_configuration(
    rows: Sequence[dict[str, Any]],
    *,
    strategy: str,
    filter_name: str,
    ef: int,
    query_nos: Sequence[int],
    repeats: int,
    bootstrap_seed: int,
    phase: str = "calibration",
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("phase") == phase
        and row.get("configured_filter_strategy") == strategy
        and row.get("filter_name") == filter_name
        and int(row.get("ef", -1)) == ef
    ]
    expected = {(int(query_no), repeat) for query_no in query_nos for repeat in range(repeats)}
    observed: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in selected:
        key = (int(row["query_no"]), int(row["repeat"]))
        if key in observed:
            duplicates += 1
        observed[key] = row
    valid = [
        row
        for row in observed.values()
        if row.get("valid") is True
        and not row.get("error")
        and not row.get("order_error")
        and int(row.get("retry_count", -1)) == 0
    ]
    complete = set(observed) == expected and duplicates == 0 and len(valid) == len(expected)
    base: dict[str, Any] = {
        "phase": phase,
        "strategy": reported_strategy(strategy),
        "configured_filter_strategy": strategy,
        "filter_name": filter_name,
        "ef": ef,
        "expected_queries": len(query_nos),
        "expected_repeats": repeats,
        "expected_samples": len(expected),
        "observed_samples": len(observed),
        "error_count": len(selected) - len(valid),
        "duplicate_pairs": duplicates,
        "missing_pairs": len(expected - set(observed)),
        "complete": complete,
        "latency_definition": "end_to_end_http_json_parse_row_id_transfer",
        "service_qps_definition": "single_client_sequential_completed_requests_per_measured_service_time",
    }
    if not complete:
        return {**base, "recall_mean": NA, "recall_lcb95": NA, "recall_ci95_low": NA, "recall_ci95_high": NA, "latency_mean_ms": NA, "latency_p50_ms": NA, "latency_p95_ms": NA, "latency_p99_ms": NA, "latency_ci95_low_ms": NA, "latency_ci95_high_ms": NA, "single_client_service_qps": NA}
    recalls = _query_level_means(valid, "recall_at_10")
    query_latencies = _query_level_means(valid, "end_to_end_ms")
    sample_latencies = [float(row["end_to_end_ms"]) for row in valid]
    recall_mean, recall_lcb, recall_low, recall_high = bootstrap_recall_stats(recalls, BOOTSTRAP_SAMPLES, bootstrap_seed)
    latency_low, latency_high = bootstrap_mean_ci(query_latencies, BOOTSTRAP_SAMPLES, bootstrap_seed + 1)
    return {
        **base,
        "recall_mean": recall_mean,
        "recall_lcb95": recall_lcb,
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
        "latency_mean_ms": statistics.fmean(query_latencies),
        "latency_p50_ms": percentile(sample_latencies, 0.50),
        "latency_p95_ms": percentile(sample_latencies, 0.95),
        "latency_p99_ms": percentile(sample_latencies, 0.99),
        "latency_ci95_low_ms": latency_low,
        "latency_ci95_high_ms": latency_high,
        "single_client_service_qps": 1000.0 / statistics.fmean(sample_latencies),
    }


def select_fastest_config(summaries: Sequence[dict[str, Any]], target: float) -> dict[str, Any] | None:
    eligible = [row for row in summaries if row.get("complete") is True and _finite_number(row.get("recall_mean")) and float(row["recall_mean"]) >= target]
    return min(eligible, key=lambda row: (float(row["latency_mean_ms"]), int(row["ef"]))) if eligible else None


def reaches_target(summary: dict[str, Any], target: float) -> bool:
    return bool(summary.get("complete") is True and _finite_number(summary.get("recall_mean")) and float(summary["recall_mean"]) >= target)


def pair_calibration_summaries(
    summaries: Sequence[dict[str, Any]], strategy: str, filter_name: str
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in summaries if row.get("configured_filter_strategy") == strategy and row.get("filter_name") == filter_name),
        key=lambda row: int(row["ef"]),
    )


def pair_reached_highest_target(
    summaries: Sequence[dict[str, Any]], highest_target: float
) -> bool:
    return any(reaches_target(summary, highest_target) for summary in summaries)


def calibration_target_status(
    candidates: Sequence[dict[str, Any]], target: float, ef_values: Sequence[int]
) -> str:
    if select_fastest_config(candidates, target) is not None:
        return "selected"
    if calibration_grid_proof(candidates, ef_values)["grid_exhausted_without_errors"]:
        return "unattainable_on_grid"
    return "incomplete_grid"


def calibration_grid_proof(candidates: Sequence[dict[str, Any]], ef_values: Sequence[int]) -> dict[str, Any]:
    expected_efs = [int(ef) for ef in ef_values]
    measured_efs = [int(row["ef"]) for row in sorted(candidates, key=lambda row: int(row["ef"]))]
    complete_without_errors = (
        measured_efs == expected_efs
        and len(candidates) == len(expected_efs)
        and all(row.get("complete") is True for row in candidates)
    )
    return {
        "measured_efs": measured_efs,
        "max_ef": expected_efs[-1],
        "max_ef_complete": bool(candidates and measured_efs[-1] == expected_efs[-1] and candidates[-1].get("complete") is True),
        "grid_exhausted_without_errors": complete_without_errors,
    }


def validate_monotone_calibration_state(
    summaries: Sequence[dict[str, Any]],
    strategies: Sequence[str],
    filters: Sequence[FilterSpec],
    ef_values: Sequence[int],
    highest_target: float,
) -> None:
    expected_efs = [int(ef) for ef in ef_values]
    expected_pairs = {(strategy, spec.name) for strategy in strategies for spec in filters}
    actual_pairs = {
        (str(row.get("configured_filter_strategy")), str(row.get("filter_name")))
        for row in summaries
    }
    if not actual_pairs <= expected_pairs:
        raise RuntimeError("checkpoint calibration summary has an unknown strategy/filter pair")
    for strategy, filter_name in expected_pairs:
        candidates = pair_calibration_summaries(summaries, strategy, filter_name)
        measured_efs = [int(row["ef"]) for row in candidates]
        if measured_efs != expected_efs[:len(measured_efs)] or len(measured_efs) != len(set(measured_efs)):
            raise RuntimeError(f"checkpoint calibration grid is not an ascending prefix: {strategy}/{filter_name}")
        reached_at = [index for index, row in enumerate(candidates) if reaches_target(row, highest_target)]
        if reached_at and reached_at[0] != len(candidates) - 1:
            raise RuntimeError(f"checkpoint violates monotone early-stop: {strategy}/{filter_name}")


def common_attainable_targets(
    strategies: Sequence[str],
    filters: Sequence[FilterSpec],
    targets: Sequence[float],
    selections: dict[tuple[str, str, float], dict[str, Any] | None],
) -> dict[str, list[float]]:
    return {
        spec.name: [
            float(target) for target in targets
            if all(selections.get((strategy, spec.name, float(target))) is not None for strategy in strategies)
        ]
        for spec in filters
    }


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def measurement_block_integrity_errors(
    raw_rows: Sequence[dict[str, Any]], summaries: Sequence[dict[str, Any]], *,
    phase: str, query_nos: Sequence[int], repeats: int, block_fields: Sequence[str],
) -> list[str]:
    """Validate every recorded measurement block without treating low recall as corruption."""
    errors: list[str] = []
    expected_pairs = {(int(query_no), repeat) for query_no in query_nos for repeat in range(repeats)}
    summary_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for summary in summaries:
        try:
            key = tuple(summary[field] for field in block_fields)
        except KeyError:
            errors.append(f"{phase} summary has no complete block identity")
            continue
        if key in summary_by_key:
            errors.append(f"duplicate {phase} summary block: {key}")
            continue
        summary_by_key[key] = summary
    for key, summary in summary_by_key.items():
        rows = [
            row for row in raw_rows
            if row.get("phase") == phase
            and tuple(row.get(field) for field in block_fields) == key
        ]
        observed: set[tuple[int, int]] = set()
        for row in rows:
            try:
                pair = int(row["query_no"]), int(row["repeat"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"malformed {phase} raw row in block {key}")
                continue
            if pair in observed:
                errors.append(f"duplicate {phase} query/repeat pair in block {key}: {pair}")
            observed.add(pair)
            if pair not in expected_pairs:
                errors.append(f"unexpected {phase} query/repeat pair in block {key}: {pair}")
            if (row.get("valid") is not True or row.get("error") or row.get("order_error")
                    or int(row.get("retry_count", -1)) != 0):
                errors.append(f"invalid {phase} raw row in block {key}: {pair}")
            for field in ("recall_at_10", "end_to_end_ms"):
                if not _finite_number(row.get(field)):
                    errors.append(f"non-finite {phase} raw metric {field} in block {key}: {pair}")
        if observed != expected_pairs or len(rows) != len(expected_pairs):
            errors.append(
                f"{phase} block coverage mismatch for {key}: "
                f"expected={len(expected_pairs)} observed={len(rows)} unique={len(observed)}"
            )
        if summary.get("complete") is not True:
            errors.append(f"incomplete {phase} summary block: {key}")
        for field in (
            "recall_mean", "recall_lcb95", "recall_ci95_low", "recall_ci95_high",
            "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "latency_ci95_low_ms", "latency_ci95_high_ms", "single_client_service_qps",
        ):
            if not _finite_number(summary.get(field)):
                errors.append(f"non-finite {phase} summary metric {field} in block {key}")
    return errors


def target_outcome_counts(
    final_summaries: Sequence[dict[str, Any]], target_statuses: Sequence[str],
) -> dict[str, int]:
    return {
        "selected_and_confirmed": sum(
            row.get("target_outcome") == "selected_and_confirmed" for row in final_summaries
        ),
        "selected_but_final_unconfirmed": sum(
            row.get("target_outcome") == "selected_but_final_unconfirmed" for row in final_summaries
        ),
        "unattainable_on_grid": sum(status == "unattainable_on_grid" for status in target_statuses),
    }


def measurement_row(
    *,
    phase: str,
    strategy: str,
    spec: FilterSpec,
    ef: int,
    query_no: int,
    query_id: int,
    repeat: int,
    result: QueryResult,
    truth: TruthEntry,
    result_distances_sq: Sequence[float],
    target: float | str = NA,
) -> dict[str, Any]:
    valid = (
        not result.error
        and not result.order_error
        and result.retry_count == 0
        and len(result.ids) <= K
        and len(result_distances_sq) == len(result.ids)
        and query_id not in result.ids
    )
    return {
        "phase": phase,
        "strategy": reported_strategy(strategy),
        "configured_filter_strategy": strategy,
        "filter_name": spec.name,
        "target_rate": spec.target_rate,
        "predicate": spec.predicate,
        "actual_selectivity": spec.actual_pct / 100.0,
        "target_recall": target,
        "ef": ef,
        "query_no": query_no,
        "query_id": query_id,
        "repeat": repeat,
        "end_to_end_ms": result.latency_ms if result.latency_ms > 0 else NA,
        "latency_definition": "end_to_end_http_json_parse_row_id_transfer",
        "recall_at_10": tie_aware_recall(result_distances_sq, truth, K) if valid else NA,
        "recall_contract": "distance_threshold_tie_aware_v1",
        "truth_filtered_rows": truth.filtered_rows,
        "truth_kth_distance_sq": truth.kth_distance_sq,
        "truth_tie_tolerance": truth.tie_tolerance,
        "truth_self_excluded": truth.self_excluded,
        "returned": len(result.ids),
        "returned_count": result.returned_count,
        "request_limit": result.request_limit,
        "shortfall": max(0, result.request_limit - result.returned_count),
        "result_ids": ",".join(str(value) for value in result.ids) if result.ids else NA,
        "result_distances_sq": ",".join(f"{value:.9g}" for value in result_distances_sq) if result_distances_sq else NA,
        "retry_count": result.retry_count,
        "order_error": result.order_error,
        "valid": valid,
        "error": result.error,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True, default=str)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sibling_outputs(out: Path) -> dict[str, Path]:
    return {
        "raw_csv": out,
        "summary_csv": out.with_name(out.stem + "_summary.csv"),
        "schema_json": out.with_name(out.stem + "_schema.json"),
        "config_json": out.with_name(out.stem + "_config.json"),
        "manifest_json": out.with_name(out.stem + "_manifest.json"),
    }


def checkpoint_path(out: Path) -> Path:
    return out.with_name(out.stem + "_checkpoint.json")


def run_specification(
    args: argparse.Namespace,
    strategies: Sequence[str],
    filters: Sequence[FilterSpec],
    query_ids: dict[int, int],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Everything that can affect reusable measurement rows, excluding --resume."""
    return {
        "version": 1,
        "class": CLASS_NAME,
        "candidate_universe": {
            "predicate": CANDIDATE_VALIDITY_PREDICATE,
            "expected_rows": EXPECTED_VALID_ROWS,
        },
        "source_hashes": source_hashes,
        "endpoint": {"host": args.host, "port": args.port},
        "configured_filter_strategies": list(strategies),
        "filters": [asdict(spec) for spec in filters],
        "ef_values": [int(ef) for ef in args.ef_values],
        "targets": [float(target) for target in args.targets],
        "k": K,
        "calibration": {
            "query_nos": list(CALIBRATION_QUERY_NOS),
            "repeats": CALIBRATION_REPEATS,
            "warmup_queries": args.warmup_queries,
        },
        "final": {"query_nos": list(FINAL_QUERY_NOS), "repeats": FINAL_REPEATS},
        "bootstrap_seed": args.bootstrap_seed,
        "query_ids": {str(query_no): int(query_id) for query_no, query_id in sorted(query_ids.items())},
    }


def run_spec_hash(specification: dict[str, Any]) -> str:
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _block_key(phase: str, strategy: str, filter_name: str, ef: int) -> tuple[str, str, str, int]:
    return phase, strategy, filter_name, int(ef)


def _block_record(phase: str, strategy: str, filter_name: str, ef: int) -> dict[str, Any]:
    query_nos = CALIBRATION_QUERY_NOS if phase == "calibration" else FINAL_QUERY_NOS
    repeats = CALIBRATION_REPEATS if phase == "calibration" else FINAL_REPEATS
    return {
        "phase": phase,
        "configured_filter_strategy": strategy,
        "filter_name": filter_name,
        "ef": int(ef),
        "query_nos": list(query_nos),
        "repeats": repeats,
    }


def _validate_checkpoint_blocks(payload: dict[str, Any], query_ids: dict[int, int]) -> None:
    state = payload.get("state")
    raw_rows = payload.get("raw_rows")
    calibration_summaries = payload.get("calibration_summaries")
    final_results = payload.get("final_results")
    if not isinstance(state, dict) or not isinstance(raw_rows, list) or not isinstance(calibration_summaries, list) or not isinstance(final_results, list):
        raise RuntimeError("checkpoint is missing required state")
    blocks = state.get("completed_blocks")
    if not isinstance(blocks, list):
        raise RuntimeError("checkpoint completed_blocks is invalid")
    seen: set[tuple[str, str, str, int]] = set()
    calibration_keys: set[tuple[str, str, str, int]] = set()
    final_keys: set[tuple[str, str, str, int]] = set()
    for block in blocks:
        if not isinstance(block, dict):
            raise RuntimeError("checkpoint contains a malformed block")
        phase = block.get("phase")
        strategy = block.get("configured_filter_strategy")
        filter_name = block.get("filter_name")
        try:
            ef = int(block.get("ef"))
        except (TypeError, ValueError):
            raise RuntimeError("checkpoint block ef is invalid") from None
        if phase not in {"calibration", "final"} or not isinstance(strategy, str) or not isinstance(filter_name, str):
            raise RuntimeError("checkpoint block identity is invalid")
        expected_record = _block_record(phase, strategy, filter_name, ef)
        if block != expected_record:
            raise RuntimeError(f"checkpoint block specification mismatch: {block}")
        key = _block_key(phase, strategy, filter_name, ef)
        if key in seen:
            raise RuntimeError(f"checkpoint has duplicate completed block: {key}")
        seen.add(key)
        expected = {(query_no, repeat) for query_no in expected_record["query_nos"] for repeat in range(expected_record["repeats"])}
        observed_rows = [
            row for row in raw_rows
            if isinstance(row, dict)
            and row.get("phase") == phase
            and row.get("configured_filter_strategy") == strategy
            and row.get("filter_name") == filter_name
            and int(row.get("ef", -1)) == ef
        ]
        observed: set[tuple[int, int]] = set()
        for row in observed_rows:
            try:
                pair = int(row["query_no"]), int(row["repeat"])
                row_query_id = int(row["query_id"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(f"checkpoint row is malformed for block {key}") from None
            if pair in observed or pair not in expected or row_query_id != query_ids.get(pair[0]):
                raise RuntimeError(f"checkpoint rows are incomplete or inconsistent for block {key}")
            observed.add(pair)
        if observed != expected or len(observed_rows) != len(expected):
            raise RuntimeError(f"checkpoint rows are incomplete for block {key}")
        (calibration_keys if phase == "calibration" else final_keys).add(key)
    summary_keys = {
        _block_key("calibration", str(row.get("configured_filter_strategy")), str(row.get("filter_name")), int(row.get("ef", -1)))
        for row in calibration_summaries
        if isinstance(row, dict)
    }
    result_keys = {
        _block_key("final", str(row.get("configured_filter_strategy")), str(row.get("filter_name")), int(row.get("ef", -1)))
        for row in final_results
        if isinstance(row, dict)
    }
    if summary_keys != calibration_keys or len(summary_keys) != len(calibration_summaries):
        raise RuntimeError("checkpoint calibration summaries do not match completed blocks")
    if result_keys != final_keys or len(result_keys) != len(final_results):
        raise RuntimeError("checkpoint final summaries do not match completed blocks")


def load_checkpoint(path: Path, specification: dict[str, Any], query_ids: dict[int, int]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
    expected_hash = run_spec_hash(specification)
    if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("run_spec_hash") != expected_hash or payload.get("run_spec") != specification:
        raise RuntimeError("checkpoint run-spec/hash mismatch; refusing to reuse measurements")
    _validate_checkpoint_blocks(payload, query_ids)
    return payload


def write_checkpoint(
    path: Path,
    specification: dict[str, Any],
    *,
    raw_rows: Sequence[dict[str, Any]],
    calibration_summaries: Sequence[dict[str, Any]],
    final_results: Sequence[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    atomic_write_json(path, {
        "version": 1,
        "run_spec": specification,
        "run_spec_hash": run_spec_hash(specification),
        "raw_rows": list(raw_rows),
        "calibration_summaries": list(calibration_summaries),
        "final_results": list(final_results),
        "state": state,
    })


def isolate_existing_outputs(outputs: dict[str, Path]) -> Path | None:
    existing = [path for path in outputs.values() if path.exists()]
    if not existing:
        return None
    anchor = outputs["raw_csv"]
    quarantine = anchor.parent / f".{anchor.stem}_stale" / f"{time.time_ns()}-{uuid.uuid4().hex}"
    quarantine.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.replace(path, quarantine / path.name)
    return quarantine


def staging_outputs(outputs: dict[str, Path]) -> dict[str, Path]:
    token = uuid.uuid4().hex
    return {
        name: path.with_name(f".{path.name}.{token}.staging")
        for name, path in outputs.items()
    }


def cleanup_staging(outputs: dict[str, Path]) -> None:
    for path in outputs.values():
        path.unlink(missing_ok=True)


def commit_output_bundle(
    outputs: dict[str, Path],
    staged: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    data_names = [name for name in outputs if name != "manifest_json"]
    missing = [name for name in data_names if not staged[name].is_file()]
    if missing:
        raise RuntimeError(f"staged artifact outputs missing: {missing}")
    output_hashes = {name: sha256_file(staged[name]) for name in data_names}
    manifest = {
        **manifest,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_sha256": output_hashes,
        "manifest_commit": "atomic_last",
    }
    for name in data_names:
        outputs[name].parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged[name], outputs[name])
    atomic_write_json(outputs["manifest_json"], manifest)
    return manifest


def rotated(values: Sequence[Any], offset: int) -> list[Any]:
    items = list(values)
    if not items:
        return []
    start = offset % len(items)
    return items[start:] + items[:start]


def calibration_configuration_schedule(
    strategies: Sequence[str], ef_values: Sequence[int]
) -> list[tuple[int, str, int]]:
    schedule: list[tuple[int, str, int]] = []
    for ef_index, ef in enumerate(ef_values):
        for strategy in rotated(strategies, ef_index):
            schedule.append((len(schedule), strategy, int(ef)))
    return schedule


def measurement_schedule(
    query_nos: Sequence[int], repeats: int, rotation: int
) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    for repeat in range(repeats):
        for query_no in rotated(query_nos, rotation + repeat):
            schedule.append((int(query_no), repeat))
    return schedule


def _find_meta_values(value: Any, wanted_key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).upper() == wanted_key.upper():
                found.append(item)
            found.extend(_find_meta_values(item, wanted_key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_meta_values(item, wanted_key))
    return found


def acorn_reporting_metadata(service_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "reported_strategy": ACORN_REPORTED_STRATEGY,
        "configured_filter_strategy": "acorn",
        "effective_path_proven": False,
        "claim": "configured ACORN with query-dependent automatic fallback; effective per-query path is not proven",
        ACORN_RATIO_ENV: {
            "runner_environment": os.environ.get(ACORN_RATIO_ENV, NA),
            "service_meta_values": _find_meta_values(service_meta, ACORN_RATIO_ENV) or [NA],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ef-values", type=int, nargs="+", default=list(DEFAULT_EF_VALUES))
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--strategies", nargs="+", choices=("acorn", "sweeping"), default=["acorn"])
    parser.add_argument("--strategy", choices=("acorn", "sweeping"), help="single-strategy alias")
    parser.add_argument("--k", type=int, choices=(K,), default=K)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--resume", action="store_true", help="resume only from an exactly matching complete-block checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run(args: argparse.Namespace) -> int:
    strategies = [args.strategy] if args.strategy else args.strategies
    print(json.dumps({"dry_run": True, "network": False, "files_read": False, "files_written": False, "class": CLASS_NAME, "k": K, "configured_filter_strategies": strategies, "reported_strategies": [reported_strategy(strategy) for strategy in strategies], "ef_values": args.ef_values, "targets": args.targets, "target_selection_rule": TARGET_SELECTION_RULE, "calibration_query_nos": [20, 99], "final_query_nos": [100, 199], "reserved_query_nos": [0, 19]}, sort_keys=True))
    return 0


def _count_query(spec: FilterSpec) -> str:
    return "{ Aggregate { " + f"{CLASS_NAME}(where:{json_to_graphql(spec.where)}) {{ meta {{ count }} }}" + " } }"


def _candidate_universe_count_query() -> str:
    where = {
        "path": ["embedding_valid"],
        "operator": "Equal",
        "valueBoolean": True,
    }
    return (
        "{ Aggregate { "
        + f"{CLASS_NAME}(where:{json_to_graphql(where)}) {{ meta {{ count }} }}"
        + " } }"
    )


def _run_measurements(
    args: argparse.Namespace,
    base_url: str,
    vectors: np.memmap,
    truth: dict[tuple[str, int], TruthEntry],
    query_ids: dict[int, int],
    spec: FilterSpec,
    strategy: str,
    ef: int,
    phase: str,
    query_nos: Sequence[int],
    repeats: int,
    target: float | str = NA,
    schedule_rotation: int = 0,
    schedule_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timed = phase in {"calibration", "final"}
    for query_no, repeat in measurement_schedule(query_nos, repeats, schedule_rotation):
        query_id = query_ids[query_no]
        result = query_once(
            base_url,
            vectors[query_id],
            spec.where,
            query_id,
            K,
            timeout=args.timeout,
            retries=0 if timed else args.retries,
        )
        try:
            result_distances_sq = exact_squared_l2(vectors, query_id, result.ids) if result.ids else ()
        except Exception as exc:
            result = QueryResult(
                result.ids,
                result.latency_ms,
                result.retry_count,
                result.order_error,
                f"{exc.__class__.__name__}: {exc}",
                result.returned_count,
                result.request_limit,
            )
            result_distances_sq = ()
        row = measurement_row(phase=phase, strategy=strategy, spec=spec, ef=ef, query_no=query_no, query_id=query_id, repeat=repeat, result=result, truth=truth[(spec.name, query_no)], result_distances_sq=result_distances_sq, target=target)
        row["schedule_index"] = schedule_index
        row["schedule_rotation"] = schedule_rotation
        rows.append(row)
    return rows


def _summary_row_for_target(
    base: dict[str, Any],
    target: float,
    selected: dict[str, Any] | None,
    final: dict[str, Any] | None,
    target_status: str,
) -> dict[str, Any]:
    result = dict(final or {"complete": False, "recall_mean": NA, "recall_lcb95": NA, "recall_ci95_low": NA, "recall_ci95_high": NA, "latency_mean_ms": NA, "latency_p50_ms": NA, "latency_p95_ms": NA, "latency_p99_ms": NA, "latency_ci95_low_ms": NA, "latency_ci95_high_ms": NA, "single_client_service_qps": NA})
    final_met = bool(result.get("complete") is True and _finite_number(result.get("recall_mean")) and float(result["recall_mean"]) >= target)
    unattainable = target_status == "unattainable_on_grid"
    target_outcome = (
        "selected_and_confirmed" if selected and final_met else
        "selected_but_final_unconfirmed" if selected else
        "unattainable_on_grid" if unattainable else "incomplete_grid"
    )
    result.update({"phase": "final", "strategy": base["strategy"], "configured_filter_strategy": base["configured_filter_strategy"], "filter_name": base["filter_name"], "target_recall": target, "selected_ef": selected["ef"] if selected else NA, "target_status": target_status, "target_met": final_met if selected else (NA if unattainable else False), "target_met_final": final_met if selected else (NA if unattainable else False), "target_outcome": target_outcome, "comparison_status": "confirmed" if target_outcome == "selected_and_confirmed" else "unconfirmed" if target_outcome == "selected_but_final_unconfirmed" else target_outcome, "calibration_recall_mean": base.get("recall_mean", NA), "calibration_recall_lcb95": base.get("recall_lcb95", NA), "calibration_complete": base.get("complete", False), "target_selection_rule": TARGET_SELECTION_RULE})
    return result


def artifact_gate_errors(
    *,
    strategies: Sequence[str],
    filters: Sequence[FilterSpec],
    targets: Sequence[float],
    selections: dict[tuple[str, str, float], dict[str, Any] | None],
    final_summaries: Sequence[dict[str, Any]],
    raw_rows: Sequence[dict[str, Any]],
    target_statuses: dict[tuple[str, str, float], str] | None = None,
    calibration_summaries: Sequence[dict[str, Any]] = (),
    final_results: Sequence[dict[str, Any]] = (),
) -> list[str]:
    errors: list[str] = []
    expected = {
        (strategy, spec.name, float(target))
        for strategy in strategies
        for spec in filters
        for target in targets
    }
    if set(selections) != expected:
        errors.append(f"selection grid mismatch: expected={len(expected)} actual={len(selections)}")
    by_key: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in final_summaries:
        key = (
            str(row.get("configured_filter_strategy")),
            str(row.get("filter_name")),
            float(row.get("target_recall")),
        )
        by_key.setdefault(key, []).append(row)
    for key in sorted(expected):
        winner = selections.get(key)
        status = (target_statuses or {}).get(key, "missing_winner")
        if winner is None and status != "unattainable_on_grid":
            errors.append(f"missing calibration winner: strategy={key[0]} filter={key[1]} target={key[2]}")
        rows = by_key.get(key, [])
        if len(rows) != 1:
            errors.append(f"final summary cardinality mismatch for {key}: {len(rows)}")
        elif winner is None and status == "unattainable_on_grid":
            if rows[0].get("target_status") != "unattainable_on_grid":
                errors.append(f"unattainable target summary mismatch for {key}")
        elif winner is not None and rows[0].get("target_outcome") not in {
                "selected_and_confirmed", "selected_but_final_unconfirmed"}:
            errors.append(f"selected target outcome is invalid for {key}")
    extras = set(by_key) - expected
    if extras:
        errors.append(f"unexpected final summary keys: {sorted(extras)}")
    for row in raw_rows:
        if row.get("phase") in {"calibration", "final"}:
            if int(row.get("retry_count", -1)) != 0:
                errors.append(
                    "timed measurement retried: "
                    f"phase={row.get('phase')} strategy={row.get('configured_filter_strategy')} "
                    f"filter={row.get('filter_name')} ef={row.get('ef')} query_no={row.get('query_no')}"
                )
            if row.get("valid") is not True or row.get("error") or row.get("order_error"):
                errors.append(
                    "invalid timed measurement: "
                    f"phase={row.get('phase')} strategy={row.get('configured_filter_strategy')} "
                    f"filter={row.get('filter_name')} ef={row.get('ef')} query_no={row.get('query_no')}"
                )
    errors.extend(measurement_block_integrity_errors(
        raw_rows, calibration_summaries, phase="calibration",
        query_nos=CALIBRATION_QUERY_NOS, repeats=CALIBRATION_REPEATS,
        block_fields=("configured_filter_strategy", "filter_name", "ef"),
    ))
    errors.extend(measurement_block_integrity_errors(
        raw_rows, final_results, phase="final",
        query_nos=FINAL_QUERY_NOS, repeats=FINAL_REPEATS,
        block_fields=("configured_filter_strategy", "filter_name", "ef"),
    ))
    return errors


def group_selected_targets(
    selections: dict[tuple[str, str, float], dict[str, Any] | None]
) -> dict[tuple[str, str, int], list[float]]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for (strategy, filter_name, target), winner in selections.items():
        if winner is not None:
            grouped.setdefault((strategy, filter_name, int(winner["ef"])), []).append(target)
    return {key: sorted(targets) for key, targets in grouped.items()}


def run(args: argparse.Namespace) -> int:
    base_url = f"http://{args.host}:{args.port}"
    outputs = sibling_outputs(args.out)
    checkpoint = checkpoint_path(args.out)
    if args.k != K:
        raise ValueError(f"formal runner requires k={K}")
    if sorted(set(args.targets)) != list(args.targets) or any(target <= 0 or target > 1 for target in args.targets):
        raise ValueError("targets must be sorted, unique, and in (0, 1]")
    if sorted(set(args.ef_values)) != list(args.ef_values) or any(ef <= 0 for ef in args.ef_values):
        raise ValueError("ef values must be sorted, unique, and positive")
    if args.retries < 0:
        raise ValueError("setup retries must be non-negative")
    if args.warmup_queries < 0 or args.warmup_queries > len(CALIBRATION_QUERY_NOS):
        raise ValueError("warmup_queries is outside the calibration query range")
    strategies = [args.strategy] if args.strategy else list(args.strategies)
    if not strategies or len(strategies) != len(set(strategies)):
        raise ValueError("configured filter strategies must be non-empty and unique")

    filters = load_filter_specs(args.filters_csv)
    vectors, vector_rows, dimensions = read_fbin_memmap(args.fbin)
    truth, query_ids = load_truth(args.truth_csv, filters, k=K)
    if any(query_id < 0 or query_id >= vector_rows for query_id in query_ids.values()):
        raise ValueError("truth query_id is outside the fbin row range")
    source_hashes = {"runner": sha256_file(Path(__file__)), "filters_csv": sha256_file(args.filters_csv), "truth_csv": sha256_file(args.truth_csv), "fbin": sha256_file(args.fbin)}
    specification = run_specification(args, strategies, filters, query_ids, source_hashes)
    quarantined_outputs = None if args.resume else isolate_existing_outputs(outputs)

    raw_rows: list[dict[str, Any]] = []
    calibration_summaries: list[dict[str, Any]] = []
    final_result_summaries: list[dict[str, Any]] = []
    completed_blocks: list[dict[str, Any]] = []
    schema_records: list[dict[str, Any]] = []
    schema_timings: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []
    if args.resume:
        payload = load_checkpoint(checkpoint, specification, query_ids)
        state = payload["state"]
        required_state = ("completed_blocks", "schema_records", "schema_timings", "node_records")
        if any(not isinstance(state.get(name), list) for name in required_state):
            raise RuntimeError("checkpoint state is incomplete")
        raw_rows = payload["raw_rows"]
        calibration_summaries = payload["calibration_summaries"]
        final_result_summaries = payload["final_results"]
        completed_blocks = state["completed_blocks"]
        schema_records = state["schema_records"]
        schema_timings = state["schema_timings"]
        node_records = state["node_records"]
        validate_monotone_calibration_state(
            calibration_summaries, strategies, filters, args.ef_values, float(args.targets[-1])
        )

    initial_schema, _ = request_json(base_url, f"/v1/schema/{CLASS_NAME}", timeout=args.timeout, retries=args.retries)
    schema_records.append({"phase": "resume_initial" if args.resume else "initial", "schema": initial_schema})
    final_summaries: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, float], dict[str, Any] | None] = {}
    target_statuses: dict[tuple[str, str, float], str] = {}
    service_meta: dict[str, Any] = {}
    total_count = 0
    candidate_universe_count = 0
    filter_counts: dict[str, int] = {}
    configuration_schedule = calibration_configuration_schedule(strategies, args.ef_values)
    schema_restore_required = False
    primary_error: Exception | None = None

    def checkpoint_state() -> dict[str, Any]:
        return {
            "completed_blocks": completed_blocks,
            "schema_records": schema_records,
            "schema_timings": schema_timings,
            "node_records": node_records,
        }

    def save_checkpoint() -> None:
        write_checkpoint(
            checkpoint,
            specification,
            raw_rows=raw_rows,
            calibration_summaries=calibration_summaries,
            final_results=final_result_summaries,
            state=checkpoint_state(),
        )

    completed_keys = {
        _block_key(block["phase"], block["configured_filter_strategy"], block["filter_name"], block["ef"])
        for block in completed_blocks
    }
    try:
        verify_schema(initial_schema)
        service_meta, _ = request_json(base_url, "/v1/meta", timeout=args.timeout, retries=args.retries)
        initial_nodes, node_retries = get_ready_nodes(base_url, args.timeout, args.retries)
        node_records.append({"phase": "initial", "retries": node_retries, "nodes": initial_nodes})
        count_data, _ = graphql(base_url, f"{{ Aggregate {{ {CLASS_NAME} {{ meta {{ count }} }} }} }}", timeout=args.timeout, retries=args.retries)
        total_count = int(count_data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])
        if total_count != EXPECTED_ROWS:
            raise RuntimeError(f"Weaviate count mismatch: expected={EXPECTED_ROWS} actual={total_count}")
        count_data, _ = graphql(
            base_url,
            _candidate_universe_count_query(),
            timeout=args.timeout,
            retries=args.retries,
        )
        candidate_universe_count = int(
            count_data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"]
        )
        if candidate_universe_count != EXPECTED_VALID_ROWS:
            raise RuntimeError(
                "Weaviate candidate universe count mismatch: "
                f"expected={EXPECTED_VALID_ROWS} actual={candidate_universe_count}"
            )
        for spec in filters:
            count_data, _ = graphql(base_url, _count_query(spec), timeout=args.timeout, retries=args.retries)
            actual = int(count_data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])
            filter_counts[spec.name] = actual
            if actual != spec.expected_rows:
                raise RuntimeError(f"filter count mismatch for {spec.name}: expected={spec.expected_rows} actual={actual}")

        highest_target = float(args.targets[-1])
        for schedule_index, strategy, ef in configuration_schedule:
            active_specs = [
                spec for spec in rotated(filters, schedule_index)
                if not pair_reached_highest_target(
                    pair_calibration_summaries(calibration_summaries, strategy, spec.name), highest_target
                )
            ]
            pending_specs = [
                spec for spec in active_specs
                if _block_key("calibration", strategy, spec.name, ef) not in completed_keys
            ]
            if not pending_specs:
                continue
            schema_restore_required = True
            updated_schema, update_ms, update_retries = put_hnsw_config(base_url, strategy, ef, args.timeout, args.retries)
            schema_records.append({"phase": "calibration", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "reported_strategy": reported_strategy(strategy), "ef": ef, "schema": updated_schema})
            schema_timings.append({"phase": "calibration", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "ef": ef, "schema_update_ms": update_ms, "schema_retries": update_retries})
            ready_nodes, node_retries = get_ready_nodes(base_url, args.timeout, args.retries)
            node_records.append({"phase": "calibration", "schedule_index": schedule_index, "configured_filter_strategy": strategy, "ef": ef, "retries": node_retries, "nodes": ready_nodes})
            if args.warmup_queries:
                for spec in pending_specs:
                    warmup = _run_measurements(args, base_url, vectors, truth, query_ids, spec, strategy, ef, "warmup", CALIBRATION_QUERY_NOS[: args.warmup_queries], 1, schedule_rotation=schedule_index, schedule_index=schedule_index)
                    for row in warmup:
                        row["recall_at_10"] = NA
                    raw_rows.extend(warmup)
            for filter_position, spec in enumerate(pending_specs):
                block_started = time.perf_counter()
                measured = _run_measurements(args, base_url, vectors, truth, query_ids, spec, strategy, ef, "calibration", CALIBRATION_QUERY_NOS, CALIBRATION_REPEATS, schedule_rotation=schedule_index + filter_position, schedule_index=schedule_index)
                raw_rows.extend(measured)
                summary = summarize_configuration(measured, strategy=strategy, filter_name=spec.name, ef=ef, query_nos=CALIBRATION_QUERY_NOS, repeats=CALIBRATION_REPEATS, bootstrap_seed=args.bootstrap_seed + schedule_index * len(filters) + filter_position)
                calibration_summaries.append({**summary, "schedule_index": schedule_index, "target_recall": NA, "target_met": {str(target): reaches_target(summary, target) for target in args.targets}})
                block = _block_record("calibration", strategy, spec.name, ef)
                completed_blocks.append(block)
                completed_keys.add(_block_key("calibration", strategy, spec.name, ef))
                save_checkpoint()
                print(f"progress strategy={strategy} ef={ef} filter={spec.name} completed=calibration active={len(pending_specs)} recall_mean={summary['recall_mean']} recall_lcb95_reported={summary['recall_lcb95']} elapsed_s={time.perf_counter() - block_started:.1f}", flush=True)

        for strategy in strategies:
            for spec in filters:
                candidates = pair_calibration_summaries(calibration_summaries, strategy, spec.name)
                for target in args.targets:
                    key = (strategy, spec.name, float(target))
                    selections[key] = select_fastest_config(candidates, target)
                    target_statuses[key] = calibration_target_status(candidates, target, args.ef_values)

        selected_groups = group_selected_targets(selections)
        resumed_final_keys = {
            (block["configured_filter_strategy"], block["filter_name"], int(block["ef"]))
            for block in completed_blocks
            if block["phase"] == "final"
        }
        if not resumed_final_keys <= set(selected_groups):
            raise RuntimeError("checkpoint final block is not selected by the restored calibration state")
        final_results = {
            (str(row["configured_filter_strategy"]), str(row["filter_name"]), int(row["ef"])): row
            for row in final_result_summaries
        }
        final_schedule_index = 0
        for _, strategy, ef in configuration_schedule:
            groups = [key for key in selected_groups if key[0] == strategy and key[2] == ef]
            if not groups:
                continue
            pending_groups = [
                key for key in groups
                if _block_key("final", strategy, key[1], ef) not in completed_keys
            ]
            if not pending_groups:
                final_schedule_index += 1
                continue
            schema_restore_required = True
            updated_schema, update_ms, update_retries = put_hnsw_config(base_url, strategy, ef, args.timeout, args.retries)
            schema_records.append({"phase": "final", "schedule_index": final_schedule_index, "configured_filter_strategy": strategy, "reported_strategy": reported_strategy(strategy), "ef": ef, "filters": sorted(key[1] for key in pending_groups), "schema": updated_schema})
            schema_timings.append({"phase": "final", "schedule_index": final_schedule_index, "configured_filter_strategy": strategy, "ef": ef, "schema_update_ms": update_ms, "schema_retries": update_retries})
            ready_nodes, node_retries = get_ready_nodes(base_url, args.timeout, args.retries)
            node_records.append({"phase": "final", "schedule_index": final_schedule_index, "configured_filter_strategy": strategy, "ef": ef, "retries": node_retries, "nodes": ready_nodes})
            selected_names = {key[1] for key in pending_groups}
            for filter_position, spec in enumerate(rotated(filters, final_schedule_index)):
                if spec.name not in selected_names:
                    continue
                block_started = time.perf_counter()
                key = (strategy, spec.name, ef)
                targets_reusing_measurement = selected_groups[key]
                if args.warmup_queries:
                    final_warmup = _run_measurements(
                        args, base_url, vectors, truth, query_ids, spec, strategy, ef,
                        "final_warmup", FINAL_QUERY_NOS[: args.warmup_queries], 1,
                        schedule_rotation=final_schedule_index + filter_position,
                        schedule_index=final_schedule_index,
                    )
                    for row in final_warmup:
                        row["recall_at_10"] = NA
                    raw_rows.extend(final_warmup)
                measured = _run_measurements(args, base_url, vectors, truth, query_ids, spec, strategy, ef, "final", FINAL_QUERY_NOS, FINAL_REPEATS, schedule_rotation=final_schedule_index + filter_position, schedule_index=final_schedule_index)
                for row in measured:
                    row["reused_for_targets"] = ",".join(str(target) for target in targets_reusing_measurement)
                raw_rows.extend(measured)
                final_results[key] = summarize_configuration(measured, strategy=strategy, filter_name=spec.name, ef=ef, query_nos=FINAL_QUERY_NOS, repeats=FINAL_REPEATS, bootstrap_seed=args.bootstrap_seed + 100_000 + final_schedule_index * len(filters) + filter_position, phase="final")
                final_result_summaries.append(final_results[key])
                block = _block_record("final", strategy, spec.name, ef)
                completed_blocks.append(block)
                completed_keys.add(_block_key("final", strategy, spec.name, ef))
                save_checkpoint()
                print(f"progress strategy={strategy} ef={ef} filter={spec.name} completed=final active={len(pending_groups)} recall_mean={final_results[key]['recall_mean']} recall_lcb95_reported={final_results[key]['recall_lcb95']} elapsed_s={time.perf_counter() - block_started:.1f}", flush=True)
            final_schedule_index += 1

        for strategy in strategies:
            for spec in filters:
                candidates = pair_calibration_summaries(calibration_summaries, strategy, spec.name)
                fallback_base = candidates[0] if candidates else {"strategy": reported_strategy(strategy), "configured_filter_strategy": strategy, "filter_name": spec.name, "complete": False, "recall_lcb95": NA}
                for target in args.targets:
                    selection_key = (strategy, spec.name, float(target))
                    winner = selections[selection_key]
                    final = final_results.get((strategy, spec.name, int(winner["ef"]))) if winner else None
                    final_summaries.append(_summary_row_for_target(winner or fallback_base, target, winner, final, target_statuses[selection_key]))
    except Exception as exc:
        primary_error = exc
    finally:
        if schema_restore_required:
            restore_started = time.perf_counter()
            try:
                restored_schema, restore_retries = put_schema_definition(
                    base_url,
                    initial_schema,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                restore_ms = (time.perf_counter() - restore_started) * 1000.0
                schema_records.append({"phase": "restore", "schema": restored_schema})
                schema_timings.append({"phase": "restore", "schema_update_ms": restore_ms, "schema_retries": restore_retries})
                ready_nodes, node_retries = get_ready_nodes(base_url, args.timeout, args.retries)
                node_records.append({"phase": "restore", "retries": node_retries, "nodes": ready_nodes})
            except Exception as restore_error:
                if primary_error is not None:
                    raise RuntimeError(
                        f"primary failure: {primary_error}; schema restore failure: {restore_error}"
                    ) from primary_error
                raise

    if primary_error is not None:
        raise primary_error

    revision = git_revision()
    measurement_errors = [str(row["error"]) for row in raw_rows if row.get("error")] + [str(row["order_error"]) for row in raw_rows if row.get("order_error")] + [f"invalid measurement block row: phase={row.get('phase')} strategy={row.get('configured_filter_strategy')} filter={row.get('filter_name')} ef={row.get('ef')} query_no={row.get('query_no')}" for row in raw_rows if row.get("phase") in {"calibration", "final"} and row.get("valid") is not True]
    gate_errors = artifact_gate_errors(
        strategies=strategies, filters=filters, targets=args.targets,
        selections=selections, final_summaries=final_summaries, raw_rows=raw_rows,
        target_statuses=target_statuses, calibration_summaries=calibration_summaries,
        final_results=final_result_summaries,
    )
    errors = measurement_errors + gate_errors
    common_targets = common_attainable_targets(strategies, filters, args.targets, selections)
    pair_grid_proofs = {
        (strategy, spec.name): calibration_grid_proof(
            pair_calibration_summaries(calibration_summaries, strategy, spec.name), args.ef_values
        )
        for strategy in strategies
        for spec in filters
    }
    selection_status_records = [
        {
            "configured_filter_strategy": strategy,
            "filter_name": filter_name,
            "target_recall": target,
            "status": target_statuses[(strategy, filter_name, target)],
            "selected_ef": winner["ef"] if winner else NA,
            "grid_proof": pair_grid_proofs[(strategy, filter_name)],
        }
        for (strategy, filter_name, target), winner in sorted(selections.items())
    ]
    config = {
        "class": CLASS_NAME,
        "git_revision": revision,
        "source_hashes": source_hashes,
        "run_spec_hash": run_spec_hash(specification),
        "vector_rows": vector_rows,
        "dimensions": dimensions,
        "k": K,
        "query_limit": K + 1,
        "recall_contract": "returned filter-valid self-excluded IDs with exact fbin squared L2 <= kth_distance_sq + tie_tolerance, capped at k",
        "candidate_universe": {
            "predicate": CANDIDATE_VALIDITY_PREDICATE,
            "expected_rows": EXPECTED_VALID_ROWS,
            "observed_rows": candidate_universe_count,
        },
        "target_selection_rule": TARGET_SELECTION_RULE,
        "filters": [asdict(spec) for spec in filters],
        "configured_filter_strategies": strategies,
        "reported_strategies": [reported_strategy(strategy) for strategy in strategies],
        "ef_values": args.ef_values,
        "targets": args.targets,
        "calibration": {
            "query_nos": list(CALIBRATION_QUERY_NOS),
            "queries": len(CALIBRATION_QUERY_NOS),
            "repeats": CALIBRATION_REPEATS,
            "configuration_schedule": configuration_schedule,
            "warmup_queries_per_filter_config": args.warmup_queries,
            "schedule": "ef-interleaved strategy rotation; rotated filter and query order",
            "selection_rule": TARGET_SELECTION_RULE,
            "unattainable_rule": "unattainable_on_grid is valid only after every ef through max_ef completed without measurement errors",
        },
        "final": {
            "query_nos": list(FINAL_QUERY_NOS),
            "queries": len(FINAL_QUERY_NOS),
            "repeats": FINAL_REPEATS,
            "final_warmup_phase": "final_warmup",
            "final_warmup_query_nos": list(FINAL_QUERY_NOS[: args.warmup_queries]),
            "final_warmup_queries_per_selected_configuration": args.warmup_queries,
            "final_warmup_excluded_from_summaries": True,
            "deduplication_key": ["configured_filter_strategy", "filter_name", "ef"],
            "runs_only_selected_configurations": True,
            "comparison_common_attainable_targets_by_filter": common_targets,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "resume": bool(args.resume),
            "block_boundary": "complete calibration or final block; completed final blocks skip warmup and timed queries",
            "run_spec_hash": run_spec_hash(specification),
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_ci_lcb": "reported_only",
        "schema_timings": schema_timings,
        "node_records": node_records,
        "measurement_mode": "single_client_sequential",
    }
    outcome_counts = target_outcome_counts(final_summaries, target_statuses.values())
    outcome_notes = [
        f"held-out target unconfirmed: strategy={row.get('configured_filter_strategy')} filter={row.get('filter_name')} target={row.get('target_recall')}"
        for row in final_summaries if row.get("target_outcome") == "selected_but_final_unconfirmed"
    ]
    manifest = {
        "artifact_valid": not errors,
        "status": "complete" if not errors else "invalid",
        "git_revision": revision,
        "source_hashes": source_hashes,
        "run_spec_hash": run_spec_hash(specification),
        "stale_outputs_quarantined_at": str(quarantined_outputs) if quarantined_outputs else NA,
        "service": {
            "meta": service_meta,
            "count": total_count,
            "candidate_universe": {
                "predicate": CANDIDATE_VALIDITY_PREDICATE,
                "count": candidate_universe_count,
            },
            "filter_counts": filter_counts,
            "result_order": "ascending _additional.distance",
            "measurement_mode": "single_client_sequential",
            "concurrency": 1,
            "qps_metric": "single_client_service_qps",
            ACORN_RATIO_ENV: {
                "runner_environment": os.environ.get(ACORN_RATIO_ENV, NA),
                "service_meta_values": _find_meta_values(service_meta, ACORN_RATIO_ENV) or [NA],
            },
            "errors": errors,
        },
        "outcome_notes": outcome_notes,
        "target_outcomes": outcome_counts,
        "target_selection_rule": TARGET_SELECTION_RULE,
        "strategy_reporting": {"acorn": acorn_reporting_metadata(service_meta)} if "acorn" in strategies else {},
        "schema": {
            "class": CLASS_NAME,
            "initial_definition_recorded": True,
            "initial_definition_restored": True,
            "update_method": "GET full definition, PUT full definition, GET readback gate",
            "flatSearchCutoff": 0,
            "distance": "l2-squared",
            "vector_index_type": "hnsw",
            "configured_filter_strategies": strategies,
        },
        "inputs": {
            "fbin_rows": vector_rows,
            "fbin_dimensions": dimensions,
            "truth_schema": "tie-aware self-excluded v1",
            "truth_grid": "q20-q99 calibration / q100-q199 held-out final; q0-q19 reserved for pgvector screen",
        },
        "calibration_selection": {
            "rule": TARGET_SELECTION_RULE,
            "bootstrap_ci_lcb": "reported_only",
            "targets": selection_status_records,
            "common_attainable_targets_by_filter": common_targets,
        },
        "selection_grid_expected": len(strategies) * len(filters) * len(args.targets),
        "selection_winners": sum(winner is not None for winner in selections.values()),
        "unattainable_on_grid": outcome_counts["unattainable_on_grid"],
        "final_targets_met": outcome_counts["selected_and_confirmed"],
        "raw_rows": len(raw_rows),
        "summary_rows": len(calibration_summaries) + len(final_summaries),
    }
    staged = staging_outputs(outputs)
    try:
        write_csv(staged["raw_csv"], raw_rows)
        write_csv(staged["summary_csv"], calibration_summaries + final_summaries)
        write_json(staged["schema_json"], {"class": CLASS_NAME, "git_revision": revision, "source_hashes": source_hashes, "records": schema_records})
        write_json(staged["config_json"], config)
        manifest = commit_output_bundle(outputs, staged, manifest)
    finally:
        cleanup_staging(staged)
    checkpoint.unlink(missing_ok=True)
    return 0 if manifest["artifact_valid"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    try:
        return run(args)
    except Exception as exc:
        outputs = sibling_outputs(args.out)
        failure = {"artifact_valid": False, "git_revision": git_revision(), "errors": [f"{exc.__class__.__name__}: {exc}"], "outputs": {name: str(path) for name, path in outputs.items()}, "output_sha256": {}, "manifest_commit": "atomic_last"}
        try:
            isolate_existing_outputs(outputs)
            atomic_write_json(outputs["manifest_json"], failure)
        except OSError:
            pass
        print(f"artifact_valid=false: {failure['errors'][0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
