import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/hybrid_vector_db/scripts/weaviate_matched_recall_baseline.py"
SPEC = importlib.util.spec_from_file_location("weaviate_matched_recall_baseline", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def comparison(path, operator, value_key, value):
    return {"path": [path], "operator": operator, value_key: value}


GOLDEN_WHERE = {
    "popular_ge1000": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 1000),
    "popular_ge1340": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 1340),
    "popular_ge1780": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 1780),
    "popular_ge2428": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 2428),
    "popular_ge3284": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 3284),
    "popular_ge4559": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 4559),
    "price_10_to_20": {
        "operator": "And",
        "operands": [
            comparison("has_price", "Equal", "valueBoolean", True),
            comparison("price", "GreaterThan", "valueNumber", 10.0),
            comparison("price", "LessThanEqual", "valueNumber", 20.0),
        ],
    },
    "popular_ge10066": comparison("item_rating_number", "GreaterThanEqual", "valueInt", 10066),
    "rating5_price_le10": {
        "operator": "And",
        "operands": [
            comparison("has_price", "Equal", "valueBoolean", True),
            comparison("price", "LessThanEqual", "valueNumber", 10.0),
            comparison("rating", "Equal", "valueNumber", 5.0),
        ],
    },
    "long_review_ge500": comparison("review_text_len", "GreaterThanEqual", "valueInt", 500),
    "grocery_rating5": {
        "operator": "And",
        "operands": [
            comparison("main_category", "Equal", "valueText", "Grocery"),
            comparison("rating", "Equal", "valueNumber", 5.0),
        ],
    },
    "grocery_helpful": {
        "operator": "And",
        "operands": [
            comparison("main_category", "Equal", "valueText", "Grocery"),
            comparison("helpful_vote", "GreaterThanEqual", "valueInt", 1),
        ],
    },
    "helpful_ge20": comparison("helpful_vote", "GreaterThanEqual", "valueInt", 20),
    "grocery_long500": {
        "operator": "And",
        "operands": [
            comparison("main_category", "Equal", "valueText", "Grocery"),
            comparison("review_text_len", "GreaterThanEqual", "valueInt", 500),
        ],
    },
}


GOLDEN_PREDICATE_AST = {
    "popular_ge1000": ("comparison", "item_rating_number", ">=", 1000),
    "popular_ge1340": ("comparison", "item_rating_number", ">=", 1340),
    "popular_ge1780": ("comparison", "item_rating_number", ">=", 1780),
    "popular_ge2428": ("comparison", "item_rating_number", ">=", 2428),
    "popular_ge3284": ("comparison", "item_rating_number", ">=", 3284),
    "popular_ge4559": ("comparison", "item_rating_number", ">=", 4559),
    "price_10_to_20": (
        "and",
        (
            ("comparison", "has_price", "=", True),
            ("comparison", "price", ">", 10.0),
            ("comparison", "price", "<=", 20.0),
        ),
    ),
    "popular_ge10066": ("comparison", "item_rating_number", ">=", 10066),
    "rating5_price_le10": (
        "and",
        (
            ("comparison", "has_price", "=", True),
            ("comparison", "price", "<=", 10.0),
            ("comparison", "rating", "=", 5.0),
        ),
    ),
    "long_review_ge500": ("comparison", "review_text_len", ">=", 500),
    "grocery_rating5": (
        "and",
        (
            ("comparison", "main_category", "=", "Grocery"),
            ("comparison", "rating", "=", 5.0),
        ),
    ),
    "grocery_helpful": (
        "and",
        (
            ("comparison", "main_category", "=", "Grocery"),
            ("comparison", "helpful_vote", ">=", 1),
        ),
    ),
    "helpful_ge20": ("comparison", "helpful_vote", ">=", 20),
    "grocery_long500": (
        "and",
        (
            ("comparison", "main_category", "=", "Grocery"),
            ("comparison", "review_text_len", ">=", 500),
        ),
    ),
}


def schema_definition(strategy="acorn", ef=500):
    properties = [
        {"name": name, "dataType": [data_type], "indexFilterable": True}
        for name, data_type in runner.PROPERTY_TYPES.items()
    ]
    return {
        "class": runner.CLASS_NAME,
        "description": "complete definition marker",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "l2-squared",
            "flatSearchCutoff": 0,
            "filterStrategy": strategy,
            "ef": ef,
            "maxConnections": 32,
        },
        "properties": properties,
        "replicationConfig": {"factor": 1},
    }


def ready_nodes(status="HEALTHY", mode="ReadWrite", indexing="READY"):
    return {
        "nodes": [
            {
                "name": "node1",
                "status": status,
                "operationalMode": mode,
                "shards": [
                    {
                        "name": "shard1",
                        "class": runner.CLASS_NAME,
                        "loaded": True,
                        "vectorIndexingStatus": indexing,
                        "vectorQueueLength": 0,
                    }
                ],
            }
        ]
    }


def truth_entry(query_no=0, query_id=99, filter_name="f", kth=1.0, tolerance=1e-6):
    return runner.TruthEntry(
        query_no=query_no,
        query_id=query_id,
        filter_name=filter_name,
        split="calibration",
        filtered_rows=100,
        k=runner.K,
        kth_distance_sq=kth,
        tie_tolerance=tolerance,
        self_excluded=True,
    )


class WeaviateMatchedRecallBaselineTests(unittest.TestCase):
    def test_formal_query_split_reserves_pgvector_screen(self):
        self.assertEqual(runner.CALIBRATION_QUERY_NOS, tuple(range(20, 100)))
        self.assertEqual(runner.FINAL_QUERY_NOS, tuple(range(100, 200)))
        self.assertTrue(set(range(20)) not in (set(runner.CALIBRATION_QUERY_NOS), set(runner.FINAL_QUERY_NOS)))

    def test_final_warmup_is_explicit_and_excluded_from_final_summary(self):
        spec = runner.FILTERS[0]
        args = SimpleNamespace(timeout=1.0, retries=1)
        vectors = np.zeros((20, 2), dtype=np.float32)
        truth = {(spec.name, 100): truth_entry(100, 19, spec.name)}
        result = runner.QueryResult(tuple(range(10)), 1.0, 0, "", "")
        with mock.patch.object(runner, "query_once", return_value=result), mock.patch.object(runner, "exact_squared_l2", return_value=(0.0,) * 10):
            rows = runner._run_measurements(
                args, "http://unused", vectors, truth, {100: 19}, spec, "acorn", 100,
                "final_warmup", [100], 1,
            )
        self.assertEqual(rows[0]["phase"], "final_warmup")
        self.assertEqual(
            runner.summarize_configuration(
                rows, strategy="acorn", filter_name=spec.name, ef=100,
                query_nos=[100], repeats=1, bootstrap_seed=1, phase="final",
            )["complete"],
            False,
        )

    def test_http_error_preserves_weaviate_response_body(self):
        failure = HTTPError(
            "http://unused/v1/graphql",
            422,
            "Unprocessable Entity",
            {},
            BytesIO(b'{"error":[{"message":"bad where operand"}]}'),
        )
        with mock.patch.object(runner, "urlopen", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "bad where operand"):
                runner.request_json("http://unused", "/v1/graphql", retries=0)

    def test_all_fourteen_predicates_match_golden_graphql_where_ast(self):
        self.assertEqual(set(GOLDEN_WHERE), set(runner.EXPECTED_FILTERS))
        self.assertEqual(set(GOLDEN_PREDICATE_AST), set(runner.EXPECTED_FILTERS))
        self.assertEqual(len(runner.FILTERS), 14)
        parsed = {spec.name: runner.predicate_ast(spec.predicate) for spec in runner.FILTERS}
        actual = {spec.name: runner.predicate_to_where(spec.predicate) for spec in runner.FILTERS}
        self.assertEqual(parsed, GOLDEN_PREDICATE_AST)
        self.assertEqual(actual, GOLDEN_WHERE)

    def test_truth_loader_rejects_retired_legacy_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            fields = ["query_no", "query_id", "filter_name", "method", "exact_filtered_topk_ids", "candidates", "query_split"]
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"query_no": 0, "query_id": 10, "filter_name": runner.FILTERS[0].name, "method": "pre_filter_exact", "exact_filtered_topk_ids": ",".join(str(i) for i in range(10)), "candidates": runner.FILTERS[0].expected_rows, "query_split": "calibration"})
            with self.assertRaisesRegex(ValueError, "retired legacy schema"):
                runner.load_truth(path, runner.FILTERS[:1], calibration_query_nos=[0], final_query_nos=[])

    def test_truth_loader_requires_tie_aware_self_excluded_contract(self):
        spec = runner.FILTERS[0]
        rows = [
            {"query_no": 0, "query_id": 10, "filter_name": spec.name, "method": "pre_filter_exact", "filtered_rows": spec.expected_rows, "k": 10, "kth_distance_sq": "0", "tie_tolerance": "1e-7", "self_excluded": "true", "query_split": "calibration"},
            {"query_no": 1, "query_id": 11, "filter_name": spec.name, "method": "pre_filter_exact", "filtered_rows": spec.expected_rows, "k": 10, "kth_distance_sq": "2.5", "tie_tolerance": "2.5e-6", "self_excluded": "true", "query_split": "final"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            truth, query_ids = runner.load_truth(path, [spec], calibration_query_nos=[0], final_query_nos=[1])
            self.assertEqual(query_ids, {0: 10, 1: 11})
            self.assertEqual(truth[(spec.name, 0)].kth_distance_sq, 0.0)
            rows[0]["self_excluded"] = "false"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "self_excluded=true"):
                runner.load_truth(path, [spec], calibration_query_nos=[0], final_query_nos=[1])

    def test_query_uses_k_plus_one_and_removes_query_object(self):
        objects = [
            {"row_id": row_id, "_additional": {"distance": float(position), "id": str(row_id)}}
            for position, row_id in enumerate([99, *range(10)])
        ]
        payload = {"data": {"Get": {runner.CLASS_NAME: objects}}}
        with mock.patch.object(runner, "graphql", return_value=(payload, 0)) as request:
            result = runner.query_once("http://unused", [0.0, 1.0], runner.FILTERS[0].where, 99, timeout=1, retries=0)
        self.assertEqual(result.ids, tuple(range(10)))
        self.assertGreater(result.latency_ms, 0.0)
        self.assertEqual(result.order_error, "")
        self.assertIn("limit:11", request.call_args.args[1])

    def test_query_preserves_short_ann_result_as_recall_evidence(self):
        objects = [
            {"row_id": 99, "_additional": {"distance": 0.0, "id": "99"}},
            {"row_id": 7, "_additional": {"distance": 0.25, "id": "7"}},
        ]
        payload = {"data": {"Get": {runner.CLASS_NAME: objects}}}
        with mock.patch.object(runner, "graphql", return_value=(payload, 0)):
            result = runner.query_once(
                "http://unused", [0.0, 1.0], runner.FILTERS[0].where, 99,
                timeout=1, retries=0,
            )
        self.assertEqual(result.ids, (7,))
        self.assertEqual(result.error, "")
        self.assertEqual(result.order_error, "")
        self.assertEqual((result.returned_count, result.request_limit, result.request_limit - result.returned_count), (2, 11, 9))
        row = runner.measurement_row(
            phase="calibration", strategy="acorn", spec=runner.FILTERS[0], ef=100,
            query_no=0, query_id=99, repeat=0, result=result,
            truth=truth_entry(query_id=99, filter_name=runner.FILTERS[0].name),
            result_distances_sq=(0.25,),
        )
        self.assertTrue(row["valid"])
        self.assertEqual(row["recall_at_10"], 0.1)
        self.assertEqual((row["returned_count"], row["shortfall"]), (2, 9))

    def test_tie_aware_recall_uses_exact_fbin_squared_l2_not_id_intersection(self):
        vectors = np.arange(11, dtype=np.float32).reshape(-1, 1)
        distances = runner.exact_squared_l2(vectors, 0, range(1, 11))
        self.assertEqual(distances[:3], (1.0, 4.0, 9.0))
        truth = truth_entry(query_id=0, kth=4.0, tolerance=0.0)
        self.assertEqual(runner.tie_aware_recall(distances, truth), 0.2)
        self.assertEqual(runner.tie_aware_recall([0.0] * 20, truth), 1.0)

    def test_schema_update_gets_and_puts_complete_definition_then_gates_readback(self):
        initial = schema_definition("sweeping", 100)
        expected = schema_definition("acorn", 250)
        responses = [(initial, 0), ({}, 0), (expected, 0)]
        with mock.patch.object(runner, "request_json", side_effect=responses) as request:
            readback, _, retries = runner.put_hnsw_config("http://unused", "acorn", 250, 1.0, 2)
        self.assertEqual(readback, expected)
        self.assertEqual(retries, 0)
        self.assertEqual(initial["vectorIndexConfig"]["filterStrategy"], "sweeping")
        put_call = request.call_args_list[1]
        self.assertEqual(put_call.kwargs["method"], "PUT")
        self.assertEqual(put_call.args[2], expected)
        self.assertIn("properties", put_call.args[2])
        self.assertIn("replicationConfig", put_call.args[2])

    def test_node_gate_requires_readwrite_healthy_ready_verbose_shard(self):
        self.assertEqual(runner.verify_node_ready(ready_nodes()), ready_nodes())
        for payload in (
            ready_nodes(status="UNHEALTHY"),
            ready_nodes(mode="ReadOnly"),
            ready_nodes(indexing="READONLY"),
            {"nodes": [{"name": "node1", "status": "HEALTHY", "operationalMode": "ReadWrite", "shards": None}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "node gate failed"):
                runner.verify_node_ready(payload)

    def test_run_restores_initial_complete_schema_when_configuration_update_fails(self):
        spec = runner.FILTERS[0]
        initial = schema_definition("sweeping", 100)
        vectors = np.zeros((200, 2), dtype=np.float32)
        query_ids = {query_no: query_no for query_no in range(200)}
        truth = {
            (spec.name, query_no): truth_entry(query_no, query_no, spec.name)
            for query_no in range(200)
        }
        args = runner.build_parser().parse_args(["--ef-values", "100", "--out", "/unused/out.csv"])
        total = {"data": {"Aggregate": {runner.CLASS_NAME: [{"meta": {"count": runner.EXPECTED_ROWS}}]}}}
        filtered = {"data": {"Aggregate": {runner.CLASS_NAME: [{"meta": {"count": spec.expected_rows}}]}}}
        with mock.patch.object(runner, "isolate_existing_outputs", return_value=None), mock.patch.object(runner, "load_filter_specs", return_value=(spec,)), mock.patch.object(runner, "read_fbin_memmap", return_value=(vectors, len(vectors), 2)), mock.patch.object(runner, "load_truth", return_value=(truth, query_ids)), mock.patch.object(runner, "request_json", side_effect=[(initial, 0), ({"version": "test"}, 0), (ready_nodes(), 0)]), mock.patch.object(runner, "graphql", side_effect=[(total, 0), (filtered, 0)]), mock.patch.object(runner, "put_hnsw_config", side_effect=RuntimeError("update failed")), mock.patch.object(runner, "put_schema_definition", return_value=(initial, 0)) as restore, mock.patch.object(runner, "get_ready_nodes", side_effect=[(ready_nodes(), 0), (ready_nodes(), 0)]):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                runner.run(args)
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(restore.call_args.args[1], initial)

    def test_run_does_not_put_schema_when_validation_fails_before_first_update(self):
        spec = runner.FILTERS[0]
        initial = schema_definition("acorn", 100)
        vectors = np.zeros((200, 2), dtype=np.float32)
        query_ids = {query_no: query_no for query_no in range(200)}
        truth = {
            (spec.name, query_no): truth_entry(query_no, query_no, spec.name)
            for query_no in range(200)
        }
        args = runner.build_parser().parse_args(["--ef-values", "100", "--out", "/unused/out.csv"])
        with mock.patch.object(runner, "isolate_existing_outputs", return_value=None), mock.patch.object(runner, "load_filter_specs", return_value=(spec,)), mock.patch.object(runner, "read_fbin_memmap", return_value=(vectors, len(vectors), 2)), mock.patch.object(runner, "load_truth", return_value=(truth, query_ids)), mock.patch.object(runner, "request_json", side_effect=[(initial, 0), ({"version": "test"}, 0)]), mock.patch.object(runner, "get_ready_nodes", return_value=(ready_nodes(), 0)), mock.patch.object(runner, "graphql", side_effect=RuntimeError("count failed")), mock.patch.object(runner, "put_schema_definition") as restore:
            with self.assertRaisesRegex(RuntimeError, "count failed"):
                runner.run(args)
        restore.assert_not_called()

    def test_schema_gate_checks_readback_and_filterable_properties(self):
        schema = schema_definition()
        self.assertEqual(runner.schema_gate(schema, "acorn", 500), [])
        bad = {**schema, "vectorIndexConfig": {**schema["vectorIndexConfig"], "flatSearchCutoff": 40000}}
        self.assertTrue(runner.schema_gate(bad, "acorn", 500))
        with self.assertRaisesRegex(RuntimeError, "schema gate"):
            runner.verify_schema(schema, "sweeping", 500)

    def test_calibration_schedule_is_interleaved_rotated_and_deterministic(self):
        expected = [
            (0, "acorn", 100),
            (1, "sweeping", 100),
            (2, "sweeping", 250),
            (3, "acorn", 250),
            (4, "acorn", 500),
            (5, "sweeping", 500),
        ]
        self.assertEqual(runner.calibration_configuration_schedule(["acorn", "sweeping"], [100, 250, 500]), expected)
        self.assertEqual(runner.measurement_schedule([0, 1, 2], 2, 1), [(1, 0), (2, 0), (0, 0), (2, 1), (0, 1), (1, 1)])

    def test_final_targets_with_same_strategy_filter_ef_share_one_group(self):
        winner = {"ef": 250}
        selections = {
            ("acorn", "f", 0.90): winner,
            ("acorn", "f", 0.95): winner,
            ("acorn", "f", 0.99): None,
        }
        self.assertEqual(runner.group_selected_targets(selections), {("acorn", "f", 250): [0.90, 0.95]})

    def test_timed_measurements_force_zero_retries_and_retry_invalidates_row(self):
        spec = runner.FILTERS[0]
        args = SimpleNamespace(timeout=1.0, retries=4)
        vectors = np.zeros((20, 2), dtype=np.float32)
        truth = {(spec.name, 0): truth_entry(0, 19, spec.name)}
        result = runner.QueryResult(tuple(range(10)), 1.0, 1, "", "")
        with mock.patch.object(runner, "query_once", return_value=result) as query, mock.patch.object(runner, "exact_squared_l2", return_value=(0.0,) * 10):
            rows = runner._run_measurements(args, "http://unused", vectors, truth, {0: 19}, spec, "acorn", 100, "calibration", [0], 1)
        self.assertEqual(query.call_args.kwargs["retries"], 0)
        self.assertFalse(rows[0]["valid"])
        summary = runner.summarize_configuration(rows, strategy="acorn", filter_name=spec.name, ef=100, query_nos=[0], repeats=1, bootstrap_seed=1)
        self.assertFalse(summary["complete"])

    def test_summary_reports_single_client_service_qps_only(self):
        row = {"phase": "calibration", "strategy": runner.ACORN_REPORTED_STRATEGY, "configured_filter_strategy": "acorn", "filter_name": "f", "ef": 100, "query_no": 0, "repeat": 0, "valid": True, "retry_count": 0, "error": "", "order_error": "", "recall_at_10": 1.0, "end_to_end_ms": 10.0}
        summary = runner.summarize_configuration([row], strategy="acorn", filter_name="f", ef=100, query_nos=[0], repeats=1, bootstrap_seed=1)
        self.assertEqual(summary["single_client_service_qps"], 100.0)
        self.assertNotIn("throughput", json.dumps(summary).lower())
        self.assertEqual(summary["strategy"], runner.ACORN_REPORTED_STRATEGY)

    def test_selection_missing_winner_is_invalid_but_final_target_miss_is_a_valid_outcome(self):
        spec = runner.FILTERS[0]
        candidates = [
            {"ef": 100, "complete": True, "recall_mean": 0.91, "recall_lcb95": 0.50, "latency_mean_ms": 10.0},
            {"ef": 250, "complete": True, "recall_mean": 0.96, "recall_lcb95": 0.50, "latency_mean_ms": 20.0},
        ]
        self.assertEqual(runner.select_fastest_config(candidates, 0.95)["ef"], 250)
        self.assertIsNone(runner.select_fastest_config(candidates, 0.99))
        missing_key = ("acorn", spec.name, 0.99)
        errors = runner.artifact_gate_errors(strategies=["acorn"], filters=[spec], targets=[0.99], selections={missing_key: None}, final_summaries=[], raw_rows=[])
        self.assertTrue(any("missing calibration winner" in error for error in errors))
        winner = {"ef": 250, "complete": True, "recall_mean": 0.99, "recall_lcb95": 0.50, "latency_mean_ms": 20.0}
        final = runner._summary_row_for_target(
            {"strategy": runner.ACORN_REPORTED_STRATEGY, "configured_filter_strategy": "acorn", "filter_name": spec.name, **winner},
            0.99, winner,
            {"complete": True, "recall_mean": 0.96, "recall_lcb95": 0.96},
            "selected",
        )
        errors = runner.artifact_gate_errors(
            strategies=["acorn"], filters=[spec], targets=[0.99],
            selections={missing_key: winner}, final_summaries=[final], raw_rows=[],
        )
        self.assertEqual(errors, [])
        self.assertEqual(final["target_outcome"], "selected_but_final_unconfirmed")
        self.assertEqual(final["recall_mean"], 0.96)

    def test_measurement_integrity_rejects_real_errors_and_missing_pairs(self):
        errors = runner.artifact_gate_errors(
            strategies=[], filters=[], targets=[], selections={}, final_summaries=[],
            raw_rows=[{"phase": "final", "valid": False, "error": "GraphQL error", "order_error": "", "retry_count": 0}],
        )
        self.assertTrue(any("invalid timed measurement" in error for error in errors))
        summary = {
            "configured_filter_strategy": "acorn", "filter_name": "f", "ef": 100,
            "complete": True, "recall_mean": 1.0, "recall_lcb95": 1.0,
            "recall_ci95_low": 1.0, "recall_ci95_high": 1.0,
            "latency_mean_ms": 1.0, "latency_p50_ms": 1.0, "latency_p95_ms": 1.0,
            "latency_p99_ms": 1.0, "latency_ci95_low_ms": 1.0,
            "latency_ci95_high_ms": 1.0, "single_client_service_qps": 1000.0,
        }
        errors = runner.measurement_block_integrity_errors(
            [], [summary], phase="calibration", query_nos=[0], repeats=2,
            block_fields=("configured_filter_strategy", "filter_name", "ef"),
        )
        self.assertTrue(any("coverage mismatch" in error for error in errors))

    def test_target_outcome_counts_keep_confirmed_unconfirmed_and_unattainable_separate(self):
        counts = runner.target_outcome_counts(
            [{"target_outcome": "selected_and_confirmed"}, {"target_outcome": "selected_but_final_unconfirmed"}],
            ["selected", "unattainable_on_grid"],
        )
        self.assertEqual(counts, {
            "selected_and_confirmed": 1,
            "selected_but_final_unconfirmed": 1,
            "unattainable_on_grid": 1,
        })

    def test_monotone_calibration_stops_pair_after_highest_target_and_keeps_fastest_low_target(self):
        candidates = [
            {"ef": 100, "complete": True, "recall_mean": 0.91, "recall_lcb95": 0.50, "latency_mean_ms": 4.0},
            {"ef": 250, "complete": True, "recall_mean": 0.99, "recall_lcb95": 0.50, "latency_mean_ms": 9.0},
        ]
        self.assertFalse(runner.pair_reached_highest_target(candidates[:1], 0.99))
        self.assertTrue(runner.pair_reached_highest_target(candidates, 0.99))
        self.assertEqual(runner.select_fastest_config(candidates, 0.90)["ef"], 100)
        self.assertEqual(runner.select_fastest_config(candidates, 0.99)["ef"], 250)

    def test_unattainable_on_grid_is_valid_when_complete_max_grid_is_proven(self):
        spec = runner.FILTERS[0]
        candidates = [
            {"ef": 100, "complete": True, "recall_lcb95": 0.91, "latency_mean_ms": 4.0},
            {"ef": 250, "complete": True, "recall_lcb95": 0.94, "latency_mean_ms": 9.0},
        ]
        key = ("acorn", spec.name, 0.99)
        self.assertEqual(runner.calibration_target_status(candidates, 0.99, [100, 250]), "unattainable_on_grid")
        errors = runner.artifact_gate_errors(
            strategies=["acorn"],
            filters=[spec],
            targets=[0.99],
            selections={key: None},
            target_statuses={key: "unattainable_on_grid"},
            final_summaries=[{"configured_filter_strategy": "acorn", "filter_name": spec.name, "target_recall": 0.99, "target_status": "unattainable_on_grid"}],
            raw_rows=[],
        )
        self.assertEqual(errors, [])

    def test_checkpoint_run_spec_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            runner.atomic_write_json(path, {"version": 1, "run_spec": {"run": "old"}, "run_spec_hash": runner.run_spec_hash({"run": "old"}), "raw_rows": [], "calibration_summaries": [], "final_results": [], "state": {"completed_blocks": []}})
            with self.assertRaisesRegex(RuntimeError, "run-spec/hash mismatch"):
                runner.load_checkpoint(path, {"run": "new"}, {})

    def test_complete_checkpoint_block_is_validated_and_available_for_resume_skip(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "CALIBRATION_QUERY_NOS", (0, 1)), mock.patch.object(runner, "CALIBRATION_REPEATS", 1):
            path = Path(tmp) / "checkpoint.json"
            specification = {"run": "same"}
            block = runner._block_record("calibration", "acorn", "f", 100)
            rows = [
                {"phase": "calibration", "configured_filter_strategy": "acorn", "filter_name": "f", "ef": 100, "query_no": query_no, "query_id": query_no, "repeat": 0}
                for query_no in (0, 1)
            ]
            summary = {"configured_filter_strategy": "acorn", "filter_name": "f", "ef": 100}
            runner.write_checkpoint(path, specification, raw_rows=rows, calibration_summaries=[summary], final_results=[], state={"completed_blocks": [block], "schema_records": [], "schema_timings": [], "node_records": []})
            restored = runner.load_checkpoint(path, specification, {0: 0, 1: 1})
            completed = {
                runner._block_key(item["phase"], item["configured_filter_strategy"], item["filter_name"], item["ef"])
                for item in restored["state"]["completed_blocks"]
            }
            self.assertIn(runner._block_key("calibration", "acorn", "f", 100), completed)

    def test_successful_run_commits_bundle_then_removes_checkpoint(self):
        spec = runner.FILTERS[0]
        initial = schema_definition("acorn", 100)
        vectors = np.zeros((20, 2), dtype=np.float32)
        query_ids = {0: 19, 1: 19}
        truth = {(spec.name, query_no): truth_entry(query_no, 19, spec.name) for query_no in query_ids}
        total = {"data": {"Aggregate": {runner.CLASS_NAME: [{"meta": {"count": runner.EXPECTED_ROWS}}]}}}
        filtered = {"data": {"Aggregate": {runner.CLASS_NAME: [{"meta": {"count": spec.expected_rows}}]}}}
        result = runner.QueryResult(tuple(range(10)), 1.0, 0, "", "")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "CALIBRATION_QUERY_NOS", (0,)), mock.patch.object(runner, "FINAL_QUERY_NOS", (1,)), mock.patch.object(runner, "CALIBRATION_REPEATS", 1), mock.patch.object(runner, "FINAL_REPEATS", 1):
            out = Path(tmp) / "run.csv"
            args = runner.build_parser().parse_args(["--ef-values", "100", "250", "--targets", "0.9", "--warmup-queries", "1", "--out", str(out)])
            with mock.patch.object(runner, "load_filter_specs", return_value=(spec,)), mock.patch.object(runner, "read_fbin_memmap", return_value=(vectors, len(vectors), 2)), mock.patch.object(runner, "load_truth", return_value=(truth, query_ids)), mock.patch.object(runner, "sha256_file", return_value="hash"), mock.patch.object(runner, "request_json", side_effect=[(initial, 0), ({"version": "test"}, 0)]), mock.patch.object(runner, "get_ready_nodes", return_value=(ready_nodes(), 0)), mock.patch.object(runner, "graphql", side_effect=[(total, 0), (filtered, 0)]), mock.patch.object(runner, "put_hnsw_config", return_value=(initial, 0.0, 0)) as put_config, mock.patch.object(runner, "put_schema_definition", return_value=(initial, 0)), mock.patch.object(runner, "query_once", return_value=result):
                self.assertEqual(runner.run(args), 0)
            self.assertEqual([call.args[2] for call in put_config.call_args_list], [100, 100])
            self.assertTrue(out.is_file())
            raw_text = runner.sibling_outputs(out)["raw_csv"].read_text(encoding="utf-8")
            summary_text = runner.sibling_outputs(out)["summary_csv"].read_text(encoding="utf-8")
            self.assertIn("final_warmup", raw_text)
            self.assertNotIn("final_warmup", summary_text)
            self.assertFalse(runner.checkpoint_path(out).exists())

    def test_acorn_reporting_records_auto_fallback_and_ratio_sources(self):
        meta = {"configuration": {runner.ACORN_RATIO_ENV: "0.6"}}
        with mock.patch.dict(os.environ, {runner.ACORN_RATIO_ENV: "0.4"}):
            record = runner.acorn_reporting_metadata(meta)
        self.assertEqual(record["reported_strategy"], runner.ACORN_REPORTED_STRATEGY)
        self.assertFalse(record["effective_path_proven"])
        self.assertEqual(record[runner.ACORN_RATIO_ENV]["runner_environment"], "0.4")
        self.assertEqual(record[runner.ACORN_RATIO_ENV]["service_meta_values"], ["0.6"])

    def test_artifact_bundle_isolates_old_files_hashes_outputs_and_commits_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = runner.sibling_outputs(Path(tmp) / "run.csv")
            for path in outputs.values():
                path.write_text("old", encoding="utf-8")
            quarantine = runner.isolate_existing_outputs(outputs)
            self.assertIsNotNone(quarantine)
            self.assertTrue(all(not path.exists() for path in outputs.values()))
            self.assertEqual(len(list(quarantine.iterdir())), len(outputs))

            staged = runner.staging_outputs(outputs)
            for name, path in staged.items():
                if name != "manifest_json":
                    path.write_text(name, encoding="utf-8")
            manifest = runner.commit_output_bundle(outputs, staged, {"artifact_valid": True})
            self.assertEqual(manifest["manifest_commit"], "atomic_last")
            self.assertTrue(outputs["manifest_json"].is_file())
            committed = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
            for name, digest in committed["output_sha256"].items():
                self.assertEqual(digest, hashlib.sha256(name.encode()).hexdigest())

    def test_k_is_forced_to_ten_and_dry_run_touches_no_network_or_files(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            runner.build_parser().parse_args(["--k", "9"])
        with mock.patch.object(runner, "urlopen", side_effect=AssertionError("network")), mock.patch.object(Path, "open", side_effect=AssertionError("file")), mock.patch.object(Path, "write_text", side_effect=AssertionError("write")):
            self.assertEqual(runner.main(["--dry-run", "--filters-csv", "/missing/filters.csv", "--truth-csv", "/missing/truth.csv", "--fbin", "/missing/vectors.fbin", "--out", "/missing/out.csv"]), 0)


if __name__ == "__main__":
    unittest.main()
