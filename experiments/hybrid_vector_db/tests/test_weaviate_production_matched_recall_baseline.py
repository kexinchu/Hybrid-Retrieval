import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/hybrid_vector_db/scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "weaviate_production_matched_recall_baseline.py"
SPEC = importlib.util.spec_from_file_location("weaviate_production_matched_recall_baseline", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def schema_definition(strategy="acorn", ef=500, cutoff=100_000):
    properties = [
        {"name": name, "dataType": [data_type], "indexFilterable": True}
        for name, data_type in runner.baseline.PROPERTY_TYPES.items()
    ]
    return {
        "class": runner.CLASS_NAME,
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "l2-squared", "filterStrategy": strategy, "ef": ef,
            "flatSearchCutoff": cutoff, "maxConnections": 32,
        },
        "properties": properties,
        "replicationConfig": {"factor": 1},
    }


def summary(
    ef, cutoff, lcb, latency, complete=True, strategy="acorn", *,
    recall_mean=None, ci_low=None, ci_high=None,
):
    return {
        "configured_filter_strategy": strategy, "filter_name": "f", "ef": ef,
        "flat_search_cutoff": cutoff, "complete": complete, "recall_lcb95": lcb,
        "recall_mean": lcb if recall_mean is None else recall_mean,
        "latency_mean_ms": latency,
        "latency_ci95_low_ms": latency - 0.5 if ci_low is None else ci_low,
        "latency_ci95_high_ms": latency + 0.5 if ci_high is None else ci_high,
    }


def filter_spec(name="f", expected_rows=50_000):
    original = runner.baseline.FILTERS[0]
    return runner.FilterSpec(
        original.target_rate, name, original.predicate, expected_rows,
        expected_rows / runner.baseline.EXPECTED_ROWS * 100.0, original.where,
    )


def checkpoint_state(original_schema, blocks=None):
    return {
        "completed_blocks": list(blocks or []),
        "schema_records": [{"phase": "original_schema_snapshot", "schema": original_schema}],
        "schema_timings": [],
        "node_records": [],
        "original_schema": original_schema,
        "original_schema_sha256": runner.schema_snapshot_hash(original_schema),
    }


class WeaviateProductionMatchedRecallBaselineTests(unittest.TestCase):
    def test_default_cutoff_grid_includes_zero_and_covers_all_amazon_filters(self):
        self.assertEqual(runner.DEFAULT_FLAT_SEARCH_CUTOFFS[0], 0)
        self.assertGreater(runner.DEFAULT_FLAT_SEARCH_CUTOFFS[-1], max(
            spec.expected_rows for spec in runner.baseline.FILTERS
        ))

    def test_effective_cutoffs_reduce_grid_to_two_semantic_routes(self):
        spec = filter_spec(expected_rows=60_000)
        grid = [0, 25_000, 100_000, 250_000]
        self.assertEqual(runner.effective_cutoffs(spec, grid), (0, 100_000))
        proof = runner.cutoff_equivalence_proof(spec, grid)
        self.assertEqual(proof["skipped_equivalent_cutoffs"], [25_000, 250_000])
        self.assertEqual(proof["representative_for_grid_value"], {
            "0": 0, "25000": 0, "100000": 100_000, "250000": 100_000,
        })
        self.assertEqual(proof["flat_configuration_equivalence"]["representative"], {
            "configured_filter_strategy": "sweeping", "ef": 100,
            "flat_search_cutoff": 100_000,
        })
        self.assertEqual(
            proof["flat_configuration_equivalence"]["equivalent_configuration_count"],
            2 * len(runner.DEFAULT_EF_VALUES) * 2,
        )
        self.assertTrue(proof["complete_effective_semantics_coverage"])

    def test_schedule_is_flat_first_then_ascending_hnsw_per_route(self):
        schedule = runner.calibration_configuration_schedule(
            ["acorn", "sweeping"], [0, 100_000, 250_000], [100, 250]
        )
        self.assertEqual(
            schedule,
            [
                (0, "sweeping", 100_000, 100),
                (1, "sweeping", 250_000, 100),
                (2, "acorn", 0, 100), (3, "sweeping", 0, 100),
                (4, "sweeping", 0, 250), (5, "acorn", 0, 250),
            ],
        )
        for strategy in ("acorn", "sweeping"):
            self.assertEqual(
                [ef for _, item_strategy, item_cutoff, ef in schedule
                 if (item_strategy, item_cutoff) == (strategy, 0)],
                [100, 250],
            )
        self.assertEqual(
            [item for item in schedule if item[2] != 0],
            [(0, "sweeping", 100_000, 100), (1, "sweeping", 250_000, 100)],
        )

    def test_target_early_stop_is_recomputed_and_local_to_hnsw_route(self):
        filters = (filter_spec(),)
        flat = summary(100, 100_000, 1.0, 5.0, strategy="sweeping")
        stopped = [flat, summary(100, 0, 0.99, 10.0), summary(250, 0, 0.99, 12.0)]
        with self.assertRaisesRegex(RuntimeError, "highest_target_reached"):
            runner.validate_monotone_calibration_state(
                stopped, filters, [0, 100_000], [100, 250], 0.99, 1.05,
            )
        different_cutoff = [
            flat,
            summary(100, 0, 0.99, 10.0),
        ]
        runner.validate_monotone_calibration_state(
            different_cutoff, filters, [0, 100_000], [100, 250], 0.99, 1.05,
        )
        with self.assertRaisesRegex(RuntimeError, "unknown strategy/filter/cutoff route"):
            runner.validate_monotone_calibration_state(
                [summary(100, 100_000, 1.0, 5.0, strategy="acorn")],
                filters, [0, 100_000], [100, 250], 0.99, 1.05,
            )
        with self.assertRaisesRegex(RuntimeError, "single representative"):
            runner.validate_monotone_calibration_state(
                [summary(250, 100_000, 1.0, 5.0, strategy="sweeping")],
                filters, [0, 100_000], [100, 250], 0.99, 1.05,
            )

    def test_dominance_requires_strict_guarded_ci_and_two_point_monotonicity(self):
        spec = filter_spec()
        flat = summary(
            100, 100_000, 1.0, 5.0, strategy="sweeping",
            recall_mean=1.0, ci_low=4.5, ci_high=5.0,
        )
        first = summary(100, 0, 0.80, 7.0, ci_low=6.0, ci_high=8.0)
        latest = summary(250, 0, 0.85, 8.0, ci_low=7.0, ci_high=9.0)
        proof = runner.hnsw_dominance_proof(
            [flat, first, latest], "acorn", spec, [0, 100_000], [100, 250, 500], 1.05,
        )
        self.assertTrue(proof["dominance_proven"])
        self.assertTrue(proof["gates"]["hnsw_efs_are_an_ascending_grid_prefix"])
        self.assertFalse(proof["source_budget_proof"]["internal_route_observed"])

        false_cases = {
            "strict_ci_guard": summary(250, 0, 0.85, 8.0, ci_low=5.25, ci_high=9.0),
            "mean_decreased": summary(250, 0, 0.85, 6.5, ci_low=6.25, ci_high=9.0),
            "ci_lower_decreased": summary(250, 0, 0.85, 8.0, ci_low=5.75, ci_high=9.0),
        }
        comparison_first = {
            "strict_ci_guard": summary(100, 0, 0.80, 7.0, ci_low=5.0, ci_high=8.0),
            "mean_decreased": summary(100, 0, 0.80, 7.0, ci_low=6.0, ci_high=8.0),
            "ci_lower_decreased": summary(100, 0, 0.80, 7.0, ci_low=6.0, ci_high=8.0),
        }
        for name, candidate in false_cases.items():
            with self.subTest(name=name):
                rejected = runner.hnsw_dominance_proof(
                    [flat, comparison_first[name], candidate], "acorn", spec,
                    [0, 100_000], [100, 250, 500], 1.05,
                )
                self.assertFalse(rejected["dominance_proven"])

    def test_dominance_termination_status_is_not_unattainable(self):
        spec = filter_spec()
        rows = [
            summary(100, 100_000, 1.0, 5.0, strategy="sweeping",
                    recall_mean=1.0, ci_low=4.5, ci_high=5.0),
            summary(100, 0, 0.80, 7.0, ci_low=6.0),
            summary(250, 0, 0.85, 8.0, ci_low=7.0),
        ]
        termination = runner.hnsw_route_termination(
            rows, "acorn", spec, [0, 100_000], [100, 250, 500], 0.99, 1.05,
        )
        self.assertEqual(termination["termination_reason"], "dominated_by_exact_flat")
        self.assertEqual(
            runner.hnsw_route_target_status(rows[1:], 0.99, termination),
            "dominated_by_exact_flat",
        )
        proof = runner.configuration_grid_proof(
            rows, spec, [0, 100_000], [100, 250, 500], [0.90, 0.99], 1.05,
        )
        self.assertEqual(
            proof["hnsw_routes"]["acorn"]["target_statuses"]["0.99"],
            "dominated_by_exact_flat",
        )
        self.assertTrue(proof["hnsw_routes"]["acorn"]["termination"]["dominance_proof"]["dominance_proven"])

    def test_target_status_uses_unattainable_only_after_dominance_proof(self):
        spec = filter_spec()
        proof = {
            "hnsw_routes": {
                "acorn": {"termination": {"termination_reason": "dominated_by_exact_flat"}},
                "sweeping": {"termination": {"termination_reason": "full_grid_exhausted"}},
            },
            "all_hnsw_routes_fully_exhausted_without_errors": False,
        }
        with mock.patch.object(runner, "select_fastest_config", return_value=None), \
                mock.patch.object(runner, "configuration_grid_proof", return_value=proof):
            self.assertEqual(
                runner.calibration_target_status(
                    [], spec, 0.99, [0, 100_000], [100, 250], [0.99], 1.05,
                ),
                "unattainable_on_grid",
            )

    def test_resume_recomputes_dominance_and_rejects_blocks_after_stop(self):
        spec = filter_spec()
        flat = summary(100, 100_000, 1.0, 5.0, strategy="sweeping",
                       recall_mean=1.0, ci_low=4.5, ci_high=5.0)
        first = summary(100, 0, 0.80, 7.0, ci_low=6.0)
        second = summary(250, 0, 0.85, 8.0, ci_low=7.0)
        runner.validate_monotone_calibration_state(
            [flat, first, second], (spec,), [0, 100_000], [100, 250, 500], 0.99, 1.05,
        )
        third = summary(500, 0, 0.86, 8.5, ci_low=7.5)
        with self.assertRaisesRegex(RuntimeError, "dominated_by_exact_flat"):
            runner.validate_monotone_calibration_state(
                [flat, first, second, third], (spec,), [0, 100_000],
                [100, 250, 500], 0.99, 1.05,
            )

        tampered_second = summary(250, 0, 0.85, 8.0, ci_low=5.2)
        nondominating_third = summary(500, 0, 0.86, 8.5, ci_low=5.1)
        runner.validate_monotone_calibration_state(
            [flat, first, tampered_second, nondominating_third], (spec,),
            [0, 100_000], [100, 250, 500], 0.99, 1.05,
        )
        with self.assertRaisesRegex(RuntimeError, "before every flat representative"):
            runner.validate_monotone_calibration_state(
                [first], (spec,), [0, 100_000], [100, 250, 500], 0.99, 1.05,
            )

    def test_selection_uses_lowest_measured_latency_across_cutoff_and_ef(self):
        candidates = [
            summary(100, 0, 0.96, 8.0),
            summary(250, 0, 0.99, 11.0),
            summary(100, 0, 0.96, 7.0, strategy="sweeping"),
            summary(100, 100_000, 1.0, 5.0, strategy="sweeping"),
        ]
        winner = runner.select_fastest_config(candidates, 0.95)
        self.assertEqual((winner["flat_search_cutoff"], winner["ef"]), (100_000, 100))
        self.assertEqual(runner.select_fastest_config(candidates, 0.99)["flat_search_cutoff"], 100_000)

    def test_selection_uses_query_level_mean_and_reports_bootstrap_lcb(self):
        spec = filter_spec()
        candidates = [
            summary(250, 0, 0.923, 2.0, strategy="sweeping", recall_mean=0.943),
            summary(100, 100_000, 1.0, 5.0, strategy="sweeping", recall_mean=1.0),
        ]
        selected = runner.select_conservative_config(
            candidates, 0.90, spec, [0, 100_000], [100, 250], 0.05
        )
        self.assertEqual(selected["selection_mode"], "calibration_query_level_mean_recall")
        self.assertEqual((selected["flat_search_cutoff"], selected["ef"]), (0, 250))

        conservative = runner.select_conservative_config(
            [
                summary(250, 0, 0.951, 2.0, strategy="sweeping"),
                candidates[1],
            ],
            0.90, spec, [0, 100_000], [100, 250], 0.05,
        )
        self.assertEqual(conservative["selection_mode"], "calibration_query_level_mean_recall")
        self.assertEqual((conservative["flat_search_cutoff"], conservative["ef"]), (0, 250))
        self.assertAlmostEqual(runner.required_calibration_lcb(0.95, 0.05), 0.95)
        self.assertAlmostEqual(runner.required_calibration_lcb(0.99, 0.05), 0.99)

    def test_conservative_selection_fails_closed_without_exact_flat_calibration(self):
        spec = filter_spec()
        self.assertIsNone(runner.select_conservative_config(
            [summary(250, 0, 0.923, 2.0, strategy="sweeping", recall_mean=0.89)],
            0.90, spec, [0, 100_000], [100, 250], 0.05,
        ))

    def test_unattainable_requires_every_cutoff_route_complete_without_errors(self):
        spec = filter_spec()
        cutoff_grid = [0, 25_000, 100_000, 250_000]
        partial = [summary(100, 0, 0.80, 5.0), summary(250, 0, 0.85, 6.0)]
        self.assertEqual(
            runner.calibration_target_status(
                partial, spec, 0.99, cutoff_grid, [100, 250], [0.99], 1.05,
            ),
            "incomplete_grid",
        )
        complete = partial + [
            summary(100, 0, 0.80, 5.5, strategy="sweeping"),
            summary(250, 0, 0.85, 6.5, strategy="sweeping"),
            summary(100, 100_000, 0.98, 8.0, strategy="sweeping"),
        ]
        self.assertEqual(
            runner.calibration_target_status(
                complete, spec, 0.99, cutoff_grid, [100, 250], [0.99], 1.05,
            ),
            "unattainable_on_grid",
        )
        proof = runner.configuration_grid_proof(
            complete, spec, cutoff_grid, [100, 250], [0.99], 1.05,
        )
        self.assertEqual(set(proof["hnsw_routes"]), {"acorn", "sweeping"})
        self.assertTrue(proof["flat_route"]["complete_without_errors"])
        self.assertEqual(
            {route["termination"]["termination_reason"] for route in proof["hnsw_routes"].values()},
            {"full_grid_exhausted"},
        )

    def test_system_selection_groups_best_route_without_strategy_commonality(self):
        acorn = {"configured_filter_strategy": "acorn", "flat_search_cutoff": 0, "ef": 100}
        flat = {"configured_filter_strategy": "sweeping", "flat_search_cutoff": 100_000, "ef": 100}
        winner = runner.select_fastest_config([
            summary(100, 0, 0.89, 3.0, strategy="acorn"),
            summary(100, 0, 0.95, 4.0, strategy="sweeping"),
            summary(100, 100_000, 1.0, 8.0, strategy="sweeping"),
        ], 0.95)
        self.assertEqual(
            (winner["configured_filter_strategy"], winner["flat_search_cutoff"]),
            ("sweeping", 0),
        )
        selections = {
            ("f", 0.90): acorn,
            ("f", 0.95): flat,
        }
        spec = filter_spec()
        grouped = runner.group_selected_targets(selections)
        self.assertEqual(
            grouped,
            {("acorn", "f", 0, 100): [0.90], ("sweeping", "f", 100_000, 100): [0.95]},
        )
        self.assertEqual(runner.add_flat_exactness_controls(
            {("acorn", "f", 0, 100): [0.90]}, (spec,), [0, 100_000], [100, 250]
        ), {
            ("acorn", "f", 0, 100): [0.90],
            ("sweeping", "f", 100_000, 100): [],
        })

    def test_all_targets_can_reuse_one_exact_flat_measurement(self):
        flat = {
            "configured_filter_strategy": "sweeping", "flat_search_cutoff": 100_000,
            "ef": 100,
        }
        groups = runner.group_selected_targets({
            ("f", 0.90): flat,
            ("f", 0.95): flat,
            ("f", 0.99): flat,
        })
        self.assertEqual(groups, {("sweeping", "f", 100_000, 100): [0.90, 0.95, 0.99]})

    def test_default_effective_calibration_query_budget(self):
        budget = runner.calibration_query_budget(
            runner.baseline.FILTERS, runner.DEFAULT_EF_VALUES, warmup_queries=1
        )
        self.assertEqual(budget["maximum_effective_blocks_before_hnsw_early_stop"], 294)
        self.assertEqual(budget["flat_representative_blocks"], 14)
        self.assertEqual(budget["maximum_hnsw_blocks"], 280)
        self.assertEqual(budget["maximum_timed_queries_before_hnsw_early_stop"], 47_040)
        self.assertEqual(budget["configured_warmup_queries"], 294)
        self.assertEqual(budget["maximum_total_service_queries"], 47_334)

    def test_flat_held_out_result_is_an_exactness_gate(self):
        spec = filter_spec()
        key = ("sweeping", "f", 100_000, 100)
        exact = {
            "configured_filter_strategy": "sweeping", "filter_name": "f",
            "flat_search_cutoff": 100_000, "ef": 100, "complete": True,
            "recall_mean": 1.0, "recall_lcb95": 1.0,
        }
        errors = runner.artifact_gate_errors(
            {key: []}, [], [], [exact], (spec,), [0, 100_000], [100, 250]
        )
        self.assertEqual(errors, [])
        inexact = {**exact, "recall_mean": 0.99}
        errors = runner.artifact_gate_errors(
            {key: []}, [], [], [inexact], (spec,), [0, 100_000], [100, 250]
        )
        self.assertTrue(any("flat held-out exactness gate failed" in error for error in errors))

    def test_selected_final_target_miss_preserves_metrics_without_invalidating_artifact(self):
        spec = filter_spec()
        hnsw_key = ("acorn", "f", 0, 100)
        flat_key = ("sweeping", "f", 100_000, 100)
        selected = summary(100, 0, 0.99, 4.0)
        held_out = {**summary(100, 0, 0.96, 5.0), "phase": "final"}
        flat = {
            **summary(100, 100_000, 1.0, 3.0, strategy="sweeping"),
            "phase": "final",
        }
        final_summary = runner._summary_row_for_target(selected, 0.99, held_out)
        errors = runner.artifact_gate_errors(
            {hnsw_key: [0.99], flat_key: []}, [final_summary], [], [held_out, flat],
            (spec,), [0, 100_000], [100, 250],
        )
        self.assertEqual(errors, [])
        self.assertEqual(final_summary["target_outcome"], "selected_but_final_unconfirmed")
        self.assertEqual(final_summary["comparison_status"], "unconfirmed")
        self.assertEqual(final_summary["recall_mean"], 0.96)
        self.assertEqual(final_summary["latency_mean_ms"], 5.0)

    def test_artifact_gate_rejects_measurement_error_and_missing_block_pairs(self):
        errors = runner.artifact_gate_errors(
            {}, [], [{"phase": "final", "valid": False, "error": "GraphQL error", "order_error": "", "retry_count": 0}],
            [], (), [0, 100_000], [100],
        )
        self.assertTrue(any("invalid timed measurement" in error for error in errors))
        calibration = [{
            **summary(100, 0, 1.0, 1.0),
            "recall_ci95_low": 1.0, "recall_ci95_high": 1.0,
            "latency_p50_ms": 1.0, "latency_p95_ms": 1.0, "latency_p99_ms": 1.0,
            "single_client_service_qps": 1000.0,
        }]
        errors = runner.artifact_gate_errors(
            {}, [], [], [], (), [0, 100_000], [100], calibration,
        )
        self.assertTrue(any("calibration block coverage mismatch" in error for error in errors))

    def test_manifest_outcome_counts_keep_all_three_categories(self):
        self.assertEqual(
            runner.baseline.target_outcome_counts(
                [{"target_outcome": "selected_and_confirmed"}, {"target_outcome": "selected_but_final_unconfirmed"}],
                ["unattainable_on_grid"],
            ),
            {
                "selected_and_confirmed": 1,
                "selected_but_final_unconfirmed": 1,
                "unattainable_on_grid": 1,
            },
        )

    def test_schema_gate_and_put_readback_gate_strategy_ef_and_cutoff(self):
        initial = schema_definition("sweeping", 100, 0)
        expected = schema_definition("acorn", 250, 100_000)
        self.assertEqual(runner.schema_gate(expected, "acorn", 250, 100_000), [])
        bad = schema_definition("acorn", 250, 0)
        self.assertTrue(any("flatSearchCutoff" in item for item in runner.schema_gate(bad, "acorn", 250, 100_000)))
        with mock.patch.object(runner.baseline, "request_json", side_effect=[(initial, 0), ({}, 0), (expected, 0)]) as request:
            readback, _, retries = runner.put_hnsw_config("http://unused", "acorn", 250, 100_000, 1.0, 2)
        self.assertEqual(readback, expected)
        self.assertEqual(retries, 0)
        self.assertEqual(request.call_args_list[1].args[2]["vectorIndexConfig"]["flatSearchCutoff"], 100_000)

    def test_service_version_and_image_digest_gate_fail_closed(self):
        identity = runner.validate_service_identity(
            {"version": "1.38.0"}, "1.38.0", "sha256:immutable"
        )
        self.assertEqual(identity["service_image_digest"], "sha256:immutable")
        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            runner.validate_service_identity({"version": "1.38.1"}, "1.38.0", "sha256:x")
        with self.assertRaisesRegex(ValueError, "service-image-digest"):
            runner.validate_service_identity({"version": "1.38.0"}, "1.38.0", "  ")
        args = runner.build_parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "service-image-digest"):
            runner._validate_args(args)
        invalid_guard = runner.build_parser().parse_args([
            "--service-image-digest", "sha256:x", "--hnsw-dominance-guard", "0.99",
        ])
        with self.assertRaisesRegex(ValueError, "dominance guard"):
            runner._validate_args(invalid_guard)
        reporting_only_margin = runner.build_parser().parse_args([
            "--service-image-digest", "sha256:x", "--calibration-lcb-margin", "0.02",
        ])
        runner._validate_args(reporting_only_margin)
        self.assertEqual(runner.main(["--dry-run"]), 0)

    def test_filter_selection_is_ordered_unique_complete_and_run_spec_bound(self):
        filters = (filter_spec("f1", 40_000), filter_spec("f2", 50_000))
        selected = runner.select_filter_specs(filters, ["f2", "f1"])
        self.assertEqual([spec.name for spec in selected], ["f2", "f1"])
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.select_filter_specs(filters, ["f1", "f1"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            runner.select_filter_specs(filters, ["missing"])
        args = runner.build_parser().parse_args([
            "--service-image-digest", "sha256:test", "--flat-search-cutoffs", "0", "100000",
            "--filter-names", "f2", "f1", "--hnsw-dominance-guard", "1.2",
        ])
        specification = runner.run_specification(args, selected, {0: 7}, {"runner": "hash"})
        self.assertEqual(specification["filter_names"], ["f2", "f1"])
        self.assertEqual(specification["service_image_digest"], "sha256:test")
        self.assertEqual(specification["hnsw_flat_dominance"]["guard"], 1.2)
        self.assertEqual(
            specification["calibration"]["selection_rule"],
            runner.baseline.TARGET_SELECTION_RULE,
        )
        self.assertEqual(
            specification["calibration"]["selection_policy"],
            runner.CALIBRATION_SELECTION_POLICY,
        )
        self.assertEqual(specification["effective_cutoffs_by_filter"], {
            "f2": [0, 100_000], "f1": [0, 100_000],
        })

    def test_checkpoint_strictly_binds_cutoff_in_run_spec_blocks_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "CALIBRATION_QUERY_NOS", (0,)), mock.patch.object(runner, "CALIBRATION_REPEATS", 1):
            path = Path(tmp) / "checkpoint.json"
            specification = {"flat_search_cutoffs": [0, 100_000]}
            original_schema = schema_definition("sweeping", 100, 0)
            block = runner._block_record("calibration", "sweeping", "f", 100_000, 100)
            row = {
                "phase": "calibration", "configured_filter_strategy": "sweeping", "filter_name": "f",
                "flat_search_cutoff": 100_000, "ef": 100, "query_no": 0, "query_id": 7,
                "repeat": 0,
            }
            runner.write_checkpoint(
                path, specification, raw_rows=[row],
                calibration_summaries=[{"configured_filter_strategy": "sweeping", "filter_name": "f", "flat_search_cutoff": 100_000, "ef": 100}],
                final_results=[], state=checkpoint_state(original_schema, [block]),
            )
            restored = runner.load_checkpoint(path, specification, {0: 7})
            self.assertEqual(restored["state"]["completed_blocks"], [block])
            with self.assertRaisesRegex(RuntimeError, "run-spec/hash mismatch"):
                runner.load_checkpoint(path, {"flat_search_cutoffs": [0]}, {0: 7})
            row["flat_search_cutoff"] = 0
            runner.write_checkpoint(
                path, specification, raw_rows=[row],
                calibration_summaries=[{"configured_filter_strategy": "sweeping", "filter_name": "f", "flat_search_cutoff": 100_000, "ef": 100}],
                final_results=[], state=checkpoint_state(original_schema, [block]),
            )
            with self.assertRaisesRegex(RuntimeError, "rows are incomplete"):
                runner.load_checkpoint(path, specification, {0: 7})

            invalid_state = checkpoint_state(original_schema)
            invalid_state.pop("original_schema")
            runner.write_checkpoint(
                path, specification, raw_rows=[], calibration_summaries=[], final_results=[],
                state=invalid_state,
            )
            with self.assertRaisesRegex(RuntimeError, "original_schema"):
                runner.load_checkpoint(path, specification, {})

    def test_original_schema_is_snapshotted_before_put_and_resume_restores_it(self):
        spec = filter_spec()
        original = schema_definition("sweeping", 111, 7)
        leftover = schema_definition("acorn", 100, 100_000)
        vectors = [[0.0]] * 200
        query_ids = {0: 0}
        total = {"data": {"Aggregate": {runner.CLASS_NAME: [{"meta": {"count": runner.baseline.EXPECTED_ROWS}}]}}}
        filtered = {"data": {"Aggregate": {runner.CLASS_NAME: [{"meta": {"count": spec.expected_rows}}]}}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.csv"
            checkpoint = Path(tmp) / "checkpoint.json"
            argv = [
                "--out", str(out), "--checkpoint", str(checkpoint),
                "--service-image-digest", "sha256:immutable", "--filter-names", "f",
                "--flat-search-cutoffs", "0", "100000", "--ef-values", "100",
                "--targets", "0.9", "--warmup-queries", "0",
            ]
            args = runner.build_parser().parse_args(argv)
            common_patches = (
                mock.patch.object(runner.baseline, "load_filter_specs", return_value=(spec,)),
                mock.patch.object(runner.baseline, "read_fbin_memmap", return_value=(vectors, len(vectors), 1)),
                mock.patch.object(runner.baseline, "load_truth", return_value=({}, query_ids)),
                mock.patch.object(runner.baseline, "sha256_file", return_value="hash"),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
                    mock.patch.object(runner.baseline, "isolate_existing_outputs", return_value=None), \
                    mock.patch.object(runner.baseline, "request_json", side_effect=[(original, 0), ({"version": "1.38.0"}, 0)]), \
                    mock.patch.object(runner.baseline, "get_ready_nodes", return_value=({"nodes": []}, 0)), \
                    mock.patch.object(runner.baseline, "graphql", side_effect=[(total, 0), (filtered, 0)]), \
                    mock.patch.object(runner, "put_hnsw_config", side_effect=RuntimeError("schema mutation crash")), \
                    mock.patch.object(runner, "put_schema_definition", return_value=(original, 0)) as restore:
                with self.assertRaisesRegex(RuntimeError, "schema mutation crash"):
                    runner.run(args)
            first_snapshot = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(first_snapshot["state"]["original_schema"], original)
            self.assertEqual(first_snapshot["state"]["schema_records"], [
                {"phase": "original_schema_snapshot", "schema": original}
            ])
            self.assertEqual(restore.call_args.args[1], original)

            resume_args = runner.build_parser().parse_args([*argv, "--resume"])
            resume_patches = (
                mock.patch.object(runner.baseline, "load_filter_specs", return_value=(spec,)),
                mock.patch.object(runner.baseline, "read_fbin_memmap", return_value=(vectors, len(vectors), 1)),
                mock.patch.object(runner.baseline, "load_truth", return_value=({}, query_ids)),
                mock.patch.object(runner.baseline, "sha256_file", return_value="hash"),
            )
            with resume_patches[0], resume_patches[1], resume_patches[2], resume_patches[3], \
                    mock.patch.object(runner.baseline, "request_json", side_effect=[(leftover, 0), ({"version": "1.39.0"}, 0)]), \
                    mock.patch.object(runner.baseline, "get_ready_nodes", return_value=({"nodes": []}, 0)), \
                    mock.patch.object(runner, "put_hnsw_config") as mutate, \
                    mock.patch.object(runner, "put_schema_definition", return_value=(original, 0)) as resume_restore:
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    runner.run(resume_args)
            mutate.assert_not_called()
            self.assertEqual(resume_restore.call_args.args[1], original)
            resumed_snapshot = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(resumed_snapshot["state"]["original_schema"], original)
            self.assertIn(
                {"phase": "resume_live_schema", "schema": leftover},
                resumed_snapshot["state"]["schema_records"],
            )

    def test_keyboard_interrupt_restores_original_schema_and_is_re_raised(self):
        spec = filter_spec()
        original = schema_definition("sweeping", 111, 7)
        vectors = [[0.0]] * 200
        query_ids = {0: 0}
        total = {
            "data": {"Aggregate": {runner.CLASS_NAME: [
                {"meta": {"count": runner.baseline.EXPECTED_ROWS}}
            ]}}
        }
        filtered = {
            "data": {"Aggregate": {runner.CLASS_NAME: [
                {"meta": {"count": spec.expected_rows}}
            ]}}
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.csv"
            checkpoint = Path(tmp) / "checkpoint.json"
            args = runner.build_parser().parse_args([
                "--out", str(out), "--checkpoint", str(checkpoint),
                "--service-image-digest", "sha256:immutable", "--filter-names", "f",
                "--flat-search-cutoffs", "0", "100000", "--ef-values", "100",
                "--targets", "0.9", "--warmup-queries", "0",
            ])
            with mock.patch.object(runner.baseline, "load_filter_specs", return_value=(spec,)), \
                    mock.patch.object(runner.baseline, "read_fbin_memmap", return_value=(vectors, len(vectors), 1)), \
                    mock.patch.object(runner.baseline, "load_truth", return_value=({}, query_ids)), \
                    mock.patch.object(runner.baseline, "sha256_file", return_value="hash"), \
                    mock.patch.object(runner.baseline, "isolate_existing_outputs", return_value=None), \
                    mock.patch.object(runner.baseline, "request_json", side_effect=[(original, 0), ({"version": "1.38.0"}, 0)]), \
                    mock.patch.object(runner.baseline, "get_ready_nodes", return_value=({"nodes": []}, 0)), \
                    mock.patch.object(runner.baseline, "graphql", side_effect=[(total, 0), (filtered, 0)]), \
                    mock.patch.object(runner, "put_hnsw_config", side_effect=KeyboardInterrupt()), \
                    mock.patch.object(runner, "put_schema_definition", return_value=(original, 0)) as restore:
                with self.assertRaises(KeyboardInterrupt):
                    runner.run(args)
            self.assertEqual(restore.call_args.args[1], original)
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"]["original_schema"], original)
            self.assertEqual(payload["state"]["completed_blocks"], [])

        with self.assertRaises(SystemExit):
            runner.raise_after_schema_restore(SystemExit(7), None)

    def test_expected_route_is_declared_inference_not_observation(self):
        spec = runner.baseline.FILTERS[-1]
        record = runner.expected_route(spec, spec.expected_rows + 1)
        self.assertEqual(record["expected_route"], "flat")
        self.assertFalse(record["route_observed"])
        self.assertIn("allowList.Len() < flatSearchCutoff", record["inference"])


if __name__ == "__main__":
    unittest.main()
