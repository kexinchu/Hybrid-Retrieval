import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner import (
    Config,
    DENSE_12_EF_SEARCH,
    DEFAULT_MODES,
    DEFAULT_BFS_INDEX,
    DEFAULT_BFS_TABLE,
    DEFAULT_INSERTION_INDEX,
    DEFAULT_INSERTION_TABLE,
    DEFAULT_TRUTH_CSV,
    SQLENS_PROFILE_REQUIRED_FIELDS,
    SQLENS_TRAVERSAL_PROFILE_REQUIRED_FIELDS,
    TIE_AWARE_RECALL_CONTRACT,
    bootstrap_mean_ci,
    acquire_formal_data_guard,
    build_configs,
    calibrate_mode_filter,
    consolidate_final,
    configs_for_mode,
    database_fingerprint,
    explicit_candidate_validity_predicate,
    formal_completion_gate,
    final_eligible_rows,
    mode_calibration_grids,
    paired_speedup_ci,
    parse_targets,
    percentile,
    plan_evidence_path,
    require_plan_evidence,
    run_d123,
    run_d123_interleaved,
    run_final_interleaved,
    seeded_calibration_block,
    select_row,
    selected_rows,
    sha256_file,
    sqlens_runtime_provenance,
    truth_query_ids,
    validate_tie_aware_raw_row,
)


class TargetRecallRunnerTests(unittest.TestCase):
    @staticmethod
    def _plan_runtime_metadata():
        identity = {
            "expected_build_id": "sqlens-v11-test",
            "expected_vector_so_sha256": "a" * 64,
            "observed_build_id": "sqlens-v11-test",
            "observed_vector_so_sha256": "a" * 64,
            "exact_match": True,
        }
        return {
            "sqlens_runtime_identity_startup": identity,
            "sqlens_runtime_identity_final": dict(identity),
            "execution_lifecycle": {
                "warmup_complete": True,
                "d3_lifecycle_complete": True,
                "backend_cpu_provenance_complete": True,
                "runtime_sqlens_identity_complete": True,
            },
            "backend_cpu_evidence": [
                {
                    "backend_pid": 123,
                    "requested_cpu_list": "",
                    "observed_cpu_list": "48-63",
                    "exact_match": None,
                    "pinning_attempted_by_runner": False,
                }
            ],
            "runtime_sqlens_identity_evidence": [
                {"mode": "original", "backend_pid": 123, **identity}
            ],
        }

    def test_parse_targets_sorts_and_deduplicates(self):
        self.assertEqual(parse_targets("0.99, 0.90,0.99,1"), [0.90, 0.99, 1.0])

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_targets("0,0.95")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_targets("0.95,1.01")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_targets(" , ")

    def test_formal_data_guard_holds_share_lock_and_records_relation_identity(self):
        cursor = mock.Mock()
        cursor.fetchone.side_effect = [
            ("public.external_queries", 30, 40),
            ("public.items", 10, 20),
            (4321, "100:100:"),
            ("public.external_queries", 30, 40),
            (200,),
            (["query_key:bigint", "query_embedding:vector"],),
        ]
        connection = mock.Mock()
        connection.cursor.return_value = cursor
        args = argparse.Namespace(
            insertion_table="public.items",
            bfs_table="public.items",
            query_table="public.external_queries",
        )
        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.psycopg.connect",
                return_value=connection,
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.pg_config_from_env",
                return_value=argparse.Namespace(conninfo="postgresql://test"),
            ),
        ):
            observed_connection, evidence = acquire_formal_data_guard(args)
        self.assertIs(observed_connection, connection)
        self.assertEqual(evidence["lock_mode"], "SHARE")
        self.assertEqual(evidence["backend_pid"], 4321)
        self.assertEqual(evidence["relations"]["public.items"]["relfilenode"], 20)
        self.assertEqual(
            evidence["query_table"],
            {
                "name": "public.external_queries",
                "oid": 30,
                "relfilenode": 40,
                "row_count": 200,
                "columns": ["query_key:bigint", "query_embedding:vector"],
            },
        )
        self.assertIn("LOCK TABLE", str(cursor.execute.call_args_list[0].args[0]))
        connection.commit.assert_not_called()
        connection.close.assert_not_called()

    def test_database_fingerprint_binds_candidate_validity_contract(self):
        cursor = mock.Mock()
        cursor.fetchone.side_effect = [
            ("16.4", "0.8.2"),
            (10, 20, 100, 1000, True, True, None),
            (11, 21, 100, 2000, True, True, "embedding_valid"),
            (12, 22, 100, 2000, True, True, "embedding_valid"),
        ]
        connection = mock.MagicMock()
        connection.cursor.return_value = cursor
        connect = mock.MagicMock()
        connect.return_value.__enter__.return_value = connection
        args = argparse.Namespace(
            insertion_table="public.items",
            insertion_index="public.items_hnsw",
            bfs_table="public.items",
            bfs_index="public.items_hnsw_bfs",
            query_table="public.queries",
            candidate_validity_predicate="embedding_valid",
        )
        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.psycopg.connect",
                connect,
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.pg_config_from_env",
                return_value=argparse.Namespace(conninfo="postgresql://test"),
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.query_relation_provenance",
                return_value={"name": "public.queries", "row_count": 200},
            ),
        ):
            fingerprint = database_fingerprint(args, "sqlens-v11-test")

        self.assertEqual(
            fingerprint["candidate_validity_predicate"],
            "embedding_valid",
        )
        self.assertEqual(len(fingerprint["candidate_validity_predicate_sha256"]), 64)
        self.assertEqual(fingerprint["query_table"]["row_count"], 200)

    def test_q200_external_truth_requires_non_self_excluded_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "q200.csv"
            truth.write_text(
                "method,filter_name,query_no,query_id,filtered_rows,kth_distance_sq,tie_tolerance,self_excluded\n"
                "pre_filter_exact,yfcc_filter,200,9001,10,0.25,1e-12,false\n",
                encoding="utf-8",
            )

            self.assertEqual(truth_query_ids(truth, expected_self_excluded=False), {200: 9001})
            with self.assertRaisesRegex(RuntimeError, "expected query contract"):
                truth_query_ids(truth, expected_self_excluded=True)

    def test_truth_candidate_validity_contract_is_optional_but_fail_closed_when_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "q200.csv"
            truth.write_text(
                "method,filter_name,query_no,query_id,filtered_rows,kth_distance_sq,"
                "tie_tolerance,self_excluded,candidate_validity_predicate\n"
                "pre_filter_exact,f,200,9001,10,0.25,1e-12,false,embedding_valid\n",
                encoding="utf-8",
            )

            self.assertEqual(
                truth_query_ids(
                    truth,
                    expected_self_excluded=False,
                    expected_candidate_validity_predicate="embedding_valid",
                ),
                {200: 9001},
            )
            with self.assertRaisesRegex(RuntimeError, "candidate contract"):
                truth_query_ids(
                    truth,
                    expected_self_excluded=False,
                    expected_candidate_validity_predicate="embedding_valid AND public",
                )

            legacy = Path(tmp) / "legacy.csv"
            legacy.write_text(
                "method,filter_name,query_no,query_id,filtered_rows,kth_distance_sq,"
                "tie_tolerance,self_excluded\n"
                "pre_filter_exact,f,200,9001,10,0.25,1e-12,false\n",
                encoding="utf-8",
            )
            self.assertEqual(
                truth_query_ids(legacy, expected_self_excluded=False),
                {200: 9001},
            )
            with self.assertRaisesRegex(RuntimeError, "missing candidate_validity_predicate"):
                truth_query_ids(
                    legacy,
                    expected_self_excluded=False,
                    expected_candidate_validity_predicate="embedding_valid",
                )

    def test_explicit_candidate_validity_respects_cli_presence_marker(self):
        self.assertIsNone(
            explicit_candidate_validity_predicate(
                argparse.Namespace(
                    candidate_validity_predicate="embedding_valid",
                    candidate_validity_predicate_explicit=False,
                )
            )
        )
        self.assertEqual(
            explicit_candidate_validity_predicate(
                argparse.Namespace(
                    candidate_validity_predicate="",
                    candidate_validity_predicate_explicit=True,
                )
            ),
            "TRUE",
        )

    def test_percentile_uses_sorted_floor_index_and_empty_is_zero(self):
        self.assertEqual(percentile([], 0.95), 0.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.0), 1.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 1.0), 3.0)

    def test_bootstrap_mean_ci_is_reproducible_for_a_seed(self):
        values = [10.0, 20.0, 40.0, 80.0]
        first = bootstrap_mean_ci(values, samples=250, seed=1234)
        second = bootstrap_mean_ci(values, samples=250, seed=1234)

        self.assertEqual(first, second)
        self.assertEqual(bootstrap_mean_ci([], samples=250, seed=1234), (0.0, 0.0))
        self.assertEqual(bootstrap_mean_ci([7.0], samples=250, seed=1234), (7.0, 7.0))

    def test_select_row_meeting_target_chooses_lowest_latency(self):
        rows = [
            {"config": "slow", "ok": 2, "errors": 0, "recall_mean": 0.99, "latency_mean_ms": 30.0},
            {"config": "fast", "ok": 2, "errors": 0, "recall_mean": 0.95, "latency_mean_ms": 10.0},
            {"config": "failed", "ok": 0, "errors": 0, "recall_mean": 1.0, "latency_mean_ms": 1.0},
            {"config": "errored", "ok": 2, "errors": 1, "recall_mean": 1.0, "latency_mean_ms": 1.0},
        ]

        selected, met = select_row(rows, target=0.95)

        self.assertTrue(met)
        self.assertEqual(selected["config"], "fast")

    def test_select_row_does_not_return_an_unattained_fallback(self):
        rows = [
            {"config": "best-recall-slow", "ok": 2, "errors": 0, "recall_mean": 0.90, "latency_mean_ms": 50.0},
            {"config": "best-recall-fast", "ok": 2, "errors": 0, "recall_mean": 0.90, "latency_mean_ms": 20.0},
            {"config": "faster-lower-recall", "ok": 2, "errors": 0, "recall_mean": 0.85, "latency_mean_ms": 1.0},
        ]

        selected, met = select_row(rows, target=0.95)

        self.assertFalse(met)
        self.assertIsNone(selected)

    def test_select_row_requires_recall_lower_bound(self):
        rows = [
            {
                "config": "uncertain-fast",
                "ok": 200,
                "errors": 0,
                "rows_complete": True,
                "recall_mean": 0.97,
                "recall_lcb95": 0.93,
                "latency_mean_ms": 10.0,
            },
            {
                "config": "confirmed",
                "ok": 200,
                "errors": 0,
                "rows_complete": True,
                "recall_mean": 0.98,
                "recall_lcb95": 0.96,
                "latency_mean_ms": 20.0,
            },
        ]

        selected, met = select_row(rows, target=0.95)

        self.assertTrue(met)
        self.assertEqual(selected["config"], "confirmed")

    def test_final_eligible_rows_requires_a_jointly_attainable_method(self):
        rows = [
            {"target_recall": 0.95, "filter_name": "f", "mode": "original", "selection_status": "selected"},
            {"target_recall": 0.95, "filter_name": "f", "mode": "d1", "selection_status": "unattainable_on_grid", "config": "fallback"},
            {"target_recall": 0.99, "filter_name": "f", "mode": "original", "selection_status": "unattainable_on_grid"},
            {"target_recall": 0.99, "filter_name": "f", "mode": "d1", "selection_status": "selected"},
        ]

        eligible = final_eligible_rows(rows)

        self.assertEqual(eligible, [])

    @staticmethod
    def _calibration_args(resume=False):
        return argparse.Namespace(
            run_spec_hash="a" * 64,
            tag="test",
            bootstrap_samples=10,
            bootstrap_seed=1,
            calibration_queries=1,
            calibration_repeats=1,
            calibration_query_offset=0,
            guidance_filter_strategy="guided_collect",
            schedule_seed=20260718,
            resume=resume,
        )

    @staticmethod
    def _calibration_summary(recall_lcb95, latency=10.0):
        return [{
            "queries": 1,
            "ok": 1,
            "errors": 0,
            "rows_complete": True,
            "recall_mean": recall_lcb95,
            "recall_lcb95": recall_lcb95,
            "latency_mean_ms": latency,
        }]

    def test_calibration_stops_after_a_complete_ef_group_reaches_max_target(self):
        configs = [
            Config(100, 1000, 8.0, "strict_order", 10),
            Config(100, 2000, 8.0, "strict_order", 10),
            Config(200, 1000, 8.0, "strict_order", 10),
        ]

        def summarize(path, *_args):
            return self._calibration_summary(0.99 if "ef100_" in path.name else 1.0)

        plan_entry = {"path": "plan.json", "sha256": "abc", "checks": [{"passed": True}]}
        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_d123",
                return_value=1.0,
            ) as run_mock,
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.summarize_raw",
                side_effect=summarize,
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.plan_evidence_manifest_entry",
                return_value=plan_entry,
            ),
        ):
            rows, evidence = calibrate_mode_filter(
                "f", "design1_bloom", configs, self._calibration_args(), [0.95, 0.99]
            )

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual([row["ef_search"] for row in rows], [100, 100])
        self.assertEqual(evidence["configs_planned"], 3)
        self.assertEqual(evidence["configs_executed"], 2)
        self.assertTrue(evidence["stopped_early"])
        self.assertEqual(evidence["max_ef_evaluated"], 100)
        self.assertFalse(evidence["grid_exhausted"])

    def test_unattained_target_is_marked_only_after_the_full_grid_completes(self):
        calibration = []
        for ef_search in (100, 200):
            calibration.append(
                {
                    "filter_name": "f",
                    "mode": "design1_bloom",
                    "config": f"ef{ef_search}",
                    "ef_search": ef_search,
                    "guided_collect_target": 1,
                    "max_scan_tuples": 1000,
                    "scan_mem_multiplier": 8.0,
                    "iterative_scan": "strict_order",
                    "ok": 1,
                    "errors": 0,
                    "rows_complete": True,
                    "recall_mean": 0.90,
                    "recall_lcb95": 0.90,
                    "latency_mean_ms": 10.0,
                    "configs_planned": 2,
                    "configs_executed": 2,
                    "max_ef_evaluated": 200,
                    "grid_exhausted": True,
                    "stopped_early": False,
                }
            )

        selected = selected_rows(calibration, ["f"], ["design1_bloom"], [0.95])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["selection_status"], "unattainable_on_grid")
        self.assertEqual(selected[0]["config"], "")
        self.assertTrue(selected[0]["grid_exhausted"])

    def test_resume_reuses_completed_block_and_stops_before_higher_ef(self):
        configs = [
            Config(100, 1000, 8.0, "strict_order", 10),
            Config(200, 1000, 8.0, "strict_order", 10),
        ]
        plan_entry = {"path": "plan.json", "sha256": "abc", "checks": [{"passed": True}]}
        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.reusable_summary",
                return_value=self._calibration_summary(0.99),
            ) as reusable_mock,
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.require_plan_evidence",
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_d123",
            ) as run_mock,
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.plan_evidence_manifest_entry",
                return_value=plan_entry,
            ),
        ):
            rows, evidence = calibrate_mode_filter(
                "f", "design1_bloom", configs, self._calibration_args(resume=True), [0.95, 0.99]
            )

        self.assertEqual(reusable_mock.call_count, 1)
        run_mock.assert_not_called()
        self.assertEqual([row["ef_search"] for row in rows], [100])
        self.assertEqual(evidence["configs_reused"], 1)
        self.assertTrue(evidence["stopped_early"])

    def test_stock_iterative_scan_families_stop_independently(self):
        configs = [
            Config(100, 5000000, 32.0, "off", 100),
            Config(200, 5000000, 32.0, "off", 200),
        ]

        def summarize(path, *_args):
            if "_off_" in path.name:
                return self._calibration_summary(0.99)
            return self._calibration_summary(0.80 if "ef100_" in path.name else 0.99)

        args = self._calibration_args()
        args.stock_iterative_scan_values = "off,strict_order"
        plan_entry = {"path": "plan.json", "sha256": "abc", "checks": [{"passed": True}]}
        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_d123",
                return_value=1.0,
            ) as run_mock,
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.summarize_raw",
                side_effect=summarize,
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.plan_evidence_manifest_entry",
                return_value=plan_entry,
            ),
        ):
            rows, evidence = calibrate_mode_filter("f", "original", configs, args, [0.99])

        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(
            [(row["iterative_scan"], row["ef_search"]) for row in rows],
            [("off", 100), ("strict_order", 100), ("strict_order", 200)],
        )
        self.assertTrue(evidence["families"]["off"]["stopped_early"])
        self.assertFalse(evidence["families"]["strict_order"]["stopped_early"])
        self.assertTrue(evidence["families"]["strict_order"]["grid_exhausted"])

    def test_calibration_family_error_prevents_early_stop_and_selection(self):
        configs = [
            Config(100, 1000, 32.0, "off", 100),
            Config(100, 2000, 32.0, "off", 100),
            Config(200, 1000, 32.0, "off", 200),
        ]

        def summarize(path, *_args):
            if "ef100_" in path.name and "max2000_" in path.name:
                return [{
                    "queries": 1,
                    "ok": 0,
                    "errors": 1,
                    "rows_complete": False,
                    "recall_mean": 0.0,
                    "recall_lcb95": 0.0,
                    "latency_mean_ms": 0.0,
                }]
            return self._calibration_summary(0.99)

        args = self._calibration_args()
        args.stock_iterative_scan_values = "strict_order"
        plan_entry = {"path": "plan.json", "sha256": "abc", "checks": [{"passed": True}]}
        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_d123",
                return_value=1.0,
            ) as run_mock,
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.summarize_raw",
                side_effect=summarize,
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.plan_evidence_manifest_entry",
                return_value=plan_entry,
            ),
        ):
            rows, evidence = calibrate_mode_filter("f", "original", configs, args, [0.95])

        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(evidence["families"]["strict_order"]["errors"], 1)
        self.assertFalse(evidence["families"]["strict_order"]["stopped_early"])
        selected = selected_rows(rows, ["f"], ["original"], [0.95])
        self.assertEqual(selected[0]["selection_status"], "incomplete_or_failed")

    def test_seeded_calibration_block_is_reproducible_and_randomized(self):
        configs = [Config(100, value, 32.0, "off", 100) for value in range(1000, 9000, 1000)]

        first = seeded_calibration_block(configs, 20260718, "f", "original", "off", 0)
        second = seeded_calibration_block(configs, 20260718, "f", "original", "off", 0)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(configs))
        self.assertNotEqual(first, configs)

    def test_manifest_mode_grids_record_stock_and_sqlens_actual_configs(self):
        configs = [Config(100, 5000000, 32.0, "off", 100)]

        grids = mode_calibration_grids(
            configs,
            ["original", "design1_bloom"],
            "off,strict_order",
        )

        self.assertEqual(
            {row["iterative_scan"] for row in grids["original"]},
            {"off", "strict_order"},
        )
        self.assertEqual([row["iterative_scan"] for row in grids["design1_bloom"]], ["off"])

    def test_formal_completion_gate_distinguishes_matrix_measurement_and_comparison(self):
        filters = [f"dataset_filter_{number}" for number in range(14)]
        modes = DEFAULT_MODES
        targets = [0.90, 0.95, 0.99]
        selected = [
            {"filter_name": filter_name, "mode": mode, "target_recall": target}
            for filter_name in filters
            for target in targets
            for mode in modes
        ]
        final = [
            {
                **row,
                "final_status": "complete",
                "rows_complete": True,
                "errors": 0,
                "target_confirmed_in_calibration": True,
                "target_confirmed_in_final": True,
                "matched_recall_comparison_valid": True,
            }
            for row in selected
        ]

        complete = formal_completion_gate(filters, modes, targets, selected, final, False)
        self.assertTrue(complete["matrix_complete"])
        self.assertTrue(complete["measurement_complete"])
        self.assertTrue(complete["comparison_valid"])
        self.assertEqual(complete["expected_cells"], 168)
        self.assertEqual(complete["status"], "complete")

        duplicate = formal_completion_gate(
            filters,
            modes,
            targets,
            selected + [dict(selected[0])],
            final,
            False,
        )
        self.assertFalse(duplicate["matrix_complete"])
        self.assertEqual(duplicate["status"], "incomplete")

        missing_final = formal_completion_gate(filters, modes, targets, selected, final[:-1], False)
        self.assertTrue(missing_final["matrix_complete"])
        self.assertFalse(missing_final["measurement_complete"])
        self.assertFalse(missing_final["comparison_valid"])

        final[-1]["matched_recall_comparison_valid"] = False
        invalid_comparison = formal_completion_gate(filters, modes, targets, selected, final, False)
        self.assertTrue(invalid_comparison["measurement_complete"])
        self.assertFalse(invalid_comparison["comparison_valid"])
        self.assertEqual(invalid_comparison["status"], "incomplete")

    def test_parent_passes_exact_build_and_vector_sha_to_child(self):
        args = argparse.Namespace(
            progress_queries=0,
            statement_timeout_ms=1000,
            guidance_filter_strategy="guided_collect",
            guidance_selectivity_max_pct=100.0,
            guidance_max_atoms=64,
            d2_page_access="off",
            d2_index_page_access="off",
            d1_cache_mb=16,
            d3_cache_mb=16,
            preferred_index_guc="hnsw.preferred_index",
            d2_graph_proof={"required": False},
            filters_csv=None,
            truth_csv=None,
            insertion_table=None,
            insertion_index=None,
            bfs_table=None,
            bfs_index=None,
            require_preferred_index_guc=False,
            warmup_all_queries=False,
            force_hnsw=True,
            backend_cpu_list="48-51",
            candidate_validity_predicate="embedding_valid",
            sqlens_runtime_provenance={
                "loaded_vector_sqlens_build_id": "sqlens-v11-exact",
                "loaded_vector_so_sha256": "a" * 64,
            },
        )
        with mock.patch(
            "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_command",
            return_value=1.0,
        ) as command_mock:
            run_d123(
                Path("out.csv"),
                "f",
                "original",
                0,
                1,
                1,
                Config(100, 1000, 8.0, "off", 100),
                args,
                None,
            )

        cmd = command_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--expected-sqlens-build-id") + 1], "sqlens-v11-exact")
        self.assertEqual(cmd[cmd.index("--expected-vector-so-sha256") + 1], "a" * 64)
        self.assertEqual(cmd[cmd.index("--backend-cpu-list") + 1], "48-51")
        self.assertEqual(
            cmd[cmd.index("--candidate-validity-predicate") + 1],
            "embedding_valid",
        )

    def test_interleaved_parent_forwards_candidate_validity_predicate(self):
        args = argparse.Namespace(
            final_queries=2,
            final_query_offset=10,
            final_repeats=1,
            schedule_seed=7,
            progress_queries=0,
            statement_timeout_ms=1000,
            guidance_filter_strategy="traversal_guided",
            guidance_selectivity_max_pct=100.0,
            guidance_max_atoms=64,
            d2_page_access="off",
            d2_index_page_access="off",
            d1_cache_mb=16,
            d3_cache_mb=16,
            preferred_index_guc="hnsw.preferred_index",
            d2_graph_proof={"required": False},
            filters_csv=None,
            truth_csv=None,
            insertion_table=None,
            insertion_index=None,
            bfs_table=None,
            bfs_index=None,
            query_table="public.queries",
            query_id_column="query_key",
            query_vector_column="query_embedding",
            candidate_validity_predicate="embedding_valid",
            expected_truth_self_excluded=False,
            backend_cpu_list=None,
            require_preferred_index_guc=False,
            warmup_all_queries=False,
            force_hnsw=True,
            sqlens_runtime_provenance={
                "loaded_vector_sqlens_build_id": "sqlens-v11-exact",
                "loaded_vector_so_sha256": "a" * 64,
            },
        )
        modes = ["original", "design1_bloom"]
        configs = {
            mode: Config(100, 1000, 8.0, "off", 100) for mode in modes
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_command",
            return_value=1.0,
        ) as command_mock:
            run_d123_interleaved(
                Path(tmp) / "out.csv",
                "f",
                modes,
                configs,
                args,
                None,
            )

        cmd = command_mock.call_args.args[0]
        self.assertEqual(
            cmd[cmd.index("--candidate-validity-predicate") + 1],
            "embedding_valid",
        )
        self.assertIn("--no-expected-truth-self-excluded", cmd)

    def test_sqlens_runtime_provenance_records_loaded_contract_and_fails_closed(self):
        profile = {
            "profile_semantics_version": 6,
            "graph_elements_visited": 1,
            "raw_index_tids_returned": 2,
            "hnsw_am_callback_ms": 0.1,
            "executor_residual_ms": 0.2,
        }
        profile.update({field: 0 for field in SQLENS_TRAVERSAL_PROFILE_REQUIRED_FIELDS})
        profile.update(
            {
                "final_path": "stock",
                "planner_proof_attempted": False,
                "planner_proof_succeeded": False,
            }
        )
        cursor = mock.MagicMock()
        binary_sha256 = "a" * 64
        cursor.fetchone.side_effect = [
            ("sqlens-v11-test", "/usr/lib/postgresql/16/lib/vector.so", binary_sha256),
            (json.dumps(profile),),
        ]
        connection = mock.MagicMock()
        connection.cursor.return_value = cursor
        connect = mock.MagicMock()
        connect.return_value.__enter__.return_value = connection

        with mock.patch(
            "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.psycopg.connect",
            connect,
        ):
            provenance = sqlens_runtime_provenance()

        self.assertEqual(provenance["loaded_vector_sqlens_build_id"], "sqlens-v11-test")
        self.assertEqual(provenance["loaded_vector_so_sha256"], binary_sha256)
        self.assertEqual(provenance["required_build_prefix"], "sqlens-v11-")
        self.assertEqual(provenance["minimum_profile_semantics_version"], 6.0)
        self.assertEqual(provenance["profile_semantics_version"], 6)
        self.assertEqual(provenance["required_profile_fields"], {
            key: profile[key]
            for key in SQLENS_PROFILE_REQUIRED_FIELDS + SQLENS_TRAVERSAL_PROFILE_REQUIRED_FIELDS
        })

        cursor.fetchone.side_effect = [
            ("sqlens-v11-test", "/usr/lib/postgresql/16/lib/vector.so", "not-a-sha"),
            (json.dumps(profile),),
        ]
        with mock.patch(
            "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.psycopg.connect",
            connect,
        ):
            with self.assertRaisesRegex(RuntimeError, "vector.so path/SHA256"):
                sqlens_runtime_provenance()

        profile.pop("executor_residual_ms")
        profile["profile_semantics_version"] = 5
        cursor.fetchone.side_effect = [
            ("sqlens-v11-test", "/usr/lib/postgresql/16/lib/vector.so", binary_sha256),
            (json.dumps(profile),),
        ]
        with mock.patch(
            "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.psycopg.connect",
            connect,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing_fields"):
                sqlens_runtime_provenance()

    def test_configs_for_mode_deduplicates_stock_but_not_sqlens(self):
        configs = [
            Config(100, 1000, 8.0, "strict_order", 100),
            Config(100, 1000, 8.0, "strict_order", 200),
            Config(200, 1000, 8.0, "strict_order", 100),
        ]

        stock = configs_for_mode(configs, "original")
        sqlens = configs_for_mode(configs, "design1_bloom")

        self.assertEqual(len(stock), 2)
        self.assertEqual([config.guided_collect_target for config in stock], [100, 100])
        self.assertEqual(sqlens, configs)

    def test_configs_for_mode_independently_tunes_stock_iterative_scan(self):
        configs = [Config(100, 5000000, 32.0, "off", 100)]

        stock = configs_for_mode(configs, "original", "off,strict_order")
        sqlens = configs_for_mode(configs, "design1_bloom", "off,strict_order")

        self.assertEqual(
            {(config.ef_search, config.iterative_scan) for config in stock},
            {(100, "off"), (100, "strict_order")},
        )
        self.assertEqual([config.iterative_scan for config in sqlens], ["off"])
        with self.assertRaisesRegex(ValueError, "stock iterative scan values"):
            configs_for_mode(configs, "original", "off,invalid")

    def test_safe_guided_config_grid_ignores_guided_collect_target(self):
        args = argparse.Namespace(
            ef_search_values="100,200",
            guided_collect_target_values="10,20,ef",
            max_scan_tuples_values="1000",
            scan_mem_multiplier_values="8",
            iterative_scan_values="strict_order",
            guidance_filter_strategy="safe_guided",
        )

        configs = build_configs(args)

        self.assertEqual(len(configs), 2)
        self.assertEqual({config.guided_collect_target for config in configs}, {1})

    def test_traversal_guided_grid_requires_off_and_targets_at_least_ef(self):
        args = argparse.Namespace(
            ef_search_values="100,200",
            guided_collect_target_values="40,ef",
            max_scan_tuples_values="1000",
            scan_mem_multiplier_values="8",
            iterative_scan_values="off",
            guidance_filter_strategy="traversal_guided",
        )

        configs = build_configs(args)

        self.assertEqual(
            {(config.ef_search, config.guided_collect_target) for config in configs},
            {(100, 100), (200, 200)},
        )
        args.iterative_scan_values = "strict_order"
        with self.assertRaisesRegex(SystemExit, "iterative-scan-values=off"):
            build_configs(args)

    def test_default_ef_grid_matches_dense_12(self):
        self.assertEqual(
            [int(value) for value in DENSE_12_EF_SEARCH.split(",")],
            [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 8500, 10000],
        )
        self.assertEqual(DEFAULT_TRUTH_CSV.name, "amazon_selectivity14_exact_truth_q200_formal.csv")
        self.assertEqual(DEFAULT_INSERTION_TABLE, DEFAULT_BFS_TABLE)
        self.assertNotEqual(DEFAULT_INSERTION_INDEX, DEFAULT_BFS_INDEX)
        self.assertIn("bfs_clone", DEFAULT_BFS_INDEX)

    def test_raw_recall_is_recomputed_from_tie_aware_distances(self):
        row = {
            "sqlens_build_id": "sqlens-v11-test",
            "vector_so_sha256": "a" * 64,
            "backend_pid": "123",
            "backend_cpu_observed": "48-63",
            "backend_cpu_requested": "",
            "backend_cpu_pinning_attempted_by_runner": "false",
            "recall_contract": TIE_AWARE_RECALL_CONTRACT,
            "truth_self_excluded": "true",
            "truth_filtered_rows": "12",
            "truth_kth_distance_sq": "1.0",
            "truth_tie_tolerance": "0.0",
            "truth_strict_closer_count": "8",
            "truth_boundary_tied": "true",
            "result_distances": "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.0]",
            "returned": "10",
            "k": "10",
            "recall": "1.0",
            "error": "",
        }

        self.assertEqual(validate_tie_aware_raw_row(row), 1.0)
        row["recall"] = "0.8"
        with self.assertRaises(ValueError):
            validate_tie_aware_raw_row(row)

    def test_formal_traversal_raw_contract_rejects_self_and_stock_fallback(self):
        row = {
            "sqlens_build_id": "sqlens-v11-test",
            "vector_so_sha256": "a" * 64,
            "backend_pid": "123",
            "backend_cpu_observed": "48-63",
            "backend_cpu_requested": "",
            "backend_cpu_pinning_attempted_by_runner": "false",
            "guidance_filter_strategy": "traversal_guided",
            "mode": "design1_bloom",
            "self_exclusion_contract": "limit_k_plus_1_client_remove_query_id",
            "scan_limit": "3",
            "raw_returned_before_self_exclusion": "3",
            "returned": "2",
            "query_id": "99",
            "ids": "10,11",
            "guidance_enabled": "true",
            "guidance_scan_verified": "true",
            "final_path": "guided",
            "planner_proof_succeeded": "true",
            "stock_bypass_requests": "0",
            "fallback_requests": "0",
            "traversal_guidance_scope": "candidate_admission_and_validation",
            "graph_expansion_pruned": "false",
            "distance_computations_pruned": "false",
            "pre_distance_membership_checks": "0",
            "distance_computations_avoided": "0",
            "neighbor_expansion_guidance_checks": "5",
            "traversal_guided_admissions": "3",
            "traversal_guided_suppressions": "2",
            "traversal_heap_tids_suppressed": "2",
            "traversal_estimated_skip_rate_valid": "true",
            "traversal_estimated_skip_rate": "0.5",
            "recall_contract": TIE_AWARE_RECALL_CONTRACT,
            "truth_self_excluded": "true",
            "truth_filtered_rows": "2",
            "truth_kth_distance_sq": "1.0",
            "truth_tie_tolerance": "0.0",
            "result_distances": "[0.1,0.2]",
            "k": "2",
            "recall": "1.0",
            "error": "",
        }
        self.assertEqual(validate_tie_aware_raw_row(row), 1.0)

        row["fallback_requests"] = "1"
        with self.assertRaisesRegex(ValueError, "fresh-stock fallback"):
            validate_tie_aware_raw_row(row)
        row["fallback_requests"] = "0"
        row["ids"] = "99,10"
        with self.assertRaisesRegex(ValueError, "still contains the query row"):
            validate_tie_aware_raw_row(row)

    def test_plan_evidence_fails_closed_on_output_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            raw.write_text("value\n1\n", encoding="utf-8")
            evidence = {
                **self._plan_runtime_metadata(),
                "status": "complete",
                "output_rows": 1,
                "output_sha256": sha256_file(raw),
                "checks": [{"passed": True, "expected_index_access_method": "hnsw"}],
            }
            plan_evidence_path(raw).write_text(json.dumps(evidence), encoding="utf-8")

            self.assertEqual(require_plan_evidence(raw)["status"], "complete")
            raw.write_text("value\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                require_plan_evidence(raw)

    def test_plan_evidence_binds_explicit_candidate_validity_to_parent_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            raw.write_text("value\n1\n", encoding="utf-8")
            predicate = "embedding_valid"
            predicate_hash = hashlib.sha256(predicate.encode("utf-8")).hexdigest()
            evidence = {
                **self._plan_runtime_metadata(),
                "status": "complete",
                "output_rows": 1,
                "output_sha256": sha256_file(raw),
                "query_contract": {
                    "candidate_validity_predicate": predicate,
                    "candidate_validity_predicate_sha256": predicate_hash,
                },
                "checks": [
                    {
                        "passed": True,
                        "expected_index_access_method": "hnsw",
                        "expected_index": predicate,
                        "expected_index_oid": 1,
                        "expected_index_predicate": predicate,
                        "expected_index_predicate_sha256": predicate_hash,
                        "expected_index_is_partial": True,
                        "catalog_index_oid": 1,
                        "catalog_index_predicate": predicate,
                        "catalog_index_predicate_sha256": predicate_hash,
                        "catalog_index_is_partial": True,
                        "catalog_index_predicate_matches": True,
                        "candidate_validity_predicate": predicate,
                        "candidate_validity_predicate_sha256": predicate_hash,
                    }
                ],
            }
            evidence_path = plan_evidence_path(raw)
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

            self.assertEqual(
                require_plan_evidence(raw, predicate)["status"],
                "complete",
            )
            parent_fingerprint = {
                "relations": {
                    predicate: {"oid": 1, "indpred": predicate, "is_partial": True}
                }
            }
            self.assertEqual(
                require_plan_evidence(raw, predicate, parent_fingerprint)["status"],
                "complete",
            )
            parent_fingerprint["relations"][predicate]["indpred"] = None
            with self.assertRaisesRegex(RuntimeError, "parent database fingerprint"):
                require_plan_evidence(raw, predicate, parent_fingerprint)
            with self.assertRaisesRegex(RuntimeError, "query contract mismatch"):
                require_plan_evidence(raw, "embedding_valid AND public")

            evidence["checks"][0]["candidate_validity_predicate_sha256"] = "0" * 64
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "plan evidence mismatch"):
                require_plan_evidence(raw, predicate)

    def test_plan_evidence_requires_valid_same_graph_proof_for_d2(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            raw.write_text("value\n1\n", encoding="utf-8")
            evidence = {
                **self._plan_runtime_metadata(),
                "status": "complete",
                "output_rows": 1,
                "output_sha256": sha256_file(raw),
                "checks": [
                    {
                        "passed": True,
                        "mode": "design1_bloom_bfs_layout",
                        "expected_index_access_method": "hnsw",
                    }
                ],
                "d2_graph_proof": {
                    "source_index": "public.source_idx",
                    "clone_index": "public.clone_idx",
                    "relations": {
                        "source": {"name": "public.source_idx", "oid": 1, "relfilenode": 11, "heap_oid": 9},
                        "clone": {"name": "public.clone_idx", "oid": 2, "relfilenode": 12, "heap_oid": 9},
                    },
                    "comparison": {
                        "format": "sqlens-hnsw-compare-v2",
                        "same_heap": True,
                        "logical_equal": True,
                        "entry_equal": True,
                        "definition_equal": True,
                        "tuple_coverage_equal": True,
                        "physical_equal": False,
                        "left_definition_digest": "sha256:" + "1" * 64,
                        "right_definition_digest": "sha256:" + "1" * 64,
                        "left_tuple_coverage_digest": "sha256:" + "2" * 64,
                        "right_tuple_coverage_digest": "sha256:" + "2" * 64,
                        "left_logical_digest": "sha256:" + "3" * 64,
                        "right_logical_digest": "sha256:" + "3" * 64,
                        "left_physical_digest": "sha256:" + "4" * 64,
                        "right_physical_digest": "sha256:" + "5" * 64,
                    },
                },
            }
            evidence["d2_graph_proof_final"] = json.loads(
                json.dumps(evidence["d2_graph_proof"])
            )
            proof_path = plan_evidence_path(raw)
            proof_path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertEqual(require_plan_evidence(raw)["status"], "complete")

            evidence["d2_graph_proof"]["comparison"]["physical_equal"] = True
            proof_path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "physical_equal"):
                require_plan_evidence(raw)

    def test_consolidate_final_calculates_speedup_vs_stock(self):
        selected = [
            {
                "target_recall": 0.95,
                "target_met_in_calibration": True,
                "target_confirmed_in_calibration": True,
                "filter_name": "popular_ge1000",
                "mode": "original",
                "config": "stock-config",
                "ef_search": 100,
                "guided_collect_target": 100,
                "max_scan_tuples": 1000,
                "scan_mem_multiplier": 8.0,
                "iterative_scan": "strict_order",
            },
            {
                "target_recall": 0.95,
                "target_met_in_calibration": True,
                "target_confirmed_in_calibration": True,
                "filter_name": "popular_ge1000",
                "mode": "design1_bloom",
                "config": "sqlens-config",
                "ef_search": 100,
                "guided_collect_target": 100,
                "max_scan_tuples": 1000,
                "scan_mem_multiplier": 8.0,
                "iterative_scan": "strict_order",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "paired.csv"
            with raw.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["filter_name", "mode", "query_no", "repeat", "end_to_end_ms", "recall", "error"],
                )
                writer.writeheader()
                writer.writerow({"filter_name": "popular_ge1000", "mode": "original", "query_no": 0, "repeat": 0, "end_to_end_ms": 100, "recall": 0.96, "error": ""})
                writer.writerow({"filter_name": "popular_ge1000", "mode": "design1_bloom", "query_no": 0, "repeat": 0, "end_to_end_ms": 40, "recall": 0.94, "error": ""})
            shared = {
                "rows_complete": True,
                "errors": 0,
                "expected_queries": 1,
                "expected_repeats": 1,
                "recall_contract": TIE_AWARE_RECALL_CONTRACT,
                "truth_self_excluded": True,
                "plan_gate_passed": True,
                "final_raw": str(raw),
                "final_execution_order": "interleaved",
                "final_schedule_id": "target0p95_popular_ge1000",
            }
            final_results = {
                (0.95, "popular_ge1000", "original", "stock-config"): {
                    **shared,
                    "recall_mean": 0.96,
                    "recall_lcb95": 0.95,
                    "latency_mean_ms": 100.0,
                },
                (0.95, "popular_ge1000", "design1_bloom", "sqlens-config"): {
                    **shared,
                    "recall_mean": 0.94,
                    "recall_lcb95": 0.93,
                    "latency_mean_ms": 40.0,
                },
            }

            rows = consolidate_final(selected, final_results)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["speedup_vs_stock"], 1.0)
        self.assertIsNone(rows[1]["speedup_vs_stock"])
        self.assertTrue(rows[0]["target_met_in_final"])
        self.assertFalse(rows[1]["target_met_in_final"])
        self.assertTrue(rows[0]["matched_recall_comparison_valid"])
        self.assertFalse(rows[1]["matched_recall_comparison_valid"])

    def test_shared_stock_config_is_kept_inside_each_target_schedule(self):
        def selected_row(target, mode, config, ef_search):
            return {
                "target_recall": target,
                "filter_name": "f",
                "mode": mode,
                "config": config,
                "ef_search": ef_search,
                "guided_collect_target": 1,
                "max_scan_tuples": 5000000,
                "scan_mem_multiplier": 32.0,
                "iterative_scan": "strict_order",
            }

        selected = [
            selected_row(0.90, "original", "shared-stock", 500),
            selected_row(0.90, "design1_bloom", "method-low", 750),
            selected_row(0.95, "original", "shared-stock", 500),
            selected_row(0.95, "design1_bloom", "method-high", 1500),
        ]
        args = argparse.Namespace(
            modes=["original", "design1_bloom"],
            final_queries=100,
            final_repeats=5,
            run_spec_hash="a" * 64,
            tag="test",
            resume=False,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )
        summary = [
            {"mode": "original", "queries": 100},
            {"mode": "design1_bloom", "queries": 100},
        ]
        plan_entry = {"path": "plan.json", "sha256": "abc", "checks": [{"passed": True}]}

        with (
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.run_d123_interleaved",
                return_value=1.0,
            ) as run_mock,
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.summarize_raw",
                return_value=summary,
            ),
            mock.patch(
                "experiments.hybrid_vector_db.scripts.pgvector_target_recall_selectivity_runner.plan_evidence_manifest_entry",
                return_value=plan_entry,
            ),
        ):
            results = run_final_interleaved(selected, args)

        self.assertEqual(run_mock.call_count, 2)
        low_stock = results[(0.90, "f", "original", "ef500_target1_max5000000_mem32p0_strict_order")]
        low_method = results[(0.90, "f", "design1_bloom", "ef750_target1_max5000000_mem32p0_strict_order")]
        high_stock = results[(0.95, "f", "original", "ef500_target1_max5000000_mem32p0_strict_order")]
        high_method = results[(0.95, "f", "design1_bloom", "ef1500_target1_max5000000_mem32p0_strict_order")]
        self.assertEqual(low_stock["final_raw"], low_method["final_raw"])
        self.assertEqual(high_stock["final_raw"], high_method["final_raw"])
        self.assertNotEqual(low_stock["final_raw"], high_stock["final_raw"])
        self.assertEqual(low_stock["final_schedule_id"], low_method["final_schedule_id"])
        self.assertEqual(high_stock["final_schedule_id"], high_method["final_schedule_id"])

    def test_paired_speedup_ci_matches_queries_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["filter_name", "mode", "query_no", "repeat", "end_to_end_ms", "recall", "error"],
                )
                writer.writeheader()
                for query_no, stock_ms, method_ms in [(0, 20.0, 10.0), (1, 40.0, 20.0)]:
                    for repeat in range(2):
                        writer.writerow(
                            {
                                "filter_name": "f",
                                "mode": "original",
                                "query_no": query_no,
                                "repeat": repeat,
                                "end_to_end_ms": stock_ms,
                                "recall": 1.0,
                                "error": "",
                            }
                        )
                        writer.writerow(
                            {
                                "filter_name": "f",
                                "mode": "design1_bloom",
                                "query_no": query_no,
                                "repeat": repeat,
                                "end_to_end_ms": method_ms,
                                "recall": 1.0,
                                "error": "",
                            }
                        )

            first = paired_speedup_ci(path, "original", path, "design1_bloom", "f", 100, 57)
            second = paired_speedup_ci(path, "original", path, "design1_bloom", "f", 100, 57)

            self.assertEqual(first, second)
            self.assertEqual(first, (2, 2.0, 2.0))

    def test_paired_speedup_ci_rejects_an_incomplete_pair_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "incomplete.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["filter_name", "mode", "query_no", "repeat", "end_to_end_ms", "recall", "error"],
                )
                writer.writeheader()
                writer.writerow({"filter_name": "f", "mode": "original", "query_no": 0, "repeat": 0, "end_to_end_ms": 20, "recall": 1, "error": ""})
                writer.writerow({"filter_name": "f", "mode": "original", "query_no": 0, "repeat": 1, "end_to_end_ms": 20, "recall": 1, "error": ""})
                writer.writerow({"filter_name": "f", "mode": "design1_bloom", "query_no": 0, "repeat": 0, "end_to_end_ms": 10, "recall": 1, "error": ""})

            result = paired_speedup_ci(path, "original", path, "design1_bloom", "f", 100, 57)

            self.assertEqual(result, (0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
