import argparse
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pgvector_design1_design2_design3_selectivity_benchmark as benchmark  # noqa: E402


class InterleavedSelectivityBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _d2_proof(checked_at: str = "2026-07-18T00:00:00+00:00") -> dict[str, object]:
        return {
            "checked_at": checked_at,
            "source_index": "public.source_idx",
            "clone_index": "public.clone_idx",
            "relations": {
                "source": {
                    "name": "public.source_idx",
                    "oid": 101,
                    "relfilenode": 1001,
                    "heap_oid": 99,
                },
                "clone": {
                    "name": "public.clone_idx",
                    "oid": 102,
                    "relfilenode": 1002,
                    "heap_oid": 99,
                },
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
        }

    @staticmethod
    def _sqlens_profile() -> dict[str, object]:
        profile = {
            "profile_semantics_version": 4,
            "graph_elements_visited": 11,
            "raw_index_tids_returned": 7,
            "hnsw_am_callback_ms": 1.25,
            "executor_residual_ms": 0.75,
        }
        profile.update({field: 0 for field in benchmark.SQLENS_TRAVERSAL_PROFILE_FIELDS})
        profile.update(
            {
                "final_path": "stock",
                "planner_proof_attempted": False,
                "planner_proof_succeeded": False,
                "planner_proof_bypass_reason": "scan_not_started",
                "stock_bypass_reason": "none",
                "fallback_reason": "none",
            }
        )
        return profile

    @staticmethod
    def _successful_traversal_profile() -> dict[str, object]:
        return {
            "guidance_checks": 5,
            "distance_compute_count": 8,
            "traversal_expanded_nodes": 4,
            "traversal_guidance_checks": 5,
            "final_path": "guided",
            "planner_proof_attempted": True,
            "planner_proof_succeeded": True,
            "planner_proof_bypass_reason": "none",
            "pre_distance_membership_checks": 5,
            "pre_distance_membership_matches": 3,
            "pre_distance_membership_misses": 2,
            "distance_computations_avoided_attempted": 2,
            "distance_computations_avoided": 2,
            "neighbor_expansion_guidance_checks": 5,
            "neighbor_expansion_guidance_matches": 3,
            "neighbor_expansion_guidance_misses": 2,
            "guided_expanded_nodes": 4,
            "guided_phase_distance_computations": 8,
            "stock_phase_expanded_nodes": 0,
            "stock_phase_distance_computations": 0,
            "stock_bypass_requests": 0,
            "stock_bypass_reason": "none",
            "fallback_requests": 0,
            "fallback_reason": "none",
            "fallback_stock_expanded_nodes": 0,
            "fallback_stock_distance_computations": 0,
            "traversal_estimated_skip_rate_valid": True,
            "traversal_estimated_skip_rate": 0.5,
        }

    def test_sqlens_v11_gate_accepts_required_build_and_profile(self):
        cursor = mock.Mock()
        cursor.fetchone.side_effect = [
            ("sqlens-v11-amazon-build",),
            (json.dumps(self._sqlens_profile()),),
        ]

        build_id, profile = benchmark.require_sqlens_provenance(cursor)

        self.assertEqual(build_id, "sqlens-v11-amazon-build")
        self.assertEqual(profile["profile_semantics_version"], 4)
        self.assertEqual(
            [call.args[0] for call in cursor.execute.call_args_list],
            [
                "SELECT vector_sqlens_build_id()",
                "SELECT vector_hnsw_last_scan_profile()",
            ],
        )

    def test_sqlens_v11_gate_rejects_old_build_before_profile_or_wrapper_ddl(self):
        cursor = mock.Mock()
        cursor.fetchone.return_value = ("sqlens-v8-old-build",)

        with self.assertRaisesRegex(RuntimeError, "expected the 'sqlens-v11-' prefix"):
            benchmark.ensure_functions(cursor)

        self.assertEqual(cursor.execute.call_count, 1)
        self.assertEqual(cursor.execute.call_args.args[0], "SELECT vector_sqlens_build_id()")

    def test_sqlens_v11_gate_rejects_missing_function_actionably(self):
        cursor = mock.Mock()
        cursor.execute.side_effect = Exception("function vector_sqlens_build_id() does not exist")

        with self.assertRaisesRegex(RuntimeError, r"vector_sqlens_build_id\(\) is unavailable"):
            benchmark.require_sqlens_provenance(cursor)

    def test_sqlens_v11_gate_rejects_missing_profile_function_actionably(self):
        cursor = mock.Mock()
        cursor.fetchone.return_value = ("sqlens-v11-amazon-build",)
        cursor.execute.side_effect = [
            None,
            Exception("function vector_hnsw_last_scan_profile() does not exist"),
        ]

        with self.assertRaisesRegex(RuntimeError, r"vector_hnsw_last_scan_profile\(\) is unavailable"):
            benchmark.require_sqlens_provenance(cursor)

    def test_sqlens_v11_gate_rejects_missing_profile_field(self):
        cursor = mock.Mock()
        profile = self._sqlens_profile()
        del profile["executor_residual_ms"]
        cursor.fetchone.side_effect = [
            ("sqlens-v11-amazon-build",),
            (json.dumps(profile),),
        ]

        with self.assertRaisesRegex(RuntimeError, "executor_residual_ms"):
            benchmark.require_sqlens_provenance(cursor)

    def test_child_exact_identity_gate_uses_server_side_vector_sha(self):
        cursor = mock.Mock()
        cursor.fetchone.return_value = (
            "sqlens-v11-exact",
            "/usr/lib/postgresql/16/lib/vector.so",
            "a" * 64,
        )

        identity = benchmark.require_exact_sqlens_identity(
            cursor,
            "sqlens-v11-exact",
            "a" * 64,
        )

        self.assertTrue(identity["exact_match"])
        self.assertEqual(identity["observed_vector_so_sha256"], "a" * 64)
        self.assertIn("pg_read_binary_file", cursor.execute.call_args.args[0])

        cursor.fetchone.return_value = (
            "sqlens-v11-exact",
            "/usr/lib/postgresql/16/lib/vector.so",
            "b" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "vector.so SHA256 mismatch"):
            benchmark.require_exact_sqlens_identity(
                cursor,
                "sqlens-v11-exact",
                "a" * 64,
            )

    def test_backend_cpu_provenance_is_db_side_and_fails_closed_on_mismatch(self):
        cursor = mock.Mock()
        cursor.fetchone.return_value = (
            4321,
            "Name:\tpostgres\nCpus_allowed_list:\t48-51\n",
        )

        provenance = benchmark.backend_cpu_provenance(cursor, "48,49-51")

        self.assertEqual(provenance["backend_pid"], 4321)
        self.assertEqual(provenance["requested_cpu_list"], "48-51")
        self.assertEqual(provenance["observed_cpu_list"], "48-51")
        self.assertTrue(provenance["exact_match"])
        self.assertFalse(provenance["pinning_attempted_by_runner"])
        self.assertEqual(
            cursor.execute.call_args.args[0],
            "SELECT pg_backend_pid(), pg_read_file('/proc/self/status')",
        )

        cursor.fetchone.return_value = (4321, "Cpus_allowed_list:\t48-63\n")
        mismatch = benchmark.backend_cpu_provenance(cursor, "48-51")
        with self.assertRaisesRegex(RuntimeError, "Docker namespace PID"):
            benchmark.enforce_backend_cpu_provenance(mismatch)

    def test_guidance_scan_contract_requires_statement_effect_and_traversal(self):
        self.assertFalse(benchmark.guidance_scan_contract_satisfied({}, "safe_guided"))
        self.assertTrue(
            benchmark.guidance_scan_contract_satisfied(
                {"guidance_checks": 3}, "safe_guided"
            )
        )
        self.assertTrue(
            benchmark.guidance_scan_contract_satisfied(
                self._successful_traversal_profile(), "traversal_guided"
            )
        )
        for field, value in (
            ("final_path", "stock_bypass"),
            ("stock_bypass_requests", 1),
            ("fallback_requests", 1),
            ("planner_proof_succeeded", False),
            ("traversal_estimated_skip_rate_valid", False),
        ):
            with self.subTest(field=field):
                profile = self._successful_traversal_profile()
                profile[field] = value
                self.assertFalse(
                    benchmark.guidance_scan_contract_satisfied(profile, "traversal_guided")
                )
        self.assertFalse(
            benchmark.guidance_scan_contract_satisfied(
                {"guidance_checks": 3, "traversal_guidance_checks": 0},
                "guided_collect",
            )
        )
        self.assertTrue(
            benchmark.guidance_scan_contract_satisfied(
                {"guidance_checks": 3, "traversal_guidance_checks": 9},
                "guided_collect",
            )
        )

    def test_mode_configs_json_validates_and_resolves_overrides(self):
        configs = benchmark.parse_mode_configs_json(
            '{"original":{"ef_search":200},'
            '"design1_bloom":{"max_scan_tuples":5000,"scan_mem_multiplier":4,'
            '"iterative_scan":"relaxed_order","guided_collect_target":300}}'
        )
        args = argparse.Namespace(
            ef_search=100,
            max_scan_tuples=1000,
            scan_mem_multiplier=8.0,
            iterative_scan="strict_order",
            guided_collect_target=100,
            mode_configs_json=configs,
        )

        self.assertEqual(benchmark.effective_mode_config(args, "original")["ef_search"], 200)
        self.assertEqual(
            benchmark.effective_mode_config(args, "design1_bloom"),
            {
                "ef_search": 100,
                "max_scan_tuples": 5000,
                "scan_mem_multiplier": 4.0,
                "iterative_scan": "relaxed_order",
                "guided_collect_target": 300,
            },
        )

    def test_mode_configs_json_rejects_unknown_or_invalid_values(self):
        invalid = [
            "[]",
            '{"not_a_mode":{"ef_search":100}}',
            '{"original":{"not_a_setting":100}}',
            '{"original":{"ef_search":true}}',
            '{"original":{"iterative_scan":"sometimes"}}',
        ]

        for value in invalid:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                benchmark.parse_mode_configs_json(value)

    def test_mode_configs_json_accepts_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.json"
            path.write_text('{"design1_bloom":{"ef_search":321}}', encoding="utf-8")

            configs = benchmark.parse_mode_configs_json(str(path))

        self.assertEqual(configs, {"design1_bloom": {"ef_search": 321}})

    def test_truth_loader_requires_self_excluded_tie_aware_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "truth.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "method",
                        "filter_name",
                        "query_no",
                        "query_id",
                        "filtered_rows",
                        "kth_distance_sq",
                        "tie_tolerance",
                        "strict_closer_count",
                        "boundary_tied",
                        "self_excluded",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method": "pre_filter_exact",
                        "filter_name": "f",
                        "query_no": 2,
                        "query_id": 99,
                        "filtered_rows": 12,
                        "kth_distance_sq": 0.5625,
                        "tie_tolerance": 1e-12,
                        "strict_closer_count": 8,
                        "boundary_tied": "true",
                        "self_excluded": "true",
                    }
                )

            truth, query_by_no = benchmark.load_tie_aware_truth(path)

            legacy = Path(tmpdir) / "legacy.csv"
            legacy.write_text(
                "method,filter_name,query_no,query_id,filtered_rows,kth_distance,self_excluded\n"
                "pre_filter_exact,f,2,99,12,0.75,true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing tie-aware fields"):
                benchmark.load_tie_aware_truth(legacy)

        self.assertEqual(query_by_no, {2: 99})
        self.assertEqual(truth[("f", 2)].kth_distance_sq, 0.5625)
        self.assertEqual(truth[("f", 2)].tie_tolerance, 1e-12)
        self.assertTrue(truth[("f", 2)].boundary_tied)
        self.assertTrue(truth[("f", 2)].self_excluded)

    def test_shuffled_modes_is_seeded_and_balanced(self):
        modes = benchmark.MODES[:3]
        first_rng = random.Random(20260718)
        second_rng = random.Random(20260718)

        first = [benchmark.shuffled_modes(modes, first_rng) for _ in range(8)]
        second = [benchmark.shuffled_modes(modes, second_rng) for _ in range(8)]

        self.assertEqual(first, second)
        self.assertTrue(all(sorted(group) == sorted(modes) for group in first))
        self.assertTrue(any(group != modes for group in first))

    def test_balanced_mode_order_rotates_every_mode_through_each_position(self):
        modes = benchmark.MODES[:4]
        orders = [benchmark.balanced_mode_order(modes, block_no, 20260718) for block_no in range(4)]

        self.assertTrue(all(sorted(order) == sorted(modes) for order in orders))
        for position in range(len(modes)):
            self.assertEqual({order[position] for order in orders}, set(modes))

    def test_d3_adaptive_activation_probes_before_admitting_guidance(self):
        cursor = SimpleNamespace(
            execute=mock.Mock(),
            fetchone=mock.Mock(
                side_effect=[
                    (0,),
                    (json.dumps({"active": False, "adaptive_state": "probing"}),),
                    (0,),
                    (json.dumps({"active": False, "adaptive_state": "probing"}),),
                    (2,),
                    (json.dumps({"active": True, "adaptive_state": "page"}),),
                ]
            ),
        )
        args = argparse.Namespace(
            bfs_table="bfs_table",
            bfs_index="bfs_index",
            insertion_table="insertion_table",
            insertion_index="insertion_index",
            filter_selectivity_by_name={"filter_a": 1.0},
            filter_atoms={"filter_a": ["a = 1"]},
            guidance_selectivity_max_pct=10.0,
            guidance_max_atoms=64,
            guidance_filter_strategy="guided_collect",
            reset_cache_per_query=False,
        )

        first = benchmark.activate(cursor, args, "design1_bloom_bfs_layout_d3", "filter_a")
        second = benchmark.activate(cursor, args, "design1_bloom_bfs_layout_d3", "filter_a")
        third = benchmark.activate(cursor, args, "design1_bloom_bfs_layout_d3", "filter_a")

        self.assertEqual((first["activation_atom_count"], first["guidance_route"]), (0, "d3_probe"))
        self.assertFalse(first["guidance_enabled"])
        self.assertEqual((second["activation_atom_count"], second["guidance_route"]), (0, "d3_probe"))
        self.assertFalse(second["guidance_enabled"])
        self.assertEqual((third["activation_atom_count"], third["adaptive_state"]), (2, "page"))
        self.assertTrue(third["guidance_enabled"])
        activation_calls = [
            call for call in cursor.execute.call_args_list
            if "vector_hnsw_guidance_activate" in call.args[0]
        ]
        self.assertEqual(len(activation_calls), 3)
        self.assertTrue(all(call.args[1][-1] == "adaptive" for call in activation_calls))

    def test_d3_measurements_preserve_admission_and_prove_warm_reuse(self):
        cursor = SimpleNamespace(execute=mock.Mock(), fetchone=mock.Mock(return_value=("{}",)))
        runtime = benchmark.ModeRuntime(
            mode="design1_bloom_bfs_layout_d3",
            config={
                "ef_search": 100,
                "max_scan_tuples": 1000,
                "scan_mem_multiplier": 8.0,
                "iterative_scan": "strict_order",
                "guided_collect_target": 100,
            },
            cache_mb=1024,
            conn=SimpleNamespace(),
            cur=cursor,
            backend_cpu_provenance={
                "backend_pid": 4321,
                "requested_cpu_list": "48-51",
                "observed_cpu_list": "48-51",
                "exact_match": True,
            },
            sqlens_runtime_identity={
                "observed_build_id": "sqlens-v11-exact",
                "observed_vector_so_sha256": "a" * 64,
                "exact_match": True,
            },
        )
        args = argparse.Namespace(
            bfs_table="bfs_table",
            bfs_index="bfs_index",
            d2_page_access="off",
            d2_index_page_access="off",
            filter_selectivity_by_name={"filter_a": 1.0},
            filter_atoms={"filter_a": ["a = 1"]},
            guidance_selectivity_max_pct=10.0,
            guidance_max_atoms=64,
            guidance_filter_strategy="guided_collect",
            k=1,
            execution_order="interleaved",
            schedule_seed=20260718,
            warmup_all_queries=True,
            sqlens_runtime_identity={
                "observed_build_id": "sqlens-v11-exact",
                "observed_vector_so_sha256": "a" * 64,
            },
        )
        activation = {
            "table": "bfs_table",
            "index": "bfs_index",
            "activation_atom_count": 2,
            "guidance_enabled": True,
            "guidance_route": "enabled",
        }
        guidance_profiles = [
            {"active": False, "adaptive_state": "probing", "adaptive_admissions": 0, "fragment_cache_hits": 0},
            {"active": True, "adaptive_state": "page", "adaptive_admissions": 1, "fragment_cache_hits": 0},
            {"active": True, "adaptive_state": "page", "adaptive_admissions": 1, "fragment_cache_hits": 0},
            {"active": True, "adaptive_state": "page", "adaptive_admissions": 1, "fragment_cache_hits": 1},
        ]
        cache_profiles = [
            {"resident_entries": 0, "resident_bytes": 0, "composed_guide_hits": 0},
            {"resident_entries": 1, "resident_bytes": 4096, "composed_guide_hits": 0},
            {"resident_entries": 1, "resident_bytes": 4096, "composed_guide_hits": 0},
            {"resident_entries": 1, "resident_bytes": 4096, "composed_guide_hits": 1},
        ]
        scan_profile = {"guidance_checks": 1, "traversal_guidance_checks": 1}

        with (
            mock.patch.object(benchmark, "activate", return_value=activation) as activate_mock,
            mock.patch.object(
                benchmark,
                "run_query",
                return_value=([10], [0.25], {"sqlens_raw_returned_before_self_exclusion": 1}),
            ) as run_query_mock,
            mock.patch.object(
                benchmark,
                "read_guidance_profile",
                side_effect=guidance_profiles,
            ),
            mock.patch.object(
                benchmark,
                "read_cache_profile",
                side_effect=cache_profiles,
            ),
            mock.patch.object(benchmark, "read_scan_profile", return_value=scan_profile),
            mock.patch.object(
                benchmark.time,
                "perf_counter",
                side_effect=[10.0, 10.1, 10.3, 20.0, 20.1, 20.3],
            ),
        ):
            truth = benchmark.TruthEntry(
                query_id=101,
                filtered_rows=1,
                kth_distance_sq=0.0625,
                tie_tolerance=0.0,
                self_excluded=True,
            )
            first = benchmark.run_measured_query(
                args, runtime, "filter_a", 1.0, "a = 1", 1, 101, 0, {("filter_a", 1): truth}, 1
            )
            truth = benchmark.TruthEntry(
                query_id=102,
                filtered_rows=1,
                kth_distance_sq=0.0625,
                tie_tolerance=0.0,
                self_excluded=True,
            )
            warm = benchmark.run_measured_query(
                args, runtime, "filter_a", 1.0, "a = 1", 2, 102, 0, {("filter_a", 2): truth}, 2
            )

        self.assertEqual(activate_mock.call_count, 2)
        self.assertFalse(first["d3_active_guidance_reused"])
        self.assertEqual(first["d3_phase"], "admission")
        self.assertTrue(first["d3_admitted_after"])
        self.assertEqual(first["d3_adaptive_admissions_before"], 0)
        self.assertEqual(first["d3_adaptive_admissions_after"], 1)
        self.assertEqual(warm["d3_phase"], "warm")
        self.assertTrue(warm["d3_admitted_before"])
        self.assertTrue(warm["d3_active_guidance_reused"])
        self.assertEqual(warm["d3_fragment_cache_hits_delta"], 1)
        self.assertEqual(warm["d3_composed_guide_hits_delta"], 1)
        self.assertTrue(first["guidance_binding_verified"])
        self.assertEqual(warm["ef_search"], 100)
        self.assertEqual(warm["result_distances"], "[0.25]")
        self.assertAlmostEqual(first["activation_ms"], 100.0)
        self.assertAlmostEqual(first["query_latency_ms"], 200.0)
        self.assertAlmostEqual(first["end_to_end_ms"], 300.0)
        self.assertEqual(first["sqlens_build_id"], "sqlens-v11-exact")
        self.assertEqual(first["vector_so_sha256"], "a" * 64)
        bindings = [call.args[5] for call in run_query_mock.call_args_list]
        self.assertEqual(bindings, [("bfs_index", ["a = 1"], "adaptive")] * 2)
        self.assertTrue(all(call.kwargs["reset_profile"] is False for call in run_query_mock.call_args_list))
        self.assertTrue(all(call.kwargs["read_profile"] is False for call in run_query_mock.call_args_list))

    def test_warmup_exception_is_evidenced_and_fails_closed(self):
        args = argparse.Namespace(warmup_evidence=[], k=1, guidance_filter_strategy="guided_collect")
        runtime = benchmark.ModeRuntime(
            mode="original",
            config={},
            cache_mb=1,
            conn=SimpleNamespace(),
            cur=SimpleNamespace(execute=mock.Mock()),
        )

        with (
            mock.patch.object(benchmark, "read_guidance_profile", return_value={"active": False}),
            mock.patch.object(benchmark, "read_cache_profile", return_value={"resident_entries": 0}),
            mock.patch.object(benchmark, "activate", side_effect=RuntimeError("warmup failed")),
            mock.patch.object(benchmark, "recover_runtime"),
            self.assertRaisesRegex(RuntimeError, "warmup failed"),
        ):
            benchmark.run_warmup(args, runtime, "filter_a", "a = 1", 101)

        self.assertEqual(len(args.warmup_evidence), 1)
        self.assertEqual(args.warmup_evidence[0]["status"], "failed")
        self.assertIn("warmup failed", args.warmup_evidence[0]["error"])

    def test_d3_mode_never_consumes_admission_in_unmeasured_warmup(self):
        mode = "design1_bloom_bfs_layout_d3"
        runtime = SimpleNamespace(mode=mode)
        args = argparse.Namespace(
            modes=[mode],
            warmup_all_queries=True,
            warmup_queries=99,
            repeats=1,
            progress_queries=0,
        )

        with (
            mock.patch.object(benchmark, "open_mode_runtime", return_value=runtime),
            mock.patch.object(benchmark, "close_mode_runtime"),
            mock.patch.object(benchmark, "run_warmup") as warmup_mock,
            mock.patch.object(
                benchmark,
                "run_measured_query",
                return_value={"mode": mode, "filter_name": "filter_a", "error": ""},
            ) as measured_mock,
        ):
            benchmark.run_mode(
                args,
                mode,
                [("filter_a", 1.0, "a = 1")],
                [1],
                {1: 101},
                truth={},
            )

        warmup_mock.assert_not_called()
        measured_mock.assert_called_once()

    def test_lifecycle_gate_checks_warmup_count_and_all_d3_phases(self):
        args = argparse.Namespace(
            modes=["original", "design1_bloom_bfs_layout_d3"],
            warmup_all_queries=False,
            warmup_queries=1,
            warmup_evidence=[{"status": "complete"}],
            d3_phase_evidence=[
                {"filter_name": "filter_a", "d3_phase": "cold"},
                {"filter_name": "filter_a", "d3_phase": "admission"},
                {"filter_name": "filter_a", "d3_phase": "warm"},
            ],
            repeats=1,
            backend_cpu_list="48-51",
            backend_cpu_evidence=[
                {
                    "mode": "original",
                    "backend_pid": 100,
                    "requested_cpu_list": "48-51",
                    "observed_cpu_list": "48-51",
                    "exact_match": True,
                    "pinning_attempted_by_runner": False,
                },
                {
                    "mode": "design1_bloom_bfs_layout_d3",
                    "backend_pid": 101,
                    "requested_cpu_list": "48-51",
                    "observed_cpu_list": "48-51",
                    "exact_match": True,
                    "pinning_attempted_by_runner": False,
                },
            ],
            expected_sqlens_build_id="sqlens-v11-exact",
            expected_vector_so_sha256="a" * 64,
            runtime_sqlens_identity_evidence=[
                {
                    "mode": "original",
                    "exact_match": True,
                    "expected_build_id": "sqlens-v11-exact",
                    "expected_vector_so_sha256": "a" * 64,
                },
                {
                    "mode": "design1_bloom_bfs_layout_d3",
                    "exact_match": True,
                    "expected_build_id": "sqlens-v11-exact",
                    "expected_vector_so_sha256": "a" * 64,
                },
            ],
        )
        filters = [("filter_a", 1.0, "a = 1")]

        evidence = benchmark.validate_execution_lifecycle(args, filters, [1, 2, 3])

        self.assertTrue(evidence["warmup_complete"])
        self.assertTrue(evidence["d3_lifecycle_complete"])
        self.assertEqual(
            evidence["d3_phase_counts"]["filter_a"],
            {"cold": 1, "admission": 1, "warm": 1},
        )

        args.warmup_evidence = []
        with self.assertRaisesRegex(RuntimeError, "warmup evidence"):
            benchmark.validate_execution_lifecycle(args, filters, [1, 2, 3])

    def test_open_d3_runtime_has_no_prewarm_bloom_and_resets_after_plan_gate(self):
        cursor = SimpleNamespace(execute=mock.Mock(), close=mock.Mock())
        connection = SimpleNamespace(cursor=mock.Mock(return_value=cursor), close=mock.Mock())
        args = argparse.Namespace(
            d3_cache_mb=1024,
            d1_cache_mb=512,
            insertion_table="insertion_table",
            insertion_index="insertion_idx",
            bfs_table="bfs_table",
            bfs_index="bfs_idx",
            plan_query_id=101,
            require_preferred_index_guc=False,
            expected_sqlens_build_id="sqlens-v11-exact",
            expected_vector_so_sha256="a" * 64,
        )
        config = {
            "ef_search": 100,
            "max_scan_tuples": 1000,
            "scan_mem_multiplier": 8.0,
            "iterative_scan": "strict_order",
            "guided_collect_target": 100,
        }

        with (
            mock.patch.object(benchmark.psycopg, "connect", return_value=connection),
            mock.patch.object(
                benchmark,
                "backend_cpu_provenance",
                return_value={
                    "backend_pid": 4321,
                    "requested_cpu_list": "",
                    "observed_cpu_list": "48-63",
                    "exact_match": None,
                    "pinning_attempted_by_runner": False,
                },
            ),
            mock.patch.object(
                benchmark,
                "require_exact_sqlens_identity",
                return_value={
                    "expected_build_id": "sqlens-v11-exact",
                    "expected_vector_so_sha256": "a" * 64,
                    "observed_build_id": "sqlens-v11-exact",
                    "observed_vector_so_sha256": "a" * 64,
                    "exact_match": True,
                },
            ),
            mock.patch.object(benchmark, "ensure_functions"),
            mock.patch.object(benchmark, "ensure_tracking"),
            mock.patch.object(benchmark, "effective_mode_config", return_value=config),
            mock.patch.object(benchmark, "configure") as configure_mock,
            mock.patch.object(benchmark, "set_preferred_index_if_supported", return_value=None),
            mock.patch.object(benchmark, "gate_runtime_plans") as gate_mock,
        ):
            runtime = benchmark.open_mode_runtime(
                args,
                "design1_bloom_bfs_layout_d3",
                [("filter_a", 1.0, "a = 1")],
            )

        self.assertEqual(configure_mock.call_count, 2)
        gate_mock.assert_called_once()
        self.assertTrue(runtime.planner_proof_verified)
        self.assertEqual(
            [call.args[0] for call in cursor.execute.call_args_list],
            [
                "SELECT vector_hnsw_metadata_cache_reset()",
                "SELECT vector_hnsw_metadata_cache_reset()",
            ],
        )
        self.assertFalse(
            any("vector_hnsw_guidance_activate" in call.args[0] for call in cursor.execute.call_args_list)
        )

    def test_run_query_returns_distances_and_self_excludes_query_id(self):
        cursor = mock.Mock()
        cursor.fetchall.return_value = [(10, 0.125), (11, 0.5)]
        cursor.fetchone.return_value = ('{"visited_tuples":2}',)

        ids, distances, profile = benchmark.run_query(cursor, "items", "rating = 5", 99, 2)

        self.assertEqual(ids, [10, 11])
        self.assertEqual(distances, [0.125, 0.5])
        self.assertEqual(
            profile,
            {
                "visited_tuples": 2,
                "sqlens_raw_returned_before_self_exclusion": 2,
            },
        )
        query_sql, query_params = cursor.execute.call_args_list[1].args
        self.assertIn("WHERE (rating = 5) AND id <> %s", query_sql)
        self.assertEqual(query_params, (99, 99))

    def test_formal_traversal_query_has_no_residual_self_qual_and_client_excludes(self):
        sql = benchmark.search_query_sql(
            "items",
            "rating = 5",
            2,
            bind_guidance=True,
            client_self_exclusion=True,
        )
        self.assertNotIn("id <>", sql)
        self.assertIn("LIMIT 3", sql)

        cursor = mock.Mock()
        cursor.fetchall.return_value = [
            (99, 0.0),
            (10, 0.125),
            (11, 0.5),
        ]
        cursor.fetchone.return_value = ('{"valid":true}',)

        ids, distances, profile = benchmark.run_query(
            cursor,
            "items",
            "rating = 5",
            99,
            2,
            client_self_exclusion=True,
        )

        self.assertEqual(ids, [10, 11])
        self.assertEqual(distances, [0.125, 0.5])
        self.assertLessEqual(len(ids), 2)
        self.assertEqual(profile["sqlens_raw_returned_before_self_exclusion"], 3)
        query_sql, query_params = cursor.execute.call_args_list[1].args
        self.assertNotIn("id <>", query_sql)
        self.assertEqual(query_params, (99,))

    def test_external_query_sql_reads_the_query_table_without_candidate_self_exclusion(self):
        sql = benchmark.search_query_sql(
            "public.candidates",
            "rating = 5",
            2,
            query_table="public.queries",
            query_id_column="query_key",
            query_vector_column="query_embedding",
            self_exclusion=False,
        )

        self.assertIn("FROM public.queries AS q", sql)
        self.assertIn('q."query_key" = %s', sql)
        self.assertIn('q."query_embedding"', sql)
        self.assertNotIn("id <>", sql)
        self.assertIn("LIMIT 2", sql)

        cursor = mock.Mock()
        cursor.fetchall.return_value = [(99, 0.125), (10, 0.5)]
        cursor.fetchone.return_value = ('{"visited_tuples":2}',)
        ids, _, _ = benchmark.run_query(
            cursor,
            "public.candidates",
            "rating = 5",
            99,
            2,
            query_table="public.queries",
            query_id_column="query_key",
            query_vector_column="query_embedding",
            self_exclusion=False,
        )

        self.assertEqual(ids, [99, 10])
        _, query_params = cursor.execute.call_args_list[1].args
        self.assertEqual(query_params, (99,))

    def test_d2_graph_proof_gate_requires_same_logical_graph_but_new_layout(self):
        valid = self._d2_proof()
        proof = benchmark.validate_d2_graph_proof(
            valid, "public.source_idx", "public.clone_idx"
        )
        self.assertFalse(proof["comparison"]["physical_equal"])
        self.assertEqual(len(proof["stable_fingerprint_sha256"]), 64)

        for field, value in (
            ("same_heap", False),
            ("logical_equal", False),
            ("entry_equal", False),
            ("tuple_coverage_equal", False),
            ("physical_equal", True),
        ):
            with self.subTest(field=field):
                invalid = json.loads(json.dumps(valid))
                invalid["comparison"][field] = value
                with self.assertRaises(benchmark.D2GraphProofGateError):
                    benchmark.validate_d2_graph_proof(
                        invalid, "public.source_idx", "public.clone_idx"
                    )

    def test_d2_stable_proof_ignores_checked_at_but_binds_relation_identity(self):
        first = benchmark.validate_d2_graph_proof(
            self._d2_proof("2026-07-18T00:00:00+00:00"),
            "public.source_idx",
            "public.clone_idx",
        )
        second = benchmark.validate_d2_graph_proof(
            self._d2_proof("2026-07-19T00:00:00+00:00"),
            "public.source_idx",
            "public.clone_idx",
        )

        self.assertEqual(
            benchmark.stable_d2_graph_proof(first),
            benchmark.stable_d2_graph_proof(second),
        )
        changed = self._d2_proof()
        changed["relations"]["clone"]["relfilenode"] = 9999
        changed_proof = benchmark.validate_d2_graph_proof(
            changed,
            "public.source_idx",
            "public.clone_idx",
        )
        self.assertNotEqual(
            first["stable_fingerprint_sha256"],
            changed_proof["stable_fingerprint_sha256"],
        )

    def test_child_d2_gate_live_revalidates_delegated_oid_and_relfilenode(self):
        delegated = benchmark.validate_d2_graph_proof(
            self._d2_proof(),
            "public.source_idx",
            "public.clone_idx",
        )
        connection = mock.Mock()
        connection.cursor.return_value = mock.Mock()
        args = argparse.Namespace(
            insertion_index="public.source_idx",
            bfs_index="public.clone_idx",
        )

        with (
            mock.patch.object(benchmark.psycopg, "connect", return_value=connection),
            mock.patch.object(benchmark, "ensure_functions"),
            mock.patch.object(
                benchmark,
                "require_d2_relation_identity",
                return_value=delegated["relations"],
            ) as identity,
            mock.patch.object(benchmark, "require_d2_graph_proof") as full_proof,
        ):
            live = benchmark.require_d2_graph_proof_from_env(args, delegated)

        self.assertTrue(live["live_revalidated"])
        self.assertFalse(live["full_graph_fingerprint_recomputed"])
        identity.assert_called_once()
        full_proof.assert_not_called()
        changed = json.loads(json.dumps(delegated))
        changed.pop("stable_fingerprint_sha256")
        changed["relations"]["clone"]["relfilenode"] = 9999
        changed = benchmark.validate_d2_graph_proof(
            changed,
            "public.source_idx",
            "public.clone_idx",
        )
        with (
            mock.patch.object(benchmark.psycopg, "connect", return_value=connection),
            mock.patch.object(benchmark, "ensure_functions"),
            mock.patch.object(
                benchmark,
                "require_d2_relation_identity",
                return_value=changed["relations"],
            ),
            self.assertRaisesRegex(RuntimeError, "live revalidation changed"),
        ):
            benchmark.require_d2_graph_proof_from_env(args, delegated)

    def test_preferred_index_guc_is_set_and_verified_as_regclass(self):
        cursor = mock.Mock()
        cursor.fetchone.side_effect = [("",), ("public.clone_idx", True)]

        observed = benchmark.set_preferred_index_if_supported(
            cursor,
            SimpleNamespace(preferred_index_guc="hnsw.preferred_index"),
            "public.clone_idx",
        )

        self.assertEqual(observed, "public.clone_idx")
        self.assertEqual(
            cursor.execute.call_args_list[1].args,
            (
                "SELECT set_config(%s, %s, false)",
                ("hnsw.preferred_index", "public.clone_idx"),
            ),
        )

    def test_formal_runtime_rejects_missing_preferred_index_guc(self):
        cursor = SimpleNamespace(execute=mock.Mock(), close=mock.Mock())
        connection = SimpleNamespace(cursor=mock.Mock(return_value=cursor), close=mock.Mock())
        args = argparse.Namespace(
            d3_cache_mb=1024,
            d1_cache_mb=512,
            insertion_table="items",
            insertion_index="source_idx",
            bfs_table="items",
            bfs_index="clone_idx",
            plan_query_id=None,
            preferred_index_guc="hnsw.preferred_index",
            require_preferred_index_guc=True,
            expected_sqlens_build_id="sqlens-v11-exact",
            expected_vector_so_sha256="a" * 64,
        )
        config = {
            "ef_search": 100,
            "max_scan_tuples": 1000,
            "scan_mem_multiplier": 8.0,
            "iterative_scan": "off",
            "guided_collect_target": 100,
        }
        with (
            mock.patch.object(benchmark.psycopg, "connect", return_value=connection),
            mock.patch.object(
                benchmark,
                "backend_cpu_provenance",
                return_value={
                    "backend_pid": 4321,
                    "requested_cpu_list": "",
                    "observed_cpu_list": "48-63",
                    "exact_match": None,
                    "pinning_attempted_by_runner": False,
                },
            ),
            mock.patch.object(
                benchmark,
                "require_exact_sqlens_identity",
                return_value={
                    "expected_build_id": "sqlens-v11-exact",
                    "expected_vector_so_sha256": "a" * 64,
                    "observed_build_id": "sqlens-v11-exact",
                    "observed_vector_so_sha256": "a" * 64,
                    "exact_match": True,
                },
            ),
            mock.patch.object(benchmark, "ensure_functions"),
            mock.patch.object(benchmark, "ensure_tracking"),
            mock.patch.object(benchmark, "effective_mode_config", return_value=config),
            mock.patch.object(benchmark, "configure"),
            mock.patch.object(benchmark, "set_preferred_index_if_supported", return_value=None),
            self.assertRaisesRegex(RuntimeError, "requires hnsw.preferred_index"),
        ):
            benchmark.open_mode_runtime(args, "design1_bloom_bfs_layout", [])

        connection.close.assert_called_once()

    def test_tie_aware_recall_uses_distance_threshold_and_boundary_capacity(self):
        truth = benchmark.TruthEntry(
            query_id=99,
            filtered_rows=12,
            kth_distance_sq=1.0,
            tie_tolerance=0.0,
            strict_closer_count=8,
            boundary_tied=True,
            self_excluded=True,
        )

        recall = benchmark.tie_aware_recall(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0],
            truth,
            10,
        )

        self.assertEqual(recall, 1.0)
        self.assertEqual(benchmark.tie_aware_recall([0.1] * 7 + [1.0, 2.0, 3.0], truth, 10), 0.8)

    def test_explain_gate_requires_actual_expected_hnsw_index_node(self):
        cursor = mock.Mock()
        metadata = (42, "items_embedding_hnsw", "public", "hnsw", 7, "items", "public")
        matching_plan = [
            {
                "Plan": {
                    "Node Type": "Limit",
                    "Plans": [
                        {
                            "Node Type": "Index Scan",
                            "Schema": "public",
                            "Relation Name": "items",
                            "Index Name": "items_embedding_hnsw",
                        }
                    ],
                }
            }
        ]
        cursor.fetchone.side_effect = [metadata, (matching_plan,)]

        evidence = benchmark.explain_hnsw_plan(
            cursor,
            "public.items",
            "public.items_embedding_hnsw",
            "rating = 5",
            99,
            10,
        )

        self.assertTrue(evidence["passed"])
        explain_sql, explain_params = cursor.execute.call_args_list[1].args
        self.assertIn("EXPLAIN (FORMAT JSON, VERBOSE)", explain_sql)
        self.assertIn("id <> %s", explain_sql)
        self.assertEqual(explain_params, (99, 99))

        cursor = mock.Mock()
        wrong_plan = [
            {
                "Plan": {
                    "Node Type": "Index Scan",
                    "Schema": "public",
                    "Relation Name": "items",
                    "Index Name": "output_label_is_not_evidence",
                }
            }
        ]
        cursor.fetchone.side_effect = [metadata, (wrong_plan,)]
        rejected = benchmark.explain_hnsw_plan(
            cursor,
            "public.items",
            "public.items_embedding_hnsw",
            "rating = 5",
            99,
            10,
        )

        self.assertFalse(rejected["passed"])

    def test_interleaved_run_reuses_one_runtime_per_mode_and_balances_warmups(self):
        modes = benchmark.MODES[:3]
        args = argparse.Namespace(
            modes=modes,
            schedule_seed=20260718,
            warmup_all_queries=False,
            warmup_queries=1,
            repeats=2,
            progress_queries=0,
        )
        filters = [("filter_a", 1.0, "a = 1"), ("filter_b", 2.0, "b = 1")]
        query_nos = [1, 2]
        query_by_no = {1: 101, 2: 102}
        warmup_modes: list[str] = []
        opened: list[SimpleNamespace] = []
        closed: list[SimpleNamespace] = []

        def fake_open(_args, mode, _filters):
            runtime = SimpleNamespace(mode=mode)
            opened.append(runtime)
            return runtime

        def fake_warmup(_args, runtime, _filter_name, _predicate, _query_id):
            warmup_modes.append(runtime.mode)

        def fake_measured(
            _args,
            runtime,
            filter_name,
            _selectivity,
            _predicate,
            query_no,
            _query_id,
            repeat,
            _truth,
            schedule_position,
            block_no=0,
            query_order_position=0,
        ):
            return {
                "mode": runtime.mode,
                "filter_name": filter_name,
                "pair_key": benchmark.pair_key(filter_name, query_no, repeat),
                "schedule_position": schedule_position,
                "block_no": block_no,
                "query_order_position": query_order_position,
                "error": "",
            }

        with (
            mock.patch.object(benchmark, "open_mode_runtime", side_effect=fake_open),
            mock.patch.object(benchmark, "close_mode_runtime", side_effect=closed.append),
            mock.patch.object(benchmark, "run_warmup", side_effect=fake_warmup),
            mock.patch.object(benchmark, "run_measured_query", side_effect=fake_measured),
        ):
            rows = benchmark.run_interleaved(args, filters, query_nos, query_by_no, truth={})

        self.assertEqual([runtime.mode for runtime in opened], modes)
        self.assertEqual({id(runtime) for runtime in opened}, {id(runtime) for runtime in closed})
        for offset in range(0, len(warmup_modes), len(modes)):
            self.assertEqual(sorted(warmup_modes[offset : offset + len(modes)]), sorted(modes))

        for offset in range(0, len(rows), len(modes)):
            group = rows[offset : offset + len(modes)]
            self.assertEqual(len({row["pair_key"] for row in group}), 1)
            self.assertEqual({row["mode"] for row in group}, set(modes))
            self.assertEqual([row["schedule_position"] for row in group], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
