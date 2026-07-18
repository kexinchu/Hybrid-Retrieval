import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import laion25m_sqlens_variants_benchmark as laion  # noqa: E402
import yfcc_overlap_sqlens_variants_benchmark as yfcc  # noqa: E402


class SqlensVariantBindingTests(unittest.TestCase):
    @staticmethod
    def _gate_cursor(build_id="sqlens-v11-test", profile=None):
        cursor = mock.MagicMock()
        profile = profile or {
            "profile_semantics_version": 4,
            "graph_elements_visited": 1,
            "raw_index_tids_returned": 1,
            "hnsw_am_callback_ms": 0.1,
            "executor_residual_ms": 0.2,
        }
        cursor.execute.side_effect = lambda sql, *args: cursor
        cursor.fetchone.side_effect = [(build_id,), (json.dumps(profile),)]
        return cursor

    def test_sqlens_v11_gate_accepts_build_and_profile_contract(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                build_id, profile = runner.require_sqlens_provenance(self._gate_cursor())
                self.assertTrue(build_id.startswith("sqlens-v11-"))
                self.assertEqual(profile["profile_semantics_version"], 4)

    def test_sqlens_v11_gate_rejects_old_build(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                with self.assertRaisesRegex(runner.SqlensProvenanceGateError, "expected the 'sqlens-v11-' prefix"):
                    runner.require_sqlens_provenance(self._gate_cursor(build_id="sqlens-v10-old"))

                cursor = self._gate_cursor(build_id="sqlens-v8-old")
                with self.assertRaises(runner.SqlensProvenanceGateError):
                    runner.ensure_functions(cursor)
                self.assertFalse(
                    any(str(call.args[0]).startswith("CREATE OR REPLACE FUNCTION") for call in cursor.execute.call_args_list)
                )

    def test_sqlens_v9_gate_rejects_missing_profile_fields(self):
        profile = {
            "profile_semantics_version": 4,
            "graph_elements_visited": 1,
            "raw_index_tids_returned": 1,
            "hnsw_am_callback_ms": 0.1,
        }
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                with self.assertRaisesRegex(runner.SqlensProvenanceGateError, "executor_residual_ms"):
                    runner.require_sqlens_provenance(self._gate_cursor(profile=profile))

    def test_guided_collect_measurements_require_tid_and_traversal_checks(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                self.assertFalse(runner.guidance_scan_contract_satisfied({}, "guided_collect"))
                self.assertFalse(
                    runner.guidance_scan_contract_satisfied(
                        {"guidance_checks": 2, "traversal_guidance_checks": 0},
                        "guided_collect",
                    )
                )
                self.assertTrue(
                    runner.guidance_scan_contract_satisfied(
                        {"guidance_checks": 2, "traversal_guidance_checks": 7},
                        "guided_collect",
                    )
                )

    def test_ensure_functions_registers_guidance_bind_for_both_runners(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                cursor = self._gate_cursor()
                runner.ensure_functions(cursor)
                statements = [call.args[0] for call in cursor.execute.call_args_list]
                self.assertIn(
                    "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_bind(regclass, text[], text) "
                    "RETURNS boolean AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
                    statements,
                )

    def test_yfcc_stock_and_disabled_profiles_do_not_bind(self):
        args = SimpleNamespace(
            stock_table="yfcc_stock",
            stock_index="yfcc_stock_idx",
            bfs_table="yfcc_bfs",
            bfs_index="yfcc_bfs_idx",
            query_table="yfcc_queries",
            guidance_kind="exact",
            k=10,
        )
        row = {"tags_list": [7, 9], "qid": 11}

        stock_sql, stock_params = yfcc.build_hybrid_query(args, "stock", row, {})
        disabled_sql, disabled_params = yfcc.build_hybrid_query(
            args, "d1_d2", row, {"guidance_enabled": False, "index": "yfcc_bfs_idx"}
        )

        for sql in (stock_sql, disabled_sql):
            self.assertNotIn("vector_hnsw_guidance_bind", sql)
            self.assertNotIn("OFFSET 0", sql)
        self.assertEqual(stock_params, ([7, 9], 11))
        self.assertEqual(disabled_params, ([7, 9], 11))

    def test_yfcc_enabled_profile_binds_matching_signature_before_query_params(self):
        args = SimpleNamespace(
            stock_table="yfcc_stock",
            stock_index="yfcc_stock_idx",
            bfs_table="yfcc_bfs",
            bfs_index="yfcc_bfs_idx",
            query_table="yfcc_queries",
            guidance_kind="exact",
            k=10,
        )
        row = {"tags_list": [7, 9], "qid": 11}
        profile = {"guidance_enabled": True, "index": "yfcc_bfs_idx"}

        sql, params = yfcc.build_hybrid_query(args, "d1_d2", row, profile)

        normalized = " ".join(sql.lower().split())
        self.assertIn(
            "(select vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) offset 0) and tags && %s::int[]",
            normalized,
        )
        self.assertEqual(
            params,
            (
                "yfcc_bfs_idx",
                ["sql:tags @> ARRAY[7]", "|", "sql:tags @> ARRAY[9]"],
                "exact",
                [7, 9],
                11,
            ),
        )

    def test_laion_stock_and_disabled_profiles_do_not_bind(self):
        args = SimpleNamespace(
            table="laion_stock",
            stock_index="laion_stock_idx",
            bfs_table="laion_bfs",
            bfs_index="laion_bfs_idx",
            query_table="laion_queries",
            guidance_kind="bloom",
            k=10,
        )
        row = {
            "predicate": "labels && ARRAY[7,9]::int[]",
            "labels_list": [7, 9],
            "workload": "label_or",
            "qid": 11,
        }

        stock_sql, stock_params = laion.build_hybrid_query(args, "stock", row, {})
        disabled_sql, disabled_params = laion.build_hybrid_query(
            args, "d1_d2", row, {"guidance_enabled": False, "index": "laion_bfs_idx"}
        )

        for sql in (stock_sql, disabled_sql):
            self.assertNotIn("vector_hnsw_guidance_bind", sql)
            self.assertNotIn("OFFSET 0", sql)
        self.assertEqual(stock_params, (11,))
        self.assertEqual(disabled_params, (11,))

    def test_laion_enabled_profile_binds_matching_signature_before_query_params(self):
        args = SimpleNamespace(
            table="laion_stock",
            stock_index="laion_stock_idx",
            bfs_table="laion_bfs",
            bfs_index="laion_bfs_idx",
            query_table="laion_queries",
            guidance_kind="bloom",
            k=10,
        )
        row = {
            "predicate": "labels && ARRAY[7,9]::int[]",
            "labels_list": [7, 9],
            "workload": "label_or",
            "qid": 11,
        }
        profile = {"guidance_enabled": True, "index": "laion_bfs_idx"}

        sql, params = laion.build_hybrid_query(args, "d1_d2", row, profile)

        normalized = " ".join(sql.lower().split())
        self.assertIn(
            "(select vector_hnsw_guidance_bind(%s::regclass, %s::text[], %s) offset 0) and (labels && array[7,9]::int[])",
            normalized,
        )
        self.assertEqual(
            params,
            (
                "laion_bfs_idx",
                ["sql:labels @> ARRAY[7]::int[]", "|", "sql:labels @> ARRAY[9]::int[]"],
                "bloom",
                11,
            ),
        )

    @staticmethod
    def _d3_args(runner):
        common = {
            "bfs_table": "bfs_table",
            "bfs_index": "bfs_idx",
            "query_table": "queries",
            "guidance_kind": "bloom",
            "guidance_filter_strategy": "guided_collect",
            "guidance_selectivity_max_pct": 100.0,
            "guidance_max_atoms": 64,
            "d3_guidance_max_atoms": 64,
            "statement_timeout_ms": 1000,
            "ef_search": 100,
            "iterative_scan": "strict_order",
            "max_scan_tuples": 1000,
            "scan_mem_multiplier": 1.0,
            "guided_collect_target": 10,
            "d1_cache_mb": 1,
            "d3_cache_mb": 1,
            "d2_page_access": "off",
            "d2_index_page_access": "off",
            "d2_page_window": 1,
            "d2_page_prefetch_min_items": 1,
            "d2_page_disable_after_no_merge": 1,
            "force_hnsw": False,
            "k": 10,
        }
        if runner is yfcc:
            common.update(stock_table="stock_table", stock_index="stock_idx")
        else:
            common.update(
                table="stock_table",
                stock_index="stock_idx",
                guidance_compose_exact_or=False,
                require_compose_exact_guc=False,
                d3_enable_policy="legacy",
                d3_min_predicted_skip_rate=0.5,
            )
        return SimpleNamespace(**common)

    @staticmethod
    def _d3_row(runner):
        if runner is yfcc:
            return {"tags_list": [7, 9], "filter_pct": 1.0, "qid": 11}
        return {
            "predicate": "labels && ARRAY[7,9]::int[]",
            "labels_list": [7, 9],
            "workload": "label_or",
            "actual_pct": 1.0,
            "qid": 11,
        }

    def test_d3_first_two_activations_are_probes_and_third_binds_adaptive(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                args = self._d3_args(runner)
                row = self._d3_row(runner)
                cursor = mock.MagicMock()
                cursor.fetchone.side_effect = [
                    (0,),
                    (json.dumps({"active": False, "adaptive_probes": 1}),),
                    (0,),
                    (json.dumps({"active": False, "adaptive_probes": 2}),),
                    (2,),
                    (json.dumps({"active": True, "adaptive_admissions": 1}),),
                ]

                first, _ = runner.activate_guidance(cursor, args, "d1_d2_d3", row)
                second, _ = runner.activate_guidance(cursor, args, "d1_d2_d3", row)
                active, _ = runner.activate_guidance(cursor, args, "d1_d2_d3", row)

                self.assertEqual(first["guidance_route"], "d3_probe")
                self.assertEqual(second["guidance_route"], "d3_probe")
                self.assertFalse(first["guidance_enabled"])
                self.assertFalse(second["guidance_enabled"])
                self.assertTrue(active["guidance_enabled"])
                activation_calls = [
                    call for call in cursor.execute.call_args_list if "vector_hnsw_guidance_activate" in call.args[0]
                ]
                self.assertEqual(len(activation_calls), 3)
                self.assertTrue(all(call.args[1][2] == "adaptive" for call in activation_calls))
                sql, params = runner.build_hybrid_query(args, "d1_d2_d3", row, active)
                self.assertIn("vector_hnsw_guidance_bind", sql)
                self.assertEqual(params[2], "adaptive")

    def test_d3_probe_fails_open_without_guidance_binding_error(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                args = self._d3_args(runner)
                row = self._d3_row(runner)
                cursor = mock.MagicMock()
                cursor.fetchall.return_value = [(1,), (2,)]
                with mock.patch.object(runner, "fetch_json", return_value={"guidance_checks": 0}):
                    ids, _, _, error = runner.run_query(
                        cursor,
                        args,
                        "d1_d2_d3",
                        row,
                        {"guidance_enabled": False, "guidance_route": "d3_probe"},
                    )
                self.assertEqual(ids, [1, 2])
                self.assertEqual(error, "")
                query_call = next(
                    call
                    for call in cursor.execute.call_args_list
                    if "SELECT id" in call.args[0]
                )
                self.assertNotIn("vector_hnsw_guidance_bind", query_call.args[0])

    def test_d3_has_no_prewarm_or_activation_reuse_path(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                self.assertFalse(hasattr(runner, "prewarm_d3"))
                self.assertFalse(hasattr(runner, "d3_guidance_signature"))
                self.assertFalse(hasattr(runner, "reuse_activation_profile"))
                self.assertFalse(runner.warmup_enabled(SimpleNamespace(warmup_all_queries=True), "d1_d2_d3"))

    def test_plan_gate_resets_and_reconfigures_before_workload(self):
        for runner in (yfcc, laion):
            with self.subTest(runner=runner.__name__):
                args = self._d3_args(runner)
                row = self._d3_row(runner)
                cursor = mock.MagicMock()
                cursor.fetchone.return_value = ([{"Plan": {"Node Type": "Index Scan", "Index Name": "bfs_idx"}}],)
                with (
                    mock.patch.object(
                        runner,
                        "activate_guidance",
                        return_value=({"guidance_enabled": False}, 0.0),
                    ),
                    mock.patch.object(runner, "build_hybrid_query", return_value=("SELECT 1", ())),
                    mock.patch.object(runner, "configure") as configure,
                ):
                    self.assertTrue(runner.gate_method_plan(cursor, args, "d1_d2_d3", row))
                statements = [call.args[0] for call in cursor.execute.call_args_list]
                self.assertIn("SELECT vector_hnsw_guidance_reset()", statements)
                self.assertIn("SELECT vector_hnsw_metadata_cache_reset()", statements)
                configure.assert_called_once_with(cursor, args, "d1_d2_d3")


if __name__ == "__main__":
    unittest.main()
