import argparse
import contextlib
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


from experiments.hybrid_vector_db.scripts import pgvector_upstream_overhead_control as runner


class Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.statements = []

    def execute(self, sql, *_args):
        self.statements.append(sql)

    def fetchone(self):
        return self.rows.pop(0)

    def fetchall(self):
        return self.rows.pop(0)


def completed_summary(config, recall, latency):
    return {
        "config_label": config.label,
        "config_family": config.family,
        "complete": True,
        "recall_mean": recall,
        "recall_lcb95": recall,
        "latency_mean_ms": latency,
        "latency_p95_ms": latency,
        "latency_p99_ms": latency,
    }


class PgvectorUpstreamOverheadControlTests(unittest.TestCase):
    def test_default_ladder_uses_only_the_official_upstream_ef_range(self):
        raw = runner.default_config_ladder()
        effective, proof = runner.effective_config_grid(raw)

        self.assertEqual(len(raw), 28)
        self.assertEqual(len(effective), 28)
        self.assertEqual(proof["dropped_equivalent_configs"], 0)
        self.assertLessEqual(
            max(config.ef_search for config in effective),
            runner.UPSTREAM_MAX_EF_SEARCH,
        )
        self.assertEqual(
            {family: sum(config.family == family for config in effective) for family in {config.family for config in effective}},
            {"off": 4, "strict_order": 12, "relaxed_order": 12},
        )
        self.assertEqual(
            len({(config.max_scan_tuples, config.scan_mem_multiplier) for config in effective if config.family == "strict_order"}),
            3,
        )

    def test_off_configs_dedup_max_scan_and_memory_as_semantically_irrelevant(self):
        configs = runner.default_config_ladder()
        configs.extend(
            [
                runner.Config(250, "off", 5_000_000, 32.0, 9),
                runner.Config(250, "off", 1_000_000, 8.0, 4),
            ]
        )

        effective, proof = runner.effective_config_grid(configs)

        self.assertEqual(len(effective), 28)
        self.assertEqual(proof["dropped_equivalent_configs"], 2)
        off_250 = [config for config in effective if config.family == "off" and config.ef_search == 250]
        self.assertEqual(len(off_250), 1)
        self.assertEqual(off_250[0].max_scan_tuples, runner.OFF_REPRESENTATIVE_MAX_SCAN)
        self.assertEqual(off_250[0].scan_mem_multiplier, runner.OFF_REPRESENTATIVE_SCAN_MEM)

        conflicting = runner.default_config_ladder() + [
            runner.Config(250, "strict_order", 100_000, 1.0, 99)
        ]
        with self.assertRaisesRegex(ValueError, "conflicting budget_rank"):
            runner.effective_config_grid(conflicting)

    def test_formal_grid_does_not_require_exploratory_relaxed_order(self):
        configs = [
            runner.Config(100, "off", 100_000, 1.0, 0),
            runner.Config(100, "strict_order", 100_000, 1.0, 1),
            runner.Config(100, "strict_order", 5_000_000, 32.0, 3),
        ]

        effective, proof = runner.effective_config_grid(configs)
        max_budget = runner.family_max_budget_configs(effective)

        self.assertEqual(proof["families"], ["off", "strict_order"])
        self.assertEqual(proof["required_formal_families"], ["off", "strict_order"])
        self.assertEqual(set(max_budget), {"off", "strict_order"})
        self.assertEqual(max_budget["strict_order"].budget_rank, 3)

    def test_default_query_count_bound_is_53200_per_implementation(self):
        counts = runner.default_query_count_bounds(28)

        self.assertEqual(counts["screen_queries"], 7_840)
        self.assertEqual(counts["max_promoted_configs_per_filter"], 9)
        self.assertEqual(counts["verification_query_upper_bound"], 20_160)
        self.assertEqual(counts["final_query_upper_bound"], 25_200)
        self.assertEqual(counts["total_query_upper_bound"], 53_200)

    def test_custom_ladder_rejects_sqlens_only_ef_values(self):
        with self.assertRaisesRegex(ValueError, "official pgvector"):
            runner.config_from_mapping(
                {
                    "ef_search": 1500,
                    "iterative_scan": "strict_order",
                    "max_scan_tuples": 5_000_000,
                    "scan_mem_multiplier": 32,
                    "budget_rank": 3,
                }
            )

        accepted = runner.config_from_mapping(
            {
                "ef_search": 10_000,
                "iterative_scan": "strict_order",
                "max_scan_tuples": 5_000_000,
                "scan_mem_multiplier": 32,
                "budget_rank": 3,
            },
            max_ef_search=10_000,
        )
        self.assertEqual(accepted.ef_search, 10_000)

    def test_candidate_validity_is_part_of_sql_but_not_the_workload_predicate(self):
        sql = runner.build_hybrid_sql(
            "public.items",
            "rating = 5",
            10,
            "embedding_valid",
        )
        normalized = " ".join(sql.split())
        self.assertIn(
            "WHERE (rating = 5) AND (embedding_valid) AND id <> %s",
            normalized,
        )
        with self.assertRaisesRegex(ValueError, "candidate-validity"):
            runner.build_hybrid_sql("public.items", "rating = 5", 10, "true; DROP")

    def test_upstream_evaluation_patch_must_equal_the_canonical_two_file_diff(self):
        canonical = Path("patches/pgvector-v0.8.2-ef-search-10000.patch").read_bytes()
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            patch = Path(temporary) / "ceiling.patch"
            patch.write_bytes(canonical)
            command_results = [
                SimpleNamespace(returncode=0, stdout=canonical, stderr=b""),
                SimpleNamespace(
                    returncode=0,
                    stdout="src/hnsw.c\nsrc/hnsw.h\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
            ]
            with mock.patch.object(runner.subprocess, "run", side_effect=command_results):
                proof = runner.upstream_parameter_ceiling_provenance(
                    source, patch, 10_000
                )

            self.assertFalse(proof["algorithm_change"])
            self.assertEqual(proof["patch_sha256"], runner.UPSTREAM_EF10000_PATCH_SHA256)
            self.assertEqual(proof["changed_files"], ["src/hnsw.c", "src/hnsw.h"])

            patch.write_bytes(canonical + b"\n")
            with self.assertRaisesRegex(runner.ProvenanceGateError, "canonical"):
                runner.upstream_parameter_ceiling_provenance(source, patch, 10_000)

    def test_promotion_includes_margin_winner_family_recall_and_max_budget_proofs(self):
        configs = [
            runner.Config(100, "off", 100_000, 1.0, 0),
            runner.Config(200, "off", 100_000, 1.0, 0),
            runner.Config(100, "strict_order", 100_000, 1.0, 1),
            runner.Config(200, "strict_order", 5_000_000, 32.0, 3),
            runner.Config(100, "relaxed_order", 100_000, 1.0, 1),
            runner.Config(200, "relaxed_order", 5_000_000, 32.0, 3),
        ]
        recalls = [0.90, 0.91, 0.96, 0.97, 0.94, 0.95]
        latencies = [1.0, 3.0, 5.0, 8.0, 2.0, 6.0]
        summaries = [
            completed_summary(config, recall, latency)
            for config, recall, latency in zip(configs, recalls, latencies)
        ]

        promoted, proof = runner.build_promotion_set(summaries, configs, [0.95], margin=0.02)
        reasons = {row["config_label"]: row["promotion_reasons"] for row in proof}

        self.assertEqual(
            {config.label for config in promoted},
            {configs[1].label, configs[3].label, configs[4].label, configs[5].label},
        )
        self.assertIn("fastest_screen_target_0.95_minus_margin_0.02", reasons[configs[4].label])
        self.assertIn("family_strict_order_max_screen_recall", reasons[configs[3].label])
        self.assertIn("global_max_screen_recall", reasons[configs[3].label])
        self.assertIn("family_off_maximum_budget_verification_boundary", reasons[configs[1].label])

    def test_promotion_and_selection_never_cross_the_declared_family(self):
        off = runner.Config(100, "off", 100_000, 1.0, 0)
        strict = runner.Config(100, "strict_order", 100_000, 1.0, 1)
        summaries = [
            completed_summary(off, 0.99, 100.0),
            completed_summary(strict, 0.96, 5.0),
        ]

        promoted, _proof = runner.build_promotion_set(
            summaries, [off, strict], [0.95], margin=0.02, family="strict_order"
        )
        selected, status, _proof = runner.select_verified_config(
            summaries,
            0.95,
            verified_config_labels=[strict.label],
            family="strict_order",
        )

        self.assertEqual(promoted, [strict])
        self.assertEqual(status, "selected")
        self.assertEqual(selected["config_label"], strict.label)

    def test_no_verified_config_is_explicitly_not_an_unattainability_claim(self):
        labels = {"off": "off-max", "strict_order": "strict-max", "relaxed_order": "relaxed-max"}
        summaries = [
            {
                "config_label": label,
                "complete": True,
                "recall_lcb95": 0.94,
                "latency_mean_ms": position + 1.0,
            }
            for position, label in enumerate(labels.values())
        ]

        selected, status, proof = runner.select_verified_config(
            summaries, 0.95, verified_config_labels=list(labels.values())
        )
        self.assertIsNone(selected)
        self.assertEqual(status, "no_verified_config_meets_target")
        self.assertFalse(proof["claims_unattainable"])
        self.assertEqual(proof["verified_configs"], sorted(labels.values()))

        selected, status, proof = runner.select_verified_config(
            summaries[:-1], 0.95, verified_config_labels=list(labels.values())
        )
        self.assertIsNone(selected)
        self.assertEqual(status, "incomplete_verification")
        self.assertEqual(proof["missing_verified_configs"], ["relaxed-max"])

    def test_verification_selection_uses_lcb_then_fastest_mean_latency(self):
        summaries = [
            {"config_label": "slow", "complete": True, "recall_lcb95": 0.98, "latency_mean_ms": 8.0},
            {"config_label": "fast", "complete": True, "recall_lcb95": 0.96, "latency_mean_ms": 3.0},
            {"config_label": "uncertain", "complete": True, "recall_lcb95": 0.94, "latency_mean_ms": 1.0},
        ]
        selected, status, _proof = runner.select_verified_config(
            summaries,
            0.95,
            verified_config_labels=["slow", "fast", "uncertain"],
        )
        self.assertEqual(status, "selected")
        self.assertEqual(selected["config_label"], "fast")

    def test_heldout_final_lcb_miss_is_never_reported_as_success(self):
        metrics = {"complete": True, "recall_lcb95": 0.94}

        self.assertEqual(
            runner.heldout_final_status("selected", 0.95, metrics),
            "missed_target",
        )
        self.assertEqual(
            runner.heldout_final_status(
                "selected", 0.95, {"complete": True, "recall_lcb95": 0.96}
            ),
            "confirmed",
        )

    def test_query_splits_are_disjoint_and_cover_q0_through_q199(self):
        contract = runner.validate_split_contract()

        self.assertEqual(contract["screen"], {"first": 0, "last": 19, "queries": 20})
        self.assertEqual(contract["verification"], {"first": 20, "last": 99, "queries": 80})
        self.assertEqual(contract["final"], {"first": 100, "last": 199, "queries": 100})
        self.assertFalse(set(runner.SCREEN_QUERY_NOS) & set(runner.VERIFICATION_QUERY_NOS))
        self.assertFalse(set(runner.VERIFICATION_QUERY_NOS) & set(runner.FINAL_QUERY_NOS))

    def test_official_runtime_gate_uses_extension_version_but_not_stale_sql_declarations(self):
        cursor = Cursor([("0.8.2",)])

        provenance = runner.gate_implementation(cursor, "official")

        self.assertFalse(provenance["runtime_sql_declarations_used_as_identity"])
        self.assertEqual(len(cursor.statements), 1)
        self.assertNotIn("vector_sqlens_build_id", cursor.statements[0])

    def test_sqlens_gate_defaults_to_v11_and_profile_semantics_four(self):
        profile = {
            "profile_semantics_version": 4,
            "graph_elements_visited": 10,
            "raw_index_tids_returned": 4,
            "hnsw_am_callback_ms": 1.0,
            "executor_residual_ms": 0.5,
        }
        cursor = Cursor([("0.8.2",), ("sqlens-v11-amazon",), (json.dumps(profile),)])

        provenance = runner.gate_implementation(cursor, "sqlens_disabled")

        self.assertEqual(provenance["loaded_vector_sqlens_build_id"], "sqlens-v11-amazon")
        self.assertEqual(provenance["profile_gate"]["required_build_prefix"], "sqlens-v11-")
        self.assertEqual(provenance["profile_gate"]["minimum_profile_semantics_version"], 4.0)

        old = profile | {"profile_semantics_version": 3}
        with self.assertRaises(runner.ProvenanceGateError):
            runner.gate_implementation(
                Cursor([("0.8.2",), ("sqlens-v11-amazon",), (json.dumps(old),)]),
                "sqlens_disabled",
            )

    def test_sqlens_disabled_resets_every_current_v11_extension_guc(self):
        cursor = Cursor([])

        statements = runner.disable_sqlens_gucs(cursor)

        self.assertEqual(statements, cursor.statements)
        for guc in (
            "hnsw.filter_strategy",
            "hnsw.page_access",
            "hnsw.index_page_access",
            "hnsw.guidance_compose_exact_or",
            "hnsw.guidance_require_epoch",
            "hnsw.require_full_memory_build",
        ):
            self.assertIn(f"SET {guc} = off", statements)
        for guc in ("hnsw.metadata_cache_max_mb", "hnsw.build_page_order", "hnsw.build_seed"):
            self.assertIn(f"RESET {guc}", statements)
        for guc in ("hnsw.clone_source", "hnsw.preferred_index"):
            self.assertIn(f"SET {guc} = ''", statements)

    def test_runtime_hnsw_guc_inventory_forces_every_nonstock_knob_safe(self):
        class GucCursor:
            def __init__(self):
                self.statements = []
                self.inventory_reads = 0

            def execute(self, sql, *_args):
                self.statements.append(sql)
                if "FROM pg_settings" in sql:
                    self.inventory_reads += 1

            def fetchall(self):
                if self.inventory_reads == 1:
                    return [
                        ("hnsw.ef_search", "integer", "40", "40"),
                        ("hnsw.clone_source", "string", "public.old", ""),
                        ("hnsw.preferred_index", "string", "public.idx", ""),
                        ("hnsw.traversal_guidance", "bool", "on", "off"),
                        ("hnsw.experimental_budget", "integer", "9", "0"),
                    ]
                return [
                    ("hnsw.ef_search", "integer", "40", "40"),
                    ("hnsw.clone_source", "string", "", ""),
                    ("hnsw.preferred_index", "string", "", ""),
                    ("hnsw.traversal_guidance", "bool", "off", "off"),
                    ("hnsw.experimental_budget", "integer", "0", "0"),
                ]

        cursor = GucCursor()
        audit = runner.enforce_hnsw_guc_allowlist(cursor)

        self.assertEqual(audit["stock_allowlist"], sorted(runner.STOCK_HNSW_GUCS))
        self.assertEqual(audit["unhandled_nonstock_gucs"], [])
        self.assertIn("SET hnsw.clone_source = ''", cursor.statements)
        self.assertIn("SET hnsw.preferred_index = ''", cursor.statements)
        self.assertIn("SET hnsw.traversal_guidance = off", cursor.statements)
        self.assertIn("RESET hnsw.experimental_budget", cursor.statements)

    def test_measurement_timer_stops_immediately_after_fetchall(self):
        config = runner.Config(100, "off", 100_000, 1.0)
        truth = runner.TruthEntry(7, 1.0, 0.0, tuple(range(10)), True)

        class MeasurementCursor:
            def execute(self, *_args):
                return None

            def fetchall(self):
                return [(1, 0.5)]

        with mock.patch.object(runner.time, "perf_counter", side_effect=[10.0, 10.125]), \
                mock.patch.object(runner, "tie_aware_recall", return_value=0.9) as recall:
            row = runner.measurement_row(
                "official", "final", "f", 100, 7, 0, config, 1,
                "[0,1]", truth, MeasurementCursor(), "SELECT 1", 10,
            )

        self.assertEqual(row["latency_ms"], 125.0)
        recall.assert_called_once()

    def test_database_fingerprint_binds_cluster_database_relations_and_epoch(self):
        cursor = Cursor([
            ("cluster-123", "db", 42, "postgres", "127.0.0.1", 55432, "17.5", "0.8.2"),
            (10_000_000, 0, 9_999_999, 100, 101),
            (200, 201, "CREATE INDEX idx ON public.items USING hnsw (embedding vector_l2_ops)"),
        ])

        fingerprint = runner.database_fingerprint(
            cursor, "public.items", "public.idx", "amazon10m-v1"
        )

        self.assertEqual(fingerprint["system_identifier"], "cluster-123")
        self.assertEqual(fingerprint["database_oid"], 42)
        self.assertEqual(fingerprint["table_oid"], 100)
        self.assertEqual(fingerprint["index_oid"], 200)
        self.assertEqual(fingerprint["data_epoch"], "amazon10m-v1")
        self.assertEqual(
            fingerprint["indexdef_sha256"],
            hashlib.sha256(
                b"CREATE INDEX idx ON public.items USING hnsw (embedding vector_l2_ops)"
            ).hexdigest(),
        )

    def test_graph_identity_requires_same_heap_logical_equivalence(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.json"
            payload = {
                "source_index": "public.source_idx",
                "clone_index": "public.clone_idx",
                "stable_fingerprint_sha256": "f" * 64,
                "comparison": {
                    "format": "sqlens-hnsw-compare-v2",
                    "same_heap": True,
                    "entry_equal": True,
                    "logical_equal": True,
                    "definition_equal": True,
                    "tuple_coverage_equal": True,
                    "physical_equal": False,
                    "left_definition_digest": "sha256:def",
                    "right_definition_digest": "sha256:def",
                    "left_tuple_coverage_digest": "sha256:tid",
                    "right_tuple_coverage_digest": "sha256:tid",
                    "left_logical_digest": "sha256:logical",
                    "right_logical_digest": "sha256:logical",
                    "left_physical_digest": "sha256:left",
                    "right_physical_digest": "sha256:right",
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            identity = runner.load_graph_identity(
                path, "public.source_idx", "public.clone_idx"
            )
            self.assertTrue(identity["logical_equal"])
            self.assertFalse(identity["physical_equal"])
            self.assertEqual(identity["logical_digest"], "sha256:logical")

            payload["comparison"]["logical_equal"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(runner.ProvenanceGateError):
                runner.load_graph_identity(
                    path, "public.source_idx", "public.clone_idx"
                )

            payload["comparison"]["logical_equal"] = True
            payload["comparison"]["right_logical_digest"] = "sha256:other"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(runner.ProvenanceGateError):
                runner.load_graph_identity(
                    path, "public.source_idx", "public.clone_idx"
                )

    def test_formal_design_is_fixed_to_fourteen_filters_three_targets_and_declared_family(self):
        filters = [{"filter_name": f"f{i}"} for i in range(14)]
        design = runner.validate_formal_design(filters, [0.90, 0.95, 0.99], "strict_order")

        self.assertEqual(design["cell_count"], 42)
        self.assertEqual(design["formal_family"], "strict_order")
        with self.assertRaisesRegex(ValueError, "exactly 14"):
            runner.validate_formal_design(filters[:-1], [0.90, 0.95, 0.99], "off")
        with self.assertRaisesRegex(ValueError, "exactly 0.90,0.95,0.99"):
            runner.validate_formal_design(filters, [0.90, 0.95], "off")
        with self.assertRaisesRegex(ValueError, "relaxed_order"):
            runner.validate_formal_design(filters, [0.90, 0.95, 0.99], "relaxed_order")

    def test_checkpoint_spec_includes_runtime_binding_and_run_uuid(self):
        spec = runner.build_checkpoint_spec(
            run_uuid="run-123",
            base_spec={"filters": ["f"]},
            database_fingerprint={"system_identifier": "cluster", "database_oid": 7},
            binary_provenance={"vector_so_sha256": "a" * 64},
            settings_audit={"after": {"hnsw.clone_source": ""}},
        )

        self.assertEqual(spec["run_uuid"], "run-123")
        self.assertEqual(spec["database_fingerprint"]["database_oid"], 7)
        self.assertEqual(spec["binary_provenance"]["vector_so_sha256"], "a" * 64)
        self.assertEqual(spec["checkpoint_spec_sha256"], runner.sha256_json(spec | {"checkpoint_spec_sha256": None}))

    def test_server_binary_gate_hashes_pg_config_vector_so_inside_container(self):
        digest = runner.OFFICIAL_UPSTREAM_VECTOR_SO_SHA256
        command = mock.Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout="pgvector-upstream:0.8.2\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="/usr/lib/postgresql/17/lib\n", stderr=""),
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{digest}  /usr/lib/postgresql/17/lib/vector.so\n",
                    stderr="",
                ),
                SimpleNamespace(
                    returncode=0,
                    stdout=f"sha256:{'c' * 64}\n",
                    stderr="",
                ),
                SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "CpusetCpus": "48-63",
                            "CpuPeriod": 100000,
                            "CpuQuota": 0,
                            "NanoCpus": 0,
                            "Memory": 68719476736,
                            "MemorySwap": 68719476736,
                        }
                    ),
                    stderr="",
                ),
            ]
        )

        provenance = runner.server_vector_binary_provenance("pg-server", digest, command)

        self.assertEqual(provenance["vector_so_sha256"], digest)
        self.assertEqual(provenance["server_image"], "pgvector-upstream:0.8.2")
        self.assertEqual(provenance["server_image_id"], f"sha256:{'c' * 64}")
        self.assertEqual(provenance["server_resource_limits"]["cpuset_cpus"], "48-63")
        self.assertEqual(
            provenance["server_resource_limits"]["memory_bytes"], 68719476736
        )
        self.assertEqual(
            command.call_args_list[2].args[0],
            [
                "docker",
                "exec",
                "pg-server",
                "sha256sum",
                "/usr/lib/postgresql/17/lib/vector.so",
            ],
        )

    def test_server_binary_gate_fails_closed_on_digest_mismatch(self):
        actual = "a" * 64
        expected = "b" * 64
        command = mock.Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout="image\n", stderr=""),
                SimpleNamespace(returncode=0, stdout="/pkglib\n", stderr=""),
                SimpleNamespace(returncode=0, stdout=f"{actual}  /pkglib/vector.so\n", stderr=""),
            ]
        )
        with self.assertRaisesRegex(runner.ProvenanceGateError, "SHA-256 mismatch"):
            runner.server_vector_binary_provenance("pg-server", expected, command)

    def test_official_formal_args_require_pinned_digest_container_and_source(self):
        args = argparse.Namespace(
            implementation="official",
            server_container="pg-server",
            expected_vector_so_sha256=runner.OFFICIAL_UPSTREAM_VECTOR_SO_SHA256,
            vector_source_tag="v0.8.2",
            vector_source_commit="abc123",
            vector_build_recipe="make",
            vector_compiler_flags="-O3",
            filters_csv=Path(__file__),
            truth_csv=Path(__file__),
            graph_identity_json=Path(__file__),
            vector_source_repo=Path(__file__).parent,
            source_index="public.source_idx",
            clone_index="public.clone_idx",
            run_uuid="run-123",
            data_epoch="amazon10m-v1",
            target_recalls=[0.90, 0.95, 0.99],
            formal_family="off",
            final_repeats=6,
            execution_stage="calibration",
            final_block=None,
            promotion_margin=0.02,
            minimum_sqlens_profile_semantics=4.0,
        )
        runner.validate_runtime_args(args)

        args.expected_vector_so_sha256 = "a" * 64
        with self.assertRaisesRegex(runner.ProvenanceGateError, "pinned upstream"):
            runner.validate_runtime_args(args)

    def test_stock_sql_is_marker_free_and_hnsw_plan_is_required(self):
        sql = runner.build_hybrid_sql("public.items", "rating = 5 AND price <= 10", 10)
        normalized = " ".join(sql.lower().split())

        self.assertIn("where (rating = 5 and price <= 10) and id <> %s", normalized)
        self.assertIn("order by embedding <-> %s::vector limit 10", normalized)
        self.assertNotIn("sqlens", normalized)
        self.assertNotIn("guidance", normalized)

        with self.assertRaisesRegex(RuntimeError, "HNSW EXPLAIN gate failed"):
            runner.assert_hnsw_explain_gate(
                [{"Plan": {"Node Type": "Seq Scan"}}], "public.items_hnsw"
            )

    def test_checkpoint_resume_accepts_only_exact_complete_measurement_key_blocks(self):
        config = runner.Config(100, "off", 100_000, 1.0, 0)
        query_nos = [0, 1]

        def row(query_no):
            key = runner.measurement_key("official", "screen", "f", query_no, 0, config.label)
            return {field: "" for field in runner.RAW_FIELDS} | {
                "implementation": "official",
                "phase": "screen",
                "query_split": "screen",
                "filter_name": "f",
                "query_no": query_no,
                "repeat": 0,
                "config_label": config.label,
                "config_family": config.family,
                "budget_rank": config.budget_rank,
                "ef_search": config.ef_search,
                "iterative_scan": config.iterative_scan,
                "max_scan_tuples": config.max_scan_tuples,
                "scan_mem_multiplier": config.scan_mem_multiplier,
                "measurement_key": key,
            }

        rows = [row(0), row(1)]
        completed = runner.validate_stage_checkpoint(
            rows, "official", "screen", {"f": [config]}, query_nos, 1
        )
        self.assertEqual(completed, {("screen", "f")})

        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_stage_checkpoint(
                rows[:1], "official", "screen", {"f": [config]}, query_nos, 1
            )
        foreign = rows + [row(1) | {"measurement_key": "foreign"}]
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_stage_checkpoint(
                foreign, "official", "screen", {"f": [config]}, query_nos, 1
            )

    def test_resume_requires_recorded_promotion_hash_before_later_rows(self):
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_derived_resume_hash(
                {"promotion_set_sha256": "old"},
                "promotion_set_sha256",
                "new",
                later_rows_exist=True,
            )
        with self.assertRaises(runner.CheckpointContractError):
            runner.validate_derived_resume_hash(
                {}, "promotion_set_sha256", "new", later_rows_exist=True
            )

    def test_manifest_only_zero_row_checkpoint_can_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "raw": root / "raw.csv",
                "manifest": root / "manifest.json",
            }
            runner.atomic_write_json(paths["manifest"], {"base_run_spec_hash": "spec"})

            rows, manifest = runner.load_checkpoint(paths, "spec", resume=True)

        self.assertEqual(rows, [])
        self.assertEqual(manifest["base_run_spec_hash"], "spec")

    def test_resume_append_only_audit_rejects_mutation(self):
        before = [{"measurement_key": "m1", "latency_ms": "1.0"}]
        after = before + [{"measurement_key": "m2", "latency_ms": 2.0}]

        audit = runner.resume_append_only_audit(before, after)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["new_measurements"], 1)
        with self.assertRaises(runner.CheckpointContractError):
            runner.resume_append_only_audit(
                before,
                [{"measurement_key": "m1", "latency_ms": "9.0"}],
            )

    def test_dry_run_has_no_file_docker_or_database_access(self):
        argv = ["runner", "--implementation", "official", "--dry-run"]
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(output),
            mock.patch.object(runner.Path, "open", side_effect=AssertionError("file access")),
            mock.patch.object(runner.Path, "exists", side_effect=AssertionError("file access")),
            mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("external command")),
            mock.patch.dict(sys.modules, {"psycopg": None}),
        ):
            runner.main()

        payload = json.loads(output.getvalue())
        bounds = payload["default_query_count_bounds_per_implementation"]
        self.assertEqual(payload["default_effective_config_count"], 28)
        self.assertEqual(payload["formal_family_effective_config_count"], 4)
        self.assertEqual(payload["formal_cell_count"], 42)
        self.assertEqual(bounds["screen_queries"], 1_120)
        self.assertEqual(bounds["total_query_upper_bound"], 35_280)
        self.assertFalse(payload["file_access"])
        self.assertFalse(payload["docker_access"])
        self.assertFalse(payload["database_access"])


if __name__ == "__main__":
    unittest.main()
