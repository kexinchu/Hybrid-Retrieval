from __future__ import annotations

import unittest

from psycopg import sql

from experiments.hybrid_vector_db.scripts.pgvector_update_concurrency_correctness import (
    DURING_BUILD_OPERATIONS,
    GUIDANCE_KINDS,
    ISOLATIONS,
    OPERATIONS,
    PHASES,
    StrictCorrectnessFailure,
    build_schedule_grid,
    _plan_index_names,
    _query_statement,
    strict_failure_status,
    validate_runtime_provenance,
    validate_schedule_completeness,
    validate_result,
)


class PgvectorUpdateConcurrencyCorrectnessTests(unittest.TestCase):
    def test_runtime_provenance_requires_loaded_binary_hash(self):
        valid = {
            "build_id": "sqlens-v11-test",
            "build_id_error": "",
            "vector_library_sha256": "a" * 64,
            "vector_library_error": "",
        }
        self.assertEqual(validate_runtime_provenance(valid), [])
        self.assertTrue(validate_runtime_provenance(valid | {"build_id": "stock"}))
        self.assertTrue(validate_runtime_provenance(valid | {"vector_library_sha256": "missing"}))

    def test_grid_covers_all_operations_and_excludes_truncate_during_build(self):
        grid = build_schedule_grid()
        keys = {(case.operation, case.phase) for case in grid}
        for operation in set(OPERATIONS) - {"truncate_tid_reuse"}:
            self.assertIn((operation, "build_before_write"), keys)
            self.assertIn((operation, "write_before_load"), keys)
        self.assertIn(("truncate_tid_reuse", "write_before_load"), keys)
        self.assertNotIn(("truncate_tid_reuse", "build_before_write"), keys)
        for operation in DURING_BUILD_OPERATIONS:
            self.assertIn((operation, "write_during_fragment_build"), keys)
        self.assertNotIn(("truncate_tid_reuse", "write_during_fragment_build"), keys)
        self.assertEqual(
            len(grid),
            len(ISOLATIONS)
            * len(GUIDANCE_KINDS)
            * (len(DURING_BUILD_OPERATIONS) * 3 + 1),
        )

    def test_plan_index_name_walker_finds_nested_hnsw(self):
        plan = [{"Plan": {"Node Type": "Limit", "Plans": [{"Index Name": "fixture_hnsw"}]}}]
        self.assertEqual(_plan_index_names(plan), ["fixture_hnsw"])

    def test_guided_statement_binds_signature_but_exact_truth_does_not(self):
        guided = _query_statement(sql, "fixture", "tenant_id = 1", exact=False).as_string(None)
        exact = _query_statement(sql, "fixture", "tenant_id = 1", exact=True).as_string(None)
        self.assertIn("vector_hnsw_guidance_bind", guided)
        self.assertIn("OFFSET 0", guided)
        self.assertNotIn("vector_hnsw_guidance_bind", exact)

    def test_grid_is_deterministic_and_repeats_are_explicit(self):
        first = build_schedule_grid(
            isolations=("repeatable_read",),
            guidance_kinds=("bloom",),
            operations=("delete",),
            phases=PHASES,
            repeats=2,
        )
        second = build_schedule_grid(
            isolations=("repeatable_read",),
            guidance_kinds=("bloom",),
            operations=("delete",),
            phases=PHASES,
            repeats=2,
        )
        self.assertEqual(first, second)
        self.assertEqual([case.repeat for case in first], [0, 0, 0, 1, 1, 1])

    def test_grid_rejects_unknown_values_and_nonpositive_repeats(self):
        with self.assertRaisesRegex(ValueError, "unknown schedule value"):
            build_schedule_grid(operations=("not_an_operation",))
        with self.assertRaisesRegex(ValueError, "repeats must be positive"):
            build_schedule_grid(repeats=0)

    def test_strict_failure_detects_false_negative(self):
        with self.assertRaisesRegex(StrictCorrectnessFailure, "false negative"):
            validate_result([10, 12], [10, 11])

    def test_strict_failure_detects_ordered_mismatch(self):
        with self.assertRaisesRegex(StrictCorrectnessFailure, "ordered mismatch"):
            validate_result([12, 10], [10, 12])

    def test_strict_status_is_nonzero_for_any_correctness_or_backend_error(self):
        self.assertEqual(strict_failure_status([{"result_ids": [1], "truth_ids": [1]}]), 0)
        self.assertEqual(strict_failure_status([{"false_negative": True}]), 1)
        self.assertEqual(strict_failure_status([{"ordered_mismatch": True}]), 1)
        self.assertEqual(strict_failure_status([{"error": "serialization failure"}]), 1)

    def test_schedule_completeness_rejects_missing_and_duplicate_rows(self):
        schedule = build_schedule_grid(
            isolations=("read_committed",),
            guidance_kinds=("page",),
            operations=("delete",),
            phases=PHASES,
        )
        records = []
        for case in schedule:
            case_id = (
                f"{case.operation}-{case.phase}-{case.isolation}-"
                f"{case.guidance_kind}-r{case.repeat}"
            )
            phases = ("precommit", "postcommit") if case.phase == "build_before_write" else ("postcommit",)
            for phase in phases:
                for query_no in range(2):
                    records.append(
                        {
                            "case_id": case_id,
                            "phase": phase,
                            "query_no": query_no,
                            "backend_role": "reader" if case.phase == "build_before_write" else "query",
                        }
                    )
        self.assertEqual(validate_schedule_completeness(records, schedule, 2), [])
        self.assertTrue(validate_schedule_completeness(records[:-1], schedule, 2))
        self.assertTrue(validate_schedule_completeness(records + [records[0]], schedule, 2))


if __name__ == "__main__":
    unittest.main()
