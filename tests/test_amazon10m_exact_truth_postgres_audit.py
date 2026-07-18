from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/hybrid_vector_db/scripts/amazon10m_exact_truth_postgres_audit.py"
SPEC = importlib.util.spec_from_file_location("amazon10m_exact_truth_postgres_audit", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def truth_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query_no": 0,
        "query_id": 10,
        "filter_name": "f",
        "predicate": "rating >= 4",
        "k": 2,
        "exact_topk_ids": "1,2",
        "exact_topk_distances_sq": "1.0,2.0",
        "exact_topk_plus_one_ids": "1,2,3",
        "exact_topk_plus_one_distances_sq": "1.0,2.0,3.0",
        "kth_distance_sq": "2.0",
        "tie_tolerance": "0.001",
        "strict_closer_count": "1",
        "boundary_tied": "False",
        "self_excluded": "True",
    }
    row.update(overrides)
    return row


class PostgresExactAuditTests(TestCase):
    def test_unsafe_predicates_are_rejected(self) -> None:
        for predicate in ("rating >= 4; DROP TABLE x", "rating >= 4 -- comment", "(rating >= 4", "SELECT 1"):
            with self.assertRaises(ValueError):
                audit.safe_sql_predicate(predicate)
        self.assertEqual(audit.safe_sql_predicate("rating >= 4 AND store = 'x'"), "rating >= 4 AND store = 'x'")

    def test_plan_gate_accepts_seq_scan_and_rejects_index_scan(self) -> None:
        good = {"Plan": {"Node Type": "Limit", "Plans": [{"Node Type": "Seq Scan", "Relation Name": "t"}]}}
        result = audit.validate_plan_gate(good)
        self.assertTrue(result["passed"])
        bad = {"Plan": {"Node Type": "Index Scan", "Index Name": "hnsw_idx"}}
        with self.assertRaisesRegex(RuntimeError, "index access path"):
            audit.validate_plan_gate(bad)
        with self.assertRaisesRegex(RuntimeError, "no node types"):
            audit.validate_plan_gate({"Plan": {"Plans": []}})

    def test_non_tie_requires_exact_ids_and_tie_allows_boundary_substitution(self) -> None:
        non_tie = audit.compare_truth_cell(truth_row(), [(1, 1.0, True, True), (2, 2.0, True, True), (99, 4.0, True, True)])
        self.assertTrue(non_tie["passed"])
        wrong = audit.compare_truth_cell(truth_row(), [(1, 1.0), (9, 2.0), (99, 4.0)])
        self.assertFalse(wrong["passed"])
        tied = truth_row(
            exact_topk_plus_one_ids="1,2,9",
            exact_topk_plus_one_distances_sq="1.0,2.0,2.0005",
            boundary_tied="True",
        )
        allowed = audit.compare_truth_cell(tied, [(1, 1.0, True, True), (9, 2.0005, True, True), (99, 4.0, True, True)])
        self.assertTrue(allowed["tie_cell"])
        self.assertTrue(allowed["passed"])
        too_far = audit.compare_truth_cell(tied, [(1, 1.0), (9, 2.01), (99, 4.0)])
        self.assertFalse(too_far["passed"])

    def test_manifest_validation_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing required proof"):
            audit.validate_audit_manifest({"artifact_valid": True})
        manifest = {
            "artifact_valid": True, "expected_cells": 1, "completed_cells": 1, "passed_cells": 1,
            "source_hashes": {"truth_csv_sha256": "0" * 64, "filters_csv_sha256": "1" * 64},
            "database": {
                "table_oid": 1,
                "table_relfilenode": 2,
                "server_version": "16",
                "vector_version": "0.8",
                "vector_binary_path": "/lib/vector.so",
                "vector_binary_sha256": "2" * 64,
            },
            "plan_contract": {"forbidden_access": ["Index Scan"]},
            "outputs": {"per_cell_csv_sha256": "a"},
        }
        with self.assertRaisesRegex(RuntimeError, "failed cell"):
            audit.validate_audit_manifest(manifest, [{"passed": False}])

    def test_truth_manifest_and_filter_hashes_and_coverage_are_required(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            filters = directory / "filters.csv"
            truth = directory / "truth.csv"
            manifest = directory / "truth-manifest.json"
            with filters.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["filter_name", "predicate"])
                writer.writeheader()
                writer.writerow({"filter_name": "f", "predicate": "rating >= 4"})
            with truth.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(truth_row()))
                writer.writeheader()
                writer.writerow(truth_row())
            manifest.write_text(json.dumps({
                "artifact_valid": True,
                "inputs": {"filters_csv": {"sha256": hashlib.sha256(filters.read_bytes()).hexdigest()}},
                "outputs": {"truth_csv_sha256": hashlib.sha256(truth.read_bytes()).hexdigest()},
            }))
            cells, proof = audit.load_truth_inputs(truth, manifest, filters, (0,))
            self.assertEqual(len(cells), 1)
            self.assertEqual(proof["expected_cells"], 1)
            manifest.write_text(json.dumps({"artifact_valid": False, "outputs": {"truth_csv_sha256": hashlib.sha256(truth.read_bytes()).hexdigest()}}))
            with self.assertRaisesRegex(ValueError, "artifact_valid"):
                audit.load_truth_inputs(truth, manifest, filters, (0,))

    @patch.dict("sys.modules", {"psycopg": None})
    def test_import_and_contract_do_not_connect(self) -> None:
        cursor = MagicMock()
        sql = audit.build_exact_sql("public.items", "rating >= 4", 10)
        self.assertIn("vector_l2_squared_distance", sql)
        self.assertIn("candidate_valid", sql)
        self.assertNotIn("query_row", sql)
        self.assertIn("v.id <> %s", sql)
        self.assertNotIn("hnsw", sql.lower())
        self.assertEqual(cursor.method_calls, [])


if __name__ == "__main__":
    import unittest

    unittest.main()
