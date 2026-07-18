from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/hybrid_vector_db/scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "prepare_weaviate_amazon10m_formal.py"
SPEC = importlib.util.spec_from_file_location("prepare_weaviate_amazon10m_formal", SCRIPT)
formal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = formal
SPEC.loader.exec_module(formal)


class PrepareWeaviateAmazon10MFormalTests(unittest.TestCase):
    def test_uuid_is_deterministic_and_row_scoped(self) -> None:
        self.assertEqual(formal.object_uuid(17), formal.object_uuid(17))
        self.assertNotEqual(formal.object_uuid(17), formal.object_uuid(18))
        with self.assertRaises(ValueError):
            formal.object_uuid(-1)
        with self.assertRaises(ValueError):
            formal.object_uuid(formal.EXPECTED_ROWS)

    def test_schema_gate_requires_exact_formal_hnsw_and_property_set(self) -> None:
        schema = formal.expected_schema()
        self.assertEqual(formal.verify_formal_schema(schema, schema), schema)

        wrong_distance = deepcopy(schema)
        wrong_distance["vectorIndexConfig"]["distance"] = "cosine"
        with self.assertRaisesRegex(formal.PreparationError, "distance"):
            formal.verify_formal_schema(wrong_distance, schema)

        missing_valid = deepcopy(schema)
        missing_valid["properties"] = [
            prop for prop in missing_valid["properties"] if prop["name"] != "embedding_valid"
        ]
        with self.assertRaisesRegex(formal.PreparationError, "embedding_valid|property set"):
            formal.verify_formal_schema(missing_valid, schema)

        extra = deepcopy(schema)
        extra["properties"].append(
            {"name": "unexpected", "dataType": ["int"], "indexFilterable": True}
        )
        with self.assertRaisesRegex(formal.PreparationError, "property set"):
            formal.verify_formal_schema(extra, schema)

        wrong_hnsw = deepcopy(schema)
        wrong_hnsw["vectorIndexConfig"]["maxConnections"] = 16
        with self.assertRaisesRegex(formal.PreparationError, "maxConnections"):
            formal.verify_formal_schema(wrong_hnsw, schema)

    def test_row_object_includes_embedding_valid_and_rejects_nonfinite(self) -> None:
        row = {
            "rating": "5.0",
            "verified_purchase": "True",
            "helpful_vote": "2",
            "review_text_len": "100",
            "main_category": "Grocery",
            "price": "9.5",
            "has_price": "True",
            "item_rating_number": "1200",
        }
        valid = formal.row_to_object(4, row, np.array([0.0, 1.0], dtype=np.float32))
        invalid = formal.row_to_object(5, row, np.zeros(2, dtype=np.float32))
        self.assertIs(valid["properties"]["embedding_valid"], True)
        self.assertIs(invalid["properties"]["embedding_valid"], False)
        self.assertEqual(valid["id"], formal.object_uuid(4))
        with self.assertRaisesRegex(formal.PreparationError, "invalid vector"):
            formal.row_to_object(6, row, np.array([0.0, np.nan], dtype=np.float32))

    def test_batch_response_checks_every_object(self) -> None:
        ids = [formal.object_uuid(0), formal.object_uuid(1)]
        response = [
            {"id": ids[0], "result": {"status": "SUCCESS"}},
            {"id": ids[1], "result": {"status": "SUCCESS"}},
        ]
        self.assertEqual(formal.validate_batch_response(response, ids)["object_count"], 2)

        failed = deepcopy(response)
        failed[1]["result"] = {
            "status": "FAILED",
            "errors": {"error": [{"message": "bad vector"}]},
        }
        with self.assertRaisesRegex(formal.PreparationError, "bad vector"):
            formal.validate_batch_response(failed, ids)
        with self.assertRaisesRegex(formal.PreparationError, "ID set mismatch"):
            formal.validate_batch_response(response[:1], ids)

    def test_batch_retry_reconciles_partial_commit_by_uuid_and_validity(self) -> None:
        row = {
            "rating": "5.0",
            "verified_purchase": "False",
            "helpful_vote": "0",
            "review_text_len": "10",
            "main_category": "Grocery",
            "price": "",
            "has_price": "False",
            "item_rating_number": "1",
        }
        objects = [
            formal.row_to_object(i, row, np.array([float(i + 1)], dtype=np.float32))
            for i in range(3)
        ]
        calls: list[list[int]] = []

        def send(values):
            calls.append([int(value["properties"]["row_id"]) for value in values])
            if len(calls) == 1:
                raise TimeoutError("ambiguous commit")
            return [
                {"id": value["id"], "result": {"status": "SUCCESS"}}
                for value in values
            ]

        def inspect(start, end):
            self.assertEqual((start, end), (0, 3))
            return {
                0: {
                    "id": objects[0]["id"],
                    "embedding_valid": True,
                }
            }

        result = formal.import_batch_with_retry(
            objects,
            send=send,
            inspect=inspect,
            max_retries=2,
            backoff_seconds=0,
        )
        self.assertEqual(calls, [[0, 1, 2], [1, 2]])
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.recovered_objects, 1)

    def test_resume_reconciliation_can_send_a_noncontiguous_missing_subset(self) -> None:
        row = {
            "rating": "5.0",
            "verified_purchase": "False",
            "helpful_vote": "0",
            "review_text_len": "10",
            "main_category": "Grocery",
            "price": "",
            "has_price": "False",
            "item_rating_number": "1",
        }
        objects = [
            formal.row_to_object(i, row, np.array([1.0], dtype=np.float32))
            for i in range(4)
        ]
        sent = []

        def inspect(start, end):
            self.assertEqual((start, end), (0, 4))
            return {
                row_id: {
                    "id": objects[row_id]["id"],
                    "embedding_valid": True,
                }
                for row_id in (0, 2)
            }

        def send(values):
            sent.append([value["properties"]["row_id"] for value in values])
            return [
                {"id": value["id"], "result": {"status": "SUCCESS"}}
                for value in values
            ]

        result = formal.import_batch_with_retry(
            objects,
            send=send,
            inspect=inspect,
            max_retries=1,
            backoff_seconds=0,
            reconcile_before_first_attempt=True,
        )
        self.assertEqual(sent, [[1, 3]])
        self.assertEqual(result.recovered_objects, 2)

    def test_checkpoint_resume_requires_same_spec_and_contiguous_ranges(self) -> None:
        spec = {"input": {"sha256": "a" * 64}, "row_range": [0, 10]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            checkpoint = formal.initialize_or_resume_checkpoint(path, spec, resume=False)
            checkpoint["completed_batches"] = [
                {
                    "batch_no": 0,
                    "row_range": [0, 5],
                    "object_count": 5,
                    "status": "acknowledged",
                    "payload_sha256": "1" * 64,
                    "response_sha256": "2" * 64,
                    "acknowledged_ids_sha256": "3" * 64,
                },
                {
                    "batch_no": 1,
                    "row_range": [5, 10],
                    "object_count": 5,
                    "status": "acknowledged",
                    "payload_sha256": "4" * 64,
                    "response_sha256": "5" * 64,
                    "acknowledged_ids_sha256": "6" * 64,
                },
            ]
            self.assertEqual(formal.completed_end(checkpoint["completed_batches"], 0, 10), 10)
            formal.atomic_write_json(path, checkpoint)
            resumed = formal.initialize_or_resume_checkpoint(path, spec, resume=True)
            self.assertEqual(resumed["specification"], spec)
            with self.assertRaisesRegex(formal.PreparationError, "provenance"):
                formal.initialize_or_resume_checkpoint(
                    path, {"input": {"sha256": "b" * 64}}, resume=True
                )

            broken = deepcopy(checkpoint["completed_batches"])
            broken[1]["row_range"] = [6, 10]
            with self.assertRaisesRegex(formal.PreparationError, "contiguous"):
                formal.completed_end(broken, 0, 10)

    def test_completion_gates_require_total_valid_and_all_14_filter_counts(self) -> None:
        filters = formal.baseline.load_filter_specs()
        values = [formal.EXPECTED_ROWS, formal.EXPECTED_VALID_ROWS] + [
            spec.expected_rows for spec in filters
        ]
        with mock.patch.object(
            formal, "query_count", side_effect=[(value, 0) for value in values]
        ) as query_count:
            gates = formal.completion_gates(
                "http://unused", filters, timeout=1, retries=0
            )
        self.assertTrue(gates["passed"])
        self.assertEqual(len(gates["filter_counts"]), 14)
        self.assertEqual(query_count.call_count, 16)

        bad_values = values[:]
        bad_values[-1] -= 1
        with mock.patch.object(
            formal, "query_count", side_effect=[(value, 0) for value in bad_values]
        ):
            with self.assertRaisesRegex(formal.PreparationError, "grocery_long500"):
                formal.completion_gates(
                    "http://unused", filters, timeout=1, retries=0
                )

    def test_image_digest_is_mandatory_and_immutable(self) -> None:
        digest = "semitechnologies/weaviate@sha256:" + "a" * 64
        identity = formal.validate_service_identity(
            {"version": "1.38.0"}, "1.38.0", digest
        )
        self.assertEqual(identity["service_image_digest"], digest)
        with self.assertRaisesRegex(formal.PreparationError, "digest"):
            formal.validate_image_digest("semitechnologies/weaviate:1.38.0")
        with self.assertRaisesRegex(formal.PreparationError, "version mismatch"):
            formal.validate_service_identity(
                {"version": "1.37.0"}, "1.38.0", digest
            )

    def test_atomic_json_never_leaves_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            formal.atomic_write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
