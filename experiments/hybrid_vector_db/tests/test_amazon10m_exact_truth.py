from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import amazon10m_exact_truth as truth  # noqa: E402


class AmazonExactTruthTest(unittest.TestCase):
    def test_validity_predicates_reject_statement_and_comment_markers(self) -> None:
        self.assertEqual(
            truth.safe_sql_predicate(" embedding_valid AND price IS NOT NULL "),
            "embedding_valid AND price IS NOT NULL",
        )
        for unsafe in ("TRUE; DROP TABLE reviews", "TRUE -- bypass", "TRUE /* bypass */", ""):
            with self.subTest(unsafe=unsafe), self.assertRaises(argparse.ArgumentTypeError):
                truth.safe_sql_predicate(unsafe)

    def test_query_validity_defaults_to_candidate_contract(self) -> None:
        candidate = "embedding_valid IS TRUE"
        self.assertEqual(truth.resolve_query_validity_predicate(candidate, None), candidate)
        self.assertEqual(
            truth.resolve_query_validity_predicate(candidate, "review_year >= 2020"),
            "review_year >= 2020",
        )

    def test_formal_validity_default_and_explicit_true_have_provenance(self) -> None:
        predicate, provenance = truth.resolve_candidate_validity_predicate(None)
        self.assertEqual(predicate, "embedding_valid")
        self.assertEqual(provenance, "formal_default_embedding_valid")
        self.assertEqual(
            truth.resolve_candidate_validity_predicate("TRUE"),
            ("TRUE", "explicit_cli_predicate"),
        )

    def test_batched_topk_matches_bruteforce(self) -> None:
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(200, 8)).astype("<f4")
        query_ids = np.asarray([2, 17, 91], dtype=np.int64)
        candidate_ids = np.asarray([row for row in range(len(vectors)) if row % 3], dtype=np.int64)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.fbin"
            with path.open("wb") as f:
                np.asarray(vectors.shape, dtype="<i4").tofile(f)
                vectors.tofile(f)
            mapped = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=vectors.shape)
            actual, distances, _ = truth.exact_topk_batch(
                mapped,
                query_ids,
                candidate_ids,
                k=10,
                chunk_rows=37,
                progress_chunks=0,
                filter_name="test",
            )

        for position, query_id in enumerate(query_ids):
            distances = np.sum((vectors[candidate_ids] - vectors[query_id]) ** 2, axis=1)
            distances[candidate_ids == query_id] = np.inf
            expected_pos = np.lexsort((candidate_ids, distances))[:10]
            self.assertEqual(actual[position][:10], candidate_ids[expected_pos].tolist())

    def test_exact_topk_excludes_self_and_breaks_ties_by_id(self) -> None:
        vectors = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype="<f4",
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.fbin"
            with path.open("wb") as f:
                np.asarray(vectors.shape, dtype="<i4").tofile(f)
                vectors.tofile(f)
            mapped = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=vectors.shape)
            ids, distances, _ = truth.exact_topk_batch(
                mapped,
                np.asarray([1], dtype=np.int64),
                np.asarray([4, 2, 1, 0, 3], dtype=np.int64),
                k=2,
                chunk_rows=2,
                progress_chunks=0,
                filter_name="ties",
            )

        self.assertEqual(ids[0][:2], [0, 2])
        self.assertEqual(distances[0][:2], [0.0, 0.0])
        self.assertNotIn(1, ids[0])

    def test_exact_topk_uses_float64_direct_cdist_per_chunk(self) -> None:
        vectors = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype="<f4")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.fbin"
            with path.open("wb") as f:
                np.asarray(vectors.shape, dtype="<i4").tofile(f)
                vectors.tofile(f)
            mapped = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=vectors.shape)
            from scipy.spatial.distance import cdist

            with mock.patch("scipy.spatial.distance.cdist", wraps=cdist) as distance_mock:
                truth.exact_topk_batch(
                    mapped,
                    np.asarray([0], dtype=np.int64),
                    np.asarray([1, 2], dtype=np.int64),
                    k=1,
                    chunk_rows=1,
                    progress_chunks=0,
                    filter_name="direct",
                )

        self.assertEqual(distance_mock.call_count, 2)
        first_call = distance_mock.call_args_list[0]
        self.assertEqual(first_call.kwargs["metric"], "sqeuclidean")
        self.assertEqual(first_call.args[0].dtype, np.dtype(np.float64))
        self.assertEqual(first_call.args[1].dtype, np.dtype(np.float64))

    def test_truth_boundary_detects_k_plus_one_tie(self) -> None:
        kth, tolerance, strict, tied = truth.truth_boundary(
            [0.0, 0.0, 1.0, 1.0], k=3
        )
        self.assertEqual(kth, 1.0)
        self.assertGreater(tolerance, 0.0)
        self.assertEqual(strict, 2)
        self.assertTrue(tied)

    def test_final_query_sampling_is_unique_and_disjoint(self) -> None:
        excluded = {1, 2, 3, 99}
        first = truth.sample_disjoint_query_ids(200, excluded, 50, seed=58)
        second = truth.sample_disjoint_query_ids(200, excluded, 50, seed=58)

        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), len(set(first)))
        self.assertFalse(set(first) & excluded)

    def test_eligible_query_sampling_is_deterministic_disjoint_and_noncontiguous(self) -> None:
        eligible = np.asarray([4, 19, 22, 105, 900, 1201, 8000, 50000], dtype=np.int64)
        calibration = truth.sample_disjoint_eligible_query_ids(eligible, set(), 4, seed=57)
        repeated = truth.sample_disjoint_eligible_query_ids(eligible, set(), 4, seed=57)
        final = truth.sample_disjoint_eligible_query_ids(
            eligible, set(int(value) for value in calibration), 3, seed=58
        )

        self.assertEqual(calibration.tolist(), repeated.tolist())
        self.assertTrue(set(calibration).issubset(set(eligible)))
        self.assertTrue(set(final).issubset(set(eligible)))
        self.assertFalse(set(calibration) & set(final))
        self.assertEqual(len(calibration), len(set(calibration)))

    def test_eligible_query_sampling_rejects_unsorted_or_duplicate_population(self) -> None:
        for invalid in (
            np.asarray([1, 3, 2], dtype=np.int64),
            np.asarray([1, 2, 2], dtype=np.int64),
        ):
            with self.subTest(ids=invalid.tolist()), self.assertRaises(ValueError):
                truth.sample_disjoint_eligible_query_ids(invalid, set(), 1, seed=1)

    def test_candidate_fetch_combines_filter_and_validity_predicates(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.sql = ""

            def execute(self, sql: str) -> None:
                self.sql = sql

            def __iter__(self):
                return iter([(9,), (31,)])

        cursor = Cursor()
        ids, _ = truth.fetch_candidate_ids(
            cursor,
            "reviews",
            "price <= 10",
            "embedding_valid IS TRUE",
        )

        self.assertEqual(
            cursor.sql,
            "SELECT id FROM reviews WHERE (price <= 10) AND (embedding_valid IS TRUE)",
        )
        self.assertEqual(ids.tolist(), [9, 31])

    def test_query_mapping_rejects_zero_fbin_vector(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.sql = ""
                self.params = None

            def execute(self, sql: str, params=None) -> None:
                self.sql = sql
                self.params = params

            def fetchall(self):
                return [(1, "[0,0]")]

        cursor = Cursor()
        vectors = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(SystemExit, "finite and nonzero"):
            truth.verify_query_vector_mapping(
                cursor,
                "reviews",
                vectors,
                np.asarray([1], dtype=np.int64),
                "embedding_valid IS TRUE",
            )
        self.assertIn("AND (embedding_valid IS TRUE)", cursor.sql)

    def test_truth_row_records_query_population_contract(self) -> None:
        row = truth.truth_row(
            0,
            19,
            {"filter_name": "cheap", "target_rate": "0.1", "predicate": "price <= 10"},
            0.09,
            90,
            [2, 3, 4],
            [0.1, 0.2, 0.3],
            1.5,
            57,
            "calibration",
            False,
            2,
            "embedding_valid",
            "embedding_valid",
            812,
            "abc123",
            "postgres_ordered_id_scan_v1:reviews",
        )

        self.assertEqual(row["candidate_validity_predicate"], "embedding_valid")
        self.assertEqual(row["query_validity_predicate"], "embedding_valid")
        self.assertEqual(row["eligible_query_population"], 812)
        self.assertEqual(row["eligible_query_ids_sha256"], "abc123")

    def test_resume_rejects_validity_contract_drift(self) -> None:
        resumed = [
            {
                "filter_name": "cheap",
                "query_no": "0",
                "query_id": "19",
                "filtered_rows": "90",
                "kth_distance_sq": "0.2",
                "tie_tolerance": "1e-9",
                "self_excluded": "true",
                "candidate_validity_predicate": "TRUE",
                "query_validity_predicate": "TRUE",
                "eligible_query_population": "1000",
                "eligible_query_ids_sha256": "old",
                "eligible_query_population_provenance": "postgres:old",
            }
        ]
        with self.assertRaisesRegex(SystemExit, "resume validity contract mismatch"):
            truth.validate_resumed_rows(
                resumed,
                np.asarray([19], dtype=np.int64),
                [{"filter_name": "cheap"}],
                "embedding_valid",
                "embedding_valid",
                812,
                "new",
                "postgres:new",
            )

    def test_resume_rejects_duplicate_query_numbers(self) -> None:
        rows = [
            {
                "filter_name": "cheap",
                "query_no": "0",
                "query_id": "19",
                "filtered_rows": "90",
                "kth_distance_sq": "0.2",
                "tie_tolerance": "1e-9",
                "self_excluded": "true",
                "candidate_validity_predicate": "embedding_valid",
                "query_validity_predicate": "embedding_valid",
            },
            {
                "filter_name": "cheap",
                "query_no": "0",
                "query_id": "23",
                "filtered_rows": "90",
                "kth_distance_sq": "0.2",
                "tie_tolerance": "1e-9",
                "self_excluded": "true",
                "candidate_validity_predicate": "embedding_valid",
                "query_validity_predicate": "embedding_valid",
            },
        ]
        with self.assertRaisesRegex(SystemExit, "query_no values are not unique"):
            truth.validate_resumed_rows(
                rows,
                np.asarray([19, 23], dtype=np.int64),
                [{"filter_name": "cheap"}],
                "embedding_valid",
                "embedding_valid",
            )

    def test_resume_checkpoint_binds_k_and_input_hashes(self) -> None:
        query_ids = np.asarray([19], dtype=np.int64)
        filters = [{"filter_name": "cheap", "target_rate": "0.1", "predicate": "price <= 10"}]
        row = truth.truth_row(
            0,
            19,
            filters[0],
            0.09,
            90,
            [2, 3],
            [0.1, 0.2],
            1.5,
            57,
            "calibration",
            False,
            2,
            "embedding_valid IS TRUE",
            "embedding_valid IS TRUE",
            812,
            "eligible-hash",
            "postgres:table",
            "formal_default_embedding_valid",
            "inherits_candidate_validity",
            "candidate-hash",
        )
        fbin = {"path": "/tmp/vectors.fbin", "sha256": "fbin-hash"}
        filters_csv = {"path": "/tmp/filters.csv", "sha256": "filters-hash"}
        table = {"name": "reviews", "rows": 100, "min_id": 0, "max_id": 99, "oid": 7, "relfilenode": 8}
        eligible = {"rows": 812, "ids_sha256": "eligible-hash", "provenance": "postgres:table"}
        checkpoint = {
            "schema_version": truth.CHECKPOINT_SCHEMA_VERSION,
            "k": 2,
            "query_count": 1,
            "query_ids_sha256": truth.ordered_ids_sha256(query_ids),
            "candidate_validity_predicate": "embedding_valid IS TRUE",
            "candidate_validity_provenance": "formal_default_embedding_valid",
            "query_validity_predicate": "embedding_valid IS TRUE",
            "query_validity_provenance": "inherits_candidate_validity",
            "fbin": fbin,
            "filters_csv": filters_csv,
            "filters": truth.filter_contract(filters),
            "table": table,
            "eligible_query_population": eligible,
            "completed_filters": {
                "cheap": {"candidate_rows": 90, "candidate_ids_sha256": "candidate-hash"}
            },
        }
        self.assertEqual(
            truth.validate_resume_checkpoint(
                checkpoint,
                [row],
                query_ids,
                filters,
                k=2,
                fbin=fbin,
                filters_csv=filters_csv,
                table_identity=table,
                eligible_population=eligible,
                candidate_validity_predicate="embedding_valid IS TRUE",
                query_validity_predicate="embedding_valid IS TRUE",
                candidate_validity_provenance="formal_default_embedding_valid",
                query_validity_provenance="inherits_candidate_validity",
            ),
            {"cheap"},
        )
        checkpoint["k"] = 3
        with self.assertRaisesRegex(SystemExit, "checkpoint contract mismatch"):
            truth.validate_resume_checkpoint(
                checkpoint,
                [row],
                query_ids,
                filters,
                k=2,
                fbin=fbin,
                filters_csv=filters_csv,
                table_identity=table,
                eligible_population=eligible,
                candidate_validity_predicate="embedding_valid IS TRUE",
                query_validity_predicate="embedding_valid IS TRUE",
                candidate_validity_provenance="formal_default_embedding_valid",
                query_validity_provenance="inherits_candidate_validity",
            )


if __name__ == "__main__":
    unittest.main()
