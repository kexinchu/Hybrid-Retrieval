from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import amazon10m_sql_native_exact_truth as truth  # noqa: E402


class _FlatL2Index:
    """Small Faiss API stand-in that makes equal-distance result order arbitrary."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32).copy()

    def search(self, queries: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        differences = self.vectors[None, :, :] - queries[:, None, :]
        distances = np.einsum("qcd,qcd->qc", differences, differences, dtype=np.float32)
        local_positions = np.arange(self.ntotal, dtype=np.int64)
        # Reverse local order for ties to prove production code imposes global-ID order.
        ordering = np.asarray(
            [np.lexsort((-local_positions, row))[:count] for row in distances], dtype=np.int64
        )
        return np.take_along_axis(distances, ordering, axis=1), ordering


class _FakeFaiss:
    __version__ = "test"
    IndexFlatL2 = _FlatL2Index

    @staticmethod
    def omp_set_num_threads(_count: int) -> None:
        pass


class Amazon10MSqlNativeExactTruthTests(unittest.TestCase):
    def test_candidate_sql_preserves_real_join_acl_rls_and_temporal_contracts(self) -> None:
        sql_by_workload = {
            workload.name: truth.build_candidate_sql("public.vectors", "rating = 5", workload)
            for workload in truth.WORKLOADS
        }
        self.assertNotIn("LIMIT", sql_by_workload["acl_only"].upper())
        self.assertIn("CURRENT_USER", sql_by_workload["acl_only"])
        self.assertIn("v.rating = 5", sql_by_workload["acl_only"])
        self.assertIn("v.embedding_valid", sql_by_workload["acl_only"])
        self.assertNotIn("valid_from <= %(as_of)s", sql_by_workload["acl_only"])
        self.assertIn("grant_row.valid_from <= %(as_of)s", sql_by_workload["grant_temporal_selectivity"])
        self.assertIn("fact.valid_from <= %(as_of)s", sql_by_workload["fact_temporal_selectivity"])
        for sql_text in sql_by_workload.values():
            truth.validate_candidate_sql(sql_text)
            self.assertNotIn("hnsw", sql_text.lower())
        spot_sql = truth.build_spot_check_sql(
            "public.vectors", "rating = 5", truth.WORKLOADS[0], "embedding_valid"
        )
        self.assertIn("v.embedding_valid", spot_sql)
        self.assertIn("query_row.embedding_valid", spot_sql)

    def test_candidate_validity_predicate_accepts_only_preregistered_universe(self) -> None:
        self.assertEqual(
            truth.validate_candidate_validity_predicate(" embedding_valid "),
            "embedding_valid",
        )
        for predicate in (
            "",
            "embedding_valid; SELECT 1",
            "embedding_valid -- x",
            "/*x*/ true",
            "embedding_valid) OR true OR (embedding_valid",
        ):
            with self.subTest(predicate=predicate):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError, "empty|must be exactly"
                ):
                    truth.validate_candidate_validity_predicate(predicate)

    def test_candidate_and_workload_predicates_have_separate_hash_contracts(self) -> None:
        args = truth.create_argument_parser().parse_args([])
        run_spec = truth.build_run_spec(
            args,
            [truth.FilterSpec("f", "1%", "rating = 5", 10, 1.0)],
            {0: 10},
            {"script": "s"},
            {"backend": "faiss"},
            {"checked_rows": 1},
        )
        self.assertEqual(run_spec["candidate_universe"]["predicate"], "embedding_valid")
        self.assertEqual(
            run_spec["candidate_universe"]["predicate_sha256"],
            truth.candidate_universe_predicate_sha256("embedding_valid"),
        )
        self.assertEqual(
            run_spec["workload_scalar_predicates"][0]["predicate_sha256"],
            truth.workload_scalar_predicate_sha256("rating = 5"),
        )
        self.assertNotEqual(
            run_spec["candidate_universe"]["predicate_sha256"],
            run_spec["workload_scalar_predicates"][0]["predicate_sha256"],
        )

    def test_chunked_topk_is_direct_float32_tie_sorted_and_self_excluded(self) -> None:
        vectors = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        )
        ids, distances, _ = truth.exact_topk_batched(
            vectors, np.asarray([1, 4]), np.asarray([0, 1, 2, 3, 4, 5]),
            k=2, chunk_rows=2, query_batch_size=1,
        )
        self.assertEqual(ids[0].tolist(), [0, 2, 3])
        self.assertTrue(np.allclose(distances[0], [0.0, 0.0, 1.0]))
        self.assertNotIn(1, ids[0].tolist())
        self.assertEqual(ids[1].tolist(), [3, 5, 0])
        meta = truth.truth_metadata(distances[0], 2)
        self.assertTrue(meta["boundary_tied"] is False)
        self.assertEqual(meta["strict_closer_count"], 0)

    def test_faiss_flat_exact_matches_numpy_with_self_ties_sparse_ids_and_k_plus_one(self) -> None:
        vectors = np.zeros((101, 2), dtype=np.float32)
        vectors[73] = [1.0, 0.0]
        vectors[100] = [1.0, 0.0]
        candidate_ids = np.asarray([100, 50, 73, 9, 3], dtype=np.int64)
        query_ids = np.asarray([50, 25], dtype=np.int64)

        expected_ids, expected_distances, _ = truth.exact_topk_batched(
            vectors, query_ids, candidate_ids, k=2, chunk_rows=2, query_batch_size=1
        )
        actual_ids, actual_distances, metadata = truth.exact_topk_faiss(
            vectors, query_ids, candidate_ids, k=2, faiss_module=_FakeFaiss, faiss_threads=1
        )

        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(actual_distances, expected_distances)
        self.assertEqual(actual_ids[0].tolist(), [3, 9, 73])
        self.assertNotIn(50, actual_ids[0].tolist())
        self.assertEqual(actual_ids[1].tolist(), [3, 9, 50])
        self.assertEqual(metadata["class"], "IndexFlatL2")
        self.assertEqual(metadata["index_ntotal"], 5)
        self.assertEqual(metadata["maximum_requested_rows"], 5)
        self.assertTrue(metadata["local_positions_mapped_to_global_ids"])
        self.assertTrue(metadata["exact"])

    def test_faiss_flat_fails_when_candidate_pool_cannot_supply_k_plus_one_nonself(self) -> None:
        vectors = np.zeros((4, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"k\+1 non-self"):
            truth.exact_topk_faiss(
                vectors, np.asarray([0]), np.asarray([0, 1]), k=2,
                faiss_module=_FakeFaiss, faiss_threads=1,
            )

    def test_calibration_final_split_is_disjoint_and_fixed(self) -> None:
        self.assertEqual([truth.query_split(query_no, 100) for query_no in (0, 99, 100, 199)],
                         ["calibration", "calibration", "final", "final"])
        self.assertEqual(truth.select_spot_query_nos(dict(enumerate(range(200))), 2), [0, 199])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.csv"
            path.write_text(
                "query_no,query_id,query_split,candidate_validity_predicate,query_validity_predicate\n"
                "0,10,calibration,embedding_valid,embedding_valid\n"
                "1,11,calibration,embedding_valid,embedding_valid\n"
                "2,10,final,embedding_valid,embedding_valid\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                truth.load_query_ids(path, 2, 1)

    def test_base_table_mapping_sampler_spans_fbin_and_includes_all_query_ids(self) -> None:
        self.assertEqual(truth.deterministic_base_table_sample_ids(10, 4), [0, 3, 6, 9])
        self.assertEqual(truth.deterministic_base_table_sample_ids(3, 10), [0, 1, 2])
        base_ids, checked_ids = truth.base_table_mapping_ids(10, 4, [8, 1, 8])
        self.assertEqual(base_ids, [0, 3, 6, 9])
        self.assertEqual(checked_ids, [0, 1, 3, 6, 8, 9])
        with self.assertRaisesRegex(RuntimeError, "outside fbin"):
            truth.base_table_mapping_ids(10, 4, [10])

    def test_base_table_mapping_mismatch_fails_closed(self) -> None:
        vectors = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "mismatch at id=2"):
            truth.validate_vector_mapping(
                vectors, [0, 2], {
                    0: np.asarray([0.0, 0.0], dtype=np.float32),
                    2: np.asarray([2.0, 2.1], dtype=np.float32),
                },
            )

    def test_query_candidate_universe_proof_rejects_invalid_query_row(self) -> None:
        cursor = mock.MagicMock()
        cursor.fetchall.return_value = [(10,)]
        with self.assertRaisesRegex(RuntimeError, "candidate-validity universe"):
            truth.verify_query_candidate_universe(
                cursor,
                "public.items",
                {0: 10, 1: 11},
                "embedding_valid",
            )
        self.assertIn("query_row.embedding_valid", cursor.execute.call_args.args[0])

    def test_truth_cohort_rejects_unrelated_filter_provenance(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            path.write_text(
                "query_no,query_id,query_split,filter_name,predicate,method,self_excluded,"
                "kth_distance_sq,candidate_validity_predicate,query_validity_predicate\n"
                "0,10,calibration,unrelated,rating = 1,pre_filter_exact,true,1.0,"
                "embedding_valid,embedding_valid\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "filter names/predicates"):
                truth.load_query_cohort(
                    path,
                    {0: "calibration"},
                    "embedding_valid",
                    source_manifest_path=Path(tmp) / "manifest.json",
                    expected_filters=[
                        truth.FilterSpec("expected", "1%", "rating = 5", 1, 1.0)
                    ],
                )

    def test_checkpoint_rejects_stale_run_or_source_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.json"
            truth.write_pair_checkpoint(path, "run-a", {"script": "old"}, {"rows": []})
            self.assertEqual(truth.load_pair_checkpoint(path, "run-a", {"script": "old"})["rows"], [])
            with self.assertRaisesRegex(RuntimeError, "run-spec"):
                truth.load_pair_checkpoint(path, "run-b", {"script": "old"})
            with self.assertRaisesRegex(RuntimeError, "source-hash"):
                truth.load_pair_checkpoint(path, "run-a", {"script": "new"})

    def test_run_spec_hash_binds_exact_backend_and_threads(self) -> None:
        args = truth.create_argument_parser().parse_args([])
        two_threads = truth.create_argument_parser().parse_args(["--faiss-threads", "2"])
        source_hashes = {"script": "source"}
        base_table_mapping = {
            "base_sample_ids": [0, 9], "base_sample_ids_sha256": "base",
            "checked_ids": [0, 9], "checked_ids_sha256": "checked", "max_abs_error": 0.0,
        }
        faiss_one = truth.build_run_spec(args, [], {}, source_hashes, {
            "backend": "faiss", "class": "IndexFlatL2", "faiss_version": "1.14.2",
            "threads": 1, "exact": True, "formal_default": True,
        }, base_table_mapping)
        faiss_two = truth.build_run_spec(two_threads, [], {}, source_hashes, {
            "backend": "faiss", "class": "IndexFlatL2", "faiss_version": "1.14.2",
            "threads": 2, "exact": True, "formal_default": True,
        }, base_table_mapping)
        self.assertEqual(faiss_one["faiss_threads"], 1)
        self.assertEqual(faiss_one["base_table_mapping"]["checked_ids_sha256"], "checked")
        self.assertNotEqual(truth.canonical_sha256(faiss_one), truth.canonical_sha256(faiss_two))

    def test_formal_dimensions_require_14_filters_q100_q100_and_faiss(self) -> None:
        args = truth.create_argument_parser().parse_args([])
        filters = truth.read_filters(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "amazon10m_selectivity14_filters.csv"
        )
        truth.validate_formal_dimensions(args, filters)
        args.backend = "numpy"
        with self.assertRaisesRegex(RuntimeError, "Faiss"):
            truth.validate_formal_dimensions(args, filters)

    def test_spot_check_compares_every_strict_rank_and_accepts_tied_substitution(self) -> None:
        valid = truth.validate_spot_check(
            [10, 11, 12], [0.0, 1.0, 1.0], [(10, 0.0), (12, 1.0), (11, 1.0)], k=2
        )
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["tie_positions"], [1, 2])
        with self.assertRaisesRegex(RuntimeError, "rank=0"):
            truth.validate_spot_check([10, 11, 12], [0.0, 1.0, 2.0], [(99, 0.0), (11, 1.0), (12, 2.0)], k=2)
        with self.assertRaisesRegex(RuntimeError, "wrong distance"):
            truth.validate_spot_check(
                [10, 11, 12], [0.0, 1.0, 1.0], [(10, 0.0), (12, 0.5), (11, 1.0)], k=2
            )

    def test_dry_run_does_not_read_inputs(self) -> None:
        missing = Path("/definitely/missing")
        self.assertEqual(truth.main(["--dry-run", "--fbin", str(missing), "--filters-csv", str(missing)]), 0)

    def test_workloads_are_acl_grant_temporal_and_fact_temporal_without_rls_collapse(self) -> None:
        self.assertEqual(
            [workload.name for workload in truth.WORKLOADS],
            ["acl_only", "grant_temporal_selectivity", "fact_temporal_selectivity"],
        )
        sql_by_workload = {
            workload.name: truth.build_candidate_sql("public.vectors", "rating = 5", workload)
            for workload in truth.WORKLOADS
        }
        self.assertNotIn("valid_from <= %(as_of)s", sql_by_workload["acl_only"])
        self.assertIn(
            "grant_row.valid_from <= %(as_of)s",
            sql_by_workload["grant_temporal_selectivity"],
        )
        self.assertNotIn(
            "fact.valid_from <= %(as_of)s",
            sql_by_workload["grant_temporal_selectivity"],
        )
        self.assertIn(
            "fact.valid_from <= %(as_of)s",
            sql_by_workload["fact_temporal_selectivity"],
        )
        self.assertNotIn(
            "grant_row.valid_from <= %(as_of)s",
            sql_by_workload["fact_temporal_selectivity"],
        )

    def test_formal_guard_locks_all_data_relations_and_rechecks_same_version(self) -> None:
        cursor = mock.MagicMock()
        with (
            mock.patch.object(
                truth,
                "fingerprint_relations",
                side_effect=[{"t": {"oid": 1}}, {"t": {"oid": 1}}],
            ),
            mock.patch.object(truth, "session_context", return_value={"snapshot": "1:2:"}),
        ):
            guard = truth.acquire_formal_data_guard(cursor, "public.t")
            released = truth.release_formal_data_guard(cursor, "public.t", guard)
        statements = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertEqual(statements[0], "BEGIN ISOLATION LEVEL REPEATABLE READ")
        self.assertIn("LOCK TABLE", statements[1])
        self.assertIn("public.amazon_review_facts", statements[1])
        self.assertTrue(released["valid"])
        self.assertEqual(released["start_hash"], released["end_hash"])
        self.assertEqual(statements[-1], "COMMIT")

        changed = mock.MagicMock()
        with (
            mock.patch.object(
                truth,
                "fingerprint_relations",
                side_effect=[{"t": {"oid": 1}}, {"t": {"oid": 2}}],
            ),
            mock.patch.object(truth, "session_context", return_value={"snapshot": "1:2:"}),
        ):
            guard = truth.acquire_formal_data_guard(changed, "public.t")
            with self.assertRaisesRegex(RuntimeError, "data version changed"):
                truth.release_formal_data_guard(changed, "public.t", guard)
        self.assertEqual(changed.execute.call_args.args[0], "ROLLBACK")

    def test_shared_fingerprint_contains_columns_policies_and_epoch(self) -> None:
        cursor = mock.MagicMock()
        cursor.fetchone.side_effect = [
            (1, 2, 3, 4, "r", True, False, 9),
        ]
        cursor.fetchall.side_effect = [
            [("id", "bigint", True)],
            [("policy", "{reader}", "SELECT", "qual", None)],
            [
                (
                    "amazon_sql_native_epoch_bump",
                    "O",
                    "public.amazon_sql_native_bump_relation_epoch()",
                    "INSERT INTO public.amazon_sql_native_relation_epoch; DO UPDATE SET epoch = epoch + 1",
                    "CREATE TRIGGER amazon_sql_native_epoch_bump AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON t FOR EACH STATEMENT EXECUTE FUNCTION amazon_sql_native_bump_relation_epoch()",
                )
            ],
        ]
        fingerprint = truth.relation_fingerprint(cursor, "public.t")
        self.assertEqual(fingerprint["columns"], [["id", "bigint", True]])
        self.assertEqual(fingerprint["policies"][0][0], "policy")
        self.assertEqual(fingerprint["data_epoch"], 9)
        self.assertEqual(fingerprint["triggers"][0][0], "amazon_sql_native_epoch_bump")

    def test_epoch_trigger_proof_rejects_disabled_or_wrong_function(self) -> None:
        valid = (
            "amazon_sql_native_epoch_bump",
            "O",
            "public.amazon_sql_native_bump_relation_epoch()",
            "INSERT INTO public.amazon_sql_native_relation_epoch VALUES (1); "
            "ON CONFLICT DO UPDATE SET epoch = epoch + 1",
            "CREATE TRIGGER amazon_sql_native_epoch_bump AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE "
            "ON t FOR EACH STATEMENT EXECUTE FUNCTION amazon_sql_native_bump_relation_epoch()",
        )
        self.assertTrue(truth.valid_epoch_trigger(valid))
        self.assertFalse(truth.valid_epoch_trigger((valid[0], "D", *valid[2:])))
        self.assertFalse(
            truth.valid_epoch_trigger((valid[0], valid[1], "public.noop()", *valid[3:]))
        )

    def test_manifest_builder_binds_data_version_proof(self) -> None:
        relations = {
            "public.amazon_review_facts": {"policies": [], "data_epoch": 7}
        }
        query_ids = {0: 10}
        query_splits = {0: "calibration"}
        cohort_hash = truth.query_cohort_sha256(query_ids, query_splits)
        candidate_universe = {
            "predicate": "embedding_valid",
            "predicate_sha256": truth.candidate_universe_predicate_sha256(
                "embedding_valid"
            ),
        }
        manifest = truth.build_artifact_manifest(
            run_spec={
                "a": 1,
                "principal": "principal",
                "query_ids": query_ids,
                "query_splits": query_splits,
                "query_cohort_sha256": cohort_hash,
                "query_cohort": {"query_cohort_sha256": cohort_hash},
                "candidate_universe": candidate_universe,
            },
            source_hashes={"script": "s"},
            fbin={"path": "f"},
            base_table_mapping={"checked_rows": 1},
            outputs={"truth_csv_sha256": "h"},
            backend={"exact": True},
            pairs=[],
            data_version_proof={
                "valid": True,
                "start_hash": truth.canonical_sha256(relations),
                "end_hash": truth.canonical_sha256(relations),
                "start_relations": relations, "end_relations": relations,
            },
            rls_security_proof={
                "current_user": "principal", "is_superuser": False,
                "bypass_rls": False, "owns_facts": False,
                "reader_membership": True, "rls_enabled": True,
                "policy_hash": truth.canonical_sha256([]),
                "positive_probe_visible": True, "negative_probe_hidden": True,
            },
        )
        self.assertTrue(manifest["artifact_valid"])
        self.assertEqual(
            manifest["data_version_proof"]["start_hash"],
            truth.canonical_sha256(relations),
        )
        self.assertEqual(manifest["query_cohort_sha256"], cohort_hash)
        self.assertEqual(manifest["relation_epoch"]["relations"], {
            "public.amazon_review_facts": 7
        })

    def test_exact_artifact_publication_fails_closed_before_writing_csv(self) -> None:
        rows = [{"workload": "acl_only", "id": 1}]
        digest = truth.sha256_text(truth.render_csv(rows))
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "truth.csv"
            manifest_path = Path(tmpdir) / "manifest.json"
            with self.assertRaisesRegex(RuntimeError, "artifact_valid"):
                truth.publish_exact_artifact(
                    out,
                    manifest_path,
                    rows,
                    {"artifact_valid": False, "outputs": {"truth_csv_sha256": digest}},
                )
            self.assertFalse(out.exists())
            self.assertFalse(manifest_path.exists())

            truth.publish_exact_artifact(
                out,
                manifest_path,
                rows,
                {"artifact_valid": True, "outputs": {"truth_csv_sha256": digest}},
            )
            self.assertEqual(truth.sha256_file(out), digest)
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
