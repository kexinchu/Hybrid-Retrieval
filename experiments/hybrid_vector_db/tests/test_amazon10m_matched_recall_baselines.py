import csv
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from experiments.hybrid_vector_db.scripts.amazon10m_matched_recall_baselines import (
    NA,
    AllowList,
    FilterSpec,
    TruthEntry,
    aggregate_measurements,
    artifact_validation_errors,
    assert_no_hnsw_index,
    balanced_order,
    bitmap_contains,
    build_allow_list,
    calibration_table,
    build_parser,
    exact_sql,
    final_summary_table,
    load_truth,
    measurement_row,
    search_faiss,
    sha256_file,
    set_bitmap_ids,
    tie_aware_recall_at_k,
    write_csv,
)
from experiments.hybrid_vector_db.scripts.finalize_amazon10m_matched_recall_baselines import (
    FinalizationFailure,
    finalize_existing,
)


SPEC = FilterSpec(
    name="filter_a",
    target_rate="10.0%",
    predicate="helpful_vote >= 1",
    expected_rows=20,
    actual_pct=10.0,
)


def measured_row(
    phase: str,
    method: str,
    query_no: int,
    repeat: int,
    latency: float,
    recall: float,
    ef_search=100,
):
    return {
        "phase": phase,
        "method": method,
        "filter_name": SPEC.name,
        "ef_search": ef_search,
        "query_no": query_no,
        "repeat": repeat,
        "search_latency_ms": latency,
        "recall_at_10": recall,
        "valid": True,
        "error": "",
    }


class Amazon10mMatchedRecallBaselineTests(unittest.TestCase):
    def test_formal_default_table_matches_the_exact_gt_and_fbin_id_space(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.table, "amazon_grocery_reviews_10m_pgvector")
        self.assertEqual(args.calibration_query_offset, 20)
        self.assertEqual(args.calibration_queries, 80)
        self.assertEqual(args.final_query_offset, 100)
        self.assertEqual(args.final_queries, 100)

    def test_target_eligibility_uses_query_mean_and_reports_lcb_only(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", 20, 0, 1.0, 1.0, 100),
            measured_row("calibration", "faiss_allowlist", 20, 1, 1.0, 0.0, 100),
        ]
        table, selected = calibration_table(
            rows, [SPEC], [100], [0.5], [20], repeats=2,
            bootstrap_samples=100, bootstrap_seed=57,
        )
        self.assertTrue(table[0]["eligible"])
        self.assertEqual(selected[(SPEC.name, 0.5)], 100)
        self.assertIn("recall_lcb95", table[0])

    def test_exact_sql_materializes_filter_and_excludes_hnsw_plan(self):
        sql = exact_sql("samegraph_insert", SPEC.predicate, 10)

        self.assertIn("WITH filtered AS MATERIALIZED", sql)
        self.assertIn("id <> %s", sql)
        self.assertIn("ORDER BY embedding <-> %s::vector, id", sql)
        self.assertIn("LIMIT 10", sql)

        scalar_plan = {
            "Node Type": "Limit",
            "Plans": [
                {
                    "Node Type": "CTE Scan",
                    "Plans": [
                        {"Node Type": "Index Scan", "Index Name": "helpful_vote_idx"}
                    ],
                }
            ],
        }
        self.assertEqual(
            assert_no_hnsw_index(scalar_plan, ["samegraph_insert_hnsw"]),
            {"helpful_vote_idx"},
        )

        hnsw_plan = {
            "Node Type": "Index Scan",
            "Index Name": "samegraph_insert_hnsw",
        }
        with self.assertRaisesRegex(RuntimeError, "used HNSW"):
            assert_no_hnsw_index(hnsw_plan, ["public.samegraph_insert_hnsw"])

    def test_streaming_bitmap_sets_faiss_little_endian_bits(self):
        bitmap = np.zeros(3, dtype=np.uint8)

        count = set_bitmap_ids(bitmap, [0, 1, 7, 8, 19], total_rows=20)

        self.assertEqual(count, 5)
        self.assertEqual(bitmap.tolist(), [0b10000011, 0b00000001, 0b00001000])
        self.assertTrue(all(bitmap_contains(bitmap, value) for value in [0, 1, 7, 8, 19]))
        self.assertFalse(bitmap_contains(bitmap, 18))
        with self.assertRaisesRegex(ValueError, "outside"):
            set_bitmap_ids(bitmap, [20], total_rows=20)

    def test_allowlist_builder_fetches_batches_and_keeps_bitmap_backing(self):
        class FakeCursor:
            def __init__(self):
                self.batches = [[(0,), (7,)], [(8,), (19,)], []]
                self.fetchmany_calls = 0
                self.sql = ""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql):
                self.sql = sql

            def fetchmany(self, size):
                self.fetchmany_calls += 1
                self.fetch_size = size
                return self.batches.pop(0)

        class FakeConnection:
            def __init__(self):
                self.server_cursor = FakeCursor()

            def transaction(self):
                return nullcontext()

            def cursor(self, name):
                self.cursor_name = name
                return self.server_cursor

        class FakeSelector:
            def __init__(self, size, bitmap):
                self.size = size
                self.bitmap = bitmap

        class FakeFaiss:
            IDSelectorBitmap = FakeSelector

            @staticmethod
            def swig_ptr(bitmap):
                return bitmap

        conn = FakeConnection()
        small_spec = FilterSpec(SPEC.name, SPEC.target_rate, SPEC.predicate, 4, SPEC.actual_pct)

        allow_list = build_allow_list(conn, FakeFaiss, "samegraph_insert", small_spec, 20, 2)

        self.assertTrue(allow_list.valid)
        self.assertEqual(allow_list.rows, 4)
        self.assertEqual(allow_list.bitmap_bytes, 3)
        self.assertEqual(allow_list.selector.size, 20)
        self.assertIs(allow_list.selector.bitmap, allow_list.bitmap)
        self.assertEqual(conn.server_cursor.fetchmany_calls, 3)
        self.assertEqual(conn.server_cursor.fetch_size, 2)
        self.assertIn("SELECT id FROM samegraph_insert", conn.server_cursor.sql)
        self.assertTrue(bitmap_contains(allow_list.bitmap, 19))

    def test_faiss_search_passes_bitmap_selector_to_hnsw_parameters(self):
        class Params:
            efSearch = 0
            sel = None

        class FakeFaiss:
            SearchParametersHNSW = Params

        class FakeIndex:
            def search(self, query, k, params):
                self.query = query
                self.k = k
                self.params = params
                return np.zeros((1, k), dtype=np.float32), np.asarray([[4, 2, -1]])

        index = FakeIndex()
        selector = object()

        ids, latency_ms = search_faiss(
            index,
            FakeFaiss,
            np.asarray([1.0, 2.0], dtype=np.float32),
            selector,
            ef_search=500,
            k=3,
        )

        self.assertEqual(ids, [4, 2])
        self.assertGreater(latency_ms, 0.0)
        self.assertEqual(index.params.efSearch, 500)
        self.assertIs(index.params.sel, selector)
        self.assertEqual(index.query.shape, (1, 2))

    def test_faiss_search_requests_extra_row_and_excludes_query_id(self):
        class Params:
            efSearch = 0
            sel = None

        class FakeFaiss:
            SearchParametersHNSW = Params

        class FakeIndex:
            def search(self, query, k, params):
                self.k = k
                return np.zeros((1, k), dtype=np.float32), np.asarray([[7, 4, 2, 1]])

        index = FakeIndex()
        ids, _ = search_faiss(
            index,
            FakeFaiss,
            np.asarray([1.0, 2.0], dtype=np.float32),
            object(),
            ef_search=500,
            k=3,
            query_id=7,
        )
        self.assertEqual(index.k, 4)
        self.assertEqual(ids, [4, 2, 1])

    def test_tie_aware_recall_accepts_different_boundary_ids(self):
        vectors = np.asarray(
            [[0.0], [0.0], [1.0], [-1.0], [2.0]], dtype=np.float32
        )
        entry = TruthEntry(0, 0, SPEC.name, "calibration", (1, 2), 20, 1.0, 1e-9, True)
        self.assertEqual(tie_aware_recall_at_k([1, 3], 0, vectors, entry, 2), 1.0)
        self.assertEqual(tie_aware_recall_at_k([1, 4], 0, vectors, entry, 2), 0.5)

    def test_balanced_order_rotates_each_config_through_each_position(self):
        configs = [250, 500, 1000]
        orders = [balanced_order(configs, block, seed=57) for block in range(3)]

        self.assertTrue(all(set(order) == set(configs) for order in orders))
        for position in range(3):
            self.assertEqual({order[position] for order in orders}, set(configs))

    def test_truth_loader_requires_disjoint_complete_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            fieldnames = [
                "query_no",
                "query_id",
                "filter_name",
                "method",
                "exact_filtered_topk_ids",
                "recall_at_10_exact_filtered",
                "filtered_rows",
                "kth_distance_sq",
                "tie_tolerance",
                "self_excluded",
                "query_split",
                "candidate_validity_predicate",
                "query_validity_predicate",
                "candidate_rows",
            ]
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "query_no": 0,
                        "query_id": 3,
                        "filter_name": SPEC.name,
                        "method": "pre_filter_exact",
                        "exact_filtered_topk_ids": "1,2",
                        "recall_at_10_exact_filtered": 1.0,
                        "filtered_rows": 20,
                        "kth_distance_sq": 1.0,
                        "tie_tolerance": 1e-9,
                        "self_excluded": True,
                        "query_split": "calibration",
                        "candidate_validity_predicate": "embedding_valid",
                        "query_validity_predicate": "embedding_valid",
                        "candidate_rows": 20,
                    }
                )
                writer.writerow(
                    {
                        "query_no": 1,
                        "query_id": 4,
                        "filter_name": SPEC.name,
                        "method": "pre_filter_exact",
                        "exact_filtered_topk_ids": "5,6",
                        "recall_at_10_exact_filtered": 1.0,
                        "filtered_rows": 20,
                        "kth_distance_sq": 1.0,
                        "tie_tolerance": 1e-9,
                        "self_excluded": True,
                        "query_split": "final",
                        "candidate_validity_predicate": "embedding_valid",
                        "query_validity_predicate": "embedding_valid",
                        "candidate_rows": 20,
                    }
                )

            truth, query_ids = load_truth(path, [SPEC], [0], [1], k=2)

            self.assertEqual(query_ids, {0: 3, 1: 4})
            self.assertEqual(truth[(SPEC.name, 1)].ids, (5, 6))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_truth(path, [SPEC], [0], [0], k=2)

    def test_truth_loader_rejects_legacy_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            path.write_text(
                "query_no,query_id,filter_name,method,exact_filtered_topk_ids,candidates\n"
                "0,3,filter_a,pre_filter_exact,1\"\"2,20\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "retired schema"):
                load_truth(path, [SPEC], [0], [], k=2)

    def test_calibration_uses_lcb_and_then_lowest_mean_latency(self):
        rows = []
        for query_no in range(4):
            for repeat in range(2):
                rows.append(
                    measured_row(
                        "calibration", "faiss_allowlist", query_no, repeat, 20.0, 1.0, 100
                    )
                )
                rows.append(
                    measured_row(
                        "calibration", "faiss_allowlist", query_no, repeat, 10.0, 0.95, 200
                    )
                )
                rows.append(
                    measured_row(
                        "calibration", "faiss_allowlist", query_no, repeat, 1.0, 0.90, 300
                    )
                )

        table, selected = calibration_table(
            rows,
            [SPEC],
            [100, 200, 300],
            [0.95, 0.99],
            [0, 1, 2, 3],
            repeats=2,
            bootstrap_samples=100,
            bootstrap_seed=57,
        )

        self.assertEqual(selected[(SPEC.name, 0.95)], 200)
        self.assertEqual(selected[(SPEC.name, 0.99)], 100)
        selected_rows = [row for row in table if row["selected"]]
        self.assertEqual({(row["target_recall"], row["ef_search"]) for row in selected_rows}, {(0.95, 200), (0.99, 100)})

    def test_incomplete_measurement_grid_is_invalid_with_na_metrics(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", 0, 0, 1.0, 1.0),
            measured_row("calibration", "faiss_allowlist", 0, 1, 1.0, 1.0),
            measured_row("calibration", "faiss_allowlist", 1, 0, 1.0, 1.0),
        ]

        summary = aggregate_measurements(
            rows,
            phase="calibration",
            method="faiss_allowlist",
            filter_name=SPEC.name,
            ef_search=100,
            query_nos=[0, 1],
            repeats=2,
            bootstrap_samples=10,
            bootstrap_seed=57,
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertEqual(summary["missing_pairs"], 1)
        self.assertEqual(summary["recall_mean"], NA)
        self.assertEqual(summary["latency_mean_ms"], NA)

    def test_complete_summary_reports_p99_and_query_mean_ci(self):
        rows = [
            measured_row("final", "faiss_allowlist", query_no, repeat, 1.0 + query_no + repeat, 1.0)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        summary = aggregate_measurements(
            rows,
            phase="final",
            method="faiss_allowlist",
            filter_name=SPEC.name,
            ef_search=100,
            query_nos=[0, 1],
            repeats=2,
            bootstrap_samples=100,
            bootstrap_seed=57,
        )
        self.assertEqual(summary["status"], "valid")
        self.assertIn("latency_p99_ms", summary)
        self.assertIn("latency_query_mean_ci95_low_ms", summary)

    def test_final_summary_rejects_missing_matched_pair(self):
        final_rows = []
        truth_ids = list(range(10))
        for repeat in range(2):
            final_rows.append(
                measurement_row(
                    phase="final",
                    method="sql_first_exact",
                    spec=SPEC,
                    query_no=100,
                    query_id=10,
                    repeat=repeat,
                    schedule_position=1,
                    block_no=repeat,
                    ef_search=NA,
                    result_ids=truth_ids,
                    truth_ids=truth_ids,
                    latency_ms=20.0,
                )
            )
        final_rows.append(
            measurement_row(
                phase="final",
                method="faiss_allowlist",
                spec=SPEC,
                query_no=100,
                query_id=10,
                repeat=0,
                schedule_position=2,
                block_no=0,
                ef_search=500,
                result_ids=truth_ids,
                truth_ids=truth_ids,
                latency_ms=5.0,
            )
        )

        summary = final_summary_table(
            final_rows,
            [SPEC],
            [0.95],
            {(SPEC.name, 0.95): 500},
            [100],
            repeats=2,
            bootstrap_samples=10,
            bootstrap_seed=57,
            allow_lists={
                SPEC.name: AllowList(object(), np.zeros(3, dtype=np.uint8), 20, 3.0, 3, True)
            },
        )

        self.assertEqual(len(summary), 2)
        self.assertTrue(all(row["status"] == "invalid" for row in summary))
        self.assertTrue(all(row["search_latency_mean_ms"] == NA for row in summary))
        self.assertTrue(all(row["speedup_vs_sql_first_exact"] == NA for row in summary))

    def test_complete_ladder_below_target_is_valid_unattainable_without_faiss_final(self):
        calibration = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 0.5, ef)
            for ef in (100, 200)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        table, selected = calibration_table(
            calibration, [SPEC], [100, 200], [0.9], [0, 1], 2, 50, 57
        )
        self.assertEqual(selected, {})
        self.assertTrue(all(row["outcome"] == "unattainable_on_grid" for row in table))
        self.assertTrue(all(row["calibration_ladder_complete"] for row in table))
        self.assertTrue(all(row["max_ef_search"] == 200 for row in table))

        final_rows = [
            measurement_row(
                phase="final", method="sql_first_exact", spec=SPEC, query_no=query_no,
                query_id=10 + query_no, repeat=repeat, schedule_position=1, block_no=repeat,
                ef_search=NA, result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
            )
            for query_no in (100, 101)
            for repeat in (0, 1)
        ]
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], selected, [100, 101], 2, 50, 57,
            calibration_outcomes={(SPEC.name, 0.9): "unattainable_on_grid"},
        )
        faiss = next(row for row in summary if row["method"] == "faiss_allowlist")
        self.assertEqual(faiss["outcome"], "unattainable_on_grid")
        self.assertEqual(faiss["status"], "valid")
        self.assertEqual(faiss["samples"], 0)
        self.assertEqual(faiss["expected_samples"], 0)
        self.assertEqual(faiss["missing_pairs"], 0)
        self.assertEqual(faiss["recall_mean"], NA)
        self.assertFalse(faiss["matched_recall_comparison_valid"])

    def test_incomplete_max_ef_is_not_unattainable(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 0.5, ef)
            for ef in (100, 200)
            for query_no in (0, 1)
            for repeat in (0, 1)
            if not (ef == 200 and query_no == 1 and repeat == 1)
        ]
        table, selected = calibration_table(rows, [SPEC], [100, 200], [0.9], [0, 1], 2, 50, 57)
        self.assertEqual(selected, {})
        self.assertTrue(all(row["outcome"] == "calibration_invalid" for row in table))
        self.assertTrue(any(row["status"] == "invalid" for row in table))
        self.assertTrue(any(not row["calibration_ladder_complete"] for row in table))

    def test_selected_complete_final_target_miss_is_valid_unconfirmed(self):
        calibration = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 1.0, 100)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        table, selected = calibration_table(calibration, [SPEC], [100], [0.9], [0, 1], 2, 50, 57)
        final_rows = []
        for query_no in (100, 101):
            for repeat in (0, 1):
                final_rows.append(measurement_row(
                    phase="final", method="sql_first_exact", spec=SPEC, query_no=query_no,
                    query_id=10 + query_no, repeat=repeat, schedule_position=1, block_no=repeat,
                    ef_search=NA, result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
                ))
                final_rows.append(measurement_row(
                    phase="final", method="faiss_allowlist", spec=SPEC, query_no=query_no,
                    query_id=10 + query_no, repeat=repeat, schedule_position=2, block_no=repeat,
                    ef_search=100, result_ids=list(range(8)), truth_ids=list(range(10)), latency_ms=5.0,
                ))
        summary = final_summary_table(
            final_rows, [SPEC], [0.9], selected, [100, 101], 2, 50, 57,
            calibration_outcomes={(SPEC.name, 0.9): "selected_pending_final"},
        )
        self.assertTrue(all(row["status"] == "valid" for row in summary))
        self.assertTrue(all(row["outcome"] == "selected_but_final_unconfirmed" for row in summary))
        self.assertTrue(all(not row["matched_recall_comparison_valid"] for row in summary))
        self.assertFalse(artifact_validation_errors(table, summary, [SPEC], [100], [0.9]))

    def test_duplicate_or_error_calibration_cannot_be_unattainable(self):
        rows = [
            measured_row("calibration", "faiss_allowlist", query_no, repeat, 2.0, 0.5, 100)
            for query_no in (0, 1)
            for repeat in (0, 1)
        ]
        duplicate = dict(rows[0])
        rows.append(duplicate)
        table, _ = calibration_table(rows, [SPEC], [100], [0.9], [0, 1], 2, 50, 57)
        self.assertTrue(all(row["outcome"] == "calibration_invalid" for row in table))
        rows[-1] = {**rows[-1], "valid": False, "error": "search failed"}
        table, _ = calibration_table(rows, [SPEC], [100], [0.9], [0, 1], 2, 50, 57)
        self.assertTrue(all(row["outcome"] == "calibration_invalid" for row in table))

    def test_finalizer_rejects_tamper_and_preserves_outputs_until_atomic_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filters = root / "filters.csv"
            filters.write_text(
                "filter_name,target_rate,predicate,count,actual_pct\n"
                "filter_a,10.0%,helpful_vote >= 1,20,10.0\n", encoding="utf-8"
            )
            truth = root / "truth.csv"
            fields = ["query_no", "query_id", "filter_name", "method", "exact_filtered_topk_ids",
                      "recall_at_10_exact_filtered", "filtered_rows", "kth_distance_sq", "tie_tolerance",
                      "self_excluded", "query_split", "candidate_validity_predicate",
                      "query_validity_predicate", "candidate_rows"]
            with truth.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for query_no, split in ((0, "calibration"), (1, "final")):
                    writer.writerow({"query_no": query_no, "query_id": query_no + 2,
                        "filter_name": SPEC.name, "method": "pre_filter_exact", "exact_filtered_topk_ids": "5,6",
                        "recall_at_10_exact_filtered": 1.0, "filtered_rows": 20, "kth_distance_sq": 1.0,
                        "tie_tolerance": 1e-9, "self_excluded": True, "query_split": split,
                        "candidate_validity_predicate": "embedding_valid",
                        "query_validity_predicate": "embedding_valid", "candidate_rows": 20})
            fbin, faiss = root / "vectors.fbin", root / "index.faiss"
            fbin.write_bytes(b"vectors")
            faiss.write_bytes(b"index")
            raw = root / "raw.csv"
            calibration = root / "calibration.csv"
            final = root / "final.csv"
            raw_rows = [measured_row("calibration", "faiss_allowlist", 0, repeat, 2.0, 0.5, ef)
                        for ef in (100, 200) for repeat in (0, 1)]
            final_rows = [measurement_row(
                phase="final", method="sql_first_exact", spec=SPEC, query_no=1, query_id=3,
                repeat=repeat, schedule_position=1, block_no=repeat, ef_search=NA,
                result_ids=list(range(10)), truth_ids=list(range(10)), latency_ms=20.0,
            ) for repeat in (0, 1)]
            raw_rows.extend(final_rows)
            write_csv(raw, raw_rows)
            table, _ = calibration_table(raw_rows, [SPEC], [100, 200], [0.9], [0], 2, 20, 57)
            write_csv(calibration, table)
            write_csv(final, final_rows)
            manifest = root / "legacy.json"
            payload = {
                "status": "invalid", "finished_at_utc": "2026-07-18T00:00:00+00:00",
                "args": {"filter_names": [SPEC.name], "calibration_query_offset": 0,
                    "calibration_queries": 1, "calibration_repeats": 2, "final_query_offset": 1,
                    "final_queries": 1, "final_repeats": 2, "target_recalls": "0.9",
                    "ef_search_values": "100,200", "k": 2, "bootstrap_samples": 20, "bootstrap_seed": 57,
                    "tag": "only-this-shard", "out_dir": str(root), "overwrite": True},
                "inputs": {"filters": {"path": str(filters)}, "truth": {"path": str(truth)},
                    "fbin": {"path": str(fbin)}, "faiss_index": {"path": str(faiss)},
                    "runner": {"path": "runner.py", "sha256": "a" * 64}},
                "postgres": {"table_oid": 1}, "outputs": {"raw": str(raw), "calibration": str(calibration), "final": str(final)},
                "query_splits": {"calibration_query_nos": [0], "final_query_nos": [1]},
                "source_hashes": {"truth": sha256_file(truth), "fbin": sha256_file(fbin), "faiss": sha256_file(faiss)},
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            prefix = root / "finalized"
            summary_path = root / "finalized_summary.csv"
            manifest_path = root / "finalized_manifest.json"
            summary_path.write_text("old-summary", encoding="utf-8")
            manifest_path.write_text("old-manifest", encoding="utf-8")
            fbin.write_bytes(b"tampered")
            with self.assertRaisesRegex(FinalizationFailure, "hash changed"):
                finalize_existing(manifest, raw, calibration, final, prefix)
            self.assertEqual(summary_path.read_text(encoding="utf-8"), "old-summary")
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "old-manifest")
            fbin.write_bytes(b"vectors")
            outputs = finalize_existing(manifest, raw, calibration, final, prefix)
            self.assertTrue(outputs["summary"].is_file())
            finalized = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(finalized["artifact_valid"])
            self.assertEqual(finalized["status"], "complete")
            self.assertEqual(finalized["software_versions"]["measurement_runner_sha256"], "a" * 64)
            self.assertNotIn("filter_names", finalized["run_contract"])
            self.assertNotIn("tag", finalized["run_contract"])
            self.assertNotIn("out_dir", finalized["run_contract"])
            self.assertTrue(Path(finalized["outputs"]["summary"]["path"]).is_absolute())
            self.assertTrue(Path(finalized["outputs"]["manifest"]).is_absolute())
            write_csv(calibration, table[:-1])
            with self.assertRaisesRegex(FinalizationFailure, "calibration key coverage"):
                finalize_existing(manifest, raw, calibration, final, root / "rejected")


if __name__ == "__main__":
    unittest.main()
