from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import amazon10m_exact_truth as truth  # noqa: E402


class AmazonExactTruthTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
