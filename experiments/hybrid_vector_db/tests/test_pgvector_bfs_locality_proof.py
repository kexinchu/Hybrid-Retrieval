import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pgvector_design1_design2_design3_selectivity_benchmark as benchmark  # noqa: E402


def locality(blocks: list[int]) -> dict[str, object]:
    pairs = max(len(blocks) - 1, 0)
    same = sum(left == right for left, right in zip(blocks, blocks[1:]))
    next_page = sum(right == left + 1 for left, right in zip(blocks, blocks[1:]))
    nondecreasing = sum(right >= left for left, right in zip(blocks, blocks[1:]))
    return {
        "format": "sqlens-hnsw-bfs-locality-v1",
        "rank_base": 0,
        "graph_nodes": len(blocks),
        "reachable_nodes": len(blocks),
        "fallback_nodes": 0,
        "sequence_nodes": len(blocks),
        "adjacent_pairs": pairs,
        "same_block_pairs": same,
        "next_block_pairs": next_page,
        "same_or_next_page_pairs": same + next_page,
        "nondecreasing_pairs": nondecreasing,
        "backward_pairs": pairs - nondecreasing,
        "total_abs_block_delta": sum(
            abs(right - left) for left, right in zip(blocks, blocks[1:])
        ),
        "max_abs_block_delta": max(
            [abs(right - left) for left, right in zip(blocks, blocks[1:])] or [0]
        ),
        "page_runs": len(blocks) and 1 + sum(
            left != right for left, right in zip(blocks, blocks[1:])
        ),
        "same_block_ratio": same / pairs if pairs else 0.0,
        "same_or_next_page_ratio": (same + next_page) / pairs if pairs else 0.0,
        "nondecreasing_ratio": nondecreasing / pairs if pairs else 0.0,
        "full_statistics": True,
        "sample_limit": 256,
        "sample_count": len(blocks),
        "sample_truncated": False,
        "sample_strategy": "evenly_spaced_inclusive",
        "rank_samples": [
            {"rank": rank, "block": block, "offset": 1}
            for rank, block in enumerate(blocks)
        ],
    }


class PgvectorBfsLocalityProofTests(unittest.TestCase):
    def test_complete_counters_and_rank_samples_are_validated(self) -> None:
        value = locality([10, 10, 11, 15, 14])
        benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

        value["sample_truncated"] = True
        with self.assertRaisesRegex(RuntimeError, "sample truncation"):
            benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

    def test_stable_d2_proof_keeps_both_symmetric_locality_objects(self) -> None:
        comparison = {
            field: True
            for field in (
                "same_heap",
                "logical_equal",
                "physical_equal",
                "entry_equal",
                "definition_equal",
                "tuple_coverage_equal",
            )
        }
        comparison.update(
            {
                "format": "sqlens-hnsw-compare-v2",
                "left_definition_digest": "sha256:" + "1" * 64,
                "right_definition_digest": "sha256:" + "1" * 64,
                "left_tuple_coverage_digest": "sha256:" + "2" * 64,
                "right_tuple_coverage_digest": "sha256:" + "2" * 64,
                "left_logical_digest": "sha256:" + "3" * 64,
                "right_logical_digest": "sha256:" + "3" * 64,
                "left_physical_digest": "sha256:" + "4" * 64,
                "right_physical_digest": "sha256:" + "5" * 64,
                "left_bfs_locality": locality([1, 2, 2]),
                "right_bfs_locality": locality([8, 8, 9]),
            }
        )
        stable = benchmark.stable_d2_graph_proof(
            {
                "source_index": "source",
                "clone_index": "clone",
                "relations": {
                    "source": {
                        "name": "source",
                        "oid": 1,
                        "relfilenode": 2,
                        "heap_oid": 3,
                    },
                    "clone": {
                        "name": "clone",
                        "oid": 4,
                        "relfilenode": 5,
                        "heap_oid": 3,
                    },
                },
                "comparison": comparison,
            }
        )
        self.assertEqual(
            stable["comparison"]["left_bfs_locality"],
            comparison["left_bfs_locality"],
        )
        self.assertEqual(
            stable["comparison"]["right_bfs_locality"],
            comparison["right_bfs_locality"],
        )

    def test_contract_is_bounded_sample_but_full_statistics(self) -> None:
        source = (ROOT.parent.parent / "third_party/pgvector-sqlens/src/hnswclone.c").read_text()
        smoke = (ROOT / "sql/pgvector_clone_formality_smoke.sql").read_text()
        for marker in (
            "sqlens-hnsw-bfs-locality-v1",
            "HNSW_BFS_LOCALITY_SAMPLE_LIMIT 256",
            "sameOrNextPagePairs",
            "nondecreasingPairs",
            "full_statistics",
            "rank_samples",
        ):
            self.assertIn(marker, source)
        self.assertIn("source/clone BFS locality comparison is not symmetric", smoke)


if __name__ == "__main__":
    unittest.main()
