from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.hybrid_vector_db.scripts.finalize_sqlens_matched_recall_comparison import (
    FinalizationFailure,
    METHODS,
    QUERY_NOS,
    finalize_artifacts,
)


FILTERS = tuple(f"filter_{number:02d}" for number in range(14))
TARGETS = (0.90, 0.95, 0.99)
GT_HASH = "a" * 64
RUNNER_HASH = "b" * 64
VECTOR_HASH = "c" * 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalizeSqlensMatchedRecallComparisonTests(unittest.TestCase):
    def _fixture(self, root: Path, *, raw_official: bool = False) -> tuple[Path, dict[str, Path]]:
        csv_paths: dict[str, Path] = {}
        manifest_paths: dict[str, Path] = {}
        for method in METHODS:
            csv_path = root / f"{method}.csv"
            manifest_path = root / f"{method}.json"
            csv_paths[method] = csv_path
            manifest_paths[method] = manifest_path
            if raw_official and method == "official_pgvector":
                rows = []
                fields = ["phase", "filter_name", "query_no", "repeat", "recall_at_10", "search_latency_ms", "valid", "error"]
                for filter_name in FILTERS:
                    for query_no in QUERY_NOS:
                        repeats = 10 if filter_name == FILTERS[0] and query_no == 100 else 1
                        for repeat in range(repeats):
                            # q100's ten zeroes make sample mean (0.9) differ
                            # from query-level mean (0.99).
                            recall = 0.0 if filter_name == FILTERS[0] and query_no == 100 else 1.0
                            rows.append({
                                "phase": "final",
                                "filter_name": filter_name,
                                "query_no": query_no,
                                "repeat": repeat,
                                "recall_at_10": recall,
                                "search_latency_ms": 10.0,
                                "valid": "true",
                                "error": "",
                            })
                self._write_csv(csv_path, fields, rows)
                kind = "raw"
            else:
                fields = [
                    "filter_name", "target_recall", "status", "recall_aggregation",
                    "recall_mean", "queries", "samples", "latency_mean_ms",
                    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                ]
                rows = [
                    {
                        "filter_name": filter_name,
                        "target_recall": target,
                        "status": "valid",
                        "recall_aggregation": "query_level_mean",
                        "recall_mean": 1.0,
                        "queries": 100,
                        "samples": 100,
                        "latency_mean_ms": 20.0,
                        "latency_p50_ms": 20.0,
                        "latency_p95_ms": 20.0,
                        "latency_p99_ms": 20.0,
                    }
                    for filter_name in FILTERS
                    for target in TARGETS
                ]
                self._write_csv(csv_path, fields, rows)
                kind = "summary"
            source_manifest = {
                "artifact": f"amazon10m_{method}_formal",
                "artifact_valid": True,
                "status": "complete",
                "dataset": "amazon10m",
                "filter_names": list(FILTERS),
                "target_recalls": list(TARGETS),
                "query_cohort": {"query_nos": list(QUERY_NOS), "gt_hash": GT_HASH},
                "latency_scope": "search_only",
                "candidate_validity_predicate": "embedding_valid",
                "candidate_universe": {"expected_rows": 9_979_556},
                "software_versions": {"measurement_runner_sha256": RUNNER_HASH},
                "binary_provenance": {"version": "formal-test", "sha256": VECTOR_HASH},
                "outputs": {kind: {"path": str(csv_path), "sha256": sha256(csv_path)}},
            }
            if method == "weaviate_production":
                source_manifest.pop("binary_provenance")
                source_manifest["service_provenance"] = {
                    "version": "1.38.0", "service_image_digest": "sha256:" + "a" * 64,
                }
            if method == "sqlens_enabled":
                source_manifest["sqlens_runtime_provenance"] = {
                    "loaded_sqlens_build_id": "sqlens-v11-test",
                    "loaded_vector_so_sha256": VECTOR_HASH,
                }
            manifest_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")

        index = {
            "artifact": "amazon10m_five_way_matched_recall_inputs",
            "dataset": "amazon10m",
            "filter_names": list(FILTERS),
            "target_recalls": list(TARGETS),
            "query_cohort": {"query_nos": list(QUERY_NOS), "gt_hash": GT_HASH},
            "latency_scope": "search_only",
            "artifacts": [
                {
                    "method": method,
                    "manifest": str(manifest_paths[method]),
                    "manifest_sha256": sha256(manifest_paths[method]),
                    "csv": str(csv_paths[method]),
                    "csv_sha256": sha256(csv_paths[method]),
                    "kind": "raw" if raw_official and method == "official_pgvector" else "summary",
                }
                for method in METHODS
            ],
        }
        input_path = root / "inputs.json"
        input_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        return input_path, csv_paths

    @staticmethod
    def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_matrix_is_published_and_query_level_mean_meets_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, _ = self._fixture(root, raw_official=True)
            outputs = finalize_artifacts(input_path, root / "comparison")
            rows = list(csv.DictReader(outputs["summary"].open(newline="", encoding="utf-8")))
            self.assertEqual(len(rows), 14 * 3 * 5)
            official = next(
                row for row in rows
                if row["method"] == "official_pgvector"
                and row["filter_name"] == FILTERS[0]
                and row["target_recall"] == "0.99"
            )
            self.assertEqual(official["cell_status"], "valid")
            self.assertEqual(official["target_met"], "True")
            self.assertAlmostEqual(float(official["recall_mean"]), 0.99)
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["matrix"]["cells"], 210)
            self.assertEqual(manifest["latency_scope"], "search_only")
            self.assertEqual(manifest["cell_counts"]["valid"], 210)

    def test_csv_tamper_is_rejected_and_existing_outputs_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, csv_paths = self._fixture(root)
            outputs = finalize_artifacts(input_path, root / "comparison")
            old_summary = outputs["summary"].read_bytes()
            old_manifest = outputs["manifest"].read_bytes()
            csv_paths["faiss_allowlist"].write_text(
                csv_paths["faiss_allowlist"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(FinalizationFailure, "CSV SHA changed"):
                finalize_artifacts(input_path, root / "comparison")
            self.assertEqual(outputs["summary"].read_bytes(), old_summary)
            self.assertEqual(outputs["manifest"].read_bytes(), old_manifest)

    def test_missing_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, csv_paths = self._fixture(root)
            path = csv_paths["weaviate_production"]
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(FinalizationFailure, "CSV SHA changed"):
                finalize_artifacts(input_path, root / "comparison")
            # Update only the declared CSV hash: the schema/key-space check must still fail.
            index = json.loads(input_path.read_text(encoding="utf-8"))
            for entry in index["artifacts"]:
                if entry["method"] == "weaviate_production":
                    entry["csv_sha256"] = sha256(path)
            input_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationFailure, "summary missing or duplicated cell"):
                finalize_artifacts(input_path, root / "comparison")

    def test_cohort_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, _ = self._fixture(root)
            manifest_path = root / "sqlens_enabled.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["query_cohort"]["gt_hash"] = "d" * 64
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            index = json.loads(input_path.read_text(encoding="utf-8"))
            for entry in index["artifacts"]:
                if entry["method"] == "sqlens_enabled":
                    entry["manifest_sha256"] = sha256(manifest_path)
            input_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationFailure, "GT/cohort hash mismatch"):
                finalize_artifacts(input_path, root / "comparison")

    def test_latency_scope_cannot_relabel_search_only_as_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, _ = self._fixture(root)
            index = json.loads(input_path.read_text(encoding="utf-8"))
            index["latency_scope"] = "end_to_end"
            input_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationFailure, "latency scope mismatch"):
                finalize_artifacts(input_path, root / "comparison")

    def test_source_must_declare_latency_scope_and_candidate_universe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, _ = self._fixture(root)
            manifest_path = root / "faiss_allowlist.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["latency_scope"]
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            index = json.loads(input_path.read_text(encoding="utf-8"))
            entry = next(item for item in index["artifacts"] if item["method"] == "faiss_allowlist")
            entry["manifest_sha256"] = sha256(manifest_path)
            input_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationFailure, "latency scope"):
                finalize_artifacts(input_path, root / "comparison")

            manifest["latency_scope"] = "search_only"
            del manifest["candidate_validity_predicate"]
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            entry["manifest_sha256"] = sha256(manifest_path)
            input_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(FinalizationFailure, "candidate universe"):
                finalize_artifacts(input_path, root / "comparison")


if __name__ == "__main__":
    unittest.main()
