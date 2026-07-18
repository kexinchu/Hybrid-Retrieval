from __future__ import annotations

import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/hybrid_vector_db/scripts/combine_matched_recall_shards.py"
SPEC = importlib.util.spec_from_file_location("combine_matched_recall_shards", SCRIPT)
assert SPEC and SPEC.loader
combiner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(combiner)


class CombineMatchedRecallShardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.expected = self.root / "expected.csv"
        with self.expected.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filter_name"])
            writer.writeheader()
            writer.writerows({"filter_name": name} for name in ("alpha", "beta"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _shard(self, name: str, *, filter_name: str | None = None, **changes: object) -> Path:
        filter_name = filter_name or name
        base = self.root / f"{name}_raw.csv"
        rows = [{"phase": "setup", "method": "faiss_allowlist", "filter_name": filter_name, "ef_search": "N/A"}]
        calibration = [{"filter_name": filter_name, "method": "faiss_allowlist", "target_recall": "0.90", "ef_search": "100"}]
        final = [{"phase": "final", "method": "sql_first_exact", "filter_name": filter_name, "query_no": "1", "query_id": "11", "repeat": "0", "ef_search": "N/A"}]
        summary = [{"filter_name": filter_name, "method": "sql_first_exact", "target_recall": "0.90", "selected_faiss_ef_search": "100"}]
        paths = {
            "raw": base,
            "calibration": base.with_name(base.stem.replace("_raw", "_calibration") + ".csv"),
            "final": base.with_name(base.stem.replace("_raw", "_final") + ".csv"),
            "summary": base.with_name(base.stem.replace("_raw", "_summary") + ".csv"),
        }
        for kind, values in {"raw": rows, "calibration": calibration, "final": final, "summary": summary}.items():
            with paths[kind].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
        manifest = {
            "artifact": "amazon10m_matched_recall_baselines",
            "artifact_valid": True,
            "status": "complete",
            "filter_names": [filter_name],
            "run_contract": {"dataset": "amazon10m", "k": 10},
            "source_db": {"table": "reviews", "snapshot": "db-1"},
            "source_hashes": {"faiss": "faiss-1", "fbin": "fbin-1", "truth": "truth-1"},
            "query_splits": {"calibration": [0], "final": [1]},
            "repeats": {"calibration": 2, "final": 5},
            "target_recalls": [0.90, 0.95],
            "ef_ladder": [100, 200],
            "software_versions": {"python": "3.10", "faiss": "1.14"},
            "outputs": {**{kind: str(path) for kind, path in paths.items()}, "manifest": str(self.root / f"{name}_manifest.json")},
        }
        manifest.update(changes)
        manifest_path = self.root / f"{name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_success_merges_stably_and_records_hashes(self) -> None:
        first = self._shard("b", filter_name="beta")
        second = self._shard("a", filter_name="alpha")
        outputs = combiner.combine([first, second], self.expected, self.root / "combined")
        self.assertTrue(outputs["raw"].exists())
        with outputs["raw"].open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["filter_name"] for row in rows], ["alpha", "beta"])
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        self.assertTrue(manifest["artifact_valid"])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["input_artifacts"]), 2)
        self.assertEqual(len(manifest["input_manifests"]), 2)
        self.assertTrue(all(item["sha256"] for item in manifest["input_manifests"]))
        self.assertEqual(set(manifest["output_sha256"]), set(combiner.ARTIFACTS))

    def test_duplicate_and_missing_filter_are_rejected(self) -> None:
        one = self._shard("one", filter_name="alpha")
        two = self._shard("two", filter_name="alpha")
        with self.assertRaises(combiner.ValidationFailure):
            combiner.combine([one, two], self.expected, self.root / "out")
        with self.assertRaises(combiner.ValidationFailure):
            combiner.combine([one], self.expected, self.root / "out2")

    def test_provenance_mismatch_is_rejected(self) -> None:
        one = self._shard("one", filter_name="alpha")
        two = self._shard("two", filter_name="beta", source_hashes={"faiss": "different", "fbin": "fbin-1", "truth": "truth-1"})
        with self.assertRaisesRegex(combiner.ValidationFailure, "provenance/run contract mismatch"):
            combiner.combine([one, two], self.expected, self.root / "out")

    def test_shard_local_run_fields_do_not_create_false_mismatch(self) -> None:
        one = self._shard(
            "one", filter_name="alpha",
            run_contract={"dataset": "amazon10m", "k": 10, "tag": "alpha-run", "filter_names": ["alpha"]},
        )
        two = self._shard(
            "two", filter_name="beta",
            run_contract={"dataset": "amazon10m", "k": 10, "tag": "beta-run", "filter_names": ["beta"]},
        )

        outputs = combiner.combine([one, two], self.expected, self.root / "out")
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["contract"]["run_contract"], {"dataset": "amazon10m", "k": 10})

    def test_invalid_shard_is_rejected(self) -> None:
        shard = self._shard("one", filter_name="alpha", artifact_valid=False)
        with self.assertRaisesRegex(combiner.ValidationFailure, "artifact_valid"):
            combiner.combine([shard], self.expected, self.root / "out")

    def test_csv_foreign_filter_and_duplicate_key_are_rejected(self) -> None:
        shard = self._shard("one", filter_name="alpha")
        manifest = json.loads(shard.read_text(encoding="utf-8"))
        final = Path(manifest["outputs"]["final"])
        with final.open("a", encoding="utf-8") as stream:
            stream.write("final,sql_first_exact,foreign,1,11,0,N/A\n")
        with self.assertRaisesRegex(combiner.ValidationFailure, "foreign filter"):
            combiner.combine([shard, self._shard("two", filter_name="beta")], self.expected, self.root / "out")

        shard = self._shard("dup", filter_name="alpha")
        manifest = json.loads(shard.read_text(encoding="utf-8"))
        raw = Path(manifest["outputs"]["raw"])
        with raw.open("a", encoding="utf-8") as stream:
            stream.write("setup,faiss_allowlist,alpha,N/A\n")
        with self.assertRaisesRegex(combiner.ValidationFailure, "duplicate key"):
            combiner.combine([shard, self._shard("two", filter_name="beta")], self.expected, self.root / "out2")

    def test_atomic_commit_failure_preserves_old_outputs(self) -> None:
        one = self._shard("one", filter_name="alpha")
        two = self._shard("two", filter_name="beta")
        prefix = self.root / "combined"
        old = {name: prefix.with_name(prefix.name + suffix) for name, suffix in combiner.OUTPUT_SUFFIXES.items()}
        old["manifest"] = prefix.with_name(prefix.name + "_manifest.json")
        for path in old.values():
            path.write_text("old\n", encoding="utf-8")
        real_replace = os.replace
        calls = 0

        def fail_once(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated commit failure")
            real_replace(source, destination)

        with mock.patch.object(combiner.os, "replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                combiner.combine([one, two], self.expected, prefix)
        self.assertTrue(all(path.read_text(encoding="utf-8") == "old\n" for path in old.values()))


if __name__ == "__main__":
    unittest.main()
