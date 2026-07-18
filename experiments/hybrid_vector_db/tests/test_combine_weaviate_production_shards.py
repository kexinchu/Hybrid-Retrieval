import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/hybrid_vector_db/scripts/combine_weaviate_production_shards.py"
SPEC = importlib.util.spec_from_file_location("combine_weaviate_production_shards", SCRIPT)
assert SPEC and SPEC.loader
combiner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = combiner
SPEC.loader.exec_module(combiner)


RAW_FIELDS = [
    "phase", "configured_filter_strategy", "filter_name", "ef", "flat_search_cutoff",
    "query_no", "query_id", "repeat", "end_to_end_ms", "latency_definition",
    "recall_at_10", "recall_contract", "retry_count", "order_error", "valid", "error",
]
SUMMARY_FIELDS = [
    "phase", "configured_filter_strategy", "filter_name", "ef", "flat_search_cutoff",
    "expected_queries", "expected_repeats", "expected_samples", "observed_samples",
    "error_count", "duplicate_pairs", "missing_pairs", "complete", "latency_definition",
    "service_qps_definition", "recall_mean", "recall_lcb95", "recall_ci95_low",
    "recall_ci95_high", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
    "latency_p99_ms", "latency_ci95_low_ms", "latency_ci95_high_ms",
    "single_client_service_qps", "target_recall", "target_status", "target_outcome",
    "comparison_status", "selected_ef", "selected_flat_search_cutoff",
]
SHARD_FILTERS = [
    ["filter_00", "filter_01", "filter_02"],
    ["filter_03", "filter_04", "filter_05"],
    ["filter_06", "filter_07", "filter_08", "filter_09"],
    ["filter_10", "filter_11", "filter_12", "filter_13"],
]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_row(phase, name, query_no, repeat):
    return {
        "phase": phase,
        "configured_filter_strategy": "acorn" if phase == "calibration" else "sweeping",
        "filter_name": name,
        "ef": 100,
        "flat_search_cutoff": 0 if phase == "calibration" else 100_000,
        "query_no": query_no,
        "query_id": query_no + 1000,
        "repeat": repeat,
        "end_to_end_ms": 1.25,
        "latency_definition": combiner.LATENCY_DEFINITION,
        "recall_at_10": 1.0,
        "recall_contract": combiner.RECALL_CONTRACT,
        "retry_count": 0,
        "order_error": "",
        "valid": True,
        "error": "",
    }


def summary_row(phase, name, target="N/A"):
    final = phase == "final"
    expected_queries = 100
    expected_repeats = 5 if final else 2
    row = {
        "phase": phase,
        "configured_filter_strategy": "sweeping" if final else "acorn",
        "filter_name": name,
        "ef": 100,
        "flat_search_cutoff": 100_000 if final else 0,
        "expected_queries": expected_queries,
        "expected_repeats": expected_repeats,
        "expected_samples": expected_queries * expected_repeats,
        "observed_samples": expected_queries * expected_repeats,
        "error_count": 0,
        "duplicate_pairs": 0,
        "missing_pairs": 0,
        "complete": True,
        "latency_definition": combiner.LATENCY_DEFINITION,
        "service_qps_definition": combiner.QPS_DEFINITION,
        "recall_mean": 1.0,
        "recall_lcb95": 1.0,
        "recall_ci95_low": 1.0,
        "recall_ci95_high": 1.0,
        "latency_mean_ms": 1.25,
        "latency_p50_ms": 1.25,
        "latency_p95_ms": 1.5,
        "latency_p99_ms": 1.5,
        "latency_ci95_low_ms": 1.0,
        "latency_ci95_high_ms": 1.5,
        "single_client_service_qps": 800.0,
        "target_recall": target,
        "target_status": "selected" if final else "",
        "target_outcome": "selected_and_confirmed" if final else "",
        "comparison_status": "confirmed" if final else "",
        "selected_ef": 100 if final else "",
        "selected_flat_search_cutoff": 100_000 if final else "",
    }
    return row


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.filters_csv = self.root / "expected_filters.csv"
        write_csv(
            self.filters_csv,
            ["filter_name", "predicate"],
            [{"filter_name": name, "predicate": "x = 1"} for group in SHARD_FILTERS for name in group],
        )
        self.filter_sha = combiner.sha256_file(self.filters_csv)
        self.manifests = [self._make_shard(index, names) for index, names in enumerate(SHARD_FILTERS)]
        self.out_prefix = self.root / "combined"

    def _make_shard(self, index, filters):
        stem = f"weaviate_production_matched_recall_shard_{index}"
        paths = {
            "raw_csv": self.root / f"{stem}.csv",
            "summary_csv": self.root / f"{stem}_summary.csv",
            "config_json": self.root / f"{stem}_config.json",
            "schema_json": self.root / f"{stem}_schema.json",
            "manifest_json": self.root / f"{stem}_manifest.json",
        }
        source_hashes = {
            "runner": digest("runner"),
            "baseline_runner": digest("baseline"),
            "filters_csv": self.filter_sha,
            "truth_csv": digest("truth"),
            "fbin": digest("fbin"),
        }
        run_spec_hash = digest(f"run-spec-{index}")
        image_digest = "registry/weaviate@sha256:" + digest("image")
        original_schema = {
            "class": "AmazonGroceryReview",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": "l2-squared", "filterStrategy": "acorn",
                "ef": -1, "flatSearchCutoff": 0,
            },
        }
        original_schema_hash = combiner._sha256_json(original_schema)
        config = {
            "class": "AmazonGroceryReview",
            "git_revision": "a" * 40,
            "source_hashes": source_hashes,
            "run_spec_hash": run_spec_hash,
            "vector_rows": 10_000_000,
            "dimensions": 128,
            "k": 10,
            "configured_filter_strategies": ["acorn", "sweeping"],
            "filter_names": filters,
            "ef_values": [100, 250],
            "flat_search_cutoffs": [0, 100_000],
            "targets": [0.90, 0.95, 0.99],
            "hnsw_dominance_guard": 1.05,
            "service_identity": {
                "actual_version": "1.38.0",
                "expected_version": "1.38.0",
                "service_image_digest": image_digest,
            },
            "measurement_mode": "single_client_sequential",
            "calibration": {
                "queries": 100,
                "repeats": 2,
                "schedule_order": "all flat representatives before HNSW",
                "selection_rule": "fastest measured LCB-qualified configuration",
            },
            "final": {
                "queries": 100,
                "repeats": 5,
                "deduplication_key": [
                    "configured_filter_strategy", "filter_name", "flat_search_cutoff", "ef",
                ],
                "runs_selected_system_configs_plus_flat_exactness_controls": True,
                "reuses_one_exact_measurement_for_multiple_targets": True,
                "retunes_after_held_out_measurement": False,
            },
            "checkpoint": {
                "path": str(self.root / f"checkpoint-{index}.json"),
                "persistence": "atomic complete-block snapshot",
                "storage": "single JSON snapshot",
                "complete_block_boundary": True,
                "original_schema_sha256": original_schema_hash,
                "run_spec_hash": run_spec_hash,
            },
            "effective_cutoffs_by_filter": {name: [0, 100_000] for name in filters},
        }
        schema = {
            "class": "AmazonGroceryReview",
            "source_hashes": source_hashes,
            "records": [
                {"phase": "original_schema_snapshot", "schema": original_schema},
                {"phase": "calibration", "schema": {**original_schema}},
                {"phase": "restore", "schema": original_schema},
            ],
        }
        raw_rows = []
        summary_rows = []
        target_records = []
        selected_and_confirmed = 0
        unattainable = 0
        for name in filters:
            raw_rows.extend(
                raw_row("calibration", name, query_no, repeat)
                for query_no in range(100)
                for repeat in range(2)
            )
            raw_rows.extend(
                raw_row("final", name, query_no, repeat)
                for query_no in range(100, 200)
                for repeat in range(5)
            )
            summary_rows.append(summary_row("calibration", name))
            for target in combiner.EXPECTED_TARGETS:
                is_unattainable = index == 3 and name == filters[-1] and target == 0.99
                if is_unattainable:
                    target_records.append({
                        "filter_name": name,
                        "target_recall": target,
                        "status": "unattainable_on_grid",
                        "selected_filter_strategy": "N/A",
                        "selected_ef": "N/A",
                        "selected_flat_search_cutoff": "N/A",
                    })
                    unattainable += 1
                else:
                    target_records.append({
                        "filter_name": name,
                        "target_recall": target,
                        "status": "selected",
                        "selected_filter_strategy": "sweeping",
                        "selected_ef": 100,
                        "selected_flat_search_cutoff": 100_000,
                    })
                    summary_rows.append(summary_row("final", name, target))
                    selected_and_confirmed += 1
        write_csv(paths["raw_csv"], RAW_FIELDS, raw_rows)
        write_csv(paths["summary_csv"], SUMMARY_FIELDS, summary_rows)
        write_json(paths["config_json"], config)
        write_json(paths["schema_json"], schema)
        manifest = {
            "artifact_valid": True,
            "status": "complete",
            "manifest_commit": "atomic_last",
            "git_revision": "a" * 40,
            "source_hashes": source_hashes,
            "run_spec_hash": run_spec_hash,
            "service": {
                "meta": {"version": "1.38.0"},
                "version": "1.38.0",
                "expected_version": "1.38.0",
                "version_gate_passed": True,
                "image_digest": image_digest,
                "count": 10_000_000,
                "filter_counts": {name: 1000 + offset for offset, name in enumerate(filters)},
                "measurement_mode": "single_client_sequential",
                "concurrency": 1,
                "errors": [],
            },
            "schema": {
                "original_schema_sha256": original_schema_hash,
                "original_definition_restored": True,
            },
            "checkpoint": {"original_schema_persisted_before_schema_put": True},
            "calibration_selection": {"targets": target_records},
            "target_outcomes": {
                "selected_and_confirmed": selected_and_confirmed,
                "selected_but_final_unconfirmed": 0,
                "unattainable_on_grid": unattainable,
            },
            "flat_held_out_exactness_gate": {
                "required_recall_mean": 1.0,
                "required_recall_lcb95": 1.0,
                "records": [
                    {
                        "filter_name": name,
                        "held_out_recall_mean": 1.0,
                        "held_out_recall_lcb95": 1.0,
                        "representative": {
                            "configured_filter_strategy": "sweeping",
                            "filter_name": name,
                            "flat_search_cutoff": 100_000,
                            "ef": 100,
                        },
                    }
                    for name in filters
                ],
            },
            "raw_rows": len(raw_rows),
            "summary_rows": len(summary_rows),
            "outputs": {name: str(path) for name, path in paths.items()},
            "output_sha256": {
                name: combiner.sha256_file(paths[name]) for name in combiner.DATA_OUTPUTS
            },
        }
        write_json(paths["manifest_json"], manifest)
        return paths["manifest_json"]

    def manifest(self, index):
        return read_json(self.manifests[index])

    def rewrite_manifest(self, index, value):
        write_json(self.manifests[index], value)

    def rewrite_output(self, index, name, value):
        manifest = self.manifest(index)
        path = Path(manifest["outputs"][name])
        if name.endswith("_csv"):
            fields = RAW_FIELDS if name == "raw_csv" else SUMMARY_FIELDS
            write_csv(path, fields, value)
        else:
            write_json(path, value)
        manifest["output_sha256"][name] = combiner.sha256_file(path)
        if name == "raw_csv":
            manifest["raw_rows"] = len(value)
        elif name == "summary_csv":
            manifest["summary_rows"] = len(value)
        self.rewrite_manifest(index, manifest)

    def output_paths(self):
        return {
            name: self.out_prefix.with_name(self.out_prefix.name + suffix)
            for name, suffix in combiner.OUTPUT_SUFFIXES.items()
        }


class CombineWeaviateProductionShardsTests(unittest.TestCase):
    def test_combines_four_valid_shards_and_preserves_explicit_unattainable_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            outputs = combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)
            self.assertEqual(outputs, fixture.output_paths())
            manifest = read_json(outputs["manifest_json"])
            self.assertTrue(manifest["artifact_valid"])
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["input_shards"]), 4)
            self.assertEqual(len(manifest["filter_names"]), 14)
            self.assertEqual(len(manifest["calibration_selection"]["targets"]), 42)
            self.assertEqual(manifest["target_outcomes"]["unattainable_on_grid"], 1)
            self.assertEqual(
                manifest["raw_rows"], sum(read_json(path)["raw_rows"] for path in fixture.manifests)
            )
            for name in combiner.DATA_OUTPUTS:
                self.assertEqual(manifest["output_sha256"][name], combiner.sha256_file(outputs[name]))
            config = read_json(outputs["config_json"])
            self.assertEqual(config["measurement_mode"], "single_client_sequential")
            self.assertEqual(config["filter_names"], [name for group in SHARD_FILTERS for name in group])
            schema = read_json(outputs["schema_json"])
            self.assertTrue(schema["original_definition_restored"])
            self.assertEqual(len(schema["shards"]), 4)

    def test_rejects_manifest_service_and_immutable_identity_failures(self):
        mutations = {
            "artifact_valid": lambda manifest: manifest.update({"artifact_valid": False}),
            "status": lambda manifest: manifest.update({"status": "invalid"}),
            "version": lambda manifest: manifest["service"].update({"version": "1.39.0"}),
            "digest": lambda manifest: manifest["service"].update({"image_digest": "latest"}),
            "count": lambda manifest: manifest["service"].update({"count": 9_999_999}),
            "concurrency": lambda manifest: manifest["service"].update({"concurrency": 2}),
            "errors": lambda manifest: manifest["service"].update({"errors": ["query failed"]}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                manifest = fixture.manifest(0)
                mutate(manifest)
                fixture.rewrite_manifest(0, manifest)
                with self.assertRaises(combiner.ValidationFailure):
                    combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)
                self.assertFalse(any(path.exists() for path in fixture.output_paths().values()))

    def test_rejects_output_hash_row_count_and_measurement_contract_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = fixture.manifest(0)
            manifest["raw_rows"] += 1
            fixture.rewrite_manifest(0, manifest)
            with self.assertRaisesRegex(combiner.ValidationFailure, "raw_rows"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            raw_path = Path(fixture.manifest(0)["outputs"]["raw_csv"])
            raw_path.write_text(raw_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(combiner.ValidationFailure, "SHA256"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = fixture.manifest(0)
            raw_path = Path(manifest["outputs"]["raw_csv"])
            with raw_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[0]["recall_contract"] = "plain_set_overlap"
            fixture.rewrite_output(0, "raw_csv", rows)
            with self.assertRaisesRegex(combiner.ValidationFailure, "recall contract"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

    def test_rejects_incompatible_source_run_contract_and_filter_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = fixture.manifest(1)
            config = read_json(Path(manifest["outputs"]["config_json"]))
            config["ef_values"] = [100, 250, 500]
            fixture.rewrite_output(1, "config_json", config)
            with self.assertRaisesRegex(combiner.ValidationFailure, "run contract"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = fixture.manifest(0)
            config = read_json(Path(manifest["outputs"]["config_json"]))
            config["filter_names"][1] = config["filter_names"][0]
            fixture.rewrite_output(0, "config_json", config)
            with self.assertRaisesRegex(combiner.ValidationFailure, "filter_names"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            fields, rows = combiner._read_csv(fixture.filters_csv, "filters")
            rows.pop()
            write_csv(fixture.filters_csv, fields, rows)
            with self.assertRaisesRegex(combiner.ValidationFailure, "exactly 14"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

    def test_rejects_missing_duplicate_or_error_pairs_and_schema_restore_gaps(self):
        mutations = {
            "missing target": lambda manifest: manifest["calibration_selection"]["targets"].pop(),
            "duplicate target": lambda manifest: manifest["calibration_selection"]["targets"].append(
                dict(manifest["calibration_selection"]["targets"][0])
            ),
            "restore attestation": lambda manifest: manifest["schema"].update(
                {"original_definition_restored": False}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                manifest = fixture.manifest(0)
                mutate(manifest)
                fixture.rewrite_manifest(0, manifest)
                with self.assertRaises(combiner.ValidationFailure):
                    combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = fixture.manifest(0)
            summary_path = Path(manifest["outputs"]["summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[0]["missing_pairs"] = "1"
            fixture.rewrite_output(0, "summary_csv", rows)
            with self.assertRaisesRegex(combiner.ValidationFailure, "missing_pairs"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            manifest = fixture.manifest(0)
            schema_path = Path(manifest["outputs"]["schema_json"])
            schema = read_json(schema_path)
            schema["records"][-1]["schema"]["vectorIndexConfig"]["ef"] = 500
            fixture.rewrite_output(0, "schema_json", schema)
            with self.assertRaisesRegex(combiner.ValidationFailure, "differs from original"):
                combiner.combine(fixture.manifests, fixture.filters_csv, fixture.out_prefix)

    def test_main_returns_nonzero_and_preserves_existing_bundle_on_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            sentinels = fixture.output_paths()
            for path in sentinels.values():
                path.write_text("existing", encoding="utf-8")
            manifest = fixture.manifest(0)
            manifest["artifact_valid"] = False
            fixture.rewrite_manifest(0, manifest)
            rc = combiner.main([
                "--input-manifests", *(str(path) for path in fixture.manifests),
                "--expected-filters-csv", str(fixture.filters_csv),
                "--out-prefix", str(fixture.out_prefix),
            ])
            self.assertEqual(rc, 2)
            self.assertTrue(all(path.read_text(encoding="utf-8") == "existing" for path in sentinels.values()))

    def test_atomic_commit_restores_every_existing_output_after_mid_publish_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destinations = {
                name: root / f"published{suffix}"
                for name, suffix in combiner.OUTPUT_SUFFIXES.items()
            }
            stage = root / "stage"
            stage.mkdir()
            staged = {name: stage / path.name for name, path in destinations.items()}
            for name in combiner.ALL_OUTPUTS:
                destinations[name].write_text(f"old-{name}", encoding="utf-8")
                staged[name].write_text(f"new-{name}", encoding="utf-8")

            real_replace = combiner.os.replace

            def fail_on_summary(source, destination):
                if Path(source) == staged["summary_csv"]:
                    raise OSError("simulated publish failure")
                return real_replace(source, destination)

            with mock.patch.object(combiner.os, "replace", side_effect=fail_on_summary):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    combiner._commit_bundle(staged, destinations)
            for name, path in destinations.items():
                self.assertEqual(path.read_text(encoding="utf-8"), f"old-{name}")

    def test_requires_exactly_four_distinct_production_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with self.assertRaisesRegex(combiner.ValidationFailure, "exactly 4"):
                combiner.resolve_manifests(fixture.manifests[:3])
            wrong = Path(tmp) / "ordinary_manifest.json"
            wrong.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(combiner.ValidationFailure, "production matched-recall"):
                combiner.combine(
                    [*fixture.manifests[:3], wrong], fixture.filters_csv, fixture.out_prefix
                )


if __name__ == "__main__":
    unittest.main()
