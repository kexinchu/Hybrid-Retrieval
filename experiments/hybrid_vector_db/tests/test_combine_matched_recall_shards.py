import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


combiner = load_module(
    "combine_matched_recall_shards",
    ROOT / "experiments/hybrid_vector_db/scripts/combine_matched_recall_shards.py",
)
finalizer = load_module(
    "finalize_sqlens_matched_recall_comparison_for_combiner_test",
    ROOT / "experiments/hybrid_vector_db/scripts/finalize_sqlens_matched_recall_comparison.py",
)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Fixture:
    filters = ("filter_a", "filter_b")

    def __init__(self, root):
        self.root = Path(root)
        self.filters_csv = self.root / "filters.csv"
        write_csv(
            self.filters_csv,
            ("filter_name", "predicate"),
            ({"filter_name": name, "predicate": "price <= 10"} for name in self.filters),
        )
        self.manifests = [self.make_shard(index, name) for index, name in enumerate(self.filters)]
        self.out_prefix = self.root / "combined"

    def make_shard(self, index, filter_name):
        stem = self.root / f"shard_{index}"
        outputs = {
            kind: stem.with_name(stem.name + suffix)
            for kind, suffix in combiner.OUTPUT_SUFFIXES.items()
        }
        raw_fields = ("phase", "filter_name", "method")
        calibration_fields = ("filter_name", "method", "target_recall", "ef_search")
        final_fields = (
            "phase", "filter_name", "method", "query_no", "query_id", "repeat", "ef_search",
        )
        summary_fields = (
            "filter_name", "method", "target_recall", "selected_faiss_ef_search",
        )
        write_csv(outputs["raw"], raw_fields, [{
            "phase": "setup", "filter_name": filter_name, "method": "faiss_allowlist",
        }])
        write_csv(outputs["calibration"], calibration_fields, [{
            "filter_name": filter_name,
            "method": "faiss_allowlist",
            "target_recall": 0.90,
            "ef_search": 1000,
        }])
        write_csv(outputs["final"], final_fields, [{
            "phase": "final",
            "filter_name": filter_name,
            "method": "faiss_allowlist",
            "query_no": 100,
            "query_id": 1100,
            "repeat": 0,
            "ef_search": 1000,
        }])
        write_csv(outputs["summary"], summary_fields, [{
            "filter_name": filter_name,
            "method": "faiss_allowlist",
            "target_recall": 0.90,
            "selected_faiss_ef_search": 1000,
        }])
        source_hashes = {
            "runner": digest("runner"),
            "faiss": digest("faiss"),
            "fbin": digest("fbin"),
            "truth": digest("truth"),
        }
        manifest = {
            "artifact": "amazon10m_matched_recall_baselines",
            "artifact_valid": True,
            "status": "complete",
            "filter_names": [filter_name],
            "run_contract": {
                "dataset": "amazon10m",
                "candidate_validity_predicate": "embedding_valid",
                "target_recalls": "0.90,0.95,0.99",
                "calibration_query_offset": 20,
                "calibration_queries": 80,
                "final_query_offset": 100,
                "final_queries": 100,
            },
            "candidate_universe": {
                "predicate": "embedding_valid",
                "expected_rows": 9_979_556,
                "observed_rows": 9_979_556,
            },
            "source_hashes": source_hashes,
            "inputs": {
                "runner": {"path": "runner.py", "sha256": source_hashes["runner"]},
                "faiss_index": {"path": "index.faiss", "sha256": source_hashes["faiss"]},
                "fbin": {"path": "vectors.fbin", "sha256": source_hashes["fbin"]},
                "truth": {"path": "truth.csv", "sha256": source_hashes["truth"]},
                "postgres_table": "amazon_grocery_reviews_10m_pgvector",
            },
            "postgres": {
                "server_version": "16.14",
                "vector_extension_version": "0.8.2",
                "table_oid": 1234,
                "table_relfilenode": 5678,
                "candidate_universe": {"predicate": "embedding_valid", "rows": 9_979_556},
            },
            "faiss_index": {"type": "IndexHNSWFlat", "ntotal": 10_000_000},
            "query_splits": {
                "reserved_query_nos": list(range(20)),
                "calibration_query_nos": list(range(20, 100)),
                "final_query_nos": list(range(100, 200)),
                "query_no_overlap": False,
            },
            "repeats": {"calibration": 2, "final": 5},
            "target_recalls": [0.90, 0.95, 0.99],
            "ef_ladder": [1000, 5000],
            "software_versions": {
                "measurement_runner_sha256": source_hashes["runner"],
                "python": "3.12",
                "faiss": "1.8.0",
            },
            "execution": {
                "latency": "search-only; allow-list construction and output I/O excluded",
            },
            "row_counts": {kind: 1 for kind in combiner.ARTIFACTS},
            "outputs": {kind: str(path) for kind, path in outputs.items()},
        }
        manifest_path = stem.with_name(stem.name + "_manifest.json")
        write_json(manifest_path, manifest)
        return manifest_path

    def manifest(self, index):
        return read_json(self.manifests[index])

    def rewrite_manifest(self, index, manifest):
        write_json(self.manifests[index], manifest)


class CombineMatchedRecallShardsTests(unittest.TestCase):
    def test_combined_manifest_retains_finalizer_consumable_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            outputs = combiner.combine(
                fixture.manifests, fixture.filters_csv, fixture.out_prefix
            )
            manifest = read_json(outputs["manifest"])
            self.assertEqual(manifest["dataset"], "amazon10m")
            self.assertEqual(manifest["filter_names"], list(fixture.filters))
            self.assertEqual(manifest["target_recalls"], [0.90, 0.95, 0.99])
            self.assertEqual(manifest["latency_scope"], "search_only")
            self.assertEqual(manifest["candidate_universe"], {
                "predicate": "embedding_valid",
                "expected_rows": 9_979_556,
                "observed_rows": 9_979_556,
            })
            self.assertEqual(
                manifest["query_splits"]["calibration_query_nos"], list(range(20, 100))
            )
            self.assertEqual(
                manifest["query_splits"]["final_query_nos"], list(range(100, 200))
            )
            self.assertEqual(manifest["query_cohort"]["gt_hash"], digest("truth"))
            self.assertEqual(manifest["source_hashes"]["runner"], digest("runner"))
            self.assertEqual(manifest["binary_provenance"]["source_db"]["server_version"], "16.14")
            finalizer._validate_metadata(
                manifest,
                fixture.filters,
                combiner.EXPECTED_TARGETS,
                digest("truth"),
                "search_only",
            )
            provenance = finalizer._provenance(manifest, "faiss_allowlist")
            self.assertEqual(provenance["runner_sha256"], digest("runner"))
            for kind in combiner.ARTIFACTS:
                self.assertEqual(
                    manifest["output_sha256"][kind], combiner.sha256_file(outputs[kind])
                )

    def test_rejects_mismatched_formal_provenance_and_protocol(self):
        mutations = {
            "candidate predicate": lambda value: value["candidate_universe"].update(
                {"predicate": "has_vector"}
            ),
            "candidate rows": lambda value: value["candidate_universe"].update(
                {"observed_rows": 9_979_555}
            ),
            "calibration cohort": lambda value: value["query_splits"].update(
                {"calibration_query_nos": list(range(21, 100))}
            ),
            "final cohort": lambda value: value["query_splits"].update(
                {"final_query_nos": list(range(99, 199))}
            ),
            "targets": lambda value: value.update({"target_recalls": [0.90, 0.95]}),
            "GT hash": lambda value: value["source_hashes"].update(
                {"truth": digest("other truth")}
            ),
            "runner hash": lambda value: value["source_hashes"].update(
                {"runner": digest("other runner")}
            ),
            "latency scope": lambda value: value["execution"].update(
                {"latency": "end_to_end"}
            ),
            "binary provenance": lambda value: value["postgres"].update(
                {"server_version": "15.9"}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(tmp)
                manifest = fixture.manifest(1)
                mutate(manifest)
                fixture.rewrite_manifest(1, manifest)
                with self.assertRaises(combiner.ValidationFailure):
                    combiner.combine(
                        fixture.manifests, fixture.filters_csv, fixture.out_prefix
                    )


if __name__ == "__main__":
    unittest.main()
