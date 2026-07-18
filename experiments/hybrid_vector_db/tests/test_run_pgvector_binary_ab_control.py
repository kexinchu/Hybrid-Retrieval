from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.hybrid_vector_db.scripts import run_pgvector_binary_ab_control as controller


SQLENS_DIGEST = "a" * 64
INITIAL_BYTES = b"initial-vector-so"
INITIAL_DIGEST = hashlib.sha256(INITIAL_BYTES).hexdigest()


def result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def controller_args(temporary: str) -> object:
    graph_identity = Path(temporary) / "graph-identity.json"
    graph_identity.write_text(
        json.dumps({
            "same_heap": True,
            "logical_equal": True,
            "logical_digest": "graph-digest",
        }),
        encoding="utf-8",
    )
    args = controller.build_parser().parse_args([
        "--server-container", "pgvector",
        "--official-vector-so", f"{temporary}/official-vector.so",
        "--sqlens-vector-so", f"{temporary}/sqlens-vector.so",
        "--sqlens-vector-so-sha256", SQLENS_DIGEST,
        "--official-vector-source-tag", "upstream-v0.8.2",
        "--official-vector-source-commit", "official-commit",
        "--sqlens-vector-source-tag", "sqlens-v11-test",
        "--sqlens-vector-source-commit", "sqlens-commit",
        "--official-vector-build-recipe", "make official",
        "--official-vector-compiler-flags=-O3",
        "--sqlens-vector-build-recipe", "make sqlens",
        "--sqlens-vector-compiler-flags=-O3",
        "--official-vector-source-repo", temporary,
        "--sqlens-vector-source-repo", temporary,
        "--graph-identity-json", str(graph_identity),
        "--run-uuid", "test-run",
        "--data-epoch", "amazon10m-v1",
        "--manifest", f"{temporary}/controller.json",
        "--out-dir", f"{temporary}/results",
        "--filters-csv", f"{temporary}/filters.csv",
        "--truth-csv", f"{temporary}/truth.csv",
        "--tag", "test-tag",
        "--target-recalls", "0.90,0.95,0.99",
        "--screen-repeats", "2",
        "--verification-repeats", "3",
        "--final-repeats", "4",
        "--pg-isready-poll-seconds", "0",
        "--pg-isready-timeout-seconds", "1",
    ])
    return args


class RunPgvectorBinaryAbControlTests(unittest.TestCase):
    def test_seeded_final_schedule_is_audited_ab_ba_and_balanced(self) -> None:
        schedule, audit = controller.counterbalanced_final_schedule("run-123", 17)

        self.assertEqual(len(schedule), 4)
        self.assertEqual(audit["pair_orders"], ["AB", "BA"])
        self.assertEqual(audit["arm_counts"], {"official": 2, "sqlens_disabled": 2})
        self.assertEqual(
            audit["first_in_pair_counts"], {"official": 1, "sqlens_disabled": 1}
        )
        self.assertTrue(audit["seeded_balance_verified"])
        self.assertEqual(
            controller.counterbalanced_final_schedule("run-123", 17),
            (schedule, audit),
        )

    def test_new_manifest_claim_never_overwrites_an_old_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text('{"run_uuid":"old"}\n', encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                controller.claim_controller_manifest(
                    path, {"run_uuid": "new"}, resume=False
                )

            self.assertEqual(json.loads(path.read_text()), {"run_uuid": "old"})

    def test_cross_arm_finalizer_requires_exact_42_cells_and_publishes_paired_ci(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filters = [f"f{i:02d}" for i in range(14)]
            targets = [0.90, 0.95, 0.99]
            shared = {
                "run_uuid": "run-123",
                "status": "arm_ready",
                "artifact_valid": True,
                "formal_design": {
                    "formal_family": "off",
                    "filters": filters,
                    "target_recalls": targets,
                    "cell_count": 42,
                },
                "source_hashes": {
                    "runner_sha256": "1" * 64,
                    "filters_sha256": "2" * 64,
                    "truth_sha256": "3" * 64,
                    "graph_identity_sha256": "9" * 64,
                    "hybrid_sql_sha256_by_filter": {name: "4" * 64 for name in filters},
                },
                "database_fingerprint": {
                    "system_identifier": "cluster",
                    "database_oid": 7,
                    "table_oid": 10,
                    "table_relfilenode": 11,
                    "index_oid": 12,
                    "index_relfilenode": 13,
                    "indexdef_sha256": "5" * 64,
                    "data_epoch": "amazon10m-v1",
                    "source_clone_graph_identity": {
                        "proof": {"same_heap": True, "logical_equal": True},
                        "source_index": {"index_oid": 12},
                        "clone_index": {"index_oid": 14},
                    },
                },
                "query_splits": {
                    "screen": {"first": 0, "last": 19, "queries": 20},
                    "verification": {"first": 20, "last": 99, "queries": 80},
                    "final": {"first": 100, "last": 199, "queries": 100},
                },
                "schedule_contract": {
                    "schedule_seed": 19,
                    "final_query_nos": list(range(100, 200)),
                    "final_repeats": 2,
                    "warmup_spec_sha256": "6" * 64,
                },
                "warmup_invocations": [
                    {
                        "execution_stage": "calibration",
                        "final_block": None,
                        "warmup_spec_sha256": "6" * 64,
                    },
                    {
                        "execution_stage": "final",
                        "final_block": 0,
                        "warmup_spec_sha256": "6" * 64,
                    },
                    {
                        "execution_stage": "final",
                        "final_block": 1,
                        "warmup_spec_sha256": "6" * 64,
                    },
                ],
                "settings_audit": {
                    "hnsw_guc_audit": {
                        "all_nonstock_forced_safe": True,
                        "unhandled_nonstock_gucs": [],
                    }
                },
                "guc_block_audits": [
                    {
                        "phase": phase,
                        "filter_name": name,
                        "after_sha256": "a" * 64,
                    }
                    for phase in ("screen", "verification", "final", "final")
                    for name in filters
                ],
                "target_selection": {
                    name: {
                        format(target, "g"): {
                            "status": "selected",
                            "config_label": f"{name}-cfg",
                        }
                        for target in targets
                    }
                    for name in filters
                },
            }
            manifests = []
            for implementation, latency, digest in (
                ("official", 20.0, controller.OFFICIAL_VECTOR_SO_SHA256),
                ("sqlens_disabled", 10.0, SQLENS_DIGEST),
            ):
                raw_path = root / f"{implementation}.csv"
                with raw_path.open("w", encoding="utf-8", newline="") as target:
                    writer = csv.DictWriter(
                        target,
                        fieldnames=[
                            "run_uuid", "implementation", "execution_stage", "final_block",
                            "phase", "query_split", "filter_name", "query_no",
                            "repeat", "config_label", "latency_ms", "valid", "error",
                            "recall_at_10", "truth_self_excluded", "pair_key",
                            "measurement_key",
                        ],
                    )
                    writer.writeheader()
                    for name in filters:
                        for query_no in range(100, 200):
                            for repeat in range(2):
                                writer.writerow({
                                    "run_uuid": "run-123",
                                    "implementation": implementation,
                                    "execution_stage": "final",
                                    "final_block": repeat,
                                    "phase": "final",
                                    "query_split": "final",
                                    "filter_name": name,
                                    "query_no": query_no,
                                    "repeat": repeat,
                                    "config_label": f"{name}-cfg",
                                    "latency_ms": latency,
                                    "valid": "True",
                                    "error": "",
                                    "recall_at_10": 1.0,
                                    "truth_self_excluded": "True",
                                    "pair_key": f"final|{name}|q{query_no}|r{repeat}",
                                    "measurement_key": (
                                        f"{implementation}|final|{name}|q{query_no}|r{repeat}|{name}-cfg"
                                    ),
                                })
                manifest = dict(shared) | {
                    "implementation": implementation,
                    "server_binary_provenance": {
                        "vector_so_sha256": digest,
                        "expected_vector_so_sha256": digest,
                        "binary_hash_matches_expected": True,
                        "server_image_id": f"sha256:{'8' * 64}",
                    },
                    "source_provenance": {
                        "source_tag": f"{implementation}-tag",
                        "source_commit": f"{implementation}-commit",
                        "build_recipe": "make",
                        "compiler_flags": "-O3",
                        "dirty_diff_sha256": "7" * 64,
                        "source_tree": f"/src/{implementation}",
                    },
                    "outputs": {"raw": str(raw_path)},
                    "output_hashes": {
                        "raw": {
                            "path": str(raw_path),
                            "sha256": controller.sha256_file(raw_path),
                        }
                    },
                }
                manifest_path = root / f"{implementation}.json"
                controller.atomic_write_json(manifest_path, manifest)
                manifests.append(manifest_path)

            final_schedule = [
                {"implementation": "official", "final_block": 0},
                {"implementation": "sqlens_disabled", "final_block": 0},
                {"implementation": "sqlens_disabled", "final_block": 1},
                {"implementation": "official", "final_block": 1},
            ]
            controller_manifest = {
                "run_uuid": "run-123",
                "calibration_order": ["official", "sqlens_disabled"],
                "final_schedule": final_schedule,
                "seeded_balance_audit": {
                    "seeded_balance_verified": True,
                    "pair_orders": ["AB", "BA"],
                },
                "runner_runs": [
                    {
                        "run_uuid": "run-123",
                        "implementation": implementation,
                        "execution_stage": stage,
                        "final_block": block,
                        "exit_code": 0,
                        "staging_manifest": {
                            "resume_append_only_audit": {"passed": True},
                            "server_vector_so_sha256": (
                                controller.OFFICIAL_VECTOR_SO_SHA256
                                if implementation == "official"
                                else SQLENS_DIGEST
                            ),
                        },
                    }
                    for stage, implementation, block in [
                        ("calibration", "official", None),
                        ("calibration", "sqlens_disabled", None),
                        ("final", "official", 0),
                        ("final", "sqlens_disabled", 0),
                        ("final", "sqlens_disabled", 1),
                        ("final", "official", 1),
                    ]
                ],
            }
            publish = root / "published" / "run-123.json"
            report = controller.finalize_ab_artifacts(
                manifests,
                publish,
                bootstrap_samples=100,
                bootstrap_seed=23,
                controller_manifest=controller_manifest,
            )

            self.assertTrue(publish.exists())
            self.assertEqual(report["paired_gate"]["cell_count"], 42)
            self.assertTrue(report["paired_gate"]["passed"])
            self.assertEqual(len(report["cells"]), 42)
            self.assertEqual(report["cells"][0]["speedup_mean"], 2.0)
            self.assertEqual(report["cells"][0]["speedup_lcb95"], 2.0)

            unconfirmed = json.loads(manifests[1].read_text())
            unconfirmed["target_selection"][filters[0]]["0.9"] = {
                "status": "no_verified_config_meets_target",
                "config_label": "",
            }
            controller.atomic_write_json(manifests[1], unconfirmed)
            staging_only = root / "published" / "staging-only.json"
            with self.assertRaisesRegex(
                controller.FinalizationError, "staging-only"
            ):
                controller.finalize_ab_artifacts(
                    manifests,
                    staging_only,
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                    controller_manifest=controller_manifest,
                )
            self.assertFalse(staging_only.exists())

            controller.atomic_write_json(
                manifests[1],
                unconfirmed
                | {
                    "target_selection": shared["target_selection"],
                },
            )
            tampered = json.loads(manifests[1].read_text())
            tampered["database_fingerprint"]["index_oid"] = 99
            controller.atomic_write_json(manifests[1], tampered)
            rejected = root / "published" / "rejected.json"
            with self.assertRaises(controller.FinalizationError):
                controller.finalize_ab_artifacts(
                    manifests,
                    rejected,
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                    controller_manifest=controller_manifest,
                )
            self.assertFalse(rejected.exists())

    def test_dry_run_performs_zero_subprocess_or_manifest_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(controller.subprocess, "run") as run, \
                mock.patch.object(controller, "atomic_write_json") as write:
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                self.assertEqual(controller.main([
                    "--dry-run",
                    "--out-dir", f"{temporary}/does-not-exist",
                    "--manifest", f"{temporary}/does-not-exist/manifest.json",
                ]), 0)
            run.assert_not_called()
            write.assert_not_called()
            self.assertFalse(Path(temporary, "does-not-exist").exists())
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["file_access"])
            self.assertFalse(payload["docker_access"])
            self.assertFalse(payload["database_access"])

    def test_host_digest_mismatch_is_rejected_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vector.so"
            path.write_bytes(b"wrong")
            source = {
                "host_path": str(path),
                "expected_digest": "b" * 64,
            }
            with self.assertRaises(controller.DigestMismatchError):
                controller.validate_host_binary(source)

    def test_active_session_gate_rejects_by_default_and_allows_explicit_override(self) -> None:
        args = SimpleNamespace(
            server_container="pgvector",
            pg_host="127.0.0.1",
            pg_port=55432,
            pg_user="postgres",
            pg_database="hybrid_vector",
            allow_active_sessions=False,
        )
        with mock.patch.object(controller.subprocess, "run", return_value=result(stdout="2\n")):
            with self.assertRaises(controller.ActiveSessionsError):
                controller.enforce_active_session_gate(args)
        args.allow_active_sessions = True
        with mock.patch.object(controller.subprocess, "run", return_value=result(stdout="2\n")):
            evidence = controller.enforce_active_session_gate(args)
        self.assertEqual(evidence["active_sessions_excluding_gate"], 2)
        self.assertTrue(evidence["allow_active_sessions"])

    def test_successful_order_calibrates_both_then_runs_ab_ba_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = controller_args(temporary)
            Path(args.official_vector_so).write_bytes(b"official")
            Path(args.sqlens_vector_so).write_bytes(b"sqlens")
            calls: list[list[str]] = []
            current_digest = {"value": INITIAL_DIGEST}

            def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
                command = list(argv)
                calls.append(command)
                if command[:2] == ["docker", "exec"] and command[3:5] == ["pg_config", "--pkglibdir"]:
                    return result(stdout="/usr/lib/postgresql/16/lib\n")
                if command[:2] == ["docker", "cp"]:
                    source, destination = command[2], command[3]
                    if destination.startswith("pgvector:"):
                        if "official-vector.so" in source:
                            current_digest["value"] = controller.OFFICIAL_VECTOR_SO_SHA256
                        elif "sqlens-vector.so" in source:
                            current_digest["value"] = SQLENS_DIGEST
                        else:
                            current_digest["value"] = INITIAL_DIGEST
                    else:
                        Path(destination).write_bytes(INITIAL_BYTES)
                    return result()
                if command[:2] == ["docker", "exec"] and command[3] == "sha256sum":
                    return result(stdout=f"{current_digest['value']}  /usr/lib/postgresql/16/lib/vector.so\n")
                if "psql" in command:
                    return result(stdout="0\n")
                if command[:4] == ["docker", "exec", "pgvector", "pg_isready"]:
                    return result()
                if command[:2] == ["docker", "exec"] and command[3:4] in (["chmod"], ["mv"]):
                    return result()
                if command[:2] == ["docker", "restart"]:
                    return result()
                if len(command) > 1 and command[1] == str(controller.RUNNER_PATH):
                    implementation = command[command.index("--implementation") + 1]
                    stage = command[command.index("--execution-stage") + 1]
                    block = (
                        int(command[command.index("--final-block") + 1])
                        if "--final-block" in command
                        else None
                    )
                    status = (
                        "calibration_complete"
                        if stage == "calibration"
                        else ("final_in_progress" if block == 0 else "arm_ready")
                    )
                    controller.atomic_write_json(
                        controller.arm_manifest_path(args, implementation),
                        {
                            "status": status,
                            "checkpoint_spec_sha256": "c" * 64,
                            "resume_append_only_audit": {"passed": True},
                            "server_binary_provenance": {
                                "vector_so_sha256": current_digest["value"]
                            },
                            "target_selection_sha256": "d" * 64,
                        },
                    )
                    return result()
                raise AssertionError(f"unexpected command: {command}")

            def fake_digest(path: Path) -> str:
                if "official-vector.so" in str(path):
                    return controller.OFFICIAL_VECTOR_SO_SHA256
                if "sqlens-vector.so" in str(path):
                    return SQLENS_DIGEST
                return INITIAL_DIGEST

            with mock.patch.object(controller.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(controller, "sha256_file", side_effect=fake_digest), \
                    mock.patch.object(
                        controller,
                        "finalize_ab_artifacts",
                        return_value={"paired_gate": {"passed": True, "cell_count": 42}},
                    ):
                manifest = controller.run_controller(args)

            self.assertEqual(
                [item["implementation"] for item in manifest["switches"]],
                [
                    "official", "sqlens_disabled",
                    "official", "sqlens_disabled",
                    "sqlens_disabled", "official",
                    "restore_initial",
                ],
            )
            self.assertEqual(
                [item["implementation"] for item in manifest["runner_runs"]],
                [
                    "official", "sqlens_disabled",
                    "official", "sqlens_disabled",
                    "sqlens_disabled", "official",
                ],
            )
            runner_positions = [index for index, command in enumerate(calls) if len(command) > 1 and command[1] == str(controller.RUNNER_PATH)]
            self.assertEqual(len(runner_positions), 6)
            self.assertLess(runner_positions[0], runner_positions[1])
            self.assertIn("--target-recalls", calls[runner_positions[0]])
            self.assertEqual(
                calls[runner_positions[0]][calls[runner_positions[0]].index("--target-recalls") + 1],
                calls[runner_positions[1]][calls[runner_positions[1]].index("--target-recalls") + 1],
            )
            stages = [
                command[command.index("--execution-stage") + 1]
                for command in (calls[position] for position in runner_positions)
            ]
            self.assertEqual(stages, ["calibration", "calibration", "final", "final", "final", "final"])
            self.assertEqual(manifest["seeded_balance_audit"]["pair_orders"], ["AB", "BA"])
            self.assertEqual(manifest["restoration"]["status"], "verified")
            self.assertNotIn("password", json.dumps(manifest).lower())

    def test_runner_failure_still_restores_and_records_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = controller_args(temporary)
            Path(args.official_vector_so).write_bytes(b"official")
            Path(args.sqlens_vector_so).write_bytes(b"sqlens")
            calls: list[list[str]] = []
            current_digest = {"value": INITIAL_DIGEST}

            def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
                command = list(argv)
                calls.append(command)
                if command[:2] == ["docker", "exec"] and command[3:5] == ["pg_config", "--pkglibdir"]:
                    return result(stdout="/usr/lib/postgresql/16/lib\n")
                if command[:2] == ["docker", "cp"]:
                    source, destination = command[2], command[3]
                    if destination.startswith("pgvector:"):
                        current_digest["value"] = (
                            controller.OFFICIAL_VECTOR_SO_SHA256
                            if "official-vector.so" in source
                            else INITIAL_DIGEST
                        )
                    else:
                        Path(destination).write_bytes(INITIAL_BYTES)
                    return result()
                if command[:2] == ["docker", "exec"] and command[3] == "sha256sum":
                    return result(stdout=f"{current_digest['value']}  vector.so\n")
                if "psql" in command:
                    return result(stdout="0\n")
                if command[:4] == ["docker", "exec", "pgvector", "pg_isready"]:
                    return result()
                if command[:2] == ["docker", "restart"]:
                    return result()
                if command[:2] == ["docker", "exec"] and command[3:4] in (["chmod"], ["mv"]):
                    return result()
                if len(command) > 1 and command[1] == str(controller.RUNNER_PATH):
                    return result(returncode=17, stderr="runner failed")
                raise AssertionError(f"unexpected command: {command}")

            def fake_digest(path: Path) -> str:
                if "official-vector.so" in str(path):
                    return controller.OFFICIAL_VECTOR_SO_SHA256
                if "sqlens-vector.so" in str(path):
                    return SQLENS_DIGEST
                return INITIAL_DIGEST

            with mock.patch.object(controller.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(controller, "sha256_file", side_effect=fake_digest):
                with self.assertRaises(controller.RunnerFailedError):
                    controller.run_controller(args)

            manifest = json.loads(Path(args.manifest).read_text())
            self.assertEqual(manifest["runner_runs"][0]["exit_code"], 17)
            self.assertEqual(
                manifest["runner_runs"][0]["child_logs"]["stderr"]["tail"],
                "runner failed",
            )
            self.assertTrue(
                Path(
                    manifest["runner_runs"][0]["child_logs"]["stderr"]["path"]
                ).is_file()
            )
            self.assertEqual(manifest["restoration"]["status"], "verified")
            self.assertEqual(manifest["switches"][-1]["implementation"], "restore_initial")
            self.assertEqual(manifest["switches"][-1]["status"], "installed_and_verified")
            self.assertTrue(any(command[:2] == ["docker", "restart"] for command in calls))

    def test_restoration_failure_takes_priority_over_runner_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = controller_args(temporary)
            Path(args.official_vector_so).write_bytes(b"official")
            Path(args.sqlens_vector_so).write_bytes(b"sqlens")
            runner_error = controller.RunnerFailedError(
                "runner failed",
                {
                    "implementation": "official",
                    "execution_stage": "calibration",
                    "final_block": None,
                    "exit_code": 17,
                },
            )

            def switch(_args, manifest, _path, source, recovery=False):
                if recovery:
                    raise controller.ControllerError("restore exploded")
                manifest["switches"].append({
                    "implementation": source["implementation"],
                    "replacement_attempted": True,
                    "recovery": False,
                    "status": "installed_and_verified",
                })
                return manifest["switches"][-1]

            with mock.patch.object(
                    controller, "validate_host_binary",
                    side_effect=[controller.OFFICIAL_VECTOR_SO_SHA256, SQLENS_DIGEST],
                ), mock.patch.object(
                    controller, "enforce_active_session_gate", return_value={"active": 0}
                ), mock.patch.object(
                    controller, "discover_vector_so", return_value="/pkglib/vector.so"
                ), mock.patch.object(
                    controller, "docker_copy"
                ), mock.patch.object(
                    controller, "fsync_existing_file"
                ), mock.patch.object(
                    controller, "sha256_file", return_value=INITIAL_DIGEST
                ), mock.patch.object(
                    controller, "switch_binary", side_effect=switch
                ), mock.patch.object(
                    controller, "run_external_runner", side_effect=runner_error
                ):
                with self.assertRaises(controller.RecoveryFailedError) as raised:
                    controller.run_controller(args)

            self.assertIs(raised.exception.original_error, runner_error)
            manifest = json.loads(Path(args.manifest).read_text())
            self.assertEqual(manifest["status"], "recovery_failed")
            self.assertEqual(manifest["restoration"]["status"], "failed")

    def test_preflight_busy_gate_never_restarts_or_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = controller_args(temporary)
            Path(args.official_vector_so).write_bytes(b"official")
            Path(args.sqlens_vector_so).write_bytes(b"sqlens")
            calls: list[list[str]] = []

            def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
                command = list(argv)
                calls.append(command)
                if "psql" in command:
                    return result(stdout="1\n")
                raise AssertionError(f"unexpected command after busy gate: {command}")

            with mock.patch.object(controller.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(
                        controller, "validate_host_binary",
                        side_effect=[controller.OFFICIAL_VECTOR_SO_SHA256, SQLENS_DIGEST],
                    ):
                with self.assertRaises(controller.ActiveSessionsError):
                    controller.run_controller(args)

            self.assertFalse(Path(args.manifest).exists())
            self.assertFalse(any(command[:2] == ["docker", "restart"] for command in calls))

    def test_post_install_digest_failure_still_restores_initial_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = controller_args(temporary)
            Path(args.official_vector_so).write_bytes(b"official")
            Path(args.sqlens_vector_so).write_bytes(b"sqlens")
            current_digest = {"value": INITIAL_DIGEST}
            server_digest_reads = 0

            def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
                nonlocal server_digest_reads
                command = list(argv)
                if "psql" in command:
                    return result(stdout="0\n")
                if command[:2] == ["docker", "exec"] and command[3:5] == ["pg_config", "--pkglibdir"]:
                    return result(stdout="/usr/lib/postgresql/16/lib\n")
                if command[:2] == ["docker", "cp"]:
                    source, destination = command[2], command[3]
                    if destination.startswith("pgvector:"):
                        current_digest["value"] = (
                            controller.OFFICIAL_VECTOR_SO_SHA256
                            if "official-vector.so" in source else INITIAL_DIGEST
                        )
                    else:
                        Path(destination).write_bytes(INITIAL_BYTES)
                    return result()
                if command[:2] == ["docker", "exec"] and command[3:4] in (["chmod"], ["mv"]):
                    return result()
                if command[:2] == ["docker", "restart"]:
                    return result()
                if command[:4] == ["docker", "exec", "pgvector", "pg_isready"]:
                    return result()
                if command[:2] == ["docker", "exec"] and command[3] == "sha256sum":
                    server_digest_reads += 1
                    digest = "f" * 64 if server_digest_reads == 1 else current_digest["value"]
                    return result(stdout=f"{digest}  vector.so\n")
                raise AssertionError(f"unexpected command: {command}")

            def fake_digest(path: Path) -> str:
                if "official-vector.so" in str(path):
                    return controller.OFFICIAL_VECTOR_SO_SHA256
                if "sqlens-vector.so" in str(path):
                    return SQLENS_DIGEST
                return INITIAL_DIGEST

            with mock.patch.object(controller.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(controller, "sha256_file", side_effect=fake_digest):
                with self.assertRaises(controller.DigestMismatchError):
                    controller.run_controller(args)

            manifest = json.loads(Path(args.manifest).read_text())
            self.assertEqual(manifest["switches"][0]["status"], "failed")
            self.assertTrue(manifest["switches"][0]["binary_replaced"])
            self.assertEqual(manifest["switches"][-1]["implementation"], "restore_initial")
            self.assertEqual(manifest["restoration"]["status"], "verified")


if __name__ == "__main__":
    unittest.main()
