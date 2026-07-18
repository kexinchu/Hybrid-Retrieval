from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import amazon10m_d3_adaptation_lifecycle_benchmark as runner  # noqa: E402


def filters() -> list[runner.FilterSpec]:
    return [runner.FilterSpec(f"f{number}", f"rating = {number}", (f"sql:rating = {number}",), 1, 1.0) for number in range(14)]


def truth(items: list[runner.FilterSpec]) -> dict[tuple[str, int], runner.TruthEntry]:
    return {(item.name, query_no): runner.TruthEntry(item.name, query_no, query_no, tuple(range(10)), 1.0, 0.0)
            for item in items for query_no in range(runner.FORMAL_Q200)}


class FakeSession:
    def __init__(self, profiles: list[dict[str, object]]) -> None:
        self.profiles = iter(profiles)
        self.calls: list[str] = []
        self.value: object = None

    def execute(self, sql: str, params=None) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(sql)
        if "metadata_cache_profile" in sql:
            self.value = json.dumps(next(self.profiles))

    def one(self):  # type: ignore[no-untyped-def]
        return self.value

    def row(self):  # type: ignore[no-untyped-def]
        return self.value

    def all(self):  # type: ignore[no-untyped-def]
        return []


class FragmentStoreSession:
    def __init__(self, audits: list[tuple[object, ...]], records: list[list[tuple[str]]], deleted: int) -> None:
        self.audits = iter(audits)
        self.records = iter(records)
        self.deleted = deleted
        self.calls: list[tuple[str, object]] = []
        self.last_sql = ""

    def execute(self, sql: str, params=None) -> None:  # type: ignore[no-untyped-def]
        self.last_sql = sql
        self.calls.append((sql, params))

    def one(self):  # type: ignore[no-untyped-def]
        raise AssertionError("fragment-store audit uses row(), not one()")

    def row(self):  # type: ignore[no-untyped-def]
        return next(self.audits)

    def all(self):  # type: ignore[no-untyped-def]
        if "DELETE FROM" in self.last_sql:
            return [(1,)] * self.deleted
        return next(self.records)


class Amazon10MD3AdaptationLifecycleTests(unittest.TestCase):
    def test_real_fourteen_amazon_predicates_and_atoms_are_loaded_without_synthesis(self) -> None:
        specs = runner.load_filters(Path(__file__).resolve().parents[1] / "configs" / "amazon10m_selectivity14_filters.csv")
        self.assertEqual(len(specs), 14)
        self.assertEqual(specs[0].predicate, "item_rating_number >= 1000")
        self.assertEqual(specs[-1].atoms, ("sql:main_category = 'Grocery'", "sql:review_text_len >= 500"))
        self.assertFalse(any("%" in spec.predicate for spec in specs))

    def test_trace_is_deterministic_q200_repeating_and_phase_shifted_hot_cold(self) -> None:
        specs = filters()
        exact = truth(specs)
        one = runner.build_trace(specs, exact, requests=200, window_size=20, seed=41)
        two = runner.build_trace(specs, exact, requests=200, window_size=20, seed=41)
        self.assertEqual(one, two)
        self.assertEqual({request.query_no for request in one} <= set(range(200)), True)
        self.assertTrue(any(request.reuse_distance is not None for request in one))
        first = Counter(request.filter_name for request in one[:100])
        second = Counter(request.filter_name for request in one[100:])
        self.assertNotEqual(first.most_common(1), second.most_common(1))
        self.assertEqual(one[99].phase, "steady_hot")
        self.assertEqual(one[100].phase, "phase_shift_hot")

    def test_adaptive_cache_gate_detects_preexisting_and_resets_empty(self) -> None:
        session = FakeSession([
            {"entries": 1, "resident_entries": 1, "resident_bytes": 8},
            {"entries": 0, "resident_entries": 0, "resident_bytes": 0},
        ])
        empty, evidence = runner.adaptive_cache_empty_gate(session)
        self.assertTrue(empty)
        self.assertFalse(evidence["before_reset_empty"])
        self.assertTrue(evidence["after_reset_empty"])
        self.assertEqual(evidence["after_reset"]["entries"], 0)
        self.assertIn("SELECT vector_hnsw_metadata_cache_reset()", session.calls)

    def test_fragment_store_reset_is_targeted_and_epoch_proven(self) -> None:
        before = {"heap_oid": 41, "count": 2, "epoch": 7, "relfilenode": 91,
                  "epoch_proof": {"valid": True, "rows_checked": 2}}
        after = {"heap_oid": 41, "count": 0, "epoch": 7, "relfilenode": 91,
                 "epoch_proof": {"valid": True, "rows_checked": 0, "epoch": 7}}
        proof = runner.validate_fragment_store_reset(before, 2, after)
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["deleted"], 2)
        self.assertEqual(proof["heap_oid"], 41)
        self.assertEqual(proof["epoch_proof"]["epoch"], 7)
        with self.assertRaisesRegex(runner.BenchmarkContractError, "persistent fragment store"):
            runner.validate_fragment_store_reset(before, 1, after)
        with self.assertRaisesRegex(runner.BenchmarkContractError, "epoch proof"):
            runner.validate_fragment_store_reset(before, 2, {**after, "epoch_proof": {"valid": False, "rows_checked": 0}})

    def test_fragment_store_audit_and_clear_use_target_heap_not_global_delete(self) -> None:
        row = json.dumps({"heap_oid": 41, "build_epoch": 7, "relfilenode": 91})
        session = FragmentStoreSession(
            [("pgvector_hnsw_fragment_store", 41, 91, 7, True), ("pgvector_hnsw_fragment_store", 41, 91, 7, True)],
            [[(row,), (row,)], []],
            2,
        )
        proof = runner.clear_fragment_store(session, "public.reviews")
        self.assertEqual(proof["prebuilt_fragments"], 0)
        delete_sql = next(sql for sql, _ in session.calls if "DELETE FROM" in sql)
        self.assertIn("WHERE heap_oid = %s::regclass::oid", delete_sql)
        self.assertIn("RETURNING heap_oid", delete_sql)
        self.assertEqual(proof["heap_oid"], 41)

    def test_old_checkpoint_schema_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "old_checkpoint.json"
            path.write_text(json.dumps({"checkpoint_schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "checkpoint schema"):
                runner.load_checkpoint(path, "unused")

    def test_lifecycle_classifies_creation_reuse_eviction_and_reason(self) -> None:
        created = runner.lifecycle_classification({"entries": 0, "evictions": 0}, {"entries": 1, "evictions": 0},
                                                 {"fragment_builds": 1, "active": True}, admitted=True, reason="admit")
        self.assertTrue(created["fragment_created"])
        reused = runner.lifecycle_classification({"entries": 1, "evictions": 0}, {"entries": 1, "evictions": 1},
                                                {"fragment_store_hits": 1, "active": True}, admitted=True, reason="reuse")
        self.assertTrue(reused["fragment_reused"])
        self.assertTrue(reused["fragment_evicted"])
        self.assertEqual(reused["admission_reason"], "reuse")

    def test_checkpoint_is_atomic_only_for_complete_cross_mode_paired_windows(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_checkpoint.json"
            rows = {mode: [{"window": 0, "request_no": n} for n in range(3)] for mode in runner.MODES}
            runner.write_checkpoint(path, "same", rows, [0], 3)
            restored = runner.load_checkpoint(path, "same")
            self.assertEqual(restored["completed_paired_windows"], [0])
            self.assertEqual(restored["resume_contract"]["cross_process_resume"], "forbidden")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "run-spec"):
                runner.load_checkpoint(path, "different")
            checkpoint_before = path.read_text(encoding="utf-8")
            partial = {mode: list(block) for mode, block in rows.items()}
            partial["adaptive"].pop()
            with self.assertRaisesRegex(runner.BenchmarkContractError, "partial paired window"):
                runner.write_checkpoint(path, "same", partial, [0], 3)
            self.assertEqual(path.read_text(encoding="utf-8"), checkpoint_before)
            tampered = json.loads(checkpoint_before)
            tampered["rows_by_mode"]["stock"][0]["request_no"] = 99
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(runner.BenchmarkContractError, "fingerprint mismatch"):
                runner.load_checkpoint(path, "same")

    def test_open_mode_backends_uses_three_distinct_persistent_connections(self) -> None:
        psycopg = mock.Mock()
        connections = [mock.Mock(name=f"connection_{mode}") for mode in runner.MODES]
        sessions = [mock.Mock(name=f"session_{mode}") for mode in runner.MODES]
        for pid, session in enumerate(sessions, start=101):
            session.one.return_value = pid
        psycopg.connect.side_effect = connections
        with mock.patch.object(runner, "CursorSession", side_effect=sessions), \
             mock.patch.object(runner, "database_provenance", return_value={"database_build_id": "build"}):
            backends = runner.open_mode_backends(psycopg, "postgresql://test", table="reviews", index="reviews_idx")
        self.assertEqual(psycopg.connect.call_count, len(runner.MODES))
        self.assertEqual([backends[mode].session for mode in runner.MODES], sessions)
        self.assertEqual([backends[mode].backend_pid for mode in runner.MODES], [101, 102, 103])
        self.assertEqual(len({id(backends[mode].session) for mode in runner.MODES}), len(runner.MODES))
        runner.close_mode_backends(backends)
        self.assertTrue(all(connection.close.called for connection in connections))

    def test_each_paired_request_rotates_modes_and_retains_each_mode_session_cache(self) -> None:
        args = runner.create_argument_parser().parse_args(["--requests", "6", "--window-size", "3"])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        trace = [runner.Request(number, "steady_hot", number // 3, "f", 0, 99, None) for number in range(6)]
        sessions = {mode: mock.Mock(name=f"{mode}_session") for mode in runner.MODES}
        backends = {
            mode: runner.ModeBackend(mode, mock.Mock(), sessions[mode], index + 1, {"database_build_id": "build"})
            for index, mode in enumerate(runner.MODES)
        }
        calls: list[tuple[str, int, object]] = []
        cache_entries = {mode: 0 for mode in runner.MODES}

        def fake_run_request(session, unused_args, mode, request, unused_filter, unused_truth, unused_provenance, *, adaptive_started_empty, online_materializations_before):  # type: ignore[no-untyped-def]
            self.assertIs(session, sessions[mode])
            self.assertTrue(adaptive_started_empty)
            if mode == "adaptive":
                self.assertEqual(online_materializations_before, int(request.request_no > 0))
            cache_entries[mode] += 1
            calls.append((mode, request.request_no, session))
            return {"mode": mode, "request_no": request.request_no, "window": request.window,
                    "fragment_builds_delta": int(mode == "adaptive" and request.request_no == 0),
                    "materialization_observed": bool(mode == "adaptive" and request.request_no == 0)}

        lifecycle_state = {"online_materializations": 0}
        with mock.patch.object(runner, "run_request", side_effect=fake_run_request):
            first = runner.run_paired_window(
                backends, args, trace, {"f": spec}, {("f", 0): exact}, {"database_build_id": "build"},
                window=0, adaptive_started_empty=True, adaptive_lifecycle_state=lifecycle_state,
            )
            second = runner.run_paired_window(
                backends, args, trace, {"f": spec}, {("f", 0): exact}, {"database_build_id": "build"},
                window=1, adaptive_started_empty=True, adaptive_lifecycle_state=lifecycle_state,
            )

        self.assertEqual(runner.paired_request_mode_order(0), runner.MODES)
        self.assertEqual(runner.paired_request_mode_order(1), ("adaptive", "eager_prebuilt", "stock"))
        self.assertEqual(runner.paired_request_mode_order(2), ("eager_prebuilt", "stock", "adaptive"))
        self.assertEqual(
            [mode for mode, _, _ in calls],
            [
                "stock", "adaptive", "eager_prebuilt",
                "adaptive", "eager_prebuilt", "stock",
                "eager_prebuilt", "stock", "adaptive",
            ] * 2,
        )
        self.assertEqual(cache_entries, {mode: 6 for mode in runner.MODES})
        self.assertEqual(first["stock"][0]["backend_pid"], 1)
        self.assertEqual(second["stock"][0]["paired_request_mode_rank"], 0)
        self.assertEqual(second["adaptive"][0]["paired_request_mode_rank"], 1)
        self.assertEqual(lifecycle_state["online_materializations"], 1)

    def test_paired_execution_rejects_shared_mode_session(self) -> None:
        shared_session = mock.Mock()
        backends = {
            mode: runner.ModeBackend(mode, mock.Mock(), shared_session, index + 1, {"database_build_id": "build"})
            for index, mode in enumerate(runner.MODES)
        }
        with self.assertRaisesRegex(runner.BenchmarkContractError, "independent persistent session/cache"):
            runner.validate_independent_mode_sessions(backends)

    def test_cross_process_resume_fails_closed_before_inputs_or_database(self) -> None:
        args = runner.create_argument_parser().parse_args(["--resume"])
        with mock.patch.object(runner, "load_filters") as load_filters:
            with self.assertRaisesRegex(runner.BenchmarkContractError, "cross-process --resume is disabled"):
                runner.execute_experiment(args)
        load_filters.assert_not_called()
        contract = runner.checkpoint_resume_contract()
        self.assertEqual(contract["cross_process_resume"], "forbidden")
        self.assertEqual(contract["cache_lifecycle_fingerprints"], "audit_only_not_replayable")

    def test_break_even_and_percentiles_follow_formal_rank_rule(self) -> None:
        adaptive = [{"request_no": 0, "e2e_ms": 12.0}, {"request_no": 1, "e2e_ms": 7.0}]
        stock = {0: {"e2e_ms": 10.0}, 1: {"e2e_ms": 10.0}}
        self.assertEqual(runner.break_even_request(adaptive, stock), 1)
        noisy = [{"request_no": 0, "e2e_ms": 9.0}, {"request_no": 1, "e2e_ms": 12.0},
                 {"request_no": 2, "e2e_ms": 8.0}]
        noisy_stock = {request_no: {"e2e_ms": 10.0} for request_no in range(3)}
        self.assertEqual(runner.break_even_request(noisy, noisy_stock), 2)
        self.assertEqual(runner.percentile([1, 2, 3, 4, 5], .95), 5)
        self.assertEqual(runner.percentile([1, 2, 3, 4, 5], .99), 5)
        summary = runner.summary_for_window([
            {"e2e_ms": value, "query_ms": value, "recall_at_10": 1.0, "fragment_reused": False,
             "guidance_checks": 1, "guidance_skips": 0, "cache_resident_bytes_after": 1, "error": ""}
            for value in (1, 2, 3, 4, 5)
        ], bootstrap_samples=20, bootstrap_seed=7)
        self.assertEqual(summary["e2e_p95_ms"], 5.0)
        self.assertEqual(summary["e2e_p99_ms"], 5.0)

    def test_summary_proves_lifecycle_deltas_and_percentile_contract(self) -> None:
        rows = [
            {"e2e_ms": 4.0, "query_ms": 3.0, "recall_at_10": 1.0, "fragment_reused": False,
             "fragment_store_hit_delta": 0, "probe_observed": True, "materialization_observed": True,
             "reuse_observed": False, "refine_observed": False, "evict_observed": False,
             "hidden_prebuilt_fragment_reused": False, "lifecycle_path": "probe->materialize",
             "guidance_checks": 1, "guidance_skips": 0, "cache_resident_bytes_after": 2, "error": ""},
            {"e2e_ms": 2.0, "query_ms": 1.0, "recall_at_10": 1.0, "fragment_reused": True,
             "fragment_store_hit_delta": 1, "probe_observed": False, "materialization_observed": False,
             "reuse_observed": True, "refine_observed": True, "evict_observed": True,
             "hidden_prebuilt_fragment_reused": False, "lifecycle_path": "reuse->refine->evict",
             "guidance_checks": 1, "guidance_skips": 1, "cache_resident_bytes_after": 1, "error": ""},
        ]
        summary = runner.summary_for_window(rows, bootstrap_samples=20, bootstrap_seed=7)
        self.assertEqual(summary["e2e_p50_ms"], 2.0)
        self.assertEqual(summary["e2e_p95_ms"], 4.0)
        self.assertEqual(summary["e2e_p99_ms"], 4.0)
        self.assertEqual(summary["fragment_store_hit_delta"], 1)
        self.assertEqual(summary["lifecycle_event_counts"]["materialize"], 1)
        self.assertEqual(summary["hidden_prebuilt_reuse_count"], 0)

    def test_run_spec_names_reused_q200_trace_and_rejects_cracking_claim(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        spec = runner.make_run_spec(args, {}, {}, [])
        self.assertEqual(spec["workload_manifest_name"], runner.FORMAL_WORKLOAD_MANIFEST_NAME)
        self.assertEqual(spec["unique_query_vectors"], runner.FORMAL_Q200)
        self.assertFalse(spec["database_cracking"])
        self.assertIn("10,000-request trace", spec["trace_contract"])
        self.assertIn("vectors are reused", spec["trace_contract"])
        self.assertNotIn("q10000", json.dumps(spec))

    def test_invalidation_catches_missing_rows_recall_planner_and_build_mismatch(self) -> None:
        trace = [runner.Request(0, "steady_hot", 0, "f", 0, 0, None)]
        base = {"request_no": 0, "e2e_ms": 1.0, "recall_at_10": 1.0, "database_build_id": "build", "profile_build_id": "build", "error": ""}
        rows = {"stock": [base], "adaptive": [{**base, "recall_at_10": .5, "activation_attempted": True,
                                                  "planner_proof_required": True,
                                                  "planner_proof_verified": False, "adaptive_cache_started_empty": False,
                                                  "database_build_id": "other"}], "eager_prebuilt": []}
        errors = runner.validate_artifact(rows, trace, recall_delta=.01, provenance={"database_build_id": "build"})
        self.assertIn("profile_build_mismatch:adaptive", errors)
        self.assertIn("planner_proof_failure:adaptive", errors)
        self.assertIn("recall_regression:adaptive", errors)
        self.assertIn("preexisting_adaptive_cache", errors)
        self.assertIn("missing_or_duplicate_windows:eager_prebuilt", errors)

    def test_adaptive_request_always_enters_extension_state_machine(self) -> None:
        args = runner.create_argument_parser().parse_args([])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        request = runner.Request(0, "steady_hot", 0, "f", 0, 99, None)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        session = mock.Mock()
        cache = {"entries": 0, "resident_entries": 0, "resident_bytes": 0}
        with mock.patch.object(runner, "json_profile", side_effect=[cache, cache, cache, cache]), \
             mock.patch.object(runner, "activate", return_value=({"active": False, "adaptive_state": "probing"}, 2.5)) as activate, \
             mock.patch.object(runner, "run_search", return_value=(list(range(10)), {}, "", 7.5)):
            row = runner.run_request(
                session, args, "adaptive", request, spec, exact,
                {"database_build_id": "build"}, adaptive_started_empty=True,
            )
        self.assertEqual(activate.call_args.args[3], "adaptive")
        self.assertTrue(row["activation_attempted"])
        self.assertFalse(row["guidance_active"])
        self.assertFalse(row["planner_proof_required"])
        self.assertEqual(row["adaptive_state"], "probing")
        self.assertEqual(row["activation_ms"], 2.5)
        self.assertEqual(row["query_ms"], 7.5)
        self.assertEqual(row["e2e_ms"], 10.0)

    def test_eager_control_uses_explicit_nonadaptive_fragment_kind(self) -> None:
        args = runner.create_argument_parser().parse_args(["--eager-kind", "page"])
        spec = runner.FilterSpec("f", "rating = 1", ("sql:rating = 1",), 10, 1.0)
        request = runner.Request(0, "steady_hot", 0, "f", 0, 99, None)
        exact = runner.TruthEntry("f", 0, 99, tuple(range(10)), 1.0, 0.0)
        session = mock.Mock()
        cache = {"entries": 1, "resident_entries": 1, "resident_bytes": 8}
        with mock.patch.object(runner, "json_profile", side_effect=[cache, cache, cache, cache]), \
             mock.patch.object(runner, "activate", return_value=({"active": True}, 1.0)) as activate, \
             mock.patch.object(runner, "run_search", return_value=(list(range(10)), {"planner_proof_succeeded": True}, "", 2.0)):
            row = runner.run_request(
                session, args, "eager_prebuilt", request, spec, exact,
                {"database_build_id": "build"}, adaptive_started_empty=True,
            )
        self.assertEqual(activate.call_args.args[3], "page")
        self.assertTrue(row["guidance_active"])
        self.assertTrue(row["planner_proof_required"])

    def test_runtime_guc_and_timer_contract_use_real_extension_names(self) -> None:
        args = runner.create_argument_parser().parse_args(["--d3-page-min-skip-rate", "0.25"])
        session = mock.Mock()
        with mock.patch.object(runner, "json_profile", return_value={}):
            runner.configure(session, args, "adaptive")
        statements = [call.args[0] for call in session.execute.call_args_list]
        self.assertIn("SET hnsw.d3_page_min_skip_rate = 0.25", statements)
        self.assertFalse(any("d3_refine_skip_rate" in statement for statement in statements))

    def test_dry_run_reads_no_inputs_or_database_and_debug_override_is_labeled(self) -> None:
        missing = Path("/definitely/missing")
        with mock.patch.object(runner, "execute_experiment") as execute:
            self.assertEqual(runner.main(["--dry-run", "--filters-csv", str(missing), "--truth", str(missing),
                                          "--requests", "200", "--window-size", "20"]), 0)
        execute.assert_not_called()
        payload = runner.dry_run_payload(runner.create_argument_parser().parse_args(["--requests", "200", "--window-size", "20"]))
        self.assertTrue(payload["debug_override_labeled_non_formal"])


if __name__ == "__main__":
    unittest.main()
