from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.hybrid_vector_db.scripts import (
    prepare_amazon10m_valid_embedding_indexes as prepare,
)


SHA256 = "a" * 64
BUILD_ID = "sqlens-v11-formal-test"


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "table": prepare.DEFAULT_TABLE,
        "source_index": prepare.DEFAULT_SOURCE_INDEX,
        "clone_index": prepare.DEFAULT_CLONE_INDEX,
        "constraint_name": prepare.DEFAULT_CONSTRAINT,
        "stage": "all",
        "expected_sqlens_build_id": BUILD_ID,
        "expected_vector_so_sha256": SHA256,
        "expected_rows": prepare.EXPECTED_ROWS,
        "maintenance_work_mem": "64GB",
        "build_seed": 57,
        "proof_output": Path("proof.json"),
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def table_state(relfilenode: int = 200) -> prepare.RelationState:
    return prepare.RelationState(
        prepare.DEFAULT_TABLE,
        oid=100,
        relfilenode=relfilenode,
        physical_relfilenode=relfilenode,
    )


def source_state(
    options: tuple[str, ...] = ("m=16", "ef_construction=64"),
    *,
    comment: str | None = None,
    oid: int = 300,
    relfilenode: int = 301,
) -> prepare.IndexState:
    contract = prepare.source_build_contract(args(), table_state())
    return prepare.IndexState(
        name=prepare.DEFAULT_SOURCE_INDEX,
        oid=oid,
        relfilenode=relfilenode,
        physical_relfilenode=relfilenode,
        heap_oid=100,
        heap_relfilenode=200,
        valid=True,
        ready=True,
        live=True,
        access_method="hnsw",
        unique=False,
        primary=False,
        key_attributes=1,
        total_attributes=1,
        indexed_column="embedding",
        opclass="vector_l2_ops",
        predicate="embedding_valid",
        reloptions=options,
        comment=prepare.provenance_comment(contract) if comment is None else comment,
        definition="CREATE INDEX source USING hnsw ... WHERE embedding_valid",
    )


def clone_state(
    source: prepare.IndexState | None = None,
    *,
    comment: str | None = None,
) -> prepare.IndexState:
    source = source or source_state()
    contract = prepare.clone_build_contract(args(), table_state(), source)
    return prepare.IndexState(
        name=prepare.DEFAULT_CLONE_INDEX,
        oid=400,
        relfilenode=401,
        physical_relfilenode=401,
        heap_oid=100,
        heap_relfilenode=200,
        valid=True,
        ready=True,
        live=True,
        access_method="hnsw",
        unique=False,
        primary=False,
        key_attributes=1,
        total_attributes=1,
        indexed_column="embedding",
        opclass="vector_l2_ops",
        predicate="(embedding_valid)",
        reloptions=("ef_construction=64", "m=16"),
        comment=prepare.provenance_comment(contract) if comment is None else comment,
        definition="CREATE INDEX clone USING hnsw ... WHERE embedding_valid",
    )


class Transaction:
    def __init__(self, owner: "FakeConnection") -> None:
        self.owner = owner

    def __enter__(self) -> "Transaction":
        self.owner.transactions_started += 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.owner.transactions_committed += 1
        else:
            self.owner.transactions_rolled_back += 1
        return False


class FakeCursor:
    def __init__(self, responses: list[object] | None = None, rowcount: int = 0) -> None:
        self.responses = list(responses or [])
        self.rowcount = rowcount
        self.statements: list[tuple[str, object]] = []
        self.closed = False

    def execute(self, statement: object, params: object = None) -> None:
        rendered = (
            statement.as_string(None)
            if hasattr(statement, "as_string")
            else str(statement)
        )
        self.statements.append((" ".join(rendered.split()), params))

    def fetchone(self) -> object:
        if not self.responses:
            raise AssertionError("unexpected fetchone")
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor | None = None) -> None:
        self._cursor = cursor or FakeCursor()
        self.transactions_started = 0
        self.transactions_committed = 0
        self.transactions_rolled_back = 0
        self.entered = False

    def __enter__(self) -> "FakeConnection":
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def transaction(self) -> Transaction:
        return Transaction(self)


class NamingAndDryRunTests(unittest.TestCase):
    def test_default_indexes_are_new_distinct_and_identifier_safe(self) -> None:
        self.assertNotEqual(prepare.DEFAULT_SOURCE_INDEX, prepare.DEFAULT_CLONE_INDEX)
        self.assertNotIn(prepare.DEFAULT_SOURCE_INDEX, prepare.LEGACY_INDEXES)
        self.assertNotIn(prepare.DEFAULT_CLONE_INDEX, prepare.LEGACY_INDEXES)
        for name in (prepare.DEFAULT_SOURCE_INDEX, prepare.DEFAULT_CLONE_INDEX):
            _, relation = prepare.parse_qualified_name(name)
            self.assertLessEqual(len(relation), 63)

    def test_create_sql_is_partial_l2_hnsw_with_fixed_build_parameters(self) -> None:
        sql = prepare.hnsw_create_sql(
            prepare.DEFAULT_SOURCE_INDEX, prepare.DEFAULT_TABLE
        )
        self.assertIn("USING hnsw (embedding vector_l2_ops)", sql)
        self.assertTrue(sql.startswith('CREATE INDEX "amazon10m_embedding_valid_hnsw_source_idx" '))
        self.assertNotIn('CREATE INDEX "public".', sql)
        self.assertIn("WITH (m = 16, ef_construction = 64)", sql)
        self.assertTrue(sql.endswith("WHERE embedding_valid"))
        self.assertNotIn("CONCURRENTLY", sql)

    def test_comment_sql_quotes_identifier_and_literal_without_bind_parameter(self) -> None:
        statement = prepare.hnsw_comment_sql(
            "public.safe_index", "provenance with ' quoted content"
        ).as_string(None)
        self.assertEqual(
            statement,
            'COMMENT ON INDEX "public"."safe_index" IS '
            "'provenance with '' quoted content'",
        )

    def test_dry_run_does_not_resolve_db_config_connect_or_write_output(self) -> None:
        parsed = args(dry_run=True, proof_output=Path("must-not-exist.json"))
        connector = mock.Mock(side_effect=AssertionError("must not connect"))
        with (
            mock.patch.object(
                prepare, "pg_config_from_env", side_effect=AssertionError("must not read config")
            ),
            mock.patch.object(
                prepare, "write_json_atomic", side_effect=AssertionError("must not write")
            ),
            mock.patch.object(
                prepare, "require_d2_graph_proof", side_effect=AssertionError("must not prove")
            ),
        ):
            plan = prepare.run(parsed, connect=connector)
        connector.assert_not_called()
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["database_connected"])
        self.assertFalse(plan["input_files_read"])
        self.assertEqual(plan["clone_source"], prepare.DEFAULT_SOURCE_INDEX)

    def test_validate_args_rejects_same_source_and_clone_name(self) -> None:
        with self.assertRaisesRegex(prepare.PreparationError, "must be different"):
            prepare.validate_args(args(clone_index=prepare.DEFAULT_SOURCE_INDEX))

    def test_create_and_args_reject_cross_schema_index(self) -> None:
        with self.assertRaisesRegex(prepare.PreparationError, "schemas must match"):
            prepare.hnsw_create_sql("other.source_idx", prepare.DEFAULT_TABLE)
        with self.assertRaisesRegex(prepare.PreparationError, "same schema"):
            prepare.validate_args(args(source_index="other.source_idx"))


class ColumnHygieneTests(unittest.TestCase):
    def test_embedding_valid_statistics_refreshes_and_requires_catalog_row(self) -> None:
        cursor = FakeCursor(responses=[(0.0, 2.0, "{t,f}", "{0.998,0.002}")])
        stats = prepare.embedding_valid_statistics(
            cursor, prepare.DEFAULT_TABLE, refresh=True
        )
        self.assertTrue(stats["refreshed"])
        self.assertTrue(cursor.statements[0][0].startswith("ANALYZE"))
        self.assertEqual(stats["n_distinct"], 2.0)

    def test_metadata_only_column_add_requires_postgresql_11_or_newer(self) -> None:
        cursor = FakeCursor(responses=[(160004,)])
        self.assertEqual(prepare.require_metadata_only_column_add(cursor), 160004)
        old_cursor = FakeCursor(responses=[(100023,)])
        with self.assertRaisesRegex(prepare.PreparationError, "PostgreSQL 11"):
            prepare.require_metadata_only_column_add(old_cursor)

    def test_column_validation_requires_boolean_not_null_true_default(self) -> None:
        valid = prepare.ColumnState("boolean", True, "true", True)
        self.assertEqual(prepare.validate_column_definition(valid), valid)
        for invalid in (
            prepare.ColumnState("integer", True, "true", True),
            prepare.ColumnState("boolean", False, "true", True),
            prepare.ColumnState("boolean", True, "false", True),
        ):
            with self.assertRaises(prepare.PreparationError):
                prepare.validate_column_definition(invalid)

    def test_constraint_validation_accepts_only_exact_norm_equivalence(self) -> None:
        valid = prepare.ConstraintState(
            prepare.DEFAULT_CONSTRAINT,
            True,
            False,
            "(embedding_valid = (vector_norm(embedding) > (0)::double precision))",
        )
        self.assertEqual(
            prepare.validate_constraint_definition(valid, require_validated=True), valid
        )
        wrong = prepare.ConstraintState(
            prepare.DEFAULT_CONSTRAINT,
            True,
            False,
            "embedding_valid = (vector_norm(embedding) >= 0)",
        )
        with self.assertRaisesRegex(prepare.PreparationError, "definition mismatch"):
            prepare.validate_constraint_definition(wrong, require_validated=True)

    def test_unvalidated_constraint_is_accepted_only_during_resume_validation(self) -> None:
        state = prepare.ConstraintState(
            prepare.DEFAULT_CONSTRAINT,
            False,
            False,
            "embedding_valid = (vector_norm(embedding) > 0::double precision)",
        )
        prepare.validate_constraint_definition(state, require_validated=False)
        with self.assertRaisesRegex(prepare.PreparationError, "not validated"):
            prepare.validate_constraint_definition(state, require_validated=True)

    def test_ensure_column_is_metadata_only_and_updates_only_inconsistent_rows(self) -> None:
        cursor = FakeCursor(rowcount=17)
        connection = FakeConnection(cursor)
        relation = table_state()
        valid_column = prepare.ColumnState("boolean", True, "true", True)
        pending_constraint = prepare.ConstraintState(
            prepare.DEFAULT_CONSTRAINT,
            False,
            False,
            "embedding_valid = (vector_norm(embedding) > 0::double precision)",
        )
        valid_constraint = prepare.ConstraintState(
            prepare.DEFAULT_CONSTRAINT,
            True,
            False,
            pending_constraint.expression,
        )
        with (
            mock.patch.object(
                prepare,
                "relation_state",
                side_effect=[relation, relation, relation],
            ),
            mock.patch.object(
                prepare, "column_state", side_effect=[None, valid_column]
            ),
            mock.patch.object(
                prepare,
                "constraint_state",
                side_effect=[None, pending_constraint, valid_constraint],
            ),
            mock.patch.object(
                prepare,
                "table_counts",
                return_value={
                    "total_rows": prepare.EXPECTED_ROWS,
                    "valid_rows": prepare.EXPECTED_ROWS - 17,
                    "invalid_rows": 17,
                    "inconsistent_rows": 0,
                },
            ),
            mock.patch.object(
                prepare,
                "embedding_valid_statistics",
                return_value={"refreshed": True, "n_distinct": 2.0},
            ),
        ):
            report = prepare.ensure_embedding_valid_column(
                connection, cursor, args()
            )

        statements = [statement for statement, _ in cursor.statements]
        update = next(statement for statement in statements if statement.startswith("UPDATE"))
        self.assertIn(
            "WHERE embedding_valid IS DISTINCT FROM (vector_norm(embedding) > 0)",
            update,
        )
        self.assertIn(
            'ALTER TABLE "public"."amazon_grocery_reviews_10m_pgvector" '
            "ADD COLUMN embedding_valid boolean NOT NULL DEFAULT true",
            statements,
        )
        self.assertTrue(any("ADD CONSTRAINT" in statement for statement in statements))
        self.assertTrue(any("VALIDATE CONSTRAINT" in statement for statement in statements))
        self.assertFalse(any("DROP " in statement for statement in statements))
        self.assertEqual(connection.transactions_committed, 4)
        self.assertEqual(report["corrected_inconsistent_rows"], 17)
        self.assertTrue(report["metadata_only_add_verified"])

    def test_metadata_add_fails_if_table_relfilenode_changes_before_any_update(self) -> None:
        cursor = FakeCursor(rowcount=99)
        connection = FakeConnection(cursor)
        with (
            mock.patch.object(
                prepare,
                "relation_state",
                side_effect=[table_state(200), table_state(201)],
            ),
            mock.patch.object(prepare, "column_state", return_value=None),
        ):
            with self.assertRaisesRegex(prepare.PreparationError, "rewrote/replaced"):
                prepare.ensure_embedding_valid_column(connection, cursor, args())
        statements = [statement for statement, _ in cursor.statements]
        self.assertFalse(any(statement.startswith("UPDATE") for statement in statements))

    def test_existing_column_resume_does_not_claim_to_observe_metadata_add(self) -> None:
        cursor = FakeCursor(rowcount=0)
        connection = FakeConnection(cursor)
        relation = table_state()
        valid_column = prepare.ColumnState("boolean", True, "true", True)
        valid_constraint = prepare.ConstraintState(
            prepare.DEFAULT_CONSTRAINT,
            True,
            False,
            "embedding_valid = (vector_norm(embedding) > 0::double precision)",
        )
        with (
            mock.patch.object(
                prepare, "relation_state", side_effect=[relation, relation]
            ),
            mock.patch.object(prepare, "column_state", return_value=valid_column),
            mock.patch.object(
                prepare, "constraint_state", return_value=valid_constraint
            ),
            mock.patch.object(
                prepare,
                "table_counts",
                return_value={
                    "total_rows": prepare.EXPECTED_ROWS,
                    "valid_rows": prepare.EXPECTED_ROWS,
                    "invalid_rows": 0,
                    "inconsistent_rows": 0,
                },
            ),
            mock.patch.object(
                prepare,
                "embedding_valid_statistics",
                return_value={"refreshed": True, "n_distinct": 2.0},
            ),
        ):
            report = prepare.ensure_embedding_valid_column(connection, cursor, args())
        self.assertIsNone(report["metadata_only_add_verified"])
        self.assertFalse(report["column_added"])
        self.assertFalse(
            any("ADD COLUMN" in statement for statement, _ in cursor.statements)
        )

    def test_counts_fail_closed_on_wrong_total_or_inconsistent_behavior(self) -> None:
        with self.assertRaisesRegex(prepare.PreparationError, "row count mismatch"):
            prepare.validate_counts(
                {
                    "total_rows": 9,
                    "valid_rows": 9,
                    "invalid_rows": 0,
                    "inconsistent_rows": 0,
                },
                10,
            )
        with self.assertRaisesRegex(prepare.PreparationError, "disagrees"):
            prepare.validate_counts(
                {
                    "total_rows": 10,
                    "valid_rows": 9,
                    "invalid_rows": 1,
                    "inconsistent_rows": 1,
                },
                10,
            )


class IndexPreparationTests(unittest.TestCase):
    def test_source_build_uses_insertion_order_and_empty_clone_source(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        source = source_state()
        with mock.patch.object(
            prepare, "index_state", side_effect=[None, source]
        ):
            observed, created, contract = prepare.build_source_index(
                connection, cursor, args(), table_state()
            )
        statements = cursor.statements
        sql = [statement for statement, _ in statements]
        self.assertTrue(created)
        self.assertEqual(observed, source)
        self.assertEqual(contract["build_page_order"], "insertion")
        self.assertFalse(contract["require_full_memory_build"])
        self.assertIn(
            "SELECT set_config('hnsw.build_page_order', 'insertion', true)", sql
        )
        self.assertIn(
            "SELECT set_config('hnsw.require_full_memory_build', 'off', true)", sql
        )
        self.assertIn("SELECT set_config('hnsw.clone_source', '', true)", sql)
        self.assertTrue(any(statement.startswith("CREATE INDEX") for statement in sql))
        self.assertTrue(any(statement.startswith("COMMENT ON INDEX") for statement in sql))
        self.assertTrue(
            all(
                params is None
                for statement, params in statements
                if statement.startswith("COMMENT ON INDEX")
            )
        )
        self.assertFalse(any("DROP INDEX" in statement for statement in sql))
        self.assertEqual(connection.transactions_committed, 1)

    def test_clone_build_uses_source_graph_full_memory_and_bfs_order(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        source = source_state()
        clone = clone_state(source)
        with mock.patch.object(
            prepare, "index_state", side_effect=[None, clone]
        ):
            observed, created, contract = prepare.build_clone_index(
                connection, cursor, args(), table_state(), source
            )
        sql = [statement for statement, _ in cursor.statements]
        self.assertTrue(created)
        self.assertEqual(observed, clone)
        self.assertEqual(contract["source_oid"], source.oid)
        self.assertEqual(contract["source_relfilenode"], source.relfilenode)
        self.assertEqual(contract["clone_source"], prepare.DEFAULT_SOURCE_INDEX)
        self.assertTrue(contract["require_full_memory_build"])
        self.assertIn(
            "SELECT set_config('hnsw.require_full_memory_build', 'on', true)", sql
        )
        self.assertIn("SELECT set_config('hnsw.build_page_order', 'bfs', true)", sql)
        clone_source_call = next(
            (statement, params)
            for statement, params in cursor.statements
            if "set_config('hnsw.clone_source'" in statement
        )
        self.assertEqual(clone_source_call[1], (prepare.DEFAULT_SOURCE_INDEX,))
        self.assertFalse(any("hnsw.build_seed" in statement for statement in sql))
        self.assertFalse(any("DROP INDEX" in statement for statement in sql))
        self.assertEqual(connection.transactions_committed, 1)

    def test_valid_existing_source_is_reused_without_ddl(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        existing = source_state()
        with mock.patch.object(prepare, "index_state", return_value=existing):
            observed, created, _ = prepare.build_source_index(
                connection, cursor, args(), table_state()
            )
        self.assertEqual(observed, existing)
        self.assertFalse(created)
        self.assertEqual(connection.transactions_started, 0)
        self.assertEqual(cursor.statements, [])

    def test_definition_mismatch_refuses_implicit_drop_or_rebuild(self) -> None:
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        mismatched = source_state(options=("m=32", "ef_construction=64"))
        with mock.patch.object(prepare, "index_state", return_value=mismatched):
            with self.assertRaisesRegex(
                prepare.PreparationError, "refusing to drop or rebuild"
            ):
                prepare.build_source_index(connection, cursor, args(), table_state())
        self.assertEqual(connection.transactions_started, 0)
        self.assertFalse(any("DROP" in statement for statement, _ in cursor.statements))

    def test_missing_provenance_rejects_an_ambiguous_independent_build(self) -> None:
        ambiguous = source_state(comment="")
        with self.assertRaisesRegex(prepare.PreparationError, "provenance mismatch"):
            prepare.validate_index_state(
                ambiguous,
                table=table_state(),
                expected_contract=prepare.source_build_contract(args(), table_state()),
                role="source",
            )

    def test_clone_provenance_rejects_replaced_source_relation(self) -> None:
        original_source = source_state(oid=300, relfilenode=301)
        clone = clone_state(original_source)
        replacement_source = source_state(oid=310, relfilenode=311)
        expected = prepare.clone_build_contract(args(), table_state(), replacement_source)
        with self.assertRaisesRegex(prepare.PreparationError, "provenance mismatch"):
            prepare.validate_index_state(
                clone,
                table=table_state(),
                expected_contract=expected,
                role="clone",
            )

    def test_index_validation_requires_same_heap_and_partial_predicate(self) -> None:
        source = source_state()
        wrong_heap = prepare.IndexState(
            **{**source.__dict__, "heap_oid": 999}
        )
        expected = prepare.source_build_contract(args(), table_state())
        with self.assertRaisesRegex(prepare.PreparationError, "definition/provenance"):
            prepare.validate_index_state(
                wrong_heap,
                table=table_state(),
                expected_contract=expected,
                role="source",
            )
        wrong_predicate = prepare.IndexState(
            **{**source.__dict__, "predicate": "true"}
        )
        with self.assertRaises(prepare.PreparationError):
            prepare.validate_index_state(
                wrong_predicate,
                table=table_state(),
                expected_contract=expected,
                role="source",
            )


class ProvenanceAndProofTests(unittest.TestCase):
    def test_exact_binary_provenance_requires_build_id_and_server_sha(self) -> None:
        cursor = FakeCursor(
            [(BUILD_ID, "/usr/lib/postgresql/17/lib/vector.so", SHA256)]
        )
        observed = prepare.exact_sqlens_provenance(cursor, BUILD_ID, SHA256)
        self.assertTrue(observed["exact_match"])
        self.assertEqual(observed["observed_vector_so_sha256"], SHA256)

        wrong = FakeCursor(
            [("sqlens-v11-wrong", "/usr/lib/postgresql/17/lib/vector.so", SHA256)]
        )
        with self.assertRaisesRegex(prepare.PreparationError, "build ID mismatch"):
            prepare.exact_sqlens_provenance(wrong, BUILD_ID, SHA256)

        wrong_sha = FakeCursor(
            [(BUILD_ID, "/usr/lib/postgresql/17/lib/vector.so", "b" * 64)]
        )
        with self.assertRaisesRegex(prepare.PreparationError, "SHA256 mismatch"):
            prepare.exact_sqlens_provenance(wrong_sha, BUILD_ID, SHA256)

    def test_hnsw_capability_gate_fails_if_clone_guc_is_missing(self) -> None:
        cursor = FakeCursor([("insertion",), ("off",), (None,)])
        with self.assertRaisesRegex(prepare.PreparationError, "clone_source"):
            prepare.hnsw_build_capabilities(cursor)

    def test_atomic_json_replaces_target_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proof.json"
            path.write_text('{"old": true}\n', encoding="utf-8")
            prepare.write_json_atomic(path, {"proof_contract": "v2", "valid": True})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"proof_contract": "v2", "valid": True},
            )
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_proof_stage_calls_canonical_gate_and_atomically_publishes(self) -> None:
        parsed = args(stage="proof")
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        source = source_state()
        clone = clone_state(source)
        canonical = {
            "proof_contract": "sqlens_same_heap_same_logical_graph_physical_layout_v2",
            "source_index": prepare.DEFAULT_SOURCE_INDEX,
            "clone_index": prepare.DEFAULT_CLONE_INDEX,
            "comparison": {"format": "sqlens-hnsw-compare-v2"},
        }
        column_report = {
            "row_counts": {
                "total_rows": prepare.EXPECTED_ROWS,
                "valid_rows": prepare.EXPECTED_ROWS,
                "invalid_rows": 0,
                "inconsistent_rows": 0,
            }
        }
        connector = mock.Mock(return_value=connection)
        with (
            mock.patch.object(
                prepare, "pg_config_from_env", return_value=SimpleNamespace(conninfo="db")
            ),
            mock.patch.object(
                prepare, "exact_sqlens_provenance", return_value={"exact_match": True}
            ),
            mock.patch.object(
                prepare, "require_metadata_only_column_add", return_value=160004
            ),
            mock.patch.object(
                prepare, "hnsw_build_capabilities", return_value={"clone": "available"}
            ),
            mock.patch.object(prepare, "acquire_advisory_lock"),
            mock.patch.object(prepare, "release_advisory_lock"),
            mock.patch.object(
                prepare, "verify_embedding_valid_column", return_value=column_report
            ),
            mock.patch.object(prepare, "relation_state", return_value=table_state()),
            mock.patch.object(
                prepare,
                "verify_source",
                return_value=(
                    source,
                    prepare.source_build_contract(parsed, table_state()),
                ),
            ),
            mock.patch.object(
                prepare,
                "verify_clone",
                return_value=(
                    clone,
                    prepare.clone_build_contract(parsed, table_state(), source),
                ),
            ),
            mock.patch.object(
                prepare, "require_d2_graph_proof", return_value=canonical
            ) as proof_gate,
            mock.patch.object(prepare, "write_json_atomic") as writer,
        ):
            payload = prepare.run(parsed, connect=connector)

        proof_gate.assert_called_once_with(
            cursor, prepare.DEFAULT_SOURCE_INDEX, prepare.DEFAULT_CLONE_INDEX
        )
        writer.assert_called_once()
        self.assertEqual(writer.call_args.args[0], parsed.proof_output)
        self.assertTrue(payload["artifact_valid"])
        self.assertEqual(payload["proof_contract"], canonical["proof_contract"])
        self.assertEqual(
            payload["preparation"]["source_clone_graph_diff"],
            canonical["comparison"],
        )
        self.assertEqual(
            payload["preparation"]["column_hygiene"]["row_counts"]["total_rows"],
            prepare.EXPECTED_ROWS,
        )
        self.assertTrue(cursor.closed)

    def test_verify_stage_recomputes_proof_without_writing_artifact(self) -> None:
        parsed = args(stage="verify")
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        source = source_state()
        clone = clone_state(source)
        connector = mock.Mock(return_value=connection)
        with (
            mock.patch.object(
                prepare, "pg_config_from_env", return_value=SimpleNamespace(conninfo="db")
            ),
            mock.patch.object(
                prepare, "exact_sqlens_provenance", return_value={"exact_match": True}
            ),
            mock.patch.object(
                prepare, "require_metadata_only_column_add", return_value=160004
            ),
            mock.patch.object(prepare, "hnsw_build_capabilities", return_value={}),
            mock.patch.object(prepare, "acquire_advisory_lock"),
            mock.patch.object(prepare, "release_advisory_lock"),
            mock.patch.object(
                prepare,
                "verify_embedding_valid_column",
                return_value={"row_counts": {"total_rows": prepare.EXPECTED_ROWS}},
            ),
            mock.patch.object(prepare, "relation_state", return_value=table_state()),
            mock.patch.object(
                prepare,
                "verify_source",
                return_value=(
                    source,
                    prepare.source_build_contract(parsed, table_state()),
                ),
            ),
            mock.patch.object(
                prepare,
                "verify_clone",
                return_value=(
                    clone,
                    prepare.clone_build_contract(parsed, table_state(), source),
                ),
            ),
            mock.patch.object(
                prepare,
                "require_d2_graph_proof",
                return_value={"proof_contract": "v2", "comparison": {}},
            ) as proof_gate,
            mock.patch.object(prepare, "write_json_atomic") as writer,
        ):
            payload = prepare.run(parsed, connect=connector)
        proof_gate.assert_called_once()
        writer.assert_not_called()
        self.assertTrue(payload["artifact_valid"])


if __name__ == "__main__":
    unittest.main()
