from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.hybrid_vector_db.scripts.pgvector_design1_bfs_layout_selectivity_benchmark import (
    SCALAR_INDEXES,
    create_same_graph_scalar_indexes,
    rebuild_same_graph_indexes,
    validate_same_graph_tables,
)


def args() -> SimpleNamespace:
    return SimpleNamespace(
        source_table="source_heap",
        insertion_table="insert_heap",
        insertion_index="insert_hnsw",
        bfs_table="bfs_heap",
        bfs_index="bfs_hnsw",
        maintenance_work_mem="4GB",
        require_full_memory_build=True,
        disable_autovacuum_during_build=True,
        hnsw_build_seed=57,
        hnsw_m=16,
        hnsw_ef_construction=100,
        scalar_maintenance_work_mem="2GB",
        scalar_parallel_workers=2,
    )


class FakeCursor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.statements = []

    def execute(self, statement, params=None):
        self.statements.append((" ".join(statement.split()), params))

    def fetchone(self):
        if not self.responses:
            raise AssertionError("unexpected fetchone")
        return self.responses.pop(0)


class SameGraphLayoutBuilderTests(unittest.TestCase):
    def test_rebuild_uses_one_cursor_and_reseeds_each_layout(self):
        cursor = FakeCursor(
            [
                (10_000_000, 0, 9_999_999),
                (10_000_000, 0, 9_999_999),
                (10_000_000, 0, 9_999_999),
                (0,),
                (0,),
                (True, True),
                (True, True),
            ]
        )

        rebuild_same_graph_indexes(cursor, args())

        sql = [statement for statement, _ in cursor.statements]
        seed_positions = [i for i, statement in enumerate(sql) if statement == "SET hnsw.build_seed = 57"]
        insert_create = next(i for i, statement in enumerate(sql) if "CREATE INDEX \"insert_hnsw\"" in statement)
        bfs_create = next(i for i, statement in enumerate(sql) if "CREATE INDEX \"bfs_hnsw\"" in statement)
        self.assertEqual(len(seed_positions), 2)
        self.assertLess(seed_positions[0], insert_create)
        self.assertLess(insert_create, seed_positions[1])
        self.assertLess(seed_positions[1], bfs_create)
        self.assertIn("SET hnsw.build_seed = -1", sql)
        self.assertIn("ALTER TABLE \"insert_heap\" RESET (autovacuum_enabled)", sql)
        self.assertIn("ALTER TABLE \"bfs_heap\" RESET (autovacuum_enabled)", sql)

    def test_validation_rejects_different_id_spaces_before_rebuild(self):
        cursor = FakeCursor([(10, 0, 9), (10, 0, 9), (9, 0, 8)])

        with self.assertRaisesRegex(RuntimeError, "ID spaces differ"):
            validate_same_graph_tables(cursor, args())

        self.assertFalse(any("DROP INDEX" in statement for statement, _ in cursor.statements))

    def test_validation_rejects_source_content_mismatch(self):
        cursor = FakeCursor(
            [
                (10, 0, 9),
                (10, 0, 9),
                (10, 0, 9),
                (1,),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "logical sample"):
            validate_same_graph_tables(cursor, args())

        logical_checks = [
            statement
            for statement, _ in cursor.statements
            if "to_jsonb(source)" in statement
        ]
        self.assertEqual(len(logical_checks), 1)
        self.assertIn('JOIN "source_heap" AS source', logical_checks[0])

    def test_scalar_only_replaces_nonunique_ids_and_mirrors_predicate_indexes(self):
        cursor = FakeCursor(
            [
                (10_000_000, 0, 9_999_999),
                (10_000_000, 0, 9_999_999),
                (10_000_000, 0, 9_999_999),
                (0,),
                (0,),
                (False,),
                (False,),
            ]
        )

        create_same_graph_scalar_indexes(cursor, args())

        sql = [statement for statement, _ in cursor.statements]
        self.assertIn('DROP INDEX "insert_heap_id_idx"', sql)
        self.assertIn('DROP INDEX "bfs_heap_id_idx"', sql)
        self.assertIn('CREATE UNIQUE INDEX "insert_heap_id_idx" ON "insert_heap" (id)', sql)
        self.assertIn('CREATE UNIQUE INDEX "bfs_heap_id_idx" ON "bfs_heap" (id)', sql)
        scalar_creates = [statement for statement in sql if "CREATE INDEX IF NOT EXISTS" in statement]
        self.assertEqual(len(scalar_creates), 2 * len(SCALAR_INDEXES))


if __name__ == "__main__":
    unittest.main()
