import csv
import io
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCHEMA = ROOT / "sql" / "amazon10m_sql_native_schema.sql"
sys.path.insert(0, str(SCRIPTS))

import prepare_amazon10m_sql_native as workload  # noqa: E402


class Amazon10MSqlNativeWorkloadTests(unittest.TestCase):
    def test_temporal_targets_are_exactly_the_required_fourteen(self):
        self.assertEqual(
            workload.TEMPORAL_TARGET_PCTS,
            tuple(
                Decimal(value)
                for value in (
                    "0.2",
                    "0.5",
                    "1",
                    "2",
                    "5",
                    "10",
                    "15",
                    "20",
                    "25",
                    "30",
                    "35",
                    "40",
                    "45",
                    "50",
                )
            ),
        )

    def test_reference_buckets_use_real_timestamps_and_record_ties(self):
        timestamps = [1000] * 4 + [2000] * 6 + [5000] * 10
        buckets = workload.reference_temporal_buckets(
            timestamps, targets=(Decimal("20"), Decimal("25"), Decimal("50"))
        )
        self.assertEqual(
            [
                (
                    bucket.target_count,
                    bucket.as_of,
                    bucket.achieved_count,
                    bucket.acl_visible_count,
                )
                for bucket in buckets
            ],
            [(4, 1000, 4, 20), (5, 2000, 10, 20), (10, 2000, 10, 20)],
        )

    def test_target_count_uses_exact_decimal_ceiling(self):
        self.assertEqual(workload.exact_target_count(500, Decimal("0.2")), 1)
        self.assertEqual(workload.exact_target_count(501, Decimal("0.2")), 2)
        self.assertEqual(workload.exact_target_count(10_000_000, Decimal("0.5")), 50_000)

    def test_workload_populations_must_not_collapse(self):
        self.assertEqual(
            workload.validate_distinct_workload_populations(1000, 650, 200),
            {"acl_only": 1000, "grant_temporal_selectivity": 650,
             "fact_temporal_selectivity": 200},
        )
        for counts in ((1000, 1000, 200), (1000, 200, 200), (1000, 0, 200)):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(RuntimeError, "workload populations"):
                    workload.validate_distinct_workload_populations(*counts)

    def test_tenant_policy_prefers_real_store_then_real_asin(self):
        store_tenant = workload.tenant_identity("  Real Store  ", "ASIN-1")
        asin_tenant = workload.tenant_identity("  ", "ASIN-1")

        self.assertEqual(store_tenant[1:], ("store", "Real Store"))
        self.assertTrue(store_tenant[0].startswith("store:"))
        self.assertEqual(asin_tenant[1:], ("asin", "ASIN-1"))
        self.assertTrue(asin_tenant[0].startswith("asin:"))
        self.assertNotEqual(store_tenant[0], asin_tenant[0])
        with self.assertRaises(ValueError):
            workload.tenant_identity("", "")

    def test_csv_header_validation_uses_builder_contract_without_scanning_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = Path(tmpdir) / "valid.csv"
            invalid_path = Path(tmpdir) / "invalid.csv"
            with valid_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(workload.CSV_COLUMNS)
                writer.writerow(["not parsed"])
            with invalid_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(reversed(workload.CSV_COLUMNS))

            self.assertEqual(workload.validate_csv_header(valid_path), workload.CSV_COLUMNS)
            with self.assertRaisesRegex(ValueError, "CSV header mismatch"):
                workload.validate_csv_header(invalid_path)

    def test_dry_run_neither_opens_csv_nor_connects(self):
        output = io.StringIO()
        missing_csv = "/definitely/not/present/amazon.csv"
        with (
            mock.patch.object(workload, "run_prepare") as prepare,
            mock.patch.object(workload, "run_validate_only") as validate,
            mock.patch("sys.stdout", output),
        ):
            status = workload.main(["--dry-run", "--csv", missing_csv])

        self.assertEqual(status, 0)
        prepare.assert_not_called()
        validate.assert_not_called()
        self.assertIn("vector_table=public.amazon_grocery_reviews_10m_pgvector", output.getvalue())
        self.assertIn("valid_from=real-csv-timestamp", output.getvalue())

    def test_schema_defines_relations_indexes_and_non_owner_rls(self):
        sql_text = SCHEMA.read_text(encoding="utf-8")
        for table in (
            "amazon_review_facts",
            "amazon_product_dim",
            "amazon_tenants",
            "amazon_principal_tenant_grants",
            "amazon_sql_native_buckets",
        ):
            self.assertIn(f"public.{table}", sql_text)
        self.assertIn(
            "ALTER TABLE public.amazon_review_facts ENABLE ROW LEVEL SECURITY",
            sql_text,
        )
        self.assertNotIn(
            "ALTER TABLE public.amazon_product_dim ENABLE ROW LEVEL SECURITY",
            sql_text,
        )
        self.assertIn("CREATE ROLE amazon10m_sql_native_reader", sql_text)
        self.assertIn("NOBYPASSRLS", sql_text)
        self.assertIn("FOR SELECT TO amazon10m_sql_native_reader", sql_text)
        lower_sql = sql_text.lower()
        self.assertIn("derived benchmark acl", lower_sql)
        self.assertIn("real store values", lower_sql)
        self.assertIn("synthetic end times are forbidden", lower_sql)
        self.assertNotIn("FORCE ROW LEVEL SECURITY", sql_text)

    def test_rls_policy_is_acl_only_and_temporal_semantics_stay_in_workload_sql(self):
        sql_text = SCHEMA.read_text(encoding="utf-8")
        policy_start = sql_text.index("CREATE POLICY amazon_review_facts_acl_select")
        policy_end = sql_text.index(";", policy_start)
        policy = sql_text[policy_start:policy_end]
        self.assertNotIn("app.as_of", policy)
        self.assertNotIn("valid_from", policy)
        self.assertNotIn("valid_to", policy)
        self.assertIn("CURRENT_USER", policy)
        self.assertIn("grant_row.can_read", policy)

    def test_schema_and_prepare_install_data_epoch_tracking(self):
        sql_text = SCHEMA.read_text(encoding="utf-8").lower()
        self.assertIn("amazon_sql_native_relation_epoch", sql_text)
        self.assertIn("amazon_sql_native_bump_relation_epoch", sql_text)
        self.assertIn("after insert or update or delete or truncate", sql_text)
        normalized = " ".join(workload.INSTALL_VECTOR_EPOCH_TRIGGER_SQL.lower().split())
        self.assertIn("create trigger amazon_sql_native_epoch_bump", normalized)
        self.assertIn("execute function public.amazon_sql_native_bump_relation_epoch()", normalized)

    def test_load_sql_preserves_csv_id_and_timestamp_without_modulo_or_fake_time(self):
        normalized = " ".join(workload.LOAD_RELATIONS_SQL.lower().split())
        self.assertIn("select id, btrim(user_id), btrim(parent_asin), timestamp", normalized)
        self.assertIn("null::bigint", normalized)
        self.assertNotRegex(normalized, r"\b(id|review_id)\s*%")
        self.assertNotIn("generate_series", normalized)
        self.assertNotIn("random()", normalized)

    def test_bucket_sql_is_acl_visible_and_orders_real_valid_from(self):
        normalized = " ".join(workload.INSERT_BUCKETS_SQL.lower().split())
        self.assertIn("principal_tenant_grants", normalized)
        self.assertIn("grant_row.principal_name", normalized)
        self.assertIn("group by valid_from", normalized)
        self.assertIn("over (order by valid_from)", normalized)
        self.assertIn("cumulative_count", normalized)
        self.assertNotIn("review_id", normalized)

    def test_acl_selection_uses_volume_coverage_not_fixed_tenant_count(self):
        normalized = " ".join(workload.INSERT_GRANTS_SQL.lower().split())
        self.assertIn("sum(review_count) over", normalized)
        self.assertIn("cumulative_count - review_count", normalized)
        self.assertIn("ceil(total_count * %s::numeric / 100.0)", normalized)
        self.assertNotIn(" limit ", f" {normalized} ")
        args = workload.create_argument_parser().parse_args([])
        self.assertEqual(args.acl_coverage_pct, Decimal("60"))

    def test_explicit_transaction_commits_or_rolls_back(self):
        cursor = mock.MagicMock()
        cursor.__enter__.return_value = cursor
        connection = mock.MagicMock()
        connection.cursor.return_value = cursor

        result = workload.run_transaction(
            connection, read_only=False, operation=lambda _cursor: "prepared"
        )
        self.assertEqual(result, "prepared")
        self.assertEqual(
            [call.args[0] for call in cursor.execute.call_args_list],
            ["BEGIN ISOLATION LEVEL REPEATABLE READ", "COMMIT"],
        )

        cursor.reset_mock()

        def fail(_cursor):
            raise RuntimeError("synthetic failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            workload.run_transaction(connection, read_only=False, operation=fail)
        self.assertEqual(
            [call.args[0] for call in cursor.execute.call_args_list],
            ["BEGIN ISOLATION LEVEL REPEATABLE READ", "ROLLBACK"],
        )

        cursor.reset_mock()
        workload.run_transaction(connection, read_only=True, operation=lambda _cursor: None)
        self.assertEqual(
            [call.args[0] for call in cursor.execute.call_args_list],
            ["BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY", "ROLLBACK"],
        )


if __name__ == "__main__":
    unittest.main()
