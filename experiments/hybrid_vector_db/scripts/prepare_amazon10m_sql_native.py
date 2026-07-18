from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Iterable, Sequence

from common_pg import pg_config_from_env, require_psycopg


DEFAULT_CSV = Path(
    "data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"
)
DEFAULT_SCHEMA_SQL = Path(__file__).resolve().parents[1] / "sql" / "amazon10m_sql_native_schema.sql"
DEFAULT_VECTOR_TABLE = "public.amazon_grocery_reviews_10m_pgvector"
READER_ROLE = "amazon10m_sql_native_reader"
DEFAULT_PRINCIPAL = "amazon10m_sql_native_benchmark"
DEFAULT_EXPECTED_ROWS = 10_000_000
GRANT_TEMPORAL_TARGET_PCT = Decimal("20")
FACT_TEMPORAL_TARGET_PCT = Decimal("5")
MAX_TEMPORAL_TARGET_ERROR_PCT_POINTS = Decimal("0.1")

CSV_COLUMNS = (
    "id",
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
    "review_text_len",
    "store",
    "main_category",
    "category_id",
    "price",
    "has_price",
    "item_avg_rating",
    "item_rating_number",
)

TEMPORAL_TARGET_PCTS = tuple(
    Decimal(value)
    for value in ("0.2", "0.5", "1", "2", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50")
)

ROLE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class ReferenceBucket:
    target_pct: Decimal
    as_of: int
    acl_visible_count: int
    target_count: int
    achieved_count: int

    @property
    def achieved_pct(self) -> Decimal:
        return Decimal(100) * Decimal(self.achieved_count) / Decimal(self.acl_visible_count)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def percentage(value: str) -> Decimal:
    parsed = Decimal(value)
    if parsed <= 0 or parsed > 100:
        raise argparse.ArgumentTypeError("percentage must be in (0, 100]")
    return parsed


def parse_qualified_name(value: str) -> tuple[str, ...]:
    parts = tuple(value.split("."))
    if len(parts) not in (1, 2) or any(not IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise argparse.ArgumentTypeError(
            "table names must be unquoted identifiers in table or schema.table form"
        )
    return tuple(part.lower() for part in parts)


def principal_name(value: str) -> str:
    if not ROLE_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "principal must be a lowercase PostgreSQL role identifier"
        )
    return value


def tenant_identity(store: str | None, parent_asin: str) -> tuple[str, str, str]:
    """Mirror the SQL benchmark tenant rule for tests and audit tooling."""
    clean_store = (store or "").strip()
    source_kind = "store" if clean_store else "asin"
    source_value = clean_store if clean_store else parent_asin.strip()
    if not source_value:
        raise ValueError("tenant source requires a real store or parent_asin")
    digest = hashlib.md5(source_value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{source_kind}:{digest}", source_kind, source_value


def exact_target_count(population: int, target_pct: Decimal) -> int:
    target = Decimal(population) * target_pct / Decimal(100)
    return max(1, int(target.to_integral_value(rounding=ROUND_CEILING)))


def validate_distinct_workload_populations(
    acl_only: int, grant_temporal: int, fact_temporal: int
) -> dict[str, int]:
    counts = {
        "acl_only": int(acl_only),
        "grant_temporal_selectivity": int(grant_temporal),
        "fact_temporal_selectivity": int(fact_temporal),
    }
    if (
        any(value <= 0 or value > counts["acl_only"] for value in counts.values())
        or len(set(counts.values())) != len(counts)
    ):
        raise RuntimeError(
            "SQL-native workload populations collapsed or are invalid: "
            + json.dumps(counts, sort_keys=True)
        )
    return counts


def validate_temporal_target(
    name: str,
    observed: int,
    population: int,
    target_pct: Decimal,
    *,
    max_error_pct_points: Decimal = MAX_TEMPORAL_TARGET_ERROR_PCT_POINTS,
) -> Decimal:
    if population <= 0 or observed <= 0 or observed > population:
        raise RuntimeError(
            f"{name} temporal population is invalid: "
            f"observed={observed} population={population}"
        )
    actual_pct = Decimal(100) * Decimal(observed) / Decimal(population)
    if abs(actual_pct - target_pct) > max_error_pct_points:
        raise RuntimeError(
            f"{name} temporal selectivity missed its target: target={target_pct}% "
            f"actual={actual_pct}% tolerance={max_error_pct_points} percentage points"
        )
    return actual_pct


def reference_temporal_buckets(
    visible_timestamps: Iterable[int],
    targets: Sequence[Decimal] = TEMPORAL_TARGET_PCTS,
) -> list[ReferenceBucket]:
    """Small-data reference for PostgreSQL percentile_disc and timestamp ties."""
    timestamps = sorted(int(value) for value in visible_timestamps)
    if not timestamps:
        raise ValueError("ACL-visible timestamps must not be empty")
    result: list[ReferenceBucket] = []
    for target_pct in targets:
        target_count = exact_target_count(len(timestamps), target_pct)
        cutoff = timestamps[target_count - 1]
        achieved_count = bisect.bisect_right(timestamps, cutoff)
        result.append(
            ReferenceBucket(
                target_pct=target_pct,
                as_of=cutoff,
                acl_visible_count=len(timestamps),
                target_count=target_count,
                achieved_count=achieved_count,
            )
        )
    return result


def validate_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = tuple(next(reader, ()))
    if header != CSV_COLUMNS:
        raise ValueError(
            f"CSV header mismatch for {path}: expected={CSV_COLUMNS!r} actual={header!r}"
        )
    return header


STAGING_DDL = """
CREATE TEMP TABLE amazon10m_sql_native_stage (
    id bigint,
    user_id text,
    parent_asin text,
    rating double precision,
    timestamp bigint,
    verified_purchase boolean,
    helpful_vote integer,
    review_text_len integer,
    store text,
    main_category text,
    category_id integer,
    price double precision,
    has_price boolean,
    item_avg_rating double precision,
    item_rating_number integer
) ON COMMIT DROP
"""

COPY_STAGING_SQL = """
COPY amazon10m_sql_native_stage
FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
"""

INSTALL_VECTOR_EPOCH_TRIGGER_SQL = """
DROP TRIGGER IF EXISTS amazon_sql_native_epoch_bump ON {vector_table};
CREATE TRIGGER amazon_sql_native_epoch_bump
AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON {vector_table}
FOR EACH STATEMENT
EXECUTE FUNCTION public.amazon_sql_native_bump_relation_epoch()
"""

PRODUCT_STAGE_SQL = """
CREATE TEMP TABLE amazon10m_sql_native_product_stage ON COMMIT DROP AS
SELECT DISTINCT ON (parent_asin)
       parent_asin,
       btrim(coalesce(store, '')) AS store,
       btrim(coalesce(main_category, '')) AS main_category,
       item_rating_number,
       CASE WHEN btrim(coalesce(store, '')) <> '' THEN 'store' ELSE 'asin' END
           AS source_kind,
       CASE WHEN btrim(coalesce(store, '')) <> ''
            THEN btrim(store)
            ELSE btrim(parent_asin)
       END AS source_value
FROM amazon10m_sql_native_stage
ORDER BY parent_asin,
         (btrim(coalesce(store, '')) <> '') DESC,
         btrim(coalesce(store, '')),
         btrim(coalesce(main_category, '')),
         item_rating_number DESC
"""

LOAD_RELATIONS_SQL = """
TRUNCATE TABLE
    public.amazon_sql_native_buckets,
    public.amazon_principal_tenant_grants,
    public.amazon_review_facts,
    public.amazon_product_dim,
    public.amazon_tenants;

INSERT INTO public.amazon_tenants (tenant_id, source_kind, source_value, policy_kind)
SELECT DISTINCT
       source_kind || ':' || md5(source_value),
       source_kind,
       source_value,
       'derived_benchmark_policy'
FROM amazon10m_sql_native_product_stage;

INSERT INTO public.amazon_product_dim
    (parent_asin, tenant_id, store, main_category, item_rating_number)
SELECT parent_asin,
       source_kind || ':' || md5(source_value),
       store,
       main_category,
       item_rating_number
FROM amazon10m_sql_native_product_stage;

INSERT INTO public.amazon_review_facts
    (review_id, user_id, parent_asin, valid_from, valid_to)
SELECT id, btrim(user_id), btrim(parent_asin), timestamp, NULL::bigint
FROM amazon10m_sql_native_stage;
"""

INSERT_GRANTS_SQL = """
WITH tenant_review_counts AS (
    SELECT product.tenant_id,
           count(*)::bigint AS review_count,
           min(fact.valid_from)::bigint AS first_valid_from
    FROM public.amazon_review_facts AS fact
    JOIN public.amazon_product_dim AS product
      ON product.parent_asin = fact.parent_asin
    GROUP BY product.tenant_id
), ranked AS (
    SELECT tenant_id,
           review_count,
           first_valid_from,
           sum(review_count) OVER (ORDER BY review_count DESC, tenant_id) AS cumulative_count,
           sum(review_count) OVER () AS total_count
    FROM tenant_review_counts
), chosen AS (
    SELECT tenant_id, first_valid_from
    FROM ranked
    WHERE cumulative_count - review_count
          < ceil(total_count * %s::numeric / 100.0)::bigint
)
INSERT INTO public.amazon_principal_tenant_grants
    (principal_name, tenant_id, can_read, valid_from, valid_to, policy_kind)
SELECT %s, tenant_id, true, first_valid_from, NULL::bigint,
       'derived_benchmark_policy'
FROM chosen
ORDER BY tenant_id
"""

INSERT_BUCKETS_SQL = """
WITH targets AS (
    SELECT target_pct
    FROM unnest(%s::numeric[]) AS value(target_pct)
), acl_visible AS (
    SELECT fact.valid_from
    FROM public.amazon_review_facts AS fact
    JOIN public.amazon_product_dim AS product
      ON product.parent_asin = fact.parent_asin
    JOIN public.amazon_principal_tenant_grants AS grant_row
      ON grant_row.tenant_id = product.tenant_id
     AND grant_row.principal_name = %s
     AND grant_row.can_read
), timestamp_counts AS (
    SELECT valid_from, count(*)::bigint AS timestamp_count
    FROM acl_visible
    GROUP BY valid_from
), cumulative AS (
    SELECT valid_from,
           sum(timestamp_count) OVER (ORDER BY valid_from)::bigint AS cumulative_count
    FROM timestamp_counts
), population AS (
    SELECT coalesce(sum(timestamp_count), 0)::bigint AS visible_count
    FROM timestamp_counts
), selected AS (
    SELECT target.target_pct,
           population.visible_count,
           greatest(
               1,
               ceil(population.visible_count * target.target_pct / 100.0)::bigint
           ) AS target_count,
           cutoff.valid_from,
           cutoff.cumulative_count
    FROM targets AS target
    CROSS JOIN population
    JOIN LATERAL (
        SELECT cumulative.valid_from, cumulative.cumulative_count
        FROM cumulative
        WHERE cumulative.cumulative_count >= greatest(
            1,
            ceil(population.visible_count * target.target_pct / 100.0)::bigint
        )
        ORDER BY cumulative.valid_from
        LIMIT 1
    ) AS cutoff ON true
    WHERE population.visible_count > 0
)
INSERT INTO public.amazon_sql_native_buckets
    (principal_name, target_pct, as_of, acl_visible_count,
     target_count, achieved_count, achieved_pct)
SELECT %s,
       target_pct,
       valid_from,
       visible_count,
       target_count,
       cumulative_count,
       100.0 * cumulative_count / visible_count
FROM selected
ORDER BY target_pct
"""

# Calibrate grant-time visibility over the ACL-visible population. Tenants are
# indivisible policy units, so the deterministic ascending-volume order keeps
# the boundary overshoot small without inventing per-review ACLs. Both active
# and inactive validity times are anchored to real review timestamps.
CALIBRATE_GRANT_VALIDITY_SQL = """
WITH target AS (
    SELECT bucket.as_of,
           bucket.target_count,
           (
               SELECT min(fact.valid_from)::bigint
               FROM public.amazon_review_facts AS fact
               JOIN public.amazon_product_dim AS product
                 ON product.parent_asin = fact.parent_asin
               JOIN public.amazon_principal_tenant_grants AS grant_row
                 ON grant_row.tenant_id = product.tenant_id
                AND grant_row.principal_name = bucket.principal_name
                AND grant_row.can_read
               WHERE fact.valid_from > bucket.as_of
           ) AS inactive_valid_from
    FROM public.amazon_sql_native_buckets AS bucket
    WHERE bucket.principal_name = %s
      AND bucket.target_pct = %s
), tenant_review_counts AS (
    SELECT grant_row.tenant_id,
           count(*)::bigint AS review_count
    FROM public.amazon_principal_tenant_grants AS grant_row
    JOIN public.amazon_product_dim AS product
      ON product.tenant_id = grant_row.tenant_id
    JOIN public.amazon_review_facts AS fact
      ON fact.parent_asin = product.parent_asin
    WHERE grant_row.principal_name = %s
      AND grant_row.can_read
    GROUP BY grant_row.tenant_id
), ranked AS (
    SELECT tenant_id,
           review_count,
           sum(review_count) OVER (ORDER BY review_count, tenant_id) AS cumulative_count
    FROM tenant_review_counts
), chosen AS (
    SELECT ranked.tenant_id
    FROM ranked
    CROSS JOIN target
    WHERE ranked.cumulative_count - ranked.review_count < target.target_count
)
UPDATE public.amazon_principal_tenant_grants AS grant_row
SET valid_from = CASE
        WHEN EXISTS (
            SELECT 1 FROM chosen WHERE chosen.tenant_id = grant_row.tenant_id
        ) THEN least(grant_row.valid_from, target.as_of)
        ELSE target.inactive_valid_from
    END,
    valid_to = NULL::bigint
FROM target
WHERE grant_row.principal_name = %s
  AND grant_row.can_read
  AND target.inactive_valid_from IS NOT NULL
"""

SECONDARY_INDEXES = (
    (
        "amazon_review_facts_parent_time_idx",
        "CREATE INDEX amazon_review_facts_parent_time_idx "
        "ON public.amazon_review_facts (parent_asin, valid_from, review_id)",
    ),
    (
        "amazon_review_facts_valid_from_idx",
        "CREATE INDEX amazon_review_facts_valid_from_idx "
        "ON public.amazon_review_facts (valid_from, review_id)",
    ),
    (
        "amazon_review_facts_user_time_idx",
        "CREATE INDEX amazon_review_facts_user_time_idx "
        "ON public.amazon_review_facts (user_id, valid_from, review_id)",
    ),
    (
        "amazon_product_dim_tenant_asin_idx",
        "CREATE INDEX amazon_product_dim_tenant_asin_idx "
        "ON public.amazon_product_dim (tenant_id, parent_asin)",
    ),
    (
        "amazon_product_dim_category_rating_count_idx",
        "CREATE INDEX amazon_product_dim_category_rating_count_idx "
        "ON public.amazon_product_dim (main_category, item_rating_number, parent_asin)",
    ),
    (
        "amazon_principal_tenant_grants_tenant_idx",
        "CREATE INDEX amazon_principal_tenant_grants_tenant_idx "
        "ON public.amazon_principal_tenant_grants (tenant_id, principal_name)",
    ),
    (
        "amazon_sql_native_buckets_as_of_idx",
        "CREATE INDEX amazon_sql_native_buckets_as_of_idx "
        "ON public.amazon_sql_native_buckets (principal_name, as_of)",
    ),
)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized Amazon-10M relations and ACL-visible temporal buckets "
            "without rewriting the vector heap."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--schema-sql", type=Path, default=DEFAULT_SCHEMA_SQL)
    parser.add_argument("--vector-table", default=DEFAULT_VECTOR_TABLE)
    parser.add_argument("--principal", type=principal_name, default=DEFAULT_PRINCIPAL)
    parser.add_argument(
        "--acl-coverage-pct",
        type=percentage,
        default=Decimal("60"),
        help="Grant volume-ranked real tenants until this review coverage is reached.",
    )
    parser.add_argument("--expected-rows", type=positive_int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--copy-chunk-mib", type=positive_int, default=8)
    parser.add_argument("--dsn", help="psycopg connection string; defaults to PG* environment variables")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the transaction plan without opening the CSV or database",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="run read-only validation of an existing prepared workload",
    )
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("mode=dry-run")
    print("transaction=BEGIN -> advisory lock -> schema -> stage/validate -> replace -> validate -> COMMIT")
    print(f"csv={args.csv}")
    print(f"schema_sql={args.schema_sql}")
    print(f"vector_table={args.vector_table} access=read-only-validation")
    print(f"expected_rows={args.expected_rows}")
    print(f"principal={args.principal} non_owner_rls=true")
    print(f"acl_coverage_pct={args.acl_coverage_pct} policy=real-store-else-real-asin")
    print("valid_from=real-csv-timestamp valid_to=NULL(no-real-end)")
    print("temporal_target_pct=" + ",".join(str(value) for value in TEMPORAL_TARGET_PCTS))


def ensure_principal(cur, sql, principal: str) -> None:
    cur.execute(
        "SELECT rolsuper, rolbypassrls FROM pg_catalog.pg_roles WHERE rolname = %s",
        (principal,),
    )
    role = cur.fetchone()
    if role is None:
        cur.execute(
            sql.SQL(
                "CREATE ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "INHERIT NOBYPASSRLS"
            ).format(sql.Identifier(principal))
        )
    elif bool(role[0]) or bool(role[1]):
        raise RuntimeError(f"principal {principal!r} must not be superuser or BYPASSRLS")
    cur.execute(
        sql.SQL("GRANT {} TO {}").format(
            sql.Identifier(READER_ROLE), sql.Identifier(principal)
        )
    )


def grant_vector_select(cur, sql, vector_table: str) -> None:
    vector_identifier = sql.Identifier(*parse_qualified_name(vector_table))
    cur.execute(
        sql.SQL("GRANT SELECT ON {} TO {}").format(
            vector_identifier,
            sql.Identifier(READER_ROLE),
        )
    )


def install_vector_epoch_trigger(cur, sql, vector_table: str) -> None:
    vector_identifier = sql.Identifier(*parse_qualified_name(vector_table))
    cur.execute(
        sql.SQL("DROP TRIGGER IF EXISTS amazon_sql_native_epoch_bump ON {}").format(
            vector_identifier
        )
    )
    cur.execute(
        sql.SQL(
            "CREATE TRIGGER amazon_sql_native_epoch_bump "
            "AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON {} "
            "FOR EACH STATEMENT EXECUTE FUNCTION "
            "public.amazon_sql_native_bump_relation_epoch()"
        ).format(vector_identifier)
    )


def copy_csv_to_staging(cur, path: Path, chunk_mib: int) -> None:
    chunk_bytes = chunk_mib * 1024 * 1024
    with path.open("rb") as source, cur.copy(COPY_STAGING_SQL) as copy:
        while chunk := source.read(chunk_bytes):
            copy.write(chunk)


def validate_staging(cur, expected_rows: int) -> dict[str, int]:
    cur.execute(
        """
        SELECT count(*)::bigint,
               count(*) FILTER (
                   WHERE id IS NULL OR id < 0
                      OR btrim(coalesce(user_id, '')) = ''
                      OR btrim(coalesce(parent_asin, '')) = ''
                      OR timestamp IS NULL OR timestamp <= 0
                      OR item_rating_number IS NULL OR item_rating_number < 0
               )::bigint
        FROM amazon10m_sql_native_stage
        """
    )
    row_count, invalid_count = (int(value) for value in cur.fetchone())
    if row_count != expected_rows:
        raise RuntimeError(
            f"CSV row count mismatch: expected={expected_rows} actual={row_count}"
        )
    if invalid_count:
        raise RuntimeError(
            f"CSV has {invalid_count} rows with missing/non-real keys, timestamps, or rating counts"
        )

    cur.execute(
        """
        SELECT id, count(*)
        FROM amazon10m_sql_native_stage
        GROUP BY id
        HAVING count(*) > 1
        LIMIT 1
        """
    )
    duplicate = cur.fetchone()
    if duplicate is not None:
        raise RuntimeError(f"duplicate CSV review id: id={duplicate[0]} count={duplicate[1]}")

    cur.execute(
        """
        SELECT parent_asin
        FROM amazon10m_sql_native_stage
        GROUP BY parent_asin
        HAVING count(DISTINCT (
            btrim(coalesce(store, '')),
            btrim(coalesce(main_category, '')),
            item_rating_number
        )) > 1
        LIMIT 1
        """
    )
    conflict = cur.fetchone()
    if conflict is not None:
        raise RuntimeError(
            f"CSV has conflicting product attributes for parent_asin={conflict[0]!r}"
        )
    return {"staged_rows": row_count, "invalid_rows": invalid_count}


def materialize_relations(cur, principal: str, acl_coverage_pct: Decimal) -> None:
    cur.execute(PRODUCT_STAGE_SQL)
    cur.execute(
        "CREATE UNIQUE INDEX amazon10m_sql_native_product_stage_asin_idx "
        "ON amazon10m_sql_native_product_stage (parent_asin)"
    )
    cur.execute(LOAD_RELATIONS_SQL)
    cur.execute(INSERT_GRANTS_SQL, (acl_coverage_pct, principal))
    cur.execute(
        "SELECT count(*) FROM public.amazon_principal_tenant_grants WHERE principal_name = %s",
        (principal,),
    )
    if int(cur.fetchone()[0]) == 0:
        raise RuntimeError("derived benchmark ACL selected no real store/ASIN tenants")
    cur.execute(
        INSERT_BUCKETS_SQL,
        ([value for value in TEMPORAL_TARGET_PCTS], principal, principal),
    )
    cur.execute(
        CALIBRATE_GRANT_VALIDITY_SQL,
        (principal, GRANT_TEMPORAL_TARGET_PCT, principal, principal),
    )
    if cur.rowcount <= 0:
        raise RuntimeError("grant temporal calibration updated no ACL rows")
    for _, ddl in SECONDARY_INDEXES:
        cur.execute(ddl)
    cur.execute("ANALYZE public.amazon_tenants")
    cur.execute("ANALYZE public.amazon_product_dim")
    cur.execute("ANALYZE public.amazon_review_facts")
    cur.execute("ANALYZE public.amazon_principal_tenant_grants")
    cur.execute("ANALYZE public.amazon_sql_native_buckets")


def _assert_schema_and_roles(cur, principal: str) -> None:
    expected_tables = (
        "amazon_tenants",
        "amazon_product_dim",
        "amazon_review_facts",
        "amazon_principal_tenant_grants",
        "amazon_sql_native_buckets",
        "amazon_sql_native_relation_epoch",
    )
    cur.execute(
        """
        SELECT c.relname
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = ANY(%s)
        """,
        (list(expected_tables),),
    )
    found = {str(row[0]) for row in cur.fetchall()}
    missing = sorted(set(expected_tables) - found)
    if missing:
        raise RuntimeError(f"missing SQL-native tables: {missing}")

    cur.execute(
        """
        SELECT rolname, rolsuper, rolbypassrls
        FROM pg_catalog.pg_roles
        WHERE rolname = ANY(%s)
        """,
        ([READER_ROLE, principal],),
    )
    roles = {str(row[0]): (bool(row[1]), bool(row[2])) for row in cur.fetchall()}
    for role_name in (READER_ROLE, principal):
        if role_name not in roles:
            raise RuntimeError(f"missing RLS role: {role_name}")
        if any(roles[role_name]):
            raise RuntimeError(f"RLS role {role_name} is superuser or BYPASSRLS")

    cur.execute("SELECT pg_has_role(%s, %s, 'MEMBER')", (principal, READER_ROLE))
    if not bool(cur.fetchone()[0]):
        raise RuntimeError(f"principal {principal!r} is not a member of {READER_ROLE!r}")

    cur.execute(
        """
        SELECT c.relname
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_roles AS owner ON owner.oid = c.relowner
        WHERE n.nspname = 'public'
          AND c.relname = ANY(%s)
          AND owner.rolname = ANY(%s)
        """,
        (list(expected_tables), [READER_ROLE, principal]),
    )
    owned = [str(row[0]) for row in cur.fetchall()]
    if owned:
        raise RuntimeError(f"RLS benchmark roles must be non-owners; owned tables={owned}")

    cur.execute(
        """
        SELECT c.relname
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'amazon_review_facts'
          AND NOT c.relrowsecurity
        """
    )
    without_rls = [str(row[0]) for row in cur.fetchall()]
    if without_rls:
        raise RuntimeError(f"RLS is not enabled on tables: {without_rls}")
    cur.execute(
        """
        SELECT policyname, coalesce(qual, '')
        FROM pg_catalog.pg_policies
        WHERE schemaname = 'public' AND tablename = 'amazon_review_facts'
        ORDER BY policyname
        """
    )
    policies = [(str(row[0]), str(row[1])) for row in cur.fetchall()]
    if len(policies) != 1 or policies[0][0] != "amazon_review_facts_acl_select":
        raise RuntimeError(f"unexpected amazon_review_facts policies: {policies}")
    normalized_policy = policies[0][1].lower()
    if "app.as_of" in normalized_policy or "valid_from" in normalized_policy:
        raise RuntimeError("RLS policy must enforce ACL only, not temporal selectivity")


def _validate_vector_ids(cur, sql, vector_table: str, facts_count: int) -> int:
    cur.execute("SELECT to_regclass(%s)", (vector_table,))
    if cur.fetchone()[0] is None:
        raise RuntimeError(f"vector table does not exist: {vector_table}")
    cur.execute(
        """
        SELECT 1
        FROM pg_catalog.pg_attribute
        WHERE attrelid = to_regclass(%s)
          AND attname = 'id'
          AND NOT attisdropped
        """,
        (vector_table,),
    )
    if cur.fetchone() is None:
        raise RuntimeError(f"vector table has no id column: {vector_table}")
    cur.execute("SELECT has_table_privilege(%s, %s, 'SELECT')", (READER_ROLE, vector_table))
    if not bool(cur.fetchone()[0]):
        raise RuntimeError(f"{READER_ROLE} lacks SELECT on vector table {vector_table}")

    vector_identifier = sql.Identifier(*parse_qualified_name(vector_table))
    cur.execute(sql.SQL("SELECT count(*)::bigint FROM {}").format(vector_identifier))
    vector_count = int(cur.fetchone()[0])
    if vector_count != facts_count:
        raise RuntimeError(
            f"vector/fact row count mismatch: vector={vector_count} facts={facts_count}"
        )

    cur.execute(
        sql.SQL(
            "SELECT fact.review_id FROM public.amazon_review_facts AS fact "
            "LEFT JOIN {} AS vector_row ON vector_row.id = fact.review_id "
            "WHERE vector_row.id IS NULL LIMIT 1"
        ).format(vector_identifier)
    )
    missing_vector = cur.fetchone()
    if missing_vector is not None:
        raise RuntimeError(f"review_id missing from vector heap: {missing_vector[0]}")

    cur.execute(
        sql.SQL(
            "SELECT vector_row.id FROM {} AS vector_row "
            "LEFT JOIN public.amazon_review_facts AS fact ON fact.review_id = vector_row.id "
            "WHERE fact.review_id IS NULL LIMIT 1"
        ).format(vector_identifier)
    )
    missing_fact = cur.fetchone()
    if missing_fact is not None:
        raise RuntimeError(f"vector id missing from amazon_review_facts: {missing_fact[0]}")
    return vector_count


def _assert_epoch_triggers(cur, vector_table: str) -> None:
    relations = (
        vector_table,
        "public.amazon_review_facts",
        "public.amazon_product_dim",
        "public.amazon_principal_tenant_grants",
        "public.amazon_sql_native_buckets",
    )
    missing: list[str] = []
    for relation in relations:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger
                WHERE tgrelid = to_regclass(%s)
                  AND tgname = 'amazon_sql_native_epoch_bump'
                  AND NOT tgisinternal
            )
            """,
            (relation,),
        )
        if not bool(cur.fetchone()[0]):
            missing.append(relation)
    if missing:
        raise RuntimeError(f"missing formal data-version epoch triggers: {missing}")


def validate_database(cur, sql, args: argparse.Namespace) -> dict[str, int]:
    _assert_schema_and_roles(cur, args.principal)
    _assert_epoch_triggers(cur, args.vector_table)

    expected_indexes = {name for name, _ in SECONDARY_INDEXES}
    cur.execute(
        """
        SELECT indexname
        FROM pg_catalog.pg_indexes
        WHERE schemaname = 'public' AND indexname = ANY(%s)
        """,
        (list(expected_indexes),),
    )
    missing_indexes = expected_indexes - {str(row[0]) for row in cur.fetchall()}
    if missing_indexes:
        raise RuntimeError(f"missing SQL-native secondary indexes: {sorted(missing_indexes)}")

    counts: dict[str, int] = {}
    for table in (
        "amazon_review_facts",
        "amazon_product_dim",
        "amazon_tenants",
        "amazon_principal_tenant_grants",
        "amazon_sql_native_buckets",
    ):
        cur.execute(sql.SQL("SELECT count(*)::bigint FROM public.{}").format(sql.Identifier(table)))
        counts[table] = int(cur.fetchone()[0])
    if counts["amazon_review_facts"] != args.expected_rows:
        raise RuntimeError(
            "prepared fact row count mismatch: "
            f"expected={args.expected_rows} actual={counts['amazon_review_facts']}"
        )
    if counts["amazon_product_dim"] == 0 or counts["amazon_tenants"] == 0:
        raise RuntimeError("prepared amazon_product_dim/amazon_tenants must not be empty")

    cur.execute(
        """
        SELECT target_pct, as_of, acl_visible_count,
               target_count, achieved_count
        FROM public.amazon_sql_native_buckets
        WHERE principal_name = %s
        ORDER BY target_pct
        """,
        (args.principal,),
    )
    bucket_rows = cur.fetchall()
    actual_targets = tuple(Decimal(row[0]) for row in bucket_rows)
    if actual_targets != TEMPORAL_TARGET_PCTS:
        raise RuntimeError(
            f"temporal targets mismatch: expected={TEMPORAL_TARGET_PCTS} actual={actual_targets}"
        )

    cur.execute(
        """
        WITH acl_visible AS (
            SELECT fact.valid_from
            FROM public.amazon_review_facts AS fact
            JOIN public.amazon_product_dim AS product
              ON product.parent_asin = fact.parent_asin
            JOIN public.amazon_principal_tenant_grants AS grant_row
              ON grant_row.tenant_id = product.tenant_id
             AND grant_row.principal_name = %s
             AND grant_row.can_read
        ), timestamp_counts AS (
            SELECT valid_from, count(*)::bigint AS timestamp_count
            FROM acl_visible
            GROUP BY valid_from
        ), cumulative AS (
            SELECT valid_from,
                   sum(timestamp_count) OVER (ORDER BY valid_from)::bigint AS cumulative_count
            FROM timestamp_counts
        )
        SELECT bucket.target_pct, cumulative.cumulative_count
        FROM public.amazon_sql_native_buckets AS bucket
        LEFT JOIN cumulative
          ON cumulative.valid_from = bucket.as_of
        WHERE bucket.principal_name = %s
        ORDER BY bucket.target_pct
        """,
        (args.principal, args.principal),
    )
    recomputed = cur.fetchall()
    if len(recomputed) != len(bucket_rows):
        raise RuntimeError("could not recompute every temporal bucket")
    for stored, checked in zip(bucket_rows, recomputed, strict=True):
        target_pct, _, visible_count, target_count, achieved_count = stored
        if checked[1] is None or int(checked[1]) != int(achieved_count):
            raise RuntimeError(f"achieved count mismatch for target_pct={target_pct}")
        expected_target_count = exact_target_count(int(visible_count), Decimal(target_pct))
        if int(target_count) != expected_target_count:
            raise RuntimeError(f"target count mismatch for target_pct={target_pct}")

    visible_count = int(bucket_rows[0][2])
    achieved_acl_pct = Decimal(100) * Decimal(visible_count) / Decimal(args.expected_rows)
    if achieved_acl_pct < args.acl_coverage_pct:
        raise RuntimeError(
            f"ACL coverage below target: target={args.acl_coverage_pct} actual={achieved_acl_pct}"
        )
    cur.execute(
        """
        SELECT min(fact.review_id) FILTER (WHERE grant_row.tenant_id IS NOT NULL),
               min(fact.review_id) FILTER (WHERE grant_row.tenant_id IS NULL)
        FROM public.amazon_review_facts AS fact
        JOIN public.amazon_product_dim AS product
          ON product.parent_asin = fact.parent_asin
        LEFT JOIN public.amazon_principal_tenant_grants AS grant_row
          ON grant_row.tenant_id = product.tenant_id
         AND grant_row.principal_name = %s
         AND grant_row.can_read
        """,
        (args.principal,),
    )
    probe_row = cur.fetchone()
    if probe_row is None or probe_row[0] is None or probe_row[1] is None:
        raise RuntimeError("controlled RLS probes require visible and hidden review facts")
    positive_probe, negative_probe = (int(probe_row[0]), int(probe_row[1]))

    cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(args.principal)))
    try:
        cur.execute("SELECT count(*)::bigint FROM public.amazon_review_facts")
        rls_visible_count = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE review_id = %s)::bigint,
                   count(*) FILTER (WHERE review_id = %s)::bigint
            FROM public.amazon_review_facts
            WHERE review_id = ANY(%s::bigint[])
            """,
            (positive_probe, negative_probe, [positive_probe, negative_probe]),
        )
        positive_count, negative_count = (int(value) for value in cur.fetchone())
    finally:
        cur.execute("RESET ROLE")
    if rls_visible_count != visible_count:
        raise RuntimeError(
            f"RLS visibility mismatch: policy={rls_visible_count} buckets={visible_count}"
        )
    if positive_count != 1 or negative_count != 0:
        raise RuntimeError(
            "controlled RLS probe failed: "
            f"positive={positive_count} negative={negative_count}"
        )

    sampled_buckets = (bucket_rows[0], bucket_rows[len(bucket_rows) // 2], bucket_rows[-1])
    cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(args.principal)))
    try:
        for target_pct, as_of, _, _, achieved_count in sampled_buckets:
            cur.execute(
                """
                SELECT count(*)::bigint
                FROM public.amazon_review_facts AS fact
                WHERE fact.valid_from <= %s
                  AND (fact.valid_to IS NULL OR fact.valid_to > %s)
                """,
                (as_of, as_of),
            )
            observed = int(cur.fetchone()[0])
            if observed != int(achieved_count):
                raise RuntimeError(
                    f"explicit fact-temporal count mismatch for target_pct={target_pct}: "
                    f"query={observed} bucket={achieved_count}"
                )
        grant_bucket = next(
            row
            for row in bucket_rows
            if Decimal(row[0]) == GRANT_TEMPORAL_TARGET_PCT
        )
        fact_bucket = next(
            row
            for row in bucket_rows
            if Decimal(row[0]) == FACT_TEMPORAL_TARGET_PCT
        )
        grant_as_of = int(grant_bucket[1])
        cur.execute(
            """
            SELECT count(*)::bigint
            FROM public.amazon_review_facts AS fact
            JOIN public.amazon_product_dim AS product
              ON product.parent_asin = fact.parent_asin
            JOIN public.amazon_principal_tenant_grants AS grant_row
              ON grant_row.tenant_id = product.tenant_id
             AND grant_row.principal_name = CURRENT_USER::text
             AND grant_row.can_read
            WHERE grant_row.valid_from <= %s
              AND (grant_row.valid_to IS NULL OR grant_row.valid_to > %s)
            """,
            (grant_as_of, grant_as_of),
        )
        grant_temporal_count = int(cur.fetchone()[0])
    finally:
        cur.execute("RESET ROLE")
    population_proof = validate_distinct_workload_populations(
        visible_count, grant_temporal_count, int(fact_bucket[4])
    )
    validate_temporal_target(
        "grant",
        grant_temporal_count,
        visible_count,
        GRANT_TEMPORAL_TARGET_PCT,
    )
    validate_temporal_target(
        "fact",
        population_proof["fact_temporal_selectivity"],
        visible_count,
        FACT_TEMPORAL_TARGET_PCT,
    )

    counts["acl_visible_reviews"] = visible_count
    counts["grant_temporal_reviews_at_20pct_as_of"] = grant_temporal_count
    counts["fact_temporal_reviews_at_5pct_as_of"] = population_proof[
        "fact_temporal_selectivity"
    ]
    counts["vector_rows"] = _validate_vector_ids(
        cur, sql, args.vector_table, counts["amazon_review_facts"]
    )
    return counts


def run_transaction(conn, *, read_only: bool, operation):
    begin = "BEGIN ISOLATION LEVEL REPEATABLE READ"
    if read_only:
        begin += " READ ONLY"
    with conn.cursor() as cur:
        cur.execute(begin)
        try:
            result = operation(cur)
        except BaseException:
            cur.execute("ROLLBACK")
            raise
        else:
            cur.execute("ROLLBACK" if read_only else "COMMIT")
            return result


def run_prepare(args: argparse.Namespace) -> dict[str, int]:
    validate_csv_header(args.csv)
    schema_sql = args.schema_sql.read_text(encoding="utf-8")
    require_psycopg()
    import psycopg
    from psycopg import sql

    conninfo = args.dsn or pg_config_from_env().conninfo
    with psycopg.connect(conninfo, autocommit=True) as conn:

        def prepare(cur):
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("amazon10m_sql_native",))
            cur.execute(schema_sql)
            ensure_principal(cur, sql, args.principal)
            grant_vector_select(cur, sql, args.vector_table)
            install_vector_epoch_trigger(cur, sql, args.vector_table)
            for index_name, _ in SECONDARY_INDEXES:
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS public.{}").format(sql.Identifier(index_name))
                )
            cur.execute(STAGING_DDL)
            copy_csv_to_staging(cur, args.csv, args.copy_chunk_mib)
            staging_report = validate_staging(cur, args.expected_rows)
            materialize_relations(cur, args.principal, args.acl_coverage_pct)
            report = validate_database(cur, sql, args)
            report.update(staging_report)
            return report

        return run_transaction(conn, read_only=False, operation=prepare)


def run_validate_only(args: argparse.Namespace) -> dict[str, int]:
    require_psycopg()
    import psycopg
    from psycopg import sql

    conninfo = args.dsn or pg_config_from_env().conninfo
    with psycopg.connect(conninfo, autocommit=True) as conn:
        return run_transaction(
            conn,
            read_only=True,
            operation=lambda cur: validate_database(cur, sql, args),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    try:
        parse_qualified_name(args.vector_table)
        if args.dry_run:
            print_dry_run(args)
            return 0
        report = run_validate_only(args) if args.validate_only else run_prepare(args)
    except (argparse.ArgumentTypeError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    mode = "validate-only" if args.validate_only else "prepare"
    print(f"mode={mode} status=ok transaction={'rolled-back-read-only' if args.validate_only else 'committed'}")
    for key in sorted(report):
        print(f"{key}={report[key]}")
    print("vector_heap_writes=0")
    print("acl_policy=derived-benchmark-policy(real-store-else-real-asin)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
