-- Relational sidecars for the Amazon-10M SQL-native workload.
--
-- review_id is validated against the existing vector heap by
-- prepare_amazon10m_sql_native.py; this DDL never rewrites that heap.

DO $role$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'amazon10m_sql_native_reader'
    ) THEN
        CREATE ROLE amazon10m_sql_native_reader
            NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT NOBYPASSRLS;
    ELSIF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'amazon10m_sql_native_reader'
          AND (rolsuper OR rolbypassrls)
    ) THEN
        RAISE EXCEPTION
            'amazon10m_sql_native_reader must be a non-superuser without BYPASSRLS';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'amazon10m_sql_native_benchmark'
    ) THEN
        CREATE ROLE amazon10m_sql_native_benchmark
            NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT NOBYPASSRLS;
    ELSIF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'amazon10m_sql_native_benchmark'
          AND (rolsuper OR rolbypassrls)
    ) THEN
        RAISE EXCEPTION
            'amazon10m_sql_native_benchmark must be a non-superuser without BYPASSRLS';
    END IF;
END
$role$;

GRANT amazon10m_sql_native_reader TO amazon10m_sql_native_benchmark;

CREATE TABLE IF NOT EXISTS public.amazon_tenants (
    tenant_id text PRIMARY KEY,
    source_kind text NOT NULL CHECK (source_kind IN ('store', 'asin')),
    source_value text NOT NULL,
    policy_kind text NOT NULL DEFAULT 'derived_benchmark_policy'
        CHECK (policy_kind = 'derived_benchmark_policy')
);

CREATE TABLE IF NOT EXISTS public.amazon_product_dim (
    parent_asin text PRIMARY KEY,
    tenant_id text NOT NULL REFERENCES public.amazon_tenants (tenant_id),
    store text NOT NULL,
    main_category text NOT NULL,
    item_rating_number integer NOT NULL CHECK (item_rating_number >= 0)
);

CREATE TABLE IF NOT EXISTS public.amazon_review_facts (
    review_id bigint PRIMARY KEY,
    user_id text NOT NULL,
    parent_asin text NOT NULL REFERENCES public.amazon_product_dim (parent_asin),
    valid_from bigint NOT NULL CHECK (valid_from > 0),
    valid_to bigint,
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);

CREATE TABLE IF NOT EXISTS public.amazon_principal_tenant_grants (
    principal_name text NOT NULL,
    tenant_id text NOT NULL REFERENCES public.amazon_tenants (tenant_id) ON DELETE CASCADE,
    can_read boolean NOT NULL DEFAULT true,
    valid_from bigint NOT NULL,
    valid_to bigint,
    policy_kind text NOT NULL DEFAULT 'derived_benchmark_policy'
        CHECK (policy_kind = 'derived_benchmark_policy'),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    PRIMARY KEY (principal_name, tenant_id)
);

CREATE TABLE IF NOT EXISTS public.amazon_sql_native_buckets (
    principal_name text NOT NULL,
    target_pct numeric(5, 2) NOT NULL CHECK (target_pct > 0 AND target_pct <= 100),
    as_of bigint NOT NULL,
    acl_visible_count bigint NOT NULL CHECK (acl_visible_count > 0),
    target_count bigint NOT NULL CHECK (target_count > 0),
    achieved_count bigint NOT NULL CHECK (achieved_count >= target_count),
    achieved_pct numeric(12, 8) NOT NULL CHECK (achieved_pct > 0 AND achieved_pct <= 100),
    PRIMARY KEY (principal_name, target_pct)
);

CREATE TABLE IF NOT EXISTS public.amazon_sql_native_relation_epoch (
    relation_name text PRIMARY KEY,
    epoch bigint NOT NULL DEFAULT 0 CHECK (epoch >= 0),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE OR REPLACE FUNCTION public.amazon_sql_native_bump_relation_epoch()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $epoch$
BEGIN
    INSERT INTO public.amazon_sql_native_relation_epoch
        (relation_name, epoch, updated_at)
    VALUES
        (TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME, 1, clock_timestamp())
    ON CONFLICT (relation_name) DO UPDATE
    SET epoch = public.amazon_sql_native_relation_epoch.epoch + 1,
        updated_at = clock_timestamp();
    RETURN NULL;
END
$epoch$;

DO $epoch_triggers$
DECLARE
    relation_name text;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'amazon_tenants',
        'amazon_product_dim',
        'amazon_review_facts',
        'amazon_principal_tenant_grants',
        'amazon_sql_native_buckets'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS amazon_sql_native_epoch_bump ON public.%I',
            relation_name
        );
        EXECUTE format(
            'CREATE TRIGGER amazon_sql_native_epoch_bump '
            'AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON public.%I '
            'FOR EACH STATEMENT EXECUTE FUNCTION '
            'public.amazon_sql_native_bump_relation_epoch()',
            relation_name
        );
    END LOOP;
END
$epoch_triggers$;

CREATE INDEX IF NOT EXISTS amazon_review_facts_parent_time_idx
    ON public.amazon_review_facts (parent_asin, valid_from, review_id);
CREATE INDEX IF NOT EXISTS amazon_review_facts_valid_from_idx
    ON public.amazon_review_facts (valid_from, review_id);
CREATE INDEX IF NOT EXISTS amazon_review_facts_user_time_idx
    ON public.amazon_review_facts (user_id, valid_from, review_id);
CREATE INDEX IF NOT EXISTS amazon_product_dim_tenant_asin_idx
    ON public.amazon_product_dim (tenant_id, parent_asin);
CREATE INDEX IF NOT EXISTS amazon_product_dim_category_rating_count_idx
    ON public.amazon_product_dim (main_category, item_rating_number, parent_asin);
CREATE INDEX IF NOT EXISTS amazon_principal_tenant_grants_tenant_idx
    ON public.amazon_principal_tenant_grants (tenant_id, principal_name);
CREATE INDEX IF NOT EXISTS amazon_sql_native_buckets_as_of_idx
    ON public.amazon_sql_native_buckets (principal_name, as_of);

COMMENT ON TABLE public.amazon_tenants IS
    'Benchmark-only tenant identities derived from real store values, falling back to real parent ASIN values.';
COMMENT ON COLUMN public.amazon_tenants.tenant_id IS
    'Compact store:/asin: MD5 key; hashing compresses a real source key and never uses review_id or modulo assignment.';
COMMENT ON COLUMN public.amazon_review_facts.valid_from IS
    'Original Amazon review timestamp, preserved as epoch milliseconds.';
COMMENT ON COLUMN public.amazon_review_facts.valid_to IS
    'NULL means open-ended because the source has no real validity end; synthetic end times are forbidden.';
COMMENT ON TABLE public.amazon_principal_tenant_grants IS
    'Derived benchmark ACL over real tenant keys, not a claim about Amazon production authorization.';
COMMENT ON TABLE public.amazon_sql_native_buckets IS
    'Tie-aware temporal targets over one principal ACL-visible review population.';

REVOKE ALL ON public.amazon_tenants FROM PUBLIC;
REVOKE ALL ON public.amazon_product_dim FROM PUBLIC;
REVOKE ALL ON public.amazon_review_facts FROM PUBLIC;
REVOKE ALL ON public.amazon_principal_tenant_grants FROM PUBLIC;
REVOKE ALL ON public.amazon_sql_native_buckets FROM PUBLIC;
REVOKE ALL ON public.amazon_sql_native_relation_epoch FROM PUBLIC;

GRANT USAGE ON SCHEMA public TO amazon10m_sql_native_reader;
GRANT SELECT ON public.amazon_tenants TO amazon10m_sql_native_reader;
GRANT SELECT ON public.amazon_product_dim TO amazon10m_sql_native_reader;
GRANT SELECT ON public.amazon_review_facts TO amazon10m_sql_native_reader;
GRANT SELECT ON public.amazon_principal_tenant_grants TO amazon10m_sql_native_reader;
GRANT SELECT ON public.amazon_sql_native_buckets TO amazon10m_sql_native_reader;
GRANT SELECT ON public.amazon_sql_native_relation_epoch TO amazon10m_sql_native_reader;

-- Only the fact relation is protected by RLS. Dimension and grant lookups stay
-- visible so the policy itself remains a genuine executor join instead of a
-- SECURITY DEFINER shortcut.
ALTER TABLE public.amazon_review_facts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS amazon_review_facts_acl_temporal_select
    ON public.amazon_review_facts;
DROP POLICY IF EXISTS amazon_review_facts_acl_select
    ON public.amazon_review_facts;
CREATE POLICY amazon_review_facts_acl_select
    ON public.amazon_review_facts
    FOR SELECT TO amazon10m_sql_native_reader
    USING (
        EXISTS (
            SELECT 1
            FROM public.amazon_product_dim AS product
            JOIN public.amazon_principal_tenant_grants AS grant_row
              ON grant_row.tenant_id = product.tenant_id
            WHERE product.parent_asin = amazon_review_facts.parent_asin
              AND grant_row.principal_name = CURRENT_USER::text
              AND grant_row.can_read
        )
    );
