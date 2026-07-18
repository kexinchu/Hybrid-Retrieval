\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS nested_binding_abort_smoke CASCADE;
CREATE TABLE nested_binding_abort_smoke (
    id bigint PRIMARY KEY,
    category integer NOT NULL,
    embedding vector(2) NOT NULL
);
INSERT INTO nested_binding_abort_smoke
SELECT value,
       CASE WHEN value % 10 = 0 THEN 1 ELSE 2 END,
       ARRAY[value::real, 0::real]::vector
FROM generate_series(1, 200) AS value;
CREATE INDEX nested_binding_abort_smoke_hnsw
ON nested_binding_abort_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 40);

-- Force the first fragment-store setup to happen in a transaction that aborts.
DROP TABLE IF EXISTS public.pgvector_hnsw_fragment_store CASCADE;
DROP TABLE IF EXISTS public.pgvector_hnsw_fragment_epoch CASCADE;
BEGIN;
SELECT vector_hnsw_fragment_tracking_enable('nested_binding_abort_smoke'::regclass);
\set ON_ERROR_STOP off
SELECT 1 / 0 AS expected_transaction_abort;
\set ON_ERROR_STOP on
ROLLBACK;

-- The next transaction must recreate the store before using the guidance API.
BEGIN;
SELECT vector_hnsw_fragment_tracking_enable('nested_binding_abort_smoke'::regclass) AS recovered_epoch;
SELECT vector_hnsw_guidance_activate(
    'nested_binding_abort_smoke_hnsw'::regclass,
    ARRAY['exact:sql:category = 1'],
    'exact'
) AS recovered_atoms;
COMMIT;

DO $$
DECLARE
    profile jsonb;
BEGIN
    IF to_regclass('public.pgvector_hnsw_fragment_store') IS NULL THEN
        RAISE EXCEPTION 'fragment store was not recreated after transaction abort';
    END IF;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO profile;
    IF NOT (profile ->> 'active')::boolean OR
       (profile ->> 'statement_bound')::boolean THEN
        RAISE EXCEPTION 'guidance API did not recover cleanly: %', profile;
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION nested_binding_abort_probe(armed boolean)
RETURNS boolean
LANGUAGE plpgsql VOLATILE
AS $$
DECLARE
    nested_result boolean;
BEGIN
    IF NOT armed THEN
        RETURN false;
    END IF;
    -- The failing SPI SELECT creates a nested QueryDesc whose ExecutorEnd is
    -- skipped. The subtransaction callback must discard only that frame.
    BEGIN
        SELECT 1 / 0 = 0 INTO nested_result;
    EXCEPTION WHEN division_by_zero THEN
        nested_result := true;
    END;
    RETURN nested_result;
END
$$;

SET enable_seqscan = off;
SET enable_sort = off;
SET enable_bitmapscan = off;
SET hnsw.ef_search = 80;
SET hnsw.iterative_scan = strict_order;
SET hnsw.max_scan_tuples = 10000;
SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_reset_scan_profile();

-- The bind runs before the PL/pgSQL SPI query; the outer binding must be
-- restored when that nested executor ends, so guidance remains effective.
DO $$
DECLARE
    guided_rows bigint;
    profile jsonb;
    guidance jsonb;
BEGIN
    SELECT count(*) INTO guided_rows
    FROM (
        SELECT id
        FROM nested_binding_abort_smoke
        WHERE (SELECT nested_binding_abort_probe(
                   vector_hnsw_guidance_bind(
                       'nested_binding_abort_smoke_hnsw'::regclass,
                       ARRAY['exact:sql:category = 1'],
                       'exact'
                   )
               ) OFFSET 0)
          AND category = 1
        ORDER BY embedding <-> '[21,0]'::vector
        LIMIT 10
    ) AS guided;

    IF guided_rows <> 10 THEN
        RAISE EXCEPTION 'matching nested binding returned % rows', guided_rows;
    END IF;
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance;
    IF (profile ->> 'guidance_checks')::bigint <= 0 OR
       (profile ->> 'guidance_skips')::bigint <= 0 THEN
        RAISE EXCEPTION 'nested executor cleared matching guidance: %', profile;
    END IF;
    IF NOT (profile ->> 'planner_proof_attempted')::boolean OR
       NOT (profile ->> 'planner_proof_succeeded')::boolean OR
       profile ->> 'planner_proof_bypass_reason' <> 'none' THEN
        RAISE EXCEPTION 'nested abort lost descriptor proof outcome: %', profile;
    END IF;
    IF (guidance ->> 'binding_matches')::bigint < 1 THEN
        RAISE EXCEPTION 'matching binding was not recorded: %', guidance;
    END IF;
    IF (guidance ->> 'statement_bound')::boolean THEN
        RAISE EXCEPTION 'statement binding leaked past ExecutorEnd: %', guidance;
    END IF;
END
$$;

RESET enable_seqscan;
RESET enable_sort;
RESET enable_bitmapscan;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.max_scan_tuples;
RESET hnsw.filter_strategy;
SELECT vector_hnsw_guidance_reset();
DROP FUNCTION nested_binding_abort_probe(boolean);
DROP TABLE nested_binding_abort_smoke CASCADE;
