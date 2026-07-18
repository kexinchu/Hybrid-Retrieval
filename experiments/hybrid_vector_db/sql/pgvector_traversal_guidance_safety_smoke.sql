\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS traversal_guidance_safety_smoke CASCADE;
CREATE TABLE traversal_guidance_safety_smoke (
    id integer PRIMARY KEY,
    eligible boolean NOT NULL,
    category integer NOT NULL,
    embedding vector(2) NOT NULL
);

INSERT INTO traversal_guidance_safety_smoke
SELECT i,
       i % 3 = 0,
       i % 4,
       ARRAY[
           (((i * 37) % 1000)::double precision + i::double precision / 100000.0)::real,
           (((i * 97) % 1000)::double precision + i::double precision / 70000.0)::real
       ]::vector
FROM generate_series(1, 4000) AS i;

-- These two rows share one vector element.  The nonmatching TID must still be
-- suppressed when the element is admitted by its matching sibling.
INSERT INTO traversal_guidance_safety_smoke VALUES
    (4001, true, 1, '[2000,2000]'),
    (4002, false, 2, '[2000,2000]');

SET hnsw.build_seed = 314159;
CREATE INDEX traversal_guidance_safety_smoke_hnsw
ON traversal_guidance_safety_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 100);
ANALYZE traversal_guidance_safety_smoke;
SELECT vector_hnsw_fragment_tracking_enable(
    'traversal_guidance_safety_smoke'::regclass
);

CREATE TEMP TABLE traversal_guidance_safety_results (
    method text PRIMARY KEY,
    ids integer[] NOT NULL,
    profile jsonb
);

CREATE TEMP TABLE traversal_guidance_page_results (
    mode text PRIMARY KEY,
    ids integer[] NOT NULL,
    distances double precision[] NOT NULL,
    profile jsonb
);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.iterative_scan = off;
SET hnsw.ef_search = 80;
SET hnsw.page_access = off;
SET hnsw.page_window = 32;
SET hnsw.index_page_access = off;
SET hnsw.traversal_guided_target = 2;
SET hnsw.traversal_guided_max_bridge_hops = 3;
SET hnsw.traversal_guided_max_bridge_work = 20000;
SET hnsw.traversal_guided_min_skip_rate = 0.5;

SET hnsw.filter_strategy = off;
SELECT vector_hnsw_guidance_reset();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'stock_eligible', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS stock;

INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'stock_residual', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE eligible AND category = 1
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS stock;

SELECT vector_hnsw_guidance_activate(
    'traversal_guidance_safety_smoke_hnsw'::regclass,
    ARRAY['sql:eligible'],
    'exact'
);

-- actual => guide is true, but guide => actual is false because category is a
-- residual predicate.  Hard traversal pruning must bypass before graph checks.
SET hnsw.filter_strategy = traversal_guided;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'strict_superset_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
      AND category = 1
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS bypass;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'strict_superset_bypass';

DO $$
DECLARE
    stock_ids integer[];
    bypass_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_residual';
    SELECT ids, traversal_guidance_safety_results.profile INTO bypass_ids, profile
    FROM traversal_guidance_safety_results
    WHERE method = 'strict_superset_bypass';

    IF bypass_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'stock_bypass_reason' <> 'no_proven_guide' OR
       profile->>'planner_proof_bypass_reason' <> 'predicate_not_implied' OR
       (profile->>'planner_proof_succeeded')::boolean OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'strict-superset guide did not fail closed: %', profile;
    END IF;
END
$$;

-- Validation-only safe_guided retains the weaker one-way proof because the
-- residual SQL predicate still runs after stock graph traversal.
SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'safe_guided_residual', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
      AND category = 1
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS safe;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'safe_guided_residual';

DO $$
DECLARE
    stock_ids integer[];
    safe_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_residual';
    SELECT ids, traversal_guidance_safety_results.profile INTO safe_ids, profile
    FROM traversal_guidance_safety_results
    WHERE method = 'safe_guided_residual';

    IF safe_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'validation_only' OR
       NOT (profile->>'planner_proof_succeeded')::boolean OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 OR
       (profile->>'heap_validation_guidance_checks')::bigint <= 0 THEN
        RAISE EXCEPTION 'safe_guided lost its validation-only residual contract: %',
            profile;
    END IF;
END
$$;

-- A mismatched marker is inert and must retain exact stock IDs.
SET hnsw.filter_strategy = traversal_guided;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'binding_mismatch', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:category = 1'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS mismatch;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'binding_mismatch';

DO $$
DECLARE
    stock_ids integer[];
    mismatch_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_eligible';
    SELECT ids, traversal_guidance_safety_results.profile
    INTO mismatch_ids, profile
    FROM traversal_guidance_safety_results WHERE method = 'binding_mismatch';

    IF mismatch_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'planner_proof_bypass_reason' <> 'no_statement_binding' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'mismatched binding changed traversal output: %', profile;
    END IF;
END
$$;

-- A generic external parameter is not substituted into a hard-pruning proof.
SET plan_cache_mode = force_generic_plan;
PREPARE traversal_guidance_param(boolean) AS
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'param_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible = $1
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS parameterized;
SELECT vector_hnsw_reset_scan_profile();
EXECUTE traversal_guidance_param(true);
DEALLOCATE traversal_guidance_param;
RESET plan_cache_mode;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'param_bypass';

DO $$
DECLARE
    stock_ids integer[];
    param_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_eligible';
    SELECT ids, traversal_guidance_safety_results.profile INTO param_ids, profile
    FROM traversal_guidance_safety_results WHERE method = 'param_bypass';

    IF param_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'planner_proof_bypass_reason' <> 'param_extern_unresolved' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'PARAM_EXTERN hard-pruning proof did not fail closed: %',
            profile;
    END IF;
END
$$;

-- Join plan quals can live above the IndexScan, so traversal guidance refuses
-- the entire join-shaped statement even when the local scan qual looks exact.
CREATE TEMP TABLE traversal_guidance_join_key (category integer PRIMARY KEY);
INSERT INTO traversal_guidance_join_key VALUES (1);
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'join_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT source.id, source.embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke AS source
    JOIN traversal_guidance_join_key AS key
      ON key.category = source.category
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND source.eligible
    ORDER BY source.embedding <-> '[500,500]'::vector
    LIMIT 10
) AS joined;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'join_bypass';

DO $$
DECLARE
    stock_ids integer[];
    join_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_residual';
    SELECT ids, traversal_guidance_safety_results.profile INTO join_ids, profile
    FROM traversal_guidance_safety_results WHERE method = 'join_bypass';

    IF join_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'planner_proof_bypass_reason' <> 'non_target_var' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'join-shaped traversal proof did not fail closed: %',
            profile;
    END IF;
END
$$;

-- RLS is rejected before any traversal membership check, including for owners
-- and superusers whose current execution might bypass the policy.
ALTER TABLE traversal_guidance_safety_smoke ENABLE ROW LEVEL SECURITY;
CREATE POLICY traversal_guidance_safety_policy
ON traversal_guidance_safety_smoke USING (eligible);
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'rls_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS protected;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'rls_bypass';
ALTER TABLE traversal_guidance_safety_smoke DISABLE ROW LEVEL SECURITY;
DROP POLICY traversal_guidance_safety_policy
ON traversal_guidance_safety_smoke;

DO $$
DECLARE
    stock_ids integer[];
    rls_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_eligible';
    SELECT ids, traversal_guidance_safety_results.profile INTO rls_ids, profile
    FROM traversal_guidance_safety_results WHERE method = 'rls_bypass';

    IF rls_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'planner_proof_bypass_reason' <> 'rls_or_security_barrier' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'RLS traversal proof did not fail closed: %', profile;
    END IF;
END
$$;

-- The shared duplicate element is admitted, but its nonmatching TID is not
-- returned to the executor.
SET hnsw.page_access = off;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'duplicate_tid_validation', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[2000,2000]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[2000,2000]'::vector
    LIMIT 2
) AS duplicates;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'duplicate_tid_validation';

DO $$
DECLARE
    result_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids, traversal_guidance_safety_results.profile INTO result_ids, profile
    FROM traversal_guidance_safety_results
    WHERE method = 'duplicate_tid_validation';

    IF result_ids[1] IS DISTINCT FROM 4001 OR 4002 = ANY(result_ids) OR
       profile->>'final_path' <> 'guided' OR
       (profile->>'traversal_heap_tids_suppressed')::bigint < 1 THEN
        RAISE EXCEPTION 'duplicate-element TID validation failed: ids %, profile %',
            result_ids, profile;
    END IF;
END
$$;

-- All public page modes must preserve the HNSW distance order.  Internally,
-- traversal_guided maps reorder to the order-preserving prefetch path.
SET hnsw.page_access = off;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_page_results
SELECT 'off', array_agg(id), array_agg(distance), NULL
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS ordered;
UPDATE traversal_guidance_page_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE mode = 'off';

SET hnsw.page_access = prefetch;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_page_results
SELECT 'prefetch', array_agg(id), array_agg(distance), NULL
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS ordered;
UPDATE traversal_guidance_page_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE mode = 'prefetch';

SET hnsw.page_access = reorder;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_page_results
SELECT 'reorder', array_agg(id), array_agg(distance), NULL
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS ordered;
UPDATE traversal_guidance_page_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE mode = 'reorder';

DO $$
DECLARE
    off_ids integer[];
    row_data record;
    i integer;
BEGIN
    SELECT ids INTO off_ids FROM traversal_guidance_page_results
    WHERE mode = 'off';

    FOR row_data IN SELECT * FROM traversal_guidance_page_results LOOP
        IF row_data.ids IS DISTINCT FROM off_ids OR cardinality(row_data.ids) <> 20 OR
           row_data.profile->>'final_path' <> 'guided' THEN
            RAISE EXCEPTION 'page mode changed guided IDs/order: %', row_data;
        END IF;
        IF row_data.mode = 'off' AND
           (row_data.profile->>'page_access_batches')::bigint <> 0 THEN
            RAISE EXCEPTION 'page_access=off unexpectedly buffered candidates: %',
                row_data.profile;
        END IF;
        IF row_data.mode <> 'off' AND
           (row_data.profile->>'page_access_batches')::bigint <= 0 THEN
            RAISE EXCEPTION 'page mode did not exercise the page buffer: %', row_data;
        END IF;
        FOR i IN 2..cardinality(row_data.distances) LOOP
            IF row_data.distances[i] < row_data.distances[i - 1] THEN
                RAISE EXCEPTION 'page mode % broke distance order at %: %',
                    row_data.mode, i, row_data.distances;
            END IF;
        END LOOP;
    END LOOP;
END
$$;

-- A relation epoch change invalidates the exact TID cache.  The first scan
-- deactivates it and returns a fresh-stock result containing the new nearest ID.
SET hnsw.page_access = off;
INSERT INTO traversal_guidance_safety_smoke
VALUES (0, true, 1, '[500,500]');
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'stale_epoch_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_safety_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS stale;
UPDATE traversal_guidance_safety_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stale_epoch_bypass';

SET hnsw.filter_strategy = off;
INSERT INTO traversal_guidance_safety_results (method, ids)
SELECT 'stock_after_stale', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_safety_smoke
    WHERE eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 20
) AS stock;

DO $$
DECLARE
    stock_ids integer[];
    stale_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM traversal_guidance_safety_results
    WHERE method = 'stock_after_stale';
    SELECT ids, traversal_guidance_safety_results.profile INTO stale_ids, profile
    FROM traversal_guidance_safety_results WHERE method = 'stale_epoch_bypass';

    IF stale_ids IS DISTINCT FROM stock_ids OR stale_ids[1] IS DISTINCT FROM 0 OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'planner_proof_bypass_reason' <> 'stale_relation' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'stale epoch did not restart exact stock: ids %, profile %',
            stale_ids, profile;
    END IF;
END
$$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.iterative_scan;
RESET hnsw.ef_search;
RESET hnsw.page_access;
RESET hnsw.page_window;
RESET hnsw.index_page_access;
RESET hnsw.filter_strategy;
RESET hnsw.traversal_guided_target;
RESET hnsw.traversal_guided_max_bridge_hops;
RESET hnsw.traversal_guided_max_bridge_work;
RESET hnsw.traversal_guided_min_skip_rate;
RESET hnsw.build_seed;
DROP TABLE traversal_guidance_safety_smoke CASCADE;
