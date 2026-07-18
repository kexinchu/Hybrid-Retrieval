\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS traversal_guidance_binding_smoke CASCADE;
CREATE TABLE traversal_guidance_binding_smoke (
    id integer PRIMARY KEY,
    eligible boolean NOT NULL,
    embedding vector(2) NOT NULL
);

INSERT INTO traversal_guidance_binding_smoke
SELECT i,
       i % 4 = 0,
       ARRAY[
           (((i * 37) % 1000)::double precision + i::double precision / 100000.0)::real,
           (((i * 97) % 1000)::double precision + i::double precision / 70000.0)::real
       ]::vector
FROM generate_series(1, 4000) AS i;

SET hnsw.build_seed = 42;
CREATE INDEX traversal_guidance_binding_smoke_hnsw
ON traversal_guidance_binding_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 100);
ANALYZE traversal_guidance_binding_smoke;
SELECT vector_hnsw_fragment_tracking_enable(
    'traversal_guidance_binding_smoke'::regclass
);

CREATE TEMP TABLE traversal_guidance_results (
    method text PRIMARY KEY,
    ids integer[] NOT NULL,
    profile jsonb
);

-- Build an independent exact matched-predicate oracle. The small per-row
-- coordinate offsets make the top-k ID ordering deterministic as well.
SET enable_indexscan = off;
SET enable_bitmapscan = off;
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'exact_eligible', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE eligible
    ORDER BY distance, id
    LIMIT 10
) AS exact;
RESET enable_indexscan;
RESET enable_bitmapscan;

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.iterative_scan = off;
SET hnsw.ef_search = 100;
SET hnsw.page_access = off;
SET hnsw.index_page_access = off;

-- Successful production D1: membership is checked before vector distance.
-- This seeded fixture has a deterministic matched-ID contract; production D1
-- remains approximate ANN and must be calibrated against stock at matched recall.
SET hnsw.filter_strategy = off;
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'stock_eligible', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS stock;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stock_eligible';

SELECT vector_hnsw_guidance_activate(
    'traversal_guidance_binding_smoke_hnsw'::regclass,
    ARRAY['sql:eligible'],
    'exact'
);
SET hnsw.filter_strategy = traversal_guided;
-- The target GUC is deliberately below both LIMIT and ef_search.  A successful
-- guided phase must still collect at least the layer-0 ef candidate batch.
SET hnsw.traversal_guided_target = 2;
SET hnsw.traversal_guided_max_bridge_hops = 3;
SET hnsw.traversal_guided_max_bridge_work = 10000;
SET hnsw.traversal_guided_min_skip_rate = 0.5;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'traversal_guided', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_binding_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS guided;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'traversal_guided';

DO $$
DECLARE
    stock_ids integer[];
    guided_ids integer[];
    exact_ids integer[];
    stock_profile jsonb;
    guided_profile jsonb;
BEGIN
    SELECT ids, profile INTO stock_ids, stock_profile
    FROM traversal_guidance_results WHERE method = 'stock_eligible';
    SELECT ids, profile INTO guided_ids, guided_profile
    FROM traversal_guidance_results WHERE method = 'traversal_guided';
    SELECT ids INTO exact_ids
    FROM traversal_guidance_results WHERE method = 'exact_eligible';

    IF guided_ids IS DISTINCT FROM exact_ids OR
       stock_ids IS DISTINCT FROM exact_ids OR cardinality(guided_ids) <> 10 OR
       EXISTS (
           SELECT 1
           FROM unnest(guided_ids) AS selected(id)
           JOIN traversal_guidance_binding_smoke AS source USING (id)
           WHERE NOT source.eligible
       ) THEN
        RAISE EXCEPTION 'seeded traversal-guided matched-result contract failed: exact %, stock %, guided %',
            exact_ids, stock_ids, guided_ids;
    END IF;
    IF (guided_profile->>'profile_semantics_version')::int <> 4 OR
       NOT (guided_profile->>'planner_proof_succeeded')::boolean OR
       guided_profile->>'final_path' <> 'guided' OR
       (guided_profile->>'pre_distance_membership_checks')::bigint <= 0 OR
       (guided_profile->>'pre_distance_membership_misses')::bigint <= 0 OR
       (guided_profile->>'traversal_guided_admissions')::bigint < 100 OR
       (guided_profile->>'distance_computations_avoided')::bigint <= 0 OR
       (guided_profile->>'miss_bridge_nodes')::bigint <= 0 OR
       (guided_profile->>'miss_bridge_edges')::bigint <= 0 OR
       (guided_profile->>'fallback_requests')::bigint <> 0 OR
       (guided_profile->>'stock_bypass_requests')::bigint <> 0 THEN
        RAISE EXCEPTION 'traversal-guided proof or pre-distance counters failed: %',
            guided_profile;
    END IF;
    IF (guided_profile->>'distance_compute_count')::bigint >=
           (stock_profile->>'distance_compute_count')::bigint THEN
        RAISE EXCEPTION 'traversal-guided did not reduce distance work: stock %, guided %',
            stock_profile, guided_profile;
    END IF;
    -- This sparse fixture does not claim fewer layer-0 expansions.  Instead it
    -- proves the expansion and avoided-distance partitions exactly.
    IF guided_profile->>'distance_compute_count' IS DISTINCT FROM
           guided_profile->>'guided_phase_distance_computations' OR
       guided_profile->>'traversal_expanded_nodes' IS DISTINCT FROM
           guided_profile->>'guided_expanded_nodes' OR
       (guided_profile->>'guided_expanded_nodes')::bigint <>
           (guided_profile->>'traversal_matching_expanded')::bigint +
           (guided_profile->>'traversal_bridge_expanded')::bigint OR
       guided_profile->>'distance_computations_avoided' IS DISTINCT FROM
           guided_profile->>'distance_computations_avoided_attempted' THEN
        RAISE EXCEPTION 'successful guided phase totals are ambiguous: %',
            guided_profile;
    END IF;
END
$$;

-- The production strategy cannot enter traversal guidance without a proven
-- marker/predicate binding. Missing proof is an exact stock bypass.
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'proof_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS bypass;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'proof_bypass';

DO $$
DECLARE
    stock_ids integer[];
    bypass_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids
    FROM traversal_guidance_results WHERE method = 'stock_eligible';
    SELECT ids, traversal_guidance_results.profile INTO bypass_ids, profile
    FROM traversal_guidance_results WHERE method = 'proof_bypass';
    IF bypass_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'stock_bypass_reason' <> 'no_proven_guide' OR
       profile->>'planner_proof_bypass_reason' <> 'no_statement_binding' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'unproven traversal request did not use exact stock: %',
            profile;
    END IF;
END
$$;

-- A rare predicate plus zero bridge hops cannot certify the target. The entire
-- guided context must be discarded before a fresh stock traversal starts.
SET hnsw.filter_strategy = off;
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'stock_rare', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE id % 20 = 0
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 5
) AS stock;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stock_rare';

SELECT vector_hnsw_guidance_activate(
    'traversal_guidance_binding_smoke_hnsw'::regclass,
    ARRAY['sql:id % 20 = 0'],
    'exact'
);
SET hnsw.filter_strategy = traversal_guided;
SET hnsw.traversal_guided_target = 100;
SET hnsw.traversal_guided_max_bridge_hops = 0;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'fresh_stock_fallback', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_binding_smoke_hnsw'::regclass,
               ARRAY['sql:id % 20 = 0'],
               'exact'
           ) OFFSET 0)
      AND id % 20 = 0
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 5
) AS fallback;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'fresh_stock_fallback';

DO $$
DECLARE
    stock_ids integer[];
    fallback_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids
    FROM traversal_guidance_results WHERE method = 'stock_rare';
    SELECT ids, traversal_guidance_results.profile INTO fallback_ids, profile
    FROM traversal_guidance_results WHERE method = 'fresh_stock_fallback';

    IF fallback_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'fresh_stock_fallback' OR
       profile->>'fallback_reason' <> 'bridge_hop_budget' OR
       (profile->>'fallback_requests')::bigint <> 1 OR
       (profile->>'pre_distance_membership_checks')::bigint <= 0 OR
       (profile->>'distance_computations_avoided_attempted')::bigint <= 0 OR
       (profile->>'distance_computations_avoided')::bigint <> 0 OR
       (profile->>'traversal_exhausted_terminations')::bigint <> 0 OR
       (profile->>'fallback_stock_expanded_nodes')::bigint <= 0 OR
       (profile->>'fallback_stock_distance_computations')::bigint <= 0 THEN
        RAISE EXCEPTION 'fresh-stock fallback failed: stock %, fallback %, profile %',
            stock_ids, fallback_ids, profile;
    END IF;
    IF (profile->>'distance_compute_count')::bigint <>
         (profile->>'guided_phase_distance_computations')::bigint +
         (profile->>'fallback_stock_distance_computations')::bigint OR
       (profile->>'traversal_expanded_nodes')::bigint <>
         (profile->>'guided_expanded_nodes')::bigint +
         (profile->>'fallback_stock_expanded_nodes')::bigint THEN
        RAISE EXCEPTION 'fallback phase totals are ambiguous: %', profile;
    END IF;
END
$$;

-- Work-budget uncertainty is independently terminal and also restarts from a
-- fresh stock entry point.  One unit cannot pay for a level-0 bridge expansion.
SET hnsw.traversal_guided_max_bridge_hops = 3;
SET hnsw.traversal_guided_max_bridge_work = 1;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'fresh_stock_work_fallback', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_binding_smoke_hnsw'::regclass,
               ARRAY['sql:id % 20 = 0'],
               'exact'
           ) OFFSET 0)
      AND id % 20 = 0
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 5
) AS fallback;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'fresh_stock_work_fallback';

DO $$
DECLARE
    stock_ids integer[];
    fallback_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids
    FROM traversal_guidance_results WHERE method = 'stock_rare';
    SELECT ids, traversal_guidance_results.profile INTO fallback_ids, profile
    FROM traversal_guidance_results WHERE method = 'fresh_stock_work_fallback';

    IF fallback_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'fresh_stock_fallback' OR
       profile->>'fallback_reason' <> 'bridge_work_budget' OR
       (profile->>'fallback_requests')::bigint <> 1 OR
       (profile->>'guided_expanded_nodes')::bigint <> 0 OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 OR
       (profile->>'distance_computations_avoided_attempted')::bigint <> 0 OR
       (profile->>'distance_computations_avoided')::bigint <> 0 OR
       (profile->>'traversal_exhausted_terminations')::bigint <> 0 OR
       (profile->>'fallback_stock_expanded_nodes')::bigint <= 0 OR
       (profile->>'fallback_stock_distance_computations')::bigint <= 0 THEN
        RAISE EXCEPTION 'fresh-stock work fallback failed: stock %, fallback %, profile %',
            stock_ids, fallback_ids, profile;
    END IF;
    IF (profile->>'distance_compute_count')::bigint <>
         (profile->>'guided_phase_distance_computations')::bigint +
         (profile->>'fallback_stock_distance_computations')::bigint OR
       (profile->>'traversal_expanded_nodes')::bigint <>
         (profile->>'guided_expanded_nodes')::bigint +
         (profile->>'fallback_stock_expanded_nodes')::bigint THEN
        RAISE EXCEPTION 'work fallback phase totals are ambiguous: %', profile;
    END IF;
END
$$;

-- A 95%-matching predicate has too little estimated benefit and must take the
-- exact stock path before any traversal membership check.
SET hnsw.filter_strategy = off;
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'stock_high_match', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE id % 20 <> 0
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS stock;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stock_high_match';

SELECT vector_hnsw_guidance_activate(
    'traversal_guidance_binding_smoke_hnsw'::regclass,
    ARRAY['sql:id % 20 <> 0'],
    'exact'
);
SET hnsw.filter_strategy = traversal_guided;
SET hnsw.traversal_guided_target = 10;
SET hnsw.traversal_guided_max_bridge_hops = 3;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'stock_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_binding_smoke_hnsw'::regclass,
               ARRAY['sql:id % 20 <> 0'],
               'exact'
           ) OFFSET 0)
      AND id % 20 <> 0
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS bypass;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stock_bypass';

DO $$
DECLARE
    stock_ids integer[];
    bypass_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids
    FROM traversal_guidance_results WHERE method = 'stock_high_match';
    SELECT ids, traversal_guidance_results.profile INTO bypass_ids, profile
    FROM traversal_guidance_results WHERE method = 'stock_bypass';

    IF bypass_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'stock_bypass_reason' <> 'low_estimated_skip_rate' OR
       (profile->>'stock_bypass_requests')::bigint <> 1 OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 OR
       (profile->>'distance_computations_avoided')::bigint <> 0 OR
       (profile->>'fallback_requests')::bigint <> 0 OR
       NOT (profile->>'traversal_estimated_skip_rate_valid')::boolean OR
       (profile->>'traversal_estimated_skip_rate')::double precision >= 0.5 THEN
        RAISE EXCEPTION 'low-benefit stock bypass failed: stock %, bypass %, profile %',
            stock_ids, bypass_ids, profile;
    END IF;
    IF profile->>'distance_compute_count' IS DISTINCT FROM
         profile->>'stock_phase_distance_computations' OR
       profile->>'traversal_expanded_nodes' IS DISTINCT FROM
         profile->>'stock_phase_expanded_nodes' THEN
        RAISE EXCEPTION 'stock bypass phase totals are ambiguous: %', profile;
    END IF;
END
$$;

-- Iterative/resume scans conservatively bypass before creating guided heaps;
-- this avoids any possibility of a late fallback after output has begun.
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
    'traversal_guidance_binding_smoke_hnsw'::regclass,
    ARRAY['sql:eligible'],
    'exact'
);
SET hnsw.iterative_scan = strict_order;
SET hnsw.traversal_guided_target = 10;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'iterative_stock_bypass', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_binding_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'exact'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS bypass;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'iterative_stock_bypass';

DO $$
DECLARE
    stock_ids integer[];
    bypass_ids integer[];
    profile jsonb;
BEGIN
    SELECT ids INTO stock_ids
    FROM traversal_guidance_results WHERE method = 'stock_eligible';
    SELECT ids, traversal_guidance_results.profile INTO bypass_ids, profile
    FROM traversal_guidance_results WHERE method = 'iterative_stock_bypass';
    IF bypass_ids IS DISTINCT FROM stock_ids OR
       profile->>'final_path' <> 'stock_bypass' OR
       profile->>'stock_bypass_reason' <> 'iterative_scan' OR
       (profile->>'pre_distance_membership_checks')::bigint <> 0 OR
       profile->>'distance_compute_count' IS DISTINCT FROM
         profile->>'stock_phase_distance_computations' OR
       profile->>'traversal_expanded_nodes' IS DISTINCT FROM
         profile->>'stock_phase_expanded_nodes' THEN
        RAISE EXCEPTION 'iterative request did not bypass before guided state: %',
            profile;
    END IF;
END
$$;
SET hnsw.iterative_scan = off;

-- Bloom fragments are the formal compact D1 representation.  Their
-- false-positive model must admit useful traversal while preserving the
-- no-false-negative membership contract.
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
    'traversal_guidance_binding_smoke_hnsw'::regclass,
    ARRAY['sql:eligible'],
    'bloom'
);
SET hnsw.traversal_guided_target = 100;
SET hnsw.traversal_guided_max_bridge_hops = 3;
SET hnsw.traversal_guided_max_bridge_work = 10000;
SET hnsw.traversal_guided_min_skip_rate = 0.5;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO traversal_guidance_results (method, ids)
SELECT 'traversal_guided_bloom', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[500,500]'::vector AS distance
    FROM traversal_guidance_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'traversal_guidance_binding_smoke_hnsw'::regclass,
               ARRAY['sql:eligible'],
               'bloom'
           ) OFFSET 0)
      AND eligible
    ORDER BY embedding <-> '[500,500]'::vector
    LIMIT 10
) AS guided;
UPDATE traversal_guidance_results
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'traversal_guided_bloom';

DO $$
DECLARE
    profile jsonb;
BEGIN
    SELECT traversal_guidance_results.profile INTO profile
    FROM traversal_guidance_results WHERE method = 'traversal_guided_bloom';
    IF profile->>'final_path' <> 'guided' OR
       NOT (profile->>'traversal_estimated_skip_rate_valid')::boolean OR
       (profile->>'traversal_estimated_skip_rate')::double precision < 0.5 OR
       (profile->>'pre_distance_membership_checks')::bigint <= 0 OR
       (profile->>'distance_computations_avoided')::bigint <= 0 THEN
        RAISE EXCEPTION 'Bloom traversal admission was not exercised: %', profile;
    END IF;
END
$$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.iterative_scan;
RESET hnsw.ef_search;
RESET hnsw.page_access;
RESET hnsw.index_page_access;
RESET hnsw.filter_strategy;
RESET hnsw.traversal_guided_target;
RESET hnsw.traversal_guided_max_bridge_hops;
RESET hnsw.traversal_guided_max_bridge_work;
RESET hnsw.traversal_guided_min_skip_rate;
RESET hnsw.build_seed;
DROP TABLE traversal_guidance_binding_smoke CASCADE;
