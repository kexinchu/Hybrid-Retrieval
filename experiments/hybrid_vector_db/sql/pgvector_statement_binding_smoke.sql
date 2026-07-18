\set ON_ERROR_STOP on

DROP TABLE IF EXISTS sqlens_binding_smoke CASCADE;
CREATE TABLE sqlens_binding_smoke (
    id bigint PRIMARY KEY,
    category integer NOT NULL,
    embedding vector(2) NOT NULL
);
INSERT INTO sqlens_binding_smoke
SELECT value,
       CASE WHEN value % 10 = 0 THEN 1 ELSE 2 END,
       ARRAY[value::real, 0::real]::vector
FROM generate_series(1, 200) AS value;
CREATE INDEX sqlens_binding_smoke_hnsw
ON sqlens_binding_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 40);
SELECT vector_hnsw_fragment_tracking_enable('sqlens_binding_smoke'::regclass);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 80;
SET hnsw.iterative_scan = strict_order;
SET hnsw.max_scan_tuples = 10000;
SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_guidance_activate(
    'sqlens_binding_smoke_hnsw'::regclass,
    ARRAY['sql:category = 1'],
    'exact'
);

CREATE TEMP TABLE sqlens_binding_results (
    method text PRIMARY KEY,
    ids bigint[] NOT NULL
);

-- An activated guide is inert unless the same SQL statement binds it.
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO sqlens_binding_results
SELECT 'stock', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE category = 2
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 3
) AS stock;
DO $$
DECLARE profile jsonb;
DECLARE stock_ids bigint[];
BEGIN
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    SELECT ids INTO stock_ids FROM sqlens_binding_results WHERE method = 'stock';
    IF (profile->>'guidance_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'unbound guidance affected a scan: %', profile;
    END IF;
    IF (profile->>'profile_semantics_version')::int <> 4 OR
       NOT (profile->>'planner_proof_attempted')::boolean OR
       (profile->>'planner_proof_succeeded')::boolean OR
       profile->>'planner_proof_bypass_reason' <> 'no_statement_binding' THEN
        RAISE EXCEPTION 'unbound active guide was not distinguished in scan profile: %',
            profile;
    END IF;
    IF coalesce(cardinality(stock_ids), 0) <> 3 THEN
        RAISE EXCEPTION 'stock binding baseline was empty or incomplete: %', stock_ids;
    END IF;
END
$$;

-- A mismatched binding marker also fails open to stock HNSW.
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO sqlens_binding_results
SELECT 'signature_mismatch', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'sqlens_binding_smoke_hnsw'::regclass,
               ARRAY['sql:category = 2'],
               'exact'
           ) OFFSET 0)
      AND category = 2
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 3
) AS mismatch;
DO $$
DECLARE profile jsonb;
DECLARE guidance jsonb;
DECLARE stock_ids bigint[];
DECLARE mismatch_ids bigint[];
BEGIN
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance;
    SELECT ids INTO stock_ids FROM sqlens_binding_results WHERE method = 'stock';
    SELECT ids INTO mismatch_ids FROM sqlens_binding_results WHERE method = 'signature_mismatch';
    IF (profile->>'guidance_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'mismatched guidance affected a scan: %', profile;
    END IF;
    IF NOT (profile->>'planner_proof_attempted')::boolean OR
       (profile->>'planner_proof_succeeded')::boolean OR
       profile->>'planner_proof_bypass_reason' <> 'no_statement_binding' THEN
        RAISE EXCEPTION 'mismatched marker scan outcome was not retained: %', profile;
    END IF;
    IF mismatch_ids IS DISTINCT FROM stock_ids THEN
        RAISE EXCEPTION 'mismatched marker changed SQL results: stock %, mismatch %',
            stock_ids, mismatch_ids;
    END IF;
    IF (guidance->>'binding_mismatches')::bigint < 1 THEN
        RAISE EXCEPTION 'binding mismatch was not recorded: %', guidance;
    END IF;
END
$$;

-- The marker matches the active guide, but the actual scan predicate does
-- not imply it. Planner proof must reject guidance while the SQL marker stays
-- true and the query returns the stock ordered result.
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO sqlens_binding_results
SELECT 'predicate_mismatch', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'sqlens_binding_smoke_hnsw'::regclass,
               ARRAY['sql:category = 1'],
               'exact'
           ) OFFSET 0)
      AND category = 2
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 3
) AS mismatch;
DO $$
DECLARE profile jsonb;
DECLARE guidance jsonb;
DECLARE stock_ids bigint[];
DECLARE mismatch_ids bigint[];
BEGIN
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance;
    SELECT ids INTO stock_ids FROM sqlens_binding_results WHERE method = 'stock';
    SELECT ids INTO mismatch_ids FROM sqlens_binding_results WHERE method = 'predicate_mismatch';
    IF mismatch_ids IS DISTINCT FROM stock_ids OR coalesce(cardinality(mismatch_ids), 0) = 0 THEN
        RAISE EXCEPTION 'planner-proof bypass changed SQL results: stock %, mismatch %',
            stock_ids, mismatch_ids;
    END IF;
    IF (profile->>'guidance_checks')::bigint <> 0 OR
       NOT (profile->>'planner_proof_attempted')::boolean OR
       (profile->>'planner_proof_succeeded')::boolean OR
       profile->>'planner_proof_bypass_reason' <> 'predicate_not_implied' OR
       guidance->>'planner_proof_bypass_reason' <> 'predicate_not_implied' OR
       (guidance->>'planner_proof_failures')::bigint < 1 THEN
        RAISE EXCEPTION 'unsafe predicate did not fail open with a reason: scan %, guide %',
            profile, guidance;
    END IF;
END
$$;

-- A matching marker is evaluated as an InitPlan before HNSW starts.
SELECT vector_hnsw_reset_scan_profile();
SELECT id
FROM sqlens_binding_smoke
WHERE (SELECT vector_hnsw_guidance_bind(
           'sqlens_binding_smoke_hnsw'::regclass,
           ARRAY['sql:category = 1'],
           'exact'
       ) OFFSET 0)
  AND category = 1
ORDER BY embedding <-> '[21,0]'::vector
LIMIT 10;
DO $$
DECLARE profile jsonb;
DECLARE guidance jsonb;
BEGIN
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance;
    IF (profile->>'guidance_checks')::bigint <= 0 OR
       (profile->>'guidance_skips')::bigint <= 0 THEN
        RAISE EXCEPTION 'matching statement binding was not used: %', profile;
    END IF;
    IF NOT (profile->>'planner_proof_attempted')::boolean OR
       NOT (profile->>'planner_proof_succeeded')::boolean OR
       profile->>'planner_proof_bypass_reason' <> 'none' OR
       (profile->>'planner_proof_plan_node_id')::int <= 0 OR
       (profile->>'planner_proof_index_oid')::bigint < 1 OR
       (profile->>'planner_proof_heap_oid')::bigint < 1 OR
       (profile->>'planner_proof_guide_generation')::bigint < 1 THEN
        RAISE EXCEPTION 'successful proof identity was not retained in scan profile: %',
            profile;
    END IF;
    IF (guidance->>'planner_proof_successes')::bigint < 1 OR
       guidance->>'planner_proof_bypass_reason' <> 'none' THEN
        RAISE EXCEPTION 'matching statement did not pass planner proof: %', guidance;
    END IF;
    IF (guidance->>'statement_bound')::boolean THEN
        RAISE EXCEPTION 'statement binding leaked past ExecutorEnd: %', guidance;
    END IF;
END
$$;

-- Generic plans retain PARAM_EXTERN nodes. Resolve both the filter parameter
-- and the runner-style self exclusion from this execution before proving.
CREATE TEMP TABLE sqlens_binding_param_baseline AS
SELECT (profile->>'planner_proof_successes')::bigint AS successes
FROM (SELECT vector_hnsw_guidance_profile()::jsonb AS profile) AS p;
SET plan_cache_mode = force_generic_plan;
PREPARE sqlens_binding_generic(integer, bigint) AS
INSERT INTO sqlens_binding_results
SELECT 'param_extern', array_agg(id ORDER BY distance)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'sqlens_binding_smoke_hnsw'::regclass,
               ARRAY['sql:category = 1'],
               'exact'
           ) OFFSET 0)
      AND category = $1
      AND id <> $2
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 3
) AS guided;
EXECUTE sqlens_binding_generic(1, 20);
DEALLOCATE sqlens_binding_generic;
RESET plan_cache_mode;

DO $$
DECLARE result_ids bigint[];
DECLARE successes_before bigint;
DECLARE guidance jsonb;
DECLARE scan_profile jsonb;
BEGIN
    SELECT r.ids INTO result_ids
    FROM sqlens_binding_results AS r
    WHERE r.method = 'param_extern';
    SELECT successes INTO successes_before FROM sqlens_binding_param_baseline;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance;
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO scan_profile;
    IF coalesce(cardinality(result_ids), 0) <> 3 OR 20 = ANY(result_ids) OR
       (guidance->>'planner_proof_successes')::bigint <> successes_before + 1 OR
       guidance->>'planner_proof_bypass_reason' <> 'none' OR
       NOT (scan_profile->>'planner_proof_succeeded')::boolean THEN
        RAISE EXCEPTION 'PARAM_EXTERN planner proof failed: ids %, profile %',
            result_ids, scan_profile;
    END IF;
END
$$;

-- Correlated subplans use PARAM_EXEC. They are deliberately not resolved for
-- proof, so the descriptor must stay stock and retain a param_exec reason.
INSERT INTO sqlens_binding_results
SELECT 'param_exec_stock',
       ARRAY(
           SELECT inner_t.id
           FROM sqlens_binding_smoke AS inner_t
           WHERE inner_t.category = 1
             AND inner_t.id <> outer_t.id
           ORDER BY inner_t.embedding <-> '[21,0]'::vector
           LIMIT 3
       )
FROM (
    SELECT id
    FROM sqlens_binding_smoke
    WHERE id = 20
    OFFSET 0
) AS outer_t;

SELECT vector_hnsw_reset_scan_profile();
INSERT INTO sqlens_binding_results
SELECT 'param_exec_bound',
       ARRAY(
           SELECT inner_t.id
           FROM sqlens_binding_smoke AS inner_t
           WHERE (SELECT vector_hnsw_guidance_bind(
                      'sqlens_binding_smoke_hnsw'::regclass,
                      ARRAY['sql:category = 1'],
                      'exact'
                  ) OFFSET 0)
             AND inner_t.category = 1
             AND inner_t.id <> outer_t.id
           ORDER BY inner_t.embedding <-> '[21,0]'::vector
           LIMIT 3
       )
FROM (
    SELECT id
    FROM sqlens_binding_smoke
    WHERE id = 20
    OFFSET 0
) AS outer_t;

DO $$
DECLARE stock_ids bigint[];
DECLARE bound_ids bigint[];
DECLARE profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM sqlens_binding_results WHERE method = 'param_exec_stock';
    SELECT ids INTO bound_ids FROM sqlens_binding_results WHERE method = 'param_exec_bound';
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    IF bound_ids IS DISTINCT FROM stock_ids OR
       NOT (profile->>'planner_proof_attempted')::boolean OR
       (profile->>'planner_proof_succeeded')::boolean OR
       profile->>'planner_proof_bypass_reason' <> 'param_exec' OR
       (profile->>'guidance_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'PARAM_EXEC scan did not fail open: stock %, bound %, profile %',
            stock_ids, bound_ids, profile;
    END IF;
END
$$;

-- Two scans of the same heap and index share a QueryDesc but not proof state.
-- Only the category=1 scan may consume the guide.
INSERT INTO sqlens_binding_results
SELECT 'same_table_stock',
       array_agg((guided.id * 1000 + stock.id)::bigint
                 ORDER BY guided.distance, stock.distance, guided.id, stock.id)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE category = 1
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 2 OFFSET 0
) AS guided
CROSS JOIN (
    SELECT id, embedding <-> '[22,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE category = 2
    ORDER BY embedding <-> '[22,0]'::vector
    LIMIT 2 OFFSET 0
) AS stock;

CREATE TEMP TABLE sqlens_binding_proof_baseline AS
SELECT (profile->>'planner_proof_successes')::bigint AS successes,
       (profile->>'planner_proof_failures')::bigint AS failures
FROM (SELECT vector_hnsw_guidance_profile()::jsonb AS profile) AS p;

INSERT INTO sqlens_binding_results
SELECT 'same_table_two_scans',
       array_agg((guided.id * 1000 + stock.id)::bigint
                 ORDER BY guided.distance, stock.distance, guided.id, stock.id)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE category = 1
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 2 OFFSET 0
) AS guided
CROSS JOIN (
    SELECT id, embedding <-> '[22,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE category = 2
    ORDER BY embedding <-> '[22,0]'::vector
    LIMIT 2 OFFSET 0
) AS stock
WHERE (SELECT vector_hnsw_guidance_bind(
           'sqlens_binding_smoke_hnsw'::regclass,
           ARRAY['sql:category = 1'],
           'exact'
       ) OFFSET 0);

DO $$
DECLARE pairs bigint[];
DECLARE stock_pairs bigint[];
DECLARE guidance jsonb;
DECLARE scan_profile jsonb;
DECLARE old_successes bigint;
DECLARE old_failures bigint;
DECLARE scan_successes int;
DECLARE scan_failures int;
BEGIN
    SELECT ids INTO pairs FROM sqlens_binding_results WHERE method = 'same_table_two_scans';
    SELECT ids INTO stock_pairs FROM sqlens_binding_results WHERE method = 'same_table_stock';
    SELECT successes, failures INTO old_successes, old_failures
    FROM sqlens_binding_proof_baseline;
    SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance;
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO scan_profile;
    SELECT count(*) FILTER (WHERE (proof->>'succeeded')::boolean),
           count(*) FILTER (WHERE NOT (proof->>'succeeded')::boolean AND
                                  proof->>'bypass_reason' = 'predicate_not_implied')
    INTO scan_successes, scan_failures
    FROM jsonb_array_elements(scan_profile->'planner_proofs') AS proofs(proof);
    IF pairs IS DISTINCT FROM stock_pairs OR cardinality(pairs) <> 4 THEN
        RAISE EXCEPTION 'same-table two-scan query changed stock rows: stock %, guided %',
            stock_pairs, pairs;
    END IF;
    IF (guidance->>'planner_proof_successes')::bigint <> old_successes + 1 OR
       (guidance->>'planner_proof_failures')::bigint <> old_failures + 1 THEN
        RAISE EXCEPTION 'same-table scan proof state crossed descriptors: %', guidance;
    END IF;
    IF (scan_profile->>'planner_proof_count')::int <> 2 OR
       scan_successes <> 1 OR scan_failures <> 1 THEN
        RAISE EXCEPTION 'same-table descriptor outcomes were not retained separately: %',
            scan_profile;
    END IF;
END
$$;

-- guided_collect binds the predicate inside layer-0 traversal: it keeps
-- nonmatching bridge nodes expandable but defers stock termination until it
-- has enough guided candidates.
SET hnsw.filter_strategy = guided_collect;
SET hnsw.ef_search = 10;
SET hnsw.guided_collect_target = 10;
SELECT vector_hnsw_reset_scan_profile();
SELECT id
FROM sqlens_binding_smoke
WHERE (SELECT vector_hnsw_guidance_bind(
           'sqlens_binding_smoke_hnsw'::regclass,
           ARRAY['sql:category = 1'],
           'exact'
       ) OFFSET 0)
  AND category = 1
ORDER BY embedding <-> '[150,0]'::vector
LIMIT 10;
DO $$
DECLARE profile jsonb;
BEGIN
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    IF (profile->>'traversal_guidance_checks')::bigint <= 0 OR
       (profile->>'neighbor_expansion_guidance_checks')::bigint <= 0 OR
       (profile->>'traversal_matching_expanded')::bigint <= 0 OR
       (profile->>'traversal_bridge_expanded')::bigint <= 0 OR
       (profile->>'traversal_stop_deferrals')::bigint <= 0 OR
       (profile->>'traversal_guided_admissions')::bigint < 10 THEN
        RAISE EXCEPTION 'guided_collect did not alter traversal: %', profile;
    END IF;
END
$$;

-- off must preserve stock behavior even when activation and binding match.
SET hnsw.filter_strategy = off;
SELECT vector_hnsw_reset_scan_profile();
SELECT id
FROM sqlens_binding_smoke
WHERE (SELECT vector_hnsw_guidance_bind(
           'sqlens_binding_smoke_hnsw'::regclass,
           ARRAY['sql:category = 1'],
           'exact'
       ) OFFSET 0)
  AND category = 1
ORDER BY embedding <-> '[21,0]'::vector
LIMIT 10;
DO $$
DECLARE profile jsonb;
BEGIN
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    IF (profile->>'guidance_checks')::bigint <> 0 THEN
        RAISE EXCEPTION 'filter_strategy=off did not preserve stock behavior: %', profile;
    END IF;
END
$$;

-- With no active guide the marker is still true, the HNSW scan is stock, and
-- the descriptor profile distinguishes this from an attempted failed proof.
SELECT vector_hnsw_guidance_reset();
SET hnsw.ef_search = 80;
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO sqlens_binding_results
SELECT 'no_active', array_agg(id ORDER BY distance, id)
FROM (
    SELECT id, embedding <-> '[21,0]'::vector AS distance
    FROM sqlens_binding_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'sqlens_binding_smoke_hnsw'::regclass,
               ARRAY['sql:category = 1'],
               'exact'
           ) OFFSET 0)
      AND category = 2
    ORDER BY embedding <-> '[21,0]'::vector
    LIMIT 3
) AS stock;
DO $$
DECLARE stock_ids bigint[];
DECLARE no_active_ids bigint[];
DECLARE profile jsonb;
BEGIN
    SELECT ids INTO stock_ids FROM sqlens_binding_results WHERE method = 'stock';
    SELECT ids INTO no_active_ids FROM sqlens_binding_results WHERE method = 'no_active';
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
    IF no_active_ids IS DISTINCT FROM stock_ids OR
       (profile->>'planner_proof_attempted')::boolean OR
       (profile->>'planner_proof_succeeded')::boolean OR
       profile->>'planner_proof_bypass_reason' <> 'no_active_guide' THEN
        RAISE EXCEPTION 'no-active marker did not remain stock: ids %, profile %',
            no_active_ids, profile;
    END IF;
END
$$;

RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.max_scan_tuples;
RESET hnsw.guided_collect_target;
RESET hnsw.filter_strategy;
DROP TABLE sqlens_binding_smoke CASCADE;
