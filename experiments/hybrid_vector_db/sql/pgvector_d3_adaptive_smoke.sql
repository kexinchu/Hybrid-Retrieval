CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS pgvector_d3_adaptive_smoke;
CREATE TABLE pgvector_d3_adaptive_smoke (
  id bigint PRIMARY KEY,
  embedding vector(3) NOT NULL,
  tenant_id int NOT NULL,
  labels int[] NOT NULL
);

-- Every heap page contains both tenants, so page guidance has a deliberately
-- low skip rate and should refine to Bloom on the following activation.
INSERT INTO pgvector_d3_adaptive_smoke
SELECT i,
       ARRAY[i::float, (i % 97)::float, (i % 31)::float]::vector,
       i % 2,
       ARRAY[(i % 5), ((i + 1) % 5)]::int[]
FROM generate_series(1, 24000) AS i;

CREATE INDEX pgvector_d3_adaptive_smoke_hnsw
ON pgvector_d3_adaptive_smoke USING hnsw (embedding vector_l2_ops);
ANALYZE pgvector_d3_adaptive_smoke;
SELECT vector_hnsw_fragment_tracking_enable('pgvector_d3_adaptive_smoke'::regclass);

CREATE TEMP TABLE d3_adaptive_results (
  phase text PRIMARY KEY,
  ids bigint[]
);

SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_metadata_cache_reset();
SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 400;
SET hnsw.iterative_scan = strict_order;
SET hnsw.max_scan_tuples = 100000;
SET hnsw.scan_mem_multiplier = 8;
SET hnsw.filter_strategy = off;
SET hnsw.d3_probe_requests = 2;
SET hnsw.d3_min_benefit_per_byte = 0;
SET hnsw.d3_max_fragment_mb = 16;
SET hnsw.d3_page_min_skip_rate = 0.05;

INSERT INTO d3_adaptive_results
SELECT 'stock', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke
  WHERE tenant_id = 1
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 20
) AS q;

SET hnsw.filter_strategy = safe_guided;

DO $$
DECLARE p jsonb;
BEGIN
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:tenant_id = 1'], 'adaptive') <> 0 THEN
    RAISE EXCEPTION 'first adaptive request unexpectedly activated guidance';
  END IF;
  p := vector_hnsw_guidance_profile()::jsonb;
  IF (p ->> 'active')::boolean OR p ->> 'adaptive_state' <> 'probing' THEN
    RAISE EXCEPTION 'first request was not an inactive probe: %', p;
  END IF;
END $$;

INSERT INTO d3_adaptive_results
SELECT 'probe1', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke WHERE tenant_id = 1
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;

DO $$
DECLARE p jsonb;
BEGIN
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:tenant_id = 1'], 'adaptive') <> 0 THEN
    RAISE EXCEPTION 'second adaptive request unexpectedly activated guidance';
  END IF;
  p := vector_hnsw_guidance_profile()::jsonb;
  IF (p ->> 'active')::boolean OR (p ->> 'adaptive_probes')::bigint < 1 THEN
    RAISE EXCEPTION 'second request did not remain an inactive probe: %', p;
  END IF;
END $$;

INSERT INTO d3_adaptive_results
SELECT 'probe2', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke WHERE tenant_id = 1
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;

DO $$
DECLARE p jsonb;
BEGIN
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:tenant_id = 1'], 'adaptive') <= 0 THEN
    RAISE EXCEPTION 'third adaptive request did not admit a page fragment';
  END IF;
  p := vector_hnsw_guidance_profile()::jsonb;
  IF NOT (p ->> 'active')::boolean OR p ->> 'adaptive_state' <> 'page' OR
     (p ->> 'adaptive_page_builds')::bigint < 1 THEN
    RAISE EXCEPTION 'page admission was not recorded: %', p;
  END IF;
END $$;

INSERT INTO d3_adaptive_results
SELECT 'page', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'pgvector_d3_adaptive_smoke_hnsw'::regclass,
             ARRAY['sql:tenant_id = 1'],
             'adaptive'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;

DO $$
DECLARE p jsonb := vector_hnsw_guidance_profile()::jsonb;
BEGIN
  IF (p ->> 'adaptive_checks')::bigint <= 0 OR
     NOT (p ->> 'adaptive_refine_pending')::boolean THEN
    RAISE EXCEPTION 'page scan did not record a low-skip refinement candidate: %', p;
  END IF;
END $$;

DO $$
DECLARE p jsonb;
BEGIN
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:tenant_id = 1'], 'adaptive') <= 0 THEN
    RAISE EXCEPTION 'low-skip page did not refine to Bloom';
  END IF;
  p := vector_hnsw_guidance_profile()::jsonb;
  IF p ->> 'adaptive_state' <> 'bloom' OR
     (p ->> 'adaptive_bloom_builds')::bigint < 1 OR
     (p ->> 'adaptive_refinements')::bigint < 1 THEN
    RAISE EXCEPTION 'Bloom refinement was not recorded: %', p;
  END IF;
END $$;

INSERT INTO d3_adaptive_results
SELECT 'bloom', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'pgvector_d3_adaptive_smoke_hnsw'::regclass,
             ARRAY['sql:tenant_id = 1'],
             'adaptive'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;

DO $$
DECLARE p jsonb;
BEGIN
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:tenant_id = 1'], 'adaptive') <= 0 THEN
    RAISE EXCEPTION 'resident Bloom did not reactivate';
  END IF;
  p := vector_hnsw_guidance_profile()::jsonb;
  IF (p ->> 'fragment_cache_hits')::bigint < 1 THEN
    RAISE EXCEPTION 'repeated adaptive request missed the backend cache: %', p;
  END IF;
END $$;

INSERT INTO d3_adaptive_results
SELECT 'bloom_hit', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'pgvector_d3_adaptive_smoke_hnsw'::regclass,
             ARRAY['sql:tenant_id = 1'],
             'adaptive'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;

DO $$
BEGIN
  -- This must be accepted by the narrowed row-local cast whitelist.
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:labels @> ARRAY[1]::int[]'], 'adaptive') <> 0 THEN
    RAISE EXCEPTION 'array-cast request should begin as a probe';
  END IF;
END $$;
SELECT vector_hnsw_guidance_reset();

-- An epoch change while Bloom is active must fail open on the next executor
-- scan, then the descriptor starts a fresh inactive probe cycle.
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_smoke_hnsw'::regclass,
  ARRAY['sql:tenant_id = 1'], 'adaptive'
);
UPDATE pgvector_d3_adaptive_smoke SET tenant_id = 0 WHERE id = 1;

INSERT INTO d3_adaptive_results
SELECT 'stale_bypass', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'pgvector_d3_adaptive_smoke_hnsw'::regclass,
             ARRAY['sql:tenant_id = 1'],
             'adaptive'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;

DO $$
DECLARE p jsonb := vector_hnsw_guidance_profile()::jsonb;
BEGIN
  IF (p ->> 'active')::boolean OR (p ->> 'adaptive_stale_bypasses')::bigint < 1 THEN
    RAISE EXCEPTION 'epoch change did not fail open: %', p;
  END IF;
END $$;

SET hnsw.filter_strategy = off;
INSERT INTO d3_adaptive_results
SELECT 'stock_after_epoch', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_smoke WHERE tenant_id = 1
  ORDER BY embedding <-> '[0,0,0]' LIMIT 20
) AS q;
SET hnsw.filter_strategy = safe_guided;

DO $$
DECLARE p jsonb;
BEGIN
  IF vector_hnsw_guidance_activate(
       'pgvector_d3_adaptive_smoke_hnsw'::regclass,
       ARRAY['sql:tenant_id = 1'], 'adaptive') <> 0 THEN
    RAISE EXCEPTION 'stale descriptor did not restart with an inactive probe';
  END IF;
  p := vector_hnsw_guidance_profile()::jsonb;
  IF (p ->> 'active')::boolean OR p ->> 'adaptive_state' <> 'probing' THEN
    RAISE EXCEPTION 'stale descriptor did not transition to probing: %', p;
  END IF;
END $$;

DO $$
DECLARE stock_ids bigint[];
DECLARE actual_ids bigint[];
DECLARE mismatch_count int;
BEGIN
  SELECT ids INTO stock_ids FROM d3_adaptive_results WHERE phase = 'stock';
  SELECT count(*) INTO mismatch_count
  FROM d3_adaptive_results
  WHERE phase IN ('probe1', 'probe2', 'page', 'bloom', 'bloom_hit')
    AND ids IS DISTINCT FROM stock_ids;
  IF mismatch_count <> 0 THEN
    RAISE EXCEPTION 'adaptive result diverged from stock SQL: stock %, mismatches %', stock_ids, mismatch_count;
  END IF;
  SELECT ids INTO stock_ids FROM d3_adaptive_results WHERE phase = 'stock_after_epoch';
  SELECT ids INTO actual_ids FROM d3_adaptive_results WHERE phase = 'stale_bypass';
  IF actual_ids IS DISTINCT FROM stock_ids THEN
    RAISE EXCEPTION 'stale fail-open result diverged from stock SQL: stock %, actual %', stock_ids, actual_ids;
  END IF;
END $$;

-- A small cache budget must evict inactive low-score atoms, while the current
-- twenty-atom Bloom guide remains protected and SQL-correct.
DROP TABLE IF EXISTS pgvector_d3_adaptive_evict;
CREATE TABLE pgvector_d3_adaptive_evict (
  id bigint PRIMARY KEY,
  embedding vector(3) NOT NULL,
  tenant_id int NOT NULL
);
INSERT INTO pgvector_d3_adaptive_evict
SELECT i, ARRAY[i::float, (i % 101)::float, (i % 37)::float]::vector, i % 2
FROM generate_series(1, 85000) AS i;
CREATE INDEX pgvector_d3_adaptive_evict_hnsw
ON pgvector_d3_adaptive_evict USING hnsw (embedding vector_l2_ops);
ANALYZE pgvector_d3_adaptive_evict;
SELECT vector_hnsw_fragment_tracking_enable('pgvector_d3_adaptive_evict'::regclass);

SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_metadata_cache_reset();
SET hnsw.metadata_cache_max_mb = 1;
SET hnsw.d3_max_fragment_mb = 2;
SET hnsw.filter_strategy = off;

INSERT INTO d3_adaptive_results
SELECT 'evict_stock', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_evict WHERE tenant_id = 1
  ORDER BY embedding <-> '[0,0,0]' LIMIT 10
) AS q;
SET hnsw.filter_strategy = safe_guided;

SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_a_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;

SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);
SELECT id
FROM pgvector_d3_adaptive_evict
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'pgvector_d3_adaptive_evict_hnsw'::regclass,
           ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
           'adaptive'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]' LIMIT 10;
SELECT vector_hnsw_guidance_activate(
  'pgvector_d3_adaptive_evict_hnsw'::regclass,
  ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
  'adaptive'
);

INSERT INTO d3_adaptive_results
SELECT 'evict_active', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM pgvector_d3_adaptive_evict
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'pgvector_d3_adaptive_evict_hnsw'::regclass,
             ARRAY(SELECT 'sql:tenant_id = 1 /* evict_b_' || i || ' */' FROM generate_series(1, 20) AS i),
             'adaptive'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]' LIMIT 10
) AS q;

DO $$
DECLARE p jsonb := vector_hnsw_guidance_profile()::jsonb;
DECLARE stock_ids bigint[];
DECLARE active_ids bigint[];
BEGIN
  SELECT ids INTO stock_ids FROM d3_adaptive_results WHERE phase = 'evict_stock';
  SELECT ids INTO active_ids FROM d3_adaptive_results WHERE phase = 'evict_active';
  IF NOT (p ->> 'active')::boolean OR (p ->> 'atoms')::int <> 20 OR
     (p ->> 'adaptive_evictions')::bigint < 1 THEN
    RAISE EXCEPTION 'small-budget adaptive eviction did not preserve active atoms: %', p;
  END IF;
  IF active_ids IS DISTINCT FROM stock_ids THEN
    RAISE EXCEPTION 'active atoms were evicted or changed SQL results: stock %, active %', stock_ids, active_ids;
  END IF;
END $$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.max_scan_tuples;
RESET hnsw.scan_mem_multiplier;
RESET hnsw.filter_strategy;
RESET hnsw.d3_probe_requests;
RESET hnsw.d3_min_benefit_per_byte;
RESET hnsw.d3_max_fragment_mb;
RESET hnsw.d3_page_min_skip_rate;
RESET hnsw.metadata_cache_max_mb;

DROP TABLE pgvector_d3_adaptive_evict;
DROP TABLE pgvector_d3_adaptive_smoke;
