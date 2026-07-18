CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS safe_guidance_smoke;
CREATE TABLE safe_guidance_smoke (
  id bigint PRIMARY KEY,
  embedding vector(3),
  tenant_id int NOT NULL,
  has_price boolean NOT NULL,
  price numeric NOT NULL,
  rating int NOT NULL,
  tags int[] NOT NULL
);

INSERT INTO safe_guidance_smoke
SELECT i,
       ARRAY[i::float, (i % 17)::float, (i % 31)::float]::vector,
       CASE WHEN i = 2999 THEN 999 ELSE i % 3 END,
       i % 2 = 0,
       (i % 20)::numeric,
       CASE WHEN i % 5 = 0 THEN 5 ELSE 4 END,
       CASE i % 3
         WHEN 0 THEN ARRAY[23]
         WHEN 1 THEN ARRAY[29]
         ELSE ARRAY[31]
       END
FROM generate_series(1, 3000) AS i;

CREATE INDEX safe_guidance_smoke_hnsw
ON safe_guidance_smoke USING hnsw (embedding vector_l2_ops);
ANALYZE safe_guidance_smoke;
SELECT vector_hnsw_fragment_tracking_enable('safe_guidance_smoke'::regclass);

CREATE TEMP TABLE safe_guidance_observations (
  method text PRIMARY KEY,
  ids bigint[],
  profile jsonb
);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 40;
SET hnsw.iterative_scan = strict_order;
SET hnsw.max_scan_tuples = 100000;
SET hnsw.scan_mem_multiplier = 8;
SET hnsw.page_access = off;
SET hnsw.index_page_access = off;

SELECT vector_hnsw_guidance_reset();
SET hnsw.filter_strategy = off;
INSERT INTO safe_guidance_observations (method, ids)
SELECT 'stock', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM safe_guidance_smoke
  WHERE tenant_id = 1
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 10
) AS q;
UPDATE safe_guidance_observations
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stock';

SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_guidance_activate(
  'safe_guidance_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);
INSERT INTO safe_guidance_observations (method, ids)
SELECT 'safe_guided', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM safe_guidance_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'safe_guidance_smoke_hnsw'::regclass,
             ARRAY['exact:sql:tenant_id = 1'],
             'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 10
) AS q;
UPDATE safe_guidance_observations
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'safe_guided';

DO $$
DECLARE guide_profile jsonb;
DECLARE scan_profile jsonb;
BEGIN
  SELECT vector_hnsw_guidance_profile()::jsonb INTO guide_profile;
  SELECT vector_hnsw_last_scan_profile()::jsonb INTO scan_profile;
  IF (guide_profile ->> 'planner_proof_successes')::bigint < 1 OR
     guide_profile ->> 'planner_proof_bypass_reason' <> 'none' OR
     NOT (scan_profile ->> 'planner_proof_attempted')::boolean OR
     NOT (scan_profile ->> 'planner_proof_succeeded')::boolean OR
     scan_profile ->> 'planner_proof_bypass_reason' <> 'none' THEN
    RAISE EXCEPTION 'safe guidance did not pass planner proof: %', scan_profile;
  END IF;
END $$;

SELECT vector_hnsw_guidance_reset();
SET hnsw.filter_strategy = off;
INSERT INTO safe_guidance_observations (method, ids)
SELECT 'stock_sparse', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM safe_guidance_smoke
  WHERE tenant_id = 999
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 1
) AS q;
UPDATE safe_guidance_observations
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'stock_sparse';

SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_guidance_activate(
  'safe_guidance_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 999'],
  'exact'
);
INSERT INTO safe_guidance_observations (method, ids)
SELECT 'safe_sparse', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM safe_guidance_smoke
  WHERE tenant_id = 999
    AND (SELECT vector_hnsw_guidance_bind(
             'safe_guidance_smoke_hnsw'::regclass,
             ARRAY['exact:sql:tenant_id = 999'],
             'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 1
) AS q;
UPDATE safe_guidance_observations
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE method = 'safe_sparse';

DO $$
DECLARE
  stock_ids bigint[];
  guided_ids bigint[];
  stock_profile jsonb;
  guided_profile jsonb;
BEGIN
  SELECT ids, profile INTO stock_ids, stock_profile
  FROM safe_guidance_observations WHERE method = 'stock';
  SELECT ids, profile INTO guided_ids, guided_profile
  FROM safe_guidance_observations WHERE method = 'safe_guided';

  IF stock_ids IS DISTINCT FROM guided_ids THEN
    RAISE EXCEPTION 'safe guidance changed the stock SQL-valid result: stock %, guided %',
      stock_ids, guided_ids;
  END IF;
  IF stock_profile ->> 'traversal_expanded_nodes'
       IS DISTINCT FROM guided_profile ->> 'traversal_expanded_nodes' THEN
    RAISE EXCEPTION 'safe guidance changed expanded nodes: stock %, guided %',
      stock_profile ->> 'traversal_expanded_nodes',
      guided_profile ->> 'traversal_expanded_nodes';
  END IF;
  IF stock_profile ->> 'distance_compute_count'
       IS DISTINCT FROM guided_profile ->> 'distance_compute_count' THEN
    RAISE EXCEPTION 'safe guidance changed distance computations: stock %, guided %',
      stock_profile ->> 'distance_compute_count',
      guided_profile ->> 'distance_compute_count';
  END IF;
  IF stock_profile ->> 'traversal_candidate_admissions'
       IS DISTINCT FROM guided_profile ->> 'traversal_candidate_admissions'
     OR stock_profile ->> 'traversal_result_admissions'
       IS DISTINCT FROM guided_profile ->> 'traversal_result_admissions' THEN
    RAISE EXCEPTION 'safe guidance changed graph candidate admission: stock %, guided %',
      stock_profile, guided_profile;
  END IF;
  IF (guided_profile ->> 'traversal_guidance_checks')::bigint <= 0
     OR (guided_profile ->> 'traversal_guided_suppressions')::bigint <= 0
     OR (guided_profile ->> 'traversal_heap_tids_suppressed')::bigint <= 0 THEN
    RAISE EXCEPTION 'safe guidance counters were not populated: %', guided_profile;
  END IF;
  IF (guided_profile ->> 'neighbor_expansion_guidance_checks')::bigint <> 0 OR
     (guided_profile ->> 'heap_validation_guidance_checks')::bigint <= 0 OR
     guided_profile ->> 'final_path' <> 'validation_only' THEN
    RAISE EXCEPTION 'safe guidance was not confined to heap-TID validation: %',
      guided_profile;
  END IF;
  IF (guided_profile ->> 'returned_tuples')::bigint
       > (stock_profile ->> 'returned_tuples')::bigint THEN
    RAISE EXCEPTION 'safe guidance returned more executor candidates: stock %, guided %',
      stock_profile ->> 'returned_tuples', guided_profile ->> 'returned_tuples';
  END IF;

  SELECT ids, profile INTO stock_ids, stock_profile
  FROM safe_guidance_observations WHERE method = 'stock_sparse';
  SELECT ids, profile INTO guided_ids, guided_profile
  FROM safe_guidance_observations WHERE method = 'safe_sparse';
  IF stock_ids IS DISTINCT FROM ARRAY[2999]::bigint[]
     OR guided_ids IS DISTINCT FROM stock_ids THEN
    RAISE EXCEPTION 'safe guidance stopped while consuming an invalid candidate stream: stock %, guided %',
      stock_ids, guided_ids;
  END IF;
  IF stock_profile ->> 'traversal_expanded_nodes'
       IS DISTINCT FROM guided_profile ->> 'traversal_expanded_nodes'
     OR stock_profile ->> 'distance_compute_count'
       IS DISTINCT FROM guided_profile ->> 'distance_compute_count' THEN
    RAISE EXCEPTION 'safe sparse guidance changed stock traversal: stock %, guided %',
      stock_profile, guided_profile;
  END IF;
  IF (guided_profile ->> 'traversal_resume_batches')::bigint <= 1 THEN
    RAISE EXCEPTION 'sparse smoke did not exercise resumed empty projections: %', guided_profile;
  END IF;
END $$;

-- Formal workload DNF semantics: adjacent atoms are ANDed. The first atom
-- intentionally contains its own AND, matching the Amazon runner shape.
SELECT vector_hnsw_guidance_reset();
SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_guidance_activate(
  'safe_guidance_smoke_hnsw'::regclass,
  ARRAY['sql:has_price AND price <= 10', 'sql:rating = 5'],
  'exact'
);
DO $$
DECLARE
  rows_found bigint;
  successes_before bigint;
  guide_profile jsonb;
  scan_profile jsonb;
BEGIN
  SELECT (vector_hnsw_guidance_profile()::jsonb ->> 'planner_proof_successes')::bigint
  INTO successes_before;
  SELECT count(*) INTO rows_found
  FROM (
    SELECT id
    FROM safe_guidance_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'safe_guidance_smoke_hnsw'::regclass,
               ARRAY['sql:has_price AND price <= 10', 'sql:rating = 5'],
               'exact'
           ) OFFSET 0)
      AND has_price AND price <= 10 AND rating = 5
      AND id <> 20
    ORDER BY embedding <-> '[0,0,0]'
    LIMIT 10
  ) AS q;
  SELECT vector_hnsw_guidance_profile()::jsonb INTO guide_profile;
  SELECT vector_hnsw_last_scan_profile()::jsonb INTO scan_profile;
  IF rows_found = 0 OR
     (guide_profile ->> 'groups')::int <> 1 OR
     (guide_profile ->> 'atoms')::int <> 2 OR
     (guide_profile ->> 'planner_proof_successes')::bigint <> successes_before + 1 OR
     guide_profile ->> 'planner_proof_bypass_reason' <> 'none' OR
     NOT (scan_profile ->> 'planner_proof_succeeded')::boolean THEN
    RAISE EXCEPTION 'AND-group planner proof failed: rows %, profile %',
      rows_found, guide_profile;
  END IF;
END $$;

-- A standalone | separates OR groups, matching YFCC/LAION runner atoms.
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
  'safe_guidance_smoke_hnsw'::regclass,
  ARRAY['sql:tags @> ARRAY[23]', '|', 'sql:tags @> ARRAY[29]'],
  'exact'
);
DO $$
DECLARE
  rows_found bigint;
  successes_before bigint;
  guide_profile jsonb;
  scan_profile jsonb;
BEGIN
  SELECT (vector_hnsw_guidance_profile()::jsonb ->> 'planner_proof_successes')::bigint
  INTO successes_before;
  SELECT count(*) INTO rows_found
  FROM (
    SELECT id
    FROM safe_guidance_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'safe_guidance_smoke_hnsw'::regclass,
               ARRAY['sql:tags @> ARRAY[23]', '|', 'sql:tags @> ARRAY[29]'],
               'exact'
           ) OFFSET 0)
      AND (tags @> ARRAY[23] OR tags @> ARRAY[29])
      AND id <> 29
    ORDER BY embedding <-> '[0,0,0]'
    LIMIT 10
  ) AS q;
  SELECT vector_hnsw_guidance_profile()::jsonb INTO guide_profile;
  SELECT vector_hnsw_last_scan_profile()::jsonb INTO scan_profile;
  IF rows_found <> 10 OR
     (guide_profile ->> 'groups')::int <> 2 OR
     (guide_profile ->> 'atoms')::int <> 2 OR
     (guide_profile ->> 'planner_proof_successes')::bigint <> successes_before + 1 OR
     guide_profile ->> 'planner_proof_bypass_reason' <> 'none' OR
     NOT (scan_profile ->> 'planner_proof_succeeded')::boolean THEN
    RAISE EXCEPTION 'OR-group planner proof failed: rows %, profile %',
      rows_found, guide_profile;
  END IF;
END $$;

-- An active multi-atom guide may exceed the residency budget. Every referenced
-- payload must remain valid until reset; eviction is allowed to target only
-- inactive fragments.
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_metadata_cache_reset();
SET hnsw.metadata_cache_max_mb = 1;
SET hnsw.filter_strategy = safe_guided;
SELECT vector_hnsw_guidance_activate(
  'safe_guidance_smoke_hnsw'::regclass,
  ARRAY(
    SELECT 'exact:sql:tenant_id >= 0 /* active_' || i || ' */'
    FROM generate_series(1, 24) AS i
  ),
  'exact'
);

DO $$
DECLARE
  guided_ids bigint[];
  guide_profile jsonb;
BEGIN
  SELECT array_agg(id ORDER BY distance) INTO guided_ids
  FROM (
    SELECT id, embedding <-> '[0,0,0]' AS distance
    FROM safe_guidance_smoke
    WHERE tenant_id = 1
      AND (SELECT vector_hnsw_guidance_bind(
               'safe_guidance_smoke_hnsw'::regclass,
               ARRAY(
                 SELECT 'exact:sql:tenant_id >= 0 /* active_' || i || ' */'
                 FROM generate_series(1, 24) AS i
               ),
               'exact'
           ) OFFSET 0)
    ORDER BY embedding <-> '[0,0,0]'
    LIMIT 10
  ) AS q;
  guide_profile := vector_hnsw_guidance_profile()::jsonb;
  IF guided_ids IS DISTINCT FROM (
       SELECT ids FROM safe_guidance_observations WHERE method = 'stock'
     ) THEN
    RAISE EXCEPTION 'active fragment eviction corrupted multi-atom results: %', guided_ids;
  END IF;
  IF (guide_profile ->> 'atoms')::int != 24 THEN
    RAISE EXCEPTION 'multi-atom guidance was not fully activated: %', guide_profile;
  END IF;
END $$;

TABLE safe_guidance_observations;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.max_scan_tuples;
RESET hnsw.scan_mem_multiplier;
RESET hnsw.page_access;
RESET hnsw.index_page_access;
RESET hnsw.filter_strategy;
RESET hnsw.metadata_cache_max_mb;

DROP TABLE safe_guidance_smoke;
