CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS hot_update_guidance_smoke;
CREATE TABLE hot_update_guidance_smoke (
  id bigint PRIMARY KEY,
  embedding vector(3) NOT NULL,
  tenant_id int NOT NULL,
  generation int NOT NULL DEFAULT 0
) WITH (fillfactor = 50);

INSERT INTO hot_update_guidance_smoke
SELECT i,
       ARRAY[(i - 1)::float / 1000, 0, 0]::vector,
       CASE WHEN i = 2 THEN 0 ELSE i % 2 END,
       0
FROM generate_series(1, 64) AS i;

CREATE INDEX hot_update_guidance_smoke_hnsw
ON hot_update_guidance_smoke USING hnsw (embedding vector_l2_ops);
ANALYZE hot_update_guidance_smoke;
SELECT vector_hnsw_fragment_tracking_enable('hot_update_guidance_smoke'::regclass);

SET enable_seqscan = off;
SET enable_sort = off;
SET enable_bitmapscan = off;
SET hnsw.ef_search = 256;
SET hnsw.max_scan_tuples = 10000;
SET hnsw.iterative_scan = strict_order;
SET hnsw.filter_strategy = safe_guided;

-- Materialize fragments before a non-indexed predicate column crosses into
-- the matching set.  PostgreSQL should perform this update through HOT.
SELECT vector_hnsw_guidance_activate(
  'hot_update_guidance_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
  'hot_update_guidance_smoke_hnsw'::regclass,
  ARRAY['bloom:sql:tenant_id = 1'],
  'bloom'
);
SELECT vector_hnsw_guidance_reset();

UPDATE hot_update_guidance_smoke
SET tenant_id = 1, generation = generation + 1
WHERE id = 2;
SELECT pg_stat_force_next_flush();

DO $$
BEGIN
  IF pg_stat_get_tuples_hot_updated('hot_update_guidance_smoke'::regclass) < 1 THEN
    RAISE EXCEPTION 'predicate-crossing update did not exercise a HOT chain';
  END IF;
END $$;

CREATE TEMP TABLE hot_update_guidance_results (
  kind text PRIMARY KEY,
  ids bigint[] NOT NULL
);

SELECT vector_hnsw_guidance_activate(
  'hot_update_guidance_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);
INSERT INTO hot_update_guidance_results
SELECT 'exact', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM hot_update_guidance_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'hot_update_guidance_smoke_hnsw'::regclass,
             ARRAY['exact:sql:tenant_id = 1'],
             'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 5
) AS q;
SELECT vector_hnsw_guidance_reset();

SELECT vector_hnsw_guidance_activate(
  'hot_update_guidance_smoke_hnsw'::regclass,
  ARRAY['bloom:sql:tenant_id = 1'],
  'bloom'
);
INSERT INTO hot_update_guidance_results
SELECT 'bloom', array_agg(id ORDER BY distance)
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM hot_update_guidance_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'hot_update_guidance_smoke_hnsw'::regclass,
             ARRAY['bloom:sql:tenant_id = 1'],
             'bloom'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 5
) AS q;

DO $$
DECLARE
  exact_ids bigint[];
  bloom_ids bigint[];
  truth_ids bigint[];
BEGIN
  SELECT ids INTO exact_ids FROM hot_update_guidance_results WHERE kind = 'exact';
  SELECT ids INTO bloom_ids FROM hot_update_guidance_results WHERE kind = 'bloom';

  SET LOCAL enable_seqscan = on;
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_indexonlyscan = off;
  SET LOCAL enable_sort = on;
  SELECT array_agg(id ORDER BY distance) INTO truth_ids
  FROM (
    SELECT id, embedding <-> '[0,0,0]' AS distance
    FROM hot_update_guidance_smoke
    WHERE tenant_id = 1
    ORDER BY embedding <-> '[0,0,0]', id
    LIMIT 5
  ) AS q;

  IF exact_ids IS DISTINCT FROM truth_ids OR bloom_ids IS DISTINCT FROM truth_ids THEN
    RAISE EXCEPTION 'HOT-root guidance mismatch: exact %, bloom %, truth %',
      exact_ids, bloom_ids, truth_ids;
  END IF;
  IF array_position(exact_ids, 2) IS NULL OR array_position(bloom_ids, 2) IS NULL THEN
    RAISE EXCEPTION 'HOT-updated matching row was suppressed: exact %, bloom %',
      exact_ids, bloom_ids;
  END IF;
END $$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET enable_bitmapscan;
RESET hnsw.ef_search;
RESET hnsw.max_scan_tuples;
RESET hnsw.iterative_scan;
RESET hnsw.filter_strategy;

DROP TABLE hot_update_guidance_smoke;
