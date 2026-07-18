CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS mvcc_scope_smoke_guidance_meta;
DROP TABLE IF EXISTS mvcc_scope_smoke;
CREATE TABLE mvcc_scope_smoke (
  id bigint PRIMARY KEY,
  embedding vector(3) NOT NULL,
  tenant_id int NOT NULL
);
INSERT INTO mvcc_scope_smoke
SELECT i, ARRAY[i::float, (i % 17)::float, (i % 31)::float]::vector, i % 2
FROM generate_series(1, 1000) AS i;
CREATE INDEX mvcc_scope_smoke_hnsw
ON mvcc_scope_smoke USING hnsw (embedding vector_l2_ops);
ANALYZE mvcc_scope_smoke;
SELECT vector_hnsw_fragment_tracking_enable('mvcc_scope_smoke'::regclass);

-- A legacy sidecar with the conventional name must not silently become the
-- fragment source because its update epoch is not tied to the vector heap.
CREATE TABLE mvcc_scope_smoke_guidance_meta (
  id bigint,
  heap_tid tid,
  tenant_id int
);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = strict_order;
SET hnsw.filter_strategy = safe_guided;

SELECT vector_hnsw_guidance_activate(
  'mvcc_scope_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

DO $$
DECLARE
  profile jsonb := vector_hnsw_guidance_profile()::jsonb;
BEGIN
  IF (profile ->> 'last_cache_rows')::bigint != 500 THEN
    RAISE EXCEPTION 'guidance did not build from the tracked heap: %', profile;
  END IF;
END $$;

-- A committed write bumps the epoch. The next ordinary executor-driven scan
-- must fail open and deactivate the stale guide before returning candidates.
UPDATE mvcc_scope_smoke SET tenant_id = 0 WHERE id = 1;
SELECT id
FROM mvcc_scope_smoke
WHERE tenant_id = 1
  AND (SELECT vector_hnsw_guidance_bind(
           'mvcc_scope_smoke_hnsw'::regclass,
           ARRAY['exact:sql:tenant_id = 1'],
           'exact'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]'
LIMIT 10;

DO $$
BEGIN
  IF (vector_hnsw_guidance_profile()::jsonb ->> 'active')::boolean THEN
    RAISE EXCEPTION 'stale guidance remained active after an epoch change';
  END IF;
END $$;

DO $$
BEGIN
  BEGIN
    PERFORM vector_hnsw_guidance_activate(
      'mvcc_scope_smoke_hnsw'::regclass,
      ARRAY['exact:sql:tenant_id IN (SELECT tenant_id FROM mvcc_scope_smoke)'],
      'exact'
    );
    RAISE EXCEPTION 'cross-relation/subquery guidance was accepted';
  EXCEPTION WHEN feature_not_supported THEN
    NULL;
  END;

  BEGIN
    PERFORM vector_hnsw_guidance_activate(
      'mvcc_scope_smoke_hnsw'::regclass,
      ARRAY['exact:sql:' || repeat('tenant_id = 1 AND ', 20) || 'tenant_id = 1'],
      'exact'
    );
    RAISE EXCEPTION 'overlength guidance atom was accepted';
  EXCEPTION WHEN name_too_long THEN
    NULL;
  END;

  BEGIN
    PERFORM *
    FROM vector_hnsw_metadata_filter_search(
      'mvcc_scope_smoke_hnsw'::regclass,
      '[0,0,0]'::vector,
      10,
      100,
      'sql:tenant_id = 1'
    );
    RAISE EXCEPTION 'retired direct-ID cache search remained callable';
  EXCEPTION WHEN feature_not_supported THEN
    NULL;
  END;
END $$;

ALTER TABLE mvcc_scope_smoke ENABLE ROW LEVEL SECURITY;
DO $$
BEGIN
  BEGIN
    PERFORM vector_hnsw_guidance_activate(
      'mvcc_scope_smoke_hnsw'::regclass,
      ARRAY['exact:sql:tenant_id = 1'],
      'exact'
    );
    RAISE EXCEPTION 'hard guidance was accepted directly on an RLS heap';
  EXCEPTION WHEN feature_not_supported THEN
    NULL;
  END;
END $$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.filter_strategy;

DROP TABLE mvcc_scope_smoke_guidance_meta;
DROP TABLE mvcc_scope_smoke;
