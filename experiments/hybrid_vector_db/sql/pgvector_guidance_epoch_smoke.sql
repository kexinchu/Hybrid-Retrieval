CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS guidance_epoch_smoke;
CREATE TABLE guidance_epoch_smoke (
  id bigint PRIMARY KEY,
  embedding vector(3),
  tenant_id int NOT NULL
);

INSERT INTO guidance_epoch_smoke
SELECT i, ARRAY[i::float, 0, 0]::vector, 0
FROM generate_series(1, 3000) AS i;

CREATE INDEX guidance_epoch_smoke_hnsw
ON guidance_epoch_smoke USING hnsw (embedding vector_l2_ops);
ANALYZE guidance_epoch_smoke;

SELECT vector_hnsw_fragment_tracking_enable('guidance_epoch_smoke'::regclass);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = strict_order;
SET hnsw.filter_strategy = guided_collect;
SET hnsw.guided_collect_target = 100;

SELECT vector_hnsw_guidance_activate(
  'guidance_epoch_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

INSERT INTO guidance_epoch_smoke VALUES (0, '[0,0,0]', 1);

DO $$
DECLARE
  actual_id bigint;
  profile json;
BEGIN
  SELECT id INTO actual_id
  FROM guidance_epoch_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'guidance_epoch_smoke_hnsw'::regclass,
             ARRAY['exact:sql:tenant_id = 1'],
             'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 1;

  IF actual_id IS DISTINCT FROM 0 THEN
    RAISE EXCEPTION 'stale guidance caused a false negative: expected id 0, got %', actual_id;
  END IF;

  SELECT vector_hnsw_guidance_profile()::json INTO profile;
  IF (profile ->> 'active')::boolean THEN
    RAISE EXCEPTION 'epoch mismatch did not deactivate stale guidance';
  END IF;
END $$;

SELECT vector_hnsw_guidance_activate(
  'guidance_epoch_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

DO $$
DECLARE
  actual_id bigint;
  profile json;
BEGIN
  SELECT id INTO actual_id
  FROM guidance_epoch_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'guidance_epoch_smoke_hnsw'::regclass,
             ARRAY['exact:sql:tenant_id = 1'],
             'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 1;

  IF actual_id IS DISTINCT FROM 0 THEN
    RAISE EXCEPTION 'reactivated guidance returned wrong row: expected id 0, got %', actual_id;
  END IF;

  SELECT vector_hnsw_guidance_profile()::json INTO profile;
  IF NOT (profile ->> 'active')::boolean OR NOT (profile ->> 'epoch_tracked')::boolean THEN
    RAISE EXCEPTION 'reactivated guidance is not epoch tracked: %', profile;
  END IF;
END $$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.filter_strategy;
RESET hnsw.guided_collect_target;

DROP TABLE guidance_epoch_smoke;
