CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS guidance_relation_a;
DROP TABLE IF EXISTS guidance_relation_b;

CREATE TABLE guidance_relation_a (
  id bigint PRIMARY KEY,
  embedding vector(3),
  tenant_id int NOT NULL
);

CREATE TABLE guidance_relation_b (
  id bigint PRIMARY KEY,
  embedding vector(3),
  tenant_id int NOT NULL
);

INSERT INTO guidance_relation_a
SELECT i, ARRAY[i::float, (i % 7)::float, (i % 11)::float]::vector, i % 2
FROM generate_series(1, 3000) AS i;

INSERT INTO guidance_relation_b
SELECT i, ARRAY[i::float, (i % 7)::float, (i % 11)::float]::vector, (i + 1) % 2
FROM generate_series(1, 3000) AS i;

CREATE INDEX guidance_relation_a_hnsw
ON guidance_relation_a USING hnsw (embedding vector_l2_ops);
CREATE INDEX guidance_relation_b_hnsw
ON guidance_relation_b USING hnsw (embedding vector_l2_ops);
ANALYZE guidance_relation_a;
ANALYZE guidance_relation_b;
SELECT vector_hnsw_fragment_tracking_enable('guidance_relation_a'::regclass);
SELECT vector_hnsw_fragment_tracking_enable('guidance_relation_b'::regclass);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = strict_order;
SET hnsw.filter_strategy = guided_collect;
SET hnsw.guided_collect_target = 100;

CREATE TEMP TABLE guidance_relation_expected AS
SELECT array_agg(id ORDER BY distance) AS ids
FROM (
  SELECT id, embedding <-> '[0,0,0]' AS distance
  FROM guidance_relation_b
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 10
) AS nearest;

SELECT vector_hnsw_guidance_activate(
  'guidance_relation_a_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

DO $$
DECLARE
  expected_ids bigint[];
  actual_ids bigint[];
  profile json;
BEGIN
  SELECT ids INTO expected_ids FROM guidance_relation_expected;
  SELECT array_agg(id ORDER BY distance) INTO actual_ids
  FROM (
    SELECT id, embedding <-> '[0,0,0]' AS distance
    FROM guidance_relation_b
    ORDER BY embedding <-> '[0,0,0]'
    LIMIT 10
  ) AS nearest;

  IF actual_ids IS DISTINCT FROM expected_ids THEN
    RAISE EXCEPTION 'guidance leaked across relations: expected %, got %', expected_ids, actual_ids;
  END IF;

  SELECT vector_hnsw_last_scan_profile()::json INTO profile;
  IF (profile ->> 'guidance_checks')::bigint <> 0 THEN
    RAISE EXCEPTION 'relation-mismatched scan performed % guidance checks', profile ->> 'guidance_checks';
  END IF;
END $$;

DO $$
DECLARE
  invalid_rows bigint;
  profile json;
BEGIN
  SELECT count(*) INTO invalid_rows
  FROM (
    SELECT id, tenant_id
    FROM guidance_relation_a
    WHERE tenant_id = 1
    ORDER BY embedding <-> '[0,0,0]'
    LIMIT 10
  ) AS nearest
  WHERE tenant_id <> 1;

  IF invalid_rows <> 0 THEN
    RAISE EXCEPTION 'same-relation guidance returned invalid rows';
  END IF;

  SELECT vector_hnsw_last_scan_profile()::json INTO profile;
  IF (profile ->> 'guidance_checks')::bigint = 0 THEN
    RAISE EXCEPTION 'same-relation scan did not apply guidance';
  END IF;
END $$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.filter_strategy;
RESET hnsw.guided_collect_target;

DROP TABLE guidance_relation_a;
DROP TABLE guidance_relation_b;
