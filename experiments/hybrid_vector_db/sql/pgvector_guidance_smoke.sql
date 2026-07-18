CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS guidance_smoke;
CREATE TABLE guidance_smoke (
  id bigserial PRIMARY KEY,
  embedding vector(3),
  color text,
  price int,
  tenant_id int
);

INSERT INTO guidance_smoke (embedding, color, price, tenant_id)
SELECT ARRAY[(i % 7)::float, (i % 11)::float, (i % 13)::float]::vector,
       CASE WHEN i % 2 = 0 THEN 'red' ELSE 'blue' END,
       i % 50,
       i % 3
FROM generate_series(1, 200) i;

CREATE INDEX guidance_smoke_embedding_idx ON guidance_smoke USING hnsw (embedding vector_l2_ops);
SELECT vector_hnsw_fragment_tracking_enable('guidance_smoke'::regclass);

SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
  'guidance_smoke_embedding_idx'::regclass,
  ARRAY['page:sql:color = ''red''', 'bloom:sql:price <= 20', '|', 'exact:sql:tenant_id = 2'],
  'exact'
) AS activated_atoms;

SELECT vector_hnsw_guidance_profile();

DO $$
BEGIN
  BEGIN
    PERFORM vector_hnsw_guidance_activate(
      'guidance_smoke_embedding_idx'::regclass,
      ARRAY['!page:sql:color = ''red'''],
      'exact'
    );
    RAISE EXCEPTION 'expected !page guidance to fail';
  EXCEPTION WHEN invalid_parameter_value THEN
    RAISE NOTICE 'negated lossy guidance rejected as expected';
  END;
END $$;

SELECT vector_hnsw_guidance_reset();
DROP TABLE guidance_smoke;

DROP TABLE IF EXISTS guidance_scan_smoke;
CREATE TABLE guidance_scan_smoke (
  id bigserial PRIMARY KEY,
  embedding vector(3),
  tenant_id int
);

INSERT INTO guidance_scan_smoke (embedding, tenant_id)
SELECT ARRAY[(i % 7)::float, (i % 11)::float, (i % 13)::float]::vector, i % 3
FROM generate_series(1, 3000) i;

CREATE INDEX guidance_scan_smoke_embedding_idx ON guidance_scan_smoke USING hnsw (embedding vector_l2_ops);
ANALYZE guidance_scan_smoke;
SELECT vector_hnsw_fragment_tracking_enable('guidance_scan_smoke'::regclass);

SET enable_seqscan = off;
SET hnsw.iterative_scan = strict_order;
SET hnsw.ef_search = 20;
SET hnsw.filter_strategy = acorn1;

SELECT vector_hnsw_guidance_activate(
  'guidance_scan_smoke_embedding_idx'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

EXPLAIN (COSTS OFF)
SELECT id, tenant_id
FROM guidance_scan_smoke
WHERE (SELECT vector_hnsw_guidance_bind(
         'guidance_scan_smoke_embedding_idx'::regclass,
         ARRAY['exact:sql:tenant_id = 1'],
         'exact'
       ) OFFSET 0)
ORDER BY embedding <-> '[0,0,0]'
LIMIT 10;

WITH guided AS (
  SELECT id, tenant_id
  FROM guidance_scan_smoke
  WHERE (SELECT vector_hnsw_guidance_bind(
           'guidance_scan_smoke_embedding_idx'::regclass,
           ARRAY['exact:sql:tenant_id = 1'],
           'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[0,0,0]'
  LIMIT 10
)
SELECT count(*) AS rows, bool_and(tenant_id = 1) AS all_tenant_1
FROM guided;

SELECT vector_hnsw_last_scan_profile();
DO $$
DECLARE
  profile jsonb := vector_hnsw_last_scan_profile()::jsonb;
BEGIN
  IF (profile ->> 'profile_semantics_version')::integer <> 7
     OR (profile ->> 'heap_fetch_ms_is_residual_proxy')::boolean IS NOT TRUE
     OR profile ->> 'graph_elements_visited' IS DISTINCT FROM profile ->> 'visited_tuples'
     OR profile ->> 'raw_index_tids_returned' IS DISTINCT FROM profile ->> 'returned_tuples'
     OR profile ->> 'hnsw_am_callback_ms' IS DISTINCT FROM profile ->> 'hnsw_search_ms'
     OR profile ->> 'executor_residual_ms' IS DISTINCT FROM profile ->> 'heap_fetch_ms'
     OR NOT profile ? 'final_path'
     OR NOT profile ? 'filter_strategy'
     OR NOT profile ? 'iterative_scan'
     OR NOT profile ? 'net_distance_saved_available'
     OR profile ->> 'traversal_guidance_scope' <>
          'candidate_admission_and_validation'
     OR (profile ->> 'graph_expansion_pruned')::boolean
     OR (profile ->> 'distance_computations_pruned')::boolean THEN
    RAISE EXCEPTION 'scan profile compatibility aliases are inconsistent: %', profile;
  END IF;
END
$$;
SELECT vector_hnsw_guidance_reset();

RESET enable_seqscan;
RESET hnsw.iterative_scan;
RESET hnsw.ef_search;
RESET hnsw.filter_strategy;

DROP TABLE guidance_scan_smoke;
