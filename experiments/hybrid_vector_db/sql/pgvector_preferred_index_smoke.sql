\set ON_ERROR_STOP on

DROP SCHEMA IF EXISTS sqlens_preferred_index_smoke CASCADE;
CREATE SCHEMA sqlens_preferred_index_smoke;
SET search_path = sqlens_preferred_index_smoke, public;
SET max_parallel_maintenance_workers = 0;
SET maintenance_work_mem = '256MB';
SET hnsw.require_full_memory_build = on;

CREATE TABLE items (
  id integer PRIMARY KEY,
  embedding vector(3) NOT NULL
);
INSERT INTO items
SELECT i, ARRAY[(i % 17)::real, (i % 29)::real, (i % 43)::real]::vector(3)
FROM generate_series(1, 600) AS i;

SET hnsw.build_seed = 31;
CREATE INDEX source_hnsw ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
BEGIN;
SET LOCAL hnsw.clone_source = 'sqlens_preferred_index_smoke.source_hnsw';
SET LOCAL hnsw.build_page_order = bfs;
CREATE INDEX clone_hnsw ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
COMMIT;

CREATE INDEX btree_idx ON items (id);
CREATE INDEX never_hnsw ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32)
WHERE id < 0;
CREATE TABLE other_items (LIKE items INCLUDING ALL);
INSERT INTO other_items SELECT * FROM items;
CREATE INDEX other_hnsw ON other_items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

DO $$
DECLARE
  comparison jsonb;
BEGIN
  comparison := vector_hnsw_graph_compare('source_hnsw'::regclass, 'clone_hnsw'::regclass);
  IF NOT (comparison->>'same_heap')::boolean OR
     NOT (comparison->>'logical_equal')::boolean OR
     NOT (comparison->>'entry_equal')::boolean OR
     NOT (comparison->>'tuple_coverage_equal')::boolean OR
     (comparison->>'physical_equal')::boolean THEN
    RAISE EXCEPTION 'same-graph precondition failed: %', comparison;
  END IF;
END
$$;

DO $$
BEGIN
  PERFORM set_config(
    'hnsw.preferred_index',
    'sqlens_preferred_index_smoke.btree_idx',
    false);
  RAISE EXCEPTION 'preferred_index unexpectedly accepted a btree index';
EXCEPTION
  WHEN wrong_object_type THEN
    IF SQLERRM NOT LIKE 'hnsw.preferred_index must name an HNSW index%' THEN
      RAISE;
    END IF;
END
$$;

SET hnsw.preferred_index = 'sqlens_preferred_index_smoke.other_hnsw';
DO $$
BEGIN
  EXECUTE 'EXPLAIN (COSTS OFF) '
          'SELECT id FROM items ORDER BY embedding <-> ''[1,2,3]''::vector LIMIT 10';
  RAISE EXCEPTION 'preferred_index unexpectedly accepted an index on another heap';
EXCEPTION
  WHEN invalid_parameter_value THEN
    IF SQLERRM NOT LIKE 'hnsw.preferred_index is on a different heap%' THEN
      RAISE;
    END IF;
END
$$;

RESET enable_seqscan;
RESET enable_sort;
SET hnsw.preferred_index = 'sqlens_preferred_index_smoke.never_hnsw';
DO $$
DECLARE
  plan jsonb;
BEGIN
  EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
          'SELECT id FROM items ORDER BY embedding <-> ''[1,2,3]''::vector LIMIT 10'
    INTO plan;
  IF plan::text NOT LIKE '%Seq Scan%' OR plan::text LIKE '%source_hnsw%' OR
     plan::text LIKE '%clone_hnsw%' THEN
    RAISE EXCEPTION 'unavailable preferred partial index did not fall back to seq scan: %', plan;
  END IF;
END
$$;

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.preferred_index = 'sqlens_preferred_index_smoke.source_hnsw';
DO $$
DECLARE
  plan jsonb;
BEGIN
  EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
          'SELECT id FROM items ORDER BY embedding <-> ''[1,2,3]''::vector LIMIT 10'
    INTO plan;
  IF plan #>> '{0,Plan,Plans,0,Index Name}' <> 'source_hnsw' THEN
    RAISE EXCEPTION 'source preference was not honored: %', plan;
  END IF;
END
$$;

SET hnsw.preferred_index = 'sqlens_preferred_index_smoke.clone_hnsw';
DO $$
DECLARE
  plan jsonb;
BEGIN
  EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
          'SELECT id FROM items ORDER BY embedding <-> ''[1,2,3]''::vector LIMIT 10'
    INTO plan;
  IF plan #>> '{0,Plan,Plans,0,Index Name}' <> 'clone_hnsw' THEN
    RAISE EXCEPTION 'clone preference was not honored: %', plan;
  END IF;
END
$$;

SET plan_cache_mode = force_generic_plan;
SET hnsw.preferred_index = 'sqlens_preferred_index_smoke.source_hnsw';
PREPARE preferred_query(vector) AS
SELECT id FROM items ORDER BY embedding <-> $1 LIMIT 10;
DO $$
DECLARE
  plan jsonb;
BEGIN
  EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
          'EXECUTE preferred_query(''[1,2,3]''::vector)'
    INTO plan;
  IF plan #>> '{0,Plan,Plans,0,Index Name}' <> 'source_hnsw' THEN
    RAISE EXCEPTION 'generic source preference was not honored: %', plan;
  END IF;
END
$$;

SET hnsw.preferred_index = 'sqlens_preferred_index_smoke.clone_hnsw';
DO $$
DECLARE
  plan jsonb;
BEGIN
  EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
          'EXECUTE preferred_query(''[1,2,3]''::vector)'
    INTO plan;
  IF plan #>> '{0,Plan,Plans,0,Index Name}' <> 'clone_hnsw' THEN
    RAISE EXCEPTION 'generic plan cache was not invalidated for clone preference: %', plan;
  END IF;
END
$$;
DEALLOCATE preferred_query;

RESET hnsw.preferred_index;
RESET hnsw.clone_source;
RESET hnsw.build_seed;
RESET hnsw.build_page_order;
RESET hnsw.require_full_memory_build;
RESET enable_seqscan;
RESET enable_sort;
RESET plan_cache_mode;
DROP SCHEMA sqlens_preferred_index_smoke CASCADE;
