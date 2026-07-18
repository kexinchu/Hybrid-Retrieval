\set ON_ERROR_STOP on

DROP SCHEMA IF EXISTS sqlens_clone_formality_smoke CASCADE;
CREATE SCHEMA sqlens_clone_formality_smoke;
SET search_path = sqlens_clone_formality_smoke, public;
SET max_parallel_maintenance_workers = 0;
SET maintenance_work_mem = '64MB';
SET hnsw.require_full_memory_build = on;

CREATE TABLE items (
  id integer PRIMARY KEY,
  embedding vector(3) NOT NULL
);
INSERT INTO items
SELECT i, ARRAY[(i % 97)::real, (i % 53)::real, (i % 31)::real]::vector(3)
FROM generate_series(1, 4000) AS i;

SET hnsw.build_seed = 71;
CREATE INDEX source_hnsw ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

SET maintenance_work_mem = '1MB';
DO $memory$
BEGIN
  PERFORM vector_hnsw_graph_fingerprint('source_hnsw'::regclass);
  RAISE EXCEPTION 'fingerprint unexpectedly ignored maintenance_work_mem';
EXCEPTION
  WHEN out_of_memory THEN
    IF SQLERRM NOT LIKE 'HNSW graph fingerprint exceeds maintenance_work_mem%' THEN
      RAISE;
    END IF;
END
$memory$;

DO $memory$
BEGIN
  PERFORM vector_hnsw_graph_compare('source_hnsw'::regclass, 'source_hnsw'::regclass);
  RAISE EXCEPTION 'compare unexpectedly ignored maintenance_work_mem';
EXCEPTION
  WHEN out_of_memory THEN
    IF SQLERRM NOT LIKE 'HNSW graph fingerprint exceeds maintenance_work_mem%' THEN
      RAISE;
    END IF;
END
$memory$;

BEGIN;
SET LOCAL hnsw.clone_source = 'sqlens_clone_formality_smoke.source_hnsw';
SET LOCAL hnsw.build_page_order = bfs;
DO $memory$
BEGIN
  EXECUTE 'CREATE INDEX clone_too_small ON items USING hnsw '
          '(embedding vector_l2_ops) WITH (m = 8, ef_construction = 32)';
  RAISE EXCEPTION 'clone unexpectedly ignored maintenance_work_mem';
EXCEPTION
  WHEN out_of_memory THEN
    IF SQLERRM NOT LIKE 'source HNSW graph does not fit into maintenance_work_mem%' THEN
      RAISE;
    END IF;
END
$memory$;
ROLLBACK;

-- Formal proof runners must choose a budget large enough for the source graph
-- plus canonicalization scratch.
SET maintenance_work_mem = '64MB';
BEGIN;
SET LOCAL hnsw.clone_source = 'sqlens_clone_formality_smoke.source_hnsw';
SET LOCAL hnsw.build_page_order = bfs;
CREATE INDEX clone_hnsw ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
COMMIT;

DO $proof$
DECLARE
  source_proof jsonb;
  clone_proof jsonb;
  comparison jsonb;
  locality jsonb;
  sample jsonb;
BEGIN
  source_proof := vector_hnsw_graph_fingerprint('source_hnsw'::regclass);
  clone_proof := vector_hnsw_graph_fingerprint('clone_hnsw'::regclass);
  comparison := vector_hnsw_graph_compare('source_hnsw'::regclass, 'clone_hnsw'::regclass);

  IF source_proof->>'definition_digest' IS NULL
     OR source_proof->>'tuple_coverage_digest' IS NULL
     OR source_proof->>'definition_digest' IS DISTINCT FROM clone_proof->>'definition_digest'
     OR source_proof->>'tuple_coverage_digest' IS DISTINCT FROM clone_proof->>'tuple_coverage_digest'
     OR NOT (comparison->>'definition_equal')::boolean
     OR NOT (comparison->>'tuple_coverage_equal')::boolean THEN
    RAISE EXCEPTION 'clone proof digests are incomplete or unequal: %, %, %',
      source_proof, clone_proof, comparison;
  END IF;

  FOREACH locality IN ARRAY ARRAY[
    source_proof->'bfs_locality', clone_proof->'bfs_locality',
    comparison->'left_bfs_locality', comparison->'right_bfs_locality'
  ] LOOP
    IF locality->>'format' IS DISTINCT FROM 'sqlens-hnsw-bfs-locality-v1'
       OR NOT (locality->>'full_statistics')::boolean
       OR (locality->>'rank_base')::integer <> 0
       OR (locality->>'sample_limit')::integer <> 256
       OR locality->>'sample_strategy' IS DISTINCT FROM 'evenly_spaced_inclusive'
       OR (locality->>'graph_nodes')::bigint <> (locality->>'sequence_nodes')::bigint
       OR (locality->>'adjacent_pairs')::bigint <> GREATEST((locality->>'sequence_nodes')::bigint - 1, 0)
       OR (locality->>'sample_count')::integer <> jsonb_array_length(locality->'rank_samples')
       OR ((locality->>'sample_truncated')::boolean IS DISTINCT FROM
           ((locality->>'sample_count')::bigint < (locality->>'sequence_nodes')::bigint)) THEN
      RAISE EXCEPTION 'invalid BFS locality proof contract: %', locality;
    END IF;
    IF (locality->>'sequence_nodes')::bigint > 0 THEN
      sample := locality->'rank_samples'->0;
      IF (sample->>'rank')::bigint <> 0 THEN
        RAISE EXCEPTION 'BFS rank sample does not start at rank zero: %', locality;
      END IF;
      sample := locality->'rank_samples'->-1;
      IF (sample->>'rank')::bigint <> (locality->>'sequence_nodes')::bigint - 1 THEN
        RAISE EXCEPTION 'BFS rank sample does not cover the final rank: %', locality;
      END IF;
    END IF;
  END LOOP;
  IF comparison->'left_bfs_locality' IS DISTINCT FROM source_proof->'bfs_locality'
     OR comparison->'right_bfs_locality' IS DISTINCT FROM clone_proof->'bfs_locality' THEN
    RAISE EXCEPTION 'source/clone BFS locality comparison is not symmetric: %', comparison;
  END IF;
END
$proof$;

CREATE INDEX coverage_even ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32) WHERE id % 2 = 0;
CREATE INDEX coverage_odd ON items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32) WHERE id % 2 = 1;

DO $coverage$
DECLARE
  comparison jsonb;
BEGIN
  comparison := vector_hnsw_graph_compare('coverage_even'::regclass, 'coverage_odd'::regclass);
  IF (comparison->>'tuple_coverage_equal')::boolean THEN
    RAISE EXCEPTION 'equal tuple counts were mistaken for equal TID coverage: %', comparison;
  END IF;
  IF (comparison->>'definition_equal')::boolean THEN
    RAISE EXCEPTION 'different partial predicates were mistaken for equal definitions: %', comparison;
  END IF;
END
$coverage$;

CREATE TABLE tombstone_items (
  id integer PRIMARY KEY,
  embedding vector(3) NOT NULL
);
INSERT INTO tombstone_items
SELECT i, ARRAY[i::real, (i % 11)::real, (i % 7)::real]::vector(3)
FROM generate_series(1, 100) AS i;
CREATE INDEX tombstone_source ON tombstone_items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
DELETE FROM tombstone_items WHERE id <= 5;
VACUUM tombstone_items;

DO $tombstone$
DECLARE
  proof jsonb;
BEGIN
  proof := vector_hnsw_graph_fingerprint('tombstone_source'::regclass);
  IF (proof->>'tombstones')::integer <> 5 THEN
    RAISE EXCEPTION 'fingerprint did not report tombstones explicitly: %', proof;
  END IF;
END
$tombstone$;

SET hnsw.clone_source = 'sqlens_clone_formality_smoke.tombstone_source';
SET hnsw.build_page_order = bfs;
DO $tombstone$
BEGIN
  EXECUTE 'CREATE INDEX tombstone_clone ON tombstone_items USING hnsw '
          '(embedding vector_l2_ops) WITH (m = 8, ef_construction = 32)';
  RAISE EXCEPTION 'clone unexpectedly accepted a tombstoned graph';
EXCEPTION
  WHEN feature_not_supported THEN
    IF SQLERRM NOT LIKE 'cannot clone HNSW index % with tombstoned elements' THEN
      RAISE;
    END IF;
END
$tombstone$;
RESET hnsw.clone_source;

CREATE UNLOGGED TABLE unlogged_items (
  id integer PRIMARY KEY,
  embedding vector(3) NOT NULL
);
INSERT INTO unlogged_items
SELECT i, ARRAY[i::real, (i % 13)::real, (i % 17)::real]::vector(3)
FROM generate_series(1, 1000) AS i;
SET hnsw.build_page_order = insertion;
CREATE INDEX unlogged_source ON unlogged_items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
SET hnsw.clone_source = 'sqlens_clone_formality_smoke.unlogged_source';
SET hnsw.build_page_order = bfs;
CREATE INDEX unlogged_clone ON unlogged_items USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
RESET hnsw.clone_source;

DO $unlogged$
DECLARE
  comparison jsonb;
BEGIN
  comparison := vector_hnsw_graph_compare('unlogged_source'::regclass, 'unlogged_clone'::regclass);
  IF NOT (comparison->>'logical_equal')::boolean
     OR pg_relation_size('unlogged_clone'::regclass, 'init') >=
        pg_relation_size('unlogged_clone'::regclass, 'main') THEN
    RAISE EXCEPTION 'UNLOGGED clone main/init fork policy failed: %, main %, init %',
      comparison,
      pg_relation_size('unlogged_clone'::regclass, 'main'),
      pg_relation_size('unlogged_clone'::regclass, 'init');
  END IF;
END
$unlogged$;

DO $permissions$
BEGIN
  IF has_function_privilege(
       'pg_monitor',
       'public.vector_hnsw_graph_fingerprint(regclass)',
       'EXECUTE')
     OR has_function_privilege(
       'pg_monitor',
       'public.vector_hnsw_graph_compare(regclass,regclass)',
       'EXECUTE') THEN
    RAISE EXCEPTION 'graph proof functions remain executable by PUBLIC';
  END IF;
END
$permissions$;

GRANT EXECUTE ON FUNCTION public.vector_hnsw_graph_fingerprint(regclass)
TO pg_monitor;
GRANT USAGE ON SCHEMA sqlens_clone_formality_smoke TO pg_monitor;
SET ROLE pg_monitor;
DO $permissions$
BEGIN
  PERFORM public.vector_hnsw_graph_fingerprint(
    'sqlens_clone_formality_smoke.source_hnsw'::regclass);
  RAISE EXCEPTION 'C owner check unexpectedly allowed a non-owner proof';
EXCEPTION
  WHEN insufficient_privilege THEN
    IF SQLERRM NOT LIKE 'must be owner of index %' THEN
      RAISE;
    END IF;
END
$permissions$;
RESET ROLE;
REVOKE EXECUTE ON FUNCTION public.vector_hnsw_graph_fingerprint(regclass)
FROM pg_monitor;
REVOKE USAGE ON SCHEMA sqlens_clone_formality_smoke FROM pg_monitor;

RESET hnsw.build_seed;
RESET hnsw.build_page_order;
RESET hnsw.require_full_memory_build;
RESET maintenance_work_mem;
DROP SCHEMA sqlens_clone_formality_smoke CASCADE;
