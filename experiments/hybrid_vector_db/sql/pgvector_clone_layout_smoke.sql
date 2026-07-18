\set ON_ERROR_STOP on

DROP SCHEMA IF EXISTS sqlens_d2_smoke CASCADE;
CREATE SCHEMA sqlens_d2_smoke;
SET search_path = sqlens_d2_smoke, public;
SET max_parallel_maintenance_workers = 0;
SET maintenance_work_mem = '256MB';
SET hnsw.require_full_memory_build = on;

CREATE TABLE clone_heap (
  id bigint PRIMARY KEY,
  grp integer NOT NULL,
  embedding vector(3) NOT NULL
);

INSERT INTO clone_heap
SELECT i,
       i % 7,
       CASE
         WHEN i <= 24 THEN '[1,2,3]'::vector(3)
         ELSE ARRAY[(i % 17)::real, (i % 29)::real, (i % 43)::real]::vector(3)
       END
FROM generate_series(1, 600) AS i;

SET hnsw.build_seed = 31;
SET hnsw.build_page_order = insertion;
CREATE INDEX clone_source
ON clone_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

BEGIN;
SET LOCAL hnsw.clone_source = 'sqlens_d2_smoke.clone_source';
SET LOCAL hnsw.require_full_memory_build = on;
SET LOCAL hnsw.build_page_order = bfs;
CREATE INDEX clone_bfs
ON clone_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
COMMIT;

DO $proof$
DECLARE
  comparison jsonb;
  source_proof jsonb;
BEGIN
  comparison := vector_hnsw_graph_compare('clone_source', 'clone_bfs');
  source_proof := vector_hnsw_graph_fingerprint('clone_source');

  IF NOT (comparison->>'same_heap')::boolean
     OR NOT (comparison->>'logical_equal')::boolean
     OR NOT (comparison->>'entry_equal')::boolean
     OR NOT (comparison->>'tuple_coverage_equal')::boolean
     OR (comparison->>'physical_equal')::boolean THEN
    RAISE EXCEPTION 'same-graph BFS proof failed: %', comparison;
  END IF;
  IF (source_proof->>'heap_tids')::bigint <> 600
     OR (source_proof->>'nodes')::bigint >= 600 THEN
    RAISE EXCEPTION 'duplicate-vector/TID bundle proof failed: %', source_proof;
  END IF;
END
$proof$;

SET hnsw.build_seed = 37;
SET hnsw.build_page_order = insertion;
CREATE INDEX expression_source
ON clone_heap USING hnsw (
  ((embedding + '[0,0,0]'::vector)::vector(3)) vector_l2_ops
)
WITH (m = 8, ef_construction = 32)
WHERE grp < 4;

BEGIN;
SET LOCAL hnsw.clone_source = 'sqlens_d2_smoke.expression_source';
SET LOCAL hnsw.require_full_memory_build = on;
SET LOCAL hnsw.build_page_order = bfs;
CREATE INDEX expression_clone
ON clone_heap USING hnsw (
  ((embedding + '[0,0,0]'::vector)::vector(3)) vector_l2_ops
)
WITH (m = 8, ef_construction = 32)
WHERE grp < 4;
COMMIT;

DO $expression$
DECLARE comparison jsonb;
BEGIN
  comparison := vector_hnsw_graph_compare('expression_source', 'expression_clone');
  IF NOT (comparison->>'logical_equal')::boolean
     OR NOT (comparison->>'entry_equal')::boolean
     OR NOT (comparison->>'tuple_coverage_equal')::boolean THEN
    RAISE EXCEPTION 'expression/partial clone proof failed: %', comparison;
  END IF;
END
$expression$;

CREATE TABLE mvcc_heap (
  id bigint PRIMARY KEY,
  embedding vector(3) NOT NULL
);
INSERT INTO mvcc_heap
SELECT i, ARRAY[i::real, (i % 11)::real, (i % 5)::real]::vector(3)
FROM generate_series(1, 160) AS i;

SET hnsw.build_seed = 19;
SET hnsw.build_page_order = insertion;
CREATE INDEX mvcc_source
ON mvcc_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

BEGIN;
DELETE FROM mvcc_heap WHERE id IN (3, 7, 11);
UPDATE mvcc_heap SET embedding = '[9,9,9]' WHERE id IN (20, 21);
INSERT INTO mvcc_heap VALUES
  (1001, '[9,9,9]'),
  (1002, '[9,9,9]');
SET LOCAL hnsw.clone_source = 'sqlens_d2_smoke.mvcc_source';
SET LOCAL hnsw.require_full_memory_build = on;
SET LOCAL hnsw.build_page_order = bfs;
CREATE INDEX mvcc_clone
ON mvcc_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
COMMIT;

DO $mvcc$
DECLARE comparison jsonb;
BEGIN
  comparison := vector_hnsw_graph_compare('mvcc_source', 'mvcc_clone');
  IF NOT (comparison->>'logical_equal')::boolean
     OR NOT (comparison->>'entry_equal')::boolean
     OR NOT (comparison->>'tuple_coverage_equal')::boolean THEN
    RAISE EXCEPTION 'MVCC clone proof failed: %', comparison;
  END IF;
END
$mvcc$;

CREATE TABLE other_heap (LIKE clone_heap INCLUDING ALL);
INSERT INTO other_heap SELECT * FROM clone_heap;

DO $guards$
DECLARE message text;
BEGIN
  PERFORM set_config('hnsw.clone_source', 'sqlens_d2_smoke.clone_source', true);
  PERFORM set_config('hnsw.require_full_memory_build', 'on', true);
  PERFORM set_config('hnsw.build_page_order', 'bfs', true);

  BEGIN
    EXECUTE 'CREATE INDEX bad_m ON clone_heap USING hnsw (embedding vector_l2_ops) WITH (m=9, ef_construction=36)';
    RAISE EXCEPTION 'm mismatch was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source and destination HNSW build options do not match%' THEN RAISE; END IF;
  END;

  BEGIN
    EXECUTE 'CREATE INDEX bad_opclass ON clone_heap USING hnsw (embedding vector_ip_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'opclass mismatch was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source and destination HNSW index definitions do not match%' THEN RAISE; END IF;
  END;

  BEGIN
    EXECUTE 'CREATE INDEX bad_expression ON clone_heap USING hnsw (((embedding + ''[0,0,0]''::vector)::vector(3)) vector_l2_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'expression mismatch was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source and destination HNSW index definitions do not match%' THEN RAISE; END IF;
  END;

  BEGIN
    EXECUTE 'CREATE INDEX bad_predicate ON clone_heap USING hnsw (embedding vector_l2_ops) WITH (m=8, ef_construction=32) WHERE grp = 1';
    RAISE EXCEPTION 'predicate mismatch was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source and destination HNSW index definitions do not match%' THEN RAISE; END IF;
  END;

  BEGIN
    EXECUTE 'CREATE INDEX bad_heap ON other_heap USING hnsw (embedding vector_l2_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'cross-heap source was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source HNSW index % is on a different heap' THEN RAISE; END IF;
  END;

  PERFORM set_config('hnsw.require_full_memory_build', 'off', true);
  BEGIN
    EXECUTE 'CREATE INDEX bad_memory_guard ON clone_heap USING hnsw (embedding vector_l2_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'full-memory guard was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'hnsw.clone_source requires hnsw.require_full_memory_build%' THEN RAISE; END IF;
  END;

  PERFORM set_config('hnsw.require_full_memory_build', 'on', true);
  PERFORM set_config('hnsw.build_page_order', 'insertion', true);
  BEGIN
    EXECUTE 'CREATE INDEX bad_layout_guard ON clone_heap USING hnsw (embedding vector_l2_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'BFS guard was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'hnsw.clone_source requires hnsw.build_page_order%' THEN RAISE; END IF;
  END;
END
$guards$;

UPDATE pg_catalog.pg_index
SET indisvalid = false
WHERE indexrelid = 'clone_source'::regclass;

DO $invalid$
DECLARE message text;
BEGIN
  PERFORM set_config('hnsw.clone_source', 'sqlens_d2_smoke.clone_source', true);
  PERFORM set_config('hnsw.require_full_memory_build', 'on', true);
  PERFORM set_config('hnsw.build_page_order', 'bfs', true);
  BEGIN
    EXECUTE 'CREATE INDEX bad_invalid ON clone_heap USING hnsw (embedding vector_l2_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'invalid source was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source HNSW index % is not valid and ready' THEN RAISE; END IF;
  END;
END
$invalid$;

UPDATE pg_catalog.pg_index
SET indisvalid = true
WHERE indexrelid = 'clone_source'::regclass;

CREATE TABLE low_memory_heap (
  id bigint PRIMARY KEY,
  embedding vector(3) NOT NULL
);
INSERT INTO low_memory_heap
SELECT i, ARRAY[(i % 97)::real, (i % 193)::real, (i % 389)::real]::vector(3)
FROM generate_series(1, 4000) AS i;
SET maintenance_work_mem = '64MB';
SET hnsw.build_seed = 61;
SET hnsw.build_page_order = insertion;
CREATE INDEX low_memory_source
ON low_memory_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

DO $low_memory$
DECLARE message text;
BEGIN
  PERFORM set_config('maintenance_work_mem', '1MB', true);
  PERFORM set_config('hnsw.clone_source', 'sqlens_d2_smoke.low_memory_source', true);
  PERFORM set_config('hnsw.require_full_memory_build', 'on', true);
  PERFORM set_config('hnsw.build_page_order', 'bfs', true);
  BEGIN
    EXECUTE 'CREATE INDEX bad_low_memory ON low_memory_heap USING hnsw (embedding vector_l2_ops) WITH (m=8, ef_construction=32)';
    RAISE EXCEPTION 'low-memory clone was accepted';
  EXCEPTION WHEN OTHERS THEN
    GET STACKED DIAGNOSTICS message = MESSAGE_TEXT;
    IF message NOT LIKE 'source HNSW graph does not fit into maintenance_work_mem%' THEN RAISE; END IF;
  END;
END
$low_memory$;

RESET hnsw.clone_source;
SET maintenance_work_mem = '256MB';
SET hnsw.build_seed = 23;
SET hnsw.build_page_order = insertion;
CREATE INDEX normal_build
ON other_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

DO $normal$
DECLARE proof jsonb;
BEGIN
  proof := vector_hnsw_graph_fingerprint('normal_build');
  IF (proof->>'heap_tids')::bigint <> (SELECT count(*) FROM other_heap) THEN
    RAISE EXCEPTION 'normal build smoke failed: %', proof;
  END IF;
END
$normal$;

SELECT vector_sqlens_build_id() AS build_id,
       vector_hnsw_graph_compare('clone_source', 'clone_bfs') AS clone_proof,
       vector_hnsw_graph_compare('mvcc_source', 'mvcc_clone') AS mvcc_proof;
