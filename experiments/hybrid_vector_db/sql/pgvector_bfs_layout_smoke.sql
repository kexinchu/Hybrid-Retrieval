CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS bfs_layout_source;
DROP TABLE IF EXISTS bfs_layout_insert;
DROP TABLE IF EXISTS bfs_layout_bfs;
CREATE TABLE bfs_layout_source (
  id bigint PRIMARY KEY,
  embedding vector(3)
);
INSERT INTO bfs_layout_source
SELECT i::bigint,
       ARRAY[
         sin(((i - 1) / 2)::float),
         cos((((i - 1) / 2) % 97)::float),
         (((i - 1) / 2) % 53)::float / 53
       ]::vector
FROM generate_series(1, 5000) AS i;
CREATE TABLE bfs_layout_insert AS SELECT * FROM bfs_layout_source ORDER BY id;
CREATE TABLE bfs_layout_bfs AS SELECT * FROM bfs_layout_source ORDER BY id;

SET maintenance_work_mem = '256MB';
SET max_parallel_maintenance_workers = 0;
SET hnsw.build_seed = 20260718;
SET hnsw.require_full_memory_build = on;
SET hnsw.build_page_order = insertion;
CREATE INDEX bfs_layout_insert_hnsw
ON bfs_layout_insert USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 100);
SET hnsw.build_seed = 20260718;
SET hnsw.build_page_order = bfs;
CREATE INDEX bfs_layout_bfs_hnsw
ON bfs_layout_bfs USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 100);
SET hnsw.build_page_order = insertion;
SET hnsw.build_seed = -1;

ANALYZE bfs_layout_insert;
ANALYZE bfs_layout_bfs;
SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 200;
SET hnsw.iterative_scan = strict_order;
SET hnsw.filter_strategy = off;

DO $$
DECLARE
  query_id bigint;
  insert_ids bigint[];
  bfs_ids bigint[];
  insert_profile jsonb;
  bfs_profile jsonb;
BEGIN
  FOREACH query_id IN ARRAY ARRAY[1, 17, 113, 997, 2048, 4093]
  LOOP
    SELECT array_agg(id ORDER BY id) INTO insert_ids
    FROM (
      SELECT id, embedding <-> (
        SELECT embedding FROM bfs_layout_insert WHERE id = query_id
      ) AS distance
      FROM bfs_layout_insert
      ORDER BY embedding <-> (
        SELECT embedding FROM bfs_layout_insert WHERE id = query_id
      )
      LIMIT 20
    ) AS q;
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO insert_profile;

    SELECT array_agg(id ORDER BY id) INTO bfs_ids
    FROM (
      SELECT id, embedding <-> (
        SELECT embedding FROM bfs_layout_bfs WHERE id = query_id
      ) AS distance
      FROM bfs_layout_bfs
      ORDER BY embedding <-> (
        SELECT embedding FROM bfs_layout_bfs WHERE id = query_id
      )
      LIMIT 20
    ) AS q;
    SELECT vector_hnsw_last_scan_profile()::jsonb INTO bfs_profile;

    IF insert_ids IS DISTINCT FROM bfs_ids THEN
      RAISE EXCEPTION 'BFS layout changed HNSW results for query %: insertion %, BFS %',
        query_id, insert_ids, bfs_ids;
    END IF;
    IF insert_profile ->> 'traversal_expanded_nodes'
         IS DISTINCT FROM bfs_profile ->> 'traversal_expanded_nodes'
       OR insert_profile ->> 'distance_compute_count'
         IS DISTINCT FROM bfs_profile ->> 'distance_compute_count' THEN
      RAISE EXCEPTION 'BFS layout changed logical traversal for query %: insertion %, BFS %',
        query_id, insert_profile, bfs_profile;
    END IF;
  END LOOP;
END $$;

RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
RESET hnsw.filter_strategy;
RESET maintenance_work_mem;
RESET max_parallel_maintenance_workers;
RESET hnsw.build_page_order;
RESET hnsw.build_seed;
RESET hnsw.require_full_memory_build;

DROP TABLE bfs_layout_source;
DROP TABLE bfs_layout_insert;
DROP TABLE bfs_layout_bfs;
