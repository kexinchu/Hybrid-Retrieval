\set ON_ERROR_STOP on

SET search_path = sqlens_d2_smoke, public;
DROP INDEX IF EXISTS concurrent_clone_guard;
SET hnsw.clone_source = 'sqlens_d2_smoke.clone_source';
SET hnsw.require_full_memory_build = on;
SET hnsw.build_page_order = bfs;

\set ON_ERROR_STOP off
CREATE INDEX CONCURRENTLY concurrent_clone_guard
ON clone_heap USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);
\set ON_ERROR_STOP on

DO $guard$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_index
    WHERE indexrelid = 'concurrent_clone_guard'::regclass
      AND NOT indisvalid
      AND NOT indisready
  ) THEN
    RAISE EXCEPTION 'concurrent clone guard did not fail inside the HNSW build';
  END IF;
END
$guard$;

DROP INDEX concurrent_clone_guard;
RESET hnsw.clone_source;
