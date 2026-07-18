CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS hnsw_build_guard_smoke;
CREATE TABLE hnsw_build_guard_smoke AS
SELECT i AS id,
       ARRAY[sin(i::float), cos(i::float), (i % 97)::float]::vector(3) AS embedding
FROM generate_series(1, 50000) AS i;
ANALYZE hnsw_build_guard_smoke;

DO $$
BEGIN
  PERFORM set_config('maintenance_work_mem', '1MB', true);
  PERFORM set_config('max_parallel_maintenance_workers', '0', true);
  PERFORM set_config('hnsw.require_full_memory_build', 'on', true);
  EXECUTE 'CREATE INDEX hnsw_build_guard_memory_idx '
          'ON hnsw_build_guard_smoke USING hnsw (embedding vector_l2_ops)';
  RAISE EXCEPTION 'require_full_memory_build unexpectedly allowed a spill';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLERRM NOT LIKE 'hnsw graph does not fit into maintenance_work_mem%' THEN
      RAISE;
    END IF;
END $$;

ALTER TABLE hnsw_build_guard_smoke SET (parallel_workers = 2);
DO $$
BEGIN
  PERFORM set_config('maintenance_work_mem', '256MB', true);
  PERFORM set_config('max_parallel_maintenance_workers', '2', true);
  PERFORM set_config('min_parallel_table_scan_size', '0', true);
  PERFORM set_config('hnsw.build_seed', '57', true);
  PERFORM set_config('hnsw.require_full_memory_build', 'off', true);
  EXECUTE 'CREATE INDEX hnsw_build_guard_parallel_idx '
          'ON hnsw_build_guard_smoke USING hnsw (embedding vector_l2_ops)';
  RAISE EXCEPTION 'deterministic build unexpectedly allowed parallel workers';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLERRM NOT LIKE 'deterministic HNSW builds require max_parallel_maintenance_workers = 0%' THEN
      RAISE;
    END IF;
END $$;

DROP TABLE hnsw_build_guard_smoke;
