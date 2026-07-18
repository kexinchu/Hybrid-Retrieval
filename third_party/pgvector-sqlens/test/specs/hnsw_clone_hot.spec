setup
{
  CREATE TABLE hot_items (id integer, embedding vector(3)) WITH (fillfactor = 50);
  INSERT INTO hot_items
  SELECT i, ARRAY[i::real, (i % 17)::real, (i % 31)::real]::vector(3)
  FROM generate_series(1, 400) AS i;
}

teardown
{
  DROP TABLE hot_items;
}

session "old_snapshot"
step "take_old_snapshot"
{
  BEGIN ISOLATION LEVEL REPEATABLE READ;
  DO $$ BEGIN PERFORM count(*) FROM hot_items; END $$;
}
step "plan_with_old_snapshot"
{
  SET enable_seqscan = off;
  SET hnsw.preferred_index = 'hot_clone';
  DO $$
  DECLARE plan jsonb;
  BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
            'SELECT id FROM hot_items '
            'ORDER BY embedding <-> ''[900,901,902]''::vector LIMIT 5'
      INTO plan;
    IF plan::text NOT LIKE '%Seq Scan%' OR plan::text LIKE '%hot_clone%' THEN
      RAISE EXCEPTION 'old snapshot used clone before its HOT horizon: %', plan;
    END IF;
  END
  $$;
}
step "finish_old_snapshot"
{
  COMMIT;
}
step "plan_with_fresh_snapshot"
{
  DO $$
  DECLARE plan jsonb;
  BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON, COSTS OFF) '
            'SELECT id FROM hot_items '
            'ORDER BY embedding <-> ''[900,901,902]''::vector LIMIT 5'
      INTO plan;
    IF plan::text NOT LIKE '%Index Scan%' OR plan::text NOT LIKE '%hot_clone%' THEN
      RAISE EXCEPTION 'fresh snapshot did not use clone after its HOT horizon: %', plan;
    END IF;
  END
  $$;
}

session "builder"
step "prepare_table"
{
  VACUUM hot_items;
}
step "hot_update"
{
  UPDATE hot_items
  SET embedding = '[900,901,902]'::vector(3)
  WHERE id = 1;
}
step "build_source"
{
  SET max_parallel_maintenance_workers = 0;
  SET maintenance_work_mem = '64MB';
  SET hnsw.require_full_memory_build = on;
  CREATE INDEX hot_source ON hot_items USING hnsw (embedding vector_l2_ops)
  WITH (m = 8, ef_construction = 32);
  DO $$
  BEGIN
    IF NOT (SELECT indcheckxmin FROM pg_index
            WHERE indexrelid = 'hot_source'::regclass) THEN
      RAISE EXCEPTION 'source index did not record a broken HOT-chain horizon';
    END IF;
  END
  $$;
}
step "build_clone"
{
  SET hnsw.build_page_order = bfs;
  SET hnsw.clone_source = 'hot_source';
  CREATE INDEX hot_clone ON hot_items USING hnsw (embedding vector_l2_ops)
  WITH (m = 8, ef_construction = 32);
  DO $$
  BEGIN
    IF NOT (SELECT indcheckxmin FROM pg_index
            WHERE indexrelid = 'hot_clone'::regclass) THEN
      RAISE EXCEPTION 'clone did not propagate source indcheckxmin';
    END IF;
  END
  $$;
}

permutation "prepare_table" "take_old_snapshot" "hot_update" "build_source" "build_clone" "plan_with_old_snapshot" "finish_old_snapshot" "plan_with_fresh_snapshot"
