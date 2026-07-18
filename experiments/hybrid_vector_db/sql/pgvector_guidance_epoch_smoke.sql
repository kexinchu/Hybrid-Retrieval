\set ON_ERROR_STOP on

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
SET hnsw.iterative_scan = off;
SET hnsw.filter_strategy = traversal_guided;

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

-- A disabled invalidation trigger makes the existing guide unsafe.  Scan
-- setup must deactivate it before any pre-distance membership check.
ALTER TABLE guidance_epoch_smoke
DISABLE TRIGGER pgvector_hnsw_fragment_epoch;

DO $$
DECLARE
  actual_id bigint;
  profile jsonb;
  guidance_profile jsonb;
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

  SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
  SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance_profile;
  IF actual_id IS DISTINCT FROM 0 OR
     (guidance_profile->>'active')::boolean OR
     profile->>'final_path' <> 'stock_bypass' OR
     profile->>'planner_proof_bypass_reason' <> 'stale_relation' OR
     (profile->>'pre_distance_membership_checks')::bigint <> 0 OR
     profile->>'filter_strategy' <> 'traversal_guided' OR
     profile->>'iterative_scan' <> 'off' THEN
    RAISE EXCEPTION 'disabled trigger did not fail open: result %, profile %, guidance %',
      actual_id, profile, guidance_profile;
  END IF;
END $$;

-- Even a superuser compatibility override cannot enable hard guidance without
-- a valid target-relation trigger.
SET hnsw.guidance_require_epoch = off;
DO $$
DECLARE
  rejected boolean := false;
  profile jsonb;
BEGIN
  BEGIN
    PERFORM vector_hnsw_guidance_activate(
      'guidance_epoch_smoke_hnsw'::regclass,
      ARRAY['exact:sql:tenant_id = 1'],
      'exact'
    );
  EXCEPTION WHEN object_not_in_prerequisite_state THEN
    rejected := true;
  END;

  SELECT vector_hnsw_guidance_profile()::jsonb INTO profile;
  IF NOT rejected OR (profile->>'active')::boolean THEN
    RAISE EXCEPTION 'guidance_require_epoch bypassed invalid trigger: %', profile;
  END IF;
END $$;
RESET hnsw.guidance_require_epoch;

-- tracking_enable repairs disabled state and advances the epoch because writes
-- could have escaped invalidation during the unsafe interval.
SELECT vector_hnsw_fragment_tracking_enable('guidance_epoch_smoke'::regclass);
SELECT vector_hnsw_guidance_activate(
  'guidance_epoch_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

-- Dropping the trigger after activation has the same scan-time fail-open
-- contract and must not leave the backend-local guide active.
DROP TRIGGER pgvector_hnsw_fragment_epoch ON guidance_epoch_smoke;
DO $$
DECLARE
  actual_id bigint;
  profile jsonb;
  guidance_profile jsonb;
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

  SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
  SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance_profile;
  IF actual_id IS DISTINCT FROM 0 OR
     (guidance_profile->>'active')::boolean OR
     profile->>'final_path' <> 'stock_bypass' OR
     profile->>'planner_proof_bypass_reason' <> 'stale_relation' OR
     (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
    RAISE EXCEPTION 'dropped trigger did not fail open: result %, profile %, guidance %',
      actual_id, profile, guidance_profile;
  END IF;
END $$;

SELECT vector_hnsw_fragment_tracking_enable('guidance_epoch_smoke'::regclass);
SELECT vector_hnsw_guidance_activate(
  'guidance_epoch_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);

-- A same-name trigger bound to the wrong function is not valid tracking.
CREATE FUNCTION pg_temp.guidance_epoch_noop_trigger() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RETURN NULL;
END
$$;
DROP TRIGGER pgvector_hnsw_fragment_epoch ON guidance_epoch_smoke;
CREATE TRIGGER pgvector_hnsw_fragment_epoch
AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON guidance_epoch_smoke
FOR EACH STATEMENT EXECUTE FUNCTION pg_temp.guidance_epoch_noop_trigger();

DO $$
DECLARE
  actual_id bigint;
  profile jsonb;
  guidance_profile jsonb;
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

  SELECT vector_hnsw_last_scan_profile()::jsonb INTO profile;
  SELECT vector_hnsw_guidance_profile()::jsonb INTO guidance_profile;
  IF actual_id IS DISTINCT FROM 0 OR
     (guidance_profile->>'active')::boolean OR
     profile->>'final_path' <> 'stock_bypass' OR
     profile->>'planner_proof_bypass_reason' <> 'stale_relation' OR
     (profile->>'pre_distance_membership_checks')::bigint <> 0 THEN
    RAISE EXCEPTION 'mismatched trigger did not fail open: result %, profile %, guidance %',
      actual_id, profile, guidance_profile;
  END IF;
END $$;

-- Repair the mismatch, then prove the recreated trigger invalidates a newly
-- activated guide on the next write.
SELECT vector_hnsw_fragment_tracking_enable('guidance_epoch_smoke'::regclass);
SELECT vector_hnsw_guidance_activate(
  'guidance_epoch_smoke_hnsw'::regclass,
  ARRAY['exact:sql:tenant_id = 1'],
  'exact'
);
INSERT INTO guidance_epoch_smoke VALUES (-1, '[-1,0,0]', 1);

DO $$
DECLARE
  actual_id bigint;
  profile jsonb;
BEGIN
  SELECT id INTO actual_id
  FROM guidance_epoch_smoke
  WHERE tenant_id = 1
    AND (SELECT vector_hnsw_guidance_bind(
             'guidance_epoch_smoke_hnsw'::regclass,
             ARRAY['exact:sql:tenant_id = 1'],
             'exact'
         ) OFFSET 0)
  ORDER BY embedding <-> '[-1,0,0]'
  LIMIT 1;

  SELECT vector_hnsw_guidance_profile()::jsonb INTO profile;
  IF actual_id IS DISTINCT FROM -1 OR (profile->>'active')::boolean THEN
    RAISE EXCEPTION 'restored trigger did not invalidate guidance: result %, profile %',
      actual_id, profile;
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

DROP TABLE guidance_epoch_smoke;
