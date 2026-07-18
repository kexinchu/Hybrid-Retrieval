\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS exact_no_id_smoke CASCADE;
CREATE TABLE exact_no_id_smoke (
    row_key text PRIMARY KEY,
    embedding vector(3) NOT NULL,
    tenant_id integer NOT NULL,
    generation integer NOT NULL DEFAULT 0
) WITH (fillfactor = 50);

INSERT INTO exact_no_id_smoke (row_key, embedding, tenant_id)
SELECT format('row-%s', lpad(value::text, 4, '0')),
       ARRAY[((value - 1)::real / 128), 0::real, 0::real]::vector,
       CASE WHEN value = 1 THEN 0 ELSE 1 END
FROM generate_series(1, 128) AS value;

CREATE INDEX exact_no_id_smoke_hnsw
ON exact_no_id_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 40);
ANALYZE exact_no_id_smoke;
SELECT vector_hnsw_fragment_tracking_enable('exact_no_id_smoke'::regclass);

SET enable_seqscan = off;
SET enable_sort = off;
SET enable_bitmapscan = off;
SET hnsw.ef_search = 128;
SET hnsw.max_scan_tuples = 10000;
SET hnsw.iterative_scan = strict_order;
SET hnsw.filter_strategy = safe_guided;

-- Building exact guidance must depend only on ctid, not on a user id column.
SELECT vector_hnsw_guidance_activate(
    'exact_no_id_smoke_hnsw'::regclass,
    ARRAY['exact:sql:tenant_id = 1'],
    'exact'
);

DO $$
DECLARE
    guided_count bigint;
    all_matching boolean;
    nearest_key text;
    profile jsonb;
BEGIN
    SELECT count(*), bool_and(tenant_id = 1), min(row_key)
      INTO guided_count, all_matching, nearest_key
    FROM (
        SELECT row_key, tenant_id
        FROM exact_no_id_smoke
        WHERE tenant_id = 1
          AND (SELECT vector_hnsw_guidance_bind(
                   'exact_no_id_smoke_hnsw'::regclass,
                   ARRAY['exact:sql:tenant_id = 1'],
                   'exact'
               ) OFFSET 0)
        ORDER BY embedding <-> '[0,0,0]'::vector
        LIMIT 10
    ) AS guided;

    IF guided_count <> 10 OR NOT all_matching OR nearest_key IS DISTINCT FROM 'row-0002' THEN
        RAISE EXCEPTION 'exact guidance filtering failed: count %, all_matching %, nearest %',
            guided_count, all_matching, nearest_key;
    END IF;

    SELECT vector_hnsw_guidance_profile()::jsonb INTO profile;
    IF (profile ->> 'binding_matches')::bigint < 1 OR
       (profile ->> 'binding_scan_matches')::bigint < 1 THEN
        RAISE EXCEPTION 'matching statement binding was not used: %', profile;
    END IF;
END
$$;

SELECT pg_stat_force_next_flush();

UPDATE exact_no_id_smoke
   SET tenant_id = 1, generation = generation + 1
 WHERE row_key = 'row-0001';
SELECT pg_stat_force_next_flush();

DO $$
BEGIN
    IF pg_stat_get_tuples_hot_updated('exact_no_id_smoke'::regclass) < 1 THEN
        RAISE EXCEPTION 'predicate-crossing update did not exercise a HOT chain';
    END IF;
END
$$;

-- The epoch mismatch must fail open so the newly matching HOT tuple is visible.
DO $$
DECLARE
    stale_key text;
    profile jsonb;
BEGIN
    SELECT row_key INTO stale_key
    FROM exact_no_id_smoke
    WHERE tenant_id = 1
      AND (SELECT vector_hnsw_guidance_bind(
               'exact_no_id_smoke_hnsw'::regclass,
               ARRAY['exact:sql:tenant_id = 1'],
               'exact'
           ) OFFSET 0)
    ORDER BY embedding <-> '[0,0,0]'::vector
    LIMIT 1;

    SELECT vector_hnsw_guidance_profile()::jsonb INTO profile;
    IF stale_key IS DISTINCT FROM 'row-0001' OR (profile ->> 'active')::boolean THEN
        RAISE EXCEPTION 'stale exact guidance did not fail open: key %, profile %',
            stale_key, profile;
    END IF;
END
$$;

SELECT vector_hnsw_guidance_activate(
    'exact_no_id_smoke_hnsw'::regclass,
    ARRAY['exact:sql:tenant_id = 1'],
    'exact'
);

DO $$
DECLARE
    reactivated_key text;
    profile jsonb;
BEGIN
    SELECT row_key INTO reactivated_key
    FROM exact_no_id_smoke
    WHERE tenant_id = 1
      AND (SELECT vector_hnsw_guidance_bind(
               'exact_no_id_smoke_hnsw'::regclass,
               ARRAY['exact:sql:tenant_id = 1'],
               'exact'
           ) OFFSET 0)
    ORDER BY embedding <-> '[0,0,0]'::vector
    LIMIT 1;

    SELECT vector_hnsw_guidance_profile()::jsonb INTO profile;
    IF reactivated_key IS DISTINCT FROM 'row-0001' OR
       NOT (profile ->> 'active')::boolean OR
       NOT (profile ->> 'epoch_tracked')::boolean THEN
        RAISE EXCEPTION 'reactivated exact guidance failed: key %, profile %',
            reactivated_key, profile;
    END IF;
END
$$;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET enable_bitmapscan;
RESET hnsw.ef_search;
RESET hnsw.max_scan_tuples;
RESET hnsw.iterative_scan;
RESET hnsw.filter_strategy;

DROP TABLE exact_no_id_smoke CASCADE;
