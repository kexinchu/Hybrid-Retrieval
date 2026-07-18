\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS guided_collect_resume_smoke CASCADE;
CREATE TABLE guided_collect_resume_smoke (
    id bigint PRIMARY KEY,
    category integer NOT NULL,
    embedding vector(2) NOT NULL
);

-- The matching vertices are sparse enough to require nonmatching bridge
-- vertices, while the matching population is larger than the requested k.
INSERT INTO guided_collect_resume_smoke
SELECT value,
       CASE WHEN value % 4 = 0 THEN 1 ELSE 2 END,
       ARRAY[value::real, 0::real]::vector
FROM generate_series(1, 160) AS value;

CREATE INDEX guided_collect_resume_smoke_hnsw
ON guided_collect_resume_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 80);
ANALYZE guided_collect_resume_smoke;
SELECT vector_hnsw_fragment_tracking_enable('guided_collect_resume_smoke'::regclass);

-- Build the expected top-k without using the HNSW index.
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;
CREATE TEMP TABLE guided_collect_resume_expected AS
SELECT array_agg(id ORDER BY distance) AS ids
FROM (
    SELECT id, embedding <-> '[80.25,0]'::vector AS distance
    FROM guided_collect_resume_smoke
    WHERE category = 1
    ORDER BY embedding <-> '[80.25,0]'::vector
    LIMIT 8
) AS truth;
RESET enable_indexscan;
RESET enable_bitmapscan;
RESET enable_seqscan;

CREATE TEMP TABLE guided_collect_resume_observed (
    phase text PRIMARY KEY,
    ids bigint[],
    profile jsonb
);

SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.ef_search = 4;
SET hnsw.guided_collect_target = 4;
SET hnsw.iterative_scan = strict_order;
SET hnsw.max_scan_tuples = 100000;
SET hnsw.scan_mem_multiplier = 8;
SET hnsw.page_access = off;
SET hnsw.index_page_access = off;
SET hnsw.filter_strategy = guided_collect;

SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
    'guided_collect_resume_smoke_hnsw'::regclass,
    ARRAY['exact:sql:category = 1'],
    'exact'
);

-- k is 8 but the first guided heap is bounded by ef_search=4.  The probe
-- records that the first result batch is smaller than k before the full
-- query consumes bridge resumes.
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO guided_collect_resume_observed (phase, ids)
SELECT 'first_batch', array_agg(id ORDER BY distance)
FROM (
    SELECT id, embedding <-> '[80.25,0]'::vector AS distance
    FROM guided_collect_resume_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'guided_collect_resume_smoke_hnsw'::regclass,
               ARRAY['exact:sql:category = 1'],
               'exact'
           ) OFFSET 0)
      AND category = 1
    ORDER BY embedding <-> '[80.25,0]'::vector
    LIMIT 4
) AS first_batch;
UPDATE guided_collect_resume_observed
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE phase = 'first_batch';

DO $$
DECLARE
    first_ids bigint[];
    first_profile jsonb;
BEGIN
    SELECT ids, profile INTO first_ids, first_profile
    FROM guided_collect_resume_observed
    WHERE phase = 'first_batch';

    IF coalesce(array_length(first_ids, 1), 0) <> 4
       OR (first_profile ->> 'returned_tuples')::bigint <> 4
       OR (first_profile ->> 'traversal_resume_batches')::bigint <> 0
       OR (first_profile ->> 'traversal_discarded_pushes')::bigint <= 0 THEN
        RAISE EXCEPTION 'first guided batch unexpectedly reached k=8: ids %, profile %',
            first_ids, first_profile;
    END IF;
END
$$;

-- The full request must resume from the discarded bridge frontier to fill k.
SELECT vector_hnsw_reset_scan_profile();
INSERT INTO guided_collect_resume_observed (phase, ids)
SELECT 'full', array_agg(id ORDER BY distance)
FROM (
    SELECT id, embedding <-> '[80.25,0]'::vector AS distance
    FROM guided_collect_resume_smoke
    WHERE (SELECT vector_hnsw_guidance_bind(
               'guided_collect_resume_smoke_hnsw'::regclass,
               ARRAY['exact:sql:category = 1'],
               'exact'
           ) OFFSET 0)
      AND category = 1
    ORDER BY embedding <-> '[80.25,0]'::vector
    LIMIT 8
) AS full_scan;
UPDATE guided_collect_resume_observed
SET profile = vector_hnsw_last_scan_profile()::jsonb
WHERE phase = 'full';

DO $$
DECLARE
    expected_ids bigint[];
    actual_ids bigint[];
    scan_profile jsonb;
    invalid_rows bigint;
BEGIN
    SELECT ids INTO expected_ids FROM guided_collect_resume_expected;
    SELECT observed.ids, observed.profile INTO actual_ids, scan_profile
    FROM guided_collect_resume_observed AS observed
    WHERE observed.phase = 'full';

    IF coalesce(array_length(expected_ids, 1), 0) < 8 THEN
        RAISE EXCEPTION 'exact truth did not contain k=8 matching rows: %', expected_ids;
    END IF;
    SELECT count(*) INTO invalid_rows
    FROM unnest(actual_ids) AS returned(id)
    LEFT JOIN guided_collect_resume_smoke AS source USING (id)
    WHERE source.id IS NULL OR source.category <> 1;

    IF coalesce(array_length(actual_ids, 1), 0) <> 8
       OR cardinality(array(
              SELECT DISTINCT returned_id
              FROM unnest(actual_ids) AS returned(returned_id)
          )) <> 8
       OR invalid_rows <> 0 THEN
        RAISE EXCEPTION 'guided_collect resume did not return eight unique matching rows: exact %, actual %, profile %',
            expected_ids, actual_ids, scan_profile;
    END IF;
    IF (scan_profile ->> 'traversal_resume_batches')::bigint <= 0
       OR (scan_profile ->> 'traversal_discarded_pushes')::bigint <= 0
       OR (scan_profile ->> 'traversal_discarded_pops')::bigint <= 0
       OR (scan_profile ->> 'traversal_strict_order_drops')::bigint <= 0 THEN
        RAISE EXCEPTION 'guided_collect did not exercise bridge resume/strict-order path: %', scan_profile;
    END IF;
END
$$;

SELECT phase, ids, profile
FROM guided_collect_resume_observed
ORDER BY phase;

SELECT vector_hnsw_guidance_reset();
RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.ef_search;
RESET hnsw.guided_collect_target;
RESET hnsw.iterative_scan;
RESET hnsw.max_scan_tuples;
RESET hnsw.scan_mem_multiplier;
RESET hnsw.page_access;
RESET hnsw.index_page_access;
RESET hnsw.filter_strategy;

DROP TABLE guided_collect_resume_smoke CASCADE;
