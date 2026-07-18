\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pageinspect;
DROP TABLE IF EXISTS deterministic_build_binding_smoke CASCADE;
CREATE TABLE deterministic_build_binding_smoke (
    id integer PRIMARY KEY,
    embedding vector(2) NOT NULL
);

-- A lattice supplies many exact distance ties without duplicate vectors.
INSERT INTO deterministic_build_binding_smoke
SELECT i + 1,
       ARRAY[(i % 32)::real, (i / 32)::real]::vector
FROM generate_series(0, 1023) AS i;

SET max_parallel_maintenance_workers = 0;
SET maintenance_work_mem = '64MB';
SET hnsw.require_full_memory_build = on;
SET hnsw.build_seed = 20260718;
CREATE INDEX deterministic_build_binding_smoke_a
ON deterministic_build_binding_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 40);

SET hnsw.build_seed = 20260718;
CREATE INDEX deterministic_build_binding_smoke_b
ON deterministic_build_binding_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 40);

SET hnsw.build_seed = 20260719;
CREATE INDEX deterministic_build_binding_smoke_c
ON deterministic_build_binding_smoke USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 40);

CREATE TEMP TABLE deterministic_build_fingerprints (
    index_name text PRIMARY KEY,
    page_count integer NOT NULL,
    graph_fingerprint text NOT NULL
);

INSERT INTO deterministic_build_fingerprints
SELECT index_name,
       page_count,
       md5(string_agg(
           encode(substring(get_raw_page(index_name, block_number) FROM 25), 'hex'),
           '' ORDER BY block_number
       ))
FROM (
    SELECT index_name,
           page_count,
           generate_series(0, page_count - 1) AS block_number
    FROM (
        SELECT index_name,
               (pg_relation_size(index_name::regclass) /
                current_setting('block_size')::integer)::integer AS page_count
        FROM (VALUES
            ('deterministic_build_binding_smoke_a'),
            ('deterministic_build_binding_smoke_b'),
            ('deterministic_build_binding_smoke_c')
        ) AS indexes(index_name)
    ) AS sizes
) AS pages
GROUP BY index_name, page_count;

DO $$
DECLARE
    a deterministic_build_fingerprints%ROWTYPE;
    b deterministic_build_fingerprints%ROWTYPE;
    c deterministic_build_fingerprints%ROWTYPE;
BEGIN
    SELECT * INTO a FROM deterministic_build_fingerprints
    WHERE index_name = 'deterministic_build_binding_smoke_a';
    SELECT * INTO b FROM deterministic_build_fingerprints
    WHERE index_name = 'deterministic_build_binding_smoke_b';
    SELECT * INTO c FROM deterministic_build_fingerprints
    WHERE index_name = 'deterministic_build_binding_smoke_c';

    IF a.page_count IS DISTINCT FROM b.page_count OR
       a.graph_fingerprint IS DISTINCT FROM b.graph_fingerprint THEN
        RAISE EXCEPTION 'same-seed full graph fingerprints differ: a %, b %', a, b;
    END IF;
    IF a.graph_fingerprint IS NOT DISTINCT FROM c.graph_fingerprint THEN
        RAISE EXCEPTION 'different build seeds produced the same full graph fingerprint: a %, c %',
            a, c;
    END IF;
END
$$;

-- Distinct page metrics are scan-local unique-page counts, not block runs.
SET enable_seqscan = off;
SET enable_sort = off;
SET hnsw.iterative_scan = off;
SET hnsw.ef_search = 200;
SET hnsw.filter_strategy = off;
SELECT vector_hnsw_reset_scan_profile();
SELECT count(*)
FROM (
    SELECT id
    FROM deterministic_build_binding_smoke
    ORDER BY embedding <-> '[15.5,15.5]'::vector
    LIMIT 100
) AS scan;

DO $$
DECLARE
    profile jsonb := vector_hnsw_last_scan_profile()::jsonb;
    neighbor_runs bigint;
    neighbor_unique bigint;
    element_runs bigint;
    element_unique bigint;
BEGIN
    neighbor_runs := (profile->>'index_page_neighbor_runs')::bigint;
    neighbor_unique := (profile->>'index_page_neighbor_distinct_pages')::bigint;
    element_runs := (profile->>'index_page_element_runs')::bigint;
    element_unique := (profile->>'index_page_element_distinct_pages')::bigint;

    IF NOT (profile->>'index_page_distinct_counts_exact')::boolean OR
       (profile->>'index_page_distinct_page_limit')::integer <> 65536 OR
       profile->>'index_page_distinct_scope' <>
           'sum_of_scan_local_unique_pages' OR
       profile->>'index_page_profile_scope' <>
           'search_neighbor_and_candidate_element_pages' OR
       neighbor_unique <= 0 OR element_unique <= 0 OR
       neighbor_unique > neighbor_runs OR element_unique > element_runs OR
       (neighbor_unique = neighbor_runs AND element_unique = element_runs) OR
       (profile->>'idx_blks_hit')::bigint +
           (profile->>'idx_blks_read')::bigint <= 0 THEN
        RAISE EXCEPTION 'index page unique/run profile contract failed: %', profile;
    END IF;
END
$$;

RESET enable_seqscan;
RESET enable_sort;
RESET hnsw.iterative_scan;
RESET hnsw.ef_search;
RESET hnsw.filter_strategy;
RESET max_parallel_maintenance_workers;
RESET maintenance_work_mem;
RESET hnsw.require_full_memory_build;
RESET hnsw.build_seed;
DROP TABLE deterministic_build_binding_smoke CASCADE;
