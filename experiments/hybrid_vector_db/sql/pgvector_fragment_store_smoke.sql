CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS fragment_store_smoke;
CREATE TABLE fragment_store_smoke (
  id bigserial PRIMARY KEY,
  embedding vector(3),
  tenant_id int,
  price int
);

INSERT INTO fragment_store_smoke (embedding, tenant_id, price)
SELECT ARRAY[(i % 7)::float, (i % 11)::float, (i % 13)::float]::vector,
       i % 4,
       i % 100
FROM generate_series(1, 5000) i;

CREATE INDEX fragment_store_smoke_embedding_idx ON fragment_store_smoke USING hnsw (embedding vector_l2_ops);

DROP TABLE IF EXISTS public.pgvector_hnsw_fragment_store;

SELECT vector_hnsw_guidance_activate(
  'fragment_store_smoke_embedding_idx'::regclass,
  ARRAY['page:sql:tenant_id = 1', 'bloom:sql:price <= 20'],
  'exact'
) AS activated_atoms;

SELECT (vector_hnsw_guidance_profile()::json ->> 'last_cache_memory_bytes')::bigint > 0 AS has_memory_bytes;

SELECT kind, rows > 0 AS has_rows, octet_length(payload) > 0 AS has_payload
FROM public.pgvector_hnsw_fragment_store
ORDER BY kind;

SET hnsw.metadata_cache_max_mb = 1;

SELECT vector_hnsw_guidance_activate(
  'fragment_store_smoke_embedding_idx'::regclass,
  ARRAY['bloom:sql:tenant_id = 2'],
  'exact'
) AS activated_atoms;

SELECT vector_hnsw_guidance_reset();
RESET hnsw.metadata_cache_max_mb;

DROP TABLE fragment_store_smoke;
