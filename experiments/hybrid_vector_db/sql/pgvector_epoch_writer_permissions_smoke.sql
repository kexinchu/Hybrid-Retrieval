\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS public.sqlens_epoch_writer_smoke;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'sqlens_epoch_writer') THEN
    EXECUTE 'DROP OWNED BY sqlens_epoch_writer';
    EXECUTE 'DROP ROLE sqlens_epoch_writer';
  END IF;
END
$$;

CREATE ROLE sqlens_epoch_writer
  NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOLOGIN;

CREATE TABLE public.sqlens_epoch_writer_smoke (
  id bigserial PRIMARY KEY,
  embedding vector(3) NOT NULL,
  payload text NOT NULL
) WITH (fillfactor = 50);

INSERT INTO public.sqlens_epoch_writer_smoke (embedding, payload)
VALUES ('[1,0,0]', 'seed-one'), ('[2,0,0]', 'seed-two');

CREATE INDEX sqlens_epoch_writer_smoke_embedding_idx
ON public.sqlens_epoch_writer_smoke USING hnsw (embedding vector_l2_ops);

SELECT vector_hnsw_fragment_tracking_enable(
  'public.sqlens_epoch_writer_smoke'::regclass
);

CREATE TEMP TABLE sqlens_epoch_writer_audit AS
SELECT 'public.sqlens_epoch_writer_smoke'::regclass::oid AS heap_oid,
       epoch AS initial_epoch
FROM public.pgvector_hnsw_fragment_epoch
WHERE heap_oid = 'public.sqlens_epoch_writer_smoke'::regclass;

SELECT pg_catalog.pg_stat_reset_single_table_counters(
  'public.sqlens_epoch_writer_smoke'::regclass
);

GRANT SELECT, INSERT, UPDATE, DELETE
ON public.sqlens_epoch_writer_smoke TO sqlens_epoch_writer;
GRANT USAGE
ON SEQUENCE public.sqlens_epoch_writer_smoke_id_seq TO sqlens_epoch_writer;

SET ROLE sqlens_epoch_writer;
INSERT INTO public.sqlens_epoch_writer_smoke (embedding, payload)
VALUES ('[3,0,0]', 'writer-insert');
UPDATE public.sqlens_epoch_writer_smoke
SET embedding = '[1.5,0,0]'
WHERE id = 1;
-- payload is not indexed, and fillfactor leaves room for this HOT update.
UPDATE public.sqlens_epoch_writer_smoke
SET payload = 'writer-hot'
WHERE id = 2;
DELETE FROM public.sqlens_epoch_writer_smoke WHERE id = 3;
RESET ROLE;

SELECT pg_catalog.pg_stat_force_next_flush();

DO $$
DECLARE
  before_epoch bigint;
  after_epoch bigint;
BEGIN
  SELECT initial_epoch INTO STRICT before_epoch
  FROM sqlens_epoch_writer_audit;

  SELECT epoch INTO STRICT after_epoch
  FROM public.pgvector_hnsw_fragment_epoch
  WHERE heap_oid = 'public.sqlens_epoch_writer_smoke'::regclass;

  IF after_epoch <> before_epoch + 4 THEN
    RAISE EXCEPTION 'writer DML advanced epoch from % to %, expected %',
      before_epoch, after_epoch, before_epoch + 4;
  END IF;

  IF pg_catalog.pg_stat_get_tuples_hot_updated(
       'public.sqlens_epoch_writer_smoke'::regclass
     ) < 1 THEN
    RAISE EXCEPTION 'writer payload update did not exercise PostgreSQL HOT';
  END IF;
END
$$;

DO $$
DECLARE
  denied boolean := false;
BEGIN
  EXECUTE 'SET LOCAL ROLE sqlens_epoch_writer';
  BEGIN
    UPDATE public.pgvector_hnsw_fragment_epoch
    SET epoch = epoch + 1000
    WHERE heap_oid = 'public.sqlens_epoch_writer_smoke'::regclass;
  EXCEPTION WHEN insufficient_privilege THEN
    denied := true;
  END;
  EXECUTE 'RESET ROLE';

  IF NOT denied THEN
    RAISE EXCEPTION 'ordinary writer directly modified fragment epoch metadata';
  END IF;
END
$$;

DO $$
BEGIN
  IF pg_catalog.has_table_privilege(
       'sqlens_epoch_writer',
       'public.pgvector_hnsw_fragment_epoch',
       'INSERT'
     ) OR
     pg_catalog.has_table_privilege(
       'sqlens_epoch_writer',
       'public.pgvector_hnsw_fragment_epoch',
       'UPDATE'
     ) OR
     pg_catalog.has_table_privilege(
       'sqlens_epoch_writer',
       'public.pgvector_hnsw_fragment_epoch',
       'DELETE'
     ) THEN
    RAISE EXCEPTION 'ordinary writer unexpectedly has fragment metadata DML privileges';
  END IF;
END
$$;

DELETE FROM public.pgvector_hnsw_fragment_epoch
WHERE heap_oid = (SELECT heap_oid FROM sqlens_epoch_writer_audit);
DROP TABLE public.sqlens_epoch_writer_smoke;
DROP OWNED BY sqlens_epoch_writer;
DROP ROLE sqlens_epoch_writer;
