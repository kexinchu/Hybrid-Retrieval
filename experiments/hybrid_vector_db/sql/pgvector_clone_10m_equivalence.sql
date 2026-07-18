\set ON_ERROR_STOP on

\if :{?heap_table}
\else
  \set heap_table public.amazon_grocery_reviews_10m_pgvector
\endif
\if :{?source_index}
\else
  \set source_index public.amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx
\endif
\if :{?clone_index}
\else
  \set clone_index public.amazon_grocery_reviews_10m_pgvector_hnsw_bfs_clone_idx
\endif

SET hnsw.ef_search = 400;
CREATE TEMP TABLE sqlens_d2_equivalence_qids (qid bigint PRIMARY KEY);
\if :{?query_id_table}
INSERT INTO sqlens_d2_equivalence_qids
SELECT qid FROM :query_id_table ORDER BY qid LIMIT 100;
\else
INSERT INTO sqlens_d2_equivalence_qids
SELECT id FROM :heap_table ORDER BY id LIMIT 100;
\endif
INSERT INTO sqlens_d2_equivalence_qids VALUES (4683119)
ON CONFLICT DO NOTHING;

CREATE TEMP TABLE sqlens_d2_equivalence_results AS
WITH queries AS (
  SELECT ids.qid, heap.embedding
  FROM sqlens_d2_equivalence_qids AS ids
  JOIN :heap_table AS heap ON heap.id = ids.qid
), source_results AS (
  SELECT q.qid, array_agg(result.id ORDER BY result.rank) AS ids
  FROM queries AS q
  CROSS JOIN LATERAL vector_hnsw_page_materialize(
    :'source_index'::regclass, q.embedding, 100, 100
  ) AS result
  GROUP BY q.qid
), clone_results AS (
  SELECT q.qid, array_agg(result.id ORDER BY result.rank) AS ids
  FROM queries AS q
  CROSS JOIN LATERAL vector_hnsw_page_materialize(
    :'clone_index'::regclass, q.embedding, 100, 100
  ) AS result
  GROUP BY q.qid
)
SELECT q.qid,
       source_results.ids AS source_ids,
       clone_results.ids AS clone_ids,
       source_results.ids IS NOT DISTINCT FROM clone_results.ids AS identical
FROM queries AS q
LEFT JOIN source_results USING (qid)
LEFT JOIN clone_results USING (qid);

CREATE TEMP TABLE sqlens_d2_graph_proof AS
SELECT vector_hnsw_graph_compare(
  :'source_index'::regclass,
  :'clone_index'::regclass
) AS comparison;

DO $equivalence$
DECLARE comparison jsonb;
BEGIN
  SELECT sqlens_d2_graph_proof.comparison
  INTO comparison
  FROM sqlens_d2_graph_proof;

  IF NOT (comparison->>'same_heap')::boolean
     OR NOT (comparison->>'logical_equal')::boolean
     OR NOT (comparison->>'entry_equal')::boolean
     OR NOT (comparison->>'tuple_coverage_equal')::boolean THEN
    RAISE EXCEPTION '10M graph equivalence failed: %', comparison;
  END IF;
  IF EXISTS (SELECT 1 FROM sqlens_d2_equivalence_results WHERE NOT identical) THEN
    RAISE EXCEPTION 'q100/qid4683119 direct-index result equivalence failed';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM sqlens_d2_equivalence_results WHERE qid = 4683119) THEN
    RAISE EXCEPTION 'qid 4683119 was not present in the target heap';
  END IF;
END
$equivalence$;

SELECT comparison AS graph_proof FROM sqlens_d2_graph_proof;
SELECT count(*) AS checked_queries,
       bool_and(identical) AS all_top100_identical
FROM sqlens_d2_equivalence_results;
