# Real-Scenario Motivation Notes

## Goal

We want to check whether the bottlenecks observed in the controlled synthetic
tests also appear on real data distributions.

The target query shape follows the Compass setting:

```sql
SELECT id
FROM documents
WHERE structured_predicate
ORDER BY vector_distance(embedding, query_embedding)
LIMIT k;
```

The scheduling question is: at runtime, how should the system allocate work
between ANN expansion, relational predicate evaluation/index scan, exact
reranking, and stopping/fallback decisions?

## First Real Dataset: 20 Newsgroups

This is a real text corpus with 11,314 training documents and 20 human labels.
It is not a production vector database dataset, but it gives us real text
length distributions, real topic labels, and real semantic clusters.

### Vector / ANNS Side

- Raw data: document text.
- Vector representation: TF-IDF followed by TruncatedSVD to dense vectors.
- ANNS index: FAISS `IndexHNSWFlat`.
- Query vectors: sampled document vectors.

### Relational Side

Stored in SQLite with B-tree indexes:

- `id INTEGER PRIMARY KEY`
- `target INTEGER`: original 20 Newsgroups class id.
- `group_id INTEGER`: coarse topic group, derived from the label prefix.
- `char_len INTEGER`: document character length.
- `word_len INTEGER`: document word count.

Indexes:

- `idx_target`
- `idx_group`
- `idx_char_len`
- `idx_word_len`
- composite indexes on `(group_id, char_len)` and `(target, char_len)`

This gives us a real relational execution substrate, not just a numpy mask.

## Query Predicates

We test predicates that resemble common metadata filters:

- `target_eq`: exact category filter.
- `group_eq`: broader topical group filter.
- `char_window_1pct`, `char_window_5pct`, `char_window_20pct`: range filters on
  document length with controlled approximate selectivity.
- `group_and_char_5pct`: conjunction of category-like and range metadata.

## Baselines

- `sqlite_prefilter_exact`: relational predicate first, then exact vector
  reranking over matched row ids.
- `post_ann_10x`: HNSW top `10k`, then predicate filter.
- `post_ann_100x`: HNSW top `100k`, then predicate filter.
- `iterative_ann`: grow HNSW candidate budget until enough filtered candidates
  are found or a cap is reached.
- `adaptive_selectivity`: use SQLite `COUNT(*)` selectivity; choose
  prefilter-exact for narrow filters, otherwise iterative ANN.
- `adaptive_probe`: run a small ANN probe, estimate local predicate pass rate,
  and choose prefilter-exact when ANN neighborhoods look unproductive.

## What We Hope To Observe

1. Fixed post-filter overfetch fails when the relational predicate is selective.
2. Iterative ANN may spend many candidates but still fail to fill top-k when
   semantic neighborhoods do not match the predicate.
3. Relational prefilter is robust but becomes slower when the predicate is broad.
4. Runtime local pass-rate probes are more informative than selectivity alone.

## Important Caveats

- 20 Newsgroups has only 11k documents, so absolute latencies are not
  production-scale.
- TF-IDF+SVD is a practical dense embedding proxy, not a neural embedding model.
- SQLite is used as a lightweight B-tree relational engine. It is enough for
  motivation, but later work should repeat the test with PostgreSQL/DuckDB and
  larger vector datasets.
- Empty or near-empty documents can become zero vectors. These are filtered out
  before building HNSW; otherwise the graph can degenerate and confound the
  filtered-search analysis with a basic vector-quality problem.
