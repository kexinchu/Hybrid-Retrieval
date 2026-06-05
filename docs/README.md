# Hybrid Retrieval

This repository currently keeps one experiment line:

```text
Amazon Grocery 10M metadata filters + FAISS HNSW vector search
```

The goal is to compare how a relational predicate branch and a vector-search
branch interact under meaningful metadata filters. The current primary setup
uses DuckDB for scalar predicates and FAISS HNSW for vector search, so both
sides run in-process and avoid PostgreSQL client/server export overhead.

## Current Layout

```text
experiments/hybrid_vector_db/
  scripts/    current data prep, index build, and benchmark scripts
  sql/        optional PostgreSQL schema

data/
  amazon_reviews_2023/
    raw_reviews/Grocery_and_Gourmet_Food.jsonl
    raw_meta_extra/meta_Grocery_and_Gourmet_Food.jsonl
    processed/grocery_reviews_10m_hybrid_sql.csv
    processed/grocery_reviews_10m_tfidf_svd128.fbin
  duckdb/amazon_grocery_10m.duckdb
  faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index

results/hybrid_vector_db/
  faiss_duckdb_attribute_filter_10m_q100_vec50000_ef1000_20260604.csv
  faiss_duckdb_attribute_filter_10m_q100_vec50000_ef1000_20260604_summary.csv
  faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv
  faiss_hnsw_sql_attribute_filter_10m_q100_20260602_summary.csv
```

## Main Benchmark

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/faiss_duckdb_attribute_filter_10m.py \
  --truth-csv results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv \
  --queries 100 \
  --vector-topn 50000 \
  --ef-search 1000 \
  --out results/hybrid_vector_db/faiss_duckdb_attribute_filter_10m_q100_vec50000_ef1000_20260604.csv
```

This compares two execution styles:

- `parallel_join`: DuckDB predicate evaluation and global FAISS HNSW search run
  independently, then results are joined and exact-reranked.
- `pre_filter_allow_list`: DuckDB produces a full allow-list, then FAISS HNSW
  searches the existing graph with an `IDSelectorBatch`.

Ground truth is exact L2 top-10 over the full filtered set:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/faiss_hnsw_sql_attribute_filter_10m.py \
  --queries 100 \
  --out results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv
```

## Data Prep

Build 10M metadata and TF-IDF + SVD vectors:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/build_amazon_reviews_hybrid_10m.py
```

Build FAISS HNSW:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/build_faiss_hnsw_from_fbin.py \
  --fbin data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin \
  --out data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index \
  --rows 10000000 \
  --m 16 \
  --ef-construction 100
```

Optional PostgreSQL comparison:

```bash
experiments/hybrid_vector_db/scripts/start_pgvector_docker.sh
.venv/bin/python experiments/hybrid_vector_db/scripts/load_amazon_reviews_hybrid_sql.py
```

## Kept Scripts

| Script | Purpose |
|---|---|
| `build_amazon_reviews_hybrid_10m.py` | Build 10M metadata CSV and vector fbin |
| `build_faiss_hnsw_from_fbin.py` | Build FAISS HNSW index |
| `faiss_duckdb_attribute_filter_10m.py` | Current DuckDB + FAISS benchmark |
| `faiss_hnsw_sql_attribute_filter_10m.py` | Exact GT and PostgreSQL diagnostics |
| `analyze_amazon_grocery_predicates.py` | Find meaningful attribute filters |
| `load_amazon_reviews_hybrid_sql.py` | Optional PostgreSQL metadata loader |
| `amazon_grocery_sql_index_sanity.py` | Optional PostgreSQL index sanity check |
| `start_pgvector_docker.sh` | Optional PostgreSQL container launcher |
| `common_pg.py` | PostgreSQL connection helper |

## Notes

- `pre_filter_exact` is ground truth, not a production execution strategy.
- Temporary per-query HNSW construction is no longer treated as the correct
  pre-filtering baseline.
- The 10M vectors are TF-IDF + SVD vectors, so these are systems experiments,
  not semantic embedding quality benchmarks.
