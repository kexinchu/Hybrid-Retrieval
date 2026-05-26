# Compass-Style OOD-ANNS Experiment Notes

## Goal

This experiment moves closer to the Compass benchmark style:

- real ANN vectors from existing `.fbin` files;
- synthetic structured attributes attached to each vector row;
- SQLite B-tree indexes for relational predicates;
- FAISS HNSW for ANN;
- filtered top-k queries.

The resulting logical table is:

```text
id | embedding | attr_random | attr_correlated | attr_anti | attr_bucket | attr_cluster
```

This matches the common filtered-vector-search benchmark pattern: the ANN data
comes from a standard vector dataset, and the SQL side is generated metadata or
available row metadata.

## Dataset

Pilot dataset:

```text
/home/kec23008/docker-sys/OOD-ANNS/data/WebVid8M/hard_random.1M.fbin
```

The file contains `250,000 x 512` float32 vectors.

## Structured Attributes

We attach Compass-style structured columns:

- `attr_random`: random uniform attribute, weakly correlated with vector space.
- `attr_corr`: correlated with the first PCA/random-projection-like direction.
- `attr_anti`: `1 - attr_corr`.
- `cluster_id`: coarse cluster from vector projection buckets.
- `bucket_random`: categorical buckets from `attr_random`.

Predicates:

- random range at 0.1%, 1%, 5%, 20% target selectivity.
- correlated range at 0.1%, 1%, 5%, 20%.
- anti-correlated range at 0.1%, 1%, 5%, 20%.
- cluster equality.
- random bucket equality.
- conjunction: cluster equality + random range.

## Baselines

- `sqlite_prefilter_exact`: SQL predicate first, exact vector rerank.
- `post_ann_10x`: ANN top `10k`, SQL filter, exact rerank within filtered candidates.
- `post_ann_100x`: ANN top `100k`, SQL filter, exact rerank.
- `iterative_ann`: grow ANN candidate budget until filtered hits are enough or a cap is reached.
- `adaptive_selectivity`: choose SQL prefilter for low selectivity, otherwise iterative ANN.
- `adaptive_probe`: run a small ANN probe; if local filtered-hit yield is low, choose SQL prefilter.

## What We Are Looking For

1. Random filters should be hard for ANN-first because vector neighborhoods do
   not preserve the predicate.
2. Correlated filters should be easier for ANN-first.
3. Very broad filters should make SQL prefilter exact more expensive.
4. A scheduler should exploit both global selectivity and local ANN pass rate.

