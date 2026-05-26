# Filtered ANN Scheduler Motivation Summary

## High-Level Question

We want to study whether ANNS + structured-data search has scheduler
optimization space.

The query shape follows Compass:

```sql
SELECT id
FROM items
WHERE structured_predicate
ORDER BY distance(embedding, query)
LIMIT k;
```

A scheduler may choose among:

- continue ANN expansion;
- use SQL/B-tree prefilter and exact rerank;
- interleave structured-index candidate injection with ANN search;
- stop early when risk is low.

## Current Best Experiment

The most relevant experiment is:

```text
experiments/filtered_ann_scheduler/run_compass_style_ood.py
```

It uses:

- ANN vectors: `/home/kec23008/docker-sys/OOD-ANNS/data/WebVid8M/hard_random.1M.fbin`
- vector shape: `250,000 x 512`
- ANNS index: FAISS `IndexHNSWFlat`
- relational engine: SQLite with B-tree indexes
- structured attributes: synthetic Compass-style columns attached to vector rows

## Attribute Construction

The attributes are generated, not naturally present in the OOD-ANNS files.

| Attribute | Construction | Purpose |
| --- | --- | --- |
| `attr_random` | Uniform random value per row | Weak vector/filter correlation |
| `attr_corr` | Rank of projection on a random vector direction | Correlated range attribute |
| `attr_anti` | `1 - attr_corr` | Anti-correlated range attribute |
| `bucket_random` | 100 buckets from `attr_random` | Equality/category predicate |
| `cluster_id` | 100 buckets from `attr_corr` | Vector-related category predicate |

This is reasonable for motivation because filtered ANN performance depends on
selectivity, predicate type, and vector-filter correlation. It is not a 100%
reproduction of Compass.

Compass uses standard ANN datasets such as `GLOVE100`, `GIST`, `CRAWL`,
`VIDEO`, and `DEEP10M`, then attaches synthetic or partially real metadata.
Our current experiment follows that pattern but uses OOD-ANNS `WebVid8M`.

## Metrics To Watch

`recall_at_k`:

- quality against exact filtered top-k ground truth;
- low recall means returned results are not the true filtered nearest neighbors.

`failed_to_fill` / fail rate:

- whether a strategy returned fewer than `k` filtered hits;
- high fail rate is the most direct post-filter ANN failure mode.

`p50_ms`, `p95_ms`:

- latency and tail latency;
- scheduler work is mainly about quality/cost tradeoffs and avoiding bad tails.

`ann_returned`:

- number of ANN candidates requested;
- large values with low recall indicate wasted ANN expansion.

`exact_distance_evals` and `match_count`:

- number of vectors reranked exactly after SQL filtering;
- large values explain why filter-first becomes expensive for broad predicates.

## Main Results From WebVid8M Pilot

At `0.1%` selectivity, ANN-first fails badly:

| Predicate | Strategy | Recall | Fail Rate | ANN Returned |
| --- | --- | ---: | ---: | ---: |
| `attr_random_sel0.001` | `post_ann_10x` | `0.1075` | `1.0000` | `100` |
| `attr_random_sel0.001` | `post_ann_100x` | `0.1625` | `1.0000` | `800.2` |
| `attr_random_sel0.001` | `iterative_ann` | `0.1638` | `1.0000` | `20000` |
| `cluster_and_random_sel0.05` | `iterative_ann` | `0.1513` | `1.0000` | `20000` |

At `1%` selectivity, iterative ANN is still often weak:

| Predicate | Strategy | Recall | Fail Rate | P95 |
| --- | --- | ---: | ---: | ---: |
| `attr_random_sel0.01` | `iterative_ann` | `0.4463` | `0.6000` | `12.8875 ms` |
| `bucket_random_eq` | `iterative_ann` | `0.4450` | `0.5750` | `12.5119 ms` |
| `cluster_eq` | `iterative_ann` | `0.5313` | `0.4125` | `11.3298 ms` |

Filter-first is robust but expensive for broad predicates:

| Predicate | Selectivity | Matches | P95 |
| --- | ---: | ---: | ---: |
| `attr_random_sel0.001` | `0.10%` | `249.6` | `0.3226 ms` |
| `attr_random_sel0.01` | `1.00%` | `2497.2` | `17.5506 ms` |
| `attr_random_sel0.05` | `5.00%` | `12512.0` | `28.2170 ms` |
| `attr_random_sel0.2` | `20.01%` | `50033.8` | `94.0281 ms` |

## Conclusion

The current evidence supports this motivation:

> A single fixed plan is unstable for filtered ANN. ANN-first fails at low
> selectivity or low local predicate yield, while SQL prefilter exact becomes
> expensive for broad predicates. A runtime scheduler should combine global
> selectivity, local ANN pass-rate probes, marginal ANN gain, and exact-rerank
> cost.

This conclusion is motivation-level. It is not yet a full Compass reproduction
or a SOTA comparison.

## Caveats

- SQL attributes are generated, not native metadata from OOD-ANNS.
- Dataset is OOD-ANNS WebVid8M, not the exact Compass dataset list.
- SQLite is used as a lightweight B-tree engine, not Compass's own B+-tree code.
- No comparison yet against Compass, ACORN, Filtered-DiskANN, or other baselines.
- The next step should implement a cooperative scheduler rather than only
  heuristic plan selection.

