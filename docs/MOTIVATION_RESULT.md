# Motivation Result

## Real Attribute Filters

The current benchmark uses meaningful Amazon Review metadata predicates rather
than synthetic `mod(id)` filters.

| Target | Filter | Predicate | Actual rate |
|---:|---|---|---:|
| 50% | `popular_ge1000` | `item_rating_number >= 1000` | 50.32% |
| 20% | `price_10_to_20` | `has_price AND price > 10 AND price <= 20` | 21.89% |
| 10% | `rating5_price_le10` | `has_price AND price <= 10 AND rating = 5` | 9.59% |
| 5% | `long_review_ge500` | `review_text_len >= 500` | 5.88% |
| 2% | `grocery_rating5` | `main_category = 'Grocery' AND rating = 5` | 2.34% |
| 1% | `grocery_helpful` | `main_category = 'Grocery' AND helpful_vote >= 1` | 1.01% |
| 0.5% | `helpful_ge20` | `helpful_vote >= 20` | 0.61% |
| 0.2% | `grocery_long500` | `main_category = 'Grocery' AND review_text_len >= 500` | 0.21% |

## DuckDB + FAISS Result

Configuration:

```text
dataset = Amazon Grocery reviews, 10M rows
queries = 100
top-k = 10
FAISS HNSW efSearch = 1000
parallel vector topN = 50000
ground truth = exact L2 top-10 over the full filtered set
```

| Target | Filter | SQL rows | DuckDB SQL ms | Parallel recall | Parallel latency ms | Pre recall | Pre latency ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50% | `popular_ge1000` | 5,031,984 | 181.59 | 0.655 | 189.66 | 0.661 | 184.73 |
| 20% | `price_10_to_20` | 2,189,009 | 287.75 | 0.642 | 288.38 | 0.641 | 290.87 |
| 10% | `rating5_price_le10` | 958,716 | 167.56 | 0.637 | 167.92 | 0.629 | 170.42 |
| 5% | `long_review_ge500` | 588,019 | 65.64 | 0.490 | 68.02 | 0.475 | 68.25 |
| 2% | `grocery_rating5` | 234,056 | 48.25 | 0.625 | 51.82 | 0.614 | 50.80 |
| 1% | `grocery_helpful` | 101,481 | 26.12 | 0.621 | 30.02 | 0.608 | 28.48 |
| 0.5% | `helpful_ge20` | 60,689 | 32.82 | 0.392 | 36.91 | 0.380 | 35.12 |
| 0.2% | `grocery_long500` | 21,317 | 34.56 | 0.252 | 38.23 | 0.249 | 36.81 |

## Interpretation

DuckDB removes most PostgreSQL client/server id-export overhead. The bottleneck
is no longer simply SQL materialization; recall is mainly governed by HNSW
behavior and by how much the vector candidate set overlaps the filtered ground
truth.

`parallel_join` and `pre_filter_allow_list` are close under the current
configuration. Narrow filters remain difficult because global ANN candidates
overlap less with the filtered top-k, while allow-list HNSW still depends on
graph traversal quality under the selector.

## PostgreSQL Caveat

PostgreSQL indexed predicates can be fast when fetching bounded result sets, but
full allow-list export can dominate latency when millions of ids cross the
client/server boundary.

For `item_rating_number >= 1000`:

| Operation | Rows | Latency |
|---|---:|---:|
| PostgreSQL `LIMIT 500` | 500 | 0.83 ms client fetch |
| PostgreSQL `LIMIT 50000` | 50,000 | 47.71 ms client fetch |
| PostgreSQL full export | 5,031,984 | 4775.65 ms client fetch |
| DuckDB full export | 5,031,984 | 181.59 ms |

This is why the current primary result uses DuckDB for scalar predicates.
