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

## 2026-07-10 SIGMOD pgvector Prototype Results

This section summarizes the current Amazon 10M PostgreSQL + pgvector prototype
results used for the SIGMOD experiment line. The main run uses:

```text
dataset = Amazon Grocery reviews, 10M rows
queries = 100
repeats = 10
top-k = 10
hnsw.ef_search = 1000
hnsw.iterative_scan = strict_order
hnsw.max_scan_tuples = 200000 unless stated otherwise
cache policy = warm-cache main run
ground truth = exact SQL-valid top-10
```

Primary raw and summary files:

- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705.csv`
- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705_table.csv`
- `results/hybrid_vector_db/sigmod_d4_calibration_q100r10_warmall_main_20260710_002705_merged_table.csv`
- `results/hybrid_vector_db/sigmod_candidate_waste_q100r10_main_20260710_002705.csv`
- `results/hybrid_vector_db/sigmod_target_recall_calibration_calib_20260710_123543.csv`
- `results/hybrid_vector_db/sigmod_c4_guidance_memory_filteroff_q25_20260710_summary.json`
- `results/hybrid_vector_db/sigmod_c4_guidance_memory_acorn1_q25_20260710_summary.json`

### D1-D4 Ablation

| Selectivity | Filter | Stock pgvector | D1 guidance | D1+D2 locality | D1+D2+D3 cache/layout | D4 adaptive | D4 route | Recall |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 50% | `popular_ge1000` | 21.63 | 10.84 | 8.31 | 6.80 | 7.42 | `hnsw_d123` | 0.844 |
| 20% | `price_10_to_20` | 21.79 | 13.82 | 8.35 | 8.11 | 6.62 | `hnsw_d123` | 0.821 |
| 10% | `rating5_price_le10` | 24.68 | 13.22 | 8.30 | 8.32 | 8.60 | `hnsw_d123` | 0.842 |
| 5% | `long_review_ge500` | 204.31 | 87.59 | 35.31 | 35.36 | 35.18 | `hnsw_d123` | 0.839 |
| 2% | `grocery_rating5` | 32.58 | 18.23 | 9.38 | 9.61 | 7.81 | `hnsw_d123` | 0.858 |
| 1% | `grocery_helpful` | 37.96 | 21.77 | 11.40 | 11.88 | 9.62 | `hnsw_d123` | 0.843 |
| 0.5% | `helpful_ge20` | 224.34 | 94.66 | 36.07 | 33.85 | 32.92 | `hnsw_d123` | 0.849 |
| 0.2% | `grocery_long500` | 1037.32 | 382.89 | 227.18 | 223.16 | 340.15 | `prefilter_exact` | 0.820 |

Observation:

- D1 reduces invalid candidate validation, but the largest latency drop comes
  when D1 is combined with D2 locality/layout.
- D3 is beneficial on several selective filters, but not uniformly.
- D4 should not be claimed as a universal win yet. The 0.2% case selected
  `prefilter_exact`, but the final q100/r10 latency was slower than D1+D2+D3.
  D4 is currently evidence for adaptive route calibration, not a finished
  end-to-end optimizer.
- Follow-up note: recent Amazon/YFCC/LAION smoke runs show that unconditional
  D1/D2/D3 is not the right abstraction for all selectivities. High-selectivity
  filters can make D1's guidance checks more expensive than stock validation,
  and D3 only helps when reusable predicate state actually avoids repeated
  activation/materialization cost. The next design thread is therefore
  **adaptive admission**: conservatively enable D1, D2, or D3 only when profile
  signals predict positive benefit. This is recorded as future work, not yet as
  a claimed result.

### Candidate Waste

Candidate waste is computed from the q100/r10 raw run as:

```text
candidate_per_valid_result = returned_tuples_mean / final_returned_mean
sql_reject_rate_est = (returned_tuples_mean - final_returned_mean) / returned_tuples_mean
```

Representative results:

| Filter | Stock cand/valid | D1+D2+D3 cand/valid | Stock reject rate | D1+D2+D3 reject rate | Stock latency | D1+D2+D3 latency |
|---|---:|---:|---:|---:|---:|---:|
| `popular_ge1000` | 1.99 | 1.01 | 0.497 | 0.007 | 21.63 | 6.80 |
| `long_review_ge500` | 733.87 | 6.93 | 0.999 | 0.856 | 204.31 | 35.36 |
| `helpful_ge20` | 664.30 | 6.32 | 0.998 | 0.842 | 224.34 | 33.85 |
| `grocery_long500` | 3226.50 | 2.38 | 1.000 | 0.579 | 1037.32 | 223.16 |

This is the strongest current mechanism result: guidance/layout reduce the
number of SQL-invalid candidates that must be returned from the HNSW scan and
validated by PostgreSQL.

### Fixed-Recall Calibration Limit

The SIGMOD plan asked for target recall@10 = 0.9. A q20/r1 calibration sweep
tested:

```text
hnsw.ef_search = 1000
hnsw.max_scan_tuples = 200000, 500000, 1000000, 2000000
hnsw.scan_mem_multiplier = 8, 32
hnsw.iterative_scan = strict_order
```

`ef_search = 1000` is the current pgvector GUC upper bound in this source tree.
Within that bound, changing `max_scan_tuples` and scan memory did not change
recall for these filters. Best observed recall:

| Filter | Best recall | Best config |
|---|---:|---|
| `popular_ge1000` | 0.835 | `ef1000_max200000_mem8` |
| `price_10_to_20` | 0.810 | `ef1000_max200000_mem8` |
| `rating5_price_le10` | 0.835 | `ef1000_max500000_mem32` |
| `long_review_ge500` | 0.825 | `ef1000_max200000_mem32` |
| `grocery_rating5` | 0.860 | `ef1000_max2000000_mem32` |
| `grocery_helpful` | 0.805 | `ef1000_max200000_mem8` |
| `helpful_ge20` | 0.850 | `ef1000_max500000_mem8` |
| `grocery_long500` | 0.800 | `ef1000_max2000000_mem8` |

Interpretation:

- The current Amazon 10M pgvector runs should be described as
  **attainable-recall** results under `ef_search=1000`, not fixed recall@10 =
  0.9 results.
- Running q100/r10 with larger `max_scan_tuples` is not useful unless the
  `ef_search` ceiling or HNSW/index configuration changes.
- If the paper needs a strict 0.9-recall figure, the next experiment must
  either patch the pgvector `ef_search` upper bound and retune, rebuild a
  higher-recall HNSW index, or use a separate exact/SQL-first baseline for that
  figure.

### Page Locality

Existing fixed-candidate q400 results show heap physical clustering reduces the
candidate page footprint but does not by itself make SQL verification faster:

| Table/layout | Candidate limit | Distinct heap pages | Distance-order page runs | Distance verify ms | Page-order verify ms |
|---|---:|---:|---:|---:|---:|
| base 10M heap | 1000 | 963.15 | 972.22 | 82.06 | 82.30 |
| vector-clustered 10M heap | 1000 | 614.28 | 844.05 | 82.28 | 82.58 |

Additional q20 diagnostic on the BFS physical HNSW index layout:

| Index page access | Mean latency | p50 | p95 | Mean index prefetches |
|---|---:|---:|---:|---:|
| off | 21.19 | 20.14 | 40.82 | 0.0 |
| prefetch | 22.16 | 20.17 | 49.85 | 5257.7 |

Interpretation:

- Heap clustering alone reduces page footprint but not validation latency in
  this warm-cache setup.
- Query-time index page prefetch on the BFS-layout index did not help in the
  q20 diagnostic.
- The strongest locality evidence remains the previously measured physical
  HNSW index layout effect: insertion-order index around 32.82 ms versus BFS
  graph-order index around 26.69 ms on the same graph and q400 workload.

### Reuse and Cache Control

The earlier C4 q400 result looked like a cache regression, but the root cause
was an unfair comparison: native mode used `hnsw.filter_strategy=off`, while
`all_memory` and `managed_cache` used `acorn1`. A q25 control isolates this:

| Strategy | Mode | Mean e2e ms | Mean visited tuples | Same ordered IDs |
|---|---|---:|---:|---:|
| off | native | 34.13 | 7016.88 | n/a |
| off | all_memory | 32.54 | 7016.88 | 25/25 |
| off | managed_cache | 33.50 | 7016.88 | 25/25 |
| acorn1 | native | 37.90 | 7016.88 | n/a |
| acorn1 | all_memory | 345.20 | 75749.12 | 18/25 |
| acorn1 | managed_cache | 382.43 | 75749.12 | 18/25 |

Interpretation:

- Cache itself is not the source of the visited-tuple blowup. With
  `filter_strategy=off`, all-memory and managed-cache modes preserve
  `visited_tuples` and result IDs, with small positive/neutral latency impact.
- ACORN1 changes HNSW traversal. To fill the filtered result queue, it crosses
  many non-matching graph nodes; these nodes count as visited graph tuples and
  increase latency.
- Future C4 plots must separate **pure cache reuse** from **ACORN1 traversal**.

## 2026-06-18 PostgreSQL + pgvector Bottleneck Tests

测试脚本：

- `experiments/hybrid_vector_db/scripts/motivation_pgvector_bottleneck_tests.py`
- 完整 SQL/materialization 结果：
  `results/hybrid_vector_db/motivation_pgvector_bottleneck_tests_20260618.csv`
- 修正 profile 后的 HNSW 轻量结果：
  `results/hybrid_vector_db/motivation_pgvector_bottleneck_tests_20260618_hnsw.csv`

测试配置：

```text
table = amazon_grocery_reviews_10m_pgvector
rows = 10,000,000
vector index = HNSW, pgvector 0.8.2
k = 10
hnsw.ef_search = 1000
hnsw.iterative_scan = strict_order
queries = 5
```

### SQL / ID Materialization

| Filter | Selectivity | Case | Rows fetched | Client fetch ms | Executor ms | Plan |
|---|---:|---|---:|---:|---:|---|
| `popular_ge1000` | 50% | `LIMIT 500` | 500 | 0.98 | 0.51 | Seq Scan + Limit |
| `popular_ge1000` | 50% | `LIMIT 50000` | 50,000 | 53.45 | 36.57 | Seq Scan + Limit |
| `popular_ge1000` | 50% | full export | 5,031,984 | 4106.75 | 3173.12 | Seq Scan |
| `popular_ge1000` | 50% | full export order by id | 5,031,984 | 5665.06 | 4498.96 | primary-key Index Scan |
| `price_10_to_20` | 20% | full export | 2,189,009 | 4009.32 | 3476.36 | Seq Scan |
| `helpful_ge20` | 0.5% | full export | 60,689 | 200.97 | 225.60 | btree Index Scan |
| `grocery_long500` | 0.2% | full export | 21,317 | 78.15 | 73.29 | composite btree Index Scan |

观察：

- 宽 filter 的瓶颈不是谓词计算本身，而是大量 tuple/id 的 executor scan +
  client materialization。50% filter 的 bounded fetch 可以在 1 ms 左右返回 500
  个 id，但 full export 需要 4-5 秒。
- `ORDER BY id` 不一定有利。50% filter 走 primary-key index scan 后仍需检查约
  497 万个 false tuple，且随机/索引路径读更多页面，延迟从 4.1s 增加到 5.7s。
- 窄 filter 如果有合适 btree/composite btree，SQL 本身并不慢：
  `grocery_long500` 全量 21,317 个 id 约 78 ms。这说明“SQL 永远是瓶颈”需要分
  场景：宽 filter 的 full materialization 是瓶颈；窄 filter 的瓶颈会转向
  filtered HNSW 的候选验证。

### pgvector Filtered HNSW

| Filter | Selectivity | Mean latency ms | Executor ms | HNSW vector ms | Mean rejected candidates | Mean qual calls | Mean shared read blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| no filter | 100% | 30.44 | 9.09 | 9.00 | 0 | 0 | 1,691.6 |
| `popular_ge1000` | 50% | 23.17 | 6.48 | 6.43 | 10.4 | 20.4 | 362.8 |
| `price_10_to_20` | 20% | 22.73 | 5.95 | 5.88 | 34.8 | 44.8 | 30.8 |
| `helpful_ge20` | 0.5% | 158.55 | 140.99 | 87.98 | 15,916.4 | 15,926.4 | 75,784.4 |
| `grocery_long500` | 0.2% | 197.46 | 193.86 | 100.45 | 27,726.4 | 27,734.6 | 106,961.6 |

观察：

- 50% 和 20% filter 下，pgvector filtered HNSW 与 no-filter baseline 接近。
  此时只需要验证几十个 tuple 就能凑够 top-k。
- 0.5% 和 0.2% filter 下，pgvector 仍然按全局 HNSW 距离顺序产生候选，然后
  由 PostgreSQL executor 对 heap tuple 做 filter。为了返回 10 个结果，平均
  要拒绝约 1.6 万到 2.8 万个候选。
- `qual_ms` 本身很小，说明标量表达式计算不是主要成本。真正贵的是“候选 tuple
  验证路径”：HNSW traversal + heap tuple fetch + visibility/filter check +
  buffer/page 访问。
- 稀疏 filter 的 shared read blocks 明显上升，说明 filtered HNSW 的候选验证
  具有强随机 I/O 或弱 locality 特征。这与 page-aware pgvector 实验中的观察一致。

### 一个重要查询形态陷阱

下面这种写法没有走 HNSW，而是退化成全表 exact sort：

```sql
WITH q AS (
  SELECT embedding
  FROM amazon_grocery_reviews_10m_pgvector
  WHERE id = 0
)
SELECT t.id
FROM amazon_grocery_reviews_10m_pgvector t, q
ORDER BY t.embedding <-> q.embedding
LIMIT 10;
```

实测约 5.34s，计划为 `Seq Scan + top-N heapsort`。原因是 pgvector 的 index
path 对 `ORDER BY embedding <-> constant/parameter` 形式更敏感；当 RHS 是 join
变量时，优化器无法使用普通 HNSW order-by index scan。实验脚本因此使用参数化
向量字面量。

### 对研究方向的直接启发

PostgreSQL + pgvector 场景下值得优化的不是单一的 “SQL latency”，而是三类不同
瓶颈：

1. **宽 filter 的 full materialization**：如果系统需要把数百万 id 从 PostgreSQL
   导出给外部 ANN/allow-list，client/server 边界和 executor scan 会吞掉优化空间。
   方向是 server-side bitmap、shared-memory allow-list、compressed idset、或者
   把 vector search 下推到 PostgreSQL 内部，避免 id 大搬运。
2. **稀疏 filter 的 candidate validation**：pgvector filtered HNSW 对稀疏谓词
   会验证大量失败候选。方向是 filter-aware graph traversal、per-page/per-cluster
   metadata pruning、posting-list/bitmap guided HNSW、或者 adaptive ef/max_scan。
3. **heap locality / page locality**：HNSW 距离顺序与 heap 物理顺序弱相关，导致
   大量随机 heap page 访问。方向是 vector-aware heap clustering、page-aware
   materialization、candidate reorder + stable top-k restore、或把常用标量列嵌入
   index tuple 以减少 heap fetch。

因此，下一步系统优化可以从 “减少 SQL 扫描时间” 转为更精确的问题：

- 对宽 filter：如何避免导出百万级 allow-list？
- 对窄 filter：如何避免 pgvector 对数万失败候选做 heap validation？
- 对混合 workload：如何在 pre-filter、post-filter、parallel filter+ANN 之间做
  cost-based routing？
