# SQLens SIGMOD 实验计划

本文档用于指导 SQLens 的论文级实验执行。目标不是把所有已有结果都塞进
Evaluation，而是围绕 SIGMOD 系统论文的核心问题组织证据：SQLens 是否在
保留 PostgreSQL/\pgvector SQL 执行边界的前提下，稳定降低 filtered vector
search 的端到端代价。

## 论文主张

SQLens 是一个面向 PostgreSQL + \pgvector 的 SQL-native filtered vector
search 优化方案。它不绕开 PostgreSQL 的 SQL、MVCC、权限与最终验证语义，
而是在 \pgvector HNSW 扫描路径中引入 SQL 派生的可见性指导、局部性友好的
执行路径，以及可复用的谓词状态，从而减少无效候选、随机页访问和重复谓词
计算带来的 DBMS 执行开销。

Evaluation 需要支撑以下四个主张：

1. 在固定 recall 或可比较 recall 下，SQLens 相比 stock PostgreSQL +
   \pgvector 显著降低端到端 FVS 延迟，并改善 throughput。
2. SQLens 的 D1/D2/D3 组件分别贡献收益：D1 减少 SQL-invalid candidate，
   D2 改善物理访问局部性，D3 复用热点谓词状态。
3. SQLens 的收益不仅出现在单个数据集上，而是在至少两个 10M 级真实工作负载
   上成立；如果 LAION-25M 结果成熟，可作为更大规模补充。
4. SQLens 的额外开销可控，包括 guidance metadata、cache footprint、构建
   时间、正确性验证、更新/失效处理和并发下的尾延迟。

## Motivation 与 Evaluation 的边界

### Motivation 中应该放什么

Motivation 只负责证明问题真实存在，以及为什么问题来自 PostgreSQL +
\pgvector 的 SQL-native 执行路径。它可以包含以下证据：

- \pgvector HNSW 返回 TID 后，SQL/MVCC/权限/策略验证仍然在 PostgreSQL
  executor 中发生。
- 在选择性过滤条件下，stock \pgvector 会向 PostgreSQL 返回大量 SQL-invalid
  candidates，造成无效验证。
- HNSW 的距离顺序与 heap/index page 物理布局不一致，导致 page locality 差。
- 真实工作负载中，SQL filter/predicate fragment 的重复概率明显高于 query
  vector 精确重复概率，因此可复用状态应围绕 predicate/visibility 而非最终
  top-k answer。

这些 observation 已经在 Motivation 的图中出现。因此 Evaluation 中不再重复
candidate waste、page locality、filter reuse 这三类“问题存在性”图。

### Evaluation 中应该放什么

Evaluation 只负责证明 SQLens 是否解决问题。它应包含：

- 端到端 fixed-recall 或 matched-recall 性能对比。
- Recall-latency 和 throughput-recall frontier。
- D1/D2/D3 组件消融。
- D3 cache/reuse 的收益、正确性与开销。
- 更新、失效、并发、内存和存储 overhead。
- 与 vector-native payload-filter 系统的边界对比。

Evaluation 中可以在文字或表格里引用 motivation 指标解释结果，但不再单独放
candidate waste/locality/reuse 的机制图。机制证据的主位置是 Motivation。

### 本轮明确不做的 Evaluation 内容

- 不单独做 `Complex SQL / Routing` 小节。
- 不把 routing 作为核心实验主张。若实现中有保守 route calibration，可放在
  Implementation 或 appendix 中作为工程细节，不作为主线结果。
- 不在 Evaluation 里重复 Motivation 已经展示过的 observation 图。
- 不使用人工构造的高重复 replay 来证明 D3，除非明确标为 stress/control。

## Evaluation 章节建议结构

推荐把 Evaluation 收敛为以下结构：

1. `Experimental Setup`
   - 硬件、PostgreSQL、\pgvector commit、SQLens commit、索引大小、表大小。
   - 数据集、向量维度、行数、谓词来源、query 来源、ground truth 计算方式。
   - 所有方法使用相同 query vectors、filter predicates、`k=10`。

2. `End-to-End Performance`
   - 主图：stock \pgvector vs SQLens-D1 vs SQLens-D1+D2 vs SQLens-D1+D2+D3。
   - 主数据集：Amazon Reviews 10M。
   - 辅助数据集：YFCC-10M；LAION-25M 若结果成熟，可放入同一图的分面或独立图。
   - 报告 mean、p50、p95、std、recall@10。

3. `Recall-Latency-Throughput Frontier`
   - 必须包含 recall-latency 与 throughput-recall 两张 frontier 图。
   - frontier 不按 selectivity 分组，而是使用尽可能完整的真实 workload。
   - 每个数据集至少 10,000 个 request，覆盖 recall 0.7 到接近 1.0 的至少
     10 个配置点。
   - Amazon 与 YFCC 是最低要求；LAION-25M 若补齐 10K workload，可作为第三组。

4. `Component Ablation`
   - 只比较 stock、D1、D1+D2、D1+D2+D3。
   - 用表格汇总每个组件带来的 latency/recall/p95/overhead 变化。
   - 可以报告 returned TIDs、valid results、guidance skips、cache hits 等
     诊断列，但不要再画 candidate waste/page locality 机制图。

5. `Overheads and Robustness`
   - metadata size、cache size、build time、activation time。
   - update/invalidation cost。
   - 1/4/8/16 clients 的 throughput 与 p95/p99。
   - cold-cache sensitivity 可放 appendix，主文保留 warm-cache。

<!-- 6. `External Payload-Filter Boundary`
   - 只用于说明边界：Milvus/Weaviate/Qdrant/FAISS/HNSWlib 能很好地处理
     payload-compatible filter，但不能自然替代 PostgreSQL 中 join、ACL、
     MVCC、temporal policy 等 SQL-defined visibility。
   - 不要把这一节写成“DBMS 全面击败 vector-native system”。重点是语义边界
     与系统适用范围。 -->

## 图表组织

主文建议保留以下图表：

| 位置 | 形式 | 内容 | 目的 |
|---|---|---|---|
| Table 1 | 表格 | 数据集、规模、维度、谓词、query 来源 | 实验可复现性 |
| Fig. E1 | 折线/分面图 | End-to-end latency vs selectivity 或 predicate group | 主性能结果 |
| Fig. E2 | 两行图 | Recall-latency frontier 与 throughput-recall frontier | ANN 标准视角 |
| Table E1 | 表格 | D1/D2/D3 ablation，含 recall/p95/overhead | 组件贡献 |
| Table E2 | 表格 | Filter repeat、query repeat、cache hit、cache-control latency | D3 复用证据 |
| Table E3 | 表格 | metadata/cache/build/update/concurrency overhead | 可部署性 |
| Table E4 | 表格 | PostgreSQL vs vector-native payload-filter boundary | 适用边界 |

不建议保留在主文 Evaluation 中的图：

- Candidate waste 机制图：放 Motivation。
- Page locality 机制图：放 Motivation。
- Filter reuse 折线图：改为 Motivation 表格或 Evaluation 的 cache 表。
- Complex SQL/routing 图：不作为主线实验。

## 数据集计划

主文至少需要两个 10M 级真实工作负载；第三个成熟后再加入。

| 数据集 | 目标规模 | 本地状态 | 谓词角色 | 论文角色 |
|---|---:|---|---|---|
| Amazon Reviews 2023 Grocery | 10M | 已加载 PostgreSQL + \pgvector | category、price、rating、helpfulness、review length | 主 SQL-native 数据集 |
| YFCC | 10M | 已加载 base/query，HNSW 与 guidance metadata 已构建 | public tag filters 与 filtered-ANNS workload | 跨论文可比的第二主数据集 |
| LAION | 25M | 已加载 25M image rows，已有 caption/width 派生谓词与 q20 结果 | caption label、width range、hybrid label/range | 大规模 image-text 扩展数据集 |

## Baselines

| Baseline | 是否主文必需 | 使用场景 | 注意事项 |
|---|---|---|---|
| Stock PostgreSQL + \pgvector HNSW | 必需 | 所有 SQLens 对比 | 调优 `ef_search`、iterative scan、scan budget |
| SQL-first exact | 必需 | ground truth 与 exact baseline | `WHERE predicate` 后 exact vector ranking |
| SQLens-D1 | 必需 | predicate guidance | 与 stock 使用相同 SQL final validation |
| SQLens-D1+D2 | 必需 | locality/layout | 不把重排/构建时间混入单查询 latency |
| SQLens-D1+D2+D3 | 必需 | cache/reusable state | 正确性必须由 PostgreSQL final recheck 保证 |
<!-- | FAISS/HNSWlib allow-list | 可选 | payload-compatible 上界 | 说明不是 SQL-native DBMS 路径 |
| Milvus/Weaviate/Qdrant | 可选 | 外部边界 | 只比较 payload filter，不比较 join/ACL/RLS 除非计入 denormalization | -->

不再把 D4/routing 作为主文 baseline。若保留实现，可作为保守工程策略或
appendix 结果。

## 统一报告标准

每个主文结果必须满足：

- 使用相同 query vectors 与 filter predicates。
- 使用 exact SQL-valid top-k 作为 ground truth。
- 主目标 recall@10 建议为 0.9；0.95 作为 sensitivity。
- 如果固定 0.9 难以达到，必须明确写为 matched-recall 或 attainable-recall，
  并报告每个点的 recall。
- 报告 mean latency、p50、p95、std、recall、throughput。
- 主文使用 warm-cache；cold-cache 放 appendix。
- 保留 PostgreSQL final SQL/MVCC/policy validation。
- 记录 PostgreSQL 版本、\pgvector commit、SQLens commit、HNSW 参数、索引
  大小、表大小、硬件、内存、存储、命令行。
- 所有 paper-facing 图表应来自可追踪 CSV/JSON，不能手工改数。

## 优先级队列

### P0：正式补齐 10K frontier

目标：

- 在 Amazon 与 YFCC 上生成标准 ANN 视角的 recall-latency 和
  throughput-recall frontier。
- 每个数据集至少 10,000 个 request，至少 10 个配置点，覆盖 recall 0.7 到
  接近 1.0。

YFCC 当前状态：

- 已有 10,000 个真实 public-query request 文件：
  `results/hybrid_vector_db/yfcc10m_full_workload_requests_10000_20260714.csv`
- 已完成 q100 calibration，并确认 PostgreSQL SQL-first exact 应作为
  self-ground-truth，因为官方 YFCC GT 与 PostgreSQL exact 存在少量 tie/order
  差异。
- 正式 10K 分片运行未完整完成，需要继续 resume。

YFCC 10K resume 命令模板：

```bash
python3 experiments/hybrid_vector_db/scripts/yfcc_full_workload_recall_sweep.py \
  --out-prefix v2_full_q10000_10cfg_w0_20260715 \
  --requests 10000 \
  --methods stock bloom exact \
  --ef-search-values 1,2,4,8,16,32,128,512,2000,5000 \
  --max-scan-tuples-values 200000 \
  --selected-queries-in results/hybrid_vector_db/yfcc10m_full_workload_requests_10000_20260714.csv \
  --num-workers 8 \
  --worker-id 0 \
  --warmup-requests 10 \
  --progress-requests 250 \
  --statement-timeout-ms 180000 \
  --resume \
  --skip-function-ddl
```

将 `worker-id` 与 `out-prefix` 改为 `0..7`。每个 worker 完整行数应为
`1250 * (10 stock + 10 bloom + 1 exact) = 26250`。

完成后合并：

```bash
python3 experiments/hybrid_vector_db/scripts/merge_full_workload_shards.py \
  --inputs results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w0_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w1_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w2_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w3_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w4_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w5_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w6_20260715.csv \
           results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w7_20260715.csv \
  --out results/hybrid_vector_db/yfcc_v2_full_q10000_10cfg_merged_20260715.csv \
  --summary-out results/hybrid_vector_db/yfcc_v2_full_q10000_10cfg_merged_20260715_summary.csv
```

用 PostgreSQL exact endpoint 重算 recall：

```bash
python3 experiments/hybrid_vector_db/scripts/recompute_full_workload_recall_from_exact.py \
  --raw results/hybrid_vector_db/yfcc_v2_full_q10000_10cfg_merged_20260715.csv \
  --out results/hybrid_vector_db/yfcc_v2_full_q10000_10cfg_sqltruth_20260715.csv \
  --summary-out results/hybrid_vector_db/yfcc_v2_full_q10000_10cfg_sqltruth_summary_20260715.csv
```

注意：

- 8 个 worker 并发运行时，latency 是 8-client closed-loop workload 下的
  per-query latency。
- throughput 图应使用并发总吞吐，例如近似为
  `8 * 1000 / latency_mean_ms`，并在图注中说明。
- 如果需要单客户端 latency frontier，应单独顺序运行 10K，不能混用并发结果。

### P0：Amazon 主性能与 ablation 整理

目标：

- 保留 Amazon 10M 作为第一条完整 SQLens 证据线。
- 重新组织为端到端主图 + ablation 表，而不是散放机制图。

已可用结果：

- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705.csv`
- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705_table.csv`
- `results/hybrid_vector_db/sigmod_target_recall_calibration_calib_20260710_123543.csv`

待确认：

- 当前 Amazon 结果是 \pgvector 配置下的 attainable recall，而非严格固定
  recall。若主文需要 fixed recall@10=0.9，应重新调高或移除 `ef_search`
  ceiling 后校准。
- 主文表格中每个方法必须并列报告 recall，避免只展示 latency。

| Sel. | Filter | Stock | D1 | D1+D2 | D1+D2+D3  | Speedup | Recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50% | popular_ge1000 | 16.73 | 16.48 | 12.58 | 10.58 | 1.58x | 0.844 |
| 45% | popular_ge1340 | 17.50 | 16.74 | 12.86 | 10.98 | 1.59x | 0.844 |
| 40% | popular_ge1780 | 18.41 | 17.87 | 13.33 | 11.91 | 1.54x | 0.846 |
| 35% | popular_ge2428 | 19.74 | 18.98 | 13.99 | 11.48 | 1.72x | 0.846 |
| 30% | popular_ge3284 | 20.12 | 19.69 | 14.71 | 12.39 | 1.62x | 0.847 |
| 25% | popular_ge4559 | 22.83 | 19.25 | 14.97 | 12.84 | 1.78x | 0.847 |
| 20% | price_10_to_20 | 22.71 | 19.79 | 15.03 | 13.76 | 1.65x | 0.821 |
| 15% | popular_ge10066 | 22.03 | 21.05 | 15.69 | 12.76 | 1.73x | 0.840 |
| 10% | rating5_price_le10 | 31.42 | 24.54 | 19.06 | 18.61 | 1.69x | 0.852 |
| 5% | long_review_ge500 | 33.12 | 27.22 | 21.22 | 19.95 | 1.66x | 0.839 |
| 2% | grocery_rating5 | 32.58 | 28.23 | 23.81 | 19.11 | 1.70x | 0.861 |
| 1% | grocery_helpful | 40.41 | 36.59 | 33.09 | 29.96 | 1.34x | 0.843 |
| 0.5% | helpful_ge20 | 155.73 | 123.27 | 77.55 | 76.52 | 2.03x | 0.849 |
| 0.2% | grocery_long500 | 225.54 | 156.96 | 122.96 | 112.81 | 1.99x | 0.819 |

- YFCC
| Sel. | Actual | Filter | Stock | D1 | D1+D2 | D1+D2+D3 | Speedup | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 53.0273% | tagor_23_29 | 72.31 | 71.11 | 38.65 | 28.34 | 2.55x | 0.990 |
| 45% | 44.7679% | tagor_23_89 | 63.55 | 61.20 | 29.70 | 18.34 | 3.47x | 0.991 |
| 40% | 39.9688% | tagor_23_3 | 62.32 | 58.81 | 29.27 | 17.75 | 3.51x | 0.990 |
| 35% | 34.9893% | tagor_23_90 | 72.94 | 67.46 | 34.70 | 22.66 | 3.22x | 0.987 |
| 30% | 30.1632% | tagor_29_8 | 61.84 | 58.15 | 29.56 | 17.63 | 3.51x | 0.988 |
| 25% | 25.0068% | tagor_89_18 | 61.64 | 57.81 | 29.47 | 17.66 | 3.49x | 0.996 |
| 20% | 20.0005% | tagor_8_28 | 61.48 | 57.38 | 28.98 | 17.67 | 3.48x | 0.986 |
| 15% | 15.0022% | tagor_20_12 | 60.89 | 57.48 | 29.68 | 17.69 | 3.44x | 0.987 |
| 10% | 10.0022% | tagor_3_515 | 62.39 | 58.43 | 30.02 | 17.98 | 3.47x | 0.984 |
| 5% | 4.9908% | tagor_24 | 96.07 | 86.48 | 46.58 | 30.69 | 3.78x | 0.979 |
| 2% | 2.0001% | tagor_36_414 | 104.94 | 105.60 | 81.63 | 47.11 | 2.23x | 0.972 |
| 1% | 1.0001% | tagor_430_3076 | 140.31 | 107.97 | 83.29 | 59.91 | 2.34x | 0.959 |
| 0.5% | 0.5000% | tagor_902 | 182.69 | 134.59 | 80.88 | 59.10 | 3.09x | 0.957 |
| 0.2% | 0.2002% | tagor_990 | 191.03 | 146.25 | 113.29 | 84.05 | 2.27x | 0.960 |

- LAION-25M
| Sel. | Actual | Filter | Stock | D1+D2+D3 | Speedup | Recall |
| ---- | ------ | ------ | ----: | ----: | ----: | ----: |
| 50% | 49.767% | labelor_top70 | 66.43 | 46.50 | 1.43x | 0.932 |
| 45% | 45.458% | labelor_top55 | 66.46 | 46.51 | 1.43x | 0.929 |
| 40% | 39.991% | labelor_top40 | 66.45 | 46.49 | 1.43x | 0.927 |
| 35% | 36.019% | labelor_top30 | 66.44 | 46.49 | 1.43x | 0.925 |
| 30% | 29.783% | labelor_top20 | 76.75 | 57.64 | 1.33x | 0.941 |
| 25% | 24.642% | labelor_top14 | 76.72 | 55.31 | 1.39x | 0.943 |
| 20% | 19.590% | labelor_top9 | 76.77 | 55.72 | 1.38x | 0.934 |
| 15% | 15.423% | labelor_top6 | 82.52 | 64.98 | 1.27x | 0.914 |
| 10% | 9.025% | labelor_top3 | 82.89 | 65.80 | 1.26x | 0.878 |
| 5% | 3.964% | label_175 | 139.16 | 96.07 | 1.45x | 0.842 |
| 2% | 2.127% | label_79 | 170.20 | 113.85 | 1.49x | 0.808 |
| 1% | 1.000% | label_2039 | 203.26 | 132.76 | 1.53x | 0.616 |
| 0.5% | 0.501% | label_1432 | 300.48 | 164.26 | 1.82x | 0.780 |
| 0.2% | 0.200% | label_281 | 393.97 | 230.23 | 1.71x | 0.770 |

### P0：D3 cache/reuse 结果整理

目标：

- 用表格说明真实 workload 中 filter repeat 高于 query repeat。
- 用 cache-control 实验证明 reusable predicate state 能在保持 ordered result
  IDs 一致的情况下减少 executor 压力。

当前可用 q400 cache-control 结果：

- Artifact:
  `results/hybrid_vector_db/sigmod_c4_guidance_memory_filteroff_q400_20260713_summary.json`
- `native`: mean end-to-end 38.51ms，p95 72.78ms，visited tuples 7589.0，
  returned tuples 228.3。
- `all_memory`: mean end-to-end 36.48ms，p95 64.97ms，visited tuples 7589.0，
  returned tuples 11.7，guidance skip rate 94.87%，400/400 ordered-id match。
- `managed_cache`: mean end-to-end 36.75ms，p95 67.30ms，visited tuples
  7589.0，returned tuples 11.7，guidance skip rate 94.87%，400/400 ordered-id
  match。
- Prebuilt fragments: 15 fragments，8.21MiB SSD payload，14.8s prebuild，
  1MiB managed cache budget。

写作原则：

- ACORN1 traversal 是诊断实验，不应与纯 cache reuse 混在一起。
- D3 主张应强调“复用 predicate/visibility state”，不是缓存最终结果。

### P0/P1：Adaptive admission 待深入

当前 raw D1/D2/D3 ablation 暴露一个需要单独处理的问题：在高
selectivity 或 candidate waste 很低的 filter 上，无条件启用 D1/D2/D3
不一定单调变好。典型现象是：

- D1 的 membership/guidance check 成本可能超过 stock pgvector 本身的
  SQL validation 成本。
- D2 的物理布局收益依赖 workload 的 index/heap locality；并不是所有
  label/tag filter 都能从 BFS 或 page-aware route 获益。
- D3 的收益来自可复用 predicate/visibility state；如果 D1/D2 baseline
  已经被 warmup、fragment cache 或 active guidance 污染，D3 的边际收益会被
  提前算进 baseline。

后续需要把 adaptive admission 作为独立机制分析，而不是现在强行把 raw
D1/D2/D3 表改成单调结论。候选 route：

- D1 admission：只在预测 invalid candidate waste 或 validation fanout 足够高
  时启用 predicate guidance，否则保留 stock route。
- D2 admission：只在预测 index/heap locality 可改善时切到 physical layout /
  page-aware route。
- D3 admission：只在 filter repeat、fragment cache hit、composed exact state
  或预算下调能覆盖 activation 开销时启用 reusable state。

需要补的实验：

- 在 Amazon/YFCC/LAION 上记录每个 query 的 selectivity、returned tuples、
  guidance checks/skips、visited tuples、idx/heap blocks、activation/cache hit。
- 用这些 profile 学一个保守 admission rule，并与 oracle best-of-route 做
  gap 分析。
- 最终主表可以报告 `Stock`、`Raw SQLens`、`SQLens + adaptive admission`，
  但在 admission 验证完成前，不把它作为已完成贡献。

### P1：LAION-25M 是否进入主文

目标：

- 如果 LAION-25M 能补齐 10K frontier 与 SQL-exact/approx recall，则作为
  第三个数据集进入主文。
- 如果只能保留 q20/q640 小规模结果，则放 appendix 或作为 scalability
  feasibility，不作为核心 4/5 分证据。

当前结果：

- `results/hybrid_vector_db/laion25m_selected_filters_q200_20260714.csv`
- `results/hybrid_vector_db/laion25m_truth_all_q20_20260714.csv`
- `results/hybrid_vector_db/laion25m_pgvector_all_q20_r3_20260714_with_recall.csv`
- `results/hybrid_vector_db/laion25m_pgvector_all_q20_r3_20260714_summary_with_recall_e2e.csv`
- `results/hybrid_vector_db/laion25m_pgvector_range_q20_r5_20260714_summary_e2e.csv`
- `results/hybrid_vector_db/laion25m_pgvector_range_q20_r1_ef10000_mts5000000_summary_e2e.csv`

### P1：Overhead 与 Robustness

必须补齐的主文/appendix 结果：

- guidance metadata size、index size、table size。
- fragment build time、cache activation time。
- update 后 invalidation/rebuild 成本。
- 1/4/8/16 client 并发下 mean/p95/p99/throughput。
- cache budget sensitivity，例如 1MiB、8MiB、64MiB。
- correctness mismatch 必须为 0；lossy summaries 可以有 false positives，
  但不能产生 false negatives。

### P2：External payload-filter boundary

目标：

- 给审稿人一个清晰边界：vector-native payload filter 很强，但它不是
  PostgreSQL SQL execution 的替代品。

当前可用结果：

- MS MARCO 1M Qdrant/PostgreSQL control：
  - `research/late_bound_visibility/results/msmarco_security_killtest_1m_q100_20260713.csv`
  - `research/late_bound_visibility/results/msmarco_security_killtest_1m_q100_20260713_summary.csv`
  - `research/late_bound_visibility/results/msmarco_security_killtest_1m_q100_20260713_faiss.csv`
- Enron 50K visibility control：
  - `research/late_bound_visibility/results/enron_visibility_benchmark_q100_20260713.csv`
  - `research/late_bound_visibility/results/enron_visibility_benchmark_q100_20260713_summary.csv`

写作原则：

- 不声称 SQLens/DBMS 在 payload-only 场景全面快于 vector-native system。
- 强调 SQL-defined visibility、join、ACL、MVCC、temporal constraints 的语义
  边界。

## 当前需要保留的结果文件

Amazon 10M：

- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705.csv`
- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705_table.csv`
- `results/hybrid_vector_db/sigmod_candidate_waste_q100r10_main_20260710_002705.csv`
- `results/hybrid_vector_db/sigmod_target_recall_calibration_calib_20260710_123543.csv`
- `results/hybrid_vector_db/sigmod_c4_guidance_memory_filteroff_q400_20260713_summary.json`

YFCC-10M：

- `results/hybrid_vector_db/yfcc10m_pgvector_manifest_20260713.csv`
- `results/hybrid_vector_db/yfcc10m_full_workload_requests_10000_20260714.csv`
- `results/hybrid_vector_db/q100_v2_calib_q100_20260715.csv`
- `results/hybrid_vector_db/q100_v2_calib_q100_20260715_sqltruth_summary.csv`
- `results/hybrid_vector_db/q100_v2_selected_configs_sqltruth_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w0_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w1_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w2_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w3_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w4_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w5_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w6_20260715.csv`
- `results/hybrid_vector_db/q10000_v2_full_q10000_10cfg_w7_20260715.csv`

LAION-25M：

- `results/hybrid_vector_db/laion25m_selected_filters_q200_20260714.csv`
- `results/hybrid_vector_db/laion25m_truth_all_q20_20260714.csv`
- `results/hybrid_vector_db/laion25m_pgvector_all_q20_r3_20260714_with_recall.csv`
- `results/hybrid_vector_db/laion25m_pgvector_all_q20_r3_20260714_summary_with_recall_e2e.csv`

Motivation 机制证据：

- `research/results/page_locality_multidataset_q100_c1000_20260713.csv`
- `research/results/page_locality_multidataset_q100_c1000_20260713_summary.csv`
- `research/results/page_locality_reordered_multidataset_q100_c1000_20260713.csv`
- `research/results/page_locality_reordered_multidataset_q100_c1000_20260713_summary.csv`

脚本：

- `experiments/hybrid_vector_db/scripts/yfcc_full_workload_recall_sweep.py`
- `experiments/hybrid_vector_db/scripts/merge_full_workload_shards.py`
- `experiments/hybrid_vector_db/scripts/recompute_full_workload_recall_from_exact.py`
- `experiments/hybrid_vector_db/scripts/select_full_workload_configs.py`
- `paper/scripts/plot_evaluation.py`

## 对另一个实验 agent 的执行提醒

1. 先完成 YFCC 10K formal run，不要把 partial shard 结果画进论文。
2. 画 frontier 前，必须用 PostgreSQL exact endpoint 重新计算 recall。
3. Evaluation 不再新增 candidate waste/page locality 机制图；这些图留在
   Motivation。
4. 不再推进 Complex SQL/Routing 主线实验。
5. 每张主文图都必须同时报告或能追踪到 recall，否则不能作为性能 claim。
6. 如果发现 SQLens 与 stock recall 不匹配，优先画 frontier 或 matched-recall
   点，不要只比较 latency。
