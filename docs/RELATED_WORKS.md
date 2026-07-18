# Related Works

## Filtered Vector Search

Filtered vector search combines nearest-neighbor search with scalar predicates:

```sql
SELECT *
FROM items
WHERE price <= 10 AND rating = 5
ORDER BY embedding <-> :query_embedding
LIMIT 10;
```

Common execution styles:

- **Post-filtering**: search the vector index globally, then apply predicates to
  returned candidates.
- **Pre-filtering with allow-list**: evaluate scalar predicates first and pass
  the matching ids into vector search as an allow-list.
- **In-filtering / integrated filtering**: apply predicate constraints inside
  the vector engine during traversal.

## Weaviate-Style Pre-Filtering

The current `pre_filter_allow_list` benchmark follows the production-style
allow-list idea:

```text
scalar index / SQL predicate -> allow-list -> existing HNSW graph search
```

This is different from building a temporary HNSW per query. HNSW traversal still
walks the existing graph; the allow-list controls which ids can enter the final
result set.

Why this matters:

- Temporary HNSW build cost dominates latency and is not representative of most
  filtered HNSW systems.
- Allow-list search avoids rebuild cost, but recall still depends on HNSW graph
  traversal, filter selectivity, attribute-vector correlation, and `efSearch`.

## Split vs Integrated Systems

This repository currently studies split-engine behavior:

```text
DuckDB or PostgreSQL scalar predicate branch + FAISS vector branch
```

DuckDB is the primary scalar engine because it is in-process and columnar. That
makes it a fairer comparison with in-memory FAISS than PostgreSQL full id export
through a client/server boundary.

PostgreSQL remains useful as a DBMS baseline, especially for checking indexed
predicate latency and understanding materialization costs.

## Query Dependency

SQL and vector sub-queries can be weakly or strongly dependent.

Weak dependency:

```text
The vector query is fixed, and SQL only filters the final candidate set.
```

Strong dependency:

```text
The SQL result changes the vector query, the vector search scope, grouping,
aggregation, or downstream reranking behavior.
```

Selectivity is related to execution dependency but not identical to semantic
dependency. A 1% filter often makes post-filtering harder because global ANN
must overfetch aggressively, but the vector and SQL sub-queries are only
semantically dependent if one changes what the other is supposed to search.

## Compass Positioning

Compass-style systems focus on filtered ANN execution inside a single query.
They coordinate vector traversal and scalar candidates to avoid failures caused
by filter-disconnected graph neighborhoods.

The current project uses Compass as motivation, but the active experiment is
narrower and more concrete: measure split SQL/vector execution under real
Amazon Review metadata filters, then compare parallel join against allow-list
pre-filtering.

## Complex SQL Patterns in Hybrid Search

本节记录 2026-06-16 对 “Hybrid Search 中复杂 SQL 到底能复杂到什么程度” 的补充调研。

### 论文里的复杂度边界

不同论文对 complex SQL 的定义不一样：

- **Filtered ANN / General Filtered Search 论文**通常把复杂度集中在 predicate 上，而不是完整 SQL skeleton。ACORN 的动机是传统方法常只支持少量 equality predicates，而实际 mixed-modality 查询需要更广泛的 predicate sets 和 query semantics。Compass 则明确把目标扩到 arbitrary conjunctions、disjunctions、range predicates，并强调 DBMS compatibility。
- **VectorSQL / Text2VectorSQL 工作**把复杂度推进到更通用 SQL：join、subquery、aggregation、comparison、semantic column embedding、不同 DB backend。Text2VectorSQL/VectorSQLBench 覆盖 PostgreSQL、ClickHouse、SQLite，以及 BIRD、Spider、arXiv、Wikipedia 数据源，说明复杂 hybrid query 不只是 `WHERE metadata AND ORDER BY vector`。
- **DBMS-oriented FVS 工作**提醒：SQL 语法不一定复杂才会慢。即使只是一个 filter，如果导致 page access、heap fetch、data retrieval、filter operation 成本很高，也会主导端到端 latency。

### 工业场景里最常见的复杂 SQL 形态

#### 1. ACL / tenant / row-level visibility

企业 RAG、知识库、客服搜索最常见。filter 往往来自 join，而不是单表 metadata：

```sql
SELECT c.chunk_id, c.doc_id
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
JOIN document_acl acl ON acl.doc_id = d.doc_id
WHERE d.tenant_id = :tenant_id
  AND acl.user_id = :user_id
  AND acl.can_read = true
  AND d.deleted_at IS NULL
  AND d.version_state = 'published'
ORDER BY c.embedding <=> :q
LIMIT 20;
```

复杂点：

- ACL 可能包括 user、group、role、organization、deny-overrides-allow。
- filter selectivity 随用户变化很大。
- 同一 tenant/group 的 filter result 可能高度复用，适合 predicate/result cache。

#### 2. 时间、版本、新鲜度过滤

新闻、日志、法律/合规、企业知识库：

```sql
SELECT chunk_id, doc_id
FROM chunks
WHERE tenant_id = :tenant
  AND valid_from <= :query_time
  AND (valid_to IS NULL OR valid_to > :query_time)
  AND ingestion_time >= :query_time - interval '30 days'
  AND source_reliability >= 0.8
ORDER BY embedding <=> :q
LIMIT 50;
```

复杂点：

- 时间谓词可能与向量相关性强，也可能完全无关。
- 版本/删除/可见性约束通常不能近似。
- freshness 常和 ACL、source、dedup 一起出现。

#### 3. 电商 / marketplace faceted search

商品、广告、招聘、推荐系统：

```sql
SELECT item_id
FROM products
WHERE category_id IN (:categories)
  AND brand_id = ANY(:brands)
  AND price BETWEEN :min_price AND :max_price
  AND rating >= 4.2
  AND inventory > 0
  AND seller_status = 'active'
ORDER BY image_embedding <=> :image_query
LIMIT 100;
```

复杂点：

- 多属性 AND/OR/range/array 组合。
- 用户 facet 每次变化，cache 不一定稳定。
- 后面通常还要 business rules、diversity、sponsored ranking、库存校验。

#### 4. Geo + vector + structured filters

本地生活、房地产、地图、线下服务：

```sql
SELECT place_id, name
FROM places
WHERE ST_DWithin(location, ST_MakePoint(:lon, :lat)::geography, :radius_m)
  AND open_now = true
  AND price_level <= 3
  AND rating_count >= 50
ORDER BY description_embedding <=> :q
LIMIT 20;
```

复杂点：

- GiST/SP-GiST geo index 与 vector index 难协同。
- radius 改变 selectivity。
- `open_now` 可能依赖时区和营业时间表，甚至需要 join。

#### 5. JSON / tag / semi-structured metadata

Observability、安全日志、开发者文档、运维知识库：

```sql
SELECT event_id
FROM events
WHERE metadata->>'service' = 'payments'
  AND metadata->'tags' ?| array['timeout', 'retry']
  AND severity IN ('warn', 'error')
  AND ts >= now() - interval '7 days'
ORDER BY message_embedding <=> :q
LIMIT 100;
```

复杂点：

- JSON/GIN、时间索引、vector index 组合。
- metadata schema 演化快。
- predicate evaluation 可能比 vector search 更贵。

#### 6. Aggregation / subquery-derived filter

分析型语义搜索、scientific search、BI + RAG：

```sql
WITH active_authors AS (
  SELECT author_id
  FROM papers
  WHERE year >= 2020
  GROUP BY author_id
  HAVING count(*) >= 5
)
SELECT p.paper_id, p.title
FROM papers p
JOIN authorship ap ON p.paper_id = ap.paper_id
JOIN active_authors aa ON aa.author_id = ap.author_id
WHERE p.venue IN ('SIGMOD', 'VLDB', 'ICDE')
ORDER BY p.abstract_embedding <=> :q
LIMIT 20;
```

复杂点：

- filter set 来自 aggregation/subquery。
- 很难让 vector index 预先感知。
- materialization、reuse、late binding 是核心问题。

#### 7. Multi-vector / multi-stage semantic predicates

多模态搜索、高级 RAG、多字段语义检索：

```sql
SELECT item_id
FROM items
WHERE tenant_id = :tenant
  AND lang = 'en'
  AND (title_embedding <=> :title_q) < 0.35
ORDER BY 0.7 * (description_embedding <=> :desc_q)
       + 0.3 * (image_embedding <=> :image_q)
LIMIT 50;
```

复杂点：

- 多个 vector columns。
- vector distance 同时出现在 filter 和 order by。
- 需要 candidate allocation、rank fusion、SQL validation。

### 复杂度分层

| Level | 查询形态 | 代表场景 | 系统难点 |
| --- | --- | --- | --- |
| L0 | 单表单属性 equality/range | category/date/rating | filtered ANN baseline |
| L1 | 多属性 AND/OR/range | 电商 facet、日志过滤 | selectivity/correlation 变化 |
| L2 | JSON/array/geo/text metadata | observability、本地生活 | GIN/GiST/BRIN 与 vector index 协同 |
| L3 | join-derived filter | ACL、tenant、group permission | filter materialization、join order、cache |
| L4 | temporal/version/visibility | 企业 RAG、合规、知识库 | row visibility、freshness、一致性 |
| L5 | aggregation/subquery-derived filter | analytics + semantic search | expensive precomputation、reuse |
| L6 | multi-vector / multi-stage retrieval | multimodal search、advanced RAG | candidate allocation、rank fusion |
| L7 | workload-level complexity | many users sharing filters | coalescing、admission control、tail latency |

### 对本项目的启发

如果要让实验更贴近论文和工业场景，不应只测随机 probability predicate。建议至少加入四类复杂 SQL：

1. **ACL / join-derived filter**：最贴近企业 RAG 和权限安全。
2. **facet + range + inventory**：最贴近电商/推荐/广告。
3. **time/version/freshness filter**：最贴近知识库和新闻/日志。
4. **JSON/tag + time**：最贴近 observability 和 semi-structured metadata。

最重要的是：这些 SQL 不一定语法很长，但它们产生的 filter set 可能来自 join、bitmap、heap visibility、JSON/GIN、time range、ACL cache。系统优化应围绕 **filter-set construction / validation / reuse**，而不是只围绕 HNSW traversal。

## 2026-06-18: PostgreSQL + pgvector Motivation Test 更新

详细实验结果见 `docs/MOTIVATION_RESULT.md` 中的 “2026-06-18 PostgreSQL +
pgvector Bottleneck Tests”。

核心结论：

- 在 PostgreSQL + pgvector 中，不能简单说 “SQL 总是慢”。更准确的分解是：
  **宽 filter 的 full id materialization 慢，窄 filter 的 pgvector candidate
  validation 慢**。
- 50% filter `popular_ge1000` 下，`LIMIT 500` 只需约 1 ms，但 full export
  5,031,984 个 id 需要约 4.1s；这说明外部 ANN + PostgreSQL allow-list 方案会被
  id 大搬运吞掉优化空间。
- 0.2% filter `grocery_long500` 下，PostgreSQL btree full export 21,317 个 id
  只需约 78 ms；此时 SQL 不再是主要问题，pgvector filtered HNSW 反而需要平均拒绝
  27,726 个候选才能返回 top-k，延迟约 197 ms。
- pgvector filtered HNSW 的 `qual_ms` 很小，真正成本在 HNSW traversal、heap tuple
  fetch、visibility/filter check、buffer/page 访问。稀疏 filter 的 shared read
  blocks 明显增大，说明候选验证 locality 很差。
- pgvector 的索引使用对 SQL 形态敏感：`ORDER BY embedding <-> parameter` 能走 HNSW，
  但把 query vector 写成 join/CTE 变量时可能退化为全表 exact sort。

由此衍生的系统研究方向：

1. **Server-side filter representation**：用 bitmap/compressed idset/shared memory
   避免百万级 id 跨 PostgreSQL client/server 边界导出。
2. **Filter-aware HNSW traversal**：把标量谓词、bitmap、posting list、cluster/page
   metadata 注入 graph search，减少稀疏 filter 下的失败候选验证。
3. **Page-aware candidate validation**：按 heap page 聚合候选、批量 fetch、再恢复
   distance order，缓解 HNSW 候选顺序与 heap 物理顺序不一致的问题。
4. **Cost-based hybrid routing**：根据 selectivity、predicate source、expected
   candidate rejection、allow-list size，在 pre-filter、post-filter、parallel
   filter+ANN、exact rerank 之间自适应切换。
