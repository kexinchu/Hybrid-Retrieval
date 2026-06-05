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
