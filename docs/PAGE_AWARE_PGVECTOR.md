# Page-Aware pgvector Traversal

## Hypothesis

Filtered vector search inside PostgreSQL is not only limited by distance
computations. pgvector's HNSW and IVFFlat access methods return heap TIDs, and
PostgreSQL then fetches heap tuples to evaluate SQL predicates. If candidate
verification follows vector-distance order, the heap access pattern can be close
to random. A page-aware verifier batches candidates by heap page, evaluates the
same candidate window with better locality, and restores distance order before
returning top-k.

This targets the system overhead highlighted by Lu et al. 2026: in a DBMS,
candidate/filter checks involve index tuples, heap TIDs, buffer manager work,
and heap tuple access rather than a single in-memory identifier lookup.

## Why this is plausible in this repository

The local pgvector source shows the right insertion point:

- HNSW search returns `scan->xs_heaptid` from `hnswgettuple` after graph
  traversal has produced candidates.
- IVFFlat already materializes `(distance, heaptid)` into a tuplesort and then
  returns one heap TID at a time.
- The SQL predicate is evaluated by the PostgreSQL executor after the index AM
  returns each TID.

So the lowest-risk prototype is not to change graph traversal first. It is to
hold a small candidate window, verify heap tuples in page order, then output the
accepted tuples in the original candidate order.

## Prototype

Run the SQL/Python harness:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_page_cluster_verify.py \
  --query-vectors-from-db \
  --queries 20 \
  --candidate-limit 5000 \
  --repeats 3 \
  --ef-search 1000 \
  --iterative-scan strict_order \
  --max-scan-tuples 200000 \
  --filter-names grocery_long500 helpful_ge20 long_review_ge500 \
  --explain \
  --out results/hybrid_vector_db/pgvector_page_cluster_verify.csv
```

The script uses real pgvector to create the same ANN candidate window for both
variants:

- `distance`: verify candidates in original ANN order.
- `page`: verify the same candidates ordered by `ctid` heap block and offset.

Both variants restore original candidate order before top-k, so `same_results`
should be true. Recall is measured against the exact filtered top-k IDs from the
existing truth CSV.

## Signals

Primary support for the idea:

- `same_results = True`
- same `recall`
- `page_verify_ms < distance_verify_ms`
- `distance_order_page_runs / distinct_heap_pages` is high
- lower `page_shared_read_blocks` or lower wall time when the working set is not
  fully cached

Important negative result:

- If `distance_order_page_runs` is already close to `distinct_heap_pages`, there
  is little locality to recover.
- If predicates are extremely cheap and all heap pages are hot in shared
  buffers, page ordering may not improve latency even when locality improves.
- If candidate windows are too large, page-aware verification can pay extra sort
  and materialization cost.

## Smoke Results

On the local 10M-row pgvector/PostgreSQL container, a small smoke run completed:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_page_cluster_verify.py \
  --query-vectors-from-db \
  --queries 3 \
  --candidate-limit 100 \
  --repeats 3 \
  --ef-search 100 \
  --filter-names popular_ge1000 price_10_to_20 \
  --out results/hybrid_vector_db/pgvector_page_cluster_verify_smoke_wide.csv
```

Summary:

- `same_results` was true for all smoke queries.
- Mean recall was `0.667` for both tested filters with this small candidate
  window.
- `popular_ge1000`: distance verification `78.65 ms`, page verification
  `77.54 ms`.
- `price_10_to_20`: distance verification `93.11 ms`, page verification
  `80.51 ms`.
- The locality signal was weak: about `97` page-runs for `96.7` distinct heap
  pages in a 100-candidate window.

A 500-candidate single-query run with `hnsw.iterative_scan = relaxed_order`
also preserved results and recall, but still showed weak locality:
`497` page-runs to `493` distinct heap pages. A 5000-candidate interactive run
was interrupted after 90 seconds while still generating candidates, so larger
windows should be run offline.

Interpretation: the harness validates the measurement path and shows that
page-ordered verification can preserve answers. The current heap layout does not
yet show strong page reuse in pgvector HNSW candidate windows, so the idea needs
either larger offline windows, different heap clustering, or workloads where ANN
candidates have stronger physical locality before it is worth a C-level
implementation.

## Large-Window And Clustered Results

Follow-up offline checks used the enhanced harness with `--locality-only` and
`--table`.

Base 10M table, large candidate window:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_page_cluster_verify.py \
  --query-vectors-from-db \
  --locality-only \
  --queries 1 \
  --query-offset 0 \
  --candidate-limit 5000 \
  --ef-search 100 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 200000 \
  --filter-names price_10_to_20 \
  --out results/hybrid_vector_db/page_locality_base_w5000_q0.csv
```

Result: `5000` candidates, `4845` distance-order heap page runs, `4693`
distinct heap pages, page-run reduction `1.03x`. Restarting the PostgreSQL
Docker container to clear shared buffers produced the same page-run counts and
similar candidate generation time (`254 ms` warm vs. `252 ms` after restart),
which means OS cache remained warm.

Base 10M table, full verification, same query with a 1000-candidate window:

- `996` page runs to `988` distinct pages, page-run reduction `1.01x`.
- Distance-order verification `81.42 ms`.
- Page-order verification `82.38 ms`.
- Same result set and recall `1.0`.

Physically vector-clustered contrast table:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/prepare_pgvector_clustered_table.py \
  --target-table amazon_grocery_reviews_10m_pgvector_anchor_q0_5k \
  --query-no 0 \
  --rows 5000 \
  --ef-search 100 \
  --max-scan-tuples 200000 \
  --maintenance-work-mem 2GB \
  --drop
```

This creates a 5000-row table physically ordered by pgvector distance to query
0, then builds an HNSW index on it.

On the clustered table:

- `1000` candidates, `134` distance-order page runs, `81` distinct heap pages,
  page-run reduction `1.65x`.
- Full verification preserved results and recall `1.0`.
- Warm verification: distance-order `80.06 ms`, page-order `80.52 ms`.
- After PostgreSQL container restart: distance-order `79.33 ms`, page-order
  `79.10 ms`.

Interpretation:

- Large windows on the original 10M heap still do not create meaningful page
  reuse; HNSW candidates are physically scattered.
- The anchor-clustered table proves the measurement is sensitive to physical
  locality: page-run reduction jumps from about `1.01x-1.03x` to `1.65x`.
- SQL-level page reordering does not yet translate into clear latency wins,
  likely because the test table is small/hot and the SQL CTE/LATERAL prototype
  adds overhead that a C-level implementation would not.
- A true cold-cache result would require dropping OS page cache or running on a
  dataset larger than memory; Docker restart only clears PostgreSQL
  shared_buffers.

## Amazon-C4 Query Results

Amazon-C4 provides 21.2k test queries with `qid`, natural-language `query`,
`item_id`, `user_id`, `ori_rating`, and `ori_review`. The local copy lives at
`data/amazon_c4/test.csv`. To use these queries with the existing 10M Grocery
pgvector table, map C4 `(item_id, user_id)` to local
`(parent_asin, user_id)`:

```bash
python experiments/hybrid_vector_db/scripts/select_amazon_c4_pgvector_queries.py \
  --max-matches 200 \
  --out results/hybrid_vector_db/amazon_c4_pgvector_queries.csv
```

This found 200 unique C4 query anchors after scanning about 5M local review
rows, so no extra product-review data was needed.

Base 10M table, C4 q39, 1000-candidate request:

- pgvector returned 380 candidates.
- `distance_order_page_runs = 338`, `distinct_heap_pages = 333`.
- Page-run reduction was only `1.015x`.
- Verification latency was `79.75 ms` in distance order and `82.99 ms` in page
  order.

After restarting the PostgreSQL container, the same query produced identical
page-run counts. `EXPLAIN (ANALYZE, BUFFERS)` showed `0` shared read blocks for
both verification orders, so this is only cold with respect to PostgreSQL shared
buffers; the host OS page cache remained warm.

Query-correlated physical layout, approximate 380-row table:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/prepare_pgvector_clustered_table.py \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries.csv \
  --query-no 39 \
  --rows 50000 \
  --target-table amazon_grocery_reviews_10m_pgvector_c4q39_clustered_50k \
  --drop
```

Because this path uses pgvector HNSW to select rows, it materialized only 380
rows for this anchor. On that table, 300 candidates had `48` page-runs over `25`
heap pages, a `1.92x` run-reduction opportunity. Latency stayed effectively
flat because the table was too small and hot.

Query-correlated physical layout, exact 5k table:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/prepare_pgvector_clustered_table.py \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries.csv \
  --query-no 39 \
  --rows 5000 \
  --target-table amazon_grocery_reviews_10m_pgvector_c4q39_exact_clustered_5k \
  --drop \
  --exact-fbin data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin
```

This exact path scans the local 5GB fbin and then materializes rows from
PostgreSQL in exact query-distance order. Results:

- 1000-candidate verification: `87` page-runs over `77` heap pages, `1.13x`
  page-run reduction, with essentially unchanged latency (`80.12 ms` vs
  `79.89 ms`).
- 5000-candidate locality-only run: pgvector returned 4783 candidates with
  `474` page-runs over `381` heap pages, `1.24x` page-run reduction.

Takeaway: with Amazon-C4 query anchors, the base 10M heap layout remains too
random for page-aware verification to help. Physically correlated tables do
create measurable page-run reduction, but in these hot-cache/small-table runs
latency is still dominated by executor/materialization overhead rather than
random disk I/O.

## Generic Vector-Cluster Layout

The anchor-clustered tables above are useful upper-bound tests, but they are
query-specific. A more general physical layout is to assign each row to a coarse
`vector_cluster_id` and store heap tuples by cluster:

```text
heap order = vector_cluster_id, distance_to_centroid
```

Build two 200k-row control tables with identical rows:

- `amazon_grocery_reviews_10m_pgvector_id_order_200k`
- `amazon_grocery_reviews_10m_pgvector_vector_clustered_200k`

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/prepare_pgvector_vector_clustered_table.py \
  --rows 200000 \
  --clusters 256 \
  --train-sample 50000 \
  --kmeans-iter 20 \
  --id-order-table amazon_grocery_reviews_10m_pgvector_id_order_200k \
  --clustered-table amazon_grocery_reviews_10m_pgvector_vector_clustered_200k
```

Implementation details:

- FAISS k-means trains 256 centroids on 50k sampled vectors.
- The first 200k vectors are assigned to nearest centroid.
- The clustered table is materialized by `(vector_cluster_id,
  distance_to_centroid)`.
- Both tables have the same 200k rows and their own HNSW index.
- Both heaps are `121 MB`; total table+index size is `274 MB`.

Using 9 Amazon-C4 query anchors with `query_id < 200000`, 1000 candidates, and
`hnsw.iterative_scan = relaxed_order`:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_page_cluster_verify.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_200k \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_under200k.csv \
  --query-vectors-from-db \
  --queries 9 \
  --candidate-limit 1000 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 300000 \
  --filter-names price_10_to_20 \
  --out results/hybrid_vector_db/page_verify_c4_vector_clustered_200k_w1000.csv
```

Locality-only result:

| Layout | Mean distinct heap pages | Mean distance-order page runs | Run/page ratio |
| --- | ---: | ---: | ---: |
| id order | 823.3 | 914.6 | 1.11x |
| vector clustered | 374.2 | 923.8 | 2.47x |

Full verification result:

| Layout | Distance verify ms | Page verify ms | Speedup | Same results |
| --- | ---: | ---: | ---: | --- |
| id order | 80.85 | 80.87 | 1.00x | true |
| vector clustered | 82.50 | 80.12 | 1.03x | true |

To isolate heap layout from HNSW candidate-set differences, a same-candidate-ID
check used candidates from the id-order HNSW index and measured their `ctid`
distribution in both heaps:

| Layout | Mean distinct heap pages | Mean distance-order page runs | Run/page ratio |
| --- | ---: | ---: | ---: |
| id order | 823.3 | 914.6 | 1.11x |
| vector clustered | 343.0 | 796.1 | 2.32x |

This is the cleanest evidence so far: generic `vector_cluster_id` physical
ordering substantially improves heap-page locality for the same candidate IDs.
The SQL-level page verification prototype converts that into only a modest
latency gain because the tables are small/hot and the prototype pays
CTE/LATERAL/materialization overhead. A C-level implementation should have a
better chance of preserving the locality benefit.

## Full 10M Vector-Cluster Layout

The 200k result was repeated on the full 10M rows. The original 10M table was
used as the id/import-order baseline, and a new full table was materialized by
coarse vector cluster:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/prepare_pgvector_vector_clustered_table.py \
  --rows 10000000 \
  --clusters 1024 \
  --train-sample 200000 \
  --kmeans-iter 25 \
  --chunk-size 200000 \
  --skip-id-order \
  --clustered-table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --maintenance-work-mem 8GB \
  --work-mem 4GB \
  --ef-construction 64
```

Build notes:

- FAISS k-means trained 1024 centroids on 200k sampled vectors.
- Assigning all 10M vectors to centroids took about 5 seconds.
- PostgreSQL materialization plus HNSW build dominated runtime.
- Total build time for the clustered table and HNSW was about 2717 seconds.
- Original table: heap `6010 MB`, total `15 GB`.
- Vector-clustered 10M table: heap `6010 MB`, total `13 GB` with pkey + HNSW.

The full-table benchmark used the first 7 Amazon-C4 query anchors, 1000
candidate requests, `hnsw.ef_search = 1000`,
`hnsw.iterative_scan = relaxed_order`, and `hnsw.max_scan_tuples = 500000`.

Locality-only result:

| Layout | Mean candidates | Mean distinct heap pages | Mean distance-order page runs | Run/page ratio |
| --- | ---: | ---: | ---: | ---: |
| original 10M | 911.4 | 886.7 | 896.0 | 1.01x |
| vector-clustered 10M | 854.3 | 531.1 | 707.9 | 1.33x |

Full verification result:

| Layout | Distance verify ms | Page verify ms | Speedup | Same results |
| --- | ---: | ---: | ---: | --- |
| original 10M | 79.99 | 79.90 | 1.00x | true |
| vector-clustered 10M | 74.84 | 70.48 | 1.06x | true |

Because rebuilding HNSW over a different heap insertion order can change the
candidate set, a same-candidate-ID check was also run. Candidates were generated
only from the original 10M HNSW index, then the same IDs were looked up in both
heaps:

| Layout | Mean candidates | Mean distinct heap pages | Mean distance-order page runs | Run/page ratio |
| --- | ---: | ---: | ---: | ---: |
| original 10M | 911.4 | 886.7 | 896.0 | 1.01x |
| vector-clustered 10M | 911.4 | 479.0 | 823.4 | 1.72x |

Interpretation:

- On the original 10M heap, candidates are still almost random with respect to
  heap pages.
- Full `vector_cluster_id` physical ordering roughly halves the number of heap
  pages touched by the same candidate IDs (`886.7 -> 479.0`).
- The SQL prototype now shows a visible verification improvement on the
  clustered layout (`74.84 ms -> 70.48 ms`, about `1.06x`).
- The clustered table's own HNSW sometimes returned fewer candidates for the
  same 1000-candidate request, so same-candidate-ID locality is the cleanest
  evidence for physical-layout impact.

## Next C-level experiment

If the SQL harness shows a consistent win, the pgvector implementation path is:

1. Add a bounded candidate buffer in HNSW/IVFFlat scan state.
2. Fill up to a GUC-controlled window size, for example
   `hnsw.page_verify_window`.
3. Sort buffered heap TIDs by `(ItemPointerGetBlockNumber,
   ItemPointerGetOffsetNumber)` for heap verification.
4. Preserve the original distance rank and emit passing tuples in rank order.
5. Report candidate pages, page runs, and returned tuples through the existing
   local `vector_hnsw_last_scan_profile()` instrumentation.

For strict top-k semantics with early stopping, keep a watermark: after finding
`k` passing candidates, all candidates with original rank earlier than the kth
passing candidate must have been verified before emitting the final answer.

## 400-Query Controlled Rerun

The 7-query result above was too small, and it also mixed two questions:

- Does a physical layout make candidates more page-local?
- Does this SQL-level page-aware prototype turn that locality into latency?

To reduce candidate-generation and HNSW-insertion-order confounds, the controlled
rerun used fixed FAISS HNSW candidate IDs, then verified the same IDs against
PostgreSQL heap layouts. This isolates heap layout and page-aware verification;
it is not a pure end-to-end pgvector HNSW benchmark.

Query selection:

```bash
python experiments/hybrid_vector_db/scripts/select_amazon_c4_pgvector_queries.py \
  --max-matches 400 \
  --fallback-item-only \
  --out results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --progress-rows 2000000
```

This produced 400 Amazon-C4-derived query anchors:

- 386 exact `(item_id, user_id)` matches to local `(parent_asin, user_id)`.
- 14 item-only fallback matches to reach 400 queries.

Cache policy:

- Restarted the PostgreSQL Docker container before the baseline group.
- Ran all 400 baseline queries continuously, preserving within-group cache
  behavior.
- Restarted the PostgreSQL Docker container again before the vector-clustered
  group.
- This clears PostgreSQL shared buffers between groups. It does not drop the
  host OS page cache.

Candidate and table setup:

- 10M-row original table: `amazon_grocery_reviews_10m_pgvector`.
- 10M-row vector-clustered table:
  `amazon_grocery_reviews_10m_pgvector_vector_clustered_10m`.
- Candidate request: 1000 per query.
- FAISS HNSW `efSearch = 1000`.
- Repeats for verification timing: 3.

Locality-only result:

| Layout | Queries | Mean candidates | Mean heap pages | Mean distance-order page runs | Run/page ratio | Mean per-query run reduction | P50 run reduction | P95 run reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original 10M | 400 | 995.4 | 963.1 | 972.2 | 1.01x | 1.01x | 1.01x | 1.03x |
| vector-clustered 10M | 400 | 995.4 | 614.3 | 844.1 | 1.37x | 1.51x | 1.27x | 2.87x |

Full fixed-candidate verification result:

| Layout | Queries | Mean heap pages | Mean page runs | Run/page ratio | Distance verify ms | Page verify ms | Speedup | Same results |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| original 10M | 400 | 963.1 | 972.2 | 1.01x | 82.06 | 82.30 | 0.997x | true |
| vector-clustered 10M | 400 | 614.3 | 844.1 | 1.37x | 82.28 | 82.58 | 0.996x | true |

Verification timing distribution:

| Layout | Distance p50 ms | Distance p95 ms | Page p50 ms | Page p95 ms | Median paired speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| original 10M | 81.25 | 87.14 | 81.05 | 88.27 | 1.002x |
| vector-clustered 10M | 81.55 | 85.69 | 81.40 | 87.27 | 1.002x |

Interpretation:

- The larger 400-query run confirms the physical-layout effect: vector-clustered
  layout reduces distinct heap pages from `963.1` to `614.3` for the same fixed
  candidate IDs.
- Page-aware ordering also has more opportunity on that layout: mean page-run
  ratio rises from `1.01x` on the original heap to `1.37x` on the clustered
  heap, and the per-query run reduction has a long tail.
- The current SQL/Python verification prototype does not convert that locality
  into latency. Average page-order verification was slightly slower on both
  layouts, while median paired speedup was essentially flat.
- The likely reason is that this verification path has a large fixed cost
  around SQL execution, array materialization, ordering, and client/server
  round-trips. The measured verify phase is about `80 ms` even for small
  candidate counts, so reducing heap page runs is not the dominant term here.
- Therefore this run supports the locality premise, but it does not yet prove
  end-to-end performance benefit. A C-level executor/index-AM prototype or a
  more IO-bound cold-cache setup is needed before claiming the idea works.

## C-Level HNSW Page-Access Prototype

The first C-level prototype was implemented inside the local pgvector HNSW scan
path under `external/pgvector-src` and installed into the
`hybrid-pgvector` PostgreSQL container.

Implementation summary:

- Changed `src/hnsw.c`, `src/hnsw.h`, `src/hnswscan.c`, `src/vector.c`, and
  `sql/vector.sql`.
- Added `hnsw.page_access = off|prefetch|reorder`.
- Added `hnsw.page_window`, default `128`.
- Added a bounded `HnswPageAccessItem` buffer in `hnswgettuple`.
- `prefetch` mode fills a window of candidate heap TIDs, sorts the window by
  heap block/offset only to issue `PrefetchBuffer()` calls, then restores the
  original candidate rank before returning TIDs to the executor.
- `reorder` mode returns TIDs in heap-page order within the window. This is
  intentionally experimental and is not semantically safe for ordinary
  `ORDER BY embedding <-> query LIMIT k`.
- Added profile functions:
  `vector_hnsw_reset_scan_profile()` and
  `vector_hnsw_last_scan_profile()`.

The extension was rebuilt and installed in the running Postgres container:

```bash
docker cp external/pgvector-src hybrid-pgvector:/tmp/pgvector-src-pageaware
docker exec hybrid-pgvector bash -lc \
  'cd /tmp/pgvector-src-pageaware && make clean && make -j$(nproc) && make install'
docker restart hybrid-pgvector
```

Correctness check:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 6 \
  --k 20 \
  --ef-search 100 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 50000 \
  --page-window 128 \
  --mode prefetch \
  --statement-timeout-ms 30000 \
  --out results/hybrid_vector_db/pgvector_hnsw_page_access_correctness_original_prefetch_q6.csv
```

For the first 6 Amazon-C4 anchors:

| Layout | Mode compared with `off` | Same returned IDs | Mean page runs | Mean distinct pages |
| --- | --- | --- | ---: | ---: |
| original 10M | prefetch | true for 6/6 | 125.8 | 125.5 |
| vector-clustered 10M | prefetch | true for 6/6 | 103.2 | 95.7 |

The same smoke run with `reorder` returned different IDs from `off`, as
expected. That confirms the important semantic boundary: an index AM cannot
simply return page-sorted TIDs for a top-k ordered scan without changing query
results. Correct page-aware batching needs either order-preserving prefetch in
the index AM or executor support that can fetch heap tuples in page order while
emitting final results in rank order.

Small-sample latency:

| Layout | Off mean ms | Prefetch mean ms | Notes |
| --- | ---: | ---: | --- |
| original 10M | 15.98 | 2313.59 | one prefetch outlier at 13.75s |
| vector-clustered 10M | 11.98 | 19.58 | prefetch slower despite fewer pages |

Interpretation:

- The safe C-level implementation preserves answers.
- The vector-clustered table again shows better page locality in the profile.
- The current `prefetch` insertion point can be slower because it fills a
  `page_window` before returning the first TID, while the ordinary scan can stop
  as soon as the executor has consumed `LIMIT k` tuples.
- A direct 400-query pgvector end-to-end run with `ef_search = 1000` was
  impractical: an early correctness attempt hit 60s statement timeouts on C4
  anchors. The 400-query C-level benchmark below uses `ef_search = 100` and
  `LIMIT 1000` to keep runtime bounded while still making the scan emit about
  1000 candidate TIDs for most queries.
- The next implementation step should move from prefetch-only to a real
  executor-aware heap batching path, or make the HNSW page window adaptive so it
  does not over-fetch for small `LIMIT` queries.

## C-Level 400-Query Result

After installing the modified `vector.so`, the existing database also needed the
new SQL functions registered:

```sql
CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() RETURNS text
AS 'vector', 'vector_hnsw_last_scan_profile'
LANGUAGE C VOLATILE PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() RETURNS void
AS 'vector', 'vector_hnsw_reset_scan_profile'
LANGUAGE C VOLATILE PARALLEL SAFE;
```

The safe C-level benchmark compared `hnsw.page_access = off` with
`hnsw.page_access = prefetch` on
`amazon_grocery_reviews_10m_pgvector_vector_clustered_10m`.

Settings:

- 400 Amazon-C4-derived query anchors.
- `LIMIT 1000`.
- `hnsw.ef_search = 100`.
- `hnsw.iterative_scan = relaxed_order`.
- `hnsw.max_scan_tuples = 50000`.
- `hnsw.page_window = 1000`.
- PostgreSQL container restarted between `off` and `prefetch` groups.
- Host OS page cache was not dropped.

Commands:

```bash
PYTHONPATH=experiments/hybrid_vector_db/scripts \
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 1000 \
  --ef-search 100 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 50000 \
  --page-window 1000 \
  --mode off \
  --statement-timeout-ms 60000 \
  --stream \
  --out results/hybrid_vector_db/c_page_access_vc_off_k1000_q400.csv

PYTHONPATH=experiments/hybrid_vector_db/scripts \
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 1000 \
  --ef-search 100 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 50000 \
  --page-window 1000 \
  --mode prefetch \
  --statement-timeout-ms 60000 \
  --stream \
  --out results/hybrid_vector_db/c_page_access_vc_prefetch_k1000_q400.csv
```

Correctness:

- `prefetch` returned the same ID sequence as `off` for `400/400` queries.
- No query timed out in either group.
- A `reorder` negative control over 6 queries returned the same 1000-candidate
  set but a different order for `6/6` queries, confirming that page-order TID
  emission is not safe for ordinary ordered index scans.

All 400 queries:

| Mode | Mean returned | Mean latency ms | P50 ms | P95 ms | Mean vector search ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 894.2 | 35.16 | 36.65 | 68.93 | 29.22 |
| prefetch | 894.2 | 35.03 | 36.99 | 68.51 | 28.86 |

Prefetch profile over all 400 queries:

| Mean candidates | Mean heap pages | Mean page runs | Run/page ratio | Per-query run/page p50 | Per-query run/page p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 894.2 | 534.6 | 757.3 | 1.42x | 1.27x | 2.86x |

Queries that returned a full 1000 candidates (`317/400`):

| Mode | Mean latency ms | Speedup off/prefetch | Median paired speedup |
| --- | ---: | ---: | ---: |
| off | 43.74 | 1.005x | 0.978x |
| prefetch | 43.51 |  |  |

Full-return prefetch locality:

| Mean candidates | Mean heap pages | Mean page runs | Run/page ratio | Per-query run/page p50 | Per-query run/page p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000.0 | 639.4 | 911.0 | 1.42x | 1.35x | 3.10x |

Interpretation:

- The C-level safe path removes the SQL prototype's candidate-table and LATERAL
  verification overhead.
- It preserves correctness because it only prefetches page-sorted blocks, then
  emits TIDs in the original distance/rank order.
- It again shows real page locality on the vector-clustered 10M heap.
- It still does not show the desired `1.06x` latency gain. Average latency is
  effectively flat, and the median paired result is slightly worse.
- The reason is architectural: `prefetch` does not change the executor's heap
  tuple fetch order. It can warm buffers but cannot batch actual heap fetches by
  page. The unsafe `reorder` mode can change fetch order, but it breaks ordered
  scan semantics unless an executor node buffers verified tuples and emits them
  by original rank.

## C-Level Top-K Rerun With ef_search 1000

The next rerun used a more realistic top-k query shape while keeping a large
HNSW candidate window:

- 400 Amazon-C4-derived query anchors.
- Table: `amazon_grocery_reviews_10m_pgvector_vector_clustered_10m`.
- `LIMIT 20`.
- `hnsw.ef_search = 1000`.
- `hnsw.iterative_scan = relaxed_order`.
- `hnsw.max_scan_tuples = 500000`.
- `hnsw.page_window = 1000`.
- PostgreSQL container restarted between `off` and `prefetch` groups.
- Host OS page cache was not dropped.

Commands:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 20 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 500000 \
  --page-window 1000 \
  --mode off \
  --statement-timeout-ms 60000 \
  --stream \
  --out results/hybrid_vector_db/c_level_pgvector_vc10m_off_q400_w1000.csv

.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 20 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 500000 \
  --page-window 1000 \
  --mode prefetch \
  --statement-timeout-ms 60000 \
  --stream \
  --out results/hybrid_vector_db/c_level_pgvector_vc10m_prefetch_q400_w1000.csv
```

Correctness:

- `prefetch` returned the same ordered ID sequence as `off` for `400/400`
  queries.
- No query timed out in either group.
- A 20-query `reorder` negative control returned different ordered IDs for
  `20/20` queries, confirming again that page-order TID emission is not a
  correct implementation for ordered top-k scans.

Latency:

| Mode | Mean latency ms | P50 ms | P95 ms | Mean vector search ms |
| --- | ---: | ---: | ---: | ---: |
| off | 33.35 | 35.52 | 64.20 | 32.03 |
| prefetch | 32.77 | 35.81 | 65.09 | 30.76 |

Overall speedup:

| Metric | Value |
| --- | ---: |
| Mean off/prefetch speedup | 1.018x |
| Median paired speedup | 0.998x |
| Paired p05 speedup | 0.726x |
| Paired p95 speedup | 1.290x |

Prefetch profile:

| Mean candidates | Mean heap pages | Mean page runs | Run/page ratio | Per-query run/page mean | Per-query run/page p50 | Per-query run/page p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 894.2 | 534.8 | 758.0 | 1.42x | 1.54x | 1.27x | 2.86x |

Interpretation:

- This is the cleanest safe C-level pgvector result so far: page-aware mode is
  inside `hnswgettuple`, not a SQL candidate-table prototype.
- Correctness is good for `prefetch` because TIDs are still emitted in original
  distance/rank order.
- The vector-clustered heap still provides page locality: the scan sees about
  `758` distance-order page runs over `535` distinct pages.
- Latency is only slightly better by the mean and flat by paired median. This is
  weaker than the earlier 7-query SQL result (`1.06x`) and should not be claimed
  as a robust speedup.
- The remaining gap is that `prefetch` warms page buffers but does not make the
  executor fetch heap tuples in page order. The requested "page-order verify,
  then restore distance order" behavior needs an executor/materialization layer
  or a custom scan node that owns heap fetching and final rank-ordered emission.

## Executor/Materialization-Level Scan

The next prototype adds a C-level materialization SRF inside the local pgvector
extension:

```sql
vector_hnsw_page_materialize(index regclass, query vector, k int4, candidate_limit int4)
```

This is not yet a planner-integrated `CustomScan`, but it moves the important
operation below the SQL prototype level:

1. Open the HNSW index and heap relation directly from C.
2. Use PostgreSQL's index scan API to collect HNSW candidate TIDs in standard
   distance order.
3. Record the original rank for each TID.
4. Sort candidate TIDs by heap block and offset.
5. Fetch heap tuples in page order with `table_tuple_fetch_row_version()`.
6. Sort the visible candidates back by original HNSW rank.
7. Emit the same top-k rank order as the standard pgvector query.

Implemented files:

- `external/pgvector-src/src/vector.c`
- `external/pgvector-src/sql/vector.sql`
- `experiments/hybrid_vector_db/scripts/pgvector_hnsw_materialize_benchmark.py`

Smoke correctness on one Amazon C4-derived query:

```text
same True
base [38386, 9302996, 9095817, 9201677, 8437986, 7795802, 3437955, 5356897, 5896764, 8054321]
mat  [38386, 9302996, 9095817, 9201677, 8437986, 7795802, 3437955, 5356897, 5896764, 8054321]
profile:
  candidates: 1000
  visible: 1000
  distance_order_page_runs: 982
  distinct_heap_pages: 639
  index_ms: 10.46
  page_fetch_ms: 5.95
```

400-query run on the 10M vector-clustered table:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_materialize_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --index amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_embedd \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 20 \
  --candidate-limit 1000 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 500000 \
  --isolate-cache \
  --container hybrid-pgvector \
  --out results/hybrid_vector_db/materialized_scan_vc10m_q400_w1000_isolated.csv \
  --summary-out results/hybrid_vector_db/materialized_scan_vc10m_q400_w1000_isolated_summary.csv
```

Correctness:

- `400/400` queries returned the same ordered top-20 IDs as standard pgvector.
- The queries came from `results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv`,
  which maps Amazon C4 query text to rows in the local Amazon Grocery pgvector
  table.
- The standard group and materialized group were separated by a PostgreSQL
  container restart. Within each group, cache state was left natural.

Latency:

| Scan | Mean latency ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| Standard pgvector executor path | 23.65 | 23.26 | 53.19 |
| C page-materialized path | 22.53 | 22.28 | 50.03 |

Overall:

| Metric | Value |
| --- | ---: |
| Mean standard/materialized speedup | 1.040x |
| Median paired speedup | 0.995x |
| Mean candidates | 894.2 |
| Mean visible candidates | 894.2 |
| Mean distinct heap pages | 534.8 |
| Mean distance-order page runs | 758.0 |
| Mean per-query run/page ratio | 1.54x |
| Mean materialized index time | 19.89 ms |
| Mean materialized page fetch time | 1.66 ms |

Interpretation:

- The materialization path validates the mechanism: HNSW candidates are still
  rank-correct, heap fetches can be done in page order, and final output can be
  restored to distance order.
- The locality signal remains meaningful: the average query has about `758`
  distance-order page runs for `535` distinct heap pages, so page-order fetching
  can reduce heap run count by roughly `1.4x` to `1.5x`.
- End-to-end latency only improves modestly by mean and is flat by median. The
  reason is now clearer: on this workload the HNSW index traversal dominates
  (`19.89 ms` mean) and materialized heap fetching is small (`1.66 ms` mean).
  Reducing heap page runs therefore cannot translate linearly into whole-query
  speedup.
- This result is stronger than the SQL prototype because the heap batching is
  implemented in C, but it is still not a full planner/executor `CustomScan`.
  A production version would need a scan node that integrates with planning,
  projection, quals, tuple slots, and EXPLAIN instead of exposing a special SRF
  that assumes an `id bigint` heap column.

## Index-Page-Aware HNSW Physical Placement

After separating heap fetches from HNSW index traversal, the dominant random
access source is the HNSW index itself. A standard top-20 pgvector query can
touch thousands of HNSW index pages while fetching only about top-k heap tuples.

pgvector's HNSW index is physically a graph on index pages:

- The metapage stores the entry point.
- Each element tuple stores vector data, heap TIDs, level, and a TID pointing
  to its neighbor tuple.
- Each neighbor tuple stores neighbor element index TIDs.
- Search expands graph candidates by loading a candidate's neighbor tuple, then
  loading neighbor element tuples by their index TIDs.

The default bulk build keeps the graph in memory, then `FlushPages()` writes the
element list in insertion-list order. In this local pgvector branch, the build
path now supports:

```sql
SET hnsw.build_page_order = insertion; -- pgvector-compatible physical order
SET hnsw.build_page_order = bfs;       -- graph-neighbor BFS physical order
```

The `bfs` mode does not change query-time HNSW traversal. It only re-links the
in-memory graph before assigning on-disk index TIDs, starting from the HNSW
entry point and walking graph neighbors. To isolate physical placement from
graph randomness, this experiment also fixes the build seed in the local branch.

Implementation:

- `external/pgvector-src/src/hnsw.c`
  - Adds `hnsw.build_page_order`.
- `external/pgvector-src/src/hnswbuild.c`
  - Adds graph-neighbor BFS re-linking before `CreateGraphPages()`.
  - Seeds the build PRNG for deterministic insertion-vs-BFS comparisons.

Build and benchmark setup:

- Table data: 10M rows copied from
  `amazon_grocery_reviews_10m_pgvector_vector_clustered_10m`.
- Comparison table:
  `amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx`.
- Both indexes are built serially on the same table with:
  - `maintenance_work_mem = '64GB'`
  - `max_parallel_maintenance_workers = 0`
  - `m = 16`
  - `ef_construction = 64`
  - fixed build seed
- Query workload: `400` Amazon C4-mapped queries from
  `results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv`.
- Query settings:
  - `hnsw.ef_search = 1000`
  - `hnsw.iterative_scan = relaxed_order`
  - `hnsw.max_scan_tuples = 500000`
  - `hnsw.page_access = off`
  - `hnsw.index_page_access = off`
  - `k = 20`

Commands:

```sql
SET maintenance_work_mem='64GB';
SET max_parallel_maintenance_workers=0;
SET hnsw.build_page_order=insertion;
CREATE INDEX amazon_grocery_reviews_10m_pgvector_vc10m_seed_insertion_hnsw
ON amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx
USING hnsw (embedding vector_l2_ops)
WITH (m=16, ef_construction=64);

SET hnsw.build_page_order=bfs;
CREATE INDEX amazon_grocery_reviews_10m_pgvector_vc10m_seed_bfs_hnsw
ON amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx
USING hnsw (embedding vector_l2_ops)
WITH (m=16, ef_construction=64);
```

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 20 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 500000 \
  --page-window 1000 \
  --mode off \
  --index-page-access off \
  --statement-timeout-ms 60000
```

Result files:

- `results/hybrid_vector_db/index_layout_vc10m_seed_insertion_off_q400_w1000.csv`
- `results/hybrid_vector_db/index_layout_vc10m_seed_bfs_off_q400_w1000.csv`
- `results/hybrid_vector_db/index_layout_vc10m_seed_insertion_vs_bfs_summary.json`

Correctness:

- Same ordered top-20 IDs: `400/400`.
- Mean top-20 overlap: `1.000`.
- Mean visited tuples, neighbor loads, and element loads are identical, so the
  comparison isolates physical page placement rather than graph/search behavior.

Latency:

| Physical HNSW index order | Mean latency ms | P50 ms | P95 ms | Mean vector search ms |
| --- | ---: | ---: | ---: | ---: |
| insertion | 32.82 | 33.77 | 66.00 | 31.61 |
| bfs graph order | 26.69 | 27.14 | 52.55 | 25.60 |

Speedup:

| Metric | Value |
| --- | ---: |
| Mean insertion/BFS speedup | 1.250x |
| Median paired speedup | 1.243x |
| Mean latency speedup | 1.230x |
| P50 latency speedup | 1.244x |
| P95 latency speedup | 1.256x |

Index locality:

| Physical HNSW index order | Mean element loads | Mean element page runs | Element run/load | Mean element distinct-page sum |
| --- | ---: | ---: | ---: | ---: |
| insertion | 7050.8 | 6913.2 | 0.980 | 6716.9 |
| bfs graph order | 7050.8 | 6528.9 | 0.926 | 5877.8 |

| Metric | Value |
| --- | ---: |
| Element page-run reduction | 1.059x |
| Element distinct-page-sum reduction | 1.143x |
| Element run/load ratio delta | 0.0545 |

Interpretation:

- This is the first result that matches the expected system-overhead story:
  the optimization targets HNSW index page locality, not heap tuple fetches.
- BFS physical placement preserves the same graph/search behavior in this
  deterministic experiment, but makes graph-adjacent element tuples more
  physically local.
- The run-count reduction is modest (`1.06x`), but latency improves more
  strongly (`~1.23x` by mean latency). This suggests cache/TLB/prefetch behavior
  and fewer cold index-page touches matter beyond the simple run-count metric.
- This is still a build-time physical layout experiment, not query-time
  page-aware traversal. The next step is to combine this with query-time
  index-page prefetch or windowed expansion while preserving recall.

## Index-Page-Aware HNSW Traversal

The heap-materialization result showed that standard pgvector top-k does not
fetch enough heap tuples for heap batching to be the main win. A one-query
`pg_statio` check made the real bottleneck clear:

```text
returned rows: 20
heap blocks touched: 21
HNSW index blocks touched: 8985
```

So the next prototype moves page awareness into HNSW graph traversal itself.

Implementation:

- New GUC: `hnsw.index_page_access = off | prefetch`.
- Location: `external/pgvector-src/src/hnswutils.c`, inside
  `HnswSearchLayer()`.
- For every expanded HNSW candidate, pgvector loads its neighbor tuple and then
  iterates unvisited neighbor element TIDs.
- In `prefetch` mode, the unvisited neighbor element TIDs are copied into a
  small block list, sorted by HNSW index block, deduplicated, and passed to
  `PrefetchBuffer(index, MAIN_FORKNUM, block)`.
- The actual HNSW candidate processing order is preserved. This means the mode
  is intended to be correctness-preserving; it only warms index pages ahead of
  the existing distance-priority traversal.

Additional profile fields in `vector_hnsw_last_scan_profile()`:

- `index_page_neighbor_loads`
- `index_page_neighbor_runs`
- `index_page_neighbor_distinct_pages`
- `index_page_element_loads`
- `index_page_element_runs`
- `index_page_element_distinct_pages`
- `index_page_prefetches`

Smoke result on 5 Amazon C4-derived queries:

```text
same ordered top-20 IDs: 5/5

Example query:
  index element loads:          7969
  index element page runs:      7944
  index element distinct pages: 7878
  index page prefetches:        7878
```

This matches the paper-style diagnosis much better than heap-page metrics:
HNSW graph traversal jumps across index pages almost every element load.

400-query run:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 20 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 500000 \
  --page-window 1 \
  --mode off \
  --index-page-access off \
  --statement-timeout-ms 60000 \
  --out results/hybrid_vector_db/index_page_off_vc10m_q400_w1000.csv

.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m \
  --query-id-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --k 20 \
  --ef-search 1000 \
  --iterative-scan relaxed_order \
  --max-scan-tuples 500000 \
  --page-window 1 \
  --mode off \
  --index-page-access prefetch \
  --statement-timeout-ms 60000 \
  --out results/hybrid_vector_db/index_page_prefetch_vc10m_q400_w1000.csv
```

The two groups were separated by a PostgreSQL container restart. Host OS cache
was not dropped.

Correctness:

- `400/400` queries returned the same ordered top-20 IDs for `off` and
  `prefetch`.

Latency:

| Mode | Mean latency ms | P50 ms | P95 ms | Mean vector search ms |
| --- | ---: | ---: | ---: | ---: |
| index-page off | 23.79 | 22.87 | 55.42 | 23.11 |
| index-page prefetch | 24.34 | 23.22 | 55.40 | 23.64 |

Paired speedup:

| Metric | Value |
| --- | ---: |
| Mean off/prefetch speedup | 1.031x |
| Median paired speedup | 0.928x |
| Paired p05 speedup | 0.526x |
| Paired p95 speedup | 1.690x |

Index-page profile:

| Metric | Mean |
| --- | ---: |
| HNSW visited tuples | 6305.9 |
| Index element loads | 6442.1 |
| Index element page runs | 6420.0 |
| Index element distinct pages, per expanded-neighbor batch sum | 6375.1 |
| Index prefetches in prefetch mode | 6375.1 |
| Index neighbor tuple loads | 817.3 |
| Index neighbor tuple page runs | 808.8 |
| Element run / distinct-page ratio | 1.007x |

Interpretation:

- This confirms that pgvector's system overhead is primarily HNSW index-page
  traversal, not heap tuple fetching, for pure top-k.
- The prefetch prototype is correctness-preserving and touches the right layer,
  but it does not produce a robust latency win on this run.
- The reason is visible in the profile: element page runs are almost equal to
  per-batch distinct pages. The traversal has very little local page reuse, and
  prefetch is issued only shortly before each page is needed. That can hide some
  misses on favorable queries, but it also adds thousands of prefetch calls and
  is not enough lookahead to reliably improve median latency.
- A stronger page-aware traversal needs a wider lookahead across multiple HNSW
  candidate expansions, for example collecting a small window from the candidate
  queue, grouping neighbor/element loads by index page, then restoring HNSW
  priority semantics as much as possible. That next step will no longer be a
  pure prefetch-only optimization; it must be evaluated by recall/latency, not
  only exact ordered-ID equality.

## Metadata-Cache Filter Prototype

After the BFS physical-layout result, the next question was whether pgvector can
keep PostgreSQL compatibility while avoiding expensive heap validation for
high-frequency filters. The prototype here keeps heap/index separation intact:
it adds a backend-local pgvector-side TID metadata cache for selected hot
predicates.

Implementation:

- Added C functions in `external/pgvector-src/src/vector.c`:
  - `vector_hnsw_metadata_cache_build(index regclass, filter text)`
  - `vector_hnsw_metadata_filter_search(index regclass, query vector, k int4,
    candidate_limit int4, filter text)`
  - `vector_hnsw_metadata_filter_profile()`
- Added declarations in `external/pgvector-src/sql/vector.sql`.
- Added benchmark harness:
  `experiments/hybrid_vector_db/scripts/pgvector_metadata_cache_benchmark.py`.
- Supported prototype filters:
  - `helpful_ge20`: `helpful_vote >= 20`
  - `grocery_long500`: `main_category = 'Grocery' AND review_text_len >= 500`
  - `grocery_helpful`: `main_category = 'Grocery' AND helpful_vote >= 1`
  - `rating5_price_le10`: `has_price AND price <= 10 AND rating = 5`

The cache is intentionally narrow and experimental. It stores only passing
`ctid -> id` entries in a backend-local hash table. HNSW still generates
candidates through the normal pgvector index scan, but false candidates are
discarded by a cheap C hash lookup instead of being returned to the PostgreSQL
executor for heap fetch + SQL qual evaluation. Passing candidates use the cached
`id`, so the search path does not fetch heap tuples in the hot-cache case.

This is not yet a production invalidation design. Updates, VACUUM movement,
table rewrites, and changing predicates require cache rebuilds. The purpose is
to measure the potential of an index-level/filter-metadata cache before
integrating it with relcache invalidation or planner/executor routing.

### 400-Query BFS-Layout Results

Configuration:

```text
table = amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx
index = amazon_grocery_reviews_10m_pgvector_vc10m_seed_bfs_hnsw
queries = 400 Amazon-C4 anchors
k = 10
hnsw.ef_search = 1000
hnsw.iterative_scan = strict_order
hnsw.max_scan_tuples = 200000
candidate_limit = 200000
```

`grocery_long500`:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_metadata_cache_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx \
  --index amazon_grocery_reviews_10m_pgvector_vc10m_seed_bfs_hnsw \
  --filter-name grocery_long500 \
  --queries 400 \
  --k 10 \
  --candidate-limit 200000 \
  --ef-search 1000 \
  --iterative-scan strict_order \
  --max-scan-tuples 200000 \
  --out results/hybrid_vector_db/metadata_cache_bfs_grocery_long500_q400.csv
```

| Path | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| Standard pgvector filtered HNSW on BFS index | 71.36 | 74.91 | 132.27 |
| BFS index + metadata cache | 37.28 | 37.62 | 74.40 |

| Metric | Value |
| --- | ---: |
| Same ordered IDs | 400 / 400 |
| Mean speedup | 2.004x |
| Median speedup | 1.925x |
| P05 speedup | 1.328x |
| P95 speedup | 2.804x |
| Cache rows | 21,317 |
| Cache build time | 1150.57 ms |
| Mean standard returned heap TIDs | 11,606.6 |
| Mean cache candidate checks | 11,606.6 |
| Mean cache matches | 7.78 |

`helpful_ge20`:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_metadata_cache_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx \
  --index amazon_grocery_reviews_10m_pgvector_vc10m_seed_bfs_hnsw \
  --filter-name helpful_ge20 \
  --queries 400 \
  --k 10 \
  --candidate-limit 200000 \
  --ef-search 1000 \
  --iterative-scan strict_order \
  --max-scan-tuples 200000 \
  --out results/hybrid_vector_db/metadata_cache_bfs_helpful_ge20_q400.csv
```

| Path | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| Standard pgvector filtered HNSW on BFS index | 45.03 | 45.21 | 89.86 |
| BFS index + metadata cache | 21.57 | 19.15 | 46.53 |

| Metric | Value |
| --- | ---: |
| Same ordered IDs | 400 / 400 |
| Mean speedup | 2.148x |
| Median speedup | 2.119x |
| P05 speedup | 1.467x |
| P95 speedup | 2.828x |
| Cache rows | 60,689 |
| Cache build time | 1120.89 ms |
| Mean standard returned heap TIDs | 5,344.1 |
| Mean cache candidate checks | 5,344.1 |
| Mean cache matches | 8.42 |

### Original-Layout Contrast

On the original 10M table and original HNSW index, `grocery_long500` also
benefits from metadata caching:

```text
table = amazon_grocery_reviews_10m_pgvector
index = amazon_grocery_reviews_10m_pgvector_embedding_hnsw_idx
```

| Path | Mean ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| Standard pgvector filtered HNSW | 123.23 | 129.18 | 225.04 |
| Metadata cache | 79.49 | 90.69 | 126.66 |

| Metric | Value |
| --- | ---: |
| Same ordered IDs | 400 / 400 |
| Mean speedup | 1.817x |
| Median speedup | 1.567x |
| Cache build time | 116.69 ms |

Combining the previous BFS physical placement with the metadata cache gives a
larger end-to-end improvement on this filtered workload:

```text
original standard pgvector grocery_long500 mean: 123.23 ms
BFS-layout standard pgvector grocery_long500 mean: 71.36 ms
BFS-layout + metadata cache grocery_long500 mean: 37.28 ms
```

So the combined mean improvement over original standard pgvector is about
`3.31x`, while the metadata cache still gives about `2.00x` over the already
optimized BFS-layout path.

### Interpretation

- This exceeds the previous `1.25x` BFS physical-layout gain for sparse filtered
  workloads.
- The speedup comes from replacing thousands of executor heap validations per
  query with cheap C hash checks. For `grocery_long500`, standard pgvector
  returned about `11.6k` heap TIDs to the executor per query but only about
  `7.8` of those candidate TIDs matched the hot filter cache.
- The cache does not reduce HNSW graph traversal itself. It attacks the
  PostgreSQL compatibility overhead highlighted by Lu et al. 2026: candidate
  validation requires heap access and SQL qual evaluation when metadata is only
  in the heap.
- For production, this should become a PostgreSQL-aware predicate cache:
  generated from btree/bitmap scans, tied to relation invalidation, aware of
  MVCC/update epochs, and selected by a cost model. The prototype proves the
  performance upside is large enough to justify that design.

### Real Amazon-C4 Query-Derived Filter Workload

The fixed-filter results above are intentionally a hot-filter upper bound. A
more realistic check uses the 400 Amazon-C4 query anchors and generates a
different structured predicate from each query's matched row metadata:

```text
rating = ori_rating
AND price bucket from the query anchor
AND item_rating_number popularity bucket from the query anchor
```

This produces a mixed faceted workload rather than one shared filter. The
benchmark script is:

```bash
.venv/bin/python experiments/hybrid_vector_db/scripts/pgvector_c4_query_filter_cache_benchmark.py \
  --table amazon_grocery_reviews_10m_pgvector_vector_clustered_10m_bfsidx \
  --index amazon_grocery_reviews_10m_pgvector_vc10m_seed_bfs_hnsw \
  --query-csv results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv \
  --queries 400 \
  --mode mixed \
  --max-cache-rows 200000 \
  --k 10 \
  --candidate-limit 200000 \
  --ef-search 1000 \
  --iterative-scan strict_order \
  --max-scan-tuples 200000 \
  --out results/hybrid_vector_db/c4_query_filter_cache_bfs_mixed_q400_admit200k_2conn.csv
```

Implementation note: the experiment uses short pgvector-side cache keys for the
15 generated mixed predicates. A dynamic `sql:<predicate>` prototype exposed a
backend-local state bug when mixed with standard HNSW scans in the same
PostgreSQL backend, so the measurement uses separate standard/cache
connections. This does not change the cache search semantics, but it should be
fixed before any production-style integration.

Workload shape:

| Metric | Value |
| --- | ---: |
| Real Amazon-C4 query anchors | 400 |
| Distinct generated predicates | 15 |
| Cached predicates under `max-cache-rows=200000` | 6 |
| Queries using a cached predicate | 59 / 400 |
| Total cached TIDs | 640,716 |
| Cache build time | 7753.69 ms |

End-to-end workload result:

| Path | Mean ms/query | P50 ms | P95 ms |
| --- | ---: | ---: | ---: |
| Standard pgvector on BFS index | 20.82 | 20.62 | 41.72 |
| Routed cache path, hot cache | 19.68 | 19.29 | 38.91 |

| Metric | Value |
| --- | ---: |
| Same ordered IDs | 400 / 400 |
| Hot-cache total speedup | 1.058x |
| Mean per-query hot speedup | 1.060x |
| Median per-query hot speedup | 1.000x |
| Amortized speedup including build | 0.533x |

For the 59 queries that actually used the cache:

| Metric | Value |
| --- | ---: |
| Cached-query mean speedup | 1.405x |
| Cached-query median speedup | 1.424x |
| Cached-query P95 speedup | 2.179x |
| Cached-query standard mean | 23.90 ms |
| Cached-query cache mean | 16.12 ms |

Interpretation:

- The user's concern is valid. The earlier `~2x` result depends on many queries
  sharing the same sparse hot filter. It is an upper-bound result for reusable
  predicates, not a representative result for arbitrary real query filters.
- On this Amazon-C4-derived mixed workload, most generated filters are broad:
  many have 460k to 990k passing rows. A sane admission policy should not cache
  those as backend-local TID hashes.
- With `max-cache-rows=200000`, only 59/400 queries use the cache. Those cached
  queries improve by about `1.4x`, but the whole workload improves only `1.06x`
  with a hot cache and loses once build cost is included.
- This shifts the research direction: metadata cache is valuable only when
  predicates are both selective and reused. For arbitrary Amazon-C4 query
  filters, the system needs predicate-frequency tracking, cache admission, and
  possibly shared/materialized bitmap caches rather than per-backend TID hashes.

### Compact Membership Cache Prototype

The exact TID-hash cache improves cached queries but has poor coverage under
real Amazon-C4-derived predicates. The next prototype tested compressed
membership structures that can cache broad predicates without storing every TID
as a hash entry. The correctness model is conservative: the cache only rejects
candidates. Any candidate that may pass is joined back to the heap and rechecked
with the original SQL predicate, so Bloom/page false positives affect latency,
not result validity.

Implemented pgvector-side functions:

- `vector_hnsw_metadata_page_cache_build(index regclass, filter text)`
- `vector_hnsw_metadata_page_filter_candidates(index regclass, query vector,
  candidate_limit int4, filter text)`
- `vector_hnsw_metadata_bloom_cache_build(index regclass, filter text)`
- `vector_hnsw_metadata_bloom_filter_candidates(index regclass, query vector,
  candidate_limit int4, filter text)`
- `vector_hnsw_metadata_bloom_filter_candidates_limited(index regclass, query
  vector, candidate_limit int4, match_limit int4, filter text)`

The Bloom prototype uses about 10 bits per passing tuple and 7 hash probes. The
limited variant scans HNSW candidates until either `candidate_limit` is reached
or `match_limit` Bloom-positive candidates have been emitted for SQL recheck.

Page-level summary result on a 20-query smoke run:

| Cache | Queries cached | Same ordered IDs | Hot total speedup | Mean cache candidates | Mean cache matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Page bitmap, `candidate_limit=200000` | 20 / 20 | 20 / 20 | 0.364x | 19,102.65 | 8,307.95 |

The page bitmap raises cache coverage, but it is too coarse: many candidates
survive to SQL recheck, so it is slower than standard filtered HNSW.

Bloom results on the full 400-query Amazon-C4-derived mixed workload:

| Cache | Match limit | Queries cached | Same ordered IDs | Hot total speedup | Mean speedup | Mean candidates | Mean matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bloom | 100 | 400 / 400 | 400 / 400 | 1.109x | 1.347x | 1,889.40 | 99.86 |
| Bloom | 50 | 400 / 400 | 400 / 400 | 1.502x | 1.645x | 944.38 | 49.98 |
| Bloom | 25 | 400 / 400 | 398 / 400 | 1.847x | 1.864x | 467.48 | 25.00 |

The best correct configuration in this sweep is Bloom with
`match_limit=50`: it caches all 400 queries and improves the hot-cache total
runtime by `1.50x`, exceeding the `1.25x` physical-layout-only improvement.
However, `match_limit=25` shows the risk of aggressive early stopping: it is
faster, but 2/400 queries no longer match the standard top-k output.

Build-cost caveat:

| Cache | Build time | Hot total speedup | Amortized speedup including build |
| --- | ---: | ---: | ---: |
| Bloom, match 100 | 23,564.60 ms | 1.109x | 0.335x |
| Bloom, match 50 | 21,031.32 ms | 1.502x | 0.378x |
| Bloom, match 25 | 23,153.90 ms | 1.847x | 0.365x |

For the correct `match_limit=50` run, the 400-query hot-path saving is about
3.55 seconds, while cache construction costs about 21.03 seconds. This means
the cache must be reused across roughly six equivalent 400-query batches before
the build cost breaks even. Production integration should therefore use shared
cache state, predicate-frequency admission, and invalidation instead of building
backend-local caches opportunistically.
