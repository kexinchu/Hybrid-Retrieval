CREATE INDEX IF NOT EXISTS bfs_helpful_vote_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (helpful_vote);

CREATE INDEX IF NOT EXISTS bfs_item_rating_number_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (item_rating_number);

CREATE INDEX IF NOT EXISTS bfs_rating_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (rating);

CREATE INDEX IF NOT EXISTS bfs_review_text_len_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (review_text_len);

CREATE INDEX IF NOT EXISTS bfs_price_rating_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (has_price, price, rating);

CREATE INDEX IF NOT EXISTS bfs_main_category_rating_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (main_category, rating);

CREATE INDEX IF NOT EXISTS bfs_cat_helpful_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (main_category, helpful_vote);

CREATE INDEX IF NOT EXISTS bfs_cat_review_len_idx
ON public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs USING btree (main_category, review_text_len);

ANALYZE public.amazon_grocery_reviews_10m_pgvector_samegraph_bfs;
