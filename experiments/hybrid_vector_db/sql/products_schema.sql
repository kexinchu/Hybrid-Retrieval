CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    store TEXT,
    main_category TEXT,
    price DOUBLE PRECISION,
    average_rating DOUBLE PRECISION,
    rating_number INTEGER,
    has_price BOOLEAN NOT NULL,
    has_description BOOLEAN NOT NULL,
    text_length INTEGER NOT NULL,
    embedding vector({dim}) NOT NULL
);

CREATE INDEX products_store_idx ON products (store);
CREATE INDEX products_rating_idx ON products (average_rating);
CREATE INDEX products_rating_number_idx ON products (rating_number);
CREATE INDEX products_price_idx ON products (price) WHERE has_price;
CREATE INDEX products_store_rating_idx ON products (store, average_rating, rating_number);

ANALYZE products;
