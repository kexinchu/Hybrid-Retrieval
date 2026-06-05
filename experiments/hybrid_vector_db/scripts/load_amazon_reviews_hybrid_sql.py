from __future__ import annotations

import argparse
import time
from pathlib import Path

from common_pg import pg_config_from_env, require_psycopg


def timed(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f"{label}: {time.perf_counter() - t0:.2f}s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"))
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            if args.drop:
                timed("drop", lambda: cur.execute(f"DROP TABLE IF EXISTS {args.table}"))
            timed(
                "create",
                lambda: cur.execute(
                    f"""
                    CREATE UNLOGGED TABLE IF NOT EXISTS {args.table} (
                        id bigint PRIMARY KEY,
                        user_id text,
                        parent_asin text,
                        rating double precision,
                        timestamp bigint,
                        verified_purchase boolean,
                        helpful_vote int,
                        review_text_len int,
                        store text,
                        main_category text,
                        category_id int,
                        price double precision,
                        has_price boolean,
                        item_avg_rating double precision,
                        item_rating_number int
                    )
                    """
                ),
            )
            cur.execute(f"SELECT count(*) FROM {args.table}")
            count = int(cur.fetchone()[0])
            if count == 0:
                def copy_csv() -> None:
                    with args.csv.open("rb") as f:
                        with cur.copy(
                            f"""
                            COPY {args.table}
                            FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                            """
                        ) as copy:
                            while chunk := f.read(1024 * 1024):
                                copy.write(chunk)

                timed("copy csv", copy_csv)
            else:
                print(f"table already non-empty count={count}", flush=True)
            indexes = [
                ("rating", "rating"),
                ("verified", "verified_purchase"),
                ("timestamp", "timestamp"),
                ("category", "category_id"),
                ("price", "price"),
                ("has_price", "has_price"),
                ("asin", "parent_asin"),
                ("item_rating_number", "item_rating_number"),
                ("helpful_vote", "helpful_vote"),
                ("review_text_len", "review_text_len"),
                ("main_category", "main_category"),
                ("category_rating", "category_id, rating"),
                ("verified_rating", "verified_purchase, rating"),
                ("price_rating", "has_price, price, rating"),
                ("main_category_rating", "main_category, rating"),
                ("main_category_price", "main_category, has_price, price"),
                ("item_rating_number_rating", "item_rating_number, rating"),
            ]
            for name, cols in indexes:
                timed(
                    f"index {name}",
                    lambda name=name, cols=cols: cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {args.table}_{name}_idx ON {args.table} ({cols})"
                    ),
                )
            timed("analyze", lambda: cur.execute(f"ANALYZE {args.table}"))
            cur.execute(f"SELECT count(*) FROM {args.table}")
            print("rows", cur.fetchone()[0], flush=True)
            cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{args.table}'))")
            print("size", cur.fetchone()[0], flush=True)


if __name__ == "__main__":
    main()
