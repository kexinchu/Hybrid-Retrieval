from __future__ import annotations

from common_pg import pg_config_from_env, require_psycopg


def main() -> None:
    require_psycopg()
    import psycopg

    table = "amazon_grocery_reviews_10m"
    predicates: list[tuple[str, str]] = [
        ("verified_purchase", "verified_purchase"),
        ("rating_ge4", "rating >= 4"),
        ("rating_eq5", "rating = 5"),
        ("has_price", "has_price"),
        ("popular_ge1000", "item_rating_number >= 1000"),
        ("popular_ge5000", "item_rating_number >= 5000"),
        ("helpful_ge1", "helpful_vote >= 1"),
        ("helpful_ge2", "helpful_vote >= 2"),
        ("helpful_ge5", "helpful_vote >= 5"),
        ("long_ge200", "review_text_len >= 200"),
        ("long_ge500", "review_text_len >= 500"),
        ("long_ge1000", "review_text_len >= 1000"),
        ("price_le10", "has_price AND price <= 10"),
        ("price_le5", "has_price AND price <= 5"),
        ("price_le3", "has_price AND price <= 3"),
        ("price_10_20", "has_price AND price > 10 AND price <= 20"),
        ("grocery", "main_category = 'Grocery'"),
        ("pantry", "main_category = 'Pantry Staples'"),
        ("rating_ge4_price_le10", "has_price AND price <= 10 AND rating >= 4"),
        ("rating5_price_le10", "has_price AND price <= 10 AND rating = 5"),
        ("verified_price_le10", "verified_purchase AND has_price AND price <= 10"),
        ("verified_helpful", "verified_purchase AND helpful_vote >= 1"),
        ("helpful_long500", "helpful_vote >= 1 AND review_text_len >= 500"),
        ("grocery_rating5", "main_category = 'Grocery' AND rating = 5"),
        ("grocery_price_le10", "main_category = 'Grocery' AND has_price AND price <= 10"),
        ("grocery_helpful", "main_category = 'Grocery' AND helpful_vote >= 1"),
        ("grocery_long500", "main_category = 'Grocery' AND review_text_len >= 500"),
        ("grocery_popular", "main_category = 'Grocery' AND item_rating_number >= 1000"),
    ]
    for value in [1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 30, 50]:
        predicates.append((f"price_le_{value:g}", f"has_price AND price <= {value}"))
    for value in [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        predicates.append((f"item_rating_number_ge_{value}", f"item_rating_number >= {value}"))
    for value in [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]:
        predicates.append((f"review_text_len_ge_{value}", f"review_text_len >= {value}"))
    for value in [1, 2, 3, 5, 10, 20, 50]:
        predicates.append((f"helpful_vote_ge_{value}", f"helpful_vote >= {value}"))

    with psycopg.connect(pg_config_from_env().conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            total = int(cur.fetchone()[0])
            print("name,count,selectivity,predicate")
            for name, predicate in predicates:
                cur.execute(f"SELECT count(*) FROM {table} WHERE {predicate}")
                count = int(cur.fetchone()[0])
                print(f"{name},{count},{count / total:.8f},{predicate}")


if __name__ == "__main__":
    main()
