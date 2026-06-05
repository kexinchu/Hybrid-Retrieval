from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from common_pg import pg_config_from_env, require_psycopg


PREDICATES = [
    ("p92_verified", "verified_purchase"),
    ("p78_rating_ge4", "rating >= 4"),
    ("p63_has_price", "has_price"),
    ("p50_item_popular", "item_rating_number >= 1000"),
    ("p26_helpful", "helpful_vote >= 1"),
    ("p14_price_le10", "has_price AND price <= 10"),
    ("p059_long_review", "review_text_len >= 500"),
    ("p036_grocery_subcat", "main_category = 'Grocery'"),
    ("p000026_pantry", "main_category = 'Pantry Staples'"),
]


INDEXES = [
    ("item_rating_number", "item_rating_number"),
    ("helpful_vote", "helpful_vote"),
    ("review_text_len", "review_text_len"),
    ("main_category", "main_category"),
    ("main_category_rating", "main_category, rating"),
    ("main_category_price", "main_category, has_price, price"),
    ("item_rating_number_rating", "item_rating_number, rating"),
]


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    print(f"{label}: {time.perf_counter() - start:.2f}s", flush=True)
    return result


def walk_plan(node: dict) -> list[dict]:
    nodes = [node]
    for child in node.get("Plans", []):
        nodes.extend(walk_plan(child))
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="amazon_grocery_reviews_10m")
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/amazon_grocery_10m_sql_index_sanity.csv"))
    args = parser.parse_args()

    require_psycopg()
    import psycopg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {args.table}")
            total_rows = int(cur.fetchone()[0])
            print(f"table={args.table} rows={total_rows}", flush=True)

            for name, cols in INDEXES:
                timed(
                    f"index {name}",
                    lambda name=name, cols=cols: cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {args.table}_{name}_idx ON {args.table} ({cols})"
                    ),
                )
            timed("analyze", lambda: cur.execute(f"ANALYZE {args.table}"))

            for pname, predicate in PREDICATES:
                print(f"explain {pname}: {predicate}", flush=True)
                cur.execute(
                    f"""
                    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                    SELECT count(*) FROM {args.table} WHERE {predicate}
                    """
                )
                plan_doc = cur.fetchone()[0][0]
                plan = plan_doc["Plan"]
                nodes = walk_plan(plan)
                node_types = sorted({n["Node Type"] for n in nodes})
                index_names = sorted(
                    {
                        n["Index Name"]
                        for n in nodes
                        if "Index Name" in n
                    }
                )
                scan_types = [n["Node Type"] for n in nodes if "Scan" in n["Node Type"]]
                rows_removed = sum(int(n.get("Rows Removed by Filter", 0)) for n in nodes)
                actual_rows = max(int(n.get("Actual Rows", 0)) for n in nodes)
                execution_ms = float(plan_doc["Execution Time"])

                cur.execute(f"SELECT count(*) FROM {args.table} WHERE {predicate}")
                count = int(cur.fetchone()[0])
                rows.append(
                    {
                        "predicate_name": pname,
                        "predicate": predicate,
                        "count": count,
                        "selectivity": count / total_rows,
                        "execution_ms": execution_ms,
                        "scan_types": ";".join(scan_types),
                        "node_types": ";".join(node_types),
                        "index_names": ";".join(index_names),
                        "actual_rows_max_node": actual_rows,
                        "rows_removed_by_filter": rows_removed,
                    }
                )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
