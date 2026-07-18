from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_c4_queries(path: Path, limit: int | None) -> dict[tuple[str, str], dict[str, str]]:
    queries: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["item_id"], row["user_id"])
            if key not in queries:
                queries[key] = row
            if limit is not None and len(queries) >= limit:
                break
    return queries


def load_c4_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_qids: set[str] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["qid"] in seen_qids:
                continue
            seen_qids.add(row["qid"])
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def make_match(c4: dict[str, str], row: dict[str, str], match_type: str) -> dict[str, str]:
    return {
        "query_no": c4["qid"],
        "query_id": row["id"],
        "item_id": c4["item_id"],
        "user_id": c4["user_id"],
        "ori_rating": c4.get("ori_rating", ""),
        "query": c4.get("query", ""),
        "ori_review": c4.get("ori_review", ""),
        "parent_asin": row["parent_asin"],
        "table_rating": row["rating"],
        "main_category": row["main_category"],
        "price": row["price"],
        "has_price": row["has_price"],
        "item_rating_number": row["item_rating_number"],
        "review_text_len": row["review_text_len"],
        "match_type": match_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Amazon-C4 test queries to local Amazon Reviews pgvector row ids."
    )
    parser.add_argument("--c4-test", type=Path, default=Path("data/amazon_c4/test.csv"))
    parser.add_argument(
        "--hybrid-csv",
        type=Path,
        default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries.csv"))
    parser.add_argument("--c4-limit", type=int)
    parser.add_argument("--max-matches", type=int, default=200)
    parser.add_argument(
        "--fallback-item-only",
        action="store_true",
        help="If exact (item_id,user_id) matches are insufficient, add C4 item_id-only matches.",
    )
    parser.add_argument("--progress-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    c4_rows = load_c4_rows(args.c4_test, args.c4_limit)
    c4_by_key = {(row["item_id"], row["user_id"]): row for row in c4_rows}
    c4_by_item = {row["item_id"]: row for row in c4_rows}
    if not c4_by_key:
        raise RuntimeError(f"no Amazon-C4 rows loaded from {args.c4_test}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matches: list[dict[str, str]] = []
    with args.hybrid_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        seen_qids: set[str] = set()
        for row_no, row in enumerate(reader, start=1):
            key = (row["parent_asin"], row["user_id"])
            c4 = c4_by_key.get(key)
            if c4 is not None and c4["qid"] not in seen_qids:
                seen_qids.add(c4["qid"])
                matches.append(make_match(c4, row, "item_user"))
                if len(matches) >= args.max_matches:
                    break
            if args.progress_rows and row_no % args.progress_rows == 0:
                print(f"scanned={row_no} matches={len(matches)}", flush=True)

    if not matches:
        raise RuntimeError("no C4 queries matched local hybrid CSV by (parent_asin, user_id)")

    if args.fallback_item_only and len(matches) < args.max_matches:
        print(f"exact matches={len(matches)}; scanning again for item-only fallback", flush=True)
        seen_qids = {row["query_no"] for row in matches}
        with args.hybrid_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row_no, row in enumerate(reader, start=1):
                c4 = c4_by_item.get(row["parent_asin"])
                if c4 is not None and c4["qid"] not in seen_qids:
                    seen_qids.add(c4["qid"])
                    matches.append(make_match(c4, row, "item_only"))
                    if len(matches) >= args.max_matches:
                        break
                if args.progress_rows and row_no % args.progress_rows == 0:
                    print(f"fallback_scanned={row_no} matches={len(matches)}", flush=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
        writer.writeheader()
        writer.writerows(matches)

    print(f"wrote {args.out} matches={len(matches)}", flush=True)
    for row in matches[:5]:
        print(
            f"qid={row['query_no']} id={row['query_id']} item={row['item_id']} "
            f"category={row['main_category']} query={row['query'][:120]}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
