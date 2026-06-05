from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def timed(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f"{label}: {time.perf_counter() - t0:.2f}s", flush=True)
    return result


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    value = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", value).strip()


def parse_price(value) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    if isinstance(value, (int, float)):
        return float(value), True
    text = str(value)
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not m:
        return 0.0, False
    return float(m.group(1)), True


def stable_hash(text: str, mod: int) -> int:
    # Deterministic enough for IDs/categories without Python's randomized hash.
    h = 1469598103934665603
    for b in text.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % mod


def load_meta(meta_path: Path) -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = row.get("parent_asin") or row.get("asin")
            if not asin:
                continue
            price, has_price = parse_price(row.get("price"))
            categories = row.get("categories") or row.get("category") or []
            if isinstance(categories, list):
                main_category = clean_text(categories[0] if categories else row.get("main_category", ""))
            else:
                main_category = clean_text(categories)
            meta[str(asin)] = {
                "store": clean_text(row.get("store", "")),
                "main_category": main_category,
                "category_id": stable_hash(main_category, 10000),
                "price": price,
                "has_price": has_price,
                "item_avg_rating": float(row.get("average_rating") or 0.0),
                "item_rating_number": int(row.get("rating_number") or 0),
                "title": clean_text(row.get("title", "")),
            }
            if line_no % 200000 == 0:
                print(f"loaded meta {line_no}", flush=True)
    print(f"meta rows={len(meta)}", flush=True)
    return meta


def iter_review_texts(review_path: Path, limit: int):
    with review_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = clean_text(row.get("title", ""))
            text = clean_text(row.get("text", ""))
            yield f"{title} {text}".strip()


def fit_text_model(review_path: Path, sample_rows: int, max_features: int, dim: int):
    print(f"fitting TF-IDF sample_rows={sample_rows} max_features={max_features} dim={dim}", flush=True)
    texts = list(iter_review_texts(review_path, sample_rows))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    x = timed("fit_transform tfidf", lambda: vectorizer.fit_transform(texts))
    svd = TruncatedSVD(n_components=dim, random_state=17)
    timed("fit svd", lambda: svd.fit(x))
    print(f"tfidf vocab={len(vectorizer.vocabulary_)} explained_var={svd.explained_variance_ratio_.sum():.4f}", flush=True)
    return vectorizer, svd


def write_outputs(
    review_path: Path,
    out_csv: Path,
    out_fbin: Path,
    meta: dict[str, dict[str, object]],
    vectorizer,
    svd,
    rows: int,
    batch_size: int,
    dim: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fbin.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
        "review_text_len",
        "store",
        "main_category",
        "category_id",
        "price",
        "has_price",
        "item_avg_rating",
        "item_rating_number",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as csv_f, out_fbin.open("wb") as vec_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        vec_f.write(struct.pack("ii", rows, dim))
        batch_texts: list[str] = []
        batch_rows: list[dict[str, object]] = []
        written = 0

        def flush_batch() -> None:
            nonlocal written, batch_texts, batch_rows
            if not batch_rows:
                return
            x = vectorizer.transform(batch_texts)
            emb = svd.transform(x).astype("float32", copy=False)
            emb = normalize(emb, norm="l2", copy=False).astype("float32", copy=False)
            writer.writerows(batch_rows)
            vec_f.write(np.ascontiguousarray(emb).tobytes())
            written += len(batch_rows)
            if written % 100000 == 0:
                print(f"written {written}/{rows}", flush=True)
            batch_texts = []
            batch_rows = []

        with review_path.open("r", encoding="utf-8") as f:
            for line in f:
                if written + len(batch_rows) >= rows:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = str(row.get("parent_asin") or row.get("asin") or "")
                m = meta.get(asin, {})
                title = clean_text(row.get("title", ""))
                text = clean_text(row.get("text", ""))
                review_text = f"{title} {text}".strip()
                batch_texts.append(review_text)
                batch_rows.append(
                    {
                        "id": written + len(batch_rows),
                        "user_id": str(row.get("user_id") or ""),
                        "parent_asin": asin,
                        "rating": float(row.get("rating") or 0.0),
                        "timestamp": int(row.get("timestamp") or 0),
                        "verified_purchase": bool(row.get("verified_purchase") or False),
                        "helpful_vote": int(row.get("helpful_vote") or 0),
                        "review_text_len": len(review_text),
                        "store": m.get("store", ""),
                        "main_category": m.get("main_category", ""),
                        "category_id": int(m.get("category_id", 0)),
                        "price": float(m.get("price", 0.0)),
                        "has_price": bool(m.get("has_price", False)),
                        "item_avg_rating": float(m.get("item_avg_rating", 0.0)),
                        "item_rating_number": int(m.get("item_rating_number", 0)),
                    }
                )
                if len(batch_rows) >= batch_size:
                    flush_batch()
        flush_batch()
    print(f"wrote csv={out_csv} fbin={out_fbin}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-jsonl", type=Path, default=Path("data/amazon_reviews_2023/raw_reviews/Grocery_and_Gourmet_Food.jsonl"))
    parser.add_argument("--meta-jsonl", type=Path, default=Path("data/amazon_reviews_2023/raw_meta_extra/meta_Grocery_and_Gourmet_Food.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"))
    parser.add_argument("--out-fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--sample-rows", type=int, default=300_000)
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    meta = timed("load meta", lambda: load_meta(args.meta_jsonl))
    vectorizer, svd = timed("fit text model", lambda: fit_text_model(args.review_jsonl, args.sample_rows, args.max_features, args.dim))
    timed(
        "write outputs",
        lambda: write_outputs(
            args.review_jsonl,
            args.out_csv,
            args.out_fbin,
            meta,
            vectorizer,
            svd,
            args.rows,
            args.batch_size,
            args.dim,
        ),
    )


if __name__ == "__main__":
    main()
