from __future__ import annotations

import argparse
import csv
import hashlib
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "but",
    "by",
    "can",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "want",
    "with",
}

PRODUCT_PATTERNS = [
    re.compile(r"\b(?:i\s+)?(?:need|want|am looking for|i'm looking for|looking for|find|buy|get)\s+(?:to\s+find\s+|to\s+buy\s+|some\s+|a\s+|an\s+|the\s+)?(?P<item>[a-z0-9][a-z0-9 /&+-]{1,55}?)(?:\s+that|\s+which|\s+for|\s+with|\.|,|$)"),
    re.compile(r"\b(?:it should be|should be)\s+(?:a\s+|an\s+)?(?P<item>[a-z0-9][a-z0-9 /&+-]{1,45}?)(?:\s+that|\s+which|\s+for|\s+with|\.|,|$)"),
]

FIELD_PATTERNS = [
    ("price", re.compile(r"\b(?:cheap|affordable|budget|bargain|inexpensive|expensive|price|worth (?:the )?(?:money|price|penny)|high price|low price|cost)\b")),
    ("rating_sentiment", re.compile(r"\b(?:excellent|perfect|great|best|amazing|love|loved|favorite|highly recommend|worth|no complaints|exceeds expectations)\b")),
    ("color", re.compile(r"\b(?:black|white|red|blue|green|yellow|pink|purple|orange|grey|gray|brown|beige|silver|gold|clear|transparent|color|colour|space grey)\b")),
    ("size_fit", re.compile(r"\b(?:small|large|medium|compact|portable|lightweight|heavy|fit|fits|size|sized|slim|wide|narrow|inch|inches|2t|12-18 month|petite)\b")),
    ("material", re.compile(r"\b(?:cotton|polyester|wood|wooden|metal|steel|plastic|silicone|leather|fabric|glass|ceramic|rubber|fur)\b")),
    ("compatibility", re.compile(r"\b(?:compatible|works with|macbook|ipad|iphone|android|apple|usb c|usb-c|cpap|rv|apartment|laptop|phone|galaxy|kindle|xbox|playstation|ps5|switch)\b")),
    ("audience", re.compile(r"\b(?:kids|children|child|toddler|baby|babies|dog|dogs|cat|cats|pet|pets|women|men|wife|husband|grandkids|students|teachers)\b")),
    ("occasion_use", re.compile(r"\b(?:halloween|christmas|birthday|wedding|summer|winter|beach|travel|camping|office|garden|home|bath|kitchen|laundry|sleep|school|work|yard)\b")),
    ("health_safety", re.compile(r"\b(?:healthy|toxic-free|non-toxic|sensitive|eczema|pain|support|air quality|dust|allergy|allergies|safe|safety)\b")),
    ("feature", re.compile(r"\b(?:waterproof|water resistant|rechargeable|automatic|adjustable|foldable|sturdy|soft|comfortable|durable|easy to|easy-to|lather|scent|pocket|kickstand|charger|ethernet|shade|support|dryer)\b")),
    ("negative_constraint", re.compile(r"\b(?:not too|doesn't|does not|won't|will not|without|no\s+[a-z]+|avoid|shouldn't|should not)\b")),
]


@dataclass(frozen=True)
class Predicate:
    field: str
    value: str
    span: tuple[int, int]


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_phrase(text: str, max_tokens: int = 5) -> str:
    text = normalize_text(text)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens[:max_tokens]) or "unknown"


def extract_product(text: str) -> Predicate | None:
    for pattern in PRODUCT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        item = canonical_phrase(match.group("item"))
        if len(item) < 2 or item in {"something", "product", "thing", "some"}:
            return None
        return Predicate("category", item, match.span("item"))
    return None


def extract_field_predicates(text: str) -> list[Predicate]:
    preds: list[Predicate] = []
    for field, pattern in FIELD_PATTERNS:
        for match in pattern.finditer(text):
            value = canonical_phrase(match.group(0), max_tokens=4)
            preds.append(Predicate(field, value, match.span()))
    return preds


def extract_predicates(query: str) -> list[Predicate]:
    text = normalize_text(query)
    preds = []
    product = extract_product(text)
    if product:
        preds.append(product)
    preds.extend(extract_field_predicates(text))
    dedup: dict[tuple[str, str], Predicate] = {}
    for pred in preds:
        dedup.setdefault((pred.field, pred.value), pred)
    return list(dedup.values())


def medium_predicate_key(pred: Predicate) -> str:
    if pred.field == "category":
        return f"category={pred.value}"
    if pred.field == "price":
        if pred.value in {"cheap", "affordable", "budget", "bargain", "inexpensive", "low price"}:
            return "price=low"
        if pred.value in {"expensive", "high price"}:
            return "price=high"
        return "price=mentioned"
    if pred.field == "rating_sentiment":
        return "rating_sentiment=positive"
    if pred.field == "negative_constraint":
        return "negative_constraint=present"
    if pred.field == "health_safety":
        return "health_safety=present"
    if pred.field == "feature":
        return "feature=present"
    if pred.field == "size_fit":
        return "size_fit=present"
    return f"{pred.field}={pred.value}"


def mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for idx in range(max(0, start), min(len(chars), end)):
            chars[idx] = " "
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def vector_terms(text: str) -> list[str]:
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    return [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 2]


def bucket_terms(terms: list[str], n: int) -> str:
    return " ".join(sorted(Counter(terms).keys())[:n]) or "empty"


def simhash_bucket(terms: list[str], bits: int) -> str:
    if not terms:
        return "empty"
    weights = [0] * 64
    for term, count in Counter(terms).items():
        h = int(hashlib.sha1(term.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            weights[bit] += count if (h >> bit) & 1 else -count
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    shift = 64 - bits
    return f"{value >> shift:0{(bits + 3) // 4}x}"


def make_keys(query: str) -> dict[str, str]:
    text = normalize_text(query)
    preds = extract_predicates(query)
    pred_fields = sorted({p.field for p in preds})
    pred_medium = sorted({medium_predicate_key(p) for p in preds})
    pred_fine = sorted(f"{p.field}={p.value}" for p in preds)
    vector_text = mask_spans(text, [p.span for p in preds])
    terms = vector_terms(vector_text)
    product = next((p.value for p in preds if p.field == "category"), "unknown")
    return {
        "sql_coarse": "|".join(pred_fields) or "no_filter",
        "sql_medium": "|".join(pred_medium) or "no_filter",
        "sql_fine": "|".join(pred_fine) or "no_filter",
        "vector_coarse": product,
        "vector_medium": f"{product}|{simhash_bucket(terms, 16)}",
        "vector_fine": " ".join(terms),
        "predicates": "; ".join(pred_fine),
        "vector_text": vector_text,
    }


def duplicate_stats(keys: list[str]) -> dict[str, float | int]:
    counts = Counter(keys)
    duplicate_requests = sum(count for count in counts.values() if count > 1)
    repeated_extra = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "requests": len(keys),
        "unique_subqueries": len(counts),
        "duplicate_request_count": duplicate_requests,
        "duplicate_request_ratio": duplicate_requests / len(keys),
        "cache_hit_upper_bound_count": repeated_extra,
        "cache_hit_upper_bound_ratio": repeated_extra / len(keys),
        "largest_group": max(counts.values()) if counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/amazon_c4/test.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_trace_cache_summary.csv"))
    parser.add_argument("--out-detail", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_trace_cache_detail.csv"))
    parser.add_argument("--topn", type=int, default=12)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.input.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys = make_keys(row["query"])
            rows.append({**row, **keys})

    key_names = ["sql_coarse", "sql_medium", "sql_fine", "vector_coarse", "vector_medium", "vector_fine"]
    summary_rows = []
    for key_name in key_names:
        stats = duplicate_stats([row[key_name] for row in rows])
        branch = "sql_filter" if key_name.startswith("sql") else "vector_search"
        granularity = key_name.split("_", 1)[1]
        summary_rows.append({"branch": branch, "granularity": granularity, **stats})

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with args.out_detail.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["qid", "item_id", "query", "predicates", "vector_text", *key_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    print("summary")
    for row in summary_rows:
        print(
            f"{row['branch']},{row['granularity']},unique={row['unique_subqueries']},"
            f"dup_req={row['duplicate_request_count']} ({row['duplicate_request_ratio']:.4f}),"
            f"hit_ub={row['cache_hit_upper_bound_count']} ({row['cache_hit_upper_bound_ratio']:.4f}),"
            f"largest={row['largest_group']}"
        )

    print("\ntop repeated keys")
    for key_name in key_names:
        counts = Counter(row[key_name] for row in rows)
        print(f"\n{key_name}")
        for key, count in counts.most_common(args.topn):
            if count <= 1:
                break
            print(f"{count}\t{key[:180]}")

    print("\nexamples")
    for row in rows[:8]:
        print(f"qid={row['qid']} sql_fine=[{row['sql_fine']}] vector_coarse=[{row['vector_coarse']}]")
        print(f"  query={row['query'][:220]}")
        print(f"  vector_text={row['vector_text'][:220]}")

    print(f"\nwrote {args.out_summary}")
    print(f"wrote {args.out_detail}")


if __name__ == "__main__":
    main()
