from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class Predicate:
    name: str
    sql: str
    params: tuple[int, ...]
    kind: str


@dataclass
class SearchResult:
    ids: np.ndarray
    latency_ms: float
    sqlite_ms: float
    exact_distance_evals: int
    ann_returned: int
    predicate_evals: int
    failed_to_fill: bool
    extra: str


def label_group(target_name: str) -> str:
    if "." in target_name:
        return target_name.split(".", 1)[0]
    return target_name


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_vectors(data_home: Path, dim: int, max_features: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        data_home=str(data_home),
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.85,
        stop_words="english",
    )
    svd = TruncatedSVD(n_components=dim, random_state=0)
    model = make_pipeline(vectorizer, svd, Normalizer(copy=False))
    vectors = model.fit_transform(dataset.data).astype("float32")
    vectors = l2_normalize(vectors).astype("float32")
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 1e-6
    texts = [text for text, should_keep in zip(dataset.data, keep) if should_keep]
    targets = dataset.target.astype(np.int32)[keep]
    vectors = np.ascontiguousarray(vectors[keep])
    return vectors, targets, texts, dataset.target_names


def build_sqlite(texts: list[str], targets: np.ndarray, target_names: list[str]) -> tuple[sqlite3.Connection, dict[str, int]]:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, target INTEGER, group_id INTEGER, char_len INTEGER, word_len INTEGER)"
    )
    group_names = sorted({label_group(name) for name in target_names})
    group_to_id = {name: i for i, name in enumerate(group_names)}
    target_to_group = {
        i: group_to_id[label_group(name)]
        for i, name in enumerate(target_names)
    }
    rows = []
    for i, (text, target) in enumerate(zip(texts, targets)):
        rows.append(
            (
                i,
                int(target),
                int(target_to_group[int(target)]),
                len(text),
                len(text.split()),
            )
        )
    conn.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute("CREATE INDEX idx_target ON docs(target)")
    conn.execute("CREATE INDEX idx_group ON docs(group_id)")
    conn.execute("CREATE INDEX idx_char_len ON docs(char_len)")
    conn.execute("CREATE INDEX idx_word_len ON docs(word_len)")
    conn.execute("CREATE INDEX idx_group_char ON docs(group_id, char_len)")
    conn.execute("CREATE INDEX idx_target_char ON docs(target, char_len)")
    conn.commit()
    return conn, target_to_group


def build_hnsw(vectors: np.ndarray, m: int, ef_construction: int) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def fetch_ids(conn: sqlite3.Connection, pred: Predicate) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    rows = conn.execute(f"SELECT id FROM docs WHERE {pred.sql}", pred.params).fetchall()
    sqlite_ms = (time.perf_counter() - t0) * 1000
    return np.array([row[0] for row in rows], dtype=np.int64), sqlite_ms


def count_ids(conn: sqlite3.Connection, pred: Predicate) -> tuple[int, float]:
    t0 = time.perf_counter()
    count = conn.execute(f"SELECT COUNT(*) FROM docs WHERE {pred.sql}", pred.params).fetchone()[0]
    sqlite_ms = (time.perf_counter() - t0) * 1000
    return int(count), sqlite_ms


def predicate_mask_for_ids(
    conn: sqlite3.Connection,
    pred: Predicate,
    ids: np.ndarray,
) -> tuple[np.ndarray, float]:
    if len(ids) == 0:
        return ids, 0.0
    placeholders = ",".join("?" for _ in ids)
    sql = f"SELECT id FROM docs WHERE id IN ({placeholders}) AND {pred.sql}"
    params = tuple(int(x) for x in ids) + pred.params
    t0 = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    sqlite_ms = (time.perf_counter() - t0) * 1000
    return np.array([row[0] for row in rows], dtype=np.int64), sqlite_ms


def exact_rank(vectors: np.ndarray, q: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    if len(ids) == 0:
        return ids.astype(np.int64)
    scores = vectors[ids] @ q
    take = min(k, len(ids))
    local = np.argpartition(-scores, take - 1)[:take]
    local = local[np.argsort(-scores[local])]
    return ids[local].astype(np.int64)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    if len(truth) == 0:
        return 1.0
    return len(set(found[:k]).intersection(truth[:k])) / min(k, len(truth))


def sqlite_prefilter_exact(
    conn: sqlite3.Connection,
    vectors: np.ndarray,
    q: np.ndarray,
    pred: Predicate,
    k: int,
) -> SearchResult:
    t0 = time.perf_counter()
    ids, sqlite_ms = fetch_ids(conn, pred)
    ranked = exact_rank(vectors, q, ids, k)
    return SearchResult(
        ids=ranked,
        latency_ms=(time.perf_counter() - t0) * 1000,
        sqlite_ms=sqlite_ms,
        exact_distance_evals=len(ids),
        ann_returned=0,
        predicate_evals=len(ids),
        failed_to_fill=len(ranked) < min(k, len(ids)),
        extra="",
    )


def ann_then_filter(
    conn: sqlite3.Connection,
    index: faiss.IndexHNSWFlat,
    vectors: np.ndarray,
    q: np.ndarray,
    pred: Predicate,
    k: int,
    ann_k: int,
    ef_search: int,
) -> SearchResult:
    index.hnsw.efSearch = ef_search
    t0 = time.perf_counter()
    _, ids = index.search(q.reshape(1, -1), ann_k)
    ids = ids[0]
    ids = ids[ids >= 0].astype(np.int64)
    filtered, sqlite_ms = predicate_mask_for_ids(conn, pred, ids)
    ranked = exact_rank(vectors, q, filtered, k)
    return SearchResult(
        ids=ranked,
        latency_ms=(time.perf_counter() - t0) * 1000,
        sqlite_ms=sqlite_ms,
        exact_distance_evals=len(filtered),
        ann_returned=len(ids),
        predicate_evals=len(ids),
        failed_to_fill=len(ranked) < k,
        extra=f"ann_k={ann_k};ef={ef_search}",
    )


def iterative_ann(
    conn: sqlite3.Connection,
    index: faiss.IndexHNSWFlat,
    vectors: np.ndarray,
    q: np.ndarray,
    pred: Predicate,
    k: int,
    start_k: int,
    max_k: int,
    ef_search: int,
) -> SearchResult:
    total_latency = 0.0
    total_sqlite = 0.0
    total_predicates = 0
    total_exact = 0
    ann_k = start_k
    rounds = 0
    last_ids = np.empty(0, dtype=np.int64)
    while True:
        rounds += 1
        res = ann_then_filter(conn, index, vectors, q, pred, k, ann_k, ef_search)
        total_latency += res.latency_ms
        total_sqlite += res.sqlite_ms
        total_predicates += res.predicate_evals
        total_exact += res.exact_distance_evals
        last_ids = res.ids
        if len(last_ids) >= k or ann_k >= max_k:
            return SearchResult(
                ids=last_ids,
                latency_ms=total_latency,
                sqlite_ms=total_sqlite,
                exact_distance_evals=total_exact,
                ann_returned=ann_k,
                predicate_evals=total_predicates,
                failed_to_fill=len(last_ids) < k,
                extra=f"rounds={rounds};final_ann_k={ann_k};ef={ef_search}",
            )
        ann_k = min(max_k, ann_k * 2)


def adaptive_selectivity(
    conn: sqlite3.Connection,
    index: faiss.IndexHNSWFlat,
    vectors: np.ndarray,
    q: np.ndarray,
    pred: Predicate,
    k: int,
    n: int,
    pre_threshold: float,
    start_k: int,
    max_k: int,
    ef_search: int,
) -> SearchResult:
    count, count_ms = count_ids(conn, pred)
    sel = count / n
    if sel <= pre_threshold:
        res = sqlite_prefilter_exact(conn, vectors, q, pred, k)
        res.latency_ms += count_ms
        res.sqlite_ms += count_ms
        res.extra = f"choice=sqlite_prefilter_exact;sel={sel:g}"
        return res
    res = iterative_ann(conn, index, vectors, q, pred, k, start_k, max_k, ef_search)
    res.sqlite_ms += count_ms
    res.latency_ms += count_ms
    res.extra = f"choice=iterative_ann;sel={sel:g};{res.extra}"
    return res


def adaptive_probe(
    conn: sqlite3.Connection,
    index: faiss.IndexHNSWFlat,
    vectors: np.ndarray,
    q: np.ndarray,
    pred: Predicate,
    k: int,
    n: int,
    pre_threshold: float,
    local_yield_threshold: float,
    probe_k: int,
    start_k: int,
    max_k: int,
    ef_search: int,
) -> SearchResult:
    count, count_ms = count_ids(conn, pred)
    sel = count / n
    probe = ann_then_filter(conn, index, vectors, q, pred, k, probe_k, ef_search)
    local_yield = probe.exact_distance_evals / max(probe.ann_returned, 1)
    if sel <= pre_threshold or local_yield < local_yield_threshold:
        res = sqlite_prefilter_exact(conn, vectors, q, pred, k)
        res.latency_ms += count_ms + probe.latency_ms
        res.sqlite_ms += count_ms + probe.sqlite_ms
        res.predicate_evals += probe.predicate_evals
        res.ann_returned += probe.ann_returned
        res.extra = f"choice=sqlite_prefilter_exact;sel={sel:g};local_yield={local_yield:g}"
        return res
    res = iterative_ann(conn, index, vectors, q, pred, k, start_k, max_k, ef_search)
    res.latency_ms += count_ms + probe.latency_ms
    res.sqlite_ms += count_ms + probe.sqlite_ms
    res.predicate_evals += probe.predicate_evals
    res.ann_returned += probe.ann_returned
    res.extra = f"choice=iterative_ann;sel={sel:g};local_yield={local_yield:g};{res.extra}"
    return res


def make_length_window(values: np.ndarray, center_id: int, selectivity: float) -> tuple[int, int]:
    n = len(values)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    width = max(1, int(n * selectivity))
    center_rank = int(ranks[center_id])
    lo_rank = max(0, min(center_rank - width // 2, n - width))
    hi_rank = min(n - 1, lo_rank + width - 1)
    return int(values[order[lo_rank]]), int(values[order[hi_rank]])


def predicates_for_query(
    conn: sqlite3.Connection,
    query_id: int,
    targets: np.ndarray,
    target_to_group: dict[int, int],
    char_lens: np.ndarray,
) -> list[Predicate]:
    target = int(targets[query_id])
    group = int(target_to_group[target])
    one_lo, one_hi = make_length_window(char_lens, query_id, 0.01)
    five_lo, five_hi = make_length_window(char_lens, query_id, 0.05)
    twenty_lo, twenty_hi = make_length_window(char_lens, query_id, 0.20)
    return [
        Predicate("target_eq", "target = ?", (target,), "category"),
        Predicate("group_eq", "group_id = ?", (group,), "category"),
        Predicate("char_window_1pct", "char_len BETWEEN ? AND ?", (one_lo, one_hi), "range"),
        Predicate("char_window_5pct", "char_len BETWEEN ? AND ?", (five_lo, five_hi), "range"),
        Predicate("char_window_20pct", "char_len BETWEEN ? AND ?", (twenty_lo, twenty_hi), "range"),
        Predicate(
            "group_and_char_5pct",
            "group_id = ? AND char_len BETWEEN ? AND ?",
            (group, five_lo, five_hi),
            "conjunction",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", type=Path, default=Path("data/sklearn"))
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--queries", type=int, default=160)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--max-ann-k", type=int, default=5000)
    parser.add_argument("--pre-threshold", type=float, default=0.02)
    parser.add_argument("--local-yield-threshold", type=float, default=0.02)
    parser.add_argument("--probe-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/real_20newsgroups.csv"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.queries = 40
        args.dim = 96
        args.max_features = 15000
        args.out = Path("results/real_20newsgroups_quick.csv")

    vectors, targets, texts, target_names = load_vectors(args.data_home, args.dim, args.max_features)
    conn, target_to_group = build_sqlite(texts, targets, target_names)
    index = build_hnsw(vectors, args.hnsw_m, args.ef_construction)
    char_lens = np.array([len(text) for text in texts], dtype=np.int64)
    n = len(texts)

    rng = np.random.default_rng(args.seed)
    query_ids = rng.choice(n, size=min(args.queries, n), replace=False)
    rows: list[dict[str, object]] = []
    for qn, qid in enumerate(query_ids):
        q = vectors[qid]
        for pred in predicates_for_query(conn, int(qid), targets, target_to_group, char_lens):
            truth_res = sqlite_prefilter_exact(conn, vectors, q, pred, args.k)
            truth = truth_res.ids
            count, count_ms = count_ids(conn, pred)
            strategies = {
                "sqlite_prefilter_exact": truth_res,
                "post_ann_10x": ann_then_filter(
                    conn, index, vectors, q, pred, args.k, args.k * 10, args.ef_search
                ),
                "post_ann_100x": ann_then_filter(
                    conn, index, vectors, q, pred, args.k, args.k * 100, args.ef_search
                ),
                "iterative_ann": iterative_ann(
                    conn, index, vectors, q, pred, args.k, args.k * 10, args.max_ann_k, args.ef_search
                ),
                "adaptive_selectivity": adaptive_selectivity(
                    conn,
                    index,
                    vectors,
                    q,
                    pred,
                    args.k,
                    n,
                    args.pre_threshold,
                    args.k * 10,
                    args.max_ann_k,
                    args.ef_search,
                ),
                "adaptive_probe": adaptive_probe(
                    conn,
                    index,
                    vectors,
                    q,
                    pred,
                    args.k,
                    n,
                    args.pre_threshold,
                    args.local_yield_threshold,
                    args.probe_k,
                    args.k * 10,
                    args.max_ann_k,
                    args.ef_search,
                ),
            }
            for name, res in strategies.items():
                rows.append(
                    {
                        "dataset": "20newsgroups_train",
                        "n": n,
                        "query_no": qn,
                        "query_id": int(qid),
                        "predicate": pred.name,
                        "predicate_kind": pred.kind,
                        "strategy": name,
                        "actual_selectivity": count / n,
                        "match_count": count,
                        "count_sqlite_ms": count_ms,
                        "truth_size": len(truth),
                        "recall_at_k": recall_at_k(res.ids, truth, args.k),
                        "returned": len(res.ids),
                        "latency_ms": res.latency_ms,
                        "sqlite_ms": res.sqlite_ms,
                        "exact_distance_evals": res.exact_distance_evals,
                        "ann_returned": res.ann_returned,
                        "predicate_evals": res.predicate_evals,
                        "failed_to_fill": int(res.failed_to_fill),
                        "extra": res.extra,
                    }
                )
        print(f"finished query {qn + 1}/{len(query_ids)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
