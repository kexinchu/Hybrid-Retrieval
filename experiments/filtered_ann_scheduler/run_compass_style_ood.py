from __future__ import annotations

import argparse
import csv
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class Predicate:
    name: str
    sql: str
    params: tuple[float | int, ...]
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


def read_fbin(path: Path, limit: int | None = None) -> np.ndarray:
    with path.open("rb") as f:
        n, dim = struct.unpack("<ii", f.read(8))
    count = n if limit is None else min(n, limit)
    mm = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, dim))
    return np.ascontiguousarray(mm[:count])


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def build_attributes(vectors: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(vectors)
    direction = rng.normal(size=vectors.shape[1]).astype("float32")
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    projection = vectors @ direction
    ranks = np.empty(n, dtype=np.float32)
    ranks[np.argsort(projection)] = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    attr_corr = ranks
    attr_anti = 1.0 - ranks
    attr_random = rng.random(n, dtype=np.float32)
    bucket_random = np.minimum((attr_random * 100).astype(np.int32), 99)
    cluster_id = np.minimum((attr_corr * 100).astype(np.int32), 99)
    return {
        "attr_random": attr_random.astype(np.float32),
        "attr_corr": attr_corr.astype(np.float32),
        "attr_anti": attr_anti.astype(np.float32),
        "bucket_random": bucket_random,
        "cluster_id": cluster_id,
    }


def build_sqlite(attrs: dict[str, np.ndarray]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            attr_random REAL,
            attr_corr REAL,
            attr_anti REAL,
            bucket_random INTEGER,
            cluster_id INTEGER
        )
        """
    )
    rows = (
        (
            int(i),
            float(attrs["attr_random"][i]),
            float(attrs["attr_corr"][i]),
            float(attrs["attr_anti"][i]),
            int(attrs["bucket_random"][i]),
            int(attrs["cluster_id"][i]),
        )
        for i in range(len(attrs["attr_random"]))
    )
    conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?)", rows)
    for idx_sql in [
        "CREATE INDEX idx_attr_random ON items(attr_random)",
        "CREATE INDEX idx_attr_corr ON items(attr_corr)",
        "CREATE INDEX idx_attr_anti ON items(attr_anti)",
        "CREATE INDEX idx_bucket_random ON items(bucket_random)",
        "CREATE INDEX idx_cluster_id ON items(cluster_id)",
        "CREATE INDEX idx_cluster_random ON items(cluster_id, attr_random)",
    ]:
        conn.execute(idx_sql)
    conn.commit()
    return conn


def build_hnsw(vectors: np.ndarray, m: int, ef_construction: int) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def fetch_ids(conn: sqlite3.Connection, pred: Predicate) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    rows = conn.execute(f"SELECT id FROM items WHERE {pred.sql}", pred.params).fetchall()
    sqlite_ms = (time.perf_counter() - t0) * 1000
    return np.array([row[0] for row in rows], dtype=np.int64), sqlite_ms


def count_ids(conn: sqlite3.Connection, pred: Predicate) -> tuple[int, float]:
    t0 = time.perf_counter()
    count = conn.execute(f"SELECT COUNT(*) FROM items WHERE {pred.sql}", pred.params).fetchone()[0]
    sqlite_ms = (time.perf_counter() - t0) * 1000
    return int(count), sqlite_ms


def filter_candidate_ids(conn: sqlite3.Connection, pred: Predicate, ids: np.ndarray) -> tuple[np.ndarray, float]:
    if len(ids) == 0:
        return ids, 0.0
    placeholders = ",".join("?" for _ in ids)
    sql = f"SELECT id FROM items WHERE id IN ({placeholders}) AND {pred.sql}"
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
    filtered, sqlite_ms = filter_candidate_ids(conn, pred, ids)
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
    total_exact = 0
    total_predicates = 0
    ann_k = start_k
    rounds = 0
    last_ids = np.empty(0, dtype=np.int64)
    while True:
        rounds += 1
        res = ann_then_filter(conn, index, vectors, q, pred, k, ann_k, ef_search)
        total_latency += res.latency_ms
        total_sqlite += res.sqlite_ms
        total_exact += res.exact_distance_evals
        total_predicates += res.predicate_evals
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
    threshold: float,
    start_k: int,
    max_k: int,
    ef_search: int,
) -> SearchResult:
    count, count_ms = count_ids(conn, pred)
    sel = count / n
    if sel <= threshold:
        res = sqlite_prefilter_exact(conn, vectors, q, pred, k)
        res.latency_ms += count_ms
        res.sqlite_ms += count_ms
        res.extra = f"choice=sqlite_prefilter_exact;sel={sel:g}"
        return res
    res = iterative_ann(conn, index, vectors, q, pred, k, start_k, max_k, ef_search)
    res.latency_ms += count_ms
    res.sqlite_ms += count_ms
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
    selectivity_threshold: float,
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
    if sel <= selectivity_threshold or local_yield < local_yield_threshold:
        res = sqlite_prefilter_exact(conn, vectors, q, pred, k)
        res.latency_ms += count_ms + probe.latency_ms
        res.sqlite_ms += count_ms + probe.sqlite_ms
        res.ann_returned += probe.ann_returned
        res.predicate_evals += probe.predicate_evals
        res.extra = f"choice=sqlite_prefilter_exact;sel={sel:g};local_yield={local_yield:g}"
        return res
    res = iterative_ann(conn, index, vectors, q, pred, k, start_k, max_k, ef_search)
    res.latency_ms += count_ms + probe.latency_ms
    res.sqlite_ms += count_ms + probe.sqlite_ms
    res.ann_returned += probe.ann_returned
    res.predicate_evals += probe.predicate_evals
    res.extra = f"choice=iterative_ann;sel={sel:g};local_yield={local_yield:g};{res.extra}"
    return res


def window(center: float, width: float) -> tuple[float, float]:
    lo = min(max(center - width / 2, 0.0), 1.0 - width)
    return lo, lo + width


def make_predicates(qid: int, attrs: dict[str, np.ndarray], selectivities: list[float]) -> list[Predicate]:
    preds: list[Predicate] = []
    for attr_name, kind in [("attr_random", "random_range"), ("attr_corr", "correlated_range"), ("attr_anti", "anti_range")]:
        center = float(attrs[attr_name][qid])
        for sel in selectivities:
            lo, hi = window(center, sel)
            preds.append(
                Predicate(
                    name=f"{attr_name}_sel{sel:g}",
                    sql=f"{attr_name} BETWEEN ? AND ?",
                    params=(lo, hi),
                    kind=kind,
                )
            )
    cluster = int(attrs["cluster_id"][qid])
    bucket = int(attrs["bucket_random"][qid])
    lo, hi = window(float(attrs["attr_random"][qid]), 0.05)
    preds.extend(
        [
            Predicate("cluster_eq", "cluster_id = ?", (cluster,), "category"),
            Predicate("bucket_random_eq", "bucket_random = ?", (bucket,), "category"),
            Predicate(
                "cluster_and_random_sel0.05",
                "cluster_id = ? AND attr_random BETWEEN ? AND ?",
                (cluster, lo, hi),
                "conjunction",
            ),
        ]
    )
    return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fbin",
        type=Path,
        default=Path("/home/kec23008/docker-sys/OOD-ANNS/data/WebVid8M/hard_random.1M.fbin"),
    )
    parser.add_argument("--limit", type=int, default=250000)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--selectivities", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.2])
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--max-ann-k", type=int, default=20000)
    parser.add_argument("--pre-threshold", type=float, default=0.02)
    parser.add_argument("--local-yield-threshold", type=float, default=0.02)
    parser.add_argument("--probe-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/compass_style_ood_webvid_random.csv"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.limit = min(args.limit, 100000)
        args.queries = 24
        args.max_ann_k = 10000
        args.out = Path("results/compass_style_ood_webvid_random_quick.csv")

    print(f"loading vectors from {args.fbin}", flush=True)
    vectors = l2_normalize(read_fbin(args.fbin, args.limit)).astype("float32")
    print(f"loaded vectors shape={vectors.shape}", flush=True)
    attrs = build_attributes(vectors, args.seed)
    conn = build_sqlite(attrs)
    print("building HNSW", flush=True)
    index = build_hnsw(vectors, args.hnsw_m, args.ef_construction)
    rng = np.random.default_rng(args.seed)
    query_ids = rng.choice(len(vectors), size=min(args.queries, len(vectors)), replace=False)

    rows: list[dict[str, object]] = []
    for qn, qid in enumerate(query_ids):
        q = vectors[qid]
        for pred in make_predicates(int(qid), attrs, args.selectivities):
            truth_res = sqlite_prefilter_exact(conn, vectors, q, pred, args.k)
            truth = truth_res.ids
            count, count_ms = count_ids(conn, pred)
            strategies = {
                "sqlite_prefilter_exact": truth_res,
                "post_ann_10x": ann_then_filter(conn, index, vectors, q, pred, args.k, args.k * 10, args.ef_search),
                "post_ann_100x": ann_then_filter(conn, index, vectors, q, pred, args.k, args.k * 100, args.ef_search),
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
                    len(vectors),
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
                    len(vectors),
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
                        "dataset": args.fbin.parent.name,
                        "fbin": str(args.fbin),
                        "n": len(vectors),
                        "dim": vectors.shape[1],
                        "query_no": qn,
                        "query_id": int(qid),
                        "predicate": pred.name,
                        "predicate_kind": pred.kind,
                        "strategy": name,
                        "actual_selectivity": count / len(vectors),
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
