from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import psycopg

try:
    from .common_pg import pg_config_from_env
    from .faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
except ImportError:  # Direct script execution puts this directory on sys.path.
    from common_pg import pg_config_from_env
    from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k


TABLE = "amazon_grocery_reviews_10m_pgvector"
INDEX = f"{TABLE}_embedding_hnsw_idx"

FILTER_ATOMS: dict[str, list[str]] = {
    "popular_ge1000": ["sql:item_rating_number >= 1000"],
    "price_10_to_20": ["sql:has_price AND price > 10 AND price <= 20"],
    "rating5_price_le10": ["sql:has_price AND price <= 10", "sql:rating = 5"],
    "long_review_ge500": ["sql:review_text_len >= 500"],
    "grocery_rating5": ["sql:main_category = 'Grocery'", "sql:rating = 5"],
    "grocery_helpful": ["sql:main_category = 'Grocery'", "sql:helpful_vote >= 1"],
    "helpful_ge20": ["sql:helpful_vote >= 20"],
    "grocery_long500": ["sql:main_category = 'Grocery'", "sql:review_text_len >= 500"],
}


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def load_truth(path: Path, method: str = "pre_filter_exact") -> tuple[dict[tuple[str, int], list[int]], dict[int, int]]:
    truth: dict[tuple[str, int], list[int]] = {}
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != method:
                continue
            qno = int(row["query_no"])
            truth[(row["filter_name"], qno)] = [int(x) for x in row["exact_filtered_topk_ids"].split(",") if x]
            query_by_no[qno] = int(row["query_id"])
    return truth, query_by_no


def configure(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET jit = off")
    if args.force_hnsw:
        cur.execute("SET enable_sort = off")


def ensure_functions(cur: psycopg.Cursor) -> None:
    function_sql = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in function_sql:
        cur.execute(sql)


def ensure_guidance_meta(cur: psycopg.Cursor, table: str) -> None:
    meta = f"{table}_guidance_meta"
    cur.execute("SELECT to_regclass(%s)", (meta,))
    exists = cur.fetchone()[0] is not None
    rebuild = not exists
    if exists:
        cur.execute(f"SELECT count(*) FROM {meta}")
        rebuild = int(cur.fetchone()[0]) == 0
    if rebuild:
        if exists:
            cur.execute(f"DROP TABLE {meta}")
        cur.execute(
            f"""
            CREATE TABLE {meta} AS
            SELECT ctid AS heap_tid, id, rating, verified_purchase, helpful_vote,
                   review_text_len, main_category, price, has_price, item_rating_number
            FROM {table}
            """
        )
    indexes = [
        ("item_rating", "item_rating_number"),
        ("price", "has_price, price"),
        ("rating", "rating"),
        ("review_len", "review_text_len"),
        ("category", "main_category"),
        ("helpful", "helpful_vote"),
    ]
    for name, cols in indexes:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {meta}_{name}_idx ON {meta} ({cols})")
    cur.execute(f"ANALYZE {meta}")


def activate_guidance(
    cur: psycopg.Cursor, index_name: str, method: str, filter_name: str
) -> dict[str, float | int | str | bool]:
    if method == "baseline":
        cur.execute("SELECT vector_hnsw_guidance_reset()")
        return {
            "active": False,
            "kind": "off",
            "atoms": 0,
            "last_cache_build_ms": 0.0,
            "last_cache_rows": 0,
            "last_cache_pages": 0,
        }

    atoms = FILTER_ATOMS[filter_name]
    cur.execute("SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], %s)", (index_name, atoms, method))
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    return json.loads(cur.fetchone()[0])


def run_query(
    cur: psycopg.Cursor,
    table: str,
    predicate: str,
    query_id: int,
    k: int,
) -> tuple[list[int], float, dict[str, object]]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")

    def execute() -> list[int]:
        cur.execute(
            f"""
            SELECT id
            FROM {table}
            WHERE {predicate}
            ORDER BY embedding <-> (SELECT embedding FROM {table} WHERE id = %s)
            LIMIT {int(k)}
            """,
            (int(query_id),),
        )
        return [int(row[0]) for row in cur.fetchall()]

    ids, latency_ms = timed_ms(execute)
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    profile = json.loads(cur.fetchone()[0])
    return ids, latency_ms, profile


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    summary = out.with_name(out.stem + "_summary.csv")
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["method"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    method_order = {"baseline": 0, "page": 1, "bloom": 2, "exact": 3}
    fields = [
        "filter",
        "filter_name",
        "method",
        "recall",
        "latency_ms",
        "vector_search_ms",
        "hnsw_visited_tuples",
        "hnsw_returned_tuples",
        "guidance_checks",
        "guidance_matches",
        "guidance_skips",
        "guidance_build_ms",
        "guidance_rows",
        "guidance_pages",
        "returned",
        "errors",
    ]
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (filter_name, method), items in sorted(
            groups.items(), key=lambda kv: (order.get(kv[0][0], 999), method_order.get(kv[0][1], 999))
        ):
            ok = [r for r in items if not r["error"]]
            first = items[0]
            writer.writerow(
                {
                    "filter": first["filter"],
                    "filter_name": filter_name,
                    "method": method,
                    "recall": statistics.mean(float(r["recall"]) for r in ok) if ok else 0.0,
                    "latency_ms": statistics.mean(float(r["latency_ms"]) for r in ok) if ok else 0.0,
                    "vector_search_ms": statistics.mean(float(r["vector_search_ms"]) for r in ok) if ok else 0.0,
                    "hnsw_visited_tuples": statistics.mean(float(r["hnsw_visited_tuples"]) for r in ok) if ok else 0.0,
                    "hnsw_returned_tuples": statistics.mean(float(r["hnsw_returned_tuples"]) for r in ok) if ok else 0.0,
                    "guidance_checks": statistics.mean(float(r["guidance_checks"]) for r in ok) if ok else 0.0,
                    "guidance_matches": statistics.mean(float(r["guidance_matches"]) for r in ok) if ok else 0.0,
                    "guidance_skips": statistics.mean(float(r["guidance_skips"]) for r in ok) if ok else 0.0,
                    "guidance_build_ms": max(float(r["guidance_build_ms"]) for r in items),
                    "guidance_rows": max(int(r["guidance_rows"]) for r in items),
                    "guidance_pages": max(int(r["guidance_pages"]) for r in items),
                    "returned": statistics.mean(float(r["returned"]) for r in ok) if ok else 0.0,
                    "errors": sum(1 for r in items if r["error"]),
                }
            )
    print(f"wrote {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pgvector HNSW predicate atom guidance.")
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_predicate_guidance.csv"))
    parser.add_argument("--methods", nargs="+", default=["baseline", "page", "bloom"], choices=["baseline", "page", "bloom", "exact"])
    parser.add_argument("--filter-names", nargs="*")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--force-hnsw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-meta", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-queries", type=int, default=5)
    args = parser.parse_args()

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    rows: list[dict[str, object]] = []
    fieldnames = [
        "filter",
        "filter_name",
        "method",
        "query_no",
        "query_id",
        "repeat",
        "recall",
        "latency_ms",
        "vector_search_ms",
        "hnsw_visited_tuples",
        "hnsw_returned_tuples",
        "guidance_checks",
        "guidance_matches",
        "guidance_skips",
        "guidance_build_ms",
        "guidance_rows",
        "guidance_pages",
        "returned",
        "ids",
        "error",
    ]
    cfg = pg_config_from_env()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                ensure_functions(cur)
                if args.prepare_meta:
                    ensure_guidance_meta(cur, args.table)
                configure(cur, args)
                for filter_name, target_rate, predicate in filters:
                    for method in args.methods:
                        print(f"activate method={method} filter={filter_name}", flush=True)
                        guidance_profile = activate_guidance(cur, args.index, method, filter_name)
                        for idx, qno in enumerate(query_nos, start=1):
                            qid = query_by_no[qno]
                            for repeat in range(args.repeats):
                                error = ""
                                ids: list[int] = []
                                latency_ms = 0.0
                                profile: dict[str, object] = {}
                                start = time.perf_counter()
                                try:
                                    ids, latency_ms, profile = run_query(cur, args.table, predicate, qid, args.k)
                                except Exception as exc:  # noqa: BLE001 - keep long benchmark moving
                                    error = exc.__class__.__name__
                                    latency_ms = (time.perf_counter() - start) * 1000.0
                                    cur.execute("ROLLBACK")
                                    configure(cur, args)
                                truth_ids = truth[(filter_name, qno)]
                                row = {
                                    "filter": target_rate,
                                    "filter_name": filter_name,
                                    "method": method,
                                    "query_no": qno,
                                    "query_id": qid,
                                    "repeat": repeat,
                                    "recall": recall_at_k(ids, truth_ids, args.k) if not error else 0.0,
                                    "latency_ms": latency_ms,
                                    "vector_search_ms": float(profile.get("vector_search_ms", 0.0)) if profile else 0.0,
                                    "hnsw_visited_tuples": int(profile.get("visited_tuples", 0)) if profile else 0,
                                    "hnsw_returned_tuples": int(profile.get("returned_tuples", 0)) if profile else 0,
                                    "guidance_checks": int(profile.get("guidance_checks", 0)) if profile else 0,
                                    "guidance_matches": int(profile.get("guidance_matches", 0)) if profile else 0,
                                    "guidance_skips": int(profile.get("guidance_skips", 0)) if profile else 0,
                                    "guidance_build_ms": float(guidance_profile.get("last_cache_build_ms", 0.0)),
                                    "guidance_rows": int(guidance_profile.get("last_cache_rows", 0)),
                                    "guidance_pages": int(guidance_profile.get("last_cache_pages", 0)),
                                    "returned": len(ids),
                                    "ids": ",".join(str(x) for x in ids),
                                    "error": error,
                                }
                                rows.append(row)
                                writer.writerow(row)
                                f.flush()
                            if args.progress_queries and idx % args.progress_queries == 0:
                                latest = [
                                    r
                                    for r in rows
                                    if r["filter_name"] == filter_name and r["method"] == method and not r["error"]
                                ]
                                if latest:
                                    print(
                                        f"progress method={method} filter={filter_name} queries={idx}/{len(query_nos)} "
                                        f"lat={statistics.mean(float(r['latency_ms']) for r in latest):.2f} "
                                        f"recall={statistics.mean(float(r['recall']) for r in latest):.3f}",
                                        flush=True,
                                    )
                        cur.execute("SELECT vector_hnsw_guidance_reset()")
    print(f"wrote {args.out}", flush=True)
    if rows:
        summarize(rows, args.out)


if __name__ == "__main__":
    main()
