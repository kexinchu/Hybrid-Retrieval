from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from common_pg import pg_config_from_env, require_psycopg
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k


TABLE = "amazon_grocery_reviews_10m_pgvector"

COMPLEXITY_SUFFIXES: dict[str, str] = {
    "simple": "TRUE",
    "numeric": """
        ((id + 17) > id)
        AND ((id %% 97) >= 0)
        AND ((coalesce(review_text_len, 0) + coalesce(helpful_vote, 0)) >= 0)
        AND ((coalesce(rating, 0) * coalesce(rating, 0)) >= 0)
    """,
    "string_regex": """
        (md5(id::text) = md5(id::text))
        AND (regexp_replace(id::text, '[^0-9]', '', 'g') = id::text)
        AND (length(coalesce(main_category, '')) >= 0)
    """,
}


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000


def vector_literal(value: object) -> str:
    if isinstance(value, str):
        return value
    return "[" + ",".join(f"{float(x):.7g}" for x in value) + "]"


def load_truth(path: Path) -> tuple[dict[tuple[str, int], list[int]], dict[int, int]]:
    truth: dict[tuple[str, int], list[int]] = {}
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "pre_filter_exact":
                continue
            qno = int(row["query_no"])
            query_by_no[qno] = int(row["query_id"])
            truth[(row["filter_name"], qno)] = [
                int(x) for x in row["exact_filtered_topk_ids"].split(",") if x
            ]
    return truth, query_by_no


def load_query_vectors(cur, query_ids: list[int]) -> dict[int, str]:
    cur.execute(
        f"""
        SELECT id, embedding
        FROM {TABLE}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    rows = cur.fetchall()
    vectors = {int(row[0]): vector_literal(row[1]) for row in rows}
    missing = [qid for qid in query_ids if qid not in vectors]
    if missing:
        raise RuntimeError(f"missing {len(missing)} query vectors")
    return vectors


def load_hnsw_profile(cur) -> dict[str, float]:
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    text = cur.fetchone()[0]
    profile = json.loads(text) if isinstance(text, str) else text
    return {
        "valid": bool(profile.get("valid", False)),
        "vector_ms": float(profile.get("vector_search_ms", 0.0)),
        "visited": float(profile.get("visited_tuples", 0.0)),
        "returned": float(profile.get("returned_tuples", 0.0)),
    }


def load_qual_profile(cur) -> dict[str, float]:
    cur.execute("SELECT hybrid_qual_profile_last()")
    text = cur.fetchone()[0]
    profile = json.loads(text) if isinstance(text, str) else text
    true_count = 0.0
    false_count = 0.0
    for entry in profile.get("entries", []) or []:
        true_count += float(entry.get("true", 0.0))
        false_count += float(entry.get("false", 0.0))
    return {
        "qual_ms": float(profile.get("qual_ms", 0.0)),
        "qual_calls": float(profile.get("qual_calls", 0.0)),
        "qual_true": true_count,
        "qual_false": false_count,
    }


def combined_predicate(base: str, complexity: str) -> str:
    suffix = COMPLEXITY_SUFFIXES[complexity]
    return f"({base}) AND ({suffix})"


def explain_plan(cur, predicate: str, query_vector: str) -> tuple[str, str]:
    cur.execute(
        f"""
        EXPLAIN (FORMAT JSON)
        SELECT id
        FROM {TABLE}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT 10
        """,
        (query_vector,),
    )
    plan = cur.fetchone()[0]
    if isinstance(plan, str):
        plan = json.loads(plan)
    nodes: list[str] = []

    def walk(node: dict) -> None:
        desc = node.get("Node Type", "")
        if "Index Name" in node:
            desc += f":{node['Index Name']}"
        if "Relation Name" in node:
            desc += f":{node['Relation Name']}"
        if "Filter" in node:
            desc += ":Filter"
        if "Sort Key" in node:
            desc += ":Sort"
        nodes.append(desc)
        for child in node.get("Plans", []) or []:
            walk(child)

    walk(plan[0]["Plan"])
    plan_text = " > ".join(nodes)
    if "embedding_hnsw_idx" in plan_text:
        return "hnsw_post_filter", plan_text
    if "Sort" in plan_text:
        return "filter_then_exact_sort", plan_text
    return "other", plan_text


def run_query(cur, predicate: str, query_vector: str, k: int) -> tuple[list[int], float, dict[str, float], dict[str, float]]:
    def execute():
        cur.execute("SELECT hybrid_qual_profile_reset()")
        cur.execute("SELECT vector_hnsw_reset_scan_profile()")
        cur.execute(
            f"""
            SELECT id
            FROM {TABLE}
            WHERE {predicate}
            ORDER BY embedding <-> %s::vector
            LIMIT {int(k)}
            """,
            (query_vector,),
        )
        ids = [int(row[0]) for row in cur.fetchall()]
        return ids

    ids, total_ms = timed(execute)
    hnsw_profile = load_hnsw_profile(cur)
    qual_profile = load_qual_profile(cur)
    return ids, total_ms, hnsw_profile, qual_profile


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["filter_name"]), str(row["complexity"])), []).append(row)

    order = {name: i for i, (name, _, _) in enumerate(ATTR_FILTERS)}
    complexity_order = {name: i for i, name in enumerate(COMPLEXITY_SUFFIXES)}
    out = []
    for (filter_name, complexity), items in sorted(
        groups.items(), key=lambda x: (order.get(x[0][0], 999), complexity_order.get(x[0][1], 999))
    ):
        out.append(
            {
                "filter": items[0]["filter"],
                "filter_name": filter_name,
                "actual_selectivity": float(items[0]["actual_selectivity"]),
                "complexity": complexity,
                "plan_class": items[0]["plan_class"],
                "recall": statistics.mean(float(r["recall"]) for r in items),
                "total_ms": statistics.mean(float(r["total_ms"]) for r in items),
                "hnsw_vector_ms": statistics.mean(float(r["hnsw_vector_ms"]) for r in items),
                "qual_ms": statistics.mean(float(r["qual_ms"]) for r in items),
                "non_hnsw_exec_ms": statistics.mean(float(r["non_hnsw_exec_ms"]) for r in items),
                "qual_calls": statistics.mean(float(r["qual_calls"]) for r in items),
                "hnsw_visited": statistics.mean(float(r["hnsw_visited"]) for r in items),
                "hnsw_returned": statistics.mean(float(r["hnsw_returned"]) for r in items),
                "returned": statistics.mean(float(r["returned"]) for r in items),
                "queries": len(items),
                "repeats": int(items[0]["repeats"]),
                "ef_search": int(items[0]["ef_search"]),
                "iterative_scan": items[0]["iterative_scan"],
                "max_scan_tuples": int(items[0]["max_scan_tuples"]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/pgvector_sql_complexity_selectivity.csv"))
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["strict_order", "relaxed_order", "off"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=1.0)
    parser.add_argument("--disable-seqscan", action="store_true")
    parser.add_argument("--jit", choices=["on", "off"], default="off")
    parser.add_argument("--complexities", nargs="+", default=list(COMPLEXITY_SUFFIXES))
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    unknown = [name for name in args.complexities if name not in COMPLEXITY_SUFFIXES]
    if unknown:
        raise SystemExit(f"unknown complexities: {unknown}")

    require_psycopg()
    import psycopg

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    query_ids = [query_by_no[qno] for qno in query_nos]
    selected = set(args.filter_names or [])
    filters = [(name, target, pred) for name, target, pred in ATTR_FILTERS if not selected or name in selected]
    rows: list[dict[str, object]] = []

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS hybrid_qual_profile")
            cur.execute("SELECT to_regprocedure('vector_hnsw_last_scan_profile()'), to_regprocedure('hybrid_qual_profile_last()')")
            if not all(cur.fetchone()):
                raise RuntimeError("profile functions are not available")
            cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
            if args.iterative_scan == "off":
                cur.execute("SET hnsw.iterative_scan = off")
            else:
                cur.execute(f"SET hnsw.iterative_scan = '{args.iterative_scan}'")
            cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
            cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
            cur.execute(f"SET enable_seqscan = {'off' if args.disable_seqscan else 'on'}")
            cur.execute(f"SET jit = {args.jit}")

            query_vectors = load_query_vectors(cur, query_ids)

            base_counts: dict[str, int] = {}
            for filter_name, _, predicate in filters:
                cur.execute(f"SELECT count(*) FROM {TABLE} WHERE {predicate}")
                base_counts[filter_name] = int(cur.fetchone()[0])

            for filter_name, target, base_predicate in filters:
                actual_selectivity = base_counts[filter_name] / 10_000_000.0
                for complexity in args.complexities:
                    predicate = combined_predicate(base_predicate, complexity)
                    plan_class, plan_text = explain_plan(cur, predicate, query_vectors[query_ids[0]])
                    case_latencies: list[float] = []
                    case_vector: list[float] = []
                    case_qual: list[float] = []
                    case_non_hnsw: list[float] = []
                    case_calls: list[float] = []
                    case_recall: list[float] = []
                    for idx, qno in enumerate(query_nos, start=1):
                        qid = query_by_no[qno]
                        ids: list[int] = []
                        total_samples: list[float] = []
                        vector_samples: list[float] = []
                        qual_samples: list[float] = []
                        calls_samples: list[float] = []
                        visited_samples: list[float] = []
                        returned_samples: list[float] = []
                        for _ in range(args.repeats):
                            ids, total_ms, hnsw, qual = run_query(cur, predicate, query_vectors[qid], args.k)
                            vector_ms = float(hnsw["vector_ms"])
                            total_samples.append(total_ms)
                            vector_samples.append(vector_ms)
                            qual_samples.append(float(qual["qual_ms"]))
                            calls_samples.append(float(qual["qual_calls"]))
                            visited_samples.append(float(hnsw["visited"]))
                            returned_samples.append(float(hnsw["returned"]))

                        total_ms = statistics.mean(total_samples)
                        hnsw_ms = statistics.mean(vector_samples)
                        qual_ms = statistics.mean(qual_samples)
                        non_hnsw_ms = max(total_ms - hnsw_ms, 0.0)
                        rec = recall_at_k(ids, truth[(filter_name, qno)], args.k)
                        row = {
                            "filter": target,
                            "filter_name": filter_name,
                            "actual_selectivity": actual_selectivity,
                            "sql_rows": base_counts[filter_name],
                            "complexity": complexity,
                            "plan_class": plan_class,
                            "plan_text": plan_text,
                            "query_no": qno,
                            "query_id": qid,
                            "recall": rec,
                            "total_ms": total_ms,
                            "hnsw_vector_ms": hnsw_ms,
                            "qual_ms": qual_ms,
                            "non_hnsw_exec_ms": non_hnsw_ms,
                            "qual_calls": statistics.mean(calls_samples),
                            "hnsw_visited": statistics.mean(visited_samples),
                            "hnsw_returned": statistics.mean(returned_samples),
                            "returned": len(ids),
                            "repeats": args.repeats,
                            "ef_search": args.ef_search,
                            "iterative_scan": args.iterative_scan,
                            "max_scan_tuples": args.max_scan_tuples,
                            "scan_mem_multiplier": args.scan_mem_multiplier,
                            "jit": args.jit,
                            "disable_seqscan": args.disable_seqscan,
                        }
                        rows.append(row)
                        case_latencies.append(total_ms)
                        case_vector.append(hnsw_ms)
                        case_qual.append(qual_ms)
                        case_non_hnsw.append(non_hnsw_ms)
                        case_calls.append(statistics.mean(calls_samples))
                        case_recall.append(rec)
                        if args.progress_every and idx % args.progress_every == 0:
                            print(
                                f"progress filter={filter_name} complexity={complexity} "
                                f"queries={idx}/{len(query_nos)} total={statistics.mean(case_latencies):.2f} "
                                f"hnsw={statistics.mean(case_vector):.2f} qual={statistics.mean(case_qual):.2f} "
                                f"non_hnsw={statistics.mean(case_non_hnsw):.2f} calls={statistics.mean(case_calls):.1f} "
                                f"recall={statistics.mean(case_recall):.3f} plan={plan_class}",
                                flush=True,
                            )

    write_csv(args.out, rows)
    summary = summarize(rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_csv(summary_path, summary)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
