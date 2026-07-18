from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from pgvector_c4_query_filter_cache_benchmark import load_workload
from pgvector_hnsw_page_access_group_benchmark import load_query_vectors


ROUTES = ["default_planner", "force_hnsw_postfilter", "force_prefilter_exact"]


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def configure_common(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET jit = off")


def configure_route(cur: psycopg.Cursor, args: argparse.Namespace, route: str) -> None:
    configure_common(cur, args)
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_indexonlyscan = on")
    cur.execute("SET enable_bitmapscan = on")
    cur.execute("SET enable_sort = on")
    cur.execute("SET enable_gathermerge = on")

    if route == "force_hnsw_postfilter":
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_bitmapscan = off")
        cur.execute("SET enable_sort = off")
    elif route == "force_prefilter_exact":
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_indexonlyscan = off")
        cur.execute("SET enable_sort = on")
    elif route != "default_planner":
        raise ValueError(route)


def flatten_plan(plan: dict) -> list[dict[str, object]]:
    nodes: list[dict[str, object]] = []

    def walk(node: dict, depth: int) -> None:
        nodes.append(
            {
                "depth": depth,
                "node_type": node.get("Node Type", ""),
                "relation": node.get("Relation Name", ""),
                "index": node.get("Index Name", ""),
                "total_cost": node.get("Total Cost", 0.0),
                "plan_rows": node.get("Plan Rows", 0),
                "sort_key": "; ".join(node.get("Sort Key", []) or []),
            }
        )
        for child in node.get("Plans", []) or []:
            walk(child, depth + 1)

    walk(plan, 0)
    return nodes


def classify_plan(nodes: list[dict[str, object]]) -> str:
    indexes = " ".join(str(n["index"]) for n in nodes).lower()
    has_sort = any(str(n["node_type"]) in ("Sort", "Incremental Sort") or n["sort_key"] for n in nodes)
    if "hnsw" in indexes or "embedding_hnsw" in indexes:
        return "hnsw_post_filter"
    if has_sort:
        return "filter_then_exact_sort"
    return "other"


def explain_route(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_vector: str) -> tuple[str, str, float]:
    cur.execute(
        f"""
        EXPLAIN (FORMAT JSON)
        SELECT id
        FROM {args.table}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT {int(args.k)}
        """,
        (query_vector,),
    )
    raw = cur.fetchone()[0]
    doc = json.loads(raw) if isinstance(raw, str) else raw
    plan = doc[0]["Plan"]
    nodes = flatten_plan(plan)
    text = " > ".join(f"{n['node_type']}:{n['index'] or n['relation']}".rstrip(":") for n in nodes)
    return classify_plan(nodes), text, float(plan.get("Total Cost", 0.0))


def load_hnsw_profile(cur: psycopg.Cursor) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    text = cur.fetchone()[0]
    return json.loads(text) if isinstance(text, str) else text


def run_query(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_vector: str) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {args.table}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT {int(args.k)}
        """,
        (query_vector,),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    return ids, load_hnsw_profile(cur)


def warmup(cur: psycopg.Cursor, args: argparse.Namespace, workload: list[dict[str, object]], vectors: dict[int, str]) -> None:
    if args.warmup_queries <= 0:
        return
    for row in workload[: args.warmup_queries]:
        qid = int(row["query_id_int"])
        predicate = str(row["predicate"])
        for route in ROUTES:
            try:
                configure_route(cur, args, route)
                run_query(cur, args, predicate, vectors[qid])
            except Exception:
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
                configure_common(cur, args)


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    ok_rows = [r for r in rows if not r["error"]]
    by_query: dict[int, list[dict[str, object]]] = collections.defaultdict(list)
    for row in ok_rows:
        by_query[int(row["query_no"])].append(row)

    cases = []
    for query_no, items in by_query.items():
        routes = {str(r["route"]): r for r in items}
        if not all(route in routes for route in ROUTES):
            continue
        fastest = min(routes.values(), key=lambda r: float(r["latency_ms"]))
        default = routes["default_planner"]
        hnsw = routes["force_hnsw_postfilter"]
        prefilter = routes["force_prefilter_exact"]
        cases.append(
            {
                "query_no": query_no,
                "predicate": default["predicate"],
                "cache_key": default["cache_key"],
                "default_plan": default["plan_class"],
                "fastest_route": fastest["route"],
                "fastest_plan": fastest["plan_class"],
                "default_ms": float(default["latency_ms"]),
                "hnsw_ms": float(hnsw["latency_ms"]),
                "prefilter_ms": float(prefilter["latency_ms"]),
                "fastest_ms": float(fastest["latency_ms"]),
                "default_matches_fastest_plan": default["plan_class"] == fastest["plan_class"],
                "oracle_speedup_vs_default": float(default["latency_ms"]) / float(fastest["latency_ms"]) if float(fastest["latency_ms"]) > 0 else 0.0,
            }
        )

    total_default = sum(c["default_ms"] for c in cases)
    total_oracle = sum(c["fastest_ms"] for c in cases)
    wrong = [c for c in cases if not c["default_matches_fastest_plan"]]
    meaningful_wrong = [c for c in wrong if c["oracle_speedup_vs_default"] >= args.meaningful_speedup]
    prefilter_wins = [c for c in cases if c["fastest_plan"] == "filter_then_exact_sort"]
    hnsw_wins = [c for c in cases if c["fastest_plan"] == "hnsw_post_filter"]

    by_predicate: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for case in cases:
        by_predicate[str(case["predicate"])].append(case)
    predicate_rows = []
    for predicate, items in by_predicate.items():
        default_total = sum(c["default_ms"] for c in items)
        oracle_total = sum(c["fastest_ms"] for c in items)
        predicate_rows.append(
            {
                "predicate": predicate,
                "queries": len(items),
                "wrong_route_count": sum(1 for c in items if not c["default_matches_fastest_plan"]),
                "prefilter_win_count": sum(1 for c in items if c["fastest_plan"] == "filter_then_exact_sort"),
                "hnsw_win_count": sum(1 for c in items if c["fastest_plan"] == "hnsw_post_filter"),
                "default_total_ms": default_total,
                "oracle_total_ms": oracle_total,
                "oracle_speedup": default_total / oracle_total if oracle_total else 0.0,
                "mean_default_ms": statistics.fmean(c["default_ms"] for c in items),
                "mean_hnsw_ms": statistics.fmean(c["hnsw_ms"] for c in items),
                "mean_prefilter_ms": statistics.fmean(c["prefilter_ms"] for c in items),
            }
        )

    return {
        "table": args.table,
        "query_csv": str(args.query_csv),
        "mode": args.mode,
        "queries_requested": args.queries,
        "cases": len(cases),
        "routes": ROUTES,
        "default_total_ms": total_default,
        "oracle_total_ms": total_oracle,
        "oracle_speedup_vs_default": total_default / total_oracle if total_oracle else 0.0,
        "wrong_route_count": len(wrong),
        "wrong_route_ratio": len(wrong) / len(cases) if cases else 0.0,
        "meaningful_speedup_threshold": args.meaningful_speedup,
        "meaningful_wrong_route_count": len(meaningful_wrong),
        "meaningful_wrong_route_ratio": len(meaningful_wrong) / len(cases) if cases else 0.0,
        "prefilter_win_count": len(prefilter_wins),
        "prefilter_win_ratio": len(prefilter_wins) / len(cases) if cases else 0.0,
        "hnsw_win_count": len(hnsw_wins),
        "hnsw_win_ratio": len(hnsw_wins) / len(cases) if cases else 0.0,
        "wrong_route_speedup_mean": statistics.fmean(c["oracle_speedup_vs_default"] for c in wrong) if wrong else 0.0,
        "wrong_route_speedup_p50": statistics.median(c["oracle_speedup_vs_default"] for c in wrong) if wrong else 0.0,
        "top_wrong_examples": sorted(wrong, key=lambda c: c["oracle_speedup_vs_default"], reverse=True)[:10],
        "predicate_summary": sorted(predicate_rows, key=lambda r: (-float(r["oracle_speedup"]), -int(r["queries"]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real PostgreSQL route audit over C4-derived pgvector workload.")
    parser.add_argument("--table", default="amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--query-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "relaxed_order", "strict_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--meaningful-speedup", type=float, default=1.1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--predicate-out", type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    workload = load_workload(args.query_csv, args.queries, args.mode)
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_no",
        "c4_query_no",
        "query_id",
        "predicate",
        "cache_key",
        "repeat",
        "route",
        "plan_class",
        "plan_text",
        "planner_total_cost",
        "latency_ms",
        "returned",
        "ids",
        "visited_tuples",
        "returned_tuples",
        "vector_search_ms",
        "error",
    ]
    rows: list[dict[str, object]] = []

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        configure_common(cur, args)
        vectors = load_query_vectors(cur, args.table, [int(row["query_id_int"]) for row in workload])
        warmup(cur, args, workload, vectors)

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for query_no, row in enumerate(workload):
                query_id = int(row["query_id_int"])
                predicate = str(row["predicate"])
                qvec = vectors[query_id]
                for repeat in range(args.repeats):
                    route_order = ROUTES[:]
                    rng.shuffle(route_order)
                    for route in route_order:
                        error = ""
                        ids: list[int] = []
                        profile: dict[str, object] = {}
                        latency = 0.0
                        try:
                            configure_route(cur, args, route)
                            plan_class, plan_text, plan_cost = explain_route(cur, args, predicate, qvec)
                            (ids, profile), latency = timed_ms(lambda: run_query(cur, args, predicate, qvec))
                        except errors.QueryCanceled as exc:
                            error = exc.__class__.__name__
                            cur.execute("SET statement_timeout = 0")
                            plan_class, plan_text, plan_cost = "", "", 0.0
                        except Exception as exc:  # noqa: BLE001
                            error = exc.__class__.__name__
                            try:
                                cur.execute("ROLLBACK")
                            except Exception:
                                pass
                            plan_class, plan_text, plan_cost = "", "", 0.0
                        valid_profile = bool(profile.get("valid", False))
                        out_row = {
                            "query_no": query_no,
                            "c4_query_no": row["query_no"],
                            "query_id": query_id,
                            "predicate": predicate,
                            "cache_key": row["cache_key"],
                            "repeat": repeat,
                            "route": route,
                            "plan_class": plan_class,
                            "plan_text": plan_text,
                            "planner_total_cost": plan_cost,
                            "latency_ms": latency,
                            "returned": len(ids),
                            "ids": ",".join(str(x) for x in ids),
                            "visited_tuples": profile.get("visited_tuples", 0) if valid_profile else 0,
                            "returned_tuples": profile.get("returned_tuples", 0) if valid_profile else 0,
                            "vector_search_ms": profile.get("vector_search_ms", 0.0) if valid_profile else 0.0,
                            "error": error,
                        }
                        rows.append(out_row)
                        writer.writerow(out_row)
                        f.flush()
                if args.progress_every and (query_no + 1) % args.progress_every == 0:
                    partial = summarize(rows, args)
                    print(
                        f"progress {query_no + 1}/{len(workload)} cases={partial['cases']} "
                        f"wrong={partial['wrong_route_count']} ({partial['wrong_route_ratio']:.3f}) "
                        f"meaningful={partial['meaningful_wrong_route_count']} ({partial['meaningful_wrong_route_ratio']:.3f}) "
                        f"oracle={partial['oracle_speedup_vs_default']:.3f}x",
                        flush=True,
                    )

    summary = summarize(rows, args)
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    predicate_out = args.predicate_out or args.out.with_name(args.out.stem + "_predicates.csv")
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    with predicate_out.open("w", newline="") as f:
        predicate_rows = summary["predicate_summary"]
        writer = csv.DictWriter(f, fieldnames=list(predicate_rows[0].keys()) if predicate_rows else ["predicate"])
        writer.writeheader()
        writer.writerows(predicate_rows)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {summary_out}", flush=True)
    print(f"wrote {predicate_out}", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in {'predicate_summary', 'top_wrong_examples'}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
