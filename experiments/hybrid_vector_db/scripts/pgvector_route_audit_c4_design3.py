from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import psycopg

from common_pg import pg_config_from_env
from pgvector_c4_query_filter_cache_benchmark import load_workload
from pgvector_route_audit_design3 import (
    ROUTES,
    configure_common,
    configure_route,
    explain_route,
    run_route,
    timed_ms,
)


def load_query_vectors(cur: psycopg.Cursor, table: str, query_ids: list[int]) -> dict[int, str]:
    cur.execute(
        f"""
        SELECT id, embedding
        FROM {table}
        WHERE id = ANY(%s::bigint[])
        """,
        (query_ids,),
    )
    out = {}
    for row in cur.fetchall():
        value = row[1]
        if isinstance(value, str):
            vector = value
        else:
            vector = "[" + ",".join(f"{float(x):.7g}" for x in value) + "]"
        out[int(row[0])] = vector
    missing = [qid for qid in query_ids if qid not in out]
    if missing:
        raise RuntimeError(f"missing query vectors: {missing[:5]}")
    return out


def warmup(cur: psycopg.Cursor, args: argparse.Namespace, workload: list[dict[str, object]], vectors: dict[int, str]) -> None:
    if args.warmup_queries <= 0:
        return
    for item in workload[: args.warmup_queries]:
        predicate = str(item["predicate"])
        qid = int(item["query_id_int"])
        for route in ROUTES:
            try:
                configure_route(cur, args, route)
                run_route(cur, args.table, predicate, vectors[qid], args.k)
            except Exception:
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
                configure_common(cur, args)


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    summary = out.with_name(out.stem + "_summary.csv")
    by_query: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_query.setdefault(str(row["workload_no"]), {})[str(row["route"])] = row

    query_rows: list[dict[str, object]] = []
    for _, routes in sorted(by_query.items(), key=lambda kv: int(kv[0])):
        ok_routes = [r for r in routes.values() if not r["error"] and float(r["latency_ms"]) > 0]
        if not ok_routes or "default_planner" not in routes:
            continue
        fastest = min(ok_routes, key=lambda r: float(r["latency_ms"]))
        default = routes["default_planner"]
        default_ms = float(default["latency_ms"])
        fastest_ms = float(fastest["latency_ms"])
        query_rows.append(
            {
                "workload_no": default["workload_no"],
                "query_no": default["query_no"],
                "query_id": default["query_id"],
                "cache_key": default["cache_key"],
                "predicate": default["predicate"],
                "default_plan_class": default["plan_class"],
                "fastest_route": fastest["route"],
                "fastest_plan_class": fastest["plan_class"],
                "default_ms": default_ms,
                "fastest_ms": fastest_ms,
                "oracle_speedup": default_ms / fastest_ms if fastest_ms else 0.0,
                "route_mismatch": default["plan_class"] != fastest["plan_class"],
                "default_planner_cost": default["planner_total_cost"],
                "fastest_planner_cost": fastest["planner_total_cost"],
            }
        )

    total = len(query_rows)
    mismatches = [r for r in query_rows if r["route_mismatch"]]
    material = [r for r in query_rows if r["route_mismatch"] and float(r["oracle_speedup"]) >= 1.1]
    big = [r for r in query_rows if r["route_mismatch"] and float(r["oracle_speedup"]) >= 1.5]
    default_total = sum(float(r["default_ms"]) for r in query_rows)
    oracle_total = sum(float(r["fastest_ms"]) for r in query_rows)
    default_plans: dict[str, int] = {}
    fastest_plans: dict[str, int] = {}
    for row in query_rows:
        default_plans[str(row["default_plan_class"])] = default_plans.get(str(row["default_plan_class"]), 0) + 1
        fastest_plans[str(row["fastest_plan_class"])] = fastest_plans.get(str(row["fastest_plan_class"]), 0) + 1

    fields = [
        "metric",
        "value",
    ]
    summary_rows = [
        {"metric": "queries", "value": total},
        {"metric": "route_mismatch_count", "value": len(mismatches)},
        {"metric": "route_mismatch_ratio", "value": len(mismatches) / total if total else 0.0},
        {"metric": "material_mismatch_1p1_count", "value": len(material)},
        {"metric": "material_mismatch_1p1_ratio", "value": len(material) / total if total else 0.0},
        {"metric": "big_mismatch_1p5_count", "value": len(big)},
        {"metric": "big_mismatch_1p5_ratio", "value": len(big) / total if total else 0.0},
        {"metric": "default_total_ms", "value": default_total},
        {"metric": "oracle_total_ms", "value": oracle_total},
        {"metric": "oracle_total_speedup", "value": default_total / oracle_total if oracle_total else 0.0},
        {"metric": "default_plan_counts", "value": default_plans},
        {"metric": "fastest_plan_counts", "value": fastest_plans},
    ]

    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    query_summary = out.with_name(out.stem + "_query_summary.csv")
    if query_rows:
        with query_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(query_rows[0].keys()))
            writer.writeheader()
            writer.writerows(query_rows)

    print(f"wrote {summary}", flush=True)
    print(f"wrote {query_summary}", flush=True)
    print(
        f"queries={total} mismatches={len(mismatches)} material>=1.1x={len(material)} "
        f"big>=1.5x={len(big)} oracle_speedup={default_total / oracle_total if oracle_total else 0.0:.3f}x",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Route audit on C4-derived pgvector workload for Design 3.")
    parser.add_argument("--workload-csv", type=Path, default=Path("results/hybrid_vector_db/amazon_c4_pgvector_queries_400.csv"))
    parser.add_argument("--mode", default="mixed", choices=["price", "popularity", "mixed", "mixed_category"])
    parser.add_argument("--table", default="amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-queries", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--statement-timeout-ms", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-queries", type=int, default=25)
    args = parser.parse_args()

    workload = load_workload(args.workload_csv, args.query_offset + args.limit, args.mode)[args.query_offset :]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []

    fieldnames = [
        "workload_no",
        "query_no",
        "query_id",
        "cache_key",
        "predicate",
        "route",
        "repeat",
        "plan_class",
        "plan_text",
        "planner_startup_cost",
        "planner_total_cost",
        "planner_rows",
        "latency_ms",
        "returned",
        "visited_tuples",
        "returned_tuples",
        "vector_search_ms",
        "ids",
        "error",
    ]

    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        cur.execute("CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE")
        configure_common(cur, args)
        vectors = load_query_vectors(cur, args.table, [int(item["query_id_int"]) for item in workload])
        warmup(cur, args, workload, vectors)

        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, item in enumerate(workload, start=1):
                predicate = str(item["predicate"])
                qid = int(item["query_id_int"])
                for repeat in range(args.repeats):
                    route_order = ROUTES[:]
                    rng.shuffle(route_order)
                    for route in route_order:
                        error = ""
                        ids: list[int] = []
                        profile: dict[str, object] = {}
                        latency_ms = 0.0
                        try:
                            configure_route(cur, args, route)
                            plan_class, plan_text, startup_cost, total_cost, planner_rows = explain_route(cur, args.table, predicate, vectors[qid])
                            (ids, profile), latency_ms = timed_ms(lambda: run_route(cur, args.table, predicate, vectors[qid], args.k))
                        except Exception as exc:  # noqa: BLE001
                            error = exc.__class__.__name__
                            try:
                                cur.execute("ROLLBACK")
                            except Exception:
                                pass
                            configure_common(cur, args)
                            plan_class, plan_text, startup_cost, total_cost, planner_rows = "", "", 0.0, 0.0, 0
                        valid_hnsw_profile = bool(profile.get("valid", False))
                        row = {
                            "workload_no": idx,
                            "query_no": item["query_no"],
                            "query_id": qid,
                            "cache_key": item["cache_key"],
                            "predicate": predicate,
                            "route": route,
                            "repeat": repeat,
                            "plan_class": plan_class,
                            "plan_text": plan_text,
                            "planner_startup_cost": startup_cost,
                            "planner_total_cost": total_cost,
                            "planner_rows": planner_rows,
                            "latency_ms": latency_ms,
                            "returned": len(ids),
                            "visited_tuples": profile.get("visited_tuples", 0) if valid_hnsw_profile else 0,
                            "returned_tuples": profile.get("returned_tuples", 0) if valid_hnsw_profile else 0,
                            "vector_search_ms": profile.get("vector_search_ms", 0.0) if valid_hnsw_profile else 0.0,
                            "ids": ",".join(str(x) for x in ids),
                            "error": error,
                        }
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                if args.progress_queries and idx % args.progress_queries == 0:
                    recent = [r for r in rows if not r["error"]]
                    parts = []
                    for route in ROUTES:
                        vals = [float(r["latency_ms"]) for r in recent if r["route"] == route]
                        if vals:
                            parts.append(f"{route}={statistics.fmean(vals):.2f}ms")
                    print(f"progress queries={idx}/{len(workload)} " + " ".join(parts), flush=True)

    print(f"wrote {args.out}", flush=True)
    summarize(rows, args.out)


if __name__ == "__main__":
    main()
