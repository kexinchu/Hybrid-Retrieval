from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import psycopg
from psycopg import errors

from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import ATTR_FILTERS, recall_at_k
from pgvector_predicate_guidance_benchmark import FILTER_ATOMS, load_truth


BFS_TABLE = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs"
BFS_INDEX = "amazon_grocery_reviews_10m_pgvector_samegraph_bfs_hnsw"
BASELINE_TABLE = Path("results/hybrid_vector_db/pgvector_d1_d2_d3_selectivity_merged_q20r2_warmall_measured_20260705.csv")


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def ensure_functions(cur: psycopg.Cursor) -> None:
    functions = [
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_activate(regclass, text[], text) "
        "RETURNS int4 AS 'vector' LANGUAGE C VOLATILE PARALLEL UNSAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_guidance_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_last_scan_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_reset_scan_profile() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_profile() "
        "RETURNS text AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
        "CREATE OR REPLACE FUNCTION vector_hnsw_metadata_cache_reset() "
        "RETURNS void AS 'vector' LANGUAGE C VOLATILE PARALLEL SAFE",
    ]
    for sql in functions:
        cur.execute(sql)
    cur.execute("SELECT vector_hnsw_metadata_cache_profile()")


def configure_common(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
    cur.execute(f"SET hnsw.iterative_scan = {args.iterative_scan}")
    cur.execute(f"SET hnsw.max_scan_tuples = {int(args.max_scan_tuples)}")
    cur.execute(f"SET hnsw.scan_mem_multiplier = {float(args.scan_mem_multiplier)}")
    cur.execute(f"SET hnsw.metadata_cache_max_mb = {int(args.d3_cache_mb)}")
    cur.execute("SET hnsw.filter_strategy = off")
    cur.execute("SET hnsw.page_access = off")
    cur.execute("SET hnsw.index_page_access = off")
    cur.execute("SET jit = off")


def configure_hnsw(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    configure_common(cur, args)
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_indexonlyscan = on")
    cur.execute("SET enable_bitmapscan = on")
    cur.execute("SET enable_sort = off")


def configure_prefilter(cur: psycopg.Cursor, args: argparse.Namespace) -> None:
    configure_common(cur, args)
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_indexonlyscan = off")
    cur.execute("SET enable_bitmapscan = on")
    cur.execute("SET enable_sort = on")


def activate_guidance(cur: psycopg.Cursor, args: argparse.Namespace, filter_name: str) -> dict[str, object]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute(
        "SELECT vector_hnsw_guidance_activate(%s::regclass, %s::text[], 'bloom')",
        (args.bfs_index, FILTER_ATOMS[filter_name]),
    )
    cur.execute("SELECT vector_hnsw_guidance_profile()")
    return json.loads(cur.fetchone()[0])


def prewarm_d3(cur: psycopg.Cursor, args: argparse.Namespace, filters: list[tuple[str, str, str]]) -> None:
    configure_hnsw(cur, args)
    cur.execute("SELECT vector_hnsw_metadata_cache_reset()")
    for filter_name, _, _ in filters:
        activate_guidance(cur, args, filter_name)
    cur.execute("SELECT vector_hnsw_guidance_reset()")


def fetch_query_vector(cur: psycopg.Cursor, table: str, query_id: int) -> str:
    cur.execute(f"SELECT embedding::text FROM {table} WHERE id = %s", (int(query_id),))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"missing query id {query_id}")
    return str(row[0])


def run_hnsw_query(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_vector: str, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {args.bfs_table}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT {int(k)}
        """,
        (query_vector,),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def run_prefilter_query(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_vector: str, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {args.bfs_table}
        WHERE {predicate}
        ORDER BY embedding <-> %s::vector
        LIMIT {int(k)}
        """,
        (query_vector,),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def run_hnsw_query_by_id(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_id: int, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {args.bfs_table}
        WHERE {predicate}
        ORDER BY embedding <-> (SELECT embedding FROM {args.bfs_table} WHERE id = %s)
        LIMIT {int(k)}
        """,
        (int(query_id),),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def run_prefilter_query_by_id(cur: psycopg.Cursor, args: argparse.Namespace, predicate: str, query_id: int, k: int) -> tuple[list[int], dict[str, object]]:
    cur.execute("SELECT vector_hnsw_guidance_reset()")
    cur.execute("SELECT vector_hnsw_reset_scan_profile()")
    cur.execute(
        f"""
        SELECT id
        FROM {args.bfs_table}
        WHERE {predicate}
        ORDER BY embedding <-> (SELECT embedding FROM {args.bfs_table} WHERE id = %s)
        LIMIT {int(k)}
        """,
        (int(query_id),),
    )
    ids = [int(row[0]) for row in cur.fetchall()]
    cur.execute("SELECT vector_hnsw_last_scan_profile()")
    return ids, json.loads(cur.fetchone()[0])


def run_chosen_query_by_id(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    route: str,
    filter_name: str,
    predicate: str,
    query_id: int,
) -> tuple[list[int], dict[str, object]]:
    if route == "prefilter_exact":
        return run_prefilter_query_by_id(cur, args, predicate, query_id, args.k)
    activate_guidance(cur, args, filter_name)
    return run_hnsw_query_by_id(cur, args, predicate, query_id, args.k)


def safe_run(cur: psycopg.Cursor, fn, args: argparse.Namespace):
    try:
        return fn(), ""
    except errors.QueryCanceled as exc:
        cur.execute("SET statement_timeout = 0")
        return None, exc.__class__.__name__
    except Exception as exc:  # noqa: BLE001
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
        configure_common(cur, args)
        return None, exc.__class__.__name__


def calibrate_filter(
    cur: psycopg.Cursor,
    args: argparse.Namespace,
    filter_name: str,
    predicate: str,
    calibration_nos: list[int],
    query_by_no: dict[int, int],
) -> dict[str, object]:
    hnsw_ms: list[float] = []
    prefilter_ms: list[float] = []
    errors: list[str] = []

    for qno in calibration_nos:
        qid = query_by_no[qno]

        configure_hnsw(cur, args)
        value, err = safe_run(
            cur,
            lambda: timed_ms(lambda: run_chosen_query_by_id(cur, args, "hnsw_d123", filter_name, predicate, qid)),
            args,
        )
        if err:
            errors.append(f"hnsw:{err}")
        else:
            hnsw_ms.append(float(value[1]))

        configure_prefilter(cur, args)
        value, err = safe_run(
            cur,
            lambda: timed_ms(lambda: run_prefilter_query_by_id(cur, args, predicate, qid, args.k)),
            args,
        )
        if err:
            errors.append(f"prefilter:{err}")
        else:
            prefilter_ms.append(float(value[1]))

    hmean = statistics.fmean(hnsw_ms) if hnsw_ms else float("inf")
    pmean = statistics.fmean(prefilter_ms) if prefilter_ms else float("inf")
    hp50 = statistics.median(hnsw_ms) if hnsw_ms else float("inf")
    pp50 = statistics.median(prefilter_ms) if prefilter_ms else float("inf")
    hcv = (statistics.pstdev(hnsw_ms) / hmean) if len(hnsw_ms) > 1 and hmean else float("inf")
    pcv = (statistics.pstdev(prefilter_ms) / pmean) if len(prefilter_ms) > 1 and pmean else float("inf")

    enough = len(hnsw_ms) >= args.min_samples and len(prefilter_ms) >= args.min_samples
    stable = max(hcv, pcv) <= args.max_cv
    gap = pmean * (1.0 + args.guard_band) < hmean
    chosen = "prefilter_exact" if enough and stable and gap else "hnsw_d123"

    return {
        "filter_name": filter_name,
        "chosen_route": chosen,
        "samples_hnsw": len(hnsw_ms),
        "samples_prefilter": len(prefilter_ms),
        "hnsw_mean_ms": hmean,
        "prefilter_mean_ms": pmean,
        "hnsw_p50_ms": hp50,
        "prefilter_p50_ms": pp50,
        "hnsw_cv": hcv,
        "prefilter_cv": pcv,
        "guard_band": args.guard_band,
        "enough_samples": enough,
        "stable": stable,
        "prefilter_gap_passed": gap,
        "errors": ";".join(errors),
    }


def load_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        return {row["Filter"]: row for row in csv.DictReader(f)}


def write_merged_table(args: argparse.Namespace, rows: list[dict[str, object]], out: Path) -> None:
    baseline = load_baseline(args.baseline_table)
    ok_by_filter: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not row["error"]:
            ok_by_filter.setdefault(str(row["filter_name"]), []).append(row)

    fields = [
        "Selectivity",
        "Filter",
        "Original pgvector",
        "Design 1",
        "Design 1 + Design 2",
        "Design 1 + Design 2 + Design 3",
        "Design 1 + Design 2 + Design 3 + Design 4",
        "D1 speedup",
        "D1+D2 speedup",
        "D1+D2 + D3 speedup",
        "D1+D2+D3 + D4 speedup",
        "D4 route",
        "D4 samples",
        "D4 recall",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for filter_name, _, _ in ATTR_FILTERS:
            base = baseline[filter_name]
            ok = ok_by_filter.get(filter_name, [])
            d4 = statistics.fmean(float(r["end_to_end_ms"]) for r in ok) if ok else 0.0
            original = float(base["Original pgvector"])
            d1 = float(base["Design 1"])
            d12 = float(base["Design 1 + Design 2"])
            d123 = float(base["Design 1 + Design 2 + Design 3"])
            writer.writerow(
                {
                    "Selectivity": base["Selectivity"],
                    "Filter": filter_name,
                    "Original pgvector": f"{original:.4f}",
                    "Design 1": f"{d1:.4f}",
                    "Design 1 + Design 2": f"{d12:.4f}",
                    "Design 1 + Design 2 + Design 3": f"{d123:.4f}",
                    "Design 1 + Design 2 + Design 3 + Design 4": f"{d4:.4f}",
                    "D1 speedup": base["D1 speedup"],
                    "D1+D2 speedup": base["D1+D2 speedup"],
                    "D1+D2 + D3 speedup": base["D1+D2 + D3 speedup"],
                    "D1+D2+D3 + D4 speedup": f"{(original / d4):.4f}" if d4 else "0.0000",
                    "D4 route": ok[0]["chosen_route"] if ok else "error",
                    "D4 samples": len(ok),
                    "D4 recall": f"{statistics.fmean(float(r['recall']) for r in ok):.4f}" if ok else "0.0000",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark D1+D2+D3+D4 historical path calibration prototype.")
    parser.add_argument("--bfs-table", default=BFS_TABLE)
    parser.add_argument("--bfs-index", default=BFS_INDEX)
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--baseline-table", type=Path, default=BASELINE_TABLE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--calibration-queries", type=int, default=8)
    parser.add_argument("--warmup-all-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--iterative-scan", default="strict_order", choices=["off", "strict_order", "relaxed_order"])
    parser.add_argument("--max-scan-tuples", type=int, default=200000)
    parser.add_argument("--scan-mem-multiplier", type=float, default=8.0)
    parser.add_argument("--d3-cache-mb", type=int, default=1024)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--max-cv", type=float, default=2.0)
    parser.add_argument("--guard-band", type=float, default=0.10)
    parser.add_argument("--progress-queries", type=int, default=10)
    args = parser.parse_args()

    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[args.query_offset : args.query_offset + args.queries]
    calibration_nos = query_nos[: args.calibration_queries]
    filters = list(ATTR_FILTERS)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        ensure_functions(cur)
        prewarm_d3(cur, args, filters)

        for filter_name, selectivity, predicate in filters:
            print(f"calibrating filter={filter_name}", flush=True)
            cal = calibrate_filter(cur, args, filter_name, predicate, calibration_nos, query_by_no)
            calibration_rows.append(cal)

            if args.warmup_all_queries:
                if cal["chosen_route"] == "prefilter_exact":
                    configure_prefilter(cur, args)
                else:
                    configure_hnsw(cur, args)
                for qno in query_nos:
                    qid = query_by_no[qno]
                    if cal["chosen_route"] == "prefilter_exact":
                        fn = lambda qid=qid, predicate=predicate: run_prefilter_query_by_id(cur, args, predicate, qid, args.k)
                    else:
                        fn = lambda qid=qid, predicate=predicate, filter_name=filter_name: run_chosen_query_by_id(cur, args, "hnsw_d123", filter_name, predicate, qid)
                    safe_run(cur, fn, args)

            if cal["chosen_route"] == "prefilter_exact":
                configure_prefilter(cur, args)
            else:
                configure_hnsw(cur, args)
            for idx, qno in enumerate(query_nos, start=1):
                qid = query_by_no[qno]
                for repeat in range(args.repeats):
                    error = ""
                    ids: list[int] = []
                    profile: dict[str, object] = {}
                    latency = 0.0
                    route = str(cal["chosen_route"])
                    value, err = safe_run(
                        cur,
                        lambda route=route, predicate=predicate, filter_name=filter_name, qid=qid: timed_ms(
                            lambda: run_chosen_query_by_id(cur, args, route, filter_name, predicate, qid)
                        ),
                        args,
                    )
                    if err:
                        error = err
                    else:
                        (ids, profile), latency = value
                    rows.append(
                        {
                            "selectivity": selectivity,
                            "filter_name": filter_name,
                            "query_no": qno,
                            "query_id": qid,
                            "repeat": repeat,
                            "chosen_route": route,
                            "end_to_end_ms": latency,
                            "recall": recall_at_k(ids, truth[(filter_name, qno)], args.k) if not error else 0.0,
                            "visited_tuples": profile.get("visited_tuples", 0),
                            "guidance_checks": profile.get("guidance_checks", 0),
                            "guidance_skips": profile.get("guidance_skips", 0),
                            "returned": len(ids),
                            "ids": ",".join(str(x) for x in ids),
                            "error": error,
                        }
                    )
                if args.progress_queries and idx % args.progress_queries == 0:
                    ok = [r for r in rows if r["filter_name"] == filter_name and not r["error"]]
                    if ok:
                        print(
                            f"progress filter={filter_name} route={cal['chosen_route']} "
                            f"queries={idx}/{len(query_nos)} e2e={statistics.fmean(float(r['end_to_end_ms']) for r in ok):.2f}ms",
                            flush=True,
                        )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cal_out = args.out.with_name(args.out.stem + "_calibration.csv")
    with cal_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calibration_rows)

    table_out = args.out.with_name(args.out.stem + "_merged_table.csv")
    write_merged_table(args, rows, table_out)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {cal_out}", flush=True)
    print(f"wrote {table_out}", flush=True)


if __name__ == "__main__":
    main()
