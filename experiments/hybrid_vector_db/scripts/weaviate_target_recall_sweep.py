from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from weaviate_prefilter_10m import (
    CLASS_NAME,
    FILTERS,
    graphql,
    json_to_graphql,
    load_truth,
    mean_observed,
    metric_delta,
    query_count,
    read_fbin_memmap,
    recall_at_k,
    scrape_filtered_metrics,
)


def get_json(base_url: str, path: str, timeout: int = 60) -> dict[str, object]:
    with urlopen(base_url + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def put_json(base_url: str, path: str, payload: dict[str, object], timeout: int = 60) -> dict[str, object]:
    req = Request(
        base_url + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8")) if body else {}


def set_hnsw_config(base_url: str, ef: int, flat_search_cutoff: int) -> None:
    schema = get_json(base_url, f"/v1/schema/{CLASS_NAME}")
    vector_config = dict(schema["vectorIndexConfig"])
    if (
        int(vector_config.get("ef", -1)) == int(ef)
        and int(vector_config.get("flatSearchCutoff", 0)) == int(flat_search_cutoff)
    ):
        return
    vector_config["ef"] = int(ef)
    vector_config["flatSearchCutoff"] = int(flat_search_cutoff)
    schema["vectorIndexConfig"] = vector_config
    put_json(base_url, f"/v1/schema/{CLASS_NAME}", schema)


def query_once(
    args: argparse.Namespace,
    vector: np.ndarray,
    where: dict[str, object],
    limit: int,
) -> tuple[list[int], float, float, float, float, float]:
    gq = f"""
    {{
      Get {{
        {CLASS_NAME}(
          nearVector:{{vector:{json_to_graphql(np.asarray(vector, dtype=np.float32).tolist())}}}
          where:{json_to_graphql(where)}
          limit:{int(limit)}
        ) {{
          row_id
          _additional {{ distance id }}
        }}
      }}
    }}
    """
    before_metrics = scrape_filtered_metrics(args)
    t0 = time.perf_counter()
    data = graphql(args.base_url, gq)
    total_ms = (time.perf_counter() - t0) * 1000
    after_metrics = scrape_filtered_metrics(args)
    if "errors" in data:
        raise RuntimeError(data["errors"])
    got = [int(obj["row_id"]) for obj in data["data"]["Get"][CLASS_NAME]]
    return (
        got,
        total_ms,
        metric_delta(before_metrics, after_metrics, "filter"),
        metric_delta(before_metrics, after_metrics, "vector"),
        metric_delta(before_metrics, after_metrics, "objects"),
        metric_delta(before_metrics, after_metrics, "sort"),
    )


def evaluate_filter(
    args: argparse.Namespace,
    xb: np.memmap,
    truth: dict[tuple[str, int], list[int]],
    query_by_no: dict[int, int],
    filter_name: str,
    where: dict[str, object],
    limit: int,
) -> dict[str, float]:
    query_nos = sorted(query_by_no)[: args.queries]
    recalls: list[float] = []
    total_ms: list[float] = []
    filter_ms: list[float] = []
    vector_ms: list[float] = []
    objects_ms: list[float] = []
    sort_ms: list[float] = []
    returned: list[int] = []
    for qno in query_nos:
        query_id = query_by_no[qno]
        got, elapsed, f_ms, v_ms, o_ms, s_ms = query_once(args, xb[query_id], where, limit)
        recalls.append(recall_at_k(got[: args.k], truth[(filter_name, qno)], args.k))
        total_ms.append(elapsed)
        filter_ms.append(f_ms)
        vector_ms.append(v_ms)
        objects_ms.append(o_ms)
        sort_ms.append(s_ms)
        returned.append(len(got))
    return {
        "recall": statistics.fmean(recalls),
        "latency_ms": statistics.fmean(total_ms),
        "filter_ms": mean_observed(filter_ms),
        "vector_search_ms": mean_observed(vector_ms),
        "objects_ms": mean_observed(objects_ms),
        "sort_ms": mean_observed(sort_ms),
        "returned": statistics.fmean(returned),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], out: Path) -> None:
    groups: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["filter_name"]), int(row["ef"]), int(row["flat_search_cutoff"]), int(row["limit"])),
            [],
        ).append(row)
    summary_rows: list[dict[str, object]] = []
    order = {name: i for i, (_, name, _) in enumerate(FILTERS)}
    for (filter_name, ef, flat_search_cutoff, limit), items in sorted(
        groups.items(), key=lambda item: (order[item[0][0]], item[0][1], item[0][2], item[0][3])
    ):
        summary_rows.append(
            {
                "filter": items[0]["filter"],
                "filter_name": filter_name,
                "ef": ef,
                "flat_search_cutoff": flat_search_cutoff,
                "limit": limit,
                "repeats": len(items),
                "recall": statistics.fmean(float(r["recall"]) for r in items),
                "latency_ms": statistics.fmean(float(r["latency_ms"]) for r in items),
                "filter_ms": mean_observed([float(r["filter_ms"]) for r in items]),
                "filtered_rows": items[0]["filtered_rows"],
                "vector_search_ms": mean_observed([float(r["vector_search_ms"]) for r in items]),
                "objects_ms": mean_observed([float(r["objects_ms"]) for r in items]),
                "sort_ms": mean_observed([float(r["sort_ms"]) for r in items]),
                "returned": statistics.fmean(float(r["returned"]) for r in items),
            }
        )
    write_csv(out.with_name(out.stem + "_summary.csv"), summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--metrics-port", type=int, default=2112)
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/weaviate_target_recall06_sweep.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.6)
    parser.add_argument("--ef-values", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000, 16000])
    parser.add_argument("--flat-search-cutoff-values", type=int, nargs="+", default=[0])
    parser.add_argument("--limit-values", type=int, nargs="+", default=[10])
    parser.add_argument("--filter-names", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"

    xb, _, _ = read_fbin_memmap(args.fbin, args.rows)
    truth, query_by_no = load_truth(args.truth_csv)
    selected_filters = set(args.filter_names or [])
    filter_counts = {name: query_count(args.base_url, where) for _, name, where in FILTERS}

    sweep_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for target_rate, filter_name, where in FILTERS:
        if selected_filters and filter_name not in selected_filters:
            continue
        candidates: list[dict[str, object]] = []
        for ef in args.ef_values:
            for flat_search_cutoff in args.flat_search_cutoff_values:
                set_hnsw_config(args.base_url, ef, flat_search_cutoff)
                time.sleep(0.2)
                for limit in args.limit_values:
                    result = evaluate_filter(args, xb, truth, query_by_no, filter_name, where, limit)
                    row = {
                        "filter": target_rate,
                        "filter_name": filter_name,
                        "ef": ef,
                        "flat_search_cutoff": flat_search_cutoff,
                        "limit": limit,
                        "filtered_rows": filter_counts[filter_name],
                        **result,
                    }
                    candidates.append(row)
                    sweep_rows.append(row)
                    print(
                        f"filter={filter_name} ef={ef} flat_cutoff={flat_search_cutoff} limit={limit} "
                        f"recall={result['recall']:.3f} latency={result['latency_ms']:.2f} "
                        f"filter_ms={result['filter_ms']:.2f} vector_ms={result['vector_search_ms']:.2f}",
                        flush=True,
                    )
        above = [r for r in candidates if float(r["recall"]) >= args.target_recall]
        chosen = min(above, key=lambda r: float(r["latency_ms"])) if above else max(candidates, key=lambda r: float(r["recall"]))
        print(
            f"selected filter={filter_name} ef={chosen['ef']} limit={chosen['limit']} "
            f"flat_cutoff={chosen['flat_search_cutoff']} recall={float(chosen['recall']):.3f}",
            flush=True,
        )
        set_hnsw_config(args.base_url, int(chosen["ef"]), int(chosen["flat_search_cutoff"]))
        for repeat in range(args.repeats):
            result = evaluate_filter(args, xb, truth, query_by_no, filter_name, where, int(chosen["limit"]))
            final_rows.append(
                {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "ef": int(chosen["ef"]),
                    "flat_search_cutoff": int(chosen["flat_search_cutoff"]),
                    "limit": int(chosen["limit"]),
                    "repeat": repeat,
                    "filtered_rows": filter_counts[filter_name],
                    **result,
                }
            )
            print(
                f"  repeat={repeat} recall={result['recall']:.3f} latency={result['latency_ms']:.2f} "
                f"filter_ms={result['filter_ms']:.2f} vector_ms={result['vector_search_ms']:.2f}",
                flush=True,
            )

    write_csv(args.out, final_rows)
    summarize(final_rows, args.out)
    sweep_out = args.out.with_name(args.out.stem + "_sweep.csv")
    write_csv(sweep_out, sweep_rows)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.out.with_name(args.out.stem + '_summary.csv')}", flush=True)
    print(f"wrote {sweep_out}", flush=True)


if __name__ == "__main__":
    main()
