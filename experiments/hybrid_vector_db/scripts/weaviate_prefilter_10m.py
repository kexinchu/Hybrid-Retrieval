from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import struct
import sys
import time
import uuid
from http.client import HTTPConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


CLASS_NAME = "AmazonGroceryReview"

FILTERS: list[tuple[str, str, dict[str, object]]] = [
    ("50.32%", "popular_ge1000", {"path": ["item_rating_number"], "operator": "GreaterThanEqual", "valueInt": 1000}),
    (
        "21.89%",
        "price_10_to_20",
        {
            "operator": "And",
            "operands": [
                {"path": ["has_price"], "operator": "Equal", "valueBoolean": True},
                {"path": ["price"], "operator": "GreaterThan", "valueNumber": 10.0},
                {"path": ["price"], "operator": "LessThanEqual", "valueNumber": 20.0},
            ],
        },
    ),
    (
        "9.59%",
        "rating5_price_le10",
        {
            "operator": "And",
            "operands": [
                {"path": ["has_price"], "operator": "Equal", "valueBoolean": True},
                {"path": ["price"], "operator": "LessThanEqual", "valueNumber": 10.0},
                {"path": ["rating"], "operator": "Equal", "valueNumber": 5.0},
            ],
        },
    ),
    ("5.88%", "long_review_ge500", {"path": ["review_text_len"], "operator": "GreaterThanEqual", "valueInt": 500}),
    (
        "2.34%",
        "grocery_rating5",
        {
            "operator": "And",
            "operands": [
                {"path": ["main_category"], "operator": "Equal", "valueText": "Grocery"},
                {"path": ["rating"], "operator": "Equal", "valueNumber": 5.0},
            ],
        },
    ),
    (
        "1.01%",
        "grocery_helpful",
        {
            "operator": "And",
            "operands": [
                {"path": ["main_category"], "operator": "Equal", "valueText": "Grocery"},
                {"path": ["helpful_vote"], "operator": "GreaterThanEqual", "valueInt": 1},
            ],
        },
    ),
    ("0.61%", "helpful_ge20", {"path": ["helpful_vote"], "operator": "GreaterThanEqual", "valueInt": 20}),
    (
        "0.21%",
        "grocery_long500",
        {
            "operator": "And",
            "operands": [
                {"path": ["main_category"], "operator": "Equal", "valueText": "Grocery"},
                {"path": ["review_text_len"], "operator": "GreaterThanEqual", "valueInt": 500},
            ],
        },
    ),
]


def post_json(base_url: str, path: str, payload: dict[str, object], timeout: int = 300) -> dict[str, object]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8")) if body else {}


def get_json(base_url: str, path: str, timeout: int = 300) -> dict[str, object]:
    with urlopen(base_url + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def object_uuid(row_id: int) -> str:
    return str(uuid.UUID(int=row_id + 1))


def class_exists(base_url: str) -> bool:
    try:
        get_json(base_url, f"/v1/schema/{CLASS_NAME}")
        return True
    except HTTPError as e:
        if e.code == 404:
            return False
        raise


def create_schema(base_url: str, vector_dim: int, ef: int | None) -> None:
    if class_exists(base_url):
        return
    vector_index_config: dict[str, object] = {
        "distance": "l2-squared",
        "efConstruction": 128,
        "maxConnections": 32,
        "flatSearchCutoff": 40000,
        "filterStrategy": "acorn",
    }
    if ef is not None:
        vector_index_config["ef"] = int(ef)
    payload = {
        "class": CLASS_NAME,
        "description": f"Amazon Grocery 10M review vectors, dim={vector_dim}",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": vector_index_config,
        "properties": [
            {"name": "row_id", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
            {"name": "rating", "dataType": ["number"], "indexFilterable": True, "indexRangeFilters": True},
            {"name": "verified_purchase", "dataType": ["boolean"], "indexFilterable": True},
            {"name": "helpful_vote", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
            {"name": "review_text_len", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
            {"name": "main_category", "dataType": ["text"], "tokenization": "field", "indexFilterable": True},
            {"name": "price", "dataType": ["number"], "indexFilterable": True, "indexRangeFilters": True},
            {"name": "has_price", "dataType": ["boolean"], "indexFilterable": True},
            {"name": "item_rating_number", "dataType": ["int"], "indexFilterable": True, "indexRangeFilters": True},
        ],
    }
    post_json(base_url, "/v1/schema", payload)


def graphql(base_url: str, query: str) -> dict[str, object]:
    return post_json(base_url, "/v1/graphql", {"query": query})


def weaviate_count(base_url: str) -> int:
    if not class_exists(base_url):
        return 0
    q = f"""
    {{
      Aggregate {{
        {CLASS_NAME} {{
          meta {{ count }}
        }}
      }}
    }}
    """
    data = graphql(base_url, q)
    return int(data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])


def json_to_graphql(value: object, key: str | None = None) -> str:
    if isinstance(value, dict):
        return "{" + " ".join(f"{k}:{json_to_graphql(v, k)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(json_to_graphql(v) for v in value) + "]"
    if isinstance(value, str):
        if key == "operator":
            return value
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def load_truth(path: Path) -> tuple[dict[tuple[str, int], list[int]], dict[int, int]]:
    truth: dict[tuple[str, int], list[int]] = {}
    query_by_no: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "pre_filter_exact":
                continue
            qno = int(row["query_no"])
            fname = row["filter_name"]
            truth[(fname, qno)] = [int(x) for x in row["exact_filtered_topk_ids"].split(",") if x]
            query_by_no[qno] = int(row["query_id"])
    return truth, query_by_no


def recall_at_k(ids: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ids[:k]) & set(truth[:k])) / min(k, len(truth))


def import_batch(conn: HTTPConnection, objects: list[dict[str, object]]) -> dict[str, object]:
    body = json.dumps({"objects": objects}, separators=(",", ":")).encode("utf-8")
    conn.request("POST", "/v1/batch/objects", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    if resp.status >= 300:
        raise RuntimeError(f"batch import failed status={resp.status} body={data[:1000]!r}")
    return json.loads(data.decode("utf-8")) if data else {}


def import_data(args: argparse.Namespace, xb: np.memmap, rows: int) -> None:
    current = weaviate_count(args.base_url)
    start = max(current, args.start_row)
    target_rows = min(rows, args.rows)
    if start >= target_rows:
        print(f"import already complete count={current} target={target_rows}", flush=True)
        return

    print(f"import start={start} target={target_rows} batch_size={args.batch_size}", flush=True)
    conn = HTTPConnection(args.host, args.port, timeout=600)
    batch: list[dict[str, object]] = []
    t0 = time.perf_counter()
    last = t0
    imported = start

    with args.csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if i >= target_rows:
                break
            price = float(row["price"]) if row["has_price"] == "True" and row["price"] else 0.0
            obj = {
                "class": CLASS_NAME,
                "id": object_uuid(i),
                "properties": {
                    "row_id": i,
                    "rating": float(row["rating"]),
                    "verified_purchase": row["verified_purchase"] == "True",
                    "helpful_vote": int(row["helpful_vote"]),
                    "review_text_len": int(row["review_text_len"]),
                    "main_category": row["main_category"],
                    "price": price,
                    "has_price": row["has_price"] == "True",
                    "item_rating_number": int(float(row["item_rating_number"])),
                },
                "vector": np.asarray(xb[i], dtype=np.float32).tolist(),
            }
            batch.append(obj)
            if len(batch) >= args.batch_size:
                res = import_batch(conn, batch)
                if any("result" in item and item["result"].get("errors") for item in res):
                    raise RuntimeError(f"batch had errors: {res[:2]}")
                imported = i + 1
                batch.clear()
                now = time.perf_counter()
                if now - last >= args.progress_seconds:
                    rate = (imported - start) / max(now - t0, 1e-9)
                    print(f"imported={imported}/{target_rows} rate={rate:.1f} rows/s elapsed={(now-t0)/60:.1f}m", flush=True)
                    last = now
        if batch:
            import_batch(conn, batch)
            imported = target_rows
    conn.close()
    print(f"import done rows={imported} elapsed_min={(time.perf_counter()-t0)/60:.2f}", flush=True)


PROM_SUM_RE = re.compile(
    r'^queries_filtered_vector_durations_ms_sum\{class_name="(?P<class>[^"]+)",operation="(?P<op>[^"]+)",shard_name="(?P<shard>[^"]+)"\} (?P<value>[-+0-9.eE]+)$'
)
PROM_COUNT_RE = re.compile(
    r'^queries_filtered_vector_durations_ms_count\{class_name="(?P<class>[^"]+)",operation="(?P<op>[^"]+)",shard_name="(?P<shard>[^"]+)"\} (?P<value>[-+0-9.eE]+)$'
)


def scrape_filtered_metrics(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    with urlopen(f"http://{args.host}:{args.metrics_port}/metrics", timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for line in text.splitlines():
        m = PROM_SUM_RE.match(line)
        if m and m.group("class") == CLASS_NAME:
            sums[m.group("op")] = sums.get(m.group("op"), 0.0) + float(m.group("value"))
        m = PROM_COUNT_RE.match(line)
        if m and m.group("class") == CLASS_NAME:
            counts[m.group("op")] = counts.get(m.group("op"), 0.0) + float(m.group("value"))
    return {op: (sums.get(op, 0.0), counts.get(op, 0.0)) for op in {"filter", "vector", "objects", "sort"}}


def metric_delta(before: dict[str, tuple[float, float]], after: dict[str, tuple[float, float]], op: str) -> float:
    sum_delta = after.get(op, (0.0, 0.0))[0] - before.get(op, (0.0, 0.0))[0]
    count_delta = after.get(op, (0.0, 0.0))[1] - before.get(op, (0.0, 0.0))[1]
    if count_delta <= 0:
        return math.nan
    return sum_delta / count_delta


def mean_observed(values: list[float]) -> float:
    observed = [x for x in values if not math.isnan(x)]
    return statistics.fmean(observed) if observed else math.nan


def query_count(base_url: str, where: dict[str, object]) -> int:
    q = f"""
    {{
      Aggregate {{
        {CLASS_NAME}(where:{json_to_graphql(where)}) {{
          meta {{ count }}
        }}
      }}
    }}
    """
    data = graphql(base_url, q)
    return int(data["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"])


def run_queries(args: argparse.Namespace, xb: np.memmap) -> None:
    truth, query_by_no = load_truth(args.truth_csv)
    query_nos = sorted(query_by_no)[: args.queries]
    rows_out: list[dict[str, object]] = []
    filter_counts = {name: query_count(args.base_url, where) for _, name, where in FILTERS}
    print(f"filter_counts={filter_counts}", flush=True)

    for target_rate, filter_name, where in FILTERS:
        recalls: list[float] = []
        total_ms: list[float] = []
        filter_ms: list[float] = []
        vector_ms: list[float] = []
        objects_ms: list[float] = []
        sort_ms: list[float] = []
        returned: list[int] = []
        for qno in query_nos:
            query_id = query_by_no[qno]
            vector = np.asarray(xb[query_id], dtype=np.float32).tolist()
            gq = f"""
            {{
              Get {{
                {CLASS_NAME}(
                  nearVector:{{vector:{json_to_graphql(vector)}}}
                  where:{json_to_graphql(where)}
                  limit:{args.k}
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
            elapsed = (time.perf_counter() - t0) * 1000
            after_metrics = scrape_filtered_metrics(args)
            if "errors" in data:
                raise RuntimeError(data["errors"])
            got = [int(obj["row_id"]) for obj in data["data"]["Get"][CLASS_NAME]]
            truth_ids = truth[(filter_name, qno)]
            recalls.append(recall_at_k(got, truth_ids, args.k))
            total_ms.append(elapsed)
            filter_ms.append(metric_delta(before_metrics, after_metrics, "filter"))
            vector_ms.append(metric_delta(before_metrics, after_metrics, "vector"))
            objects_ms.append(metric_delta(before_metrics, after_metrics, "objects"))
            sort_ms.append(metric_delta(before_metrics, after_metrics, "sort"))
            returned.append(len(got))
            rows_out.append(
                {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "query_no": qno,
                    "query_id": query_id,
                    "sql_rows": filter_counts[filter_name],
                    "pre_recall": recalls[-1],
                    "pre_latency_ms": elapsed,
                    "pre_sql_ms": filter_ms[-1],
                    "pre_vector_search_ms": vector_ms[-1],
                    "objects_ms": objects_ms[-1],
                    "sort_ms": sort_ms[-1],
                    "returned": returned[-1],
                }
            )
        print(
            f"filter={target_rate} name={filter_name} rows={filter_counts[filter_name]} "
            f"recall={statistics.mean(recalls):.3f} latency={statistics.mean(total_ms):.2f} "
            f"filter_ms={mean_observed(filter_ms):.2f} "
            f"vector_ms={mean_observed(vector_ms):.2f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    summary_out = args.out.with_name(args.out.stem + "_summary.csv")
    with summary_out.open("w", newline="") as f:
        fieldnames = ["filter", "filter_name", "pre_recall", "pre_latency_ms", "pre_sql_ms", "sql_rows", "pre_vector_search_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for target_rate, filter_name, _ in FILTERS:
            items = [r for r in rows_out if r["filter_name"] == filter_name]
            writer.writerow(
                {
                    "filter": target_rate,
                    "filter_name": filter_name,
                    "pre_recall": statistics.mean(float(r["pre_recall"]) for r in items),
                    "pre_latency_ms": statistics.mean(float(r["pre_latency_ms"]) for r in items),
                    "pre_sql_ms": mean_observed([float(r["pre_sql_ms"]) for r in items]),
                    "sql_rows": items[0]["sql_rows"],
                    "pre_vector_search_ms": mean_observed([float(r["pre_vector_search_ms"]) for r in items]),
                }
            )
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {summary_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--metrics-port", type=int, default=2112)
    parser.add_argument("--csv", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv"))
    parser.add_argument("--fbin", type=Path, default=Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin"))
    parser.add_argument("--truth-csv", type=Path, default=Path("results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hybrid_vector_db/weaviate_prefilter_10m_q100_20260606.csv"))
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--progress-seconds", type=int, default=60)
    parser.add_argument("--ef", type=int)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--query-only", action="store_true")
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"

    xb, rows, dim = read_fbin_memmap(args.fbin, args.rows)
    create_schema(args.base_url, dim, args.ef)
    if not args.query_only and not args.skip_import:
        import_data(args, xb, rows)
    run_queries(args, xb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
