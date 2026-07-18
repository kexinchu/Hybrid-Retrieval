from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from common_pg import pg_config_from_env
from faiss_hnsw_sql_attribute_filter_10m import read_fbin_memmap


DEFAULT_FBIN = Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin")
DEFAULT_FILTERS = Path("experiments/hybrid_vector_db/configs/amazon10m_selectivity14_filters.csv")
DEFAULT_OUT = Path("results/hybrid_vector_db/amazon_selectivity14_exact_truth_q200_formal.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_filters(path: Path, selected: set[str]) -> list[dict[str, str]]:
    rows = read_csv(path)
    if selected:
        rows = [row for row in rows if row["filter_name"] in selected]
        missing = selected - {row["filter_name"] for row in rows}
        if missing:
            raise SystemExit(f"missing filter specs: {sorted(missing)}")
    if not rows:
        raise SystemExit(f"no filters in {path}")
    return rows


def sample_disjoint_query_ids(
    rows: int,
    excluded: set[int],
    count: int,
    seed: int,
) -> np.ndarray:
    if count > rows - len(excluded):
        raise ValueError("not enough rows for a disjoint final query set")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    batch = max(1024, count * 4)
    while len(chosen) < count:
        for query_id in rng.integers(0, rows, size=batch, endpoint=False):
            value = int(query_id)
            if value in excluded or value in chosen_set:
                continue
            chosen.append(value)
            chosen_set.add(value)
            if len(chosen) == count:
                break
    return np.asarray(chosen, dtype=np.int64)


def update_topk(
    top_dist: np.ndarray,
    top_ids: np.ndarray,
    distances: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> None:
    if distances.shape[0] == 0:
        return
    take = min(k, distances.shape[0])
    for query_pos in range(distances.shape[1]):
        column = distances[:, query_pos]
        threshold = np.partition(column, take - 1)[take - 1]
        local_pos = np.flatnonzero(column <= threshold)
        local_order = np.lexsort((candidate_ids[local_pos], column[local_pos]))[:take]
        selected = local_pos[local_order]
        merged_dist = np.concatenate((top_dist[query_pos], column[selected]))
        merged_ids = np.concatenate((top_ids[query_pos], candidate_ids[selected]))
        finite = np.isfinite(merged_dist) & (merged_ids >= 0)
        merged_dist = merged_dist[finite]
        merged_ids = merged_ids[finite]
        order = np.lexsort((merged_ids, merged_dist))[:k]
        top_dist[query_pos].fill(np.inf)
        top_ids[query_pos].fill(-1)
        top_dist[query_pos, : len(order)] = merged_dist[order]
        top_ids[query_pos, : len(order)] = merged_ids[order]


def exact_topk_batch(
    xb: np.memmap,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    chunk_rows: int,
    progress_chunks: int,
    filter_name: str,
) -> tuple[list[list[int]], list[list[float]], float]:
    queries = np.asarray(xb[query_ids], dtype=np.float32)
    query_t = np.ascontiguousarray(queries.T)
    query_norm = np.einsum("ij,ij->i", queries, queries)
    # Over-retain before the final direct-L2 pass so the matrix-product
    # candidate phase cannot decide a close boundary using cancellation noise.
    retained = max(k + 1, k * 4)
    top_dist = np.full((len(query_ids), retained), np.inf, dtype=np.float32)
    top_ids = np.full((len(query_ids), retained), -1, dtype=np.int64)
    started = time.perf_counter()

    for chunk_no, start in enumerate(range(0, len(candidate_ids), chunk_rows), start=1):
        ids = candidate_ids[start : start + chunk_rows]
        vectors = np.asarray(xb[ids], dtype=np.float32)
        vector_norm = np.einsum("ij,ij->i", vectors, vectors)
        distances = vector_norm[:, None] + query_norm[None, :] - 2.0 * (vectors @ query_t)
        np.maximum(distances, 0.0, out=distances)
        for query_pos, query_id in enumerate(query_ids):
            self_positions = np.flatnonzero(ids == query_id)
            if self_positions.size:
                distances[self_positions, query_pos] = np.inf
        update_topk(top_dist, top_ids, distances, ids, retained)
        if progress_chunks > 0 and chunk_no % progress_chunks == 0:
            elapsed = (time.perf_counter() - started) / 60.0
            done = min(start + chunk_rows, len(candidate_ids))
            print(
                f"filter={filter_name} exact rows={done}/{len(candidate_ids)} elapsed={elapsed:.1f} min",
                flush=True,
            )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids_out: list[list[int]] = []
    distances_out: list[list[float]] = []
    for query_pos, query_id in enumerate(query_ids):
        retained_ids = top_ids[query_pos][top_ids[query_pos] >= 0]
        retained_vectors = np.asarray(xb[retained_ids], dtype=np.float32)
        differences = retained_vectors - np.asarray(xb[int(query_id)], dtype=np.float32)
        direct_distances = np.einsum("ij,ij->i", differences, differences)
        order = np.lexsort((retained_ids, direct_distances))
        ids_out.append([int(value) for value in retained_ids[order]])
        distances_out.append([float(value) for value in direct_distances[order]])
    return ids_out, distances_out, elapsed_ms


def fetch_candidate_ids(cur: psycopg.Cursor, table: str, predicate: str) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} WHERE {predicate}")
    ids = np.fromiter((int(row[0]) for row in cur), dtype=np.int64)
    return ids, (time.perf_counter() - started) * 1000.0


def distance_tolerance(distance_sq: float) -> float:
    return max(1e-9, abs(distance_sq) * 1e-6)


def parse_vector_text(value: str) -> np.ndarray:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("unexpected PostgreSQL vector text")
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def verify_query_vector_mapping(
    cur: psycopg.Cursor,
    table: str,
    xb: np.memmap,
    query_ids: np.ndarray,
) -> dict[str, Any]:
    absolute_tolerance = 1e-7
    relative_tolerance = 1e-6
    wanted = [int(value) for value in query_ids]
    cur.execute(
        f"SELECT id, embedding::text FROM {table} WHERE id = ANY(%s::bigint[])",
        (wanted,),
    )
    observed = {int(row[0]): parse_vector_text(str(row[1])) for row in cur.fetchall()}
    missing = sorted(set(wanted) - set(observed))
    if missing:
        raise SystemExit(f"PostgreSQL/fbin mapping check is missing query IDs: {missing[:10]}")
    max_abs_error = 0.0
    for query_id in wanted:
        database_vector = observed[query_id]
        file_vector = np.asarray(xb[query_id], dtype=np.float32)
        if database_vector.shape != file_vector.shape:
            raise SystemExit(
                f"PostgreSQL/fbin dimension mismatch at id={query_id}: "
                f"database={database_vector.shape} fbin={file_vector.shape}"
            )
        error = float(np.max(np.abs(database_vector - file_vector)))
        max_abs_error = max(max_abs_error, error)
        if not np.allclose(
            database_vector,
            file_vector,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise SystemExit(
                f"PostgreSQL/fbin vector mismatch at id={query_id}: max_abs_error={error}"
            )
    return {
        "checked_query_rows": len(wanted),
        "comparison": "float32_allclose",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_abs_error": max_abs_error,
    }


def truth_boundary(distances: list[float], k: int) -> tuple[float, float, int, bool]:
    if len(distances) < k:
        raise ValueError(f"exact result has {len(distances)} distances, expected at least {k}")
    kth = float(distances[k - 1])
    tolerance = distance_tolerance(kth)
    strict = sum(value < kth - tolerance for value in distances[:k])
    boundary_tied = len(distances) > k and distances[k] <= kth + tolerance
    return kth, tolerance, strict, boundary_tied


def truth_row(
    query_no: int,
    query_id: int,
    filter_spec: dict[str, str],
    actual_selectivity: float,
    candidate_count: int,
    exact_ids: list[int],
    exact_distances: list[float],
    amortized_ms: float,
    seed: int,
    split: str,
    self_in_filter: bool,
    k: int,
) -> dict[str, Any]:
    top_ids = exact_ids[:k]
    top_distances = exact_distances[:k]
    ids = ",".join(str(value) for value in top_ids)
    kth, tolerance, strict, boundary_tied = truth_boundary(exact_distances, k)
    return {
        "query_no": query_no,
        "query_id": query_id,
        "filter_name": filter_spec["filter_name"],
        "target_rate": filter_spec["target_rate"],
        "predicate": filter_spec["predicate"],
        "actual_selectivity": actual_selectivity,
        "method": "pre_filter_exact",
        "k": k,
        "post_overfetch": "",
        "post_ef_search": "",
        "in_ef_search": "",
        "latency_ms": amortized_ms,
        "recall_at_10_exact_filtered": 1.0,
        "returned": len(top_ids),
        "candidates": candidate_count,
        "filtered_rows": candidate_count,
        "search_candidate_rows": candidate_count - int(self_in_filter),
        "result_ids": ids,
        "exact_filtered_topk_ids": ids,
        "exact_filtered_topk_distances_sq": ",".join(f"{value:.9g}" for value in top_distances),
        "kth_distance_sq": f"{kth:.9g}",
        "tie_tolerance": f"{tolerance:.9g}",
        "strict_closer_count": strict,
        "boundary_tied": boundary_tied,
        "self_excluded": True,
        "query_split": split,
        "query_seed": seed,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["filter_name"]), str(row.get("query_split", "calibration"))), []).append(row)
    summary: list[dict[str, Any]] = []
    for (filter_name, split), items in grouped.items():
        summary.append(
            {
                "filter_name": filter_name,
                "query_split": split,
                "queries": len(items),
                "target_rate": items[0]["target_rate"],
                "predicate": items[0]["predicate"],
                "actual_selectivity": statistics.median(float(row["actual_selectivity"]) for row in items),
                "candidate_count": int(float(items[0]["candidates"])),
                "boundary_tied_queries": sum(str(row["boundary_tied"]).lower() == "true" for row in items),
                "zero_kth_distance_queries": sum(float(row["kth_distance_sq"]) == 0.0 for row in items),
                "exact_latency_amortized_mean_ms": statistics.fmean(float(row["latency_ms"]) for row in items),
            }
        )
    write_csv(path.with_name(path.stem + "_summary.csv"), summary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def validate_resumed_rows(
    rows: list[dict[str, str]], query_ids: np.ndarray, filters: list[dict[str, str]]
) -> set[str]:
    expected_queries = {position: int(query_id) for position, query_id in enumerate(query_ids)}
    expected_filters = {row["filter_name"] for row in filters}
    completed: set[str] = set()
    for filter_name in expected_filters:
        items = [row for row in rows if row.get("filter_name") == filter_name]
        if not items:
            continue
        if len(items) != len(query_ids):
            raise SystemExit(f"resume filter={filter_name} is partial: rows={len(items)}")
        for row in items:
            query_no = int(row["query_no"])
            if expected_queries.get(query_no) != int(row["query_id"]):
                raise SystemExit(f"resume query mapping mismatch for filter={filter_name} q={query_no}")
            required = {"filtered_rows", "kth_distance_sq", "tie_tolerance", "self_excluded"}
            if not required.issubset(row) or str(row["self_excluded"]).lower() != "true":
                raise SystemExit("resume artifact uses the retired non-tie-aware truth schema")
        completed.add(filter_name)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate disjoint q100/q100 exact filtered L2 truth for Amazon-10M.")
    parser.add_argument("--fbin", type=Path, default=DEFAULT_FBIN)
    parser.add_argument("--table", default="amazon_grocery_reviews_10m_pgvector")
    parser.add_argument("--filters-csv", type=Path, default=DEFAULT_FILTERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--filter-names", nargs="*", default=[])
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--calibration-queries", type=int, default=100)
    parser.add_argument("--calibration-seed", type=int, default=57)
    parser.add_argument("--final-queries", type=int, default=100)
    parser.add_argument("--final-seed", type=int, default=58)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--progress-chunks", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    xb, vector_rows, _ = read_fbin_memmap(args.fbin, args.rows)
    filters = load_filters(args.filters_csv, set(args.filter_names))
    calibration_query_ids = sample_disjoint_query_ids(
        vector_rows, set(), args.calibration_queries, args.calibration_seed
    )
    final_query_ids = sample_disjoint_query_ids(
        vector_rows,
        set(int(value) for value in calibration_query_ids),
        args.final_queries,
        args.final_seed,
    )
    query_ids = np.concatenate((calibration_query_ids, final_query_ids))
    split_by_query_no = {
        query_no: ("calibration" if query_no < len(calibration_query_ids) else "final")
        for query_no in range(len(query_ids))
    }
    seed_by_query_no = {
        query_no: (args.calibration_seed if query_no < len(calibration_query_ids) else args.final_seed)
        for query_no in range(len(query_ids))
    }

    selected_names = {row["filter_name"] for row in filters}
    rows_out: list[dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and args.out.exists():
        resumed = read_csv(args.out)
        rows_out = [row for row in resumed if row["filter_name"] in selected_names]
        completed = validate_resumed_rows(rows_out, query_ids, filters)

    print(
        json.dumps(
            {
                "vectors": vector_rows,
                "filters": len(filters),
                "calibration_queries": len(calibration_query_ids),
                "calibration_seed": args.calibration_seed,
                "final_queries": len(final_query_ids),
                "final_seed": args.final_seed,
                "disjoint": not bool(set(final_query_ids) & set(calibration_query_ids)),
                "self_excluded": True,
                "tie_aware": True,
                "completed_filters": sorted(completed),
            },
            indent=2,
        ),
        flush=True,
    )

    database_source: dict[str, Any] = {}
    with psycopg.connect(pg_config_from_env().conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT count(*), min(id), max(id), %s::regclass::oid::bigint, "
            f"pg_relation_filenode(%s::regclass)::bigint FROM {args.table}",
            (args.table, args.table),
        )
        table_rows, min_id, max_id, table_oid, table_relfilenode = (
            int(value) for value in cur.fetchone()
        )
        if table_rows != vector_rows:
            raise SystemExit(f"table/vector row mismatch: table={table_rows}, vectors={vector_rows}")
        if (min_id, max_id) != (0, vector_rows - 1):
            raise SystemExit(
                f"table/vector ID-space mismatch: table=({min_id}, {max_id}) "
                f"fbin=(0, {vector_rows - 1})"
            )
        database_source = {
            "table": args.table,
            "rows": table_rows,
            "min_id": min_id,
            "max_id": max_id,
            "table_oid": table_oid,
            "table_relfilenode": table_relfilenode,
            "query_vector_mapping": verify_query_vector_mapping(cur, args.table, xb, query_ids),
        }

        for position, filter_spec in enumerate(filters, start=1):
            filter_name = filter_spec["filter_name"]
            if filter_name in completed:
                print(f"filter={filter_name} already complete; skipping", flush=True)
                continue
            candidate_ids, sql_ms = fetch_candidate_ids(cur, args.table, filter_spec["predicate"])
            if candidate_ids.size < args.k:
                raise SystemExit(f"filter={filter_name} only has {candidate_ids.size} candidates")
            if int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= vector_rows:
                raise SystemExit(f"filter={filter_name} has id outside vector row range")
            candidate_ids.sort()
            query_in_filter = np.isin(query_ids, candidate_ids, assume_unique=True)
            print(
                f"filter={filter_name} ({position}/{len(filters)}) candidates={len(candidate_ids)} "
                f"selectivity={len(candidate_ids) / table_rows:.6f} sql_ms={sql_ms:.1f}",
                flush=True,
            )
            exact_ids, exact_distances, exact_ms = exact_topk_batch(
                xb,
                query_ids,
                candidate_ids,
                args.k,
                args.chunk_rows,
                args.progress_chunks,
                filter_name,
            )
            amortized_ms = exact_ms / len(query_ids)
            actual_selectivity = len(candidate_ids) / table_rows
            rows_out.extend(
                truth_row(
                    query_pos,
                    int(query_id),
                    filter_spec,
                    actual_selectivity,
                    len(candidate_ids),
                    exact_ids[query_pos],
                    exact_distances[query_pos],
                    amortized_ms,
                    seed_by_query_no[query_pos],
                    split_by_query_no[query_pos],
                    bool(query_in_filter[query_pos]),
                    args.k,
                )
                for query_pos, query_id in enumerate(query_ids)
            )
            rows_out.sort(key=lambda row: (str(row["filter_name"]), int(row["query_no"])))
            write_csv(args.out, rows_out)
            write_summary(args.out, rows_out)
            print(f"filter={filter_name} exact_ms={exact_ms:.1f}; checkpointed {args.out}", flush=True)

    write_summary(args.out, rows_out)
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    manifest = {
        "artifact_valid": len(rows_out) == len(filters) * len(query_ids),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "method": "exact_filtered_l2_tie_aware",
        "self_excluded": True,
        "recall_contract": "returned SQL-valid IDs with squared L2 <= kth_distance_sq + tie_tolerance, capped at k",
        "rows": vector_rows,
        "k": args.k,
        "calibration": {"queries": len(calibration_query_ids), "seed": args.calibration_seed},
        "final": {"queries": len(final_query_ids), "seed": args.final_seed},
        "query_ids_disjoint": not bool(set(final_query_ids) & set(calibration_query_ids)),
        "filters": len(filters),
        "truth_rows": len(rows_out),
        "inputs": {
            "fbin": {"path": str(args.fbin.resolve()), "sha256": sha256_file(args.fbin)},
            "filters_csv": {"path": str(args.filters_csv.resolve()), "sha256": sha256_file(args.filters_csv)},
            "postgres": database_source,
        },
        "outputs": {
            "truth_csv": {"path": str(args.out.resolve()), "sha256": sha256_file(args.out)},
            "summary_csv": {
                "path": str(args.out.with_name(args.out.stem + "_summary.csv").resolve()),
                "sha256": sha256_file(args.out.with_name(args.out.stem + "_summary.csv")),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not manifest["artifact_valid"]:
        raise SystemExit("truth artifact is incomplete")
    print(f"wrote {args.out} rows={len(rows_out)}", flush=True)


if __name__ == "__main__":
    main()
