from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from common_pg import pg_config_from_env
from prepare_yfcc_pgvector import DIM, spmat_fields, xbin_mmap


DATA_DIR = Path(os.environ.get("YFCC10M_DATA_DIR", Path(os.environ.get("OOD_ANNS_DATA", "data/ood_anns")) / "YFCC10M"))
TARGET_BANDS = [50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tag_text(labels: tuple[int, ...]) -> str:
    return " ".join(str(int(x)) for x in labels)


def predicate(labels: tuple[int, ...]) -> str:
    return "tags && ARRAY[" + ",".join(str(int(x)) for x in labels) + "]::int[]"


def load_query_ids(query_table: str, queries_per_filter: int) -> list[int]:
    cfg = pg_config_from_env()
    with psycopg.connect(cfg.conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT qid
            FROM {query_table}
            ORDER BY md5(qid::text)
            LIMIT %s
            """,
            (int(queries_per_filter),),
        )
        return [int(row[0]) for row in cur.fetchall()]


def choose_filters(
    nrow: int,
    ncol: int,
    indptr: np.memmap,
    indices: np.memmap,
    targets: list[float],
    pair_top_tags: int,
) -> list[dict[str, Any]]:
    freq = np.bincount(np.asarray(indices, dtype=np.int32), minlength=ncol)
    order = np.argsort(freq)[::-1]
    top_labels = [int(x) for x in order[:pair_top_tags]]

    started = time.perf_counter()
    print(f"building row lists for top {len(top_labels)} YFCC tags", flush=True)
    hit_mask = np.isin(indices, np.asarray(top_labels, dtype=np.int32), assume_unique=False)
    positions = np.flatnonzero(hit_mask)
    hit_labels = np.asarray(indices[positions], dtype=np.int32)
    rows = np.searchsorted(indptr, positions, side="right").astype(np.int64) - 1
    sort_order = np.argsort(hit_labels, kind="stable")
    hit_labels = hit_labels[sort_order]
    rows = rows[sort_order].astype(np.int32, copy=False)

    row_lists: dict[int, np.ndarray] = {}
    starts = np.r_[0, np.flatnonzero(np.diff(hit_labels)) + 1, len(hit_labels)]
    for start, end in zip(starts[:-1], starts[1:]):
        label = int(hit_labels[start])
        row_lists[label] = np.unique(rows[start:end])
    print(f"built {len(row_lists)} row lists in {time.perf_counter() - started:.1f}s", flush=True)

    candidates: list[tuple[int, tuple[int, ...]]] = []
    for label in range(ncol):
        if freq[label] > 0:
            candidates.append((int(freq[label]), (int(label),)))

    print("enumerating YFCC top-tag OR pairs", flush=True)
    for i, a in enumerate(top_labels):
        a_rows = row_lists[a]
        for b in top_labels[i + 1 :]:
            b_rows = row_lists[b]
            overlap = np.intersect1d(a_rows, b_rows, assume_unique=True).size
            candidates.append((int(len(a_rows) + len(b_rows) - overlap), (int(a), int(b))))
        if (i + 1) % 20 == 0:
            print(f"  pair enumeration {i + 1}/{len(top_labels)}", flush=True)

    selected = []
    used: set[tuple[int, ...]] = set()
    for target in targets:
        ordered = sorted(candidates, key=lambda item, t=target: (abs(100.0 * item[0] / nrow - t), len(item[1]), item[1]))
        count, labels = next((count, labels) for count, labels in ordered if labels not in used)
        used.add(labels)
        actual = 100.0 * count / nrow
        name = "tagor_" + "_".join(str(x) for x in labels)
        selected.append(
            {
                "target_band_pct": float(target),
                "filter_pct": actual,
                "filter_rows": int(count),
                "filter_name": name,
                "tags": tag_text(labels),
                "predicate": predicate(labels),
                "labels_tuple": labels,
            }
        )
        print(f"target={target:g}% filter={name} actual={actual:.4f}% rows={count}", flush=True)
    return selected


def label_mask_for_chunk(
    indptr: np.memmap,
    indices: np.memmap,
    global_start: int,
    global_end: int,
    labels: tuple[int, ...],
) -> np.ndarray:
    mask = np.zeros(global_end - global_start, dtype=bool)
    lo = int(indptr[global_start])
    hi = int(indptr[global_end])
    if hi <= lo:
        return mask
    segment = indices[lo:hi]
    hits = np.nonzero(np.isin(segment, np.asarray(labels, dtype=np.int32), assume_unique=False))[0]
    if hits.size == 0:
        return mask
    absolute = hits.astype(np.int64, copy=False) + lo
    local_rows = np.searchsorted(indptr[global_start : global_end + 1], absolute, side="right") - 1
    mask[np.unique(local_rows)] = True
    return mask


def update_topk(
    top_dist: np.ndarray,
    top_ids: np.ndarray,
    spec_pos: int,
    candidate_dist: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> None:
    if candidate_dist.size == 0:
        return
    take = min(k, candidate_dist.size)
    local = np.argpartition(candidate_dist, take - 1)[:take]
    merged_dist = np.concatenate([top_dist[spec_pos], candidate_dist[local]])
    merged_ids = np.concatenate([top_ids[spec_pos], candidate_ids[local]])
    keep = np.argpartition(merged_dist, k - 1)[:k]
    keep = keep[np.argsort(merged_dist[keep], kind="stable")]
    top_dist[spec_pos] = merged_dist[keep]
    top_ids[spec_pos] = merged_ids[keep]


def generate_truth(
    filter_rows: list[dict[str, Any]],
    qids: list[int],
    indptr: np.memmap,
    indices: np.memmap,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    xb = xbin_mmap(args.base_u8bin, "uint8")
    xq = xbin_mmap(args.query_u8bin, "uint8")
    if xb.shape[1] != DIM or xq.shape[1] != DIM:
        raise SystemExit(f"bad YFCC dimensions: base={xb.shape} query={xq.shape}")

    selected_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for batch_start in range(0, len(qids), args.query_batch):
        batch_qids = qids[batch_start : batch_start + args.query_batch]
        specs: list[tuple[int, dict[str, Any]]] = [(qid, row) for qid in batch_qids for row in filter_rows]
        qid_pos = {qid: pos for pos, qid in enumerate(batch_qids)}
        q = np.asarray(xq[batch_qids], dtype=np.float32)
        q_t = np.ascontiguousarray(q.T)
        q_norm = np.einsum("ij,ij->i", q, q)
        top_dist = np.full((len(specs), args.k), np.inf, dtype=np.float32)
        top_ids = np.full((len(specs), args.k), -1, dtype=np.int32)

        for chunk_start in range(0, xb.shape[0], args.chunk_rows):
            chunk_end = min(chunk_start + args.chunk_rows, xb.shape[0])
            xb_chunk = np.asarray(xb[chunk_start:chunk_end], dtype=np.float32)
            dots = xb_chunk @ q_t
            xb_norm = np.einsum("ij,ij->i", xb_chunk, xb_chunk)
            dist = xb_norm[:, None] + q_norm[None, :] - 2.0 * dots
            candidate_ids = np.arange(chunk_start, chunk_end, dtype=np.int32)
            mask_cache: dict[tuple[int, ...], np.ndarray] = {}

            for spec_pos, (qid, row) in enumerate(specs):
                labels = row["labels_tuple"]
                mask = mask_cache.get(labels)
                if mask is None:
                    mask = label_mask_for_chunk(indptr, indices, chunk_start, chunk_end, labels)
                    mask_cache[labels] = mask
                if np.any(mask):
                    update_topk(top_dist, top_ids, spec_pos, dist[mask, qid_pos[qid]], candidate_ids[mask], args.k)

            if args.progress_chunks and ((chunk_start // args.chunk_rows) + 1) % args.progress_chunks == 0:
                elapsed = (time.perf_counter() - started) / 60.0
                print(
                    f"truth qids {batch_start + 1}-{batch_start + len(batch_qids)}/{len(qids)} "
                    f"at row {chunk_end}/{xb.shape[0]} elapsed={elapsed:.1f} min",
                    flush=True,
                )

        for spec_pos, (qid, row) in enumerate(specs):
            ids = [int(x) for x in top_ids[spec_pos] if int(x) >= 0]
            base = {
                "target_band_pct": row["target_band_pct"],
                "filter_pct": row["filter_pct"],
                "filter_rows": row["filter_rows"],
                "filter_name": row["filter_name"],
                "qid": int(qid),
                "tags": row["tags"],
                "predicate": row["predicate"],
            }
            selected_rows.append({**base, "gt": " ".join(str(x) for x in ids)})
            truth_rows.append(
                {
                    "truth_key": f"overlap|{float(row['target_band_pct'])}|{int(qid)}|{row['filter_name']}",
                    **base,
                    "gt": " ".join(str(x) for x in ids),
                }
            )
        write_csv(args.selected_out, selected_rows)
        write_csv(args.truth_out, truth_rows)
        elapsed = (time.perf_counter() - started) / 60.0
        print(f"finished exact qids {batch_start + len(batch_qids)}/{len(qids)} elapsed={elapsed:.1f} min", flush=True)

    return selected_rows, truth_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YFCC10M tags-overlap workload and exact SQL-valid top-k truth.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--query-table", default="yfcc10m_queries")
    parser.add_argument("--selected-out", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_overlap_selectivity14_selected_q100_20260716.csv"))
    parser.add_argument("--truth-out", type=Path, default=Path("results/hybrid_vector_db/yfcc10m_overlap_selectivity14_truth_q100_20260716.csv"))
    parser.add_argument("--target-bands", type=float, nargs="+", default=TARGET_BANDS)
    parser.add_argument("--queries-per-filter", type=int, default=100)
    parser.add_argument("--pair-top-tags", type=int, default=80)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=100000)
    parser.add_argument("--progress-chunks", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    args.base_u8bin = args.data_dir / "base.10M.u8bin"
    args.query_u8bin = args.data_dir / "query.public.100K.u8bin"
    base_metadata = args.data_dir / "base.metadata.10M.spmat"

    nrow, ncol, _, indptr, indices, _ = spmat_fields(base_metadata)
    filters = choose_filters(nrow, ncol, indptr, indices, [float(x) for x in args.target_bands], int(args.pair_top_tags))
    qids = load_query_ids(args.query_table, int(args.queries_per_filter))
    print(f"using {len(qids)} qids for each of {len(filters)} filters", flush=True)

    if args.select_only:
        rows = []
        for row in filters:
            for qid in qids:
                base = {key: value for key, value in row.items() if key != "labels_tuple"}
                rows.append({**base, "qid": int(qid), "gt": ""})
        write_csv(args.selected_out, rows)
        print(f"wrote selected workload {args.selected_out}", flush=True)
        return

    selected_rows, truth_rows = generate_truth(filters, qids, indptr, indices, args)
    print(f"wrote selected workload {args.selected_out} rows={len(selected_rows)}", flush=True)
    print(f"wrote exact truth {args.truth_out} rows={len(truth_rows)}", flush=True)


if __name__ == "__main__":
    main()
