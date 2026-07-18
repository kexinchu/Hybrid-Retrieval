from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from prepare_laion25m_pgvector import DIM, PROCESSED, QUERY_ROWS, base_plan, img_path, npy


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_labels(text: Any) -> tuple[int, ...]:
    value = str(text or "").strip()
    if not value:
        return ()
    value = value.strip("{}[]")
    sep = "," if "," in value else None
    return tuple(sorted({int(x) for x in value.split(sep) if str(x).strip()}))


def parse_optional_int(text: Any) -> int | None:
    value = str(text or "").strip()
    if not value:
        return None
    return int(float(value))


def truth_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["workload"]),
            str(float(row["target_band_pct"])),
            str(int(row["qid"])),
            str(row["filter_name"]),
        ]
    )


def read_fbin(path: Path) -> np.memmap:
    with path.open("rb") as f:
        header = np.fromfile(f, dtype="<i4", count=2)
    if len(header) != 2:
        raise SystemExit(f"bad fbin header: {path}")
    rows, dim = int(header[0]), int(header[1])
    if dim != DIM:
        raise SystemExit(f"expected dim {DIM}, got {dim} in {path}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def load_selected(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    workloads = set(args.workloads or [])
    targets = {float(x) for x in args.target_bands} if args.target_bands else set()
    filtered: list[dict[str, Any]] = []
    per_group: dict[tuple[str, float], int] = {}
    for row in rows:
        row["qid"] = int(row["qid"])
        row["target_band_pct"] = float(row["target_band_pct"])
        row["actual_pct"] = float(row["actual_pct"])
        row["filter_rows"] = int(row["filter_rows"])
        row["labels_tuple"] = parse_labels(row.get("labels"))
        row["range_l_int"] = parse_optional_int(row.get("range_l"))
        row["range_r_int"] = parse_optional_int(row.get("range_r"))
        if workloads and row["workload"] not in workloads:
            continue
        if targets and float(row["target_band_pct"]) not in targets:
            continue
        group = (str(row["workload"]), float(row["target_band_pct"]))
        if args.limit_per_group > 0 and per_group.get(group, 0) >= args.limit_per_group:
            continue
        per_group[group] = per_group.get(group, 0) + 1
        filtered.append(row)
    if not filtered:
        raise SystemExit("no selected rows after filtering")
    return filtered


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
    order = np.argsort(merged_dist[keep], kind="stable")
    keep = keep[order]
    top_dist[spec_pos] = merged_dist[keep]
    top_ids[spec_pos] = merged_ids[keep]


def label_mask_for_chunk(
    offsets: np.memmap,
    labels_flat: np.memmap,
    global_start: int,
    global_end: int,
    wanted: tuple[int, ...],
) -> np.ndarray:
    mask = np.zeros(global_end - global_start, dtype=bool)
    if not wanted:
        return mask
    lo = int(offsets[global_start])
    hi = int(offsets[global_end])
    if hi <= lo:
        return mask
    segment = labels_flat[lo:hi]
    hits = np.nonzero(np.isin(segment, np.asarray(wanted, dtype=np.int32), assume_unique=False))[0]
    if hits.size == 0:
        return mask
    absolute_label_pos = hits.astype(np.int64, copy=False) + lo
    local_rows = np.searchsorted(offsets[global_start : global_end + 1], absolute_label_pos, side="right") - 1
    mask[np.unique(local_rows)] = True
    return mask


def row_mask(
    row: dict[str, Any],
    width_chunk: np.ndarray,
    offsets: np.memmap,
    labels_flat: np.memmap,
    global_start: int,
    global_end: int,
    label_cache: dict[tuple[int, ...], np.ndarray],
    range_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    workload = str(row["workload"])
    labels = row["labels_tuple"]
    lo = row["range_l_int"]
    hi = row["range_r_int"]

    range_mask: np.ndarray | None = None
    if lo is not None and hi is not None:
        key = (int(lo), int(hi))
        range_mask = range_cache.get(key)
        if range_mask is None:
            range_mask = (width_chunk >= int(lo)) & (width_chunk < int(hi))
            range_cache[key] = range_mask

    label_mask: np.ndarray | None = None
    if labels:
        label_mask = label_cache.get(labels)
        if label_mask is None:
            label_mask = label_mask_for_chunk(offsets, labels_flat, global_start, global_end, labels)
            label_cache[labels] = label_mask

    if workload == "range":
        if range_mask is None:
            raise ValueError("range workload without range bounds")
        return range_mask
    if workload in {"label", "label_or"}:
        if label_mask is None:
            raise ValueError(f"{workload} workload without labels")
        return label_mask
    if workload == "hybrid":
        if range_mask is None or label_mask is None:
            raise ValueError("hybrid workload without labels/range")
        return range_mask | label_mask
    raise ValueError(workload)


def generate_truth(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    query_fbin = read_fbin(args.query_fbin)
    width = np.memmap(PROCESSED / "base_width.int32", dtype=np.int32, mode="r", shape=(25_000_000,))
    offsets = np.memmap(PROCESSED / "base_label_offsets.int64", dtype=np.int64, mode="r", shape=(25_000_001,))
    labels_flat = np.memmap(PROCESSED / "base_labels.int32", dtype=np.int32, mode="r", shape=(int(offsets[-1]),))

    qids = sorted({int(row["qid"]) for row in rows})
    rows_by_qid: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_qid.setdefault(int(row["qid"]), []).append(row)

    out: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_start in range(0, len(qids), args.query_batch):
        batch_qids = qids[batch_start : batch_start + args.query_batch]
        specs = [row for qid in batch_qids for row in rows_by_qid[qid]]
        qid_pos = {qid: pos for pos, qid in enumerate(batch_qids)}
        q = np.asarray(query_fbin[batch_qids], dtype=np.float32)
        q_t = np.ascontiguousarray(q.T)
        q_norm = np.einsum("ij,ij->i", q, q)
        top_dist = np.full((len(specs), args.k), np.inf, dtype=np.float32)
        top_ids = np.full((len(specs), args.k), -1, dtype=np.int32)

        row_base = 0
        chunk_no = 0
        for shard, take_rows in base_plan():
            xb_shard = npy(img_path(shard))
            for local_start in range(0, take_rows, args.chunk_rows):
                local_end = min(local_start + args.chunk_rows, take_rows)
                global_start = row_base + local_start
                global_end = row_base + local_end
                xb = np.asarray(xb_shard[local_start:local_end], dtype=np.float32)
                dots = xb @ q_t
                xb_norm = np.einsum("ij,ij->i", xb, xb)
                dist = xb_norm[:, None] + q_norm[None, :] - 2.0 * dots
                width_chunk = np.asarray(width[global_start:global_end], dtype=np.int32)
                candidate_ids = np.arange(global_start, global_end, dtype=np.int32)
                label_cache: dict[tuple[int, ...], np.ndarray] = {}
                range_cache: dict[tuple[int, int], np.ndarray] = {}
                for spec_pos, row in enumerate(specs):
                    mask = row_mask(row, width_chunk, offsets, labels_flat, global_start, global_end, label_cache, range_cache)
                    if np.any(mask):
                        update_topk(top_dist, top_ids, spec_pos, dist[mask, qid_pos[int(row["qid"])]], candidate_ids[mask], args.k)
                chunk_no += 1
                if args.progress_chunks and chunk_no % args.progress_chunks == 0:
                    elapsed = (time.perf_counter() - started) / 60.0
                    print(
                        f"truth batch {batch_start + 1}-{batch_start + len(batch_qids)}/{len(qids)} "
                        f"at row {global_end}/25000000 elapsed={elapsed:.1f} min",
                        flush=True,
                    )
            row_base += take_rows

        for spec_pos, row in enumerate(specs):
            ids = [int(x) for x in top_ids[spec_pos] if int(x) >= 0]
            out.append(
                {
                    "truth_key": truth_key(row),
                    "workload": row["workload"],
                    "target_band_pct": row["target_band_pct"],
                    "actual_pct": row["actual_pct"],
                    "filter_rows": row["filter_rows"],
                    "qid": row["qid"],
                    "filter_name": row["filter_name"],
                    "predicate": row["predicate"],
                    "gt": " ".join(str(x) for x in ids),
                }
            )
        write_csv(args.truth_out, out)
        elapsed = (time.perf_counter() - started) / 60.0
        print(f"finished truth qids {batch_start + len(batch_qids)}/{len(qids)} elapsed={elapsed:.1f} min", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact filtered L2 top-k truth for LAION25M selected predicates.")
    parser.add_argument("--selected-queries-in", type=Path, required=True)
    parser.add_argument("--truth-out", type=Path, default=Path("results/hybrid_vector_db/laion25m_truth_20260714.csv"))
    parser.add_argument("--query-fbin", type=Path, default=PROCESSED / "query.text_emb_10k.fbin")
    parser.add_argument("--workloads", nargs="*", default=[])
    parser.add_argument("--target-bands", type=float, nargs="*", default=[])
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--query-batch", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=100000)
    parser.add_argument("--progress-chunks", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    rows = load_selected(args.selected_queries_in, args)
    print(
        json.dumps(
            {
                "selected_rows": len(rows),
                "unique_qids": len({int(row["qid"]) for row in rows}),
                "workloads": sorted({str(row["workload"]) for row in rows}),
                "targets": sorted({float(row["target_band_pct"]) for row in rows}),
            },
            indent=2,
        ),
        flush=True,
    )
    generate_truth(rows, args)
    print(f"wrote truth {args.truth_out}", flush=True)


if __name__ == "__main__":
    main()
