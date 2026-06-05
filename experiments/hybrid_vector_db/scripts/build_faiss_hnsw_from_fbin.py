from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np


def read_fbin_memmap(path: Path, limit: int | None = None) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    rows = min(n, limit) if limit else n
    arr = np.memmap(path, dtype="float32", mode="r", offset=8, shape=(n, d))
    return arr[:rows], rows, d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--metric", default="l2", choices=["l2", "ip"])
    args = parser.parse_args()

    import faiss

    args.out.parent.mkdir(parents=True, exist_ok=True)
    xb, rows, dim = read_fbin_memmap(args.fbin, args.rows)
    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(dim, args.m, metric)
    index.hnsw.efConstruction = args.ef_construction
    print(
        f"building HNSW rows={rows} dim={dim} M={args.m} efC={args.ef_construction} metric={args.metric}",
        flush=True,
    )
    t0 = time.perf_counter()
    for start in range(0, rows, args.batch_size):
        end = min(rows, start + args.batch_size)
        index.add(np.ascontiguousarray(xb[start:end]))
        if end % (args.batch_size * 10) == 0 or end == rows:
            elapsed = time.perf_counter() - t0
            print(f"added {end}/{rows} elapsed={elapsed:.1f}s", flush=True)
    faiss.write_index(index, str(args.out))
    print(f"wrote {args.out} ntotal={index.ntotal} total_s={time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
