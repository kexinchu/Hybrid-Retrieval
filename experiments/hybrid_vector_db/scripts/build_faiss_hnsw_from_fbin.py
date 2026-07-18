from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def read_fbin_memmap(
    path: Path, limit: int | None = None
) -> tuple[np.ndarray, int, int, int]:
    with path.open("rb") as source:
        header = source.read(8)
    if len(header) != 8:
        raise ValueError("fbin header is truncated")
    total_rows, dimensions = struct.unpack("<ii", header)
    if total_rows <= 0 or dimensions <= 0:
        raise ValueError("fbin header has invalid dimensions")
    rows = min(total_rows, limit) if limit else total_rows
    expected_size = 8 + total_rows * dimensions * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"fbin size mismatch: expected={expected_size} actual={path.stat().st_size}"
        )
    mapped = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=8,
        shape=(total_rows, dimensions),
    )
    return mapped[:rows], rows, dimensions, total_rows


def index_contract(index: Any, faiss_module: Any) -> dict[str, Any]:
    storage = index.storage if hasattr(index, "storage") else None
    dimensions = int(index.d)
    metric = int(index.metric_type)
    return {
        "type": type(index).__name__,
        "rows": int(index.ntotal),
        "dimensions": dimensions,
        "metric_type": metric,
        "metric": (
            "l2" if metric == int(faiss_module.METRIC_L2) else
            "ip" if metric == int(faiss_module.METRIC_INNER_PRODUCT) else
            str(metric)
        ),
        "m": int(index.hnsw.nb_neighbors(0) // 2),
        "layer0_neighbors": int(index.hnsw.nb_neighbors(0)),
        "ef_construction": int(index.hnsw.efConstruction),
        "storage_dimensions": int(storage.d) if storage is not None else dimensions,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    output = args.out.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output.with_name(output.name + ".manifest.json")
    )
    if output == manifest_path:
        raise ValueError("index and manifest paths must differ")
    for path in (output, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_index.unlink(missing_ok=True)

    source = args.fbin.resolve(strict=True)
    vectors, rows, dimensions, source_rows = read_fbin_memmap(source, args.rows)
    if rows != args.rows:
        raise ValueError(f"requested rows={args.rows}, fbin only contains {source_rows}")
    faiss.omp_set_num_threads(args.threads)
    metric = (
        faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2
    )
    index = faiss.IndexHNSWFlat(dimensions, args.m, metric)
    index.hnsw.efConstruction = args.ef_construction
    index.hnsw.rng = faiss.RandomGenerator(args.seed)

    manifest: dict[str, Any] = {
        "artifact": "faiss_hnsw_index_build",
        "artifact_valid": False,
        "status": "building",
        "started_at": utc_now(),
        "inputs": {
            "fbin": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "header_rows": source_rows,
                "dimensions": dimensions,
                "dtype": "little-endian float32",
            },
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "configuration": {
            "rows": rows,
            "dimensions": dimensions,
            "m": args.m,
            "ef_construction": args.ef_construction,
            "batch_size": args.batch_size,
            "metric": args.metric,
            "threads": args.threads,
            "seed": args.seed,
            "determinism": (
                "deterministic serial insertion"
                if args.threads == 1
                else "seeded levels with parallel insertion; output SHA is authoritative"
            ),
        },
        "software": {
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "faiss_compile_options": str(faiss.get_compile_options()),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "progress": [],
        "outputs": {
            "index": str(output),
            "manifest": str(manifest_path),
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        f"building HNSW rows={rows} dim={dimensions} M={args.m} "
        f"efC={args.ef_construction} metric={args.metric} threads={args.threads} "
        f"seed={args.seed}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        for start in range(0, rows, args.batch_size):
            end = min(rows, start + args.batch_size)
            index.add(np.ascontiguousarray(vectors[start:end], dtype=np.float32))
            if end % (args.batch_size * args.progress_batches) == 0 or end == rows:
                elapsed = time.perf_counter() - started
                progress = {
                    "rows": end,
                    "target_rows": rows,
                    "elapsed_seconds": elapsed,
                }
                manifest["progress"].append(progress)
                atomic_json(manifest_path, manifest)
                print(
                    f"added {end}/{rows} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        faiss.write_index(index, str(temporary_index))
        with temporary_index.open("rb") as target:
            os.fsync(target.fileno())
        readback = faiss.read_index(str(temporary_index))
        contract = index_contract(readback, faiss)
        expected_contract = {
            "rows": rows,
            "dimensions": dimensions,
            "metric": args.metric,
            "m": args.m,
            "ef_construction": args.ef_construction,
        }
        for field, expected in expected_contract.items():
            if contract[field] != expected:
                raise RuntimeError(
                    f"Faiss index read-back mismatch for {field}: "
                    f"expected={expected!r} actual={contract[field]!r}"
                )
        output_sha256 = sha256_file(temporary_index)
        output_size = temporary_index.stat().st_size
        os.replace(temporary_index, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        manifest.update(
            {
                "artifact_valid": True,
                "status": "complete",
                "finished_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "index_contract": contract,
                "output_identity": {
                    "path": str(output),
                    "size_bytes": output_size,
                    "sha256": output_sha256,
                },
            }
        )
        atomic_json(manifest_path, manifest)
        print(
            f"wrote {output} ntotal={readback.ntotal} "
            f"total_s={manifest['elapsed_seconds']:.1f} sha256={output_sha256}",
            flush=True,
        )
        return manifest
    except BaseException as exc:
        temporary_index.unlink(missing_ok=True)
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            manifest.update(
                {
                    "status": "failed",
                    "artifact_valid": False,
                    "finished_at": utc_now(),
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                }
            )
            atomic_json(manifest_path, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a provenance-bound Faiss HNSW index from fbin"
    )
    parser.add_argument("--fbin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rows", type=positive_int, default=10_000_000)
    parser.add_argument("--m", type=positive_int, default=16)
    parser.add_argument("--ef-construction", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=100_000)
    parser.add_argument("--progress-batches", type=positive_int, default=10)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--metric", default="l2", choices=["l2", "ip"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
