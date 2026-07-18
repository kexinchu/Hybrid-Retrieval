from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.hybrid_vector_db.scripts import build_faiss_hnsw_from_fbin as builder


def write_fbin(path: Path, vectors: np.ndarray) -> None:
    with path.open("wb") as target:
        target.write(struct.pack("<ii", *vectors.shape))
        target.write(np.asarray(vectors, dtype="<f4").tobytes())


def test_formal_builder_publishes_readable_index_and_provenance(tmp_path: Path) -> None:
    rng = np.random.default_rng(57)
    vectors = rng.normal(size=(200, 8)).astype(np.float32)
    fbin = tmp_path / "vectors.fbin"
    output = tmp_path / "vectors.index"
    write_fbin(fbin, vectors)
    args = builder.build_parser().parse_args(
        [
            "--fbin", str(fbin),
            "--out", str(output),
            "--rows", "200",
            "--m", "8",
            "--ef-construction", "40",
            "--batch-size", "50",
            "--progress-batches", "1",
            "--threads", "1",
            "--seed", "57",
        ]
    )

    manifest = builder.build(args)
    manifest_path = output.with_name(output.name + ".manifest.json")

    assert output.is_file()
    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest["artifact_valid"] is True
    assert manifest["index_contract"]["rows"] == 200
    assert manifest["index_contract"]["m"] == 8
    assert manifest["index_contract"]["ef_construction"] == 40
    assert manifest["configuration"]["determinism"] == "deterministic serial insertion"
    assert manifest["output_identity"]["sha256"] == builder.sha256_file(output)


def test_builder_rejects_truncated_fbin_and_existing_output(tmp_path: Path) -> None:
    truncated = tmp_path / "bad.fbin"
    truncated.write_bytes(struct.pack("<ii", 2, 4) + b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        builder.read_fbin_memmap(truncated)

    vectors = np.ones((10, 4), dtype=np.float32)
    fbin = tmp_path / "vectors.fbin"
    output = tmp_path / "vectors.index"
    write_fbin(fbin, vectors)
    output.write_bytes(b"existing")
    args = builder.build_parser().parse_args(
        ["--fbin", str(fbin), "--out", str(output), "--rows", "10"]
    )
    with pytest.raises(FileExistsError):
        builder.build(args)
