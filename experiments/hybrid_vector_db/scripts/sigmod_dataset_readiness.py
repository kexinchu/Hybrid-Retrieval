from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import psycopg

from common_pg import pg_config_from_env


DATA_ROOT = Path(os.environ.get("OOD_ANNS_DATA", "data/ood_anns"))

DATASETS = [
    {
        "dataset": "amazon_reviews_2023_grocery",
        "target": "10M",
        "role": "main_sql_native",
        "pg_tables": ["amazon_grocery_reviews_10m_pgvector"],
        "files": [Path("data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin")],
        "notes": "Loaded PostgreSQL dataset with real product/review metadata.",
    },
    {
        "dataset": "yfcc",
        "target": "10M",
        "role": "main_cross_paper",
        "pg_tables": ["yfcc10m_pgvector", "yfcc10m_queries"],
        "files": [
            DATA_ROOT / "YFCC10M" / "base.10M.u8bin",
            DATA_ROOT / "YFCC10M" / "query.public.100K.u8bin",
            DATA_ROOT / "YFCC10M" / "base.metadata.10M.spmat",
            DATA_ROOT / "YFCC10M" / "query.metadata.public.100K.spmat",
            DATA_ROOT / "YFCC10M" / "GT.public.ibin",
        ],
        "notes": "Official BigANN YFCC filtered-track files; PostgreSQL full load is active.",
    },
    {
        "dataset": "tripclick",
        "target": "~1M",
        "role": "main_real_hybrid_text",
        "pg_tables": [],
        "files": [DATA_ROOT / "TripClick", DATA_ROOT / "tripclick"],
        "notes": "ACORN-style real hybrid query candidate; requires ingestion and embeddings.",
    },
    {
        "dataset": "laion",
        "target": "25M_preferred_10M_fallback",
        "role": "main_large_image_text",
        "pg_tables": [],
        "files": [
            DATA_ROOT / "LAION25M",
            DATA_ROOT / "LAION10M" / "base.10M.fbin",
            DATA_ROOT / "LAION10M" / "query.10k.fbin",
            DATA_ROOT / "LAION10M" / "base.additional.10M.fbin",
        ],
        "notes": "Use 25M only if data and storage are available; current fallback is LAION10M.",
    },
    {
        "dataset": "msmarco",
        "target": "1M",
        "role": "motivation_control",
        "pg_tables": ["msmarco_kill_passages"],
        "files": [Path("data/msmarco/raw/collection.first1m.tsv")],
        "notes": "Loaded control/motivation dataset, not the new main evaluation dataset.",
    },
    {
        "dataset": "enron",
        "target": "50K",
        "role": "motivation_control_acl",
        "pg_tables": ["enron_messages"],
        "files": [Path("data/enron/processed/enron_sample.csv")],
        "notes": "Loaded ACL/control dataset, not the new main evaluation dataset.",
    },
    {
        "dataset": "text2image",
        "target": "10M",
        "role": "controlled_scalability_only",
        "pg_tables": [],
        "files": [
            DATA_ROOT / "Text2Image10M" / "base.10M.fbin",
            DATA_ROOT / "Text2Image10M" / "query.10K.fbin",
            DATA_ROOT / "Text2Image10M" / "gt.10k.diskann.ibin",
            DATA_ROOT / "Text2Image10M" / "groundtruth-computed.10k.ibin",
        ],
        "notes": "Local vector/GT files exist, but no natural SQL metadata was found.",
    },
    {
        "dataset": "webvid",
        "target": "8M",
        "role": "controlled_scalability_only",
        "pg_tables": [],
        "files": [
            DATA_ROOT / "WebVid8M" / "base.8M.fbin",
            DATA_ROOT / "WebVid8M" / "query.test.10k.fbin",
            DATA_ROOT / "WebVid8M" / "gt.test.10k.ibin",
        ],
        "notes": "Local vector/GT files exist, but SQL predicate metadata still needs verification.",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def fbin_header(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 8 or path.suffix != ".fbin":
        return {}
    with path.open("rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    expected = 8 + int(n) * int(d) * 4
    return {
        "fbin_n": int(n),
        "fbin_dim": int(d),
        "fbin_expected_bytes": expected,
        "fbin_size_matches_header": expected == path.stat().st_size,
    }


def disk_free(path: Path) -> dict[str, int]:
    stat = os.statvfs(path)
    return {
        "disk_total_bytes": stat.f_frsize * stat.f_blocks,
        "disk_free_bytes": stat.f_frsize * stat.f_bavail,
    }


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True)
        return int(out.split()[0])
    except Exception:
        return 0


def pg_table_counts(conninfo: str, table_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not table_names:
        return out
    try:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            cur = conn.cursor()
            for table in table_names:
                cur.execute(
                    """
                    SELECT EXISTS (
                      SELECT 1
                      FROM information_schema.tables
                      WHERE table_schema = 'public' AND table_name = %s
                    )
                    """,
                    (table,),
                )
                exists = bool(cur.fetchone()[0])
                out[f"pg_{table}_exists"] = exists
                if exists:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    out[f"pg_{table}_rows"] = int(cur.fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        out["pg_error"] = f"{exc.__class__.__name__}: {exc}"
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SIGMOD dataset readiness.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    root = repo_root()
    conninfo = pg_config_from_env().conninfo
    disk = disk_free(DATA_ROOT)
    rows: list[dict[str, Any]] = []

    for spec in DATASETS:
        pg_info = pg_table_counts(conninfo, list(spec["pg_tables"]))
        for raw_path in spec["files"]:
            path = raw_path if raw_path.is_absolute() else root / raw_path
            row: dict[str, Any] = {
                "dataset": spec["dataset"],
                "target": spec["target"],
                "role": spec["role"],
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": path_size(path),
                "size_gib": round(path_size(path) / 1024**3, 3),
                "notes": spec["notes"],
                **disk,
                **pg_info,
                **fbin_header(path),
            }
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
