# Hybrid Retrieval

This repository contains prototype code and experiment harnesses for
SQL-native filtered vector search in PostgreSQL with pgvector. The main goal is
to study how SQL predicate state, HNSW traversal, page locality, and reusable
filter metadata interact in hybrid vector queries.

The repository does not include raw datasets, generated indexes, PostgreSQL
data directories, or experiment outputs. Those files should be regenerated
locally.

## Repository Layout

| Path | Contents |
|---|---|
| `docs/` | Project notes, related work, and experiment summaries |
| `experiments/hybrid_vector_db/scripts/` | Data preparation, benchmark, and result-summary scripts |
| `experiments/hybrid_vector_db/sql/` | SQL schemas and smoke-test queries |
| `experiments/hybrid_vector_db/pg_ext/` | Small PostgreSQL helper extension used for profiling |
| `patches/pgvector-sqlens.patch` | Patch against upstream pgvector with SQLens instrumentation and HNSW changes |

## Dependencies

Install Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The PostgreSQL scripts read connection settings from standard environment
variables:

```bash
export PGHOST=127.0.0.1
export PGPORT=55432
export PGDATABASE=hybrid_vector
export PGUSER=postgres
export PGPASSWORD=postgres
```

Dataset locations can be configured with:

```bash
export OOD_ANNS_DATA=/path/to/ood-anns/data
export LAION25M_DATA_DIR=/path/to/LAION25M
export LAION10M_DATA_DIR=/path/to/LAION10M
export YFCC10M_DATA_DIR=/path/to/YFCC10M
```

## pgvector Patch

The SQLens prototype is distributed as a patch instead of vendoring the full
pgvector source tree:

```bash
git clone https://github.com/pgvector/pgvector.git external/pgvector-src
cd external/pgvector-src
git checkout cab9da72c04353f143bb06b42ab70a403daac64a
git apply ../../patches/pgvector-sqlens.patch
make
make install
```

## Reproducibility Notes

- Generated data and results are intentionally ignored by Git.
- Exact result filenames and experiment status are tracked in `docs/`.
- The benchmark scripts preserve PostgreSQL final validation semantics; cached
  or guided metadata is used only to reduce work before final SQL/MVCC checks.
