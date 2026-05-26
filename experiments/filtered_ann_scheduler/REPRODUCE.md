# Reproducing Filtered ANN Scheduler Experiments

This directory contains motivation experiments for filtered ANN search:

```sql
SELECT id
FROM items
WHERE structured_predicate
ORDER BY distance(embedding, query)
LIMIT k;
```

The main target is Compass-style ANNS + structured-data search, with room for a
runtime scheduler that decides when to use ANN expansion, relational prefilter,
or hybrid candidate injection.

## Environment

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The experiments use:

- `faiss-cpu`
- `numpy`
- `pandas`
- `scikit-learn`
- `sqlite3` from the Python standard library

## Data Required

The most Compass-like run expects OOD-ANNS vector files at:

```text
/home/kec23008/docker-sys/OOD-ANNS/data
```

The default file is:

```text
/home/kec23008/docker-sys/OOD-ANNS/data/WebVid8M/hard_random.1M.fbin
```

It contains `250,000 x 512` float32 vectors. If using a different machine,
either keep the same path or pass `--fbin /path/to/file.fbin`.

The runner reads `.fbin` files with this layout:

```text
int32 n
int32 dim
float32[n, dim] vectors
```

## Main Compass-Style Experiment

Quick smoke test:

```bash
python experiments/filtered_ann_scheduler/run_compass_style_ood.py --quick
python experiments/filtered_ann_scheduler/summarize_compass_style_ood.py \
  results/compass_style_ood_webvid_random_quick.csv
python experiments/filtered_ann_scheduler/validate_real_results.py \
  results/compass_style_ood_webvid_random_quick.csv
```

Full pilot used in the current notes:

```bash
python experiments/filtered_ann_scheduler/run_compass_style_ood.py \
  --limit 250000 \
  --queries 80 \
  --out results/compass_style_ood_webvid_random_full.csv

python experiments/filtered_ann_scheduler/summarize_compass_style_ood.py \
  results/compass_style_ood_webvid_random_full.csv

python experiments/filtered_ann_scheduler/validate_real_results.py \
  results/compass_style_ood_webvid_random_full.csv
```

## What This Experiment Uses

ANNS side:

- real vectors from OOD-ANNS / WebVid8M `.fbin`
- FAISS `IndexHNSWFlat`

Structured side:

- SQLite in-memory table
- B-tree indexes
- Compass-style synthetic attributes attached to each vector row

Generated columns:

- `attr_random`: random uniform attribute
- `attr_corr`: projection-rank attribute correlated with a vector direction
- `attr_anti`: `1 - attr_corr`
- `bucket_random`: 100 random categorical buckets
- `cluster_id`: 100 projection buckets

This is not a 100% reproduction of Compass. It is a Compass-style motivation
benchmark using real ANN vectors and generated SQL attributes.

## Other Experiments

Synthetic scalar filtered ANN:

```bash
python experiments/filtered_ann_scheduler/run_benchmark.py --quick
python experiments/filtered_ann_scheduler/summarize_results.py \
  results/filtered_ann_scheduler_quick.csv
```

Synthetic multimodal proxy:

```bash
python experiments/filtered_ann_scheduler/run_multimodal_proxy.py --quick
python experiments/filtered_ann_scheduler/summarize_multimodal.py \
  results/multimodal_proxy_quick.csv
```

20 Newsgroups real-data proxy:

```bash
python experiments/filtered_ann_scheduler/run_real_20newsgroups.py --quick
python experiments/filtered_ann_scheduler/summarize_real.py \
  results/real_20newsgroups_quick.csv
```

The first run may download `20 Newsgroups`; the local `data/` directory is
ignored by git.

## Outputs

CSV outputs are written to `results/`, which is ignored by git. This keeps the
repository portable. Re-run the commands above on a new machine to regenerate
the CSV files.

