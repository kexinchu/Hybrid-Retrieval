# Filtered ANN Scheduler Motivation Tests

This harness is for motivation experiments around ANNS plus structured
filters. It creates a controlled vector+attribute dataset, runs several
filtered search strategies, and reports quality/cost bottlenecks.

## What It Tests

- Filter selectivity: when does pre-filtering beat ANN-first search?
- Filter/vector correlation: when do graph/search candidates become biased?
- Fixed overfetch fragility: how often does post-filtering return too few hits?
- Runtime scheduling: can a cheap adaptive rule avoid the worst cases?

## Strategies

- `pre_exact`: filter first, then exact distances on matching rows.
- `post_ann`: ANN first with a fixed overfetch budget, then filter.
- `iterative_ann`: ANN first, but grows the candidate budget until enough
  filtered hits are found or a cap is reached.
- `adaptive`: uses estimated selectivity to choose `pre_exact` for narrow
  filters, otherwise uses `iterative_ann`.

## Quick Start

```bash
python experiments/filtered_ann_scheduler/run_benchmark.py --quick
```

Write CSV results:

```bash
python experiments/filtered_ann_scheduler/run_benchmark.py \
  --n 50000 --queries 80 --out results/filtered_ann_pilot.csv
```

Summarize a CSV:

```bash
python experiments/filtered_ann_scheduler/summarize_results.py \
  results/filtered_ann_pilot.csv
```

