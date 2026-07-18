# Hybrid Retrieval Documentation

This directory is the canonical place for project notes and experiment
summaries. Keep new Markdown here unless the file is a top-level repository
entry point.

## Documents

| File | Purpose |
|---|---|
| `SIGMOD_EXPERIMENT_PLAN.md` | Paper-facing experiment plan, current status, and next experiment queue |
| `MOTIVATION_RESULT.md` | Current motivation and mechanism results with concrete result files |
| `RELATED_WORKS.md` | Related systems and paper notes |
| `PAGE_AWARE_PGVECTOR.md` | Page-aware pgvector implementation notes and locality experiments |

## Current Main Experiment Line

The current paper line is SQL-native filtered vector search in PostgreSQL +
pgvector on Amazon Grocery 10M. The four implemented design components are:

| Component | Role |
|---|---|
| D1 | Predicate guidance for HNSW |
| D2 | Locality/layout-aware execution |
| D3 | Reusable/cache state |
| D4 | Adaptive route calibration |

Primary result files:

- `results/hybrid_vector_db/sigmod_d123_selectivity_q100r10_warmall_main_20260710_002705_table.csv`
- `results/hybrid_vector_db/sigmod_d4_calibration_q100r10_warmall_main_20260710_002705_merged_table.csv`
- `results/hybrid_vector_db/sigmod_candidate_waste_q100r10_main_20260710_002705.csv`
- `results/hybrid_vector_db/sigmod_target_recall_calibration_calib_20260710_123543.csv`
- `results/hybrid_vector_db/sigmod_c4_guidance_memory_filteroff_q25_20260710_summary.json`
- `results/hybrid_vector_db/sigmod_c4_guidance_memory_acorn1_q25_20260710_summary.json`

## Key Scripts

| Script | Purpose |
|---|---|
| `experiments/hybrid_vector_db/scripts/run_sigmod_overnight.py` | Runs the D1-D4 overnight experiment chain |
| `experiments/hybrid_vector_db/scripts/sigmod_result_summary.py` | Summarizes raw SIGMOD CSV/JSON outputs |
| `experiments/hybrid_vector_db/scripts/sigmod_candidate_waste_summary.py` | Builds candidate-waste summary tables |
| `experiments/hybrid_vector_db/scripts/pgvector_target_recall_selectivity_runner.py` | Calibrates pgvector parameters against a target recall |
| `experiments/hybrid_vector_db/scripts/pgvector_c4_guidance_memory_benchmark.py` | Runs C4 cache/guidance controls |
| `experiments/hybrid_vector_db/scripts/pgvector_hnsw_page_access_group_benchmark.py` | Runs page/index locality diagnostics |

## Current Interpretation

- The q100/r10 Amazon 10M table is an attainable-recall result under
  `hnsw.ef_search = 1000`, not a fixed recall@10 = 0.9 result.
- Candidate-waste reduction is the strongest current mechanism result.
- Cache and ACORN1 traversal must be reported separately. Pure cache with
  `filter_strategy=off` is neutral/slightly positive; ACORN1 can greatly
  increase visited graph tuples on mixed C4 predicates.
- D4 is still a route-calibration diagnostic, not yet a universal win.
