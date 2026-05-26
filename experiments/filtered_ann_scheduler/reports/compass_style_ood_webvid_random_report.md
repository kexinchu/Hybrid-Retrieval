# Compass-Style OOD-ANNS Report: WebVid8M

This report summarizes the current best motivation run. For full context and
reproduction commands, see `../EXPERIMENT_SUMMARY.md` and `../REPRODUCE.md`.

## Setup

- ANN data: `/home/kec23008/docker-sys/OOD-ANNS/data/WebVid8M/hard_random.1M.fbin`
- Vector shape: `250,000 x 512`
- ANNS: FAISS `IndexHNSWFlat`
- SQL side: SQLite in-memory table with B-tree indexes
- Structured attributes: generated Compass-style attributes
- Queries: `80`
- `k = 10`

## Key Finding

Low-selectivity post-filter ANN fails even with large ANN expansion, while SQL
prefilter exact becomes expensive for broad predicates.

At around `0.1%` selectivity:

| Predicate | Strategy | Recall | Fail Rate | ANN Returned |
| --- | --- | ---: | ---: | ---: |
| `attr_random_sel0.001` | `post_ann_10x` | `0.1075` | `1.0000` | `100` |
| `attr_random_sel0.001` | `post_ann_100x` | `0.1625` | `1.0000` | `800.2` |
| `attr_random_sel0.001` | `iterative_ann` | `0.1638` | `1.0000` | `20000` |
| `cluster_and_random_sel0.05` | `iterative_ann` | `0.1513` | `1.0000` | `20000` |

Filter-first cost grows with match count:

| Predicate | Selectivity | Matches | P95 Latency |
| --- | ---: | ---: | ---: |
| `attr_random_sel0.001` | `0.10%` | `249.6` | `0.3226 ms` |
| `attr_random_sel0.01` | `1.00%` | `2497.2` | `17.5506 ms` |
| `attr_random_sel0.05` | `5.00%` | `12512.0` | `28.2170 ms` |
| `attr_random_sel0.2` | `20.01%` | `50033.8` | `94.0281 ms` |

## Interpretation

This supports a Compass-like scheduler design:

```text
probe ANN -> estimate local pass rate
if narrow and local yield low:
    use SQLite/B-tree candidate injection + exact rerank
elif broad:
    continue ANN-first with bounded expansion
else:
    interleave ANN expansion and structured-index candidate injection
```

