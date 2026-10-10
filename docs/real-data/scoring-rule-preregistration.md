# Pre-registration: one scoring rule, tested once on BGL

**Written and committed before** the end-of-session reruns finished and before any model was run on BGL. The git history of this file is the record.

## Why

Earlier evals reported the best of several scoring variants per dataset, and the best variant differed by dataset. That flatters the model. This fixes **one** rule from existing datasets, then tests it **once** on a dataset none of this work has touched.

## Candidates

All use the end-of-session marker run, with IDs, timestamps and counters masked. `bpb_mean` is excluded because it scores IDs and timestamps.

| Candidate | Definition |
|-----------|------------|
| `bpb_content` | Mean bits-per-byte over the session |
| `bpb_max_event` | Highest per-line bits-per-byte |
| `bpb_top10` | Mean bits of the 10% most surprising tokens |
| `bits_max_field` | Bits of the most surprising field value |
| `bits_end` | Bits of the end-of-session marker |
| `combo_field_end` | Percentile rank of `bits_max_field` + percentile rank of `bits_end` (ranks within the scored set; no labels) |
| `combo_content_end` | Percentile rank of `bpb_content` + percentile rank of `bits_end` |

## Development datasets and views

| # | Dataset | View |
|---|---------|------|
| 1 | Lab, error/latency faults (`pooled-20260918`) | all eval sessions |
| 2 | Lab, rule-proof faults (`pooled-20260925-ruleproof`) | checkout sessions |
| 3 | Lab, value drift (`pooled-20260926-valuedrift`) | checkout sessions |
| 4 | LogHub HDFS_v1 | all eval sessions |
| 5 | RCAEval RE3, Online Boutique | traces touching the faulted service |
| 6 | RCAEval RE3, Train Ticket | traces touching the faulted service |

## Selection rule

For each candidate, take AUROC averaged over seeds on each of the six views, then average the six. **The candidate with the highest six-view mean AUROC is the rule.** No other consideration enters. Computed by `eval/select_rule.py`.

## The test

LogHub BGL, protocol as fixed in `corpus/ingest/loghub_bgl.py` (20-line chronological windows; train on the first 5,000 normal windows; 10,000 later windows sampled with seed 0; 863 incidents). Model: 120 s × 5 seeds, end-of-session marker on, ≤24-line training chunks.

Reported, whatever they are: the chosen rule's AUROC and PR-AUC, every baseline's, and every other candidate's (so it's visible whether a different candidate would have done better). The result goes in `docs/real-data/loghub-bgl.md`.
