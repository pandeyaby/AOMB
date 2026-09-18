# Public accuracy eval — multi-seed aggregate

**Claim status:** `not_published`

Multi-seed aggregate only. Public claim language forbidden until docs/public-accuracy-eval.md checklist passes.

Seeds (5): [0, 1, 2, 3, 4]
Score method: session_bpb_train_then_score

## mean ± std

| Metric | mean | std | n |
|--------|------|-----|---|
| auroc | 0.665031 | 0.045039 | 5 |
| pr_auc | 0.676026 | 0.108295 | 5 |
| precision@10 | 0.600000 | 0.412311 | 5 |
| precision@32 | 0.687500 | 0.385276 | 5 |

## Per-seed

- seed=0: AUROC=0.640857 PR-AUC=0.556675
- seed=1: AUROC=0.641352 PR-AUC=0.733149
- seed=2: AUROC=0.643257 PR-AUC=0.563072
- seed=3: AUROC=0.654727 PR-AUC=0.738119
- seed=4: AUROC=0.744961 PR-AUC=0.789116
