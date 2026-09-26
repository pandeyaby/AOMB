# Public accuracy eval — multi-seed aggregate

**Claim status:** `not_published`

Multi-seed aggregate only. Public claim language forbidden until docs/public-accuracy-eval.md checklist passes.

Seeds (5): [0, 1, 2, 3, 4]
Score method: session_bpb_train_then_score

## mean ± std

| Metric | mean | std | n |
|--------|------|-----|---|
| auroc | 0.582565 | 0.006957 | 5 |
| pr_auc | 0.631494 | 0.008618 | 5 |
| precision@10 | 0.900000 | 0.000000 | 5 |
| precision@290 | 0.896552 | 0.033432 | 5 |

## Per-seed

- seed=0: AUROC=0.588754 PR-AUC=0.641108
- seed=1: AUROC=0.588338 PR-AUC=0.634048
- seed=2: AUROC=0.582379 PR-AUC=0.631188
- seed=3: AUROC=0.571550 PR-AUC=0.617573
- seed=4: AUROC=0.581804 PR-AUC=0.633553
