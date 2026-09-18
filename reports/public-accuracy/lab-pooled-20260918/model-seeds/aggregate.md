# Public accuracy eval — multi-seed aggregate

**Claim status:** `not_published`

Multi-seed aggregate only. Public claim language forbidden until docs/public-accuracy-eval.md checklist passes.

Seeds (5): [0, 1, 2, 3, 4]
Score method: session_bpb_train_then_score

## mean ± std

| Metric | mean | std | n |
|--------|------|-----|---|
| auroc | 0.765977 | 0.011546 | 5 |
| pr_auc | 0.814170 | 0.012434 | 5 |
| precision@10 | 0.900000 | 0.000000 | 5 |
| precision@290 | 0.967586 | 0.011067 | 5 |

## Per-seed

- seed=0: AUROC=0.783439 PR-AUC=0.829378
- seed=1: AUROC=0.759348 PR-AUC=0.798966
- seed=2: AUROC=0.752807 PR-AUC=0.804163
- seed=3: AUROC=0.769173 PR-AUC=0.817465
- seed=4: AUROC=0.765121 PR-AUC=0.820879
