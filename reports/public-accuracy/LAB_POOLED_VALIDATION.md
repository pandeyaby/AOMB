# Lab pooled labeled ranking eval — NOT a public accuracy claim

> **Canonical narrative:** [`docs/lab/ranking-validation.md`](../../docs/lab/ranking-validation.md)
>
> **claim_status=`not_published`.** Not a public accuracy claim.

## Results (mean ± std over 5 seeds)

| Method | AUROC | PR-AUC | precision@10 | precision@290 |
|--------|-------|--------|--------------|---------------|
| `length_baseline` | 0.6428 ± 0.0000 | 0.6159 ± 0.0000 | 0.0000 ± 0.0000 | 0.8621 ± 0.0000 |
| `event_count_baseline` | 0.4762 ± 0.0000 | 0.4849 ± 0.0000 | 0.0000 ± 0.0000 | 0.4655 ± 0.0000 |
| `session_bpb_train_then_score_300s` | 0.7660 ± 0.0115 | 0.8142 ± 0.0124 | 0.9000 ± 0.0000 | 0.9676 ± 0.0111 |

Pool: 2900 scorable sessions. Seeds 0..4. Mac MPS. Captures not vendored.
Aggregates: [`lab-pooled-20260918/`](lab-pooled-20260918/).
