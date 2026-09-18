# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-14T15:06:27Z |
| Seed | 3 |
| Score method | session_bpb_train_then_score |
| Git HEAD | `e8400eb9f41c5cbec8dddaf5907c898820ef5ed7` |
| train.py SHA | `3895c14507d056cef8e4b43aeae3778bcb9f6231` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | fixture-demo |
| Corpus content SHA-256 | `d9cd31e6b3fc88b89ece8268242d424c0c3bf002da1aa7a9a3d4c90e0ca136b9` |
| Sessions (scorable) | 2 (pos=1, neg=1) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 1.000000 |
| PR-AUC | 1.000000 |
| precision@1 | 1.000000 |
| mean score (label 0) | 5.814721 |
| mean score (label 1) | 6.839449 |

## Random ranking baseline

Draws: 64 (seed=3)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.484375 | 0.503706 |
| pr_auc | 0.742188 | 0.251853 |
| precision@1 | 0.484375 | 0.503706 |

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
