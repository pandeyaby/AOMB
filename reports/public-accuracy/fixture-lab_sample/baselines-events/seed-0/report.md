# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-14T14:49:43Z |
| Seed | 0 |
| Score method | event_count_baseline |
| Git HEAD | `e8400eb9f41c5cbec8dddaf5907c898820ef5ed7` |
| train.py SHA | `3895c14507d056cef8e4b43aeae3778bcb9f6231` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | fixture-demo |
| Corpus content SHA-256 | `d9cd31e6b3fc88b89ece8268242d424c0c3bf002da1aa7a9a3d4c90e0ca136b9` |
| Sessions (scorable) | 2 (pos=1, neg=1) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.500000 |
| PR-AUC | 0.500000 |
| precision@1 | 0.000000 |
| mean score (label 0) | 3.000000 |
| mean score (label 1) | 3.000000 |

## Random ranking baseline

Draws: 64 (seed=0)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.421875 | 0.497763 |
| pr_auc | 0.710938 | 0.248881 |
| precision@1 | 0.421875 | 0.497763 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
