# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-25T12:14:49Z |
| Seed | 1 |
| Score method | length_baseline |
| Git HEAD | `9900d447dfd9f775cc3e2ceb6f2b78b913036ef1` |
| train.py SHA | `bc736cfeb17e6f190d143d0193e7397f4de70766` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `49403cb3dd005e7e5af1de510ba3f5ed6e83df566e9ba3d46b4baed66367cfc5` |
| Sessions (scorable) | 36 (pos=18, neg=18) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.663580 |
| PR-AUC | 0.705060 |
| precision@3 | 1.000000 |
| precision@10 | 0.700000 |
| mean score (label 0) | 592.055556 |
| mean score (label 1) | 696.111111 |

## Random ranking baseline

Draws: 64 (seed=1)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.511815 | 0.103713 |
| pr_auc | 0.552823 | 0.090741 |
| precision@3 | 0.500000 | 0.308607 |
| precision@10 | 0.507812 | 0.141763 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
