# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-25T12:14:53Z |
| Seed | 3 |
| Score method | event_count_baseline |
| Git HEAD | `9900d447dfd9f775cc3e2ceb6f2b78b913036ef1` |
| train.py SHA | `bc736cfeb17e6f190d143d0193e7397f4de70766` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `49403cb3dd005e7e5af1de510ba3f5ed6e83df566e9ba3d46b4baed66367cfc5` |
| Sessions (scorable) | 36 (pos=18, neg=18) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.601852 |
| PR-AUC | 0.574747 |
| precision@3 | 1.000000 |
| precision@10 | 0.700000 |
| mean score (label 0) | 3.444444 |
| mean score (label 1) | 3.944444 |

## Random ranking baseline

Draws: 64 (seed=3)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.508873 | 0.099803 |
| pr_auc | 0.557174 | 0.088527 |
| precision@3 | 0.510417 | 0.302656 |
| precision@10 | 0.518750 | 0.159239 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
