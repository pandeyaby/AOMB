# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T15:14:35Z |
| Seed | 2 |
| Score method | length_baseline |
| Git HEAD | `f8e5aa5428eabe8394b3a00a6dceedcb82107f6c` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `f8a8288b1e37225b5cda9f8eef8e1e0879e3468797ce969b595ddfb17fdb4e1c` |
| Sessions (scorable) | 6 (pos=3, neg=3) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.555556 |
| PR-AUC | 0.722222 |
| precision@1 | 1.000000 |
| precision@3 | 0.666667 |
| mean score (label 0) | 847.666667 |
| mean score (label 1) | 814.000000 |

## Random ranking baseline

Draws: 64 (seed=2)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.510417 | 0.240614 |
| pr_auc | 0.650174 | 0.170298 |
| precision@1 | 0.500000 | 0.503953 |
| precision@3 | 0.515625 | 0.196348 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
