# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T15:28:24Z |
| Seed | 0 |
| Score method | length_baseline |
| Git HEAD | `dd081ed3edec2bfb820f6ffc7848d7a6c6402cd7` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `49403cb3dd005e7e5af1de510ba3f5ed6e83df566e9ba3d46b4baed66367cfc5` |
| Sessions (scorable) | 36 (pos=18, neg=18) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.700617 |
| PR-AUC | 0.732859 |
| precision@3 | 1.000000 |
| precision@10 | 0.700000 |
| mean score (label 0) | 687.055556 |
| mean score (label 1) | 810.444444 |

## Random ranking baseline

Draws: 64 (seed=0)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.487365 | 0.101303 |
| pr_auc | 0.529526 | 0.075237 |
| precision@3 | 0.442708 | 0.230441 |
| precision@10 | 0.496875 | 0.136822 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
