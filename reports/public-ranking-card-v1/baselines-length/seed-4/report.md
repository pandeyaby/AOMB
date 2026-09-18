# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T15:21:45Z |
| Seed | 4 |
| Score method | length_baseline |
| Git HEAD | `d180bf8d50bd39302844b54ed8cefcb179f5e908` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `8c9ad74ae718bcc9541671ccd3d212e75f5a3ae0013890504508788a1b6d309e` |
| Sessions (scorable) | 24 (pos=12, neg=12) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.812500 |
| PR-AUC | 0.860101 |
| precision@2 | 1.000000 |
| precision@10 | 0.800000 |
| mean score (label 0) | 651.750000 |
| mean score (label 1) | 862.416667 |

## Random ranking baseline

Draws: 64 (seed=4)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.494683 | 0.132565 |
| pr_auc | 0.553336 | 0.107873 |
| precision@2 | 0.437500 | 0.372678 |
| precision@10 | 0.493750 | 0.140153 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
