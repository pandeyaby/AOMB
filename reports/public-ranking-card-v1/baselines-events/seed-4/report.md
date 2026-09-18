# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T14:40:04Z |
| Seed | 4 |
| Score method | event_count_baseline |
| Git HEAD | `dd0280cd9b1daeb9df4c8fa278657f61dbf70f71` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `7dae2e232f78276026eb067fc3d2bc10a695764d019ce8fd2daf6f69aa0eb8ce` |
| Sessions (scorable) | 16 (pos=8, neg=8) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.523438 |
| PR-AUC | 0.559127 |
| precision@1 | 1.000000 |
| precision@8 | 0.625000 |
| mean score (label 0) | 3.875000 |
| mean score (label 1) | 4.000000 |

## Random ranking baseline

Draws: 64 (seed=4)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.491211 | 0.173365 |
| pr_auc | 0.578547 | 0.119390 |
| precision@1 | 0.546875 | 0.501733 |
| precision@8 | 0.503906 | 0.159004 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
