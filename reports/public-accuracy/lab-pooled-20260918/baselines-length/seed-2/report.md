# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T09:37:06Z |
| Seed | 2 |
| Score method | length_baseline |
| Git HEAD | `dd0280cd9b1daeb9df4c8fa278657f61dbf70f71` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | pooled-20260918 |
| Corpus content SHA-256 | `effb9a3a70075d7e0681c3de6d4a9e236de8677b14dba3acd851a945b18e53f5` |
| Sessions (scorable) | 2900 (pos=1444, neg=1456) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.642801 |
| PR-AUC | 0.615926 |
| precision@10 | 0.000000 |
| precision@290 | 0.862069 |
| mean score (label 0) | 679.338599 |
| mean score (label 1) | 667.899584 |

## Random ranking baseline

Draws: 64 (seed=2)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.502977 | 0.009406 |
| pr_auc | 0.503299 | 0.008693 |
| precision@10 | 0.529688 | 0.137644 |
| precision@290 | 0.511692 | 0.027754 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
