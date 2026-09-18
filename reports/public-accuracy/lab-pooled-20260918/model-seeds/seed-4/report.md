# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T10:10:06Z |
| Seed | 4 |
| Score method | session_bpb_train_then_score |
| Git HEAD | `dd0280cd9b1daeb9df4c8fa278657f61dbf70f71` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | pooled-20260918 |
| Corpus content SHA-256 | `effb9a3a70075d7e0681c3de6d4a9e236de8677b14dba3acd851a945b18e53f5` |
| Sessions (scorable) | 2900 (pos=1444, neg=1456) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.765121 |
| PR-AUC | 0.820879 |
| precision@10 | 0.900000 |
| precision@290 | 0.975862 |
| mean score (label 0) | 5.376523 |
| mean score (label 1) | 6.124250 |

## Random ranking baseline

Draws: 64 (seed=4)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.499634 | 0.010699 |
| pr_auc | 0.499089 | 0.008889 |
| precision@10 | 0.500000 | 0.174574 |
| precision@290 | 0.501239 | 0.028636 |

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
