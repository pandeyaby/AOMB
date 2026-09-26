# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-25T11:35:00Z |
| Seed | 3 |
| Score method | length_baseline |
| Git HEAD | `9900d447dfd9f775cc3e2ceb6f2b78b913036ef1` |
| train.py SHA | `bc736cfeb17e6f190d143d0193e7397f4de70766` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | pooled-20260918 |
| Corpus content SHA-256 | `effb9a3a70075d7e0681c3de6d4a9e236de8677b14dba3acd851a945b18e53f5` |
| Sessions (scorable) | 2900 (pos=1444, neg=1456) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.543396 |
| PR-AUC | 0.527583 |
| precision@10 | 0.000000 |
| precision@290 | 0.696552 |
| mean score (label 0) | 597.338599 |
| mean score (label 1) | 570.123961 |

## Random ranking baseline

Draws: 64 (seed=3)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.501493 | 0.013349 |
| pr_auc | 0.499592 | 0.010101 |
| precision@10 | 0.453125 | 0.140259 |
| precision@290 | 0.500808 | 0.028927 |

## Notes

Fixture/length baselines are for harness verification only. They are not a public accuracy claim.

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
