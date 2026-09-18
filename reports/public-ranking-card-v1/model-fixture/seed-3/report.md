# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-18T15:12:20Z |
| Seed | 3 |
| Score method | session_bpb_fixture_train_then_score |
| Git HEAD | `f8e5aa5428eabe8394b3a00a6dceedcb82107f6c` |
| train.py SHA | `fd4cbb674fd2eb29b977355de6467c5d0ae6f9ae` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | public_ranking_card_v1 |
| Corpus content SHA-256 | `f5a0183bbe4ea67d5efd13af7f5fe2a432226e33ba718d81380dd39994e282bb` |
| Sessions (scorable) | 6 (pos=3, neg=3) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 1.000000 |
| PR-AUC | 1.000000 |
| precision@1 | 1.000000 |
| precision@3 | 1.000000 |
| mean score (label 0) | 0.656370 |
| mean score (label 1) | 6.523833 |

## Random ranking baseline

Draws: 64 (seed=3)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.449653 | 0.243617 |
| pr_auc | 0.617188 | 0.167756 |
| precision@1 | 0.484375 | 0.503706 |
| precision@3 | 0.437500 | 0.213107 |

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
