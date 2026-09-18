# Public accuracy eval report

**Claim status:** `not_published`

Scaffolding / measurement report only. Do not cite as a public accuracy claim until the protocol checklist passes. Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy.

## Identity

| Field | Value |
|-------|-------|
| Generated (UTC) | 2026-09-14T15:38:13Z |
| Seed | 4 |
| Score method | session_bpb_train_then_score |
| Git HEAD | `b9c93c4c7acd73c8115d873a273b30720304c052` |
| train.py SHA | `3895c14507d056cef8e4b43aeae3778bcb9f6231` |
| prepare.py SHA | `b71a0d440d5bb05af85fba6b575b7a245af38a7a` |
| Corpus capture_id | 20260911T183259Z |
| Corpus content SHA-256 | `9e96ce893f75cb63c83b011b3ac613db73b404ebb759bd3335cb384053507a59` |
| Sessions (scorable) | 324 (pos=161, neg=163) |

## Metrics (this seed)

| Metric | Value |
|--------|-------|
| AUROC | 0.744961 |
| PR-AUC | 0.789116 |
| precision@10 | 0.900000 |
| precision@32 | 0.968750 |
| mean score (label 0) | 5.239449 |
| mean score (label 1) | 5.644169 |

## Random ranking baseline

Draws: 64 (seed=4)

| Metric | mean | std |
|--------|------|-----|
| auroc | 0.496340 | 0.030137 |
| pr_auc | 0.502965 | 0.024875 |
| precision@10 | 0.498438 | 0.144191 |
| precision@32 | 0.498047 | 0.071059 |

## Checklist reminder

See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any public claim language. This file alone is not a published claim.
