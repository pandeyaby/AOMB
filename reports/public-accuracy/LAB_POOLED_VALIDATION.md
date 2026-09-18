# Lab pooled labeled ranking eval — NOT a public accuracy claim

> **claim_status=`not_published`**. Protocol checklist in `docs/public-accuracy-eval.md` is **not** fully greenlit for public accuracy language. Do not market these as product accuracy.

## Corpus

| Field | Value |
|-------|--------|
| Pool id | `lab/captures/pooled-20260918` |
| Sources | `20260911T183259Z` + latency / errors / both / kill_redis (2026-09-18) |
| Scorable sessions | **2900** (1456 normal / 1444 incident; 7 unknown excluded) |
| Seeds | 0..4 |
| Model path | train-then-score **300s**/seed (`TIME_BUDGET`) |
| Hardware | Mac Apple Silicon (MPS) |

## Results (mean ± std over 5 seeds)

| Method | AUROC | PR-AUC | precision@10 | precision@290 |
|--------|-------|--------|--------------|---------------|
| `length_baseline` | 0.6428 ± 0.0000 | 0.6159 ± 0.0000 | 0.0000 ± 0.0000 | 0.8621 ± 0.0000 |
| `event_count_baseline` | 0.4762 ± 0.0000 | 0.4849 ± 0.0000 | 0.0000 ± 0.0000 | 0.4655 ± 0.0000 |
| `session_bpb_train_then_score_300s` | 0.7660 ± 0.0115 | 0.8142 ± 0.0124 | 0.9000 ± 0.0000 | 0.9676 ± 0.0111 |

## Notes

- Session BPB beats length and event-count baselines on AUROC / PR-AUC / precision@k.
- Still a **lab** corpus (fault-injected), not production multi-tenant traffic.
- CRISP `val_bpb` lanes (0.407753 / 0.4309 / 0.3682) remain separate training facts — **not** these ranking metrics.
- Random ranking baseline is included in per-seed `report.json` files under this directory.

