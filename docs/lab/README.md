# Lab docs (private evidence lane)

Fault-injected OpenTelemetry captures used for **development and ranking harness checks**.
Captures themselves are **not published** in this repository (JSONL stays on the operator machine).

> **Public README** describes method only. Numeric lab AUROC / PR-AUC / precision@k live in
> [`ranking-validation.md`](ranking-validation.md) with `claim_status=not_published`.

| Lane | Location | Public accuracy? |
|------|----------|------------------|
| **Public ranking card v1 (fixture)** | [`docs/public-ranking-card-v1.md`](../public-ranking-card-v1.md) + [`corpus/fixtures/public_ranking_card_v1/`](../../corpus/fixtures/public_ranking_card_v1/) | Fixture-scoped claim only — see `reports/public-ranking-card-v1/CARD.md`; **never** lab-pool AUROC |
| **Private lab pool** | [`ranking-validation.md`](ranking-validation.md) / local `lab/captures/` | Lab evidence only — **never** the public card |
| **CRISP `val_bpb`** | [`docs/crisp-val-bpb-baseline.md`](../crisp-val-bpb-baseline.md) | Training fact — not ranking accuracy |

Do **not** copy private lab-pool ranking metrics onto the public ranking card,
README accuracy claims, or fixture reports.

See also: [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md).
