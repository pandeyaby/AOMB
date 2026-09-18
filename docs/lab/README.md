# Lab docs (private evidence lane)

Lab captures and any **private lab pooled ranking** numbers live in this tree
(and under `lab/captures/`, which is gitignored).

| Lane | Location | Public accuracy? |
|------|----------|------------------|
| **Public ranking card v1 (fixture)** | [`docs/public-ranking-card-v1.md`](../public-ranking-card-v1.md) + [`corpus/fixtures/public_ranking_card_v1/`](../../corpus/fixtures/public_ranking_card_v1/) | Future claim path only — still `claim_status=not_published` |
| **Private lab pool** | This directory / local captures | Lab evidence only — **never** the public card |
| **CRISP `val_bpb`** | [`docs/crisp-val-bpb-baseline.md`](../crisp-val-bpb-baseline.md) | Training fact — not ranking accuracy |

Do **not** copy private lab-pool ranking metrics onto the public ranking card,
README accuracy claims, or fixture reports.

See also: [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md).
