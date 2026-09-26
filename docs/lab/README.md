# Lab docs (labelled evidence lane)

These are fault-injected OpenTelemetry captures from the lab stack in [`lab/`](../../lab/). They're used for labelled ranking evaluation.
The raw captures are **not** in this repository; the JSONL stays on the operator's machine. Content hashes and per-session scores are published.

> **Published results (2026-09-25):** zero-shot AUROC **0.583** ([`ranking-validation.md`](ranking-validation.md)) and in-domain **0.688**, compared against a simple rule at **0.776** ([`in-domain-eval.md`](in-domain-eval.md)). On rule-proof faults, the model alone catches value drift and misses missing calls ([`rule-proof-eval.md`](rule-proof-eval.md)).

**Publish gate:** [`publish-checklist.md`](publish-checklist.md). Every new lab number has to pass it on its own run.

| Lane | Location | Public accuracy? |
|------|----------|------------------|
| **Public ranking card v1 (fixture)** | [`docs/public-ranking-card-v1.md`](../public-ranking-card-v1.md) + [`corpus/fixtures/public_ranking_card_v1/`](../../corpus/fixtures/public_ranking_card_v1/) | `published_fixture_card` = harness smoke only — **never** lab-pool / production AUROC |
| **Lab pool** | [`ranking-validation.md`](ranking-validation.md) / local `lab/captures/` | **Published** (zero-shot, with limitations). It is **never** copied onto the fixture card |
| **CRISP `val_bpb`** | [`docs/crisp-val-bpb-baseline.md`](../crisp-val-bpb-baseline.md) | Training fact — not ranking accuracy |

Do **not** copy lab-pool metrics onto the public fixture card or fixture reports.

See also: [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md).

## Redacted public pack v0

See [`lab-public-pack-v0.md`](lab-public-pack-v0.md) and [`corpus/fixtures/lab_public_pack_v0/`](../../corpus/fixtures/lab_public_pack_v0/).
