# Baselines table (harness smoke) — NOT a public accuracy claim

Capture: `corpus/fixtures/lab_sample` (fixture). Seeds: 0..4.

| Method | AUROC mean±std | PR-AUC mean±std | precision@1 mean±std | Notes |
|--------|----------------|-----------------|----------------------|-------|
| length_baseline | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | Fixture-sized; verifies harness wiring only |
| event_count_baseline | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | Random-ish on tiny fixture |
| session_bpb (model) | *pending* | *pending* | *pending* | Blocked on lab label fix (ts=now bug); then 5×300s train-then-score |

**Claim status:** `not_published`. Real lab capture `20260911T183259Z` currently yields label_counts={'unknown':327} until OTLP timestamp fix lands.

CRISP scale (local): 500k spans / 7110 sessions (provenance `crisp_zenodo_20260914T144906Z.json`).
