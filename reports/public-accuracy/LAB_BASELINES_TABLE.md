# Lab labeled eval — multi-seed (NOT a public accuracy claim)

Capture: `lab/captures/20260911T183259Z` (post-#11 OTel nanos fix). Seeds: 0..4.
Labels: ~163 normal / ~161 incident. Model path: train-then-score 300s/seed.
**Claim status:** `not_published`.

| Method | AUROC mean±std | PR-AUC mean±std | precision@k mean±std |
|--------|----------------|-----------------|----------------------|
| length_baseline | 0.6265 ± 0.0000 | 0.5551 ± 0.0000 | nan ± nan |
| event_count_baseline | 0.4749 ± 0.0000 | 0.5000 ± 0.0000 | nan ± nan |
| session_bpb_train_then_score | 0.6650 ± 0.0450 | 0.6760 ± 0.1083 | nan ± nan |
