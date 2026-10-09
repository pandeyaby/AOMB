# `val_bpb` audit (2026-10-08): raw results

Raw output behind [`docs/val-bpb-audit.md`](../../docs/val-bpb-audit.md). Each file is one historic commit's `train.py`, run unmodified for the standard 5-minute budget on the corpus it was originally measured on (`eval/val_bpb_audit.py`).

| Field | Meaning |
|-------|---------|
| `true_bpb` | Cross-entropy from logits (the fixed `prepare.evaluate_bpb`) |
| `legacy_bpb` | The old formula, through the model's own loss path: what the project reported |
| `legacy_over_true` | Their ratio on the same trained model |
