# In-domain lab eval (2026-09-25): raw output

Raw output behind [`docs/lab/in-domain-eval.md`](../../../docs/lab/in-domain-eval.md). Produced by `uv run python -m eval.in_domain --capture lab/captures/pooled-20260918 --seeds 0..4 --train-seconds 120`.

| Path | Contents |
|------|----------|
| `results.json` / `results.md` | All methods (baselines + model variants), per-seed and per-capture AUROC |
| `sessions.json` | Every eval session's id, label, capture, and model scores per seed |
| `heatmap-seed0.html` | Per-token surprise for the most/least surprising and missed sessions (IDs and timestamps faded = not scored) |
| `sweep-seed0/` | Training-length sweep (10/30/60/300 s, seed 0); `summary.json` has held-out BPB vs AUROC |
