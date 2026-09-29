# RCAEval RE3 (online-boutique), in-domain eval (2026-09-27): raw output

Raw output behind [`docs/real-data/rcaeval-re3.md`](../../../docs/real-data/rcaeval-re3.md). RCAEval data isn't vendored (MIT, download from Hugging Face); the doc has the rebuild commands and session-file hashes.

| Path | Contents |
|------|----------|
| `results.json` / `results.md` | Whole-set and `touches_root_cause` subset AUROC for every baseline and model variant, per seed and per case |
| `sessions.json` | Every eval trace's id, label, case, and model scores per seed |
| `heatmap-seed0.html` | Per-token surprise for the most/least surprising and missed traces |
