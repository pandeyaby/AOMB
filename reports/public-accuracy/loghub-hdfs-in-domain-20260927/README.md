# LogHub HDFS_v1, in-domain eval (2026-09-27): raw output

Raw output behind [`docs/real-data/loghub-hdfs.md`](../../../docs/real-data/loghub-hdfs.md). The HDFS data itself isn't vendored (CC BY 4.0, download from Zenodo); the doc has the exact rebuild commands and the session-file hash.

| Path | Contents |
|------|----------|
| `results.json` / `results.md` | AUROC / PR-AUC / P@10 for every baseline and model variant, per seed |
| `best_f1.json` | Best-threshold F1 (upper bound, for comparison with published work) |
| `sessions.json` | Every eval block's id, label, and model scores per seed |
| `heatmap-seed0.html` | Per-token surprise for the most/least surprising and missed blocks |
