# Agent loop vs detection

Primary score (fixed in advance): `bpb_top10`. Spearman ρ(val_bpb, primary AUROC) = **0.750** over 7 runs (negative = lower val_bpb goes with higher AUROC).

| # | Commit | val_bpb | AUROC (primary) | PR-AUC (primary) | AUROC (max event) | AUROC (per-field) | Change |
|---|--------|---------|-----------------|------------------|-------------------|-------------------|--------|
| 0 | `fdd9b57` | 0.4682 | 0.842 | 0.240 | 0.931 | 0.796 | branch start (before any agent change) |
| 1 | `6eb53fd` | 0.5533 | 0.911 | 0.521 | 0.887 | 0.791 | [inject_auroc=0.5782] [Δ=+0.0268] [val_bpb=0.6098] [change: ** In `GPT |
| 2 | `62f644a` | 0.6353 | 0.922 | 0.602 | 0.892 | 0.807 | [inject_auroc=0.7423] [Δ=+0.1640] [val_bpb=0.5442] [change: ** `MATRIX |
| 3 | `3af4972` | 0.5764 | 0.957 | 0.591 | 0.875 | 0.807 | [inject_auroc=0.7660] [Δ=+0.0237] [val_bpb=0.5581] [change: smaller ba |
| 4 | `d69869c` | 0.6314 | 0.958 | 0.641 | 0.926 | 0.805 | [inject_auroc=0.7957] [Δ=+0.0297] [val_bpb=0.6061] [change: self._adam |
| 5 | `fdd9b57` | 0.4687 | 0.842 | 0.434 | 0.876 | 0.788 | branch start (before any agent change) |
| 6 | `d69869c` | 0.5828 | 0.952 | 0.625 | 0.888 | 0.803 | [inject_auroc=0.7957] [Δ=+0.0297] [val_bpb=0.6061] [change: self._adam |
