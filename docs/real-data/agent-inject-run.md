# Agent run on the injected-fault objective (HDFS)

> **claim_status=`published`** (2026-10-11).
> Optimising injected-fault AUROC overnight raised detection of real HDFS anomalies the agent never saw, from 0.842 to 0.958 AUROC under the fixed rule. It is the first agent run in this project that produced a measured detection gain. The gain brings `train.py` up to what the simpler fixture trainer already reached (0.964); it doesn't beat the hand-built check (0.977).

## Setup

- **Objective.** The loop kept a `train.py` change only when `inject_auroc` rose by more than a noise margin. `inject_auroc` is the mean AUROC for ranking corrupted copies of 300 held-out *normal* HDFS blocks above the clean originals, with the pre-registered rule `bpb_top10` and the end-of-session marker. It is computed by [`eval/objective.py`](../../eval/objective.py), which the agent doesn't write. See [`injection-objective.md`](injection-objective.md).
- **Bar.** Base model measured three times: 0.559, 0.545, 0.550 (mean 0.5514). Margin 2 × sd = 0.0113.
- **Run.** 120 experiments, 5 minutes of training each, on the Claude Code login. 4 kept, 116 rolled back, no crashes.
- **Test.** After the run, each kept commit was retrained once and scored on 10,000 labelled HDFS blocks (271 anomalous) that the loop never touched. Base and final commit were run twice.

Reproduce with `scripts/agent_inject_experiment.sh` (`prepare`, `baseline`, `start`, `evaluate`, `restore`).

## Result

| Commit | Change | Injected-fault AUROC (loop) | Real-fault AUROC (fixed rule) | Real-fault PR-AUC | val_bpb |
|--------|--------|-----------------------------|-------------------------------|-------------------|---------|
| `fdd9b57` base | n/a | 0.551 | 0.842, 0.842 | 0.240, 0.434 | 0.468, 0.469 |
| `6eb53fd` exp 1 | focal/anomaly-weighted training loss → plain cross-entropy | 0.578 | 0.911 | 0.521 | 0.553 |
| `62f644a` exp 4 | `MATRIX_LR` 0.02 → 0.04 | 0.742 | 0.922 | 0.602 | 0.635 |
| `3af4972` exp 5 | smaller batch, more optimizer steps | 0.766 | 0.957 | 0.591 | 0.576 |
| `d69869c` exp 76 | `MATRIX_LR` 0.04 → 0.06 | 0.796 | 0.958, 0.952 | 0.641, 0.625 | 0.631, 0.583 |

Two values in a cell are the two repeat runs.

- **Real-fault AUROC rose at every kept step**, 0.842 → 0.958. The two base runs and the two final runs agree to within 0.006, so the 0.11 gain is well outside run-to-run noise.
- **`val_bpb` got worse while detection got better.** Across the seven runs, Spearman ρ(val_bpb, real AUROC) = +0.75: higher bits-per-byte went with *better* detection. An agent optimising `val_bpb` on this corpus would have rejected all four changes.
- **The biggest real gain came from the smallest injected gain.** Removing the focal loss (exp 1) moved the injected score by 0.027 and the real score by 0.069. The learning-rate change (exp 4) moved the injected score by 0.164 and the real score by 0.011. The objective pointed the right way at each step, but its size is not a guide to the size of the real effect.

## How much to trust this

- **One dataset, one run, one training seed per commit.** Only the end points were repeated.
- **It closes a gap rather than setting a record.** The fixture trainer used for the published HDFS numbers already reaches 0.964 ± 0.020 with the same rule ([`end-of-session-marker.md`](end-of-session-marker.md)), and value rarity plus a too-short rule reaches 0.977. The base `train.py` was held back by the focal training loss it inherited (see the [`val_bpb` audit](../val-bpb-audit.md)); the agent's first kept change removed it.
- **Only the optimised rule improved.** Under other scoring rules the same models show no gain: max-event AUROC was 0.931 and 0.876 for the base, 0.926 and 0.888 for the final commit; per-field was 0.796 → 0.805. The loop optimised `bpb_top10` and that is what moved.
- **PR-AUC is noisy.** The base scored 0.240 and 0.434 on two runs, so the rise to about 0.63 is real in direction but its size is uncertain.
- **Kept changes are selected for a higher injected score by construction**, so "injected and real rise together" over five commits is weaker evidence than it looks. The stronger evidence is that the held-out real score rose at all, by an amount far outside noise.

## What it supports

An agent loop can improve real anomaly detection without incident labels when its objective is built from corrupted normal data and computed outside the code it edits. On the same corpus, the earlier run that optimised `val_bpb` produced [no genuine improvement in 120 experiments](../val-bpb-audit.md).

Raw data: [`reports/public-accuracy/agent-loop-inject-20261011/`](../../reports/public-accuracy/agent-loop-inject-20261011/).
