# `val_bpb` audit: the metric was not bits-per-byte

> **Published 2026-10-08.** From 2026-03-10 until this fix, every `val_bpb` this project reported was a **focal-weighted loss**, not true bits-per-byte. Re-measured properly, the "focal loss breakthrough" made the model slightly *worse*, and neither overnight run improved true bits-per-byte. **Those claims are withdrawn.** Detection results (AUROC) are not affected.

## What went wrong

`prepare.py` holds the fixed measuring stick, `evaluate_bpb`. It got its per-token losses by calling the model's own loss path:

```python
loss_flat = model(x, y, reduction='none')   # train.py decides what this returns
```

`train.py` is the one file the agent edits. On 2026-03-10, experiment 17 (`c2026cc`, "focal loss with anomaly token weighting") changed that path to return `(1 − p)^γ × cross-entropy`. The factor is at most 1, so from that commit on, `val_bpb` was pushed **below** true bits-per-byte, by more as γ grew. The agent loop then kept every change that lowered this number.

Nobody intended this. The loop was asked to minimise a number, and the number was computed by code the loop could change.

## How it was found

**The agent found it.** On 2026-09-29 the agent loop ran 120 experiments on LogHub HDFS, to test whether `val_bpb` improvements raise detection AUROC. In experiment 7 the agent (Claude, proposing changes to `train.py` as usual) wrote this before making any tuning change:

> "Before tuning anything, I found a problem in how the number is produced. `GPT.forward` applies focal loss (γ=3) and the anomaly and cascade weights to every call that passes `targets`. That includes the `reduction='none'` call the fixed `evaluate_bpb` makes. […] the reported val_bpb is a focal-weighted loss, not bits-per-byte."

It couldn't see `prepare.py` and said so. It proposed returning plain cross-entropy from the evaluation path, predicted the honest number would be *higher* and that its change would probably be reverted as "worse", and recommended re-measuring the bar first. On HDFS the true number happened to come out lower (0.3334 vs 0.3437), so the loop kept the change. It was the only change kept in 120 experiments, and the trained model was identical. No experiment improved the model.

The March run that introduced the distortion and the September run that caught it were the same loop with the same instructions. The evidence is in [`reports/val-bpb-audit/hdfs-agent-run/`](../reports/val-bpb-audit/hdfs-agent-run/): the agent's full rationale, the 26-line patch, and all 120 results.

## The fix

`evaluate_bpb` now calls `model(x)` for raw logits and computes cross-entropy itself, so nothing in `train.py` can redefine the metric. `tests/test_evaluate_bpb_integrity.py` checks this with a model whose loss path returns zeros: the evaluator must still report the true value and must never call the loss path.

## Re-measuring the record

Each run below takes a historic commit's `train.py`, **unmodified**, trains it for the standard 5-minute budget on the corpus it was originally measured on, and measures both numbers **on the same trained model** (`eval/val_bpb_audit.py`). The synthetic corpus was regenerated from the March generator; surviving March shards match byte for byte.

| Run | Commit | Reported at the time | Reported-style, re-measured | **True bits-per-byte** |
|-----|--------|----------------------|------------------------------|------------------------|
| Synthetic, exp 16 (last pre-focal) | `dc4df4a` | 0.4297 | 0.4297 | **0.4297** |
| Synthetic, exp 17 ("focal breakthrough") | `c2026cc` | 0.3950 | 0.3951 | **0.4358** |
| Synthetic, best of 120 experiments | `983ee44` | 0.3682 | 0.3685 | **0.4483** |
| CRISP 200k, pre-overnight base | `d386d09` | 0.4588 ¹ | 0.4442 | **0.5211** |
| CRISP 200k, overnight best | `73b1645` | 0.4309 | 0.4289 | **0.5206** |
| CRISP 500k, single run | `dd0280c` | 0.4078 | 0.3937 | **0.4665** |
| Tale of Errors 200k, single run | `7083fc1` | 1.3795 | 1.3718 | **1.4557** |

The re-measured reported-style numbers reproduce the originals to within about 0.4% on the synthetic corpus and CRISP 200k best, which shows the audit is measuring the same thing the project measured then.

¹ The README's 0.4588 starting point was a separate single run; the commit recorded for it couldn't be pinned exactly, so the re-measured base (0.4442) differs by 3%. The true-bits-per-byte comparison uses the same two commits either way.

## What this changes

| Claim | Status |
|-------|--------|
| "Focal loss breakthrough" at experiment 17 (0.4297 → 0.3950) | **Withdrawn.** True bits-per-byte went from 0.4297 to 0.4358, slightly worse |
| Synthetic: "0.4372 → 0.3682, a 15.8% improvement over 120 experiments" | **Withdrawn.** The final model is at 0.4483, about 2.5% *worse* than the 0.4372 baseline. The loop's real best was experiment 16 at 0.4297, a 1.7% improvement |
| CRISP: "0.4588 → 0.4309, −6.1% over 20 overnight experiments" | **Withdrawn.** True bits-per-byte went from 0.5211 to 0.5206: no change |
| CRISP 500k `0.4078`, Tale `1.3795` | **Corrected** to 0.4665 and 1.4557 |
| LogHub HDFS overnight run (120 experiments) | No model improvement; see above |
| Lab, HDFS and RCAEval detection results (AUROC, PR-AUC, F1) | **Unaffected.** They compute surprise from logits and never used the model's loss path |
| Lab training-length sweep ("held-out BPB −44%, AUROC 0.72 → 0.75") | **Unaffected**, for the same reason |

## What it means for the project

The overnight agent has not yet been shown to improve a model on any honest metric beyond the first 16 synthetic experiments (1.7%). After the metric drifted, a hundred further experiments made the real number worse while the reported number improved. That's the textbook failure of optimising a measure the optimiser can influence.

The loop deserves credit for one thing: given the same instructions six months later, it noticed the distortion, explained it, and fixed it against its own short-term interest. The fix in `prepare.py` makes that vigilance unnecessary.

## Reproduce

```bash
# cache must hold the corpus the commit was measured on (see docs/crisp-val-bpb-baseline.md)
uv run python -m eval.val_bpb_audit --sha c2026cc --label "exp 17" --out /tmp/exp17.json
```

Raw results: [`reports/val-bpb-audit/`](../reports/val-bpb-audit/).
