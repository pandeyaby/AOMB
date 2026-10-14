# Agent run on the injected-fault objective (2026-10-10): evidence

The 120-experiment run described in [`docs/real-data/agent-inject-run.md`](../../../../docs/real-data/agent-inject-run.md). The loop ran on a local experiment branch that was never pushed; these files are what's kept of it.

| File | Contents |
|------|----------|
| `experiments.csv` | All 120 experiments: `inject_auroc` and `val_bpb` as the loop measured them, kept or rolled back |
| `baseline_runs.txt` | The three base-model runs that set the bar (mean 0.5514) and the noise margin (0.0113) |
| `expNNN_kept_change.patch` | Each of the four kept changes to `train.py`, as a diff against the previous kept state |
| `expNNN_agent_rationale.txt` | The agent's own explanation for each kept change |
| `all_kept_changes.patch` | All four together, as a diff against `train.py` at `fdd9b57` |

`train.py` on `main` does not include these changes: they were tuned on HDFS only. To get the final model's training script, run `git apply reports/public-accuracy/agent-loop-inject-20261011/evidence/all_kept_changes.patch` on a checkout where `train.py` is unchanged since `fdd9b57`.
