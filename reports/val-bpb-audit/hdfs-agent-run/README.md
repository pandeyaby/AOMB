# HDFS agent-loop run (2026-09-29): evidence

The 120-experiment run that surfaced the `val_bpb` problem (see [`docs/val-bpb-audit.md`](../../../docs/val-bpb-audit.md)). The loop ran on a local experiment branch that was never pushed; these files are what's kept of it.

| File | Contents |
|------|----------|
| `experiments.csv` | All 120 experiments: reported `val_bpb`, training time, kept or rolled back. The bar was the base model's 0.3437 |
| `exp07_agent_rationale.txt` | The agent's own explanation for experiment 7, where it diagnoses the metric problem |
| `exp07_kept_change.patch` | The one change that was kept: `train.py`'s evaluation path returns plain cross-entropy |

Reported values before experiment 7 are focal-weighted; from experiment 8 on, the kept change made them true bits-per-byte.
