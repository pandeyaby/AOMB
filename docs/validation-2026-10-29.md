# Independent validation, 2026-10-29

> Every published result was re-measured end to end by separate agents that were given the procedures and the code but not the published numbers. **All of them reproduced.** The review also found a broken baseline on RCAEval Online Boutique, two ways the agent loop's score could have been faked, and several overstatements. This page records what was checked, what came back, and what changed.

## How it was run

Five Claude Code subagents, each working only from the repository at `b5538b7` and the raw datasets:

| Validator | Task |
|-----------|------|
| Lab | Rerun the three lab pools (5 seeds × 120 s, end marker) and the checkout views |
| HDFS + BGL | Rebuild both session files from the raw LogHub logs, rerun both evaluations, check the ingest for label leakage |
| RCAEval | Rebuild Online Boutique and Train Ticket from the raw RE3 data, rerun both, check for leakage and confounds |
| Agent run | Apply the agent's four kept `train.py` changes to `fdd9b57`, retrain base and patched three times each, score the labelled HDFS blocks |
| Methodology audit | Adversarial code review: leakage, contamination, baseline fairness, metric correctness, selection effects, objective tamper resistance |

They were told not to open the README, `docs/`, `reports/`, commit messages or the project brief, and to record their own numbers without comparing. Training is time-budgeted, so runs took turns behind a shared lock.

**Blinding was partial.** The agents' environment loaded the maintainer's private project notes, which contain earlier published numbers, and each validator said so. The reproductions are still meaningful because their numbers are produced by code the validators ran, not by judgement; the audit's findings carry file and line evidence and reproducer scripts.

## Results

AUROC of the pre-registered rule `bpb_top10` (mean ± sd over seeds) and of the best hand-built check, published vs re-measured:

| View | Published model | Re-measured model | Best check, published → re-measured | Verdict |
|------|-----------------|-------------------|-------------------------------------|---------|
| Lab, errors & latency | 0.738 ± 0.005 | 0.734 ± 0.009 | 0.776 → 0.776 | Reproduced |
| Lab, rule-proof (checkout) | 0.931 ± 0.026 | 0.930 ± 0.028 | 0.890 → 0.890 | Reproduced |
| Lab, value drift (checkout) | 0.712 ± 0.019 | 0.722 ± 0.021 | 0.833 → 0.833 | Reproduced |
| LogHub HDFS | 0.964 ± 0.020 | 0.966 ± 0.012 | 0.977 → 0.977 | Reproduced |
| LogHub BGL | 0.814 ± 0.019 | 0.829 ± 0.021 | 0.932 → 0.932 | Reproduced (within 1 sd) |
| RCAEval Online Boutique (faulted service) | 0.927 ± 0.036 | 0.927 ± 0.024 | 0.952 → 0.952 | Reproduced; **baseline broken** (below) |
| RCAEval Train Ticket (faulted service) | 0.860 ± 0.013 | 0.859 ± 0.019 | 0.882 → 0.882 | Reproduced |
| Agent run, base → kept changes (HDFS) | 0.842 → 0.958 | 0.856 ± 0.005 → 0.955 ± 0.026 | n/a | Reproduced (+0.10; ranges don't overlap) |

All four rebuilt session files (HDFS, BGL, Online Boutique, Train Ticket) were byte-identical to the published ones. The lab capture hashes matched. No label text reached any scored session; every split was temporal and disjoint; AUROC and PR-AUC matched scikit-learn to 1e-16, including heavy ties. `uv run pytest`: 484 passed, 3 skipped.

## What changed

### Fixed in code

| Finding | Severity | Fix |
|---------|----------|-----|
| The agent loop parsed the last score line from stdout **and stderr** and never checked the exit code, so a run that crashed after printing a score, or printed one to stderr, could be kept | Critical (loop) | `agent_loop.run_training` only accepts a score from a run that exits 0 within the time limit, read from stdout |
| `eval/objective.py` ran the agent's `train.py` in the same process as the scorer, so `train.py` could replace the scoring functions (a reproducer got 1.0 with no model) | Critical (loop) | `train.py` now runs in a child process that receives an unlabelled, shuffled list of texts and returns only per-token surprise. The parent tokenizes each text itself, checks one finite value per token, applies the rule and computes the AUROCs, and drops any score line the child prints. `tests/test_objective_isolation.py` |
| The error rule only matched `status=error`, so it never fired on Online Boutique, whose failures carry gRPC status codes | Major (baseline) | `_ERROR` also matches HTTP 4xx/5xx and non-zero gRPC codes. Only Online Boutique changes: its best check goes from 0.952 to **0.998** on the faulted-service traces. Reports recomputed without retraining |
| Spearman correlation in `eval/agent_commits.py` didn't average ties | Minor | Average ranks; the published correlation (0.75) has no ties and is unchanged |

Neither critical hole affected a published result: the agent's kept changes only touch the loss and hyperparameters, and the real-data numbers were measured outside the loop.

### Corrected in the write-ups

- **RCAEval Online Boutique:** the model is clearly behind a failed-status count (0.927 vs 0.998), not "slightly behind a structural check". Its faulted-service view is also mostly one fault: only 5 of 30 cases have both classes in it. ([rcaeval-re3.md](real-data/rcaeval-re3.md))
- **RCAEval Train Ticket:** pooling across cases inflates the result because traffic shifts after a fault. Averaged within each case, the model is slightly ahead (0.779 ± 0.005 vs 0.756), the reverse of the pooled order.
- **Agent run:** the kept changes made max-event scoring **worse** (AUROC 0.941 → 0.918, PR-AUC 0.853 → 0.671), not flat, and made training about 5× noisier between runs. ([agent-inject-run.md](real-data/agent-inject-run.md))
- **Pre-registration:** the BGL "rule fixed first" order rests on commit history (BGL ingest committed 11 minutes before the selection), and HDFS helped choose the rule that later judged the agent run. ([scoring-rule-preregistration.md](real-data/scoring-rule-preregistration.md))
- **HDFS:** 1,304 eval blocks are longer than the model's context and are scored in windows. ([loghub-hdfs.md](real-data/loghub-hdfs.md))

### Open, not fixed here

- The objective's 300 clean sessions are also the validation shard, which `train.py` could train on.
- The loop compares about 120 tries against a running best with a margin from three base runs and a fixed training seed, which biases the best score upward.
- In the `val_bpb` objective, `train.py` prints its own `val_bpb`, so nothing forces that number to come from `prepare.evaluate_bpb`.
- In the lab and RCAEval, every incident comes after every eval normal in time. A time-shift test on the baselines came back at chance (0.47–0.56), but the model itself wasn't tested this way.
- Model sessions scoring NaN are dropped, so a model variant can be scored on fewer sessions than the baselines.
- The end-of-session score on its own varies a lot between seeds (AUROC sd 0.11–0.15 on HDFS).

## Evidence

Each validator's report and machine-readable results: [`reports/validation-20261029/`](../reports/validation-20261029/). The audit's reproducers (`repro_forge.py`, `repro_monkeypatch.py`) target the code at `b5538b7`, before these fixes. The per-case RCAEval numbers come from `reports/validation-20261029/rcaeval/per_case_auroc.py`.
