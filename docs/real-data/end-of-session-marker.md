# End-of-session marker

> **claim_status=`published`** (2026-10-09).
> Adding an explicit end-of-session line fixes the model's most consistent blind spot: sessions that stop early. It lifts HDFS from 0.851 to 0.964 AUROC and changes nothing elsewhere. On HDFS, a hand-built check with one added "too short" rule does just as well (0.977).

## The problem

Next-token surprise scores what a session *contains*. A request or block that simply stops produces nothing surprising. That showed up three times: the lab's missing-call fault, HDFS blocks that end after two lines, and RCAEval traces cut short by code-level faults.

## The change

With `--end-marker`, every training session ends with a line `[end_of_session]`, and every scored session gets the same line appended. The model learns where sessions normally end, so a marker arriving early is surprising. `bits_end` reports that surprise on its own; the other scores include it.

## Effect on the fixed scoring rule

`bpb_top10`, the rule selected in [`scoring-rule-preregistration.md`](scoring-rule-preregistration.md), AUROC over seeds:

| View | Without marker | With marker | Best hand-built check |
|------|----------------|-------------|------------------------|
| Lab, error/latency | 0.732 ± 0.008 | 0.738 ± 0.005 | **0.776** rule |
| Lab, rule-proof (checkout) | 0.933 ± 0.017 | **0.931 ± 0.026** | 0.890 rule + novelty |
| Lab, value drift (checkout) | 0.715 ± 0.040 | 0.712 ± 0.019 | **0.833** value pair |
| HDFS | 0.851 ± 0.010 | 0.964 ± 0.020 | **0.977** rarity or too-short |
| RCAEval Online Boutique (faulted service) | 0.928 ± 0.027 | 0.927 ± 0.036 | **0.952** rule + novelty |
| RCAEval Train Ticket (faulted service) | 0.868 ± 0.005 | 0.860 ± 0.013 | **0.882** rule + novelty |

The marker helps where sessions stop early (HDFS) and is neutral elsewhere. The marker score alone separates the lab's missing-call fault perfectly in a one-seed check (1.000 on checkout sessions).

## What moved on HDFS, and the baseline it exposed

The whole HDFS gain comes from 42 anomalous blocks of 1–3 lines, which moved from the bottom of the ranking (median percentile 0.13) to near the top (0.92). Longer anomalies were already ranked at 0.98–1.00 and didn't move.

Every normal block in the eval set has 13 or more lines. So those short anomalies are also caught by a trivial rule: *flag a session with fewer than half as many lines as the shortest training session.* That rule wasn't among the baselines, because the existing length baseline only treats longer sessions as suspicious. It is now (`too_short`, and `rarity_or_short` combined with value rarity):

| HDFS | AUROC | PR-AUC |
|------|-------|--------|
| Value rarity | 0.822 | 0.605 |
| **Value rarity + too-short rule** | **0.977** | **0.833** |
| Model + marker, session mean | 0.974 ± 0.006 | 0.639 ± 0.054 |
| Model + marker, max event | 0.963 ± 0.015 | 0.785 ± 0.040 |
| Model + marker, fixed rule (`bpb_top10`) | 0.964 ± 0.020 | 0.621 ± 0.063 |

So on HDFS the model with the marker **matches** a two-rule hand-built check; it doesn't beat it. The earlier published HDFS comparison (model 0.878 vs best check 0.822) lacked the too-short rule. The too-short rule was added after looking at this data, which favours the baseline, so read its 0.977 as an upper bound for simple checks. It's at chance (0.500) on the other five views.

Reports: the `*-eos-20261009` folders in [`reports/public-accuracy/`](../../reports/public-accuracy/).
