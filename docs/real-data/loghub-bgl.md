# LogHub BGL: the pre-registered test

> **claim_status=`published`** (2026-10-09).
> **The model lost.** With the scoring rule fixed in advance, it scored AUROC 0.814 on BGL. A plain count of high-severity lines scored 0.925.

## What was tested

The scoring rule was chosen from six earlier datasets and committed before any model was run on BGL; see [`scoring-rule-preregistration.md`](scoring-rule-preregistration.md), whose git history is the record. The rule is **`bpb_top10`**: the mean surprise of the 10% most surprising tokens in a session, with the end-of-session marker on and IDs, timestamps and counters masked.

| Item | Value |
|------|-------|
| Data | [LogHub BGL](https://github.com/logpai/loghub) (Oliner & Stearley, DSN 2007; CC BY 4.0). 4.75M log lines from a Blue Gene/L supercomputer; each line labelled normal or alert |
| Session | 20 consecutive lines in chronological order; an incident if any line is an alert |
| Split | Train on the first 5,000 all-normal windows; score 10,000 windows sampled (seed 0) from later ones: 9,137 normal, 863 incident |
| Normalised | Node ids, hex addresses, IPs, runs of 4+ digits. The label column never enters the text. Severity levels are kept |
| Model | 120 s × 5 seeds, ≤24-line training chunks, none cropped |

Rebuild: `uv run python -m corpus.ingest.loghub_bgl --input <BGL dir> --out <file>` (expected `sessions_sha256` `e70abf32…e99f09`), then `uv run python -m eval.in_domain --sessions <file> --seeds 0..4 --train-seconds 120 --train-chunk-lines 24 --end-marker --out-dir <out>`.

## Result

| Method | Kind | AUROC | PR-AUC |
|--------|------|-------|--------|
| Session length | baseline | 0.624 | 0.117 |
| Template/shape novelty | baseline | 0.733 | 0.155 |
| Value rarity | baseline | 0.890 | 0.326 |
| **Severity count** (lines at ERROR / FATAL / SEVERE / FAILURE) | baseline | 0.925 | 0.386 |
| **Severity count + novelty** | baseline | **0.932** | **0.427** |
| **Model, pre-registered rule (`bpb_top10`)** | model | **0.814 ± 0.019** | **0.203 ± 0.025** |

Every other candidate, as promised (none would have done better):

| Candidate | AUROC | PR-AUC |
|-----------|-------|--------|
| `bpb_top10` (selected) | 0.814 ± 0.019 | 0.203 ± 0.025 |
| `bpb_content` | 0.800 ± 0.003 | 0.177 ± 0.003 |
| `combo_content_end` | 0.784 ± 0.072 | 0.178 ± 0.043 |
| `bpb_max_event` | 0.779 ± 0.009 | 0.160 ± 0.005 |
| `bits_end` | 0.732 ± 0.119 | 0.160 ± 0.052 |
| `combo_field_end` | 0.722 ± 0.115 | 0.160 ± 0.061 |
| `bits_max_field` | 0.689 ± 0.048 | 0.126 ± 0.015 |

## Why it lost

Looking at the eval windows:

- **Alerts are bursts of high-severity lines.** An incident window has a median of 20 severe lines out of 20. 85% of normal windows have none. A severity count separates those almost by definition.
- **Windows are highly repetitive in both classes.** In the median window, normal or incident, a single message fills all 20 lines. A language model finds a repeated line predictable after its first occurrence, so a burst of twenty identical `FATAL` lines costs it little more than one. Surprise rewards novelty, not volume, and BGL's alerts are mostly volume.
- **The model still ranks incidents fairly high** (median percentile 0.84–0.88), but normal windows with unfamiliar benign messages rank high too. Training covers the first 5,000 windows of a log spanning months, and only 184 of them contain any `FATAL` line.

## What it means

The selection procedure worked as intended: it produced a number nobody could have tuned. And the number says a model fixed in advance does **not** generalise to a new log dataset well enough to beat a one-line severity rule.

Taken with the six development views, where the same fixed rule wins one, is within 0.03 on three, and loses two, the supported claim is narrow. The model is useful where faults change *structure or content without changing severity* (the rule-proof lab faults). Where severity, error codes or latency already carry the signal, simple checks are better.

Reports: [`reports/public-accuracy/loghub-bgl-20261009/`](../../reports/public-accuracy/loghub-bgl-20261009/).
