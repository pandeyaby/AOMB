# Injected faults as a label-free objective

> **claim_status=`published`** (2026-10-09). A validation, not a detection result.
> Across models of varying quality, how well a model detects *injected* faults tracks how well it detects *real* ones (Spearman 0.72 and 0.79 on two datasets), and tracks it better than bits-per-byte does. The sample is small: 10 models per dataset.

## Why

The agent loop optimises `val_bpb`, how well the model predicts normal traffic. The [audit](../val-bpb-audit.md) showed the loop has produced no real gain on that metric, and the link from bits-per-byte to detection is weak. A better target would reward detection directly, but real incident labels are scarce.

`eval/inject.py` builds a detection test from normal sessions alone. Each clean session gets corrupted copies, one per corruption type, and the model is scored on ranking corrupted copies above clean ones:

| Corruption | Mimics |
|------------|--------|
| `value_swap` | a field takes another session's value (value drift) |
| `value_cross` | a field takes a value never seen in that field |
| `drop_line` | a missing call |
| `dup_line` | a retry |
| `truncate` | a session that stops early |
| `reorder` | a changed call order |
| `latency` | a duration multiplied by 10–100 |

IDs, timestamps and counters are never edited. Everything is deterministic given the seed.

## The question

Is this worth optimising? Only if a model that scores better on injected faults also scores better on real ones. If it doesn't, the agent would be chasing another number that doesn't matter.

## The test

On two datasets with real labels, train models of deliberately varying quality (5, 15, 40, 120 and 300 seconds, two seeds each: 10 models per dataset). For each model, measure on the same eval set:
- **injected-fault AUROC**: mean over corruption types, 300 clean eval normals each;
- **real-fault AUROC**: the labelled incidents;
- **bits-per-byte on held-out normals**: the in-domain analogue of `val_bpb`.

All three use the pre-registered scoring rule (`bpb_top10`) with the end-of-session marker.

## Result

Spearman rank correlation with real-fault AUROC across the 10 models:

| Dataset | Injected-fault AUROC | Bits-per-byte (lower is better) |
|---------|----------------------|----------------------------------|
| Lab, rule-proof faults (checkout sessions) | **+0.72** | −0.45 |
| LogHub HDFS | **+0.79** | −0.71 |

On both datasets the injected-fault score ranks models closer to their real detection ability than bits-per-byte does.

Per corruption type, the correlation with real-fault AUROC:

| Corruption | Lab rule-proof | HDFS |
|------------|----------------|------|
| `truncate` | +0.90 | +0.93 |
| `dup_line` | +0.78 | +0.82 |
| `reorder` | +0.48 | +0.78 |
| `drop_line` | +0.85 | −0.48 |
| `value_swap` | −0.24 | +0.76 |
| `value_cross` | −0.58 | +0.64 |
| `latency` | −0.61 | n/a (no durations) |

`truncate` and `dup_line` track real detection on both datasets. The value corruptions track it on HDFS but go the wrong way on the lab, and `drop_line` does the reverse. So the mean is a better guide than any single type, and the mix of corruptions probably needs to match the kind of faults a system actually has.

## How much to trust this

- **Small sample.** Ten models per dataset; a correlation of 0.7 from ten points is suggestive, not established.
- **Most of the spread comes from training length.** Under-trained models are bad at everything, which makes every sensible metric correlate. Among the six better-trained HDFS models (40 s and up) the top injected score and the top real score belong to the same model, but six points can't carry a claim.
- **The injected faults are harder to detect than the real ones** on HDFS (AUROC 0.49–0.68 against 0.87–0.98), because they're single small edits. The two scales differ; only the ranking of models is compared.

## Next

This is enough to justify one overnight run with the agent optimising injected-fault AUROC rather than `val_bpb`, computed by code the agent can't edit, with real-fault AUROC measured afterwards on held-out labels. It isn't enough to claim the objective works.

Raw data: [`reports/public-accuracy/injection-validation-20261009/sweep.json`](../../reports/public-accuracy/injection-validation-20261009/sweep.json).
