# In-domain lab eval: does the model beat a simple rule?

> **claim_status=`published`** (2026-09-25; **corrected 2026-09-26**, see [Corrections](#corrections); per-field score added 2026-09-27).
> On error and latency faults, the model with per-field scoring **matches** a 5-line rule (0.788 vs 0.776) with no rules written, but it adds little *beyond* that rule.

The zero-shot result in [`ranking-validation.md`](ranking-validation.md) trains on Uber CRISP and scores a lab it has never seen. That isn't the product pitch. The pitch is *"a model trained on your own telemetry"*. This eval tests the pitch directly: the model sees only the lab's own **normal** traffic, then ranks held-out sessions. It also compares against the baselines an SRE would actually write.

## Setup

| Item | Value |
|------|-------|
| Corpus | [`lab/published/pooled-20260918/`](../../lab/published/pooled-20260918/) (same as the zero-shot eval; content SHA-256 `effb9a3a…18e53f5`) |
| Split | Temporal, per capture. The **earlier half of normal sessions** trains the model (726 sessions). The later normals (730) plus **all incidents (1444)** form the eval set. No incident text is used for training or fitting |
| Model | BPE tokenizer (vocab 512) + `train.py` GPT, fit on the 726 training normals only. **The context is sized to the longest training session (768 tokens), so nothing is cropped.** 120 s per seed (~475 steps, MPS); seeds 0–4 |
| Scoring | Per-token surprise with **trace/span/parent IDs, timestamps and monotonic counters masked out.** Random hex is pure noise; clock-like values in later eval windows are always "new" |
| Code | `eval/in_domain.py` |

## Results

| Method | Kind | AUROC | PR-AUC |
|--------|------|-------|--------|
| Session length | baseline | 0.538 | 0.682 |
| Error-line count | baseline | 0.621 | 0.746 |
| Max duration z-score per (op, service) | baseline | 0.715 | 0.869 |
| **Rule: error lines, then duration z** | baseline | **0.776** | **0.896** |
| Template/shape novelty | baseline | 0.561 | 0.705 |
| Value novelty (unseen field value) | baseline | 0.623 | 0.747 |
| Session BPB, all tokens | model | 0.703 ± 0.016 | 0.842 ± 0.011 |
| Session BPB, IDs/timestamps/counters masked | model | 0.737 ± 0.009 | 0.873 ± 0.005 |
| Max per-event BPB (masked) | model | 0.744 ± 0.009 | 0.879 ± 0.006 |
| Top-10% token surprise (masked) | model | 0.732 ± 0.008 | 0.870 ± 0.005 |
| **Most surprising field value (per-field)** | model | **0.788 ± 0.003** | **0.905 ± 0.001** |

Per capture (within-capture AUROC):

| Capture / fault | Rule | Model (per-field) | Zero-shot model |
|-----------------|------|-------------------|-----------------|
| `20260911T183259Z` latency | **0.891** | 0.881 | 0.464 |
| `…052312Z` latency | **0.849** | 0.835 | 0.502 |
| `…052510Z` errors | 0.670 | **0.671** | 0.627 |
| `…052553Z` latency + errors | 0.857 | **0.868** | 0.656 |
| `…052749Z` Redis killed | 0.712 | **0.751** | 0.590 |

Reports: [`reports/public-accuracy/lab-pooled-in-domain-20260925/`](../../reports/public-accuracy/lab-pooled-in-domain-20260925/).

## What this shows

1. **Training on your own normal traffic helps a lot.** AUROC rose from 0.583 (zero-shot) to 0.788, and latency faults went from invisible (~0.50) to detected (~0.84–0.88).
2. **How you score matters as much as the model.** Averaging surprise over a session gives 0.737. Taking the single most surprising *field value* (a duration, a status, an error message) gives 0.788. A slow request is one surprising `duration_ms` value, and averaging buries it.
3. **Per-field scoring matches the rule, with no rules written: 0.788 vs 0.776.** It's within a hair of the rule on latency faults and slightly ahead on errors and the Redis outage.
4. **It adds little beyond the rule.** On the 1,456 eval sessions where the rule sees nothing (no error line, no duration beyond 3σ), the per-field model scores 0.582, a small signal above chance (the masked session score gets 0.528). Ranking by model and rule together scores 0.774, no better than the rule alone.

The per-field score was written up as the next experiment (in [`value-drift-eval.md`](value-drift-eval.md)) before it was run, so it isn't a scoring method picked after seeing results.

## Does a better model detect better? (training-length sweep)

This is the question the whole agent loop rests on: does lowering `val_bpb` improve detection? Seed 0, same split, trained for different lengths. "Held-out normal BPB" is the in-domain analogue of `val_bpb`.

| Train time | Steps | Held-out normal BPB | Incident BPB | AUROC (masked mean) | AUROC (max-event) |
|------------|-------|---------------------|--------------|---------------------|-------------------|
| 10 s | 39 | 0.223 | 0.431 | 0.724 | 0.686 |
| 30 s | 119 | 0.154 | 0.397 | 0.730 | 0.735 |
| 60 s | 233 | 0.127 | 0.399 | 0.733 | 0.739 |
| 120 s | 478 | **0.125** | 0.463 | **0.753** | **0.756** |
| 300 s | 1095 | 0.144 (overfit) | 0.541 | 0.745 | 0.755 |

**Better compression bought somewhat better detection.** Held-out BPB fell 44%, and AUROC rose from 0.72 to 0.75 before the model started to overfit. The effect is modest, but it goes the direction the agent loop assumes. It's one seed on one small lab. Summary: [`sweep-seed0/summary.json`](../../reports/public-accuracy/lab-pooled-in-domain-20260925/sweep-seed0/summary.json).

## Looking at the data (heatmap findings)

- The most surprising tokens are genuine: Redis error messages the model never saw in normal traffic.
- The "missed" incidents are mostly frontend requests at 5–7 ms with `status=ok`, captured during a fault window but never touched by the fault (partly because of the half-active-fault lab bug below).
- Three confounds turned up by looking at per-token surprise, and are now masked: random IDs, timestamps, and a Redis counter that only goes up. Each one makes later eval windows look "new" for reasons unrelated to faults.

## Corrections

The 2026-09-25 version of this page reported model AUROC **0.688** and said the training-length sweep showed **flat AUROC**. Both are superseded:
- **Training cropped every session at 256 tokens.** The packing dataloader crops long sessions, and checkout log lines start around token 430, so the model never learned them. The context is now sized to the longest training session, and the eval refuses to run if any session would be cropped.
- **A monotonic counter (`hits=`) was scored.** It rises across the day, so incident windows always carried unseen values. It's now masked with IDs and timestamps.

The conclusion that the model adds little beyond the rule is unchanged. The sweep conclusion flipped: it now shows modest improvement. On 2026-09-27 the per-field score was added and all model numbers were regenerated in one run; the per-field model now matches the rule.

## Limitations

- **Faults were only half on.** A lab bug meant every capture in this pool had its fault active on only one of the API's two gunicorn workers, so about half of API requests. `docker-compose.yml` hardcoded `FAULT_MODE: none`, and the runtime switch reached one worker. It was fixed on 2026-09-25, and later captures have the fault on every request.
- A small lab and scripted faults. The capture is public at [`lab/published/pooled-20260918/`](../../lab/published/pooled-20260918/), and `./scripts/reproduce_lab_evals.sh` reruns this eval.
- Training uses only 726 short sessions; by 300 s the model is memorising them (held-out BPB rises again).
- The rule baseline is tuned only in the loosest sense (errors first, then z-score), and the 3σ threshold for the "rule sees nothing" subset was fixed before looking at model scores.
