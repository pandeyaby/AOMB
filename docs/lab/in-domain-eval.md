# In-domain lab eval: does the model beat a simple rule?

> **claim_status=`published`** (2026-09-25; **corrected 2026-09-26**, see [Corrections](#corrections)).
> On error and latency faults, a 5-line rule still beats the model (0.776 vs 0.740), and the model adds little beyond it.

The zero-shot result in [`ranking-validation.md`](ranking-validation.md) trains on Uber CRISP and scores a lab it has never seen. That isn't the product pitch. The pitch is *"a model trained on your own telemetry"*. This eval tests the pitch directly: the model sees only the lab's own **normal** traffic, then ranks held-out sessions. It also compares against the baselines an SRE would actually write.

## Setup

| Item | Value |
|------|-------|
| Corpus | [`lab/published/pooled-20260918/`](../../lab/published/pooled-20260918/) (same as the zero-shot eval; content SHA-256 `effb9a3a…18e53f5`) |
| Split | Temporal, per capture. The **earlier half of normal sessions** trains the model (726 sessions). The later normals (730) plus **all incidents (1444)** form the eval set. No incident text is used for training or fitting |
| Model | BPE tokenizer (vocab 512) + `train.py` GPT, fit on the 726 training normals only. **The context is sized to the longest training session (768 tokens), so nothing is cropped.** 120 s per seed (~478 steps, MPS); seeds 0–4 |
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
| Session BPB, all tokens | model | 0.704 ± 0.009 | 0.843 ± 0.011 |
| Session BPB, IDs/timestamps/counters masked | model | 0.735 ± 0.015 | 0.871 ± 0.009 |
| **Max per-event BPB (masked)** | model | **0.740 ± 0.012** | **0.876 ± 0.008** |
| Top-10% token surprise (masked) | model | 0.730 ± 0.016 | 0.868 ± 0.009 |

Per capture (within-capture AUROC):

| Capture / fault | Rule | Model (max-event) | Zero-shot model |
|-----------------|------|-------------------|-----------------|
| `20260911T183259Z` latency | **0.891** | 0.799 | 0.464 |
| `…052312Z` latency | **0.849** | 0.769 | 0.502 |
| `…052510Z` errors | **0.670** | 0.638 | 0.627 |
| `…052553Z` latency + errors | **0.857** | 0.796 | 0.656 |
| `…052749Z` Redis killed | 0.712 | **0.734** | 0.590 |

Reports: [`reports/public-accuracy/lab-pooled-in-domain-20260925/`](../../reports/public-accuracy/lab-pooled-in-domain-20260925/).

## What this shows

1. **Training on your own normal traffic helps a lot.** AUROC rose from 0.583 (zero-shot) to 0.740, and latency faults went from invisible (~0.50) to detected (~0.77–0.80).
2. **A simple rule still wins on these faults: 0.776 vs 0.740.** Every fault here is an error code or a slow span, which is exactly what the rule is built to catch. The model wins only on the Redis outage, where new error messages appear.
3. **The model adds little beyond the rule.** On the 1,456 eval sessions where the rule sees nothing (no error line, no duration beyond 3σ), the model scores **0.51–0.52**, close to chance.

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

The conclusions that the rule wins on these faults and the model adds little beyond it are unchanged. The sweep conclusion flipped: it now shows modest improvement.

## Limitations

- **Faults were only half on.** A lab bug meant every capture in this pool had its fault active on only one of the API's two gunicorn workers, so about half of API requests. `docker-compose.yml` hardcoded `FAULT_MODE: none`, and the runtime switch reached one worker. It was fixed on 2026-09-25, and later captures have the fault on every request.
- A small lab and scripted faults. The capture is public at [`lab/published/pooled-20260918/`](../../lab/published/pooled-20260918/), and `./scripts/reproduce_lab_evals.sh` reruns this eval.
- Training uses only 726 short sessions; by 300 s the model is memorising them (held-out BPB rises again).
- The rule baseline is tuned only in the loosest sense (errors first, then z-score), and the 3σ threshold for the "rule sees nothing" subset was fixed before looking at model scores.
