# In-domain lab eval: does the model beat a simple rule?

> **claim_status=`published`** (2026-09-25). This is a negative-leaning result, published on purpose.
> On this lab, an error-plus-latency rule beats the model, and the model adds nothing beyond that rule.

The zero-shot result in [`ranking-validation.md`](ranking-validation.md) trains on Uber CRISP and scores a lab it has never seen. That isn't the product pitch. The pitch is *"a model trained on your own telemetry"*. This eval tests the pitch directly: the model sees only the lab's own **normal** traffic, then ranks held-out sessions. It also compares against the baselines an SRE would actually write.

## Setup

| Item | Value |
|------|-------|
| Corpus | `lab/captures/pooled-20260918` (same as the zero-shot eval; content SHA-256 `effb9a3a…18e53f5`) |
| Split | Temporal, per capture. The **earlier half of normal sessions** trains the model (726 sessions). The later normals (730) plus **all incidents (1444)** form the eval set. No incident text is used for training or fitting |
| Model | BPE tokenizer (vocab 512) + `train.py` GPT, both fit on the 726 training normals only; 120 s per seed (~305 steps, MPS); seeds 0–4 |
| Scoring | Per-token surprise. **Trace/span/parent IDs and timestamps are masked out:** random hex is pure noise, and clock times inside eval windows never appear in training |
| Code | `eval/in_domain.py` (base commit `9900d44` plus this change) |

## Results

| Method | Kind | AUROC | PR-AUC |
|--------|------|-------|--------|
| Session length | baseline | 0.538 | 0.682 |
| Error-line count | baseline | 0.621 | 0.746 |
| Max duration z-score per (op, service) | baseline | 0.715 | 0.869 |
| **Rule: error lines, then duration z** | baseline | **0.776** | **0.896** |
| Session BPB, all tokens (old AOMB score) | model | 0.620 ± 0.011 | 0.765 ± 0.007 |
| Session BPB, IDs/timestamps masked | model | 0.679 ± 0.005 | 0.826 ± 0.005 |
| Max per-event BPB | model | 0.627 ± 0.002 | 0.757 ± 0.001 |
| **Top-10% token surprise** | model | **0.688 ± 0.003** | **0.834 ± 0.006** |

Per capture (within-capture AUROC):

| Capture / fault | Rule | Model (top-10%) | Zero-shot model |
|-----------------|------|-----------------|-----------------|
| `20260911T183259Z` latency | 0.891 | 0.696 | 0.464 |
| `…052312Z` latency | 0.849 | 0.687 | 0.502 |
| `…052510Z` errors | 0.670 | 0.628 | 0.627 |
| `…052553Z` latency + errors | 0.857 | 0.752 | 0.656 |
| `…052749Z` Redis killed | 0.712 | 0.678 | 0.590 |

Reports: [`reports/public-accuracy/lab-pooled-in-domain-20260925/`](../../reports/public-accuracy/lab-pooled-in-domain-20260925/). It holds per-seed metrics, every eval session's scores, and a per-token surprise heatmap.

## What this shows

1. **Training on your own normal traffic helps a lot.** AUROC rose from 0.583 (zero-shot) to 0.688, and latency faults went from invisible (~0.50) to detected (~0.69). This supports the "your own telemetry" framing over the zero-shot one.
2. **Masking IDs and timestamps matters.** Masking them alone takes the plain session score from 0.620 to 0.679. Random trace IDs cost a similar number of bits in every session and drown out the real signal.
3. **A simple rule still wins on this lab: 0.776 vs 0.688.** Every fault here is an error code or a slow span, which is exactly what the rule is built to catch.
4. **The model adds no information beyond the rule.** On the 1,456 eval sessions where the rule sees nothing (no error line, no duration beyond 3σ), the model scores **0.48 AUROC, which is chance**. Ranking by model and rule together (0.735) scores *worse* than the rule alone.

## What it means

This lab can't show the thing an Infrastructure Language Model is for. Its faults are all expressible as "error happened" or "span got slow", and a 5-line rule does that better. An ILM earns its keep only on anomalies that no one wrote a rule for:
- a log message never seen before, while the status is still `ok`
- a call sequence that changed (retry storms, a skipped cache, a new downstream dependency)
- a config change that alters payload shape but not latency

**Next experiment:** add fault types like these to `lab/` (status stays `ok`, latency stays normal, but behaviour changes), and rerun this exact eval. *Done: see [`rule-proof-eval.md`](rule-proof-eval.md).* If the model beats the rule there, that's the claim worth making. If it doesn't, the ILM framing needs rethinking.

## Looking at the data (heatmap findings)

- The most surprising tokens are genuine: Redis error messages the model never saw in normal traffic, spelled out character by character at 17–20 bits each.
- The "missed" incidents are frontend requests at 5–7 ms with `status=ok`, captured during a fault window but never touched by the fault. Window-level labels include sessions like these, so no detector can score perfectly here.
- An early attempt to bound the best achievable AUROC ("fraction of incident sessions with an error or an out-of-range duration") came out *below* what the model reached on the errors capture. Failed requests also change a trace's shape (the database and cache child spans disappear). The bound was withdrawn.

## Does a better model detect better? (training-length sweep)

This is the question the whole agent loop rests on: does lowering `val_bpb` improve detection? Seed 0, same split, trained for different lengths. "Held-out normal BPB" is the in-domain analogue of `val_bpb` (content tokens, eval-set normals).

| Train time | Steps | Held-out normal BPB | Incident BPB | AUROC (top-10%) |
|------------|-------|---------------------|--------------|-----------------|
| 10 s | 25 | 0.342 | 0.514 | 0.681 |
| 30 s | 75 | 0.280 | 0.488 | 0.689 |
| 60 s | 154 | **0.272** | 0.501 | 0.686 |
| 120 s | 307 | 0.289 | 0.519 | 0.686 |
| 300 s | 777 | 0.347 (overfit) | 0.648 | 0.681 |

**On this lab, better compression did not buy better ranking.** Held-out BPB improved by about 20% (0.342 to 0.272), and AUROC stayed flat at 0.68–0.69 from 25 steps onward. The simplest reading: once the model knows what an error or a slow span looks like, extra modelling of normal traffic doesn't help, because nothing else distinguishes these faults.

That's a warning for the overnight agent loop, which optimises `val_bpb` alone. Until an eval with rule-proof faults shows `val_bpb` and AUROC moving together, "lower `val_bpb` means a better detector" is a hypothesis, not a result. This sweep is a single seed on one small lab.

## Limitations

- **Faults were only half on.** A lab bug meant every capture in this pool had its fault active on only one of the API's two gunicorn workers, so about half of API requests. `docker-compose.yml` hardcoded `FAULT_MODE: none`, and the runtime switch reached one worker. That's a big part of why so many incident-window sessions look normal. It was fixed on 2026-09-25, and later captures (`20260925v2-*`) have the fault on every request.

Everything in [`ranking-validation.md` → Limitations](ranking-validation.md#limitations) applies: a small lab, scripted faults, and a raw capture that isn't public yet. In addition:
- Training uses only 726 short sessions; by 300 s the model is memorising them (held-out BPB rises again in the sweep above).
- The rule baseline is tuned only in the loosest sense (errors first, then z-score), and its 3σ threshold for the "rule sees nothing" subset was fixed before looking at model scores.
