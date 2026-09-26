# Lab ranking validation — zero-shot, labelled

> **claim_status=`published`** (2026-09-25). This is AOMB's first public labelled accuracy result.
> It's a **ranking** result from a small lab, not a production benchmark. Read the limitations before citing it.

## The claim

A model trained **only on Uber's public CRISP traces** had never seen the lab application. It still ranked the lab's fault-injected sessions above normal ones by session surprise (bits-per-byte):

| Method | AUROC | PR-AUC | precision@10 | precision@290 |
|--------|-------|--------|--------------|---------------|
| random ranking | 0.500 ± 0.010 | 0.498 ± 0.009 | 0.48 ± 0.15 | 0.49 |
| event-count baseline | 0.476 | 0.485 | 0.00 | 0.47 |
| length baseline | 0.543 | 0.528 | 0.00 | 0.70 |
| **session BPB (AOMB, zero-shot)** | **0.583 ± 0.007** | **0.631 ± 0.009** | **0.90 ± 0.00** | **0.90 ± 0.03** |

The table shows mean ± std over 5 seeds. The random row is the mean ± std of 64 shuffled rankings. The length and event-count baselines are deterministic.

> **Follow-up:** trained on the lab's own normal traffic instead, the model reaches 0.688, but a simple error-and-latency rule scores 0.776 on the same data. See [`in-domain-eval.md`](in-domain-eval.md).

**In plain terms:** the signal is real but modest. It beats every baseline across all 5 seeds, and 9 of the 10 most surprising sessions are genuine incidents. It's nowhere near a finished detector.

## Breakdown by fault

Within each capture, AUROC compares that run's incident sessions against **the same run's** normal sessions. This rules out day-to-day or load differences as the explanation.

| Capture | Fault | Normal | Incident | AUROC (mean ± std, 5 seeds) |
|---------|-------|--------|----------|------------------------------|
| `20260918T052553Z-both` | latency + errors | 324 | 321 | **0.656 ± 0.010** |
| `20260918T052510Z-errors` | errors (injected API failures) | 323 | 321 | **0.627 ± 0.008** |
| `20260918T052749Z-kill_redis` | Redis killed | 323 | 320 | **0.590 ± 0.009** |
| `20260918T052312Z-latency` | added API latency | 323 | 321 | 0.502 ± 0.007 |
| `20260911T183259Z` | added API latency | 163 | 161 | 0.464 ± 0.018 |

The model catches **error-type** failures: failed requests, dependency outages, and mixed faults. It **does not detect pure latency faults**, which score at chance (`inject_faults.sh` defaults: 800 ms added latency, 50% error rate). A latency fault changes only the `duration_ms` numbers, and a byte-level surprise score averaged over a whole session barely notices one number growing. That's the most obvious gap to work on.

## Method

| Item | Value |
|------|-------|
| Eval corpus | `lab/captures/pooled-20260918`: five fault-injection runs on the lab stack in [`lab/`](../../lab/) (frontend → API → Postgres + Redis, real OpenTelemetry) |
| Content SHA-256 | `effb9a3a70075d7e0681c3de6d4a9e236de8677b14dba3acd851a945b18e53f5` (traces + logs + provenance) |
| Labels | From capture windows in `provenance.json` (normal load vs fault active). There's no per-event tagging |
| Sessions | **2900** scored (1456 normal / 1444 incident); 7 with `unknown` label excluded |
| Train corpus | Uber CRISP, 500k spans (`crisp_zenodo_20260914T144906Z`; 20 train + 1 val shard, shard-set SHA-256 `d40d47a5…0e884`). **No lab data used in training** |
| Budget | 300 s train per seed (`TIME_BUDGET`), ~560 steps, then score every session |
| Seeds | `0, 1, 2, 3, 4` |
| Hardware | Mac Apple Silicon (MPS), macOS 26.6 |
| Code | Base commit `9900d44` plus the label-leak fix in `eval/labels.py` (merged with this write-up). `train.py` blob `bc736cf`, `prepare.py` blob `b71a0d4` (unchanged) |
| Harness | `eval.run_multiseed` → `eval.run_eval --scores-from model --train-seconds 300`; breakdown via `eval.lab_breakdown` |

Reports: [`reports/public-accuracy/lab-pooled-crisp-zeroshot-20260925/`](../../reports/public-accuracy/lab-pooled-crisp-zeroshot-20260925/). These include per-seed JSON with every session's score, the aggregates, the baselines and the breakdown.

To reproduce, you need the capture; see limitation 2.

```bash
# training cache = CRISP-500k shards + tokenizer (see docs/crisp-val-bpb-baseline.md)
uv run python -m eval.run_multiseed --capture lab/captures/pooled-20260918 \
  --seeds 0..4 --scores-from model --train-seconds 300 --out-dir <out>/model-seeds
uv run python -m eval.run_multiseed --capture lab/captures/pooled-20260918 \
  --seeds 0..4 --scores-from length --out-dir <out>/baselines-length
uv run python -m eval.lab_breakdown --capture lab/captures/pooled-20260918 \
  --seeds-dir <out>/model-seeds
```

## Correction: the earlier 0.766 is withdrawn

An earlier internal run (2026-09-18, `reports/public-accuracy/lab-pooled-20260918/`) reported **AUROC 0.766**. That number was inflated by a **label leak**. Each scored session began with a provenance line containing `window=incident fault=<mode>`. The model had only ever seen `window=normal` during training, so it found the label text itself surprising. The same line inflated the length baseline, from 0.543 to 0.643.

`eval/labels.py` now strips these lines before scoring, and a regression test makes sure they stay out. The 0.766 figure was never published and should not be cited.

## Limitations

1. **Faults were only half on.** A lab bug meant every capture in this pool had its fault active on only one of the API's two gunicorn workers, so about half of API requests. `docker-compose.yml` hardcoded `FAULT_MODE: none`, and the runtime switch reached one worker. That's a big part of why so many incident-window sessions look normal. It was fixed on 2026-09-25, and later captures (`20260925v2-*`) have the fault on every request.
2. **Small lab, scripted faults.** One three-service app with four fault types over a few minutes each. It isn't production traffic and it isn't multi-tenant. Results may not transfer.
3. **The capture isn't in the repo yet.** The raw traces and logs (~6 MB) stay on the operator's machine. The repo holds the content hash, provenance and per-session scores. A redacted sample is in [`corpus/fixtures/lab_public_pack_v0/`](../../corpus/fixtures/lab_public_pack_v0/).
4. **Some lab artifacts remain in the text.** About 20 sessions, spread over both classes, contain the fault-switch control calls (`POST /admin/fault`). The lab app's error message is the literal string `induced_fault_error`. In real incidents the error text would differ; the `status=error` signal itself is genuine.
5. **Short training.** It's one 5-minute training run per seed, not an overnight-bred model. Whether overnight breeding (lower CRISP `val_bpb`) improves ranking hasn't been measured.
6. **Session-level score only.** A session is scored by its mean surprise. Localising which event is anomalous, and per-field scoring (which could catch latency faults), are future work.
7. **Separate from `val_bpb`.** Don't compare or blend this with CRISP `val_bpb` (0.4309 / 0.407753), Tale (1.379520) or synthetic (0.3682). Those numbers measure training fitness, not detection.

Protocol: [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) · checklist: [`publish-checklist.md`](publish-checklist.md).
