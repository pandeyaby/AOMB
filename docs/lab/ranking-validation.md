# Lab pooled ranking validation

> **claim_status=`not_published`**
>
> Not a public accuracy claim, product benchmark, or press number.
> Do not market these figures until the checklist in
> [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) passes **and** Abhinav
> explicitly greenlights claim language.
>
> GRAX: HOLD merge / public hero tables until re-skim + Abhinav yes.

## What this is

Frozen multi-seed **ranking** eval on **pooled lab** OpenTelemetry captures
(fault-injected org stack). Unit of scoring: session-level next-token surprise / BPB.
Higher surprise should rank incident windows above normal.

This is **lab evidence** for the method — not a bake-off against commercial APM, not
production multi-tenant traffic, and not interchangeable with CRISP / synthetic `val_bpb`.

## Method (honest)

| Item | Value |
|------|--------|
| Pool id | `lab/captures/pooled-20260918` (local; **not vendored**) |
| Sources | prior capture `20260911T183259Z` + latency / errors / both / kill_redis (2026-09-18) |
| Scorable sessions | **2900** (1456 normal / 1444 incident; 7 unknown excluded) |
| Seeds | `0..4` (5 seeds) |
| Model path | train-then-score **300s** / seed (`TIME_BUDGET` from `prepare.py`) |
| Hardware | Mac Apple Silicon (MPS) |
| Baselines | length; event-count; random ranking (in per-seed JSON) |
| Harness | `eval/run_multiseed.py` — `prepare.evaluate_bpb` **unchanged** |

Raw aggregates / seed reports (truncated session rows):
[`reports/public-accuracy/lab-pooled-20260918/`](../../reports/public-accuracy/lab-pooled-20260918/).
Summary twin: [`reports/public-accuracy/LAB_POOLED_VALIDATION.md`](../../reports/public-accuracy/LAB_POOLED_VALIDATION.md).

## Results (mean ± std over 5 seeds)

| Method | AUROC | PR-AUC | precision@10 | precision@290 |
|--------|-------|--------|--------------|---------------|
| length baseline | 0.6428 ± 0.0000 | 0.6159 ± 0.0000 | 0.0000 ± 0.0000 | 0.8621 ± 0.0000 |
| event-count baseline | 0.4762 ± 0.0000 | 0.4849 ± 0.0000 | 0.0000 ± 0.0000 | 0.4655 ± 0.0000 |
| session BPB (model) | 0.7660 ± 0.0115 | 0.8142 ± 0.0124 | 0.9000 ± 0.0000 | 0.9676 ± 0.0111 |

Session BPB beats length and event-count baselines on AUROC / PR-AUC / precision@k **on this lab pool**.

## Limitations (read before citing)

1. **Lab only** — scripted faults on a local multi-service stack, not live SpaceX / customer prod.
2. **Captures not public** — JSONL traces/logs stay local; repo has provenance pointer + reports only.
3. **Not a public accuracy claim** — checklist still gates marketing / README hero numbers.
4. **Do not blend** with CRISP `val_bpb` (0.407753 / 0.4309 / 0.458756) or synthetic `0.3682`.
5. **Single TIME_BUDGET** train-then-score per seed — not a long overnight breeding claim.
6. Precision@k depends on pool size (`k=10` and `k≈10%` of n); do not cherry-pick one seed.

## Protocol link

Pass/fail gate: [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md).
