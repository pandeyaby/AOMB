# Lab pooled, CRISP zero-shot (2026-09-25): raw harness output

This folder is the raw output behind the **published** result in [`docs/lab/ranking-validation.md`](../../../docs/lab/ranking-validation.md).

The harness stamps every report `claim_status=not_published` by default. The publish decision is recorded in that doc and its checklist, not in these files.

| Path | Contents |
|------|----------|
| `model-seeds/seed-{0..4}/report.json` | Per-seed metrics, provenance, and **every session's score** (2900 rows) |
| `model-seeds/aggregate.{json,md}` | Mean ± std across seeds |
| `model-seeds/breakdown.{json,md}` | Within-capture (per-fault) AUROC from `eval.lab_breakdown` |
| `baselines-length/`, `baselines-events/` | Deterministic baselines (session rows truncated to 5; rerun to regenerate) |

The random ranking baseline (64 draws) is inside each `report.json` under `metrics.random_baseline`.
