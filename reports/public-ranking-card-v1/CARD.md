# Public ranking card v1 — fixture report

**Card id:** `public_ranking_card_v1`  
**claim_status:** `not_published`  
**Protocol:** [`docs/public-ranking-card-v1.md`](../../docs/public-ranking-card-v1.md)

> Fixture baselines only (unless a model section is present).  
> **Not** private lab-pool AUROC. **Not** CRISP `val_bpb`.  
> Do not publish until checklist + Abhinav greenlight.

## Identity

| Field | Value |
|-------|-------|
| Fixture | `corpus/fixtures/public_ranking_card_v1` |
| Fixture content SHA-256 | `7dae2e232f78276026eb067fc3d2bc10a695764d019ce8fd2daf6f69aa0eb8ce` |
| Seeds | `[0, 1, 2, 3, 4]` |
| ε (deterministic baselines) | `1e-06` |

## Fixture baselines (mean ± std over seeds)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| length | 0.609375 | 0.000000 | 0.718155 | 0.000000 |
| events | 0.523438 | 0.000000 | 0.559127 | 0.000000 |

Random ranking baseline is included inside each per-seed `report.json`.

## Lane reminders

- Private lab pool metrics are **out of scope** for this card (never copy them here).
- CRISP / synthetic `val_bpb` are training facts, not ranking accuracy.
- `prepare.evaluate_bpb` is sacred and unused by this harness path.
