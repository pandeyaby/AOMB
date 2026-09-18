# Public ranking card v1 — fixture report

**Card id:** `public_ranking_card_v1`  
**claim_status:** `published`  
**Protocol:** [`docs/public-ranking-card-v1.md`](../../docs/public-ranking-card-v1.md)

> **Scope:** synthetic Public ranking card v1 fixture eval split only.  
> **Not** private lab-pool AUROC. **Not** CRISP `val_bpb`.  
> **Not** a production support / SLO metric.  
> Train corpus for the model path = fixture train-split sessions only (no CRISP / prepare shards).

**Claim gate:** Fixture-only model mean AUROC 1.000000 beats length 0.555556 and events 0.444444 on the frozen eval split. Scope: this synthetic public fixture card only — not lab pool, not CRISP val_bpb, not production support.

## Identity

| Field | Value |
|-------|-------|
| Fixture | `corpus/fixtures/public_ranking_card_v1` |
| Fixture content SHA-256 | `f8a8288b1e37225b5cda9f8eef8e1e0879e3468797ce969b595ddfb17fdb4e1c` |
| Split | `split.json` (train=see split / eval scored) |
| Seeds | `[0, 1, 2, 3, 4]` |
| ε (deterministic baselines) | `1e-06` |
| ε (model golden, if checked) | `0.01` |

## Fixture eval-split baselines (mean ± std over seeds)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| length | 0.555556 | 0.000000 | 0.722222 | 0.000000 |
| events | 0.444444 | 0.000000 | 0.500000 | 0.000000 |

Random ranking baseline is included inside each per-seed `report.json`.

## Fixture-only model (train 45s × seeds, eval split)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| session BPB (fixture train→eval) | 1.000000 | 0.000000 | 1.000000 | 0.000000 |

Model mean AUROC **beats** length and events on this fixture eval set.

## Explicit non-claims

- Not the private lab pool (including any lab-pool AUROC such as 0.766).
- Not CRISP / synthetic `val_bpb`.
- Not a production support or incident-response SLO metric.
- `prepare.evaluate_bpb` is sacred and unused by this harness path.

## Lane reminders

- Private lab pool metrics stay in `docs/lab/` (`not_published` lane).
- This card’s `published` status (if set) is **fixture-scoped only**.
