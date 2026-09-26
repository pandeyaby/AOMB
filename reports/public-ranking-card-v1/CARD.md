# Public ranking card v1 — fixture report

**Card id:** `public_ranking_card_v1`  
**claim_status:** `published_fixture_card`  
**Protocol:** [`docs/public-ranking-card-v1.md`](../../docs/public-ranking-card-v1.md)

> ## Limitations (read first)
>
> - **n_eval = 36** labeled sessions on a **committed local fixture** — harness smoke, not a field study.
> - **High / perfect AUROC on this toy pack ≠ general public accuracy** and ≠ production AUROC.
> - Text patterns may be stylized; separation can be easy.
> - **Not** the lab-pool AUROC (`docs/lab/ranking-validation.md`). **Not** CRISP `val_bpb`. **Not** a support/SLO metric.
> - **Not** a README AUROC hero. Metrics below are computed from real local fixture scores.
> - Train corpus (model path) = fixture train-split **normal** texts only (no CRISP / prepare shards).

**Claim gate:** Fixture-only model mean AUROC 1.000000 beats length 0.663580 and events 0.601852 on the frozen synthetic eval split. Status = published fixture card / harness smoke only — NOT production AUROC, NOT general public accuracy, NOT lab pool, NOT a README hero.

## Identity

| Field | Value |
|-------|-------|
| Fixture | `corpus/fixtures/public_ranking_card_v1` |
| Fixture content SHA-256 | `49403cb3dd005e7e5af1de510ba3f5ed6e83df566e9ba3d46b4baed66367cfc5` |
| Split | `split.json` (eval n=36) |
| Seeds | `[0, 1, 2, 3, 4]` |
| ε (deterministic baselines) | `1e-06` |
| ε (model golden, if checked) | `0.01` |

## Fixture eval-split baselines (n=36; mean ± std over seeds)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| length | 0.663580 | 0.000000 | 0.705060 | 0.000000 |
| events | 0.601852 | 0.000000 | 0.574747 | 0.000000 |

Random ranking baseline is included inside each per-seed `report.json`.

## Fixture-only model (train 45s × seeds, eval n=36)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| session BPB (fixture train→eval) | 1.000000 | 0.000000 | 1.000000 | 0.000000 |

If AUROC is ~1.0 on this synthetic pack, treat it as **toy separation / harness smoke**, not a marketable production accuracy number or README hero.

## Explicit non-claims

- Not general public accuracy or production AUROC.
- Not a README / marketing AUROC hero.
- Not the lab pool (see docs/lab/ranking-validation.md for that result).
- Not CRISP / synthetic `val_bpb`.
- Not a production support or incident-response SLO metric.
- `prepare.evaluate_bpb` is sacred and unused by this harness path.

## Lane reminders

- Private lab pool metrics stay in `docs/lab/` (`not_published` lane).
- `published_fixture_card` = fixture harness smoke that beat baselines — still not production / not README hero.
- Per-session length/events baselines: `session_baseline_scores.json` (always `not_published`).
