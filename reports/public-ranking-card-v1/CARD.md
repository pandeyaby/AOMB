# Public ranking card v1 — fixture report

**Card id:** `public_ranking_card_v1`  
**claim_status:** `published_fixture_card`  
**Protocol:** [`docs/public-ranking-card-v1.md`](../../docs/public-ranking-card-v1.md)

> ## Limitations (read first)
>
> - **n_eval = 24** labeled sessions on a **synthetic** fixture — harness smoke, not a field study.
> - **High / perfect AUROC on this toy pack ≠ general public accuracy** and ≠ production AUROC.
> - Text patterns are stylized (catalog_ok vs checkout_failed / redis_unavailable); separation can be easy.
> - **Not** private lab-pool AUROC (incl. 0.766). **Not** CRISP `val_bpb`. **Not** a support/SLO metric.
> - Train corpus = fixture train-split **normal** texts only (no CRISP / prepare shards).

**Claim gate:** Fixture-only model mean AUROC 1.000000 beats length 0.812500 and events 0.715278 on the frozen synthetic eval split. Status = published fixture card / harness smoke only — NOT production AUROC, NOT general public accuracy, NOT lab pool.

## Identity

| Field | Value |
|-------|-------|
| Fixture | `corpus/fixtures/public_ranking_card_v1` |
| Fixture content SHA-256 | `8c9ad74ae718bcc9541671ccd3d212e75f5a3ae0013890504508788a1b6d309e` |
| Split | `split.json` (eval n=24) |
| Seeds | `[0, 1, 2, 3, 4]` |
| ε (deterministic baselines) | `1e-06` |
| ε (model golden, if checked) | `0.01` |

## Fixture eval-split baselines (n=24; mean ± std over seeds)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| length | 0.812500 | 0.000000 | 0.860101 | 0.000000 |
| events | 0.715278 | 0.000000 | 0.707011 | 0.000000 |

Random ranking baseline is included inside each per-seed `report.json`.

## Fixture-only model (train 45s × seeds, eval n=24)

| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |
|--------|------------|-----------|-------------|------------|
| session BPB (fixture train→eval) | 1.000000 | 0.000000 | 1.000000 | 0.000000 |

If AUROC is ~1.0 on this synthetic pack, treat it as **toy separation / harness smoke**, not a marketable production accuracy number.

## Explicit non-claims

- Not general public accuracy or production AUROC.
- Not the private lab pool (including any lab-pool AUROC such as 0.766).
- Not CRISP / synthetic `val_bpb`.
- Not a production support or incident-response SLO metric.
- `prepare.evaluate_bpb` is sacred and unused by this harness path.

## Lane reminders

- Private lab pool metrics stay in `docs/lab/` (`not_published` lane).
- `published_fixture_card` = fixture harness smoke that beat baselines — still not production.
