# Public ranking card v1 (frozen protocol)

**Card id:** `public_ranking_card_v1`  
**Status:** Synthetic fixture + harness-smoke path.  
**claim_status:** `published_fixture_card` only when model mean AUROC beats length+events on the frozen eval split — see [`reports/public-ranking-card-v1/CARD.md`](../reports/public-ranking-card-v1/CARD.md).

> **Honest framing:** harness smoke only. Lab ranking stays **`not_published`**. **No AUROC hero** on README or marketing.  
> Do **not** promote private lab-pool AUROC (incl. 0.766 — stays in `docs/lab/`).  
> Do **not** cite CRISP / synthetic `val_bpb` as ranking accuracy.  
> Do **not** market fixture AUROC as production / general public accuracy.  
> Calibration grading for paired probes lives in [DIPTYCH](https://github.com/pandeyaby/DIPTYCH) — separate from this card.

Parent protocol: [`public-accuracy-eval.md`](public-accuracy-eval.md).

---

## Limitations (loud)

- **Synthetic** stylized sessions only.
- Eval size **n=36** held-out labeled sessions (72 total; balanced 36/36 split).
- **High / perfect AUROC here = toy separation / harness smoke**, not field performance.
- Not a production support / SLO metric.

---

## Task definition

**Unit:** one session (trace-grouped session text).

**Labels:** `normal`→0; `incident`/`cascade`/`anomalous`→1; unknown excluded.

**Eval:** held-out **eval split** of [`corpus/fixtures/public_ranking_card_v1/`](../corpus/fixtures/public_ranking_card_v1/).

**Train (model):** fixture train-split **NORMAL** texts only (ephemeral dataloader). No CRISP. No `prepare.make_dataloader`.

---

## Splits (`split.json`)

| Role | Count | Composition (balanced) |
|------|------:|------------------------|
| **train** | 36 | 18 normal + 12 incident + 6 cascade (LM uses **normals only**) |
| **eval** | 36 | 18 normal + 12 incident + 6 cascade (**both classes**) |

---

## Seeds / budget

| Setting | Value |
|---------|-------|
| Seeds | `0..4` |
| Model budget | `--train-seconds 45` |

---

## Metrics (eval split)

AUROC, PR-AUC, precision@k + length / events / random baselines. Mean±std over seeds.

---

## Reproducibility ε

| Path | ε |
|------|---|
| length / events | `1e-6` vs `REFERENCE_baselines-*.json` |
| model golden | `1e-2` vs `REFERENCE_model-fixture.json` |

---

## One-command reproduce

```bash
./scripts/run_public_ranking_card_v1.sh --with-model --check-eps
```

`prepare.py` untouched.

---

## claim_status rules

| State | Meaning |
|-------|---------|
| **`published_fixture_card`** | Model mean AUROC beats length+events on this synthetic eval split. **Harness smoke only** — not production AUROC. |
| **`not_published`** | Model missing or does not beat both baselines. |

Merge remains **HOLD** until GRAX + Abhinav yes for any stronger claim language. Fixture card itself stays harness smoke.
