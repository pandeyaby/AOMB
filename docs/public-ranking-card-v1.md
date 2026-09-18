# Public ranking card v1 (frozen protocol)

**Card id:** `public_ranking_card_v1`  
**Status:** Fixture card with publishable fixture-scoped claim when the gate below passes.  
**claim_status:** see [`reports/public-ranking-card-v1/CARD.md`](../reports/public-ranking-card-v1/CARD.md) (`published` only for this synthetic fixture card when model mean AUROC beats length and events on the frozen eval split).

> **HOLD merge** for GRAX skim + Abhinav yes.  
> Do **not** promote private lab-pool AUROC as this public card  
> (including any private lab figure such as 0.766 — that lane stays in `docs/lab/`).  
> Do **not** cite CRISP / synthetic `val_bpb` as ranking accuracy.

Parent protocol (broader claim language): [`public-accuracy-eval.md`](public-accuracy-eval.md).  
This document freezes the **v1 public fixture card** specifically.

---

## Task definition

**Unit:** one session (trace-grouped session text from the public fixture pack).

**Task:** rank sessions by a scalar anomaly score (higher = more anomalous). Binary labels from `provenance.json` windows:

| Window label | Binary |
|--------------|--------|
| `normal` | 0 (negative) |
| `incident`, `cascade`, `anomalous` | 1 (positive) |
| `unknown` / other | **exclude** |

**Eval corpus (this card):** held-out **eval split** of [`corpus/fixtures/public_ranking_card_v1/`](../corpus/fixtures/public_ranking_card_v1/) — synthetic, no customer data. See `split.json` + fixture README.

**Train corpus (model path):** **fixture train-split NORMAL session texts only** (ephemeral in-memory dataloader). No CRISP. No `prepare.make_dataloader` / data download. Positives listed under `train_session_ids` are excluded from the LM objective so held-out incident/cascade remain surprising.

---

## Explicit non-goals / lane separation

| Lane | Role on this card |
|------|-------------------|
| **Public ranking card v1 fixture** | Only allowed public-card numbers (fixture eval-split baselines + fixture-only model) |
| **Private lab pool** (`docs/lab/`, private captures) | Lab evidence only — **never** copy lab-pool AUROC onto this card |
| **CRISP `val_bpb`** (0.407753 / 0.4309 / 0.458756) | Factual training metric — **not** ranking accuracy |
| **Synthetic smoke `val_bpb` 0.3682** | Legacy breeding — **not** ranking accuracy |
| **`prepare.evaluate_bpb`** | Sacred shard metric — **do not modify**; session BPB uses a separate path |
| **Production support / SLO** | **Not** this card |

---

## Splits

Frozen in [`corpus/fixtures/public_ranking_card_v1/split.json`](../corpus/fixtures/public_ranking_card_v1/split.json):

| Role | Count | Composition |
|------|------:|-------------|
| **train** | 10 | 5 normal + 3 incident + 2 cascade (LM uses **normals only**) |
| **eval** | 6 | 3 normal + 2 incident + 1 cascade (**both classes**; all ranking metrics) |

Do not silently change this fixture without bumping the card id / split id.

---

## Seeds

| Setting | Value |
|---------|-------|
| Seeds | `0,1,2,3,4` (five seeds) |
| Default CLI | `--seeds 0..4` |
| Model budget | `--train-seconds 45` (CPU-ok short train; Mac MPS welcome) |

Baselines (`length`, `events`) are deterministic on the eval split; seed only affects the **random ranking baseline** draws and model init.

---

## Metrics

Reported per seed and as **mean ± std** across seeds, on the **eval split only**:

1. **AUROC**
2. **PR-AUC**
3. **precision@k** with protocol defaults `k ∈ {min(10, n_pos), max(1, n // 10)}`

Also report per-class mean score, counts, **random** / **length** / **events** baselines on the same eval set.

---

## Baselines + model (every card run)

| Method | Score | Notes |
|--------|-------|-------|
| `length` | session character count | Deterministic on eval split |
| `events` | session event-line count | Deterministic on eval split |
| `random` | Uniform(0,1) draws | Mean±std over draws; seed-dependent |
| `model` | session BPB after fixture-only short train | Trains on train-split normals; scores eval |

No overnight / API spend. No `prepare.py` private corpus path.

---

## Reproducibility ε

| Path | ε / rule |
|------|----------|
| Deterministic baselines (`length`, `events`) | AUROC / PR-AUC / precision@k within **`1e-6`** vs `REFERENCE_baselines-*.json` (same fixture SHA) |
| Random baseline | Same `random_draws` + seed → identical mean±std |
| Model path | Mean metrics within **`1e-2`** of `REFERENCE_model-fixture.json` when CI/golden check runs |

Fixture content SHA-256 includes `provenance.json`, traces/logs, and `split.json`.

---

## One-command reproduce

```bash
# Baselines + fixture-only model (default card path) + ε check
./scripts/run_public_ranking_card_v1.sh --with-model --check-eps

# Baselines only
./scripts/run_public_ranking_card_v1.sh --baselines-only --check-eps

# Equivalent module entry
python -m eval.run_public_ranking_card --with-model --train-seconds 45 --seeds 0..4
```

Harness: `eval.run_public_ranking_card` → `eval.run_multiseed` → `eval.run_eval` with `--session-split eval` and `--train-corpus fixture-train`. `prepare.py` is untouched.

---

## claim_status rules (this fixture card only)

| State | When |
|-------|------|
| **`published`** | Fixture-only model **mean AUROC** beats **length** and **events** on the frozen eval split (report mean±std). Scope banner required. |
| **`not_published`** | Model missing, or model does **not** beat both baselines — report honestly; do not invent a claim. |

**Still never:**

1. Place private lab-pool metrics on README as the public card.
2. Cite CRISP / synthetic `val_bpb` as ranking accuracy.
3. Treat this as a production support metric.
4. Modify `prepare.evaluate_bpb`.

Merge remains **HOLD** until GRAX skim + Abhinav yes, even when fixture `claim_status=published`.

---

## Pass / fail checklist (gate for merge + public wording)

- [x] Eval corpus is this versioned public fixture with labels + content hash + frozen `split.json`
- [x] Private lab-pool AUROC **not** cited as the public card
- [x] CRISP / synthetic `val_bpb` **not** cited as ranking accuracy
- [x] `prepare.evaluate_bpb` unchanged
- [x] Seeds `0..4`; mean±std reported on **eval split**
- [x] AUROC, PR-AUC, precision@k + length/events/random baselines present
- [x] Deterministic baselines within ε=`1e-6` of committed references (same fixture SHA)
- [x] Model trains fixture-only (no CRISP / prepare shards) and beats length+events **or** honest `not_published`
- [ ] Claim wording is ranking / surprise only (see parent protocol)
- [ ] **Abhinav explicit yes** recorded
- [ ] **GRAX skim** complete; merge HOLD lifted

**Fail any remaining box → do not merge.** Fixture `claim_status` follows the model-vs-baseline gate above.
