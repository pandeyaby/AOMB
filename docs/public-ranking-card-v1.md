# Public ranking card v1 (frozen protocol)

**Card id:** `public_ranking_card_v1`  
**Status:** Fixture + harness path only.  
**claim_status=`not_published`** until the checklist below passes **and** Abhinav explicit greenlight.

> **HOLD merge** for GRAX skim + Abhinav yes.  
> Do **not** promote private lab-pool AUROC as this public card  
> (including any private lab figure such as 0.766 — that lane stays in `docs/lab/`).  
> Do **not** cite CRISP / synthetic `val_bpb` as ranking accuracy.

This card unlocks a **future** honest public accuracy claim path. Until published, all reports and README pointers must keep `claim_status=not_published`.

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

**Eval corpus (this card):** [`corpus/fixtures/public_ranking_card_v1/`](../corpus/fixtures/public_ranking_card_v1/) — synthetic, no customer data. See fixture README for provenance + content hash after each run.

**Train corpus (optional model path):** whatever shards `train.py` / demo loaders use locally. Model path is **optional** for this card; CI runs **baselines only**.

---

## Explicit non-goals / lane separation

| Lane | Role on this card |
|------|-------------------|
| **Public ranking card v1 fixture** | Only allowed public-card numbers (fixture baselines; optional short model) |
| **Private lab pool** (`docs/lab/`, private captures) | Lab evidence only — **never** copy lab-pool AUROC onto this card |
| **CRISP `val_bpb`** (0.407753 / 0.4309 / 0.458756) | Factual training metric — **not** ranking accuracy |
| **Synthetic smoke `val_bpb` 0.3682** | Legacy breeding — **not** ranking accuracy |
| **`prepare.evaluate_bpb`** | Sacred shard metric — **do not modify**; session BPB uses a separate path |

---

## Splits

v1 uses a **single frozen fixture pack** (no train/val split inside the card). All 16 scorable sessions are scored and ranked together.

Future cards may introduce held-out labeled packs; do not silently change this fixture without bumping the card id.

---

## Seeds

Protocol: **3–5** seeds. This card freezes:

| Setting | Value |
|---------|-------|
| Seeds | `0,1,2,3,4` (five seeds) |
| Default CLI | `--seeds 0..4` |

Baselines (`length`, `events`) are deterministic given the fixture; seed only affects the **random ranking baseline** draws and optional model init.

---

## Metrics

Reported per seed and as **mean ± std** across seeds:

1. **AUROC**
2. **PR-AUC**
3. **precision@k** with protocol defaults `k ∈ {min(10, n_pos), max(1, n // 10)}`

Also report:

- Per-class mean score
- Counts: `n_normal`, `n_positive`, excluded
- **Random ranking baseline** (Uniform scores, fixed draws)
- **Length baseline** and **event-count baseline**

Higher score = more anomalous (same convention as `eval/`).

---

## Baselines (required on every card run)

| Baseline | Score | Notes |
|----------|-------|-------|
| `length` | session character count | Deterministic on fixture |
| `events` | session event-line count | Deterministic on fixture |
| `random` | Uniform(0,1) draws | Mean±std over draws; seed-dependent |

Optional: `--scores-from model --train-seconds N` (short train-then-score). **Not** required for CI. No overnight / API spend on this card path.

---

## Reproducibility ε

| Path | ε / rule |
|------|----------|
| Deterministic baselines (`length`, `events`) | AUROC / PR-AUC / precision@k must match within **`1e-6`** absolute vs committed reference aggregates under `reports/public-ranking-card-v1/` (same fixture SHA) |
| Random baseline | Same `random_draws` + seed → identical mean±std (stdlib `random`) |
| Model path | Optional; not ε-gated in CI |

Fixture content SHA-256 is recorded in each report (`corpus.content_sha256`). Changing the fixture without bumping card id / updating reports fails reproducibility checks.

---

## One-command reproduce

```bash
# Baselines only (CI / default) — writes reports/public-ranking-card-v1/
./scripts/run_public_ranking_card_v1.sh

# Equivalent module entry
python -m eval.run_public_ranking_card --baselines-only

# Optional short model path (local only; not CI)
python -m eval.run_public_ranking_card --with-model --train-seconds 30 --seeds 0..2
```

Harness reuses `eval.run_eval` + `eval.run_multiseed`. `prepare.py` is untouched.

---

## claim_status rules

| State | When |
|-------|------|
| **`not_published`** | Default for all v1 reports, README pointers, and this protocol until checklist **and** Abhinav greenlight |
| **`published`** | Only after checklist complete + Abhinav yes + public wording matches the claim statement in `public-accuracy-eval.md` |

**Rules:**

1. Every JSON/markdown report for this card must include `claim_status=not_published` until publication.
2. Never place private lab-pool metrics on README as the public card.
3. Fixture baseline numbers (only) may appear with explicit “fixture baseline / not a claim” labeling.
4. Fail any checklist box → remain `not_published`.

---

## Pass / fail checklist (gate for publication)

- [ ] Eval corpus is this versioned public fixture (or a successor card id) with labels + content hash
- [ ] Private lab-pool AUROC **not** cited as the public card
- [ ] CRISP / synthetic `val_bpb` **not** cited as ranking accuracy
- [ ] `prepare.evaluate_bpb` unchanged
- [ ] Seeds `0..4` (or documented 3–5) completed; mean±std reported
- [ ] AUROC, PR-AUC, precision@k + length/events/random baselines present
- [ ] Deterministic baselines within ε=`1e-6` of committed references (same fixture SHA)
- [ ] Claim wording is ranking / surprise only (see parent protocol)
- [ ] **Abhinav explicit yes** recorded
- [ ] **GRAX skim** complete; merge HOLD lifted

**Fail any box → do not publish. claim_status stays `not_published`.**
