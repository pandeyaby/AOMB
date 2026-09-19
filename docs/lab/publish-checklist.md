# Lab publish checklist — gate only (no AUROC claim yet)

**This PR / doc does not flip `claim_status`.** It is the **plan gate** for when private lab ranking evidence may become a public number. Empty boxes = **`claim_status` stays `not_published`**. No invented AUROC. No README hero AUROC.

Modeled after the CUDA gate in [`docs/compute-paths.md`](../compute-paths.md): requirements only, not a result.

`prepare.py` remains **sacred** — never edited to “unlock” a lab claim.

---

## Today (real)

| Surface | Status | Where |
|---------|--------|--------|
| Private lab-pool ranking | **`not_published`** | [`ranking-validation.md`](ranking-validation.md) |
| Redacted pack v0 | **`not_published`** | [`lab-public-pack-v0.md`](lab-public-pack-v0.md) · [`corpus/fixtures/lab_public_pack_v0/`](../../corpus/fixtures/lab_public_pack_v0/) |
| Public fixture card | **`published_fixture_card` / harness smoke** | [`docs/public-ranking-card-v1.md`](../public-ranking-card-v1.md) — **not** lab AUROC |
| Protocol scaffolding | Checklist / harness only | [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) |

**Do not invent:** lab AUROC on the README hero, marketing copy, or stranger badges. CUDA does not unlock lab AUROC ([`compute-paths.md`](../compute-paths.md)).

---

## Publish checklist (empty = no claim)

A public lab AUROC (or PR-AUC / precision@k) may be claimed **only after** all boxes below are filled with **measured** evidence on a named run **and** Abhinav explicit yes. Empty boxes = **no claim**. This section is a gate, not a result.

### 1. Multi-seed protocol frozen

- [ ] Seed list frozen (protocol: 3–5 seeds, typically `0..N-1`) — recorded in the claim report
- [ ] Same `TIME_BUDGET` (or fixed-step count) for every seed
- [ ] Mean ± std published for AUROC / PR-AUC / precision@k — **not** a single cherry-picked seed
- [ ] Protocol matches [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) (labeled sessions; CRISP-only unlabeled val **not** used as the claim corpus)

### 2. Baselines table

- [ ] Random ranking baseline in the same report (mean ± std)
- [ ] Length and/or event-count baseline present for sanity (not as the product claim)
- [ ] Model mean AUROC **>** random mean, with disclosed std — no silent omission of baselines

### 3. Redacted pack reproducibility

- [ ] Public-safe pack path named (e.g. `corpus/fixtures/lab_public_pack_v0/`) **or** explicit “private-only; pack TBD”
- [ ] Clone-repro smoke documented (length / harness path) without private JSONL
- [ ] Private `lab/captures/` stays local — not vendored into the claim PR
- [ ] Pack `claim_status` only flips when this whole checklist is green (not by renaming a file)

### 4. README / hero honesty

- [ ] **No README hero AUROC** until this checklist is fully green
- [ ] Stranger badges / Codespaces paths still prove harness smoke only — not lab AUROC
- [ ] CRISP `val_bpb` remains train fitness only — never pasted as AUROC

### 5. Sacred code + approval

- [ ] **`prepare.py` untouched** — `evaluate_bpb` contract unchanged
- [ ] `train.py` / harness commit SHAs recorded on the claim report
- [ ] Hardware named (today: Mac Apple Silicon MPS for lab pool; CUDA does not auto-unlock)
- [ ] **Abhinav yes** recorded (explicit go-ahead for claim language + merge)

### 6. Allowed metrics (when claiming)

| Allowed after checklist green | Never from an empty checklist |
|------------------------------|-------------------------------|
| Lab-pool AUROC / PR-AUC / precision@k as **mean ± std** over frozen seeds | Invented or single-seed “hero” AUROC |
| Redacted-pack ranking metrics with provenance | Private pooled numbers pasted onto the public fixture card |
| Clear separation from CRISP `val_bpb` and stranger CPU gates | CUDA wall-clock or CUDA-alone “unlock” of lab AUROC |

---

## Explicit non-claims for this doc / PR

- This file **does not** set `claim_status=published` anywhere.
- This file **does not** put lab AUROC on the README hero.
- Empty checkboxes above = **`not_published`** remains correct.
- Product Mac MPS train fitness (`val_bpb`) is a **different lane** — see [`docs/product-mac-path.md`](../product-mac-path.md).

Related: [`lab/README.md`](README.md) · [`ranking-validation.md`](ranking-validation.md) · [`lab-public-pack-v0.md`](lab-public-pack-v0.md) · [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) · [`docs/public-wins.md`](../public-wins.md) · [`docs/compute-paths.md`](../compute-paths.md).
