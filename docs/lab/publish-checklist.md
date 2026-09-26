# Lab publish checklist

This is the gate a lab ranking number must pass before it becomes public. **Passed on 2026-09-25** for the zero-shot pooled-lab run in [`ranking-validation.md`](ranking-validation.md). Any new lab number has to pass it again on its own run.

Modeled after the CUDA gate in [`docs/compute-paths.md`](../compute-paths.md): requirements only, not a result.

`prepare.py` remains **sacred** — never edited to “unlock” a lab claim.

---

## Today (real)

| Surface | Status | Where |
|---------|--------|--------|
| Lab-pool ranking (zero-shot, 2026-09-25) | **`published`**: AUROC 0.583 ± 0.007 | [`ranking-validation.md`](ranking-validation.md) |
| Redacted pack v0 | **`not_published`** | [`lab-public-pack-v0.md`](lab-public-pack-v0.md) · [`corpus/fixtures/lab_public_pack_v0/`](../../corpus/fixtures/lab_public_pack_v0/) |
| Public fixture card | **`published_fixture_card` / harness smoke** | [`docs/public-ranking-card-v1.md`](../public-ranking-card-v1.md) — **not** lab AUROC |
| Protocol | Checklist passed for the run above | [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) |

**Do not invent** lab numbers, and don't attach them to stranger badges; those prove the harness only. CUDA does not unlock lab AUROC ([`compute-paths.md`](../compute-paths.md)).

---

## Publish checklist (filled in for the 2026-09-25 run)

A public lab AUROC (or PR-AUC / precision@k) may be claimed **only after** all boxes below are filled with **measured** evidence on a named run **and** Abhinav explicit yes. Empty boxes = **no claim**. This section is a gate, not a result.

### 1. Multi-seed protocol frozen

- [x] Seed list frozen (protocol: 3–5 seeds, typically `0..N-1`) — recorded in the claim report
- [x] Same `TIME_BUDGET` (or fixed-step count) for every seed
- [x] Mean ± std published for AUROC / PR-AUC / precision@k — **not** a single cherry-picked seed
- [x] Protocol matches [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) (labeled sessions; CRISP-only unlabeled val **not** used as the claim corpus)

### 2. Baselines table

- [x] Random ranking baseline in the same report (mean ± std)
- [x] Length and/or event-count baseline present for sanity (not as the product claim)
- [x] Model mean AUROC **>** random mean, with disclosed std — no silent omission of baselines

### 3. Redacted pack reproducibility

- [x] Public-safe pack path named (e.g. `corpus/fixtures/lab_public_pack_v0/`) **or** explicit “private-only; pack TBD” (sample pack named; full capture private-only, content hash published)
- [x] Clone-repro smoke documented (length / harness path) without private JSONL
- [x] Private `lab/captures/` stays local — not vendored into the claim PR
- [x] Pack `claim_status` only flips when this whole checklist is green (not by renaming a file) (the pack stays `not_published`; only the pooled-lab result is published)

### 4. README / hero honesty

- [x] **No README hero AUROC** until this checklist is fully green (README cites the number in its results table, with limitations linked)
- [x] Stranger badges / Codespaces paths still prove harness smoke only — not lab AUROC
- [x] CRISP `val_bpb` remains train fitness only — never pasted as AUROC

### 5. Sacred code + approval

- [x] **`prepare.py` untouched** — `evaluate_bpb` contract unchanged
- [x] `train.py` / harness commit SHAs recorded on the claim report
- [x] Hardware named (today: Mac Apple Silicon MPS for lab pool; CUDA does not auto-unlock)
- [x] **Abhinav yes** recorded (explicit go-ahead for claim language + merge) (2026-09-25)

### 6. Allowed metrics (when claiming)

| Allowed after checklist green | Never from an empty checklist |
|------------------------------|-------------------------------|
| Lab-pool AUROC / PR-AUC / precision@k as **mean ± std** over frozen seeds | Invented or single-seed “hero” AUROC |
| Redacted-pack ranking metrics with provenance | Private pooled numbers pasted onto the public fixture card |
| Clear separation from CRISP `val_bpb` and stranger CPU gates | CUDA wall-clock or CUDA-alone “unlock” of lab AUROC |

---

## What this does not cover

- The redacted pack (`lab_public_pack_v0`) stays `not_published`. Only the pooled-lab run is published.
- Product Mac MPS train fitness (`val_bpb`) is a **different lane**; see [`docs/product-mac-path.md`](../product-mac-path.md).

Related: [`lab/README.md`](README.md) · [`ranking-validation.md`](ranking-validation.md) · [`lab-public-pack-v0.md`](lab-public-pack-v0.md) · [`docs/public-accuracy-eval.md`](../public-accuracy-eval.md) · [`docs/public-wins.md`](../public-wins.md) · [`docs/compute-paths.md`](../compute-paths.md).
