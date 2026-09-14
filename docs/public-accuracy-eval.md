# Public accuracy claim — evaluation protocol

Frozen protocol for any **public accuracy claim** about AOMB’s anomaly-ranking behavior.

> **Status:** Protocol + harness scaffolding only.  
> **No claim is published** until the pass/fail checklist below is complete.  
> Do **not** invent AUROC / PR-AUC / precision@k numbers, and do **not** market incomplete runs.  
> **GRAX:** HOLD merge until Abhinav explicit yes. Protocol/checklist only until labeled ranking metrics exist.

---

## Two `val_bpb` numbers — keep separate (never mix)

These are **not** interchangeable and **neither** is a public accuracy claim:

| Lane | Value | What it is | What it is not |
|------|-------|------------|----------------|
| **Synthetic / smoke-era (legacy)** | **`0.3682`** | Historical overnight `agent_loop` best on `generate_observability_corpus.py` | Not reference-corpus product truth; not ranking accuracy |
| **CRISP factual baseline (README)** | **`0.458756`** | Single recorded 5‑min run on a capped Uber CRISP subset ([`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md)) | Not a marketing number; CRISP has **no incident labels** |

**Rules:**

- Always cite them in **separate tables / paragraphs** (README already does).
- **Never** 1:1 compare, average, or blend `0.3682` with `0.458756`.
- **Never** reuse either as AUROC, “accuracy %”, or public claim language.
- Public accuracy (this protocol) starts only after **labeled** session ranking metrics exist and the checklist passes.

---

## Claim statement (precise, non-hype)

**Proposed claim (only after checklist passes):**

> On a versioned, labeled eval corpus of held-out sessions, a model trained with AOMB’s next-token objective assigns **higher session-level surprise / bits-per-byte (BPB)** to **incident / cascade** windows than to **normal** windows, as measured by ranking metrics (AUROC, PR-AUC, precision@k) over multi-seed runs, beating a random ranking baseline.

This is a **ranking** claim about surprise scores vs labels — not a latency SLO claim, not a vendor bake-off, and not a substitute for `val_bpb` on unlabeled shards.

**What is explicitly not this claim:**

| Number / artifact | Role | Allowed as public accuracy? |
|-------------------|------|-----------------------------|
| CRISP subset `val_bpb=0.458756` | README / factual training baseline ([`docs/crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md)) | **No** |
| Synthetic smoke-era best `0.3682` | Historical overnight breeding on generator data | **No** — keep in separate table |
| `demo_anomaly.py` class-mean BPB gaps | Qualitative story / smoke | **No** — not multi-seed ranking metrics |
| Invented or placeholder AUROC | — | **Never** |

CRISP alone is **insufficient** for this claim: the CRISP dump has **no incident window labels** (windows = normal only). Prefer lab captures with provenance windows; AIOps Challenge is cite+fetch for labeled eval only (non-commercial — do not redistribute).

---

## Required artifacts (versioned provenance)

Before any public claim language, record and publish (or attach to the claim post / release notes):

| Artifact | Requirement |
|----------|-------------|
| **Eval corpus id + content hash** | Stable id (e.g. capture_id / provenance filename) and SHA-256 of the eval session set or capture tree |
| **Corpus provenance** | Source kind (`lab_capture` preferred), license, window labels (`normal` / `incident` / `cascade`), capture scripts / fault modes |
| **Train corpus id** (if distinct) | What the model was trained on (may include CRISP for train-only; eval labels must still come from labeled sources) |
| **`train.py` SHA** | Full git commit SHA of the scoring/training code |
| **`prepare.py` SHA** | Confirm `evaluate_bpb` contract unchanged |
| **Seeds** | Exact seed list used (protocol: 3–5 seeds, typically `0..N-1`) |
| **Hardware** | Machine class, device (`mps` / `cpu` / `cuda`), OS |
| **Budget** | Full `TIME_BUDGET` from `prepare.py` **or** fixed-step count; same for every seed |
| **Harness report** | JSON + markdown from `eval/` (metrics, seed, corpus id, model SHA) — no fabricated fields |

---

## Metrics

Unit of scoring: **one session** (trace-grouped session text), scored with session-level next-token **surprise / BPB** (same information-theoretic spirit as `evaluate_bpb`, applied per session — **do not change** `prepare.evaluate_bpb`).

Binary labels for ranking:

| Label | Class |
|-------|-------|
| `normal` | negative (0) |
| `incident`, `cascade` | positive (1) |
| `unknown` / other | **exclude** from ranking metrics (log count) |

**Primary metrics** (higher surprise → ranked more anomalous):

1. **AUROC** — ranking quality of session BPB vs binary labels  
2. **PR-AUC** — precision-recall area (prefer when positives are rare)  
3. **precision@k** — among the `k` highest-BPB sessions, fraction that are positive (`k` fixed in the report, e.g. `k=min(10, n_pos)` and/or `k=max(1, n//10)`)

**Report also:**

- Per-class mean ± std of session BPB (descriptive; not the claim by itself)
- Counts: `n_normal`, `n_positive`, excluded
- **Random ranking baseline** (same labels, scores ~ Uniform or shuffled ranks) — mean±std over baseline draws
- Optional: **length baseline** (score = session character or token length) and/or a simple threshold on a scalar feature — for sanity, not as the product claim

**Multi-seed:** Run **3–5** full `TIME_BUDGET` (or fixed-step) train-then-score (or score-from-checkpoint) runs with seeds `0..N-1`. Publish **mean ± std** of AUROC, PR-AUC, precision@k across seeds. A single seed is scaffolding only — not a public claim.

---

## Data policy

| Source | Use for this protocol |
|--------|------------------------|
| **Lab captures** (`lab/` + `provenance.json` windows) | **Preferred** for first honest ranking eval — labels from capture metadata |
| **AIOps Challenge 2020** | Labeled eval only; **non-commercial**; cite + fetch locally (`corpus.ingest.fetch_aiops_challenge`); **do not redistribute** |
| **Uber CRISP** | Train corpus optional; **not** sufficient alone for labeled accuracy (no incident labels) |
| **Synthetic generator** | Smoke / CI / `demo_anomaly.py` only — not a public accuracy corpus |

Hypothesis (accepted for scaffolding): lab capture sessions + window labels in `provenance.json` are enough for a **first** ranking eval path. Larger labeled sets come later.

---

## Harness (this repo)

```bash
# Metrics + labels only (no model) — useful for CI / fixtures
uv run python -m eval.run_eval \
  --capture corpus/fixtures/lab_sample \
  --scores-from length \
  --out-dir /tmp/aomb-eval-smoke

# Multi-seed aggregation (consumes per-seed JSON reports)
uv run python -m eval.run_multiseed --seeds 0,1,2 --out-dir /tmp/aomb-eval-multiseed
```

Full train-then-score (optional; reuses demo-style session BPB, does **not** modify `prepare.evaluate_bpb`):

```bash
uv run python -m eval.run_eval \
  --capture lab/captures/<id> \
  --train-seconds 300 \
  --seed 0 \
  --out-dir reports/public-accuracy/seed0
```

See [`eval/README.md`](../eval/README.md).

---

## Pass / fail checklist (gate for public claim language)

Mark each item before any blog post, README “accuracy”, press, or social claim:

- [ ] Eval corpus is **labeled** (normal vs incident/cascade); CRISP-only unlabeled val is **not** used as the claim corpus
- [ ] Corpus id + content hash + provenance recorded
- [ ] `train.py` / harness commit SHAs recorded; `prepare.evaluate_bpb` **unchanged**
- [ ] Hardware + TIME_BUDGET (or fixed steps) recorded and identical across seeds
- [ ] **≥ 3** seeds completed; metrics reported as **mean ± std** (not a single cherry-picked run)
- [ ] AUROC, PR-AUC, and precision@k present in the harness JSON + markdown reports
- [ ] **Random ranking baseline** included in the same report; model mean AUROC **>** random mean (with disclosed std)
- [ ] No citation of CRISP `val_bpb=0.458756` or synthetic `0.3682` as the public accuracy number
- [ ] Synthetic `0.3682` and CRISP `0.458756` remain in **separate** factual lanes (no blend / no 1:1 compare in claim copy)
- [ ] AIOps data (if used) cited; not redistributed from this repo
- [ ] Claim wording matches the **Claim statement** section above (ranking / surprise), without hype extras
- [ ] **Merge HOLD:** Abhinav explicit yes recorded before merge (GRAX)

**Fail any box → do not publish an accuracy claim.** Protocol scaffolding and empty/fixture reports are fine to land in-tree. **Do not merge this work until Abhinav yes.**
