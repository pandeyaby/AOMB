# Stranger demo — cross-platform public entry

**Audience:** someone who just cloned [pandeyaby/AOMB](https://github.com/pandeyaby/AOMB) on Linux (or CI), with **no Apple MPS** and **no API keys**.

**Goal:** verify the public deterministic gates in a few minutes. Not the overnight Mac research loop.

**Pair path (cite without cloning):** green [`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) badge / Codespaces — see [`stranger-verify.md`](stranger-verify.md). Same honesty limits; different entry (linkable Action vs local clone).

---

## Honesty banner (read first)

| Claim surface | Honest status |
|---------------|---------------|
| **Public ranking card** | **`published_fixture_card` / harness smoke** only. Tiny-n synthetic fixture (eval **n=36**). High AUROC here = toy separation — **not** production / field accuracy. |
| **Lab AUROC** | Stays **`not_published`**. **No invented AUROC.** Do not promote private lab-pool numbers on README. |
| **CRISP `val_bpb`** | **Train fitness only** when cited — **not** ranking accuracy. |
| **DIPTYCH** | Companion grading repo: [pandeyaby/DIPTYCH](https://github.com/pandeyaby/DIPTYCH). AOMB **emits** paired probes; DIPTYCH **grades**. Do not fork DIPTYCH product code into AOMB beyond the existing adapter emit path. |
| **Overnight agent / MPS train** | **Out of scope** for this path. Needs Mac Silicon (+ API keys for `agent_loop`). |

`prepare.py` is **sacred** — never modified by the stranger path.

---

## One path (numbered)

```bash
# 0) Clone
git clone https://github.com/pandeyaby/AOMB.git && cd AOMB

# 1) Install
# Preferred:
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
# Fallback (no uv):
pip install pyarrow numpy rustbpe tiktoken
# For full ranking-card model smoke (CPU torch — no MPS required):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Stranger demo (no keys, no MPS)
./scripts/stranger_demo.sh
```

That script runs:

1. `./scripts/run_diptych_full8.sh` — full-8 operators + **`gate_axis_mutate`** (all `aomb=green`, `axis_power=true`)
2. `./scripts/run_public_ranking_card_v1.sh` — default **full harness smoke** (`--with-model --check-eps`) on **CPU** torch

### Faster subset (baselines only)

If you skip torch or want under ~1 minute:

```bash
STRANGER_FAST=1 ./scripts/stranger_demo.sh
# equivalent card step:
./scripts/run_public_ranking_card_v1.sh --baselines-only --check-eps
```

**Honest:** baselines-only proves length/events ε against committed refs. It does **not** re-train the fixture model or refresh `published_fixture_card` model means. Full model smoke is still **CPU** (CI already runs it on `ubuntu-latest`) — **not** MPS-gated.

### Cite path vs clone path

| Entry | Command / link | Default card mode |
|-------|----------------|-------------------|
| **Clone demo** (this doc) | `./scripts/stranger_demo.sh` | Full `--with-model` unless `STRANGER_FAST=1` |
| **Cite / verify** | [`stranger-verify`](stranger-verify.md) badge or `./scripts/stranger_verify.sh` | Defaults `STRANGER_FAST=1` (baselines ε); delegates to demo when present |

---

## What a stranger can verify vs what needs Mac MPS

| In under 5–10 min on Linux / CI | Still needs Mac MPS (+ often keys) |
|------------------------------|-------------------------------------|
| DIPTYCH full-8 + `gate_axis_mutate` | Overnight `agent_loop` / morning report |
| Ranking card baselines ε (seconds) | Product Uber CRISP train on MPS |
| Ranking card fixture model smoke (~4–10 min CPU) | Claiming lab AUROC as public |
| Adapter / stranger CI badges green on `main` | Editing `prepare.py` (don’t) |

---

## CI badges (ubuntu)

| Workflow | What it exercises |
|----------|-------------------|
| [![diptych-adapter-gate](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml) | Full-8 + `gate_axis_mutate` on every push to `main` |
| [![public-ranking-card-v1](https://github.com/pandeyaby/AOMB/actions/workflows/public-ranking-card-v1.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/public-ranking-card-v1.yml) | Fixture card baselines + CPU model smoke (path-filtered) |
| [![stranger-demo](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml) | End-to-end `./scripts/stranger_demo.sh` (fast baselines mode on PR; full model on `main` / dispatch) |
| [![stranger-verify](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) | Cite path — `./scripts/stranger_verify.sh` (default fast; optional CPU model via dispatch) |

---

## Loud refusals

```bash
STRANGER_EXPECT_MPS=1 ./scripts/stranger_demo.sh     # fails with clear message
STRANGER_EXPECT_OVERNIGHT=1 ./scripts/stranger_demo.sh
./scripts/stranger_demo.sh --overnight               # fails
```

---

## Related docs

- [`docs/anomaly-story.md`](anomaly-story.md) — **~30 min thesis:** `val_bpb` / session surprise *is* the anomaly signal (`demo_anomaly.py`)
- [`docs/stranger-60s.md`](stranger-60s.md) — 60-second outsider cheatsheet
- [`docs/public-wins.md`](public-wins.md) — outsider landing (badges + one-click verify)
- [`docs/stranger-verify.md`](stranger-verify.md) — cite without cloning (Actions / Codespaces)
- [`docs/compute-paths.md`](compute-paths.md) — CPU stranger vs MPS product train (CUDA = checklist only)
- [DIPTYCH](https://github.com/pandeyaby/DIPTYCH) — companion grading (AOMB emits; DIPTYCH grades)
- [`docs/public-ranking-card-v1.md`](public-ranking-card-v1.md) — fixture limits
- [`docs/crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) — train fitness only
- [`docs/lab/`](lab/) — lab lane (`not_published`)
- [`docs/paired-probes/`](paired-probes/) — AOMB emit path for DIPTYCH
- Optional after the story: [`product-mac-path.md`](product-mac-path.md) · [`byo-and-scorer.md`](byo-and-scorer.md)