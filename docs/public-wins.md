# Public wins — what a stranger can verify today

> **Verify in 60s** — [![stranger-verify](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) · [Open in Codespaces](https://codespaces.new/pandeyaby/AOMB) · [Example green run](https://github.com/pandeyaby/AOMB/actions/runs/35471119285)

**Audience:** outsiders (no Mac, no API keys, no private lab access).  
**Rule:** only list checks that are green or clone-reproducible **now**. No invented AUROC. No CUDA claim.

`prepare.py` is **sacred** — never edited for these paths.

Cheatsheet (what green means): **[`stranger-60s.md`](stranger-60s.md)**.  
Share / paste snip (markdown + plain text): **[`share-snip.md`](share-snip.md)**.  
Contributing as a stranger: **[`contributing-stranger.md`](contributing-stranger.md)**.

---

## One-click verify (no clone required)

| Action | Link |
|--------|------|
| **stranger-verify badge** | [Live workflow / badge](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) — cite a green check |
| **Open in Codespaces** | [codespaces.new/pandeyaby/AOMB](https://codespaces.new/pandeyaby/AOMB) → then `STRANGER_FAST=1 ./scripts/stranger_verify.sh` |
| **Example green run** | [run 35471119285](https://github.com/pandeyaby/AOMB/actions/runs/35471119285) — pinned **example** cite (main · 2026-09-19 · `7abdfd5`); badge/workflow remains canonical |
| **Run Action (workflow_dispatch)** | [stranger-verify → Run workflow](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) — default `full_model=false` ≡ fast path (`STRANGER_FAST=1`); optional `full_model=true` = CPU torch card smoke, **not** lab AUROC |
| **Same for local demo CI** | [stranger-demo → Run workflow](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml) |

Both workflows already expose **`workflow_dispatch`**. Prefer **stranger-verify** when you only need a citeable green check. Input alias: **`full_model=false`** (default) ↔ `STRANGER_FAST=1`.

**Example green run proves (when green):** DIPTYCH full-8 emit + `gate_axis_mutate` (`aomb=green`, `axis_power=true`) + ranking-card baselines ε. **Does not prove:** lab AUROC, CUDA, production / field accuracy. Tip of `main` may move — the live badge remains canonical.

---

## Live badges (cite these)

[![diptych-adapter-gate](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml)
[![stranger-demo](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml)
[![stranger-verify](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml)
[![public-ranking-card-v1](https://github.com/pandeyaby/AOMB/actions/workflows/public-ranking-card-v1.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/public-ranking-card-v1.yml)

| Badge / path | What a green check proves | Doc |
|--------------|---------------------------|-----|
| **stranger-demo** | Clone path end-to-end: full-8 + `gate_axis_mutate` + ranking-card harness (CPU) | [`stranger-demo.md`](stranger-demo.md) · `./scripts/stranger_demo.sh` |
| **stranger-verify** | Same gates via Actions / Codespaces — cite without cloning | [`stranger-verify.md`](stranger-verify.md) · `./scripts/stranger_verify.sh` |
| **diptych-adapter-gate** | Full-8 emit + `gate_axis_mutate` on every push to `main` (`aomb=green`, `axis_power=true`) | [`paired-probes/`](paired-probes/) |
| **public-ranking-card-v1** | Fixture card baselines + CPU model smoke — **`published_fixture_card` / harness smoke** only | [`public-ranking-card-v1.md`](public-ranking-card-v1.md) |

---

## Clone path (still no MPS / no keys)

```bash
git clone https://github.com/pandeyaby/AOMB.git && cd AOMB
uv sync   # or: pip install pyarrow numpy rustbpe tiktoken
./scripts/stranger_demo.sh
# faster: STRANGER_FAST=1 ./scripts/stranger_demo.sh
```

---

## Compute honesty

| Path | Hardware today | Honest claim |
|------|----------------|--------------|
| Stranger demo / verify / CI badges | **CPU** (Linux / Actions / Codespaces) | Public gates only — no MPS, no API keys, **no CUDA** |
| Product overnight breed / `TIME_BUDGET` train | **Apple Silicon MPS** | Train fitness **`val_bpb`** — **not** a CUDA claim, **not** lab AUROC |
| Future CUDA | Checklist only | Empty boxes = **no claim** — see [`compute-paths.md`](compute-paths.md) |

---

## DIPTYCH companion pin

- Companion (not the same product): **[DIPTYCH](https://github.com/pandeyaby/DIPTYCH)** grades **2-safety / calibration** on paired probes.
- AOMB **emits** fixtures/probes; DIPTYCH **grades**. Do not merge the products.
- Do not fork DIPTYCH harness/product code into this repo beyond the existing adapter emit path.
- Local emit + gate: `./scripts/run_diptych_full8.sh` · architecture: [`docs/images/aomb-diptych-architecture.svg`](images/aomb-diptych-architecture.svg)

---

## Explicit non-claims

| Surface | Status |
|---------|--------|
| **Lab AUROC** | **`not_published`** — no invented AUROC; private numbers stay under [`lab/`](lab/) |
| **CUDA** | **No claim yet** — checklist only in [`compute-paths.md`](compute-paths.md) |
| **Public ranking card** | **`published_fixture_card` / harness smoke** — tiny-n synthetic; not production / field accuracy |
| **CRISP `val_bpb`** | Train fitness only when cited — **not** ranking accuracy |
| **Overnight `agent_loop`** | Mac + API keys — **out of scope** for stranger paths |

---

## Related

- **60-second verify:** [`stranger-60s.md`](stranger-60s.md)
- **Share snip (paste):** [`share-snip.md`](share-snip.md)
- **Contributing as a stranger:** [`contributing-stranger.md`](contributing-stranger.md)
- **Org trust (optional):** [`SECURITY.md`](../SECURITY.md) · [`SUPPORT.md`](../SUPPORT.md) · [`NOTICE`](../NOTICE) · [`CONTRIBUTING.md`](../CONTRIBUTING.md) · review ownership via [`.github/CODEOWNERS`](../.github/CODEOWNERS) (`@pandeyaby`)
- **License:** [`LICENSE`](../LICENSE)
- README blurb: [Public wins](../README.md#public-wins-what-an-outsider-can-verify-today)
- Compute honesty: [`compute-paths.md`](compute-paths.md) · DIPTYCH: [pandeyaby/DIPTYCH](https://github.com/pandeyaby/DIPTYCH)
- Three lanes: README · [`corpus-v1.md`](corpus-v1.md) · [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md)
- Lab lane: [`lab/`](lab/) (`not_published` by default)
