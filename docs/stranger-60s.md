# Reproduce in 60 seconds — stranger cheatsheet

**Audience:** someone who never cloned AOMB.  
**Hardware:** CPU only (Actions / Codespaces). **No MPS. No API keys. No CUDA.**  
`prepare.py` is **sacred** — this path never touches it.

Companion (not this product): **[DIPTYCH](https://github.com/pandeyaby/DIPTYCH)** grades paired-probe 2-safety; AOMB **emits**, DIPTYCH **grades**.

---

## Steps

1. **Cite a green check (no clone)** — prefer this pinned **example** (or the live badge):
   - **Example green cite (main)** — 2026-09-19 · `7abdfd5` · [run 35471119285](https://github.com/pandeyaby/AOMB/actions/runs/35471119285) (`stranger-verify`)
   - **Canonical:** live [`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) badge / workflow — the pinned URL is an **example**, not a forever-frozen claim.
2. **Or run it yourself** — either:
   - **Codespaces:** [codespaces.new/pandeyaby/AOMB](https://codespaces.new/pandeyaby/AOMB) → then:
     ```bash
     STRANGER_FAST=1 ./scripts/stranger_verify.sh
     ```
   - **Actions `workflow_dispatch`:** [Run workflow](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) — leave **`full_model=false`** (default). That input is the UI alias for the fast path (`STRANGER_FAST=1` / baselines-only ε). Set `full_model=true` only for optional CPU torch `--with-model` smoke — still **not** lab AUROC.
3. **Read the outcome** — process exits 0 / Action is green.

---

## Green means

| Proven | Not claimed |
|--------|-------------|
| DIPTYCH full-8 emit + `gate_axis_mutate` → `aomb=green`, `axis_power=true` | **Lab AUROC** (`not_published` — **no invented AUROC**) |
| Ranking-card harness smoke (default: baselines-only ε) | Production / field accuracy |
| Public gates on **CPU** Linux | MPS product train / overnight `agent_loop` / **CUDA** |

Tiny-n fixture limits apply. Details: [`stranger-verify.md`](stranger-verify.md) · index: [`public-wins.md`](public-wins.md) · hardware honesty: [`compute-paths.md`](compute-paths.md).
