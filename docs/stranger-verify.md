# Stranger verify — cite without cloning

**Audience:** outsiders who want a green public check they can link, without a local Mac/MPS setup or API keys.

**Check name (badge-friendly):** `stranger-verify`

**Pair path (clone first):** [`stranger-demo.md`](stranger-demo.md) / `./scripts/stranger_demo.sh` — same gates, local entry. Prefer demo when you already have a clone and want optional full CPU `--with-model` by default.

---

## What this proves (when green)

| Proven | How |
|--------|-----|
| DIPTYCH full-8 emit path + `gate_axis_mutate` | all 8 operators `aomb=green` with `axis_power=true` |
| Public ranking card harness smoke | default **baselines-only ε** (`STRANGER_FAST=1`); optional CPU `--with-model` via Actions dispatch |

Runs on **Linux** (GitHub Actions / Codespaces). **No Apple MPS. No API keys. No GPU spend.**

---

## What this does **not** prove

- **Lab AUROC** — stays `not_published`. **No invented AUROC.**
- **Production / field accuracy** — fixture pack is **tiny-n synthetic** harness smoke only.
- **MPS product train** / Uber CRISP overnight `val_bpb` as ranking accuracy.
- **Overnight `agent_loop`** (needs keys; usually Mac).

Honesty parent docs: [`public-ranking-card-v1.md`](public-ranking-card-v1.md) · README three lanes · [`lab/`](lab/) · [`stranger-demo.md`](stranger-demo.md).

---

## Cite without cloning

1. **Open in Codespaces (one click)** — [codespaces.new/pandeyaby/AOMB](https://codespaces.new/pandeyaby/AOMB) (uses [`.devcontainer/`](../.devcontainer/)). Then:

```bash
STRANGER_FAST=1 ./scripts/stranger_verify.sh
```

2. **Run workflow (one click)** — [`stranger-verify` → Run workflow](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) (`workflow_dispatch`). Optional **`full_model=true`** installs CPU torch and runs ranking-card `--with-model` — still **not** lab AUROC / MPS train.
3. **Badge / past runs** — green check on the same workflow page; open a successful run → share that URL.
4. **Clone (optional)** — same command after `pip install pyarrow numpy rustbpe tiktoken` (or `uv sync`). For the fuller local default (model smoke when torch is present), use [`stranger_demo.sh`](../scripts/stranger_demo.sh) instead — see [`stranger-demo.md`](stranger-demo.md).

**60-second cheatsheet:** [`stranger-60s.md`](stranger-60s.md). Full outsider index: [`public-wins.md`](public-wins.md).

`workflow_dispatch` input alias: **`full_model=false`** (default) ↔ `STRANGER_FAST=1` (baselines-only ε); `full_model=true` ↔ `STRANGER_FAST=0` (CPU `--with-model` only — still not lab AUROC / CUDA).

---

## Script behavior

[`scripts/stranger_verify.sh`](../scripts/stranger_verify.sh):

- **Delegates** to [`scripts/stranger_demo.sh`](../scripts/stranger_demo.sh) with `STRANGER_FAST` (demo is the shared runner).
- Fallback (if demo were absent): vendors `./scripts/run_diptych_full8.sh` + ranking-card `--baselines-only --check-eps`.

Default `STRANGER_FAST=1`. Set `STRANGER_FAST=0` for full fixture-model CPU smoke when torch is installed.

`prepare.py` is never touched. No DIPTYCH harness contamination — AOMB emits probes only.

---

## Related

- **60-second verify:** [`stranger-60s.md`](stranger-60s.md)
- Outsider landing: [`public-wins.md`](public-wins.md)
- Clone-first stranger path: [`stranger-demo.md`](stranger-demo.md) / `scripts/stranger_demo.sh`
- Compute honesty: [`compute-paths.md`](compute-paths.md) · DIPTYCH: [pandeyaby/DIPTYCH](https://github.com/pandeyaby/DIPTYCH)
- Adapter gate CI: [`.github/workflows/diptych-adapter-gate.yml`](../.github/workflows/diptych-adapter-gate.yml)
- Ranking card CI: [`.github/workflows/public-ranking-card-v1.yml`](../.github/workflows/public-ranking-card-v1.yml)
- Cite CI: [`.github/workflows/stranger-verify.yml`](../.github/workflows/stranger-verify.yml)
