# Stranger verify — cite without cloning

**Audience:** outsiders who want a green public check they can link, without a local Mac/MPS setup or API keys.

**Check name (badge-friendly):** `stranger-verify`

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

Honesty parent docs: [`public-ranking-card-v1.md`](public-ranking-card-v1.md) · README three lanes · [`lab/`](lab/).

---

## Cite without cloning

1. **Badge / Actions** — green check on workflow [`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml). Open a successful run → share that URL.
2. **Codespaces** — [Open in GitHub Codespaces](https://codespaces.new/pandeyaby/AOMB) (uses [`.devcontainer/`](../.devcontainer/)). Then:

```bash
STRANGER_FAST=1 ./scripts/stranger_verify.sh
```

3. **Clone (optional)** — same command after `pip install pyarrow numpy rustbpe tiktoken` (or `uv sync`).

`workflow_dispatch` with **full_model=true** installs CPU torch and runs ranking-card `--with-model` — still **not** lab AUROC / MPS train.

---

## Script behavior

[`scripts/stranger_verify.sh`](../scripts/stranger_verify.sh):

- If [`scripts/stranger_demo.sh`](../scripts/stranger_demo.sh) is present (e.g. after the stranger-demo PR merges), **delegates** to it with `STRANGER_FAST`.
- Otherwise vendors: `./scripts/run_diptych_full8.sh` + ranking-card `--baselines-only --check-eps`.

Default `STRANGER_FAST=1`. Set `STRANGER_FAST=0` for full fixture-model CPU smoke when torch is installed.

`prepare.py` is never touched. No DIPTYCH harness contamination — AOMB emits probes only.

---

## Related

- Clone-first stranger path (when merged): [`stranger-demo.md`](stranger-demo.md) / `scripts/stranger_demo.sh`
- Adapter gate CI: [`.github/workflows/diptych-adapter-gate.yml`](../.github/workflows/diptych-adapter-gate.yml)
- Ranking card CI: [`.github/workflows/public-ranking-card-v1.yml`](../.github/workflows/public-ranking-card-v1.yml)
