# Contributing as a stranger

**Audience:** outsiders with no Mac, no API keys, no private lab access.  
**Goal:** a small, honest PR that org adopters can trust.

`prepare.py` is **sacred** — do not edit it in contribution PRs.

---

## Before you open a PR

1. **Run stranger verify first** (CPU; no keys):
   ```bash
   STRANGER_FAST=1 ./scripts/stranger_verify.sh
   ```
   Or cite a green check without cloning: [`stranger-verify.md`](stranger-verify.md) · [`stranger-60s.md`](stranger-60s.md).
2. **No keys** — do not add Anthropic/OpenAI secrets, `.env` samples with tokens, or overnight `agent_loop` as a PR requirement.
3. **No AUROC claims in PRs** — lab AUROC stays `not_published`. Do not invent numbers. Public ranking card = **`published_fixture_card` / harness smoke** only.
4. **No fake CUDA** — empty boxes in [`compute-paths.md`](compute-paths.md) mean **no claim**.

---

## Point here (public surfaces)

| Doc | Why |
|-----|-----|
| [`public-wins.md`](public-wins.md) | What an outsider can verify **today** |
| [`share-snip.md`](share-snip.md) | Ready-to-paste cite blurb |
| [`compute-paths.md`](compute-paths.md) | CPU stranger vs MPS product train; CUDA checklist only |
| [`stranger-demo.md`](stranger-demo.md) / [`stranger-verify.md`](stranger-verify.md) | Clone vs cite-without-clone paths |

License / attribution: root [`LICENSE`](../LICENSE) + [`NOTICE`](../NOTICE).

---

## Keep PRs short

- Prefer docs / CI honesty / stranger-path polish over new metrics.
- Do not merge DIPTYCH product code into AOMB beyond the existing adapter emit path.
- Codespaces already prints the verify command; **do not** add `postCreateCommand` auto-run of `stranger_verify` unless a separate reviewed change asks for it.
