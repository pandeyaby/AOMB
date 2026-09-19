---
name: Bug report
about: Something broken in docs, scripts, CI, or stranger verify — keep claims honest
title: "[bug] "
labels: []
assignees: []
---

## What broke

(One sentence. Docs / scripts / CI / adapter emit — not a metric claim.)

## Stranger verify / CI cite

Paste a **green** [`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) run URL, or confirm you ran:

```bash
STRANGER_FAST=1 ./scripts/stranger_verify.sh
```

- Run / Action URL:
- Commit SHA (if known):

## Expected vs actual

-

## Honesty (required)

- [ ] I did **not** invent AUROC (lab AUROC stays `not_published`)
- [ ] I did **not** invent CUDA / production accuracy claims
- [ ] I did **not** propose editing `prepare.py`

Index: https://github.com/pandeyaby/AOMB/blob/main/docs/public-wins.md · security: https://github.com/pandeyaby/AOMB/blob/main/SECURITY.md
