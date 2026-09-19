## Summary

(One short paragraph. Prefer docs / CI honesty / stranger-path polish.)

## Stranger verify / CI (required when touching probes or docs)

Paste a **green** [`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml) run URL, or confirm:

```bash
STRANGER_FAST=1 ./scripts/stranger_verify.sh
```

- Run / Action URL:
- N/A because (docs-only / no probe change):

## Honesty (required)

- [ ] I did **not** invent AUROC (lab AUROC stays `not_published`)
- [ ] I did **not** invent CUDA / production accuracy claims
- [ ] I did **not** edit `prepare.py` (`prepare.py` is sacred)
- [ ] I read [`docs/public-wins.md`](https://github.com/pandeyaby/AOMB/blob/main/docs/public-wins.md) and [`docs/contributing-stranger.md`](https://github.com/pandeyaby/AOMB/blob/main/docs/contributing-stranger.md)

## Checklist

- [ ] Change is small and claim-honest
- [ ] No secrets / API keys / overnight `agent_loop` as a PR requirement
