# Lab public pack v0 (redacted)

Public-safe labeled sessions derived from local lab captures.

## Lane

| Pack | Role |
|------|------|
| `corpus/fixtures/lab_public_pack_v0/` | Redacted **labeled** pack people can download with the repo |
| Private `lab/captures/` | Raw captures — stay on the operator machine |
| `public_ranking_card_v1` | Synthetic CI/harness card — different lane |

**claim_status=`not_published`.** This redacted pack is a sample, not the eval corpus. The published lab result (pooled captures, zero-shot) is in [`ranking-validation.md`](ranking-validation.md). Don't paste that AUROC here as if this pack produced it.

## Redaction

IPs, UUIDs, long hex ids, emails, and hostnames stripped; service names remapped to `svc-*`.

## Smoke

```bash
uv run python -m eval.run_eval \
  --capture corpus/fixtures/lab_public_pack_v0 \
  --scores-from length \
  --out-dir /tmp/lab-public-pack-v0-length
```
