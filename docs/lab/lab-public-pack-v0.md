# Lab public pack v0 (redacted)

Public-safe labeled sessions derived from local lab captures.

## Lane

| Pack | Role |
|------|------|
| `corpus/fixtures/lab_public_pack_v0/` | Redacted **labeled** pack people can download with the repo |
| Private `lab/captures/` | Raw captures — stay on the operator machine |
| `public_ranking_card_v1` | Synthetic CI/harness card — different lane |

**claim_status=`not_published`.** Do not market AUROC from this pack on the README hero until Abhinav greenlights a ranking run write-up. Never paste private pooled-lab AUROC (e.g. 0.766) here as a public claim.

## Redaction

IPs, UUIDs, long hex ids, emails, and hostnames stripped; service names remapped to `svc-*`.

## Smoke

```bash
uv run python -m eval.run_eval \
  --capture corpus/fixtures/lab_public_pack_v0 \
  --scores-from length \
  --out-dir /tmp/lab-public-pack-v0-length
```
