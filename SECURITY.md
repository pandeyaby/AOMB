# Security Policy

## Scope

**AOMB** is a **defensive** observability / evaluation research project: breed an infrastructure language model on telemetry, emit paired probes, and score surprise as an anomaly signal.

This is **not** an offensive-security toolkit. We do not publish or accept exploit write-ups, attack PoCs, or weaponized harnesses.

Companion grading lives in **[DIPTYCH](https://github.com/pandeyaby/DIPTYCH)** (2-safety / calibration on paired probes). AOMB **emits**; DIPTYCH **grades**. Do not merge the products.

`prepare.py` is **sacred** — never edit it to “enable” claims, CUDA, or metrics.

## Honest claims (no fake metrics)

- **No invented AUROC.** Lab AUROC stays `not_published` until an independent publish path is green.
- Public ranking card = **`published_fixture_card` / harness smoke** only — not production or field accuracy.
- **No fake CUDA** numbers. Empty boxes in [`docs/compute-paths.md`](docs/compute-paths.md) mean **no claim**.

Stranger (outsider) verify path — CPU, no keys: [`docs/public-wins.md`](docs/public-wins.md) · [`docs/stranger-verify.md`](docs/stranger-verify.md) · [`docs/contributing-stranger.md`](docs/contributing-stranger.md).

## Reporting a vulnerability

1. Prefer **[GitHub private vulnerability reporting](https://github.com/pandeyaby/AOMB/security/advisories/new)** (Security Advisories) when available.
2. Otherwise open a **private** channel with the maintainer; do **not** file a public issue with exploit details, payloads, or PoCs.
3. Include: affected surface (docs / scripts / CI / adapter emit), impact on defensive eval honesty, and a minimal repro that does **not** include an attack PoC.

We will acknowledge in good faith and keep disclosure coordinated. Please give a reasonable window before any public discussion.

## Explicitly out of scope

| Request | Response |
|---------|----------|
| Exploit / PoC / weaponized repro | **Refused** — report impact + defensive fix only |
| Invented AUROC / marketing metrics | **Refused** — see [`docs/public-wins.md`](docs/public-wins.md) |
| Fake CUDA / GPU claims | **Refused** — see [`docs/compute-paths.md`](docs/compute-paths.md) |
| Editing `prepare.py` for a “win” | **Refused** — sacred |
| Forking DIPTYCH product code into AOMB | **Refused** — companion only; adapter emit path stays thin |

## Related

- Public wins (what outsiders can verify): [`docs/public-wins.md`](docs/public-wins.md)
- Contributing as a stranger: [`docs/contributing-stranger.md`](docs/contributing-stranger.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Compute honesty: [`docs/compute-paths.md`](docs/compute-paths.md)
- DIPTYCH: [pandeyaby/DIPTYCH](https://github.com/pandeyaby/DIPTYCH)
