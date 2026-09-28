## ZERODAY Antares localization (CI)

> **Human review required.** File-level localization candidates — **not** proof of exploitability. No auto-merge. No exploits, PoCs, or payloads.

| | |
|--|--|
| Advisory | `ghsa-gc5v-m9x4-r6x2` → `CWE-377` |
| Mode | `live` (live Antares — local completions endpoint) |
| Model | `fdtn-ai/antares-1b` |
| Findings | **1** ranked file(s) |
| Incomplete | no |

### Ranked candidate files

#### 1. `corpus/ingest/tale_stream_extract.py`

- **Title:** Insecure Temporary File
- **CWEs:** CWE-377
- **Evidence:** `corpus/ingest/tale_stream_extract.py` — Antares submitted this file as a localization candidate.

<details>
<summary>Exploration trace (1 steps)</summary>

1. **other** `antares query` — Live Antares CLI run (trace not present in report.json; see Antares private history under ANTARES_DATA_DIR).

</details>

### Posture

- Localization only · not exploitability proof
- SARIF uploaded at **note** severity (GitHub Code Scanning)
- Foundry Detector-lane **candidate** — true-positive waits for human triage
- Sister pieces: Foundry Security Spec · CodeGuard (compose, don’t replace)

_Powered by [ZERODAY](https://github.com/pandeyaby/ZERODAY) around Cisco Foundation AI [Antares](https://cisco-foundation-ai.github.io/antares/)._
