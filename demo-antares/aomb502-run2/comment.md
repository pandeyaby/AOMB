## ZERODAY Antares localization (CI)

> **Human review required.** File-level localization candidates — **not** proof of exploitability. No auto-merge. No exploits, PoCs, or payloads.

| | |
|--|--|
| Advisory | `CWE-502` → `CWE-502` |
| Mode | `live` (live Antares — local completions endpoint) |
| Model | `fdtn-ai/antares-1b` |
| Findings | **2** ranked file(s) |
| Incomplete | no |

### Ranked candidate files

#### 1. `eval/score_cli.py`

- **Title:** Deserialization of Untrusted Data
- **CWEs:** CWE-502
- **Evidence:** `eval/score_cli.py` — Antares submitted this file as a localization candidate.
- **Evidence:** `eval/score_cli.py:412-412` — Rules agree: Value is built at runtime (not a constant) — check whether it can carry untrusted input. [rule cwe-502/deser-torch-load]

#### 2. `prepare.py`

- **Title:** Deserialization of Untrusted Data
- **CWEs:** CWE-502
- **Evidence:** `prepare.py` — Antares submitted this file as a localization candidate.
- **Evidence:** `prepare.py:230-230` — Rules agree: Value is built at runtime (not a constant) — check whether it can carry untrusted input. [rule cwe-502/deser]

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
