# ZERODAY scan — AOMB

> **Human review required.** Candidates, not proof of exploitability. No exploit code. No auto-merge.

Rules: 10 CWEs. Antares (`fdtn-ai/antares-1b`): 8 CWEs chosen for this repository by `antares plan`, 164 tool calls in 63s.

## By CWE

| CWE | Rules | Antares | Both |
|-----|------:|--------:|-----:|
| CWE-22 | 0 | 0 | 0 |
| CWE-78 | 0 | 1 | 0 |
| CWE-94 | 1 | 1 | 0 |
| CWE-502 | 2 | 1 | 1 |
| CWE-327 | 0 | 1 | 0 |
| CWE-77 | 0 | 9 | 0 |
| CWE-338 | 0 | 4 | 0 |
| CWE-409 | 0 | 2 | 0 |

## Files

### 1. `demo_anomaly.py` — CWE-338, CWE-94

Flagged by: antares + rules

- line 139: CWE-94: Value is built at runtime (not a constant) — check whether it can carry untrusted input. [rule cwe-94/code-eval]

  ```
      sys.modules["prepare"] = module
      exec(compile(source, str(REPO_ROOT / "prepare.py"), "exec"), module.__dict__)
      return module
  ```
- Antares (CWE-338): Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)

### 2. `prepare.py` — CWE-502

Flagged by: antares + rules

- line 230: CWE-502: Value is built at runtime (not a constant) — check whether it can carry untrusted input. [rule cwe-502/deser]

  ```
          with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
              enc = pickle.load(f)
          return cls(enc)
  ```
- Antares (CWE-502): Deserialization of Untrusted Data

### 3. `agent_loop.py` — CWE-77, CWE-78, CWE-94

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')
- Antares (CWE-78): Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')
- Antares (CWE-94): Improper Control of Generation of Code ('Code Injection')

### 4. `best_val_bpb.py` — CWE-338

Flagged by: antares

- Antares (CWE-338): Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)

### 5. `corpus/ingest/fetch_tale_of_errors.py` — CWE-327

Flagged by: antares

- Antares (CWE-327): Use of a Broken or Risky Cryptographic Algorithm

### 6. `corpus/ingest/tale_capped_pipeline.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 7. `corpus/ingest/tale_stream_extract.py` — CWE-409

Flagged by: antares

- Antares (CWE-409): Improper Handling of Highly Compressed Data (Data Amplification)

### 8. `eval/in_domain.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 9. `eval/report.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 10. `eval/run_public_ranking_card.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 11. `generate_observability_corpus.py` — CWE-338

Flagged by: antares

- Antares (CWE-338): Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)

### 12. `morning_report.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 13. `tests/test_stranger_scripts.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 14. `tests/test_tale_script_refusals.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 15. `tests/test_tale_stream_extract.py` — CWE-409

Flagged by: antares

- Antares (CWE-409): Improper Handling of Highly Compressed Data (Data Amplification)

### 16. `tests/test_train_refusals.py` — CWE-77

Flagged by: antares

- Antares (CWE-77): Improper Neutralization of Special Elements used in a Command ('Command Injection')

### 17. `visualize_corpus.py` — CWE-338

Flagged by: antares

- Antares (CWE-338): Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)

### 18. `eval/score_cli.py` — CWE-502

Flagged by: rules

- line 412: CWE-502: Value is built at runtime (not a constant) — check whether it can carry untrusted input. [rule cwe-502/deser-torch-load]

  ```
          )
      payload = torch.load(ckpt, map_location="cpu", weights_only=False)
      if isinstance(payload, dict) and "model_state_dict" in payload:
  ```

