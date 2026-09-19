# AOMB ↔ DIPTYCH adapter (diptych_schema 0.2)

Canonical DIPTYCH specs are vendored under [`diptych/`](diptych/) (CONTRACT, GATING, OPERATORS, aomb, OPERATOR_TABLE, ONEPAGER).

## Emit path

```bash
./scripts/run_diptych_full8.sh
```

Probes: `diptych-probes/<OP>/{conforming,violating}/probe.json`  
Coverage: `coverage/matrix.json` (AOMB column)  
Report: `reports/paired-probes/full8_gate_report.json`

## Operator matrix (this PR)

| Operator | Coupling | Conforming | Violating | AOMB cell |
|----------|----------|------------|-----------|-----------|
| SIGNFLIP | open_loop | pass | fail | green when CI passes |
| TRAJSWAP | **crn_closed_loop** | pass | fail | green when CI passes |
| VARSCALE | **crn_closed_loop** | pass | fail | green when CI passes |
| SATEXTEND | open_loop | pass | fail | green when CI passes |
| HISTSWAP | open_loop | pass | fail | green when CI passes |
| FREEZEDRY | open_loop | pass | fail | green when CI passes |
| RESEED | open_loop | pass | fail | green when CI passes |
| SCHEMAX | open_loop | pass | fail | green when CI passes |

## Envelope (hard keys)

`diptych_schema`, `source`, `operator`, `coupling`, `probe_id`, `control_role`, `traces` (≥2), `expected_verdict`.

See [`diptych/CONTRACT.md`](diptych/CONTRACT.md). Do **not** invent AUROC / model-grade fields.

## What this cannot claim

- Not production ranking accuracy / AUROC
- Lab ranking stays `not_published` under `docs/lab/`
- Public ranking card remains **harness smoke** (`published_fixture_card`) — separate lane
- Single-trace / tiny-pair smoke cannot claim field performance
- `prepare.py` untouched

## Gates (must fail)

1. Missing operator or missing conforming/violating twin  
2. Identical twins / same expected_verdict  
3. Stub / TODO / NotImplemented / hardcoded pass  
4. TRAJSWAP or VARSCALE without `crn_closed_loop`  
5. Grader verdict ≠ expected_verdict on either control  
