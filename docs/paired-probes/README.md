# AOMB ↔ DIPTYCH adapter (diptych_schema 0.2)

AOMB **emits** fixtures; **[DIPTYCH](https://github.com/pandeyaby/DIPTYCH)** **grades** calibration as **2-safety hyperproperties** (coupled pairs, not single-run scores). Companion layer only — do not merge the products. AOMB’s full-8 adapters are already merged; this tree emits the product probes DIPTYCH grades.

Architecture: [`docs/images/aomb-diptych-architecture.svg`](../images/aomb-diptych-architecture.svg) · pipeline: [`docs/images/aomb-calibration-pipeline.svg`](../images/aomb-calibration-pipeline.svg) · paper notes [`docs/images/README.md`](../images/README.md).

**Pinned Origin URLs** (offline mirror pointers): [`diptych/README.md`](diptych/README.md).  
Offline vendor copies: [`diptych/`](diptych/) (CONTRACT, GATING, OPERATORS, aomb, OPERATOR_TABLE, ONEPAGER).

## Emit path

```bash
./scripts/run_diptych_full8.sh
```

Probes: `diptych-probes/<OP>/{conforming,violating}/probe.json`  
Coverage: `coverage/matrix.json` (AOMB column)  
Report: `reports/paired-probes/full8_gate_report.json`  
Adapter validator: `adapters/aomb.py`

## Adapter CI / `gate_axis_mutate` (required)

Workflow: [`.github/workflows/diptych-adapter-gate.yml`](../../.github/workflows/diptych-adapter-gate.yml) (runs on every push to `main`).

- Manifest + contrast + axis presence must pass for all 8 operators.
- **`gate_axis_mutate`** must flip conforming `pass` → mutated `fail` on the operator axis only (`eval/diptych/gates.py`).
- `aomb=green` in `coverage/matrix.json` **only** when twin OK **and** `axis_power=true`.
- Forbidden sole edits (do not count as power): `expected_verdict`-only flip, SARIF rename, AUROC inject.
- `prepare.py` remains sacred / untouched by this path.

## AOMB channel sketches (exact)

| Operator | Coupling | Channels / meta |
|----------|----------|-----------------|
| FREEZEDRY | open_loop | freeze `rng`/`clock` → identical `channels.graded.values` (+ `meta.decision_fingerprint`); leak → diverge |
| RESEED | open_loop | `meta.seed` differs; `channels.stability.values` + `meta.epsilon` |
| SCHEMAX | open_loop | `channels.schema.keys` equal vs rename/drop |
| SIGNFLIP | open_loop | `meta.signflip_channel` + values; odd-symmetric holds vs breaks |
| SATEXTEND | open_loop | `meta.sat_lo` / `meta.sat_hi` + clipped target values |
| HISTSWAP | open_loop | `channels.history` + `meta.hist_splice_at` |
| TRAJSWAP | **crn_closed_loop** | `channels.trajectory.*` + `channels.closed_loop_residual.values` |
| VARSCALE | **crn_closed_loop** | `channels.variance_proxy` / `meta.var_scale` mean-matched — **not AUROC** |

`expected_verdict`: `pass` \| `fail` \| `inconclusive` only. Cell green only for conforming→pass and violating→fail.

## What this cannot claim

- Not production ranking accuracy / AUROC
- Lab ranking stays `not_published` under `docs/lab/`
- Public ranking card remains **harness smoke**
- `prepare.py` untouched
