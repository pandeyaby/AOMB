# AOMB ↔ DIPTYCH adapter (diptych_schema 0.2)

AOMB **emits** fixtures; **[DIPTYCH](https://github.com/pandeyaby/DIPTYCH)** **grades** 2-safety / calibration on paired probes. Companion layer only — do not merge the products.

Architecture: [`docs/images/aomb-diptych-architecture.svg`](../images/aomb-diptych-architecture.svg) · paper notes [`docs/images/README.md`](../images/README.md).

**Pinned Origin URLs** (canonical): [`diptych/README.md`](diptych/README.md).  
Offline vendor copies: [`diptych/`](diptych/) (CONTRACT, GATING, OPERATORS, aomb, OPERATOR_TABLE, ONEPAGER).

## Emit path

```bash
./scripts/run_diptych_full8.sh
```

Probes: `diptych-probes/<OP>/{conforming,violating}/probe.json`  
Coverage: `coverage/matrix.json` (AOMB column)  
Report: `reports/paired-probes/full8_gate_report.json`  
Adapter validator: `adapters/aomb.py`

**`aomb=green` requires `gate_axis_mutate`:** twin conf/viol is necessary but not sufficient. CI asserts `axis_power=true` on every operator (`diptych-adapter-gate` on every push to `main`).

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
