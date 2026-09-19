# Architecture — AOMB emit → DIPTYCH grade

Companion diagram for the IEEE / product one-pager path.

**Story:** breed/score loop → fixtures → DIPTYCH paired probes (2-safety / calibration).

| Artifact | Path |
|----------|------|
| SVG (preferred) | [`../images/aomb-diptych-architecture.svg`](../images/aomb-diptych-architecture.svg) |
| Mermaid source | [`../diagrams/aomb-diptych-architecture.mmd`](../diagrams/aomb-diptych-architecture.mmd) |

![AOMB breed/score → fixtures → DIPTYCH paired probes](../images/aomb-diptych-architecture.svg)

## Boundaries

- **AOMB** emits fixtures (`diptych-probes/`, public ranking card smoke). Training fitness stays **`val_bpb`**.
- **[DIPTYCH](https://github.com/pandeyaby/DIPTYCH)** grades hyperproperty / 2-safety on **paired** probes. Calibration claims live there — not as AOMB AUROC heroes.
- Products stay separate. Companion layer only.
- **`aomb=green`** in `coverage/matrix.json` requires **`gate_axis_mutate`** power-on-axis (`axis_power=true`). Twin conf/viol alone is not enough.

## Reproduce locally

```bash
./scripts/run_diptych_full8.sh
```

CI: `.github/workflows/diptych-adapter-gate.yml` (always on `push` to `main`).
