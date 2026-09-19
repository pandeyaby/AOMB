# DIPTYCH adapter specs (pinned Origin sources)

**Schema:** `diptych_schema="0.2"`  
**Hyperproperty grading harness:** https://github.com/pandeyaby/DIPTYCH  
**Pinned Origin mirror (offline):** https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a

DIPTYCH grades calibration as 2-safety hyperproperties (coupled pairs). AOMB full-8 adapters are landed; adapter CI requires twin contrast **and** `gate_axis_mutate` before `aomb=green`.

Local Markdown copies in this directory are offline mirrors for CI/review. When specs drift, prefer the **GitHub DIPTYCH** docs, then the **pinned Origin raw URLs** below.

## Pinned Origin raw URLs (stable)

| Doc | Origin raw URL |
|-----|----------------|
| CONTRACT | https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a/raw/main/docs/adapters/CONTRACT.md |
| GATING | https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a/raw/main/docs/adapters/GATING.md |
| AOMB binding | https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a/raw/main/docs/adapters/aomb.md |
| OPERATORS | https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a/raw/main/docs/OPERATORS.md |
| ONEPAGER | https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a/raw/main/docs/adapters/ONEPAGER.md |

## Local vendor copies

| Doc | Path |
|-----|------|
| CONTRACT | [`CONTRACT.md`](CONTRACT.md) |
| GATING | [`GATING.md`](GATING.md) |
| AOMB | [`aomb.md`](aomb.md) |
| OPERATORS | [`OPERATORS.md`](OPERATORS.md) |
| OPERATOR_TABLE | [`OPERATOR_TABLE.md`](OPERATOR_TABLE.md) |
| ONEPAGER | [`ONEPAGER.md`](ONEPAGER.md) |

## Coverage note

DIPTYCH reports `diptych_core` green for all 8 operators. The **aomb** column is green when twin contrast **and** `gate_axis_mutate` both pass (`coverage/matrix.json`, adapter CI).

AOMB emit path: `diptych-probes/<OP>/{conforming,violating}/probe.json` · validator: `adapters/aomb.py` · gate: `./scripts/run_diptych_full8.sh` · CI: `.github/workflows/diptych-adapter-gate.yml`.
