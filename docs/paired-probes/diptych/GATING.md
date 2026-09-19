# How DIPTYCH gates thin implementations

> **Pinned Origin (normative source of truth):**  
> https://origin.cursor.com/abhinavpandey/tmp-c44b600dec44401a/raw/main/docs/adapters/GATING.md  
>
> This file is the AOMB **vendor mirror**. Prefer Origin when reachable.  
> *(Origin raw returned HTTP 404 “could not be routed” from CI/agent egress at vendor time; content below tracks the pinned `gate_axis_mutate` contract + live AOMB harness symbols.)*

1. **Manifest gate:** every source in `{diptych_core,zeroday,aomb}` must declare 8 operators × {conforming,violating} artifact paths (`eval/diptych/gates.py::gate_manifest`).
2. **Contrast gate:** for each pair, run harness; require asymmetric verdicts (pass vs fail) unless both honestly `inconclusive` with documented reason (inconclusive≠green). Identical twins → fail (`gate_contrast`).
3. **Axis presence gate:** conforming + violating must carry the operator's axis fields / coupling (`gate_axis`). `TRAJSWAP` / `VARSCALE` require `crn_closed_loop`.
4. **`gate_axis_mutate` (power-on-axis):** after contrast + axis presence, mutate *only* the operator axis on the conforming fixture and re-grade. Baseline must be `pass`; mutated must be `fail`. Axis fingerprint (`axis_fingerprint`) must change. Cosmetic sole edits do **not** count as power (see forbidden list). Implemented in `eval/diptych/gates.py::gate_axis_mutate` + `eval/diptych/mutate_axis.py`.
5. **Stub detectors:** reject strings/markers `TODO`, `NotImplemented`, `stub`, empty traces, `expected_verdict` hardcoded without running grader.
6. **Coverage matrix:** `coverage/matrix.json` may set `aomb=green` **only** when twin contrast **and** `gate_axis_mutate` both pass (`axis_power: true`). Do not keep `aomb=green` unless mutate gate passes.
7. **PR policy:** DIPTYCH flags product PRs that only smoke 1–2 operators; GRAX informed; coverage matrix stays non-green. **HOLD merge** on adapter PRs until full-8 + axis power are green.

## Mutation table (`gate_axis_mutate`)

Mutate **only** these axes on conforming (then re-grade → must fail):

| Operator   | Allowed axis mutation |
|------------|------------------------|
| FREEZEDRY  | clear `freeze_channels` or leak rng/clock into graded/fp |
| RESEED     | L∞(stability) > epsilon |
| SCHEMAX    | rename/drop schema key on trace b |
| SIGNFLIP   | flip `signflip_channel` values without odd-symmetric repair |
| SATEXTEND  | push value outside `legal_lo`/`legal_hi` |
| HISTSWAP   | break history/`alt_history` cross or set `history_corrupt` |
| TRAJSWAP   | break traj alignment or residual > bound (keep `crn_closed_loop`) |
| VARSCALE   | raise `var_scale`/proxy above `var_scale_bound` (keep mean-match honest) |

AOMB mutator one-liners (live `MUTATION_DESCRIPTIONS`):

- **FREEZEDRY** — clear `freeze_channels` and diverge graded series + fingerprint
- **RESEED** — inflate `stability.values` on twin B beyond `meta.epsilon`
- **SCHEMAX** — rename/drop a required `channels.schema.keys` entry on twin B
- **SIGNFLIP** — replace `-a` with `+a` on `signflip_channel`; break `sign_normalized_fingerprint`
- **SATEXTEND** — push a clipped value above `legal_hi` while keeping sat bounds covering it
- **HISTSWAP** — corrupt history at `hist_splice_at` and break `alt_history` cross-link
- **TRAJSWAP** — offset trajectory (break CRN noise) and blow `closed_loop_residual` past bound
- **VARSCALE** — raise `meta.var_scale` past `var_scale_bound` (CRN values / mean-match honest)

### Forbidden sole edits (do not count as power)

- `expected_verdict`-only flip
- SARIF level rename
- AUROC / fabricated scores

Negative coverage (AOMB mirrors DIPTYCH harness cosmetic-relabel rejection; DIPTYCH reference suite ≈64 tests):

- `tests/test_diptych_full8.py::test_verdict_only_flip_is_not_axis_power`
- `tests/test_diptych_full8.py::test_cosmetic_relabel_is_not_axis_power` (SARIF rename + AUROC inject)
- `tests/test_diptych_full8.py::test_forbidden_cosmetics_leave_axis_fingerprint_unchanged`

## AOMB wiring

- Entry: `./scripts/run_diptych_full8.sh` → `eval.diptych.run_full8` → `run_gates()`
- Order inside `run_gates()`: manifest → stubs → **contrast → axis presence → `gate_axis_mutate`** → grade twins → matrix
- Unit proof: `tests/test_diptych_full8.py::test_axis_mutate_flips_each_operator` (all 8)

Contract paths on DIPTYCH box (canonical until landed in repo):

- `/workspace/diptych-spec/adapters/CONTRACT.md`
- `/workspace/diptych-spec/adapters/GATING.md`
- `/workspace/diptych-spec/adapters/zeroday.md`
- `/workspace/diptych-spec/adapters/aomb.md`
- `/workspace/diptych-spec/OPERATORS.md`
