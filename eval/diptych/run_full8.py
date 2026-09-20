#!/usr/bin/env python3
"""Run full-8 DIPTYCH probe gates for AOMB and emit coverage matrix.

Also supports ``--emit-dir`` to regenerate probe-pair JSON from fixtures
(same path as ``python -m eval.diptych.emit``).

Never invents AUROC / val_bpb. Never stub-passes. CUDA gate stays skipped.
``prepare.py`` is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.diptych.emit import (
    EXIT_EMIT_ERROR,
    EXIT_PATH_ERROR,
    EXIT_REFUSED_FLAG,
    REFUSED_METRIC_FLAGS,
    emit_probes,
    _refuse_loud_flags,
)
from eval.diptych.gates import run_gates, write_matrix
from eval.diptych.contract import ContractError


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _refuse_loud_flags(argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "paired-probes" / "full8_gate_report.json",
        help="Gate report JSON path",
    )
    p.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "coverage" / "matrix.json",
        help="Coverage matrix JSON path",
    )
    p.add_argument(
        "--emit-dir",
        type=Path,
        default=None,
        help=(
            "Also regenerate probe-pair JSON under this directory "
            "(deterministic fixture emit; same as eval.diptych.emit)"
        ),
    )
    p.add_argument(
        "--emit-only",
        action="store_true",
        help="Only emit probe JSON (--emit-dir required); skip full-8 gate",
    )
    args = p.parse_args(argv)

    if args.emit_only and args.emit_dir is None:
        print(
            "ERROR: --emit-only requires --emit-dir DIR.\n"
            "  Example: python -m eval.diptych.run_full8 --emit-only "
            "--emit-dir /tmp/diptych-emit\n"
            "  Or:      python -m eval.diptych.emit --out /tmp/diptych-emit",
            file=sys.stderr,
        )
        return EXIT_PATH_ERROR

    if args.emit_dir is not None:
        try:
            manifest = emit_probes(out_dir=args.emit_dir.resolve(), dry_run=False)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return EXIT_PATH_ERROR
        except (ContractError, ValueError) as e:
            print(f"ERROR: emit refused: {e}", file=sys.stderr)
            return EXIT_EMIT_ERROR
        print(
            f"emitted {manifest['n_probes']} probe pairs → {args.emit_dir} "
            f"(schema={manifest['diptych_schema']})"
        )
        if args.emit_only:
            print("EMIT OK")
            return 0

    report = run_gates()
    write_matrix(report.matrix, args.matrix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"diptych_schema={report.matrix.get('diptych_schema')} source={report.matrix.get('source_row')}")
    print("operator matrix (aomb column; green only after gate_axis_mutate):")
    for op, cell in report.matrix.get("operators", {}).items():
        power = report.axis_power.get(op, {})
        print(
            f"  {op:12} aomb={cell.get('aomb')} "
            f"axis_power={cell.get('axis_power')} "
            f"mutate={power.get('baseline_verdict')}→{power.get('mutated_verdict')}"
        )
    if report.failures:
        print("FAILURES:")
        for f in report.failures:
            print(f"  [{f.gate}] {f.detail}")
    print(f"report -> {args.out}")
    print(f"matrix -> {args.matrix}")
    print("GATE", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


# Re-export for tests / shell sync checks
__all__ = [
    "main",
    "REFUSED_METRIC_FLAGS",
    "EXIT_REFUSED_FLAG",
    "EXIT_PATH_ERROR",
    "EXIT_EMIT_ERROR",
]


if __name__ == "__main__":
    raise SystemExit(main())
