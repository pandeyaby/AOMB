#!/usr/bin/env python3
"""Run full-8 DIPTYCH probe gates for AOMB and emit coverage matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.diptych.gates import run_gates, write_matrix


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "paired-probes" / "full8_gate_report.json",
    )
    p.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "coverage" / "matrix.json",
    )
    args = p.parse_args(argv)

    report = run_gates()
    write_matrix(report.matrix, args.matrix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"diptych_schema={report.matrix.get('diptych_schema')} source={report.matrix.get('source_row')}")
    print("operator matrix (aomb column):")
    for op, cell in report.matrix.get("operators", {}).items():
        print(f"  {op:12} aomb={cell.get('aomb')}")
    if report.failures:
        print("FAILURES:")
        for f in report.failures:
            print(f"  [{f.gate}] {f.detail}")
    print(f"report -> {args.out}")
    print(f"matrix -> {args.matrix}")
    print("GATE", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
