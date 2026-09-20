#!/usr/bin/env python3
"""Emit DIPTYCH probe-pair JSON from committed AOMB fixtures (deterministic).

Product path: strangers regenerate validated ``diptych_schema=0.2`` probe pairs
from ``diptych-probes/`` without inventing AUROC / val_bpb or stub-passing.

Does **not** invent lab AUROC, published ranking, or val_bpb. Does **not**
hardcoded-pass / stub-pass operators. ``prepare.py`` is untouched.

Usage::

    python -m eval.diptych.emit --out /tmp/diptych-emit
    python -m eval.diptych.emit --out /tmp/diptych-emit --operator SIGNFLIP
    python -m eval.diptych.emit --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.aomb import validate_aomb_probe
from eval.diptych import OPERATORS, SCHEMA, SOURCE, STUB_MARKERS
from eval.diptych.contract import ContractError, load_probe

PROBES = ROOT / "diptych-probes"
ROLES = ("conforming", "violating")

# Loud refusals — emit regenerates fixture probe JSON only (never invent metrics).
# Keep in sync with scripts/run_diptych_full8.sh and eval.diptych.run_full8.
REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--invent-metrics",
        "--invent-auroc",
        "--claim-auroc",
        "--readme-hero",
        "--publish-readme",
        "--hero-auroc",
        "--val-bpb",
        "--invent-val-bpb",
        "--stub-pass",
        "--hardcoded-pass",
        "--force-pass",
        "--fake-green",
        "--cuda",
    }
)

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_PATH_ERROR = 2
EXIT_EMIT_ERROR = 3


def _refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on invent / stub-pass / AUROC flags before argparse."""
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            print(
                f"ERROR: Refusing '{key}'.\n"
                "  DIPTYCH emit regenerates committed fixture probe-pair JSON only.\n"
                "  Never invents AUROC / val_bpb / published ranking.\n"
                "  Never stub-passes or hardcoded-passes operators.\n"
                "  CUDA gate stays skipped — use fixtures + gate_axis_mutate.\n"
                "  Use --out DIR, --operator OP, --dry-run, or --help.",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED_FLAG)


def probe_path(probes_root: Path, op: str, role: str) -> Path:
    return probes_root / op / role / "probe.json"


def list_fixture_probes(
    probes_root: Path = PROBES,
    *,
    operators: tuple[str, ...] | None = None,
) -> list[tuple[str, str, Path]]:
    """Return (operator, role, path) for every fixture probe to emit."""
    ops = operators if operators is not None else OPERATORS
    out: list[tuple[str, str, Path]] = []
    for op in ops:
        for role in ROLES:
            out.append((op, role, probe_path(probes_root, op, role)))
    return out


def _refuse_stub_content(path: Path, doc: dict[str, Any]) -> None:
    """Refuse stub markers / asymmetric stub-pass shapes before writing."""
    text = json.dumps(doc, sort_keys=True)
    for marker in STUB_MARKERS:
        if marker in text:
            raise ContractError(
                f"refusing stub emit for {path}: contains stub token {marker!r}"
            )
    role = doc.get("control_role")
    verdict = doc.get("expected_verdict")
    if role == "violating" and verdict == "pass":
        raise ContractError(
            f"refusing stub-pass emit for {path}: violating expected_verdict=pass"
        )
    if role == "conforming" and verdict == "fail":
        raise ContractError(
            f"refusing stub emit for {path}: conforming expected_verdict=fail"
        )
    traces = doc.get("traces") or []
    if len(traces) < 2:
        raise ContractError(f"refusing stub emit for {path}: traces length < 2")


def load_and_validate_probe(path: Path) -> dict[str, Any]:
    """Load fixture probe, validate envelope + AOMB axes, refuse stubs."""
    if not path.is_file():
        raise FileNotFoundError(f"missing fixture probe: {path}")
    # Stub-pass / stub-token check on raw JSON before envelope (loud refuse).
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError(f"refusing emit for {path}: envelope must be object")
    _refuse_stub_content(path, raw)
    doc = load_probe(path)
    validate_aomb_probe(doc)
    return doc


def emit_probe_json(doc: dict[str, Any]) -> str:
    """Deterministic probe JSON (sorted keys, stable indent, trailing newline)."""
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def emit_probes(
    *,
    out_dir: Path | None,
    probes_root: Path = PROBES,
    operators: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate fixtures and emit probe-pair JSON under ``out_dir``.

    Returns a manifest dict describing what was (or would be) emitted.
    """
    entries = list_fixture_probes(probes_root, operators=operators)
    missing_display: list[str] = []
    for _, _, p in entries:
        if not p.is_file():
            try:
                missing_display.append(str(p.relative_to(ROOT)))
            except ValueError:
                missing_display.append(str(p))
    if missing_display:
        raise FileNotFoundError(
            "missing fixture probe(s):\n  " + "\n  ".join(missing_display)
        )

    emitted: list[dict[str, Any]] = []
    for op, role, src in entries:
        doc = load_and_validate_probe(src)
        rel = f"{op}/{role}/probe.json"
        record: dict[str, Any] = {
            "operator": op,
            "control_role": role,
            "probe_id": doc.get("probe_id"),
            "source_path": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
            "emit_path": rel,
            "expected_verdict": doc.get("expected_verdict"),
            "coupling": doc.get("coupling"),
        }
        if not dry_run:
            if out_dir is None:
                raise ValueError("--out DIR required unless --dry-run")
            dest = out_dir / op / role / "probe.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(emit_probe_json(doc), encoding="utf-8")
            record["written"] = str(dest)
        else:
            record["written"] = None
        emitted.append(record)

    manifest: dict[str, Any] = {
        "diptych_schema": SCHEMA,
        "source": SOURCE,
        "mode": "dry-run" if dry_run else "emit",
        "n_probes": len(emitted),
        "operators": list(operators if operators is not None else OPERATORS),
        "probes": emitted,
        "honesty": (
            "Fixture probe-pair JSON only. Never invents AUROC / val_bpb. "
            "Never stub-passes. Companion DIPTYCH grades; AOMB emits."
        ),
    }
    if not dry_run and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        man_path = out_dir / "emit_manifest.json"
        man_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(man_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _refuse_loud_flags(argv)

    p = argparse.ArgumentParser(
        description=(
            "Emit DIPTYCH probe-pair JSON from committed AOMB fixtures. "
            "Deterministic regenerate for strangers. Never invents AUROC/val_bpb; "
            "never stub-passes. prepare.py untouched."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for regenerated probe pairs (required unless --dry-run)",
    )
    p.add_argument(
        "--probes-root",
        type=Path,
        default=PROBES,
        help="Fixture root (default: diptych-probes/)",
    )
    p.add_argument(
        "--operator",
        action="append",
        dest="operators",
        metavar="OP",
        help="Emit only this operator (repeatable). Default: all 8.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fixtures and print manifest; do not write files",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print emit manifest as JSON on stdout",
    )
    args = p.parse_args(argv)

    if not args.dry_run and args.out is None:
        print(
            "ERROR: --out DIR is required unless --dry-run.\n"
            "  Example: python -m eval.diptych.emit --out /tmp/diptych-emit\n"
            "  Or:      python -m eval.diptych.emit --dry-run",
            file=sys.stderr,
        )
        return EXIT_PATH_ERROR

    ops: tuple[str, ...] | None = None
    if args.operators:
        unknown = [op for op in args.operators if op.upper() not in OPERATORS]
        if unknown:
            print(
                f"ERROR: unknown operator(s): {unknown}. "
                f"Known: {list(OPERATORS)}",
                file=sys.stderr,
            )
            return EXIT_PATH_ERROR
        # Preserve OPERATORS order for determinism
        wanted = {op.upper() for op in args.operators}
        ops = tuple(op for op in OPERATORS if op in wanted)

    probes_root = args.probes_root.resolve()
    if not probes_root.is_dir():
        print(
            f"ERROR: probes root not found: {probes_root}\n"
            "  Expected committed fixtures under diptych-probes/.",
            file=sys.stderr,
        )
        return EXIT_PATH_ERROR

    try:
        manifest = emit_probes(
            out_dir=None if args.dry_run else args.out.resolve(),
            probes_root=probes_root,
            operators=ops,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PATH_ERROR
    except (ContractError, ValueError) as e:
        print(f"ERROR: emit refused: {e}", file=sys.stderr)
        return EXIT_EMIT_ERROR

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        mode = manifest["mode"]
        print(
            f"diptych_emit schema={manifest['diptych_schema']} "
            f"source={manifest['source']} mode={mode} n={manifest['n_probes']}"
        )
        for row in manifest["probes"]:
            dest = row.get("written") or "(dry-run)"
            print(
                f"  {row['operator']:12} {row['control_role']:11} "
                f"→ {dest}"
            )
        if manifest.get("manifest_path"):
            print(f"manifest -> {manifest['manifest_path']}")
        print("EMIT", "OK" if mode == "emit" else "DRY-RUN")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
