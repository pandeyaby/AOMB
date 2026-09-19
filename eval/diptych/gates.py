"""DIPTYCH gates for AOMB (GATING.md / CONTRACT.md).

Fails on: incomplete manifest, missing violating twin, identical twins,
asymmetric-verdict failure, stub markers, wrong coupling on CRN ops.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.diptych import CRN_REQUIRED, OPERATORS, SCHEMA, SOURCE, STUB_MARKERS
from eval.diptych.contract import ContractError, load_probe, validate_envelope
from eval.diptych.grade import GradeResult, grade_document

ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / "diptych-probes"
MATRIX = ROOT / "coverage" / "matrix.json"


@dataclass
class Failure:
    gate: str
    detail: str


@dataclass
class Report:
    ok: bool
    failures: list[Failure] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    matrix: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": [f.__dict__ for f in self.failures],
            "results": self.results,
            "matrix": self.matrix,
        }


def _path(op: str, role: str) -> Path:
    return PROBES / op / role / "probe.json"


def gate_manifest() -> list[Failure]:
    out: list[Failure] = []
    if not PROBES.is_dir():
        return [Failure("manifest", f"missing {PROBES}")]
    for op in OPERATORS:
        for role in ("conforming", "violating"):
            p = _path(op, role)
            if not p.is_file():
                out.append(Failure("manifest", f"missing {p.relative_to(ROOT)}"))
    return out


def gate_stubs(paths: list[Path]) -> list[Failure]:
    out: list[Failure] = []
    # Probe JSON: reject empty traces / inverted expected_verdict / literal stub tokens in values
    for path in paths:
        text = path.read_text(encoding="utf-8")
        doc = json.loads(text)
        if not doc.get("traces") or len(doc["traces"]) < 2:
            out.append(Failure("stub", f"{path.relative_to(ROOT)} empty/short traces"))
        if path.parent.name == "violating" and doc.get("expected_verdict") == "pass":
            out.append(Failure("stub", f"{path.relative_to(ROOT)} violating expected pass"))
        if path.parent.name == "conforming" and doc.get("expected_verdict") == "fail":
            out.append(Failure("stub", f"{path.relative_to(ROOT)} conforming expected fail"))
        for marker in ("TODO", "NotImplemented", "STUB_OPERATOR", "hardcoded_pass"):
            if f'"{marker}"' in text or f": {marker}" in text:
                out.append(Failure("stub", f"{path.relative_to(ROOT)} contains stub token {marker!r}"))
    # Grader sources: reject NotImplementedError / ellipsis bodies / hardcoded True returns
    for path in (ROOT / "eval" / "diptych").glob("*.py"):
        if path.name in {"gates.py", "__init__.py"}:
            continue
        src = path.read_text(encoding="utf-8")
        if "raise NotImplementedError" in src or "raise NotImplemented" in src:
            out.append(Failure("stub", f"{path.relative_to(ROOT)} raises NotImplemented"))
        if re.search(r"return True\s*#\s*(stub|hardcoded|TODO)", src):
            out.append(Failure("stub", f"{path.relative_to(ROOT)} hardcoded True stub"))
        if re.search(r"def grade_\w+\([^)]*\):\s*\.\.\.", src):
            out.append(Failure("stub", f"{path.relative_to(ROOT)} ellipsis grader"))
    return out


def gate_contrast(op: str, conf: dict, viol: dict) -> list[Failure]:
    out: list[Failure] = []
    if json.dumps(conf["traces"], sort_keys=True) == json.dumps(viol["traces"], sort_keys=True):
        out.append(Failure("contrast", f"{op}: identical twins (no axis mutation)"))
    if conf["expected_verdict"] == viol["expected_verdict"]:
        out.append(Failure("contrast", f"{op}: expected_verdict not asymmetric"))
    return out


def gate_axis(op: str, conf: dict, viol: dict) -> list[Failure]:
    out: list[Failure] = []
    if op in CRN_REQUIRED:
        for role, doc in (("conforming", conf), ("violating", viol)):
            if doc.get("coupling") != "crn_closed_loop":
                out.append(Failure("axis", f"{op}/{role}: coupling must be crn_closed_loop"))
    checks = {
        "RESEED": lambda d: (
            "stability" in d["traces"][0]["channels"]
            and "epsilon" in d["traces"][0]["meta"]
            and d["traces"][0]["meta"].get("seed") != d["traces"][1]["meta"].get("seed")
        ),
        "SCHEMAX": lambda d: "schema" in d["traces"][0]["channels"]
        and isinstance(d["traces"][0]["channels"]["schema"].get("keys"), list),
        "FREEZEDRY": lambda d: (
            "graded" in d["traces"][0]["channels"]
            and "freeze_channels" in d["traces"][0]["meta"]
            and "decision_fingerprint" in d["traces"][0]["meta"]
        ),
        "SIGNFLIP": lambda d: "signflip_channel" in d["traces"][0]["meta"],
        "SATEXTEND": lambda d: "sat_lo" in d["traces"][0]["meta"] and "sat_hi" in d["traces"][0]["meta"],
        "HISTSWAP": lambda d: (
            "history" in d["traces"][0]["channels"]
            and "hist_splice_at" in d["traces"][0]["meta"]
        ),
        "TRAJSWAP": lambda d: (
            "trajectory" in d["traces"][0]["channels"]
            and "closed_loop_residual" in d["traces"][0]["channels"]
        ),
        "VARSCALE": lambda d: (
            "variance_proxy" in d["traces"][0]["channels"]
            and "var_scale" in d["traces"][0]["meta"]
        ),
    }
    fn = checks[op]
    for role, doc in (("conforming", conf), ("violating", viol)):
        try:
            if not fn(doc):
                out.append(Failure("axis", f"{op}/{role}: missing axis fields"))
        except Exception as e:  # noqa: BLE001
            out.append(Failure("axis", f"{op}/{role}: {e}"))
    return out


def run_gates() -> Report:
    failures = gate_manifest()
    if failures:
        return Report(ok=False, failures=failures)

    paths = [_path(op, role) for op in OPERATORS for role in ("conforming", "violating")]
    failures.extend(gate_stubs(paths))

    results: list[dict[str, Any]] = []
    matrix_ops: dict[str, Any] = {}

    for op in OPERATORS:
        conf = load_probe(_path(op, "conforming"))
        viol = load_probe(_path(op, "violating"))
        failures.extend(gate_contrast(op, conf, viol))
        failures.extend(gate_axis(op, conf, viol))

        grades: list[GradeResult] = []
        for doc in (conf, viol):
            try:
                validate_envelope(doc)
                g = grade_document(doc)
            except ContractError as e:
                failures.append(Failure("grade", f"{op}: {e}"))
                matrix_ops[op] = {"diptych_core": "pending", "zeroday": "pending", "aomb": "stub"}
                continue
            grades.append(g)
            results.append(g.to_dict())
            if not g.matches_expected:
                failures.append(
                    Failure(
                        "contrast",
                        f"{op}/{g.control_role}: expected {g.expected_verdict} got {g.actual_verdict} ({g.reason})",
                    )
                )

        green = (
            len(grades) == 2
            and all(g.matches_expected for g in grades)
            and grades[0].actual_verdict == "pass"
            and grades[1].actual_verdict == "fail"
        )
        matrix_ops[op] = {
            "diptych_core": "pending",
            "zeroday": "pending",
            "aomb": "green" if green else "pending",
        }

    matrix = {"diptych_schema": SCHEMA, "source_row": SOURCE, "operators": matrix_ops}
    ok = not failures and all(v["aomb"] == "green" for v in matrix_ops.values())
    if not ok and not failures:
        failures.append(Failure("matrix", "not all aomb cells green"))
    return Report(ok=ok, failures=failures, results=results, matrix=matrix)


def write_matrix(matrix: dict[str, Any], path: Path = MATRIX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
