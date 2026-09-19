"""AOMB → DIPTYCH adapter (diptych_schema 0.2).

Validates probe-pair envelopes and required channel sketches from
docs/paired-probes/diptych/OPERATOR_TABLE.md. Does not invent AUROC/scores.
"""

from __future__ import annotations

from typing import Any

from eval.diptych import CRN_REQUIRED, OPERATORS
from eval.diptych.contract import ContractError, validate_envelope

# Exact AOMB channel sketches (OPERATOR_TABLE + DIPTYCH confirmation).
AXIS_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "FREEZEDRY": ("channels.graded.values", "meta.freeze_channels", "meta.decision_fingerprint"),
    "RESEED": ("channels.stability.values", "meta.seed", "meta.epsilon"),
    "SCHEMAX": ("channels.schema.keys",),
    "SIGNFLIP": ("meta.signflip_channel",),
    "SATEXTEND": ("meta.sat_lo", "meta.sat_hi"),
    "HISTSWAP": ("channels.history.values", "meta.hist_splice_at"),
    "TRAJSWAP": ("channels.trajectory.values", "channels.closed_loop_residual.values"),
    "VARSCALE": ("channels.variance_proxy.values", "meta.var_scale"),
}


def _has_path(trace: dict[str, Any], dotted: str) -> bool:
    parts = dotted.split(".")
    cur: Any = {"channels": trace.get("channels"), "meta": trace.get("meta")}
    # paths are channels.X.values or meta.Y
    if parts[0] == "channels":
        block = (trace.get("channels") or {}).get(parts[1])
        if not isinstance(block, dict):
            return False
        if len(parts) == 3:
            return parts[2] in block and bool(block[parts[2]])
        return True
    if parts[0] == "meta":
        return parts[1] in (trace.get("meta") or {})
    return False


def validate_aomb_probe(doc: dict[str, Any]) -> dict[str, Any]:
    """Validate envelope + AOMB axis presence for one probe pair."""
    doc = validate_envelope(doc)
    if doc.get("source") != "aomb":
        raise ContractError("adapters.aomb requires source=aomb")
    op = doc["operator"]
    if op not in OPERATORS:
        raise ContractError(f"unknown operator {op}")
    if op in CRN_REQUIRED and doc.get("coupling") != "crn_closed_loop":
        raise ContractError(f"{op} requires coupling=crn_closed_loop")
    reqs = AXIS_REQUIREMENTS[op]
    for tr in doc["traces"]:
        for path in reqs:
            if not _has_path(tr, path):
                raise ContractError(f"{op}: missing {path} on trace {tr.get('trace_id')}")
    # SIGNFLIP also needs the named channel values
    if op == "SIGNFLIP":
        for tr in doc["traces"]:
            target = tr["meta"].get("signflip_channel")
            if not target or target not in tr.get("channels", {}):
                raise ContractError("SIGNFLIP: channels.<signflip_channel>.values required")
            if "values" not in tr["channels"][target]:
                raise ContractError("SIGNFLIP: target channel missing values")
    # SATEXTEND needs clipped target values
    if op == "SATEXTEND":
        for tr in doc["traces"]:
            target = tr["meta"].get("sat_channel", "actuator")
            if target not in tr.get("channels", {}) or "values" not in tr["channels"][target]:
                raise ContractError("SATEXTEND: clipped target channel values required")
    # Forbidden score fields already rejected by validate_envelope
    return doc


def describe_axes() -> dict[str, tuple[str, ...]]:
    """Return the canonical AOMB channel sketch map."""
    return dict(AXIS_REQUIREMENTS)
