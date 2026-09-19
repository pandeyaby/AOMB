"""AOMB DIPTYCH adapter — diptych_schema 0.2 full-8."""
from __future__ import annotations

OPERATORS: tuple[str, ...] = (
    "SIGNFLIP",
    "TRAJSWAP",
    "VARSCALE",
    "SATEXTEND",
    "HISTSWAP",
    "FREEZEDRY",
    "RESEED",
    "SCHEMAX",
)
CRN_REQUIRED = frozenset({"TRAJSWAP", "VARSCALE"})
SCHEMA = "0.2"
SOURCE = "aomb"
CONTROL_ROLES = frozenset({"conforming", "violating"})
VERDICTS = frozenset({"pass", "fail", "inconclusive"})
COUPLINGS = frozenset({"open_loop", "crn_closed_loop"})
STUB_MARKERS = ("TODO", "NotImplemented", "not_implemented", "STUB_OPERATOR", "hardcoded_pass")
