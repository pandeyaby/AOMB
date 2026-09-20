"""Pure helpers: extract factual measured val_bpb from log text.

Never invents a number. Missing / malformed / non-finite / exploded → None.

Used by morning_report (git subjects + train logs) for Mac measured-fill later.
Does not touch prepare.py. Does not invent AUROC.
"""

from __future__ import annotations

import math
import re

# train.py final summary: "val_bpb:          0.430912"
_TRAIN_LINE_RE = re.compile(r"(?m)^val_bpb:\s+(\S+)\s*$")

# agent_loop commit subject: "[val_bpb=0.4309] [Δ=-0.0123] ..."
_COMMIT_EQ_RE = re.compile(r"val_bpb=([^\s\[\]]+)")

# Reject exploded / nonsense fitness (same spirit as agent_loop.parse_val_bpb).
_MAX_SANE_VAL_BPB = 50.0


def _coerce_measured(token: str) -> float | None:
    """Turn a raw token into a finite measured val_bpb, or None (refuse invent)."""
    raw = (token or "").strip().strip("[](){}").rstrip(",;")
    if not raw:
        return None
    # Explicit non-values sometimes printed by broken runs
    if raw.lower() in {"nan", "inf", "-inf", "+inf", "none", "null", "n/a", "na", "pending"}:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    if val < 0.0 or val > _MAX_SANE_VAL_BPB:
        return None
    return val


def parse_val_bpb_from_train_log(text: str) -> float | None:
    """Extract the FINAL ``val_bpb:`` line from train.py (or smoke) log text.

    Returns None when absent or malformed — never invents a substitute.
    """
    if not text:
        return None
    matches = _TRAIN_LINE_RE.findall(text)
    if not matches:
        return None
    return _coerce_measured(matches[-1])


def parse_val_bpb_from_commit_message(msg: str) -> float | None:
    """Extract ``val_bpb=`` from a git commit subject / morning-report line.

    Returns None when absent or malformed — never invents a substitute.
    """
    if not msg or "val_bpb" not in msg:
        return None
    m = _COMMIT_EQ_RE.search(msg)
    if not m:
        return None
    return _coerce_measured(m.group(1))


def parse_val_bpb(text: str) -> float | None:
    """Prefer train-log ``val_bpb:`` form; fall back to commit ``val_bpb=``.

    Factual extract only. None if neither yields a sane measured value.
    """
    got = parse_val_bpb_from_train_log(text)
    if got is not None:
        return got
    return parse_val_bpb_from_commit_message(text)
