"""
Robust telemetry timestamp parsing for lab captures and OTLP JSONL.

OTel ProtoJSON serializes fixed64 fields such as ``startTimeUnixNano`` /
``timeUnixNano`` as decimal *strings* so JSON numbers do not lose precision.
Python's ``datetime.fromisoformat`` must not see those digit strings first:
on 3.11+ it can mis-parse them as a distant Gregorian date (e.g. year 1726),
which then fails every provenance window match and yields label=unknown.

Also accepts RFC3339 / ISO-8601 strings and unix epoch int/float in ns, µs,
ms, or seconds (magnitude heuristic).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

# Epoch magnitude bands (approx. year 2001+):
#   seconds ~1e9, milliseconds ~1e12, microseconds ~1e15, nanoseconds ~1e18
_NS_MIN = 10**17
_US_MIN = 10**14
_MS_MIN = 10**11
_S_MIN = 10**9


def parse_telemetry_timestamp(value: Any) -> Optional[datetime]:
    """Parse OTel / lab timestamp fields into timezone-aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Numeric strings (ProtoJSON fixed64) before fromisoformat.
        if _looks_like_epoch_number(s):
            try:
                if "." in s or "e" in s.lower():
                    return parse_telemetry_timestamp(float(s))
                return parse_telemetry_timestamp(int(s))
            except (TypeError, ValueError, OverflowError):
                return None
        s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    if isinstance(value, bool):
        # bool is a subclass of int — never treat True/False as epoch.
        return None

    if isinstance(value, (int, float)):
        try:
            n = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return _from_epoch_int(n)

    return None


def _looks_like_epoch_number(s: str) -> bool:
    if s[0] in "+-" and len(s) > 1:
        s = s[1:]
    if not s:
        return False
    if s.isdigit():
        return True
    # float / scientific (collector may emit float nanos)
    try:
        float(s)
    except ValueError:
        return False
    return any(c.isdigit() for c in s) and not any(
        c.isalpha() and c not in "eE" for c in s
    )


def _from_epoch_int(n: int) -> datetime:
    a = abs(n)
    if a >= _NS_MIN:
        seconds = n / 1e9
    elif a >= _US_MIN:
        seconds = n / 1e6
    elif a >= _MS_MIN:
        seconds = n / 1e3
    elif a >= _S_MIN:
        seconds = float(n)
    else:
        # Sub-second epoch values are ambiguous; treat as milliseconds.
        seconds = n / 1e3
    return datetime.fromtimestamp(seconds, tz=timezone.utc)
