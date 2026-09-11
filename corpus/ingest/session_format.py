"""
Shared session / event text format for AOMB parquet shards.

Must stay compatible with prepare.py (column "text") and the heuristics
used by visualize_corpus.py / demo_anomaly.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def format_ts(dt: datetime) -> str:
    """ISO-8601 UTC with millisecond precision, trailing Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def format_ts_from_unix_nano(nano: int | float | None) -> str:
    if nano is None:
        return format_ts(datetime.now(timezone.utc))
    # Accept ns, µs, or ms heuristically
    n = int(nano)
    if n > 1_000_000_000_000_000:  # ns
        seconds = n / 1_000_000_000
    elif n > 1_000_000_000_000:  # µs
        seconds = n / 1_000_000
    else:  # ms
        seconds = n / 1_000
    return format_ts(datetime.fromtimestamp(seconds, tz=timezone.utc))


def _kv(key: str, value: Any) -> str:
    if value is None:
        return f"{key}=n/a"
    if isinstance(value, float):
        return f"{key}={value:.6g}"
    s = str(value).replace("\n", " ").replace(" ", "_") if key in {"msg", "op"} else str(value)
    s = s.replace("\n", " ").strip()
    if " " in s:
        s = s.replace(" ", "_")
    return f"{key}={s}"


def meta_line(
    *,
    source: str,
    window: str = "unknown",
    capture_id: str = "",
    fault: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    parts = [f"# aomb_meta source={source} window={window}"]
    if capture_id:
        parts.append(f"capture_id={capture_id}")
    if fault:
        parts.append(f"fault={fault}")
    if extra:
        for k, v in extra.items():
            parts.append(_kv(k, v))
    return " ".join(parts)


def span_event_line(
    *,
    ts: str,
    trace_id: str,
    span_id: str,
    parent: str = "n/a",
    op: str = "unknown",
    svc: str = "unknown",
    duration_ms: int | float = 0,
    status: str = "ok",
    http_status: Any = None,
    http_method: str | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    parts = [
        f"[ts={ts}]",
        "[src=OTel]",
        _kv("trace_id", trace_id or "n/a"),
        _kv("span_id", span_id or "n/a"),
        _kv("parent", parent or "n/a"),
        _kv("op", op),
        _kv("svc", svc),
        _kv("duration_ms", int(duration_ms)),
        _kv("status", status),
    ]
    if http_status is not None:
        parts.append(_kv("http_status", http_status))
    if http_method:
        parts.append(_kv("http_method", http_method))
    if extra:
        for k, v in extra.items():
            if v is not None and v != "":
                parts.append(_kv(k, v))
    return " ".join(parts)


def log_event_line(
    *,
    ts: str,
    level: str = "INFO",
    svc: str = "unknown",
    msg: str = "",
    trace_id: str | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    parts = [
        f"[ts={ts}]",
        "[src=OTelLog]",
        _kv("level", level),
        _kv("svc", svc),
        _kv("msg", msg or "n/a"),
    ]
    if trace_id:
        parts.append(_kv("trace_id", trace_id))
    if extra:
        for k, v in extra.items():
            if v is not None and v != "":
                parts.append(_kv(k, v))
    return " ".join(parts)


def join_session(lines: list[str]) -> str:
    return "\n".join(lines)
