"""
SourceAdapter interface for AOMB corpus ingest.

Adapters load real telemetry into a common in-memory representation;
otlp_to_sessions.py turns that into prepare.py-compatible session docs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional


@dataclass
class TimeWindow:
    label: str  # "normal" | "incident" | "unknown"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    fault: str = ""
    notes: str = ""

    def contains(self, ts: datetime) -> bool:
        if self.start is None and self.end is None:
            return True
        if self.start is not None and ts < self.start:
            return False
        if self.end is not None and ts > self.end:
            return False
        return True


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    name: str = ""
    service_name: str = ""
    start_time: Optional[datetime] = None
    duration_ms: float = 0.0
    status_code: str = "ok"  # ok | error | unset
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogRecord:
    timestamp: Optional[datetime] = None
    severity: str = "INFO"
    body: str = ""
    service_name: str = ""
    trace_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceBundle:
    """One load from a source: spans, logs, windows, provenance stub."""

    source_id: str
    source_kind: str  # public_real | lab_capture
    license: str
    license_url: str = ""
    citation: str = ""
    spans: list[SpanRecord] = field(default_factory=list)
    logs: list[LogRecord] = field(default_factory=list)
    windows: list[TimeWindow] = field(default_factory=list)
    capture_id: str = ""
    extra_provenance: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(ABC):
    """Load a real telemetry source into SourceBundle(s)."""

    name: str = "base"

    @abstractmethod
    def load(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        """Yield one or more SourceBundles from input_path."""


def resolve_window(ts: Optional[datetime], windows: list[TimeWindow]) -> TimeWindow:
    if ts is None:
        return TimeWindow(label="unknown")
    for w in windows:
        if w.contains(ts):
            return w
    return TimeWindow(label="unknown")
