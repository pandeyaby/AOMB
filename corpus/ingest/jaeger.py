"""
Parse Jaeger HTTP API JSON traces into SpanRecord lists.

Compatible with Uber CRISP / Tale of Errors sanitized production dumps
(each file is typically one Jaeger API response or one trace object).

Session key = traceID (multi-service spans share one session).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from corpus.ingest.adapters.base import SpanRecord


def _tags_to_dict(tags: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(tags, list):
        return out
    for item in tags:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key is None:
            continue
        out[str(key)] = item.get("value")
    return out


def _us_to_dt(us: Any) -> Optional[datetime]:
    if us is None:
        return None
    try:
        return datetime.fromtimestamp(int(us) / 1_000_000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _parent_span_id(span: dict[str, Any]) -> str:
    for ref in span.get("references") or []:
        if not isinstance(ref, dict):
            continue
        if str(ref.get("refType", "")).upper() in {"CHILD_OF", "CHILD-OF", "FOLLOWS_FROM"}:
            # Prefer CHILD_OF; FOLLOWS_FROM only if no CHILD_OF found later
            if str(ref.get("refType", "")).upper().replace("-", "_") == "CHILD_OF":
                return str(ref.get("spanID") or ref.get("spanId") or "")
    for ref in span.get("references") or []:
        if isinstance(ref, dict) and ref.get("spanID"):
            return str(ref.get("spanID"))
    return ""


def _status_from_tags(attrs: dict[str, Any]) -> str:
    err = attrs.get("error")
    if err is True or str(err).lower() in {"true", "1"}:
        return "error"
    code = attrs.get("http.status_code") or attrs.get("http.statusCode")
    try:
        if code is not None and int(code) >= 500:
            return "error"
    except (TypeError, ValueError):
        pass
    return "ok"


def iter_traces(doc: Any) -> Iterator[dict[str, Any]]:
    """Yield individual Jaeger trace objects from various wrapper shapes."""
    if doc is None:
        return
    if isinstance(doc, list):
        for item in doc:
            yield from iter_traces(item)
        return
    if not isinstance(doc, dict):
        return
    if "data" in doc and isinstance(doc["data"], list):
        for item in doc["data"]:
            if isinstance(item, dict) and ("spans" in item or "traceID" in item):
                yield item
        return
    if "spans" in doc or "traceID" in doc or "traceId" in doc:
        yield doc


def jaeger_trace_to_spans(trace: dict[str, Any]) -> list[SpanRecord]:
    processes = trace.get("processes") or {}
    trace_id = str(trace.get("traceID") or trace.get("traceId") or "")
    spans_out: list[SpanRecord] = []
    for span in trace.get("spans") or []:
        if not isinstance(span, dict):
            continue
        tid = str(span.get("traceID") or span.get("traceId") or trace_id)
        pid_key = span.get("processID") or span.get("processId")
        proc = processes.get(pid_key, {}) if isinstance(processes, dict) else {}
        if not proc and isinstance(span.get("process"), dict):
            proc = span["process"]
        attrs = _tags_to_dict(span.get("tags"))
        duration_us = span.get("duration")
        try:
            duration_ms = float(duration_us) / 1000.0 if duration_us is not None else 0.0
        except (TypeError, ValueError):
            duration_ms = 0.0
        spans_out.append(
            SpanRecord(
                trace_id=tid,
                span_id=str(span.get("spanID") or span.get("spanId") or ""),
                parent_span_id=_parent_span_id(span),
                name=str(span.get("operationName") or span.get("operation_name") or ""),
                service_name=str(
                    proc.get("serviceName")
                    or proc.get("service_name")
                    or attrs.get("service.name")
                    or "unknown"
                ),
                start_time=_us_to_dt(span.get("startTime") or span.get("start_time")),
                duration_ms=duration_ms,
                status_code=_status_from_tags(attrs),
                status_message="",
                attributes=attrs,
            )
        )
    return spans_out


def load_jaeger_json(doc: Any) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    for trace in iter_traces(doc):
        spans.extend(jaeger_trace_to_spans(trace))
    return spans
