"""
Convert OTLP-derived SpanRecord / LogRecord lists into AOMB session documents.

Hypothesis: group by trace_id → multi-line session; attach window label via
capture metadata (TimeWindow), not invented per-event anomaly flags.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from corpus.ingest.adapters.base import (
    LogRecord,
    SourceBundle,
    SpanRecord,
    TimeWindow,
    resolve_window,
)
from corpus.ingest.session_format import (
    format_ts,
    join_session,
    log_event_line,
    meta_line,
    span_event_line,
)


def _status_from_code(code: str | int | None) -> str:
    if code is None:
        return "ok"
    if isinstance(code, int):
        # OTLP: 0=UNSET, 1=OK, 2=ERROR
        return {0: "ok", 1: "ok", 2: "error"}.get(code, "ok")
    s = str(code).lower()
    if s in {"2", "error", "status_code_error", "error"}:
        return "error"
    return "ok"


def _span_line(span: SpanRecord) -> str:
    attrs = span.attributes or {}
    http_status = (
        attrs.get("http.status_code")
        or attrs.get("http.response.status_code")
        or attrs.get("http_status")
    )
    http_method = attrs.get("http.method") or attrs.get("http.request.method")
    # Never invent wall-clock "now" — missing OTel timestamps must stay missing
    # so provenance window labeling does not silently drift off-window.
    ts = format_ts(span.start_time) if span.start_time else "n/a"
    return span_event_line(
        ts=ts,
        trace_id=span.trace_id,
        span_id=span.span_id,
        parent=span.parent_span_id or "n/a",
        op=span.name or "unknown",
        svc=span.service_name or "unknown",
        duration_ms=span.duration_ms,
        status=_status_from_code(span.status_code),
        http_status=http_status,
        http_method=http_method,
        extra={"db": attrs.get("db.system")} if attrs.get("db.system") else None,
    )


def _log_line(log: LogRecord) -> str:
    ts = format_ts(log.timestamp) if log.timestamp else "n/a"
    return log_event_line(
        ts=ts,
        level=(log.severity or "INFO").upper(),
        svc=log.service_name or "unknown",
        msg=log.body or "",
        trace_id=log.trace_id or None,
    )


def _session_window(
    spans: list[SpanRecord],
    logs: list[LogRecord],
    windows: list[TimeWindow],
) -> TimeWindow:
    times: list[datetime] = []
    for s in spans:
        if s.start_time:
            times.append(s.start_time)
    for lg in logs:
        if lg.timestamp:
            times.append(lg.timestamp)
    if not times:
        return TimeWindow(label="unknown")
    # Majority label by midpoint of span times
    mid = sorted(times)[len(times) // 2]
    return resolve_window(mid, windows)


_MIN = datetime.min.replace(tzinfo=timezone.utc)


def bundle_to_sessions(
    bundle: SourceBundle,
    *,
    min_events: int = 1,
    include_meta: bool = True,
) -> list[tuple[str, TimeWindow]]:
    """
    Return list of (session_text, window) from a SourceBundle.

    Spans grouped by trace_id. Logs with matching trace_id join that session.
    Orphan logs (no trace_id) are grouped into small time-bucket sessions
    labeled by the enclosing capture window.
    """
    by_trace: dict[str, list[SpanRecord]] = defaultdict(list)
    for span in bundle.spans:
        tid = (span.trace_id or "").strip()
        if not tid:
            tid = f"_orphan_span_{span.span_id or id(span)}"
        by_trace[tid].append(span)

    logs_by_trace: dict[str, list[LogRecord]] = defaultdict(list)
    orphan_logs: list[LogRecord] = []
    for lg in bundle.logs:
        tid = (lg.trace_id or "").strip()
        if tid and tid in by_trace:
            logs_by_trace[tid].append(lg)
        elif tid:
            logs_by_trace[tid].append(lg)
        else:
            orphan_logs.append(lg)

    sessions: list[tuple[str, TimeWindow]] = []

    all_trace_ids = sorted(set(by_trace) | set(logs_by_trace))
    for tid in all_trace_ids:
        spans = sorted(
            by_trace.get(tid, []),
            key=lambda s: s.start_time or _MIN,
        )
        logs = sorted(
            logs_by_trace.get(tid, []),
            key=lambda lg: lg.timestamp or _MIN,
        )
        if len(spans) + len(logs) < min_events:
            continue
        window = _session_window(spans, logs, bundle.windows)
        lines: list[str] = []
        if include_meta:
            lines.append(
                meta_line(
                    source=bundle.source_id,
                    window=window.label,
                    capture_id=bundle.capture_id,
                    fault=window.fault,
                )
            )
        # Interleave by timestamp
        events: list[tuple[datetime, str]] = []
        for s in spans:
            events.append((s.start_time or _MIN, _span_line(s)))
        for lg in logs:
            events.append((lg.timestamp or _MIN, _log_line(lg)))
        events.sort(key=lambda x: x[0])
        lines.extend(e[1] for e in events)
        sessions.append((join_session(lines), window))

    # Orphan logs: one session per capture window bucket (or single mixed)
    if orphan_logs:
        by_label: dict[str, list[LogRecord]] = defaultdict(list)
        for lg in orphan_logs:
            w = resolve_window(lg.timestamp, bundle.windows)
            by_label[w.label].append(lg)
        label_to_window = {w.label: w for w in bundle.windows}
        for label, group in by_label.items():
            group = sorted(group, key=lambda lg: lg.timestamp or _MIN)
            if len(group) < min_events:
                continue
            window = label_to_window.get(label, TimeWindow(label=label))
            lines = []
            if include_meta:
                lines.append(
                    meta_line(
                        source=bundle.source_id,
                        window=window.label,
                        capture_id=bundle.capture_id,
                        fault=window.fault,
                    )
                )
            lines.extend(_log_line(lg) for lg in group)
            sessions.append((join_session(lines), window))

    return sessions


def sessions_only(bundle: SourceBundle, **kwargs) -> list[str]:
    return [text for text, _ in bundle_to_sessions(bundle, **kwargs)]
