"""
Lab-capture adapter: JSONL exported by lab/scripts/capture.sh.

Expected layout:

  <capture_dir>/
    provenance.json          # windows, faults, capture_id
    traces.jsonl             # one OTLP-ish span object per line
    logs.jsonl               # one log record per line (optional)

Span JSON (flexible keys — flat or OTLP resourceSpans):
  Flat:
  {
    "trace_id": "...", "span_id": "...", "parent_span_id": "...",
    "name": "GET /checkout", "service_name": "api",
    "start_time_unix_nano": 1.7e18, "duration_ms": 42,
    "status_code": "ok"|"error"|0|1|2,
    "attributes": {...}
  }
  OTel file exporter (ProtoJSON): resourceSpans with string startTimeUnixNano.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from corpus.ingest.adapters.base import (
    LogRecord,
    SourceAdapter,
    SourceBundle,
    SpanRecord,
    TimeWindow,
)
from corpus.ingest.timestamps import parse_telemetry_timestamp


def _parse_dt(value: Any):
    """Parse provenance / OTel timestamps (RFC3339, unix ms/ns, ProtoJSON strings)."""
    return parse_telemetry_timestamp(value)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _windows_from_provenance(prov: dict[str, Any]) -> list[TimeWindow]:
    out: list[TimeWindow] = []
    for w in prov.get("windows") or []:
        out.append(
            TimeWindow(
                label=str(w.get("label", "unknown")),
                start=_parse_dt(w.get("start")),
                end=_parse_dt(w.get("end")),
                fault=str(w.get("fault") or ""),
                notes=str(w.get("notes") or ""),
            )
        )
    return out


class LabCaptureAdapter(SourceAdapter):
    name = "lab_capture"

    def load(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        prov_path = os.path.join(input_path, "provenance.json")
        if not os.path.exists(prov_path):
            raise FileNotFoundError(
                f"Missing {prov_path}. Lab captures must include provenance.json "
                "(see lab/scripts/run_capture_session.sh)."
            )
        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)

        spans_raw = _load_jsonl(os.path.join(input_path, "traces.jsonl"))
        # Also accept collector file exporter dumps
        if not spans_raw:
            spans_raw = _load_jsonl(os.path.join(input_path, "spans.jsonl"))
        logs_raw = _load_jsonl(os.path.join(input_path, "logs.jsonl"))

        spans: list[SpanRecord] = []
        for row in spans_raw:
            # Flatten OTLP resourceSpans if present
            if "resourceSpans" in row:
                spans.extend(_flatten_otlp_traces(row))
                continue
            attrs = row.get("attributes") or {}
            start_raw = (
                row.get("startTimeUnixNano")
                or row.get("start_time_unix_nano")
                or row.get("start_time")
                or row.get("timestamp")
            )
            end_raw = (
                row.get("endTimeUnixNano")
                or row.get("end_time_unix_nano")
                or row.get("end_time")
            )
            start = _parse_dt(start_raw)
            dur = row.get("duration_ms")
            if dur is None and start_raw is not None and end_raw is not None:
                try:
                    dur = (int(end_raw) - int(start_raw)) / 1e6
                except (TypeError, ValueError):
                    # String/float nanos already handled by int(); else leave 0
                    start_dt = start
                    end_dt = _parse_dt(end_raw)
                    if start_dt is not None and end_dt is not None:
                        dur = (end_dt - start_dt).total_seconds() * 1000.0
                    else:
                        dur = 0
            spans.append(
                SpanRecord(
                    trace_id=str(row.get("trace_id") or row.get("traceId") or ""),
                    span_id=str(row.get("span_id") or row.get("spanId") or ""),
                    parent_span_id=str(
                        row.get("parent_span_id")
                        or row.get("parentSpanId")
                        or ""
                    ),
                    name=str(row.get("name") or ""),
                    service_name=str(
                        row.get("service_name")
                        or row.get("serviceName")
                        or attrs.get("service.name")
                        or "unknown"
                    ),
                    start_time=start,
                    duration_ms=float(dur or 0),
                    status_code=str(
                        row.get("status_code")
                        or (row.get("status") or {}).get("code")
                        or "ok"
                    ),
                    status_message=str(
                        row.get("status_message")
                        or (row.get("status") or {}).get("message")
                        or ""
                    ),
                    attributes=attrs if isinstance(attrs, dict) else {},
                )
            )

        logs: list[LogRecord] = []
        for row in logs_raw:
            if "resourceLogs" in row:
                logs.extend(_flatten_otlp_logs(row))
                continue
            attrs = row.get("attributes") or {}
            logs.append(
                LogRecord(
                    timestamp=_parse_dt(
                        row.get("timeUnixNano")
                        or row.get("observedTimeUnixNano")
                        or row.get("timestamp")
                        or row.get("time_unix_nano")
                        or row.get("observed_time_unix_nano")
                    ),
                    severity=str(
                        row.get("severity")
                        or row.get("severity_text")
                        or "INFO"
                    ),
                    body=str(row.get("body") or row.get("message") or ""),
                    service_name=str(
                        row.get("service_name")
                        or attrs.get("service.name")
                        or "unknown"
                    ),
                    trace_id=str(row.get("trace_id") or row.get("traceId") or ""),
                    attributes=attrs if isinstance(attrs, dict) else {},
                )
            )

        yield SourceBundle(
            source_id=str(prov.get("source_id") or "lab-aomb-stack"),
            source_kind="lab_capture",
            license=str(prov.get("license") or "Apache-2.0"),
            license_url=str(prov.get("license_url") or ""),
            citation=str(
                prov.get("citation")
                or "AOMB lab stack capture (lab/docker-compose.yml)"
            ),
            spans=spans,
            logs=logs,
            windows=_windows_from_provenance(prov),
            capture_id=str(prov.get("capture_id") or os.path.basename(input_path)),
            extra_provenance={
                "input_path": os.path.abspath(input_path),
                "provenance": prov,
            },
        )


def _attr_list_to_dict(attrs: Any) -> dict[str, Any]:
    if isinstance(attrs, dict):
        return attrs
    out: dict[str, Any] = {}
    if not isinstance(attrs, list):
        return out
    for item in attrs:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        val = item.get("value") or {}
        if key is None:
            continue
        for vk in (
            "stringValue",
            "intValue",
            "doubleValue",
            "boolValue",
        ):
            if vk in val:
                out[key] = val[vk]
                break
    return out


def _flatten_otlp_traces(doc: dict[str, Any]) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    for rs in doc.get("resourceSpans") or []:
        res_attrs = _attr_list_to_dict(
            (rs.get("resource") or {}).get("attributes")
        )
        svc = res_attrs.get("service.name", "unknown")
        for ss in rs.get("scopeSpans") or []:
            for sp in ss.get("spans") or []:
                start_ns = sp.get("startTimeUnixNano")
                end_ns = sp.get("endTimeUnixNano")
                dur = 0.0
                if start_ns is not None and end_ns is not None:
                    try:
                        dur = (int(end_ns) - int(start_ns)) / 1e6
                    except (TypeError, ValueError):
                        dur = 0.0
                status = sp.get("status") or {}
                spans.append(
                    SpanRecord(
                        trace_id=str(sp.get("traceId") or ""),
                        span_id=str(sp.get("spanId") or ""),
                        parent_span_id=str(sp.get("parentSpanId") or ""),
                        name=str(sp.get("name") or ""),
                        service_name=str(svc),
                        start_time=_parse_dt(start_ns),
                        duration_ms=dur,
                        status_code=str(status.get("code", "ok")),
                        status_message=str(status.get("message") or ""),
                        attributes=_attr_list_to_dict(sp.get("attributes")),
                    )
                )
    return spans


def _flatten_otlp_logs(doc: dict[str, Any]) -> list[LogRecord]:
    logs: list[LogRecord] = []
    for rl in doc.get("resourceLogs") or []:
        res_attrs = _attr_list_to_dict(
            (rl.get("resource") or {}).get("attributes")
        )
        svc = res_attrs.get("service.name", "unknown")
        for sl in rl.get("scopeLogs") or []:
            for lr in sl.get("logRecords") or []:
                body = lr.get("body") or {}
                if isinstance(body, dict):
                    body_s = str(
                        body.get("stringValue")
                        or body.get("string_value")
                        or body
                    )
                else:
                    body_s = str(body)
                logs.append(
                    LogRecord(
                        timestamp=_parse_dt(
                            lr.get("timeUnixNano")
                            or lr.get("observedTimeUnixNano")
                        ),
                        severity=str(
                            lr.get("severityText") or lr.get("severityNumber") or "INFO"
                        ),
                        body=body_s,
                        service_name=str(svc),
                        trace_id=str(lr.get("traceId") or ""),
                        attributes=_attr_list_to_dict(lr.get("attributes")),
                    )
                )
    return logs
