"""
Public-real adapter: smithclay/otel-demo-telemetry (Apache-2.0).

Expects local parquet trees downloaded by fetch_otel_demo.py:

  <input>/otlp_traces/**/*.parquet
  <input>/otlp_logs/**/*.parquet   (optional)

Column names follow the HF dataset / duckdb-otlp layout (snake_case).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from glob import glob
from typing import Any, Iterator, Optional

from corpus.ingest.adapters.base import (
    LogRecord,
    SourceAdapter,
    SourceBundle,
    SpanRecord,
    TimeWindow,
)

DATASET_ID = "smithclay/otel-demo-telemetry"
LICENSE = "Apache-2.0"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
CITATION = (
    "Hugging Face dataset smithclay/otel-demo-telemetry "
    "(OTLP from open-telemetry/opentelemetry-demo via duckdb-otlp); "
    "license Apache-2.0."
)


def _parse_attrs(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _to_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    # pandas / pyarrow may give Timestamp
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        n = int(value)
        if n > 1_000_000_000_000_000:
            return datetime.fromtimestamp(n / 1e9, tz=timezone.utc)
        if n > 1_000_000_000_000:
            return datetime.fromtimestamp(n / 1e6, tz=timezone.utc)
        return datetime.fromtimestamp(n / 1e3, tz=timezone.utc)
    return None


def _duration_ms(row: dict[str, Any], start: Optional[datetime]) -> float:
    for key in (
        "duration_time_unix_nano",
        "Duration",
        "duration",
        "duration_nano",
    ):
        if key in row and row[key] is not None:
            try:
                n = int(row[key])
                # duckdb-otlp stores duration in ns
                if n > 1_000_000:
                    return n / 1_000_000.0
                return float(n)
            except (TypeError, ValueError):
                pass
    return 0.0


def _status(row: dict[str, Any]) -> str:
    for key in ("status_code", "StatusCode", "status"):
        if key in row and row[key] is not None:
            return str(row[key])
    return "0"


def _read_parquet_rows(path: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.column_names
    return [
        {col: table.column(col)[i].as_py() for col in cols}
        for i in range(table.num_rows)
    ]


def _find_parquets(root: str, *subdirs: str) -> list[str]:
    files: list[str] = []
    for sub in subdirs:
        pattern = os.path.join(root, sub, "**", "*.parquet")
        files.extend(glob(pattern, recursive=True))
        # also flat
        files.extend(glob(os.path.join(root, sub, "*.parquet")))
    return sorted(set(files))


class OtelDemoHfAdapter(SourceAdapter):
    name = "otel_demo_hf"

    def load(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        max_spans = int(kwargs.get("max_spans", 0) or 0)
        max_logs = int(kwargs.get("max_logs", 0) or 0)

        trace_files = _find_parquets(input_path, "otlp_traces", "traces")
        log_files = _find_parquets(input_path, "otlp_logs", "logs")

        if not trace_files and not log_files:
            raise FileNotFoundError(
                f"No otlp_traces/ or otlp_logs/ parquet under {input_path}. "
                "Run: python -m corpus.ingest.fetch_otel_demo "
                "or place files manually (see docs/corpus-v1.md)."
            )

        spans: list[SpanRecord] = []
        for fp in trace_files:
            for row in _read_parquet_rows(fp):
                start = _to_dt(
                    row.get("start_time_unix_nano")
                    or row.get("Timestamp")
                    or row.get("start_time")
                )
                attrs = _parse_attrs(
                    row.get("span_attributes") or row.get("SpanAttributes")
                )
                spans.append(
                    SpanRecord(
                        trace_id=str(
                            row.get("trace_id") or row.get("TraceId") or ""
                        ),
                        span_id=str(
                            row.get("span_id") or row.get("SpanId") or ""
                        ),
                        parent_span_id=str(
                            row.get("parent_span_id")
                            or row.get("ParentSpanId")
                            or ""
                        ),
                        name=str(row.get("name") or row.get("SpanName") or ""),
                        service_name=str(
                            row.get("service_name")
                            or row.get("ServiceName")
                            or attrs.get("service.name")
                            or "unknown"
                        ),
                        start_time=start,
                        duration_ms=_duration_ms(row, start),
                        status_code=_status(row),
                        status_message=str(
                            row.get("status_status_message")
                            or row.get("StatusMessage")
                            or ""
                        ),
                        attributes=attrs,
                    )
                )
                if max_spans and len(spans) >= max_spans:
                    break
            if max_spans and len(spans) >= max_spans:
                break

        logs: list[LogRecord] = []
        for fp in log_files:
            for row in _read_parquet_rows(fp):
                attrs = _parse_attrs(
                    row.get("log_attributes")
                    or row.get("LogAttributes")
                    or row.get("attributes")
                )
                logs.append(
                    LogRecord(
                        timestamp=_to_dt(
                            row.get("timestamp")
                            or row.get("Timestamp")
                            or row.get("time_unix_nano")
                        ),
                        severity=str(
                            row.get("severity_text")
                            or row.get("SeverityText")
                            or row.get("severity")
                            or "INFO"
                        ),
                        body=str(
                            row.get("body")
                            or row.get("Body")
                            or row.get("message")
                            or ""
                        ),
                        service_name=str(
                            row.get("service_name")
                            or row.get("ServiceName")
                            or "unknown"
                        ),
                        trace_id=str(
                            row.get("trace_id") or row.get("TraceId") or ""
                        ),
                        attributes=attrs,
                    )
                )
                if max_logs and len(logs) >= max_logs:
                    break
            if max_logs and len(logs) >= max_logs:
                break

        times = [s.start_time for s in spans if s.start_time]
        times += [lg.timestamp for lg in logs if lg.timestamp]
        windows: list[TimeWindow] = []
        if times:
            windows.append(
                TimeWindow(
                    label="normal",
                    start=min(times),
                    end=max(times),
                    notes=(
                        "Public OTel Demo capture has no explicit incident "
                        "windows; labeled normal unless you supply custom windows."
                    ),
                )
            )

        yield SourceBundle(
            source_id=DATASET_ID,
            source_kind="public_real",
            license=LICENSE,
            license_url=LICENSE_URL,
            citation=CITATION,
            spans=spans,
            logs=logs,
            windows=windows,
            capture_id="otel-demo-hf",
            extra_provenance={
                "dataset": DATASET_ID,
                "trace_files": len(trace_files),
                "log_files": len(log_files),
                "input_path": os.path.abspath(input_path),
            },
        )
