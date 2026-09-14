"""
Bring-your-own (BYO) telemetry adapter.

Accepts user-provided dumps without inventing fake events:

  1. OTLP JSONL directory — traces.jsonl / spans.jsonl / logs.jsonl
     (+ optional provenance.json for windows / capture_id / license)
  2. Jaeger JSON — directory of .json files or a single Jaeger API JSON file
  3. Parquet sessions — shard_*.parquet (or any parquet) with column ``text``
     already in AOMB session line format (prepare.py contract)

Detected format is recorded in provenance. Reuses session_format via
otlp_to_sessions for OTLP/Jaeger; parquet rows pass through as ready sessions.

Scoring / shard building from BYO dumps is **not** a public accuracy claim
until the labeled checklist in docs/public-accuracy-eval.md passes.
"""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from corpus.ingest.adapters.base import (
    SourceAdapter,
    SourceBundle,
    TimeWindow,
)
from corpus.ingest.adapters.lab_capture import (
    LabCaptureAdapter,
    _load_jsonl,
    _parse_dt,
    _windows_from_provenance,
)
from corpus.ingest.jaeger import load_jaeger_json
from corpus.ingest.otlp_to_sessions import bundle_to_sessions

FORMAT_OTLP_JSONL = "otlp_jsonl"
FORMAT_JAEGER_JSON = "jaeger_json"
FORMAT_PARQUET_SESSIONS = "parquet_sessions"


def detect_format(input_path: str) -> str:
    """
    Infer dump format from path layout.

    Priority:
      - parquet file, or directory containing *.parquet → parquet_sessions
      - directory with traces.jsonl / spans.jsonl / logs.jsonl → otlp_jsonl
      - .json file, or directory of Jaeger-looking JSON → jaeger_json
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"BYO input not found: {input_path}")

    if path.is_file():
        if path.suffix.lower() == ".parquet":
            return FORMAT_PARQUET_SESSIONS
        if path.suffix.lower() == ".json":
            return FORMAT_JAEGER_JSON
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return FORMAT_OTLP_JSONL
        raise ValueError(
            f"Unrecognized BYO file type {path.suffix!r}. "
            "Expected .parquet, .json (Jaeger), or .jsonl (OTLP)."
        )

    # Directory
    parquet_files = list(path.glob("*.parquet")) + list(path.glob("shard_*.parquet"))
    if parquet_files:
        return FORMAT_PARQUET_SESSIONS

    for name in ("traces.jsonl", "spans.jsonl", "logs.jsonl"):
        if (path / name).is_file():
            return FORMAT_OTLP_JSONL

    json_files = list(path.glob("*.json")) + list(path.glob("**/*.json"))
    # Exclude provenance.json-only dirs without traces
    json_files = [p for p in json_files if p.name != "provenance.json"]
    if json_files:
        return FORMAT_JAEGER_JSON

    raise ValueError(
        f"Cannot detect BYO format under {input_path}. "
        "Provide OTLP JSONL (traces.jsonl/spans.jsonl), Jaeger JSON, "
        "or parquet with a text column. See docs/byo-and-scorer.md."
    )


def _default_provenance(input_path: str, fmt: str) -> dict[str, Any]:
    return {
        "source_id": "byo-user-dump",
        "source_kind": "byo",
        "license": "user-provided",
        "license_url": "",
        "citation": f"User-provided {fmt} dump (not redistributed by AOMB)",
        "capture_id": Path(input_path).name or "byo",
        "windows": [],
        "format": fmt,
    }


def _load_optional_provenance(input_path: str, fmt: str) -> dict[str, Any]:
    """Merge optional provenance.json with BYO defaults."""
    base = _default_provenance(input_path, fmt)
    candidates = [
        os.path.join(input_path, "provenance.json")
        if os.path.isdir(input_path)
        else os.path.join(os.path.dirname(input_path), "provenance.json"),
        os.path.join(input_path, "byo_provenance.json")
        if os.path.isdir(input_path)
        else "",
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                user = json.load(f)
            base.update({k: v for k, v in user.items() if v is not None})
            base["format"] = fmt
            return base
    return base


def _looks_like_jaeger(doc: Any) -> bool:
    if isinstance(doc, dict):
        if "data" in doc and isinstance(doc["data"], list):
            return True
        if "spans" in doc and (
            "traceID" in doc or "traceId" in doc or "processes" in doc
        ):
            return True
    if isinstance(doc, list) and doc:
        return _looks_like_jaeger(doc[0])
    return False


def _discover_jaeger_files(root: str) -> list[str]:
    if os.path.isfile(root):
        return [root]
    patterns = [
        os.path.join(root, "**", "*.json"),
        os.path.join(root, "*.json"),
        os.path.join(root, "traces", "**", "*.json"),
    ]
    skip = {
        "package.json",
        "package-lock.json",
        "provenance.json",
        "byo_provenance.json",
        "tsconfig.json",
    }
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for fp in sorted(glob(pat, recursive=True)):
            if os.path.basename(fp) in skip:
                continue
            if fp in seen:
                continue
            seen.add(fp)
            out.append(fp)
    return out


class ByoAdapter(SourceAdapter):
    """Load user OTLP JSONL / Jaeger JSON / parquet session dumps."""

    name = "byo"

    def load(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        fmt = kwargs.get("format") or detect_format(input_path)
        if fmt == FORMAT_PARQUET_SESSIONS:
            # Parquet sessions are yielded via iter_sessions (already text docs).
            return
        if fmt == FORMAT_OTLP_JSONL:
            yield from self._load_otlp_jsonl(input_path, **kwargs)
            return
        if fmt == FORMAT_JAEGER_JSON:
            yield from self._load_jaeger(input_path, **kwargs)
            return
        raise ValueError(f"Unknown BYO format {fmt!r}")

    def iter_sessions(
        self, input_path: str, **kwargs: Any
    ) -> Iterator[tuple[str, TimeWindow, SourceBundle]]:
        """
        Yield (session_text, window, bundle_stub) for build_shards.

        Parquet path yields ready texts; OTLP/Jaeger go through bundle_to_sessions.
        """
        fmt = kwargs.pop("format", None) or detect_format(input_path)
        include_meta = kwargs.pop("include_meta", True)
        if fmt == FORMAT_PARQUET_SESSIONS:
            yield from self._iter_parquet_sessions(
                input_path, format=fmt, **kwargs
            )
            return

        for bundle in self.load(input_path, format=fmt, **kwargs):
            for text, window in bundle_to_sessions(
                bundle, include_meta=include_meta
            ):
                yield text, window, bundle

    def _load_otlp_jsonl(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        """Reuse lab_capture parsers; allow missing provenance.json."""
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() in {".jsonl", ".ndjson"}:
            # Single JSONL file of spans
            tmp_dir_note = str(path.parent)
            prov = _load_optional_provenance(tmp_dir_note, FORMAT_OTLP_JSONL)
            rows = _load_jsonl(str(path))
            # Delegate flattening via a minimal lab-shaped dir is awkward;
            # write through LabCaptureAdapter helpers by synthesizing load.
            from corpus.ingest.adapters.lab_capture import (
                _flatten_otlp_logs,
                _flatten_otlp_traces,
            )
            from corpus.ingest.adapters.base import LogRecord, SpanRecord

            spans = []
            logs = []
            for row in rows:
                if "resourceSpans" in row:
                    spans.extend(_flatten_otlp_traces(row))
                elif "resourceLogs" in row:
                    logs.extend(_flatten_otlp_logs(row))
                else:
                    # Flat span row (same as lab export)
                    attrs = row.get("attributes") or {}
                    start = _parse_dt(
                        row.get("start_time_unix_nano")
                        or row.get("start_time")
                        or row.get("timestamp")
                    )
                    dur = row.get("duration_ms") or 0
                    spans.append(
                        SpanRecord(
                            trace_id=str(
                                row.get("trace_id") or row.get("traceId") or ""
                            ),
                            span_id=str(
                                row.get("span_id") or row.get("spanId") or ""
                            ),
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
            max_spans = int(kwargs.get("max_spans", 0) or 0)
            max_logs = int(kwargs.get("max_logs", 0) or 0)
            if max_spans:
                spans = spans[:max_spans]
            if max_logs:
                logs = logs[:max_logs]
            yield SourceBundle(
                source_id=str(prov.get("source_id") or "byo-user-dump"),
                source_kind="byo",
                license=str(prov.get("license") or "user-provided"),
                license_url=str(prov.get("license_url") or ""),
                citation=str(prov.get("citation") or ""),
                spans=spans,
                logs=logs,
                windows=_windows_from_provenance(prov),
                capture_id=str(prov.get("capture_id") or path.stem),
                extra_provenance={
                    "input_path": os.path.abspath(input_path),
                    "format": FORMAT_OTLP_JSONL,
                    "byo": True,
                    "provenance": prov,
                },
            )
            return

        # Directory: prefer LabCaptureAdapter when provenance exists; else soft-missing
        prov_path = os.path.join(input_path, "provenance.json")
        if os.path.isfile(prov_path):
            for bundle in LabCaptureAdapter().load(input_path, **kwargs):
                bundle.source_kind = "byo"
                bundle.extra_provenance = {
                    **(bundle.extra_provenance or {}),
                    "format": FORMAT_OTLP_JSONL,
                    "byo": True,
                    "via": "lab_capture_parser",
                }
                if not str(bundle.source_id).startswith("byo"):
                    # Keep user source_id from provenance; tag kind only
                    pass
                yield bundle
            return

        # No provenance — synthesize minimal; parse jsonl like lab_capture
        soft = _default_provenance(input_path, FORMAT_OTLP_JSONL)
        from corpus.ingest.adapters.lab_capture import (
            _flatten_otlp_logs,
            _flatten_otlp_traces,
        )
        from corpus.ingest.adapters.base import LogRecord, SpanRecord

        spans_raw = _load_jsonl(os.path.join(input_path, "traces.jsonl"))
        if not spans_raw:
            spans_raw = _load_jsonl(os.path.join(input_path, "spans.jsonl"))
        logs_raw = _load_jsonl(os.path.join(input_path, "logs.jsonl"))

        spans = []
        for row in spans_raw:
            if "resourceSpans" in row:
                spans.extend(_flatten_otlp_traces(row))
                continue
            attrs = row.get("attributes") or {}
            start = _parse_dt(
                row.get("start_time_unix_nano")
                or row.get("start_time")
                or row.get("timestamp")
            )
            dur = row.get("duration_ms")
            if dur is None and row.get("end_time_unix_nano") and row.get(
                "start_time_unix_nano"
            ):
                try:
                    dur = (
                        int(row["end_time_unix_nano"])
                        - int(row["start_time_unix_nano"])
                    ) / 1e6
                except (TypeError, ValueError, KeyError):
                    dur = 0
            spans.append(
                SpanRecord(
                    trace_id=str(row.get("trace_id") or row.get("traceId") or ""),
                    span_id=str(row.get("span_id") or row.get("spanId") or ""),
                    parent_span_id=str(
                        row.get("parent_span_id") or row.get("parentSpanId") or ""
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
                        row.get("timestamp")
                        or row.get("time_unix_nano")
                        or row.get("observed_time_unix_nano")
                    ),
                    severity=str(
                        row.get("severity") or row.get("severity_text") or "INFO"
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

        max_spans = int(kwargs.get("max_spans", 0) or 0)
        max_logs = int(kwargs.get("max_logs", 0) or 0)
        if max_spans:
            spans = spans[:max_spans]
        if max_logs:
            logs = logs[:max_logs]

        if not spans and not logs:
            raise RuntimeError(
                f"BYO OTLP JSONL under {input_path}: no spans or logs found "
                "(expected traces.jsonl / spans.jsonl / logs.jsonl)."
            )

        yield SourceBundle(
            source_id=str(soft.get("source_id") or "byo-user-dump"),
            source_kind="byo",
            license=str(soft.get("license") or "user-provided"),
            license_url="",
            citation=str(soft.get("citation") or ""),
            spans=spans,
            logs=logs,
            windows=[],
            capture_id=str(soft.get("capture_id") or Path(input_path).name),
            extra_provenance={
                "input_path": os.path.abspath(input_path),
                "format": FORMAT_OTLP_JSONL,
                "byo": True,
                "note": "No provenance.json; windows=unknown until user supplies labels.",
            },
        )

    def _load_jaeger(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        prov = _load_optional_provenance(
            input_path if os.path.isdir(input_path) else os.path.dirname(input_path) or ".",
            FORMAT_JAEGER_JSON,
        )
        files = _discover_jaeger_files(input_path)
        if not files:
            raise FileNotFoundError(
                f"No Jaeger JSON under {input_path}. See docs/byo-and-scorer.md."
            )

        max_files = int(kwargs.get("max_files", 0) or 0)
        max_spans = int(kwargs.get("max_spans", 0) or 0)
        spans = []
        used = 0
        skipped = 0
        for fp in files:
            if max_files and used >= max_files:
                break
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped += 1
                continue
            if not _looks_like_jaeger(doc):
                skipped += 1
                continue
            batch = load_jaeger_json(doc)
            if not batch:
                skipped += 1
                continue
            spans.extend(batch)
            used += 1
            if max_spans and len(spans) >= max_spans:
                spans = spans[:max_spans]
                break

        if not spans:
            raise RuntimeError(
                f"BYO Jaeger input {input_path}: found {len(files)} JSON file(s) "
                "but none parsed as Jaeger traces."
            )

        windows = _windows_from_provenance(prov)
        if not windows:
            times = [s.start_time for s in spans if s.start_time]
            if times:
                windows = [
                    TimeWindow(
                        label="unknown",
                        start=min(times),
                        end=max(times),
                        notes="BYO Jaeger dump without labeled windows",
                    )
                ]

        yield SourceBundle(
            source_id=str(prov.get("source_id") or "byo-user-dump"),
            source_kind="byo",
            license=str(prov.get("license") or "user-provided"),
            license_url=str(prov.get("license_url") or ""),
            citation=str(prov.get("citation") or ""),
            spans=spans,
            logs=[],
            windows=windows,
            capture_id=str(
                prov.get("capture_id") or Path(input_path).name or "byo-jaeger"
            ),
            extra_provenance={
                "input_path": os.path.abspath(input_path),
                "format": FORMAT_JAEGER_JSON,
                "byo": True,
                "json_files_used": used,
                "json_files_skipped": skipped,
                "provenance": prov,
            },
        )

    def _iter_parquet_sessions(
        self, input_path: str, **kwargs: Any
    ) -> Iterator[tuple[str, TimeWindow, SourceBundle]]:
        path = Path(input_path)
        if path.is_file():
            files = [path]
        else:
            files = sorted(path.glob("shard_*.parquet")) or sorted(
                path.glob("*.parquet")
            )
        if not files:
            raise FileNotFoundError(
                f"No parquet files under {input_path} (expected text column)."
            )

        prov = _load_optional_provenance(
            str(path if path.is_dir() else path.parent), FORMAT_PARQUET_SESSIONS
        )
        windows = _windows_from_provenance(prov)
        default_window = windows[0] if windows else TimeWindow(label="unknown")

        stub = SourceBundle(
            source_id=str(prov.get("source_id") or "byo-user-dump"),
            source_kind="byo",
            license=str(prov.get("license") or "user-provided"),
            license_url=str(prov.get("license_url") or ""),
            citation=str(prov.get("citation") or ""),
            spans=[],
            logs=[],
            windows=windows,
            capture_id=str(
                prov.get("capture_id")
                or (path.stem if path.is_file() else path.name)
            ),
            extra_provenance={
                "input_path": os.path.abspath(input_path),
                "format": FORMAT_PARQUET_SESSIONS,
                "byo": True,
                "parquet_files": [str(f) for f in files],
                "note": "Sessions already in AOMB text format; passed through.",
            },
        )

        max_docs = int(kwargs.get("max_spans", 0) or 0)  # reuse cap as doc cap
        n = 0
        for fp in files:
            table = pq.read_table(fp)
            if "text" not in table.column_names:
                raise ValueError(
                    f"{fp}: parquet must have a 'text' column "
                    "(prepare.py / AOMB session contract)."
                )
            for text in table.column("text").to_pylist():
                if text is None:
                    continue
                s = str(text)
                if not s.strip():
                    continue
                yield s, default_window, stub
                n += 1
                if max_docs and n >= max_docs:
                    return
