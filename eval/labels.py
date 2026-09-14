"""
Load labeled sessions from lab_capture provenance windows.

Labels come from capture metadata (provenance.json), not invented per-event flags.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


POSITIVE_LABELS = frozenset({"incident", "cascade", "anomalous"})
NEGATIVE_LABELS = frozenset({"normal"})


@dataclass
class LabeledSession:
    session_id: str
    text: str
    label: str  # raw window label
    binary: Optional[int]  # 0 / 1 / None if excluded
    fault: str = ""
    capture_id: str = ""
    n_chars: int = 0
    n_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # text can be large; callers may strip for reports
        return d


def binary_from_label(label: str) -> Optional[int]:
    """Map window label → 0/1; unknown labels → None (exclude from ranking)."""
    key = (label or "").strip().lower()
    if key in NEGATIVE_LABELS:
        return 0
    if key in POSITIVE_LABELS:
        return 1
    return None


def load_provenance(capture_dir: str | Path) -> dict[str, Any]:
    path = Path(capture_dir) / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Lab captures must include provenance.json "
            "(see lab/scripts/run_capture_session.sh)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_windows(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize provenance windows for fixtures / reports."""
    out: list[dict[str, Any]] = []
    for w in provenance.get("windows") or []:
        label = str(w.get("label", "unknown"))
        out.append(
            {
                "label": label,
                "binary": binary_from_label(label),
                "start": w.get("start"),
                "end": w.get("end"),
                "fault": str(w.get("fault") or ""),
                "notes": str(w.get("notes") or ""),
            }
        )
    return out


def load_lab_sessions(capture_dir: str | Path) -> tuple[list[LabeledSession], dict[str, Any]]:
    """
    Load lab_capture dir → labeled sessions via existing ingest path.

    Returns (sessions, corpus_meta) where corpus_meta includes capture_id,
    source_id, window summary, and content hash of traces+logs+provenance.
    """
    from corpus.ingest.adapters.lab_capture import LabCaptureAdapter
    from corpus.ingest.otlp_to_sessions import bundle_to_sessions

    capture_dir = Path(capture_dir)
    prov = load_provenance(capture_dir)
    bundles = list(LabCaptureAdapter().load(str(capture_dir)))
    if not bundles:
        raise RuntimeError(f"No SourceBundle loaded from {capture_dir}")

    sessions: list[LabeledSession] = []
    for bi, bundle in enumerate(bundles):
        for si, (text, window) in enumerate(bundle_to_sessions(bundle)):
            label = window.label or "unknown"
            n_events = max(0, text.count("\n"))  # meta + events; approx
            sid = f"{bundle.capture_id or capture_dir.name}:{bi}:{si}"
            sessions.append(
                LabeledSession(
                    session_id=sid,
                    text=text,
                    label=label,
                    binary=binary_from_label(label),
                    fault=window.fault or "",
                    capture_id=str(bundle.capture_id or prov.get("capture_id") or ""),
                    n_chars=len(text),
                    n_events=n_events,
                )
            )

    meta = {
        "capture_dir": str(capture_dir.resolve()),
        "capture_id": str(prov.get("capture_id") or capture_dir.name),
        "source_id": str(prov.get("source_id") or ""),
        "source_kind": str(prov.get("source_kind") or "lab_capture"),
        "corpus_version": str(prov.get("corpus_version") or ""),
        "license": str(prov.get("license") or ""),
        "windows": parse_windows(prov),
        "content_sha256": content_hash_capture(capture_dir),
        "session_count": len(sessions),
        "n_scorable": sum(1 for s in sessions if s.binary is not None),
        "n_excluded": sum(1 for s in sessions if s.binary is None),
        "label_counts": _label_counts(sessions),
    }
    return sessions, meta


def content_hash_capture(capture_dir: str | Path) -> str:
    """SHA-256 over provenance.json + traces/logs jsonl bytes (sorted paths)."""
    root = Path(capture_dir)
    names = ["provenance.json", "traces.jsonl", "spans.jsonl", "logs.jsonl"]
    h = hashlib.sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        h.update(name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _label_counts(sessions: Iterable[LabeledSession]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in sessions:
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts


def filter_scorable(
    sessions: list[LabeledSession],
) -> tuple[list[int], list[LabeledSession]]:
    """Return (y_true, sessions_kept) for sessions with binary labels."""
    kept = [s for s in sessions if s.binary is not None]
    y_true = [int(s.binary) for s in kept]  # type: ignore[arg-type]
    return y_true, kept
