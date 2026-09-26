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
META_PREFIX = "# aomb_meta"
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


def strip_meta_lines(text: str) -> str:
    """
    Drop ``# aomb_meta`` provenance lines before scoring.

    The meta line carries ``window=<label>`` and ``fault=<mode>``. Leaving it in
    the scored text leaks the label into session BPB (and into length
    baselines), so every eval path scores telemetry events only.
    """
    return "\n".join(
        line for line in text.split("\n") if not line.startswith(META_PREFIX)
    )


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
    n_missing_event_ts = 0
    for bi, bundle in enumerate(bundles):
        for sp in bundle.spans:
            if sp.start_time is None:
                n_missing_event_ts += 1
        for lg in bundle.logs:
            if lg.timestamp is None:
                n_missing_event_ts += 1
        for si, (raw_text, window) in enumerate(bundle_to_sessions(bundle)):
            text = strip_meta_lines(raw_text)
            label = window.label or "unknown"
            n_events = text.count("\n") + 1 if text else 0
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

    label_counts = _label_counts(sessions)
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
        "label_counts": label_counts,
        "n_events_missing_timestamp": n_missing_event_ts,
    }
    return sessions, meta


def content_hash_capture(capture_dir: str | Path) -> str:
    """SHA-256 over provenance + traces/logs + optional split.json (sorted paths)."""
    root = Path(capture_dir)
    names = [
        "provenance.json",
        "traces.jsonl",
        "spans.jsonl",
        "logs.jsonl",
        "split.json",
    ]
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


def apply_session_split(
    sessions: list[LabeledSession],
    capture_dir: str | Path,
    split_role: str,
) -> tuple[list[LabeledSession], dict[str, Any] | None]:
    """
    Filter sessions by frozen split.json role.

    split_role: all | train | eval
    Returns (filtered_sessions, split_meta_or_None).
    """
    role = (split_role or "all").strip().lower()
    if role in {"", "all"}:
        return sessions, None
    if role not in {"train", "eval"}:
        raise ValueError(f"unknown session split role: {split_role!r}")

    from eval.fixture_train import load_split, partition_by_split

    split = load_split(capture_dir)
    train_s, eval_s = partition_by_split(sessions, split)
    chosen = train_s if role == "train" else eval_s
    meta = {
        "split_id": split.get("split_id"),
        "split_role": role,
        "n_train": len(train_s),
        "n_eval": len(eval_s),
        "train_session_ids": list(split.get("train_session_ids") or []),
        "eval_session_ids": list(split.get("eval_session_ids") or []),
    }
    return chosen, meta


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
