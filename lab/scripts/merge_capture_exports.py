#!/usr/bin/env python3
"""Merge lab collector exports into a capture directory.

Used by run_capture_session.sh. Kept as a standalone module so we can
regression-test the "rotated _active still has the traces" failure mode:

  If the otel-collector keeps a file descriptor open across an ``mv`` of
  ``captures/_active/traces.jsonl`` → ``captures/_active_prev_*/traces.jsonl``,
  subsequent spans land in the prev dir while ``_active/traces.jsonl`` is
  empty/missing. Final merge must still recover those lines when asked.

Default merge uses window-scoped files + ``_active`` only (so old archives do
not pollute a new capture). Pass ``--prev-after`` / ``prev_after=`` to also
scan ``_active_prev_*`` directories modified at-or-after a session start time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from typing import Iterable, Optional


SIGNAL_NAMES = ("traces", "logs")


def _read_export_file(path: str) -> list[str]:
    """Return NDJSON lines from a collector file-exporter dump."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    lines: list[str] = []
    # File exporter may write one JSON array or NDJSON
    if text.startswith("["):
        try:
            arr = json.loads(text)
            for item in arr:
                lines.append(json.dumps(item, separators=(",", ":")))
            return lines
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _collect(patterns: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            lines.extend(_read_export_file(path))
    return lines


def _dedupe_preserve(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _parse_after(value: Optional[str | float | int]) -> Optional[float]:
    """Parse an epoch seconds or ISO-8601 timestamp into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise SystemExit(f"Invalid --prev-after timestamp: {value!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _prev_dirs(captures_root: str, after_epoch: Optional[float]) -> list[str]:
    root = os.path.abspath(captures_root)
    dirs = sorted(glob.glob(os.path.join(root, "_active_prev_*")))
    out: list[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        if after_epoch is not None:
            # Directory mtime OR any file mtime — covers "created at session
            # start then written into via stale FD".
            mtimes = [os.path.getmtime(d)]
            for dirpath, _, filenames in os.walk(d):
                for name in filenames:
                    try:
                        mtimes.append(os.path.getmtime(os.path.join(dirpath, name)))
                    except OSError:
                        continue
            if max(mtimes) < after_epoch:
                continue
        out.append(d)
    return out


def _prev_patterns(captures_root: str, signal: str, after_epoch: Optional[float]) -> list[str]:
    patterns: list[str] = []
    for d in _prev_dirs(captures_root, after_epoch):
        patterns.append(os.path.join(d, f"{signal}.jsonl"))
        patterns.append(os.path.join(d, f"{signal}.json"))
        patterns.append(os.path.join(d, f"{signal}*.jsonl"))
    return patterns


def merge_capture_exports(
    dest: str,
    active: str,
    *,
    captures_root: str | None = None,
    prev_after: str | float | int | None = None,
) -> dict[str, int]:
    """Write dest/{traces,logs}.jsonl from window + active (+ optional prev).

    Prefer window-scoped files when present (disjoint normal/incident).
    Always also scan ``active``. When ``prev_after`` is set, also scan
    ``_active_prev_*`` siblings under ``captures_root`` whose mtime is at or
    after that timestamp (recover rotate-while-open data loss).
    """
    dest = os.path.abspath(dest)
    active = os.path.abspath(active)
    if captures_root is None:
        captures_root = os.path.dirname(active)
    after_epoch = _parse_after(prev_after)
    os.makedirs(dest, exist_ok=True)

    counts: dict[str, int] = {}
    for signal in SIGNAL_NAMES:
        patterns = [
            os.path.join(dest, f"normal_{signal}.jsonl"),
            os.path.join(dest, f"incident_{signal}.jsonl"),
            os.path.join(active, f"{signal}.jsonl"),
            os.path.join(active, f"{signal}.json"),
            os.path.join(active, f"{signal}*.jsonl"),
        ]
        if after_epoch is not None:
            patterns.extend(_prev_patterns(captures_root, signal, after_epoch))
        merged = _dedupe_preserve(_collect(patterns))
        out_path = os.path.join(dest, f"{signal}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(merged) + ("\n" if merged else ""))
        counts[signal] = len(merged)
    return counts


def ensure_window_copies(dest: str, active: str, window: str) -> list[str]:
    """Copy active signal files into dest/<window>_{traces,logs}.jsonl.

    Returns list of paths written. Safe no-op when active files are missing.
    """
    dest = os.path.abspath(dest)
    active = os.path.abspath(active)
    os.makedirs(dest, exist_ok=True)
    written: list[str] = []
    for signal in SIGNAL_NAMES:
        src = os.path.join(active, f"{signal}.jsonl")
        candidates = sorted(glob.glob(os.path.join(active, f"{signal}*.jsonl")))
        if os.path.isfile(src):
            out = os.path.join(dest, f"{window}_{signal}.jsonl")
            with open(src, "r", encoding="utf-8", errors="replace") as inf, open(
                out, "w", encoding="utf-8"
            ) as outf:
                outf.write(inf.read())
            written.append(out)
        elif candidates:
            lines = _dedupe_preserve(_collect(candidates))
            out = os.path.join(dest, f"{window}_{signal}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dest", help="Capture output dir (lab/captures/<id>)")
    p.add_argument("active", help="Collector active dir (lab/captures/_active)")
    p.add_argument(
        "--captures-root",
        default=None,
        help="Parent of _active / _active_prev_* (default: dirname(active))",
    )
    p.add_argument(
        "--prev-after",
        default=None,
        help=(
            "Also merge from _active_prev_* modified at/after this ISO time "
            "or epoch seconds (recovery for rotate-while-open)"
        ),
    )
    args = p.parse_args(argv)
    counts = merge_capture_exports(
        args.dest,
        args.active,
        captures_root=args.captures_root,
        prev_after=args.prev_after,
    )
    print(
        f"Merged {counts['traces']} trace lines, {counts['logs']} log lines "
        f"→ {args.dest}"
    )
    # Hint when traces empty — common symptom of the rotate bug.
    if counts["traces"] == 0 and args.prev_after is None:
        print(
            "WARNING: 0 trace lines merged. If collector was rotated while "
            "running, re-run with --prev-after <session-start-ISO>.",
            file=__import__("sys").stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
