"""Regression tests for lab capture merge (rotate-while-open safety)."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_merge_mod():
    path = ROOT / "lab" / "scripts" / "merge_capture_exports.py"
    spec = importlib.util.spec_from_file_location("merge_capture_exports", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestMergeCaptureExports(unittest.TestCase):
    def setUp(self):
        self.mod = _load_merge_mod()

    def test_recovers_traces_from_active_prev(self):
        """Bug 20260911T183259Z: empty _active, real spans in _active_prev_*."""
        with tempfile.TemporaryDirectory() as tmp:
            captures = Path(tmp)
            dest = captures / "20260911T183259Z"
            active = captures / "_active"
            prev = captures / "_active_prev_20260911T183259Z"
            dest.mkdir()
            active.mkdir()
            prev.mkdir()

            # Logs still in _active (as observed); traces only in prev.
            (active / "logs.jsonl").write_text(
                json.dumps({"body": "ok"}) + "\n", encoding="utf-8"
            )
            (active / "traces.jsonl").write_text("", encoding="utf-8")
            span = {
                "trace_id": "abc",
                "span_id": "def",
                "name": "GET /checkout",
                "service_name": "api",
            }
            (prev / "traces.jsonl").write_text(
                json.dumps(span) + "\n", encoding="utf-8"
            )
            # Touch prev after session start so --prev-after includes it.
            session_start = time.time() - 5
            os.utime(prev / "traces.jsonl", (session_start + 1, session_start + 1))

            counts = self.mod.merge_capture_exports(
                str(dest),
                str(active),
                captures_root=str(captures),
                prev_after=session_start,
            )
            self.assertEqual(counts["traces"], 1)
            self.assertEqual(counts["logs"], 1)
            traces = (dest / "traces.jsonl").read_text(encoding="utf-8").strip()
            self.assertIn("GET /checkout", traces)
            # Intentional archives must not be scanned
            archive = captures / "_active_archive_old"
            archive.mkdir()
            (archive / "traces.jsonl").write_text(
                json.dumps({"name": "SHOULD_NOT_MERGE"}) + "\n", encoding="utf-8"
            )
            os.utime(archive / "traces.jsonl", (session_start + 2, session_start + 2))
            counts2 = self.mod.merge_capture_exports(
                str(dest),
                str(active),
                captures_root=str(captures),
                prev_after=session_start,
            )
            merged = (dest / "traces.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("SHOULD_NOT_MERGE", merged)
            self.assertEqual(counts2["traces"], 1)

    def test_window_scoped_files_merge_without_dupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            captures = Path(tmp)
            dest = captures / "cap1"
            active = captures / "_active"
            dest.mkdir()
            active.mkdir()
            line_n = json.dumps({"name": "normal_span", "trace_id": "1"})
            line_i = json.dumps({"name": "incident_span", "trace_id": "2"})
            (dest / "normal_traces.jsonl").write_text(line_n + "\n", encoding="utf-8")
            (dest / "incident_traces.jsonl").write_text(line_i + "\n", encoding="utf-8")
            # Stale active copy of incident (would duplicate without dedupe)
            (active / "traces.jsonl").write_text(line_i + "\n", encoding="utf-8")
            (dest / "normal_logs.jsonl").write_text(
                json.dumps({"body": "n"}) + "\n", encoding="utf-8"
            )
            (dest / "incident_logs.jsonl").write_text(
                json.dumps({"body": "i"}) + "\n", encoding="utf-8"
            )

            counts = self.mod.merge_capture_exports(str(dest), str(active))
            self.assertEqual(counts["traces"], 2)
            self.assertEqual(counts["logs"], 2)
            text = (dest / "traces.jsonl").read_text(encoding="utf-8")
            self.assertEqual(text.count("normal_span"), 1)
            self.assertEqual(text.count("incident_span"), 1)

    def test_ensure_window_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            captures = Path(tmp)
            dest = captures / "cap1"
            active = captures / "_active"
            dest.mkdir()
            active.mkdir()
            (active / "traces.jsonl").write_text(
                json.dumps({"name": "s"}) + "\n", encoding="utf-8"
            )
            (active / "logs.jsonl").write_text(
                json.dumps({"body": "l"}) + "\n", encoding="utf-8"
            )
            written = self.mod.ensure_window_copies(str(dest), str(active), "normal")
            self.assertEqual(len(written), 2)
            self.assertTrue((dest / "normal_traces.jsonl").is_file())
            self.assertTrue((dest / "normal_logs.jsonl").is_file())

    def test_old_prev_excluded_without_prev_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            captures = Path(tmp)
            dest = captures / "cap1"
            active = captures / "_active"
            prev = captures / "_active_prev_old"
            dest.mkdir()
            active.mkdir()
            prev.mkdir()
            (prev / "traces.jsonl").write_text(
                json.dumps({"name": "old"}) + "\n", encoding="utf-8"
            )
            counts = self.mod.merge_capture_exports(str(dest), str(active))
            self.assertEqual(counts["traces"], 0)


if __name__ == "__main__":
    unittest.main()
