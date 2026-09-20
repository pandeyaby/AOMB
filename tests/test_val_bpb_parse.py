"""Unit tests: factual val_bpb parse hardening (fixture log snippets).

No MPS. No Zenodo. No invented AUROC / val_bpb. prepare.py untouched.
CUDA gate stays skipped.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "val_bpb_logs"

sys.path.insert(0, str(ROOT))

from val_bpb_parse import (  # noqa: E402
    parse_val_bpb,
    parse_val_bpb_from_commit_message,
    parse_val_bpb_from_train_log,
)


def _load(name: str) -> str:
    path = FIXTURES / name
    return path.read_text(encoding="utf-8")


class TestParseValBpbTrainLog(unittest.TestCase):
    def test_success_extracts_factual(self):
        val = parse_val_bpb_from_train_log(_load("train_success.txt"))
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 0.430912, places=6)

    def test_missing_refuses_invent(self):
        self.assertIsNone(parse_val_bpb_from_train_log(_load("train_missing.txt")))

    def test_malformed_pending_refuses(self):
        self.assertIsNone(
            parse_val_bpb_from_train_log(_load("train_malformed_pending.txt"))
        )

    def test_malformed_nan_refuses(self):
        self.assertIsNone(parse_val_bpb_from_train_log(_load("train_malformed_nan.txt")))

    def test_malformed_exploded_refuses(self):
        self.assertIsNone(
            parse_val_bpb_from_train_log(_load("train_malformed_exploded.txt"))
        )

    def test_final_line_wins(self):
        val = parse_val_bpb_from_train_log(_load("train_final_wins.txt"))
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 0.407753, places=6)

    def test_empty_and_none_like(self):
        self.assertIsNone(parse_val_bpb_from_train_log(""))
        self.assertIsNone(parse_val_bpb_from_train_log("val_bpb:          "))
        self.assertIsNone(parse_val_bpb_from_train_log("note: val_bpb mentioned"))


class TestParseValBpbCommitMessage(unittest.TestCase):
    def test_success(self):
        val = parse_val_bpb_from_commit_message(_load("commit_success.txt"))
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 0.4309, places=4)

    def test_malformed_refuses(self):
        self.assertIsNone(parse_val_bpb_from_commit_message(_load("commit_malformed.txt")))

    def test_missing_refuses(self):
        self.assertIsNone(parse_val_bpb_from_commit_message(_load("commit_missing.txt")))


class TestParseValBpbUnified(unittest.TestCase):
    def test_prefers_train_form(self):
        blob = _load("train_success.txt") + "\n" + _load("commit_success.txt")
        val = parse_val_bpb(blob)
        self.assertAlmostEqual(val, 0.430912, places=6)

    def test_falls_back_to_commit(self):
        val = parse_val_bpb(_load("commit_success.txt"))
        self.assertAlmostEqual(val, 0.4309, places=4)


class TestMorningReportFromLogCLI(unittest.TestCase):
    def test_from_log_success_prints_val(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "morning_report.py"),
                "--from-log",
                str(FIXTURES / "train_success.txt"),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("val_bpb: 0.430912", proc.stdout)
        self.assertNotIn("AUROC", proc.stdout.upper())

    def test_from_log_missing_refuses_exit_1(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "morning_report.py"),
                "--from-log",
                str(FIXTURES / "train_missing.txt"),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("REFUSED", proc.stderr)
        self.assertIn("invent", proc.stderr.lower())
        # Must not print a fabricated metric on stdout
        self.assertNotRegex(proc.stdout, r"val_bpb:\s*[0-9]")

    def test_from_log_malformed_refuses(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "morning_report.py"),
                "--from-log",
                str(FIXTURES / "train_malformed_pending.txt"),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("REFUSED", proc.stderr)

    def test_from_log_missing_file_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.log"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "morning_report.py"),
                    "--from-log",
                    str(missing),
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("REFUSED", proc.stderr)


class TestMorningReportGitLogUsesHelper(unittest.TestCase):
    def test_skips_malformed_commit_subjects(self):
        import morning_report

        fake = (
            "aaaaaaaa|feat: unrelated|2026-01-01\n"
            "bbbbbbbb|[val_bpb=pending] [change: bad]|2026-01-02\n"
            "cccccccc|[val_bpb=0.4309] [Δ=-0.01] [change: ok] [hypothesis: h]|2026-01-03\n"
        )
        with mock.patch("morning_report.subprocess.run") as run:
            run.return_value = mock.Mock(stdout=fake, returncode=0)
            exps = morning_report.parse_git_log()
        self.assertEqual(len(exps), 1)
        self.assertAlmostEqual(exps[0]["val_bpb"], 0.4309, places=4)
        self.assertEqual(exps[0]["sha"], "cccccccc")


class TestHonestyNoInventedMetricsInModule(unittest.TestCase):
    def test_helper_source_has_no_auroc_default(self):
        src = (ROOT / "val_bpb_parse.py").read_text(encoding="utf-8")
        self.assertNotRegex(src.lower(), r"auroc\s*=\s*0\.")
        self.assertIn("Never invents", src)


if __name__ == "__main__":
    unittest.main()
