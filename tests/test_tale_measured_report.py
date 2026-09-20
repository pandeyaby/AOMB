"""Unit tests: measured Tale val_bpb report emitter (fixture log snippets).

No MPS. No Zenodo. No invented AUROC / val_bpb. prepare.py untouched.
CUDA gate stays skipped.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "val_bpb_logs"

sys.path.insert(0, str(ROOT))

from eval.tale_measured_report import (  # noqa: E402
    EXIT_OK,
    EXIT_PATH_ERROR,
    EXIT_REFUSED_FLAG,
    build_card,
    claim_status_for,
    emit_from_log,
    refuse_loud_flags,
)


class TestClaimStatus(unittest.TestCase):
    def test_pending_when_null(self):
        self.assertEqual(claim_status_for(None), "pending")

    def test_measured_not_published_when_present(self):
        self.assertEqual(claim_status_for(0.430912), "measured_not_published")


class TestBuildCard(unittest.TestCase):
    def test_fields(self):
        card = build_card(
            val_bpb=0.430912,
            max_spans=50,
            source_id="fixture",
            git_sha="abc123",
            log_path="/tmp/train.log",
        )
        self.assertEqual(card["corpus"], "tale_of_errors")
        self.assertEqual(card["max_spans"], 50)
        self.assertEqual(card["source_id"], "fixture")
        self.assertEqual(card["git_sha"], "abc123")
        self.assertEqual(card["claim_status"], "measured_not_published")
        self.assertAlmostEqual(card["val_bpb"], 0.430912, places=6)

    def test_null_pending_no_git_sha(self):
        card = build_card(
            val_bpb=None,
            max_spans=None,
            source_id="x",
            git_sha=None,
            log_path="missing.log",
        )
        self.assertIsNone(card["val_bpb"])
        self.assertEqual(card["claim_status"], "pending")
        self.assertNotIn("git_sha", card)


class TestEmitFromLog(unittest.TestCase):
    def test_success_writes_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code, path, card = emit_from_log(
                log_path=FIXTURES / "train_success.txt",
                out_dir=out,
                max_spans=40,
                source_id="unit",
                git_sha="deadbeef",
            )
            self.assertEqual(code, EXIT_OK)
            self.assertTrue(path.is_file())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(loaded["val_bpb"], 0.430912, places=6)
            self.assertEqual(loaded["claim_status"], "measured_not_published")
            self.assertEqual(loaded["corpus"], "tale_of_errors")
            self.assertEqual(loaded["max_spans"], 40)
            self.assertEqual(loaded["git_sha"], "deadbeef")

    def test_missing_log_exit_2_writes_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            missing = out / "nope.log"
            code, path, card = emit_from_log(
                log_path=missing,
                out_dir=out,
                max_spans=10,
                source_id="unit",
                git_sha=None,
            )
            self.assertEqual(code, EXIT_PATH_ERROR)
            self.assertIsNone(card["val_bpb"])
            self.assertEqual(card["claim_status"], "pending")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(loaded["val_bpb"])
            self.assertEqual(loaded["claim_status"], "pending")

    def test_malformed_exit_2_writes_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code, path, card = emit_from_log(
                log_path=FIXTURES / "train_malformed_pending.txt",
                out_dir=out,
                max_spans=None,
                source_id="unit",
                git_sha=None,
            )
            self.assertEqual(code, EXIT_PATH_ERROR)
            self.assertIsNone(card["val_bpb"])
            self.assertEqual(card["claim_status"], "pending")


class TestRefuseLoudFlags(unittest.TestCase):
    def test_auroc_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            refuse_loud_flags(["--log", "x", "--auroc"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_publish_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            refuse_loud_flags(["--publish"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_cuda_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            refuse_loud_flags(["--cuda"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_invent_synonym_exits_1(self):
        with self.assertRaises(SystemExit) as ctx:
            refuse_loud_flags(["--invent-metrics"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)


class TestCLISubprocess(unittest.TestCase):
    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "eval.tale_measured_report", *args],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)},
        )

    def test_cli_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(
                [
                    "--log",
                    str(FIXTURES / "train_success.txt"),
                    "--out-dir",
                    tmp,
                    "--max-spans",
                    "50",
                    "--source-id",
                    "cli",
                ]
            )
            self.assertEqual(proc.returncode, EXIT_OK, proc.stderr)
            card = json.loads((Path(tmp) / "measured.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(card["val_bpb"], 0.430912, places=6)
            self.assertEqual(card["claim_status"], "measured_not_published")
            self.assertNotIn("AUROC", proc.stdout.upper())

    def test_cli_missing_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "absent.log")
            proc = self._run(["--log", missing, "--out-dir", tmp])
            self.assertEqual(proc.returncode, EXIT_PATH_ERROR)
            card = json.loads((Path(tmp) / "measured.json").read_text(encoding="utf-8"))
            self.assertIsNone(card["val_bpb"])
            self.assertEqual(card["claim_status"], "pending")

    def test_cli_malformed_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(
                [
                    "--log",
                    str(FIXTURES / "train_malformed_nan.txt"),
                    "--out-dir",
                    tmp,
                ]
            )
            self.assertEqual(proc.returncode, EXIT_PATH_ERROR)
            card = json.loads((Path(tmp) / "measured.json").read_text(encoding="utf-8"))
            self.assertIsNone(card["val_bpb"])
            self.assertEqual(card["claim_status"], "pending")

    def test_cli_auroc_refused(self):
        proc = self._run(["--log", str(FIXTURES / "train_success.txt"), "--auroc"])
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("Refusing", proc.stderr)

    def test_cli_publish_refused(self):
        proc = self._run(["--log", str(FIXTURES / "train_success.txt"), "--publish"])
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)

    def test_cli_cuda_refused(self):
        proc = self._run(["--log", str(FIXTURES / "train_success.txt"), "--cuda"])
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)

    def test_shell_wrapper_auroc_refused(self):
        script = ROOT / "scripts" / "tale_measured_report.sh"
        proc = subprocess.run(
            ["bash", str(script), "--auroc", "--log", str(FIXTURES / "train_success.txt")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)


class TestHonestyNoInventedMetrics(unittest.TestCase):
    def test_module_has_no_auroc_default(self):
        src = (ROOT / "eval" / "tale_measured_report.py").read_text(encoding="utf-8")
        self.assertNotRegex(src.lower(), r"auroc\s*=\s*0\.")
        self.assertNotRegex(src, r"val_bpb\s*=\s*0\.\d+")
        self.assertIn("Never invents", src)

    def test_prepare_untouched(self):
        # Sacred: this PR must not modify prepare.py
        import subprocess as sp

        diff = sp.run(
            ["git", "diff", "--name-only", "main", "--", "prepare.py"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(diff.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
