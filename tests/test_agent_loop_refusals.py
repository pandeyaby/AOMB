"""agent_loop invent-flag refusals — mirror stranger/tale refusal tests.

No MPS / Zenodo / overnight run. prepare.py untouched. No invented AUROC.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS  # noqa: E402
from agent_loop import cli, find_refused_invent_flag  # noqa: E402


class TestFindRefusedInventFlag(unittest.TestCase):
    def test_core_flags(self):
        for flag in (
            "--auroc",
            "--lab-auroc",
            "--accuracy",
            "--ranking",
            "--publish",
            "--claim",
            "--cuda",
            "--gpu",
            "--invent-auroc",
            "--val-bpb",
            "--readme-hero",
        ):
            self.assertIn(flag, REFUSED_METRIC_FLAGS)
            self.assertEqual(find_refused_invent_flag([flag]), flag)

    def test_equals_form(self):
        self.assertEqual(
            find_refused_invent_flag(["--auroc=0.99"]), "--auroc"
        )

    def test_clean_argv(self):
        self.assertIsNone(find_refused_invent_flag([]))
        self.assertIsNone(find_refused_invent_flag(["--help-me-not-a-flag"]))


class TestAgentLoopCliRefuse(unittest.TestCase):
    def test_cli_refuse_returns_one(self):
        for flag in ("--auroc", "--publish", "--cuda", "--lab-auroc"):
            with mock.patch("agent_loop.main") as main_mock:
                code = cli([flag])
            self.assertEqual(code, EXIT_REFUSED_FLAG, msg=flag)
            main_mock.assert_not_called()

    def test_cli_clean_calls_main(self):
        with mock.patch("agent_loop.main") as main_mock:
            code = cli([])
        self.assertEqual(code, 0)
        main_mock.assert_called_once_with()

    def test_subprocess_refuse(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "agent_loop.py"), "--auroc"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("Refusing", proc.stderr)
        self.assertIn("--auroc", proc.stderr)
        self.assertIn("Never invents", proc.stderr)


if __name__ == "__main__":
    unittest.main()
