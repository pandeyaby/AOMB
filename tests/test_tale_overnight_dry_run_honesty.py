"""Cross-entrypoint honesty lock: --tale-overnight-dry-run never starts agent_loop.

Subprocess + module spies across:
  - scripts/product_mac_smoke.sh
  - scripts/stranger_demo.sh / stranger_verify.sh
  - morning_report.py

Dry-run only (never --run). No paid APIs. prepare.py untouched. No MPS/Zenodo.
Never invents AUROC / val_bpb. Missing invent-flag coverage asserted exit 1.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CARD = ROOT / "reports" / "tale-capped" / "measured_capped_200k.json"
PRODUCT_SMOKE = ROOT / "scripts" / "product_mac_smoke.sh"
STRANGER_DEMO = ROOT / "scripts" / "stranger_demo.sh"
STRANGER_VERIFY = ROOT / "scripts" / "stranger_verify.sh"
MORNING_REPORT = ROOT / "morning_report.py"

# Phrases that would imply a real overnight start (must never appear).
STARTED_MARKERS = (
    "starting agent_loop",
    "agent_loop started",
    "os.execv",
    "invoking agent_loop with --run",
)


def _run(cmd: list[str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*cmd, *extra],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(ROOT)},
        check=False,
    )


def _blob(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stdout or "") + (proc.stderr or "")


def _assert_dry_run_success(test: unittest.TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    test.assertEqual(proc.returncode, 0, _blob(proc))
    out = proc.stdout or ""
    low = _blob(proc).lower()
    test.assertIn("dry-run", low)
    test.assertIn("aomb_corpus=tale_of_errors", low)
    test.assertIn("no agent_loop", low)
    # Help may mention "--run" as a *next* step; must still say it was not started.
    test.assertIn("not started", low)
    for marker in STARTED_MARKERS:
        test.assertNotIn(marker, low)
    test.assertNotIn("starting agent_loop", low)
    # PR #79: every entrypoint must pass through best_val + source (never invent).
    test.assertIn("best_val:", out)
    test.assertIn("best_val_source:", out)
    m = re.search(r"best_val_source:\s*(\S+)", out)
    test.assertIsNotNone(m, out)
    test.assertIn(m.group(1), {"env", "measured_card", "git", "missing"})


class TestCrossEntrypointSubprocessDryRun(unittest.TestCase):
    """Real subprocess entrypoints with committed measured card."""

    @classmethod
    def setUpClass(cls):
        if not CARD.is_file():
            raise unittest.SkipTest(f"committed Tale measured card missing: {CARD}")

    def test_product_mac_smoke(self):
        proc = _run(["bash", str(PRODUCT_SMOKE), "--tale-overnight-dry-run"])
        _assert_dry_run_success(self, proc)

    def test_stranger_demo(self):
        proc = _run(["bash", str(STRANGER_DEMO), "--tale-overnight-dry-run"])
        _assert_dry_run_success(self, proc)

    def test_stranger_verify(self):
        proc = _run(["bash", str(STRANGER_VERIFY), "--tale-overnight-dry-run"])
        _assert_dry_run_success(self, proc)

    def test_morning_report(self):
        proc = _run([sys.executable, str(MORNING_REPORT), "--tale-overnight-dry-run"])
        _assert_dry_run_success(self, proc)


class TestCrossEntrypointInventRefuse(unittest.TestCase):
    """Invent / publish / CUDA still exit 1 on every overnight dry-run entrypoint."""

    def test_invent_refuse_each_entrypoint(self):
        cases = [
            (["bash", str(PRODUCT_SMOKE)], "--tale-overnight-dry-run"),
            (["bash", str(STRANGER_DEMO)], "--tale-overnight-dry-run"),
            (["bash", str(STRANGER_VERIFY)], "--tale-overnight-dry-run"),
            ([sys.executable, str(MORNING_REPORT)], "--tale-overnight-dry-run"),
        ]
        for base, mode in cases:
            for flag in ("--auroc", "--publish", "--cuda"):
                with self.subTest(cmd=base[-1], flag=flag, order="mode-first"):
                    proc = _run([*base, mode, flag])
                    self.assertEqual(proc.returncode, 1, _blob(proc))
                    self.assertNotIn("starting agent_loop", _blob(proc).lower())
                with self.subTest(cmd=base[-1], flag=flag, order="flag-first"):
                    proc = _run([*base, flag, mode])
                    self.assertEqual(proc.returncode, 1, _blob(proc))


class TestHelperSpyDryRunOnly(unittest.TestCase):
    """If overnight helper is called, argv is dry-run only (--run absent)."""

    def test_product_mac_path_spy(self):
        from eval.product_mac_path import EXIT_OK, main

        if not CARD.is_file():
            self.skipTest("committed Tale measured card missing")
        with mock.patch("eval.tale_overnight_launch.card_ok", return_value=(True, "ok")):
            with mock.patch(
                "eval.tale_overnight_launch.main", return_value=0
            ) as overnight:
                with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
                    with mock.patch("sys.stdout", io.StringIO()), mock.patch(
                        "sys.stderr", io.StringIO()
                    ):
                        rc = main(["--tale-overnight-dry-run"])
        self.assertEqual(rc, EXIT_OK)
        overnight.assert_called_once()
        argv = overnight.call_args[0][0]
        self.assertIn("--dry-run", argv)
        self.assertNotIn("--run", argv)
        start.assert_not_called()

    def test_stranger_path_spy(self):
        from eval.stranger_path import EXIT_OK, main

        if not CARD.is_file():
            self.skipTest("committed Tale measured card missing")
        with mock.patch("eval.tale_overnight_launch.card_ok", return_value=(True, "ok")):
            with mock.patch(
                "eval.tale_overnight_launch.main", return_value=0
            ) as overnight:
                with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
                    with mock.patch("sys.stdout", io.StringIO()), mock.patch(
                        "sys.stderr", io.StringIO()
                    ):
                        rc = main(["--tale-overnight-dry-run"])
        self.assertEqual(rc, EXIT_OK)
        overnight.assert_called_once()
        argv = overnight.call_args[0][0]
        self.assertIn("--dry-run", argv)
        self.assertNotIn("--run", argv)
        start.assert_not_called()

    def test_morning_report_spy(self):
        from morning_report import EXIT_OK, cli

        if not CARD.is_file():
            self.skipTest("committed Tale measured card missing")
        with mock.patch("eval.tale_overnight_launch.card_ok", return_value=(True, "ok")):
            with mock.patch(
                "eval.tale_overnight_launch.main", return_value=0
            ) as overnight:
                with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
                    with mock.patch("sys.stdout", io.StringIO()), mock.patch(
                        "sys.stderr", io.StringIO()
                    ):
                        rc = cli(["--tale-overnight-dry-run"])
        self.assertEqual(rc, EXIT_OK)
        overnight.assert_called_once()
        argv = overnight.call_args[0][0]
        self.assertIn("--dry-run", argv)
        self.assertNotIn("--run", argv)
        start.assert_not_called()


class TestSourceNeverPassesRun(unittest.TestCase):
    """Static lock: entrypoint sources must not pass --run to overnight helper."""

    def test_shell_and_py_sources(self):
        paths = [
            PRODUCT_SMOKE,
            STRANGER_DEMO,
            STRANGER_VERIFY,
            ROOT / "eval" / "product_mac_path.py",
            ROOT / "eval" / "stranger_path.py",
            MORNING_REPORT,
        ]
        for path in paths:
            src = path.read_text(encoding="utf-8")
            self.assertIn("--tale-overnight-dry-run", src, path.name)
            # Overnight dry-run arms must call helper with --dry-run, never --run.
            # Narrow: any line that mentions tale_overnight / overnight_main / dry-run
            # mode must not contain a bare "--run" token as an argv literal.
            overnight_lines = [
                ln
                for ln in src.splitlines()
                if "tale_overnight" in ln
                or "tale-overnight" in ln
                or "overnight_main" in ln
                or '["--dry-run"' in ln
                or "['--dry-run'" in ln
            ]
            self.assertTrue(overnight_lines, f"{path.name}: no overnight dry-run lines")
            for ln in overnight_lines:
                # Allow comments talking about "never --run"
                if "never" in ln.lower() and "--run" in ln:
                    continue
                if "not" in ln.lower() and "--run" in ln:
                    continue
                # Disallow argv that includes --run as a launch flag.
                if '"--run"' in ln or "'--run'" in ln:
                    self.fail(f"{path.name}: overnight path must not pass --run: {ln}")


if __name__ == "__main__":
    unittest.main()
