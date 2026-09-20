"""Tale overnight launch helper — dry-run / refuse (no agent_loop, no APIs).

No MPS train / no paid APIs. prepare.py untouched. Never invents AUROC / floors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.tale_overnight_launch import (  # noqa: E402
    EXIT_OK,
    EXIT_PLATFORM,
    EXIT_REFUSED_FLAG,
    card_ok,
    main,
    planned_env,
)


def _write_card(path: Path, **fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


GOOD = dict(
    claim_status="measured_not_published",
    val_bpb=1.37952,
    max_spans=200000,
)


class TestPlannedEnv(unittest.TestCase):
    def test_defaults(self):
        env = planned_env()
        self.assertEqual(env["AOMB_CORPUS"], "tale_of_errors")
        self.assertEqual(env["AOMB_SOURCE_ID"], "tale_capped_200k")
        self.assertEqual(env["AOMB_BEST_VAL_FROM_CARD"], "1")


class TestCardOk(unittest.TestCase):
    def test_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_card(Path(tmp) / "c.json", **GOOD)
            ok, msg = card_ok(path)
        self.assertTrue(ok)
        self.assertIn("1.379520", msg)

    def test_missing(self):
        ok, msg = card_ok(Path("/no/such/card.json"))
        self.assertFalse(ok)
        self.assertIn("missing", msg)


class TestCli(unittest.TestCase):
    def test_dry_run_default_exit_0(self):
        code = main([])
        self.assertEqual(code, EXIT_OK)
        code = main(["--dry-run"])
        self.assertEqual(code, EXIT_OK)

    def test_dry_run_with_missing_card_still_0(self):
        code = main(["--dry-run", "--card", "/no/such/measured.json"])
        self.assertEqual(code, EXIT_OK)

    def test_invent_refuse(self):
        for flag in ("--auroc", "--publish", "--cuda"):
            self.assertEqual(main([flag]), EXIT_REFUSED_FLAG)
            self.assertEqual(main(["--dry-run", flag]), EXIT_REFUSED_FLAG)

    def test_run_missing_card_exit_2(self):
        code = main(["--run", "--card", "/no/such/measured.json"])
        self.assertEqual(code, EXIT_PLATFORM)

    def test_run_refused_when_not_darwin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_card(Path(tmp) / "c.json", **GOOD)
            with mock.patch(
                "eval.tale_overnight_launch.darwin_mps_ok",
                return_value=(False, "not Darwin (detected: Linux)"),
            ):
                with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
                    code = main(["--run", "--card", str(path)])
        self.assertEqual(code, EXIT_PLATFORM)
        start.assert_not_called()

    def test_run_starts_agent_when_platform_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_card(Path(tmp) / "c.json", **GOOD)
            with mock.patch(
                "eval.tale_overnight_launch.darwin_mps_ok",
                return_value=(True, "Darwin + MPS available"),
            ):
                with mock.patch(
                    "eval.tale_overnight_launch.start_agent_loop",
                    return_value=0,
                ) as start:
                    code = main(["--run", "--card", str(path)])
        self.assertEqual(code, EXIT_OK)
        start.assert_called_once_with()

    def test_shell_dry_run(self):
        script = ROOT / "scripts" / "tale_overnight_launch.sh"
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_OK)
        self.assertIn("AOMB_CORPUS=tale_of_errors", proc.stdout)
        self.assertIn("AOMB_BEST_VAL_FROM_CARD=1", proc.stdout)
        self.assertIn("dry-run", proc.stdout)
        self.assertNotIn("starting agent_loop", proc.stdout)

    def test_shell_invent_refuse(self):
        script = ROOT / "scripts" / "tale_overnight_launch.sh"
        proc = subprocess.run(
            ["bash", str(script), "--auroc"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": sys.executable},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)


if __name__ == "__main__":
    unittest.main()
