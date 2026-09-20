"""
Stranger demo / verify honesty refusals + exit-code parity.

Torch-free: subprocess wraps scripts/stranger_demo.sh and
scripts/stranger_verify.sh for --auroc / --publish / --cuda invent
refusals. Also covers eval.stranger_path thin helper.

No Zenodo / no MPS / no invented AUROC. prepare.py untouched.
CUDA gate stays skipped. Lab claim_status stays not_published.
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
DEMO_SCRIPT = ROOT / "scripts" / "stranger_demo.sh"
VERIFY_SCRIPT = ROOT / "scripts" / "stranger_verify.sh"


def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON": sys.executable},
        check=False,
    )


def _shell_metric_flags(script: Path) -> set[str]:
    src = script.read_text(encoding="utf-8")
    m = re.search(
        r'case "\$key" in\n\s+(--[^\n]+)\n\s+die_refuse "Refusing',
        src,
    )
    assert m is not None, f"{script.name}: metric refuse case arm missing"
    return {
        f.rstrip(")")
        for f in m.group(1).split("|")
        if f.startswith("--")
    }


class TestStrangerPathModule(unittest.TestCase):
    def test_refuse_auroc_before_anything(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--auroc"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        msg = buf.getvalue().lower()
        self.assertIn("auroc", msg)
        self.assertIn("not_published", msg)

    def test_refuse_publish_cuda_and_metric_matrix(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS, main

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    with self.assertRaises(SystemExit) as ctx:
                        main([flag])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
                self.assertIn(flag, buf.getvalue())

    def test_refuse_equals_form(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--publish=1", "--cuda=true"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        self.assertIn("Refusing", buf.getvalue())

    def test_refuse_product_overnight_flags(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_PRODUCT_FLAGS, main

        for flag in sorted(REFUSED_PRODUCT_FLAGS):
            with self.subTest(flag=flag):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    with self.assertRaises(SystemExit) as ctx:
                        main([flag])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_help_exit_0(self):
        from eval.stranger_path import EXIT_OK, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            rc = main(["--help"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("not_published", buf.getvalue().lower())

    def test_no_args_honesty_ok(self):
        from eval.stranger_path import EXIT_OK, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            rc = main([])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("not_published", buf.getvalue().lower())

    def test_covers_session_scorer_flags_plus_cuda(self):
        from eval.score_cli import REFUSED_METRIC_FLAGS as SCORE_FLAGS
        from eval.stranger_path import REFUSED_METRIC_FLAGS

        # Session scorer invent set ⊆ stranger refusals
        self.assertTrue(SCORE_FLAGS.issubset(REFUSED_METRIC_FLAGS))
        self.assertIn("--cuda", REFUSED_METRIC_FLAGS)
        self.assertIn("--gpu", REFUSED_METRIC_FLAGS)
        self.assertIn("--publish", REFUSED_METRIC_FLAGS)
        self.assertIn("--auroc", REFUSED_METRIC_FLAGS)

    def test_no_auroc_calculator(self):
        src = (ROOT / "eval" / "stranger_path.py").read_text(encoding="utf-8")
        self.assertNotIn("def compute_auroc", src)
        self.assertIn("not_published", src)
        self.assertIn("EXIT_REFUSED_FLAG", src)
        self.assertIn("CUDA gate stays skipped", src)


class TestStrangerShellParity(unittest.TestCase):
    def test_scripts_exist_executable_bits_ok(self):
        self.assertTrue(DEMO_SCRIPT.is_file())
        self.assertTrue(VERIFY_SCRIPT.is_file())

    def test_shell_metric_flags_match_module(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        for script in (DEMO_SCRIPT, VERIFY_SCRIPT):
            with self.subTest(script=script.name):
                listed = _shell_metric_flags(script)
                self.assertEqual(listed, set(REFUSED_METRIC_FLAGS), script.name)
                src = script.read_text(encoding="utf-8")
                self.assertIn(f"EXIT_REFUSED_FLAG={EXIT_REFUSED_FLAG}", src)
                self.assertIn("prepare.py is sacred", src.lower().replace("—", " "))
                self.assertIn("CUDA gate stays skipped", src)

    def test_demo_and_verify_same_metric_arm(self):
        self.assertEqual(
            _shell_metric_flags(DEMO_SCRIPT),
            _shell_metric_flags(VERIFY_SCRIPT),
        )


class TestStrangerDemoShell(unittest.TestCase):
    def test_refuse_auroc_exit_1(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_script(DEMO_SCRIPT, "--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("auroc", blob)
        self.assertIn("not_published", blob)

    def test_refuse_publish_and_cuda(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        for flag in ("--publish", "--cuda", "--gpu", "--invent-auroc"):
            with self.subTest(flag=flag):
                proc = _run_script(DEMO_SCRIPT, flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)
                self.assertIn("Refusing", proc.stderr + proc.stdout)

    def test_refuse_equals_form(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_script(DEMO_SCRIPT, "--publish=yes", "--cuda=1")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("--publish", proc.stderr + proc.stdout)

    def test_refuse_all_metric_flags(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                proc = _run_script(DEMO_SCRIPT, flag)
                self.assertEqual(
                    proc.returncode,
                    EXIT_REFUSED_FLAG,
                    msg=f"{flag}: {proc.stderr}",
                )

    def test_refuse_overnight(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_script(DEMO_SCRIPT, "--overnight")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)

    def test_help_exit_0(self):
        proc = _run_script(DEMO_SCRIPT, "--help")
        self.assertEqual(proc.returncode, 0)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("not_published", blob)
        self.assertIn("auroc", blob)


class TestStrangerVerifyShell(unittest.TestCase):
    def test_refuse_auroc_exit_1(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_script(VERIFY_SCRIPT, "--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("auroc", blob)
        self.assertIn("not_published", blob)

    def test_refuse_publish_cuda_parity_with_demo(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        for flag in ("--publish", "--cuda", "--lab-auroc", "--val-bpb"):
            with self.subTest(flag=flag):
                demo = _run_script(DEMO_SCRIPT, flag)
                verify = _run_script(VERIFY_SCRIPT, flag)
                self.assertEqual(demo.returncode, EXIT_REFUSED_FLAG)
                self.assertEqual(verify.returncode, EXIT_REFUSED_FLAG)
                self.assertEqual(demo.returncode, verify.returncode)

    def test_refuse_equals_form(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_script(VERIFY_SCRIPT, "--auroc=true")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)

    def test_help_exit_0(self):
        proc = _run_script(VERIFY_SCRIPT, "--help")
        self.assertEqual(proc.returncode, 0)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("not_published", blob)

    def test_mps_expect_env_refused(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = subprocess.run(
            ["bash", str(VERIFY_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "STRANGER_EXPECT_MPS": "1"},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("mps", (proc.stderr + proc.stdout).lower())


class TestSubprocessModuleCli(unittest.TestCase):
    def test_module_cli_refuse_auroc(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = subprocess.run(
            [sys.executable, "-m", "eval.stranger_path", "--auroc"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("auroc", proc.stderr.lower())

    def test_module_cli_refuse_cuda(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = subprocess.run(
            [sys.executable, "-m", "eval.stranger_path", "--cuda"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("cuda", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
