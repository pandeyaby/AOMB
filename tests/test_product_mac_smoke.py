"""
Product Mac smoke honesty refusals + dry-run / help-only (Linux CI).

Torch-free: subprocess wraps scripts/product_mac_smoke.sh for
--auroc / --publish / --cuda invent refusals and --dry-run / --help-only.
Also covers eval.product_mac_path thin helper.

No Zenodo / no MPS / no invented AUROC. prepare.py untouched.
CUDA gate stays skipped. Lab claim_status stays not_published.
Real Darwin+MPS path is not exercised here (EXIT_PLATFORM expected on Linux).
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
SCRIPT = ROOT / "scripts" / "product_mac_smoke.sh"


def _run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(ROOT)}
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=merged,
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


class TestProductMacPathModule(unittest.TestCase):
    def test_refuse_auroc_before_anything(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--auroc"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        msg = buf.getvalue().lower()
        self.assertIn("auroc", msg)
        self.assertIn("not_published", msg)

    def test_refuse_publish_cuda_and_metric_matrix(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS, main

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    with self.assertRaises(SystemExit) as ctx:
                        main([flag])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
                self.assertIn(flag, buf.getvalue())

    def test_refuse_equals_form(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--publish=1", "--cuda=true"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        self.assertIn("Refusing", buf.getvalue())

    def test_refuse_even_with_dry_run(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--dry-run", "--auroc"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_dry_run_exit_0(self):
        from eval.product_mac_path import EXIT_OK, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            rc = main(["--dry-run"])
        self.assertEqual(rc, EXIT_OK)
        msg = buf.getvalue().lower()
        self.assertIn("not_published", msg)
        self.assertIn("cuda gate stays skipped", msg)
        self.assertIn("prepare.py sacred", msg)

    def test_help_only_exit_0(self):
        from eval.product_mac_path import EXIT_OK, main

        for flag in ("--help-only", "--help", "-h"):
            with self.subTest(flag=flag):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    rc = main([flag])
                self.assertEqual(rc, EXIT_OK)
                self.assertIn("not_published", buf.getvalue().lower())

    def test_covers_session_scorer_flags_plus_cuda(self):
        from eval.product_mac_path import REFUSED_METRIC_FLAGS
        from eval.score_cli import REFUSED_METRIC_FLAGS as SCORE_FLAGS

        self.assertTrue(SCORE_FLAGS.issubset(REFUSED_METRIC_FLAGS))
        self.assertIn("--cuda", REFUSED_METRIC_FLAGS)
        self.assertIn("--gpu", REFUSED_METRIC_FLAGS)
        self.assertIn("--publish", REFUSED_METRIC_FLAGS)
        self.assertIn("--auroc", REFUSED_METRIC_FLAGS)
        self.assertIn("--invent-val-bpb", REFUSED_METRIC_FLAGS)

    def test_exit_codes_stable(self):
        from eval.product_mac_path import EXIT_OK, EXIT_PLATFORM, EXIT_REFUSED_FLAG

        self.assertEqual(EXIT_OK, 0)
        self.assertEqual(EXIT_REFUSED_FLAG, 1)
        self.assertEqual(EXIT_PLATFORM, 2)

    def test_no_auroc_calculator(self):
        src = (ROOT / "eval" / "product_mac_path.py").read_text(encoding="utf-8")
        self.assertNotIn("def compute_auroc", src)
        self.assertIn("not_published", src)
        self.assertIn("EXIT_REFUSED_FLAG", src)
        self.assertIn("CUDA gate stays skipped", src)
        self.assertIn("prepare.py is sacred", src)


class TestProductMacShellParity(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file())

    def test_shell_metric_flags_match_module(self):
        from eval.product_mac_path import EXIT_PLATFORM, EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        listed = _shell_metric_flags(SCRIPT)
        self.assertEqual(listed, set(REFUSED_METRIC_FLAGS))
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(f"EXIT_REFUSED_FLAG={EXIT_REFUSED_FLAG}", src)
        self.assertIn(f"EXIT_PLATFORM={EXIT_PLATFORM}", src)
        self.assertIn("prepare.py is sacred", src.lower().replace("—", " "))
        self.assertIn("CUDA gate stays skipped", src)
        self.assertIn("--dry-run", src)
        self.assertIn("--help-only", src)

    def test_prepare_py_not_edited_by_this_change(self):
        # Sanity: sacred file exists; smoke script invokes but must not rewrite it.
        prepare = ROOT / "prepare.py"
        self.assertTrue(prepare.is_file())
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotRegex(src, r"(^|[^a-zA-Z])sed\s")
        self.assertNotIn("> prepare.py", src)
        self.assertNotIn(">> prepare.py", src)
        # May invoke prepare.py; must not open it for write/edit in-script.
        self.assertIn("prepare.py sacred", src.lower().replace("—", " "))


class TestProductMacSmokeShell(unittest.TestCase):
    def test_refuse_auroc_exit_1(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        proc = _run_script("--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("auroc", blob)
        self.assertIn("not_published", blob)

    def test_refuse_publish_and_cuda(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        for flag in ("--publish", "--cuda", "--gpu", "--invent-auroc", "--invent-val-bpb"):
            with self.subTest(flag=flag):
                proc = _run_script(flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)
                self.assertIn("Refusing", proc.stderr + proc.stdout)

    def test_refuse_equals_form(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        proc = _run_script("--publish=yes", "--cuda=1")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("--publish", proc.stderr + proc.stdout)

    def test_refuse_all_metric_flags(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                proc = _run_script(flag)
                self.assertEqual(
                    proc.returncode,
                    EXIT_REFUSED_FLAG,
                    msg=f"{flag}: {proc.stderr}",
                )

    def test_refuse_invent_even_with_dry_run(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        proc = _run_script("--dry-run", "--auroc")
        # First matching invent flag in argv wins; either order must refuse.
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        proc2 = _run_script("--auroc", "--dry-run")
        self.assertEqual(proc2.returncode, EXIT_REFUSED_FLAG)

    def test_fake_cuda_env_refused(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        for var in ("PRODUCT_MAC_ALLOW_CUDA", "PRODUCT_MAC_FAKE_CUDA"):
            with self.subTest(var=var):
                proc = _run_script("--dry-run", env={var: "1"})
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
                blob = (proc.stderr + proc.stdout).lower()
                self.assertIn("cuda", blob)

    def test_dry_run_exit_0_on_linux(self):
        from eval.product_mac_path import EXIT_OK

        proc = _run_script("--dry-run")
        self.assertEqual(proc.returncode, EXIT_OK, proc.stderr + proc.stdout)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("dry-run", blob)
        self.assertIn("not_published", blob)
        self.assertIn("cuda gate stays skipped", blob)
        self.assertIn("prepare.py sacred", blob)
        # Must not attempt train / invent auroc
        self.assertNotIn("val_bpb: 0.", blob)
        self.assertNotRegex(blob, r"auroc\s*=\s*0\.")

    def test_help_only_exit_0(self):
        for flag in ("--help-only", "--help"):
            with self.subTest(flag=flag):
                proc = _run_script(flag)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                blob = (proc.stderr + proc.stdout).lower()
                self.assertIn("not_published", blob)
                self.assertIn("dry-run", blob)

    def test_real_path_platform_fail_on_linux(self):
        """Without --dry-run, Linux must loud-fail platform (not invent metrics)."""
        from eval.product_mac_path import EXIT_PLATFORM

        if sys.platform == "darwin":
            self.skipTest("real path platform fail is for non-Darwin CI")
        proc = _run_script()
        self.assertEqual(proc.returncode, EXIT_PLATFORM, proc.stderr)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("darwin", blob)
        self.assertIn("dry-run", blob)


class TestSubprocessModuleCli(unittest.TestCase):
    def test_module_cli_refuse_auroc(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        proc = subprocess.run(
            [sys.executable, "-m", "eval.product_mac_path", "--auroc"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("auroc", proc.stderr.lower())

    def test_module_cli_dry_run(self):
        from eval.product_mac_path import EXIT_OK

        proc = subprocess.run(
            [sys.executable, "-m", "eval.product_mac_path", "--dry-run"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_OK)
        self.assertIn("not_published", proc.stderr.lower())

    def test_module_cli_refuse_cuda(self):
        from eval.product_mac_path import EXIT_REFUSED_FLAG

        proc = subprocess.run(
            [sys.executable, "-m", "eval.product_mac_path", "--cuda"],
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
