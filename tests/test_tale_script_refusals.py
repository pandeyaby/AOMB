"""
Tale script invent-flag refusals — subprocess only (no Zenodo / no MPS).

Covers scripts/tale_capped_baseline.sh and scripts/tale_scale_smoke.sh.
Mirrors stranger / product_mac invent set: --auroc / --publish / --cuda
(+ invent synonyms) → exit 1. Also keeps --download-all / full-decompress
refusals. prepare.py untouched. No invented AUROC / val_bpb.
CUDA gate stays skipped. Lab claim_status stays not_published.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts" / "tale_capped_baseline.sh"
SMOKE_SCRIPT = ROOT / "scripts" / "tale_scale_smoke.sh"

# Keep in sync with eval.stranger_path.REFUSED_METRIC_FLAGS /
# eval.product_mac_path.REFUSED_METRIC_FLAGS and the shell case arms.
REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--invent-metrics",
        "--invent-auroc",
        "--claim-auroc",
        "--val-bpb",
        "--invent-val-bpb",
        "--readme-hero",
        "--publish-readme",
        "--hero-auroc",
        "--cuda",
        "--gpu",
    }
)

# Shared bulk / decompress refusals (script-specific synonyms may add more).
REFUSED_DOWNLOAD_FLAGS_SHARED = frozenset(
    {
        "--download-all",
        "--fetch-all",
        "--full-decompress",
        "--decompress-all",
    }
)

EXIT_REFUSED_FLAG = 1


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


def _shell_download_flags(script: Path) -> set[str]:
    src = script.read_text(encoding="utf-8")
    # Download / decompress arm (second die_refuse after invent set).
    download_m = re.search(
        r"(--(?:download|fetch-all|full-decompress)[^\n]+)\n\s+die_refuse",
        src,
    )
    assert download_m is not None, f"{script.name}: download arm missing"
    return {
        f.rstrip(")")
        for f in download_m.group(1).split("|")
        if f.startswith("--")
    }


class TestTaleScriptShellParity(unittest.TestCase):
    def test_scripts_exist(self):
        self.assertTrue(BASELINE_SCRIPT.is_file())
        self.assertTrue(SMOKE_SCRIPT.is_file())

    def test_shell_metric_flags_match_stranger_set(self):
        from eval.stranger_path import REFUSED_METRIC_FLAGS as STRANGER_FLAGS

        self.assertEqual(set(REFUSED_METRIC_FLAGS), set(STRANGER_FLAGS))
        for script in (BASELINE_SCRIPT, SMOKE_SCRIPT):
            with self.subTest(script=script.name):
                listed = _shell_metric_flags(script)
                self.assertEqual(listed, set(REFUSED_METRIC_FLAGS), script.name)
                src = script.read_text(encoding="utf-8")
                self.assertIn(f"EXIT_REFUSED_FLAG={EXIT_REFUSED_FLAG}", src)
                self.assertIn("die_refuse", src)
                self.assertIn("CUDA gate stays skipped", src)
                self.assertIn("not_published", src)

    def test_baseline_and_smoke_same_metric_arm(self):
        self.assertEqual(
            _shell_metric_flags(BASELINE_SCRIPT),
            _shell_metric_flags(SMOKE_SCRIPT),
        )

    def test_download_refusals_kept(self):
        for script in (BASELINE_SCRIPT, SMOKE_SCRIPT):
            with self.subTest(script=script.name):
                listed = _shell_download_flags(script)
                self.assertTrue(
                    REFUSED_DOWNLOAD_FLAGS_SHARED.issubset(listed),
                    msg=f"{script.name}: missing shared download refusals; got {listed}",
                )
                self.assertIn("--download-all", listed)


class TestTaleCappedBaselineRefusals(unittest.TestCase):
    def test_refuse_auroc_exit_1(self):
        proc = _run_script(BASELINE_SCRIPT, "--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("auroc", blob)
        self.assertIn("not_published", blob)

    def test_refuse_publish_and_cuda(self):
        for flag in ("--publish", "--cuda", "--gpu", "--invent-auroc"):
            with self.subTest(flag=flag):
                proc = _run_script(BASELINE_SCRIPT, flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)
                self.assertIn("Refusing", proc.stderr + proc.stdout)

    def test_refuse_equals_form(self):
        proc = _run_script(BASELINE_SCRIPT, "--publish=yes", "--cuda=1")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("--publish", proc.stderr + proc.stdout)

    def test_refuse_all_metric_flags(self):
        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                proc = _run_script(BASELINE_SCRIPT, flag)
                self.assertEqual(
                    proc.returncode,
                    EXIT_REFUSED_FLAG,
                    msg=f"{flag}: {proc.stderr}",
                )

    def test_refuse_download_all_and_full_decompress(self):
        for flag in (
            "--download-all",
            "--fetch-all",
            "--full-decompress",
            "--decompress-all",
            "--decompress",
            "--assemble-all",
        ):
            with self.subTest(flag=flag):
                proc = _run_script(BASELINE_SCRIPT, flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)
                blob = (proc.stderr + proc.stdout).lower()
                self.assertIn("refusing", blob)

    def test_help_exit_0(self):
        proc = _run_script(BASELINE_SCRIPT, "--help")
        self.assertEqual(proc.returncode, 0)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("not_published", blob)
        self.assertIn("auroc", blob)
        self.assertIn("cuda", blob)


class TestTaleScaleSmokeRefusals(unittest.TestCase):
    def test_refuse_auroc_exit_1(self):
        proc = _run_script(SMOKE_SCRIPT, "--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("auroc", blob)
        self.assertIn("not_published", blob)

    def test_refuse_publish_cuda_parity_with_baseline(self):
        for flag in ("--publish", "--cuda", "--lab-auroc", "--val-bpb", "--invent-metrics"):
            with self.subTest(flag=flag):
                baseline = _run_script(BASELINE_SCRIPT, flag)
                smoke = _run_script(SMOKE_SCRIPT, flag)
                self.assertEqual(baseline.returncode, EXIT_REFUSED_FLAG)
                self.assertEqual(smoke.returncode, EXIT_REFUSED_FLAG)
                self.assertEqual(baseline.returncode, smoke.returncode)

    def test_refuse_equals_form(self):
        proc = _run_script(SMOKE_SCRIPT, "--auroc=true", "--cuda=1")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("--auroc", proc.stderr + proc.stdout)

    def test_refuse_all_metric_flags(self):
        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                proc = _run_script(SMOKE_SCRIPT, flag)
                self.assertEqual(
                    proc.returncode,
                    EXIT_REFUSED_FLAG,
                    msg=f"{flag}: {proc.stderr}",
                )

    def test_refuse_download_all(self):
        for flag in ("--download", "--download-all", "--fetch-all", "--full-decompress"):
            with self.subTest(flag=flag):
                proc = _run_script(SMOKE_SCRIPT, flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)

    def test_help_exit_0(self):
        proc = _run_script(SMOKE_SCRIPT, "--help")
        self.assertEqual(proc.returncode, 0)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("not_published", blob)
        self.assertIn("auroc", blob)

    def test_no_zenodo_or_mps_on_refuse_path(self):
        """Refuse path must not touch network/MPS — exits before any train."""
        proc = _run_script(SMOKE_SCRIPT, "--cuda")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertNotIn("zenodo", blob)
        self.assertNotIn("mps available", blob)
        self.assertIn("cuda", blob)


class TestNoInventedMetricsInScripts(unittest.TestCase):
    def test_scripts_do_not_compute_auroc(self):
        for script in (BASELINE_SCRIPT, SMOKE_SCRIPT):
            with self.subTest(script=script.name):
                src = script.read_text(encoding="utf-8")
                self.assertNotIn("compute_auroc", src)
                self.assertIn("EXIT_REFUSED_FLAG", src)
                self.assertIn("not_published", src)
                self.assertIn("prepare.py is sacred", src.lower().replace("—", " "))


if __name__ == "__main__":
    unittest.main()
