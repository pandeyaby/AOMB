"""
demo_anomaly / anomaly-story scorer smoke hardening tests.

Torch-free: flag refusals, dry-run tiny fixture, honesty / claim_status,
exit codes. No MPS / no invented AUROC. prepare.py untouched.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class TestFlagRefusals(unittest.TestCase):
    def test_refuse_auroc_before_argparse(self):
        from demo_anomaly import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--auroc", "--dry-run", "--per-class", "1"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        msg = buf.getvalue().lower()
        self.assertIn("auroc", msg)
        self.assertIn("not_published", msg)

    def test_refuse_publish_and_ranking_matrix(self):
        from demo_anomaly import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS, main

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    with self.assertRaises(SystemExit) as ctx:
                        main([flag, "--dry-run", "--per-class", "1"])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
                self.assertIn(flag, buf.getvalue())

    def test_refuse_equals_form(self):
        from demo_anomaly import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--publish=1", "--dry-run"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        self.assertIn("Refusing", buf.getvalue())

    def test_no_auroc_calculator_in_module(self):
        src = (ROOT / "demo_anomaly.py").read_text(encoding="utf-8")
        self.assertNotIn("def compute_auroc", src)
        self.assertIn("not_published", src)
        self.assertIn("REFUSED_METRIC_FLAGS", src)
        self.assertIn("EXIT_REFUSED_FLAG", src)
        self.assertIn("EXIT_PATH_ERROR", src)
        self.assertIn("EXIT_GAP_NOT_OBVIOUS", src)


class TestDryRunTinyFixture(unittest.TestCase):
    def test_dry_run_exit_0_honesty_and_no_bpb_invent(self):
        from demo_anomaly import EXIT_OK, main

        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.stdout", buf), mock.patch("sys.stderr", err):
            rc = main(["--dry-run", "--per-class", "2"])
        self.assertEqual(rc, EXIT_OK)
        out = buf.getvalue()
        self.assertIn("dry-run", out.lower())
        self.assertIn("not_published", out.lower())
        self.assertIn("normal", out)
        self.assertIn("anomalous", out)
        self.assertIn("cascade", out)
        # Must not invent numeric AUROC / published ranking
        self.assertNotIn("auroc=", out.lower())
        self.assertNotIn('"auroc"', out.lower())

    def test_dry_run_json_report_claim_status(self):
        from demo_anomaly import EXIT_OK, main

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "anomaly_dry.json")
            rc = main(
                [
                    "--dry-run",
                    "--per-class",
                    "2",
                    "--out",
                    out,
                    "--json",
                ]
            )
            self.assertEqual(rc, EXIT_OK)
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(report["mode"], "dry_run")
            self.assertEqual(report["classes"]["normal"]["n"], 2)
            self.assertEqual(report["classes"]["anomalous"]["n"], 2)
            self.assertEqual(report["classes"]["cascade"]["n"], 2)
            for label in ("normal", "anomalous", "cascade"):
                for s in report["classes"][label]["sessions"]:
                    self.assertIsNone(s["bpb"])
            blob = json.dumps(report).lower()
            # No AUROC / ranking *keys* invented (disclaimer may mention honesty)
            self.assertNotIn('"auroc"', blob)
            self.assertNotIn('"ranking"', blob)
            self.assertNotIn("auroc=", blob)

    def test_tiny_fixture_sessions_deterministic(self):
        from demo_anomaly import make_tiny_fixture_sessions

        a = make_tiny_fixture_sessions(2, seed=7)
        b = make_tiny_fixture_sessions(2, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a["normal"]), 2)
        self.assertIn("ERROR", a["anomalous"][0])
        self.assertIn("cascade", a["cascade"][0].lower())


class TestPathValidation(unittest.TestCase):
    def test_validate_smoke_paths_missing_corpus(self):
        from demo_anomaly import validate_smoke_paths

        class _Fake:
            DATA_DIR = "/no/such/aomb/corpus"
            TOKENIZER_DIR = "/no/such/aomb/tok"
            VAL_FILENAME = "val.parquet"

        err = validate_smoke_paths(_Fake())
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("Missing corpus", err)
        self.assertIn("--dry-run", err)

    def test_main_smoke_missing_corpus_exit_2(self):
        """Smoke path without corpus → EXIT_PATH_ERROR (mocked prepare)."""
        from demo_anomaly import EXIT_PATH_ERROR, main

        class _FakePrep:
            DATA_DIR = "/no/such/aomb/corpus"
            TOKENIZER_DIR = "/no/such/aomb/tok"
            VAL_FILENAME = "val.parquet"

        with mock.patch("demo_anomaly._load_prepare", return_value=_FakePrep()):
            buf = io.StringIO()
            with mock.patch("sys.stderr", buf):
                # Avoid torch import failure before path check: ensure_device
                # is called first — mock it to a simple namespace.
                fake_dev = mock.Mock()
                fake_dev.type = "cpu"
                with mock.patch("demo_anomaly.ensure_device", return_value=fake_dev):
                    with mock.patch.dict(sys.modules, {"torch": mock.Mock()}):
                        rc = main(["--seconds", "1", "--per-class", "1"])
        self.assertEqual(rc, EXIT_PATH_ERROR)
        self.assertIn("Missing corpus", buf.getvalue())


class TestGapExitCode(unittest.TestCase):
    def test_gap_helper_mean_median(self):
        from demo_anomaly import _mean, _median

        self.assertEqual(_mean([1.0, 3.0]), 2.0)
        self.assertEqual(_median([1.0, 2.0, 3.0]), 2.0)
        self.assertTrue(_mean([]) != _mean([]))  # nan


class TestCliSubprocessSmoke(unittest.TestCase):
    """Real process smoke — torch-free dry-run + auroc refuse."""

    def test_subprocess_dry_run(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "demo_anomaly.py"),
                "--dry-run",
                "--per-class",
                "1",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("not_published", proc.stdout.lower())
        self.assertIn("dry_run", proc.stdout.lower() + proc.stderr.lower())

    def test_subprocess_refuse_auroc(self):
        from demo_anomaly import EXIT_REFUSED_FLAG

        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "demo_anomaly.py"),
                "--auroc",
                "--dry-run",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("auroc", proc.stderr.lower())
        self.assertIn("not_published", proc.stderr.lower())


class TestImportTorchFree(unittest.TestCase):
    def test_import_demo_anomaly_without_torch(self):
        """Importing demo_anomaly must not require torch until DEVICE/smoke."""
        # Fresh interpreter check via subprocess
        code = (
            "import sys\n"
            "sys.modules['torch'] = None  # poison if eagerly imported\n"
            "import importlib\n"
            "sys.path.insert(0, %r)\n"
            "# Remove poison — we only care that import itself does not load torch\n"
            "del sys.modules['torch']\n"
            "import demo_anomaly as d\n"
            "assert d.EXIT_OK == 0\n"
            "assert '--auroc' in d.REFUSED_METRIC_FLAGS\n"
            "assert 'torch' not in sys.modules\n"
            "rc = d.main(['--dry-run', '--per-class', '1'])\n"
            "assert rc == 0\n"
            "assert 'torch' not in sys.modules\n"
        ) % str(ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
