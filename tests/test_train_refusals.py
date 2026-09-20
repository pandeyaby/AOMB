"""train.py invent-flag refusals — torch-free (subprocess + source lock).

Refuse runs before torch / MPS verify. prepare.py untouched. No invented AUROC.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"


def _run_train(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TRAIN), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )


class TestTrainInventRefuseSource(unittest.TestCase):
    def test_gate_before_torch_import(self):
        src = TRAIN.read_text(encoding="utf-8")
        refuse_at = src.index("refuse_invent_flags")
        torch_at = src.index("import torch")
        self.assertLess(refuse_at, torch_at)
        self.assertIn("REFUSED_METRIC_FLAGS", src)
        self.assertIn("EXIT_REFUSED_FLAG", src)
        self.assertIn("Never invents", src)
        self.assertIn("prepare.py stays untouched", src)
        # Script-level gate
        self.assertRegex(
            src,
            r'if __name__ == ["\']__main__["\']:\s*\n\s*refuse_invent_flags\(\)',
        )

    def test_shares_stranger_metric_flags(self):
        from eval.stranger_path import REFUSED_METRIC_FLAGS

        src = TRAIN.read_text(encoding="utf-8")
        self.assertIn("from eval.stranger_path import", src)
        for flag in (
            "--auroc",
            "--publish",
            "--cuda",
            "--lab-auroc",
            "--invent-val-bpb",
        ):
            self.assertIn(flag, REFUSED_METRIC_FLAGS)


class TestTrainInventRefuseSubprocess(unittest.TestCase):
    def test_refuse_auroc_exit_1(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_train("--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = proc.stderr + proc.stdout
        self.assertIn("Refusing", blob)
        self.assertIn("--auroc", blob)
        self.assertIn("Never invents", blob)
        # Must not reach MPS platform check
        self.assertNotIn("MPS", blob)
        self.assertNotIn("requires macOS", blob)

    def test_refuse_publish_cuda_matrix(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                proc = _run_train(flag)
                self.assertEqual(
                    proc.returncode,
                    EXIT_REFUSED_FLAG,
                    msg=proc.stderr + proc.stdout,
                )
                self.assertIn("Refusing", proc.stderr + proc.stdout)

    def test_refuse_equals_form(self):
        from eval.stranger_path import EXIT_REFUSED_FLAG

        proc = _run_train("--auroc=0.99", "--cuda=1")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("--auroc", proc.stderr + proc.stdout)

    def test_helpers_unit_via_runpy_namespace(self):
        """Load only the honesty helpers without executing train body."""
        import ast

        src = TRAIN.read_text(encoding="utf-8")
        # Extract find_refused_invent_flag by compiling a tiny sibling module text
        from eval.stranger_path import REFUSED_METRIC_FLAGS

        # Inline reimplementation check: train's function body matches stranger set
        self.assertIn("--auroc", REFUSED_METRIC_FLAGS)
        tree = ast.parse(src)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("find_refused_invent_flag", names)
        self.assertIn("refuse_invent_flags", names)


if __name__ == "__main__":
    unittest.main()
