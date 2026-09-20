"""Unit tests: public-wins Tale measured-card one-liner (temp cards).

No MPS / Zenodo. Never invents AUROC / val_bpb. prepare.py untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.public_wins_tale_line import (  # noqa: E402
    EXIT_OK,
    EXIT_PATH_ERROR,
    EXIT_REFUSED_FLAG,
    factual_line_from_card,
    load_measured_card,
    main,
    run,
)


def _write(path: Path, **fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


class TestFactualLine(unittest.TestCase):
    def test_good_card(self):
        line = factual_line_from_card(
            {
                "claim_status": "measured_not_published",
                "val_bpb": 1.37952,
                "max_spans": 200000,
            }
        )
        self.assertIsNotNone(line)
        assert line is not None
        self.assertIn("val_bpb=1.379520", line)
        self.assertIn("claim_status=measured_not_published", line)
        self.assertIn("max_spans=200000", line)
        self.assertIn("train fitness only, not AUROC", line)
        self.assertNotIn("AUROC=", line)

    def test_pending_refuses_print(self):
        self.assertIsNone(
            factual_line_from_card(
                {"claim_status": "pending", "val_bpb": 1.37952, "max_spans": 200000}
            )
        )

    def test_null_val(self):
        self.assertIsNone(
            factual_line_from_card(
                {
                    "claim_status": "measured_not_published",
                    "val_bpb": None,
                    "max_spans": 200000,
                }
            )
        )

    def test_nan_val(self):
        self.assertIsNone(
            factual_line_from_card(
                {
                    "claim_status": "measured_not_published",
                    "val_bpb": float("nan"),
                    "max_spans": 200000,
                }
            )
        )


class TestRunAndCli(unittest.TestCase):
    def test_missing_card(self):
        code, line = run(Path("/no/such/measured_card.json"))
        self.assertEqual(code, EXIT_PATH_ERROR)
        self.assertIn("unavailable", line)

    def test_malformed_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")
            code, line = run(path)
        self.assertEqual(code, EXIT_PATH_ERROR)
        self.assertIn("unavailable", line)

    def test_good_temp_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "c.json",
                claim_status="measured_not_published",
                val_bpb=1.37952,
                max_spans=200000,
            )
            code, line = run(path)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("1.379520", line)

    def test_cli_refuse(self):
        for flag in ("--auroc", "--publish", "--cuda"):
            self.assertEqual(main([flag]), EXIT_REFUSED_FLAG)

    def test_cli_with_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "c.json",
                claim_status="measured_not_published",
                val_bpb=1.37952,
                max_spans=200000,
            )
            self.assertEqual(main(["--card", str(path)]), EXIT_OK)

    def test_committed_card_smoke(self):
        card = ROOT / "reports" / "tale-capped" / "measured_capped_200k.json"
        if not card.is_file():
            self.skipTest("committed card not present")
        data = load_measured_card(card)
        self.assertIsNotNone(data)
        code, line = run(card)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("train fitness only, not AUROC", line)

    def test_shell_wrapper_refuse(self):
        script = ROOT / "scripts" / "public_wins_tale_line.sh"
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
