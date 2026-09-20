"""Unit tests: morning_report Tale measured-card read (temp fixtures).

No MPS / Zenodo. No invented AUROC / val_bpb. prepare.py untouched.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from morning_report import (  # noqa: E402
    EXIT_REFUSED_FLAG,
    TaleCardSnapshot,
    cli,
    format_tale_card_section,
    read_tale_card,
    refuse_loud_flags,
    resolve_tale_card_path,
)


def _write_card(path: Path, **fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


class TestReadTaleCard(unittest.TestCase):
    def test_good_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _write_card(
                Path(tmp) / "measured.json",
                claim_status="measured_not_published",
                val_bpb=1.37952,
            )
            snap = read_tale_card(card)
            self.assertEqual(snap.state, "ok")
            self.assertAlmostEqual(snap.val_bpb, 1.37952, places=5)
            self.assertEqual(snap.claim_status, "measured_not_published")
            self.assertFalse(snap.is_public_accuracy_claim)

    def test_missing_card(self):
        snap = read_tale_card(Path("/no/such/measured_card.json"))
        self.assertEqual(snap.state, "missing")
        self.assertIsNone(snap.val_bpb)

    def test_null_val_bpb(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _write_card(
                Path(tmp) / "c.json",
                claim_status="measured_not_published",
                val_bpb=None,
            )
            snap = read_tale_card(card)
            self.assertEqual(snap.state, "unavailable")
            self.assertIsNone(snap.val_bpb)

    def test_pending_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _write_card(
                Path(tmp) / "c.json",
                claim_status="pending",
                val_bpb=1.37952,
            )
            snap = read_tale_card(card)
            self.assertEqual(snap.state, "pending")
            self.assertIsNone(snap.val_bpb)

    def test_wrong_status_still_no_invent(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _write_card(
                Path(tmp) / "c.json",
                claim_status="published",
                val_bpb=1.37952,
            )
            snap = read_tale_card(card)
            # Finite val may still display with claim_status, but never as public claim
            self.assertEqual(snap.state, "ok")
            self.assertAlmostEqual(snap.val_bpb, 1.37952, places=5)
            self.assertFalse(snap.is_public_accuracy_claim)
            text = "\n".join(format_tale_card_section(snap))
            self.assertIn("NOT a public accuracy", text)
            self.assertNotIn("AUROC=", text)

    def test_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")
            snap = read_tale_card(path)
            self.assertEqual(snap.state, "malformed")
            self.assertIsNone(snap.val_bpb)

    def test_format_pending_unavailable_wording(self):
        snap = TaleCardSnapshot(
            path=Path("x.json"),
            state="pending",
            val_bpb=None,
            claim_status="pending",
        )
        text = "\n".join(format_tale_card_section(snap))
        self.assertIn("pending / unavailable", text)
        self.assertIn("no invented val_bpb", text)


class TestResolveTaleCardPath(unittest.TestCase):
    def test_cli_wins(self):
        path = Path("/tmp/card.json")
        self.assertEqual(
            resolve_tale_card_path(cli_path=path, environ={}),
            path,
        )

    def test_env_enables_default(self):
        from morning_report import DEFAULT_TALE_CARD

        got = resolve_tale_card_path(
            cli_path=None, environ={"AOMB_TALE_CARD": "1"}
        )
        self.assertEqual(got, DEFAULT_TALE_CARD)

    def test_env_off(self):
        self.assertIsNone(
            resolve_tale_card_path(cli_path=None, environ={})
        )


class TestCliRefuseAndCard(unittest.TestCase):
    def test_refuse_invent_flags(self):
        for flag in ("--auroc", "--publish", "--cuda", "--invent-val-bpb"):
            self.assertEqual(refuse_loud_flags([flag]), flag)
            self.assertEqual(cli([flag]), EXIT_REFUSED_FLAG)

    def test_cli_tale_card_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _write_card(
                Path(tmp) / "measured.json",
                claim_status="measured_not_published",
                val_bpb=1.37952,
            )
            with mock.patch("morning_report.parse_git_log", return_value=[]):
                with mock.patch("builtins.print") as pr:
                    code = cli(["--tale-card", str(card)])
            self.assertEqual(code, 0)
            joined = "\n".join(
                str(c.args[0]) if c.args else "" for c in pr.call_args_list
            )
            self.assertIn("1.379520", joined)
            self.assertIn("measured_not_published", joined)
            self.assertIn("NOT a public accuracy", joined)

    def test_cli_tale_card_missing(self):
        with mock.patch("morning_report.parse_git_log", return_value=[]):
            with mock.patch("builtins.print") as pr:
                code = cli(["--tale-card", "/no/such/card.json"])
        self.assertEqual(code, 0)
        joined = "\n".join(
            str(c.args[0]) if c.args else "" for c in pr.call_args_list
        )
        self.assertIn("unavailable", joined)
        self.assertNotRegex(joined, r"card val_bpb\s*:\s*1\.")


if __name__ == "__main__":
    unittest.main()
