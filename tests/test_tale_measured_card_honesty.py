"""Honesty lock: committed Tale measured card + docs stay non-inventing.

Loads reports/tale-capped/measured_capped_200k.json (source of truth after
#60). Never invents AUROC / val_bpb. prepare.py untouched. No MPS / Zenodo.
"""

from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "reports" / "tale-capped" / "measured_capped_200k.json"
DOCS_PATH = ROOT / "docs" / "tale-val-bpb-baseline.md"

# Keys that would invent ranking / accuracy claims on this train-lane card.
FORBIDDEN_KEY_SUBSTR = (
    "auroc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "ranking",
    "publish",
    "accuracy",
    "hero",
)


def _load_card() -> dict:
    raw = CARD_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert isinstance(data, dict)
    return data


class TestTaleMeasuredCardHonestyLock(unittest.TestCase):
    def test_card_file_present(self):
        self.assertTrue(CARD_PATH.is_file(), f"missing card: {CARD_PATH}")

    def test_claim_status_measured_not_published(self):
        card = _load_card()
        self.assertEqual(card.get("claim_status"), "measured_not_published")

    def test_val_bpb_finite_round_trip(self):
        """Read val_bpb from the committed card — never invent a substitute."""
        text = CARD_PATH.read_text(encoding="utf-8")
        card = json.loads(text)
        val = card.get("val_bpb")
        self.assertIsInstance(val, (int, float))
        self.assertNotIsInstance(val, bool)
        self.assertTrue(math.isfinite(float(val)), msg=repr(val))
        again = json.loads(json.dumps({"val_bpb": val}))["val_bpb"]
        self.assertEqual(again, val)
        footer = f"{float(val):.6f}"
        self.assertAlmostEqual(float(footer), float(val), places=6)
        # Sanity vs committed Mac measured value (file is source of truth).
        self.assertAlmostEqual(float(val), 1.37952, places=5)

    def test_corpus_and_max_spans(self):
        card = _load_card()
        self.assertEqual(card.get("corpus"), "tale_of_errors")
        self.assertEqual(card.get("max_spans"), 200000)

    def test_no_invent_metric_keys(self):
        card = _load_card()
        keys_lower = {str(k).lower() for k in card.keys()}
        for needle in FORBIDDEN_KEY_SUBSTR:
            for key in keys_lower:
                self.assertNotIn(
                    needle,
                    key,
                    msg=f"forbidden invent-ish key {key!r} contains {needle!r}",
                )

    def test_source_id_tale_lane(self):
        card = _load_card()
        self.assertEqual(card.get("source_id"), "uber-tale-of-errors")


class TestTaleBaselineDocsSyncOptional(unittest.TestCase):
    """If docs cite a numeric capped val_bpb, it must match the card (SoT)."""

    def test_docs_numeric_val_bpb_matches_card(self):
        self.assertTrue(DOCS_PATH.is_file(), f"missing docs: {DOCS_PATH}")
        docs = DOCS_PATH.read_text(encoding="utf-8")
        card = _load_card()
        card_val = float(card["val_bpb"])
        card_footer = f"{card_val:.6f}"

        measured_section = docs
        if "## Measured row" in docs:
            measured_section = docs.split("## Measured row", 1)[1]
            nxt = measured_section.find("\n## ")
            if nxt != -1:
                measured_section = measured_section[:nxt]

        pending_as_value = re.search(
            r"\*\*val_bpb\*\*\s*\|\s*\*\*pending\*\*",
            measured_section,
            re.IGNORECASE,
        )
        self.assertIsNone(
            pending_as_value,
            "docs still mark capped val_bpb as pending; card is source of truth",
        )

        nums = re.findall(
            r"val_bpb\*\*\s*\|\s*\*\*([0-9]+\.[0-9]+)\*\*",
            measured_section,
            re.IGNORECASE,
        )
        nums += re.findall(
            r"\bval_bpb\b[^0-9]{0,40}([0-9]+\.[0-9]+)",
            measured_section,
            re.IGNORECASE,
        )
        self.assertTrue(
            nums,
            "measured section should cite the factual card val_bpb numerically",
        )
        for raw in nums:
            self.assertAlmostEqual(
                float(raw),
                card_val,
                places=5,
                msg=f"docs val_bpb {raw} != card {card_val}",
            )
        self.assertIn(card_footer, docs)


if __name__ == "__main__":
    unittest.main()
