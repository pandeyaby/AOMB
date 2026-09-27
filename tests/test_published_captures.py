"""Published lab captures must be byte-identical to what the reports evaluated."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from eval.labels import content_hash_capture, load_lab_sessions

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "lab" / "published"

# capture dir → report file whose recorded content hash it must match
REPORTS = {
    "pooled-20260918": "reports/public-accuracy/lab-pooled-crisp-zeroshot-20260925/model-seeds/seed-0/report.json",
    "pooled-20260925-ruleproof": "reports/public-accuracy/lab-ruleproof-in-domain-20260925/results.json",
    "pooled-20260926-valuedrift": "reports/public-accuracy/lab-valuedrift-in-domain-20260926/results.json",
}


def _recorded_hash(report: dict) -> str:
    return report.get("content_sha256") or report["corpus"]["content_sha256"]


class TestPublishedCaptures(unittest.TestCase):
    def test_every_published_capture_is_hash_pinned(self):
        dirs = sorted(p.name for p in PUBLISHED.iterdir() if p.is_dir())
        self.assertTrue(dirs)
        self.assertEqual(set(dirs), set(REPORTS), "add new published captures to REPORTS")

    def test_hashes_match_reports(self):
        for name, rel in REPORTS.items():
            with self.subTest(capture=name):
                report = json.loads((ROOT / rel).read_text(encoding="utf-8"))
                self.assertEqual(content_hash_capture(PUBLISHED / name), _recorded_hash(report))

    def test_captures_load_with_both_classes(self):
        for name in REPORTS:
            with self.subTest(capture=name):
                sessions, meta = load_lab_sessions(PUBLISHED / name)
                labels = {s.label for s in sessions}
                self.assertTrue({"normal", "incident"} <= labels)
                self.assertEqual(meta["capture_id"], name)


if __name__ == "__main__":
    unittest.main()
