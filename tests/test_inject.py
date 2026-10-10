"""Fault injector: deterministic, touches only what it says, never IDs or clocks."""

from __future__ import annotations

import unittest

from eval.inject import CORRUPTIONS, Injector, injection_auroc


def _session(region="us-east", currency="USD", dur=5, hits=7):
    return "\n".join([
        f"[ts=1.0] [src=OTel] trace_id=aa11 span_id=bb22 parent=n/a op=GET_/api/checkout svc=api duration_ms={dur} status=ok",
        f"[ts=1.1] [src=OTel] trace_id=aa11 span_id=cc33 parent=bb22 op=db.ping svc=api duration_ms=1 status=ok",
        f"[ts=1.2] [src=OTelLog] level=INFO svc=api msg=checkout_ok_db=ok_hits={hits}_region={region}_currency={currency}",
    ])


REFERENCE = [_session("us-east", "USD"), _session("eu-west", "EUR")]


class TestInjector(unittest.TestCase):
    def setUp(self):
        self.inj = Injector(REFERENCE, seed=0)
        self.clean = [_session("us-east", "USD", hits=11), _session("eu-west", "EUR", hits=12)]
        self.out = self.inj.corrupt(self.clean)

    def test_every_corruption_applies_and_changes_the_text(self):
        self.assertEqual(set(self.out), set(CORRUPTIONS))
        for name, items in self.out.items():
            self.assertTrue(items, name)
            for i, text in items:
                self.assertNotEqual(text, self.clean[i], name)

    def test_deterministic(self):
        self.assertEqual(self.out, Injector(REFERENCE, seed=0).corrupt(self.clean))
        self.assertNotEqual(self.out, Injector(REFERENCE, seed=1).corrupt(self.clean))

    def test_ids_timestamps_and_counters_are_never_edited(self):
        for name in ("value_swap", "value_cross", "latency"):
            for i, text in self.out[name]:
                for keep in ("trace_id=aa11", "span_id=bb22", "parent=bb22", "[ts=1.0]", f"hits={11 + i}"):
                    self.assertIn(keep, text, name)

    def test_structural_corruptions_change_line_counts_as_stated(self):
        n = self.clean[0].count("\n") + 1
        count = lambda name: {i: t.count("\n") + 1 for i, t in self.out[name]}[0]
        self.assertEqual(count("drop_line"), n - 1)
        self.assertEqual(count("dup_line"), n + 1)
        self.assertLess(count("truncate"), n)
        self.assertEqual(count("reorder"), n)

    def _changed_field(self, name):
        from eval.inject import _fields

        text = dict(self.out[name])[0]
        for old_line, new_line in zip(self.clean[0].split("\n"), text.split("\n")):
            if old_line != new_line:
                old = {k: v for k, v, _a, _b in _fields(old_line)}
                for k, v, _a, _b in _fields(new_line):
                    if old.get(k) != v:
                        return k, v
        self.fail(f"{name}: no changed field found")

    def test_value_swap_uses_a_seen_value_and_cross_uses_an_unseen_one(self):
        k, v = self._changed_field("value_swap")
        self.assertIn(v, self.inj.pool[k])  # familiar for that field
        k, v = self._changed_field("value_cross")
        self.assertNotIn(v, self.inj.pool[k])  # never seen in that field
        self.assertIn(v, self.inj.all_values)

    def test_latency_multiplies_a_duration(self):
        import re

        before = [int(x) for x in re.findall(r"duration_ms=(\d+)", self.clean[0])]
        after = [int(x) for x in re.findall(r"duration_ms=(\d+)", dict(self.out["latency"])[0])]
        self.assertEqual(sum(a != b for a, b in zip(after, before)), 1)
        self.assertGreaterEqual(max(a / max(b, 1) for a, b in zip(after, before)), 10)

    def test_single_line_session_skips_inapplicable_corruptions(self):
        one = ["[ts=1] [src=OTelLog] level=INFO svc=api msg=hello"]
        out = Injector(REFERENCE).corrupt(one)
        self.assertEqual(out["truncate"], [])
        self.assertEqual(out["drop_line"], [])
        self.assertEqual(out["latency"], [])

    def test_injection_auroc(self):
        self.assertEqual(injection_auroc([0.1, 0.2], [0.8, 0.9]), 1.0)
        self.assertEqual(injection_auroc([0.8, 0.9], [0.1, 0.2]), 0.0)


if __name__ == "__main__":
    unittest.main()
