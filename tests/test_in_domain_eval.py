"""Tests for the in-domain lab eval (eval/in_domain.py) — pure logic, no torch."""

from __future__ import annotations

import math
import unittest

from eval.in_domain import (
    _noise_mask,
    baseline_scores,
    fit_duration_stats,
    fit_novelty,
    log_template,
    model_scores_for,
    temporal_split,
)
from eval.labels import LabeledSession


def _sess(sid, label, text, fault=""):
    return LabeledSession(
        session_id=sid,
        text=text,
        label=label,
        binary={"normal": 0, "incident": 1}[label],
        fault=fault,
        n_chars=len(text),
    )


def _span(ts, op, dur, status="ok"):
    return (
        f"[ts=2026-09-18T05:23:{ts:02d}.000Z] [src=OTel] trace_id=ab12 span_id=cd34 "
        f"parent=n/a op={op} svc=api duration_ms={dur} status={status}"
    )


class TestTemporalSplit(unittest.TestCase):
    def test_earlier_normals_train_later_normals_and_incidents_eval(self):
        sessions = [_sess(f"n{i}", "normal", _span(i, "GET", 5)) for i in range(4)]
        sessions.append(_sess("i0", "incident", _span(30, "GET", 800)))
        cap = {s.session_id: "c1" for s in sessions}
        train, ev = temporal_split(sessions, cap)
        self.assertEqual([s.session_id for s in train], ["n0", "n1"])
        self.assertEqual({s.session_id for s in ev}, {"n2", "n3", "i0"})
        self.assertTrue(all(s.binary == 0 for s in train))


class TestBaselines(unittest.TestCase):
    def test_duration_z_and_error_rule(self):
        train = [_sess(f"n{i}", "normal", _span(i, "GET", 4 + i % 3)) for i in range(6)]
        stats = fit_duration_stats(train)
        ev = [
            _sess("ok", "normal", _span(10, "GET", 5)),
            _sess("slow", "incident", _span(11, "GET", 800)),
            _sess("err", "incident", _span(12, "GET", 5, status="error")),
        ]
        sc = baseline_scores(ev, stats)
        self.assertGreater(sc["duration_z"][1], 5 * max(sc["duration_z"][0], 0.1))
        self.assertEqual(sc["error_lines"], [0.0, 0.0, 1.0])
        # error outranks everything under the rule; slow outranks ok
        self.assertEqual(sorted(range(3), key=lambda i: sc["rule"][i]), [0, 1, 2])


class TestNoveltyBaseline(unittest.TestCase):
    def _log(self, msg, level="INFO"):
        return f"[ts=2026-09-18T05:23:01.000Z] [src=OTelLog] level={level} svc=api msg={msg}"

    def test_template_masks_values_but_not_new_messages(self):
        self.assertEqual(
            log_template("INFO", "api", "checkout_ok_db=ok_hits=7"),
            log_template("INFO", "api", "checkout_ok_db=replica_hits=123"),
        )
        self.assertNotEqual(
            log_template("INFO", "api", "checkout_ok_db=ok_hits=7"),
            log_template("WARN", "api", "pricing_fallback_source=static"),
        )

    def test_novelty_flags_new_template_op_and_shape_only(self):
        normal = "\n".join([_span(1, "GET", 5), self._log("checkout_ok_db=ok_hits=7")])
        train = [_sess("n0", "normal", normal)]
        stats = fit_duration_stats(train)
        seen = fit_novelty(train)
        ev = [
            _sess("same", "normal", normal.replace("hits=7", "hits=9")),
            _sess("drift", "incident", normal.replace("db=ok", "db=replica")),
            _sess("newlog", "incident", normal + "\n" + self._log("pricing_fallback_x=1", "WARN")),
            _sess("retry", "incident", normal + "\n" + _span(2, "GET", 5)),
        ]
        nov = baseline_scores(ev, stats, seen)["novelty"]
        self.assertEqual(nov[0], 0.0)
        self.assertEqual(nov[1], 0.0)  # value drift is invisible to templates
        self.assertEqual(nov[2], 1.0)  # new template
        self.assertEqual(nov[3], 1.0)  # new trace shape (extra span)


class TestModelScoring(unittest.TestCase):
    def test_noise_mask_covers_ids_and_timestamp_only(self):
        text = _span(1, "GET", 150)
        toks = [(c, 1.0, 1) for c in text]
        masked = "".join("_" if m else c for c, m in zip(text, _noise_mask(text, toks)))
        self.assertIn("[ts=________________________]", masked)
        self.assertIn("trace_id=____", masked)
        self.assertIn("parent=___", masked)
        self.assertIn("duration_ms=150", masked)

    def test_content_score_ignores_surprise_in_ids(self):
        text = _span(1, "GET", 5)
        toks = [(c, 1.0, 1) for c in text]
        base = model_scores_for(toks, text)
        # make every ID/timestamp char hugely surprising — content score must not move
        mask = _noise_mask(text, toks)
        noisy = [(c, 50.0 if m else n, b) for (c, n, b), m in zip(toks, mask)]
        out = model_scores_for(noisy, text)
        self.assertAlmostEqual(out["bpb_content"], base["bpb_content"])
        self.assertGreater(out["bpb_mean"], base["bpb_mean"])
        self.assertAlmostEqual(base["bpb_content"], 1.0 / math.log(2))


if __name__ == "__main__":
    unittest.main()
