"""LogHub HDFS and RCAEval adapters: rendering and identifier normalisation."""

from __future__ import annotations

import unittest

from corpus.ingest.loghub_hdfs import render, select_blocks
from corpus.ingest.loghub_bgl import render as bgl_render
from corpus.ingest.rcaeval import normalise, render_trace
from eval.in_domain import _noise_mask


class TestHdfsAdapter(unittest.TestCase):
    def test_render_masks_identifiers_and_keeps_message(self):
        line = (
            "081110 224621 33 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: "
            "/user/root/randtxt3/_temporary/_task_200811101024_0007_m_001253_0/part-01253. "
            "blk_7554337092993489054"
        )
        out = render(line)
        self.assertTrue(out.startswith("[ts=081110T224621] [src=OTelLog] level=INFO svc=dfs.FSNamesystem msg="))
        for leaked in ("randtxt3", "200811101024", "01253", "7554337092993489054", " 33 "):
            self.assertNotIn(leaked, out)
        self.assertIn("allocateBlock", out)

    def test_render_masks_ips_but_keeps_sizes(self):
        out = render(
            "081111 064819 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: "
            "blockMap updated: 10.251.194.147:50010 is added to blk_779714437703947168 size 64376891"
        )
        self.assertNotIn("10.251", out)
        self.assertIn("size_64376891", out)

    def test_select_blocks_trains_on_early_normals_only(self):
        labels = [("b0", "Normal"), ("b1", "Anomaly"), ("b2", "Normal"), ("b3", "Normal"), ("b4", "Anomaly")]
        train, ev = select_blocks(labels, n_train=2, n_eval=10, seed=0)
        self.assertEqual(train, ["b0", "b2"])
        self.assertEqual(sorted(ev), ["b3", "b4"])  # strictly after the last training block


class TestBglAdapter(unittest.TestCase):
    NODE = "R30-M0-N9-C:J16-U01"

    def _line(self, label, level, msg):
        return f"{label} 1118536327 2005.06.11 {self.NODE} 2005-06-11-17.32.07.581048 {self.NODE} RAS KERNEL {level} {msg}".rstrip()

    def test_label_column_never_reaches_the_text(self):
        is_alert, text = bgl_render(self._line("KERNDTLB", "FATAL", "data TLB error interrupt"))
        self.assertTrue(is_alert)
        self.assertNotIn("KERNDTLB", text)
        self.assertEqual(text, "[ts=1118536327] [src=OTelLog] level=FATAL svc=RAS.KERNEL msg=data_TLB_error_interrupt")

    def test_identifiers_are_masked_and_empty_messages_kept(self):
        _, text = bgl_render(self._line("-", "INFO", f"63543 exceptions at 0x00544eb8 on {self.NODE}"))
        self.assertIn("msg=NUM_exceptions_at_HEX_on_NODE", text)
        is_alert, empty = bgl_render(self._line("-", "FATAL", ""))
        self.assertFalse(is_alert)
        self.assertTrue(empty.endswith("level=FATAL svc=RAS.KERNEL msg=empty"))

    def test_severity_rule_counts_bgl_levels(self):
        from eval.in_domain import _ERROR

        _, fatal = bgl_render(self._line("-", "FATAL", "x"))
        _, info = bgl_render(self._line("-", "INFO", "x"))
        self.assertEqual(len(_ERROR.findall(fatal)), 1)
        self.assertEqual(len(_ERROR.findall(info)), 0)


class TestRcaevalAdapter(unittest.TestCase):
    def test_render_trace_uses_span_format_and_masks_ids(self):
        spans = [
            {"traceID": "t1", "spanID": "s2", "parentSpanID": "s1", "serviceName": "cart",
             "operationName": "GET /cart/0f8fad5b-d9cb-469f-a165-70867728950e", "startTimeMillis": 1733591047900,
             "startTime": 1733591047900000, "duration": 11371, "statusCode": None},
            {"traceID": "t1", "spanID": "s1", "parentSpanID": "None", "serviceName": "frontend",
             "operationName": "GET /cart", "startTimeMillis": 1733591047847,
             "startTime": 1733591047847000, "duration": 18682, "statusCode": 500},
        ]
        text = render_trace(spans)
        first, second = text.split("\n")
        self.assertIn("svc=frontend duration_ms=18 status=500", first)  # sorted by start time
        self.assertIn("parent=n/a", first)
        self.assertIn("op=GET_/cart/UUID", second)
        masked = "".join("_" if m else c for c, m in zip(text, _noise_mask(text, [(c, 1.0, 1) for c in text])))
        self.assertNotIn("1733591047", masked)  # timestamps masked for scoring

    def test_normalise_keeps_short_numbers(self):
        self.assertEqual(normalise("GET /api/v1/items 42"), "GET_/api/v1/items_42")
        self.assertEqual(normalise("order 123456"), "order_NUM")


if __name__ == "__main__":
    unittest.main()
