"""val_bpb must come from logits — never from the agent-editable loss path."""

from __future__ import annotations

import math
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - CI jobs without torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestEvaluateBpbIgnoresModelLoss(unittest.TestCase):
    VOCAB = 8

    def _model(self):
        vocab = self.VOCAB

        class Lying(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
                self.loss_path_calls = 0

            def forward(self, idx, targets=None, reduction="mean"):
                if targets is not None:
                    # what a gamed train.py does: a "loss" that isn't cross-entropy
                    self.loss_path_calls += 1
                    return torch.zeros(targets.numel())
                return torch.zeros(*idx.shape, vocab)  # uniform logits

        return Lying()

    def test_uniform_logits_give_log2_vocab_bits_per_token(self):
        import prepare

        x = torch.zeros(2, 4, dtype=torch.long)
        y = torch.ones(2, 4, dtype=torch.long)

        def loader(*_a, **_k):
            while True:
                yield x, y, 1

        model = self._model()
        with mock.patch.object(prepare, "get_token_bytes", lambda device="cpu": torch.ones(self.VOCAB, dtype=torch.int32)), \
             mock.patch.object(prepare, "make_dataloader", loader), \
             mock.patch.object(prepare, "EVAL_TOKENS", 2 * 2 * prepare.MAX_SEQ_LEN):
            bpb = prepare.evaluate_bpb(model, tokenizer=None, batch_size=2)

        # 1 byte per token, uniform over 8 tokens → exactly 3 bits per byte
        self.assertAlmostEqual(bpb, math.log2(self.VOCAB), places=5)
        self.assertEqual(model.loss_path_calls, 0, "evaluate_bpb must not use the model's loss path")


if __name__ == "__main__":
    unittest.main()
