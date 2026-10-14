"""
Train with ``train.py``, then score the model on injected faults.

The agent loop's alternative objective (``AOMB_OBJECTIVE=inject``). It runs
``train.py`` from the working directory unmodified, then measures how well the
trained model ranks corrupted copies of held-out *normal* sessions above the
clean originals (eval/inject.py), and prints::

    val_bpb:       <true bits-per-byte, from train.py's own final eval>
    inject_auroc:  <mean AUROC over corruption types>

The metric is computed here, from the model's logits, by code the agent loop
never rewrites (it only writes ``train.py``). The agent never sees labelled
incidents: the sessions scored are normal ones, read from

    AOMB_OBJECTIVE_VAL    JSONL of held-out normal sessions (scored)
    AOMB_OBJECTIVE_TRAIN  JSONL whose ``split == "train"`` rows fit the value pool

Scoring is fixed: the pre-registered rule ``bpb_top10`` with the end-of-session
marker (docs/real-data/scoring-rule-preregistration.md).

This guards against a metric drifting because ``train.py`` changed how a loss
is computed, which is what happened to val_bpb. It is not a sandbox: train.py
runs in the same process.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "bpb_top10"
N_CLEAN = 300


def read_texts(path: str, split: str | None) -> list[str]:
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (split is None or r.get("split") == split) and r.get("label") == "normal" and r["text"]:
            out.append(r["text"])
    return out


def inject_scores(model, tokenizer, token_bytes, max_seq_len: int, clean: list[str], reference: list[str]) -> dict:
    """{corruption: AUROC, ..., 'mean': mean AUROC} for the fixed rule."""
    from eval.in_domain import END_MARKER, model_scores_for, token_surprise
    from eval.inject import Injector, injection_auroc

    def score(text: str) -> float:
        text = text + "\n" + END_MARKER
        return model_scores_for(token_surprise(model, tokenizer, token_bytes, text, max_seq_len), text)[RULE]

    clean_scores = [score(t) for t in clean]
    out: dict[str, float] = {}
    for name, items in Injector(reference, seed=0).corrupt(clean).items():
        if items:
            out[name] = injection_auroc(clean_scores, [score(t) for _i, t in items])
    out["mean"] = sum(out.values()) / len(out) if out else float("nan")
    return out


def main() -> int:
    val_path = os.environ.get("AOMB_OBJECTIVE_VAL")
    train_path = os.environ.get("AOMB_OBJECTIVE_TRAIN")
    if not (val_path and train_path):
        print("ERROR: set AOMB_OBJECTIVE_VAL and AOMB_OBJECTIVE_TRAIN", file=sys.stderr)
        return 2
    clean = read_texts(val_path, None)[:N_CLEAN]
    reference = read_texts(train_path, "train")

    source = (ROOT / "train.py").read_text(encoding="utf-8")
    module = types.ModuleType("_aomb_objective_train")
    module.__file__ = str(ROOT / "train.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "train.py", "exec"), module.__dict__)  # trains; prints val_bpb

    import torch

    import prepare

    ns = module.__dict__
    model = ns["model"]
    model.eval()
    token_bytes = prepare.get_token_bytes(device=str(next(model.parameters()).device))
    with torch.no_grad():
        scores = inject_scores(model, ns["tokenizer"], token_bytes, prepare.MAX_SEQ_LEN, clean, reference)

    print("--- injected-fault objective")
    for name, v in scores.items():
        if name != "mean":
            print(f"inject_{name}: {v:.6f}")
    print(f"inject_auroc:     {scores['mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
