"""
Train with ``train.py``, then score the model on injected faults.

The agent loop's alternative objective (``AOMB_OBJECTIVE=inject``). It runs
``train.py`` from the working directory unmodified, then measures how well the
trained model ranks corrupted copies of held-out *normal* sessions above the
clean originals (eval/inject.py), and prints::

    val_bpb:       <true bits-per-byte, from train.py's own final eval>
    inject_auroc:  <mean AUROC over corruption types>

The agent never sees labelled incidents: the sessions scored are normal ones, read from

    AOMB_OBJECTIVE_VAL    JSONL of held-out normal sessions (scored)
    AOMB_OBJECTIVE_TRAIN  JSONL whose ``split == "train"`` rows fit the value pool

Scoring is fixed: the pre-registered rule ``bpb_top10`` with the end-of-session
marker (docs/real-data/scoring-rule-preregistration.md).

Isolation. ``train.py`` is written by the agent, so it runs in a **child
process** (``--child``) that never imports the scoring code's state and is not
told which texts are corrupted. The parent builds the clean and corrupted
texts, shuffles them into one unlabelled list, and asks the child only for the
model's per-token surprise (nats) on each. The parent then tokenizes each text
itself, checks the child returned one finite number per token, applies the
scoring rule and computes the AUROCs. A ``train.py`` that monkeypatches
``eval.inject``, ``eval.in_domain`` or ``eval.metrics`` therefore changes
nothing the parent computes; the parent also drops any ``inject_*`` line the
child prints, so it can't forge the score the loop reads.

What remains trusted: the per-token probabilities come from the agent's model
code, which is the thing being optimised. ``train.py`` also runs with the
user's file permissions, so this is a guard against accidental or casual
metric gaming, not a sandbox against a hostile program.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "bpb_top10"
N_CLEAN = 300
_ENV_KEYS = ("AOMB_OBJECTIVE_VAL", "AOMB_OBJECTIVE_TRAIN")


def read_texts(path: str, split: str | None) -> list[str]:
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (split is None or r.get("split") == split) and r.get("label") == "normal" and r["text"]:
            out.append(r["text"])
    return out


def with_marker(text: str) -> str:
    from eval.in_domain import END_MARKER

    return text + "\n" + END_MARKER


# ---------------------------------------------------------------- child (agent code)


def child(texts_path: str, out_path: str) -> int:
    """Run train.py, then write the trained model's per-token nats for each text."""
    texts = [json.loads(line)["text"] for line in Path(texts_path).read_text(encoding="utf-8").splitlines()]

    source = (ROOT / "train.py").read_text(encoding="utf-8")
    module = types.ModuleType("_aomb_objective_train")
    module.__file__ = str(ROOT / "train.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "train.py", "exec"), module.__dict__)  # trains; prints val_bpb

    import torch

    import prepare
    from eval.in_domain import token_surprise

    ns = module.__dict__
    model = ns["model"]
    model.eval()
    token_bytes = prepare.get_token_bytes(device=str(next(model.parameters()).device))
    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            toks = token_surprise(model, ns["tokenizer"], token_bytes, text, prepare.MAX_SEQ_LEN)
            f.write(json.dumps([n for _t, n, _b in toks]) + "\n")
    return 0


# ---------------------------------------------------------------- parent (trusted)


def score_texts(texts: list[str], nats: list[list[float]]) -> list[float]:
    """Apply the fixed rule to the child's per-token nats, using the parent's own tokens."""
    import prepare
    from eval.in_domain import model_scores_for

    tokenizer = prepare.Tokenizer.from_directory()
    token_bytes = prepare.get_token_bytes(device="cpu").tolist()
    if len(nats) != len(texts):
        raise ValueError(f"child returned {len(nats)} score rows for {len(texts)} texts")
    out = []
    for i, (text, row) in enumerate(zip(texts, nats)):
        ids = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())[1:]
        if len(row) != len(ids) or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in row):
            raise ValueError(f"text {i}: child returned {len(row)} values for {len(ids)} tokens, or non-finite values")
        toks = [(tokenizer.decode([tid]), float(n), int(token_bytes[tid])) for tid, n in zip(ids, row)]
        out.append(model_scores_for(toks, text)[RULE])
    return out


def build_items(clean: list[str], reference: list[str]) -> list[tuple[str, str, int]]:
    """[(corruption or 'clean', text, label)] for every clean session and its corrupted copies."""
    from eval.inject import Injector

    items = [("clean", with_marker(t), 0) for t in clean]
    for name, corrupted in Injector(reference, seed=0).corrupt(clean).items():
        items += [(name, with_marker(t), 1) for _i, t in corrupted]
    return items


def aurocs(items: list[tuple[str, str, int]], scores: list[float]) -> dict[str, float]:
    from eval.inject import injection_auroc

    clean = [s for (name, _t, _y), s in zip(items, scores) if name == "clean"]
    out: dict[str, float] = {}
    for name in dict.fromkeys(n for n, _t, _y in items if n != "clean"):
        out[name] = injection_auroc(clean, [s for (n, _t, _y), s in zip(items, scores) if n == name])
    out["mean"] = sum(out.values()) / len(out) if out else float("nan")
    return out


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3])

    val_path = os.environ.get("AOMB_OBJECTIVE_VAL")
    train_path = os.environ.get("AOMB_OBJECTIVE_TRAIN")
    if not (val_path and train_path):
        print("ERROR: set AOMB_OBJECTIVE_VAL and AOMB_OBJECTIVE_TRAIN", file=sys.stderr)
        return 2
    items = build_items(read_texts(val_path, None)[:N_CLEAN], read_texts(train_path, "train"))
    order = list(range(len(items)))
    random.Random(0).shuffle(order)  # the child sees one unlabelled, shuffled list

    with tempfile.TemporaryDirectory(prefix="aomb-objective-") as tmp:
        texts_path, out_path = Path(tmp) / "texts.jsonl", Path(tmp) / "nats.jsonl"
        texts_path.write_text("".join(json.dumps({"text": items[i][1]}) + "\n" for i in order), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
        proc = subprocess.run(
            [sys.executable, "-m", "eval.objective", "--child", str(texts_path), str(out_path)],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        # pass the child's log through, minus anything that looks like our score
        sys.stdout.write("".join(ln for ln in proc.stdout.splitlines(True) if not ln.lstrip().startswith("inject_")))
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"ERROR: train.py child exited with code {proc.returncode}", file=sys.stderr)
            return 1
        nats = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    shuffled = score_texts([items[i][1] for i in order], nats)
    scores = [0.0] * len(items)
    for pos, i in enumerate(order):
        scores[i] = shuffled[pos]
    result = aurocs(items, scores)

    print("--- injected-fault objective")
    for name, v in result.items():
        if name != "mean":
            print(f"inject_{name}: {v:.6f}")
    print(f"inject_auroc:     {result['mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
