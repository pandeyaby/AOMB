"""
Audit a historic commit's val_bpb: reported (loss-path) vs true bits-per-byte.

From 2026-03-10 (exp 17, c2026cc) until prepare.evaluate_bpb was fixed, val_bpb
was computed from ``model(x, y, reduction='none')``. train.py owns that code
path and returned a focal-weighted loss from it, so reported val_bpb was not
bits-per-byte. This runs one commit's ``train.py`` unmodified on the current
prepare.py cache and measures, on the SAME trained model:

- ``true_bpb``    cross-entropy from logits (prepare.evaluate_bpb, fixed)
- ``legacy_bpb``  the old formula, through the model's own loss path

Usage (cache must already hold the corpus the commit was measured on):
    uv run python -m eval.val_bpb_audit --sha <commit> --label "<what>" --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def legacy_bpb(model, tokenizer, batch_size, autocast_ctx=None) -> float:
    """The pre-fix evaluate_bpb: per-token loss taken from the model's loss path."""
    import torch

    import prepare

    device = next(model.parameters()).device
    token_bytes = prepare.get_token_bytes(device=device)
    loader = prepare.make_dataloader(tokenizer, batch_size, prepare.MAX_SEQ_LEN, "val")
    steps = prepare.EVAL_TOKENS // (batch_size * prepare.MAX_SEQ_LEN)
    total_nats, total_bytes = 0.0, 0
    with torch.no_grad():
        for _ in range(steps):
            x, y, _ = next(loader)
            if autocast_ctx is not None:
                with autocast_ctx:
                    loss_flat = model(x, y, reduction="none").view(-1)
            else:
                loss_flat = model(x, y, reduction="none").view(-1)
            nbytes = token_bytes[y.view(-1)]
            total_nats += (loss_flat * (nbytes > 0)).sum().item()
            total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


def run(sha: str, label: str) -> dict:
    source = subprocess.run(
        ["git", "show", f"{sha}:train.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    module = types.ModuleType("_aomb_audit_commit")
    module.__file__ = str(ROOT / "train.py")
    sys.modules[module.__name__] = module
    t0 = time.time()
    exec(compile(source, f"train.py@{sha[:7]}", "exec"), module.__dict__)
    ns = module.__dict__
    model = ns["model"]
    model.eval()
    legacy = legacy_bpb(model, ns["tokenizer"], ns["DEVICE_BATCH_SIZE"], ns.get("autocast_ctx"))
    true = float(ns["val_bpb"])  # train.py's own final eval, now via the fixed prepare.evaluate_bpb
    return {
        "sha": sha,
        "label": label,
        "true_bpb": true,
        "legacy_bpb": legacy,
        "legacy_over_true": legacy / true,
        "num_steps": int(ns.get("step", -1)),
        "depth": ns.get("DEPTH"),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sha", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = run(a.sha, a.label)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
