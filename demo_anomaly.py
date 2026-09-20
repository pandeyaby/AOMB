"""
demo_anomaly.py
━━━━━━━━━━━━━━━
Prove the AOMB thesis in one command: next-token surprise *is* anomaly detection.

After corpus + prepare (+ optional 60s smoke), this script:
  1. Runs a short smoke-style train on the existing observability shards
  2. Scores held-out sessions labeled normal / anomalous / cascade
  3. Prints per-class bits-per-byte (surprise) so the gap is obvious

Uses the same generator as generate_observability_corpus.py — labels stay in
memory for the held-out set (parquet only stores text; no parallel eval stack).

Device: Apple Silicon MPS preferred; honest CPU fallback. No API keys. No spend.

Honesty (aligned with session scorer):
  Session / class BPB is diagnostic surprise — same family as train-lane val_bpb.
  Never invents AUROC / published ranking. Lab claim_status stays not_published.
  Refuses --auroc / --publish / ranking flags (exit 1).
  prepare.py is sacred. CUDA gate stays skipped.

Usage:
    uv run python demo_anomaly.py
    uv run python demo_anomaly.py --seconds 45 --per-class 16
    uv run python demo_anomaly.py --dry-run --per-class 2   # CI / no torch
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import re
import sys
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent

# Exit codes — refuse invent AUROC (1); missing corpus/paths (2); gap not yet
# obvious after short smoke (3). Shared spirit with eval.score_cli.
EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_PATH_ERROR = 2
EXIT_GAP_NOT_OBVIOUS = 3

# Loud refusals — anomaly story prints BPB/surprise only (never AUROC).
# Keep aligned with eval.score_cli.REFUSED_METRIC_FLAGS.
REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--invent-metrics",
        "--invent-auroc",
        "--claim-auroc",
    }
)

HONESTY_LINES = (
    "HONESTY: same family as train-lane val_bpb (surprise / bits-per-byte).",
    "         Not AUROC. Lab stays not_published. Fixture card = harness smoke.",
    "         prepare.py sacred. CUDA gate stays skipped.",
)

DISCLAIMER = (
    "NOTE: Class / session BPB scores are diagnostic surprise, not a public "
    "accuracy claim (claim_status=not_published until labeled checklist passes)."
)


def _refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on ranking / AUROC / publish flags before argparse.

    Raises SystemExit with code EXIT_REFUSED_FLAG.
    Applies to --dry-run and full smoke-train paths alike.
    """
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            print(
                f"ERROR: Refusing '{key}'.\n"
                "  Anomaly story prints class / session BPB / surprise only.\n"
                "  Never invents AUROC / published ranking accuracy.\n"
                "  Lab claim_status stays not_published.\n"
                "  Use --dry-run (metadata) or smoke train (default) for BPB.",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED_FLAG)


def pick_device():
    """MPS preferred; honest CPU fallback. No fake CUDA path."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_device():
    """Lazy device init so --dry-run / flag refusals stay torch-free."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return pick_device()


def __getattr__(name: str):
    # External callers (eval.score_cli / fixture_train) use demo.DEVICE.
    if name == "DEVICE":
        return ensure_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_prepare():
    """Import prepare.py without the macOS-only gate (demo selects device itself)."""
    if "prepare" in sys.modules:
        return sys.modules["prepare"]
    source = (REPO_ROOT / "prepare.py").read_text()
    source = re.sub(
        r"\nverify_macos_env\(\)\n",
        "\n# verify_macos_env() skipped by demo_anomaly — MPS preferred, CPU ok\n",
        source,
        count=1,
    )
    module = type(sys)("prepare")
    module.__file__ = str(REPO_ROOT / "prepare.py")
    sys.modules["prepare"] = module
    exec(compile(source, str(REPO_ROOT / "prepare.py"), "exec"), module.__dict__)
    return module


def _load_train_symbols(prepare_mod):
    """
    Load GPT / hyperparams from train.py without running its module-level
    5-minute training loop (train.py is a script, not a library).
    """
    import torch

    source = (REPO_ROOT / "train.py").read_text()
    source = re.sub(
        r"\nverify_macos_env\(\)\n",
        "\n# verify_macos_env() skipped by demo_anomaly\n",
        source,
        count=1,
    )
    marker = "# ---------------------------------------------------------------------------\n# Setup: tokenizer, model, optimizer, dataloader"
    idx = source.find(marker)
    if idx < 0:
        raise RuntimeError(
            "demo_anomaly: could not find train.py Setup marker — train.py layout changed?"
        )
    source = source[:idx]
    # Register a real module so @dataclass (GPTConfig) can resolve __module__
    mod_name = "_aomb_train_demo"
    module = types.ModuleType(mod_name)
    module.__file__ = str(REPO_ROOT / "train.py")
    module.__dict__.update(
        {
            "os": os,
            "sys": sys,
            "gc": gc,
            "time": time,
            "math": math,
            "torch": torch,
        }
    )
    sys.modules[mod_name] = module
    # ensure `from prepare import ...` resolves to our shim
    sys.modules["prepare"] = prepare_mod
    exec(compile(source, str(REPO_ROOT / "train.py"), "exec"), module.__dict__)
    # MuonAdamW.__init__ reads module-level device_type (normally set in Setup)
    module.device_type = ensure_device().type
    return module.__dict__


def _autocast_ctx(device_type: str):
    import torch

    if device_type == "cpu":
        # bfloat16 autocast on CPU is optional; keep float32 for widest compatibility
        return contextlib.nullcontext()
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()  # MPS: match train.py


def _sync(device_type: str):
    import torch

    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def _session_bpb_imports():
    """Local torch.nn.functional for session_bpb (avoid module-level torch)."""
    import torch
    import torch.nn.functional as F

    yield torch, F


def session_bpb(model, tokenizer, token_bytes, text: str, max_seq_len: int) -> float:
    """
    Bits-per-byte surprise for one session (chunked to max_seq_len).
    Uses raw next-token CE (not focal train loss) so the number is true surprise.
    """
    with _session_bpb_imports() as (torch, F):
        device = next(model.parameters()).device
        bos = tokenizer.get_bos_token_id()
        ids = tokenizer.encode(text, prepend=bos)
        if len(ids) < 2:
            return float("nan")

        total_nats = 0.0
        total_bytes = 0
        pos = 0
        while pos + 1 < len(ids):
            chunk = ids[pos : pos + max_seq_len + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)  # no targets → logits
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                y.view(-1),
                reduction="none",
            )
            y_flat = y.view(-1)
            nbytes = token_bytes[y_flat]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += int(nbytes.sum().item())
            pos += max_seq_len

        if total_bytes == 0:
            return float("nan")
        return total_nats / (math.log(2) * total_bytes)


def make_heldout_sessions(per_class: int, seed: int = 20260315):
    """Labeled held-out sessions from the same generator (labels in memory only)."""
    from generate_observability_corpus import generate_session

    rng_state = random.getstate()
    random.seed(seed)
    base = datetime(2026, 6, 1, 0, 0, 0)
    out = {"normal": [], "anomalous": [], "cascade": []}
    for _ in range(per_class):
        base += timedelta(seconds=random.uniform(5, 40))
        out["normal"].append(generate_session(base, anomalous=False, cascade=False))
        base += timedelta(seconds=random.uniform(5, 40))
        out["anomalous"].append(generate_session(base, anomalous=True, cascade=False))
        base += timedelta(seconds=random.uniform(5, 40))
        out["cascade"].append(generate_session(base, anomalous=True, cascade=True))
    random.setstate(rng_state)
    return out


def make_tiny_fixture_sessions(per_class: int = 2, seed: int = 7):
    """Tiny synthetic texts for dry-run / unit tests (no generator / no torch)."""
    rng = random.Random(seed)
    out = {"normal": [], "anomalous": [], "cascade": []}
    for i in range(per_class):
        n = rng.randint(10, 40)
        out["normal"].append(
            f"ts=2026-06-01T00:00:{i:02d}Z level=INFO svc=api "
            f"msg=ok latency_ms={n} request_id=n-{i}"
        )
        out["anomalous"].append(
            f"ts=2026-06-01T00:01:{i:02d}Z level=ERROR svc=api "
            f"msg=timeout FAIL CRITICAL request_id=a-{i}"
        )
        out["cascade"].append(
            f"ts=2026-06-01T00:02:{i:02d}Z level=CRITICAL svc=api "
            f"msg=circuit_breaker escalate pagerduty cascade request_id=c-{i}\n"
            f"ts=2026-06-01T00:02:{i:02d}Z level=ERROR svc=db msg=connection_refused"
        )
    return out


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def _print_honesty_banner(*, dry_run: bool = False) -> None:
    print("=" * 72)
    print("  AOMB ANOMALY STORY — next-token surprise IS anomaly detection")
    print("=" * 72)
    for line in HONESTY_LINES:
        print(f"  {line}")
    print(f"  {DISCLAIMER}")
    if dry_run:
        print("  Mode            : dry-run (session metadata only — no torch / no BPB invent)")
    print()


def _sessions_report(
    heldout: dict[str, list[str]],
    scores: Optional[dict[str, list[float]]] = None,
    *,
    mode: str,
) -> dict[str, Any]:
    """Machine-readable summary — never includes AUROC / invented metrics."""
    classes = {}
    for label, texts in heldout.items():
        entry: dict[str, Any] = {
            "n": len(texts),
            "mean_chars": _mean([float(len(t)) for t in texts]),
            "sessions": [
                {
                    "session_id": f"{label}-{i}",
                    "label": label,
                    "n_chars": len(t),
                    "bpb": None
                    if scores is None
                    else (
                        None
                        if scores[label][i] != scores[label][i]
                        else float(scores[label][i])
                    ),
                }
                for i, t in enumerate(texts)
            ],
        }
        if scores is not None:
            entry["mean_bpb"] = _mean(scores[label])
            entry["median_bpb"] = _median(scores[label])
        classes[label] = entry
    return {
        "claim_status": "not_published",
        "disclaimer": DISCLAIMER,
        "mode": mode,
        "classes": classes,
    }


def run_dry_run(
    *,
    per_class: int = 2,
    use_generator: bool = False,
    out_path: Optional[str] = None,
    as_json: bool = False,
) -> int:
    """Torch-free dry-run: held-out session metadata + honesty, no BPB invent."""
    _print_honesty_banner(dry_run=True)
    print(f"  Sessions/class  : {per_class}")
    print(f"  Generator       : {'observability' if use_generator else 'tiny_fixture'}")
    print()

    if use_generator:
        heldout = make_heldout_sessions(per_class)
    else:
        heldout = make_tiny_fixture_sessions(per_class)

    print("  Held-out session metadata (dry-run — bpb not scored)")
    print("  " + "─" * 60)
    print(f"  {'class':<12} {'n':>4}  {'mean_chars':>10}  {'bpb':>10}")
    for label in ("normal", "anomalous", "cascade"):
        texts = heldout[label]
        mean_c = _mean([float(len(t)) for t in texts])
        print(f"  {label:<12} {len(texts):>4}  {mean_c:10.1f}  {'dry_run':>10}")
        if as_json:
            for i, t in enumerate(texts):
                print(
                    json.dumps(
                        {
                            "session_id": f"{label}-{i}",
                            "label": label,
                            "n_chars": len(t),
                            "bpb": None,
                            "dry_run": True,
                        }
                    )
                )
    print("  " + "─" * 60)
    print()
    print("  claim_status=not_published — no AUROC, no invented val_bpb.")
    print("  Next: full smoke (omit --dry-run) after prepare.py for real BPB.")
    print("=" * 72)

    report = _sessions_report(heldout, scores=None, mode="dry_run")
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  Wrote {path}", file=sys.stderr)
    return EXIT_OK


def validate_smoke_paths(prepare_mod) -> Optional[str]:
    """Return an error message if corpus/tokenizer missing, else None."""
    data_dir = prepare_mod.DATA_DIR
    tok_dir = prepare_mod.TOKENIZER_DIR
    val_path = os.path.join(data_dir, prepare_mod.VAL_FILENAME)
    tok_pkl = os.path.join(tok_dir, "tokenizer.pkl")

    if not os.path.isdir(data_dir) or not any(
        f.endswith(".parquet") for f in os.listdir(data_dir)
    ):
        return (
            f"ERROR: Missing corpus at {data_dir}\n"
            "  Run:  uv run python generate_observability_corpus.py\n"
            "  Or use --dry-run for metadata-only (no torch)."
        )
    if not os.path.exists(val_path):
        return (
            f"ERROR: Missing pinned val shard: {val_path}\n"
            "  Run:  uv run python generate_observability_corpus.py"
        )
    if not os.path.exists(tok_pkl):
        return (
            f"ERROR: Missing tokenizer at {tok_dir}\n"
            "  Run:  uv run python prepare.py --num-shards 20"
        )
    return None


def run_smoke_train_and_score(
    *,
    seconds: float = 60.0,
    per_class: int = 20,
    batch_size: Optional[int] = None,
    out_path: Optional[str] = None,
) -> int:
    """Full product path: short train → score held-out classes → print BPB gap."""
    import torch

    device = ensure_device()

    _print_honesty_banner(dry_run=False)
    print(
        f"  Device          : {device.type}"
        + (" (Apple Silicon MPS)" if device.type == "mps" else " (honest CPU fallback)")
    )
    print(f"  Train budget    : {seconds:.0f}s")
    print(f"  Sessions/class  : {per_class}")
    print()

    prepare = _load_prepare()
    path_err = validate_smoke_paths(prepare)
    if path_err:
        print(path_err, file=sys.stderr)
        return EXIT_PATH_ERROR

    data_dir = prepare.DATA_DIR
    tok_dir = prepare.TOKENIZER_DIR

    train_ns = _load_train_symbols(prepare)
    GPT = train_ns["GPT"]
    GPTConfig = train_ns["GPTConfig"]
    DEPTH = train_ns["DEPTH"]
    ASPECT_RATIO = train_ns["ASPECT_RATIO"]
    HEAD_DIM = train_ns["HEAD_DIM"]
    WINDOW_PATTERN = train_ns["WINDOW_PATTERN"]
    TOTAL_BATCH_SIZE = train_ns["TOTAL_BATCH_SIZE"]
    EMBEDDING_LR = train_ns["EMBEDDING_LR"]
    UNEMBEDDING_LR = train_ns["UNEMBEDDING_LR"]
    MATRIX_LR = train_ns["MATRIX_LR"]
    SCALAR_LR = train_ns["SCALAR_LR"]
    WEIGHT_DECAY = train_ns["WEIGHT_DECAY"]
    ADAM_BETAS = train_ns["ADAM_BETAS"]
    WARMUP_RATIO = train_ns["WARMUP_RATIO"]
    WARMDOWN_RATIO = train_ns["WARMDOWN_RATIO"]
    FINAL_LR_FRAC = train_ns["FINAL_LR_FRAC"]

    MAX_SEQ_LEN = prepare.MAX_SEQ_LEN
    Tokenizer = prepare.Tokenizer
    make_dataloader = prepare.make_dataloader
    get_token_bytes = prepare.get_token_bytes

    device_type = device.type
    if batch_size is not None:
        device_batch_size = batch_size
    else:
        device_batch_size = 16 if device_type == "mps" else 2

    # Shrink total batch on CPU so grad_accum stays sane
    total_batch_size = TOTAL_BATCH_SIZE
    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    if total_batch_size % tokens_per_fwdbwd != 0:
        # pick largest multiple of tokens_per_fwdbwd that is <= TOTAL_BATCH_SIZE
        total_batch_size = max(
            tokens_per_fwdbwd,
            (TOTAL_BATCH_SIZE // tokens_per_fwdbwd) * tokens_per_fwdbwd,
        )
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size      : {vocab_size:,}")
    print(f"  Device batch    : {device_batch_size}  (grad_accum={grad_accum_steps})")
    print(f"  Data dir        : {data_dir}")
    print(f"  Tokenizer       : {tok_dir}")
    print()

    # Anomaly token ids (same spirit as train.py — used only if forward accepts mask)
    anomaly_keywords = [
        "error",
        "ERROR",
        "timeout",
        "TIMEOUT",
        "fail",
        "FAIL",
        "CRITICAL",
        "critical",
        "alert",
        "ALERT",
        "VIOLATED",
        "violated",
        "circuit_breaker",
        "escalate",
        "pagerduty",
        "anomal",
        "ANOMAL",
        "drift_score",
        "STALL",
        "stall",
        "degraded",
        "DEGRADED",
        "down",
        "DOWN",
        "outage",
        "OUTAGE",
    ]
    anomaly_token_ids = [
        tid
        for tid in range(vocab_size)
        if any(k in tokenizer.decode([tid]) for k in anomaly_keywords)
    ]
    anomaly_token_ids_tensor = torch.tensor(
        anomaly_token_ids, dtype=torch.long, device=device
    )

    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
    print(f"  Model           : depth={DEPTH} dim={model_dim} window={WINDOW_PATTERN}")
    print()
    print("  Training (smoke)…")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    if device.type == "cpu":
        # train.py keeps embeddings/RoPE in bf16 for MPS; CPU matmul needs float32
        model.float()

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    if device.type == "cpu":
        # MuonAdamW torch.compiles on cpu/cuda; skip inductor here (no C++ toolchain required)
        optimizer.adamw_step_fused = train_ns["adamw_step_fused"]
        optimizer.muon_step_fused = train_ns["muon_step_fused"]

    train_loader = make_dataloader(tokenizer, device_batch_size, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)
    autocast_ctx = _autocast_ctx(device_type)

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        if progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    time_budget = float(seconds)
    t_train0 = time.time()
    total_training_time = 0.0
    step = 0
    smooth_train_loss = 0.0
    last_loss = float("nan")

    model.train()
    while True:
        _sync(device_type)
        t0 = time.time()
        for _micro in range(grad_accum_steps):
            anomaly_mask = torch.isin(y, anomaly_token_ids_tensor)
            with autocast_ctx:
                try:
                    loss = model(x, y, anomaly_mask=anomaly_mask)
                except TypeError:
                    loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / time_budget, 1.0) if time_budget > 0 else 1.0
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_wd = WEIGHT_DECAY * (1 - progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd
        optimizer.step()
        model.zero_grad(set_to_none=True)

        last_loss = train_loss.item()
        if last_loss > 100:
            print("\nFAIL: loss exploded", file=sys.stderr)
            return EXIT_PATH_ERROR

        _sync(device_type)
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * last_loss
        debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
        remaining = max(0.0, time_budget - total_training_time)
        print(
            f"\r  step {step:05d} | loss: {debiased:.4f} | "
            f"tok/s: {int(total_batch_size / dt):,} | remaining: {remaining:.0f}s   ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1
        if step > 10 and total_training_time >= time_budget:
            break
        # Safety: tiny budgets / slow CPU — still produce an eval
        if time.time() - t_train0 > time_budget + 120 and step > 10:
            break

    print()
    print(
        f"  Trained         : {total_training_time:.1f}s steady-state over {step} steps "
        f"(wall {time.time() - t_train0:.1f}s)"
    )
    print(f"  Final train loss: {last_loss:.4f}")
    print()

    # ── held-out surprise by class ───────────────────────────────────────────
    print("  Scoring held-out sessions (normal / anomalous / cascade)…")
    heldout = make_heldout_sessions(per_class)
    token_bytes = get_token_bytes(device=str(device))
    model.eval()

    scores = {k: [] for k in ("normal", "anomalous", "cascade")}
    with autocast_ctx:
        for label, texts in heldout.items():
            for text in texts:
                scores[label].append(
                    session_bpb(model, tokenizer, token_bytes, text, MAX_SEQ_LEN)
                )

    normal_mean = _mean(scores["normal"])
    print()
    print("  Held-out session surprise (bits-per-byte)")
    print("  " + "─" * 60)
    print(f"  {'class':<12} {'n':>4}  {'mean_bpb':>10}  {'median_bpb':>10}  {'vs_normal':>10}")
    for label in ("normal", "anomalous", "cascade"):
        xs = scores[label]
        mean_b = _mean(xs)
        med_b = _median(xs)
        if label == "normal" or not (normal_mean == normal_mean) or normal_mean == 0:
            rel = "—"
        else:
            rel = f"{(mean_b / normal_mean - 1.0) * 100:+.0f}%"
        print(f"  {label:<12} {len(xs):>4}  {mean_b:10.4f}  {med_b:10.4f}  {rel:>10}")

    print("  " + "─" * 60)
    print()
    print("  What to look for:")
    print("    • anomalous / cascade mean_bpb  >  normal  (the gap is the detector)")
    print("    • after longer train, cascade often ≥ anomalous (denser failures)")
    print("  Same objective as val_bpb — no labels at train time, no thresholds.")
    print("  Not claimed: lab AUROC · CUDA · production accuracy · fixture-card heroes.")
    print("  claim_status=not_published")
    print("=" * 72)

    report = _sessions_report(heldout, scores, mode="smoke_train_score")
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  Wrote {path}", file=sys.stderr)

    # Non-zero exit if the story failed directionally (helps CI / newcomers)
    anom_mean = _mean(scores["anomalous"])
    casc_mean = _mean(scores["cascade"])
    if not (anom_mean > normal_mean and casc_mean > normal_mean):
        print(
            "\nNote: gap not yet obvious after this short train — "
            "re-run with --seconds 90 or after a longer overnight experiment.",
            file=sys.stderr,
        )
        return EXIT_GAP_NOT_OBVIOUS
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AOMB anomaly story: show surprise/BPB gap across session classes "
            "(claim_status=not_published; never invents AUROC)"
        ),
        epilog=(
            "Refuses --auroc / --publish / ranking flags. "
            "prepare.py is sacred — not modified. "
            "CUDA gate stays skipped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="Smoke train time budget in seconds (default: 60)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Held-out sessions per class (default: 20; dry-run often uses 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override device batch size (default: 16 on MPS, 2 on CPU)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Session metadata only (no torch / no train / no invented BPB)",
    )
    parser.add_argument(
        "--use-generator",
        action="store_true",
        help="With --dry-run: use observability generator instead of tiny fixture",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON report path (claim_status=not_published)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --dry-run: print per-session JSON lines (bpb=null)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    _refuse_loud_flags(raw)

    args = build_parser().parse_args(raw)

    if args.dry_run:
        return run_dry_run(
            per_class=args.per_class,
            use_generator=args.use_generator,
            out_path=args.out,
            as_json=args.json,
        )

    return run_smoke_train_and_score(
        seconds=args.seconds,
        per_class=args.per_class,
        batch_size=args.batch_size,
        out_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
