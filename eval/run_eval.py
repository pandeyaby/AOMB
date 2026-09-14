"""
Single-seed public accuracy eval runner.

Default path: lab_capture sessions + length baseline (no torch) for scaffolding.
Optional: precomputed scores, or --train-seconds model path (requires prepared env).

Does not modify prepare.evaluate_bpb. Does not invent claim numbers.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.labels import filter_scorable, load_lab_sessions
from eval.metrics import summarize_ranking
from eval.report import build_report, write_report
from eval.score import resolve_scores


def _hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }
    try:
        import torch

        if torch.backends.mps.is_available():
            info["device"] = "mps"
        elif torch.cuda.is_available():
            info["device"] = "cuda"
        else:
            info["device"] = "cpu"
    except Exception:
        info["device"] = "unknown"
    return info


def _train_then_score(sessions, seed: int, train_seconds: float):
    """Optional short train then session BPB — mirrors demo_anomaly loading pattern."""
    # Lazy imports keep CI metrics path torch-free
    from eval.score import session_bpb_texts

    # Reuse demo_anomaly loaders without running its main()
    import demo_anomaly as demo

    prepare = demo._load_prepare()
    train_ns = demo._load_train_symbols(prepare)
    tokenizer = prepare.Tokenizer.from_directory()
    token_bytes = prepare.get_token_bytes(device=demo.DEVICE)
    MAX_SEQ_LEN = prepare.MAX_SEQ_LEN

    GPT = train_ns["GPT"]
    GPTConfig = train_ns["GPTConfig"]
    DEPTH = train_ns["DEPTH"]
    ASPECT_RATIO = train_ns["ASPECT_RATIO"]
    HEAD_DIM = train_ns["HEAD_DIM"]
    WINDOW_PATTERN = train_ns["WINDOW_PATTERN"]

    import torch

    torch.manual_seed(seed)
    vocab_size = tokenizer.get_vocab_size()
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
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=demo.DEVICE)
    model.init_weights()
    if demo.DEVICE.type == "cpu":
        model.float()

    # Minimal train loop (time-budget) — same spirit as demo_anomaly, not overnight
    import time

    make_dataloader = prepare.make_dataloader
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

    device_batch_size = 16 if demo.DEVICE.type == "mps" else 2
    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    total_batch_size = TOTAL_BATCH_SIZE
    if total_batch_size % tokens_per_fwdbwd != 0:
        total_batch_size = max(
            tokens_per_fwdbwd,
            (TOTAL_BATCH_SIZE // tokens_per_fwdbwd) * tokens_per_fwdbwd,
        )
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    if demo.DEVICE.type == "cpu":
        optimizer.adamw_step_fused = train_ns["adamw_step_fused"]
        optimizer.muon_step_fused = train_ns["muon_step_fused"]

    train_loader = make_dataloader(tokenizer, device_batch_size, MAX_SEQ_LEN, "train")
    x, y, _epoch = next(train_loader)
    autocast_ctx = demo._autocast_ctx(demo.DEVICE.type)

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

    t0 = time.time()
    step = 0
    model.train()
    while True:
        for micro in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y) / grad_accum_steps
            loss.backward()
            x, y, _epoch = next(train_loader)
        progress = min((time.time() - t0) / max(train_seconds, 1e-6), 1.0)
        for g in optimizer.param_groups:
            g["lr"] = g.get("initial_lr", g["lr"]) * get_lr_multiplier(progress)
            if g.get("group_name") == "matrix":
                g["momentum"] = get_muon_momentum(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if time.time() - t0 >= train_seconds:
            break

    demo._sync(demo.DEVICE.type)
    scores = session_bpb_texts(
        model, tokenizer, token_bytes, [s.text for s in sessions], MAX_SEQ_LEN
    )
    train_meta = {
        "mode": "train_then_score",
        "train_seconds": train_seconds,
        "num_steps": step,
        "device": demo.DEVICE.type,
        "note": "Session BPB via model forward; prepare.evaluate_bpb untouched.",
    }
    return scores, train_meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AOMB public accuracy eval (single seed) — protocol scaffolding"
    )
    p.add_argument(
        "--capture",
        type=str,
        default=str(ROOT / "corpus" / "fixtures" / "lab_sample"),
        help="lab_capture directory with provenance.json",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scores-from",
        type=str,
        default="length",
        choices=["length", "events", "precomputed", "model"],
        help="Score source (default: length baseline — no model)",
    )
    p.add_argument(
        "--precomputed-scores",
        type=str,
        default=None,
        help="JSON path when --scores-from precomputed",
    )
    p.add_argument(
        "--train-seconds",
        type=float,
        default=0.0,
        help="If >0 with --scores-from model, short train-then-score budget",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--random-draws",
        type=int,
        default=64,
        help="Random ranking baseline draws",
    )
    args = p.parse_args(argv)

    sessions, corpus_meta = load_lab_sessions(args.capture)
    y_true, kept = filter_scorable(sessions)
    if len(kept) < 2 or len(set(y_true)) < 2:
        print(
            "ERROR: need both normal and incident/cascade labeled sessions for ranking. "
            f"Got label_counts={corpus_meta.get('label_counts')} scorable={len(kept)}. "
            "Sessions are labeled by matching event timestamps "
            "(OTel startTimeUnixNano / timeUnixNano) to provenance.json window "
            "start/end. If everything is 'unknown', check timestamp parsing / "
            "window alignment (see eval/README.md).",
            file=sys.stderr,
        )
        if corpus_meta.get("n_events_missing_timestamp"):
            print(
                f"  hint: {corpus_meta['n_events_missing_timestamp']} span/log "
                "events had no parseable timestamp.",
                file=sys.stderr,
            )
        return 2

    train_meta: dict = {}
    if args.scores_from == "model":
        if args.train_seconds <= 0:
            print(
                "ERROR: --scores-from model requires --train-seconds > 0 "
                "(checkpoint loading not yet wired; train.py does not save checkpoints).",
                file=sys.stderr,
            )
            return 2
        scores, train_meta = _train_then_score(kept, args.seed, args.train_seconds)
        method = "session_bpb_train_then_score"
    else:
        scores, method = resolve_scores(
            kept,
            scores_from=args.scores_from,
            precomputed_path=args.precomputed_scores,
        )

    # Drop NaN scores from ranking
    paired = [(y, s) for y, s in zip(y_true, scores) if s == s]
    if len(paired) < 2:
        print("ERROR: insufficient finite scores", file=sys.stderr)
        return 2
    y_f = [y for y, _ in paired]
    s_f = [s for _, s in paired]
    if len(set(y_f)) < 2:
        print("ERROR: only one class left after dropping NaN scores", file=sys.stderr)
        return 2

    metrics = summarize_ranking(
        y_f, s_f, random_draws=args.random_draws, random_seed=args.seed
    )
    report = build_report(
        metrics=metrics,
        corpus=corpus_meta,
        seed=args.seed,
        score_method=method,
        hardware=_hardware(),
        train_meta=train_meta,
        repo_root=ROOT,
        claim_status="not_published",
        notes=(
            "Fixture/length baselines are for harness verification only. "
            "They are not a public accuracy claim."
            if method.endswith("baseline")
            else ""
        ),
    )
    # Attach per-session rows without full text (keep reports small)
    report["sessions"] = [
        {
            "session_id": s.session_id,
            "label": s.label,
            "binary": s.binary,
            "fault": s.fault,
            "n_chars": s.n_chars,
            "score": float(sc) if sc == sc else None,
        }
        for s, sc in zip(kept, scores)
    ]

    json_path, md_path = write_report(report, args.out_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"seed={args.seed} method={method} "
        f"AUROC={metrics['auroc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
        f"(claim_status=not_published)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
