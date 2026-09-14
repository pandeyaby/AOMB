"""
Shippable session scorer CLI.

Print per-session surprise / bits-per-byte (BPB) for input sessions or dumps.

Does **not** call or modify ``prepare.evaluate_bpb`` (sacred shard-level metric).
Reuses ``eval.score.session_bpb_texts`` and ``prepare.Tokenizer``.

Scoring output is **not** a public accuracy claim until the labeled checklist in
docs/public-accuracy-eval.md passes.

Usage::

    uv run python -m eval.score_cli --input corpus/fixtures/lab_sample --dry-run
    uv run python -m score_session --input path/to/dump --train-seconds 30
    uv run python -m eval.score_cli --input shards/ --checkpoint model.pt

Modes:
  --dry-run          Load sessions and print ids/lengths (no torch / no train)
  --train-seconds N  Short smoke train then score (requires prepared tokenizer)
  --checkpoint PATH  Load a previously saved scorer checkpoint then score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DISCLAIMER = (
    "NOTE: Session BPB scores are diagnostic surprise, not a public accuracy "
    "claim. See docs/public-accuracy-eval.md (claim_status=not_published until "
    "labeled checklist passes)."
)


def load_session_texts(
    input_path: str,
    *,
    adapter: str = "auto",
    max_sessions: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load session texts from BYO / lab / crisp / parquet paths.

    Returns (rows, meta) where each row has session_id, text, label, n_chars.
    """
    path = Path(input_path)
    meta: dict[str, Any] = {"input_path": str(path.resolve())}

    # Explicit parquet file or dir of shards
    if path.is_file() and path.suffix.lower() == ".parquet":
        return _from_parquet([path], meta, max_sessions)
    if path.is_dir() and (
        list(path.glob("shard_*.parquet")) or list(path.glob("*.parquet"))
    ):
        files = sorted(path.glob("shard_*.parquet")) or sorted(path.glob("*.parquet"))
        return _from_parquet(files, meta, max_sessions)

    # Prefer labeled lab loader when provenance + lab layout present
    if (path / "provenance.json").is_file() and (
        (path / "traces.jsonl").is_file() or (path / "spans.jsonl").is_file()
    ):
        if adapter in {"auto", "lab_capture"}:
            from eval.labels import load_lab_sessions

            sessions, corpus_meta = load_lab_sessions(path)
            rows = [
                {
                    "session_id": s.session_id,
                    "text": s.text,
                    "label": s.label,
                    "n_chars": s.n_chars,
                    "n_events": s.n_events,
                }
                for s in sessions
            ]
            if max_sessions:
                rows = rows[:max_sessions]
            meta.update(corpus_meta)
            meta["loader"] = "lab_capture"
            return rows, meta

    # BYO / crisp-style via byo adapter
    from corpus.ingest.adapters.byo import (
        FORMAT_JAEGER_JSON,
        FORMAT_OTLP_JSONL,
        FORMAT_PARQUET_SESSIONS,
        ByoAdapter,
        detect_format,
    )

    if adapter in {"auto", "byo"}:
        fmt = detect_format(str(path))
    elif adapter in {
        FORMAT_OTLP_JSONL,
        FORMAT_JAEGER_JSON,
        FORMAT_PARQUET_SESSIONS,
        "otlp",
        "jaeger",
        "parquet",
    }:
        fmt = {
            "otlp": FORMAT_OTLP_JSONL,
            "jaeger": FORMAT_JAEGER_JSON,
            "parquet": FORMAT_PARQUET_SESSIONS,
        }.get(adapter, adapter)
    else:
        fmt = detect_format(str(path))

    byo = ByoAdapter()
    rows = []
    bundles_meta = []
    for i, (text, window, bundle) in enumerate(
        byo.iter_sessions(str(path), format=fmt)
    ):
        rows.append(
            {
                "session_id": f"{bundle.capture_id}:{i}",
                "text": text,
                "label": window.label,
                "n_chars": len(text),
                "n_events": max(0, text.count("\n")),
            }
        )
        if not bundles_meta:
            bundles_meta.append(
                {
                    "source_id": bundle.source_id,
                    "source_kind": bundle.source_kind,
                    "license": bundle.license,
                    "format": (bundle.extra_provenance or {}).get("format"),
                }
            )
        if max_sessions and len(rows) >= max_sessions:
            break
    meta["loader"] = "byo"
    meta["format"] = fmt
    meta["sources"] = bundles_meta
    meta["session_count"] = len(rows)
    return rows, meta


def _from_parquet(
    files: list[Path], meta: dict[str, Any], max_sessions: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for fp in files:
        table = pq.read_table(fp)
        if "text" not in table.column_names:
            raise ValueError(f"{fp}: expected column 'text'")
        for j, text in enumerate(table.column("text").to_pylist()):
            if text is None or not str(text).strip():
                continue
            s = str(text)
            rows.append(
                {
                    "session_id": f"{fp.stem}:{j}",
                    "text": s,
                    "label": "unknown",
                    "n_chars": len(s),
                    "n_events": max(0, s.count("\n")),
                }
            )
            if max_sessions and len(rows) >= max_sessions:
                meta["loader"] = "parquet"
                meta["parquet_files"] = [str(f) for f in files]
                meta["session_count"] = len(rows)
                return rows, meta
    meta["loader"] = "parquet"
    meta["parquet_files"] = [str(f) for f in files]
    meta["session_count"] = len(rows)
    return rows, meta


def _build_model(prepare, train_ns, device):
    import torch

    GPT = train_ns["GPT"]
    GPTConfig = train_ns["GPTConfig"]
    DEPTH = train_ns["DEPTH"]
    ASPECT_RATIO = train_ns["ASPECT_RATIO"]
    HEAD_DIM = train_ns["HEAD_DIM"]
    WINDOW_PATTERN = train_ns["WINDOW_PATTERN"]
    MAX_SEQ_LEN = prepare.MAX_SEQ_LEN

    tokenizer = prepare.Tokenizer.from_directory()
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
    model.to_empty(device=device)
    model.init_weights()
    if device.type == "cpu":
        model.float()
    return model, tokenizer, config


def _short_train(model, prepare, train_ns, device, demo, train_seconds: float, seed: int):
    import time
    import torch

    torch.manual_seed(seed)
    MAX_SEQ_LEN = prepare.MAX_SEQ_LEN
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

    tokenizer = prepare.Tokenizer.from_directory()
    device_batch_size = 16 if device.type == "mps" else 2
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
    if device.type == "cpu":
        optimizer.adamw_step_fused = train_ns["adamw_step_fused"]
        optimizer.muon_step_fused = train_ns["muon_step_fused"]

    train_loader = make_dataloader(tokenizer, device_batch_size, MAX_SEQ_LEN, "train")
    x, y, _epoch = next(train_loader)
    autocast_ctx = demo._autocast_ctx(device.type)

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
        for _micro in range(grad_accum_steps):
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
    demo._sync(device.type)
    return {"train_seconds": train_seconds, "num_steps": step, "seed": seed}


def save_checkpoint(path: str | Path, model, meta: dict[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "meta": meta,
        "format": "aomb_session_scorer_v1",
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        return payload.get("meta") or {}
    # Raw state_dict
    model.load_state_dict(payload)
    return {}


def score_with_model(
    rows: list[dict[str, Any]],
    *,
    train_seconds: float = 0.0,
    checkpoint: Optional[str] = None,
    save_ckpt: Optional[str] = None,
    seed: int = 0,
) -> tuple[list[float], dict[str, Any]]:
    """Train-short and/or load checkpoint, then session_bpb_texts."""
    import demo_anomaly as demo
    from eval.score import session_bpb_texts

    prepare = demo._load_prepare()
    train_ns = demo._load_train_symbols(prepare)
    device = demo.DEVICE
    model, tokenizer, _config = _build_model(prepare, train_ns, device)
    token_bytes = prepare.get_token_bytes(device=device)
    train_meta: dict[str, Any] = {
        "device": device.type,
        "note": "Session BPB via model forward; prepare.evaluate_bpb untouched.",
    }

    if checkpoint:
        ckpt_meta = load_checkpoint(checkpoint, model)
        train_meta["checkpoint"] = str(checkpoint)
        train_meta["checkpoint_meta"] = ckpt_meta
        train_meta["mode"] = "checkpoint"
    elif train_seconds > 0:
        train_meta.update(
            _short_train(
                model, prepare, train_ns, device, demo, train_seconds, seed
            )
        )
        train_meta["mode"] = "train_then_score"
    else:
        raise ValueError(
            "Model scoring requires --checkpoint or --train-seconds > 0 "
            "(use --dry-run to load sessions without a model)."
        )

    if save_ckpt:
        save_checkpoint(
            save_ckpt,
            model,
            {
                "seed": seed,
                "train_seconds": train_seconds,
                "mode": train_meta.get("mode"),
            },
        )
        train_meta["saved_checkpoint"] = str(save_ckpt)

    scores = session_bpb_texts(
        model,
        tokenizer,
        token_bytes,
        [r["text"] for r in rows],
        prepare.MAX_SEQ_LEN,
    )
    return scores, train_meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "AOMB session scorer — per-session surprise/BPB "
            "(not a public accuracy claim)"
        )
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Session dump: lab_capture / BYO OTLP|Jaeger|parquet path",
    )
    p.add_argument(
        "--adapter",
        default="auto",
        help="auto | byo | lab_capture (default: auto-detect)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load sessions and print metadata only (no torch train/score)",
    )
    p.add_argument(
        "--train-seconds",
        type=float,
        default=0.0,
        help="Short train budget before scoring (requires prepared env)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load scorer checkpoint (.pt) instead of / after init",
    )
    p.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Optional path to save model after train-then-score",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Cap sessions scored (0=all)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON report path",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON lines to stdout",
    )
    args = p.parse_args(argv)

    rows, meta = load_session_texts(
        args.input, adapter=args.adapter, max_sessions=args.max_sessions
    )
    if not rows:
        print("ERROR: no sessions loaded from", args.input, file=sys.stderr)
        return 2

    print(DISCLAIMER, file=sys.stderr)
    print(
        f"Loaded {len(rows)} session(s) via {meta.get('loader')} "
        f"from {meta.get('input_path')}",
        file=sys.stderr,
    )

    if args.dry_run:
        scores = [None] * len(rows)
        train_meta = {"mode": "dry_run"}
        for r in rows:
            line = {
                "session_id": r["session_id"],
                "label": r["label"],
                "n_chars": r["n_chars"],
                "n_events": r["n_events"],
                "bpb": None,
                "dry_run": True,
            }
            if args.json:
                print(json.dumps(line))
            else:
                print(
                    f"{r['session_id']}\tlabel={r['label']}\t"
                    f"n_chars={r['n_chars']}\tn_events={r['n_events']}\t"
                    f"bpb=dry_run"
                )
    else:
        try:
            scores, train_meta = score_with_model(
                rows,
                train_seconds=args.train_seconds,
                checkpoint=args.checkpoint,
                save_ckpt=args.save_checkpoint,
                seed=args.seed,
            )
        except Exception as e:
            print(f"ERROR: scoring failed: {e}", file=sys.stderr)
            return 2
        for r, bpb in zip(rows, scores):
            bpb_s = None if bpb != bpb else float(bpb)
            line = {
                "session_id": r["session_id"],
                "label": r["label"],
                "n_chars": r["n_chars"],
                "n_events": r["n_events"],
                "bpb": bpb_s,
            }
            if args.json:
                print(json.dumps(line))
            else:
                bpb_disp = "nan" if bpb_s is None else f"{bpb_s:.6f}"
                print(
                    f"{r['session_id']}\tlabel={r['label']}\t"
                    f"n_chars={r['n_chars']}\tbpb={bpb_disp}"
                )

    report = {
        "claim_status": "not_published",
        "disclaimer": DISCLAIMER,
        "meta": meta,
        "train_meta": train_meta,
        "sessions": [
            {
                "session_id": r["session_id"],
                "label": r["label"],
                "n_chars": r["n_chars"],
                "n_events": r["n_events"],
                "bpb": (
                    None
                    if (s is None or (isinstance(s, float) and s != s))
                    else float(s)
                ),
            }
            for r, s in zip(rows, scores)
        ],
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
