"""
Fixture-only train-then-score for the public ranking card.

Trains on frozen train-split session texts from the public fixture only —
no CRISP download, no prepare.make_dataloader shards, no private corpus.

prepare.evaluate_bpb is never called or modified.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence

from eval.labels import LabeledSession

# Fixture tokenizer: small vocab fit to ~10 short synthetic sessions.
FIXTURE_VOCAB_SIZE = 512
FIXTURE_SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
FIXTURE_BOS_TOKEN = "<|reserved_0|>"
FIXTURE_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# Short context: fixture sessions are << 2k tokens after BPE.
FIXTURE_MAX_SEQ_LEN = 256


def load_split(capture_dir: str | Path) -> dict[str, Any]:
    path = Path(capture_dir) / "split.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Public ranking card fixture-only train requires "
            "a frozen split.json (train/eval session ids)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def partition_by_split(
    sessions: Sequence[LabeledSession],
    split: dict[str, Any],
) -> tuple[list[LabeledSession], list[LabeledSession]]:
    """Partition scorable sessions into (train, eval) by frozen session ids."""
    train_ids = set(split["train_session_ids"])
    eval_ids = set(split["eval_session_ids"])
    if train_ids & eval_ids:
        raise ValueError("split.json train/eval session ids overlap")
    train = [s for s in sessions if s.session_id in train_ids]
    eval_s = [s for s in sessions if s.session_id in eval_ids]
    missing_train = train_ids - {s.session_id for s in train}
    missing_eval = eval_ids - {s.session_id for s in eval_s}
    if missing_train or missing_eval:
        raise ValueError(
            f"split.json ids not found in loaded sessions: "
            f"train_missing={sorted(missing_train)} eval_missing={sorted(missing_eval)}"
        )
    return train, eval_s


class FixtureTokenizer:
    """Minimal tiktoken wrapper matching prepare.Tokenizer interface."""

    def __init__(self, enc, bos_token: str = FIXTURE_BOS_TOKEN):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(bos_token)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
            )
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            return ids
        if isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            return ids
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.enc.decode(ids)


def train_fixture_tokenizer(texts: Sequence[str]) -> tuple[FixtureTokenizer, Any]:
    """
    Train a BPE tokenizer on fixture train-split texts only.

    Returns (tokenizer, token_bytes tensor on CPU).
    """
    import rustbpe
    import tiktoken
    import torch

    if not texts:
        raise ValueError("need train-split texts to build fixture tokenizer")

    rbpe = rustbpe.Tokenizer()
    vocab_no_special = FIXTURE_VOCAB_SIZE - len(FIXTURE_SPECIAL_TOKENS)
    rbpe.train_from_iterator(list(texts), vocab_no_special, pattern=FIXTURE_SPLIT_PATTERN)

    pattern = rbpe.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in rbpe.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(FIXTURE_SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="aomb_public_ranking_card_v1",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    tokenizer = FixtureTokenizer(enc)

    special_set = set(FIXTURE_SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32)
    return tokenizer, token_bytes


def make_text_dataloader(
    tokenizer: FixtureTokenizer,
    texts: Sequence[str],
    B: int,
    T: int,
    device: str,
    buffer_size: int = 64,
):
    """
    Infinite BOS-aligned packed dataloader over an in-memory text list.

    Same packing spirit as prepare.make_dataloader, but documents come only
    from the provided texts (fixture train split) — never from CRISP/cache.
    """
    import torch

    assert texts, "empty train text list"
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()
    # Pre-tokenize once; cycle forever
    docs = tokenizer.encode(list(texts), prepend=bos_token)
    doc_buffer: list[list[int]] = []
    epoch = 1
    doc_i = 0

    def refill_buffer():
        nonlocal epoch, doc_i
        while len(doc_buffer) < buffer_size:
            doc_buffer.append(list(docs[doc_i]))
            doc_i += 1
            if doc_i >= len(docs):
                doc_i = 0
                epoch += 1

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < max(4, min(buffer_size, len(docs) * 2)):
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer)
        yield inputs, targets, epoch


def train_fixture_lm(
    train_texts: Sequence[str],
    *,
    seed: int,
    train_seconds: float,
) -> tuple[Any, FixtureTokenizer, Any, dict[str, Any]]:
    """
    Fit a BPE tokenizer and a short GPT on ``train_texts`` only.

    Returns (model, tokenizer, token_bytes on device, info). Shared by the
    public ranking card and the in-domain lab eval (eval.in_domain).
    """
    import time

    import torch

    import demo_anomaly as demo

    prepare = demo._load_prepare()
    train_ns = demo._load_train_symbols(prepare)

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

    tokenizer, token_bytes_cpu = train_fixture_tokenizer(train_texts)
    token_bytes = token_bytes_cpu.to(demo.DEVICE)

    torch.manual_seed(seed)
    vocab_size = tokenizer.get_vocab_size()
    max_seq_len = FIXTURE_MAX_SEQ_LEN
    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    config = GPTConfig(
        sequence_len=max_seq_len,
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

    device_batch_size = 8 if demo.DEVICE.type == "mps" else 2
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    total_batch_size = TOTAL_BATCH_SIZE
    # Shrink total batch for tiny fixture + CPU so we take more optimizer steps
    # inside the time budget without huge grad accum.
    if demo.DEVICE.type == "cpu":
        total_batch_size = max(tokens_per_fwdbwd, tokens_per_fwdbwd * 4)
    elif total_batch_size % tokens_per_fwdbwd != 0:
        total_batch_size = max(
            tokens_per_fwdbwd,
            (TOTAL_BATCH_SIZE // tokens_per_fwdbwd) * tokens_per_fwdbwd,
        )
    grad_accum_steps = max(1, total_batch_size // tokens_per_fwdbwd)

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

    train_loader = make_text_dataloader(
        tokenizer,
        train_texts,
        device_batch_size,
        max_seq_len,
        device=str(demo.DEVICE),
    )
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

    demo._sync(demo.DEVICE.type)
    model.eval()
    info = {
        "num_steps": step,
        "device": demo.DEVICE.type,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "grad_accum_steps": grad_accum_steps,
        "device_batch_size": device_batch_size,
    }
    return model, tokenizer, token_bytes, info


def train_then_score_fixture(
    train_sessions: Sequence[LabeledSession],
    eval_sessions: Sequence[LabeledSession],
    *,
    seed: int,
    train_seconds: float,
) -> tuple[list[float], dict[str, Any]]:
    """
    Short GPT train on fixture train-split texts, then session BPB on eval sessions.

    Uses train.py GPT + hyperparams via demo_anomaly loaders, but replaces the
    prepare.make_dataloader("train") path with an in-memory fixture dataloader.
    """
    from eval.score import session_bpb_texts

    train_texts = [s.text for s in train_sessions if int(s.binary or 0) == 0]
    if not train_texts:
        # Fallback: all train-split texts (should not happen on the public card)
        train_texts = [s.text for s in train_sessions]
    n_train_normal = sum(1 for s in train_sessions if int(s.binary or 0) == 0)
    n_train_pos_skipped = sum(1 for s in train_sessions if int(s.binary or 0) == 1)
    model, tokenizer, token_bytes, info = train_fixture_lm(
        train_texts, seed=seed, train_seconds=train_seconds
    )
    scores = session_bpb_texts(
        model,
        tokenizer,
        token_bytes,
        [s.text for s in eval_sessions],
        info["max_seq_len"],
    )
    train_meta = {
        "mode": "fixture_train_then_score",
        "train_corpus": "fixture_train_split_normals_only",
        "train_seconds": train_seconds,
        "num_steps": info["num_steps"],
        "device": info["device"],
        "n_train_sessions_listed": len(train_sessions),
        "n_train_normal_texts": n_train_normal,
        "n_train_positive_excluded_from_lm": n_train_pos_skipped,
        "n_eval_sessions": len(eval_sessions),
        "vocab_size": info["vocab_size"],
        "max_seq_len": info["max_seq_len"],
        "grad_accum_steps": info["grad_accum_steps"],
        "device_batch_size": info["device_batch_size"],
        "note": (
            "LM trained on public fixture train-split NORMAL session texts only "
            "(positives in the train split are excluded from the LM objective so "
            "held-out incident/cascade remain surprising). No CRISP / prepare data "
            "download. prepare.evaluate_bpb untouched. Scores are session BPB on "
            "held-out eval-split sessions."
        ),
    }
    return scores, train_meta


def persist_tokenizer_sidecar(
    tokenizer: FixtureTokenizer,
    token_bytes,
    out_dir: str | Path,
) -> Path:
    """Optional helper: write tokenizer artifacts under a temp/report dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer.enc, f)
    import torch

    torch.save(token_bytes.cpu(), out / "token_bytes.pt")
    return out
