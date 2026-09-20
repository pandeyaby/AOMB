"""Measured Tale val_bpb report emitter (factual extract only).

Reads a train / morning-report-style log, extracts factual ``val_bpb`` via
``val_bpb_parse`` (or None), and writes a JSON card under ``reports/tale-capped/``.

Never invents AUROC or val_bpb. ``prepare.py`` is sacred (untouched).
CUDA gate stays skipped.

Exit codes:
  0 — card written (measured or pending)
  1 — refused invent / publish / cuda / auroc flags
  2 — missing or unreadable log (still writes null + pending card when out-dir ok)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from val_bpb_parse import parse_val_bpb  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_PATH_ERROR = 2

DEFAULT_OUT_DIR = ROOT / "reports" / "tale-capped"
DEFAULT_CORPUS = "tale_of_errors"

# Loud invent / publish / cuda refusals (mirror score_cli + tale baseline spirit).
REFUSED_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--cuda",
        "--gpu",
        "--invent-metrics",
        "--invent-auroc",
        "--invent-val-bpb",
        "--claim-auroc",
        "--invent",
    }
)


def refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on invent / publish / cuda flags before argparse."""
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_FLAGS:
            print(
                f"ERROR: Refusing '{key}'.\n"
                "  Tale measured report emits factual val_bpb only "
                "(or null + pending).\n"
                "  Never invents AUROC / published ranking / CUDA path.\n"
                "  claim_status stays pending or measured_not_published.\n"
                "  Use --log PATH [--out-dir DIR] [--max-spans N] "
                "[--source-id ID].",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED_FLAG)


def claim_status_for(val_bpb: Optional[float]) -> str:
    """pending when null; measured_not_published when a factual value exists."""
    if val_bpb is None:
        return "pending"
    return "measured_not_published"


def build_card(
    *,
    val_bpb: Optional[float],
    max_spans: Optional[int],
    source_id: str,
    git_sha: Optional[str],
    log_path: str,
    corpus: str = DEFAULT_CORPUS,
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "val_bpb": val_bpb,
        "claim_status": claim_status_for(val_bpb),
        "corpus": corpus,
        "max_spans": max_spans,
        "source_id": source_id,
        "log_path": log_path,
        "emitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if git_sha:
        card["git_sha"] = git_sha
    return card


def write_card(card: dict[str, Any], out_dir: Path, filename: str = "measured.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def emit_from_log(
    *,
    log_path: Path,
    out_dir: Path,
    max_spans: Optional[int],
    source_id: str,
    git_sha: Optional[str],
    filename: str = "measured.json",
) -> tuple[int, Path, dict[str, Any]]:
    """Parse log, write card. Returns (exit_code, card_path, card).

    Missing / unreadable / unparseable → exit 2, val_bpb null, claim_status pending.
    """
    text: Optional[str] = None
    path_error = False
    if not log_path.is_file():
        path_error = True
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        print(
            "  Will not invent val_bpb. Pass a real train / morning-report log.",
            file=sys.stderr,
        )
    else:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            path_error = True
            print(f"ERROR: cannot read log: {log_path} ({exc})", file=sys.stderr)
            print("  Will not invent val_bpb.", file=sys.stderr)

    val: Optional[float] = None
    if text is not None:
        val = parse_val_bpb(text)
        if val is None:
            path_error = True
            print(
                f"ERROR: no factual val_bpb in log: {log_path}",
                file=sys.stderr,
            )
            print(
                "  Missing / malformed → null + pending. Never invent a floor.",
                file=sys.stderr,
            )

    card = build_card(
        val_bpb=val,
        max_spans=max_spans,
        source_id=source_id,
        git_sha=git_sha,
        log_path=str(log_path),
    )
    out_path = write_card(card, out_dir, filename=filename)
    print(f"wrote {out_path}")
    print(f"val_bpb: {card['val_bpb']}")
    print(f"claim_status: {card['claim_status']}")
    return (EXIT_PATH_ERROR if path_error else EXIT_OK, out_path, card)


def main(argv: Optional[list[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    refuse_loud_flags(raw)

    parser = argparse.ArgumentParser(
        description=(
            "Emit a factual Tale val_bpb JSON card (or null + pending). "
            "Never invents AUROC / val_bpb. prepare.py untouched."
        )
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to train / morning-report-style log text",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--max-spans",
        type=int,
        default=None,
        help="Optional span cap metadata for the card",
    )
    parser.add_argument(
        "--source-id",
        default="tale-measured",
        help="Card source_id (default: tale-measured)",
    )
    parser.add_argument(
        "--filename",
        default="measured.json",
        help="Card filename inside out-dir (default: measured.json)",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Optional git sha (default: GIT_SHA or GITHUB_SHA env)",
    )
    args = parser.parse_args(raw)

    git_sha = args.git_sha or os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA")
    code, _, _ = emit_from_log(
        log_path=Path(args.log),
        out_dir=Path(args.out_dir),
        max_spans=args.max_spans,
        source_id=args.source_id,
        git_sha=git_sha,
        filename=args.filename,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
