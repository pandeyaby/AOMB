"""Public-wins one-liner from the committed Tale measured card (factual only).

Reads ``reports/tale-capped/measured_capped_200k.json`` and prints one honest
line: val_bpb + claim_status + max_spans + train-fitness disclaimer.

Never invents AUROC / val_bpb. ``prepare.py`` is sacred. No CUDA invent.

Exit codes:
  0 — printed factual line
  1 — refused invent / publish / cuda flags
  2 — card missing / malformed / pending / non-finite (prints unavailable)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS  # noqa: E402

EXIT_OK = 0
EXIT_PATH_ERROR = 2

DEFAULT_CARD = ROOT / "reports" / "tale-capped" / "measured_capped_200k.json"
REQUIRED_CLAIM = "measured_not_published"


def find_refused_invent_flag(argv: list[str]) -> str | None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            return key
    return None


def load_measured_card(path: Path) -> dict[str, Any] | None:
    """Return card dict or None when missing / malformed."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def factual_line_from_card(card: dict[str, Any]) -> str | None:
    """Build the public-wins line, or None if not honestly printable."""
    claim = card.get("claim_status")
    if claim != REQUIRED_CLAIM:
        return None
    val_raw = card.get("val_bpb")
    if val_raw is None or isinstance(val_raw, bool):
        return None
    try:
        val = float(val_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    max_spans = card.get("max_spans")
    spans_s = str(max_spans) if max_spans is not None else "unknown"
    return (
        f"tale_capped val_bpb={val:.6f} "
        f"claim_status={REQUIRED_CLAIM} "
        f"max_spans={spans_s} — "
        "train fitness only, not AUROC"
    )


def format_unavailable(reason: str) -> str:
    return f"unavailable: {reason} — never invent val_bpb / AUROC"


def run(card_path: Path) -> tuple[int, str]:
    """Return (exit_code, stdout_line)."""
    card = load_measured_card(card_path)
    if card is None:
        return EXIT_PATH_ERROR, format_unavailable(
            f"card missing or malformed ({card_path})"
        )
    line = factual_line_from_card(card)
    if line is None:
        claim = card.get("claim_status")
        return EXIT_PATH_ERROR, format_unavailable(
            f"card not printable (claim_status={claim!r}, val_bpb={card.get('val_bpb')!r})"
        )
    return EXIT_OK, line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="public_wins_tale_line",
        description=(
            "Print one factual Tale measured-card line for public-wins. "
            "Never invents AUROC. measured_not_published is not a published "
            "accuracy claim."
        ),
    )
    p.add_argument(
        "--card",
        type=Path,
        default=DEFAULT_CARD,
        help=f"Measured card path (default: {DEFAULT_CARD})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = find_refused_invent_flag(argv)
    if bad is not None:
        print(
            f"ERROR: Refusing '{bad}'.\n"
            "  public-wins Tale line prints factual measured-card fields only.\n"
            "  Never invents AUROC / published ranking / accuracy claims.\n"
            "  Re-run without invent flags.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_FLAG

    args = build_parser().parse_args(argv)
    code, line = run(Path(args.card))
    if code == EXIT_OK:
        print(line)
    else:
        print(line, file=sys.stderr)
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
