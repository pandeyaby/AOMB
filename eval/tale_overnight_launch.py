"""Tale overnight launch helper — dry-run first; factual card floor only.

Documents / sets lane env for agent_loop breeding on capped Tale:
  AOMB_CORPUS=tale_of_errors
  AOMB_SOURCE_ID=tale_capped_200k
  AOMB_BEST_VAL_FROM_CARD=1

Default ``--dry-run`` prints the plan and exits 0 (no agent_loop, no APIs).
``--run`` requires the measured card + Darwin/MPS; never invents a floor.

Never invents AUROC / val_bpb. prepare.py sacred. No workflow spend in CI.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.stranger_path import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS  # noqa: E402
from eval.public_wins_tale_line import (  # noqa: E402
    DEFAULT_CARD,
    factual_line_from_card,
    load_measured_card,
)

EXIT_OK = 0
EXIT_PLATFORM = 2  # missing card / not Darwin+MPS on --run (same family as path errors)

DEFAULT_CORPUS = "tale_of_errors"
DEFAULT_SOURCE_ID = "tale_capped_200k"
DEFAULT_FROM_CARD = "1"


def find_refused_invent_flag(argv: list[str]) -> str | None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            return key
    return None


def planned_env(
    *,
    corpus: str = DEFAULT_CORPUS,
    source_id: str = DEFAULT_SOURCE_ID,
    from_card: str = DEFAULT_FROM_CARD,
    card_path: Path = DEFAULT_CARD,
) -> dict[str, str]:
    return {
        "AOMB_CORPUS": corpus,
        "AOMB_SOURCE_ID": source_id,
        "AOMB_BEST_VAL_FROM_CARD": from_card,
        "AOMB_BEST_VAL_CARD": str(card_path),
    }


def card_ok(card_path: Path) -> tuple[bool, str]:
    """Return (ok, message). ok only when measured_not_published + finite val_bpb."""
    card = load_measured_card(card_path)
    if card is None:
        return False, f"measured card missing or malformed: {card_path}"
    line = factual_line_from_card(card)
    if line is None:
        return (
            False,
            f"measured card not printable "
            f"(claim_status={card.get('claim_status')!r}, "
            f"val_bpb={card.get('val_bpb')!r}): {card_path}",
        )
    return True, line


def darwin_mps_ok() -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, f"not Darwin (detected: {platform.system()})"
    try:
        import torch  # type: ignore
    except ImportError:
        return False, "torch not importable (needed for MPS check)"
    try:
        ok = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        return False, f"MPS check failed: {exc}"
    if not ok:
        return False, "MPS not available"
    return True, "Darwin + MPS available"


def format_dry_run(
    env: dict[str, str],
    *,
    card_path: Path,
    card_status: str,
) -> str:
    lines = [
        "tale_overnight_launch: dry-run (no agent_loop, no APIs)",
        f"  would set AOMB_CORPUS={env['AOMB_CORPUS']}",
        f"  would set AOMB_SOURCE_ID={env['AOMB_SOURCE_ID']}",
        f"  would set AOMB_BEST_VAL_FROM_CARD={env['AOMB_BEST_VAL_FROM_CARD']}",
        f"  would set AOMB_BEST_VAL_CARD={env['AOMB_BEST_VAL_CARD']}",
        f"  measured card: {card_path}",
        f"  card status: {card_status}",
        "  honesty: train fitness floor from card only — never invent AUROC / val_bpb",
        "  next: ./scripts/tale_overnight_launch.sh --run   # Darwin+MPS + card required",
        "  agent: python agent_loop.py   # not started in dry-run",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tale_overnight_launch",
        description=(
            "Tale overnight launch helper. Default --dry-run prints lane env "
            "and exits without starting agent_loop. --run requires measured "
            "card + Darwin/MPS. Never invents AUROC / floors."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned env and exit 0 (default if neither mode set)",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Start agent_loop with Tale lane env (Darwin+MPS + card required)",
    )
    p.add_argument(
        "--card",
        type=Path,
        default=DEFAULT_CARD,
        help=f"Measured card path (default: {DEFAULT_CARD})",
    )
    p.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"AOMB_CORPUS value (default: {DEFAULT_CORPUS})",
    )
    p.add_argument(
        "--source-id",
        default=DEFAULT_SOURCE_ID,
        help=f"AOMB_SOURCE_ID value (default: {DEFAULT_SOURCE_ID})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = find_refused_invent_flag(argv)
    if bad is not None:
        print(
            f"ERROR: Refusing '{bad}'.\n"
            "  tale_overnight_launch breeds on factual Tale val_bpb only.\n"
            "  Never invents AUROC / published ranking / accuracy claims.\n"
            "  Re-run without invent flags. prepare.py stays untouched.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_FLAG

    args = build_parser().parse_args(argv)
    do_run = bool(args.run)
    # Default to dry-run when neither flag given.
    do_dry = bool(args.dry_run) or not do_run

    env = planned_env(
        corpus=args.corpus,
        source_id=args.source_id,
        card_path=Path(args.card),
    )
    ok, card_msg = card_ok(Path(args.card))

    if do_dry and not do_run:
        status = card_msg if ok else f"unavailable ({card_msg}) — --run would exit 2"
        print(format_dry_run(env, card_path=Path(args.card), card_status=status))
        return EXIT_OK

    # --run path
    if not ok:
        print(
            f"ERROR: {card_msg}\n"
            "  Cannot start overnight without a factual measured card floor.\n"
            "  Never invents val_bpb. Emit card via eval.tale_measured_report first.",
            file=sys.stderr,
        )
        return EXIT_PLATFORM

    plat_ok, plat_msg = darwin_mps_ok()
    if not plat_ok:
        print(
            f"ERROR: platform check failed: {plat_msg}\n"
            "  --run requires Darwin + MPS. Use --dry-run on Linux/CI.\n"
            "  No agent_loop started; no API spend.",
            file=sys.stderr,
        )
        return EXIT_PLATFORM

    print(f"tale_overnight_launch: starting agent_loop with card floor:\n  {card_msg}")
    for k, v in env.items():
        os.environ[k] = v
    return start_agent_loop()


def start_agent_loop() -> int:
    """Start agent_loop.py (os.execv). Tests patch this — never call paid APIs in CI."""
    agent = ROOT / "agent_loop.py"
    if not agent.is_file():
        print(f"ERROR: missing {agent}", file=sys.stderr)
        return EXIT_PLATFORM
    os.execv(sys.executable, [sys.executable, str(agent)])
    return EXIT_OK  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
