"""
Stranger path honesty — shared refusals + exit codes for public scripts.

Source of truth for scripts/stranger_demo.sh and scripts/stranger_verify.sh.
Keep shell case arms in sync with REFUSED_METRIC_FLAGS.

Public stranger path = DIPTYCH full-8 + ranking-card harness smoke only.
Never invents AUROC / val_bpb / published ranking. CUDA gate stays skipped.
prepare.py is sacred. No Zenodo / no MPS / no overnight agent.
CI may use --tale-card-line / --tale-overnight-dry-run (dry-run only; no agent_loop).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Session-scorer / demo_anomaly invent set + CUDA (gate stays skipped).
# Keep in sync with scripts/stranger_demo.sh and scripts/stranger_verify.sh.
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
        "--val-bpb",
        "--invent-val-bpb",
        "--readme-hero",
        "--publish-readme",
        "--hero-auroc",
        "--cuda",
        "--gpu",
    }
)

# Overnight / MPS train flags — also EXIT_REFUSED_FLAG (not a product path).
REFUSED_PRODUCT_FLAGS = frozenset(
    {
        "--overnight",
        "--agent",
        "--mps-train",
    }
)

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_PATH_ERROR = 2  # missing/malformed Tale card


def refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on invent / CUDA / overnight flags.

    Raises SystemExit with code EXIT_REFUSED_FLAG.
    """
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            print(
                f"ERROR: Refusing '{key}'.\n"
                "  Stranger path proves DIPTYCH full-8 + ranking-card harness smoke only.\n"
                "  Never invents AUROC / val_bpb / published ranking accuracy.\n"
                "  Lab claim_status stays not_published.\n"
                "  CUDA gate stays skipped — use CPU stranger verify / demo.\n"
                "  Run without invent flags. Optional: STRANGER_FAST=1 for baselines-only.",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED_FLAG)
        if key in REFUSED_PRODUCT_FLAGS:
            print(
                f"ERROR: Refusing '{key}'.\n"
                "  Stranger path = no overnight agent, no MPS train, no API keys.\n"
                "  Run without those flags. Optional: STRANGER_FAST=1 for baselines-only.\n"
                "  On Apple Silicon with keys, see README Quickstart (agent_loop / train.py).",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED_FLAG)


TALE_OVERNIGHT_DRY_RUN_FLAGS = frozenset({"--tale-overnight-dry-run"})


def wants_tale_card_line(argv: list[str]) -> bool:
    """True when argv requests the Tale measured-card one-liner (CI-safe)."""
    for arg in argv:
        if arg.split("=", 1)[0] == "--tale-card-line":
            return True
    return False


def wants_tale_overnight_dry_run(argv: list[str]) -> bool:
    """True when argv requests Tale overnight launch dry-run (CI-safe; no agent_loop)."""
    for arg in argv:
        if arg.split("=", 1)[0] in TALE_OVERNIGHT_DRY_RUN_FLAGS:
            return True
    return False


def run_tale_overnight_dry_run(card_path: Path | None = None) -> int:
    """Invoke eval.tale_overnight_launch --dry-run only (never --run / agent_loop).

    Missing/malformed measured card → EXIT_PATH_ERROR (2), same honesty as
    product_mac / helper card gate. Invent flags must already be refused by caller.
    """
    from eval.tale_overnight_launch import (
        DEFAULT_CARD,
        card_ok,
        main as overnight_main,
    )

    card = Path(card_path) if card_path is not None else DEFAULT_CARD
    ok, msg = card_ok(card)
    if not ok:
        print(
            f"ERROR: {msg}\n"
            "  stranger --tale-overnight-dry-run requires a factual measured card.\n"
            "  Never invents val_bpb / AUROC. No agent_loop started; no API spend.\n"
            "  Helper --run would also exit 2 without this card.",
            file=sys.stderr,
        )
        return EXIT_PATH_ERROR
    # Dry-run only — never pass --run; never call start_agent_loop.
    return overnight_main(["--dry-run", "--card", str(card)])


def main(argv: list[str] | None = None) -> int:
    """Thin CLI: refuse invent flags; optional tale CI modes; else help."""
    args = list(sys.argv[1:] if argv is None else argv)
    refuse_loud_flags(args)
    if wants_tale_overnight_dry_run(args):
        return run_tale_overnight_dry_run()
    if wants_tale_card_line(args):
        from eval.public_wins_tale_line import DEFAULT_CARD, run as tale_run

        code, line = tale_run(DEFAULT_CARD)
        if code == EXIT_OK:
            print(line)
        else:
            print(line, file=sys.stderr)
            print(line)
        return code
    if args and args[0] in ("-h", "--help", "help"):
        print(
            "AOMB stranger path honesty helper.\n"
            "  Scripts: ./scripts/stranger_demo.sh · ./scripts/stranger_verify.sh\n"
            "  CI: --tale-card-line / --tale-overnight-dry-run (Linux OK; no agent_loop).\n"
            "  Refuses --auroc / --publish / --cuda / invent flags (exit 1).\n"
            "  Missing measured card on overnight dry-run → exit 2.\n"
            "  Lab claim_status stays not_published. prepare.py sacred.\n"
            "  CUDA gate stays skipped.",
            file=sys.stderr,
        )
        return EXIT_OK
    if args:
        print(
            f"ERROR: unknown stranger-path arg {args[0]!r}.\n"
            "  Use ./scripts/stranger_demo.sh or ./scripts/stranger_verify.sh.\n"
            "  Or: python -m eval.stranger_path --help",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_REFUSED_FLAG)
    print(
        "stranger_path: honesty OK (no invent flags).\n"
        "  claim_status=not_published · CUDA gate skipped · prepare.py sacred",
        file=sys.stderr,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
