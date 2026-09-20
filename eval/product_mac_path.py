"""
Product Mac path honesty — shared refusals + exit codes for MPS smoke.

Source of truth for scripts/product_mac_smoke.sh.
Keep shell case arms in sync with REFUSED_METRIC_FLAGS.

Product Mac path = Darwin + MPS train fitness (factual val_bpb only).
Never invents AUROC / published ranking. CUDA gate stays skipped.
prepare.py is sacred. CI uses --dry-run / --help-only (no MPS / no TIME_BUDGET).
"""

from __future__ import annotations

import sys

# Session-scorer invent set + CUDA + invent-val_bpb flags.
# Keep in sync with scripts/product_mac_smoke.sh.
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

# CI-safe modes: no Darwin/MPS/train required.
DRY_RUN_FLAGS = frozenset({"--dry-run", "--help-only", "-h", "--help", "help"})

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_PLATFORM = 2  # not Darwin / MPS unavailable on real path


def refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on invent / CUDA / publish flags.

    Raises SystemExit with code EXIT_REFUSED_FLAG.
    Applies to --dry-run and real MPS paths alike.
    """
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            print(
                f"ERROR: Refusing '{key}'.\n"
                "  Product Mac smoke reports factual val_bpb on Darwin + MPS only.\n"
                "  Never invents AUROC / published ranking accuracy.\n"
                "  Lab claim_status stays not_published.\n"
                "  CUDA gate stays skipped — no --cuda / --gpu claim path.\n"
                "  CI: --dry-run or --help-only (no MPS / no full TIME_BUDGET).",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_REFUSED_FLAG)


def is_dry_run(argv: list[str]) -> bool:
    """True when argv requests CI dry-run / help-only (no MPS)."""
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in DRY_RUN_FLAGS:
            return True
    return False


def dry_run_message() -> str:
    return (
        "product_mac_path: dry-run / help-only OK\n"
        "  claim_status=not_published · CUDA gate stays skipped · prepare.py sacred\n"
        "  Real path: Darwin + MPS → factual val_bpb (not AUROC; not CUDA)\n"
        "  CI: no MPS required · no full TIME_BUDGET · no invent metrics"
    )


def main(argv: list[str] | None = None) -> int:
    """Thin CLI: refuse invent flags; dry-run/help exit 0; else honesty hint."""
    args = list(sys.argv[1:] if argv is None else argv)
    refuse_loud_flags(args)

    if is_dry_run(args):
        # Prefer help text when only help was asked.
        help_only = any(
            a.split("=", 1)[0] in ("-h", "--help", "help", "--help-only") for a in args
        )
        if help_only and "--dry-run" not in {a.split("=", 1)[0] for a in args}:
            print(
                "AOMB product Mac path honesty helper.\n"
                "  Script: ./scripts/product_mac_smoke.sh\n"
                "  Real: Darwin + MPS → factual val_bpb only (no AUROC / no CUDA).\n"
                "  CI: --dry-run or --help-only (Linux OK; no MPS / no TIME_BUDGET).\n"
                "  Refuses --auroc / --publish / --cuda / invent flags (exit 1).\n"
                "  Lab claim_status stays not_published. prepare.py sacred.\n"
                "  CUDA gate stays skipped. Platform fail on real path: exit 2.",
                file=sys.stderr,
            )
            return EXIT_OK
        print(dry_run_message(), file=sys.stderr)
        return EXIT_OK

    if args:
        print(
            f"ERROR: unknown product-mac-path arg {args[0]!r}.\n"
            "  Use ./scripts/product_mac_smoke.sh\n"
            "  CI: python -m eval.product_mac_path --dry-run\n"
            "  Or: python -m eval.product_mac_path --help",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_REFUSED_FLAG)

    print(
        "product_mac_path: honesty OK (no invent flags).\n"
        "  Real smoke: ./scripts/product_mac_smoke.sh on Darwin + MPS\n"
        "  CI: --dry-run · claim_status=not_published · CUDA gate skipped",
        file=sys.stderr,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
