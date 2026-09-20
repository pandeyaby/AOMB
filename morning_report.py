"""
morning_report.py — Autonomous Observability Model Breeder
Run after an overnight autoresearch-macos session to summarize progress.

Usage:
    uv run python morning_report.py
    uv run python morning_report.py --plot   # save overnight_progress.png
    uv run python morning_report.py --from-log train.log  # factual val_bpb only
    uv run python morning_report.py --tale-card  # optional Tale measured card
    uv run python morning_report.py --tale-overnight-dry-run  # overnight helper dry-run
    AOMB_TALE_CARD=1 uv run python morning_report.py

Honesty: never invents AUROC / val_bpb. measured_not_published is not a
public accuracy claim — morning_report only echoes factual card fields.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from val_bpb_parse import parse_val_bpb_from_commit_message, parse_val_bpb_from_train_log

ROOT = Path(__file__).resolve().parent
DEFAULT_TALE_CARD = ROOT / "reports" / "tale-capped" / "measured_capped_200k.json"

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_PATH_ERROR = 2  # missing/malformed card on --tale-overnight-dry-run

# Share invent-flag set with best_val_bpb / stranger CLIs.
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



def refuse_loud_flags(argv: list[str]) -> str | None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            return key
    return None


def _truthy_env(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def resolve_tale_card_path(
    *,
    cli_path: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path | None:
    """Return card path when --tale-card / AOMB_TALE_CARD enables it; else None."""
    env = os.environ if environ is None else environ
    if cli_path is not None:
        return Path(cli_path)
    if _truthy_env(env.get("AOMB_TALE_CARD")):
        override = (env.get("AOMB_BEST_VAL_CARD") or "").strip()
        if override:
            return Path(override).expanduser()
        return DEFAULT_TALE_CARD
    return None


@dataclass(frozen=True)
class TaleCardSnapshot:
    """Factual card view for morning_report (never invents)."""

    path: Path
    state: str  # ok | pending | unavailable | missing | malformed
    val_bpb: float | None
    claim_status: str | None

    @property
    def is_public_accuracy_claim(self) -> bool:
        # measured_not_published must never be treated as a published claim.
        return False


def read_tale_card(path: Path | str) -> TaleCardSnapshot:
    """Load Tale measured card for display. Never invents a val_bpb."""
    card_path = Path(path)
    if not card_path.is_file():
        return TaleCardSnapshot(
            path=card_path,
            state="missing",
            val_bpb=None,
            claim_status=None,
        )
    try:
        data: Any = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return TaleCardSnapshot(
            path=card_path,
            state="malformed",
            val_bpb=None,
            claim_status=None,
        )
    if not isinstance(data, dict):
        return TaleCardSnapshot(
            path=card_path,
            state="malformed",
            val_bpb=None,
            claim_status=None,
        )

    claim_raw = data.get("claim_status")
    claim = claim_raw if isinstance(claim_raw, str) else None

    if claim in (None, "pending"):
        return TaleCardSnapshot(
            path=card_path,
            state="pending",
            val_bpb=None,
            claim_status=claim or "pending",
        )

    val_raw = data.get("val_bpb")
    val: float | None = None
    if val_raw is not None and not isinstance(val_raw, bool):
        try:
            if isinstance(val_raw, str):
                token = val_raw.strip().lower()
                if token in {"", "nan", "inf", "-inf", "+inf", "null", "none", "pending", "n/a"}:
                    val = None
                else:
                    cand = float(val_raw)
                    val = cand if math.isfinite(cand) else None
            else:
                cand = float(val_raw)
                val = cand if math.isfinite(cand) else None
        except (TypeError, ValueError):
            val = None

    if val is None:
        return TaleCardSnapshot(
            path=card_path,
            state="unavailable",
            val_bpb=None,
            claim_status=claim,
        )

    # Finite val present — print it with claim_status. Still not a public claim.
    return TaleCardSnapshot(
        path=card_path,
        state="ok",
        val_bpb=val,
        claim_status=claim,
    )


def format_tale_card_section(snap: TaleCardSnapshot) -> list[str]:
    """Human lines for the Tale card block (no invented numbers / no hero AUROC)."""
    lines = [
        "",
        "  Tale measured card",
        f"  path                 : {snap.path}",
    ]
    if snap.state == "ok" and snap.val_bpb is not None:
        claim = snap.claim_status or "unknown"
        lines.append(f"  card val_bpb         : {snap.val_bpb:.6f}")
        lines.append(f"  claim_status         : {claim}")
        lines.append(
            "  note                 : factual card only — "
            "measured_not_published is NOT a public accuracy / AUROC claim"
        )
    elif snap.state == "pending":
        lines.append("  card val_bpb         : pending / unavailable")
        lines.append(
            f"  claim_status         : {snap.claim_status or 'pending'}"
        )
        lines.append("  note                 : no invented val_bpb")
    elif snap.state == "missing":
        lines.append("  card val_bpb         : unavailable (card missing)")
        lines.append("  note                 : no invented val_bpb")
    elif snap.state == "malformed":
        lines.append("  card val_bpb         : unavailable (card malformed)")
        lines.append("  note                 : no invented val_bpb")
    else:
        lines.append("  card val_bpb         : pending / unavailable")
        if snap.claim_status:
            lines.append(f"  claim_status         : {snap.claim_status}")
        lines.append("  note                 : no invented val_bpb")
    return lines


# ── Git log parsing ───────────────────────────────────────────────────────────

def parse_git_log() -> list[dict]:
    result = subprocess.run(
        ["git", "log", "--format=%H|%s|%ai"],
        capture_output=True, text=True, cwd=Path(__file__).parent
    )
    experiments = []
    for line in result.stdout.strip().split("\n"):
        if "|" not in line or "val_bpb" not in line:
            continue
        parts = line.split("|", 2)
        sha, msg = parts[0][:8], parts[1]
        timestamp = parts[2].strip() if len(parts) > 2 else ""

        # Factual extract only — skip inventing when the token is missing/malformed.
        val_bpb = parse_val_bpb_from_commit_message(msg)
        if val_bpb is None:
            continue

        delta_m  = re.search(r"Δ=([+-]?\d+\.\d+)", msg)
        change_m = re.search(r"\[change: ([^\]]+)\]", msg)
        hyp_m    = re.search(r"\[hypothesis: ([^\]]+)\]", msg)

        experiments.append({
            "sha":       sha,
            "val_bpb":   val_bpb,
            "delta":     float(delta_m.group(1)) if delta_m else None,
            "change":    change_m.group(1) if change_m else msg[:60],
            "hypothesis": hyp_m.group(1) if hyp_m else "",
            "timestamp": timestamp,
        })

    return list(reversed(experiments))  # chronological


def report_val_bpb_from_log(path: Path) -> int:
    """Print factual val_bpb from a train log, or refuse (exit 1) if missing.

    For Mac measured-fill later: cite only what train.py printed.
    """
    if not path.is_file():
        print(f"REFUSED: log file not found: {path}", file=sys.stderr)
        print("No invented val_bpb. Re-run train.py and pass its log.", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    val = parse_val_bpb_from_train_log(text)
    if val is None:
        print(f"REFUSED: no factual val_bpb: line in {path}", file=sys.stderr)
        print(
            "Will not invent AUROC or val_bpb. Wait for train.py to print "
            "`val_bpb: <float>` (final eval), then re-run --from-log.",
            file=sys.stderr,
        )
        return 1
    # Machine-readable single line for Mac measured-fill scripts.
    print(f"val_bpb: {val:.6f}")
    return 0


# ── Sparkline ─────────────────────────────────────────────────────────────────

def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1e-9
    chars = "▁▂▃▄▅▆▇█"
    # Invert: lower val_bpb (better) = taller bar
    return "".join(chars[int((1 - (v - mn) / rng) * (len(chars) - 1))] for v in values)


# ── Main report ───────────────────────────────────────────────────────────────

def main(plot: bool = False, tale_card: Path | None = None):
    W = 72
    print("=" * W)
    print("  AUTONOMOUS OBSERVABILITY MODEL BREEDER — MORNING REPORT")
    print("  Cisco / Splunk / AppDynamics Telemetry Foundation Model")
    print("=" * W)

    if tale_card is not None:
        snap = read_tale_card(tale_card)
        for line in format_tale_card_section(snap):
            print(line)

    exps = parse_git_log()

    if not exps:
        print("\n  No experiments found in git log.")
        print("  Did the agent loop run overnight? Check: git log --oneline\n")
        return

    bpb_vals = [e["val_bpb"] for e in exps]
    best_bpb  = min(bpb_vals)
    worst_bpb = bpb_vals[0]
    best_exp  = min(exps, key=lambda e: e["val_bpb"])
    improvement = worst_bpb - best_bpb
    pct_improve = 100 * improvement / worst_bpb if worst_bpb > 0 else 0

    print(f"\n  Experiments completed  : {len(exps)}")
    print(f"  Starting val_bpb       : {worst_bpb:.4f}")
    print(f"  Best val_bpb           : {best_bpb:.4f}  ({improvement:.4f} = {pct_improve:.1f}% improvement)")
    print(f"  Best experiment        : #{[e['sha'] for e in exps].index(best_exp['sha'])+1} ({best_exp['sha']})")
    print(f"  Best change            : {best_exp['change']}")

    # Anomaly detection readiness
    print()
    if best_bpb < 0.80:
        verdict = "EXCELLENT — strong implicit anomaly detector ready for production"
    elif best_bpb < 1.00:
        verdict = "VERY GOOD — model understands telemetry patterns well"
    elif best_bpb < 1.20:
        verdict = "GOOD — continue experiments through Tier 3/4"
    elif best_bpb < 1.50:
        verdict = "FAIR — architecture/LR tuning still ongoing"
    else:
        verdict = "EARLY STAGE — run more experiments; check for OOM/NaN failures"
    print(f"  Anomaly detector status: {verdict}")

    # Progress sparkline
    print(f"\n  val_bpb trend (lower = better):")
    spark = sparkline(bpb_vals)
    # Print in chunks of 60
    for i in range(0, len(spark), 60):
        prefix = f"  exp {i+1:3d}–{min(i+60, len(spark)):3d}  "
        print(prefix + spark[i:i+60])

    # Full table
    print(f"\n  {'#':>4}  {'SHA':8}  {'val_bpb':>8}  {'Δ':>7}  Change")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*38}")
    for i, exp in enumerate(exps):
        delta_str = f"{exp['delta']:+.4f}" if exp["delta"] is not None else "       "
        marker = " ◀ BEST" if exp["val_bpb"] == best_bpb else ""
        change_short = exp["change"][:38]
        print(f"  {i+1:>4}  {exp['sha']:8}  {exp['val_bpb']:>8.4f}  {delta_str:>7}  {change_short}{marker}")

    # Best 5 hypotheses
    top5 = sorted(exps, key=lambda e: e["val_bpb"])[:5]
    print(f"\n  Top-5 winning changes (by val_bpb):")
    for rank, exp in enumerate(top5, 1):
        print(f"  {rank}. val_bpb={exp['val_bpb']:.4f}  {exp['change'][:55]}")
        if exp["hypothesis"]:
            print(f"       why: {exp['hypothesis'][:65]}")

    # Next actions
    print(f"\n  Recommended next steps:")
    if best_bpb > 1.5:
        print("  → Run Tier 1 experiments: DEPTH, ASPECT_RATIO, WINDOW_PATTERN sweeps")
    elif best_bpb > 1.2:
        print("  → Run Tier 2/3 experiments: LR and schedule tuning")
    elif best_bpb > 1.0:
        print("  → Run Tier 4 experiments: focal loss, n_kv_head reduction")
    else:
        print("  → Run Tier 5: combine best config, push deeper (DEPTH=8)")

    print(f"\n  Restore best model:")
    print(f"  git checkout {best_exp['sha']} -- train.py")

    print(f"\n  Start another overnight session:")
    print(f"  uv run train.py   # verify best config still runs cleanly first")

    print("=" * W)

    # Optional matplotlib plot
    if plot:
        _plot_progress(exps, bpb_vals)


def _plot_progress(exps, bpb_vals):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\n  matplotlib not available: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    steps = list(range(1, len(exps) + 1))

    ax.plot(steps, bpb_vals, marker="o", markersize=4, linewidth=1.5,
            color="#00a8e0", label="val_bpb per experiment")

    # Annotate drops > 0.02
    for i in range(1, len(bpb_vals)):
        if bpb_vals[i] < bpb_vals[i - 1] - 0.02:
            ax.annotate(exps[i]["change"][:25],
                        (steps[i], bpb_vals[i]),
                        textcoords="offset points", xytext=(4, 8),
                        fontsize=6.5, rotation=25, ha="left", color="#333")

    # Reference lines
    for threshold, label, color in [
        (1.5, "Fair", "#f0a030"),
        (1.2, "Good", "#80c040"),
        (1.0, "Very good", "#30a060"),
        (0.8, "Excellent (prod-ready)", "#0070c0"),
    ]:
        ax.axhline(threshold, linestyle="--", color=color, alpha=0.5,
                   linewidth=0.9, label=label)

    ax.set(
        title="Autonomous Observability Model Breeder — val_bpb Progress",
        xlabel="Experiment #",
        ylabel="val_bpb (lower = better)",
        ylim=(max(0, min(bpb_vals) - 0.05), max(bpb_vals) + 0.1),
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = "overnight_progress.png"
    plt.savefig(out, dpi=160)
    print(f"\n  Plot saved: {out}")



def run_tale_overnight_dry_run(card_path: Path | None = None) -> int:
    """Invoke eval.tale_overnight_launch --dry-run only (never --run / agent_loop).

    Missing/malformed measured card → EXIT_PATH_ERROR (2). Invent flags must
    already be refused by cli(). No API spend.
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
            "  morning_report --tale-overnight-dry-run requires a factual measured card.\n"
            "  Never invents val_bpb / AUROC. No agent_loop started; no API spend.\n"
            "  Helper --run would also exit 2 without this card.",
            file=sys.stderr,
        )
        return EXIT_PATH_ERROR
    return overnight_main(["--dry-run", "--card", str(card)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AOMB morning report. Factual val_bpb only — never invents AUROC. "
            "Optional Tale measured card is not a public accuracy claim."
        ),
    )
    parser.add_argument("--plot", action="store_true", help="Save matplotlib progress chart")
    parser.add_argument(
        "--from-log",
        type=Path,
        metavar="PATH",
        help="Parse factual val_bpb from a train log (refuse invent if missing)",
    )
    parser.add_argument(
        "--tale-card",
        nargs="?",
        const=DEFAULT_TALE_CARD,
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Print factual val_bpb + claim_status from a Tale measured card "
            f"(default path: {DEFAULT_TALE_CARD}). "
            "Also enabled by AOMB_TALE_CARD=1. Never invents; "
            "measured_not_published is not a public accuracy claim."
        ),
    )
    parser.add_argument(
        "--tale-overnight-dry-run",
        action="store_true",
        help=(
            "Invoke Tale overnight launch helper in dry-run only "
            "(never --run / agent_loop / API spend). Requires measured card "
            "(missing → exit 2). Linux CI OK."
        ),
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = refuse_loud_flags(argv)
    if bad is not None:
        print(
            f"ERROR: refusing invent / publish / CUDA flag {bad}. "
            "morning_report never invents AUROC / val_bpb.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_FLAG

    args = build_parser().parse_args(argv)
    if args.tale_overnight_dry_run:
        # Dry-run only path — never starts agent_loop; card required (exit 2).
        return run_tale_overnight_dry_run()
    if args.from_log is not None:
        return report_val_bpb_from_log(args.from_log)

    tale_path = resolve_tale_card_path(cli_path=args.tale_card)
    main(plot=args.plot, tale_card=tale_path)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(cli())
