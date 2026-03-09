"""
morning_report.py — Autonomous Observability Model Breeder
Run after an overnight autoresearch-macos session to summarize progress.

Usage:
    uv run python morning_report.py
    uv run python morning_report.py --plot   # save overnight_progress.png
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


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

        bpb_m   = re.search(r"val_bpb=(\d+\.\d+)", msg)
        delta_m  = re.search(r"Δ=([+-]?\d+\.\d+)", msg)
        change_m = re.search(r"\[change: ([^\]]+)\]", msg)
        hyp_m    = re.search(r"\[hypothesis: ([^\]]+)\]", msg)

        if bpb_m:
            experiments.append({
                "sha":       sha,
                "val_bpb":   float(bpb_m.group(1)),
                "delta":     float(delta_m.group(1)) if delta_m else None,
                "change":    change_m.group(1) if change_m else msg[:60],
                "hypothesis": hyp_m.group(1) if hyp_m else "",
                "timestamp": timestamp,
            })

    return list(reversed(experiments))  # chronological


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

def main(plot: bool = False):
    W = 72
    print("=" * W)
    print("  AUTONOMOUS OBSERVABILITY MODEL BREEDER — MORNING REPORT")
    print("  Cisco / Splunk / AppDynamics Telemetry Foundation Model")
    print("=" * W)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOMB morning report")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib progress chart")
    args = parser.parse_args()
    main(plot=args.plot)
