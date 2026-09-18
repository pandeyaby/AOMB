"""
Public ranking card v1 — one-command reproduce entry.

Runs multiseed baselines (length + events) on the public fixture pack and
writes reports under reports/public-ranking-card-v1/.

claim_status=not_published. Does not modify prepare.py.
Does not promote private lab-pool AUROC as the public card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.labels import content_hash_capture
from eval.run_multiseed import main as multiseed_main
from eval.run_multiseed import parse_seeds

CARD_ID = "public_ranking_card_v1"
FIXTURE = ROOT / "corpus" / "fixtures" / "public_ranking_card_v1"
REPORT_ROOT = ROOT / "reports" / "public-ranking-card-v1"
PROTOCOL = "docs/public-ranking-card-v1.md"
# Deterministic baseline AUROC/PR-AUC/precision must match within this ε
EPS = 1e-6
DEFAULT_SEEDS = "0..4"
DEFAULT_RANDOM_DRAWS = 64


def _run_multiseed(
    *,
    scores_from: str,
    out_dir: Path,
    seeds: str,
    random_draws: int,
    train_seconds: float = 0.0,
) -> int:
    argv = [
        "--capture",
        str(FIXTURE),
        "--scores-from",
        scores_from,
        "--seeds",
        seeds,
        "--out-dir",
        str(out_dir),
        "--random-draws",
        str(random_draws),
    ]
    if scores_from == "model":
        argv += ["--train-seconds", str(train_seconds)]
    return multiseed_main(argv)


def _load_agg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_eps(live: dict, ref: dict | None, label: str) -> list[str]:
    """Return list of ε failures (empty if ok / no reference)."""
    if ref is None:
        return []
    failures: list[str] = []
    live_ms = live.get("metrics_mean_std") or {}
    ref_ms = ref.get("metrics_mean_std") or {}
    for metric in ("auroc", "pr_auc"):
        lv = (live_ms.get(metric) or {}).get("mean")
        rv = (ref_ms.get(metric) or {}).get("mean")
        if lv is None or rv is None:
            continue
        if abs(float(lv) - float(rv)) > EPS:
            failures.append(
                f"{label} {metric}: |{lv} - {rv}| > {EPS}"
            )
    live_pk = live_ms.get("precision_at_k") or {}
    ref_pk = ref_ms.get("precision_at_k") or {}
    for k, ref_block in ref_pk.items():
        if k not in live_pk:
            failures.append(f"{label} precision@{k}: missing in live")
            continue
        lv = live_pk[k].get("mean")
        rv = ref_block.get("mean")
        if lv is None or rv is None:
            continue
        if abs(float(lv) - float(rv)) > EPS:
            failures.append(
                f"{label} precision@{k}: |{lv} - {rv}| > {EPS}"
            )
    return failures


def _write_card_summary(
    *,
    fixture_sha: str,
    seeds: list[int],
    length_agg: dict,
    events_agg: dict,
    model_agg: dict | None,
    out_path: Path,
) -> None:
    length_auroc = length_agg["metrics_mean_std"]["auroc"]
    events_auroc = events_agg["metrics_mean_std"]["auroc"]
    lines = [
        "# Public ranking card v1 — fixture report",
        "",
        f"**Card id:** `{CARD_ID}`  ",
        f"**claim_status:** `not_published`  ",
        f"**Protocol:** [`{PROTOCOL}`](../../{PROTOCOL})",
        "",
        "> Fixture baselines only (unless a model section is present).  ",
        "> **Not** private lab-pool AUROC. **Not** CRISP `val_bpb`.  ",
        "> Do not publish until checklist + Abhinav greenlight.",
        "",
        "## Identity",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Fixture | `corpus/fixtures/public_ranking_card_v1` |",
        f"| Fixture content SHA-256 | `{fixture_sha}` |",
        f"| Seeds | `{seeds}` |",
        f"| ε (deterministic baselines) | `{EPS}` |",
        "",
        "## Fixture baselines (mean ± std over seeds)",
        "",
        "| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |",
        "|--------|------------|-----------|-------------|------------|",
        (
            f"| length | {length_auroc['mean']:.6f} | {length_auroc['std']:.6f} | "
            f"{length_agg['metrics_mean_std']['pr_auc']['mean']:.6f} | "
            f"{length_agg['metrics_mean_std']['pr_auc']['std']:.6f} |"
        ),
        (
            f"| events | {events_auroc['mean']:.6f} | {events_auroc['std']:.6f} | "
            f"{events_agg['metrics_mean_std']['pr_auc']['mean']:.6f} | "
            f"{events_agg['metrics_mean_std']['pr_auc']['std']:.6f} |"
        ),
        "",
        "Random ranking baseline is included inside each per-seed `report.json`.",
        "",
    ]
    if model_agg is not None:
        ma = model_agg["metrics_mean_std"]["auroc"]
        mp = model_agg["metrics_mean_std"]["pr_auc"]
        lines += [
            "## Optional short model path (not CI)",
            "",
            "| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |",
            "|--------|------------|-----------|-------------|------------|",
            (
                f"| session BPB (short train) | {ma['mean']:.6f} | {ma['std']:.6f} | "
                f"{mp['mean']:.6f} | {mp['std']:.6f} |"
            ),
            "",
            "Model numbers here are **fixture-only** and still `not_published`.",
            "",
        ]
    lines += [
        "## Lane reminders",
        "",
        "- Private lab pool metrics are **out of scope** for this card (never copy them here).",
        "- CRISP / synthetic `val_bpb` are training facts, not ranking accuracy.",
        "- `prepare.evaluate_bpb` is sacred and unused by this harness path.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Public ranking card v1 reproduce (fixture baselines)"
    )
    p.add_argument(
        "--baselines-only",
        action="store_true",
        default=True,
        help="Run length + events baselines (default)",
    )
    p.add_argument(
        "--with-model",
        action="store_true",
        help="Also run optional short train-then-score model path",
    )
    p.add_argument("--train-seconds", type=float, default=30.0)
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    p.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPORT_ROOT),
        help="Report root (default: reports/public-ranking-card-v1)",
    )
    p.add_argument(
        "--check-eps",
        action="store_true",
        help="Fail if deterministic baseline means diverge >1e-6 from committed refs",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate/check existing reports under out-dir",
    )
    args = p.parse_args(argv)

    if not FIXTURE.is_dir():
        print(f"ERROR: missing fixture {FIXTURE}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    fixture_sha = content_hash_capture(FIXTURE)

    length_dir = out_root / "baselines-length"
    events_dir = out_root / "baselines-events"

    if not args.skip_run:
        print(f"[{CARD_ID}] fixture_sha={fixture_sha}")
        print(f"[{CARD_ID}] seeds={seeds} claim_status=not_published")
        rc = _run_multiseed(
            scores_from="length",
            out_dir=length_dir,
            seeds=args.seeds,
            random_draws=args.random_draws,
        )
        if rc != 0:
            return rc
        rc = _run_multiseed(
            scores_from="events",
            out_dir=events_dir,
            seeds=args.seeds,
            random_draws=args.random_draws,
        )
        if rc != 0:
            return rc

        if args.with_model:
            model_dir = out_root / "model-short"
            rc = _run_multiseed(
                scores_from="model",
                out_dir=model_dir,
                seeds=args.seeds,
                random_draws=args.random_draws,
                train_seconds=args.train_seconds,
            )
            if rc != 0:
                return rc

    length_agg_path = length_dir / "aggregate.json"
    events_agg_path = events_dir / "aggregate.json"
    if not length_agg_path.is_file() or not events_agg_path.is_file():
        print(
            f"ERROR: missing aggregates under {out_root} "
            "(run without --skip-run first)",
            file=sys.stderr,
        )
        return 2

    length_agg = _load_agg(length_agg_path)
    events_agg = _load_agg(events_agg_path)
    # Stamp card identity onto aggregates (non-destructive copy fields)
    for agg, method in ((length_agg, "length"), (events_agg, "events")):
        agg["card_id"] = CARD_ID
        agg["protocol"] = PROTOCOL
        agg["claim_status"] = "not_published"
        agg["fixture_content_sha256"] = fixture_sha
        agg["epsilon"] = EPS
        agg["baseline"] = method
        (out_root / f"baselines-{method}" / "aggregate.json").write_text(
            json.dumps(agg, indent=2) + "\n", encoding="utf-8"
        )

    model_agg = None
    model_agg_path = out_root / "model-short" / "aggregate.json"
    if model_agg_path.is_file():
        model_agg = _load_agg(model_agg_path)
        model_agg["card_id"] = CARD_ID
        model_agg["protocol"] = PROTOCOL
        model_agg["claim_status"] = "not_published"
        model_agg["fixture_content_sha256"] = fixture_sha
        model_agg_path.write_text(json.dumps(model_agg, indent=2) + "\n", encoding="utf-8")

    summary_path = out_root / "CARD.md"
    _write_card_summary(
        fixture_sha=fixture_sha,
        seeds=seeds,
        length_agg=length_agg,
        events_agg=events_agg,
        model_agg=model_agg,
        out_path=summary_path,
    )
    print(f"Wrote {summary_path}")

    # Reference aggregates for ε (committed next to reports)
    ref_length = out_root / "REFERENCE_baselines-length.json"
    ref_events = out_root / "REFERENCE_baselines-events.json"
    failures: list[str] = []
    if args.check_eps:
        for live, ref_path, label in (
            (length_agg, ref_length, "length"),
            (events_agg, ref_events, "events"),
        ):
            if not ref_path.is_file():
                failures.append(f"missing reference {ref_path}")
                continue
            ref = _load_agg(ref_path)
            ref_sha = ref.get("fixture_content_sha256")
            if ref_sha and ref_sha != fixture_sha:
                failures.append(
                    f"{label}: fixture SHA mismatch live={fixture_sha} ref={ref_sha}"
                )
            failures.extend(_check_eps(live, ref, label))
        if failures:
            print("ERROR: ε check failed:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 3
        print(f"ε check passed (≤{EPS})")

    # Always refresh references when running a full baseline pass (for commit)
    if not args.skip_run and not args.with_model:
        for agg, path in ((length_agg, ref_length), (events_agg, ref_events)):
            path.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {path}")

    la = length_agg["metrics_mean_std"]["auroc"]["mean"]
    ea = events_agg["metrics_mean_std"]["auroc"]["mean"]
    print(
        f"[{CARD_ID}] length AUROC={la:.6f} events AUROC={ea:.6f} "
        f"(claim_status=not_published)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
