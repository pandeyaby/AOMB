"""
Public ranking card v1 — one-command reproduce entry.

Runs multiseed length/events baselines (and optional fixture-only model) on the
public fixture pack's **held-out eval split**, writing reports under
reports/public-ranking-card-v1/.

Model path (--with-model) trains ONLY on frozen train-split session texts
(ephemeral in-memory dataloader). No CRISP. No prepare data download.
prepare.py is untouched.

claim_status becomes `published_fixture_card` (harness smoke) when model mean
AUROC beats length and events on the synthetic eval split — NOT production AUROC.
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
# Model golden ε is looser (CPU float / short train variance across machines)
MODEL_EPS = 1e-2
DEFAULT_SEEDS = "0..4"
DEFAULT_RANDOM_DRAWS = 64
DEFAULT_TRAIN_SECONDS = 45.0


def _run_multiseed(
    *,
    scores_from: str,
    out_dir: Path,
    seeds: str,
    random_draws: int,
    train_seconds: float = 0.0,
    train_corpus: str = "prepare",
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
        "--session-split",
        "eval",
    ]
    if scores_from == "model":
        argv += [
            "--train-seconds",
            str(train_seconds),
            "--train-corpus",
            train_corpus,
        ]
    return multiseed_main(argv)


def _load_agg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_eps(live: dict, ref: dict | None, label: str, eps: float = EPS) -> list[str]:
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
        if abs(float(lv) - float(rv)) > eps:
            failures.append(
                f"{label} {metric}: |{lv} - {rv}| > {eps}"
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
        if abs(float(lv) - float(rv)) > eps:
            failures.append(
                f"{label} precision@{k}: |{lv} - {rv}| > {eps}"
            )
    return failures


def _decide_claim_status(
    length_agg: dict,
    events_agg: dict,
    model_agg: dict | None,
) -> tuple[str, str]:
    """
    Fixture-card / harness-smoke status only.

    `published_fixture_card` when model mean AUROC beats length and events on
    the same eval split. This is NOT production AUROC / general public accuracy.
    """
    if model_agg is None:
        return (
            "not_published",
            "No fixture-only model aggregate; baselines only (harness incomplete).",
        )
    m = float(model_agg["metrics_mean_std"]["auroc"]["mean"])
    l = float(length_agg["metrics_mean_std"]["auroc"]["mean"])
    e = float(events_agg["metrics_mean_std"]["auroc"]["mean"])
    if m > l and m > e:
        return (
            "published_fixture_card",
            (
                f"Fixture-only model mean AUROC {m:.6f} beats length {l:.6f} "
                f"and events {e:.6f} on the frozen synthetic eval split. "
                "Status = published fixture card / harness smoke only — "
                "NOT production AUROC, NOT general public accuracy, NOT lab pool."
            ),
        )
    return (
        "not_published",
        (
            f"Fixture-only model mean AUROC {m:.6f} does not beat both "
            f"length ({l:.6f}) and events ({e:.6f}) on the eval split. "
            "Honest failure — no invented claim."
        ),
    )


def _write_card_summary(
    *,
    fixture_sha: str,
    seeds: list[int],
    length_agg: dict,
    events_agg: dict,
    model_agg: dict | None,
    claim_status: str,
    claim_reason: str,
    train_seconds: float,
    n_eval: int,
    out_path: Path,
) -> None:
    length_auroc = length_agg["metrics_mean_std"]["auroc"]
    events_auroc = events_agg["metrics_mean_std"]["auroc"]
    lines = [
        "# Public ranking card v1 — fixture report",
        "",
        f"**Card id:** `{CARD_ID}`  ",
        f"**claim_status:** `{claim_status}`  ",
        f"**Protocol:** [`{PROTOCOL}`](../../{PROTOCOL})",
        "",
        "> ## Limitations (read first)",
        ">",
        f"> - **n_eval = {n_eval}** labeled sessions on a **synthetic** fixture — harness smoke, not a field study.",
        "> - **High / perfect AUROC on this toy pack ≠ general public accuracy** and ≠ production AUROC.",
        "> - Text patterns are stylized (catalog_ok vs checkout_failed / redis_unavailable); separation can be easy.",
        "> - **Not** private lab-pool AUROC (incl. 0.766). **Not** CRISP `val_bpb`. **Not** a support/SLO metric.",
        "> - Train corpus = fixture train-split **normal** texts only (no CRISP / prepare shards).",
        "",
        f"**Claim gate:** {claim_reason}",
        "",
        "## Identity",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Fixture | `corpus/fixtures/public_ranking_card_v1` |",
        f"| Fixture content SHA-256 | `{fixture_sha}` |",
        f"| Split | `split.json` (eval n={n_eval}) |",
        f"| Seeds | `{seeds}` |",
        f"| ε (deterministic baselines) | `{EPS}` |",
        f"| ε (model golden, if checked) | `{MODEL_EPS}` |",
        "",
        f"## Fixture eval-split baselines (n={n_eval}; mean ± std over seeds)",
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
            f"## Fixture-only model (train {train_seconds:g}s × seeds, eval n={n_eval})",
            "",
            "| Method | AUROC mean | AUROC std | PR-AUC mean | PR-AUC std |",
            "|--------|------------|-----------|-------------|------------|",
            (
                f"| session BPB (fixture train→eval) | {ma['mean']:.6f} | "
                f"{ma['std']:.6f} | {mp['mean']:.6f} | {mp['std']:.6f} |"
            ),
            "",
            (
                "If AUROC is ~1.0 on this synthetic pack, treat it as **toy separation / harness smoke**, "
                "not a marketable production accuracy number."
            ),
            "",
        ]
    lines += [
        "## Explicit non-claims",
        "",
        "- Not general public accuracy or production AUROC.",
        "- Not the private lab pool (including any lab-pool AUROC such as 0.766).",
        "- Not CRISP / synthetic `val_bpb`.",
        "- Not a production support or incident-response SLO metric.",
        "- `prepare.evaluate_bpb` is sacred and unused by this harness path.",
        "",
        "## Lane reminders",
        "",
        "- Private lab pool metrics stay in `docs/lab/` (`not_published` lane).",
        "- `published_fixture_card` = fixture harness smoke that beat baselines — still not production.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Public ranking card v1 reproduce (fixture eval-split)"
    )
    p.add_argument(
        "--baselines-only",
        action="store_true",
        default=False,
        help="Run length + events baselines only (skip model even if golden exists)",
    )
    p.add_argument(
        "--with-model",
        action="store_true",
        help="Run fixture-only short train-then-score on eval split",
    )
    p.add_argument("--train-seconds", type=float, default=DEFAULT_TRAIN_SECONDS)
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
        help="Fail if means diverge from committed refs (baselines; model if present)",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate/check existing reports under out-dir",
    )
    p.add_argument(
        "--write-references",
        action="store_true",
        help="Refresh REFERENCE_*.json from live aggregates (for commit)",
    )
    args = p.parse_args(argv)

    if not FIXTURE.is_dir():
        print(f"ERROR: missing fixture {FIXTURE}", file=sys.stderr)
        return 2
    if not (FIXTURE / "split.json").is_file():
        print(f"ERROR: missing frozen split {FIXTURE / 'split.json'}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    fixture_sha = content_hash_capture(FIXTURE)

    length_dir = out_root / "baselines-length"
    events_dir = out_root / "baselines-events"
    model_dir = out_root / "model-fixture"

    run_model = args.with_model and not args.baselines_only

    if not args.skip_run:
        print(f"[{CARD_ID}] fixture_sha={fixture_sha}")
        print(f"[{CARD_ID}] seeds={seeds} session_split=eval")
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

        if run_model:
            print(
                f"[{CARD_ID}] fixture-only model train_seconds={args.train_seconds} "
                "(no CRISP / prepare shards)"
            )
            rc = _run_multiseed(
                scores_from="model",
                out_dir=model_dir,
                seeds=args.seeds,
                random_draws=args.random_draws,
                train_seconds=args.train_seconds,
                train_corpus="fixture-train",
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

    split = json.loads((FIXTURE / "split.json").read_text(encoding="utf-8"))
    n_eval = int(split.get("n_eval") or len(split.get("eval_session_ids") or []))

    model_agg = None
    model_agg_path = model_dir / "aggregate.json"
    if model_agg_path.is_file():
        model_agg = _load_agg(model_agg_path)

    claim_status, claim_reason = _decide_claim_status(length_agg, events_agg, model_agg)

    # Stamp card identity onto aggregates
    for agg, method in ((length_agg, "length"), (events_agg, "events")):
        agg["card_id"] = CARD_ID
        agg["protocol"] = PROTOCOL
        agg["claim_status"] = claim_status
        agg["claim_reason"] = claim_reason
        agg["fixture_content_sha256"] = fixture_sha
        agg["epsilon"] = EPS
        agg["baseline"] = method
        agg["session_split"] = "eval"
        agg["n_eval"] = n_eval
        (out_root / f"baselines-{method}" / "aggregate.json").write_text(
            json.dumps(agg, indent=2) + "\n", encoding="utf-8"
        )

    if model_agg is not None:
        model_agg["card_id"] = CARD_ID
        model_agg["protocol"] = PROTOCOL
        model_agg["claim_status"] = claim_status
        model_agg["claim_reason"] = claim_reason
        model_agg["fixture_content_sha256"] = fixture_sha
        model_agg["session_split"] = "eval"
        model_agg["n_eval"] = n_eval
        model_agg["train_corpus"] = "fixture_train_split_only"
        model_agg["epsilon_model"] = MODEL_EPS
        model_agg_path.write_text(json.dumps(model_agg, indent=2) + "\n", encoding="utf-8")

    summary_path = out_root / "CARD.md"
    _write_card_summary(
        fixture_sha=fixture_sha,
        seeds=seeds,
        length_agg=length_agg,
        events_agg=events_agg,
        model_agg=model_agg,
        claim_status=claim_status,
        claim_reason=claim_reason,
        train_seconds=args.train_seconds,
        n_eval=n_eval,
        out_path=summary_path,
    )
    print(f"Wrote {summary_path}")

    ref_length = out_root / "REFERENCE_baselines-length.json"
    ref_events = out_root / "REFERENCE_baselines-events.json"
    ref_model = out_root / "REFERENCE_model-fixture.json"
    failures: list[str] = []
    if args.check_eps:
        for live, ref_path, label, eps in (
            (length_agg, ref_length, "length", EPS),
            (events_agg, ref_events, "events", EPS),
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
            failures.extend(_check_eps(live, ref, label, eps=eps))
        if model_agg is not None and ref_model.is_file():
            ref = _load_agg(ref_model)
            ref_sha = ref.get("fixture_content_sha256")
            if ref_sha and ref_sha != fixture_sha:
                failures.append(
                    f"model: fixture SHA mismatch live={fixture_sha} ref={ref_sha}"
                )
            failures.extend(_check_eps(model_agg, ref, "model", eps=MODEL_EPS))
        if failures:
            print("ERROR: ε check failed:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 3
        print(f"ε check passed (baselines ≤{EPS}" + (f", model ≤{MODEL_EPS}" if model_agg and ref_model.is_file() else "") + ")")

    # Refresh references when explicitly requested or after a fresh baseline/model run
    if args.write_references or (not args.skip_run and not args.check_eps):
        for agg, path in ((length_agg, ref_length), (events_agg, ref_events)):
            path.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {path}")
        if model_agg is not None:
            ref_model.write_text(json.dumps(model_agg, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {ref_model}")

    la = length_agg["metrics_mean_std"]["auroc"]["mean"]
    ea = events_agg["metrics_mean_std"]["auroc"]["mean"]
    msg = f"[{CARD_ID}] length AUROC={la:.6f} events AUROC={ea:.6f}"
    if model_agg is not None:
        ma = model_agg["metrics_mean_std"]["auroc"]["mean"]
        ms = model_agg["metrics_mean_std"]["auroc"]["std"]
        msg += f" model AUROC={ma:.6f}±{ms:.6f}"
    msg += f" (claim_status={claim_status})"
    print(msg)
    print(f"[{CARD_ID}] {claim_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
