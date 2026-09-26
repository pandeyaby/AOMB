"""
Public ranking card v1 — honest fixture runner.

Scores **committed** local fixtures (public_ranking_card_v1 / lab_public_pack_v0)
and emits only harness-allowed outputs:

  - length / events baselines (deterministic, no torch)
  - optional fixture-only session-BPB model path (--with-model)
  - ranking aggregates computed from those real scores (never invented)

Does **not** invent AUROC heroes, README marketing numbers, lab-pool AUROC,
or val_bpb. Loudly refuses invent / publish / README-hero flags.

Model path (--with-model) trains ONLY on frozen train-split session texts
(ephemeral in-memory dataloader). No CRISP. No prepare data download.
prepare.py is untouched.

claim_status becomes `published_fixture_card` (harness smoke) only when the
frozen public_ranking_card_v1 model mean AUROC beats length and events on the
synthetic eval split — NOT production AUROC, NOT a README hero.
lab_public_pack_v0 stays `not_published`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.labels import content_hash_capture, filter_scorable, load_lab_sessions
from eval.run_multiseed import main as multiseed_main
from eval.run_multiseed import parse_seeds
from eval.score import score_event_count, score_length

CARD_ID = "public_ranking_card_v1"
FIXTURES_ROOT = ROOT / "corpus" / "fixtures"
DEFAULT_FIXTURE_NAME = "public_ranking_card_v1"
KNOWN_FIXTURES = frozenset(
    {
        "public_ranking_card_v1",
        "lab_public_pack_v0",
    }
)
REPORT_ROOT = ROOT / "reports" / "public-ranking-card-v1"
PROTOCOL = "docs/public-ranking-card-v1.md"
# Deterministic baseline AUROC/PR-AUC/precision must match within this ε
EPS = 1e-6
# Model golden ε is looser (CPU float / short train variance across machines)
MODEL_EPS = 1e-2
DEFAULT_SEEDS = "0..4"
DEFAULT_RANDOM_DRAWS = 64
DEFAULT_TRAIN_SECONDS = 45.0

# Loud refusals — never invent AUROC heroes / README publish / val_bpb.
# The harness still *computes* ranking metrics from real fixture scores;
# these flags would invent or market claims without that honest path.
_REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--publish",
        "--claim",
        "--invent-metrics",
        "--invent-auroc",
        "--claim-auroc",
        "--readme-hero",
        "--publish-readme",
        "--hero-auroc",
        "--val-bpb",
        "--invent-val-bpb",
    }
)


def _refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on invent / publish / README-hero flags before argparse."""
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in _REFUSED_METRIC_FLAGS:
            raise SystemExit(
                f"ERROR: Refusing '{key}'.\n"
                "  Ranking-card runner scores committed local fixtures only.\n"
                "  Emits length/events baselines (+ optional session BPB model).\n"
                "  Never invents AUROC heroes, README marketing numbers, or val_bpb.\n"
                "  Lab / lab_public_pack stay claim_status=not_published.\n"
                "  Use --baselines-only, --with-model, --fixture NAME, or --help."
            )


def resolve_fixture(name_or_path: str) -> Path:
    """Resolve a known fixture name or absolute/relative capture path."""
    raw = Path(name_or_path)
    if raw.is_dir():
        return raw.resolve()
    # Explicit path that does not exist yet — surface as missing dir (validate).
    if raw.is_absolute() or "/" in name_or_path or name_or_path.startswith("."):
        return raw.resolve()
    name = name_or_path.strip().strip("/")
    if name in KNOWN_FIXTURES:
        return (FIXTURES_ROOT / name).resolve()
    # Allow corpus/fixtures/<name> style
    candidate = FIXTURES_ROOT / name
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Unknown fixture {name_or_path!r}. "
        f"Known committed fixtures: {sorted(KNOWN_FIXTURES)}"
    )


def validate_fixture(fixture: Path) -> Optional[str]:
    """
    Validate committed fixture layout before scoring.

    Returns an error message string, or None when usable.
    """
    if not fixture.is_dir():
        return (
            f"ERROR: fixture directory not found: {fixture}\n"
            f"  Known committed fixtures: {sorted(KNOWN_FIXTURES)}\n"
            "  Or pass --fixture public_ranking_card_v1 | lab_public_pack_v0."
        )
    if not (fixture / "provenance.json").is_file():
        return (
            f"ERROR: missing provenance.json under {fixture}\n"
            "  Ranking card requires a labeled lab_capture-style fixture.\n"
            "  Or use --help."
        )
    has_traces = (fixture / "traces.jsonl").is_file() or (
        fixture / "spans.jsonl"
    ).is_file()
    if not has_traces:
        return (
            f"ERROR: missing traces.jsonl / spans.jsonl under {fixture}\n"
            "  Fixture must contain session telemetry to score.\n"
            "  Or use --help."
        )
    return None


def emit_session_baseline_scores(
    fixture: Path,
    *,
    session_split: str = "all",
    out_path: Path,
) -> dict[str, Any]:
    """
    Emit per-session length/events baseline scores from a real local fixture.

    No torch. No invented AUROC. claim_status stays not_published on this
    sidecar (aggregates may still report harness ranking metrics separately).
    """
    sessions, meta = load_lab_sessions(fixture)
    y_true, kept = filter_scorable(sessions)
    split_meta: dict[str, Any] = {"split_role": "all"}
    if session_split != "all":
        from eval.fixture_train import load_split, partition_by_split

        split = load_split(fixture)
        train_s, eval_s = partition_by_split(kept, split)
        kept = eval_s if session_split == "eval" else train_s
        y_true = [int(s.binary) for s in kept]  # type: ignore[arg-type]
        split_meta = {
            "split_role": session_split,
            "split_id": split.get("split_id"),
            "n_train": int(split.get("n_train") or len(train_s)),
            "n_eval": int(split.get("n_eval") or len(eval_s)),
        }

    length_scores = score_length(kept)
    events_scores = score_event_count(kept)
    rows = []
    for s, y, ls, es in zip(kept, y_true, length_scores, events_scores):
        rows.append(
            {
                "session_id": s.session_id,
                "label": s.label,
                "binary": y,
                "n_chars": s.n_chars,
                "n_events": s.n_events,
                "score_length": float(ls),
                "score_events": float(es),
            }
        )

    payload: dict[str, Any] = {
        "card_id": CARD_ID,
        "claim_status": "not_published",
        "disclaimer": (
            "Per-session length/events baselines from a committed local fixture. "
            "Not a published ranking claim and not lab-pool accuracy. "
            "Harness ranking aggregates (when run) are computed from these "
            "real scores — never invented."
        ),
        "fixture": str(fixture.relative_to(ROOT)) if fixture.is_relative_to(ROOT) else str(fixture),
        "fixture_content_sha256": meta.get("content_sha256")
        or content_hash_capture(fixture),
        "capture_id": meta.get("capture_id"),
        "session_split": split_meta,
        "n_sessions": len(rows),
        "sessions": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _run_multiseed(
    *,
    fixture: Path,
    scores_from: str,
    out_dir: Path,
    seeds: str,
    random_draws: int,
    session_split: str = "eval",
    train_seconds: float = 0.0,
    train_corpus: str = "prepare",
) -> int:
    argv = [
        "--capture",
        str(fixture),
        "--scores-from",
        scores_from,
        "--seeds",
        seeds,
        "--out-dir",
        str(out_dir),
        "--random-draws",
        str(random_draws),
        "--session-split",
        session_split,
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
    *,
    fixture_name: str,
    allow_publish_fixture_card: bool,
    baselines_only: bool = False,
) -> tuple[str, str]:
    """
    Fixture-card / harness-smoke status only.

    `published_fixture_card` only for the frozen public_ranking_card_v1 protocol
    when model mean AUROC beats length and events on the same eval split.
    This is NOT production AUROC / general public accuracy / README hero.
    lab_public_pack and baselines-only stay not_published.
    """
    if baselines_only:
        return (
            "not_published",
            (
                "Baselines-only run (length/events from real local fixture scores). "
                "No fixture-only model aggregate — claim_status stays not_published. "
                "No AUROC hero, no README publish."
            ),
        )
    if not allow_publish_fixture_card:
        return (
            "not_published",
            (
                f"Fixture {fixture_name!r} is outside the frozen "
                f"{DEFAULT_FIXTURE_NAME} publish lane. Harness metrics may be "
                "computed from real local scores; claim_status stays "
                "not_published — no AUROC hero, no README publish."
            ),
        )
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
                "NOT production AUROC, NOT general public accuracy, NOT lab pool, "
                "NOT a README hero."
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
    fixture: Path,
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
    fixture_disp = (
        str(fixture.relative_to(ROOT)) if fixture.is_relative_to(ROOT) else str(fixture)
    )
    lines = [
        "# Public ranking card v1 — fixture report",
        "",
        f"**Card id:** `{CARD_ID}`  ",
        f"**claim_status:** `{claim_status}`  ",
        f"**Protocol:** [`{PROTOCOL}`](../../{PROTOCOL})",
        "",
        "> ## Limitations (read first)",
        ">",
        f"> - **n_eval = {n_eval}** labeled sessions on a **committed local fixture** — harness smoke, not a field study.",
        "> - **High / perfect AUROC on this toy pack ≠ general public accuracy** and ≠ production AUROC.",
        "> - Text patterns may be stylized; separation can be easy.",
        "> - **Not** the lab-pool AUROC (`docs/lab/ranking-validation.md`). **Not** CRISP `val_bpb`. **Not** a support/SLO metric.",
        "> - **Not** a README AUROC hero. Metrics below are computed from real local fixture scores.",
        "> - Train corpus (model path) = fixture train-split **normal** texts only (no CRISP / prepare shards).",
        "",
        f"**Claim gate:** {claim_reason}",
        "",
        "## Identity",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Fixture | `{fixture_disp}` |",
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
                "not a marketable production accuracy number or README hero."
            ),
            "",
        ]
    lines += [
        "## Explicit non-claims",
        "",
        "- Not general public accuracy or production AUROC.",
        "- Not a README / marketing AUROC hero.",
        "- Not the lab pool (see docs/lab/ranking-validation.md for that result).",
        "- Not CRISP / synthetic `val_bpb`.",
        "- Not a production support or incident-response SLO metric.",
        "- `prepare.evaluate_bpb` is sacred and unused by this harness path.",
        "",
        "## Lane reminders",
        "",
        "- Private lab pool metrics stay in `docs/lab/` (`not_published` lane).",
        "- `published_fixture_card` = fixture harness smoke that beat baselines — still not production / not README hero.",
        "- Per-session length/events baselines: `session_baseline_scores.json` (always `not_published`).",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_public_ranking_card",
        description=(
            "Honest public ranking-card fixture runner — scores committed "
            "local fixtures (length/events baselines + optional session BPB). "
            "Never invents AUROC heroes / README publish / val_bpb "
            "(claim_status=not_published unless frozen card harness gate passes)."
        ),
        epilog=(
            "Refuses --auroc / --publish / --readme-hero / invent flags. "
            "prepare.py is sacred — not modified. "
            "Fixtures: public_ranking_card_v1 · lab_public_pack_v0."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fixture",
        type=str,
        default=DEFAULT_FIXTURE_NAME,
        help=(
            "Committed fixture name or path "
            f"(default: {DEFAULT_FIXTURE_NAME}; also: lab_public_pack_v0)"
        ),
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
    p.add_argument(
        "--session-scores-only",
        action="store_true",
        help=(
            "Emit session_baseline_scores.json from the fixture and exit "
            "(no multiseed ranking aggregates; no invented AUROC)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    _refuse_loud_flags(raw)

    args = build_parser().parse_args(raw)

    try:
        fixture = resolve_fixture(args.fixture)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    path_err = validate_fixture(fixture)
    if path_err:
        print(path_err, file=sys.stderr)
        return 2

    fixture_name = fixture.name
    has_split = (fixture / "split.json").is_file()
    # Frozen card publish lane only for the committed public_ranking_card_v1 pack.
    allow_publish_fixture_card = (
        fixture_name == DEFAULT_FIXTURE_NAME and has_split
    )
    # Eval split when frozen split exists; otherwise score all scorable sessions.
    session_split = "eval" if has_split else "all"

    if fixture_name == DEFAULT_FIXTURE_NAME and not has_split:
        print(
            f"ERROR: missing frozen split {fixture / 'split.json'}",
            file=sys.stderr,
        )
        return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    fixture_sha = content_hash_capture(fixture)

    # Always emit honest per-session baseline scores from the real fixture.
    session_scores_path = out_root / "session_baseline_scores.json"
    session_payload = emit_session_baseline_scores(
        fixture,
        session_split=session_split,
        out_path=session_scores_path,
    )
    print(
        f"[{CARD_ID}] wrote {session_scores_path} "
        f"(n={session_payload['n_sessions']} session baselines; "
        f"claim_status={session_payload['claim_status']})"
    )

    if args.session_scores_only:
        print(
            f"[{CARD_ID}] session-scores-only — no multiseed aggregates, "
            "no invented AUROC heroes."
        )
        return 0

    length_dir = out_root / "baselines-length"
    events_dir = out_root / "baselines-events"
    model_dir = out_root / "model-fixture"

    # Model path only on frozen card fixture with split (fixture-train needs it).
    run_model = (
        args.with_model
        and not args.baselines_only
        and allow_publish_fixture_card
    )
    if args.with_model and not allow_publish_fixture_card:
        print(
            f"[{CARD_ID}] NOTE: --with-model ignored for fixture {fixture_name!r} "
            "(fixture-train / published_fixture_card lane requires "
            f"{DEFAULT_FIXTURE_NAME} + split.json). Running baselines only.",
            file=sys.stderr,
        )

    if not args.skip_run:
        print(f"[{CARD_ID}] fixture={fixture_name} fixture_sha={fixture_sha}")
        print(
            f"[{CARD_ID}] seeds={seeds} session_split={session_split} "
            f"allow_publish_fixture_card={allow_publish_fixture_card}"
        )
        rc = _run_multiseed(
            fixture=fixture,
            scores_from="length",
            out_dir=length_dir,
            seeds=args.seeds,
            random_draws=args.random_draws,
            session_split=session_split,
        )
        if rc != 0:
            return rc
        rc = _run_multiseed(
            fixture=fixture,
            scores_from="events",
            out_dir=events_dir,
            seeds=args.seeds,
            random_draws=args.random_draws,
            session_split=session_split,
        )
        if rc != 0:
            return rc

        if run_model:
            print(
                f"[{CARD_ID}] fixture-only model train_seconds={args.train_seconds} "
                "(no CRISP / prepare shards)"
            )
            rc = _run_multiseed(
                fixture=fixture,
                scores_from="model",
                out_dir=model_dir,
                seeds=args.seeds,
                random_draws=args.random_draws,
                session_split=session_split,
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

    if has_split:
        split = json.loads((fixture / "split.json").read_text(encoding="utf-8"))
        n_eval = int(split.get("n_eval") or len(split.get("eval_session_ids") or []))
    else:
        n_eval = int(session_payload["n_sessions"])

    model_agg = None
    model_agg_path = model_dir / "aggregate.json"
    if model_agg_path.is_file() and run_model:
        model_agg = _load_agg(model_agg_path)
    elif model_agg_path.is_file() and allow_publish_fixture_card and not args.baselines_only:
        # skip-run / prior model aggregate on frozen card
        model_agg = _load_agg(model_agg_path)

    claim_status, claim_reason = _decide_claim_status(
        length_agg,
        events_agg,
        model_agg,
        fixture_name=fixture_name,
        allow_publish_fixture_card=allow_publish_fixture_card,
        baselines_only=bool(args.baselines_only),
    )

    # Stamp card identity onto aggregates
    for agg, method in ((length_agg, "length"), (events_agg, "events")):
        agg["card_id"] = CARD_ID
        agg["protocol"] = PROTOCOL
        agg["claim_status"] = claim_status
        agg["claim_reason"] = claim_reason
        agg["fixture_content_sha256"] = fixture_sha
        agg["fixture_name"] = fixture_name
        agg["epsilon"] = EPS
        agg["baseline"] = method
        agg["session_split"] = session_split
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
        model_agg["fixture_name"] = fixture_name
        model_agg["session_split"] = session_split
        model_agg["n_eval"] = n_eval
        model_agg["train_corpus"] = "fixture_train_split_only"
        model_agg["epsilon_model"] = MODEL_EPS
        model_agg_path.write_text(json.dumps(model_agg, indent=2) + "\n", encoding="utf-8")

    summary_path = out_root / "CARD.md"
    _write_card_summary(
        fixture=fixture,
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
    # ε check only against committed refs for the frozen public card fixture.
    if args.check_eps:
        if not allow_publish_fixture_card:
            print(
                "ERROR: --check-eps only applies to frozen "
                f"{DEFAULT_FIXTURE_NAME} references "
                f"(got fixture={fixture_name!r})",
                file=sys.stderr,
            )
            return 2
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
        print(
            f"ε check passed (baselines ≤{EPS}"
            + (
                f", model ≤{MODEL_EPS}"
                if model_agg and ref_model.is_file()
                else ""
            )
            + ")"
        )

    # Refresh references when explicitly requested or after a fresh baseline/model
    # run on the frozen public card only (never write hero refs for lab pack).
    if allow_publish_fixture_card and (
        args.write_references or (not args.skip_run and not args.check_eps)
    ):
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
