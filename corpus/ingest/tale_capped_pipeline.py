"""
Orchestrate stream-capped Tale extract → shards → optional prepare / train / score.

Designed for Mac MPS product path; CI uses fixture or synthetic ``.tar.zst`` and
stops before prepare/train (or uses ``--score-dry-run``).

Does **not** invent ``val_bpb`` / AUROC. ``prepare.py`` is sacred — invoked only,
never edited. Lab ``claim_status`` stays ``not_published``.

Usage::

  uv run python -m corpus.ingest.tale_capped_pipeline --help
  uv run python -m corpus.ingest.tale_capped_pipeline \\
    --fixture --max-spans 50 --data-dir /tmp/aomb-tale-pipe --score-dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FIXTURE = ROOT / "corpus" / "fixtures" / "tale_of_errors_sample"
DEFAULT_DATA_DIR = Path(
    os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "data")
)

_REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--val-bpb",
        "--invent-val-bpb",
        "--claim-val-bpb",
        "--invent-metrics",
    }
)
_REFUSED_DECOMPRESS_FLAGS = frozenset(
    {
        "--full-decompress",
        "--decompress-all",
        "--uncapped",
        "--download-all",
    }
)


@dataclass
class PipelineResult:
    """Factual run summary — never a published ranking / AUROC claim."""

    claim_status: str = "not_published"
    stages: list[str] = field(default_factory=list)
    extract: Optional[dict[str, Any]] = None
    shards: Optional[dict[str, Any]] = None
    prepare: Optional[dict[str, Any]] = None
    train: Optional[dict[str, Any]] = None
    score: Optional[dict[str, Any]] = None
    jaeger_tree: Optional[str] = None
    data_dir: Optional[str] = None
    max_spans: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _refuse_loud_flags(argv: list[str]) -> None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in _REFUSED_METRIC_FLAGS:
            raise SystemExit(
                f"ERROR: Refusing '{key}'.\n"
                "  This pipeline is the public-real *train/score* lane on a\n"
                "  stream-capped Tale subset only.\n"
                "  Tale dumps have no AOMB incident labels → no AUROC.\n"
                "  Never invents val_bpb / ranking accuracy.\n"
                "  Lab claim_status stays not_published."
            )
        if key in _REFUSED_DECOMPRESS_FLAGS:
            raise SystemExit(
                f"ERROR: Refusing '{key}'.\n"
                "  Full Tale decompress is OUT OF SCOPE (300–500 GB/archive).\n"
                "  Pass --max-spans (and optional extract --max-files/--max-bytes).\n"
                "  Use corpus.ingest.tale_stream_extract for capped stream extract."
            )


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _looks_like_jaeger_tree(path: Path) -> bool:
    if not path.is_dir():
        return False
    for pattern in ("**/*.json",):
        for candidate in path.glob(pattern):
            name = candidate.name.lower()
            if name in {
                "package.json",
                "package-lock.json",
                "provenance.json",
                "extract_provenance.json",
            }:
                continue
            return True
    return False


def run_extract(
    extract_input: str,
    extract_out: str,
    *,
    max_spans: int = 0,
    max_files: int = 0,
    max_bytes: int = 0,
    prefix: str = "trace1_",
    concat_out: Optional[str] = None,
) -> dict[str, Any]:
    from corpus.ingest.tale_stream_extract import extract_from_input

    if not (max_spans or max_files or max_bytes):
        raise SystemExit(
            "ERROR: Extract stage refuses uncapped run.\n"
            "  Pass at least one of: --max-spans / --extract-max-files / "
            "--extract-max-bytes."
        )
    src = _resolve_path(extract_input)
    if not src.exists():
        raise SystemExit(
            f"ERROR: Extract input not found: {src}\n"
            "  Pass --extract-input PATH to a .tar.zst / trace*_ pieces dir,\n"
            "  or use --fixture / --jaeger-tree for CI without Zenodo."
        )
    stats = extract_from_input(
        str(src),
        extract_out,
        prefix=prefix,
        concat_out=concat_out,
        max_spans=max_spans,
        max_files=max_files,
        max_bytes=max_bytes,
    )
    return {
        "input": str(src),
        "out": extract_out,
        "files_written": stats.files_written,
        "spans_written": stats.spans_written,
        "bytes_written": stats.bytes_written,
        "stopped_reason": stats.stopped_reason,
        "members_seen": stats.members_seen,
    }


def run_build_shards(
    jaeger_tree: str,
    *,
    max_spans: int,
    num_train_shards: int = 1,
    data_dir: str,
    write_val_shard: bool = True,
) -> dict[str, Any]:
    from corpus.ingest.build_shards import build

    if max_spans <= 0:
        raise SystemExit(
            "ERROR: --max-spans N is required (positive integer).\n"
            "  No silent uncapped shard build on this pipeline."
        )
    tree = _resolve_path(jaeger_tree)
    if not tree.exists():
        raise SystemExit(
            f"ERROR: Jaeger tree not found: {tree}\n"
            "  Missing input — refuse rather than invent corpus."
        )
    if not _looks_like_jaeger_tree(tree):
        raise SystemExit(
            f"ERROR: No Jaeger JSON under: {tree}\n"
            "  Point --jaeger-tree / --fixture at a traces/*.json tree,\n"
            "  or run extract first with --extract-input."
        )
    os.makedirs(data_dir, exist_ok=True)
    meta = build(
        "tale_of_errors",
        str(tree),
        num_train_shards=num_train_shards,
        write_val_shard=write_val_shard,
        data_dir=data_dir,
        max_spans=max_spans,
    )
    return {
        "adapter": "tale_of_errors",
        "input": str(tree),
        "max_spans": max_spans,
        "num_train_shards": num_train_shards,
        "data_dir": data_dir,
        "build_meta": meta,
    }


def run_prepare(num_shards: int, data_dir: str) -> dict[str, Any]:
    """Invoke sacred prepare.py — never edit it. Darwin + Metal only."""
    if platform.system() != "Darwin":
        raise SystemExit(
            f"ERROR: prepare.py requires macOS + Metal (this host: {platform.system()}).\n"
            "  Shard + score-dry-run stages are enough on Linux / CI.\n"
            "  On a Mac: pass --prepare after shards land in the default data dir."
        )
    default = str(DEFAULT_DATA_DIR)
    if os.path.abspath(data_dir) != os.path.abspath(default):
        raise SystemExit(
            f"ERROR: Refusing --prepare with non-default --data-dir ({data_dir}).\n"
            f"  prepare.py expects {default}.\n"
            "  Re-run with --data-dir set to the default cache, or skip --prepare."
        )
    cmd = [sys.executable, str(ROOT / "prepare.py"), "--num-shards", str(num_shards)]
    # Prefer uv when available (caller shell usually wraps); keep module self-contained.
    if _have_uv():
        cmd = ["uv", "run", "python", str(ROOT / "prepare.py"), "--num-shards", str(num_shards)]
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"ERROR: prepare.py failed (exit {proc.returncode})")
    return {"invoked": True, "num_shards": num_shards, "sacred": True}


def run_train_smoke(train_seconds: float) -> dict[str, Any]:
    """Bounded train.py smoke (Mac MPS). Does not invent val_bpb numbers."""
    if train_seconds <= 0:
        raise SystemExit("ERROR: --train-seconds must be > 0 when --train is set.")
    if platform.system() != "Darwin":
        raise SystemExit(
            f"ERROR: Short train requires Darwin + MPS (this host: {platform.system()}).\n"
            "  On Linux / CI use --score-dry-run after shards instead.\n"
            "  This pipeline never fabricates a val_bpb number."
        )
    # Bounded wall-clock via alarm — same spirit as product_mac_smoke.sh
    code = f"""
import signal, sys
secs = float({train_seconds!r})
def _alarm(s, f):
    print("TRAIN_SMOKE_ALARM: stopping after", secs, "s (no invented val_bpb)", flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGALRM, _alarm)
signal.setitimer(signal.ITIMER_REAL, secs)
sys.argv = ["train.py"]
import runpy
runpy.run_path({str(ROOT / "train.py")!r}, run_name="__main__")
"""
    cmd = [sys.executable, "-c", code]
    if _have_uv():
        cmd = ["uv", "run", "python", "-c", code]
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return {
        "invoked": True,
        "train_seconds": train_seconds,
        "exit_code": proc.returncode,
        "note": "Cite only val_bpb printed by train.py — never invent one here.",
    }


def run_score(
    score_input: str,
    *,
    dry_run: bool = True,
    out: Optional[str] = None,
    max_sessions: int = 0,
    train_seconds: float = 0.0,
) -> dict[str, Any]:
    from eval.score_cli import main as score_main

    argv = ["--input", score_input]
    if dry_run:
        argv.append("--dry-run")
    elif train_seconds > 0:
        argv.extend(["--train-seconds", str(train_seconds)])
    if max_sessions:
        argv.extend(["--max-sessions", str(max_sessions)])
    if out:
        argv.extend(["--out", out])
    argv.append("--json")
    rc = score_main(argv)
    if rc != 0:
        raise SystemExit(f"ERROR: score_cli failed (exit {rc})")
    report: dict[str, Any] = {"exit_code": rc, "dry_run": dry_run, "input": score_input}
    if out and Path(out).is_file():
        loaded = json.loads(Path(out).read_text(encoding="utf-8"))
        report["claim_status"] = loaded.get("claim_status", "not_published")
        report["n_sessions"] = len(loaded.get("sessions") or [])
        report["train_meta"] = loaded.get("train_meta")
        if report["claim_status"] != "not_published" and dry_run:
            raise SystemExit(
                "ERROR: unexpected claim_status from dry-run score "
                f"({report['claim_status']!r}); expected not_published."
            )
    else:
        report["claim_status"] = "not_published"
    return report


def _have_uv() -> bool:
    from shutil import which

    return which("uv") is not None


def run_pipeline(
    *,
    max_spans: int,
    fixture: bool = False,
    jaeger_tree: Optional[str] = None,
    extract_input: Optional[str] = None,
    extract_out: Optional[str] = None,
    extract_max_files: int = 0,
    extract_max_bytes: int = 0,
    extract_prefix: str = "trace1_",
    concat_out: Optional[str] = None,
    num_train_shards: int = 1,
    data_dir: Optional[str] = None,
    do_prepare: bool = False,
    do_train: bool = False,
    train_seconds: float = 60.0,
    do_score: bool = False,
    score_dry_run: bool = False,
    score_input: Optional[str] = None,
    score_out: Optional[str] = None,
    score_max_sessions: int = 0,
    score_train_seconds: float = 0.0,
) -> PipelineResult:
    if max_spans <= 0:
        raise SystemExit(
            "ERROR: --max-spans N is required (positive integer; no uncapped path)."
        )

    data = str(_resolve_path(data_dir) if data_dir else Path("/tmp/aomb-tale-capped-pipeline"))
    result = PipelineResult(
        claim_status="not_published",
        max_spans=max_spans,
        data_dir=data,
        notes=[
            "Train lane only — factual val_bpb when you train on Mac; no AUROC.",
            "claim_status=not_published",
        ],
    )

    tree: Optional[Path] = None

    if fixture:
        tree = DEFAULT_FIXTURE
        if not tree.exists():
            raise SystemExit(f"ERROR: In-repo fixture missing: {tree}")
        result.stages.append("fixture")
        result.notes.append("Using in-repo fixture (not a public-real baseline number).")
    elif jaeger_tree:
        tree = _resolve_path(jaeger_tree)
        if not tree.exists():
            raise SystemExit(f"ERROR: Jaeger tree not found: {tree}")
        result.stages.append("jaeger_tree")
    elif extract_input:
        out_dir = extract_out or str(Path(data) / "extracted")
        # Prefer extract-stage caps; fall back to shard max_spans so one flag works.
        ex_spans = max_spans
        ex_files = extract_max_files
        ex_bytes = extract_max_bytes
        extract_meta = run_extract(
            extract_input,
            out_dir,
            max_spans=ex_spans,
            max_files=ex_files,
            max_bytes=ex_bytes,
            prefix=extract_prefix,
            concat_out=concat_out,
        )
        result.extract = extract_meta
        result.stages.append("extract")
        tree = Path(out_dir)
        if extract_meta.get("stopped_reason") not in {
            "max-spans",
            "max-files",
            "max-bytes",
            "exhausted",
        }:
            result.notes.append(
                f"extract stopped_reason={extract_meta.get('stopped_reason')!r}"
            )
    else:
        raise SystemExit(
            "ERROR: Specify one of --fixture, --jaeger-tree PATH, or --extract-input PATH.\n"
            "  Missing input — refuse rather than invent corpus / metrics."
        )

    assert tree is not None
    result.jaeger_tree = str(tree)

    shard_meta = run_build_shards(
        str(tree),
        max_spans=max_spans,
        num_train_shards=num_train_shards,
        data_dir=data,
    )
    result.shards = shard_meta
    result.stages.append("build_shards")

    if do_prepare:
        result.prepare = run_prepare(num_train_shards, data)
        result.stages.append("prepare")

    if do_train:
        result.train = run_train_smoke(train_seconds)
        result.stages.append("train")

    if do_score or score_dry_run:
        s_in = score_input or str(tree)
        dry = score_dry_run or not (score_train_seconds > 0)
        # On Linux CI, force dry-run unless caller explicitly asked for model score
        # with train seconds (which will fail loud on non-Darwin via score_cli deps).
        if score_dry_run:
            dry = True
        result.score = run_score(
            s_in,
            dry_run=dry,
            out=score_out,
            max_sessions=score_max_sessions,
            train_seconds=0.0 if dry else score_train_seconds,
        )
        result.stages.append("score_dry_run" if dry else "score")
        if result.score.get("claim_status"):
            result.claim_status = result.score["claim_status"]

    result.claim_status = result.claim_status or "not_published"
    if result.claim_status != "not_published":
        # Safety: this orchestration must not flip publish state.
        raise SystemExit(
            f"ERROR: Refusing claim_status={result.claim_status!r}. "
            "Pipeline keeps not_published."
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m corpus.ingest.tale_capped_pipeline",
        description=(
            "Stream-capped Tale extract → tale_of_errors shards → optional "
            "prepare / short train / score. Mac MPS product path; CI uses "
            "fixture or synthetic archive + --score-dry-run."
        ),
        epilog=(
            "Honesty: no AUROC, no invented val_bpb, no full decompress, "
            "no uncapped shards. prepare.py is sacred. "
            "claim_status=not_published."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("input (pick one)")
    src.add_argument(
        "--fixture",
        action="store_true",
        help=f"Use in-repo fixture ({DEFAULT_FIXTURE})",
    )
    src.add_argument(
        "--jaeger-tree",
        default=None,
        help="Existing local Jaeger JSON tree (e.g. prior capped extract out/)",
    )
    src.add_argument(
        "--extract-input",
        default=None,
        help="Path to .tar.zst or directory of trace*_ pieces (stream-capped extract)",
    )

    p.add_argument(
        "--extract-out",
        default=None,
        help="Extract output dir (default: <data-dir>/extracted)",
    )
    p.add_argument("--extract-prefix", default="trace1_")
    p.add_argument("--concat-out", default=None, help="Optional concatenated .tar.zst path")
    p.add_argument(
        "--extract-max-files",
        type=int,
        default=0,
        help="Extract-stage file cap (in addition to --max-spans)",
    )
    p.add_argument(
        "--extract-max-bytes",
        type=int,
        default=0,
        help="Extract-stage byte cap",
    )

    p.add_argument(
        "--max-spans",
        type=int,
        required=True,
        help="Required span cap for extract (if used) and build_shards (no uncapped)",
    )
    p.add_argument("--num-train-shards", type=int, default=1)
    p.add_argument(
        "--data-dir",
        default="/tmp/aomb-tale-capped-pipeline",
        help="Shard output directory",
    )

    p.add_argument(
        "--prepare",
        action="store_true",
        help="Invoke sacred prepare.py (Darwin + Metal; default data dir only)",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Bounded train.py smoke (Darwin + MPS)",
    )
    p.add_argument(
        "--train-seconds",
        type=float,
        default=60.0,
        help="Wall-clock bound for --train (default: 60)",
    )

    p.add_argument(
        "--score",
        action="store_true",
        help="Run eval.score_cli on the Jaeger tree / --score-input",
    )
    p.add_argument(
        "--score-dry-run",
        action="store_true",
        help="Score load-only (no torch) — CI-safe; claim_status=not_published",
    )
    p.add_argument(
        "--score-input",
        default=None,
        help="Override score input (default: Jaeger tree used for shards)",
    )
    p.add_argument("--score-out", default=None, help="Optional JSON score report path")
    p.add_argument("--score-max-sessions", type=int, default=0)
    p.add_argument(
        "--score-train-seconds",
        type=float,
        default=0.0,
        help="If >0 with --score (not dry-run), short train-then-score",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print PipelineResult JSON to stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    _refuse_loud_flags(raw)

    args = build_parser().parse_args(raw)
    sources = sum(
        bool(x) for x in (args.fixture, args.jaeger_tree, args.extract_input)
    )
    if sources != 1:
        print(
            "ERROR: Specify exactly one of --fixture, --jaeger-tree, --extract-input.",
            file=sys.stderr,
        )
        return 2

    result = run_pipeline(
        max_spans=args.max_spans,
        fixture=args.fixture,
        jaeger_tree=args.jaeger_tree,
        extract_input=args.extract_input,
        extract_out=args.extract_out,
        extract_max_files=args.extract_max_files,
        extract_max_bytes=args.extract_max_bytes,
        extract_prefix=args.extract_prefix,
        concat_out=args.concat_out,
        num_train_shards=args.num_train_shards,
        data_dir=args.data_dir,
        do_prepare=args.prepare,
        do_train=args.train,
        train_seconds=args.train_seconds,
        do_score=args.score,
        score_dry_run=args.score_dry_run,
        score_input=args.score_input,
        score_out=args.score_out,
        score_max_sessions=args.score_max_sessions,
        score_train_seconds=args.score_train_seconds,
    )

    print(
        f"OK: tale capped pipeline stages={result.stages} "
        f"claim_status={result.claim_status} data_dir={result.data_dir}"
    )
    if result.extract:
        print(
            f"  extract: stopped={result.extract.get('stopped_reason')} "
            f"files={result.extract.get('files_written')} "
            f"spans={result.extract.get('spans_written')}"
        )
    if result.shards:
        print(
            f"  shards: adapter=tale_of_errors max_spans={result.max_spans} "
            f"→ {result.data_dir}"
        )
    if result.score:
        print(
            f"  score: dry_run={result.score.get('dry_run')} "
            f"claim_status={result.score.get('claim_status')}"
        )
    print(
        "Honesty: no invented val_bpb / AUROC. Full decompress OUT OF SCOPE. "
        "prepare.py sacred."
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
