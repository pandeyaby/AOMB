"""Safe autoresearch cache lane quarantine / restore (CRISP ↔ Tale).

Renames the active ``data`` + ``tokenizer`` directories under
``~/.cache/autoresearch/`` to stamped quarantine dirs, and restores them
honestly without silent clobber.

Never invents AUROC / val_bpb. Never touches prepare.py.
No MPS / Zenodo / network.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1
EXIT_USAGE = 2
EXIT_STATE = 3

ACTIVE_DATA = "data"
ACTIVE_TOKENIZER = "tokenizer"

# Keep in sync with eval.stranger_path / product_mac invent set (subset + synonyms).
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

_LANE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_STAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def default_cache_root(home: Path | None = None) -> Path:
    base = Path(home) if home is not None else Path.home()
    return base / ".cache" / "autoresearch"


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_lane(lane: str) -> str:
    lane = (lane or "").strip()
    if not _LANE_RE.match(lane):
        raise ValueError(
            f"invalid lane {lane!r}: use crisp|tale|label "
            "(letters/digits/_/- , start with a letter, max 64)"
        )
    return lane


def quarantine_names(lane: str, stamp: str) -> tuple[str, str]:
    lane = sanitize_lane(lane)
    if not _STAMP_RE.match(stamp):
        raise ValueError(f"invalid stamp {stamp!r}: want YYYYMMDD_HHMMSS")
    return f"data_{lane}_{stamp}", f"tokenizer_{lane}_{stamp}"


@dataclass(frozen=True)
class QuarantineResult:
    cache_root: Path
    lane: str
    stamp: str
    data_src: Path
    tokenizer_src: Path
    data_dst: Path
    tokenizer_dst: Path


def quarantine(
    *,
    cache_root: Path,
    lane: str,
    stamp: str | None = None,
    dry_run: bool = False,
) -> QuarantineResult:
    """Rename active data+tokenizer → stamped quarantine dirs.

    Fails if either active path is missing, or destination already exists.
    """
    lane = sanitize_lane(lane)
    stamp = stamp or _now_stamp()
    data_name, tok_name = quarantine_names(lane, stamp)
    data_src = cache_root / ACTIVE_DATA
    tok_src = cache_root / ACTIVE_TOKENIZER
    data_dst = cache_root / data_name
    tok_dst = cache_root / tok_name

    if not data_src.is_dir():
        raise FileNotFoundError(f"active data missing: {data_src}")
    if not tok_src.is_dir():
        raise FileNotFoundError(f"active tokenizer missing: {tok_src}")
    if data_dst.exists():
        raise FileExistsError(f"quarantine target exists: {data_dst}")
    if tok_dst.exists():
        raise FileExistsError(f"quarantine target exists: {tok_dst}")

    if not dry_run:
        cache_root.mkdir(parents=True, exist_ok=True)
        data_src.rename(data_dst)
        tok_src.rename(tok_dst)

    return QuarantineResult(
        cache_root=cache_root,
        lane=lane,
        stamp=stamp,
        data_src=data_src,
        tokenizer_src=tok_src,
        data_dst=data_dst,
        tokenizer_dst=tok_dst,
    )


def _tokenizer_sibling_for_data(data_dir: Path) -> Path:
    name = data_dir.name
    if not name.startswith("data_"):
        raise ValueError(
            f"--from must be a quarantined data_* directory, got {name!r}"
        )
    return data_dir.parent / ("tokenizer_" + name[len("data_") :])


def _latest_data_for_lane(cache_root: Path, lane: str) -> Path | None:
    lane = sanitize_lane(lane)
    prefix = f"data_{lane}_"
    candidates = [
        p
        for p in cache_root.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def restore(
    *,
    cache_root: Path,
    from_dir: Path | None = None,
    lane: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> QuarantineResult:
    """Restore quarantined data+tokenizer → active names.

    Refuses to overwrite active ``data`` / ``tokenizer`` unless ``force``.
    """
    if (from_dir is None) == (lane is None):
        raise ValueError("restore requires exactly one of --from or --lane")

    if from_dir is not None:
        data_src = Path(from_dir).expanduser().resolve()
        if not data_src.is_dir():
            raise FileNotFoundError(f"--from not a directory: {data_src}")
        tok_src = _tokenizer_sibling_for_data(data_src)
        rest = data_src.name[len("data_") :]
        parts = rest.rsplit("_", 2)
        if len(parts) >= 3 and _STAMP_RE.match(f"{parts[-2]}_{parts[-1]}"):
            stamp = f"{parts[-2]}_{parts[-1]}"
            lane_s = "_".join(parts[:-2]) or "restored"
        else:
            stamp = _now_stamp()
            lane_s = "restored"
    else:
        assert lane is not None
        lane_s = sanitize_lane(lane)
        data_src = _latest_data_for_lane(cache_root, lane_s)
        if data_src is None:
            raise FileNotFoundError(
                f"no quarantined data_{lane_s}_* under {cache_root}"
            )
        tok_src = _tokenizer_sibling_for_data(data_src)
        rest = data_src.name[len("data_") :]
        stamp = (
            rest[len(lane_s) + 1 :]
            if rest.startswith(lane_s + "_")
            else _now_stamp()
        )

    if not tok_src.is_dir():
        raise FileNotFoundError(f"matching tokenizer quarantine missing: {tok_src}")

    data_dst = cache_root / ACTIVE_DATA
    tok_dst = cache_root / ACTIVE_TOKENIZER

    if data_dst.exists() or tok_dst.exists():
        if not force:
            raise FileExistsError(
                "active data/tokenizer present; refuse silent clobber "
                "(pass --force to overwrite)"
            )
        if not dry_run:
            if data_dst.exists():
                shutil.rmtree(data_dst)
            if tok_dst.exists():
                shutil.rmtree(tok_dst)

    if not dry_run:
        cache_root.mkdir(parents=True, exist_ok=True)
        data_src.rename(data_dst)
        tok_src.rename(tok_dst)

    return QuarantineResult(
        cache_root=cache_root,
        lane=lane_s,
        stamp=stamp,
        data_src=data_src,
        tokenizer_src=tok_src,
        data_dst=data_dst,
        tokenizer_dst=tok_dst,
    )


@dataclass(frozen=True)
class StatusReport:
    cache_root: Path
    active_data: bool
    active_tokenizer: bool
    quarantines: tuple[tuple[str, str], ...]  # (data_name, tokenizer_name-or-"")


def status(cache_root: Path) -> StatusReport:
    active_data = (cache_root / ACTIVE_DATA).is_dir()
    active_tok = (cache_root / ACTIVE_TOKENIZER).is_dir()
    pairs: list[tuple[str, str]] = []
    if cache_root.is_dir():
        data_dirs = sorted(
            p.name
            for p in cache_root.iterdir()
            if p.is_dir() and p.name.startswith("data_") and p.name != ACTIVE_DATA
        )
        for dname in data_dirs:
            tname = "tokenizer_" + dname[len("data_") :]
            tpath = cache_root / tname
            pairs.append((dname, tname if tpath.is_dir() else ""))
    return StatusReport(
        cache_root=cache_root,
        active_data=active_data,
        active_tokenizer=active_tok,
        quarantines=tuple(pairs),
    )


def refuse_loud_flags(argv: Iterable[str]) -> str | None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            return key
    return None


def _print_status(rep: StatusReport) -> None:
    print(f"cache_root={rep.cache_root}")
    print(
        f"active: data={'yes' if rep.active_data else 'no'} "
        f"tokenizer={'yes' if rep.active_tokenizer else 'no'}"
    )
    if not rep.quarantines:
        print("quarantined: (none)")
        return
    print("quarantined:")
    for dname, tname in rep.quarantines:
        tok = tname if tname else "(tokenizer missing)"
        print(f"  {dname}  <->  {tok}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cache_lane",
        description=(
            "Quarantine / restore ~/.cache/autoresearch data+tokenizer lanes "
            "(CRISP <-> Tale) without silent clobber. No AUROC / val_bpb / CUDA."
        ),
        epilog=(
            "Examples:\n"
            "  python -m corpus.ingest.cache_lane status\n"
            "  python -m corpus.ingest.cache_lane quarantine --lane tale\n"
            "  python -m corpus.ingest.cache_lane restore --lane crisp\n"
            "  python -m corpus.ingest.cache_lane restore "
            "--from ~/.cache/autoresearch/data_tale_20260919_232229 --force\n"
            "  ./scripts/cache_lane.sh quarantine --lane crisp"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override cache root (default: ~/.cache/autoresearch)",
    )
    p.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Override HOME for default cache root (tests)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quarantine", help="Rename active data+tokenizer to stamped dirs")
    q.add_argument(
        "--lane",
        required=True,
        help="Lane label: crisp | tale | custom label",
    )
    q.add_argument(
        "--stamp",
        default=None,
        help="Optional YYYYMMDD_HHMMSS (default: now)",
    )
    q.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("restore", help="Restore quarantined lane to active data+tokenizer")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--from",
        dest="from_dir",
        type=Path,
        help="Quarantined data_* directory to restore",
    )
    g.add_argument(
        "--lane",
        help="Restore latest data_<lane>_* (+ matching tokenizer_)",
    )
    r.add_argument(
        "--force",
        action="store_true",
        help="Required to overwrite existing active data/tokenizer (non-interactive)",
    )
    r.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Show active vs quarantined lanes (no secrets)")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = refuse_loud_flags(argv)
    if bad is not None:
        print(
            f"ERROR: refusing invent / publish / CUDA flag {bad}. "
            "cache_lane only moves directories; never invents metrics.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_FLAG

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if args.cache_root is not None:
        cache_root = Path(args.cache_root).expanduser().resolve()
    else:
        cache_root = default_cache_root(
            Path(args.home).expanduser() if args.home else None
        )

    try:
        if args.cmd == "status":
            _print_status(status(cache_root))
            return EXIT_OK
        if args.cmd == "quarantine":
            result = quarantine(
                cache_root=cache_root,
                lane=args.lane,
                stamp=args.stamp,
                dry_run=args.dry_run,
            )
            verb = "would quarantine" if args.dry_run else "quarantined"
            print(f"{verb}: lane={result.lane} stamp={result.stamp}")
            print(f"  {result.data_src.name} -> {result.data_dst.name}")
            print(f"  {result.tokenizer_src.name} -> {result.tokenizer_dst.name}")
            return EXIT_OK
        if args.cmd == "restore":
            result = restore(
                cache_root=cache_root,
                from_dir=args.from_dir,
                lane=args.lane,
                force=args.force,
                dry_run=args.dry_run,
            )
            verb = "would restore" if args.dry_run else "restored"
            print(f"{verb}: lane={result.lane} stamp={result.stamp}")
            print(f"  {result.data_src.name} -> {result.data_dst.name}")
            print(f"  {result.tokenizer_src.name} -> {result.tokenizer_dst.name}")
            return EXIT_OK
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_STATE

    parser.error(f"unknown command {args.cmd!r}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
