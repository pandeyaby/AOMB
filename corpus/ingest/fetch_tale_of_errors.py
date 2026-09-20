"""
Fetch Uber Tale of Errors artifacts from Zenodo (flagship-scale path).

  Part 1: doi:10.5281/zenodo.13947828
  Part 2: doi:10.5281/zenodo.13952897
  License: CC BY 4.0 — cite Lee, Zhang, Parwal, Chabbi (SIGMETRICS 2025)

Split pieces reassemble to trace{1,2}-sanitized.tar.zst. Each decompressed
archive needs ~300–500 GB — full download must NOT run in CI.

Default prints instructions only. Use --list-only to enumerate Zenodo files,
or --download FILE… to pull selected pieces (HTTP resume supported).

Do not mix sanitization mapping with CRISP (Zenodo 13956078).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

DEFAULT_OUT = os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch", "corpus-v1", "tale_of_errors"
)
DOI_PART1 = "10.5281/zenodo.13947828"
DOI_PART2 = "10.5281/zenodo.13952897"
RECORDS: tuple[tuple[str, str], ...] = (
    ("13947828", DOI_PART1),
    ("13952897", DOI_PART2),
)
ZENODO_API = "https://zenodo.org/api/records"
DECOMPRESSED_DISK_NOTE = "300–500 GB per archive after zstd decompress"
COMPRESSED_SCALE_NOTE = (
    "~35 GB compressed (part 1) + ~37 GB compressed (part 2); "
    "full reassembly + decompress needs hundreds of GB free"
)

# Env vars that indicate CI / non-interactive runners.
_CI_ENV_KEYS = (
    "CI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "BUILDKITE",
    "TF_BUILD",  # Azure Pipelines
    "JENKINS_URL",
)


def in_ci(environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    for key in _CI_ENV_KEYS:
        val = env.get(key, "").strip().lower()
        if not val:
            continue
        if key == "JENKINS_URL":
            return True
        if val in {"1", "true", "yes", "on"}:
            return True
    return False


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit, div in (("KB", 1024), ("MB", 1024**2), ("GB", 1024**3), ("TB", 1024**4)):
        if n < div * 1024 or unit == "TB":
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def list_record_files(record_id: str) -> list[dict[str, Any]]:
    """Return Zenodo file metadata for one record (key, size, checksum, links)."""
    import requests

    url = f"{ZENODO_API}/{record_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("files") or []
    out: list[dict[str, Any]] = []
    for f in files:
        key = f.get("key") or ""
        links = f.get("links") or {}
        out.append(
            {
                "record_id": record_id,
                "key": key,
                "size": int(f.get("size") or 0),
                "checksum": f.get("checksum") or "",
                "download_url": links.get("download")
                or f"https://zenodo.org/records/{record_id}/files/{key}?download=1",
            }
        )
    out.sort(key=lambda x: x["key"])
    return out


def list_all_files() -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for record_id, _doi in RECORDS:
        files.extend(list_record_files(record_id))
    return files


def print_file_listing(files: list[dict[str, Any]]) -> None:
    by_record: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        by_record.setdefault(f["record_id"], []).append(f)
    for record_id, doi in RECORDS:
        group = by_record.get(record_id, [])
        total = sum(x["size"] for x in group)
        print(f"\nZenodo {record_id} ({doi}) — {len(group)} files, {_format_bytes(total)}")
        print(f"  https://zenodo.org/records/{record_id}")
        for f in group:
            print(
                f"  {f['key']:28s} {_format_bytes(f['size']):>12s}  {f['checksum']}"
            )


EXIT_REFUSED_FLAG = 1

# Invent / publish / cuda refusals (mirror score_cli / tale baseline spirit).
REFUSED_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--cuda",
        "--gpu",
        "--invent-metrics",
        "--invent-auroc",
        "--invent-val-bpb",
        "--claim-auroc",
        "--invent",
    }
)


def refuse_loud_flags(argv: list[str]) -> None:
    """Fail loud on invent / publish / cuda flags before argparse."""
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_FLAGS:
            msg = (
                f"ERROR: Refusing '{key}'.\n"
                "  Tale Zenodo fetch downloads corpus pieces only.\n"
                "  Never invents AUROC / published ranking / CUDA path.\n"
                "  Use --list-only or --download KEY…"
            )
            print(msg, file=sys.stderr)
            raise SystemExit(EXIT_REFUSED_FLAG)


def partial_path(dest: str) -> str:
    """In-progress resume target for a final destination path."""
    return dest + ".partial"


def assert_size_ok(path: str, expected: int, *, label: str) -> None:
    """Loud size check — never claim success on mismatch."""
    got = os.path.getsize(path) if os.path.exists(path) else -1
    if expected > 0 and got != expected:
        raise RuntimeError(
            f"Size mismatch for {label}: got {got}, expected {expected}. "
            "Leaving .partial in place; not promoting corrupt final."
        )


def verify_checksum(path: str, checksum: str, *, label: str) -> None:
    """Verify Zenodo-style checksum (e.g. md5:hex). Empty checksum → skip."""
    checksum = (checksum or "").strip()
    if not checksum:
        return
    if ":" not in checksum:
        raise RuntimeError(f"Unrecognized checksum for {label}: {checksum!r}")
    algo, expected_hex = checksum.split(":", 1)
    algo = algo.lower().strip()
    expected_hex = expected_hex.strip().lower()
    if algo != "md5":
        # Zenodo currently publishes md5; refuse silent skip of unknown algos.
        raise RuntimeError(
            f"Unsupported checksum algo for {label}: {algo!r} (expected md5)"
        )
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected_hex:
        raise RuntimeError(
            f"Checksum mismatch for {label}: got md5:{got}, expected {checksum}. "
            "Not promoting to final; .partial retained for retry."
        )


def promote_partial(
    tmp: str,
    dest: str,
    *,
    expected: int,
    checksum: str = "",
    label: str = "",
) -> str:
    """Size (+ optional checksum) check, then atomic replace into final path."""
    name = label or os.path.basename(dest)
    if not os.path.exists(tmp):
        raise RuntimeError(f"Missing .partial for {name}: {tmp}")
    assert_size_ok(tmp, expected, label=name)
    verify_checksum(tmp, checksum, label=name)
    os.replace(tmp, dest)
    return dest


def prepare_resume_state(
    dest: str,
    expected: int,
    *,
    key: str,
) -> tuple[str, int, str, dict[str, str]]:
    """Return (tmp, existing_bytes, open_mode, headers) for a download attempt.

    Rules:
      - Final dest with exact expected size → caller should skip (handled upstream).
      - Corrupt final (wrong size) is removed; never left as silent success.
      - Oversized .partial is deleted and restarted.
      - Exact-sized .partial is promoted by caller (not here).
      - Undersized .partial becomes Range resume target.
    """
    tmp = partial_path(dest)

    if os.path.exists(dest):
        dest_size = os.path.getsize(dest)
        if expected > 0 and dest_size == expected:
            return tmp, -1, "wb", {}  # sentinel: already complete
        # Corrupt / incomplete final — do not trust it.
        print(
            f"WARNING: removing corrupt/incomplete final for {key}: "
            f"{dest} ({_format_bytes(dest_size)}"
            + (f", expected {_format_bytes(expected)})" if expected else ")"),
            file=sys.stderr,
        )
        os.remove(dest)

    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expected > 0 and existing > expected:
        print(
            f"WARNING: oversized .partial for {key} "
            f"({_format_bytes(existing)} > {_format_bytes(expected)}); "
            "deleting and restarting from byte 0",
            file=sys.stderr,
        )
        os.remove(tmp)
        existing = 0

    headers: dict[str, str] = {}
    mode = "wb"
    if existing > 0 and (not expected or existing < expected):
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"Resuming {key} from {_format_bytes(existing)} (.partial)")
    return tmp, existing, mode, headers


def download_file(
    meta: dict[str, Any],
    out_dir: str,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Download one Zenodo file with HTTP Range resume into out_dir.

    Writes only to ``K.partial`` until size (+ optional checksum) checks pass,
    then atomically renames to final ``K``. Never leaves a silent corrupt final.
    On failure, ``.partial`` is retained for retry; success is not claimed.
    """
    import requests

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, meta["key"])
    expected = int(meta["size"] or 0)
    checksum = str(meta.get("checksum") or "")

    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        # Final already matches size; optional checksum verify.
        try:
            verify_checksum(dest, checksum, label=meta["key"])
        except RuntimeError:
            print(
                f"WARNING: final checksum failed for {meta['key']}; "
                "removing and re-downloading",
                file=sys.stderr,
            )
            os.remove(dest)
        else:
            print(f"Already complete: {dest} ({_format_bytes(expected)})")
            return dest

    tmp, existing, mode, headers = prepare_resume_state(
        dest, expected, key=meta["key"]
    )
    if existing == -1:
        print(f"Already complete: {dest} ({_format_bytes(expected)})")
        return dest

    # Exact-sized leftover .partial → promote after checks (no network).
    if expected > 0 and existing == expected and os.path.exists(tmp):
        promote_partial(
            tmp, dest, expected=expected, checksum=checksum, label=meta["key"]
        )
        print(f"Already complete (from .partial): {dest}")
        return dest

    url = meta["download_url"]
    print(f"Downloading {meta['key']} ({_format_bytes(expected)})")
    print(f"  ← {url}")
    print(f"  → {dest}  (via {tmp})")
    try:
        with requests.get(url, stream=True, timeout=120, headers=headers) as r:
            # Some hosts ignore Range; restart from scratch if 200 after Range request.
            if existing and r.status_code == 200:
                mode = "wb"
                existing = 0
                print("  server ignored Range — restarting from byte 0")
            r.raise_for_status()
            with open(tmp, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        size_now = os.path.getsize(tmp)
                        if expected:
                            pct = 100.0 * size_now / expected
                            print(
                                f"  … {_format_bytes(size_now)} / {_format_bytes(expected)} "
                                f"({pct:.1f}%)",
                                flush=True,
                            )
                        else:
                            print(f"  … {_format_bytes(size_now)}", flush=True)
        promote_partial(
            tmp, dest, expected=expected, checksum=checksum, label=meta["key"]
        )
    except Exception:
        # Keep .partial for resume; never promote on failure.
        if os.path.exists(dest) and expected and os.path.getsize(dest) != expected:
            # Should not happen (we only replace after checks), but be defensive.
            print(
                f"ERROR: refusing corrupt final for {meta['key']}; removing {dest}",
                file=sys.stderr,
            )
            os.remove(dest)
        raise
    return dest


def refuse_full_download_in_ci(
    *,
    force_ci_override: bool = False,
    environ: dict[str, str] | None = None,
) -> None:
    """Exit if CI would pull the multi-GB / hundreds-of-GB corpus."""
    if force_ci_override:
        return
    if not in_ci(environ):
        return
    print(
        "ERROR: Tale of Errors full / bulk download refused in CI.\n"
        f"  Detected CI via one of: {', '.join(_CI_ENV_KEYS)}\n"
        f"  Disk need: {COMPRESSED_SCALE_NOTE}; decompress {DECOMPRESSED_DISK_NOTE}.\n"
        "  Use a local machine with --list-only / --download <file>, or set\n"
        "  --allow-ci-download only for intentional non-default runners.\n"
        "  Smoke tests use corpus/fixtures/tale_of_errors_sample/ — no Zenodo pull.",
        file=sys.stderr,
    )
    raise SystemExit(3)


def manual_instructions(out_dir: str) -> None:
    print(
        f"""
Uber Tale of Errors — flagship-scale public-real corpus (optional)
  Part 1:   {DOI_PART1}  https://zenodo.org/records/13947828
  Part 2:   {DOI_PART2}  https://zenodo.org/records/13952897
  License:  CC BY 4.0 — cite Lee, Zhang, Parwal, Chabbi (SIGMETRICS 2025)
  Disk:     {COMPRESSED_SCALE_NOTE}
            Decompress: {DECOMPRESSED_DISK_NOTE}
  CI:       Full download must NOT run in CI (this script refuses it).

  Do NOT mix sanitization mapping with CRISP (Zenodo 13956078).

List files (API, no download):
  uv run python -m corpus.ingest.fetch_tale_of_errors --list-only

Download one split piece (resume-safe), e.g. first part-1 shard:
  uv run python -m corpus.ingest.fetch_tale_of_errors \\
    --download trace1_aa --out {out_dir}

Assemble + decompress (after downloading all trace1_* / trace2_* pieces):
  cd {out_dir}
  cat trace1_* > trace1-sanitized.tar.zst
  cat trace2_* > trace2-sanitized.tar.zst
  zstd -d trace1-sanitized.tar.zst   # needs {DECOMPRESSED_DISK_NOTE}
  zstd -d trace2-sanitized.tar.zst
  tar -xf trace1-sanitized.tar
  tar -xf trace2-sanitized.tar
  # Arrange Jaeger JSON under a single tree, then:

  uv run python -m corpus.ingest.build_shards \\
    --adapter tale_of_errors \\
    --input /path/to/assembled/jaeger/tree \\
    --max-spans N \\
    --num-train-shards 8 --write-val-shard

Smoke / CI (no Zenodo):
  uv run python -m corpus.ingest.build_shards \\
    --adapter tale_of_errors \\
    --input corpus/fixtures/tale_of_errors_sample \\
    --max-spans 100 --num-train-shards 1 --write-val-shard
"""
    )


def _index_by_key(files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for f in files:
        key = f["key"]
        if key in index:
            raise SystemExit(f"Duplicate Zenodo key across records: {key}")
        index[key] = f
    return index


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    refuse_loud_flags(raw)
    p = argparse.ArgumentParser(
        description=(
            "Fetch Uber Tale of Errors from Zenodo "
            "(list / selective download; refuse full pull in CI)"
        )
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="Download directory")
    p.add_argument(
        "--list-only",
        action="store_true",
        help="List Zenodo files via API and exit (no download)",
    )
    p.add_argument(
        "--download",
        nargs="+",
        metavar="FILE",
        help="Download selected Zenodo file key(s) with resume (not full corpus)",
    )
    p.add_argument(
        "--download-all",
        action="store_true",
        help=(
            "Download every file from both Zenodo records "
            f"({COMPRESSED_SCALE_NOTE}). Refused in CI."
        ),
    )
    p.add_argument(
        "--allow-ci-download",
        action="store_true",
        help="Override CI guard (dangerous; never use in default CI jobs)",
    )
    args = p.parse_args(raw)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.list_only:
        files = list_all_files()
        print_file_listing(files)
        print(
            f"\nDisk note: {COMPRESSED_SCALE_NOTE}. "
            f"Decompress: {DECOMPRESSED_DISK_NOTE}."
        )
        print("Do not mix sanitization mapping with CRISP (Zenodo 13956078).")
        return 0

    if args.download_all:
        refuse_full_download_in_ci(force_ci_override=args.allow_ci_download)
        files = list_all_files()
        print(
            f"WARNING: downloading ALL {len(files)} files "
            f"({_format_bytes(sum(f['size'] for f in files))}). "
            f"Decompress later needs {DECOMPRESSED_DISK_NOTE}."
        )
        for meta in files:
            download_file(meta, out_dir)
        print(f"Done. Files in {out_dir}")
        print("Next: cat trace1_* > trace1-sanitized.tar.zst  (and same for trace2_*)")
        return 0

    if args.download:
        # Selective download: still refuse if CI and user somehow asks for everything
        # via an explicit huge selection — but primarily block --download-all.
        # Also block any download attempt in CI unless override (accidental CI spend).
        refuse_full_download_in_ci(force_ci_override=args.allow_ci_download)
        files = list_all_files()
        index = _index_by_key(files)
        missing = [k for k in args.download if k not in index]
        if missing:
            print(f"Unknown file key(s): {missing}", file=sys.stderr)
            print("Use --list-only to see available keys.", file=sys.stderr)
            return 1
        for key in args.download:
            download_file(index[key], out_dir)
        print(f"Done. Files in {out_dir}")
        return 0

    manual_instructions(out_dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
