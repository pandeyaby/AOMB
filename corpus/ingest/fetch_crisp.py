"""
Fetch Uber CRISP artifact from Zenodo (v1 bootstrap public-real corpus).

  doi:10.5281/zenodo.13956078
  File: CRISP-main.zip (~2.33 GB, md5 efc646e625270685734e8988fc5ef8ec)
  License: CC BY 4.0 — cite Zhang et al., USENIX ATC'22

Default does NOT download (too large for CI). Use --download explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

DEFAULT_OUT = os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch", "corpus-v1", "crisp"
)
ZENODO_RECORD = "13956078"
FILENAME = "CRISP-main.zip"
EXPECTED_MD5 = "efc646e625270685734e8988fc5ef8ec"
DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/{FILENAME}?download=1"
)
DOI = "10.5281/zenodo.13956078"


def _md5_file(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download(out_dir: str) -> str:
    import requests

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, FILENAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000:
        print(f"Zip already present: {dest} ({os.path.getsize(dest)} bytes)")
        return dest

    print(f"Downloading {DOWNLOAD_URL}")
    print(f"  → {dest}")
    print("  (~2.33 GB — this takes a while; not for CI)")
    with requests.get(DOWNLOAD_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest + ".partial"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    print(f"  … {os.path.getsize(tmp) / 1e9:.2f} GB", flush=True)
        os.replace(tmp, dest)
    return dest


def verify_md5(path: str) -> bool:
    print(f"Verifying md5 of {path} …")
    got = _md5_file(path)
    ok = got == EXPECTED_MD5
    print(f"  got={got} expected={EXPECTED_MD5} {'OK' if ok else 'MISMATCH'}")
    return ok


def extract(zip_path: str, out_dir: str) -> str:
    extract_dir = os.path.join(out_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    marker = os.path.join(extract_dir, ".aomb_extracted")
    if os.path.exists(marker):
        print(f"Already extracted: {extract_dir}")
        return extract_dir
    print(f"Extracting {zip_path} → {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(DOI + "\n")
    return extract_dir


def manual_instructions(out_dir: str) -> None:
    print(
        f"""
Uber CRISP — v1 bootstrap public-real corpus
  DOI:      {DOI}
  Record:   https://zenodo.org/records/{ZENODO_RECORD}
  File:     {FILENAME} (~2.33 GB)
  md5:      {EXPECTED_MD5}
  License:  CC BY 4.0 — cite Zhang et al., USENIX ATC'22

Manual:
  1. Download {FILENAME} from Zenodo into {out_dir}/
  2. unzip {FILENAME} -d {out_dir}/extracted
  3. uv run python -m corpus.ingest.build_shards \\
       --adapter crisp_zenodo \\
       --input {out_dir}/extracted \\
       --num-train-shards 8 --write-val-shard

Or: uv run python -m corpus.ingest.fetch_crisp --download
"""
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch Uber CRISP from Zenodo")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--download",
        action="store_true",
        help="Actually download ~2.33 GB CRISP-main.zip (not for CI)",
    )
    p.add_argument(
        "--extract-only",
        metavar="ZIP",
        help="Extract an already-downloaded zip",
    )
    p.add_argument("--skip-md5", action="store_true")
    args = p.parse_args(argv)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.extract_only:
        path = os.path.abspath(args.extract_only)
        if not args.skip_md5 and not verify_md5(path):
            return 1
        extract_dir = extract(path, out_dir)
        print(f"Ready: {extract_dir}")
        print(
            "Next: uv run python -m corpus.ingest.build_shards "
            f"--adapter crisp_zenodo --input {extract_dir} "
            "--num-train-shards 8 --write-val-shard"
        )
        return 0

    if not args.download:
        manual_instructions(out_dir)
        zip_path = os.path.join(out_dir, FILENAME)
        if os.path.exists(zip_path):
            print(f"Found existing zip at {zip_path}")
            if not args.skip_md5:
                verify_md5(zip_path)
            extract_dir = extract(zip_path, out_dir)
            print(f"Ready: {extract_dir}")
            return 0
        return 2

    zip_path = download(out_dir)
    if not args.skip_md5 and not verify_md5(zip_path):
        return 1
    extract_dir = extract(zip_path, out_dir)
    print(f"Done. Extracted → {extract_dir}")
    print(
        "Next: uv run python -m corpus.ingest.build_shards "
        f"--adapter crisp_zenodo --input {extract_dir} "
        "--num-train-shards 8 --write-val-shard"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
