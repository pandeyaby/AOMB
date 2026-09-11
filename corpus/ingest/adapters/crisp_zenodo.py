"""
Public-real adapter: Uber CRISP production Jaeger traces (v1 bootstrap).

Source: Zenodo 10.5281/zenodo.13956078 — CRISP-main.zip (~2.33 GB)
License: CC BY 4.0
~100k sanitized production Jaeger traces (multi-service). Session = traceID.

Do NOT mix with Tale of Errors traces (different sanitization mapping).
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any, Iterator

from corpus.ingest.adapters.base import (
    SourceAdapter,
    SourceBundle,
    TimeWindow,
)
from corpus.ingest.jaeger import load_jaeger_json

SOURCE_ID = "uber-crisp-zenodo-13956078"
DOI = "10.5281/zenodo.13956078"
LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CITATION = (
    "Zhang et al., CRISP: Critical Path Analysis of Large-Scale Microservice "
    "Architectures, USENIX ATC'22. Artifact: https://doi.org/10.5281/zenodo.13956078 "
    "(CC BY 4.0). Cite the paper when using these traces."
)
ZENODO_RECORD = "13956078"
ZENODO_FILE = "CRISP-main.zip"
MD5 = "efc646e625270685734e8988fc5ef8ec"


def _looks_like_jaeger(doc: Any) -> bool:
    if isinstance(doc, dict):
        if "data" in doc and isinstance(doc["data"], list):
            return True
        if "spans" in doc and ("traceID" in doc or "traceId" in doc or "processes" in doc):
            return True
    return False


def _discover_json_files(root: str) -> list[str]:
    """Find Jaeger JSON files under an extracted CRISP tree or flat traces dir."""
    patterns = [
        os.path.join(root, "**", "*.json"),
        os.path.join(root, "*.json"),
    ]
    # Prefer common artifact layouts if present
    preferred_globs = [
        os.path.join(root, "**", "traces", "**", "*.json"),
        os.path.join(root, "**", "trace", "**", "*.json"),
        os.path.join(root, "CRISP-main", "**", "*.json"),
    ]
    files: list[str] = []
    for pat in preferred_globs + patterns:
        files.extend(glob(pat, recursive=True))
    # Dedupe, skip obvious non-trace JSON (package.json, etc.)
    skip_names = {
        "package.json",
        "package-lock.json",
        "composer.json",
        "tsconfig.json",
        ".eslintrc.json",
    }
    out: list[str] = []
    seen: set[str] = set()
    for fp in sorted(set(files)):
        base = os.path.basename(fp)
        if base in skip_names:
            continue
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
    return out


class CrispZenodoAdapter(SourceAdapter):
    name = "crisp_zenodo"

    def load(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        max_files = int(kwargs.get("max_files", 0) or 0)
        max_spans = int(kwargs.get("max_spans", 0) or 0)

        if not os.path.isdir(input_path):
            raise FileNotFoundError(
                f"CRISP input not found: {input_path}. "
                "Run: python -m corpus.ingest.fetch_crisp  "
                "(downloads CRISP-main.zip from Zenodo — ~2.33 GB) "
                "or point --input at an extracted tree. See docs/corpus-v1.md."
            )

        files = _discover_json_files(input_path)
        if not files:
            raise FileNotFoundError(
                f"No JSON files under {input_path}. Expected Jaeger trace JSON "
                f"from {ZENODO_FILE} (doi:{DOI})."
            )

        spans = []
        used_files = 0
        skipped = 0
        for fp in files:
            if max_files and used_files >= max_files:
                break
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped += 1
                continue
            if not _looks_like_jaeger(doc):
                skipped += 1
                continue
            batch = load_jaeger_json(doc)
            if not batch:
                skipped += 1
                continue
            spans.extend(batch)
            used_files += 1
            if max_spans and len(spans) >= max_spans:
                spans = spans[:max_spans]
                break

        if not spans:
            raise RuntimeError(
                f"Found {len(files)} JSON files under {input_path} but none parsed "
                "as Jaeger traces. Check extraction layout (docs/corpus-v1.md)."
            )

        times = [s.start_time for s in spans if s.start_time]
        windows = []
        if times:
            windows.append(
                TimeWindow(
                    label="normal",
                    start=min(times),
                    end=max(times),
                    notes=(
                        "CRISP artifact is production traces with randomized start "
                        "offsets; no incident window labels in the dump. Use lab "
                        "captures or Tale of Errors / AIOps for fault-labeled eval."
                    ),
                )
            )

        yield SourceBundle(
            source_id=SOURCE_ID,
            source_kind="public_real",
            license=LICENSE,
            license_url=LICENSE_URL,
            citation=CITATION,
            spans=spans,
            logs=[],
            windows=windows,
            capture_id="crisp-zenodo-13956078",
            extra_provenance={
                "doi": DOI,
                "zenodo_record": ZENODO_RECORD,
                "zenodo_file": ZENODO_FILE,
                "md5": MD5,
                "json_files_used": used_files,
                "json_files_skipped": skipped,
                "input_path": os.path.abspath(input_path),
                "paper": "USENIX ATC'22 CRISP",
                "note": "Do not mix with Tale of Errors sanitization mapping.",
            },
        )
