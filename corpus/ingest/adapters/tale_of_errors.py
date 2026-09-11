"""
Flagship-scale path (optional / later): Uber Tale of Errors.

~1.4M sanitized production Jaeger traces, CC BY 4.0.
DOIs: 10.5281/zenodo.13947828 + 10.5281/zenodo.13952897
Decompressed size: ~300–500 GB per archive — do NOT require in CI.

This adapter documents the path and can load a *local sample* of Jaeger JSON
the same way as CRISP. Full multi-part download is out of scope for v1 bootstrap.
Do not mix sanitization mappings with CRISP (Zenodo note).
"""

from __future__ import annotations

from typing import Any, Iterator

from corpus.ingest.adapters.crisp_zenodo import CrispZenodoAdapter
from corpus.ingest.adapters.base import SourceAdapter, SourceBundle

SOURCE_ID = "uber-tale-of-errors"
DOI_PART1 = "10.5281/zenodo.13947828"
DOI_PART2 = "10.5281/zenodo.13952897"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Lee, Zhang, Parwal, Chabbi — The Tale of Errors in Microservices "
    "(SIGMETRICS 2025). Artifacts: "
    f"https://doi.org/{DOI_PART1} and https://doi.org/{DOI_PART2} (CC BY 4.0)."
)


class TaleOfErrorsAdapter(SourceAdapter):
    """Reuse Jaeger discovery; override provenance for Tale of Errors."""

    name = "tale_of_errors"

    def load(self, input_path: str, **kwargs: Any) -> Iterator[SourceBundle]:
        # Same on-disk Jaeger JSON layout as CRISP once assembled locally.
        for bundle in CrispZenodoAdapter().load(input_path, **kwargs):
            bundle.source_id = SOURCE_ID
            bundle.license = LICENSE
            bundle.license_url = "https://creativecommons.org/licenses/by/4.0/"
            bundle.citation = CITATION
            bundle.capture_id = "tale-of-errors"
            bundle.extra_provenance = {
                **bundle.extra_provenance,
                "doi_part1": DOI_PART1,
                "doi_part2": DOI_PART2,
                "scale": "~1.4M traces; 300–500GB decompressed per archive",
                "ci": "full download not required / not run in CI",
                "note": "Do not mix sanitization mapping with CRISP artifact.",
            }
            yield bundle
