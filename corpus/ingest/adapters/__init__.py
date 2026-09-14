"""Adapters for corpus source loaders."""

from corpus.ingest.adapters.byo import ByoAdapter
from corpus.ingest.adapters.crisp_zenodo import CrispZenodoAdapter
from corpus.ingest.adapters.lab_capture import LabCaptureAdapter
from corpus.ingest.adapters.tale_of_errors import TaleOfErrorsAdapter

__all__ = [
    "ByoAdapter",
    "CrispZenodoAdapter",
    "LabCaptureAdapter",
    "TaleOfErrorsAdapter",
]
