"""Tests: Tale Zenodo fetch .partial resume hardening (no network).

prepare.py untouched. No invented AUROC / val_bpb. CUDA gate skipped.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from corpus.ingest.fetch_tale_of_errors import (
    EXIT_REFUSED_FLAG,
    download_file,
    main,
    partial_path,
    prepare_resume_state,
    promote_partial,
    refuse_loud_flags,
    verify_checksum,
)


def _meta(key: str, data: bytes, *, checksum: bool = True) -> dict:
    md5 = hashlib.md5(data).hexdigest()
    return {
        "key": key,
        "size": len(data),
        "checksum": f"md5:{md5}" if checksum else "",
        "download_url": f"https://example.test/{key}",
        "record_id": "0",
    }


class TestPartialHelpers(unittest.TestCase):
    def test_partial_path_suffix(self):
        self.assertEqual(partial_path("/tmp/trace1_am"), "/tmp/trace1_am.partial")

    def test_promote_exact_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            part = partial_path(dest)
            payload = b"hello-zenodo-bytes"
            Path(part).write_bytes(payload)
            out = promote_partial(
                part,
                dest,
                expected=len(payload),
                checksum=f"md5:{hashlib.md5(payload).hexdigest()}",
                label="trace1_am",
            )
            self.assertEqual(out, dest)
            self.assertTrue(os.path.exists(dest))
            self.assertFalse(os.path.exists(part))
            self.assertEqual(Path(dest).read_bytes(), payload)

    def test_promote_size_mismatch_keeps_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            part = partial_path(dest)
            Path(part).write_bytes(b"short")
            with self.assertRaises(RuntimeError) as ctx:
                promote_partial(part, dest, expected=100, label="trace1_am")
            self.assertIn("Size mismatch", str(ctx.exception))
            self.assertTrue(os.path.exists(part))
            self.assertFalse(os.path.exists(dest))

    def test_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f")
            Path(path).write_bytes(b"abc")
            with self.assertRaises(RuntimeError) as ctx:
                verify_checksum(path, "md5:" + ("0" * 32), label="f")
            self.assertIn("Checksum mismatch", str(ctx.exception))

    def test_oversized_partial_deleted_for_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            part = partial_path(dest)
            Path(part).write_bytes(b"x" * 50)
            out_tmp, existing, mode, headers = prepare_resume_state(
                dest, expected=10, key="trace1_am"
            )
            self.assertEqual(out_tmp, part)
            self.assertEqual(existing, 0)
            self.assertEqual(mode, "wb")
            self.assertEqual(headers, {})
            self.assertFalse(os.path.exists(part))

    def test_undersized_partial_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            part = partial_path(dest)
            Path(part).write_bytes(b"0123")
            _tmp, existing, mode, headers = prepare_resume_state(
                dest, expected=10, key="trace1_am"
            )
            self.assertEqual(existing, 4)
            self.assertEqual(mode, "ab")
            self.assertEqual(headers.get("Range"), "bytes=4-")

    def test_corrupt_final_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            Path(dest).write_bytes(b"nope")
            _tmp, existing, mode, headers = prepare_resume_state(
                dest, expected=10, key="trace1_am"
            )
            self.assertFalse(os.path.exists(dest))
            self.assertEqual(existing, 0)
            self.assertEqual(mode, "wb")


class _FakeResponse:
    def __init__(self, payload: bytes, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=0):
        yield self._payload


class TestDownloadFileMocked(unittest.TestCase):
    def test_fresh_download_promotes_partial(self):
        payload = b"full-file-bytes-here"
        meta = _meta("trace1_am", payload)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "requests.get", return_value=_FakeResponse(payload, status_code=200)
            ):
                dest = download_file(meta, tmp)
            self.assertEqual(Path(dest).read_bytes(), payload)
            self.assertFalse(os.path.exists(partial_path(dest)))

    def test_resume_from_leftover_partial(self):
        payload = b"ABCDEFGHIJ"
        meta = _meta("trace1_am", payload)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            part = partial_path(dest)
            Path(part).write_bytes(payload[:4])  # leftover partial
            # Server returns remaining bytes with 206
            with mock.patch(
                "requests.get",
                return_value=_FakeResponse(payload[4:], status_code=206),
            ) as get:
                out = download_file(meta, tmp)
                self.assertIn("Range", get.call_args.kwargs.get("headers") or get.call_args[1].get("headers", {}))
            self.assertEqual(Path(out).read_bytes(), payload)
            self.assertFalse(os.path.exists(part))

    def test_exact_partial_promotes_without_network(self):
        payload = b"already-complete-partial"
        meta = _meta("trace1_am", payload)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "trace1_am")
            Path(partial_path(dest)).write_bytes(payload)
            with mock.patch("requests.get") as get:
                out = download_file(meta, tmp)
                get.assert_not_called()
            self.assertEqual(Path(out).read_bytes(), payload)

    def test_size_mismatch_keeps_partial_no_final(self):
        payload = b"too-short"
        meta = _meta("trace1_am", payload)
        meta["size"] = 100  # lie about size
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "requests.get", return_value=_FakeResponse(payload, status_code=200)
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    download_file(meta, tmp)
            self.assertIn("Size mismatch", str(ctx.exception))
            dest = os.path.join(tmp, "trace1_am")
            self.assertFalse(os.path.exists(dest))
            self.assertTrue(os.path.exists(partial_path(dest)))


class TestFetchRefusals(unittest.TestCase):
    def test_auroc_refused(self):
        with self.assertRaises(SystemExit) as ctx:
            refuse_loud_flags(["--download", "x", "--auroc"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_publish_cuda_refused_via_main(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--publish"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        with self.assertRaises(SystemExit) as ctx:
            main(["--cuda", "--list-only"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)


if __name__ == "__main__":
    unittest.main()
