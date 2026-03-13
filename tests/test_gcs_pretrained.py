import contextlib
import datetime as dt
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path


def _load_gcs_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "es_finetuning" / "gcs.py"
    spec = importlib.util.spec_from_file_location("gcs_module", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gcs_module = _load_gcs_module()
GCSClient = gcs_module.GCSClient


class _FakeBlob:
    def __init__(self, name, size, updated):
        self.name = name
        self.size = size
        self.updated = updated


class GcsPretrainedTests(unittest.TestCase):
    def test_resolve_pretrained_checkpoint_prefers_local_file(self):
        client = GCSClient(bucket_name="test-bucket", project="test-project")
        client.list_blobs = lambda prefix: self.fail("should not query GCS when local cache exists")
        client.download_file = lambda gcs_path, local_path: self.fail("should not download")

        with tempfile.TemporaryDirectory() as tmpdir:
            older = Path(tmpdir) / "older.pt"
            newer = Path(tmpdir) / "newer.pt"
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                resolved = client.resolve_pretrained_checkpoint(tmpdir)

        self.assertEqual(resolved, str(newer))
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Using local NAMM checkpoint:", stderr.getvalue())

    def test_download_latest_pretrained_uses_cached_file_and_logs_to_stderr(self):
        client = GCSClient(bucket_name="test-bucket", project="test-project")
        blob = _FakeBlob(
            "NAMM_checkpoints/pretrained/model.pt",
            4,
            dt.datetime(2026, 3, 13, 1, 0, 0),
        )
        client.list_blobs = lambda prefix: [blob]
        client.download_file = lambda gcs_path, local_path: self.fail("should not download")

        with tempfile.TemporaryDirectory() as tmpdir:
            cached_path = Path(tmpdir) / "model.pt"
            cached_path.write_bytes(b"abcd")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                resolved = client.download_latest_pretrained(tmpdir)

        self.assertEqual(resolved, str(cached_path))
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("NAMM checkpoint cached:", stderr.getvalue())

    def test_download_latest_pretrained_downloads_file_and_logs_to_stderr(self):
        client = GCSClient(bucket_name="test-bucket", project="test-project")
        blob = _FakeBlob(
            "NAMM_checkpoints/pretrained/model.pt",
            6,
            dt.datetime(2026, 3, 13, 2, 0, 0),
        )
        client.list_blobs = lambda prefix: [blob]

        def _download(gcs_path, local_path):
            self.assertEqual(gcs_path, blob.name)
            Path(local_path).write_bytes(b"abcdef")

        client.download_file = _download

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                resolved = client.download_latest_pretrained(tmpdir)

            self.assertEqual(Path(resolved).read_bytes(), b"abcdef")

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(
            "Downloading gs://test-bucket/NAMM_checkpoints/pretrained/model.pt",
            stderr.getvalue(),
        )
        self.assertIn("Saved:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
