"""Small synthetic archives for the host extraction utility; no SCOUT I/O."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("extract_scout", Path(__file__).resolve().parents[1] / "extract-scout.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scout-extraction-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "source.tar.gz"
        self.staging = self.root / "staging"

    def create(self, entries=None):
        entries = entries or [("scout/README.md", b"sample"),
                              ("scout/images/sequence_1/cam_0/image_0.jpg", b"synthetic bytes")]
        with tarfile.open(self.archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
            directory = tarfile.TarInfo("scout/")
            directory.type = tarfile.DIRTYPE
            tar.addfile(directory)
            for name, value in entries:
                if isinstance(name, tarfile.TarInfo):
                    tar.addfile(name)
                else:
                    member = tarfile.TarInfo(name)
                    member.size = len(value)
                    tar.addfile(member, io.BytesIO(value))
        return hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_regular_files_extract_and_archive_is_preserved(self):
        checksum = self.create()
        original = self.archive.read_bytes()
        reports = []
        result = extract.extract_archive(self.archive, self.staging, checksum, 0, reports.append)
        self.assertEqual(result["files_written"], 2)
        self.assertEqual(result["written_bytes"], len(b"sample") + len(b"synthetic bytes"))
        self.assertEqual((self.staging / "scout/README.md").read_bytes(), b"sample")
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertTrue(result["archive_sha256_matches_verified_download"])
        self.assertEqual(reports[-1]["compressed_progress_percent"], 100)
        self.assertEqual((self.staging / "scout/README.md").stat().st_mode & 0o777, 0o600)

    def test_unsafe_paths_are_rejected_without_outside_writes(self):
        for i, name in enumerate(("../escape", "scout/../../escape", "/tmp/scout-escape", "other/a", "scout/dir\\a")):
            with self.subTest(name=name):
                checksum = self.create([(name, b"unsafe")])
                with self.assertRaisesRegex(ValueError, "archive path"):
                    extract.extract_archive(self.archive, self.root / ("bad-" + str(i)), checksum, 0)
        self.assertFalse((self.root / "escape").exists())

    def test_symlinks_hardlinks_devices_and_sparse_members_rejected(self):
        for typ in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE):
            with self.subTest(type=typ):
                member = tarfile.TarInfo("scout/link")
                member.type, member.linkname = typ, "../../outside"
                with self.assertRaisesRegex(ValueError, "links, special"):
                    extract.member_path(member)
        sparse = tarfile.TarInfo("scout/sparse")
        sparse.sparse = [(0, 10)]
        with self.assertRaises(ValueError):
            extract.member_path(sparse)

    def test_existing_staging_and_duplicate_members_are_not_overwritten(self):
        checksum = self.create()
        self.staging.mkdir()
        marker = self.staging / "keep"
        marker.write_text("existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            extract.extract_archive(self.archive, self.staging, checksum, 0)
        self.assertEqual(marker.read_text(), "existing")
        checksum = self.create([("scout/file", b"first"), ("scout/file", b"second")])
        other = self.root / "duplicate"
        with self.assertRaises(FileExistsError):
            extract.extract_archive(self.archive, other, checksum, 0)
        self.assertEqual((other / "scout/file").read_bytes(), b"first")

    def test_disk_reserve_prevents_file_writes(self):
        checksum = self.create()
        with mock.patch.object(extract.shutil, "disk_usage", return_value=types.SimpleNamespace(free=5)):
            with self.assertRaisesRegex(RuntimeError, "reserve"):
                extract.extract_archive(self.archive, self.staging, checksum, 10)
        self.assertEqual(list(self.staging.rglob("*")), [])

    def test_next_file_must_fit_above_reserve(self):
        checksum = self.create([("scout/large", b"123456789012345")])
        with mock.patch.object(extract.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20)):
            with self.assertRaisesRegex(RuntimeError, "next file"):
                extract.extract_archive(self.archive, self.staging, checksum, 10)
        self.assertFalse((self.staging / "scout/large").exists())

    def test_hash_mismatch_cannot_be_reported_as_success(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            extract.extract_archive(self.archive, self.staging, "0"*64, 0)
        self.assertTrue(self.staging.exists())

    def test_archive_modified_during_extraction_is_rejected(self):
        checksum = self.create()
        changed = []
        def progress(data):
            if not changed:
                st = self.archive.stat()
                os.utime(self.archive, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
                changed.append(True)
        with self.assertRaisesRegex(ValueError, "changed during"):
            extract.extract_archive(self.archive, self.staging, checksum, 0, progress)


if __name__ == "__main__":
    unittest.main()
