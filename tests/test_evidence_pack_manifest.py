"""
test_evidence_pack_manifest.py

Unit tests for batch_runs.evidence_pack.manifest module.

Uses stdlib unittest only - no third-party dependencies.

Run with:
    python3 -m unittest tests.test_evidence_pack_manifest -v
    
Or via discover:
    python3 -m unittest discover -s tests -p 'test_*.py' -q
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from batch_runs.evidence_pack import (
    ManifestError,
    compute_sha256,
    count_manifest_files,
    verify_manifest,
    write_manifest,
)


class TestComputeSha256(unittest.TestCase):
    """Tests for compute_sha256 function."""

    def test_computes_correct_hash(self):
        """Test that compute_sha256 returns correct hash for known content."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"hello world")
            temp_path = Path(f.name)

        try:
            result = compute_sha256(temp_path)
            # SHA-256 of "hello world"
            expected = hashlib.sha256(b"hello world").hexdigest()
            self.assertEqual(result, expected)
        finally:
            temp_path.unlink()

    def test_empty_file(self):
        """Test hash of empty file."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = compute_sha256(temp_path)
            expected = hashlib.sha256(b"").hexdigest()
            self.assertEqual(result, expected)
        finally:
            temp_path.unlink()

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with self.assertRaises(FileNotFoundError):
            compute_sha256(Path("/nonexistent/path/file.txt"))


class TestWriteManifest(unittest.TestCase):
    """Tests for write_manifest function."""

    def setUp(self):
        """Create a temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.evidence_dir = Path(self.temp_dir) / "evidence_pack"
        self.evidence_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_manifest_json(self):
        """Test that write_manifest creates manifest.json."""
        # Create a test file
        test_file = self.evidence_dir / "test.txt"
        test_file.write_text("test content")

        manifest_path = write_manifest(self.evidence_dir, {"test_key": "test_value"})

        self.assertTrue(manifest_path.exists())
        self.assertEqual(manifest_path.name, "manifest.json")

    def test_manifest_contains_required_fields(self):
        """Test that manifest contains all required fields."""
        test_file = self.evidence_dir / "test.txt"
        test_file.write_text("test content")

        write_manifest(self.evidence_dir, {"seed": 12345})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("created_utc", manifest)
        self.assertIn("metadata", manifest)
        self.assertIn("files", manifest)
        self.assertIn("python_version", manifest["metadata"])
        self.assertEqual(manifest["metadata"]["seed"], 12345)

    def test_manifest_files_sorted_by_path(self):
        """Test that files in manifest are sorted by path."""
        # Create multiple files
        (self.evidence_dir / "z_file.txt").write_text("z")
        (self.evidence_dir / "a_file.txt").write_text("a")
        (self.evidence_dir / "m_file.txt").write_text("m")

        write_manifest(self.evidence_dir, {})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        paths = [f["path"] for f in manifest["files"]]
        self.assertEqual(paths, sorted(paths))

    def test_manifest_excludes_itself(self):
        """Test that manifest.json is not included in file list."""
        test_file = self.evidence_dir / "test.txt"
        test_file.write_text("test content")

        write_manifest(self.evidence_dir, {})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        paths = [f["path"] for f in manifest["files"]]
        self.assertNotIn("manifest.json", paths)

    def test_manifest_contains_correct_hashes(self):
        """Test that manifest contains correct SHA-256 hashes."""
        test_file = self.evidence_dir / "test.txt"
        content = b"test content for hashing"
        test_file.write_bytes(content)

        write_manifest(self.evidence_dir, {})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        file_info = manifest["files"][0]
        expected_hash = hashlib.sha256(content).hexdigest()
        self.assertEqual(file_info["sha256"], expected_hash)

    def test_nested_directories(self):
        """Test that nested directories are handled correctly."""
        subdir = self.evidence_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")
        (self.evidence_dir / "root.txt").write_text("root")

        write_manifest(self.evidence_dir, {})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        paths = [f["path"] for f in manifest["files"]]
        self.assertIn("root.txt", paths)
        self.assertIn("subdir/nested.txt", paths)

    def test_raises_on_nonexistent_directory(self):
        """Test that FileNotFoundError is raised for nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            write_manifest(Path("/nonexistent/path"), {})


class TestVerifyManifest(unittest.TestCase):
    """Tests for verify_manifest function."""

    def setUp(self):
        """Create a temporary directory with valid evidence pack."""
        self.temp_dir = tempfile.mkdtemp()
        self.evidence_dir = Path(self.temp_dir) / "evidence_pack"
        self.evidence_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verify_passes_for_valid_manifest(self):
        """Test that verify passes when manifest matches files."""
        # Create test files
        (self.evidence_dir / "file1.txt").write_text("content1")
        (self.evidence_dir / "file2.txt").write_text("content2")

        # Write manifest
        write_manifest(self.evidence_dir, {})

        # Verify should not raise
        verify_manifest(self.evidence_dir)

    def test_verify_fails_for_missing_manifest(self):
        """Test that verify fails when manifest.json is missing."""
        (self.evidence_dir / "file.txt").write_text("content")

        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)

        self.assertIn("Manifest not found", str(ctx.exception))

    def test_verify_fails_for_missing_file(self):
        """Test that verify fails when a listed file is missing."""
        (self.evidence_dir / "file.txt").write_text("content")
        write_manifest(self.evidence_dir, {})

        # Delete the file
        (self.evidence_dir / "file.txt").unlink()

        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)

        self.assertIn("Missing file", str(ctx.exception))
        self.assertIn("file.txt", str(ctx.exception))

    def test_verify_fails_for_hash_mismatch(self):
        """Test that verify fails when file content changes."""
        test_file = self.evidence_dir / "file.txt"
        test_file.write_text("original content")
        write_manifest(self.evidence_dir, {})

        # Modify the file
        test_file.write_text("modified content")

        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)

        self.assertIn("mismatch", str(ctx.exception).lower())
        self.assertIn("file.txt", str(ctx.exception))

    def test_verify_fails_for_extra_file(self):
        """Test that verify fails when extra files exist (default)."""
        (self.evidence_dir / "file.txt").write_text("content")
        write_manifest(self.evidence_dir, {})

        # Add extra file after manifest
        (self.evidence_dir / "extra.txt").write_text("extra")

        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)

        self.assertIn("Extra files", str(ctx.exception))
        self.assertIn("extra.txt", str(ctx.exception))

    def test_verify_passes_for_extra_file_when_allowed(self):
        """Test that verify passes when extra files exist and allow_extra=True."""
        (self.evidence_dir / "file.txt").write_text("content")
        write_manifest(self.evidence_dir, {})

        # Add extra file after manifest
        (self.evidence_dir / "extra.txt").write_text("extra")

        # Should not raise when allow_extra=True
        verify_manifest(self.evidence_dir, allow_extra=True)

    def test_verify_actionable_error_message(self):
        """Test that error messages include expected and actual values."""
        test_file = self.evidence_dir / "file.txt"
        test_file.write_text("original")
        write_manifest(self.evidence_dir, {})

        # Get expected hash before modification
        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        expected_hash = manifest["files"][0]["sha256"]

        # Modify the file
        test_file.write_text("modified")

        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)

        error_msg = str(ctx.exception)
        # Should include expected hash
        self.assertIn(expected_hash, error_msg)
        # Should include "Actual SHA-256"
        self.assertIn("Actual SHA-256", error_msg)


class TestCountManifestFiles(unittest.TestCase):
    """Tests for count_manifest_files function."""

    def setUp(self):
        """Create a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.evidence_dir = Path(self.temp_dir) / "evidence_pack"
        self.evidence_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_counts_files_correctly(self):
        """Test that file count matches number of files in manifest."""
        (self.evidence_dir / "file1.txt").write_text("1")
        (self.evidence_dir / "file2.txt").write_text("2")
        (self.evidence_dir / "file3.txt").write_text("3")

        write_manifest(self.evidence_dir, {})

        count = count_manifest_files(self.evidence_dir / "manifest.json")
        self.assertEqual(count, 3)

    def test_empty_evidence_pack(self):
        """Test count for evidence pack with no files."""
        write_manifest(self.evidence_dir, {})

        count = count_manifest_files(self.evidence_dir / "manifest.json")
        self.assertEqual(count, 0)


class TestManifestDeterminism(unittest.TestCase):
    """Tests to ensure manifest generation is deterministic."""

    def setUp(self):
        """Create a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.evidence_dir = Path(self.temp_dir) / "evidence_pack"
        self.evidence_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_order_is_deterministic(self):
        """Test that files are always in the same order regardless of creation order."""
        # Create files in random order
        for name in ["z.txt", "a.txt", "m.txt", "b.txt"]:
            (self.evidence_dir / name).write_text(name)

        write_manifest(self.evidence_dir, {})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        paths = [f["path"] for f in manifest["files"]]
        expected = ["a.txt", "b.txt", "m.txt", "z.txt"]
        self.assertEqual(paths, expected)

    def test_posix_paths_on_all_platforms(self):
        """Test that paths use forward slashes (POSIX-style)."""
        subdir = self.evidence_dir / "sub" / "dir"
        subdir.mkdir(parents=True)
        (subdir / "file.txt").write_text("content")

        write_manifest(self.evidence_dir, {})

        manifest_path = self.evidence_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        path = manifest["files"][0]["path"]
        self.assertEqual(path, "sub/dir/file.txt")
        self.assertNotIn("\\", path)


class TestIntegration(unittest.TestCase):
    """Integration tests for write_manifest + verify_manifest workflow."""

    def setUp(self):
        """Create a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.evidence_dir = Path(self.temp_dir) / "evidence_pack"
        self.evidence_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_then_verify_succeeds(self):
        """Test that write followed by verify succeeds."""
        # Create realistic evidence pack structure
        (self.evidence_dir / "phase_ab_manifest.json").write_text('{"test": true}')
        (self.evidence_dir / "confidence_report.json").write_text('{"decision": "HOLD"}')
        (self.evidence_dir / "confidence_report.md").write_text("# Report\n")

        # Write and verify
        write_manifest(self.evidence_dir, {"seed": 42, "cli_args": ["smoke"]})
        verify_manifest(self.evidence_dir)  # Should not raise

    def test_round_trip_with_nested_structure(self):
        """Test write+verify with nested directory structure."""
        # Create nested structure
        reports = self.evidence_dir / "reports"
        reports.mkdir()
        (reports / "summary.json").write_text('{}')
        (reports / "details.md").write_text("# Details")
        (self.evidence_dir / "manifest_data.json").write_text('{"key": "value"}')

        write_manifest(self.evidence_dir, {})
        verify_manifest(self.evidence_dir)  # Should not raise


if __name__ == "__main__":
    unittest.main()

