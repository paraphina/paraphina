"""
Unit tests for the Docs Integrity Gate.

Tests the reference validation and canonical spec hash locking functionality
WITHOUT depending on the real repo docs content. All tests use temp dirs
and sample strings.
"""

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

# Import from tools package (no sys.path manipulation)
from tools.check_docs_integrity import (
    CANONICAL_MARKER,
    KNOWN_EXTENSIONS,
    RefError,
    check_canonical_hash,
    compute_canonical_hash,
    is_path_like,
    parse_implemented_refs,
    run_checks,
    validate_implemented_refs,
    validate_ref,
)


class TestIsPathLike(unittest.TestCase):
    """Test the is_path_like function."""
    
    def test_valid_rust_path(self):
        """Rust paths with / should be recognized."""
        self.assertTrue(is_path_like("paraphina/src/state.rs"))
        self.assertTrue(is_path_like("src/lib.rs"))
    
    def test_valid_python_path(self):
        """Python paths should be recognized."""
        self.assertTrue(is_path_like("batch_runs/metrics.py"))
        self.assertTrue(is_path_like("tools/check_docs_integrity.py"))
    
    def test_valid_config_paths(self):
        """Config file paths should be recognized."""
        self.assertTrue(is_path_like("config/settings.yml"))
        self.assertTrue(is_path_like("config/settings.yaml"))
        self.assertFalse(is_path_like("Cargo.toml"))  # No slash, should fail
        self.assertTrue(is_path_like("paraphina/Cargo.toml"))
    
    def test_valid_md_path(self):
        """Markdown paths should be recognized."""
        self.assertTrue(is_path_like("docs/WHITEPAPER.md"))
    
    def test_path_with_symbol_annotation(self):
        """Paths with :: symbol annotations should be recognized."""
        self.assertTrue(is_path_like("paraphina/src/state.rs::GlobalState"))
        self.assertTrue(is_path_like("src/mm.rs::compute_mm_quotes"))
        self.assertTrue(is_path_like("src/hedge.rs::Symbol::method"))
    
    def test_rejects_no_slash(self):
        """Tokens without / should be rejected."""
        self.assertFalse(is_path_like("GlobalState"))
        self.assertFalse(is_path_like("config.rs"))
        self.assertFalse(is_path_like("Cargo.toml"))
    
    def test_rejects_unknown_extension(self):
        """Tokens with unknown extensions should be rejected."""
        self.assertFalse(is_path_like("path/to/file.txt"))
        self.assertFalse(is_path_like("path/to/file.json"))
        self.assertFalse(is_path_like("path/to/file"))


class TestParseImplementedRefs(unittest.TestCase):
    """Test parsing Implemented: references from content."""
    
    def test_parse_single_ref(self):
        """Should parse a single Implemented: reference."""
        content = "- **Implemented: `paraphina/src/state.rs`**"
        refs = parse_implemented_refs(content, "test.md")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0], (1, "paraphina/src/state.rs"))
    
    def test_parse_ref_with_symbol(self):
        """Should parse reference with symbol annotation."""
        content = "- **Implemented: `paraphina/src/state.rs::GlobalState`**"
        refs = parse_implemented_refs(content, "test.md")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0], (1, "paraphina/src/state.rs::GlobalState"))
    
    def test_parse_multiple_refs_same_line(self):
        """Should parse multiple refs on the same line."""
        content = "- **Implemented: `src/a.rs`, `src/b.rs`**"
        refs = parse_implemented_refs(content, "test.md")
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0][1], "src/a.rs")
        self.assertEqual(refs[1][1], "src/b.rs")
    
    def test_parse_multiple_lines(self):
        """Should parse refs across multiple lines."""
        content = """Line 1
- **Implemented: `src/a.rs`**
Line 3
- **Implemented: `src/b.rs::Symbol`**
"""
        refs = parse_implemented_refs(content, "test.md")
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0], (2, "src/a.rs"))
        self.assertEqual(refs[1], (4, "src/b.rs::Symbol"))
    
    def test_ignores_non_path_tokens(self):
        """Should ignore backticked tokens that don't look like paths."""
        content = "- **Implemented: `GlobalState` in `paraphina/src/state.rs`**"
        refs = parse_implemented_refs(content, "test.md")
        # Only the path should be found
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0][1], "paraphina/src/state.rs")
    
    def test_ignores_lines_without_implemented(self):
        """Should ignore lines without 'Implemented:' keyword."""
        content = """Some text with `path/to/file.rs` here
- **Implemented: `src/real.rs`**
More text with `another/path.py`"""
        refs = parse_implemented_refs(content, "test.md")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0][1], "src/real.rs")


class TestValidateRef(unittest.TestCase):
    """Test reference validation against actual files."""
    
    def test_valid_file_exists(self):
        """Should succeed when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Create test file
            (repo_root / "src").mkdir()
            (repo_root / "src/test.rs").write_text("fn main() {}")
            
            is_valid, error = validate_ref("src/test.rs", repo_root)
            self.assertTrue(is_valid)
            self.assertEqual(error, "")  # Empty string for validated success
    
    def test_missing_file_fails(self):
        """Should fail with 'missing file' when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Create the directory so it's not treated as an example path
            (repo_root / "src").mkdir()
            
            is_valid, error = validate_ref("src/nonexistent.rs", repo_root)
            self.assertFalse(is_valid)
            self.assertIn("missing file", error)
            self.assertIn("src/nonexistent.rs", error)
    
    def test_example_path_skipped(self):
        """Example paths (first dir doesn't exist) should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Don't create any directories - this simulates example paths
            
            is_valid, error = validate_ref("path/to/file.rs", repo_root)
            self.assertTrue(is_valid)
            self.assertIsNone(error)  # None indicates skipped, not validated
    
    def test_symbol_present(self):
        """Should succeed when symbol is present in file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            (repo_root / "src/test.rs").write_text("pub struct GlobalState { }")
            
            is_valid, error = validate_ref("src/test.rs::GlobalState", repo_root)
            self.assertTrue(is_valid)
            self.assertEqual(error, "")  # Empty string for validated success
    
    def test_symbol_missing_fails(self):
        """Should fail with 'missing symbol' when symbol not in file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            (repo_root / "src/test.rs").write_text("pub struct OtherStruct { }")
            
            is_valid, error = validate_ref("src/test.rs::GlobalState", repo_root)
            self.assertFalse(is_valid)
            self.assertIn("missing symbol", error)
            self.assertIn("GlobalState", error)
    
    def test_nested_symbol_uses_last_identifier(self):
        """Should check only the last identifier in nested symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            # File contains 'method' but not 'Symbol'
            (repo_root / "src/test.rs").write_text("fn method() {}")
            
            is_valid, error = validate_ref("src/test.rs::Symbol::method", repo_root)
            self.assertTrue(is_valid)  # Should pass because 'method' exists
    
    def test_numeric_symbol_skipped(self):
        """Symbols that don't match identifier pattern should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            (repo_root / "src/test.rs").write_text("pub fn test() {}")
            
            # 123 doesn't match identifier pattern [A-Za-z_][A-Za-z0-9_]*
            # so symbol check is skipped (file just needs to exist)
            is_valid, error = validate_ref("src/test.rs::123", repo_root)
            self.assertTrue(is_valid)  # Should pass because 123 doesn't match identifier pattern
            self.assertEqual(error, "")  # Empty string for validated success


class TestValidateImplementedRefs(unittest.TestCase):
    """Test full validation of Implemented: refs."""
    
    def test_all_refs_valid(self):
        """Should return no errors when all refs are valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "src").mkdir()
            (repo_root / "src/a.rs").write_text("content a")
            (repo_root / "src/b.rs").write_text("pub struct Thing {}")
            
            content = """# Test
- **Implemented: `src/a.rs`**
- **Implemented: `src/b.rs::Thing`**
"""
            errors = validate_implemented_refs(content, repo_root)
            self.assertEqual(len(errors), 0)
    
    def test_missing_file_error(self):
        """Should report error for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Create the directory so it's not treated as an example path
            (repo_root / "src").mkdir()
            
            content = "- **Implemented: `src/missing.rs`**"
            errors = validate_implemented_refs(content, repo_root)
            
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].line, 1)
            self.assertIn("missing file", errors[0].message)
    
    def test_errors_sorted_deterministically(self):
        """Errors should be sorted by (doc_path, line, message)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Create src directory so refs are not treated as example paths
            (repo_root / "src").mkdir()
            # Create one file so we can test missing file detection
            (repo_root / "src/exists.rs").write_text("content")
            
            content = """Line 1
- **Implemented: `src/z_missing.rs`**
Line 3
- **Implemented: `src/a_missing.rs`**
"""
            errors = validate_implemented_refs(content, repo_root)
            
            # Should be sorted by line number
            self.assertEqual(len(errors), 2)
            self.assertEqual(errors[0].line, 2)
            self.assertEqual(errors[1].line, 4)


class TestComputeCanonicalHash(unittest.TestCase):
    """Test canonical hash computation."""
    
    def test_computes_hash_after_marker(self):
        """Should compute SHA256 of content after marker."""
        content = f"""# Part I
Some content
{CANONICAL_MARKER}
# Part II
Canonical content here
"""
        hash_hex, error = compute_canonical_hash(content, CANONICAL_MARKER)
        
        self.assertIsNotNone(hash_hex)
        self.assertEqual(error, "")
        
        # Verify manually
        expected_content = """# Part II
Canonical content here
"""
        expected_hash = hashlib.sha256(expected_content.encode('utf-8')).hexdigest()
        self.assertEqual(hash_hex, expected_hash)
    
    def test_missing_marker_fails(self):
        """Should fail if marker not found."""
        content = "# No marker here"
        hash_hex, error = compute_canonical_hash(content, CANONICAL_MARKER)
        
        self.assertIsNone(hash_hex)
        self.assertIn("marker", error.lower())
    
    def test_normalizes_line_endings(self):
        """Should normalize CRLF to LF."""
        content_lf = f"Before\n{CANONICAL_MARKER}\nCanonical\n"
        content_crlf = f"Before\r\n{CANONICAL_MARKER}\r\nCanonical\r\n"
        
        hash_lf, _ = compute_canonical_hash(content_lf, CANONICAL_MARKER)
        hash_crlf, _ = compute_canonical_hash(content_crlf, CANONICAL_MARKER)
        
        self.assertEqual(hash_lf, hash_crlf)


class TestCheckCanonicalHash(unittest.TestCase):
    """Test canonical hash checking and update logic."""
    
    def test_hash_match_succeeds(self):
        """Should succeed when hash matches baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.txt"
            content = f"Before\n{CANONICAL_MARKER}\nCanonical content\n"
            
            # Compute expected hash
            expected = hashlib.sha256("Canonical content\n".encode('utf-8')).hexdigest()
            baseline_path.write_text(expected + "\n")
            
            success, msg, _ = check_canonical_hash(content, baseline_path, update_mode=False)
            self.assertTrue(success)
            self.assertIn("OK", msg)
    
    def test_hash_mismatch_fails(self):
        """Should fail with exit 2 semantics when hash mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.txt"
            content = f"Before\n{CANONICAL_MARKER}\nCanonical content\n"
            
            # Write wrong hash
            baseline_path.write_text("wronghash\n")
            
            success, msg, _ = check_canonical_hash(content, baseline_path, update_mode=False)
            self.assertFalse(success)
            self.assertIn("mismatch", msg.lower())
            self.assertIn("Expected:", msg)
            self.assertIn("Actual:", msg)
    
    def test_missing_baseline_fails(self):
        """Should fail when baseline file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "nonexistent.txt"
            content = f"Before\n{CANONICAL_MARKER}\nCanonical\n"
            
            success, msg, _ = check_canonical_hash(content, baseline_path, update_mode=False)
            self.assertFalse(success)
            self.assertIn("missing", msg.lower())
            self.assertIn("--update-canonical-hash", msg)
    
    def test_update_mode_writes_baseline(self):
        """Update mode should write the baseline hash file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.txt"
            content = f"Before\n{CANONICAL_MARKER}\nCanonical content\n"
            
            success, msg, _ = check_canonical_hash(content, baseline_path, update_mode=True)
            self.assertTrue(success)
            self.assertTrue(baseline_path.exists())
            
            # Verify written hash
            expected = hashlib.sha256("Canonical content\n".encode('utf-8')).hexdigest()
            written = baseline_path.read_text().strip()
            self.assertEqual(written, expected)


class TestRunChecks(unittest.TestCase):
    """Test the full run_checks function with exit code semantics."""
    
    def test_all_pass_returns_0(self):
        """Should return 0 when all checks pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "docs").mkdir()
            (repo_root / "src").mkdir()
            
            # Create valid file
            (repo_root / "src/test.rs").write_text("pub fn main() {}")
            
            # Create whitepaper with valid ref and marker
            whitepaper = f"""# Part I
- **Implemented: `src/test.rs::main`**

{CANONICAL_MARKER}
# Part II
Canonical content
"""
            (repo_root / "docs/WHITEPAPER.md").write_text(whitepaper)
            
            # Create baseline hash
            expected = hashlib.sha256("# Part II\nCanonical content\n".encode('utf-8')).hexdigest()
            (repo_root / "docs/CANONICAL_SPEC_V1_SHA256.txt").write_text(expected + "\n")
            
            # Capture stdout to avoid noisy output during tests
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exit_code = run_checks(repo_root, update_mode=False)
            self.assertEqual(exit_code, 0)
    
    def test_missing_ref_returns_2(self):
        """Should return 2 for integrity violation (missing file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "docs").mkdir()
            # Create src directory so it's not treated as an example path
            (repo_root / "src").mkdir()
            
            whitepaper = f"""# Part I
- **Implemented: `src/missing.rs`**

{CANONICAL_MARKER}
# Part II
Canonical content
"""
            (repo_root / "docs/WHITEPAPER.md").write_text(whitepaper)
            
            # Create baseline hash (valid)
            expected = hashlib.sha256("# Part II\nCanonical content\n".encode('utf-8')).hexdigest()
            (repo_root / "docs/CANONICAL_SPEC_V1_SHA256.txt").write_text(expected + "\n")
            
            # Capture stdout to avoid noisy output during tests
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exit_code = run_checks(repo_root, update_mode=False)
            self.assertEqual(exit_code, 2)
    
    def test_hash_mismatch_returns_2(self):
        """Should return 2 for integrity violation (hash mismatch)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "docs").mkdir()
            
            whitepaper = f"""# Part I

{CANONICAL_MARKER}
# Part II
Canonical content
"""
            (repo_root / "docs/WHITEPAPER.md").write_text(whitepaper)
            
            # Write wrong baseline hash
            (repo_root / "docs/CANONICAL_SPEC_V1_SHA256.txt").write_text("wronghash\n")
            
            # Capture stdout to avoid noisy output during tests
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exit_code = run_checks(repo_root, update_mode=False)
            self.assertEqual(exit_code, 2)
    
    def test_update_mode_creates_baseline(self):
        """Update mode should create the baseline hash file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "docs").mkdir()
            
            whitepaper = f"""# Part I

{CANONICAL_MARKER}
# Part II
Canonical content
"""
            (repo_root / "docs/WHITEPAPER.md").write_text(whitepaper)
            
            # Capture stdout to avoid noisy output during tests
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exit_code = run_checks(repo_root, update_mode=True)
            self.assertEqual(exit_code, 0)
            
            baseline_path = repo_root / "docs/CANONICAL_SPEC_V1_SHA256.txt"
            self.assertTrue(baseline_path.exists())


class TestRefErrorNamedTuple(unittest.TestCase):
    """Test the RefError named tuple."""
    
    def test_ref_error_has_all_fields(self):
        """RefError should have all required fields."""
        err = RefError(
            doc_path="docs/WHITEPAPER.md",
            line=42,
            message="missing file: src/test.rs",
        )
        
        self.assertEqual(err.doc_path, "docs/WHITEPAPER.md")
        self.assertEqual(err.line, 42)
        self.assertEqual(err.message, "missing file: src/test.rs")


if __name__ == '__main__':
    unittest.main()
