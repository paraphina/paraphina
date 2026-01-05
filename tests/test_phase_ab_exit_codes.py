"""
test_phase_ab_exit_codes.py

Unit tests to lock the exit-code contract for Phase AB CLI.

This test module ensures:
1. Smoke mode exit codes: PASS/HOLD => 0, REJECT => 2, ERROR => 3
2. Strict/gate mode exit codes: PASS => 0, FAIL => 1, HOLD => 2, ERROR => 3
3. Evidence pack verification with tampering detection

Uses only stdlib unittest - no third-party dependencies.

Run with:
    python3 -m unittest tests.test_phase_ab_exit_codes -v

Or via discover:
    python3 -m unittest discover -s tests -p 'test_*.py' -q
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from batch_runs.phase_ab.cli import (
    ci_exit_code_for_decision,
    get_exit_code_description,
    SMOKE_EXIT_PASS,
    SMOKE_EXIT_REJECT,
    SMOKE_EXIT_ERROR,
    GATE_EXIT_PASS,
    GATE_EXIT_FAIL,
    GATE_EXIT_HOLD,
    GATE_EXIT_ERROR,
)


# =============================================================================
# Tests: Smoke Mode Exit Codes
# =============================================================================

class TestSmokeExitCodes(unittest.TestCase):
    """Test smoke mode exit code contract: HOLD is CI pass (exit 0)."""
    
    def test_promote_returns_0(self):
        """PROMOTE returns exit code 0 in smoke mode."""
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("promote", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("Promote", "smoke"), 0)
    
    def test_hold_returns_0(self):
        """HOLD returns exit code 0 in smoke mode (CI pass - pipeline succeeded)."""
        self.assertEqual(ci_exit_code_for_decision("HOLD", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("hold", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("Hold", "smoke"), 0)
    
    def test_reject_returns_2(self):
        """REJECT returns exit code 2 in smoke mode."""
        self.assertEqual(ci_exit_code_for_decision("REJECT", "smoke"), 2)
        self.assertEqual(ci_exit_code_for_decision("reject", "smoke"), 2)
    
    def test_error_returns_3(self):
        """ERROR returns exit code 3 in smoke mode."""
        self.assertEqual(ci_exit_code_for_decision("ERROR", "smoke"), 3)
        self.assertEqual(ci_exit_code_for_decision("error", "smoke"), 3)
    
    def test_unknown_returns_3(self):
        """Unknown decisions return exit code 3 in smoke mode."""
        self.assertEqual(ci_exit_code_for_decision("UNKNOWN", "smoke"), 3)
        self.assertEqual(ci_exit_code_for_decision("invalid", "smoke"), 3)
        self.assertEqual(ci_exit_code_for_decision("", "smoke"), 3)
    
    def test_default_mode_is_smoke(self):
        """Default CI mode is smoke (HOLD = 0)."""
        # Without explicit mode, should use smoke semantics
        self.assertEqual(ci_exit_code_for_decision("HOLD"), 0)
        self.assertEqual(ci_exit_code_for_decision("PROMOTE"), 0)
        self.assertEqual(ci_exit_code_for_decision("REJECT"), 2)
        self.assertEqual(ci_exit_code_for_decision("ERROR"), 3)
    
    def test_smoke_exit_constants(self):
        """Smoke exit code constants are correct."""
        self.assertEqual(SMOKE_EXIT_PASS, 0)
        self.assertEqual(SMOKE_EXIT_REJECT, 2)
        self.assertEqual(SMOKE_EXIT_ERROR, 3)


# =============================================================================
# Tests: Strict/Gate Mode Exit Codes
# =============================================================================

class TestStrictExitCodes(unittest.TestCase):
    """Test strict/gate mode exit code contract: deterministic institutional exit codes."""
    
    def test_promote_returns_0(self):
        """PROMOTE (PASS) returns exit code 0 in strict mode."""
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "strict"), 0)
        self.assertEqual(ci_exit_code_for_decision("promote", "strict"), 0)
        self.assertEqual(ci_exit_code_for_decision("Promote", "strict"), 0)
    
    def test_reject_returns_1(self):
        """REJECT (FAIL) returns exit code 1 in strict mode."""
        self.assertEqual(ci_exit_code_for_decision("REJECT", "strict"), 1)
        self.assertEqual(ci_exit_code_for_decision("reject", "strict"), 1)
    
    def test_hold_returns_2(self):
        """HOLD returns exit code 2 in strict mode (insufficient evidence)."""
        self.assertEqual(ci_exit_code_for_decision("HOLD", "strict"), 2)
        self.assertEqual(ci_exit_code_for_decision("hold", "strict"), 2)
        self.assertEqual(ci_exit_code_for_decision("Hold", "strict"), 2)
    
    def test_error_returns_3(self):
        """ERROR returns exit code 3 in strict mode."""
        self.assertEqual(ci_exit_code_for_decision("ERROR", "strict"), 3)
        self.assertEqual(ci_exit_code_for_decision("error", "strict"), 3)
    
    def test_unknown_returns_3(self):
        """Unknown decisions return exit code 3 in strict mode."""
        self.assertEqual(ci_exit_code_for_decision("UNKNOWN", "strict"), 3)
        self.assertEqual(ci_exit_code_for_decision("invalid", "strict"), 3)
        self.assertEqual(ci_exit_code_for_decision("", "strict"), 3)
    
    def test_gate_exit_constants(self):
        """Gate exit code constants are correct."""
        self.assertEqual(GATE_EXIT_PASS, 0)
        self.assertEqual(GATE_EXIT_FAIL, 1)
        self.assertEqual(GATE_EXIT_HOLD, 2)
        self.assertEqual(GATE_EXIT_ERROR, 3)


# =============================================================================
# Tests: Exit Code Contract Consistency
# =============================================================================

class TestExitCodeContract(unittest.TestCase):
    """Test exit code contract consistency between modes."""
    
    def test_promote_same_in_both_modes(self):
        """PROMOTE returns 0 in both smoke and strict modes."""
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "strict"), 0)
    
    def test_hold_differs_between_modes(self):
        """HOLD has different exit codes between modes."""
        smoke_code = ci_exit_code_for_decision("HOLD", "smoke")
        strict_code = ci_exit_code_for_decision("HOLD", "strict")
        
        self.assertEqual(smoke_code, 0)  # CI pass in smoke
        self.assertEqual(strict_code, 2)  # Insufficient evidence in strict
        self.assertNotEqual(smoke_code, strict_code)
    
    def test_reject_differs_between_modes(self):
        """REJECT has different exit codes between modes."""
        smoke_code = ci_exit_code_for_decision("REJECT", "smoke")
        strict_code = ci_exit_code_for_decision("REJECT", "strict")
        
        self.assertEqual(smoke_code, 2)  # CI fail in smoke
        self.assertEqual(strict_code, 1)  # FAIL in strict
    
    def test_error_same_in_both_modes(self):
        """ERROR returns 3 in both modes."""
        self.assertEqual(ci_exit_code_for_decision("ERROR", "smoke"), 3)
        self.assertEqual(ci_exit_code_for_decision("ERROR", "strict"), 3)
    
    def test_strict_codes_are_distinct(self):
        """All strict mode exit codes are distinct."""
        codes = [
            ci_exit_code_for_decision("PROMOTE", "strict"),
            ci_exit_code_for_decision("REJECT", "strict"),
            ci_exit_code_for_decision("HOLD", "strict"),
            ci_exit_code_for_decision("ERROR", "strict"),
        ]
        self.assertEqual(len(codes), len(set(codes)), "Exit codes should be distinct")
    
    def test_strict_codes_sequential(self):
        """Strict mode exit codes are 0, 1, 2, 3."""
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "strict"), 0)
        self.assertEqual(ci_exit_code_for_decision("REJECT", "strict"), 1)
        self.assertEqual(ci_exit_code_for_decision("HOLD", "strict"), 2)
        self.assertEqual(ci_exit_code_for_decision("ERROR", "strict"), 3)


# =============================================================================
# Tests: Exit Code Documentation
# =============================================================================

class TestExitCodeDocumentation(unittest.TestCase):
    """Test that exit code documentation is available."""
    
    def test_smoke_description_exists(self):
        """Smoke mode has exit code description."""
        desc = get_exit_code_description("smoke")
        self.assertIn("smoke mode", desc.lower())
        self.assertIn("0", desc)
        self.assertIn("HOLD", desc)
    
    def test_strict_description_exists(self):
        """Strict mode has exit code description."""
        desc = get_exit_code_description("strict")
        self.assertIn("strict", desc.lower())
        self.assertIn("0", desc)
        self.assertIn("1", desc)
        self.assertIn("2", desc)
        self.assertIn("3", desc)
    
    def test_descriptions_differ(self):
        """Smoke and strict descriptions are different."""
        smoke_desc = get_exit_code_description("smoke")
        strict_desc = get_exit_code_description("strict")
        self.assertNotEqual(smoke_desc, strict_desc)


# =============================================================================
# Tests: Evidence Pack Verification with Tampering
# =============================================================================

class TestEvidencePackTampering(unittest.TestCase):
    """Test evidence pack verification detects tampering."""
    
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
        """Verification passes for valid evidence pack."""
        from batch_runs.evidence_pack import write_manifest, verify_manifest
        
        # Create test files
        (self.evidence_dir / "file1.txt").write_text("content1")
        (self.evidence_dir / "file2.txt").write_text("content2")
        
        # Write manifest
        write_manifest(self.evidence_dir, {"seed": 42})
        
        # Verify should not raise
        verify_manifest(self.evidence_dir)
    
    def test_verify_fails_on_content_tampering(self):
        """Verification fails when file content is tampered."""
        from batch_runs.evidence_pack import write_manifest, verify_manifest, ManifestError
        
        # Create test file
        test_file = self.evidence_dir / "important_data.json"
        test_file.write_text('{"decision": "PROMOTE"}')
        
        # Write manifest (captures original hash)
        write_manifest(self.evidence_dir, {})
        
        # Tamper with file content
        test_file.write_text('{"decision": "REJECT"}')
        
        # Verification should fail with ManifestError
        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)
        
        error_msg = str(ctx.exception)
        self.assertIn("important_data.json", error_msg)
        self.assertIn("mismatch", error_msg.lower())
    
    def test_verify_fails_on_file_deletion(self):
        """Verification fails when file is deleted."""
        from batch_runs.evidence_pack import write_manifest, verify_manifest, ManifestError
        
        # Create test file
        test_file = self.evidence_dir / "critical_file.txt"
        test_file.write_text("critical content")
        
        # Write manifest
        write_manifest(self.evidence_dir, {})
        
        # Delete the file
        test_file.unlink()
        
        # Verification should fail
        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)
        
        self.assertIn("critical_file.txt", str(ctx.exception))
        self.assertIn("Missing file", str(ctx.exception))
    
    def test_verify_fails_on_extra_file(self):
        """Verification fails when extra file is added (unless allowed)."""
        from batch_runs.evidence_pack import write_manifest, verify_manifest, ManifestError
        
        # Create original file
        (self.evidence_dir / "original.txt").write_text("original")
        
        # Write manifest
        write_manifest(self.evidence_dir, {})
        
        # Add extra file after manifest
        (self.evidence_dir / "injected.txt").write_text("malicious content")
        
        # Verification should fail by default
        with self.assertRaises(ManifestError) as ctx:
            verify_manifest(self.evidence_dir)
        
        error_msg = str(ctx.exception)
        self.assertIn("Extra files", error_msg)
        self.assertIn("injected.txt", error_msg)
    
    def test_verify_passes_extra_file_when_allowed(self):
        """Verification passes with extra files when allow_extra=True."""
        from batch_runs.evidence_pack import write_manifest, verify_manifest
        
        # Create original file
        (self.evidence_dir / "original.txt").write_text("original")
        
        # Write manifest
        write_manifest(self.evidence_dir, {})
        
        # Add extra file
        (self.evidence_dir / "extra.txt").write_text("extra content")
        
        # Verification should pass with allow_extra=True
        verify_manifest(self.evidence_dir, allow_extra=True)
    
    def test_sha256_hash_is_deterministic(self):
        """SHA-256 hash computation is deterministic."""
        from batch_runs.evidence_pack import compute_sha256
        
        # Create test file
        test_file = self.evidence_dir / "test.txt"
        content = b"deterministic content for hashing"
        test_file.write_bytes(content)
        
        # Compute hash multiple times
        hashes = [compute_sha256(test_file) for _ in range(5)]
        
        # All hashes should be identical
        self.assertTrue(all(h == hashes[0] for h in hashes))
        
        # Hash should match expected value
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(hashes[0], expected)


# =============================================================================
# Tests: CLI Module Imports
# =============================================================================

class TestCLIModuleExports(unittest.TestCase):
    """Test that CLI module exports required functions and constants."""
    
    def test_exit_code_function_exported(self):
        """ci_exit_code_for_decision is exported."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertTrue(callable(ci_exit_code_for_decision))
    
    def test_exit_code_constants_exported(self):
        """Exit code constants are exported."""
        from batch_runs.phase_ab.cli import (
            SMOKE_EXIT_PASS,
            SMOKE_EXIT_REJECT,
            SMOKE_EXIT_ERROR,
            GATE_EXIT_PASS,
            GATE_EXIT_FAIL,
            GATE_EXIT_HOLD,
            GATE_EXIT_ERROR,
        )
        
        # All should be integers
        self.assertIsInstance(SMOKE_EXIT_PASS, int)
        self.assertIsInstance(SMOKE_EXIT_REJECT, int)
        self.assertIsInstance(SMOKE_EXIT_ERROR, int)
        self.assertIsInstance(GATE_EXIT_PASS, int)
        self.assertIsInstance(GATE_EXIT_FAIL, int)
        self.assertIsInstance(GATE_EXIT_HOLD, int)
        self.assertIsInstance(GATE_EXIT_ERROR, int)
    
    def test_gate_command_exists(self):
        """Gate command handler exists."""
        from batch_runs.phase_ab.cli import cmd_gate
        self.assertTrue(callable(cmd_gate))
    
    def test_gate_help_works(self):
        """Gate command --help works without error."""
        from batch_runs.phase_ab.cli import main
        
        with self.assertRaises(SystemExit) as ctx:
            main(["gate", "--help"])
        
        # --help exits with code 0
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

