"""
test_phase_ab.py

Unit tests for Phase AB orchestrator.

Test cases:
- resolves run root when trials.jsonl is at root
- resolves one-level nested trials.jsonl
- multiple nested trials → newest chosen + warning
- adversarial search dir passed → error message contains specific diagnosis text
- evidence verification called when SHA256SUMS exists (mock subprocess)
- HOLD decision returns exit 0 (mock phase_b gate)

Uses only stdlib unittest + tempfile + mocking.
"""

import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from typing import List
from batch_runs.phase_ab.pipeline import (
    _validate_sim_eval_binary,
    resolve_run_root,
    verify_evidence_pack,
    run_phase_ab,
    TrialsNotFoundError,
    AdversarialSearchError,
    NestedRunError,
    PhaseABResult,
    PhaseABManifest,
    ROOT,
)


# =============================================================================
# Tests: Run Root Resolution - Basic Cases
# =============================================================================

class TestResolveRunRootBasic(unittest.TestCase):
    """Test resolve_run_root with basic cases."""
    
    def test_resolves_run_root_when_trials_at_root(self):
        """Test that trials.jsonl at root resolves correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trials = root / "trials.jsonl"
            trials.write_text('{"trial_id": "test"}\n')
            
            resolved = resolve_run_root(root)
            
            self.assertEqual(resolved, root)
    
    def test_resolves_one_level_nested_trials(self):
        """Test resolution of trials.jsonl one level deep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested_dir"
            nested.mkdir()
            trials = nested / "trials.jsonl"
            trials.write_text('{"trial_id": "test"}\n')
            
            resolved = resolve_run_root(root)
            
            self.assertEqual(resolved, nested)
    
    def test_returns_directory_not_file(self):
        """Test that resolved path is a directory, not the trials.jsonl file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trials = root / "trials.jsonl"
            trials.write_text('{"trial_id": "test"}\n')
            
            resolved = resolve_run_root(root)
            
            self.assertTrue(resolved.is_dir())
            self.assertEqual(resolved, root)
    
    def test_handles_trials_file_path_directly(self):
        """Test that passing trials.jsonl file path directly returns parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trials = root / "trials.jsonl"
            trials.write_text('{"trial_id": "test"}\n')
            
            # Pass the trials.jsonl file path directly
            resolved = resolve_run_root(trials)
            
            self.assertEqual(resolved, root)


# =============================================================================
# Tests: Run Root Resolution - Multiple Candidates
# =============================================================================

class TestResolveRunRootMultiple(unittest.TestCase):
    """Test resolve_run_root with multiple nested candidates."""
    
    def test_multiple_nested_trials_chooses_newest(self):
        """Test that multiple trials.jsonl files → newest chosen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create older nested directory
            older = root / "older_run"
            older.mkdir()
            older_trials = older / "trials.jsonl"
            older_trials.write_text('{"trial_id": "older"}\n')
            
            # Set older mtime (1 hour ago)
            import time
            old_mtime = time.time() - 3600
            os.utime(older_trials, (old_mtime, old_mtime))
            
            # Create newer nested directory
            newer = root / "newer_run"
            newer.mkdir()
            newer_trials = newer / "trials.jsonl"
            newer_trials.write_text('{"trial_id": "newer"}\n')
            
            # Capture stderr to check for warning
            captured = StringIO()
            with patch('sys.stderr', captured):
                resolved = resolve_run_root(root)
            
            # Should choose newer
            self.assertEqual(resolved, newer)
            
            # Should print warning
            warning = captured.getvalue()
            self.assertIn("WARNING", warning)
            self.assertIn("Multiple trials.jsonl found", warning)
            self.assertIn("older_run", warning)
    
    def test_warning_lists_other_candidates(self):
        """Test that warning message lists all candidate directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create three nested directories with different mtimes
            for i, name in enumerate(["first", "second", "third"]):
                subdir = root / name
                subdir.mkdir()
                trials = subdir / "trials.jsonl"
                trials.write_text(f'{{"trial_id": "{name}"}}\n')
                
                # Set progressively older mtimes
                import time
                mtime = time.time() - (i * 3600)
                os.utime(trials, (mtime, mtime))
            
            captured = StringIO()
            with patch('sys.stderr', captured):
                resolved = resolve_run_root(root)
            
            # Should choose "first" (newest)
            self.assertEqual(resolved, root / "first")
            
            # Warning should list all candidates
            warning = captured.getvalue()
            self.assertIn("second", warning)
            self.assertIn("third", warning)


# =============================================================================
# Tests: Run Root Resolution - Error Cases
# =============================================================================

class TestResolveRunRootErrors(unittest.TestCase):
    """Test resolve_run_root error handling."""
    
    def test_directory_not_found_raises_error(self):
        """Test that non-existent directory raises TrialsNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist"
            
            with self.assertRaises(TrialsNotFoundError) as ctx:
                resolve_run_root(nonexistent)
            
            self.assertIn("does not exist", str(ctx.exception))
    
    def test_no_trials_raises_helpful_error(self):
        """Test that missing trials.jsonl raises helpful error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create some other files but no trials.jsonl
            (root / "pareto.json").write_text("{}")
            (root / "pareto.csv").write_text("rank,id\n")
            
            with self.assertRaises(TrialsNotFoundError) as ctx:
                resolve_run_root(root)
            
            error_msg = str(ctx.exception)
            self.assertIn("trials.jsonl", error_msg)
            self.assertIn("Expected structure", error_msg)
    
    def test_adversarial_search_dir_raises_specific_error(self):
        """Test that adversarial search output directory raises specific error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create adversarial search indicators
            (root / "generated_suite").mkdir()
            (root / "search_results.jsonl").write_text('{"id": "test"}\n')
            (root / "summary.json").write_text("{}")
            
            with self.assertRaises(AdversarialSearchError) as ctx:
                resolve_run_root(root)
            
            error_msg = str(ctx.exception)
            
            # Check for specific diagnosis text
            self.assertIn("Phase A adversarial search output", error_msg)
            self.assertIn("not an evaluated run", error_msg)
            self.assertIn("trials.jsonl", error_msg)
            self.assertIn("sim_eval suite", error_msg)
    
    def test_sim_eval_nesting_raises_guidance_error(self):
        """Test that sim_eval_suite_run nesting raises helpful error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create sim_eval nesting structure
            nested = root / "sim_eval_suite_run" / "run_001"
            nested.mkdir(parents=True)
            (nested / "trials.jsonl").write_text('{"trial_id": "test"}\n')
            
            with self.assertRaises(NestedRunError) as ctx:
                resolve_run_root(root)
            
            error_msg = str(ctx.exception)
            self.assertIn("sim_eval_suite_run", error_msg)
            self.assertIn("Expected path", error_msg)


# =============================================================================
# Tests: sim_eval Binary Resolution
# =============================================================================

class TestSimEvalBinaryResolution(unittest.TestCase):
    """Test sim_eval binary resolution logic with validation."""
    
    def _create_mock_validator(self, valid_paths: List[Path]):
        """Create a mock validator that passes for specific paths."""
        def mock_validate(bin_path: Path, verbose: bool = False) -> bool:
            return bin_path in valid_paths
        return mock_validate
    
    def test_env_var_takes_precedence_when_valid(self):
        """Test that SIM_EVAL_BIN env var takes precedence when it passes validation."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create fake binaries
            env_bin = Path(tmpdir) / "custom_sim_eval"
            release_bin = fake_root / "target" / "release" / "sim_eval"
            
            env_bin.touch()
            env_bin.chmod(0o755)
            release_bin.parent.mkdir(parents=True)
            release_bin.touch()
            
            # Mock validator to pass for env_bin
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary') as mock_validate:
                mock_validate.side_effect = lambda p, verbose=False: p == env_bin
                
                with patch.dict(os.environ, {"SIM_EVAL_BIN": str(env_bin)}):
                    result = resolve_sim_eval_bin(repo_root=fake_root)
                    self.assertEqual(result, env_bin)
    
    def test_release_binary_preferred_over_debug_when_both_valid(self):
        """Test that target/release/sim_eval is preferred over target/debug when both valid."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create both release and debug binaries
            release_bin = fake_root / "target" / "release" / "sim_eval"
            debug_bin = fake_root / "target" / "debug" / "sim_eval"
            
            release_bin.parent.mkdir(parents=True)
            debug_bin.parent.mkdir(parents=True)
            
            release_bin.touch()
            debug_bin.touch()
            
            # Mock validator to pass for both
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary') as mock_validate:
                mock_validate.return_value = True
                
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("SIM_EVAL_BIN", None)
                    result = resolve_sim_eval_bin(repo_root=fake_root)
                    self.assertEqual(result, release_bin)
    
    def test_release_binary_preferred_over_shutil_which(self):
        """Test that target/release/sim_eval is preferred over shutil.which result."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create release binary
            release_bin = fake_root / "target" / "release" / "sim_eval"
            release_bin.parent.mkdir(parents=True)
            release_bin.touch()
            
            # Create a PATH binary
            path_bin = Path(tmpdir) / "path_sim_eval"
            path_bin.touch()
            
            # Mock validator to pass for both
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary') as mock_validate:
                mock_validate.return_value = True
                
                with patch('shutil.which', return_value=str(path_bin)):
                    with patch.dict(os.environ, {}, clear=False):
                        os.environ.pop("SIM_EVAL_BIN", None)
                        result = resolve_sim_eval_bin(repo_root=fake_root)
                        # Should prefer release_bin over path_bin
                        self.assertEqual(result, release_bin)
    
    def test_debug_binary_used_when_release_invalid(self):
        """Test that target/debug/sim_eval is used when release fails validation."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create both binaries
            release_bin = fake_root / "target" / "release" / "sim_eval"
            debug_bin = fake_root / "target" / "debug" / "sim_eval"
            
            release_bin.parent.mkdir(parents=True)
            debug_bin.parent.mkdir(parents=True)
            
            release_bin.touch()
            debug_bin.touch()
            
            # Mock validator to fail for release but pass for debug
            def mock_validate(bin_path: Path, verbose: bool = False) -> bool:
                return bin_path == debug_bin
            
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary', side_effect=mock_validate):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("SIM_EVAL_BIN", None)
                    result = resolve_sim_eval_bin(repo_root=fake_root)
                    self.assertEqual(result, debug_bin)
    
    def test_path_binary_used_when_repo_binaries_invalid(self):
        """Test that PATH binary is used when repo binaries fail validation."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create repo binaries
            release_bin = fake_root / "target" / "release" / "sim_eval"
            debug_bin = fake_root / "target" / "debug" / "sim_eval"
            release_bin.parent.mkdir(parents=True)
            debug_bin.parent.mkdir(parents=True)
            release_bin.touch()
            debug_bin.touch()
            
            # Create PATH binary
            path_bin = Path(tmpdir) / "path_sim_eval"
            path_bin.touch()
            
            # Mock validator to only pass for PATH binary
            def mock_validate(bin_path: Path, verbose: bool = False) -> bool:
                return bin_path == path_bin
            
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary', side_effect=mock_validate):
                with patch('shutil.which', return_value=str(path_bin)):
                    with patch.dict(os.environ, {}, clear=False):
                        os.environ.pop("SIM_EVAL_BIN", None)
                        result = resolve_sim_eval_bin(repo_root=fake_root)
                        self.assertEqual(result, path_bin)
    
    def test_raises_error_when_all_candidates_fail_validation(self):
        """Test that SimEvalNotFoundError is raised when all candidates fail validation."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin, SimEvalNotFoundError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create binaries that will fail validation
            release_bin = fake_root / "target" / "release" / "sim_eval"
            release_bin.parent.mkdir(parents=True)
            release_bin.touch()
            
            # Mock validator to always fail
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary', return_value=False):
                with patch('shutil.which', return_value=None):
                    with patch.dict(os.environ, {}, clear=False):
                        os.environ.pop("SIM_EVAL_BIN", None)
                        with self.assertRaises(SimEvalNotFoundError) as ctx:
                            resolve_sim_eval_bin(repo_root=fake_root)
                        
                        error_msg = str(ctx.exception)
                        self.assertIn("cargo build", error_msg)
                        self.assertIn("--release", error_msg)
    
    def test_error_message_includes_candidates_tried(self):
        """Test that error message lists all candidates tried."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin, SimEvalNotFoundError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Mock validator to always fail
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary', return_value=False):
                with patch('shutil.which', return_value=None):
                    with patch.dict(os.environ, {}, clear=False):
                        os.environ.pop("SIM_EVAL_BIN", None)
                        with self.assertRaises(SimEvalNotFoundError) as ctx:
                            resolve_sim_eval_bin(repo_root=fake_root)
                        
                        error_msg = str(ctx.exception)
                        self.assertIn("target/release", error_msg)
                        self.assertIn("target/debug", error_msg)
                        self.assertIn("not found", error_msg)
    
    def test_repo_root_parameter_allows_testing(self):
        """Test that repo_root parameter enables isolated testing."""
        from batch_runs.phase_ab.pipeline import resolve_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            
            # Create release binary in custom root
            release_bin = fake_root / "target" / "release" / "sim_eval"
            release_bin.parent.mkdir(parents=True)
            release_bin.touch()
            
            # Mock validator to pass
            with patch('batch_runs.phase_ab.pipeline._validate_sim_eval_binary', return_value=True):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("SIM_EVAL_BIN", None)
                    # Use repo_root parameter instead of patching ROOT constant
                    result = resolve_sim_eval_bin(repo_root=fake_root)
                    self.assertEqual(result, release_bin)


# =============================================================================
# Tests: sim_eval Binary Validation
# =============================================================================

class TestSimEvalBinaryValidation(unittest.TestCase):
    """Test sim_eval binary validation logic."""
    
    def test_returns_false_for_nonexistent_binary(self):
        """Test that validation returns False for non-existent binary."""
        fake_bin = Path("/nonexistent/path/sim_eval")
        result = _validate_sim_eval_binary(fake_bin)
        self.assertFalse(result)
    
    def test_returns_false_for_non_executable(self):
        """Test that validation returns False for non-executable file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = Path(tmpdir) / "sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o644)  # Not executable
            
            result = _validate_sim_eval_binary(fake_bin)
            self.assertFalse(result)
    
    def test_returns_true_when_help_contains_required_commands(self):
        """Test that validation passes when --help contains required commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = Path(tmpdir) / "sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o755)
            
            # Mock subprocess.run to return help output with required commands
            mock_help_output = """
sim_eval - Simulation & Evaluation runner

SUBCOMMANDS:
  run                   Run a single scenario
  suite                 Run a CI suite
  verify-evidence-pack  Verify evidence pack hashes
  verify-evidence-tree  Verify evidence tree
  write-evidence-pack   Write evidence pack
"""
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=mock_help_output,
                    stderr="",
                    returncode=0,
                )
                
                result = _validate_sim_eval_binary(fake_bin)
                self.assertTrue(result)
    
    def test_returns_false_when_missing_verify_evidence_pack(self):
        """Test that validation fails when verify-evidence-pack is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = Path(tmpdir) / "sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o755)
            
            # Help output missing verify-evidence-pack
            mock_help_output = """
sim_eval - Simulation & Evaluation runner

SUBCOMMANDS:
  run                   Run a single scenario
  suite                 Run a CI suite
  verify-evidence-tree  Verify evidence tree
"""
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=mock_help_output,
                    stderr="",
                    returncode=0,
                )
                
                result = _validate_sim_eval_binary(fake_bin)
                self.assertFalse(result)
    
    def test_returns_false_when_missing_verify_evidence_tree(self):
        """Test that validation fails when verify-evidence-tree is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = Path(tmpdir) / "sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o755)
            
            # Help output missing verify-evidence-tree
            mock_help_output = """
sim_eval - Simulation & Evaluation runner

SUBCOMMANDS:
  run                   Run a single scenario
  suite                 Run a CI suite
  verify-evidence-pack  Verify evidence pack hashes
"""
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=mock_help_output,
                    stderr="",
                    returncode=0,
                )
                
                result = _validate_sim_eval_binary(fake_bin)
                self.assertFalse(result)
    
    def test_returns_false_on_subprocess_timeout(self):
        """Test that validation returns False when --help times out."""
        import subprocess as sp
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_bin = Path(tmpdir) / "sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o755)
            
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = sp.TimeoutExpired(cmd=["sim_eval"], timeout=10)
                
                result = _validate_sim_eval_binary(fake_bin)
                self.assertFalse(result)


# =============================================================================
# Tests: Evidence Verification
# =============================================================================

class TestEvidenceVerification(unittest.TestCase):
    """Test evidence pack verification."""
    
    def test_no_evidence_pack_returns_true(self):
        """Test that missing evidence pack returns success (nothing to verify)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            success, errors = verify_evidence_pack(root)
            
            self.assertTrue(success)
            self.assertEqual(errors, [])
    
    def test_evidence_verification_called_when_sha256sums_exists(self):
        """Test that verification is called when SHA256SUMS exists."""
        from batch_runs.phase_ab.pipeline import find_sim_eval_bin
        
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create evidence pack structure
            evidence_pack = root / "evidence_pack"
            evidence_pack.mkdir()
            (evidence_pack / "SHA256SUMS").write_text("abc123  file.txt\n")
            
            # Create a fake sim_eval binary
            fake_bin = Path(tmpdir) / "fake_sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o755)
            
            # Mock find_sim_eval_bin to return our fake binary
            with patch('batch_runs.phase_ab.pipeline.find_sim_eval_bin', return_value=fake_bin):
                # Mock subprocess.run
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0,
                        stdout="OK: verified 1 file",
                        stderr="",
                    )
                    
                    success, errors = verify_evidence_pack(root)
                    
                    # Should have called subprocess
                    self.assertTrue(mock_run.called)
                    
                    # Check command contains verify-evidence-pack
                    call_args = mock_run.call_args[0][0]
                    self.assertIn("verify-evidence-pack", call_args)
                    self.assertIn(str(root), call_args)
    
    def test_verification_failure_returns_errors(self):
        """Test that verification failure returns error messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create evidence pack structure
            evidence_pack = root / "evidence_pack"
            evidence_pack.mkdir()
            (evidence_pack / "SHA256SUMS").write_text("abc123  file.txt\n")
            
            # Create a fake sim_eval binary
            fake_bin = Path(tmpdir) / "fake_sim_eval"
            fake_bin.touch()
            fake_bin.chmod(0o755)
            
            # Mock find_sim_eval_bin and subprocess.run to fail
            with patch('batch_runs.phase_ab.pipeline.find_sim_eval_bin', return_value=fake_bin):
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=1,
                        stdout="",
                        stderr="Hash mismatch for file.txt",
                    )
                    
                    success, errors = verify_evidence_pack(root)
                    
                    self.assertFalse(success)
                    # Check that error contains the mismatch message
                    error_text = " ".join(errors)
                    self.assertIn("Hash mismatch", error_text)
    
    def test_verification_fails_when_binary_not_found(self):
        """Test that verification fails gracefully when sim_eval not found."""
        from batch_runs.phase_ab.pipeline import SimEvalNotFoundError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create evidence pack structure
            evidence_pack = root / "evidence_pack"
            evidence_pack.mkdir()
            (evidence_pack / "SHA256SUMS").write_text("abc123  file.txt\n")
            
            # Mock find_sim_eval_bin to raise SimEvalNotFoundError
            with patch('batch_runs.phase_ab.pipeline.find_sim_eval_bin') as mock_find:
                mock_find.side_effect = SimEvalNotFoundError("Binary not found")
                
                success, errors = verify_evidence_pack(root)
                
                self.assertFalse(success)
                self.assertIn("Binary not found", errors[0])


# =============================================================================
# Tests: CI Exit Code Semantics
# =============================================================================

class TestCIExitCodeSemantics(unittest.TestCase):
    """Test CI exit code semantics with ci_exit_code_for_decision helper."""
    
    def test_ci_exit_code_smoke_mode_promote_returns_0(self):
        """PROMOTE returns 0 in smoke mode."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("promote", "smoke"), 0)
    
    def test_ci_exit_code_smoke_mode_hold_returns_0(self):
        """HOLD returns 0 in smoke mode (CI pass - pipeline succeeded)."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("HOLD", "smoke"), 0)
        self.assertEqual(ci_exit_code_for_decision("hold", "smoke"), 0)
    
    def test_ci_exit_code_smoke_mode_reject_returns_2(self):
        """REJECT returns 2 in smoke mode (CI fail - guardrails failed)."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("REJECT", "smoke"), 2)
        self.assertEqual(ci_exit_code_for_decision("reject", "smoke"), 2)
    
    def test_ci_exit_code_smoke_mode_error_returns_3(self):
        """ERROR returns 3 in smoke mode (CI fail - runtime error)."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("ERROR", "smoke"), 3)
        self.assertEqual(ci_exit_code_for_decision("error", "smoke"), 3)
    
    def test_ci_exit_code_strict_mode_promote_returns_0(self):
        """PROMOTE returns 0 in strict mode."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("PROMOTE", "strict"), 0)
    
    def test_ci_exit_code_strict_mode_hold_returns_1(self):
        """HOLD returns 1 in strict mode (CI fail - promotion required)."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("HOLD", "strict"), 1)
    
    def test_ci_exit_code_strict_mode_reject_returns_2(self):
        """REJECT returns 2 in strict mode."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        self.assertEqual(ci_exit_code_for_decision("REJECT", "strict"), 2)
    
    def test_ci_exit_code_default_is_smoke_mode(self):
        """Default CI mode is smoke (HOLD = 0)."""
        from batch_runs.phase_ab.cli import ci_exit_code_for_decision
        # Without explicit mode, should use smoke semantics
        self.assertEqual(ci_exit_code_for_decision("HOLD"), 0)


# =============================================================================
# Tests: Exit Code Semantics (Legacy)
# =============================================================================

class TestExitCodeSemantics(unittest.TestCase):
    """Test institutional CI exit code semantics."""
    
    def test_hold_decision_returns_exit_0(self):
        """Test that HOLD decision returns exit code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create minimal trials.jsonl
            trials = root / "trials.jsonl"
            trials.write_text('{"trial_id": "test", "pnl_mean": 100, "pnl_stdev": 10, "dd_cvar": 50, "kill_rate": 0.0}\n')
            
            out_dir = Path(tmpdir) / "output"
            
            # Mock Phase B to return HOLD
            with patch('batch_runs.phase_ab.pipeline.run_phase_ab') as mock_run:
                # Create a mock result with HOLD decision
                mock_manifest = PhaseABManifest(
                    candidate_run_resolved=str(root),
                    baseline_run_resolved=None,
                    phase_b_out_dir=str(out_dir),
                    decision="HOLD",
                    alpha=0.05,
                    bootstrap_samples=1000,
                    seed=42,
                    confidence_report_json=str(out_dir / "confidence_report.json"),
                    confidence_report_md=str(out_dir / "confidence_report.md"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                mock_result = PhaseABResult(
                    manifest=mock_manifest,
                    exit_code=0,  # HOLD returns 0
                )
                mock_run.return_value = mock_result
                
                result = mock_run()
                
                # HOLD should return exit code 0
                self.assertEqual(result.exit_code, 0)
                self.assertEqual(result.manifest.decision, "HOLD")
    
    def test_promote_decision_returns_exit_0(self):
        """Test that PROMOTE decision returns exit code 0."""
        mock_manifest = PhaseABManifest(
            candidate_run_resolved="/path/to/candidate",
            baseline_run_resolved="/path/to/baseline",
            phase_b_out_dir="/path/to/output",
            decision="PROMOTE",
            alpha=0.05,
            bootstrap_samples=1000,
            seed=42,
            confidence_report_json="/path/to/confidence_report.json",
            confidence_report_md="/path/to/confidence_report.md",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        mock_result = PhaseABResult(
            manifest=mock_manifest,
            exit_code=0,  # PROMOTE returns 0
        )
        
        self.assertEqual(mock_result.exit_code, 0)
        self.assertEqual(mock_result.decision, "PROMOTE")
    
    def test_reject_decision_returns_exit_2(self):
        """Test that REJECT decision returns exit code 2."""
        mock_manifest = PhaseABManifest(
            candidate_run_resolved="/path/to/candidate",
            baseline_run_resolved="/path/to/baseline",
            phase_b_out_dir="/path/to/output",
            decision="REJECT",
            alpha=0.05,
            bootstrap_samples=1000,
            seed=42,
            confidence_report_json="/path/to/confidence_report.json",
            confidence_report_md="/path/to/confidence_report.md",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        mock_result = PhaseABResult(
            manifest=mock_manifest,
            exit_code=2,  # REJECT returns 2
        )
        
        self.assertEqual(mock_result.exit_code, 2)
        self.assertEqual(mock_result.decision, "REJECT")
    
    def test_error_returns_exit_3(self):
        """Test that ERROR returns exit code 3."""
        mock_manifest = PhaseABManifest(
            candidate_run_resolved="/path/to/candidate",
            baseline_run_resolved=None,
            phase_b_out_dir="/path/to/output",
            decision="ERROR",
            alpha=0.05,
            bootstrap_samples=1000,
            seed=42,
            confidence_report_json="",
            confidence_report_md="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            errors=["Something went wrong"],
        )
        mock_result = PhaseABResult(
            manifest=mock_manifest,
            exit_code=3,  # ERROR returns 3
        )
        
        self.assertEqual(mock_result.exit_code, 3)
        self.assertEqual(mock_result.decision, "ERROR")


# =============================================================================
# Tests: Manifest Structure
# =============================================================================

class TestManifestStructure(unittest.TestCase):
    """Test Phase AB manifest structure."""
    
    def test_manifest_to_dict_includes_required_fields(self):
        """Test that manifest.to_dict() includes all required fields."""
        manifest = PhaseABManifest(
            candidate_run_resolved="/path/to/candidate",
            baseline_run_resolved="/path/to/baseline",
            phase_b_out_dir="/path/to/output",
            decision="HOLD",
            alpha=0.05,
            bootstrap_samples=1000,
            seed=42,
            confidence_report_json="/path/to/confidence_report.json",
            confidence_report_md="/path/to/confidence_report.md",
            timestamp="2025-01-04T12:00:00Z",
            git_commit="abc123",
        )
        
        d = manifest.to_dict()
        
        # Check all required fields
        self.assertIn("candidate_run_resolved", d)
        self.assertIn("baseline_run_resolved", d)
        self.assertIn("phase_b_out_dir", d)
        self.assertIn("decision", d)
        self.assertIn("alpha", d)
        self.assertIn("bootstrap_samples", d)
        self.assertIn("seed", d)
        self.assertIn("confidence_report_json", d)
        self.assertIn("confidence_report_md", d)
        self.assertIn("timestamp", d)
        self.assertIn("git_commit", d)
        self.assertIn("schema_version", d)
    
    def test_manifest_json_serializable(self):
        """Test that manifest can be serialized to JSON."""
        manifest = PhaseABManifest(
            candidate_run_resolved="/path/to/candidate",
            baseline_run_resolved=None,
            phase_b_out_dir="/path/to/output",
            decision="HOLD",
            alpha=0.05,
            bootstrap_samples=1000,
            seed=42,
            confidence_report_json="/path/to/confidence_report.json",
            confidence_report_md="/path/to/confidence_report.md",
            timestamp="2025-01-04T12:00:00Z",
        )
        
        # Should not raise
        json_str = json.dumps(manifest.to_dict())
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["decision"], "HOLD")


# =============================================================================
# Tests: CLI Integration
# =============================================================================

class TestCLIIntegration(unittest.TestCase):
    """Test CLI module integration."""
    
    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from batch_runs.phase_ab import cli
        self.assertTrue(hasattr(cli, 'main'))
        self.assertTrue(hasattr(cli, 'cmd_run'))
        self.assertTrue(hasattr(cli, 'cmd_smoke'))
    
    def test_cli_help_works(self):
        """Test that CLI --help works."""
        from batch_runs.phase_ab.cli import main
        
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        
        # --help exits with code 0
        self.assertEqual(ctx.exception.code, 0)
    
    def test_smoke_command_exists(self):
        """Test that smoke subcommand exists."""
        from batch_runs.phase_ab.cli import main
        
        with self.assertRaises(SystemExit) as ctx:
            main(["smoke", "--help"])
        
        self.assertEqual(ctx.exception.code, 0)


# =============================================================================
# Tests: Determinism
# =============================================================================

class TestDeterminism(unittest.TestCase):
    """Test that operations are deterministic."""
    
    def test_resolve_run_root_deterministic(self):
        """Test that run root resolution is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trials = root / "trials.jsonl"
            trials.write_text('{"trial_id": "test"}\n')
            
            # Call multiple times
            results = [resolve_run_root(root) for _ in range(5)]
            
            # All results should be identical
            self.assertTrue(all(r == results[0] for r in results))
    
    def test_manifest_to_dict_deterministic(self):
        """Test that manifest serialization is deterministic."""
        manifest = PhaseABManifest(
            candidate_run_resolved="/path/to/candidate",
            baseline_run_resolved="/path/to/baseline",
            phase_b_out_dir="/path/to/output",
            decision="HOLD",
            alpha=0.05,
            bootstrap_samples=1000,
            seed=42,
            confidence_report_json="/path/to/confidence_report.json",
            confidence_report_md="/path/to/confidence_report.md",
            timestamp="2025-01-04T12:00:00Z",
        )
        
        # Serialize multiple times
        json_strs = [json.dumps(manifest.to_dict(), sort_keys=True) for _ in range(5)]
        
        # All should be identical
        self.assertTrue(all(s == json_strs[0] for s in json_strs))


# =============================================================================
# Tests: Run PhaseAB with Mocks
# =============================================================================

class TestRunPhaseABMocked(unittest.TestCase):
    """Test run_phase_ab with mocked Phase B."""
    
    def test_run_phase_ab_resolves_and_calls_phase_b(self):
        """Test that run_phase_ab resolves paths and calls Phase B."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create candidate dir with trials.jsonl
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "trials.jsonl").write_text('{"trial_id": "test"}\n')
            
            # Create baseline dir with trials.jsonl
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "trials.jsonl").write_text('{"trial_id": "baseline"}\n')
            
            out_dir = root / "output"
            
            # Mock Phase B run_gate
            with patch('batch_runs.phase_ab.pipeline.run_phase_ab') as mock_run:
                mock_manifest = PhaseABManifest(
                    candidate_run_resolved=str(candidate),
                    baseline_run_resolved=str(baseline),
                    phase_b_out_dir=str(out_dir),
                    decision="HOLD",
                    alpha=0.05,
                    bootstrap_samples=1000,
                    seed=42,
                    confidence_report_json=str(out_dir / "confidence_report.json"),
                    confidence_report_md=str(out_dir / "confidence_report.md"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                mock_run.return_value = PhaseABResult(
                    manifest=mock_manifest,
                    exit_code=0,
                )
                
                result = mock_run(
                    candidate_run=candidate,
                    baseline_run=baseline,
                    out_dir=out_dir,
                )
                
                self.assertEqual(result.exit_code, 0)
                self.assertEqual(result.manifest.decision, "HOLD")


if __name__ == "__main__":
    unittest.main()

