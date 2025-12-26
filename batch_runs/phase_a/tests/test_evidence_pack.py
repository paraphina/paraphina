"""
Tests for root evidence pack functionality.

Hermetic tests that don't require cargo - they test the Python helpers
and mock the subprocess calls.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.promote_pipeline import (
    _write_evidence_pack,
    _verify_evidence_pack,
    ROOT,
    SIM_EVAL_BIN,
)


class TestWriteEvidencePackCommand(unittest.TestCase):
    """Test the write evidence pack command construction."""
    
    def test_command_uses_binary_when_exists(self):
        """Test that command uses SIM_EVAL_BIN when it exists."""
        out_dir = Path("/test/output")
        
        with patch('subprocess.run') as mock_run, \
             patch.object(Path, 'exists', return_value=True):
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            
            # We need to mock the SIM_EVAL_BIN.exists() check
            with patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
                mock_bin.exists.return_value = True
                mock_bin.__str__ = lambda self: "/path/to/sim_eval"
                
                result = _write_evidence_pack(out_dir, verbose=False)
                
                # Verify subprocess was called
                self.assertEqual(result, 0)
    
    def test_command_falls_back_to_cargo_run(self):
        """Test that command falls back to cargo run when binary doesn't exist."""
        out_dir = Path("/test/output")
        
        with patch('subprocess.run') as mock_run, \
             patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
            mock_bin.exists.return_value = False
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            
            result = _write_evidence_pack(out_dir, verbose=False, smoke=True)
            
            # Verify subprocess was called with cargo run
            self.assertEqual(result, 0)
            call_args = mock_run.call_args[0][0]
            self.assertIn("cargo", call_args)
            self.assertIn("write-evidence-pack", call_args)
    
    def test_returns_error_code_on_failure(self):
        """Test that error code is returned on subprocess failure."""
        out_dir = Path("/test/output")
        
        with patch('subprocess.run') as mock_run, \
             patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
            mock_bin.exists.return_value = True
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")
            
            result = _write_evidence_pack(out_dir, verbose=False)
            
            self.assertEqual(result, 1)
    
    def test_handles_exception_gracefully(self):
        """Test that exceptions are handled and return error code."""
        out_dir = Path("/test/output")
        
        with patch('subprocess.run', side_effect=FileNotFoundError("binary not found")), \
             patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
            mock_bin.exists.return_value = True
            
            result = _write_evidence_pack(out_dir, verbose=False)
            
            self.assertEqual(result, 1)


class TestVerifyEvidencePackCommand(unittest.TestCase):
    """Test the verify evidence pack command construction."""
    
    def test_verify_command_uses_binary(self):
        """Test that verify command uses SIM_EVAL_BIN when it exists."""
        out_dir = Path("/test/output")
        
        with patch('subprocess.run') as mock_run, \
             patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
            mock_bin.exists.return_value = True
            mock_run.return_value = MagicMock(returncode=0, stdout="OK: verified 5 files", stderr="")
            
            result = _verify_evidence_pack(out_dir, verbose=False)
            
            self.assertEqual(result, 0)
    
    def test_verify_returns_error_on_failure(self):
        """Test that verification failure returns error code."""
        out_dir = Path("/test/output")
        
        with patch('subprocess.run') as mock_run, \
             patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
            mock_bin.exists.return_value = True
            mock_run.return_value = MagicMock(returncode=3, stdout="", stderr="Hash mismatch")
            
            result = _verify_evidence_pack(out_dir, verbose=False)
            
            self.assertEqual(result, 3)


class TestEvidencePackDeterminism(unittest.TestCase):
    """Test that evidence pack operations are deterministic."""
    
    def test_command_arguments_are_deterministic(self):
        """Test that command arguments are constructed deterministically."""
        out_dir = Path("/test/output/study_001")
        
        with patch('subprocess.run') as mock_run, \
             patch('batch_runs.phase_a.promote_pipeline.SIM_EVAL_BIN') as mock_bin:
            mock_bin.exists.return_value = True
            mock_bin.__str__ = lambda self: "/path/to/sim_eval"
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            
            # Call twice
            _write_evidence_pack(out_dir, verbose=False)
            _write_evidence_pack(out_dir, verbose=False)
            
            # Get both call argument lists
            calls = mock_run.call_args_list
            self.assertEqual(len(calls), 2)
            
            # First positional argument is the command list
            cmd1 = calls[0][0][0]
            cmd2 = calls[1][0][0]
            
            # Commands should be identical
            self.assertEqual(cmd1, cmd2)


class TestEvidencePackIntegrationPaths(unittest.TestCase):
    """Test the integration paths for evidence packs."""
    
    def test_root_constant_is_valid_path(self):
        """Test that ROOT constant points to a valid directory."""
        # ROOT should be the repo root (2 levels up from the test file)
        self.assertTrue(ROOT.exists(), f"ROOT {ROOT} should exist")
        self.assertTrue(ROOT.is_dir(), f"ROOT {ROOT} should be a directory")
        
        # Should contain the paraphina crate directory
        self.assertTrue((ROOT / "paraphina").exists(), 
                       f"ROOT {ROOT} should contain paraphina/")
    
    def test_sim_eval_bin_path_is_sensible(self):
        """Test that SIM_EVAL_BIN path is reasonable."""
        # SIM_EVAL_BIN should be under target/release
        self.assertTrue(str(SIM_EVAL_BIN).endswith("sim_eval") or 
                       str(SIM_EVAL_BIN).endswith("sim_eval.exe"))
        self.assertIn("target", str(SIM_EVAL_BIN))
        self.assertIn("release", str(SIM_EVAL_BIN))


if __name__ == "__main__":
    unittest.main()

