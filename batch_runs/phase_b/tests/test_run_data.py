"""
Tests for Phase B run_data module.

Tests:
- test_discover_trials_direct: Direct trials.jsonl in run_root
- test_discover_trials_nested: Nested discovery with shallowest precedence
- test_discover_trials_jsonl_file: Direct JSONL file as input
- test_parse_trials_jsonl: JSONL parsing with metric extraction
- test_no_usable_observations_error: Error when no valid observations
- test_missing_trials_file_error: Error when no trials.jsonl found
- test_metric_extraction_fallbacks: Resilient metric extraction
- test_status_filtering: Filtering by status field
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_b.run_data import (
    discover_trials_file,
    parse_trials_jsonl,
    load_run_data,
    TrialsFileNotFoundError,
    NoUsableObservationsError,
    TrialObservation,
    LoadedRunData,
)


class TestDiscoverTrialsFile(unittest.TestCase):
    """Tests for trials.jsonl discovery."""
    
    def test_discover_trials_direct(self):
        """Test discovery of direct trials.jsonl in run_root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            trials_file = run_root / "trials.jsonl"
            trials_file.write_text('{"pnl": 100, "kill_ci": 0.1}\n')
            
            discovered = discover_trials_file(run_root)
            self.assertEqual(discovered, trials_file)
    
    def test_discover_trials_nested(self):
        """Test discovery of nested trials.jsonl with shallowest precedence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            
            # Create nested structure with trials.jsonl at different depths
            # No run_root/trials.jsonl - should search nested
            shallow = run_root / "subdir"
            shallow.mkdir(parents=True)
            shallow_trials = shallow / "trials.jsonl"
            shallow_trials.write_text('{"pnl": 100, "kill_ci": 0.1}\n')
            
            deep = run_root / "some" / "deep" / "path"
            deep.mkdir(parents=True)
            deep_trials = deep / "trials.jsonl"
            deep_trials.write_text('{"pnl": 200, "kill_ci": 0.2}\n')
            
            discovered = discover_trials_file(run_root)
            # Should pick the shallowest one
            self.assertEqual(discovered, shallow_trials)
    
    def test_discover_trials_jsonl_file(self):
        """Test discovery when run_root is a JSONL file directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            jsonl_file = run_root / "my_trials.jsonl"
            jsonl_file.write_text('{"pnl": 100, "kill_ci": 0.1}\n')
            
            discovered = discover_trials_file(jsonl_file)
            self.assertEqual(discovered, jsonl_file)
    
    def test_discover_trials_not_found(self):
        """Test error when no trials.jsonl found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            # Empty directory
            
            with self.assertRaises(TrialsFileNotFoundError) as ctx:
                discover_trials_file(run_root)
            
            self.assertIn(str(run_root), str(ctx.exception))


class TestParseTrialsJsonl(unittest.TestCase):
    """Tests for JSONL parsing."""
    
    def test_parse_basic_trials(self):
        """Test parsing basic trials.jsonl with pnl and kill_ci."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100.5, "kill_ci": 0.05}',
                '{"pnl": 150.0, "kill_ci": 0.10}',
                '{"pnl": 75.25, "kill_ci": 0.02}',
            ])
            trials_file.write_text(content)
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 3)
            self.assertEqual(observations[0].pnl, 100.5)
            self.assertEqual(observations[0].kill_ci, 0.05)
            self.assertEqual(observations[1].pnl, 150.0)
            self.assertEqual(observations[2].pnl, 75.25)
            self.assertEqual(len(rejection_counts), 0)
    
    def test_parse_with_blank_lines(self):
        """Test that blank lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100, "kill_ci": 0.05}',
                '',
                '{"pnl": 150, "kill_ci": 0.10}',
                '   ',
                '{"pnl": 75, "kill_ci": 0.02}',
            ])
            trials_file.write_text(content)
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 3)
    
    def test_parse_phase_a_format(self):
        """Test parsing actual Phase A output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            # This matches the actual Phase A output format
            content = json.dumps({
                "candidate_id": "abc123",
                "trial_id": "trial_0000_abc123",
                "is_valid": True,
                "pnl_mean": 34.32,
                "mc_mean_pnl": 34.32,
                "kill_rate": 0.0,
                "kill_ucb": 0.213,
                "mc_kill_prob_ci_upper": 0.278,
                "dd_cvar": 0.0,
                "mc_drawdown_cvar": 0.0,
            })
            trials_file.write_text(content + '\n')
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 1)
            self.assertAlmostEqual(observations[0].pnl, 34.32, places=2)
            # Should use kill_ucb as kill_ci (fallback)
            self.assertAlmostEqual(observations[0].kill_ci, 0.213, places=3)


class TestMetricExtraction(unittest.TestCase):
    """Tests for resilient metric extraction."""
    
    def test_metric_from_top_level(self):
        """Test metric extraction from top-level keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            trials_file.write_text('{"pnl": 100, "kill_ci": 0.05}\n')
            
            observations, _ = parse_trials_jsonl(trials_file)
            
            self.assertEqual(observations[0].pnl, 100)
            self.assertEqual(observations[0].kill_ci, 0.05)
    
    def test_metric_from_metrics_subkey(self):
        """Test metric extraction from metrics.<key>."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = json.dumps({
                "metrics": {"pnl": 200, "kill_ci": 0.15}
            })
            trials_file.write_text(content + '\n')
            
            observations, _ = parse_trials_jsonl(trials_file)
            
            self.assertEqual(observations[0].pnl, 200)
            self.assertEqual(observations[0].kill_ci, 0.15)
    
    def test_metric_from_result_subkey(self):
        """Test metric extraction from result.<key>."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = json.dumps({
                "result": {"pnl": 300, "kill_ci": 0.25}
            })
            trials_file.write_text(content + '\n')
            
            observations, _ = parse_trials_jsonl(trials_file)
            
            self.assertEqual(observations[0].pnl, 300)
            self.assertEqual(observations[0].kill_ci, 0.25)
    
    def test_pnl_fallback_keys(self):
        """Test PnL extraction with fallback keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            
            # Test pnl_mean fallback
            content = json.dumps({"pnl_mean": 100, "kill_ci": 0.1})
            trials_file.write_text(content + '\n')
            observations, _ = parse_trials_jsonl(trials_file)
            self.assertEqual(observations[0].pnl, 100)
            
            # Test mc_mean_pnl fallback
            content = json.dumps({"mc_mean_pnl": 200, "kill_ci": 0.1})
            trials_file.write_text(content + '\n')
            observations, _ = parse_trials_jsonl(trials_file)
            self.assertEqual(observations[0].pnl, 200)
    
    def test_kill_ci_fallback_keys(self):
        """Test kill_ci extraction with fallback keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            
            # Test kill_ucb fallback
            content = json.dumps({"pnl": 100, "kill_ucb": 0.15})
            trials_file.write_text(content + '\n')
            observations, _ = parse_trials_jsonl(trials_file)
            self.assertEqual(observations[0].kill_ci, 0.15)
            
            # Test mc_kill_prob_ci_upper fallback
            content = json.dumps({"pnl": 100, "mc_kill_prob_ci_upper": 0.25})
            trials_file.write_text(content + '\n')
            observations, _ = parse_trials_jsonl(trials_file)
            self.assertEqual(observations[0].kill_ci, 0.25)
            
            # Test kill_rate fallback when no CI available
            content = json.dumps({"pnl": 100, "kill_rate": 0.05})
            trials_file.write_text(content + '\n')
            observations, _ = parse_trials_jsonl(trials_file)
            self.assertEqual(observations[0].kill_ci, 0.05)


class TestStatusFiltering(unittest.TestCase):
    """Tests for status-based filtering."""
    
    def test_accept_valid_status_values(self):
        """Test that valid status values are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100, "kill_ci": 0.1, "status": "OK"}',
                '{"pnl": 200, "kill_ci": 0.1, "status": "PASS"}',
                '{"pnl": 300, "kill_ci": 0.1, "status": "VALID"}',
            ])
            trials_file.write_text(content)
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 3)
            self.assertEqual(len(rejection_counts), 0)
    
    def test_reject_invalid_status(self):
        """Test that invalid status values are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100, "kill_ci": 0.1, "status": "OK"}',
                '{"pnl": 200, "kill_ci": 0.1, "status": "FAILED"}',
                '{"pnl": 300, "kill_ci": 0.1, "status": "ERROR"}',
            ])
            trials_file.write_text(content)
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].pnl, 100)
            self.assertIn("status=FAILED", rejection_counts)
            self.assertIn("status=ERROR", rejection_counts)
    
    def test_accept_without_status(self):
        """Test that records without status field are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = '{"pnl": 100, "kill_ci": 0.1}\n'
            trials_file.write_text(content)
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 1)
    
    def test_reject_is_valid_false(self):
        """Test that is_valid=False records are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_file = Path(tmpdir) / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100, "kill_ci": 0.1, "is_valid": true}',
                '{"pnl": 200, "kill_ci": 0.1, "is_valid": false}',
            ])
            trials_file.write_text(content)
            
            observations, rejection_counts = parse_trials_jsonl(trials_file)
            
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].pnl, 100)
            self.assertIn("is_valid=False", rejection_counts)


class TestNoUsableObservationsError(unittest.TestCase):
    """Tests for no usable observations error."""
    
    def test_empty_file_error(self):
        """Test error on empty trials file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            trials_file = run_root / "trials.jsonl"
            trials_file.write_text('')
            
            with self.assertRaises(NoUsableObservationsError) as ctx:
                load_run_data(run_root)
            
            error = ctx.exception
            self.assertEqual(error.trials_file, trials_file)
            self.assertEqual(error.lines_parsed, 0)
    
    def test_missing_metrics_error(self):
        """Test error when required metrics are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            trials_file = run_root / "trials.jsonl"
            # Missing kill_ci
            content = '\n'.join([
                '{"pnl": 100}',
                '{"pnl": 200}',
            ])
            trials_file.write_text(content)
            
            with self.assertRaises(NoUsableObservationsError) as ctx:
                load_run_data(run_root)
            
            error = ctx.exception
            self.assertEqual(error.trials_file, trials_file)
            self.assertEqual(error.lines_parsed, 2)
            self.assertIn("missing_kill_ci", error.rejection_counts)
            self.assertEqual(error.rejection_counts["missing_kill_ci"], 2)
    
    def test_error_message_includes_details(self):
        """Test that error message includes file path and rejection counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            trials_file = run_root / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100}',  # missing kill_ci
                '{"kill_ci": 0.1}',  # missing pnl
                '{"invalid json',  # parse error
            ])
            trials_file.write_text(content)
            
            with self.assertRaises(NoUsableObservationsError) as ctx:
                load_run_data(run_root)
            
            error_str = str(ctx.exception)
            self.assertIn(str(trials_file), error_str)
            self.assertIn("Lines parsed:", error_str)
            self.assertIn("Rejections:", error_str)


class TestLoadRunData(unittest.TestCase):
    """Integration tests for load_run_data."""
    
    def test_load_valid_run_data(self):
        """Test loading valid run data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            trials_file = run_root / "trials.jsonl"
            content = '\n'.join([
                '{"pnl": 100, "kill_ci": 0.05, "kill_rate": 0.0}',
                '{"pnl": 150, "kill_ci": 0.10, "kill_rate": 0.0}',
                '{"pnl": 75, "kill_ci": 0.02, "kill_rate": 0.5}',
            ])
            trials_file.write_text(content)
            
            loaded = load_run_data(run_root)
            
            self.assertEqual(loaded.n_observations, 3)
            self.assertEqual(loaded.trials_file, trials_file)
            
            # Test pnl arrays
            pnl_arrays = loaded.to_pnl_arrays()
            self.assertEqual(len(pnl_arrays), 3)
            self.assertEqual(pnl_arrays[0][0], 100)
            
            # Test kill flags
            kill_flags = loaded.to_kill_flags()
            self.assertEqual(len(kill_flags), 3)
            self.assertFalse(kill_flags[0])  # kill_rate = 0
            self.assertFalse(kill_flags[1])  # kill_rate = 0
            self.assertTrue(kill_flags[2])   # kill_rate = 0.5 > 0
    
    def test_load_from_nested_directory(self):
        """Test loading from nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            nested = run_root / "some" / "subdir"
            nested.mkdir(parents=True)
            trials_file = nested / "trials.jsonl"
            trials_file.write_text('{"pnl": 100, "kill_ci": 0.05}\n')
            
            loaded = load_run_data(run_root)
            
            self.assertEqual(loaded.n_observations, 1)
            self.assertEqual(loaded.trials_file, trials_file)


class TestObservationConversion(unittest.TestCase):
    """Tests for TrialObservation conversion methods."""
    
    def test_to_pnl_array(self):
        """Test conversion to PnL array."""
        obs = TrialObservation(pnl=123.45, kill_ci=0.1)
        arr = obs.to_pnl_array()
        
        self.assertEqual(len(arr), 1)
        self.assertEqual(arr[0], 123.45)
    
    def test_is_killed_property(self):
        """Test is_killed property."""
        obs_not_killed = TrialObservation(pnl=100, kill_ci=0.1, kill_rate=0.0)
        self.assertFalse(obs_not_killed.is_killed)
        
        obs_killed = TrialObservation(pnl=100, kill_ci=0.1, kill_rate=0.5)
        self.assertTrue(obs_killed.is_killed)


if __name__ == "__main__":
    unittest.main()

