"""
Tests for suite YAML parsing and env_overrides merging.

Hermetic tests - no cargo invocation.
"""

import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.promote_pipeline import (
    parse_suite_yaml,
    merge_env_overlays,
)


class TestParseSuiteYaml(unittest.TestCase):
    """Test suite YAML parsing."""
    
    def test_parse_basic_suite(self):
        """Test parsing a basic suite YAML with path-based scenarios."""
        yaml_content = """
suite_id: research_v1
suite_version: 1
repeat_runs: 1

out_dir: runs/research

scenarios:
  - path: scenarios/v1/synth_baseline.yaml
  - path: scenarios/v1/synth_jump.yaml
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "test_suite.yaml"
            suite_path.write_text(yaml_content)
            
            result = parse_suite_yaml(suite_path)
            
            self.assertEqual(result.get("suite_id"), "research_v1")
            self.assertEqual(result.get("out_dir"), "runs/research")
            self.assertEqual(result.get("repeat_runs"), 1)
            self.assertEqual(len(result.get("scenarios", [])), 2)
            self.assertEqual(result["scenarios"][0].get("path"), "scenarios/v1/synth_baseline.yaml")
    
    def test_parse_inline_env_overrides(self):
        """Test parsing suite with inline env_overrides."""
        yaml_content = """
suite_id: adversarial_regression_v1
suite_version: 1
repeat_runs: 2
out_dir: runs/adversarial_regression

scenarios:
  - id: smoke_balanced_0
    seed: 42
    profile: balanced
    env_overrides:
      PARAPHINA_RISK_PROFILE: balanced
      PARAPHINA_VOL_REF: "0.2000"
      PARAPHINA_MM_SIZE_ETA: "0.5000"
      PARAPHINA_INIT_Q_TAO: "-30.00"
      PARAPHINA_DAILY_LOSS_LIMIT: "1000.00"

  - id: smoke_aggressive_2
    seed: 44
    profile: aggressive
    env_overrides:
      PARAPHINA_RISK_PROFILE: aggressive
      PARAPHINA_VOL_REF: "0.2000"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "adversarial.yaml"
            suite_path.write_text(yaml_content)
            
            result = parse_suite_yaml(suite_path)
            
            self.assertEqual(result.get("suite_id"), "adversarial_regression_v1")
            self.assertEqual(result.get("repeat_runs"), 2)
            self.assertEqual(len(result.get("scenarios", [])), 2)
            
            # Check first scenario
            scenario1 = result["scenarios"][0]
            self.assertEqual(scenario1.get("id"), "smoke_balanced_0")
            self.assertEqual(scenario1.get("seed"), 42)
            self.assertEqual(scenario1.get("profile"), "balanced")
            
            env_overrides = scenario1.get("env_overrides", {})
            self.assertEqual(env_overrides.get("PARAPHINA_RISK_PROFILE"), "balanced")
            self.assertEqual(env_overrides.get("PARAPHINA_VOL_REF"), "0.2000")
            self.assertEqual(env_overrides.get("PARAPHINA_INIT_Q_TAO"), "-30.00")
    
    def test_parse_empty_file(self):
        """Test parsing empty suite file returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "empty.yaml"
            suite_path.write_text("")
            
            result = parse_suite_yaml(suite_path)
            
            self.assertEqual(result.get("scenarios", []), [])
    
    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file returns empty dict."""
        result = parse_suite_yaml(Path("/nonexistent/path.yaml"))
        self.assertEqual(result, {})
    
    def test_parse_mixed_scenarios(self):
        """Test parsing suite with both path and inline scenarios."""
        yaml_content = """
suite_id: mixed_v1
out_dir: runs/mixed

scenarios:
  - path: scenarios/v1/synth_baseline.yaml
  - id: stress_test_1
    seed: 100
    env_overrides:
      PARAPHINA_VOL_REF: "0.3"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "mixed.yaml"
            suite_path.write_text(yaml_content)
            
            result = parse_suite_yaml(suite_path)
            
            self.assertEqual(len(result.get("scenarios", [])), 2)
            self.assertIn("path", result["scenarios"][0])
            self.assertIn("id", result["scenarios"][1])


class TestMergeEnvOverlays(unittest.TestCase):
    """Test env_overrides merging rules."""
    
    def test_candidate_env_only(self):
        """Test merge with only candidate env."""
        candidate_env = {
            "PARAPHINA_RISK_PROFILE": "balanced",
            "PARAPHINA_HEDGE_BAND_BASE": "0.05",
        }
        suite_env = {}
        
        result = merge_env_overlays(candidate_env, suite_env)
        
        self.assertEqual(result, candidate_env)
    
    def test_suite_env_overrides_candidate(self):
        """Test that suite env_overrides take precedence over candidate."""
        candidate_env = {
            "PARAPHINA_RISK_PROFILE": "balanced",
            "PARAPHINA_VOL_REF": "0.10",
            "PARAPHINA_MM_SIZE_ETA": "1.0",
        }
        suite_env = {
            "PARAPHINA_VOL_REF": "0.20",  # Override
            "PARAPHINA_INIT_Q_TAO": "-30.0",  # New key
        }
        
        result = merge_env_overlays(candidate_env, suite_env)
        
        # Suite overrides candidate
        self.assertEqual(result["PARAPHINA_VOL_REF"], "0.20")
        # Suite adds new keys
        self.assertEqual(result["PARAPHINA_INIT_Q_TAO"], "-30.0")
        # Candidate values preserved if not overridden
        self.assertEqual(result["PARAPHINA_RISK_PROFILE"], "balanced")
        self.assertEqual(result["PARAPHINA_MM_SIZE_ETA"], "1.0")
    
    def test_merge_is_deterministic(self):
        """Test that merge produces deterministic results."""
        candidate_env = {"A": "1", "B": "2", "C": "3"}
        suite_env = {"B": "22", "D": "4"}
        
        result1 = merge_env_overlays(candidate_env, suite_env)
        result2 = merge_env_overlays(candidate_env, suite_env)
        
        self.assertEqual(result1, result2)
    
    def test_merge_does_not_mutate_inputs(self):
        """Test that merge does not mutate input dictionaries."""
        candidate_env = {"A": "1"}
        suite_env = {"B": "2"}
        original_candidate = dict(candidate_env)
        original_suite = dict(suite_env)
        
        merge_env_overlays(candidate_env, suite_env)
        
        self.assertEqual(candidate_env, original_candidate)
        self.assertEqual(suite_env, original_suite)
    
    def test_merge_with_empty_inputs(self):
        """Test merge with empty inputs."""
        self.assertEqual(merge_env_overlays({}, {}), {})
        self.assertEqual(merge_env_overlays({"A": "1"}, {}), {"A": "1"})
        self.assertEqual(merge_env_overlays({}, {"A": "1"}), {"A": "1"})
    
    def test_adversarial_overrides_example(self):
        """Test realistic adversarial suite override scenario."""
        # Candidate config from tuning
        candidate_env = {
            "PARAPHINA_RISK_PROFILE": "balanced",
            "PARAPHINA_HEDGE_BAND_BASE": "0.05",
            "PARAPHINA_MM_SIZE_ETA": "1.0",
            "PARAPHINA_VOL_REF": "0.10",
            "PARAPHINA_DAILY_LOSS_LIMIT": "1000.0",
        }
        
        # Adversarial scenario env_overrides (stress test)
        suite_env = {
            "PARAPHINA_VOL_REF": "0.20",  # 2x stress
            "PARAPHINA_MM_SIZE_ETA": "0.5",  # Reduced size
            "PARAPHINA_INIT_Q_TAO": "-30.0",  # Starting position
        }
        
        result = merge_env_overlays(candidate_env, suite_env)
        
        # Verify adversarial params override candidate
        self.assertEqual(result["PARAPHINA_VOL_REF"], "0.20")
        self.assertEqual(result["PARAPHINA_MM_SIZE_ETA"], "0.5")
        self.assertEqual(result["PARAPHINA_INIT_Q_TAO"], "-30.0")
        
        # Verify candidate params preserved where not overridden
        self.assertEqual(result["PARAPHINA_RISK_PROFILE"], "balanced")
        self.assertEqual(result["PARAPHINA_HEDGE_BAND_BASE"], "0.05")
        self.assertEqual(result["PARAPHINA_DAILY_LOSS_LIMIT"], "1000.0")


class TestEnvMergePriority(unittest.TestCase):
    """Test that env merge follows correct priority order."""
    
    def test_priority_order(self):
        """
        Test priority: suite env_overrides > candidate config.
        
        This is critical for adversarial testing where the suite
        must be able to override candidate settings to stress test.
        """
        candidate = {"KEY": "candidate_value"}
        suite = {"KEY": "suite_value"}
        
        result = merge_env_overlays(candidate, suite)
        
        # Suite wins
        self.assertEqual(result["KEY"], "suite_value")
    
    def test_additive_keys(self):
        """Test that keys from both sources are included."""
        candidate = {"ONLY_IN_CANDIDATE": "1"}
        suite = {"ONLY_IN_SUITE": "2"}
        
        result = merge_env_overlays(candidate, suite)
        
        self.assertIn("ONLY_IN_CANDIDATE", result)
        self.assertIn("ONLY_IN_SUITE", result)


if __name__ == "__main__":
    unittest.main()

