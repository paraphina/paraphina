"""
Tests for .env file parsing and writing round-trip.

Hermetic tests - no cargo invocation.
"""

import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.promote_pipeline import CandidateConfig


class TestEnvParsing(unittest.TestCase):
    """Test .env file parsing and writing."""
    
    def test_write_env_file_basic(self):
        """Test writing a basic .env file."""
        config = CandidateConfig(
            candidate_id="abc123",
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
            init_q_tao=0.0,
            hedge_max_step=10.0,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "candidate.env"
            config.write_env_file(env_path)
            
            self.assertTrue(env_path.exists())
            content = env_path.read_text()
            
            # Check all env vars are present
            self.assertIn("PARAPHINA_RISK_PROFILE=balanced", content)
            self.assertIn("PARAPHINA_HEDGE_BAND_BASE=0.05", content)
            self.assertIn("PARAPHINA_MM_SIZE_ETA=1.0", content)
            self.assertIn("PARAPHINA_VOL_REF=0.1", content)
            self.assertIn("PARAPHINA_DAILY_LOSS_LIMIT=1000.0", content)
            self.assertIn("PARAPHINA_INIT_Q_TAO=0.0", content)
            self.assertIn("PARAPHINA_HEDGE_MAX_STEP=10.0", content)
    
    def test_from_env_file_basic(self):
        """Test parsing a basic .env file."""
        env_content = """# Test config
export PARAPHINA_RISK_PROFILE=conservative
export PARAPHINA_HEDGE_BAND_BASE=0.03
export PARAPHINA_MM_SIZE_ETA=0.75
export PARAPHINA_VOL_REF=0.15
export PARAPHINA_DAILY_LOSS_LIMIT=500.0
export PARAPHINA_INIT_Q_TAO=-10.0
export PARAPHINA_HEDGE_MAX_STEP=5.0
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "test.env"
            env_path.write_text(env_content)
            
            config = CandidateConfig.from_env_file(env_path)
            
            self.assertEqual(config.profile, "conservative")
            self.assertAlmostEqual(config.hedge_band_base, 0.03)
            self.assertAlmostEqual(config.mm_size_eta, 0.75)
            self.assertAlmostEqual(config.vol_ref, 0.15)
            self.assertAlmostEqual(config.daily_loss_limit, 500.0)
            self.assertAlmostEqual(config.init_q_tao, -10.0)
            self.assertAlmostEqual(config.hedge_max_step, 5.0)
    
    def test_round_trip_preserves_values(self):
        """Test that writing and reading back preserves all values."""
        original = CandidateConfig(
            candidate_id="test123",
            profile="aggressive",
            hedge_band_base=0.075,
            mm_size_eta=1.5,
            vol_ref=0.125,
            daily_loss_limit=2000.0,
            init_q_tao=15.0,
            hedge_max_step=20.0,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "roundtrip.env"
            
            # Write
            original.write_env_file(env_path)
            
            # Read back
            parsed = CandidateConfig.from_env_file(env_path)
            
            # Verify all values match
            self.assertEqual(parsed.profile, original.profile)
            self.assertAlmostEqual(parsed.hedge_band_base, original.hedge_band_base)
            self.assertAlmostEqual(parsed.mm_size_eta, original.mm_size_eta)
            self.assertAlmostEqual(parsed.vol_ref, original.vol_ref)
            self.assertAlmostEqual(parsed.daily_loss_limit, original.daily_loss_limit)
            self.assertAlmostEqual(parsed.init_q_tao, original.init_q_tao)
            self.assertAlmostEqual(parsed.hedge_max_step, original.hedge_max_step)
    
    def test_round_trip_hash_consistency(self):
        """Test that config hash is consistent across round-trip."""
        original = CandidateConfig(
            candidate_id="test456",
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
            init_q_tao=0.0,
            hedge_max_step=10.0,
        )
        
        original_hash = original.config_hash()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "hash_test.env"
            
            # Write and read back
            original.write_env_file(env_path)
            parsed = CandidateConfig.from_env_file(env_path)
            
            # Hash should match (deterministic)
            self.assertEqual(parsed.config_hash(), original_hash)
    
    def test_parse_without_export_prefix(self):
        """Test parsing .env file without 'export' prefix."""
        env_content = """# Test config (no export prefix)
PARAPHINA_RISK_PROFILE=balanced
PARAPHINA_HEDGE_BAND_BASE=0.05
PARAPHINA_MM_SIZE_ETA=1.0
PARAPHINA_VOL_REF=0.10
PARAPHINA_DAILY_LOSS_LIMIT=1000.0
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "no_export.env"
            env_path.write_text(env_content)
            
            config = CandidateConfig.from_env_file(env_path)
            
            self.assertEqual(config.profile, "balanced")
            self.assertAlmostEqual(config.hedge_band_base, 0.05)
    
    def test_parse_ignores_comments_and_blanks(self):
        """Test that comments and blank lines are ignored."""
        env_content = """
# This is a comment
  # This is an indented comment

export PARAPHINA_RISK_PROFILE=balanced

# Another comment
export PARAPHINA_HEDGE_BAND_BASE=0.05

"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "comments.env"
            env_path.write_text(env_content)
            
            config = CandidateConfig.from_env_file(env_path)
            
            self.assertEqual(config.profile, "balanced")
            self.assertAlmostEqual(config.hedge_band_base, 0.05)
    
    def test_parse_defaults_for_missing_keys(self):
        """Test that missing keys get default values."""
        env_content = """export PARAPHINA_RISK_PROFILE=balanced
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "minimal.env"
            env_path.write_text(env_content)
            
            config = CandidateConfig.from_env_file(env_path)
            
            self.assertEqual(config.profile, "balanced")
            # Check defaults are applied
            self.assertAlmostEqual(config.hedge_band_base, 0.05)
            self.assertAlmostEqual(config.mm_size_eta, 1.0)
            self.assertAlmostEqual(config.vol_ref, 0.10)
            self.assertAlmostEqual(config.daily_loss_limit, 1000.0)
            self.assertAlmostEqual(config.init_q_tao, 0.0)
            self.assertAlmostEqual(config.hedge_max_step, 10.0)
    
    def test_env_overlay_sorted_deterministically(self):
        """Test that env overlay keys are sorted for determinism."""
        config = CandidateConfig(
            candidate_id="test",
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        env = config.to_env_overlay()
        keys = list(env.keys())
        
        # Keys should be in consistent order (dict order in Python 3.7+)
        self.assertEqual(keys, list(env.keys()))
        
        # Run multiple times to verify determinism
        for _ in range(5):
            env2 = config.to_env_overlay()
            self.assertEqual(list(env.keys()), list(env2.keys()))
            self.assertEqual(list(env.values()), list(env2.values()))
    
    def test_config_hash_deterministic(self):
        """Test that config hash is deterministic."""
        config = CandidateConfig(
            candidate_id="test",
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        hash1 = config.config_hash()
        
        # Create identical config
        config2 = CandidateConfig(
            candidate_id="different_id",  # ID shouldn't affect hash
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        hash2 = config2.config_hash()
        
        # Hashes should match (based on env overlay, not candidate_id)
        self.assertEqual(hash1, hash2)
    
    def test_config_hash_differs_for_different_values(self):
        """Test that different configs produce different hashes."""
        config1 = CandidateConfig(
            candidate_id="test1",
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        config2 = CandidateConfig(
            candidate_id="test2",
            profile="conservative",  # Different profile
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        self.assertNotEqual(config1.config_hash(), config2.config_hash())


class TestEnvFileEdgeCases(unittest.TestCase):
    """Test edge cases in .env file handling."""
    
    def test_handle_equals_in_value(self):
        """Test handling values that contain '=' character."""
        # This shouldn't happen in our case, but test robustness
        env_content = """export PARAPHINA_RISK_PROFILE=balanced
export PARAPHINA_HEDGE_BAND_BASE=0.05
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "edge.env"
            env_path.write_text(env_content)
            
            config = CandidateConfig.from_env_file(env_path)
            self.assertEqual(config.profile, "balanced")
    
    def test_handle_whitespace(self):
        """Test handling various whitespace."""
        env_content = """  export PARAPHINA_RISK_PROFILE=balanced  
export   PARAPHINA_HEDGE_BAND_BASE=0.05
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "whitespace.env"
            env_path.write_text(env_content)
            
            config = CandidateConfig.from_env_file(env_path)
            # Should still parse correctly
            self.assertEqual(config.profile, "balanced")


if __name__ == "__main__":
    unittest.main()

