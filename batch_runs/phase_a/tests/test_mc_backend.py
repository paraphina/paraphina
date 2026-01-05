"""
Tests for MC backend selection logic in the promotion pipeline.

These tests verify:
1. Backend selection: direct (shards=1) vs sharded (shards>1)
2. Validation: mc_shards > mc_runs is rejected
3. Metadata: trial records include correct mc_backend info
"""

import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Minimal reproduction of relevant classes for testing
# This avoids needing to run Rust binaries in unit tests

@dataclass
class MockCandidateConfig:
    """Mock candidate config for testing."""
    candidate_id: str
    profile: str = "balanced"
    hedge_band_base: float = 0.05
    mm_size_eta: float = 1.0
    vol_ref: float = 0.10
    daily_loss_limit: float = 1000.0


@dataclass
class MockTrialResult:
    """Mock trial result with mc_backend fields."""
    trial_id: str
    mc_backend: str = "direct"
    mc_shards: int = 1
    mc_runs: int = 10
    mc_dir: str = "mc"
    mc_scale_manifest: Optional[str] = None
    
    def to_mc_backend_dict(self) -> Dict[str, Any]:
        """Return mc_backend metadata as dict."""
        return {
            "backend": self.mc_backend,
            "shards": self.mc_shards,
            "runs": self.mc_runs,
            "dir": self.mc_dir,
            "scale_manifest": self.mc_scale_manifest,
        }


def select_mc_backend(mc_shards: int) -> str:
    """
    Determine MC backend based on shard count.
    
    This mirrors the logic in promote_pipeline.py.
    """
    return "sharded" if mc_shards > 1 else "direct"


def validate_mc_shards(mc_shards: int, mc_runs: int) -> tuple:
    """
    Validate mc_shards parameter.
    
    Returns:
        (is_valid, error_message)
    """
    if mc_shards < 1:
        return False, f"--mc-shards must be >= 1, got {mc_shards}"
    if mc_shards > mc_runs:
        return False, f"--mc-shards ({mc_shards}) cannot exceed --mc-runs ({mc_runs})"
    return True, None


class TestMCBackendSelection(unittest.TestCase):
    """Test MC backend selection logic."""
    
    def test_direct_backend_when_shards_is_one(self):
        """Verify direct backend when --mc-shards=1."""
        backend = select_mc_backend(mc_shards=1)
        self.assertEqual(backend, "direct")
    
    def test_sharded_backend_when_shards_greater_than_one(self):
        """Verify sharded backend when --mc-shards>1."""
        backend = select_mc_backend(mc_shards=2)
        self.assertEqual(backend, "sharded")
        
        backend = select_mc_backend(mc_shards=3)
        self.assertEqual(backend, "sharded")
        
        backend = select_mc_backend(mc_shards=10)
        self.assertEqual(backend, "sharded")
    
    def test_default_is_direct(self):
        """Verify default (1 shard) uses direct backend."""
        backend = select_mc_backend(mc_shards=1)
        self.assertEqual(backend, "direct")


class TestMCShardsValidation(unittest.TestCase):
    """Test --mc-shards validation logic."""
    
    def test_valid_shards_equal_to_runs(self):
        """Verify mc_shards == mc_runs is valid."""
        is_valid, error = validate_mc_shards(mc_shards=10, mc_runs=10)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_valid_shards_less_than_runs(self):
        """Verify mc_shards < mc_runs is valid."""
        is_valid, error = validate_mc_shards(mc_shards=3, mc_runs=12)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_valid_single_shard(self):
        """Verify mc_shards=1 is valid."""
        is_valid, error = validate_mc_shards(mc_shards=1, mc_runs=50)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_invalid_shards_exceeds_runs(self):
        """Verify mc_shards > mc_runs is rejected with clear error."""
        is_valid, error = validate_mc_shards(mc_shards=15, mc_runs=10)
        self.assertFalse(is_valid)
        self.assertIn("cannot exceed", error)
        self.assertIn("15", error)
        self.assertIn("10", error)
    
    def test_invalid_zero_shards(self):
        """Verify mc_shards=0 is rejected."""
        is_valid, error = validate_mc_shards(mc_shards=0, mc_runs=10)
        self.assertFalse(is_valid)
        self.assertIn(">= 1", error)
    
    def test_invalid_negative_shards(self):
        """Verify negative mc_shards is rejected."""
        is_valid, error = validate_mc_shards(mc_shards=-1, mc_runs=10)
        self.assertFalse(is_valid)
        self.assertIn(">= 1", error)


class TestTrialMetadata(unittest.TestCase):
    """Test trial result mc_backend metadata."""
    
    def test_direct_mode_metadata(self):
        """Verify trial metadata for direct mode."""
        result = MockTrialResult(
            trial_id="trial_0000_abc123",
            mc_backend="direct",
            mc_shards=1,
            mc_runs=50,
            mc_dir="trial_0000_abc123/mc",
            mc_scale_manifest=None,
        )
        
        metadata = result.to_mc_backend_dict()
        
        self.assertEqual(metadata["backend"], "direct")
        self.assertEqual(metadata["shards"], 1)
        self.assertEqual(metadata["runs"], 50)
        self.assertIsNone(metadata["scale_manifest"])
    
    def test_sharded_mode_metadata(self):
        """Verify trial metadata for sharded mode."""
        result = MockTrialResult(
            trial_id="trial_0001_def456",
            mc_backend="sharded",
            mc_shards=3,
            mc_runs=12,
            mc_dir="trial_0001_def456/mc",
            mc_scale_manifest="trial_0001_def456/mc/mc_scale_manifest.json",
        )
        
        metadata = result.to_mc_backend_dict()
        
        self.assertEqual(metadata["backend"], "sharded")
        self.assertEqual(metadata["shards"], 3)
        self.assertEqual(metadata["runs"], 12)
        self.assertIsNotNone(metadata["scale_manifest"])
        self.assertIn("mc_scale_manifest.json", metadata["scale_manifest"])
    
    def test_metadata_contains_all_required_fields(self):
        """Verify all required fields are present in metadata."""
        result = MockTrialResult(
            trial_id="trial_0002_ghi789",
            mc_backend="direct",
            mc_shards=1,
            mc_runs=10,
            mc_dir="mc",
        )
        
        metadata = result.to_mc_backend_dict()
        
        required_fields = ["backend", "shards", "runs", "dir", "scale_manifest"]
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing field: {field}")


class TestMCScaleIntegration(unittest.TestCase):
    """Test mc_scale module importability and function signature."""
    
    def test_run_mc_scale_is_importable(self):
        """Verify run_mc_scale can be imported from mc_scale module."""
        try:
            from batch_runs.phase_a.mc_scale import run_mc_scale
            # Verify it's callable
            self.assertTrue(callable(run_mc_scale))
        except ImportError as e:
            self.fail(f"Failed to import run_mc_scale: {e}")
    
    def test_run_mc_scale_accepts_expected_parameters(self):
        """Verify run_mc_scale accepts the expected parameters."""
        from batch_runs.phase_a.mc_scale import run_mc_scale
        import inspect
        
        sig = inspect.signature(run_mc_scale)
        params = list(sig.parameters.keys())
        
        # Required parameters
        expected_params = ["out_dir", "seed", "runs", "shards", "ticks"]
        for param in expected_params:
            self.assertIn(param, params, f"Missing required parameter: {param}")
        
        # Optional parameters
        optional_params = ["monte_carlo_bin", "sim_eval_bin", "quiet"]
        for param in optional_params:
            self.assertIn(param, params, f"Missing optional parameter: {param}")
    
    def test_compute_shard_ranges_is_importable(self):
        """Verify compute_shard_ranges can be imported."""
        from batch_runs.phase_a.mc_scale import compute_shard_ranges
        
        # Test basic functionality
        ranges = compute_shard_ranges(runs=12, shards=3)
        self.assertEqual(len(ranges), 3)
        
        # Verify contiguous
        total = sum(r["end"] - r["start"] for r in ranges)
        self.assertEqual(total, 12)


class TestBackendSelectionIntegration(unittest.TestCase):
    """
    Integration tests for backend selection.
    
    These tests verify that the TrialResult dataclass properly
    stores mc_backend metadata based on configuration.
    """
    
    def test_trial_result_fields_exist_in_real_class(self):
        """Verify TrialResult in promote_pipeline has mc_backend fields."""
        try:
            from batch_runs.phase_a.promote_pipeline import TrialResult
            import dataclasses
            
            field_names = [f.name for f in dataclasses.fields(TrialResult)]
            
            expected_fields = [
                "mc_backend",
                "mc_shards", 
                "mc_runs",
                "mc_dir",
                "mc_scale_manifest",
            ]
            
            for field in expected_fields:
                self.assertIn(field, field_names, f"TrialResult missing field: {field}")
                
        except ImportError as e:
            self.fail(f"Failed to import TrialResult: {e}")


if __name__ == "__main__":
    unittest.main()

