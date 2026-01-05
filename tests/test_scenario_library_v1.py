"""
test_scenario_library_v1.py

Unit tests for batch_runs.phase_a.scenario_library_v1 module.

Uses stdlib unittest only - no third-party dependencies.

Tests:
- Generator determinism (same seed -> identical hashes)
- Manifest check fails on tampering
- Suite file references exist
- Scenario format validity

Run with:
    python3 -m unittest tests.test_scenario_library_v1 -v
    
Or via discover:
    python3 -m unittest discover -s tests -p 'test_*.py' -q
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from batch_runs.phase_a.scenario_library_v1 import (
    MODULE_PATH,
    __version__,
    ScenarioSpec,
    get_scenario_library,
    compute_sha256,
    compute_content_hash,
    generate_library,
    generate_suite,
    check_manifest,
    check_suite_references,
    LIBRARY_DIR,
    SUITE_PATH,
    MANIFEST_PATH,
)


class TestScenarioSpec(unittest.TestCase):
    """Tests for ScenarioSpec dataclass and YAML generation."""

    def test_to_yaml_content_produces_valid_yaml_like_content(self):
        """Test that to_yaml_content produces parseable YAML-like content."""
        spec = ScenarioSpec(
            scenario_id="test_scenario",
            category="test",
            description="Test scenario",
            base_seed=42,
            num_seeds=3,
            steps=1000,
            dt_seconds=0.25,
            risk_profile="balanced",
            init_q_tao=0.0,
            process="gbm",
            vol=0.015,
            drift=0.0,
            jump_intensity=0.0,
            jump_sigma=0.0,
            fees_bps_maker=0.0,
            fees_bps_taker=0.5,
            latency_ms=2.0,
        )
        
        content = spec.to_yaml_content()
        
        # Check required fields are present
        self.assertIn("scenario_id: test_scenario", content)
        self.assertIn("scenario_version: 1", content)
        self.assertIn("engine: rl_sim_env", content)
        self.assertIn("steps: 1000", content)
        self.assertIn("base_seed: 42", content)
        self.assertIn("vol: 0.015000", content)

    def test_to_yaml_content_includes_jump_params_for_jump_diffusion(self):
        """Test that jump parameters are included for jump_diffusion_stub process."""
        spec = ScenarioSpec(
            scenario_id="test_jump",
            category="test",
            description="Test jump scenario",
            base_seed=100,
            num_seeds=2,
            steps=500,
            dt_seconds=0.25,
            risk_profile="conservative",
            init_q_tao=5.0,
            process="jump_diffusion_stub",
            vol=0.025,
            drift=0.0,
            jump_intensity=0.0005,
            jump_sigma=0.03,
            fees_bps_maker=0.0,
            fees_bps_taker=1.0,
            latency_ms=5.0,
        )
        
        content = spec.to_yaml_content()
        
        self.assertIn("process: jump_diffusion_stub", content)
        self.assertIn("jump_intensity:", content)
        self.assertIn("jump_sigma:", content)

    def test_to_yaml_content_excludes_jump_params_for_gbm(self):
        """Test that jump parameters are excluded for gbm process."""
        spec = ScenarioSpec(
            scenario_id="test_gbm",
            category="test",
            description="Test GBM scenario",
            base_seed=200,
            num_seeds=2,
            steps=500,
            dt_seconds=0.25,
            risk_profile="balanced",
            init_q_tao=0.0,
            process="gbm",
            vol=0.015,
            drift=0.0,
            jump_intensity=0.0,
            jump_sigma=0.0,
            fees_bps_maker=0.0,
            fees_bps_taker=0.5,
            latency_ms=2.0,
        )
        
        content = spec.to_yaml_content()
        
        self.assertIn("process: gbm", content)
        self.assertNotIn("jump_intensity:", content)
        self.assertNotIn("jump_sigma:", content)

    def test_to_yaml_content_includes_shock_params(self):
        """Test that optional shock parameters are included when set."""
        spec = ScenarioSpec(
            scenario_id="test_shock",
            category="liquidity_shock",
            description="Test shock scenario",
            base_seed=300,
            num_seeds=2,
            steps=500,
            dt_seconds=0.25,
            risk_profile="conservative",
            init_q_tao=10.0,
            process="gbm",
            vol=0.018,
            drift=0.0,
            jump_intensity=0.0,
            jump_sigma=0.0,
            fees_bps_maker=0.0,
            fees_bps_taker=1.5,
            latency_ms=10.0,
            spread_shock_bps=50.0,
            depth_shock_pct=-60.0,
        )
        
        content = spec.to_yaml_content()
        
        self.assertIn("spread_shock_bps: 50.0", content)
        self.assertIn("depth_shock_pct: -60.0", content)


class TestGetScenarioLibrary(unittest.TestCase):
    """Tests for get_scenario_library function."""

    def test_returns_list_of_scenario_specs(self):
        """Test that get_scenario_library returns a list of ScenarioSpec."""
        scenarios = get_scenario_library()
        
        self.assertIsInstance(scenarios, list)
        self.assertTrue(len(scenarios) > 0)
        for s in scenarios:
            self.assertIsInstance(s, ScenarioSpec)

    def test_returns_exactly_10_scenarios(self):
        """Test that get_scenario_library returns exactly 10 scenarios."""
        scenarios = get_scenario_library()
        self.assertEqual(len(scenarios), 10)

    def test_scenario_ids_are_unique(self):
        """Test that all scenario IDs are unique."""
        scenarios = get_scenario_library()
        ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_categories_represented(self):
        """Test that all expected categories are represented."""
        scenarios = get_scenario_library()
        categories = set(s.category for s in scenarios)
        
        expected = {
            "vol_regime",
            "liquidity_shock",
            "venue_outage",
            "funding_inversion",
            "basis_spike",
        }
        
        self.assertEqual(categories, expected)

    def test_base_seeds_are_unique(self):
        """Test that all base seeds are unique (for determinism)."""
        scenarios = get_scenario_library()
        seeds = [s.base_seed for s in scenarios]
        self.assertEqual(len(seeds), len(set(seeds)))


class TestHashFunctions(unittest.TestCase):
    """Tests for hashing utility functions."""

    def test_compute_sha256_correct_hash(self):
        """Test compute_sha256 returns correct hash."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content for hashing")
            temp_path = Path(f.name)
        
        try:
            result = compute_sha256(temp_path)
            expected = hashlib.sha256(b"test content for hashing").hexdigest()
            self.assertEqual(result, expected)
        finally:
            temp_path.unlink()

    def test_compute_content_hash_correct_hash(self):
        """Test compute_content_hash returns correct hash for string content."""
        content = "test string content"
        result = compute_content_hash(content)
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.assertEqual(result, expected)


class TestGeneratorDeterminism(unittest.TestCase):
    """Tests for generator determinism - same inputs should always produce same outputs."""

    def setUp(self):
        """Create a temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_library_dir = LIBRARY_DIR

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scenario_content_is_deterministic(self):
        """Test that scenario content generation is deterministic."""
        scenarios = get_scenario_library()
        
        # Generate content twice
        contents_1 = [s.to_yaml_content() for s in scenarios]
        contents_2 = [s.to_yaml_content() for s in scenarios]
        
        # Should be identical
        self.assertEqual(contents_1, contents_2)

    def test_scenario_hashes_are_deterministic(self):
        """Test that scenario hashes are deterministic across runs."""
        scenarios = get_scenario_library()
        
        # Compute hashes twice
        hashes_1 = [compute_content_hash(s.to_yaml_content()) for s in scenarios]
        hashes_2 = [compute_content_hash(s.to_yaml_content()) for s in scenarios]
        
        # Should be identical
        self.assertEqual(hashes_1, hashes_2)

    def test_scenario_order_is_stable(self):
        """Test that scenario order is stable across calls."""
        scenarios_1 = get_scenario_library()
        scenarios_2 = get_scenario_library()
        
        ids_1 = [s.scenario_id for s in scenarios_1]
        ids_2 = [s.scenario_id for s in scenarios_2]
        
        self.assertEqual(ids_1, ids_2)


class TestManifestVerification(unittest.TestCase):
    """Tests for manifest verification (check command)."""

    def setUp(self):
        """Store original paths for restoration."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_check_manifest_passes_on_valid_library(self):
        """Test that check_manifest passes on the generated library."""
        # This test uses the actual generated library
        if not MANIFEST_PATH.exists():
            self.skipTest("Manifest not generated yet")
        
        success, errors = check_manifest(verbose=False)
        self.assertTrue(success, f"Manifest check failed: {errors}")

    def test_check_manifest_detects_missing_manifest(self):
        """Test that check_manifest detects missing manifest file."""
        with mock.patch('batch_runs.phase_a.scenario_library_v1.MANIFEST_PATH', 
                       Path(self.temp_dir) / "nonexistent.json"):
            success, errors = check_manifest(verbose=False)
            
            self.assertFalse(success)
            self.assertTrue(any("not found" in e.lower() for e in errors))


class TestManifestTamperingDetection(unittest.TestCase):
    """Tests that verify manifest check fails on tampering."""

    def setUp(self):
        """Create a temporary library with valid manifest."""
        self.temp_dir = tempfile.mkdtemp()
        self.lib_dir = Path(self.temp_dir) / "scenario_library_v1"
        self.lib_dir.mkdir(parents=True)
        
        # Create a simple scenario file
        self.scenario_file = self.lib_dir / "test_scenario.yaml"
        self.scenario_file.write_text("scenario_id: test\nscenario_version: 1\n")
        
        # Create a valid manifest
        content_hash = compute_sha256(self.scenario_file)
        self.manifest_path = self.lib_dir / "manifest_sha256.json"
        manifest = {
            "schema_version": 1,
            "generated_by": {"module": MODULE_PATH, "version": __version__},
            "scenario_count": 1,
            "files": {"test_scenario.yaml": content_hash},
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tamper_detection_content_change(self):
        """Test that changing file content is detected."""
        # Tamper with the file
        self.scenario_file.write_text("scenario_id: tampered\nscenario_version: 1\n")
        
        with mock.patch('batch_runs.phase_a.scenario_library_v1.LIBRARY_DIR', self.lib_dir):
            with mock.patch('batch_runs.phase_a.scenario_library_v1.MANIFEST_PATH', self.manifest_path):
                success, errors = check_manifest(verbose=False)
        
        self.assertFalse(success)
        self.assertTrue(any("mismatch" in e.lower() for e in errors))

    def test_tamper_detection_extra_file(self):
        """Test that extra files are detected."""
        # Add an extra file
        extra_file = self.lib_dir / "extra.yaml"
        extra_file.write_text("extra content")
        
        with mock.patch('batch_runs.phase_a.scenario_library_v1.LIBRARY_DIR', self.lib_dir):
            with mock.patch('batch_runs.phase_a.scenario_library_v1.MANIFEST_PATH', self.manifest_path):
                success, errors = check_manifest(verbose=False)
        
        self.assertFalse(success)
        self.assertTrue(any("extra" in e.lower() for e in errors))

    def test_tamper_detection_missing_file(self):
        """Test that missing files are detected."""
        # Delete the scenario file
        self.scenario_file.unlink()
        
        with mock.patch('batch_runs.phase_a.scenario_library_v1.LIBRARY_DIR', self.lib_dir):
            with mock.patch('batch_runs.phase_a.scenario_library_v1.MANIFEST_PATH', self.manifest_path):
                success, errors = check_manifest(verbose=False)
        
        self.assertFalse(success)
        self.assertTrue(any("missing" in e.lower() for e in errors))


class TestSuiteReferences(unittest.TestCase):
    """Tests for suite file reference validation."""

    def test_suite_references_exist(self):
        """Test that all scenario files referenced in suite exist."""
        if not SUITE_PATH.exists():
            self.skipTest("Suite not generated yet")
        
        success, errors = check_suite_references(verbose=False)
        self.assertTrue(success, f"Suite reference check failed: {errors}")

    def test_suite_references_valid_format(self):
        """Test that suite file has valid format."""
        if not SUITE_PATH.exists():
            self.skipTest("Suite not generated yet")
        
        with open(SUITE_PATH) as f:
            content = f.read()
        
        # Check required fields
        self.assertIn("suite_id:", content)
        self.assertIn("suite_version:", content)
        self.assertIn("scenarios:", content)
        self.assertIn("- path:", content)


class TestSmokeCLI(unittest.TestCase):
    """Tests for smoke CLI command construction."""

    def test_smoke_command_uses_seed(self):
        """Test that smoke command respects seed parameter."""
        # This is a unit test for the smoke run parameters
        # We don't actually run smoke here (that's an integration test)
        from batch_runs.phase_a.scenario_library_v1 import run_smoke
        
        # Verify the function signature accepts seed
        import inspect
        sig = inspect.signature(run_smoke)
        params = list(sig.parameters.keys())
        
        self.assertIn("seed", params)
        self.assertIn("out_dir", params)


class TestModuleMetadata(unittest.TestCase):
    """Tests for module metadata consistency."""

    def test_module_path_constant(self):
        """Test MODULE_PATH constant is correct."""
        self.assertEqual(MODULE_PATH, "batch_runs.phase_a.scenario_library_v1")

    def test_version_format(self):
        """Test __version__ has valid semver format."""
        import re
        pattern = r"^\d+\.\d+\.\d+$"
        self.assertTrue(
            re.match(pattern, __version__),
            f"Version '{__version__}' does not match semver pattern"
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for generate + check workflow."""

    def test_generate_then_check_succeeds(self):
        """Test that generate followed by check succeeds."""
        # Use the actual generated library
        if not MANIFEST_PATH.exists():
            self.skipTest("Library not generated yet")
        
        success, errors = check_manifest(verbose=False)
        self.assertTrue(success, f"Integration check failed: {errors}")

    def test_regenerate_produces_identical_hashes(self):
        """Test that regenerating with same seed produces identical file hashes."""
        if not MANIFEST_PATH.exists():
            self.skipTest("Library not generated yet")
        
        # Read original manifest
        with open(MANIFEST_PATH) as f:
            original_manifest = json.load(f)
        
        # Get the seed from original manifest (or use default)
        seed = original_manifest.get("seed", 20260105)
        
        # Regenerate with the same seed (this overwrites files)
        new_manifest = generate_library(seed=seed, verbose=False)
        
        # Compare file hashes
        self.assertEqual(
            original_manifest["files"],
            new_manifest["files"],
            "Regeneration produced different hashes - determinism violated"
        )


class TestSeedDeterminism(unittest.TestCase):
    """Tests for seed-based determinism of generate command."""

    def setUp(self):
        """Create temporary directories for test outputs."""
        self.temp_dir_1 = tempfile.mkdtemp()
        self.temp_dir_2 = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir_1, ignore_errors=True)
        shutil.rmtree(self.temp_dir_2, ignore_errors=True)

    def test_same_seed_produces_identical_manifest_hash(self):
        """Test that generate --seed X produces the same manifest hash across two runs."""
        seed = 20260105
        out_dir_1 = Path(self.temp_dir_1) / "lib1"
        out_dir_2 = Path(self.temp_dir_2) / "lib2"
        
        # Generate twice with the same seed
        manifest_1 = generate_library(seed=seed, out_dir=out_dir_1, verbose=False)
        manifest_2 = generate_library(seed=seed, out_dir=out_dir_2, verbose=False)
        
        # Compare file hashes
        self.assertEqual(
            manifest_1["files"],
            manifest_2["files"],
            "Same seed produced different file hashes"
        )
        
        # Compare seeds in manifest
        self.assertEqual(manifest_1["seed"], seed)
        self.assertEqual(manifest_2["seed"], seed)
        
        # Compare full manifest content hash
        manifest_path_1 = out_dir_1 / "manifest_sha256.json"
        manifest_path_2 = out_dir_2 / "manifest_sha256.json"
        
        hash_1 = compute_sha256(manifest_path_1)
        hash_2 = compute_sha256(manifest_path_2)
        
        self.assertEqual(hash_1, hash_2, "Same seed produced different manifest hashes")

    def test_manifest_includes_seed(self):
        """Test that generated manifest includes the seed."""
        seed = 12345
        out_dir = Path(self.temp_dir_1) / "lib"
        
        manifest = generate_library(seed=seed, out_dir=out_dir, verbose=False)
        
        self.assertIn("seed", manifest)
        self.assertEqual(manifest["seed"], seed)
        
        # Also verify it's in the file
        manifest_path = out_dir / "manifest_sha256.json"
        with open(manifest_path) as f:
            saved_manifest = json.load(f)
        
        self.assertEqual(saved_manifest["seed"], seed)

    def test_generate_with_different_seeds_mentioned_in_readme(self):
        """Test that README includes the seed used for generation."""
        seed = 99999
        out_dir = Path(self.temp_dir_1) / "lib"
        
        generate_library(seed=seed, out_dir=out_dir, verbose=False)
        
        readme_path = out_dir / "README.md"
        with open(readme_path) as f:
            readme_content = f.read()
        
        # Check seed is mentioned in README
        self.assertIn(str(seed), readme_content)
        self.assertIn("Seed:", readme_content)


class TestGenerateCLI(unittest.TestCase):
    """Tests for generate CLI command."""

    def test_generate_cli_accepts_seed(self):
        """Test that generate CLI accepts --seed argument."""
        from batch_runs.phase_a.scenario_library_v1 import main
        import sys
        
        # Test that --seed is recognized (mock to avoid actual generation)
        with mock.patch('sys.argv', ['scenario_library_v1', 'generate', '--seed', '12345']):
            with mock.patch('batch_runs.phase_a.scenario_library_v1.generate_library') as mock_gen:
                with mock.patch('batch_runs.phase_a.scenario_library_v1.generate_suite'):
                    with mock.patch('batch_runs.phase_a.scenario_library_v1.check_manifest', return_value=(True, [])):
                        with mock.patch('batch_runs.phase_a.scenario_library_v1.check_suite_references', return_value=(True, [])):
                            mock_gen.return_value = {"files": {}}
                            main()
                            
                            # Verify generate_library was called with seed
                            mock_gen.assert_called_once()
                            call_kwargs = mock_gen.call_args[1]
                            self.assertEqual(call_kwargs["seed"], 12345)


if __name__ == "__main__":
    unittest.main()

