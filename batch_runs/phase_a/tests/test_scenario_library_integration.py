"""
test_scenario_library_integration.py

Unit tests for Scenario Library v1 integration into the Phase A promotion pipeline.

Tests:
- Suite selection logic (smoke vs full)
- --skip-scenario-library removes scenario library from pipeline
- Skip reason is recorded in promotion record structure
- ScenarioLibraryConfig dataclass behavior
- ScenarioLibraryResult serialization

Uses stdlib unittest only - no third-party dependencies.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from batch_runs.phase_a.promote_pipeline import (
    ScenarioLibraryConfig,
    ScenarioLibraryResult,
    get_scenario_library_suite,
    check_scenario_library_integrity,
    validate_scenario_library_suite,
    SCENARIO_LIBRARY_FULL_SUITE,
    SCENARIO_LIBRARY_SMOKE_SUITE,
    SCENARIO_LIBRARY_DIR,
    SCENARIO_LIBRARY_MANIFEST,
)


class TestGetScenarioLibrarySuite(unittest.TestCase):
    """Tests for get_scenario_library_suite function."""

    def test_smoke_mode_returns_smoke_suite(self):
        """Test that smoke=True returns the smoke suite path."""
        result = get_scenario_library_suite(smoke=True, override_path=None)
        self.assertEqual(result, SCENARIO_LIBRARY_SMOKE_SUITE)

    def test_full_mode_returns_full_suite(self):
        """Test that smoke=False returns the full suite path."""
        result = get_scenario_library_suite(smoke=False, override_path=None)
        self.assertEqual(result, SCENARIO_LIBRARY_FULL_SUITE)

    def test_override_path_takes_precedence(self):
        """Test that override_path takes precedence over smoke mode."""
        custom_path = Path("/custom/suite.yaml")
        
        result = get_scenario_library_suite(smoke=True, override_path=custom_path)
        self.assertEqual(result, custom_path)
        
        result = get_scenario_library_suite(smoke=False, override_path=custom_path)
        self.assertEqual(result, custom_path)


class TestScenarioLibraryConfig(unittest.TestCase):
    """Tests for ScenarioLibraryConfig dataclass."""

    def test_default_is_enabled(self):
        """Test that scenario library is enabled by default."""
        config = ScenarioLibraryConfig()
        self.assertTrue(config.enabled)

    def test_default_is_not_smoke(self):
        """Test that default is not smoke mode."""
        config = ScenarioLibraryConfig()
        self.assertFalse(config.smoke_mode)

    def test_default_suite_path_is_none(self):
        """Test that default suite_path is None (auto-select)."""
        config = ScenarioLibraryConfig()
        self.assertIsNone(config.suite_path)

    def test_get_suite_path_returns_smoke_when_smoke_mode(self):
        """Test get_suite_path returns smoke suite when smoke_mode=True."""
        config = ScenarioLibraryConfig(smoke_mode=True)
        self.assertEqual(config.get_suite_path(), SCENARIO_LIBRARY_SMOKE_SUITE)

    def test_get_suite_path_returns_full_when_not_smoke(self):
        """Test get_suite_path returns full suite when smoke_mode=False."""
        config = ScenarioLibraryConfig(smoke_mode=False)
        self.assertEqual(config.get_suite_path(), SCENARIO_LIBRARY_FULL_SUITE)

    def test_get_suite_path_respects_override(self):
        """Test get_suite_path respects suite_path override."""
        custom = Path("/custom/path.yaml")
        config = ScenarioLibraryConfig(suite_path=custom, smoke_mode=True)
        self.assertEqual(config.get_suite_path(), custom)


class TestScenarioLibraryResult(unittest.TestCase):
    """Tests for ScenarioLibraryResult dataclass and serialization."""

    def test_default_values(self):
        """Test that default values are sensible."""
        result = ScenarioLibraryResult()
        self.assertFalse(result.ran)
        self.assertFalse(result.skipped)
        self.assertIsNone(result.skip_reason)
        self.assertIsNone(result.suite_path)
        self.assertFalse(result.passed)
        self.assertEqual(result.errors, [])
        self.assertFalse(result.evidence_verified)

    def test_to_dict_serialization(self):
        """Test that to_dict produces valid JSON-serializable dict."""
        result = ScenarioLibraryResult(
            ran=True,
            skipped=False,
            suite_path="/path/to/suite.yaml",
            output_dir="/path/to/output",
            passed=True,
            evidence_verified=True,
        )
        
        d = result.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d["ran"], True)
        self.assertEqual(d["skipped"], False)
        self.assertEqual(d["suite_path"], "/path/to/suite.yaml")
        self.assertEqual(d["passed"], True)
        self.assertEqual(d["evidence_verified"], True)
        
        # Should be JSON serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

    def test_skipped_result_serialization(self):
        """Test serialization of a skipped result."""
        result = ScenarioLibraryResult(
            ran=False,
            skipped=True,
            skip_reason="--skip-scenario-library flag set",
        )
        
        d = result.to_dict()
        
        self.assertEqual(d["ran"], False)
        self.assertEqual(d["skipped"], True)
        self.assertEqual(d["skip_reason"], "--skip-scenario-library flag set")

    def test_failed_result_with_errors(self):
        """Test serialization of a failed result with errors."""
        result = ScenarioLibraryResult(
            ran=True,
            skipped=False,
            passed=False,
            errors=["Suite execution failed", "Missing scenario file"],
        )
        
        d = result.to_dict()
        
        self.assertEqual(d["ran"], True)
        self.assertEqual(d["passed"], False)
        self.assertEqual(len(d["errors"]), 2)
        self.assertIn("Suite execution failed", d["errors"])


class TestSuiteSelectionForPipeline(unittest.TestCase):
    """Tests for suite selection logic in the pipeline context."""

    def test_smoke_mode_selects_smoke_suite(self):
        """Test that --smoke mode selects the scenario_library_smoke_v1.yaml suite."""
        config = ScenarioLibraryConfig(enabled=True, smoke_mode=True)
        suite_path = config.get_suite_path()
        
        self.assertEqual(suite_path.name, "scenario_library_smoke_v1.yaml")

    def test_non_smoke_mode_selects_full_suite(self):
        """Test that non-smoke mode selects the scenario_library_v1.yaml suite."""
        config = ScenarioLibraryConfig(enabled=True, smoke_mode=False)
        suite_path = config.get_suite_path()
        
        self.assertEqual(suite_path.name, "scenario_library_v1.yaml")


class TestSkipScenarioLibraryLogic(unittest.TestCase):
    """Tests for --skip-scenario-library flag behavior."""

    def test_skip_disables_scenario_library(self):
        """Test that enabled=False disables scenario library."""
        config = ScenarioLibraryConfig(enabled=False)
        self.assertFalse(config.enabled)

    def test_skip_result_records_skip(self):
        """Test that skipped result is properly recorded."""
        result = ScenarioLibraryResult(
            ran=False,
            skipped=True,
            skip_reason="Scenario library skipped (--skip-scenario-library)",
        )
        
        d = result.to_dict()
        
        self.assertFalse(d["ran"])
        self.assertTrue(d["skipped"])
        self.assertIn("--skip-scenario-library", d["skip_reason"])

    def test_promotion_record_includes_scenario_library(self):
        """Test that scenario library result can be included in promotion record."""
        # This tests the data structure, not the actual promotion logic
        result = ScenarioLibraryResult(
            ran=True,
            skipped=False,
            suite_path="scenarios/suites/scenario_library_v1.yaml",
            output_dir="runs/trial_0001/suite/scenario_library_v1",
            passed=True,
            evidence_verified=True,
        )
        
        # Simulate promotion record structure
        promotion_record = {
            "schema_version": 3,
            "study_name": "test_study",
            "scenario_library": result.to_dict(),
        }
        
        # Verify structure
        self.assertIn("scenario_library", promotion_record)
        self.assertEqual(promotion_record["scenario_library"]["ran"], True)
        self.assertEqual(promotion_record["scenario_library"]["passed"], True)


class TestIntegrityCheck(unittest.TestCase):
    """Tests for scenario library integrity checking."""

    def test_integrity_check_passes_on_valid_library(self):
        """Test that integrity check passes on the existing library."""
        if not SCENARIO_LIBRARY_MANIFEST.exists():
            self.skipTest("Scenario library not generated yet")
        
        success, errors = check_scenario_library_integrity(verbose=False)
        self.assertTrue(success, f"Integrity check failed: {errors}")

    def test_integrity_check_fails_on_missing_manifest(self):
        """Test that integrity check fails when manifest is missing."""
        with mock.patch('batch_runs.phase_a.promote_pipeline.SCENARIO_LIBRARY_MANIFEST',
                       Path("/nonexistent/manifest.json")):
            success, errors = check_scenario_library_integrity(verbose=False)
            
            self.assertFalse(success)
            self.assertTrue(any("not found" in e.lower() for e in errors))


class TestSuiteValidation(unittest.TestCase):
    """Tests for scenario library suite validation."""

    def test_validate_existing_smoke_suite(self):
        """Test that smoke suite validation passes."""
        if not SCENARIO_LIBRARY_SMOKE_SUITE.exists():
            self.skipTest("Smoke suite not generated yet")
        
        success, errors = validate_scenario_library_suite(
            SCENARIO_LIBRARY_SMOKE_SUITE, verbose=False
        )
        self.assertTrue(success, f"Suite validation failed: {errors}")

    def test_validate_existing_full_suite(self):
        """Test that full suite validation passes."""
        if not SCENARIO_LIBRARY_FULL_SUITE.exists():
            self.skipTest("Full suite not generated yet")
        
        success, errors = validate_scenario_library_suite(
            SCENARIO_LIBRARY_FULL_SUITE, verbose=False
        )
        self.assertTrue(success, f"Suite validation failed: {errors}")

    def test_validate_fails_on_missing_suite(self):
        """Test that validation fails for missing suite file."""
        success, errors = validate_scenario_library_suite(
            Path("/nonexistent/suite.yaml"), verbose=False
        )
        
        self.assertFalse(success)
        self.assertTrue(any("not found" in e.lower() for e in errors))

    def test_validate_fails_on_empty_suite(self):
        """Test that validation fails for suite with 0 scenarios."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("suite_id: empty_suite\nscenarios:\n")
            temp_path = Path(f.name)
        
        try:
            success, errors = validate_scenario_library_suite(temp_path, verbose=False)
            
            self.assertFalse(success)
            self.assertTrue(any("0 scenarios" in e for e in errors))
        finally:
            temp_path.unlink()


class TestFullSuiteContainsAllScenarios(unittest.TestCase):
    """Tests to verify full suite contains all expected scenarios."""

    def test_full_suite_has_10_scenarios(self):
        """Test that full suite contains all 10 scenarios."""
        if not SCENARIO_LIBRARY_FULL_SUITE.exists():
            self.skipTest("Full suite not generated yet")
        
        content = SCENARIO_LIBRARY_FULL_SUITE.read_text()
        
        # Count scenario paths
        scenario_count = content.count("- path:")
        self.assertEqual(scenario_count, 10)

    def test_full_suite_scenarios_sorted_by_filename(self):
        """Test that full suite scenarios are in sorted order by filename."""
        if not SCENARIO_LIBRARY_FULL_SUITE.exists():
            self.skipTest("Full suite not generated yet")
        
        content = SCENARIO_LIBRARY_FULL_SUITE.read_text()
        
        # Extract scenario filenames
        import re
        paths = re.findall(r'- path: .*/(slib_v1_[^/]+\.yaml)', content)
        
        # Verify sorted order
        self.assertEqual(paths, sorted(paths))

    def test_smoke_suite_has_5_scenarios(self):
        """Test that smoke suite contains 5 scenarios (one per category)."""
        if not SCENARIO_LIBRARY_SMOKE_SUITE.exists():
            self.skipTest("Smoke suite not generated yet")
        
        content = SCENARIO_LIBRARY_SMOKE_SUITE.read_text()
        
        # Count scenario paths
        scenario_count = content.count("- path:")
        self.assertEqual(scenario_count, 5)


if __name__ == "__main__":
    unittest.main()

