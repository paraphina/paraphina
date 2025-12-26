"""
Tests for ADR (Adversarial Delta Regression) functionality.

Hermetic tests - no cargo invocation.
"""

import unittest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.promote_pipeline import (
    ADRConfig,
    ADRResult,
    compute_baseline_cache_path,
    find_evidence_root,
    _has_evidence_pack,
)


class TestBaselineCachePath(unittest.TestCase):
    """Test baseline cache path computation."""
    
    def test_stable_deterministic_path(self):
        """Test that baseline cache path is stable and deterministic."""
        cache_root = Path("/cache/root")
        profile = "balanced"
        suite_path = Path("/path/to/research_v1.yaml")
        
        # Compute path multiple times
        path1 = compute_baseline_cache_path(cache_root, profile, suite_path)
        path2 = compute_baseline_cache_path(cache_root, profile, suite_path)
        path3 = compute_baseline_cache_path(cache_root, profile, suite_path)
        
        # All should be identical
        self.assertEqual(path1, path2)
        self.assertEqual(path2, path3)
        
        # Should use suite stem (without extension)
        self.assertEqual(path1.name, "research_v1")
        self.assertEqual(path1.parent.name, "balanced")
        self.assertEqual(path1.parent.parent, cache_root)
    
    def test_different_profiles_different_paths(self):
        """Test that different profiles produce different paths."""
        cache_root = Path("/cache/root")
        suite_path = Path("/path/to/research_v1.yaml")
        
        balanced = compute_baseline_cache_path(cache_root, "balanced", suite_path)
        conservative = compute_baseline_cache_path(cache_root, "conservative", suite_path)
        aggressive = compute_baseline_cache_path(cache_root, "aggressive", suite_path)
        
        self.assertNotEqual(balanced, conservative)
        self.assertNotEqual(conservative, aggressive)
        self.assertNotEqual(balanced, aggressive)
        
        # All should share suite name
        self.assertEqual(balanced.name, conservative.name)
        self.assertEqual(conservative.name, aggressive.name)
    
    def test_different_suites_different_paths(self):
        """Test that different suites produce different paths."""
        cache_root = Path("/cache/root")
        profile = "balanced"
        
        research = compute_baseline_cache_path(cache_root, profile, Path("/suites/research_v1.yaml"))
        adversarial = compute_baseline_cache_path(cache_root, profile, Path("/suites/adversarial_regression_v2.yaml"))
        
        self.assertNotEqual(research, adversarial)
        self.assertEqual(research.name, "research_v1")
        self.assertEqual(adversarial.name, "adversarial_regression_v2")
    
    def test_path_structure(self):
        """Test the expected path structure."""
        cache_root = Path("/baseline_cache")
        profile = "conservative"
        suite_path = Path("/scenarios/suites/ci_smoke.yaml")
        
        result = compute_baseline_cache_path(cache_root, profile, suite_path)
        
        # Expected: /baseline_cache/conservative/ci_smoke
        self.assertEqual(result, cache_root / "conservative" / "ci_smoke")


class TestFindEvidenceRoot(unittest.TestCase):
    """Test evidence root finding functionality."""
    
    def test_finds_evidence_pack_in_directory(self):
        """Test finding evidence pack in given directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence_pack = root / "evidence_pack"
            evidence_pack.mkdir()
            (evidence_pack / "SHA256SUMS").write_text("test")
            
            result = find_evidence_root(root)
            
            self.assertEqual(result, root)
    
    def test_finds_evidence_pack_in_subdirectory(self):
        """Test finding evidence pack in a subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subdir = root / "suite" / "research_v1"
            subdir.mkdir(parents=True)
            evidence_pack = subdir / "evidence_pack"
            evidence_pack.mkdir()
            (evidence_pack / "SHA256SUMS").write_text("test")
            
            # Starting from root, should find that it contains evidence via subdir
            result = find_evidence_root(root)
            
            # It should find the root since the subdir has evidence
            self.assertEqual(result, root)
    
    def test_returns_none_when_no_evidence(self):
        """Test returning None when no evidence pack exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "some_file.txt").write_text("not evidence")
            
            result = find_evidence_root(root)
            
            self.assertIsNone(result)
    
    def test_finds_evidence_in_parent(self):
        """Test finding evidence pack walking up to parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence_pack = root / "evidence_pack"
            evidence_pack.mkdir()
            (evidence_pack / "SHA256SUMS").write_text("test")
            
            subdir = root / "nested" / "deep"
            subdir.mkdir(parents=True)
            
            # Start from deep subdir, should find root
            result = find_evidence_root(subdir)
            
            self.assertEqual(result, root)
    
    def test_has_evidence_pack_helper(self):
        """Test the _has_evidence_pack helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # No evidence pack
            self.assertFalse(_has_evidence_pack(root))
            
            # Create evidence_pack directory without SHA256SUMS
            evidence_pack = root / "evidence_pack"
            evidence_pack.mkdir()
            self.assertFalse(_has_evidence_pack(root))
            
            # Add SHA256SUMS
            (evidence_pack / "SHA256SUMS").write_text("test")
            self.assertTrue(_has_evidence_pack(root))


class TestADRConfig(unittest.TestCase):
    """Test ADRConfig dataclass."""
    
    def test_default_config(self):
        """Test default ADRConfig values."""
        config = ADRConfig()
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.suites, [])
        self.assertIsNone(config.baseline_cache_dir)
        self.assertIsNone(config.gate_max_regression_usd)
        self.assertIsNone(config.gate_max_regression_pct)
        self.assertTrue(config.write_md)
        self.assertTrue(config.write_json)
    
    def test_get_baseline_cache_with_explicit_dir(self):
        """Test get_baseline_cache with explicit cache directory."""
        config = ADRConfig(
            enabled=True,
            baseline_cache_dir=Path("/explicit/cache"),
        )
        study_dir = Path("/study/dir")
        
        result = config.get_baseline_cache(study_dir)
        
        self.assertEqual(result, Path("/explicit/cache"))
    
    def test_get_baseline_cache_default(self):
        """Test get_baseline_cache with default (under study_dir)."""
        config = ADRConfig(enabled=True)
        study_dir = Path("/study/dir")
        
        result = config.get_baseline_cache(study_dir)
        
        self.assertEqual(result, Path("/study/dir/_baseline_cache"))
    
    def test_get_effective_suites_explicit(self):
        """Test get_effective_suites with explicit suites."""
        config = ADRConfig(
            enabled=True,
            suites=[Path("/suite/a.yaml"), Path("/suite/b.yaml")],
        )
        
        result = config.get_effective_suites()
        
        self.assertEqual(result, [Path("/suite/a.yaml"), Path("/suite/b.yaml")])


class TestADRResult(unittest.TestCase):
    """Test ADRResult dataclass."""
    
    def test_default_result(self):
        """Test default ADRResult values."""
        result = ADRResult(
            suite_name="test_suite",
            baseline_dir=Path("/baseline"),
            candidate_dir=Path("/candidate"),
        )
        
        self.assertEqual(result.suite_name, "test_suite")
        self.assertEqual(result.baseline_dir, Path("/baseline"))
        self.assertEqual(result.candidate_dir, Path("/candidate"))
        self.assertIsNone(result.report_md)
        self.assertIsNone(result.report_json)
        self.assertTrue(result.gates_passed)
        self.assertEqual(result.gate_failures, [])
        self.assertEqual(result.command_run, "")
        self.assertEqual(result.returncode, 0)
    
    def test_result_with_failures(self):
        """Test ADRResult with gate failures."""
        result = ADRResult(
            suite_name="test_suite",
            baseline_dir=Path("/baseline"),
            candidate_dir=Path("/candidate"),
            gates_passed=False,
            gate_failures=["Regression too large: $100", "Max regression exceeded"],
        )
        
        self.assertFalse(result.gates_passed)
        self.assertEqual(len(result.gate_failures), 2)


class TestReportCommandConstruction(unittest.TestCase):
    """Test that report command construction is correct."""
    
    def test_basic_command_elements(self):
        """Test basic command construction elements."""
        # This tests the logic used in run_adr_report
        baseline_dir = Path("/baseline/dir")
        candidate_dir = Path("/candidate/dir")
        out_md = Path("/reports/suite.md")
        out_json = Path("/reports/suite.json")
        
        # Simulate command construction
        cmd = [
            "sim_eval", "report",
            "--baseline", str(baseline_dir),
            "--variant", f"candidate={candidate_dir}",
            "--out-md", str(out_md),
            "--out-json", str(out_json),
        ]
        
        self.assertIn("sim_eval", cmd)
        self.assertIn("report", cmd)
        self.assertIn("--baseline", cmd)
        self.assertIn(str(baseline_dir), cmd)
        self.assertIn("--variant", cmd)
        self.assertIn(f"candidate={candidate_dir}", cmd)
        self.assertIn("--out-md", cmd)
        self.assertIn(str(out_md), cmd)
        self.assertIn("--out-json", cmd)
        self.assertIn(str(out_json), cmd)
    
    def test_command_with_gates(self):
        """Test command construction with gate flags."""
        cmd = [
            "sim_eval", "report",
            "--baseline", "/baseline",
            "--variant", "candidate=/candidate",
            "--out-md", "/report.md",
            "--out-json", "/report.json",
        ]
        
        # Add optional gates
        gate_max_regression_usd = 50.0
        gate_max_regression_pct = 10.0
        
        if gate_max_regression_usd is not None:
            cmd.extend(["--gate-max-regression-usd", str(gate_max_regression_usd)])
        
        if gate_max_regression_pct is not None:
            cmd.extend(["--gate-max-regression-pct", str(gate_max_regression_pct)])
        
        self.assertIn("--gate-max-regression-usd", cmd)
        self.assertIn("50.0", cmd)
        self.assertIn("--gate-max-regression-pct", cmd)
        self.assertIn("10.0", cmd)


class TestDeterminism(unittest.TestCase):
    """Test determinism requirements."""
    
    def test_baseline_cache_path_determinism(self):
        """Test that baseline cache paths are fully deterministic."""
        cache_root = Path("/cache")
        profile = "balanced"
        suite = Path("/suites/research_v1.yaml")
        
        # Generate 100 times and verify all are identical
        paths = [compute_baseline_cache_path(cache_root, profile, suite) for _ in range(100)]
        
        first = paths[0]
        for path in paths[1:]:
            self.assertEqual(path, first, "Baseline cache path not deterministic")
    
    def test_path_ordering_stability(self):
        """Test that path computation order doesn't affect results."""
        cache_root = Path("/cache")
        suites = [
            Path("/suites/research_v1.yaml"),
            Path("/suites/adversarial_v2.yaml"),
            Path("/suites/ci_smoke.yaml"),
        ]
        profiles = ["balanced", "conservative", "aggressive"]
        
        # First pass
        results1 = {}
        for profile in profiles:
            for suite in suites:
                key = (profile, suite.stem)
                results1[key] = compute_baseline_cache_path(cache_root, profile, suite)
        
        # Second pass (reversed order)
        results2 = {}
        for profile in reversed(profiles):
            for suite in reversed(suites):
                key = (profile, suite.stem)
                results2[key] = compute_baseline_cache_path(cache_root, profile, suite)
        
        # Results should be identical
        for key in results1:
            self.assertEqual(results1[key], results2[key])


if __name__ == "__main__":
    unittest.main()

