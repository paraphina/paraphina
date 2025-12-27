"""
Tests for confidence-aware statistics functions.

Hermetic tests - no cargo invocation.
"""

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.stats import (
    wilson_ucb,
    wilson_lcb,
    wilson_ci,
    normal_lcb,
    normal_ucb,
    normal_ci,
    get_statistics_metadata,
    KNOWN_Z_QUANTILES,
    _z_quantile,
)


class TestWilsonUCB(unittest.TestCase):
    """Tests for Wilson score upper confidence bound."""
    
    def test_known_value_zero_zero(self):
        """Test 0/0 raises ValueError."""
        with self.assertRaises(ValueError):
            wilson_ucb(0, 0)
    
    def test_known_value_zero_n(self):
        """Test 0/100 case: UCB should be small but non-zero."""
        # 0 kills in 100 runs at alpha=0.05
        ucb = wilson_ucb(0, 100, 0.05)
        self.assertGreater(ucb, 0.0)
        self.assertLess(ucb, 0.05)  # Should be well below 5%
    
    def test_known_value_half(self):
        """Test 50/100 case: UCB should be just above 0.5."""
        # 50 kills in 100 runs at alpha=0.05
        ucb = wilson_ucb(50, 100, 0.05)
        self.assertGreater(ucb, 0.5)
        self.assertLess(ucb, 0.65)  # Should be around 0.58
    
    def test_known_value_all(self):
        """Test 100/100 case: UCB should be 1.0 or very close."""
        ucb = wilson_ucb(100, 100, 0.05)
        self.assertGreaterEqual(ucb, 0.95)
        self.assertLessEqual(ucb, 1.0)
    
    def test_known_value_small_n(self):
        """Test 1/5 case with Wilson interval."""
        ucb = wilson_ucb(1, 5, 0.05)
        # Wilson interval for 1/5 at 95%:
        # p_hat = 0.2, z ≈ 1.645 for one-sided
        # UCB should be between 0.2 and 0.7
        self.assertGreater(ucb, 0.2)
        self.assertLess(ucb, 0.75)
    
    def test_known_value_2_100(self):
        """Test 2/100 - known value from Wilson formula."""
        # 2 kills in 100 runs
        # Point estimate = 0.02
        # Wilson UCB should be around 0.05-0.06
        ucb = wilson_ucb(2, 100, 0.05)
        self.assertGreater(ucb, 0.02)  # Above point estimate
        self.assertLess(ucb, 0.10)     # Below 10%
        # More precisely:
        self.assertGreater(ucb, 0.04)
        self.assertLess(ucb, 0.08)
    
    def test_monotonicity_in_k(self):
        """Test that UCB increases with k."""
        n = 100
        ucb_0 = wilson_ucb(0, n, 0.05)
        ucb_10 = wilson_ucb(10, n, 0.05)
        ucb_50 = wilson_ucb(50, n, 0.05)
        
        self.assertLess(ucb_0, ucb_10)
        self.assertLess(ucb_10, ucb_50)
    
    def test_monotonicity_in_alpha(self):
        """Test that UCB decreases with alpha (more confident = lower bound)."""
        k, n = 10, 100
        ucb_01 = wilson_ucb(k, n, 0.01)  # More confident
        ucb_05 = wilson_ucb(k, n, 0.05)
        ucb_10 = wilson_ucb(k, n, 0.10)  # Less confident
        
        # Lower alpha = higher z = higher UCB
        self.assertGreater(ucb_01, ucb_05)
        self.assertGreater(ucb_05, ucb_10)
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise ValueError."""
        with self.assertRaises(ValueError):
            wilson_ucb(-1, 100, 0.05)  # Negative k
        with self.assertRaises(ValueError):
            wilson_ucb(101, 100, 0.05)  # k > n
        with self.assertRaises(ValueError):
            wilson_ucb(10, 100, 0.0)  # alpha = 0
        with self.assertRaises(ValueError):
            wilson_ucb(10, 100, 1.0)  # alpha = 1


class TestWilsonLCB(unittest.TestCase):
    """Tests for Wilson score lower confidence bound."""
    
    def test_known_value_all(self):
        """Test 100/100 case: LCB should be high."""
        lcb = wilson_lcb(100, 100, 0.05)
        self.assertGreater(lcb, 0.95)
    
    def test_known_value_zero(self):
        """Test 0/100 case: LCB should be 0."""
        lcb = wilson_lcb(0, 100, 0.05)
        self.assertEqual(lcb, 0.0)
    
    def test_symmetry_with_ucb(self):
        """Test that LCB < point estimate < UCB."""
        k, n = 50, 100
        lcb = wilson_lcb(k, n, 0.05)
        ucb = wilson_ucb(k, n, 0.05)
        point = k / n
        
        self.assertLess(lcb, point)
        self.assertGreater(ucb, point)


class TestWilsonCI(unittest.TestCase):
    """Tests for Wilson score two-sided confidence interval."""
    
    def test_two_sided_contains_point(self):
        """Test that two-sided CI contains point estimate."""
        k, n = 25, 100
        lower, upper = wilson_ci(k, n, 0.05)
        point = k / n
        
        self.assertLess(lower, point)
        self.assertGreater(upper, point)
    
    def test_two_sided_narrower_at_higher_n(self):
        """Test that CI narrows with more observations."""
        lower_10, upper_10 = wilson_ci(5, 10, 0.05)
        lower_100, upper_100 = wilson_ci(50, 100, 0.05)
        
        width_10 = upper_10 - lower_10
        width_100 = upper_100 - lower_100
        
        self.assertGreater(width_10, width_100)


class TestNormalLCB(unittest.TestCase):
    """Tests for normal lower confidence bound."""
    
    def test_known_value_simple(self):
        """Test simple case with known values."""
        # mean=100, stdev=20, n=100, alpha=0.05
        # SE = 20 / sqrt(100) = 2
        # z(0.95) ≈ 1.645
        # LCB = 100 - 1.645 * 2 = 96.71
        lcb = normal_lcb(100.0, 20.0, 100, 0.05)
        self.assertAlmostEqual(lcb, 96.71, delta=0.1)
    
    def test_lcb_below_mean(self):
        """Test that LCB is always below mean."""
        lcb = normal_lcb(50.0, 10.0, 50, 0.05)
        self.assertLess(lcb, 50.0)
    
    def test_monotonicity_lower_alpha_lower_lcb(self):
        """Test that lower alpha gives lower LCB (more conservative)."""
        mean, stdev, n = 100.0, 20.0, 100
        lcb_01 = normal_lcb(mean, stdev, n, 0.01)  # More confident
        lcb_05 = normal_lcb(mean, stdev, n, 0.05)
        lcb_10 = normal_lcb(mean, stdev, n, 0.10)  # Less confident
        
        # Lower alpha = higher z = lower LCB
        self.assertLess(lcb_01, lcb_05)
        self.assertLess(lcb_05, lcb_10)
    
    def test_monotonicity_higher_n_higher_lcb(self):
        """Test that higher n gives higher LCB (narrower interval)."""
        mean, stdev, alpha = 100.0, 20.0, 0.05
        lcb_10 = normal_lcb(mean, stdev, 10, alpha)
        lcb_100 = normal_lcb(mean, stdev, 100, alpha)
        lcb_1000 = normal_lcb(mean, stdev, 1000, alpha)
        
        self.assertLess(lcb_10, lcb_100)
        self.assertLess(lcb_100, lcb_1000)
    
    def test_edge_case_n_equals_1(self):
        """Test n=1 returns mean (documented behavior)."""
        lcb = normal_lcb(100.0, 20.0, 1, 0.05)
        self.assertEqual(lcb, 100.0)
    
    def test_edge_case_stdev_zero(self):
        """Test stdev=0 returns mean."""
        lcb = normal_lcb(100.0, 0.0, 50, 0.05)
        self.assertEqual(lcb, 100.0)
    
    def test_edge_case_nan_mean(self):
        """Test NaN mean returns NaN."""
        lcb = normal_lcb(float("nan"), 20.0, 50, 0.05)
        self.assertTrue(math.isnan(lcb))
    
    def test_edge_case_nan_stdev(self):
        """Test NaN stdev returns NaN."""
        lcb = normal_lcb(100.0, float("nan"), 50, 0.05)
        self.assertTrue(math.isnan(lcb))
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise ValueError."""
        with self.assertRaises(ValueError):
            normal_lcb(100.0, 20.0, 0, 0.05)  # n = 0
        with self.assertRaises(ValueError):
            normal_lcb(100.0, -20.0, 50, 0.05)  # Negative stdev
        with self.assertRaises(ValueError):
            normal_lcb(100.0, 20.0, 50, 0.0)  # alpha = 0
        with self.assertRaises(ValueError):
            normal_lcb(100.0, 20.0, 50, 1.0)  # alpha = 1


class TestNormalUCB(unittest.TestCase):
    """Tests for normal upper confidence bound."""
    
    def test_ucb_above_mean(self):
        """Test that UCB is always above mean."""
        ucb = normal_ucb(50.0, 10.0, 50, 0.05)
        self.assertGreater(ucb, 50.0)
    
    def test_symmetry_with_lcb(self):
        """Test that LCB and UCB are symmetric around mean."""
        mean, stdev, n, alpha = 100.0, 20.0, 100, 0.05
        lcb = normal_lcb(mean, stdev, n, alpha)
        ucb = normal_ucb(mean, stdev, n, alpha)
        
        # Check symmetry
        delta_lower = mean - lcb
        delta_upper = ucb - mean
        self.assertAlmostEqual(delta_lower, delta_upper, delta=0.01)


class TestNormalCI(unittest.TestCase):
    """Tests for normal two-sided confidence interval."""
    
    def test_two_sided_contains_mean(self):
        """Test that two-sided CI contains mean."""
        mean, stdev, n = 100.0, 20.0, 100
        lower, upper = normal_ci(mean, stdev, n, 0.05)
        
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
    
    def test_two_sided_symmetric(self):
        """Test that two-sided CI is symmetric around mean."""
        mean, stdev, n = 100.0, 20.0, 100
        lower, upper = normal_ci(mean, stdev, n, 0.05)
        
        delta_lower = mean - lower
        delta_upper = upper - mean
        self.assertAlmostEqual(delta_lower, delta_upper, delta=0.01)


class TestZQuantile(unittest.TestCase):
    """Tests for internal z-quantile function."""
    
    def test_known_quantiles(self):
        """Test against known z-quantile values."""
        # z(0.95) for one-sided 95% ≈ 1.6449
        z_95 = _z_quantile(0.95)
        self.assertAlmostEqual(z_95, KNOWN_Z_QUANTILES[0.05], delta=0.001)
        
        # z(0.975) for two-sided 95% ≈ 1.96
        z_975 = _z_quantile(0.975)
        self.assertAlmostEqual(z_975, KNOWN_Z_QUANTILES[0.025], delta=0.001)
    
    def test_symmetry(self):
        """Test that z(p) = -z(1-p)."""
        z_low = _z_quantile(0.05)
        z_high = _z_quantile(0.95)
        self.assertAlmostEqual(z_low, -z_high, delta=0.0001)


class TestStatisticsMetadata(unittest.TestCase):
    """Tests for statistics metadata generation."""
    
    def test_metadata_structure(self):
        """Test that metadata has required fields."""
        meta = get_statistics_metadata(0.05)
        
        self.assertIn("alpha", meta)
        self.assertIn("methods", meta)
        self.assertIn("z_source", meta)
        self.assertIn("description", meta)
        
        self.assertEqual(meta["alpha"], 0.05)
        self.assertEqual(meta["methods"]["kill_ucb"], "wilson")
        self.assertEqual(meta["methods"]["pnl_lcb"], "normal")
    
    def test_metadata_different_alpha(self):
        """Test that alpha is correctly reflected in metadata."""
        meta_05 = get_statistics_metadata(0.05)
        meta_01 = get_statistics_metadata(0.01)
        
        self.assertEqual(meta_05["alpha"], 0.05)
        self.assertEqual(meta_01["alpha"], 0.01)


class TestDeterminism(unittest.TestCase):
    """Tests for deterministic behavior."""
    
    def test_wilson_deterministic(self):
        """Test that wilson_ucb is deterministic."""
        results = [wilson_ucb(10, 100, 0.05) for _ in range(10)]
        self.assertEqual(len(set(results)), 1)  # All same
    
    def test_normal_deterministic(self):
        """Test that normal_lcb is deterministic."""
        results = [normal_lcb(100.0, 20.0, 50, 0.05) for _ in range(10)]
        self.assertEqual(len(set(results)), 1)  # All same


if __name__ == "__main__":
    unittest.main()

