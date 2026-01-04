"""
Tests for Phase B confidence module (bootstrap and CI computation).

Tests:
- test_bootstrap_ci_stability: CI stability across repeated runs with same seed
- test_estimator_correctness: Correctness of mean, median, CVaR, max_drawdown
- test_block_bootstrap_preserves_structure: Block bootstrap maintains temporal structure
"""

import math
import random
import unittest
from pathlib import Path
import sys
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_b.confidence import (
    BlockBootstrap,
    ConfidenceInterval,
    RunMetrics,
    bootstrap_ci,
    compute_ci,
    compute_cvar,
    compute_max_drawdown,
    compute_max_drawdowns_per_run,
    compute_mean,
    compute_median,
    compute_run_metrics,
)


class TestEstimators(unittest.TestCase):
    """Tests for statistical estimators."""
    
    def test_compute_mean_basic(self):
        """Test mean computation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(compute_mean(data), 3.0)
    
    def test_compute_mean_empty(self):
        """Test mean with empty list."""
        data: List[float] = []
        self.assertTrue(math.isnan(compute_mean(data)))
    
    def test_compute_median_odd(self):
        """Test median with odd number of elements."""
        data = [1.0, 5.0, 3.0]
        self.assertAlmostEqual(compute_median(data), 3.0)
    
    def test_compute_median_even(self):
        """Test median with even number of elements."""
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(compute_median(data), 2.5)
    
    def test_compute_cvar_basic(self):
        """Test CVaR computation."""
        # 10 values, alpha=0.2 means worst 20% = 2 values
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        cvar = compute_cvar(data, alpha=0.2)
        # Worst 20% = [1, 2], mean = 1.5
        self.assertAlmostEqual(cvar, 1.5)
    
    def test_compute_cvar_alpha_05(self):
        """Test CVaR at 5% level."""
        data = [float(i) for i in range(1, 101)]  # 1 to 100
        cvar = compute_cvar(data, alpha=0.05)
        # Worst 5% of 100 values = 5 values: [1, 2, 3, 4, 5]
        # Mean = 3
        self.assertAlmostEqual(cvar, 3.0)
    
    def test_compute_cvar_invalid_alpha(self):
        """Test CVaR with invalid alpha."""
        data = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            compute_cvar(data, alpha=0.0)
        with self.assertRaises(ValueError):
            compute_cvar(data, alpha=1.5)
    
    def test_compute_max_drawdown_simple(self):
        """Test max drawdown computation."""
        # Equity curve: [10, 15, 8, 12, 5, 20]
        # Running max: [10, 15, 15, 15, 15, 20]
        # Drawdowns:   [0,  0,  7,  3, 10, 0]
        # Max DD = 10
        equity = [10.0, 15.0, 8.0, 12.0, 5.0, 20.0]
        dd = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd, 10.0)
    
    def test_compute_max_drawdown_no_drawdown(self):
        """Test max drawdown with monotonic increase."""
        equity = [1.0, 2.0, 3.0, 4.0, 5.0]
        dd = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd, 0.0)
    
    def test_compute_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        equity = [10.0]
        dd = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd, 0.0)


class TestBlockBootstrap(unittest.TestCase):
    """Tests for block bootstrap resampling."""
    
    def test_resample_length(self):
        """Test that resampled data has same length as original."""
        data = [float(i) for i in range(100)]
        bootstrap = BlockBootstrap(block_size=10, n_bootstrap=1, seed=42)
        rng = random.Random(42)
        resampled = bootstrap.resample(data, rng)
        self.assertEqual(len(resampled), len(data))
    
    def test_resample_values_from_original(self):
        """Test that resampled values come from original data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        bootstrap = BlockBootstrap(block_size=2, n_bootstrap=1, seed=42)
        rng = random.Random(42)
        resampled = bootstrap.resample(data, rng)
        for val in resampled:
            self.assertIn(val, data)
    
    def test_bootstrap_samples_deterministic(self):
        """Test that same seed gives same results."""
        rng = random.Random(42)
        data = [rng.gauss(0, 1) for _ in range(100)]
        
        bootstrap1 = BlockBootstrap(n_bootstrap=100, seed=42)
        samples1 = bootstrap1.bootstrap_samples(data, compute_mean)
        
        bootstrap2 = BlockBootstrap(n_bootstrap=100, seed=42)
        samples2 = bootstrap2.bootstrap_samples(data, compute_mean)
        
        self.assertEqual(samples1, samples2)
    
    def test_bootstrap_samples_different_seeds(self):
        """Test that different seeds give different results."""
        rng = random.Random(42)
        data = [rng.gauss(0, 1) for _ in range(100)]
        
        bootstrap1 = BlockBootstrap(n_bootstrap=100, seed=42)
        samples1 = bootstrap1.bootstrap_samples(data, compute_mean)
        
        bootstrap2 = BlockBootstrap(n_bootstrap=100, seed=123)
        samples2 = bootstrap2.bootstrap_samples(data, compute_mean)
        
        # Should not be equal
        self.assertNotEqual(samples1, samples2)


class TestBootstrapCIStability(unittest.TestCase):
    """Test CI stability across repeated runs with same seed."""
    
    def test_bootstrap_ci_stability(self):
        """Test that CI is stable across repeated calls with same seed."""
        rng = random.Random(42)
        data = [rng.gauss(0, 1) for _ in range(100)]
        
        # Run bootstrap CI multiple times
        results = []
        for _ in range(5):
            lower, point, upper = bootstrap_ci(
                data, compute_mean,
                alpha=0.05,
                n_bootstrap=500,
                seed=42,
            )
            results.append((lower, point, upper))
        
        # All results should be identical
        for i in range(1, len(results)):
            self.assertAlmostEqual(results[0][0], results[i][0])
            self.assertAlmostEqual(results[0][1], results[i][1])
            self.assertAlmostEqual(results[0][2], results[i][2])
    
    def test_confidence_interval_contains_point(self):
        """Test that CI contains point estimate."""
        rng = random.Random(42)
        data = [rng.gauss(0, 1) + 5 for _ in range(100)]  # Mean around 5
        lower, point, upper = bootstrap_ci(
            data, compute_mean,
            alpha=0.05,
            n_bootstrap=1000,
            seed=42,
        )
        
        self.assertLess(lower, point)
        self.assertGreater(upper, point)
    
    def test_ci_width_decreases_with_sample_size(self):
        """Test that CI narrows with more data."""
        rng = random.Random(42)
        
        data_small = [rng.gauss(0, 1) for _ in range(50)]
        data_large = [rng.gauss(0, 1) for _ in range(500)]
        
        lower_s, _, upper_s = bootstrap_ci(data_small, compute_mean, seed=42)
        lower_l, _, upper_l = bootstrap_ci(data_large, compute_mean, seed=42)
        
        width_small = upper_s - lower_s
        width_large = upper_l - lower_l
        
        self.assertGreater(width_small, width_large)


class TestConfidenceInterval(unittest.TestCase):
    """Tests for ConfidenceInterval dataclass."""
    
    def test_width(self):
        """Test width property."""
        ci = ConfidenceInterval(
            lower=1.0, point=2.0, upper=3.0,
            alpha=0.05, n_samples=100, n_bootstrap=1000,
            estimator_name="mean"
        )
        self.assertAlmostEqual(ci.width, 2.0)
    
    def test_contains_zero(self):
        """Test contains_zero property."""
        ci_yes = ConfidenceInterval(
            lower=-1.0, point=0.5, upper=2.0,
            alpha=0.05, n_samples=100, n_bootstrap=1000,
            estimator_name="mean"
        )
        self.assertTrue(ci_yes.contains_zero)
        
        ci_no = ConfidenceInterval(
            lower=1.0, point=2.0, upper=3.0,
            alpha=0.05, n_samples=100, n_bootstrap=1000,
            estimator_name="mean"
        )
        self.assertFalse(ci_no.contains_zero)
    
    def test_to_dict(self):
        """Test JSON serialization."""
        ci = ConfidenceInterval(
            lower=1.0, point=2.0, upper=3.0,
            alpha=0.05, n_samples=100, n_bootstrap=1000,
            estimator_name="mean"
        )
        d = ci.to_dict()
        
        self.assertEqual(d["lower"], 1.0)
        self.assertEqual(d["point"], 2.0)
        self.assertEqual(d["upper"], 3.0)
        self.assertEqual(d["alpha"], 0.05)
        self.assertEqual(d["n_samples"], 100)


class TestComputeRunMetrics(unittest.TestCase):
    """Tests for compute_run_metrics function."""
    
    def test_basic_metrics(self):
        """Test basic metrics computation."""
        # Create synthetic run data
        rng = random.Random(42)
        pnl_per_run = [[rng.gauss(0, 1) for _ in range(100)] for _ in range(20)]
        kill_flags = [False] * 18 + [True] * 2  # 10% kill rate
        
        metrics = compute_run_metrics(
            pnl_per_run, kill_flags,
            alpha=0.05, n_bootstrap=100, seed=42
        )
        
        self.assertEqual(metrics.n_runs, 20)
        self.assertEqual(metrics.kill_count, 2)
        self.assertAlmostEqual(metrics.kill_rate, 0.1)
        
        # Check that CIs are computed
        self.assertIsNotNone(metrics.pnl_mean)
        self.assertFalse(math.isnan(metrics.pnl_mean.point))
    
    def test_empty_runs(self):
        """Test with empty run list."""
        metrics = compute_run_metrics([], [], seed=42)
        
        self.assertEqual(metrics.n_runs, 0)
        self.assertTrue(math.isnan(metrics.kill_rate))
    
    def test_metrics_to_dict(self):
        """Test RunMetrics JSON serialization."""
        rng = random.Random(42)
        pnl_per_run = [[rng.gauss(0, 1) for _ in range(50)] for _ in range(10)]
        kill_flags = [False] * 10
        
        metrics = compute_run_metrics(
            pnl_per_run, kill_flags,
            alpha=0.05, n_bootstrap=50, seed=42
        )
        
        d = metrics.to_dict()
        
        self.assertIn("pnl_mean", d)
        self.assertIn("max_drawdown", d)
        self.assertIn("kill_rate", d)
        self.assertIn("n_runs", d)


class TestDeterminism(unittest.TestCase):
    """Tests for deterministic behavior."""
    
    def test_bootstrap_is_deterministic(self):
        """Test that bootstrap is deterministic with seed."""
        rng = random.Random(123)
        data = [rng.gauss(0, 1) for _ in range(100)]
        
        # Multiple calls with same seed should give same result
        results = []
        for _ in range(3):
            lower, point, upper = bootstrap_ci(
                data, compute_mean,
                n_bootstrap=500,
                seed=42,
            )
            results.append((lower, point, upper))
        
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])
    
    def test_run_metrics_deterministic(self):
        """Test that run metrics are deterministic."""
        rng = random.Random(42)
        pnl_per_run = [[rng.gauss(0, 1) for _ in range(50)] for _ in range(10)]
        kill_flags = [False] * 8 + [True] * 2
        
        metrics1 = compute_run_metrics(
            pnl_per_run, kill_flags,
            alpha=0.05, n_bootstrap=100, seed=42
        )
        
        metrics2 = compute_run_metrics(
            pnl_per_run, kill_flags,
            alpha=0.05, n_bootstrap=100, seed=42
        )
        
        self.assertEqual(metrics1.pnl_mean.lower, metrics2.pnl_mean.lower)
        self.assertEqual(metrics1.pnl_mean.upper, metrics2.pnl_mean.upper)


if __name__ == "__main__":
    unittest.main()
