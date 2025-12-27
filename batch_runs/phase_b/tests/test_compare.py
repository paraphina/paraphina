"""
Tests for Phase B comparison module (dominance and CI overlap).

Tests:
- test_dominance_logic: One-sided dominance checks
- test_ci_overlap_detection: Metric-by-metric CI overlap tests
- test_comparison_result: Full comparison result structure
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_b.confidence import ConfidenceInterval
from batch_runs.phase_b.compare import (
    ComparisonOutcome,
    ComparisonResult,
    MetricComparison,
    MetricDirection,
    check_dominance,
    check_non_inferiority,
    check_strict_dominance,
    compare_metric,
    compare_metric_threshold,
    compare_runs,
)


def make_ci(lower: float, point: float, upper: float, name: str = "test") -> ConfidenceInterval:
    """Helper to create a ConfidenceInterval."""
    return ConfidenceInterval(
        lower=lower, point=point, upper=upper,
        alpha=0.05, n_samples=100, n_bootstrap=1000,
        estimator_name=name,
    )


class TestDominanceLogic(unittest.TestCase):
    """Tests for one-sided dominance checking."""
    
    def test_candidate_dominates_higher_is_better(self):
        """Test candidate dominates when lower CI > baseline upper CI."""
        candidate = make_ci(10.0, 12.0, 14.0)  # Lower bound is 10
        baseline = make_ci(5.0, 7.0, 9.0)       # Upper bound is 9
        
        outcome, reason = check_dominance(
            candidate, baseline, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertEqual(outcome, ComparisonOutcome.CANDIDATE_DOMINATES)
        self.assertIn("10.0", reason)
        self.assertIn("9.0", reason)
    
    def test_baseline_dominates_higher_is_better(self):
        """Test baseline dominates when its lower CI > candidate upper CI."""
        candidate = make_ci(1.0, 3.0, 5.0)   # Upper bound is 5
        baseline = make_ci(7.0, 9.0, 11.0)   # Lower bound is 7
        
        outcome, reason = check_dominance(
            candidate, baseline, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertEqual(outcome, ComparisonOutcome.BASELINE_DOMINATES)
    
    def test_no_significant_diff_higher_is_better(self):
        """Test no significant difference when CIs overlap."""
        candidate = make_ci(5.0, 8.0, 11.0)  # [5, 11]
        baseline = make_ci(7.0, 10.0, 13.0)  # [7, 13]
        
        outcome, reason = check_dominance(
            candidate, baseline, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertEqual(outcome, ComparisonOutcome.NO_SIGNIFICANT_DIFF)
        self.assertIn("overlap", reason.lower())
    
    def test_candidate_dominates_lower_is_better(self):
        """Test candidate dominates when upper CI < baseline lower CI."""
        candidate = make_ci(1.0, 2.0, 3.0)   # Upper bound is 3
        baseline = make_ci(5.0, 7.0, 9.0)    # Lower bound is 5
        
        outcome, reason = check_dominance(
            candidate, baseline, MetricDirection.LOWER_IS_BETTER
        )
        
        self.assertEqual(outcome, ComparisonOutcome.CANDIDATE_DOMINATES)
    
    def test_baseline_dominates_lower_is_better(self):
        """Test baseline dominates when its upper CI < candidate lower CI."""
        candidate = make_ci(10.0, 12.0, 14.0)  # Lower bound is 10
        baseline = make_ci(1.0, 3.0, 5.0)      # Upper bound is 5
        
        outcome, reason = check_dominance(
            candidate, baseline, MetricDirection.LOWER_IS_BETTER
        )
        
        self.assertEqual(outcome, ComparisonOutcome.BASELINE_DOMINATES)
    
    def test_no_significant_diff_lower_is_better(self):
        """Test no significant difference for lower-is-better with overlap."""
        candidate = make_ci(5.0, 8.0, 11.0)  # [5, 11]
        baseline = make_ci(7.0, 10.0, 13.0)  # [7, 13]
        
        outcome, reason = check_dominance(
            candidate, baseline, MetricDirection.LOWER_IS_BETTER
        )
        
        self.assertEqual(outcome, ComparisonOutcome.NO_SIGNIFICANT_DIFF)


class TestCompareMetric(unittest.TestCase):
    """Tests for compare_metric function."""
    
    def test_metric_comparison_pass_when_candidate_dominates(self):
        """Test that comparison passes when candidate dominates."""
        candidate = make_ci(10.0, 12.0, 14.0)
        baseline = make_ci(5.0, 7.0, 9.0)
        
        result = compare_metric(
            "pnl_mean", candidate, baseline, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertTrue(result.is_pass)
        self.assertEqual(result.outcome, ComparisonOutcome.CANDIDATE_DOMINATES)
    
    def test_metric_comparison_pass_when_no_diff(self):
        """Test that comparison passes when no significant difference."""
        candidate = make_ci(5.0, 8.0, 11.0)
        baseline = make_ci(7.0, 10.0, 13.0)
        
        result = compare_metric(
            "pnl_mean", candidate, baseline, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertTrue(result.is_pass)  # Overlapping = pass
        self.assertEqual(result.outcome, ComparisonOutcome.NO_SIGNIFICANT_DIFF)
    
    def test_metric_comparison_fail_when_baseline_dominates(self):
        """Test that comparison fails when baseline dominates."""
        candidate = make_ci(1.0, 3.0, 5.0)
        baseline = make_ci(7.0, 9.0, 11.0)
        
        result = compare_metric(
            "pnl_mean", candidate, baseline, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertFalse(result.is_pass)
        self.assertEqual(result.outcome, ComparisonOutcome.BASELINE_DOMINATES)


class TestCompareMetricThreshold(unittest.TestCase):
    """Tests for threshold-based metric comparison."""
    
    def test_passes_threshold_higher_is_better(self):
        """Test passing when lower CI >= threshold (higher is better)."""
        candidate = make_ci(10.0, 12.0, 14.0)  # Lower CI = 10
        
        result = compare_metric_threshold(
            "pnl_mean", candidate, 8.0, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertTrue(result.is_pass)
        self.assertEqual(result.outcome, ComparisonOutcome.CANDIDATE_PASSES)
    
    def test_fails_threshold_higher_is_better(self):
        """Test failing when lower CI < threshold (higher is better)."""
        candidate = make_ci(5.0, 7.0, 9.0)  # Lower CI = 5
        
        result = compare_metric_threshold(
            "pnl_mean", candidate, 8.0, MetricDirection.HIGHER_IS_BETTER
        )
        
        self.assertFalse(result.is_pass)
        self.assertEqual(result.outcome, ComparisonOutcome.CANDIDATE_FAILS)
    
    def test_passes_threshold_lower_is_better(self):
        """Test passing when upper CI <= threshold (lower is better)."""
        candidate = make_ci(1.0, 3.0, 5.0)  # Upper CI = 5
        
        result = compare_metric_threshold(
            "kill_rate", candidate, 10.0, MetricDirection.LOWER_IS_BETTER
        )
        
        self.assertTrue(result.is_pass)
        self.assertEqual(result.outcome, ComparisonOutcome.CANDIDATE_PASSES)
    
    def test_fails_threshold_lower_is_better(self):
        """Test failing when upper CI > threshold (lower is better)."""
        candidate = make_ci(8.0, 10.0, 12.0)  # Upper CI = 12
        
        result = compare_metric_threshold(
            "kill_rate", candidate, 10.0, MetricDirection.LOWER_IS_BETTER
        )
        
        self.assertFalse(result.is_pass)
        self.assertEqual(result.outcome, ComparisonOutcome.CANDIDATE_FAILS)


class TestComparisonResult(unittest.TestCase):
    """Tests for ComparisonResult aggregation."""
    
    def test_all_pass(self):
        """Test all_pass when all metrics pass."""
        comparisons = [
            MetricComparison(
                metric_name="pnl_mean",
                direction=MetricDirection.HIGHER_IS_BETTER,
                candidate_ci=make_ci(10.0, 12.0, 14.0),
                baseline_ci=make_ci(5.0, 7.0, 9.0),
                outcome=ComparisonOutcome.CANDIDATE_DOMINATES,
                is_pass=True,
                reason="Candidate dominates",
            ),
            MetricComparison(
                metric_name="max_drawdown",
                direction=MetricDirection.LOWER_IS_BETTER,
                candidate_ci=make_ci(1.0, 2.0, 3.0),
                baseline_ci=make_ci(5.0, 7.0, 9.0),
                outcome=ComparisonOutcome.CANDIDATE_DOMINATES,
                is_pass=True,
                reason="Candidate dominates",
            ),
        ]
        
        # Create a mock RunMetrics (simplified)
        from batch_runs.phase_b.confidence import RunMetrics
        mock_metrics = RunMetrics(
            pnl_mean=make_ci(10.0, 12.0, 14.0, "pnl_mean"),
            pnl_median=make_ci(10.0, 12.0, 14.0, "pnl_median"),
            pnl_cvar=make_ci(5.0, 7.0, 9.0, "pnl_cvar"),
            max_drawdown=make_ci(1.0, 2.0, 3.0, "max_drawdown"),
            kill_rate=0.05,
            kill_count=5,
            n_runs=100,
        )
        
        result = ComparisonResult(
            candidate_metrics=mock_metrics,
            baseline_metrics=mock_metrics,
            metric_comparisons=comparisons,
        )
        
        self.assertTrue(result.all_pass)
        self.assertEqual(result.n_pass, 2)
        self.assertEqual(result.n_fail, 0)
        self.assertEqual(len(result.fail_reasons), 0)
    
    def test_one_fail(self):
        """Test that one failing metric sets all_pass to False."""
        comparisons = [
            MetricComparison(
                metric_name="pnl_mean",
                direction=MetricDirection.HIGHER_IS_BETTER,
                candidate_ci=make_ci(10.0, 12.0, 14.0),
                baseline_ci=make_ci(5.0, 7.0, 9.0),
                outcome=ComparisonOutcome.CANDIDATE_DOMINATES,
                is_pass=True,
                reason="Candidate dominates",
            ),
            MetricComparison(
                metric_name="kill_rate",
                direction=MetricDirection.LOWER_IS_BETTER,
                candidate_ci=make_ci(0.1, 0.15, 0.2),
                baseline_ci=None,
                outcome=ComparisonOutcome.CANDIDATE_FAILS,
                is_pass=False,
                reason="Kill rate too high",
                threshold=0.1,
            ),
        ]
        
        from batch_runs.phase_b.confidence import RunMetrics
        mock_metrics = RunMetrics(
            pnl_mean=make_ci(10.0, 12.0, 14.0, "pnl_mean"),
            pnl_median=make_ci(10.0, 12.0, 14.0, "pnl_median"),
            pnl_cvar=make_ci(5.0, 7.0, 9.0, "pnl_cvar"),
            max_drawdown=make_ci(1.0, 2.0, 3.0, "max_drawdown"),
            kill_rate=0.15,
            kill_count=15,
            n_runs=100,
        )
        
        result = ComparisonResult(
            candidate_metrics=mock_metrics,
            baseline_metrics=None,
            metric_comparisons=comparisons,
        )
        
        self.assertFalse(result.all_pass)
        self.assertEqual(result.n_pass, 1)
        self.assertEqual(result.n_fail, 1)
        self.assertEqual(len(result.fail_reasons), 1)
        self.assertIn("kill_rate", result.fail_reasons[0])


class TestToDict(unittest.TestCase):
    """Tests for JSON serialization."""
    
    def test_metric_comparison_to_dict(self):
        """Test MetricComparison to_dict."""
        comparison = MetricComparison(
            metric_name="pnl_mean",
            direction=MetricDirection.HIGHER_IS_BETTER,
            candidate_ci=make_ci(10.0, 12.0, 14.0),
            baseline_ci=make_ci(5.0, 7.0, 9.0),
            outcome=ComparisonOutcome.CANDIDATE_DOMINATES,
            is_pass=True,
            reason="Test reason",
        )
        
        d = comparison.to_dict()
        
        self.assertEqual(d["metric_name"], "pnl_mean")
        self.assertEqual(d["direction"], "higher_is_better")
        self.assertEqual(d["outcome"], "candidate_dominates")
        self.assertTrue(d["is_pass"])
        self.assertEqual(d["reason"], "Test reason")


if __name__ == "__main__":
    unittest.main()

