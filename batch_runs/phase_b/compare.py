"""
compare.py

Baseline Comparison for Phase B: Confidence-Aware Statistical Gating.

Implements:
- Candidate vs baseline comparison
- One-sided dominance checks
- Metric-by-metric CI overlap tests
- Clear pass/fail reasons per metric

Dominance Testing:
A candidate dominates baseline on a metric if:
- For "higher is better" metrics (PnL): lower_ci(candidate) > upper_ci(baseline)
- For "lower is better" metrics (drawdown): upper_ci(candidate) < lower_ci(baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from batch_runs.phase_b.confidence import (
    ConfidenceInterval,
    RunMetrics,
)


# =============================================================================
# Enums
# =============================================================================

class MetricDirection(Enum):
    """Direction of metric improvement."""
    HIGHER_IS_BETTER = "higher_is_better"  # e.g., PnL
    LOWER_IS_BETTER = "lower_is_better"    # e.g., drawdown, kill rate


class ComparisonOutcome(Enum):
    """Outcome of a single metric comparison."""
    CANDIDATE_DOMINATES = "candidate_dominates"  # Candidate is statistically better
    BASELINE_DOMINATES = "baseline_dominates"    # Baseline is statistically better
    NO_SIGNIFICANT_DIFF = "no_significant_diff"  # CIs overlap, inconclusive
    CANDIDATE_PASSES = "candidate_passes"        # Candidate meets threshold
    CANDIDATE_FAILS = "candidate_fails"          # Candidate fails threshold


# =============================================================================
# Metric Comparison Result
# =============================================================================

@dataclass
class MetricComparison:
    """
    Result of comparing a single metric between candidate and baseline.
    
    Attributes:
        metric_name: Name of the metric
        direction: Whether higher or lower is better
        candidate_ci: Candidate's confidence interval
        baseline_ci: Baseline's confidence interval
        outcome: Comparison outcome
        is_pass: Whether the comparison passes (candidate not worse)
        reason: Human-readable explanation
    """
    metric_name: str
    direction: MetricDirection
    candidate_ci: ConfidenceInterval
    baseline_ci: Optional[ConfidenceInterval]
    outcome: ComparisonOutcome
    is_pass: bool
    reason: str
    threshold: Optional[float] = None  # For threshold-based checks
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "candidate_ci": self.candidate_ci.to_dict(),
            "baseline_ci": self.baseline_ci.to_dict() if self.baseline_ci else None,
            "outcome": self.outcome.value,
            "is_pass": self.is_pass,
            "reason": self.reason,
            "threshold": self.threshold,
        }


# =============================================================================
# Dominance Checking
# =============================================================================

def check_dominance(
    candidate_ci: ConfidenceInterval,
    baseline_ci: ConfidenceInterval,
    direction: MetricDirection,
) -> Tuple[ComparisonOutcome, str]:
    """
    Check if candidate dominates baseline on a single metric.
    
    Dominance is determined by non-overlapping confidence intervals:
    - Higher is better: candidate.lower > baseline.upper
    - Lower is better: candidate.upper < baseline.lower
    
    Args:
        candidate_ci: Candidate's confidence interval
        baseline_ci: Baseline's confidence interval
        direction: Whether higher or lower values are better
        
    Returns:
        Tuple of (outcome, reason)
    """
    if direction == MetricDirection.HIGHER_IS_BETTER:
        # Candidate dominates if its lower bound exceeds baseline's upper
        if candidate_ci.lower > baseline_ci.upper:
            return (
                ComparisonOutcome.CANDIDATE_DOMINATES,
                f"Candidate lower CI ({candidate_ci.lower:.4f}) > "
                f"baseline upper CI ({baseline_ci.upper:.4f})"
            )
        # Baseline dominates if its lower bound exceeds candidate's upper
        elif baseline_ci.lower > candidate_ci.upper:
            return (
                ComparisonOutcome.BASELINE_DOMINATES,
                f"Baseline lower CI ({baseline_ci.lower:.4f}) > "
                f"candidate upper CI ({candidate_ci.upper:.4f})"
            )
        else:
            return (
                ComparisonOutcome.NO_SIGNIFICANT_DIFF,
                f"CIs overlap: candidate [{candidate_ci.lower:.4f}, {candidate_ci.upper:.4f}] "
                f"vs baseline [{baseline_ci.lower:.4f}, {baseline_ci.upper:.4f}]"
            )
    else:  # LOWER_IS_BETTER
        # Candidate dominates if its upper bound is below baseline's lower
        if candidate_ci.upper < baseline_ci.lower:
            return (
                ComparisonOutcome.CANDIDATE_DOMINATES,
                f"Candidate upper CI ({candidate_ci.upper:.4f}) < "
                f"baseline lower CI ({baseline_ci.lower:.4f})"
            )
        # Baseline dominates if its upper bound is below candidate's lower
        elif baseline_ci.upper < candidate_ci.lower:
            return (
                ComparisonOutcome.BASELINE_DOMINATES,
                f"Baseline upper CI ({baseline_ci.upper:.4f}) < "
                f"candidate lower CI ({candidate_ci.lower:.4f})"
            )
        else:
            return (
                ComparisonOutcome.NO_SIGNIFICANT_DIFF,
                f"CIs overlap: candidate [{candidate_ci.lower:.4f}, {candidate_ci.upper:.4f}] "
                f"vs baseline [{baseline_ci.lower:.4f}, {baseline_ci.upper:.4f}]"
            )


def compare_metric(
    metric_name: str,
    candidate_ci: ConfidenceInterval,
    baseline_ci: ConfidenceInterval,
    direction: MetricDirection,
) -> MetricComparison:
    """
    Compare a single metric between candidate and baseline.
    
    Args:
        metric_name: Name of the metric
        candidate_ci: Candidate's confidence interval
        baseline_ci: Baseline's confidence interval
        direction: Whether higher or lower values are better
        
    Returns:
        MetricComparison result
    """
    outcome, reason = check_dominance(candidate_ci, baseline_ci, direction)
    
    # Candidate passes if it dominates or there's no significant difference
    # (Fail closed: baseline dominance = fail)
    is_pass = outcome != ComparisonOutcome.BASELINE_DOMINATES
    
    return MetricComparison(
        metric_name=metric_name,
        direction=direction,
        candidate_ci=candidate_ci,
        baseline_ci=baseline_ci,
        outcome=outcome,
        is_pass=is_pass,
        reason=reason,
    )


def compare_metric_threshold(
    metric_name: str,
    candidate_ci: ConfidenceInterval,
    threshold: float,
    direction: MetricDirection,
) -> MetricComparison:
    """
    Compare a metric against a threshold.
    
    For threshold checks:
    - Higher is better: candidate.lower >= threshold
    - Lower is better: candidate.upper <= threshold
    
    Args:
        metric_name: Name of the metric
        candidate_ci: Candidate's confidence interval
        threshold: Threshold value to compare against
        direction: Whether higher or lower values are better
        
    Returns:
        MetricComparison result
    """
    if direction == MetricDirection.HIGHER_IS_BETTER:
        passes = candidate_ci.lower >= threshold
        if passes:
            outcome = ComparisonOutcome.CANDIDATE_PASSES
            reason = f"Candidate lower CI ({candidate_ci.lower:.4f}) >= threshold ({threshold:.4f})"
        else:
            outcome = ComparisonOutcome.CANDIDATE_FAILS
            reason = f"Candidate lower CI ({candidate_ci.lower:.4f}) < threshold ({threshold:.4f})"
    else:  # LOWER_IS_BETTER
        passes = candidate_ci.upper <= threshold
        if passes:
            outcome = ComparisonOutcome.CANDIDATE_PASSES
            reason = f"Candidate upper CI ({candidate_ci.upper:.4f}) <= threshold ({threshold:.4f})"
        else:
            outcome = ComparisonOutcome.CANDIDATE_FAILS
            reason = f"Candidate upper CI ({candidate_ci.upper:.4f}) > threshold ({threshold:.4f})"
    
    return MetricComparison(
        metric_name=metric_name,
        direction=direction,
        candidate_ci=candidate_ci,
        baseline_ci=None,
        outcome=outcome,
        is_pass=passes,
        reason=reason,
        threshold=threshold,
    )


# =============================================================================
# Full Comparison Result
# =============================================================================

@dataclass
class ComparisonResult:
    """
    Complete comparison result between candidate and baseline.
    
    Attributes:
        candidate_metrics: Candidate run metrics
        baseline_metrics: Baseline run metrics (optional)
        metric_comparisons: List of individual metric comparisons
        all_pass: Whether all comparisons pass
        n_pass: Number of passing comparisons
        n_fail: Number of failing comparisons
        fail_reasons: List of failure reasons
    """
    candidate_metrics: RunMetrics
    baseline_metrics: Optional[RunMetrics]
    metric_comparisons: List[MetricComparison]
    all_pass: bool = False
    n_pass: int = 0
    n_fail: int = 0
    fail_reasons: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute aggregate statistics."""
        self.n_pass = sum(1 for c in self.metric_comparisons if c.is_pass)
        self.n_fail = len(self.metric_comparisons) - self.n_pass
        self.all_pass = self.n_fail == 0
        self.fail_reasons = [
            f"{c.metric_name}: {c.reason}"
            for c in self.metric_comparisons
            if not c.is_pass
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "metric_comparisons": [c.to_dict() for c in self.metric_comparisons],
            "all_pass": self.all_pass,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "fail_reasons": self.fail_reasons,
        }


# =============================================================================
# Main Comparison Function
# =============================================================================

def compare_runs(
    candidate: RunMetrics,
    baseline: Optional[RunMetrics] = None,
    kill_threshold: Optional[float] = None,
    drawdown_threshold: Optional[float] = None,
    pnl_threshold: Optional[float] = None,
) -> ComparisonResult:
    """
    Compare candidate run against baseline with optional thresholds.
    
    Comparison rules (fail closed):
    1. PnL mean: candidate.lower > baseline.upper (higher is better)
    2. Drawdown: candidate.upper < baseline.upper (lower is better)
    3. Kill rate: candidate <= threshold (if provided)
    
    If no baseline, only threshold checks are performed.
    
    Args:
        candidate: Candidate run metrics
        baseline: Baseline run metrics (optional)
        kill_threshold: Maximum allowed kill rate (optional)
        drawdown_threshold: Maximum allowed drawdown upper CI (optional)
        pnl_threshold: Minimum required PnL lower CI (optional)
        
    Returns:
        ComparisonResult with all metric comparisons
    """
    comparisons: List[MetricComparison] = []
    
    # 1. PnL Mean comparison
    if baseline is not None:
        comparisons.append(compare_metric(
            "pnl_mean",
            candidate.pnl_mean,
            baseline.pnl_mean,
            MetricDirection.HIGHER_IS_BETTER,
        ))
    elif pnl_threshold is not None:
        comparisons.append(compare_metric_threshold(
            "pnl_mean",
            candidate.pnl_mean,
            pnl_threshold,
            MetricDirection.HIGHER_IS_BETTER,
        ))
    
    # 2. Drawdown comparison
    if baseline is not None:
        comparisons.append(compare_metric(
            "max_drawdown",
            candidate.max_drawdown,
            baseline.max_drawdown,
            MetricDirection.LOWER_IS_BETTER,
        ))
    elif drawdown_threshold is not None:
        comparisons.append(compare_metric_threshold(
            "max_drawdown",
            candidate.max_drawdown,
            drawdown_threshold,
            MetricDirection.LOWER_IS_BETTER,
        ))
    
    # 3. Kill rate threshold check (always if provided)
    if kill_threshold is not None:
        # Create a synthetic CI for kill rate (point estimate with small interval)
        kill_ci = ConfidenceInterval(
            lower=candidate.kill_rate,
            point=candidate.kill_rate,
            upper=candidate.kill_rate,
            alpha=candidate.alpha,
            n_samples=candidate.n_runs,
            n_bootstrap=0,
            estimator_name="kill_rate",
        )
        comparisons.append(compare_metric_threshold(
            "kill_rate",
            kill_ci,
            kill_threshold,
            MetricDirection.LOWER_IS_BETTER,
        ))
    
    # 4. CVaR comparison (higher is better for PnL CVaR)
    if baseline is not None:
        comparisons.append(compare_metric(
            "pnl_cvar",
            candidate.pnl_cvar,
            baseline.pnl_cvar,
            MetricDirection.HIGHER_IS_BETTER,
        ))
    
    return ComparisonResult(
        candidate_metrics=candidate,
        baseline_metrics=baseline,
        metric_comparisons=comparisons,
    )


# =============================================================================
# Strict Dominance Check
# =============================================================================

def check_strict_dominance(
    candidate: RunMetrics,
    baseline: RunMetrics,
) -> Tuple[bool, List[str]]:
    """
    Check if candidate strictly dominates baseline on all key metrics.
    
    Strict dominance requires:
    - PnL: candidate.lower > baseline.upper
    - Drawdown: candidate.upper < baseline.lower
    
    This is a conservative requirement for promotion.
    
    Args:
        candidate: Candidate run metrics
        baseline: Baseline run metrics
        
    Returns:
        Tuple of (dominates, reasons)
    """
    reasons = []
    
    # PnL check
    pnl_outcome, pnl_reason = check_dominance(
        candidate.pnl_mean,
        baseline.pnl_mean,
        MetricDirection.HIGHER_IS_BETTER,
    )
    if pnl_outcome != ComparisonOutcome.CANDIDATE_DOMINATES:
        reasons.append(f"PnL: {pnl_reason}")
    
    # Drawdown check
    dd_outcome, dd_reason = check_dominance(
        candidate.max_drawdown,
        baseline.max_drawdown,
        MetricDirection.LOWER_IS_BETTER,
    )
    if dd_outcome != ComparisonOutcome.CANDIDATE_DOMINATES:
        reasons.append(f"Drawdown: {dd_reason}")
    
    dominates = len(reasons) == 0
    return dominates, reasons


def check_non_inferiority(
    candidate: RunMetrics,
    baseline: RunMetrics,
) -> Tuple[bool, List[str]]:
    """
    Check if candidate is not inferior to baseline on any key metric.
    
    Non-inferiority requires:
    - Baseline does NOT dominate candidate on any metric
    - i.e., for each metric, candidate is either better or overlapping
    
    This is a less conservative requirement than strict dominance.
    
    Args:
        candidate: Candidate run metrics
        baseline: Baseline run metrics
        
    Returns:
        Tuple of (non_inferior, reasons)
    """
    reasons = []
    
    # PnL check
    pnl_outcome, pnl_reason = check_dominance(
        candidate.pnl_mean,
        baseline.pnl_mean,
        MetricDirection.HIGHER_IS_BETTER,
    )
    if pnl_outcome == ComparisonOutcome.BASELINE_DOMINATES:
        reasons.append(f"PnL: {pnl_reason}")
    
    # Drawdown check
    dd_outcome, dd_reason = check_dominance(
        candidate.max_drawdown,
        baseline.max_drawdown,
        MetricDirection.LOWER_IS_BETTER,
    )
    if dd_outcome == ComparisonOutcome.BASELINE_DOMINATES:
        reasons.append(f"Drawdown: {dd_reason}")
    
    non_inferior = len(reasons) == 0
    return non_inferior, reasons

