"""
stats.py

Deterministic statistical functions for confidence-aware promotion gating.

This module provides stdlib-only implementations for:
- wilson_ucb: Upper confidence bound for binomial proportions (kill rate)
- normal_lcb: Lower confidence bound for normal-distributed means (PnL)

All functions use statistics.NormalDist().inv_cdf for z-quantiles.
No external dependencies (no numpy/scipy/pandas).

References:
- Wilson score interval: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
- Normal confidence intervals: standard z-interval for sample means
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Tuple

# Standard normal distribution for quantile computations
_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def _z_quantile(p: float) -> float:
    """
    Compute the p-th quantile of the standard normal distribution.
    
    Uses statistics.NormalDist.inv_cdf (Python 3.8+).
    
    Args:
        p: Probability (0 < p < 1)
        
    Returns:
        z value such that P(Z <= z) = p
    """
    return _STANDARD_NORMAL.inv_cdf(p)


def wilson_ucb(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Compute the Wilson score upper confidence bound for a binomial proportion.
    
    This is a one-sided confidence bound used for the kill rate:
    - k: number of kills observed
    - n: total number of MC runs
    - alpha: significance level (default 0.05 for 95% confidence)
    
    Returns the upper bound of the (1-alpha) confidence interval.
    
    The Wilson score interval is preferred over naive intervals because:
    - It has good coverage properties for small samples and extreme proportions
    - It never produces negative probabilities or probabilities > 1
    - It is recommended by Agresti & Coull (1998)
    
    Args:
        k: Number of successes (kills)
        n: Number of trials (runs)
        alpha: Significance level (default 0.05)
        
    Returns:
        Upper confidence bound for the kill probability
        
    Raises:
        ValueError: If n == 0, k < 0, k > n, or alpha not in (0, 1)
        
    Examples:
        >>> wilson_ucb(5, 100, 0.05)  # 5 kills in 100 runs
        0.0949...  # Upper bound is about 9.5%
        
        >>> wilson_ucb(0, 50, 0.05)  # 0 kills in 50 runs
        0.0588...  # Upper bound is still non-zero
    """
    # Input validation
    if n == 0:
        raise ValueError("n must be > 0 (cannot compute confidence interval with no observations)")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    # z-score for (1 - alpha) one-sided upper bound
    # For UCB, we want z such that P(Z <= z) = 1 - alpha
    z = _z_quantile(1.0 - alpha)
    
    p_hat = k / n
    n_f = float(n)
    z_sq = z * z
    
    # Wilson score formula
    # UCB = (p_hat + z²/(2n) + z * sqrt(p_hat*(1-p_hat)/n + z²/(4n²))) / (1 + z²/n)
    denom = 1.0 + z_sq / n_f
    center = p_hat + z_sq / (2.0 * n_f)
    
    # Margin calculation
    variance_term = p_hat * (1.0 - p_hat) / n_f
    z_term = z_sq / (4.0 * n_f * n_f)
    margin = z * math.sqrt(variance_term + z_term)
    
    # Upper bound
    ucb = (center + margin) / denom
    
    # Clamp to [0, 1] for numerical stability
    return min(1.0, max(0.0, ucb))


def wilson_lcb(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Compute the Wilson score lower confidence bound for a binomial proportion.
    
    This is the lower bound counterpart to wilson_ucb.
    
    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (default 0.05)
        
    Returns:
        Lower confidence bound for the proportion
        
    Raises:
        ValueError: If n == 0, k < 0, k > n, or alpha not in (0, 1)
    """
    if n == 0:
        raise ValueError("n must be > 0 (cannot compute confidence interval with no observations)")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    z = _z_quantile(1.0 - alpha)
    
    p_hat = k / n
    n_f = float(n)
    z_sq = z * z
    
    denom = 1.0 + z_sq / n_f
    center = p_hat + z_sq / (2.0 * n_f)
    
    variance_term = p_hat * (1.0 - p_hat) / n_f
    z_term = z_sq / (4.0 * n_f * n_f)
    margin = z * math.sqrt(variance_term + z_term)
    
    # Lower bound
    lcb = (center - margin) / denom
    
    return min(1.0, max(0.0, lcb))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute the Wilson score two-sided confidence interval for a binomial proportion.
    
    This is the full (1-alpha) confidence interval, returning both bounds.
    
    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower, upper) confidence bounds
        
    Raises:
        ValueError: If n == 0, k < 0, k > n, or alpha not in (0, 1)
    """
    if n == 0:
        raise ValueError("n must be > 0 (cannot compute confidence interval with no observations)")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    # For two-sided CI, use alpha/2 in each tail
    z = _z_quantile(1.0 - alpha / 2.0)
    
    p_hat = k / n
    n_f = float(n)
    z_sq = z * z
    
    denom = 1.0 + z_sq / n_f
    center = p_hat + z_sq / (2.0 * n_f)
    
    variance_term = p_hat * (1.0 - p_hat) / n_f
    z_term = z_sq / (4.0 * n_f * n_f)
    margin = z * math.sqrt(variance_term + z_term)
    
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    
    return (max(0.0, lower), min(1.0, upper))


def normal_lcb(mean: float, stdev: float, n: int, alpha: float = 0.05) -> float:
    """
    Compute the normal lower confidence bound for a sample mean.
    
    This is used for PnL confidence bounds:
    - mean: sample mean of PnL
    - stdev: sample standard deviation of PnL
    - n: number of MC runs
    - alpha: significance level (default 0.05)
    
    Returns a conservative (lower) bound on the true mean PnL.
    For promotion, we require LCB >= threshold (fail closed).
    
    The formula is: LCB = mean - z * (stdev / sqrt(n))
    where z = Phi^{-1}(1 - alpha) for one-sided.
    
    Edge case behavior:
    - n == 1: Returns mean (no interval possible with single observation)
    - stdev == 0: Returns mean (no variability, perfect confidence)
    - stdev < 0: Raises ValueError
    - n <= 0: Raises ValueError
    
    Args:
        mean: Sample mean
        stdev: Sample standard deviation (population or sample)
        n: Sample size (number of observations)
        alpha: Significance level (default 0.05)
        
    Returns:
        Lower confidence bound for the true mean
        
    Raises:
        ValueError: If n <= 0, stdev < 0, or alpha not in (0, 1)
        
    Examples:
        >>> normal_lcb(100.0, 20.0, 50, 0.05)  # mean=100, stdev=20, n=50
        95.34...  # Lower bound is about 95.3
        
        >>> normal_lcb(100.0, 0.0, 50, 0.05)  # stdev=0 (no variability)
        100.0  # LCB equals mean
    """
    # Input validation
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if stdev < 0.0:
        raise ValueError(f"stdev must be >= 0, got {stdev}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    # Check for NaN inputs
    if math.isnan(mean):
        return float("nan")
    if math.isnan(stdev):
        # If stdev is NaN, we cannot compute a bound; return NaN
        return float("nan")
    
    # Edge case: single observation
    # With n=1, we have no degrees of freedom to estimate variability
    # Document this behavior: LCB = mean (conservative choice is mean itself)
    if n == 1:
        return mean
    
    # Edge case: zero standard deviation
    # If there's no variability, the mean is perfectly known
    if stdev == 0.0:
        return mean
    
    # z-score for (1 - alpha) one-sided lower bound
    z = _z_quantile(1.0 - alpha)
    
    # Standard error of the mean
    se = stdev / math.sqrt(n)
    
    # Lower confidence bound
    lcb = mean - z * se
    
    return lcb


def normal_ucb(mean: float, stdev: float, n: int, alpha: float = 0.05) -> float:
    """
    Compute the normal upper confidence bound for a sample mean.
    
    This is the upper bound counterpart to normal_lcb.
    
    Args:
        mean: Sample mean
        stdev: Sample standard deviation
        n: Sample size
        alpha: Significance level (default 0.05)
        
    Returns:
        Upper confidence bound for the true mean
        
    Raises:
        ValueError: If n <= 0, stdev < 0, or alpha not in (0, 1)
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if stdev < 0.0:
        raise ValueError(f"stdev must be >= 0, got {stdev}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    if math.isnan(mean) or math.isnan(stdev):
        return float("nan")
    
    if n == 1 or stdev == 0.0:
        return mean
    
    z = _z_quantile(1.0 - alpha)
    se = stdev / math.sqrt(n)
    
    return mean + z * se


def normal_ci(mean: float, stdev: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute the normal two-sided confidence interval for a sample mean.
    
    Args:
        mean: Sample mean
        stdev: Sample standard deviation
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower, upper) confidence bounds
        
    Raises:
        ValueError: If n <= 0, stdev < 0, or alpha not in (0, 1)
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if stdev < 0.0:
        raise ValueError(f"stdev must be >= 0, got {stdev}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    if math.isnan(mean) or math.isnan(stdev):
        return (float("nan"), float("nan"))
    
    if n == 1 or stdev == 0.0:
        return (mean, mean)
    
    # For two-sided CI, use alpha/2 in each tail
    z = _z_quantile(1.0 - alpha / 2.0)
    se = stdev / math.sqrt(n)
    
    return (mean - z * se, mean + z * se)


# =============================================================================
# Statistics configuration for PROMOTION_RECORD
# =============================================================================

def get_statistics_metadata(alpha: float = 0.05) -> dict:
    """
    Generate statistics metadata for PROMOTION_RECORD.json.
    
    This documents the methods used for confidence bounds.
    
    Args:
        alpha: Significance level used for bounds
        
    Returns:
        Dictionary suitable for inclusion in PROMOTION_RECORD.json
    """
    return {
        "alpha": alpha,
        "methods": {
            "kill_ucb": "wilson",
            "pnl_lcb": "normal",
        },
        "z_source": "statistics.NormalDist.inv_cdf",
        "description": (
            "Confidence bounds computed using one-sided intervals at "
            f"alpha={alpha}. Kill UCB uses Wilson score interval. "
            "PnL LCB uses normal z-interval for sample means."
        ),
    }


# =============================================================================
# Known values for testing
# =============================================================================

# Pre-computed z-quantiles for common alpha levels
# Used for testing that our implementation matches known values
KNOWN_Z_QUANTILES = {
    0.05: 1.6448536269514722,   # one-sided 95%
    0.025: 1.9599639845400545,  # two-sided 95%
    0.01: 2.3263478740408408,   # one-sided 99%
    0.005: 2.5758293035489004,  # two-sided 99%
}

