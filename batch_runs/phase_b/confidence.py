"""
confidence.py

Bootstrap & CI Engine for Phase B: Confidence-Aware Statistical Gating.

Implements:
- Block bootstrap with configurable block size (for time-series data)
- Mean, median, CVaR, max drawdown estimators
- Percentile-based confidence intervals
- Deterministic RNG seeding for reproducibility

This module uses only Python stdlib (no numpy/scipy) for statistical computations.

References:
- Block bootstrap: Künsch (1989), "The Jackknife and the Bootstrap for General Stationary Observations"
- CVaR: Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk"
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union


# =============================================================================
# Type aliases
# =============================================================================

# Use List[float] instead of numpy.ndarray
Array = List[float]
Estimator = Callable[[Array], float]


# =============================================================================
# Helper Functions (stdlib replacements for numpy)
# =============================================================================

def _cumsum(data: List[float]) -> List[float]:
    """Compute cumulative sum of a list."""
    if not data:
        return []
    result = [0.0] * len(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = result[i - 1] + data[i]
    return result


def _running_max(data: List[float]) -> List[float]:
    """Compute running maximum of a list."""
    if not data:
        return []
    result = [0.0] * len(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = max(result[i - 1], data[i])
    return result


def _percentile(sorted_data: List[float], p: float) -> float:
    """
    Compute p-th percentile of sorted data using linear interpolation.
    
    Args:
        sorted_data: Sorted list of values
        p: Percentile (0-100)
        
    Returns:
        Interpolated percentile value
    """
    if not sorted_data:
        return float("nan")
    
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    
    # Linear interpolation method (matches numpy's default)
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_data[int(k)]
    
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


# =============================================================================
# Statistical Estimators
# =============================================================================

def compute_mean(data: Array) -> float:
    """
    Compute the arithmetic mean.
    
    Args:
        data: List of values
        
    Returns:
        Mean of the data
    """
    if len(data) == 0:
        return float("nan")
    return statistics.mean(data)


def compute_median(data: Array) -> float:
    """
    Compute the median.
    
    Args:
        data: List of values
        
    Returns:
        Median of the data
    """
    if len(data) == 0:
        return float("nan")
    return statistics.median(data)


def compute_cvar(data: Array, alpha: float = 0.05) -> float:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR at level alpha is the expected value of the worst alpha*100% outcomes.
    For PnL, this is the expected loss in the worst tail.
    
    Args:
        data: List of values (e.g., PnL)
        alpha: Tail probability (default 0.05 for 5% CVaR)
        
    Returns:
        CVaR (mean of the worst alpha fraction of data)
        
    Note:
        For loss-oriented CVaR (drawdown), pass negative values or interpret accordingly.
    """
    if len(data) == 0:
        return float("nan")
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    
    sorted_data = sorted(data)
    n = len(data)
    
    # Number of observations in the tail
    k = max(1, math.ceil(n * alpha))
    
    # CVaR is mean of the k worst observations
    return statistics.mean(sorted_data[:k])


def compute_max_drawdown(equity_curve: Array) -> float:
    """
    Compute maximum drawdown from an equity curve.
    
    Maximum drawdown is the largest peak-to-trough decline.
    
    Args:
        equity_curve: List of cumulative PnL values
        
    Returns:
        Maximum drawdown (positive value representing the loss)
    """
    if len(equity_curve) == 0:
        return float("nan")
    if len(equity_curve) == 1:
        return 0.0
    
    # Running maximum
    running_max = _running_max(equity_curve)
    
    # Drawdown at each point
    drawdowns = [rm - ec for rm, ec in zip(running_max, equity_curve)]
    
    return max(drawdowns)


def compute_drawdown_series(pnl_series: Array) -> Array:
    """
    Compute drawdown series from PnL series.
    
    Args:
        pnl_series: List of per-period PnL values
        
    Returns:
        List of drawdowns at each point
    """
    if len(pnl_series) == 0:
        return []
    
    # Convert to equity curve
    equity = _cumsum(pnl_series)
    
    # Running maximum
    running_max = _running_max(equity)
    
    # Drawdown at each point
    return [rm - eq for rm, eq in zip(running_max, equity)]


def compute_max_drawdowns_per_run(pnl_per_run: List[Array]) -> Array:
    """
    Compute max drawdown for each run.
    
    Args:
        pnl_per_run: List of PnL lists, one per run
        
    Returns:
        List of max drawdowns, one per run
    """
    return [compute_max_drawdown(_cumsum(pnl)) for pnl in pnl_per_run]


# =============================================================================
# Block Bootstrap
# =============================================================================

@dataclass
class BlockBootstrap:
    """
    Block bootstrap resampling for time-series data.
    
    Block bootstrap preserves temporal dependencies by resampling contiguous
    blocks rather than individual observations.
    
    Attributes:
        block_size: Size of each block (default: sqrt(n))
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
    """
    block_size: Optional[int] = None
    n_bootstrap: int = 1000
    seed: int = 42
    
    def resample(self, data: Array, rng: random.Random) -> Array:
        """
        Generate a single bootstrap sample using block resampling.
        
        Args:
            data: List to resample
            rng: Python random.Random instance
            
        Returns:
            Bootstrap sample of same length as data
        """
        n = len(data)
        if n == 0:
            return list(data)
        
        # Default block size: sqrt(n) as per Künsch (1989)
        block_size = self.block_size if self.block_size else max(1, int(math.sqrt(n)))
        
        # Number of blocks needed (may slightly overshoot)
        n_blocks = math.ceil(n / block_size)
        
        # Randomly select block starting positions
        # Each block starts at a random position in [0, n - block_size]
        max_start = max(0, n - block_size)
        if max_start == 0:
            # Data shorter than block size - just return shuffled copy
            return [data[rng.randrange(n)] for _ in range(n)]
        
        # Extract blocks and concatenate
        blocks: List[float] = []
        for _ in range(n_blocks):
            start = rng.randint(0, max_start)
            end = min(start + block_size, n)
            blocks.extend(data[start:end])
        
        # Trim to original length
        return blocks[:n]
    
    def bootstrap_samples(
        self,
        data: Array,
        estimator: Estimator,
    ) -> Array:
        """
        Generate bootstrap distribution of an estimator.
        
        Args:
            data: List of observations
            estimator: Function mapping list to scalar statistic
            
        Returns:
            List of bootstrap estimates (length n_bootstrap)
        """
        rng = random.Random(self.seed)
        
        estimates: List[float] = []
        for _ in range(self.n_bootstrap):
            sample = self.resample(data, rng)
            estimates.append(estimator(sample))
        
        return estimates
    
    def confidence_interval(
        self,
        data: Array,
        estimator: Estimator,
        alpha: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Compute percentile confidence interval via bootstrap.
        
        Args:
            data: List of observations
            estimator: Function mapping list to scalar statistic
            alpha: Significance level (default 0.05 for 95% CI)
            
        Returns:
            Tuple of (lower, point_estimate, upper)
        """
        if len(data) == 0:
            return (float("nan"), float("nan"), float("nan"))
        
        # Point estimate
        point = estimator(data)
        
        # Bootstrap distribution
        boot_estimates = self.bootstrap_samples(data, estimator)
        
        # Percentile confidence interval
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        
        sorted_estimates = sorted(boot_estimates)
        lower = _percentile(sorted_estimates, lower_pct)
        upper = _percentile(sorted_estimates, upper_pct)
        
        return (lower, point, upper)


# =============================================================================
# Convenience Functions
# =============================================================================

def bootstrap_ci(
    data: Union[Array, List[float]],
    estimator: Estimator,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: List of observations
        estimator: Function mapping list to scalar statistic
        alpha: Significance level (default 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for block bootstrap (None = sqrt(n))
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (lower, point_estimate, upper)
        
    Example:
        >>> data = [random.gauss(0, 1) for _ in range(100)]
        >>> lower, point, upper = bootstrap_ci(data, compute_mean, alpha=0.05)
    """
    arr = list(data)
    bootstrap = BlockBootstrap(
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return bootstrap.confidence_interval(arr, estimator, alpha)


# =============================================================================
# Confidence Interval Result
# =============================================================================

@dataclass
class ConfidenceInterval:
    """
    Result of a confidence interval computation.
    
    Attributes:
        lower: Lower bound of the CI
        point: Point estimate
        upper: Upper bound of the CI
        alpha: Significance level
        n_samples: Number of observations
        n_bootstrap: Number of bootstrap samples used
        estimator_name: Name of the estimator (e.g., "mean", "cvar")
    """
    lower: float
    point: float
    upper: float
    alpha: float
    n_samples: int
    n_bootstrap: int
    estimator_name: str
    
    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower
    
    @property
    def contains_zero(self) -> bool:
        """Check if the CI contains zero."""
        return self.lower <= 0 <= self.upper
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lower": self.lower,
            "point": self.point,
            "upper": self.upper,
            "alpha": self.alpha,
            "n_samples": self.n_samples,
            "n_bootstrap": self.n_bootstrap,
            "estimator_name": self.estimator_name,
            "width": self.width,
        }


def compute_ci(
    data: Union[Array, List[float]],
    estimator: Estimator,
    estimator_name: str,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    seed: int = 42,
) -> ConfidenceInterval:
    """
    Compute a ConfidenceInterval object for a statistic.
    
    Args:
        data: List of observations
        estimator: Function mapping list to scalar statistic
        estimator_name: Name of the estimator for reporting
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for block bootstrap
        seed: Random seed
        
    Returns:
        ConfidenceInterval object
    """
    arr = list(data)
    lower, point, upper = bootstrap_ci(
        arr, estimator, alpha, n_bootstrap, block_size, seed
    )
    return ConfidenceInterval(
        lower=lower,
        point=point,
        upper=upper,
        alpha=alpha,
        n_samples=len(arr),
        n_bootstrap=n_bootstrap,
        estimator_name=estimator_name,
    )


# =============================================================================
# Metrics from Run Data
# =============================================================================

@dataclass
class RunMetrics:
    """
    Computed metrics with confidence intervals for a simulation run.
    
    Attributes:
        pnl_mean: Mean PnL with CI
        pnl_median: Median PnL with CI
        pnl_cvar: PnL CVaR (tail risk) with CI
        max_drawdown: Maximum drawdown with CI
        kill_rate: Kill switch activation rate
        n_runs: Number of simulation runs
    """
    pnl_mean: ConfidenceInterval
    pnl_median: ConfidenceInterval
    pnl_cvar: ConfidenceInterval
    max_drawdown: ConfidenceInterval
    kill_rate: float
    kill_count: int
    n_runs: int
    alpha: float = 0.05
    seed: int = 42
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pnl_mean": self.pnl_mean.to_dict(),
            "pnl_median": self.pnl_median.to_dict(),
            "pnl_cvar": self.pnl_cvar.to_dict(),
            "max_drawdown": self.max_drawdown.to_dict(),
            "kill_rate": self.kill_rate,
            "kill_count": self.kill_count,
            "n_runs": self.n_runs,
            "alpha": self.alpha,
            "seed": self.seed,
        }


def compute_run_metrics(
    pnl_per_run: List[Array],
    kill_flags: List[bool],
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    seed: int = 42,
    cvar_alpha: float = 0.05,
) -> RunMetrics:
    """
    Compute all metrics with confidence intervals from simulation runs.
    
    Args:
        pnl_per_run: List of PnL lists, one per run
        kill_flags: List of boolean kill switch flags, one per run
        alpha: Significance level for CIs
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for bootstrap
        seed: Random seed
        cvar_alpha: Alpha for CVaR computation (tail size)
        
    Returns:
        RunMetrics object with all computed metrics
    """
    n_runs = len(pnl_per_run)
    
    if n_runs == 0:
        nan_ci = ConfidenceInterval(
            lower=float("nan"),
            point=float("nan"),
            upper=float("nan"),
            alpha=alpha,
            n_samples=0,
            n_bootstrap=n_bootstrap,
            estimator_name="",
        )
        return RunMetrics(
            pnl_mean=nan_ci,
            pnl_median=nan_ci,
            pnl_cvar=nan_ci,
            max_drawdown=nan_ci,
            kill_rate=float("nan"),
            kill_count=0,
            n_runs=0,
            alpha=alpha,
            seed=seed,
        )
    
    # Aggregate metrics per run
    final_pnl = [sum(pnl) for pnl in pnl_per_run]
    max_drawdowns = compute_max_drawdowns_per_run(pnl_per_run)
    
    # Kill rate
    kill_count = sum(kill_flags)
    kill_rate = kill_count / n_runs
    
    # Compute CIs for each metric
    pnl_mean_ci = compute_ci(
        final_pnl, compute_mean, "pnl_mean",
        alpha, n_bootstrap, block_size, seed
    )
    
    pnl_median_ci = compute_ci(
        final_pnl, compute_median, "pnl_median",
        alpha, n_bootstrap, block_size, seed + 1  # Different seed for independence
    )
    
    # CVaR estimator with fixed alpha
    def cvar_estimator(data: Array) -> float:
        return compute_cvar(data, cvar_alpha)
    
    pnl_cvar_ci = compute_ci(
        final_pnl, cvar_estimator, "pnl_cvar",
        alpha, n_bootstrap, block_size, seed + 2
    )
    
    max_dd_ci = compute_ci(
        max_drawdowns, compute_mean, "max_drawdown",
        alpha, n_bootstrap, block_size, seed + 3
    )
    
    return RunMetrics(
        pnl_mean=pnl_mean_ci,
        pnl_median=pnl_median_ci,
        pnl_cvar=pnl_cvar_ci,
        max_drawdown=max_dd_ci,
        kill_rate=kill_rate,
        kill_count=kill_count,
        n_runs=n_runs,
        alpha=alpha,
        seed=seed,
    )
