// src/tail_risk.rs
//
// Tail risk metrics for Monte Carlo analysis (Phase A).
//
// Provides:
// - VaR (Value at Risk) at configurable alpha levels
// - CVaR (Conditional VaR / Expected Shortfall)
// - Quantile computations for PnL and drawdown distributions
// - Wilson score confidence intervals for kill probability
//
// All functions are deterministic and avoid HashMap to ensure stable ordering.

use serde::Serialize;

/// Default alpha level for VaR/CVaR (95% confidence).
pub const DEFAULT_VAR_ALPHA: f64 = 0.95;

/// Standard quantile levels for tail risk reporting.
pub const STANDARD_QUANTILES: [f64; 5] = [0.01, 0.05, 0.50, 0.95, 0.99];

// ============================================================================
// Quantile computation
// ============================================================================

/// Compute a single quantile from a sorted slice using linear interpolation.
/// `p` is in [0, 1]. Returns NaN for empty slices.
pub fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let p = p.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n.saturating_sub(1) as f64);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi || lo >= n {
        return sorted[lo.min(n - 1)];
    }
    let w = idx - (lo as f64);
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// Compute multiple quantiles at once. Input is sorted in-place.
/// Returns quantile values in the same order as `ps`.
pub fn compute_quantiles(data: &mut [f64], ps: &[f64]) -> Vec<f64> {
    // Filter non-finite values and sort
    let mut finite: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    ps.iter().map(|&p| quantile_sorted(&finite, p)).collect()
}

// ============================================================================
// VaR and CVaR
// ============================================================================

/// Value at Risk at confidence level alpha.
/// For loss distributions (lower = worse), VaR_alpha is the (1-alpha) quantile.
/// For PnL (higher = better), VaR_alpha at 0.95 gives the 5th percentile (worst 5%).
///
/// This function computes the (1-alpha) quantile of the input data.
pub fn var_at_alpha(data: &mut [f64], alpha: f64) -> f64 {
    let p = 1.0 - alpha;
    compute_quantiles(data, &[p])[0]
}

/// Conditional VaR (Expected Shortfall) at confidence level alpha.
/// CVaR_alpha is the mean of all values below (or equal to) VaR_alpha.
///
/// For a 95% confidence level, this is the average of the worst 5% of outcomes.
pub fn cvar_at_alpha(data: &mut [f64], alpha: f64) -> f64 {
    let var = var_at_alpha(data, alpha);
    if !var.is_finite() {
        return f64::NAN;
    }

    // Collect all values <= VaR
    let tail: Vec<f64> = data
        .iter()
        .copied()
        .filter(|&x| x.is_finite() && x <= var)
        .collect();

    if tail.is_empty() {
        return var; // Edge case: return VaR itself
    }

    tail.iter().sum::<f64>() / tail.len() as f64
}

// ============================================================================
// Wilson confidence interval for proportions
// ============================================================================

/// Wilson score confidence interval for a binomial proportion.
///
/// Given `k` successes out of `n` trials, compute the (1-alpha) CI.
/// Uses Wilson score interval which handles edge cases (0/n, n/n, small n) well.
///
/// Returns (lower, upper) bounds.
/// For z, we use the standard normal quantile; for 95% CI, z ≈ 1.96.
pub fn wilson_ci(k: u64, n: u64, alpha: f64) -> (f64, f64) {
    if n == 0 {
        // Undefined case: no data
        return (0.0, 1.0);
    }

    // z-score for (1-alpha) confidence
    // For alpha=0.05, z ≈ 1.96
    let z = normal_quantile(1.0 - alpha / 2.0);

    let p_hat = k as f64 / n as f64;
    let n_f = n as f64;

    let denom = 1.0 + z * z / n_f;
    let center = p_hat + z * z / (2.0 * n_f);
    let margin = z * ((p_hat * (1.0 - p_hat) / n_f) + (z * z / (4.0 * n_f * n_f))).sqrt();

    let lower = ((center - margin) / denom).max(0.0);
    let upper = ((center + margin) / denom).min(1.0);

    (lower, upper)
}

/// Approximate inverse normal CDF (quantile function).
/// Uses Abramowitz-Stegun approximation. Good for p in (0.001, 0.999).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Use symmetry: if p > 0.5, compute for (1-p) and negate
    let (sign, p_adj) = if p > 0.5 { (1.0, 1.0 - p) } else { (-1.0, p) };

    // Rational approximation for tail
    let t = (-2.0 * p_adj.ln()).sqrt();

    // Coefficients from Abramowitz-Stegun 26.2.23
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;

    sign * (t - numerator / denominator)
}

// ============================================================================
// Output structures (stable field ordering via struct, not HashMap)
// ============================================================================

/// Quantile set for a metric (deterministic field order).
#[derive(Debug, Clone, Serialize)]
pub struct QuantileSet {
    pub p01: f64,
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl QuantileSet {
    pub fn from_data(data: &mut [f64]) -> Self {
        let qs = compute_quantiles(data, &STANDARD_QUANTILES);
        Self {
            p01: qs[0],
            p05: qs[1],
            p50: qs[2],
            p95: qs[3],
            p99: qs[4],
        }
    }
}

/// VaR/CVaR pair at a specific alpha level.
#[derive(Debug, Clone, Serialize)]
pub struct VaRCVaR {
    pub alpha: f64,
    pub var: f64,
    pub cvar: f64,
}

impl VaRCVaR {
    pub fn from_data(data: &mut [f64], alpha: f64) -> Self {
        Self {
            alpha,
            var: var_at_alpha(data, alpha),
            cvar: cvar_at_alpha(data, alpha),
        }
    }
}

/// Wilson confidence interval for kill probability.
#[derive(Debug, Clone, Serialize)]
pub struct KillProbabilityCI {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub ci_level: f64, // e.g., 0.95 for 95% CI
    pub kill_count: u64,
    pub total_runs: u64,
}

impl KillProbabilityCI {
    pub fn new(kill_count: u64, total_runs: u64, ci_level: f64) -> Self {
        let point_estimate = if total_runs > 0 {
            kill_count as f64 / total_runs as f64
        } else {
            0.0
        };
        let alpha = 1.0 - ci_level;
        let (ci_lower, ci_upper) = wilson_ci(kill_count, total_runs, alpha);
        Self {
            point_estimate,
            ci_lower,
            ci_upper,
            ci_level,
            kill_count,
            total_runs,
        }
    }
}

/// Complete tail risk metrics for mc_summary.json.
#[derive(Debug, Clone, Serialize)]
pub struct TailRiskMetrics {
    pub schema_version: u32,
    pub pnl_quantiles: QuantileSet,
    pub pnl_var_cvar: VaRCVaR,
    pub max_drawdown_quantiles: QuantileSet,
    pub max_drawdown_var_cvar: VaRCVaR,
    pub kill_probability: KillProbabilityCI,
}

impl TailRiskMetrics {
    /// Compute tail risk metrics from Monte Carlo samples.
    ///
    /// # Arguments
    /// * `pnl_samples` - Final PnL values from each run
    /// * `drawdown_samples` - Max drawdown values from each run
    /// * `kill_count` - Number of runs that triggered kill switch
    /// * `var_alpha` - Alpha level for VaR/CVaR (default 0.95)
    /// * `ci_level` - Confidence level for Wilson CI (default 0.95)
    pub fn compute(
        pnl_samples: &[f64],
        drawdown_samples: &[f64],
        kill_count: u64,
        var_alpha: f64,
        ci_level: f64,
    ) -> Self {
        let total_runs = pnl_samples.len() as u64;

        let mut pnl = pnl_samples.to_vec();
        let mut dd = drawdown_samples.to_vec();

        Self {
            schema_version: 1,
            pnl_quantiles: QuantileSet::from_data(&mut pnl),
            pnl_var_cvar: VaRCVaR::from_data(&mut pnl, var_alpha),
            max_drawdown_quantiles: QuantileSet::from_data(&mut dd),
            max_drawdown_var_cvar: VaRCVaR::from_data(&mut dd, var_alpha),
            kill_probability: KillProbabilityCI::new(kill_count, total_runs, ci_level),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps || (a.is_nan() && b.is_nan())
    }

    // ------------------------------------------------------------------------
    // VaR/CVaR tests on known vectors
    // ------------------------------------------------------------------------

    #[test]
    fn test_var_simple_vector() {
        // Simple sorted vector: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let mut data: Vec<f64> = (1..=10).map(|x| x as f64).collect();

        // VaR at 95% = 5th percentile = p=0.05
        // For 10 elements, index = 0.05 * 9 = 0.45
        // Linear interp: sorted[0] * 0.55 + sorted[1] * 0.45 = 1 * 0.55 + 2 * 0.45 = 1.45
        let var = var_at_alpha(&mut data, 0.95);
        assert!(
            approx_eq(var, 1.45, 0.01),
            "VaR(0.95) expected ~1.45, got {}",
            var
        );
    }

    #[test]
    fn test_var_all_negative() {
        // Loss vector: [-100, -80, -60, -40, -20]
        let mut data = vec![-100.0, -80.0, -60.0, -40.0, -20.0];

        // VaR at 95% = 5th percentile
        // For 5 elements at p=0.05, index = 0.05 * 4 = 0.2
        // Linear interp: sorted[0] * 0.8 + sorted[1] * 0.2 = -100 * 0.8 + -80 * 0.2 = -96
        let var = var_at_alpha(&mut data, 0.95);
        assert!(
            approx_eq(var, -96.0, 0.01),
            "VaR(0.95) expected -96, got {}",
            var
        );
    }

    #[test]
    fn test_cvar_simple_vector() {
        // Sorted vector: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let mut data: Vec<f64> = (1..=10).map(|x| x as f64).collect();

        // VaR at 95% ≈ 1.45
        // CVaR = mean of values <= 1.45 = just [1]
        let cvar = cvar_at_alpha(&mut data, 0.95);
        assert!(
            approx_eq(cvar, 1.0, 0.01),
            "CVaR(0.95) expected 1.0, got {}",
            cvar
        );
    }

    #[test]
    fn test_cvar_larger_sample() {
        // 100 values: 1..=100
        let mut data: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        // VaR at 95% = 5th percentile
        // p=0.05, index = 0.05 * 99 = 4.95
        // sorted[4] = 5, sorted[5] = 6
        // VaR ≈ 5 * 0.05 + 6 * 0.95 = 5.95
        let var = var_at_alpha(&mut data, 0.95);
        assert!((5.0..=6.0).contains(&var), "VaR(0.95) should be ~5.95");

        // CVaR = mean of values <= 5.95 = mean([1,2,3,4,5]) = 3
        let cvar = cvar_at_alpha(&mut data, 0.95);
        assert!(
            approx_eq(cvar, 3.0, 0.1),
            "CVaR(0.95) expected ~3.0, got {}",
            cvar
        );
    }

    #[test]
    fn test_var_cvar_empty() {
        let mut data: Vec<f64> = vec![];
        let var = var_at_alpha(&mut data, 0.95);
        let cvar = cvar_at_alpha(&mut data, 0.95);
        assert!(var.is_nan(), "VaR of empty should be NaN");
        assert!(cvar.is_nan(), "CVaR of empty should be NaN");
    }

    #[test]
    fn test_var_single_element() {
        let mut data = vec![42.0];
        let var = var_at_alpha(&mut data, 0.95);
        let cvar = cvar_at_alpha(&mut data, 0.95);
        assert!(
            approx_eq(var, 42.0, EPSILON),
            "VaR of single element should be that element"
        );
        assert!(
            approx_eq(cvar, 42.0, EPSILON),
            "CVaR of single element should be that element"
        );
    }

    #[test]
    fn test_var_ignores_nan() {
        let mut data = vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0];
        let var = var_at_alpha(&mut data, 0.95);
        // Should compute on [1,2,3,4,5] ignoring NaN
        assert!(var.is_finite(), "VaR should ignore NaN");
    }

    // ------------------------------------------------------------------------
    // Wilson CI tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_wilson_ci_zero_zero() {
        // 0/0 case
        let (lo, hi) = wilson_ci(0, 0, 0.05);
        assert!(
            approx_eq(lo, 0.0, EPSILON),
            "0/0 lower should be 0, got {}",
            lo
        );
        assert!(
            approx_eq(hi, 1.0, EPSILON),
            "0/0 upper should be 1, got {}",
            hi
        );
    }

    #[test]
    fn test_wilson_ci_zero_n() {
        // 0/100 case: point estimate is 0
        let (lo, hi) = wilson_ci(0, 100, 0.05);
        assert!(
            (0.0..0.01).contains(&lo),
            "0/100 lower should be very small"
        );
        assert!(hi > 0.0 && hi < 0.1, "0/100 upper should be small but > 0");
    }

    #[test]
    fn test_wilson_ci_n_n() {
        // 100/100 case: point estimate is 1
        let (lo, hi) = wilson_ci(100, 100, 0.05);
        assert!(lo > 0.9 && lo < 1.0, "100/100 lower should be near 1");
        assert!(
            approx_eq(hi, 1.0, 0.001),
            "100/100 upper should be 1, got {}",
            hi
        );
    }

    #[test]
    fn test_wilson_ci_small_n() {
        // 1/5 case: point estimate is 0.2
        let (lo, hi) = wilson_ci(1, 5, 0.05);
        assert!((0.0..0.2).contains(&lo), "1/5 lower should be < 0.2");
        assert!(hi > 0.2 && hi <= 1.0, "1/5 upper should be > 0.2");

        // Wilson interval is known to be conservative for small n
        // Should bracket the true proportion
    }

    #[test]
    fn test_wilson_ci_half() {
        // 50/100 case: point estimate is 0.5
        let (lo, hi) = wilson_ci(50, 100, 0.05);
        assert!(lo > 0.35 && lo < 0.5, "50/100 lower should be ~0.40");
        assert!(hi > 0.5 && hi < 0.65, "50/100 upper should be ~0.60");

        // Check symmetry around 0.5
        let center = (lo + hi) / 2.0;
        assert!(
            approx_eq(center, 0.5, 0.01),
            "50/100 CI should be centered near 0.5"
        );
    }

    #[test]
    fn test_wilson_ci_large_n() {
        // 500/10000 case: point estimate is 0.05
        let (lo, hi) = wilson_ci(500, 10000, 0.05);
        assert!(
            lo > 0.045 && lo < 0.05,
            "Large n lower bound should be tight"
        );
        assert!(
            hi > 0.05 && hi < 0.055,
            "Large n upper bound should be tight"
        );
    }

    #[test]
    fn test_wilson_ci_wider_at_higher_alpha() {
        let (lo_95, hi_95) = wilson_ci(50, 100, 0.05);
        let (lo_99, hi_99) = wilson_ci(50, 100, 0.01);

        let width_95 = hi_95 - lo_95;
        let width_99 = hi_99 - lo_99;

        assert!(
            width_99 > width_95,
            "99% CI should be wider than 95% CI: {} vs {}",
            width_99,
            width_95
        );
    }

    // ------------------------------------------------------------------------
    // Quantile tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_quantiles_standard() {
        let mut data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let qs = QuantileSet::from_data(&mut data);

        // p01 ≈ 1.99, p05 ≈ 5.95, p50 = 50.5, p95 ≈ 95.05, p99 ≈ 99.01
        assert!(qs.p01 >= 1.0 && qs.p01 <= 2.0, "p01 should be ~1-2");
        assert!(qs.p05 >= 5.0 && qs.p05 <= 6.0, "p05 should be ~5-6");
        assert!(
            approx_eq(qs.p50, 50.5, 0.1),
            "p50 should be 50.5, got {}",
            qs.p50
        );
        assert!(qs.p95 >= 95.0 && qs.p95 <= 96.0, "p95 should be ~95-96");
        assert!(qs.p99 >= 99.0 && qs.p99 <= 100.0, "p99 should be ~99-100");
    }

    // ------------------------------------------------------------------------
    // Integration test: TailRiskMetrics
    // ------------------------------------------------------------------------

    #[test]
    fn test_tail_risk_metrics_integration() {
        let pnl: Vec<f64> = (1..=100).map(|x| (x as f64) - 50.0).collect(); // -49 to +50
        let drawdown: Vec<f64> = (1..=100).map(|x| x as f64).collect(); // 1 to 100
        let kill_count = 10u64;

        let metrics = TailRiskMetrics::compute(&pnl, &drawdown, kill_count, 0.95, 0.95);

        assert_eq!(metrics.schema_version, 1);
        assert!(metrics.pnl_var_cvar.alpha == 0.95);
        assert!(metrics.kill_probability.kill_count == 10);
        assert!(metrics.kill_probability.total_runs == 100);
        assert!(approx_eq(
            metrics.kill_probability.point_estimate,
            0.1,
            EPSILON
        ));
    }

    // ------------------------------------------------------------------------
    // Normal quantile tests (helper function)
    // ------------------------------------------------------------------------

    #[test]
    fn test_normal_quantile_basic() {
        // z for 97.5% ≈ 1.96
        let z = normal_quantile(0.975);
        assert!(
            approx_eq(z, 1.96, 0.01),
            "normal_quantile(0.975) should be ~1.96, got {}",
            z
        );

        // z for 50% = 0
        let z_50 = normal_quantile(0.5);
        assert!(
            approx_eq(z_50, 0.0, 0.001),
            "normal_quantile(0.5) should be 0"
        );

        // Symmetry: z(0.025) ≈ -1.96
        let z_lo = normal_quantile(0.025);
        assert!(
            approx_eq(z_lo, -1.96, 0.01),
            "normal_quantile(0.025) should be ~-1.96"
        );
    }
}
