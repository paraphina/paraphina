// src/engine.rs
//
// Paraphina MM engine:
//  - seeds synthetic per-venue mids/spreads/depth and local vols,
//  - aggregates a cross-venue fair value observation,
//  - runs a simple EWMA “Kalman-lite” filter on price,
//  - updates effective volatility + control scalars,
//  - runs toxicity + venue health,
//  - updates risk limits + risk regime,
//  - delegates post-fill recomputations to GlobalState.

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
use crate::toxicity::update_toxicity_and_health;

pub struct Engine {
    cfg: Config,
}

impl Engine {
    pub fn new(cfg: &Config) -> Self {
        Self { cfg: cfg.clone() }
    }

    /// Seed synthetic mids / spreads / depth for all venues and update
    /// per-venue local volatility estimates.
    ///
    /// This is a deterministic function of `now_ms` and venue index. The goal
    /// is not to be realistic, but to give the strategy a smooth price path
    /// with non-zero returns so that the volatility machinery is exercised.
    pub fn seed_dummy_mids(&self, state: &mut GlobalState, now_ms: i64) {
        let base = 250.0 + (now_ms as f64) * 0.001;
        let vol_cfg = &self.cfg.volatility;

        for (idx, v) in state.venues.iter_mut().enumerate() {
            let offset = idx as f64 * 0.4;
            let mid_prev = v.mid;

            let mid = base + offset;
            let spread = 0.4 + idx as f64 * 0.02;
            let depth = 10_000.0;

            // Update order-book snapshot.
            v.mid = Some(mid);
            v.spread = Some(spread);
            v.last_mid_update_ms = Some(now_ms);
            v.depth_near_mid = depth;

            // --- Local per-venue volatility (short / long EWMA of log returns) ---
            if let Some(prev) = mid_prev {
                if prev > 0.0 && mid > 0.0 {
                    let r = (mid / prev).ln();
                    let r2 = r * r;

                    let alpha_short = vol_cfg.fv_vol_alpha_short;
                    let alpha_long = vol_cfg.fv_vol_alpha_long;

                    let var_short_prev = v.local_vol_short * v.local_vol_short;
                    let var_long_prev = v.local_vol_long * v.local_vol_long;

                    let var_short_new = (1.0 - alpha_short) * var_short_prev + alpha_short * r2;
                    let var_long_new = (1.0 - alpha_long) * var_long_prev + alpha_long * r2;

                    v.local_vol_short = var_short_new.max(0.0).sqrt();
                    v.local_vol_long = var_long_new.max(0.0).sqrt();
                }
            }
        }
    }

    /// Main per-tick engine step.
    ///
    /// Sections (roughly matching the whitepaper):
    ///   5) fair value / filtering
    ///   6) volatility + control scalars
    ///   7) toxicity / venue health
    ///   8, 14) risk limits + risk regime / kill switch
    pub fn main_tick(&self, state: &mut GlobalState, now_ms: i64) {
        // 1) Fair value + volatility + control scalars ------------------------
        self.update_fair_value_and_vol(state, now_ms);

        // 2) Toxicity + venue health -----------------------------------------
        update_toxicity_and_health(state, &self.cfg);

        // 3) Risk limits + risk regime / kill switch -------------------------
        self.update_risk_limits_and_regime(state);
    }

    /// Aggregate fair value observation, run a “Kalman-lite” EWMA filter,
    /// and update sigma_eff + spread/size/band multipliers.
    fn update_fair_value_and_vol(&self, state: &mut GlobalState, now_ms: i64) {
        let (s_obs, avg_spread, avg_depth) = match self.compute_agg_mid_spread(state) {
            Some(t) => t,
            None => {
                // No usable books this tick – keep previous fair value / sigma.
                let _ = now_ms;
                return;
            }
        };

        // ----- Price filter (very simple EWMA over the aggregated mid) -----
        let prev_fair = state.fair_value.unwrap_or(s_obs);
        // Small smoothing factor – “Kalman-lite”.
        let alpha_price = 0.2_f64;
        let s_t = alpha_price * s_obs + (1.0 - alpha_price) * prev_fair;

        state.fair_value_prev = prev_fair;
        state.fair_value = Some(s_t);

        // We keep KF fields populated even though we’re not running a full KF yet.
        state.kf_x_hat = s_t.ln();
        state.kf_p = 1.0;
        state.kf_last_update_ms = now_ms;

        // ----- Volatility EWMAs -> sigma_eff --------------------------------
        let vol_cfg = &self.cfg.volatility;

        let ret = if prev_fair > 0.0 && s_t > 0.0 {
            (s_t / prev_fair).ln()
        } else {
            0.0
        };
        let r2 = ret * ret;

        // Interpret sigma_eff^2 as the current variance estimate,
        // and run a single-scale EWMA on variance.
        let var_prev = if state.sigma_eff > 0.0 {
            state.sigma_eff * state.sigma_eff
        } else {
            vol_cfg.sigma_min * vol_cfg.sigma_min
        };

        let alpha_var = vol_cfg.fv_vol_alpha_short; // use short-horizon alpha
        let var_new = (1.0 - alpha_var) * var_prev + alpha_var * r2;

        let mut sigma_eff = var_new.max(0.0).sqrt();
        if !sigma_eff.is_finite() || sigma_eff < vol_cfg.sigma_min {
            sigma_eff = vol_cfg.sigma_min;
        }

        state.sigma_eff = sigma_eff;

        // ----- Map sigma_eff -> vol_ratio_clipped + control scalars ---------
        let mut vol_ratio = if vol_cfg.vol_ref > 0.0 {
            sigma_eff / vol_cfg.vol_ref
        } else {
            1.0
        };
        if !vol_ratio.is_finite() {
            vol_ratio = 1.0;
        }

        let vol_ratio_clipped = vol_ratio
            .max(vol_cfg.vol_ratio_min)
            .min(vol_cfg.vol_ratio_max);

        state.vol_ratio_clipped = vol_ratio_clipped;

        let bump = (vol_ratio_clipped - 1.0).max(0.0);

        // spread_mult grows with volatility; size_mult and band_mult shrink.
        state.spread_mult = 1.0 + vol_cfg.spread_vol_mult_coeff * bump;
        state.size_mult = 1.0 / (1.0 + vol_cfg.size_vol_mult_coeff * bump);
        state.band_mult = 1.0 / (1.0 + vol_cfg.band_vol_mult_coeff * bump);

        // avg_spread / avg_depth are currently just diagnostics, but keep
        // them wired to avoid “unused” warnings when we start logging them.
        let _ = (avg_spread, avg_depth);
    }

    /// Cross-venue liquidity-weighted mid + average spread + depth.
    fn compute_agg_mid_spread(&self, state: &GlobalState) -> Option<(f64, f64, f64)> {
        let mut weight_sum: f64 = 0.0;
        let mut mid_num: f64 = 0.0;
        let mut spread_sum: f64 = 0.0;
        let mut depth_sum: f64 = 0.0;
        let mut count: f64 = 0.0;

        for v in &state.venues {
            // Only use venues that currently have a usable mid + spread.
            if let (Some(mid), Some(spread)) = (v.mid, v.spread) {
                if v.depth_near_mid <= 0.0 {
                    continue;
                }

                let depth = v.depth_near_mid;
                let weight = depth; // simple depth-proportional weighting

                weight_sum += weight;
                mid_num += weight * mid;
                spread_sum += spread;
                depth_sum += depth;
                count += 1.0;
            }
        }

        if weight_sum <= 0.0 || count < 1.0 {
            return None;
        }

        let mid = mid_num / weight_sum;
        let avg_spread = spread_sum / count.max(1.0);
        let avg_depth = depth_sum / count.max(1.0);

        Some((mid, avg_spread, avg_depth))
    }

    /// Vol-scaled risk limits + discrete risk regime + kill switch.
    fn update_risk_limits_and_regime(&self, state: &mut GlobalState) {
        let risk_cfg = &self.cfg.risk;

        // ---- 1) Vol-scaled delta limit + static basis limits ---------------
        let vol_ratio = state.vol_ratio_clipped.max(0.25);
        state.delta_limit_usd = risk_cfg.delta_hard_limit_usd_base / vol_ratio;

        state.basis_limit_hard_usd = risk_cfg.basis_hard_limit_usd;
        state.basis_limit_warn_usd = risk_cfg.basis_hard_limit_usd * risk_cfg.basis_warn_frac;

        // ---- 2) Aggregate daily PnL fields --------------------------------
        // NOTE: field names in GlobalState today are `daily_realised_pnl`
        // and `daily_unrealised_pnl`.
        state.daily_pnl_total = state.daily_realised_pnl + state.daily_unrealised_pnl;

        // ---- 3) Choose risk regime based on delta / basis / PnL -----------
        let delta_abs = state.dollar_delta_usd.abs();
        let basis_abs = state.basis_usd.abs();
        let pnl = state.daily_pnl_total;

        let delta_warn = risk_cfg.delta_warn_frac * state.delta_limit_usd;
        let basis_warn = risk_cfg.basis_warn_frac * risk_cfg.basis_hard_limit_usd;
        // All of these are negative numbers; pnl_warn is a “less negative”
        // early-warning threshold.
        let pnl_warn = risk_cfg.pnl_warn_frac * risk_cfg.daily_loss_limit;

        let mut regime = RiskRegime::Normal;

        // Hard-limit regime if *any* hard constraint breached.
        if delta_abs >= state.delta_limit_usd
            || basis_abs >= risk_cfg.basis_hard_limit_usd
            || pnl <= risk_cfg.daily_loss_limit
        {
            regime = RiskRegime::HardLimit;
        }
        // Warning regime if *any* warning threshold breached
        // (and hard limit not already hit).
        else if delta_abs >= delta_warn || basis_abs >= basis_warn || pnl <= pnl_warn {
            regime = RiskRegime::Warning;
        }

        // Kill-switch if we are in HardLimit and PnL has breached the hard
        // daily loss limit. This can be made stricter later if desired.
        let kill_switch =
            matches!(regime, RiskRegime::HardLimit) && pnl <= risk_cfg.daily_loss_limit;

        state.risk_regime = regime;
        state.kill_switch = kill_switch;
    }

    /// Delegate post-fill recomputations (inventory, basis, PnL, etc.)
    /// to GlobalState. This assumes you have:
    ///   impl GlobalState {
    ///       pub fn recompute_after_fills(&mut self, cfg: &Config) { ... }
    ///   }
    pub fn recompute_after_fills(&self, state: &mut GlobalState) {
        state.recompute_after_fills(&self.cfg);
    }
}
