// src/engine.rs
//
// Paraphina MM engine:
//
//  - seeds synthetic per-venue mids/spreads/depth and local vols,
//  - aggregates a cross-venue fair value observation,
//  - runs a 1D log-price Kalman filter (with spread/depth-aware noise),
//  - updates effective volatility + control scalars,
//  - runs toxicity + venue health,
//  - updates risk limits + risk regime,
//  - delegates post-fill recomputations to GlobalState.

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
use crate::toxicity::update_toxicity_and_health;
use crate::types::VenueStatus;

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
    ///   5) fair value / filtering (Kalman over log price)
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

    /// Aggregate fair value observation, run a 1D log-price Kalman filter,
    /// and update sigma_eff + spread/size/band multipliers.
    fn update_fair_value_and_vol(&self, state: &mut GlobalState, now_ms: i64) {
        let book_cfg = &self.cfg.book;
        let k_cfg = &self.cfg.kalman;

        // --- 1) Aggregate a cross-venue mid / spread / depth ----------------
        let obs = self.compute_agg_mid_spread(state, now_ms);
        let prev_fair_opt = state.fair_value;
        let prev_fair = prev_fair_opt.unwrap_or_else(|| {
            // Fallback: if we have a KF state, use that; otherwise we’ll
            // initialise from the first observation.
            if state.kf_x_hat != 0.0 {
                state.kf_x_hat.exp()
            } else {
                0.0
            }
        });

        // Time delta for the process noise (seconds).
        let dt_s = if state.kf_last_update_ms > 0 {
            let dt_ms = (now_ms - state.kf_last_update_ms).max(1);
            (dt_ms as f64) / 1000.0
        } else {
            1.0
        };

        // Initialise KF state if needed.
        let mut x_hat = if state.kf_x_hat != 0.0 {
            state.kf_x_hat
        } else if let Some((mid, _, _)) = obs {
            mid.max(1e-6).ln()
        } else if prev_fair > 0.0 {
            prev_fair.ln()
        } else {
            // Arbitrary but finite.
            (250.0_f64).ln()
        };

        let mut p = if state.kf_p > 0.0 {
            state.kf_p
        } else {
            k_cfg.p_init
        };

        // --- 2) Prediction step: random walk in log price -------------------
        let q = (k_cfg.q_base * dt_s).max(0.0);
        p += q;

        // --- 3) Optional measurement update --------------------------------
        let mut used_obs = false;

        let fair_new = if let Some((mid_obs, avg_spread, avg_depth)) = obs {
            // Outlier gating vs previous fair value.
            let gate_ref = if prev_fair > 0.0 { prev_fair } else { mid_obs };
            let rel_jump = ((mid_obs - gate_ref) / gate_ref).abs();

            if prev_fair_opt.is_none() || rel_jump <= book_cfg.max_mid_jump_pct {
                // Observation noise R(s, d) = clip(r_min + a * s^2 + b / d, r_min, r_max).
                let spread = avg_spread.max(1e-6);
                let depth = avg_depth.max(1.0);

                let mut r = k_cfg.r_min + k_cfg.r_a * spread * spread + k_cfg.r_b / depth;
                if !r.is_finite() || r <= 0.0 {
                    r = k_cfg.r_max;
                }
                r = r.clamp(k_cfg.r_min, k_cfg.r_max);

                let y = mid_obs.max(1e-6).ln();
                let s = p + r;
                let k_gain = if s > 0.0 { p / s } else { 0.0 };

                x_hat += k_gain * (y - x_hat);
                p *= 1.0 - k_gain;

                used_obs = true;
            }

            x_hat.exp()
        } else {
            // No usable observation this tick.
            x_hat.exp()
        };

        // Keep KF variance sane.
        if !p.is_finite() || p <= 0.0 {
            p = k_cfg.p_init;
        }

        // --- 4) Volatility update and control scalars -----------------------
        let fair_prev_for_ret = if prev_fair > 0.0 { prev_fair } else { fair_new };
        let ret = if fair_prev_for_ret > 0.0 {
            (fair_new / fair_prev_for_ret).ln()
        } else {
            0.0
        };

        self.update_vol_and_scalars(state, ret);

        // --- 5) Store KF and fair value back into state ---------------------
        state.fair_value_prev = fair_prev_for_ret;
        state.fair_value = Some(fair_new);
        state.kf_x_hat = x_hat;
        state.kf_p = p;
        state.kf_last_update_ms = now_ms;

        // We don't yet use avg_spread / avg_depth downstream, but we keep
        // them wired via `compute_agg_mid_spread` so they are easy to log.
        let _ = used_obs;
    }

    /// Single-scale variance EWMA -> sigma_eff + vol_ratio -> spread/size/band.
    fn update_vol_and_scalars(&self, state: &mut GlobalState, ret: f64) {
        let vol_cfg = &self.cfg.volatility;

        let r2 = ret * ret;

        // Interpret sigma_eff^2 as the current variance estimate.
        let var_prev = if state.sigma_eff > 0.0 {
            state.sigma_eff * state.sigma_eff
        } else {
            vol_cfg.sigma_min * vol_cfg.sigma_min
        };

        let alpha_var = vol_cfg.fv_vol_alpha_short;
        let var_new = (1.0 - alpha_var) * var_prev + alpha_var * r2;

        let mut sigma_eff = var_new.max(0.0).sqrt();
        if !sigma_eff.is_finite() || sigma_eff < vol_cfg.sigma_min {
            sigma_eff = vol_cfg.sigma_min;
        }
        state.sigma_eff = sigma_eff;

        // Map sigma_eff → vol_ratio_clipped.
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
    }

    /// Cross-venue liquidity-weighted mid + average spread + depth.
    ///
    /// Uses only non-disabled venues with non-stale books. If fewer than
    /// `book.min_healthy_for_kf` such venues are available, returns None.
    fn compute_agg_mid_spread(&self, state: &GlobalState, now_ms: i64) -> Option<(f64, f64, f64)> {
        let book_cfg = &self.cfg.book;

        let mut weight_sum: f64 = 0.0;
        let mut mid_num: f64 = 0.0;
        let mut spread_sum: f64 = 0.0;
        let mut depth_sum: f64 = 0.0;
        let mut used_count: u32 = 0;

        for (i, v) in state.venues.iter().enumerate() {
            // Skip disabled venues entirely.
            if matches!(v.status, VenueStatus::Disabled) {
                continue;
            }

            // Require a reasonably fresh book.
            if let Some(ts) = v.last_mid_update_ms {
                let age_ms = now_ms - ts;
                if age_ms > book_cfg.stale_ms {
                    continue;
                }
            } else {
                continue;
            }

            if let (Some(mid), Some(spread)) = (v.mid, v.spread) {
                let depth = v.depth_near_mid;
                if depth <= 0.0 {
                    continue;
                }

                // Simple depth-proportional weighting.
                let weight = depth;

                weight_sum += weight;
                mid_num += weight * mid;
                spread_sum += spread;
                depth_sum += depth;
                used_count += 1;

                // Silence "unused" warning if we ever remove fields.
                let _ = i;
            }
        }

        if weight_sum <= 0.0 {
            return None;
        }
        if used_count < book_cfg.min_healthy_for_kf {
            return None;
        }

        let mid = mid_num / weight_sum;
        let avg_spread = spread_sum / (used_count as f64).max(1.0);
        let avg_depth = depth_sum / (used_count as f64).max(1.0);

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
        state.daily_pnl_total = state.daily_realised_pnl + state.daily_unrealised_pnl;

        // ---- 3) Choose risk regime based on delta / basis / PnL -----------
        let delta_abs = state.dollar_delta_usd.abs();
        let basis_abs = state.basis_usd.abs();
        let pnl = state.daily_pnl_total;

        let delta_warn = risk_cfg.delta_warn_frac * state.delta_limit_usd;
        let basis_warn = risk_cfg.basis_warn_frac * risk_cfg.basis_hard_limit_usd;

        // Daily loss limit semantics:
        // - Config stores a *positive* USD number (e.g. 2000.0).
        // - Losses are negative PnL, so the actual hard limit is -daily_loss_limit.
        let loss_limit = -risk_cfg.daily_loss_limit.abs();
        let pnl_warn = -risk_cfg.daily_loss_limit.abs() * risk_cfg.pnl_warn_frac;

        let mut regime = RiskRegime::Normal;

        // Hard-limit regime if *any* hard constraint is breached.
        if delta_abs >= state.delta_limit_usd
            || basis_abs >= risk_cfg.basis_hard_limit_usd
            || pnl <= loss_limit
        {
            regime = RiskRegime::HardLimit;
        }
        // Warning regime if *any* warning threshold breached
        // (and hard limit not already hit).
        else if delta_abs >= delta_warn || basis_abs >= basis_warn || pnl <= pnl_warn {
            regime = RiskRegime::Warning;
        }

        // Kill-switch if we are in HardLimit and PnL has breached the hard
        // daily loss limit.
        let kill_switch = matches!(regime, RiskRegime::HardLimit) && pnl <= loss_limit;

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
