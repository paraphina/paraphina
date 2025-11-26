// src/engine.rs
//
// Core strategy engine scaffolding: main loop tick, Kalman fair value,
// volatility EWMAs, volatility-driven control scalars, inventory/basis
// recomputation, and risk-regime classification. All mapped to
// Sections 5, 6, 8 and 14 of the whitepaper.

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
use crate::types::{TimestampMs, VenueStatus};

pub struct Engine<'a> {
    cfg: &'a Config,
}

impl<'a> Engine<'a> {
    pub fn new(cfg: &'a Config) -> Self {
        Self { cfg }
    }

    /// TEMP helper: seed synthetic mids/spreads/depth into all venues so that
    /// the Kalman filter can initialise before we have real exchange data.
    pub fn seed_dummy_mids(&self, state: &mut GlobalState, now_ms: TimestampMs) {
        let base_price = 250.0;
        let spread = 1.0;
        let depth = 1_000.0;

        for (i, v) in state.venues.iter_mut().enumerate() {
            let mid = base_price + i as f64 * 0.5;
            v.mid = Some(mid);
            v.spread = Some(spread);
            v.depth_near_mid = depth;
            v.last_mid_update_ms = Some(now_ms);
        }
    }

    /// One "main loop" tick as per Section 16.1, currently restricted to:
    ///  - fair value & volatility updates (Sections 5 & 6),
    ///  - volatility scalars,
    ///  - inventory & basis recomputation (Section 8),
    ///  - risk regime & kill switch (Section 14).
    pub fn main_tick(&self, state: &mut GlobalState, now_ms: TimestampMs) {
        self.update_fair_value_and_vol(state, now_ms);
        self.update_volatility_scalars(state);

        // NEW: toxicity + venue health (Section 7).
        crate::toxicity::update_toxicity_and_health(self.cfg, state);

        self.recompute_inventory_and_basis(state);
        self.update_risk_regime(state);
        // Later we will add:
        //  - more detailed toxicity/venue gating,
        //  - quoting & order management,
        //  - hedging & exits.
    }

    // ----------------- Section 5 & 6: Kalman + vol -----------------

    fn update_fair_value_and_vol(&self, state: &mut GlobalState, now_ms: TimestampMs) {
        let book_cfg = &self.cfg.book;
        let kalman_cfg = &self.cfg.kalman;
        let vol_cfg = &self.cfg.volatility;

        // ----- Time update -----
        if let Some(last_ms) = state.kf_last_update_ms {
            let dt_ms = (now_ms - last_ms).max(0);
            let dt_sec = dt_ms as f64 / 1000.0;
            state.kf_p += kalman_cfg.q_base * dt_sec;
        }
        state.kf_last_update_ms = Some(now_ms);

        // ----- Initialisation: seed x_hat from venue mids if needed -----
        if state.kf_x_hat.is_none() {
            let mut mids: Vec<f64> = state
                .venues
                .iter()
                .filter(|v| v.status == VenueStatus::Healthy)
                .filter_map(|v| v.mid)
                .collect();

            if !mids.is_empty() {
                mids.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if mids.len() % 2 == 1 {
                    mids[mids.len() / 2]
                } else {
                    (mids[mids.len() / 2 - 1] + mids[mids.len() / 2]) * 0.5
                };

                state.kf_x_hat = Some(median.ln());
                state.kf_p = kalman_cfg.p_init;
                state.fair_value_prev = state.fair_value;
                state.fair_value = Some(median);
            } else {
                // No mids anywhere; cannot update KF yet.
                return;
            }
        }

        let mut x_hat = state.kf_x_hat.unwrap();
        let mut p = state.kf_p;

        // ----- Build eligible observations -----
        struct Obs {
            mid: f64,
            spread: f64,
            depth: f64,
        }

        let mut obs: Vec<Obs> = Vec::new();

        for v in state.venues.iter() {
            let mid = match v.mid {
                Some(m) => m,
                None => continue,
            };
            let spread = match v.spread {
                Some(s) if s > 0.0 => s,
                _ => continue,
            };
            if v.depth_near_mid <= 0.0 {
                continue;
            }
            if v.status != VenueStatus::Healthy {
                continue;
            }
            if let Some(last_mid_ms) = v.last_mid_update_ms {
                if now_ms - last_mid_ms > book_cfg.stale_ms as i64 {
                    continue;
                }
            }

            // Outlier vs previous fair value.
            if let Some(prev_fv) = state.fair_value {
                let dev = ((mid - prev_fv).abs()) / prev_fv.max(1e-9);
                if dev > book_cfg.max_mid_jump_pct {
                    continue;
                }
            }

            obs.push(Obs {
                mid,
                spread,
                depth: v.depth_near_mid,
            });
        }

        // If not enough healthy venues, skip observation updates.
        if obs.len() < book_cfg.min_healthy_for_kf as usize {
            // Keep the time-updated x_hat / P only.
        } else {
            // ----- Sequential scalar KF updates for each observation -----
            for ob in obs {
                let mut r =
                    kalman_cfg.r_a * ob.spread * ob.spread +
                    kalman_cfg.r_b / (ob.depth + 1e-9);

                if r < kalman_cfg.r_min {
                    r = kalman_cfg.r_min;
                } else if r > kalman_cfg.r_max {
                    r = kalman_cfg.r_max;
                }

                let s = p + r;
                let k = p / s;
                let y = ob.mid.ln();
                x_hat = x_hat + k * (y - x_hat);
                p = (1.0 - k) * p;
            }
        }

        state.kf_x_hat = Some(x_hat);
        state.kf_p = p;

        // ----- Map Kalman state to fair value -----
        let s_t = x_hat.exp();
        let prev_fv = state.fair_value;
        state.fair_value_prev = prev_fv;
        state.fair_value = Some(s_t);

        // ----- Volatility EWMA updates on log returns -----
        if let Some(prev) = prev_fv {
            if prev > 0.0 {
                let r_t = (s_t / prev).ln();

                let var_short = state.fv_short_vol * state.fv_short_vol;
                let var_long = state.fv_long_vol * state.fv_long_vol;

                let alpha_s = vol_cfg.fv_vol_alpha_short;
                let alpha_l = vol_cfg.fv_vol_alpha_long;

                let var_short_new =
                    (1.0 - alpha_s) * var_short + alpha_s * r_t * r_t;
                let var_long_new =
                    (1.0 - alpha_l) * var_long + alpha_l * r_t * r_t;

                state.fv_short_vol = var_short_new.max(0.0).sqrt();
                state.fv_long_vol = var_long_new.max(0.0).sqrt();
            }
        }

        // Effective sigma: sigma_eff = max(short_vol, SIGMA_MIN).
        let sigma_eff = state.fv_short_vol.max(vol_cfg.sigma_min);
        state.sigma_eff = sigma_eff;
    }

    // ----------------- Section 6: vol scalars -----------------

    fn update_volatility_scalars(&self, state: &mut GlobalState) {
        let vol_cfg = &self.cfg.volatility;
        let sigma_eff = state.sigma_eff;

        if vol_cfg.vol_ref <= 0.0 {
            state.vol_ratio_clipped = 1.0;
            state.spread_mult = 1.0;
            state.size_mult = 1.0;
            state.band_mult = 1.0;
            return;
        }

        let vol_ratio = sigma_eff / vol_cfg.vol_ref;
        let vol_ratio_clipped = vol_ratio
            .max(vol_cfg.vol_ratio_min)
            .min(vol_cfg.vol_ratio_max);

        let x = (vol_ratio_clipped - 1.0).max(0.0);
        let spread_mult = 1.0 + vol_cfg.spread_vol_mult_coeff * x;
        let size_mult = 1.0 / (1.0 + vol_cfg.size_vol_mult_coeff * x);
        let band_mult = 1.0 / (1.0 + vol_cfg.band_vol_mult_coeff * x);

        state.vol_ratio_clipped = vol_ratio_clipped;
        state.spread_mult = spread_mult;
        state.size_mult = size_mult;
        state.band_mult = band_mult;
    }

    // ----------------- Section 8: inventory & basis -----------------

    fn recompute_inventory_and_basis(&self, state: &mut GlobalState) {
        // Global inventory q_t = sum_v q_v.
        let mut q_t = 0.0;
        for v in &state.venues {
            q_t += v.position_tao;
        }
        state.q_global_tao = q_t;

        // If we don't have a fair value yet, zero the derived metrics.
        let Some(s_t) = state.fair_value else {
            state.dollar_delta_usd = 0.0;
            state.basis_usd = 0.0;
            state.basis_gross_usd = 0.0;
            return;
        };

        state.dollar_delta_usd = q_t * s_t;

        let mut basis = 0.0;
        let mut basis_gross = 0.0;

        for v in &state.venues {
            let q_v = v.position_tao;
            if let Some(mid) = v.mid {
                let b_v = mid - s_t;
                basis += q_v * b_v;
                basis_gross += q_v.abs() * b_v.abs();
            }
        }

        state.basis_usd = basis;
        state.basis_gross_usd = basis_gross;
    }

    // ----------------- Section 14: risk regime & kill switch -----------------

    fn update_risk_regime(&self, state: &mut GlobalState) {
        let risk_cfg = &self.cfg.risk;

        // Delta limit scaled by current volatility ratio.
        let vol_ratio = if state.vol_ratio_clipped > 0.0 {
            state.vol_ratio_clipped
        } else {
            1.0
        };
        let delta_limit =
            risk_cfg.delta_hard_limit_usd_base / vol_ratio.max(1e-9);

        // Basis Warning threshold.
        let basis_warn =
            risk_cfg.basis_warn_frac * risk_cfg.basis_hard_limit_usd;

        // Total daily PnL = realised + unrealised.
        let pnl_total =
            state.daily_realised_pnl + state.daily_unrealised_pnl;
        state.daily_pnl_total = pnl_total;

        let abs_delta = state.dollar_delta_usd.abs();
        let abs_basis = state.basis_usd.abs();

        // Decide regime.
        let mut regime = RiskRegime::Normal;
        let mut kill = false;

        // Critical conditions.
        if abs_delta > delta_limit
            || abs_basis > risk_cfg.basis_hard_limit_usd
            || pnl_total <= risk_cfg.daily_loss_limit
        {
            regime = RiskRegime::Critical;
            kill = true;
        } else {
            // Warning conditions.
            let delta_warn_level =
                risk_cfg.delta_warn_frac * delta_limit;
            let basis_warn_level = basis_warn;
            let pnl_warn_level =
                risk_cfg.pnl_warn_frac * risk_cfg.daily_loss_limit;

            if abs_delta > delta_warn_level
                || abs_basis > basis_warn_level
                || pnl_total < pnl_warn_level
            {
                regime = RiskRegime::Warning;
            }
        }

        state.risk_regime = regime;
        state.kill_switch = kill;

        state.delta_limit_usd = delta_limit;
        state.basis_limit_warn_usd = basis_warn;
        state.basis_limit_hard_usd = risk_cfg.basis_hard_limit_usd;
    }
}
