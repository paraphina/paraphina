// src/engine.rs
//
// Paraphina MM engine:
//
//  - seeds synthetic per-venue mids/spreads/depth and local vols,
//  - runs a robust 1D log-price Kalman filter over *multiple venue observations*,
//    with spread/depth-based observation noise + health/staleness/outlier gating,
//  - maintains short/long FV volatility and derives σ_eff,
//  - updates volatility-driven control scalars,
//  - runs toxicity + venue health,
//  - marks positions to fair (updates delta/basis/unrealised),
//  - updates risk limits + risk regime + latching kill switch.

use crate::config::Config;
use crate::state::{GlobalState, KillReason, RiskRegime};
use crate::toxicity::update_toxicity_and_health;
use crate::types::VenueStatus;

pub struct Engine<'a> {
    cfg: &'a Config,
}

impl<'a> Engine<'a> {
    pub fn new(cfg: &'a Config) -> Self {
        Self { cfg }
    }

    /// Seed synthetic mids / spreads / depth for all venues and update
    /// per-venue local volatility estimates.
    ///
    /// Deterministic function of `now_ms` and venue index. This is for
    /// exercising the control loops / tests, not realism.
    ///
    /// # Optimization Notes (Opt7)
    ///
    /// This function is already highly optimized for the hot path:
    ///
    /// - **Zero allocations**: All operations mutate VenueState fields in-place.
    ///   No Vec, String, Arc, or other heap-allocating operations occur.
    /// - **Precomputed decay factors**: EWMA alphas and (1-alpha) values are
    ///   computed once before the venue loop, avoiding per-venue config lookups.
    /// - **Explicit indexed loop**: Uses `for idx in 0..len` pattern to ensure
    ///   deterministic venue ordering (no iterator-based nondeterminism).
    /// - **No scratch buffers needed**: Unlike `main_tick` helpers, this function
    ///   doesn't collect intermediate data—it directly mutates venue state.
    /// - **Early exit not applicable**: Every tick requires updating all venue
    ///   mids/spreads, so no early-exit optimization is possible.
    ///
    /// # Determinism Guarantees
    ///
    /// Given identical `now_ms` and initial state, this function produces
    /// bit-exact identical results. The floating-point computation order is
    /// fixed (venue-by-venue, short-vol then long-vol) to ensure reproducibility.
    pub fn seed_dummy_mids(&self, state: &mut GlobalState, now_ms: i64) {
        let base = 250.0 + (now_ms as f64) * 0.001;

        // Precompute EWMA decay factors outside the loop to avoid per-venue
        // repeated config access and (1.0 - alpha) computation.
        let alpha_short = self.cfg.volatility.fv_vol_alpha_short;
        let alpha_long = self.cfg.volatility.fv_vol_alpha_long;
        let one_minus_alpha_short = 1.0 - alpha_short;
        let one_minus_alpha_long = 1.0 - alpha_long;

        // Explicit indexed loop for deterministic venue ordering.
        for idx in 0..state.venues.len() {
            let v = &mut state.venues[idx];

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
            // Only update if we have a valid previous mid for computing returns.
            if let Some(prev) = mid_prev {
                if prev > 0.0 && mid > 0.0 {
                    let r = (mid / prev).ln();
                    let r2 = r * r;

                    let var_short_prev = v.local_vol_short * v.local_vol_short;
                    let var_long_prev = v.local_vol_long * v.local_vol_long;

                    let var_short_new = one_minus_alpha_short * var_short_prev + alpha_short * r2;
                    let var_long_new = one_minus_alpha_long * var_long_prev + alpha_long * r2;

                    v.local_vol_short = var_short_new.max(0.0).sqrt();
                    v.local_vol_long = var_long_new.max(0.0).sqrt();
                }
            }
        }
    }

    /// Main per-tick engine step.
    ///
    /// Whitepaper mapping:
    ///   5) fair value / filtering (Kalman over log price)
    ///   6) volatility + control scalars
    ///   7) toxicity / venue health
    ///   8) inventory/basis
    ///  14) risk limits + risk regime / kill switch
    pub fn main_tick(&self, state: &mut GlobalState, now_ms: i64) {
        // 1) Fair value + volatility + control scalars
        self.update_fair_value_and_vol(state, now_ms);

        // 2) Mark-to-fair inventory / basis / unrealised PnL (spec-consistent)
        state.recompute_after_fills(self.cfg);

        // 3) Toxicity + venue health (now processes pending markouts)
        update_toxicity_and_health(state, self.cfg, now_ms);

        // 4) Risk limits + regime / kill switch (latched)
        self.update_risk_limits_and_regime(state);
    }

    // ---------------------------------------------------------------------
    // Fair value: robust multi-venue KF (sequential observation updates)
    // ---------------------------------------------------------------------

    fn update_fair_value_and_vol(&self, state: &mut GlobalState, now_ms: i64) {
        let book_cfg = &self.cfg.book;
        let k_cfg = &self.cfg.kalman;

        // Prior fair reference for gating and return computation.
        let prev_fair_opt = state.fair_value;
        let prev_fair = prev_fair_opt.unwrap_or(state.fair_value_prev);

        // Time delta (seconds) for process noise.
        let dt_ms: i64 = if state.kf_last_update_ms > 0 {
            (now_ms - state.kf_last_update_ms).max(1)
        } else {
            1000
        };
        let dt_s = (dt_ms as f64) / 1000.0;

        // If KF is uninitialised, initialise from median of eligible mids
        // (or fall back to prev_fair or a constant).
        let kf_uninit = state.kf_last_update_ms == 0 || state.kf_x_hat == 0.0;

        // Compute median mid using scratch buffer (no allocation).
        let init_mid_median = self.median_mid_from_books(state, now_ms);

        let mut x_hat = if !kf_uninit {
            state.kf_x_hat
        } else if let Some(med) = init_mid_median {
            med.max(1e-6).ln()
        } else if prev_fair > 0.0 {
            prev_fair.max(1e-6).ln()
        } else {
            250.0_f64.ln()
        };

        let mut p = if !kf_uninit && state.kf_p.is_finite() && state.kf_p > 0.0 {
            state.kf_p
        } else {
            k_cfg.p_init
        };

        // --- Prediction step (random walk in log price)
        let q_proc = (k_cfg.q_base * dt_s).max(0.0);
        p += q_proc;

        // --- Measurement update (sequential over eligible venues)
        // Gate around the prior fair value if we have one.
        let gate_ref = if prev_fair.is_finite() && prev_fair > 0.0 {
            Some(prev_fair)
        } else {
            None
        };

        // Collect observations into scratch buffer (no allocation).
        self.collect_kf_observations(state, now_ms, gate_ref);

        // Milestone D: min-healthy gating - require >= min_healthy_for_kf observations
        // to apply a measurement update; otherwise skip measurement update (time update only).
        let min_healthy = book_cfg.min_healthy_for_kf as usize;
        let obs_len = state.scratch_kf_obs.len();
        let fv_available = obs_len >= min_healthy;

        // Track which venues were used for telemetry.
        // Reuse healthy_venues_used Vec in state (clear and repopulate).
        state.healthy_venues_used.clear();
        if fv_available {
            for (idx, _, _) in &state.scratch_kf_obs {
                state.healthy_venues_used.push(*idx);
            }
        }

        if fv_available {
            // Explicit indexed loop over scratch buffer for KF update.
            for i in 0..state.scratch_kf_obs.len() {
                let (_, y, r) = state.scratch_kf_obs[i];
                // Standard scalar KF update:
                //   S = P + R
                //   K = P / S
                //   x = x + K (y - x)
                //   P = (1 - K) P
                let s = p + r;
                if s.is_finite() && s > 0.0 {
                    let k_gain = p / s;
                    x_hat += k_gain * (y - x_hat);
                    p *= 1.0 - k_gain;
                }
            }
        }
        // else: time update only - FV unchanged, fv_available = false

        // Keep KF variance sane.
        if !p.is_finite() || p <= 0.0 {
            p = k_cfg.p_init;
        }

        // Current fair value.
        let mut fair_new = x_hat.exp();
        if !fair_new.is_finite() || fair_new <= 0.0 {
            // Fallback to previous fair or a constant.
            fair_new = if prev_fair.is_finite() && prev_fair > 0.0 {
                prev_fair
            } else {
                250.0
            };
            x_hat = fair_new.max(1e-6).ln();
        }

        // Compute return r_t = log(S_t / S_{t-1}) for vol updates.
        let fair_prev_for_ret = if prev_fair.is_finite() && prev_fair > 0.0 {
            prev_fair
        } else {
            fair_new
        };
        let ret = if fair_prev_for_ret > 0.0 && fair_new > 0.0 {
            (fair_new / fair_prev_for_ret).ln()
        } else {
            0.0
        };

        // Update vol EWMAs + scalars from σ_eff.
        self.update_vol_and_scalars(state, ret);

        // Store back into state.
        state.fair_value_prev = fair_prev_for_ret;
        state.fair_value = Some(fair_new);
        state.kf_x_hat = x_hat;
        state.kf_p = p;
        state.kf_last_update_ms = now_ms;

        // Milestone D: FV gating telemetry fields.
        state.fv_available = fv_available;
        state.healthy_venues_used_count = state.healthy_venues_used.len();
    }

    /// Compute median mid across venues with fresh books (no outlier gating).
    ///
    /// Optimization notes:
    /// - Reuses `state.scratch_mids` buffer to avoid per-tick heap allocation.
    /// - Buffer is cleared and repopulated each call; capacity is preserved.
    /// - Explicit indexed loop maintains deterministic venue ordering.
    fn median_mid_from_books(&self, state: &mut GlobalState, now_ms: i64) -> Option<f64> {
        let book_cfg = &self.cfg.book;
        let stale_ms = book_cfg.stale_ms;

        // Reuse scratch buffer: clear but preserve capacity.
        state.scratch_mids.clear();

        // Explicit indexed loop for deterministic ordering.
        for idx in 0..state.venues.len() {
            let v = &state.venues[idx];

            if matches!(v.status, VenueStatus::Disabled) {
                continue;
            }

            let Some(ts) = v.last_mid_update_ms else {
                continue;
            };
            if now_ms - ts > stale_ms {
                continue;
            }

            let Some(mid) = v.mid else {
                continue;
            };
            if !mid.is_finite() || mid <= 0.0 {
                continue;
            }

            state.scratch_mids.push(mid);
        }

        if state.scratch_mids.is_empty() {
            return None;
        }

        state
            .scratch_mids
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = state.scratch_mids.len();
        if n % 2 == 1 {
            Some(state.scratch_mids[n / 2])
        } else {
            Some(0.5 * (state.scratch_mids[n / 2 - 1] + state.scratch_mids[n / 2]))
        }
    }

    /// Collect eligible KF observations as (venue_index, y=log(mid), r=variance).
    ///
    /// Eligibility follows the spec (Milestone D gating policy):
    ///  - venue not disabled,
    ///  - status is Healthy (not Warning) for KF contributions,
    ///  - non-stale book (staleness gating),
    ///  - valid mid/spread/depth,
    ///  - optional outlier gating vs gate_ref (outlier gating).
    ///
    /// Returns a slice reference to the scratch buffer containing observations.
    /// The buffer is cleared and repopulated each call.
    ///
    /// Optimization notes:
    /// - Reuses `state.scratch_kf_obs` buffer to avoid per-tick heap allocation.
    /// - Config values are cached in locals to avoid repeated struct access.
    /// - Explicit indexed loop maintains deterministic venue ordering.
    fn collect_kf_observations(&self, state: &mut GlobalState, now_ms: i64, gate_ref: Option<f64>) {
        let book_cfg = &self.cfg.book;
        let k_cfg = &self.cfg.kalman;

        // Cache config values to avoid repeated struct access in loop.
        let stale_ms = book_cfg.stale_ms;
        let max_mid_jump_pct = book_cfg.max_mid_jump_pct;
        let r_min = k_cfg.r_min;
        let r_max = k_cfg.r_max;
        let r_a = k_cfg.r_a;
        let r_b = k_cfg.r_b;

        // Reuse scratch buffer: clear but preserve capacity.
        state.scratch_kf_obs.clear();

        // Explicit indexed loop for deterministic ordering.
        for venue_idx in 0..state.venues.len() {
            let v = &state.venues[venue_idx];

            // --- Disabled venue gating ---
            if matches!(v.status, VenueStatus::Disabled) {
                continue;
            }
            // Spec: use Healthy venues for FV updates.
            if !matches!(v.status, VenueStatus::Healthy) {
                continue;
            }

            // --- Staleness gating (Milestone D) ---
            let Some(ts) = v.last_mid_update_ms else {
                continue;
            };
            if now_ms - ts > stale_ms {
                continue;
            }

            let (Some(mid), Some(spread)) = (v.mid, v.spread) else {
                continue;
            };

            let depth = v.depth_near_mid;

            if !mid.is_finite() || mid <= 0.0 {
                continue;
            }
            if !spread.is_finite() || spread <= 0.0 {
                continue;
            }
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }

            // --- Outlier gating (Milestone D) ---
            // Reject venues whose mid deviates too far from the reference fair value.
            if let Some(ref_price) = gate_ref {
                if ref_price.is_finite() && ref_price > 0.0 {
                    let dev = ((mid - ref_price) / ref_price).abs();
                    if dev > max_mid_jump_pct {
                        continue;
                    }
                }
            }

            // Observation noise model:
            // We observe y = log(mid). Use spread scaled to a relative measure.
            let spread_rel = (spread / mid).max(1e-9);
            let mut r = r_min + r_a * spread_rel * spread_rel + r_b / (depth + 1e-9);

            if !r.is_finite() || r <= 0.0 {
                r = r_max;
            }
            r = r.clamp(r_min, r_max);

            let y = mid.max(1e-6).ln();
            state.scratch_kf_obs.push((venue_idx, y, r));
        }
    }

    // ---------------------------------------------------------------------
    // Volatility: maintain short/long EWMAs and derive σ_eff + scalars
    // ---------------------------------------------------------------------

    fn update_vol_and_scalars(&self, state: &mut GlobalState, ret: f64) {
        let vol_cfg = &self.cfg.volatility;
        let r2 = ret * ret;

        // Short vol EWMA
        let var_short_prev = state.fv_short_vol * state.fv_short_vol;
        let var_short_new =
            (1.0 - vol_cfg.fv_vol_alpha_short) * var_short_prev + vol_cfg.fv_vol_alpha_short * r2;
        let mut sigma_short = var_short_new.max(0.0).sqrt();
        if !sigma_short.is_finite() {
            sigma_short = vol_cfg.sigma_min;
        }

        // Long vol EWMA
        let var_long_prev = state.fv_long_vol * state.fv_long_vol;
        let var_long_new =
            (1.0 - vol_cfg.fv_vol_alpha_long) * var_long_prev + vol_cfg.fv_vol_alpha_long * r2;
        let mut sigma_long = var_long_new.max(0.0).sqrt();
        if !sigma_long.is_finite() {
            sigma_long = vol_cfg.sigma_min;
        }

        state.fv_short_vol = sigma_short;
        state.fv_long_vol = sigma_long;

        // Effective vol floor.
        let mut sigma_eff = sigma_short.max(vol_cfg.sigma_min);
        if !sigma_eff.is_finite() || sigma_eff <= 0.0 {
            sigma_eff = vol_cfg.sigma_min;
        }
        state.sigma_eff = sigma_eff;

        // vol_ratio and clipped.
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

        // spread_mult grows with vol; size_mult and band_mult shrink.
        state.spread_mult = 1.0 + vol_cfg.spread_vol_mult_coeff * bump;
        state.size_mult = 1.0 / (1.0 + vol_cfg.size_vol_mult_coeff * bump);
        state.band_mult = 1.0 / (1.0 + vol_cfg.band_vol_mult_coeff * bump);
    }

    // ---------------------------------------------------------------------
    // Risk: vol-scaled limits, discrete regime, latched kill switch
    // ---------------------------------------------------------------------

    /// Vol-scaled risk limits + discrete risk regime + *latched* kill switch.
    ///
    /// Milestone C implementation:
    /// - Centralizes all risk regime determination in one place.
    /// - Critical/HardLimit regime implies kill_switch = true (latching).
    /// - Kill switch triggers on ANY hard breach: PnL, delta, basis, or liquidation distance.
    /// - Once kill_switch is true, it stays true until manual reset.
    pub fn update_risk_limits_and_regime(&self, state: &mut GlobalState) {
        let risk_cfg = &self.cfg.risk;

        // ---- 1) Vol-scaled delta limit + static basis limits ---------------
        let vol_ratio = state.vol_ratio_clipped.max(1e-6);
        state.delta_limit_usd = risk_cfg.delta_hard_limit_usd_base / vol_ratio;

        state.basis_limit_hard_usd = risk_cfg.basis_hard_limit_usd;
        state.basis_limit_warn_usd = risk_cfg.basis_hard_limit_usd * risk_cfg.basis_warn_frac;

        // ---- 2) Aggregate daily PnL fields --------------------------------
        state.daily_pnl_total = state.daily_realised_pnl + state.daily_unrealised_pnl;

        // ---- 3) Compute minimum liquidation distance across all venues -----
        let mut min_liq_dist_sigma = f64::INFINITY;
        for v in &state.venues {
            if v.dist_liq_sigma.is_finite() && v.dist_liq_sigma >= 0.0 {
                min_liq_dist_sigma = min_liq_dist_sigma.min(v.dist_liq_sigma);
            }
        }

        // ---- 4) Determine hard breach conditions ---------------------------
        let delta_abs = state.dollar_delta_usd.abs();
        let basis_abs = state.basis_usd.abs();
        let pnl = state.daily_pnl_total;

        let delta_warn = risk_cfg.delta_warn_frac * state.delta_limit_usd;
        let basis_warn = state.basis_limit_warn_usd;

        // Daily loss limit semantics:
        // - Config stores a *positive* USD number (e.g. 2000.0).
        // - Losses are negative PnL, so the actual hard limit is -daily_loss_limit.
        let loss_limit = -risk_cfg.daily_loss_limit.abs();
        let pnl_warn = loss_limit * risk_cfg.pnl_warn_frac; // e.g. -2000 * 0.5 = -1000

        // Hard breach conditions (any triggers HardLimit/Critical regime)
        let pnl_hard_breach = pnl <= loss_limit;
        let delta_hard_breach = delta_abs >= state.delta_limit_usd;
        let basis_hard_breach = basis_abs >= state.basis_limit_hard_usd;
        let liq_hard_breach =
            min_liq_dist_sigma.is_finite() && min_liq_dist_sigma <= risk_cfg.liq_crit_sigma;

        // ---- 5) Choose risk regime based on breach conditions --------------
        let mut regime = RiskRegime::Normal;

        // Hard-limit (Critical) regime if *any* hard constraint is breached.
        if pnl_hard_breach || delta_hard_breach || basis_hard_breach || liq_hard_breach {
            regime = RiskRegime::HardLimit;
        }
        // Warning regime if *any* warning threshold breached.
        else if delta_abs >= delta_warn
            || basis_abs >= basis_warn
            || pnl <= pnl_warn
            || (min_liq_dist_sigma.is_finite() && min_liq_dist_sigma <= risk_cfg.liq_warn_sigma)
        {
            regime = RiskRegime::Warning;
        }

        // ---- 6) Kill switch trigger + latch -------------------------------
        //
        // Milestone C: kill switch triggers on ANY hard breach, not just PnL.
        // This is the canonical "Critical implies kill_switch" behavior.
        //
        // Once triggered, kill_switch is LATCHED (stays true until manual reset).
        // We also preserve the first reason for kill activation.

        // Determine new kill trigger and reason (only if not already killed)
        if !state.kill_switch {
            // Check in priority order: PnL > Delta > Basis > Liquidation
            if pnl_hard_breach {
                state.kill_switch = true;
                state.kill_reason = KillReason::PnlHardBreach;
            } else if delta_hard_breach {
                state.kill_switch = true;
                state.kill_reason = KillReason::DeltaHardBreach;
            } else if basis_hard_breach {
                state.kill_switch = true;
                state.kill_reason = KillReason::BasisHardBreach;
            } else if liq_hard_breach {
                state.kill_switch = true;
                state.kill_reason = KillReason::LiquidationDistanceBreach;
            }
        }

        // If kill is latched, force HardLimit regime for clarity/telemetry.
        state.risk_regime = if state.kill_switch {
            RiskRegime::HardLimit
        } else {
            regime
        };
    }

    /// Delegate post-fill recomputations (inventory, basis, PnL, etc.)
    /// to GlobalState.
    pub fn recompute_after_fills(&self, state: &mut GlobalState) {
        state.recompute_after_fills(self.cfg);
    }
}
