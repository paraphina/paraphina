// src/state.rs
//
// Global engine state + per-venue state for Paraphina.
//
// This file is intentionally "spec-shaped":
//  - Per-venue: book-derived mid/spread/depth + local vols + toxicity/health,
//    position + funding + (synthetic) margin/liquidation.
//  - Global: fair value + vol scalars, inventories + basis exposures, PnL,
//    risk regime + kill switch.
//
// Notes:
//  - We keep the current 3-state regime enum {Normal, Warning, HardLimit}.
//    In the whitepaper this maps to {Normal, Warning, Critical}, but in this
//    codebase HardLimit is the "circuit-breaker" state.
//  - Kill-switch is a separate boolean and is intended to be *latched*
//    by the Engine (once set, it stays set until manual reset).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::types::{Side, TimestampMs, VenueStatus};

/// A pending markout evaluation entry.
///
/// Created at fill time; evaluated at `t_eval_ms` to compute realized markout.
#[derive(Debug, Clone)]
pub struct PendingMarkout {
    /// Timestamp when the fill occurred.
    pub t_fill_ms: TimestampMs,
    /// Timestamp when we should evaluate the markout (t_fill_ms + horizon).
    pub t_eval_ms: TimestampMs,
    /// Side of the fill (Buy or Sell).
    pub side: Side,
    /// Size of the fill in TAO.
    pub size_tao: f64,
    /// Fill price in USD/TAO.
    pub price: f64,
    /// Fair value at fill time (for optional analysis).
    pub fair_at_fill: f64,
    /// Venue mid price at fill time (used for markout calculation).
    pub mid_at_fill: f64,
}

/// Per-venue state (one per perp venue / subaccount).
#[derive(Debug, Clone)]
pub struct VenueState {
    /// Stable identifier matching Config.venues[i].id
    pub id: String,

    // ----- Order book & local vols -----
    /// Current mid price from order book (if any).
    pub mid: Option<f64>,
    /// Current best spread from order book (if any).
    pub spread: Option<f64>,
    /// Depth near mid (sum within top-K levels).
    pub depth_near_mid: f64,
    /// Last time (ms) we updated the mid / spread.
    pub last_mid_update_ms: Option<TimestampMs>,
    /// Short-horizon EWMA local vol of log mid.
    pub local_vol_short: f64,
    /// Long-horizon EWMA local vol of log mid.
    pub local_vol_long: f64,

    // ----- Health / toxicity -----
    pub status: VenueStatus,
    /// Toxicity score ∈ [0,1] (EWMA of markout-based instantaneous toxicity).
    pub toxicity: f64,
    /// Pending markout evaluations (ring buffer, bounded by config).
    pub pending_markouts: VecDeque<PendingMarkout>,
    /// Running EWMA of markout in USD/TAO for telemetry/debugging.
    pub markout_ewma_usd_per_tao: f64,

    // ----- Position & funding -----
    /// Net TAO-equivalent perp position.
    pub position_tao: f64,
    /// Current 8h funding rate (dimensionless).
    pub funding_8h: f64,
    /// Volume-weighted average entry price of the current position (USD per TAO).
    /// Defined only when `position_tao != 0.0`, otherwise 0.0.
    pub avg_entry_price: f64,

    // ----- Margin & liquidation (synthetic for now) -----
    /// Total margin balance in USD (if known).
    pub margin_balance_usd: f64,
    /// Margin currently used in USD.
    pub margin_used_usd: f64,
    /// Margin available for new positions in USD.
    pub margin_available_usd: f64,
    /// Estimated liquidation price (if known).
    pub price_liq: Option<f64>,
    /// Distance to liquidation in sigma units (approx).
    pub dist_liq_sigma: f64,
}

/// High-level risk regime (whitepaper Section 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRegime {
    Normal,
    Warning,
    /// Hard limit / circuit-breaker regime (whitepaper "Critical").
    /// When this regime is active, kill_switch MUST be latched true.
    HardLimit,
}

/// Reason code for kill switch activation.
///
/// The kill switch can be triggered by multiple conditions. This enum
/// captures the primary reason for activation. Once latched, the reason
/// is preserved until manual reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KillReason {
    /// No kill triggered (default state).
    None,
    /// Daily PnL loss limit breached.
    PnlHardBreach,
    /// Volatility-scaled delta limit breached.
    DeltaHardBreach,
    /// Basis exposure hard limit breached.
    BasisHardBreach,
    /// Liquidation distance too close (below critical sigma threshold).
    LiquidationDistanceBreach,
}

/// Global engine state shared across strategy components.
#[derive(Debug, Clone)]
pub struct GlobalState {
    // ----- Per-venue -----
    pub venues: Vec<VenueState>,

    // ----- Fair value / Kalman filter backbone -----
    /// Last time the fair-value filter was updated.
    pub kf_last_update_ms: TimestampMs,
    /// 1D KF variance term P.
    pub kf_p: f64,
    /// Filtered log-price x_t = log(S_t).
    pub kf_x_hat: f64,
    /// Previous fair value S_{t-1}.
    pub fair_value_prev: f64,
    /// Current fair value S_t (if we have one).
    pub fair_value: Option<f64>,
    /// Short-horizon fair value vol (σ_short).
    pub fv_short_vol: f64,
    /// Long-horizon fair value vol (σ_long).
    pub fv_long_vol: f64,
    /// Effective volatility σ_eff = max(σ_short, σ_min).
    pub sigma_eff: f64,

    // ----- Milestone D: FV gating telemetry -----
    /// Whether fair value was successfully updated this tick.
    ///
    /// True if >= min_healthy_for_kf observations passed gating and the
    /// measurement update was applied. False if we only did a time update
    /// (prediction step) due to insufficient healthy venue data.
    pub fv_available: bool,
    /// Indices of venues whose observations were used in the last KF update.
    ///
    /// Empty if fv_available is false (time update only).
    pub healthy_venues_used: Vec<usize>,
    /// Count of healthy venues used in the last KF update (convenience field).
    pub healthy_venues_used_count: usize,

    // ----- Volatility-driven control scalars (whitepaper Section 6) -----
    /// Vol ratio clipped into [vol_ratio_min, vol_ratio_max].
    pub vol_ratio_clipped: f64,
    pub spread_mult: f64,
    pub size_mult: f64,
    pub band_mult: f64,

    // ----- Inventory & basis (whitepaper Section 8) -----
    /// Global net trading inventory q_t in TAO.
    pub q_global_tao: f64,
    /// Dollar delta q_t * S_t.
    pub dollar_delta_usd: f64,
    /// Net basis exposure B_t in USD (signed): sum_v q_v * (m_v - S_t).
    pub basis_usd: f64,
    /// Gross basis exposure in USD: sum_v |q_v| * |m_v - S_t|.
    pub basis_gross_usd: f64,

    // ----- PnL -----
    pub daily_realised_pnl: f64,
    pub daily_unrealised_pnl: f64,
    pub daily_pnl_total: f64,
    /// Optional PnL anchor fair value (kept for spec / future).
    pub pnl_ref_fair_value: Option<f64>,

    // ----- Risk regime & limits -----
    pub risk_regime: RiskRegime,
    /// Kill switch: once true, stays true until manual reset (latching).
    /// When kill_switch is true, risk_regime MUST be HardLimit.
    pub kill_switch: bool,
    /// Primary reason for kill switch activation (if any).
    /// Preserved once set until manual reset.
    pub kill_reason: KillReason,
    /// Volatility-scaled delta limit (updated in engine).
    pub delta_limit_usd: f64,
    /// Basis warning limit in USD (soft).
    pub basis_limit_warn_usd: f64,
    /// Basis hard limit in USD.
    pub basis_limit_hard_usd: f64,
}

impl GlobalState {
    /// Create a new state with per-venue scaffolding derived from config.
    pub fn new(cfg: &Config) -> Self {
        let mut venues = Vec::with_capacity(cfg.venues.len());

        for vcfg in &cfg.venues {
            venues.push(VenueState {
                id: vcfg.id.clone(),

                mid: None,
                spread: None,
                depth_near_mid: 0.0,
                last_mid_update_ms: None,
                local_vol_short: 0.0,
                local_vol_long: 0.0,

                status: VenueStatus::Healthy,
                toxicity: 0.0,
                pending_markouts: VecDeque::new(),
                markout_ewma_usd_per_tao: 0.0,

                position_tao: 0.0,
                funding_8h: 0.0,
                avg_entry_price: 0.0,

                // Synthetic defaults; real implementations poll these.
                margin_balance_usd: 0.0,
                margin_used_usd: 0.0,
                margin_available_usd: 10_000.0,
                price_liq: None,
                dist_liq_sigma: 10.0,
            });
        }

        GlobalState {
            venues,

            // Fair value filter
            kf_last_update_ms: 0,
            kf_p: cfg.kalman.p_init,
            kf_x_hat: 0.0,
            fair_value_prev: 0.0,
            fair_value: None,
            fv_short_vol: 0.0,
            fv_long_vol: 0.0,
            sigma_eff: cfg.volatility.sigma_min,

            // Milestone D: FV gating telemetry
            fv_available: false,
            healthy_venues_used: Vec::new(),
            healthy_venues_used_count: 0,

            // Vol control scalars
            vol_ratio_clipped: 1.0,
            spread_mult: 1.0,
            size_mult: 1.0,
            band_mult: 1.0,

            // Inventory & basis
            q_global_tao: 0.0,
            dollar_delta_usd: 0.0,
            basis_usd: 0.0,
            basis_gross_usd: 0.0,

            // PnL
            daily_realised_pnl: 0.0,
            daily_unrealised_pnl: 0.0,
            daily_pnl_total: 0.0,
            pnl_ref_fair_value: None,

            // Risk
            risk_regime: RiskRegime::Normal,
            kill_switch: false,
            kill_reason: KillReason::None,
            delta_limit_usd: cfg.risk.delta_hard_limit_usd_base,
            basis_limit_warn_usd: cfg.risk.basis_warn_frac * cfg.risk.basis_hard_limit_usd,
            basis_limit_hard_usd: cfg.risk.basis_hard_limit_usd,
        }
    }
}

// --- Perp-fill application + realised PnL accounting ------------------------
//
//  - maintain per-venue position q_v and VWAP entry price,
//  - on each fill, compute realised PnL for the portion that *closes*
//    existing inventory,
//  - subtract trading fees,
//  - accumulate into daily_realised_pnl; total PnL will be updated when
//    we recompute marks/unrealised.

impl GlobalState {
    /// Apply a single perp fill into the state.
    ///
    /// - `venue_index`: index into `self.venues`.
    /// - `side`: Buy or Sell.
    /// - `size_tao`: filled size in TAO (non-negative).
    /// - `price`: fill price in USD per TAO.
    /// - `fee_bps`: net fee in basis points (positive = cost, negative = rebate).
    pub fn apply_perp_fill(
        &mut self,
        venue_index: usize,
        side: Side,
        size_tao: f64,
        price: f64,
        fee_bps: f64,
    ) {
        if size_tao <= 0.0 || !price.is_finite() || price <= 0.0 {
            return;
        }

        let Some(v) = self.venues.get_mut(venue_index) else {
            return;
        };

        // Signed trade size: + for buy, - for sell.
        let trade = match side {
            Side::Buy => size_tao,
            Side::Sell => -size_tao,
        };

        let q_old = v.position_tao;
        let p_old = v.avg_entry_price;
        let p_trade = price;

        // Total fee in USD (positive reduces PnL; negative is a rebate).
        let fee = (fee_bps / 10_000.0) * p_trade * size_tao;

        let mut realised = 0.0_f64;

        if q_old == 0.0 {
            // Opening a fresh position.
            v.position_tao = trade;
            v.avg_entry_price = p_trade;
        } else {
            let same_dir = q_old.signum() == trade.signum();

            if same_dir {
                // Add to existing position, update VWAP entry price.
                let q_new = q_old + trade;
                if q_new != 0.0 {
                    let w_old = q_old.abs();
                    let w_trade = trade.abs();
                    v.avg_entry_price = (p_old * w_old + p_trade * w_trade) / (w_old + w_trade);
                    v.position_tao = q_new;
                } else {
                    // Exactly flat (rare).
                    v.position_tao = 0.0;
                    v.avg_entry_price = 0.0;
                }
            } else {
                // Closing or flipping some/all of the existing position.
                let close_qty = trade.abs().min(q_old.abs());

                if close_qty > 0.0 {
                    if q_old > 0.0 {
                        // Closing a long: sell closes.
                        realised += (p_trade - p_old) * close_qty;
                    } else {
                        // Closing a short: buy closes.
                        realised += (p_old - p_trade) * close_qty;
                    }
                }

                let q_new = q_old + trade;

                if q_old.abs() > trade.abs() {
                    // Partial close; keep original entry price.
                    v.position_tao = q_new;
                    v.avg_entry_price = p_old;
                } else if q_old.abs() < trade.abs() {
                    // Closed and flipped into a new position.
                    v.position_tao = q_new;
                    v.avg_entry_price = p_trade;
                } else {
                    // Exactly flat.
                    v.position_tao = 0.0;
                    v.avg_entry_price = 0.0;
                }
            }
        }

        // Fees always apply to the executed trade.
        realised -= fee;

        self.daily_realised_pnl += realised;
        // Unrealised is marked in `recompute_after_fills`.
        self.daily_pnl_total = self.daily_realised_pnl + self.daily_unrealised_pnl;
    }

    /// Recompute inventory, basis and mark-to-market PnL.
    ///
    /// Despite the historical name, this is safe and intended to be called:
    ///  - after a batch of fills, and
    ///  - on every main tick after fair value updates (whitepaper: mark-to-S_t).
    pub fn recompute_after_fills(&mut self, _cfg: &Config) {
        // Prefer current fair; else fall back to previous fair.
        let s_t = self.fair_value.unwrap_or(self.fair_value_prev);

        // If we still don't have a sane fair value, keep everything at 0.
        if !s_t.is_finite() || s_t <= 0.0 {
            self.q_global_tao = 0.0;
            self.dollar_delta_usd = 0.0;
            self.basis_usd = 0.0;
            self.basis_gross_usd = 0.0;
            self.daily_unrealised_pnl = 0.0;
            self.daily_pnl_total = self.daily_realised_pnl;
            return;
        }

        self.q_global_tao = 0.0;
        self.dollar_delta_usd = 0.0;
        self.basis_usd = 0.0;
        self.basis_gross_usd = 0.0;

        let mut unrealised = 0.0_f64;

        for v in &self.venues {
            let q = v.position_tao;

            self.q_global_tao += q;
            self.dollar_delta_usd += q * s_t;

            // Per-venue basis b_v = m_v - S_t (if no mid, assume m_v == S_t => b_v=0).
            let b_v = v.mid.unwrap_or(s_t) - s_t;

            // Net basis exposure: sum q_v * b_v
            self.basis_usd += q * b_v;

            // Gross basis exposure: sum |q_v| * |b_v|
            self.basis_gross_usd += q.abs() * b_v.abs();

            // Mark-to-fair unrealised PnL.
            if q != 0.0 {
                unrealised += q * (s_t - v.avg_entry_price);
            }
        }

        self.daily_unrealised_pnl = unrealised;
        self.daily_pnl_total = self.daily_realised_pnl + self.daily_unrealised_pnl;
    }

    /// Record a pending markout evaluation for a fill.
    ///
    /// Called immediately after a fill is applied. The markout will be
    /// evaluated at `now_ms + horizon_ms` in `update_toxicity_and_health`.
    pub fn record_pending_markout(&mut self, record: PendingMarkoutRecord) {
        if record.size_tao <= 0.0 || !record.price.is_finite() || record.price <= 0.0 {
            return;
        }

        let Some(v) = self.venues.get_mut(record.venue_index) else {
            return;
        };

        let entry = PendingMarkout {
            t_fill_ms: record.now_ms,
            t_eval_ms: record.now_ms + record.horizon_ms,
            side: record.side,
            size_tao: record.size_tao,
            price: record.price,
            fair_at_fill: record.fair,
            mid_at_fill: record.mid,
        };

        v.pending_markouts.push_back(entry);

        // Enforce bounded queue: drop oldest entries if over limit.
        while v.pending_markouts.len() > record.max_pending {
            v.pending_markouts.pop_front();
        }
    }
}

/// Parameters for recording a pending markout evaluation.
///
/// Bundles all parameters needed by `record_pending_markout` to avoid
/// clippy::too_many_arguments.
pub struct PendingMarkoutRecord {
    /// Index into `GlobalState.venues`.
    pub venue_index: usize,
    /// Buy or Sell.
    pub side: Side,
    /// Filled size in TAO.
    pub size_tao: f64,
    /// Fill price in USD/TAO.
    pub price: f64,
    /// Current timestamp.
    pub now_ms: TimestampMs,
    /// Current fair value (for optional analysis).
    pub fair: f64,
    /// Current venue mid price.
    pub mid: f64,
    /// Time until markout evaluation.
    pub horizon_ms: i64,
    /// Maximum queue size (older entries dropped).
    pub max_pending: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::Side;

    fn approx(label: &str, expected: f64, actual: f64) {
        let diff = (expected - actual).abs();
        assert!(
            diff < 1e-6,
            "{}: left={} right={} diff={}",
            label,
            expected,
            actual,
            diff
        );
    }

    #[test]
    fn basis_and_unrealised_pnl_two_venues() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        // Set fair value.
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;

        // Venue 0: long 5 @ 95, mid 102.
        {
            let v0 = &mut state.venues[0];
            v0.position_tao = 5.0;
            v0.avg_entry_price = 95.0;
            v0.mid = Some(102.0);
        }

        // Venue 1: short 3 @ 110, mid 98.
        {
            let v1 = &mut state.venues[1];
            v1.position_tao = -3.0;
            v1.avg_entry_price = 110.0;
            v1.mid = Some(98.0);
        }

        state.recompute_after_fills(&cfg);

        approx("q_global_tao", 2.0, state.q_global_tao);
        approx("dollar_delta_usd", 200.0, state.dollar_delta_usd);

        // Basis contributions:
        // v0: 5 * (102 - 100) = 10
        // v1: -3 * (98 - 100) =  6
        approx("basis_usd", 16.0, state.basis_usd);
        approx("basis_gross_usd", 16.0, state.basis_gross_usd);

        // Unrealised PnL is marked vs fair value:
        // v0: 5 * (100 - 95) = 25
        // v1: -3 * (100 - 110) = 30
        approx("daily_unrealised_pnl", 55.0, state.daily_unrealised_pnl);
        approx("daily_pnl_total", 55.0, state.daily_pnl_total);
    }

    #[test]
    fn long_open_and_close_no_fees() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        let venue_index = 0;
        let size = 10.0;
        let entry_price = 100.0;
        let exit_price = 110.0;

        // Open long.
        state.apply_perp_fill(venue_index, Side::Buy, size, entry_price, 0.0);
        // Close long.
        state.apply_perp_fill(venue_index, Side::Sell, size, exit_price, 0.0);

        // Mark at exit price.
        state.fair_value = Some(exit_price);
        state.fair_value_prev = exit_price;
        state.recompute_after_fills(&cfg);

        approx(
            "realised PnL long round-trip no fees",
            (exit_price - entry_price) * size,
            state.daily_realised_pnl,
        );
        approx("unrealised PnL flat book", 0.0, state.daily_unrealised_pnl);
        approx("q_global_tao flat book", 0.0, state.q_global_tao);
    }

    #[test]
    fn short_open_and_close_with_fees() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        let venue_index = 0;
        let size = 10.0;
        let entry_price = 110.0;
        let exit_price = 100.0;
        let fee_bps = 5.0;

        // Open short: sell first.
        state.apply_perp_fill(venue_index, Side::Sell, size, entry_price, fee_bps);
        // Close short: buy back.
        state.apply_perp_fill(venue_index, Side::Buy, size, exit_price, fee_bps);

        // Mark at exit price (flat book).
        state.fair_value = Some(exit_price);
        state.fair_value_prev = exit_price;
        state.recompute_after_fills(&cfg);

        let fee_entry = (fee_bps / 10_000.0) * entry_price * size;
        let fee_exit = (fee_bps / 10_000.0) * exit_price * size;
        let expected_realised = (entry_price - exit_price) * size - fee_entry - fee_exit;

        approx(
            "realised PnL short round-trip with fees",
            expected_realised,
            state.daily_realised_pnl,
        );
        approx("unrealised PnL flat book", 0.0, state.daily_unrealised_pnl);
        approx("q_global_tao flat book", 0.0, state.q_global_tao);
    }
}
