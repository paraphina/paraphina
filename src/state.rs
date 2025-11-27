// src/state.rs
//
// Canonical engine state for Paraphina, aligned with whitepaper Sections 3, 4.3, 8 & 14.
//
// - Per-venue state: orderbook-derived fields, inventory, perp-style PnL buckets,
//   funding, margin & liquidation distance, toxicity & health.
// - Global state: Kalman fair value, volatility, control scalars,
//   global inventory, basis exposure, PnL and risk regime.
// - Helper methods: perp fill + funding payment application.

use crate::config::Config;
use crate::types::{Side, TimestampMs, VenueStatus};

/// High-level risk regime (Section 14.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRegime {
    Normal,
    Warning,
    Critical,
}

/// Per-venue strategy state.
/// This is a simplified version of the whitepaper's per-venue state with
/// enough fields to support the current engine / MM / hedge scaffolds.
#[derive(Debug, Clone)]
pub struct VenueState {
    /// Stable identifier (e.g. "extended").
    pub id: String,
    /// Human-readable venue name.
    pub name: String,

    /// Health status used by engine for selection / gating.
    pub status: VenueStatus,
    /// Toxicity score in [0,1] (placeholder for now).
    pub toxicity: f64,

    // -------- Order book / derived fields (Section 3) --------

    /// Current mid price m_{v,t}, if known.
    pub mid: Option<f64>,
    /// Current best-ask - best-bid spread, if known.
    pub spread: Option<f64>,
    /// Total size within some band around mid (depth_near_mid_v).
    pub depth_near_mid: f64,
    /// Timestamp of last mid / book update.
    pub last_mid_update_ms: Option<TimestampMs>,
    /// Short-horizon EWMA vol of log mid.
    pub local_vol_short: f64,
    /// Long-horizon EWMA vol of log mid.
    pub local_vol_long: f64,

    // -------- Positions & PnL --------

    /// Net TAO-equivalent perp position q_v (TAO).
    pub position_tao: f64,
    /// VWAP of current open position (for unrealised PnL vs fair value).
    pub vwap: f64,

    /// Realised trade PnL on this venue (excluding funding & fees).
    pub pnl_realised: f64,
    /// Unrealised PnL marked vs global fair value S_t.
    pub pnl_unrealised: f64,
    /// Cumulative funding PnL on this venue.
    pub pnl_funding: f64,
    /// Cumulative fee PnL on this venue (negative = paid fees).
    pub pnl_fees: f64,

    // -------- Margin & liquidation (Section 3 / 4.2) --------

    pub margin_balance: f64,
    pub margin_used: f64,
    pub margin_available: f64,
    /// Estimated liquidation price for current position (if any).
    pub price_liq: f64,
    /// Estimated sigma-distance to liquidation (dist_liq_sigma_v).
    pub dist_liq_sigma: f64,

    // -------- Funding (Section 2: funding_{8h,v,t}) --------

    /// Current 8h funding rate at this venue (dimensionless).
    pub funding_8h: f64,
}

impl VenueState {
    /// Apply a single perp fill on this venue, updating:
    ///  - position_tao,
    ///  - VWAP,
    ///  - realised trade PnL,
    ///  - fee PnL.
    ///
    /// All PnL values are in quote units (USD).
    pub fn apply_perp_fill(
        &mut self,
        side: Side,
        size_tao: f64,
        price: f64,
        fee_bps: f64,
    ) {
        if size_tao <= 0.0 {
            return;
        }

        let signed_size = match side {
            Side::Buy => size_tao,
            Side::Sell => -size_tao,
        };

        let q0 = self.position_tao;
        let p0 = self.vwap;
        let q_trade = signed_size;
        let p_trade = price;

        let mut pnl_realised = 0.0;

        if q0 == 0.0 {
            // Opening a fresh position.
            self.position_tao = q_trade;
            self.vwap = p_trade;
        } else if q0.signum() == q_trade.signum() {
            // Adding to an existing position in the same direction.
            let q_new = q0 + q_trade;
            if q_new != 0.0 {
                let notional_old = q0 * p0;
                let notional_new = q_trade * p_trade;
                self.vwap = (notional_old + notional_new) / q_new;
            }
            self.position_tao = q_new;
        } else {
            // Trade is reducing or flipping the position.
            let abs_q0 = q0.abs();
            let abs_trade = q_trade.abs();
            let sign_q0 = q0.signum();

            if abs_trade < abs_q0 {
                // Partial close: remain same direction.
                let closing_size = abs_trade;
                pnl_realised += closing_size * (p_trade - p0) * sign_q0;
                self.position_tao = q0 + q_trade; // same sign as q0
                // VWAP remains p0 (remaining part is old inventory).
            } else if abs_trade == abs_q0 {
                // Full close: flat after this.
                let closing_size = abs_q0;
                pnl_realised += closing_size * (p_trade - p0) * sign_q0;
                self.position_tao = 0.0;
                self.vwap = 0.0;
            } else {
                // Flip: fully close old, open new in opposite direction.
                let closing_size = abs_q0;
                pnl_realised += closing_size * (p_trade - p0) * sign_q0;

                let q_new = q0 + q_trade; // sign of q_trade
                self.position_tao = q_new;
                self.vwap = p_trade; // treat leftover as new position at trade price.
            }
        }

        self.pnl_realised += pnl_realised;

        // Fees: fee_bps is positive for fees. We store fee PnL as negative.
        if fee_bps != 0.0 {
            let notional_quote = price * size_tao.abs();
            let fee_quote = fee_bps / 10_000.0 * notional_quote;
            self.pnl_fees -= fee_quote;
        }
    }

    /// Apply a funding payment (positive = receive, negative = pay).
    pub fn apply_funding_payment(&mut self, funding_pnl_quote: f64) {
        self.pnl_funding += funding_pnl_quote;
    }
}

/// Global engine state.
/// Owned by the strategy engine; mutated by loops & event handlers.
#[derive(Debug)]
pub struct GlobalState {
    // ---- Per-venue collection ----
    pub venues: Vec<VenueState>,

    // ---- Fair value & Kalman filter (Section 5) ----
    /// x_hat = log(S_t), if initialised.
    pub kf_x_hat: Option<f64>,
    /// Kalman variance P_t.
    pub kf_p: f64,
    /// Last KF update time.
    pub kf_last_update_ms: Option<TimestampMs>,
    /// Fair value S_t.
    pub fair_value: Option<f64>,
    /// Previous fair value S_{t-1}.
    pub fair_value_prev: Option<f64>,

    // ---- Volatility & control scalars (Section 5 & 6) ----
    /// Short-horizon EWMA volatility of log S_t.
    pub fv_short_vol: f64,
    /// Long-horizon EWMA volatility of log S_t.
    pub fv_long_vol: f64,
    /// Effective volatility sigma_eff = max(short_vol, sigma_min).
    pub sigma_eff: f64,

    /// Volatility ratio (sigma_eff / sigma_ref) after clipping.
    pub vol_ratio_clipped: f64,
    /// Spread multiplier spread_mult(t).
    pub spread_mult: f64,
    /// Size multiplier size_mult(t).
    pub size_mult: f64,
    /// Hedge band multiplier band_mult(t).
    pub band_mult: f64,

    // ---- Inventory, dollar delta & basis (Section 8) ----
    /// Global TAO inventory q_t = sum_v q_v.
    pub q_global_tao: f64,
    /// Dollar delta Δ^{USD}_t = q_t * S_t.
    pub dollar_delta_usd: f64,
    /// Net basis exposure B_t in USD units (sum_v q_v * b_v).
    pub basis_usd: f64,
    /// Gross basis exposure B_t^{gross}.
    pub basis_gross_usd: f64,

    // ---- PnL buckets (Section 14.2) ----
    /// Daily realised PnL across venues (trades + funding + fees).
    pub daily_realised_pnl: f64,
    /// Daily unrealised PnL across venues (marked to S_t).
    pub daily_unrealised_pnl: f64,
    /// Daily total PnL = realised + unrealised.
    pub daily_pnl_total: f64,

    // ---- Risk regime & limits (Section 14) ----
    pub risk_regime: RiskRegime,
    pub kill_switch: bool,

    /// Vol-scaled hard dollar-delta limit L_t^{delta}.
    pub delta_limit_usd: f64,
    /// Warning basis limit |B_t| (basis_warn_frac * basis_hard_limit_usd).
    pub basis_limit_warn_usd: f64,
    /// Hard basis limit |B_t| (basis_hard_limit_usd).
    pub basis_limit_hard_usd: f64,
}

impl GlobalState {
    /// Construct initial state from config, with one VenueState per configured venue.
    pub fn new(cfg: &Config) -> Self {
        let venues = cfg
            .venues
            .iter()
            .map(|vc| VenueState {
                id: vc.id.clone(),
                name: vc.name.clone(),
                status: VenueStatus::Healthy,
                toxicity: 0.0,

                mid: None,
                spread: None,
                depth_near_mid: 0.0,
                last_mid_update_ms: None,
                local_vol_short: 0.0,
                local_vol_long: 0.0,

                position_tao: 0.0,
                vwap: 0.0,
                pnl_realised: 0.0,
                pnl_unrealised: 0.0,
                pnl_funding: 0.0,
                pnl_fees: 0.0,

                margin_balance: 10_000.0,
                margin_used: 0.0,
                margin_available: 10_000.0,
                price_liq: 0.0,
                dist_liq_sigma: f64::INFINITY,

                funding_8h: 0.0,
            })
            .collect();

        GlobalState {
            venues,

            kf_x_hat: None,
            kf_p: 0.0,
            kf_last_update_ms: None,
            fair_value: None,
            fair_value_prev: None,

            fv_short_vol: 0.0,
            fv_long_vol: 0.0,
            sigma_eff: 0.0,

            vol_ratio_clipped: 1.0,
            spread_mult: 1.0,
            size_mult: 1.0,
            band_mult: 1.0,

            q_global_tao: 0.0,
            dollar_delta_usd: 0.0,
            basis_usd: 0.0,
            basis_gross_usd: 0.0,

            daily_realised_pnl: 0.0,
            daily_unrealised_pnl: 0.0,
            daily_pnl_total: 0.0,

            risk_regime: RiskRegime::Normal,
            kill_switch: false,

            delta_limit_usd: 0.0,
            basis_limit_warn_usd: 0.0,
            basis_limit_hard_usd: 0.0,
        }
    }

    /// Convenience wrapper to apply a perp fill by venue index.
    pub fn apply_perp_fill(
        &mut self,
        venue_index: usize,
        side: Side,
        size_tao: f64,
        price: f64,
        fee_bps: f64,
    ) {
        if let Some(v) = self.venues.get_mut(venue_index) {
            v.apply_perp_fill(side, size_tao, price, fee_bps);
        }
    }

    /// Convenience wrapper to apply a funding payment by venue index.
    pub fn apply_funding_payment(&mut self, venue_index: usize, funding_pnl_quote: f64) {
        if let Some(v) = self.venues.get_mut(venue_index) {
            v.apply_funding_payment(funding_pnl_quote);
        }
    }
}
