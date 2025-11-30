// src/state.rs
//
// Global engine state + per-venue state for Paraphina.

use crate::config::Config;
use crate::types::{Side, TimestampMs, VenueStatus};

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
    /// Toxicity score ∈ [0,1].
    pub toxicity: f64,

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

/// High-level risk regime (Section 14 of the whitepaper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRegime {
    Normal,
    Warning,
    /// Hard limit / circuit-breaker regime.
    HardLimit,
}

/// Global engine state shared across strategy components.
#[derive(Debug, Clone)]
pub struct GlobalState {
    // ----- Per-venue -----
    pub venues: Vec<VenueState>,

    // ----- Fair value / “Kalman-lite” filter -----
    /// Last time the fair-value filter was updated.
    pub kf_last_update_ms: TimestampMs,
    /// Dummy variance term kept for future KF upgrades.
    pub kf_p: f64,
    /// Filtered log-price x_t = log(S_t).
    pub kf_x_hat: f64,
    /// Previous fair value S_{t-1}.
    pub fair_value_prev: f64,
    /// Current fair value S_t (if we have one).
    pub fair_value: Option<f64>,
    /// Short-horizon fair value vol (σ_short) – kept for future use.
    pub fv_short_vol: f64,
    /// Long-horizon fair value vol (σ_long) – kept for future use.
    pub fv_long_vol: f64,
    /// Effective volatility σ_eff (used by quoting / risk).
    pub sigma_eff: f64,

    // ----- Volatility-driven control scalars (Section 6) -----
    /// Vol ratio clipped into [vol_ratio_min, vol_ratio_max].
    pub vol_ratio_clipped: f64,
    pub spread_mult: f64,
    pub size_mult: f64,
    pub band_mult: f64,

    // ----- Inventory & basis (Section 8) -----
    /// Global trading inventory q_t in TAO.
    pub q_global_tao: f64,
    /// Dollar delta q_t * S_t.
    pub dollar_delta_usd: f64,
    /// Net basis exposure in USD (signed).
    pub basis_usd: f64,
    /// Gross basis exposure in USD (sum of absolute venue bases).
    pub basis_gross_usd: f64,

    // ----- PnL -----
    pub daily_realised_pnl: f64,
    pub daily_unrealised_pnl: f64,
    pub daily_pnl_total: f64,
    /// Optional PnL anchor fair value (not used yet but kept for spec).
    pub pnl_ref_fair_value: Option<f64>,

    // ----- Risk regime & limits -----
    pub risk_regime: RiskRegime,
    pub kill_switch: bool,
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

                position_tao: 0.0,
                funding_8h: 0.0,
                avg_entry_price: 0.0,

                // For now we just give each venue a synthetic chunk of
                // available margin and set liquidation far away.
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
//    we recompute after fills.

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
        if size_tao <= 0.0 {
            return;
        }

        if let Some(v) = self.venues.get_mut(venue_index) {
            // Signed trade size: + for buy, - for sell.
            let trade = match side {
                Side::Buy => size_tao,
                Side::Sell => -size_tao,
            };

            let q_old = v.position_tao;
            let p_old = v.avg_entry_price;
            let p_trade = price;

            // Total fee in USD (always reduces realised PnL if fee_bps > 0).
            let fee = (fee_bps / 10_000.0) * p_trade * size_tao.abs();

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
                        let p_new =
                            (p_old * w_old + p_trade * w_trade) / (w_old + w_trade);
                        v.position_tao = q_new;
                        v.avg_entry_price = p_new;
                    } else {
                        // Pathological cancellation.
                        v.position_tao = 0.0;
                        v.avg_entry_price = 0.0;
                    }
                } else {
                    // Closing or flipping some or all of the existing position.
                    let close_qty = trade.abs().min(q_old.abs());

                    if close_qty > 0.0 {
                        if q_old > 0.0 {
                            // Closing a long: PnL = (sell_price - entry_price) * qty.
                            realised += (p_trade - p_old) * close_qty;
                        } else {
                            // Closing a short: PnL = (entry_price - buy_price) * qty.
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

            // Subtract fees from realised PnL.
            realised -= fee;

            // Accumulate into global realised PnL.
            self.daily_realised_pnl += realised;
            // `daily_unrealised_pnl` will be updated in `recompute_after_fills`.
            self.daily_pnl_total =
                self.daily_realised_pnl + self.daily_unrealised_pnl;
        }
    }

    /// Recompute inventory, basis and mark-to-market PnL after a batch of fills.
    /// Called by the Engine after synthetic / hedge fills.
    pub fn recompute_after_fills(&mut self, _cfg: &Config) {
        let s_t = self.fair_value.unwrap_or(self.fair_value_prev);

        self.q_global_tao = 0.0;
        self.dollar_delta_usd = 0.0;
        self.basis_usd = 0.0;
        self.basis_gross_usd = 0.0;

        let mut unrealised = 0.0_f64;

        for v in &self.venues {
            self.q_global_tao += v.position_tao;
            self.dollar_delta_usd += v.position_tao * s_t;

            let mark = v.mid.unwrap_or(s_t);
            let basis_v = v.position_tao * (mark - s_t);
            self.basis_usd += basis_v;
            self.basis_gross_usd += basis_v.abs();

            if v.position_tao != 0.0 {
                unrealised += v.position_tao * (s_t - v.avg_entry_price);
            }
        }

        self.daily_unrealised_pnl = unrealised;
        self.daily_pnl_total =
            self.daily_realised_pnl + self.daily_unrealised_pnl;
    }
}
