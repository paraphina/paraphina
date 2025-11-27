// src/state.rs
//
// Global engine state + per-venue state for Paraphina.

use crate::config::Config;
use crate::types::{TimestampMs, VenueStatus};

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
    Critical,
}

/// Global engine state shared across strategy components.
#[derive(Debug, Clone)]
pub struct GlobalState {
    // ----- Per-venue -----
    pub venues: Vec<VenueState>,

    // ----- Fair value / Kalman -----
    pub kf_last_update_ms: Option<TimestampMs>,
    pub kf_p: f64,
    pub kf_x_hat: Option<f64>,
    pub fair_value_prev: Option<f64>,
    pub fair_value: Option<f64>,
    /// Short-horizon fair value vol (σ_short).
    pub fv_short_vol: f64,
    /// Long-horizon fair value vol (σ_long).
    pub fv_long_vol: f64,
    /// Effective volatility σ_eff = max(σ_short, σ_min).
    pub sigma_eff: f64,

    // ----- Volatility-driven control scalars (Section 6) -----
    pub vol_ratio_clipped: f64,
    pub spread_mult: f64,
    pub size_mult: f64,
    pub band_mult: f64,

    // ----- Inventory & basis (Section 8) -----
    /// Global trading inventory q_t in TAO.
    pub q_global_tao: f64,
    /// Dollar delta q_t * S_t.
    pub dollar_delta_usd: f64,
    /// Net basis exposure in USD.
    pub basis_usd: f64,
    /// Gross basis exposure in USD.
    pub basis_gross_usd: f64,

    // ----- PnL -----
    pub daily_realised_pnl: f64,
    pub daily_unrealised_pnl: f64,
    pub daily_pnl_total: f64,
    /// Reference fair value for daily PnL marking (e.g. S_ref at start of day).
    pub pnl_ref_fair_value: Option<f64>,

    // ----- Risk regime & limits -----
    pub risk_regime: RiskRegime,
    pub kill_switch: bool,
    /// Volatility-scaled delta limit (updated in engine).
    pub delta_limit_usd: f64,
    pub basis_limit_warn_usd: f64,
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

            kf_last_update_ms: None,
            kf_p: cfg.kalman.p_init,
            kf_x_hat: None,
            fair_value_prev: None,
            fair_value: None,
            fv_short_vol: 0.0,
            fv_long_vol: 0.0,
            sigma_eff: cfg.volatility.sigma_min,

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
            pnl_ref_fair_value: None,

            risk_regime: RiskRegime::Normal,
            kill_switch: false,

            delta_limit_usd: cfg.risk.delta_hard_limit_usd_base,
            basis_limit_warn_usd: cfg.risk.basis_warn_frac * cfg.risk.basis_hard_limit_usd,
            basis_limit_hard_usd: cfg.risk.basis_hard_limit_usd,
        }
    }
}

// --- TEMP: simple perp-fill application stub ---
//
// Add this at the very bottom of src/state.rs,
// after the existing `impl GlobalState` block(s).

impl GlobalState {
    /// Apply a single perp fill into the state.
    ///
    /// For now this only updates the venue position in TAO.
    /// Detailed realised/unrealised PnL accounting will be added later.
    pub fn apply_perp_fill(
        &mut self,
        venue_index: usize,
        side: crate::types::Side,
        size_tao: f64,
        _price: f64,
        _fee_bps: f64,
    ) {
        if let Some(v) = self.venues.get_mut(venue_index) {
            let signed = match side {
                crate::types::Side::Buy => size_tao,
                crate::types::Side::Sell => -size_tao,
            };
            v.position_tao += signed;
        }

        // PnL fields (daily_realised_pnl, daily_unrealised_pnl) are left
        // unchanged for now so the risk engine still sees 0 PnL unless
        // we set something manually.
    }
}
