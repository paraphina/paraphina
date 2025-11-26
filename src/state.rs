// src/state.rs
//
// Global and per-venue state representation, mirroring Sections 3 & 8
// of the whitepaper.

use crate::config::{Config, VenueConfig};
use crate::types::{TimestampMs, VenueStatus};

/// Risk regime as per the whitepaper: Normal / Warning / Critical.
#[derive(Debug, Clone, Copy)]
pub enum RiskRegime {
    Normal,
    Warning,
    Critical,
}

/// Per-venue state for a perp venue v.
#[derive(Debug, Clone)]
pub struct VenueState {
    /// Stable identifier (e.g. "extended", "hyperliquid").
    pub id: String,
    /// Human-readable name.
    pub name: String,

    // ---------- Market data ----------
    /// Current mid price m_{v,t} (if known).
    pub mid: Option<f64>,
    /// Current spread (best_ask - best_bid).
    pub spread: Option<f64>,
    /// Short-horizon EWMA vol of log mid.
    pub local_vol_short: f64,
    /// Long-horizon EWMA vol of log mid.
    pub local_vol_long: f64,
    /// Total size within K levels around mid (depth_near_mid_v).
    pub depth_near_mid: f64,
    /// Timestamp of last mid update (ms).
    pub last_mid_update_ms: Option<TimestampMs>,

    // ---------- Toxicity & health ----------
    /// Toxicity score in [0,1].
    pub toxicity: f64,
    /// Venue health classification.
    pub status: VenueStatus,

    // ---------- Position & margin ----------
    /// Net TAO-equivalent perp position q_v (TAO).
    pub position_tao: f64,
    /// Margin balance (USD).
    pub margin_balance: f64,
    /// Margin available (USD).
    pub margin_available: f64,
    /// Current 8h funding rate.
    pub funding_8h: f64,
    /// Estimated liquidation price (if known).
    pub price_liq: Option<f64>,
    /// Distance to liquidation in sigma units.
    pub dist_liq_sigma: f64,
}

impl VenueState {
    /// Construct a fresh VenueState from a VenueConfig, with all dynamic fields
    /// initialised to "no data yet" or 0.
    pub fn new(cfg: &VenueConfig) -> Self {
        Self {
            id: cfg.id.clone(),
            name: cfg.name.clone(),

            mid: None,
            spread: None,
            local_vol_short: 0.0,
            local_vol_long: 0.0,
            depth_near_mid: 0.0,
            last_mid_update_ms: None,

            toxicity: 0.0,
            status: VenueStatus::Healthy,

            position_tao: 0.0,
            margin_balance: 0.0,
            margin_available: 0.0,
            funding_8h: 0.0,
            price_liq: None,
            dist_liq_sigma: f64::INFINITY,
        }
    }
}

/// Global engine state, aggregating all venues plus fair value, vols,
/// global inventory q_t, basis exposure B_t, PnL and risk regime.
#[derive(Debug)]
pub struct GlobalState {
    /// Current risk regime of the engine.
    pub risk_regime: RiskRegime,
    /// Global kill switch (true in Critical regime).
    pub kill_switch: bool,

    // ---------- Fair value & volatility ----------
    /// Fair value S_t (synthetic spot) if initialised.
    pub fair_value: Option<f64>,
    /// Previous fair value S_{t-1}.
    pub fair_value_prev: Option<f64>,
    /// Short-horizon EWMA volatility of log S_t.
    pub fv_short_vol: f64,
    /// Long-horizon EWMA volatility of log S_t.
    pub fv_long_vol: f64,
    /// Effective volatility sigma_t^{eff}.
    pub sigma_eff: f64,

    // ---------- Kalman state ----------
    /// Kalman filter state x_hat = log S_t, if initialised.
    pub kf_x_hat: Option<f64>,
    /// Kalman variance P_t.
    pub kf_p: f64,
    /// Last Kalman update time (ms).
    pub kf_last_update_ms: Option<TimestampMs>,

    // ---------- Volatility-driven scalars ----------
    /// Clipped volatility ratio vol_ratio_clipped.
    pub vol_ratio_clipped: f64,
    /// Spread multiplier spread_mult(t).
    pub spread_mult: f64,
    /// Size multiplier size_mult(t).
    pub size_mult: f64,
    /// Hedge-band multiplier band_mult(t).
    pub band_mult: f64,

    // ---------- Inventory & basis ----------
    /// Per-venue states.
    pub venues: Vec<VenueState>,
    /// Global inventory q_t in TAO.
    pub q_global_tao: f64,
    /// Dollar delta Δ^{USD}_t = q_t * S_t.
    pub dollar_delta_usd: f64,
    /// Net basis exposure B_t (USD).
    pub basis_usd: f64,
    /// Gross basis exposure B_t^{gross} (USD).
    pub basis_gross_usd: f64,

    // ---------- Risk limits (last computed, for debugging/metrics) ----------
    /// Last computed delta limit L_t^{delta} in USD.
    pub delta_limit_usd: f64,
    /// Last computed Warning basis threshold in USD.
    pub basis_limit_warn_usd: f64,
    /// Hard basis limit in USD (copied from config).
    pub basis_limit_hard_usd: f64,

    // ---------- PnL ----------
    /// Realised PnL for the current day.
    pub daily_realised_pnl: f64,
    /// Unrealised PnL (marking to S_t).
    pub daily_unrealised_pnl: f64,
    /// Total daily PnL.
    pub daily_pnl_total: f64,
}

impl GlobalState {
    /// Construct an initial GlobalState from static config.
    pub fn new(cfg: &Config) -> Self {
        let venues = cfg.venues.iter().map(VenueState::new).collect();

        Self {
            risk_regime: RiskRegime::Normal,
            kill_switch: false,

            fair_value: None,
            fair_value_prev: None,
            fv_short_vol: 0.0,
            fv_long_vol: 0.0,
            sigma_eff: 0.0,

            kf_x_hat: None,
            kf_p: 0.0,
            kf_last_update_ms: None,

            vol_ratio_clipped: 1.0,
            spread_mult: 1.0,
            size_mult: 1.0,
            band_mult: 1.0,

            venues,
            q_global_tao: 0.0,
            dollar_delta_usd: 0.0,
            basis_usd: 0.0,
            basis_gross_usd: 0.0,

            delta_limit_usd: 0.0,
            basis_limit_warn_usd: 0.0,
            basis_limit_hard_usd: 0.0,

            daily_realised_pnl: 0.0,
            daily_unrealised_pnl: 0.0,
            daily_pnl_total: 0.0,
        }
    }
}
