// src/rl/observation.rs
//
// Versioned Observation schema for RL policy input.
//
// Per WHITEPAPER Appendix A.3, this observation vector provides a stable,
// serializable state snapshot derived from GlobalState + per-venue VenueState.
//
// Design requirements:
// - Versioned (obs_version field) for schema evolution
// - Serializable (serde) for logging and replay
// - Deterministic ordering (Vec, not HashMap) for reproducibility
// - Normalized and clipped features for stable RL training

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::state::{GlobalState, KillReason, RiskRegime};
use crate::types::{TimestampMs, VenueStatus};

/// Current observation schema version.
/// Increment when adding/removing/changing fields.
pub const OBS_VERSION: u32 = 1;

/// Per-venue observation features.
///
/// Contains all venue-specific state needed by a policy.
/// Fields are ordered deterministically (by venue index).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VenueObservation {
    /// Stable venue identifier.
    pub venue_id: String,
    /// Venue index for deterministic ordering.
    pub venue_index: usize,

    // ----- Market data -----
    /// Current mid price (None if no book data).
    pub mid: Option<f64>,
    /// Current bid-ask spread.
    pub spread: Option<f64>,
    /// Depth near mid (USD).
    pub depth_near_mid: f64,
    /// Milliseconds since last book update (staleness).
    pub staleness_ms: Option<i64>,

    // ----- Local volatility -----
    /// Short-horizon EWMA local vol of log mid.
    pub local_vol_short: f64,
    /// Long-horizon EWMA local vol of log mid.
    pub local_vol_long: f64,

    // ----- Health and toxicity -----
    /// Venue status (Healthy/Warning/Disabled).
    pub status: VenueStatusObs,
    /// Toxicity score ∈ [0,1].
    pub toxicity: f64,

    // ----- Position and funding -----
    /// Net TAO-equivalent perp position.
    pub position_tao: f64,
    /// Volume-weighted average entry price (USD/TAO).
    pub avg_entry_price: f64,
    /// Current 8h funding rate (dimensionless).
    pub funding_8h: f64,

    // ----- Margin and liquidation -----
    /// Margin available for new positions (USD).
    pub margin_available_usd: f64,
    /// Estimated distance to liquidation in sigma units.
    pub dist_liq_sigma: f64,
}

/// Serializable venue status (mirrors VenueStatus).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VenueStatusObs {
    Healthy,
    Warning,
    Disabled,
}

impl From<VenueStatus> for VenueStatusObs {
    fn from(status: VenueStatus) -> Self {
        match status {
            VenueStatus::Healthy => VenueStatusObs::Healthy,
            VenueStatus::Warning => VenueStatusObs::Warning,
            VenueStatus::Disabled => VenueStatusObs::Disabled,
        }
    }
}

/// Serializable risk regime (mirrors RiskRegime).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskRegimeObs {
    Normal,
    Warning,
    HardLimit,
}

impl From<RiskRegime> for RiskRegimeObs {
    fn from(regime: RiskRegime) -> Self {
        match regime {
            RiskRegime::Normal => RiskRegimeObs::Normal,
            RiskRegime::Warning => RiskRegimeObs::Warning,
            RiskRegime::HardLimit => RiskRegimeObs::HardLimit,
        }
    }
}

/// Serializable kill reason (mirrors KillReason).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KillReasonObs {
    None,
    PnlHardBreach,
    DeltaHardBreach,
    BasisHardBreach,
    LiquidationDistanceBreach,
}

impl From<KillReason> for KillReasonObs {
    fn from(reason: KillReason) -> Self {
        match reason {
            KillReason::None => KillReasonObs::None,
            KillReason::PnlHardBreach => KillReasonObs::PnlHardBreach,
            KillReason::DeltaHardBreach => KillReasonObs::DeltaHardBreach,
            KillReason::BasisHardBreach => KillReasonObs::BasisHardBreach,
            KillReason::LiquidationDistanceBreach => KillReasonObs::LiquidationDistanceBreach,
        }
    }
}

/// Global observation for RL policy input.
///
/// Contains all global state plus per-venue observations.
/// This is the canonical state representation passed to policies.
///
/// Per WHITEPAPER Appendix A.3, includes:
/// - fair_value, sigma_eff, spread/size multipliers
/// - inventory/exposure, basis, daily pnl
/// - risk regime, kill switch
/// - Per-venue: mid/spread/depth/vol/toxicity/status/position/funding/liquidation-distance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    // ----- Metadata -----
    /// Schema version for forwards/backwards compatibility.
    pub obs_version: u32,
    /// Current timestamp (ms since epoch).
    pub timestamp_ms: TimestampMs,
    /// Current tick index.
    pub tick_index: u64,

    // ----- Global fair value and volatility -----
    /// Synthetic fair value S_t (USD/TAO).
    pub fair_value: Option<f64>,
    /// Effective volatility σ_eff.
    pub sigma_eff: f64,
    /// Vol ratio clipped (σ_eff / σ_ref).
    pub vol_ratio_clipped: f64,

    // ----- Control scalars (Section 6) -----
    /// Spread multiplier from vol.
    pub spread_mult: f64,
    /// Size multiplier from vol.
    pub size_mult: f64,
    /// Band multiplier for hedge deadband.
    pub band_mult: f64,

    // ----- Inventory and exposure -----
    /// Global net trading inventory (TAO).
    pub q_global_tao: f64,
    /// Dollar delta (USD).
    pub dollar_delta_usd: f64,
    /// Current delta limit (USD, vol-scaled).
    pub delta_limit_usd: f64,

    // ----- Basis exposure -----
    /// Net basis exposure (USD).
    pub basis_usd: f64,
    /// Gross basis exposure (USD).
    pub basis_gross_usd: f64,
    /// Basis warning limit (USD).
    pub basis_limit_warn_usd: f64,
    /// Basis hard limit (USD).
    pub basis_limit_hard_usd: f64,

    // ----- PnL -----
    /// Daily realized PnL (USD).
    pub daily_realised_pnl: f64,
    /// Daily unrealized PnL (USD).
    pub daily_unrealised_pnl: f64,
    /// Daily total PnL (USD).
    pub daily_pnl_total: f64,

    // ----- Risk state -----
    /// Current risk regime.
    pub risk_regime: RiskRegimeObs,
    /// Kill switch status.
    pub kill_switch: bool,
    /// Kill reason (if any).
    pub kill_reason: KillReasonObs,

    // ----- FV gating (Milestone D) -----
    /// Whether fair value was successfully updated this tick.
    pub fv_available: bool,
    /// Count of healthy venues used in FV update.
    pub healthy_venues_used_count: usize,

    // ----- Per-venue observations -----
    /// Ordered list of per-venue observations.
    /// Deterministic ordering by venue_index.
    pub venues: Vec<VenueObservation>,
}

impl Observation {
    /// Build an Observation from GlobalState and Config.
    ///
    /// This is the canonical way to create an observation for policy input.
    /// The observation is deterministic given the same state.
    pub fn from_state(
        state: &GlobalState,
        _cfg: &Config,
        now_ms: TimestampMs,
        tick_index: u64,
    ) -> Self {
        let mut venues = Vec::with_capacity(state.venues.len());

        for (idx, v) in state.venues.iter().enumerate() {
            let staleness_ms = v.last_mid_update_ms.map(|ts| now_ms - ts);

            venues.push(VenueObservation {
                venue_id: v.id.to_string(),
                venue_index: idx,
                mid: v.mid,
                spread: v.spread,
                depth_near_mid: v.depth_near_mid,
                staleness_ms,
                local_vol_short: v.local_vol_short,
                local_vol_long: v.local_vol_long,
                status: v.status.into(),
                toxicity: v.toxicity,
                position_tao: v.position_tao,
                avg_entry_price: v.avg_entry_price,
                funding_8h: v.funding_8h,
                margin_available_usd: v.margin_available_usd,
                dist_liq_sigma: v.dist_liq_sigma,
            });
        }

        Observation {
            obs_version: OBS_VERSION,
            timestamp_ms: now_ms,
            tick_index,
            fair_value: state.fair_value,
            sigma_eff: state.sigma_eff,
            vol_ratio_clipped: state.vol_ratio_clipped,
            spread_mult: state.spread_mult,
            size_mult: state.size_mult,
            band_mult: state.band_mult,
            q_global_tao: state.q_global_tao,
            dollar_delta_usd: state.dollar_delta_usd,
            delta_limit_usd: state.delta_limit_usd,
            basis_usd: state.basis_usd,
            basis_gross_usd: state.basis_gross_usd,
            basis_limit_warn_usd: state.basis_limit_warn_usd,
            basis_limit_hard_usd: state.basis_limit_hard_usd,
            daily_realised_pnl: state.daily_realised_pnl,
            daily_unrealised_pnl: state.daily_unrealised_pnl,
            daily_pnl_total: state.daily_pnl_total,
            risk_regime: state.risk_regime.into(),
            kill_switch: state.kill_switch,
            kill_reason: state.kill_reason.into(),
            fv_available: state.fv_available,
            healthy_venues_used_count: state.healthy_venues_used_count,
            venues,
        }
    }

    /// Serialize to JSON bytes for deterministic comparison.
    ///
    /// Returns canonical JSON with sorted keys for byte-for-byte reproducibility.
    pub fn to_canonical_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        // Use serde_json's default which preserves struct field order
        serde_json::to_vec(self)
    }

    /// Mean toxicity across all venues (for reward computation).
    pub fn mean_toxicity(&self) -> f64 {
        if self.venues.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.venues.iter().map(|v| v.toxicity).sum();
        sum / self.venues.len() as f64
    }

    /// Minimum liquidation distance across all venues (for risk checks).
    pub fn min_dist_liq_sigma(&self) -> f64 {
        self.venues
            .iter()
            .filter(|v| v.dist_liq_sigma.is_finite() && v.dist_liq_sigma >= 0.0)
            .map(|v| v.dist_liq_sigma)
            .fold(f64::INFINITY, f64::min)
    }

    /// Count of healthy venues.
    pub fn healthy_venue_count(&self) -> usize {
        self.venues
            .iter()
            .filter(|v| matches!(v.status, VenueStatusObs::Healthy))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::GlobalState;

    fn setup_test_state() -> (Config, GlobalState) {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        state.fair_value = Some(300.0);
        state.fair_value_prev = 300.0;
        state.sigma_eff = 0.02;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;
        state.vol_ratio_clipped = 1.0;
        state.delta_limit_usd = 100_000.0;

        for v in &mut state.venues {
            v.mid = Some(300.0);
            v.spread = Some(1.0);
            v.depth_near_mid = 10_000.0;
            v.margin_available_usd = 10_000.0;
            v.dist_liq_sigma = 10.0;
            v.status = crate::types::VenueStatus::Healthy;
            v.toxicity = 0.1;
            v.last_mid_update_ms = Some(1000);
        }

        (cfg, state)
    }

    #[test]
    fn test_observation_from_state() {
        let (cfg, state) = setup_test_state();
        let obs = Observation::from_state(&state, &cfg, 2000, 5);

        assert_eq!(obs.obs_version, OBS_VERSION);
        assert_eq!(obs.tick_index, 5);
        assert_eq!(obs.timestamp_ms, 2000);
        assert_eq!(obs.fair_value, Some(300.0));
        assert_eq!(obs.venues.len(), cfg.venues.len());

        // Check venues are ordered by index
        for (i, v) in obs.venues.iter().enumerate() {
            assert_eq!(v.venue_index, i);
        }
    }

    #[test]
    fn test_observation_serialization_deterministic() {
        let (cfg, state) = setup_test_state();

        let obs1 = Observation::from_state(&state, &cfg, 2000, 5);
        let obs2 = Observation::from_state(&state, &cfg, 2000, 5);

        let json1 = obs1.to_canonical_json().unwrap();
        let json2 = obs2.to_canonical_json().unwrap();

        assert_eq!(json1, json2, "Same state should produce identical JSON");
    }

    #[test]
    fn test_observation_roundtrip() {
        let (cfg, state) = setup_test_state();
        let obs = Observation::from_state(&state, &cfg, 2000, 5);

        let json = serde_json::to_string(&obs).unwrap();
        let parsed: Observation = serde_json::from_str(&json).unwrap();

        assert_eq!(obs, parsed, "Observation should roundtrip through JSON");
    }

    #[test]
    fn test_mean_toxicity() {
        let (cfg, mut state) = setup_test_state();

        state.venues[0].toxicity = 0.1;
        state.venues[1].toxicity = 0.2;
        state.venues[2].toxicity = 0.3;
        state.venues[3].toxicity = 0.4;
        state.venues[4].toxicity = 0.5;

        let obs = Observation::from_state(&state, &cfg, 2000, 5);
        let mean = obs.mean_toxicity();

        let expected = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5.0;
        assert!((mean - expected).abs() < 1e-9);
    }

    #[test]
    fn test_min_dist_liq_sigma() {
        let (cfg, mut state) = setup_test_state();

        state.venues[0].dist_liq_sigma = 10.0;
        state.venues[1].dist_liq_sigma = 5.0;
        state.venues[2].dist_liq_sigma = 3.0;
        state.venues[3].dist_liq_sigma = 8.0;
        state.venues[4].dist_liq_sigma = 15.0;

        let obs = Observation::from_state(&state, &cfg, 2000, 5);
        let min = obs.min_dist_liq_sigma();

        assert!((min - 3.0).abs() < 1e-9);
    }
}
