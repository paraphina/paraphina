// src/config.rs
//
// Central configuration for the Paraphina engine.
// This is the single source of truth that maps directly onto the
// whitepaper parameters (venues, Kalman fair value, vols, risk,
// MM quoting, hedging, toxicity / venue health).
//
// It also carries a small number of "simulation environment"
// parameters such as the initial global inventory q0 in TAO.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Serialize;

#[derive(Debug, Clone)]
pub struct Config {
    /// Human-readable config / release version.
    pub version: &'static str,
    /// Signed initial global position in TAO.
    ///
    /// Sign convention:
    /// - `q0 > 0`  → start long `q0` TAO
    /// - `q0 < 0`  → start short `|q0|` TAO
    /// - `|q0| ≈ 0` → start flat
    pub initial_q_tao: f64,
    /// Static config per venue (Extended, Hyperliquid, Aster, Lighter, Paradex).
    pub venues: Vec<VenueConfig>,
    /// Orderbook / fair-value observation config.
    pub book: BookConfig,
    /// Fill aggregation window (ms) for batching fills (§4.3).
    pub fill_agg_interval_ms: i64,
    /// Main loop interval (ms) (§16).
    pub main_loop_interval_ms: i64,
    /// Hedge loop interval (ms) (§16).
    pub hedge_loop_interval_ms: i64,
    /// Risk loop interval (ms) (§16).
    pub risk_loop_interval_ms: i64,
    /// Kalman filter config for log fair value.
    pub kalman: KalmanConfig,
    /// Volatility / control-scalar config.
    pub volatility: VolatilityConfig,
    /// Global risk / limit config.
    pub risk: RiskConfig,
    /// Funding staleness + boundary policy.
    pub funding: FundingPolicyConfig,
    /// Quoting (Avellaneda–Stoikov + funding/basis) config.
    pub mm: MmConfig,
    /// Hedge engine (global LQ controller + band).
    pub hedge: HedgeConfig,
    /// Exit engine (cross-venue profit-only exits).
    pub exit: ExitConfig,
    /// Toxicity scoring + venue health config.
    pub toxicity: ToxicityConfig,
}

/// Coarse risk profile preset used by the CLI / research harness.
///
/// These presets only tweak a small set of hyperparameters on top of the
/// whitepaper-spec default (which we treat as "Balanced").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskProfile {
    Conservative,
    Balanced,
    Aggressive,
}

impl RiskProfile {
    /// Return a stable lowercase name for the profile (used in logs/telemetry).
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskProfile::Conservative => "conservative",
            RiskProfile::Balanced => "balanced",
            RiskProfile::Aggressive => "aggressive",
        }
    }

    /// Parse a profile name (case-insensitive). Returns None if unrecognized.
    pub fn parse(s: &str) -> Option<RiskProfile> {
        match s.trim().to_ascii_lowercase().as_str() {
            "balanced" | "bal" | "b" => Some(RiskProfile::Balanced),
            "conservative" | "cons" | "c" => Some(RiskProfile::Conservative),
            "aggressive" | "agg" | "a" | "loose" | "l" => Some(RiskProfile::Aggressive),
            _ => None,
        }
    }
}

/// Source of the effective risk profile (for logging/debugging precedence).
///
/// Precedence order (highest to lowest):
/// 1. CLI argument (--profile)
/// 2. Environment variable (PARAPHINA_RISK_PROFILE)
/// 3. Scenario file (if applicable)
/// 4. Default (Balanced)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileSource {
    /// Explicitly provided via CLI argument (highest priority).
    Cli,
    /// Loaded from PARAPHINA_RISK_PROFILE environment variable.
    Env,
    /// Loaded from scenario YAML file (sim_eval only).
    Scenario,
    /// Default fallback (Balanced).
    Default,
}

impl ProfileSource {
    /// Return a stable lowercase name for the source (used in logs/telemetry).
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfileSource::Cli => "cli",
            ProfileSource::Env => "env",
            ProfileSource::Scenario => "scenario",
            ProfileSource::Default => "default",
        }
    }
}

/// Resolved profile with its source for logging.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveProfile {
    pub profile: RiskProfile,
    pub source: ProfileSource,
}

impl EffectiveProfile {
    /// Log the effective profile at startup (INFO level to stderr).
    ///
    /// Format: `effective_risk_profile=<profile> source=<source>`
    pub fn log_startup(&self) {
        eprintln!(
            "effective_risk_profile={} source={}",
            self.profile.as_str(),
            self.source.as_str()
        );
    }
}

/// Resolve the effective risk profile using standard precedence rules.
///
/// Precedence (highest to lowest):
/// 1. `cli_profile` - if Some, use it (source=cli)
/// 2. `PARAPHINA_RISK_PROFILE` env var - if set and parseable (source=env)
/// 3. `scenario_profile` - if Some and parseable (source=scenario, for sim_eval)
/// 4. Default Balanced (source=default)
///
/// # Arguments
/// * `cli_profile` - Profile from CLI argument (--profile), if provided
/// * `scenario_profile` - Profile from scenario spec, if applicable (sim_eval)
///
/// # Returns
/// `EffectiveProfile` with the resolved profile and its source.
pub fn resolve_effective_profile(
    cli_profile: Option<RiskProfile>,
    scenario_profile: Option<&str>,
) -> EffectiveProfile {
    // 1. CLI takes highest precedence
    if let Some(p) = cli_profile {
        return EffectiveProfile {
            profile: p,
            source: ProfileSource::Cli,
        };
    }

    // 2. Environment variable
    if let Ok(env_val) = std::env::var("PARAPHINA_RISK_PROFILE") {
        if !env_val.is_empty() {
            if let Some(p) = RiskProfile::parse(&env_val) {
                return EffectiveProfile {
                    profile: p,
                    source: ProfileSource::Env,
                };
            }
            // Non-empty but unparseable: warn and fall through
            eprintln!(
                "[config] WARN: invalid PARAPHINA_RISK_PROFILE={:?}; ignoring",
                env_val
            );
        }
    }

    // 3. Scenario profile (for sim_eval)
    if let Some(s) = scenario_profile {
        if let Some(p) = RiskProfile::parse(s) {
            return EffectiveProfile {
                profile: p,
                source: ProfileSource::Scenario,
            };
        }
    }

    // 4. Default
    EffectiveProfile {
        profile: RiskProfile::Balanced,
        source: ProfileSource::Default,
    }
}

#[derive(Debug, Clone)]
pub struct VenueConfig {
    /// Stable identifier used in logs / routing (e.g. "extended").
    pub id: String,
    /// Arc-wrapped identifier for cheap cloning in hot paths.
    /// This is computed from `id` at construction time.
    pub id_arc: Arc<str>,
    /// Human-readable venue name.
    pub name: String,
    /// Smallest price tick size for this venue.
    pub tick_size: f64,
    /// Base per-order size in TAO (before vol / risk scaling).
    pub base_order_size: f64,
    /// Hard max per-order size in TAO.
    pub max_order_size: f64,
    /// Maker fee in basis points (positive = fee).
    pub maker_fee_bps: f64,
    /// Taker fee in basis points (positive = fee).
    pub taker_fee_bps: f64,
    /// Maker rebate in basis points (positive = rebate).
    pub maker_rebate_bps: f64,
    /// Avellaneda–Stoikov risk aversion γ_v.
    pub gamma: f64,
    /// Avellaneda–Stoikov intensity decay k_v.
    pub k: f64,
    /// Liquidity weight w_v^{liq} for venue inventory targeting.
    pub w_liq: f64,
    /// Funding weight w_v^{fund} for venue inventory targeting.
    pub w_fund: f64,
    /// Whether this venue is allowed to be used for hedging.
    pub is_hedge_allowed: bool,
    /// Whether this venue is allowed to contribute observations to the global
    /// fair-value filter / healthy-venues-used set.
    pub contributes_to_fv: bool,
    /// Optional per-venue gate on spread quality before the venue may
    /// contribute to fair value. When set, the venue is excluded from the
    /// global KF on ticks where `spread / mid * 10_000` exceeds this threshold.
    pub fv_max_spread_bps: Option<f64>,
    /// Minimum lot size in TAO (orders smaller than this are rejected).
    pub lot_size_tao: f64,
    /// Size step/increment in TAO (orders must be multiples of this).
    pub size_step_tao: f64,
    /// Minimum notional value in USD (orders below this are skipped).
    pub min_notional_usd: f64,
    /// Per-venue override for stale_ms threshold. If Some, this venue uses this
    /// threshold instead of the global `book.stale_ms`. Useful for high-latency
    /// venues (e.g., Hyperliquid) that need a larger staleness window.
    pub stale_ms_override: Option<i64>,
    /// Per-venue max order requests per second (overrides global rate limit).
    pub rate_limit_rps: Option<f64>,
    /// Per-venue rate limit burst capacity.
    pub rate_limit_burst: Option<u32>,
}

impl VenueConfig {
    /// Returns the effective stale_ms for this venue: the override if set,
    /// otherwise the provided global fallback.
    #[inline]
    pub fn effective_stale_ms(&self, global_stale_ms: i64) -> i64 {
        self.stale_ms_override.unwrap_or(global_stale_ms)
    }
}

#[derive(Debug, Clone)]
pub struct BookConfig {
    /// Number of levels to track per side in each order book.
    pub depth_levels: usize,
    /// Max age (ms) before a book is considered stale.
    pub stale_ms: i64,
    /// Minimum healthy venues required for KF observation update.
    pub min_healthy_for_kf: u32,
    /// Max allowed relative mid move vs last fair before treating as outlier.
    pub max_mid_jump_pct: f64,
}

#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// State noise q per second for log price.
    pub q_base: f64,
    /// Coefficient a for observation noise vs spread.
    pub r_a: f64,
    /// Coefficient b for observation noise vs depth.
    pub r_b: f64,
    /// Min observation noise variance.
    pub r_min: f64,
    /// Max observation noise variance.
    pub r_max: f64,
    /// Initial variance P_0 for the Kalman filter.
    pub p_init: f64,
}

#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// EWMA alpha for short-horizon fair value volatility.
    pub fv_vol_alpha_short: f64,
    /// EWMA alpha for long-horizon fair value volatility.
    pub fv_vol_alpha_long: f64,
    /// Minimum effective volatility σ_min.
    pub sigma_min: f64,
    /// Reference volatility σ_ref for vol_ratio.
    pub vol_ref: f64,
    /// Min vol_ratio used when clipping.
    pub vol_ratio_min: f64,
    /// Max vol_ratio used when clipping.
    pub vol_ratio_max: f64,
    /// Coefficient c_s in spread_mult(t).
    pub spread_vol_mult_coeff: f64,
    /// Coefficient c_q in size_mult(t).
    pub size_vol_mult_coeff: f64,
    /// Coefficient c_band in band_mult(t).
    pub band_vol_mult_coeff: f64,
    /// Timescale (seconds) that `vol_ref` was calibrated for.
    /// Used to auto-scale `vol_ref` to per-tick cadence:
    ///   vol_ref_tick = vol_ref * sqrt(tick_sec / vol_ref_cadence_sec)
    /// Default: 86400.0 (1 day) — vol_ref = 0.028125 ≈ 44.6% annualized.
    pub vol_ref_cadence_sec: f64,
    /// Timescale (seconds) that `sigma_min` was calibrated for.
    /// Same scaling logic as `vol_ref_cadence_sec`.
    /// Default: 86400.0.
    pub sigma_min_cadence_sec: f64,
}

#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Base dollar-delta limit before vol scaling (at vol_ratio ≈ 1).
    pub delta_hard_limit_usd_base: f64,
    /// Fraction of delta limit where Warning regime begins.
    pub delta_warn_frac: f64,
    /// Hard limit on basis exposure |B_t| in USD.
    pub basis_hard_limit_usd: f64,
    /// Fraction of basis limit where Warning begins.
    pub basis_warn_frac: f64,
    /// Daily loss limit (realised + unrealised), in absolute USD.
    /// The engine interprets this as a positive loss threshold.
    pub daily_loss_limit: f64,
    /// Fraction of loss limit where Warning regime begins.
    pub pnl_warn_frac: f64,
    /// Extra spread multiplier applied in Warning regime.
    pub spread_warn_mult: f64,
    /// Max per-order size (TAO) in Warning regime.
    pub q_warn_cap: f64,
    /// Safety factor for MM margin sizing (MM_MARGIN_SAFETY).
    pub mm_margin_safety: f64,
    /// Max leverage assumption for MM sizing (MM_MAX_LEVERAGE).
    pub mm_max_leverage: f64,
    /// Sigma distance where liq Warning starts (LIQ_WARN_SIGMA).
    pub liq_warn_sigma: f64,
    /// Sigma distance where liq is considered “too close” (LIQ_CRIT_SIGMA).
    pub liq_crit_sigma: f64,
}

#[derive(Debug, Clone)]
pub struct FundingPolicyConfig {
    /// Funding data staleness threshold (ms).
    pub stale_ms: i64,
    /// Avoid using funding within this window (ms) of next funding timestamp.
    pub avoid_window_ms: i64,
}

#[derive(Debug, Clone)]
pub struct MmConfig {
    /// Weight of basis in reservation price (β_b in spec).
    pub basis_weight: f64,
    /// Weight of funding in reservation price (β_f in spec).
    pub funding_weight: f64,
    /// Enable funding-aware MM logic (default: false).
    pub funding_enabled: bool,
    /// Minimum per-unit edge for local MM quotes (in USD).
    pub edge_local_min: f64,
    /// Optional per-venue edge floor overrides keyed by venue id.
    pub edge_local_min_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue bid edge floor overrides keyed by venue id.
    pub edge_local_min_bid_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue ask edge floor overrides keyed by venue id.
    pub edge_local_min_ask_by_venue: BTreeMap<String, f64>,
    /// Multiplier for volatility-based edge buffer.
    pub edge_vol_mult: f64,
    /// Optional multiplier that adds estimated hedge/forced-unwind cost to MM
    /// local edge floors. Default is zero to preserve legacy behavior.
    pub hedge_cost_edge_mult: f64,
    /// Risk parameter η in size objective J(Q)=eQ - 0.5 η Q^2.
    pub size_eta: f64,
    /// λ_inv ∈ [0,1] controlling anchoring to per-venue targets.
    pub lambda_inv: f64,
    /// Quote horizon T (seconds) in the AS model.
    pub quote_horizon_sec: f64,
    /// Slope of funding→inventory skew map.
    pub funding_skew_slope: f64,
    /// Clip for funding-driven skew magnitude.
    pub funding_skew_clip: f64,

    // ----- Order management (Section 11) -----
    /// Minimum quote lifetime before replacement (milliseconds).
    pub min_quote_lifetime_ms: i64,
    /// Optional per-venue quote lifetime overrides keyed by venue id.
    pub min_quote_lifetime_ms_by_venue: BTreeMap<String, i64>,
    /// Optional Paradex-only extension for keeping a safe same-side order when the
    /// control layer temporarily suppresses the desired side.
    pub paradex_post_control_suppression_grace_ms: Option<i64>,
    /// Optional Paradex-only grace for keeping a safe same-side order when the
    /// current order remains near the local edge floor but the desired quote is
    /// temporarily absent because edge dipped below minimum.
    pub paradex_edge_under_min_grace_ms: Option<i64>,
    /// Optional Paradex-only allowance for how far below the local edge floor a
    /// safe same-side current order may drift while we preserve queue position.
    pub paradex_edge_under_min_band_usd: Option<f64>,
    /// Optional Hyperliquid-only grace for keeping a safe same-side order when
    /// the current order remains near the local edge floor but the desired quote
    /// is temporarily absent because edge dipped below minimum.
    pub hyperliquid_edge_under_min_grace_ms: Option<i64>,
    /// Optional Hyperliquid-only allowance for how far below the local edge floor
    /// a safe same-side current order may drift while we preserve queue
    /// position.
    pub hyperliquid_edge_under_min_band_usd: Option<f64>,
    /// Optional Extended-only grace for keeping a safe same-side order when the
    /// current order remains near the local edge floor but the desired quote is
    /// temporarily absent because edge dipped below minimum.
    pub extended_edge_under_min_grace_ms: Option<i64>,
    /// Optional Extended-only allowance for how far below the local edge floor a
    /// safe same-side current order may drift while we preserve queue position.
    pub extended_edge_under_min_band_usd: Option<f64>,
    /// Optional Paradex-only extension for preserving supported-replace same-side
    /// visibility across open-order snapshot gaps.
    pub paradex_supported_replace_snapshot_gap_grace_ms: Option<i64>,
    /// Optional per-venue extension for preserving supported-replace same-side
    /// visibility across open-order snapshot gaps.
    pub supported_replace_snapshot_gap_grace_ms_by_venue: BTreeMap<String, i64>,
    /// Optional Paradex-only guard for preserving just-placed MM orders while
    /// awaiting the first accepted/open-order truth from the venue.
    pub paradex_pending_place_grace_ms: Option<i64>,
    /// Optional per-venue guard for preserving just-placed MM orders while
    /// awaiting the first accepted/open-order truth from the venue.
    pub pending_place_grace_ms_by_venue: BTreeMap<String, i64>,
    /// Price tolerance in ticks before triggering order replacement.
    pub price_tol_ticks: f64,
    /// Optional per-venue price tolerance overrides keyed by venue id.
    pub price_tol_ticks_by_venue: BTreeMap<String, f64>,
    /// Size tolerance (relative) before triggering order replacement.
    pub size_tol_rel: f64,
    /// Optional per-venue size tolerance overrides keyed by venue id.
    pub size_tol_rel_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue max local spread in absolute USD before skipping MM quotes.
    pub max_quote_spread_abs_usd_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue max local spread in basis points before skipping MM quotes.
    pub max_quote_spread_bps_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue max generated two-sided quote spread in basis points.
    pub max_generated_quote_spread_bps_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue hard cap on MM quote size in TAO.
    pub max_quote_size_tao_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue economic role overrides keyed by venue id.
    pub venue_role_by_venue: BTreeMap<String, MmVenueRole>,
    /// Optional per-venue global inventory threshold that starts tapering the
    /// inventory-worsening side before the soft governor fully blocks it.
    pub pre_soft_taper_global_position_tao_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue local inventory threshold that starts tapering the
    /// inventory-worsening side before the soft governor fully blocks it.
    pub pre_soft_taper_venue_position_tao_by_venue: BTreeMap<String, f64>,
    /// Optional per-venue size multiplier applied to the worsening side once a
    /// pre-soft taper threshold is breached.
    pub pre_soft_taper_size_multiplier_by_venue: BTreeMap<String, f64>,

    // ----- Funding target inventory (Section 9) -----
    /// Scale for funding rate in target inventory calculation.
    /// phi(funding_8h) = clip(funding_8h / this, -1, 1) * FUNDING_TARGET_MAX_TAO
    pub funding_target_rate_scale: f64,
    /// Maximum TAO shift from funding preference per venue.
    pub funding_target_max_tao: f64,
    /// Maximum venue book age (ms) before skipping MM quotes for that venue.
    /// If None, uses the venue's effective stale_ms threshold.
    /// This is a fail-fast guard to prevent quoting on stale data.
    pub quote_max_age_ms: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MmVenueRole {
    Fill,
    Probationary,
    Anchor,
    Noise,
}

impl MmVenueRole {
    pub fn as_str(self) -> &'static str {
        match self {
            MmVenueRole::Fill => "fill",
            MmVenueRole::Probationary => "probationary",
            MmVenueRole::Anchor => "anchor",
            MmVenueRole::Noise => "noise",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fill" | "fills" | "primary" => Some(MmVenueRole::Fill),
            "probationary" | "probation" | "trial" | "candidate" => Some(MmVenueRole::Probationary),
            "anchor" | "reference" | "ref" => Some(MmVenueRole::Anchor),
            "noise" | "noisy" | "suppress" | "suppressed" => Some(MmVenueRole::Noise),
            _ => None,
        }
    }
}

impl MmConfig {
    #[inline]
    pub fn edge_local_min_bid_for(&self, venue_id: &str) -> f64 {
        self.edge_local_min_bid_by_venue
            .get(venue_id)
            .copied()
            .or_else(|| self.edge_local_min_by_venue.get(venue_id).copied())
            .unwrap_or(self.edge_local_min)
            .max(0.0)
    }

    #[inline]
    pub fn edge_local_min_ask_for(&self, venue_id: &str) -> f64 {
        self.edge_local_min_ask_by_venue
            .get(venue_id)
            .copied()
            .or_else(|| self.edge_local_min_by_venue.get(venue_id).copied())
            .unwrap_or(self.edge_local_min)
            .max(0.0)
    }

    #[inline]
    pub fn min_quote_lifetime_ms_for(&self, venue_id: &str) -> i64 {
        self.min_quote_lifetime_ms_by_venue
            .get(venue_id)
            .copied()
            .unwrap_or(self.min_quote_lifetime_ms)
            .max(1)
    }

    #[inline]
    pub fn post_control_suppression_grace_ms_for(&self, venue_id: &str) -> i64 {
        let base = self.min_quote_lifetime_ms_for(venue_id);
        if venue_id.eq_ignore_ascii_case("paradex") {
            self.paradex_post_control_suppression_grace_ms
                .unwrap_or(base)
                .max(base)
        } else {
            base
        }
    }

    #[inline]
    pub fn price_tol_ticks_for(&self, venue_id: &str) -> f64 {
        self.price_tol_ticks_by_venue
            .get(venue_id)
            .copied()
            .unwrap_or(self.price_tol_ticks)
            .max(0.0)
    }

    #[inline]
    pub fn size_tol_rel_for(&self, venue_id: &str) -> f64 {
        self.size_tol_rel_by_venue
            .get(venue_id)
            .copied()
            .unwrap_or(self.size_tol_rel)
            .max(0.0)
    }

    #[inline]
    pub fn max_quote_spread_abs_usd_for(&self, venue_id: &str) -> Option<f64> {
        self.max_quote_spread_abs_usd_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    #[inline]
    pub fn max_quote_spread_bps_for(&self, venue_id: &str) -> Option<f64> {
        self.max_quote_spread_bps_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    #[inline]
    pub fn max_generated_quote_spread_bps_for(&self, venue_id: &str) -> Option<f64> {
        self.max_generated_quote_spread_bps_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    #[inline]
    pub fn max_quote_size_tao_for(&self, venue_id: &str) -> Option<f64> {
        self.max_quote_size_tao_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    #[inline]
    pub fn venue_role_for(&self, venue_id: &str) -> MmVenueRole {
        self.venue_role_by_venue
            .get(venue_id)
            .copied()
            .unwrap_or(MmVenueRole::Fill)
    }

    #[inline]
    pub fn pre_soft_taper_global_position_tao_for(&self, venue_id: &str) -> Option<f64> {
        self.pre_soft_taper_global_position_tao_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    #[inline]
    pub fn pre_soft_taper_venue_position_tao_for(&self, venue_id: &str) -> Option<f64> {
        self.pre_soft_taper_venue_position_tao_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    #[inline]
    pub fn pre_soft_taper_size_multiplier_for(&self, venue_id: &str) -> Option<f64> {
        self.pre_soft_taper_size_multiplier_by_venue
            .get(venue_id)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
    }

    #[inline]
    pub fn edge_under_min_grace_ms_for(&self, venue_id: &str) -> i64 {
        if venue_id.eq_ignore_ascii_case("paradex") {
            return self
                .paradex_edge_under_min_grace_ms
                .unwrap_or(self.min_quote_lifetime_ms_for(venue_id))
                .max(1);
        }
        if venue_id.eq_ignore_ascii_case("hyperliquid") {
            return self
                .hyperliquid_edge_under_min_grace_ms
                .unwrap_or(self.min_quote_lifetime_ms_for(venue_id))
                .max(1);
        }
        if venue_id.eq_ignore_ascii_case("extended") {
            return self
                .extended_edge_under_min_grace_ms
                .unwrap_or(self.min_quote_lifetime_ms_for(venue_id))
                .max(1);
        }
        self.min_quote_lifetime_ms_for(venue_id)
    }

    #[inline]
    pub fn edge_under_min_band_usd_for(&self, venue_id: &str) -> Option<f64> {
        if venue_id.eq_ignore_ascii_case("paradex") {
            return self
                .paradex_edge_under_min_band_usd
                .filter(|v| v.is_finite() && *v >= 0.0);
        }
        if venue_id.eq_ignore_ascii_case("hyperliquid") {
            return self
                .hyperliquid_edge_under_min_band_usd
                .filter(|v| v.is_finite() && *v >= 0.0);
        }
        if venue_id.eq_ignore_ascii_case("extended") {
            return self
                .extended_edge_under_min_band_usd
                .filter(|v| v.is_finite() && *v >= 0.0);
        }
        None
    }

    #[inline]
    pub fn supported_replace_snapshot_gap_grace_ms_for(&self, venue_id: &str) -> i64 {
        const DEFAULT_SUPPORTED_REPLACE_SNAPSHOT_GAP_GRACE_MS: i64 = 2_000;
        if let Some(value) = self
            .supported_replace_snapshot_gap_grace_ms_by_venue
            .get(venue_id)
        {
            return (*value).max(1);
        }
        if venue_id.eq_ignore_ascii_case("paradex") {
            return self
                .paradex_supported_replace_snapshot_gap_grace_ms
                .unwrap_or(DEFAULT_SUPPORTED_REPLACE_SNAPSHOT_GAP_GRACE_MS)
                .max(1);
        }
        DEFAULT_SUPPORTED_REPLACE_SNAPSHOT_GAP_GRACE_MS
    }

    #[inline]
    pub fn pending_place_grace_ms_for(&self, venue_id: &str) -> i64 {
        if let Some(value) = self.pending_place_grace_ms_by_venue.get(venue_id) {
            return (*value).max(0);
        }
        if venue_id.eq_ignore_ascii_case("paradex") {
            return self.paradex_pending_place_grace_ms.unwrap_or(0).max(0);
        }
        0
    }
}

#[derive(Debug, Clone)]
pub struct HedgeConfig {
    // ----- Deadband + LQ control (Section 13.1) -----
    /// Base half-band in TAO (before vol scaling).
    /// `band_vol = band_base_tao * (1 + band_vol_mult * vol_ratio_clipped)`
    pub band_base_tao: f64,

    /// Volatility multiplier for the deadband.
    pub band_vol_mult: f64,

    /// LQ controller gain k_hedge = alpha / (alpha + beta).
    /// The global hedge step is ΔH_raw = k_hedge * X.
    pub k_hedge: f64,

    /// Max TAO we are allowed to move in one hedge action (global cap).
    pub max_step_tao: f64,

    // ----- Per-venue allocation (Section 13.2) -----
    /// Max TAO per single venue per tick.
    pub max_venue_tao_per_tick: f64,

    /// Fraction of venue depth_near_mid (converted to TAO) we consume per hedge.
    pub depth_fraction: f64,

    /// Minimum depth_near_mid (USD) required to consider a venue for hedging.
    pub min_depth_usd: f64,

    /// Weight for funding benefit in the per-venue cost model.
    /// Positive => prefer venues with favorable funding.
    pub funding_weight: f64,
    /// Enable funding-aware hedge logic (default: false).
    pub funding_enabled: bool,

    /// Weight for basis edge in the per-venue cost model.
    /// Positive => prefer venues where we can capture basis spread.
    pub basis_weight: f64,

    /// Funding horizon (seconds) used for approximating funding benefit.
    pub funding_horizon_sec: f64,

    /// Slippage buffer constant (USD/TAO) added to execution cost.
    pub slippage_buffer: f64,

    /// Guard price multiplier for IOC orders.
    /// Guard price = ask + guard_mult * spread (buy) or bid - guard_mult * spread (sell).
    pub guard_mult: f64,

    /// Fragmentation penalty (USD/TAO) applied when opening a new position on a venue.
    pub frag_penalty: f64,

    /// Liquidation warning sigma threshold (copied from risk config for gating).
    /// Penalize venues as dist_liq_sigma approaches this.
    pub liq_warn_sigma: f64,

    /// Liquidation critical sigma threshold.
    /// Hard-skip venues at or below this.
    pub liq_crit_sigma: f64,

    /// Penalty scale for liquidation proximity (USD/TAO per sigma below warn).
    pub liq_penalty_scale: f64,

    // ----- Margin constraints (Milestone F) -----
    /// Safety buffer for margin-based hedge caps.
    /// Applied as: additional_cap = (margin_available * max_leverage * safety_buffer) / price
    /// Default: 0.95
    pub margin_safety_buffer: f64,

    /// Max leverage assumption for hedge margin calculations.
    /// Default: 10.0
    pub max_leverage: f64,

    // ----- Multi-chunk allocation (Milestone F) -----
    /// Chunk size in TAO for multi-chunk allocation.
    /// If <= 0, a default is computed from max_venue_tao_per_tick / 4.
    /// Default: 0.0 (use computed default)
    pub chunk_size_tao: f64,

    /// Convexity cost per chunk in basis points.
    /// Each subsequent chunk on the same venue adds this to its unit cost.
    /// Enables "spreading" across venues when > 0.
    /// Default: 0.0 (no convexity, preserves existing behavior)
    pub chunk_convexity_cost_bps: f64,

    // ----- Legacy compat (for migration) -----
    /// Legacy: Base half-band in TAO. Alias for band_base_tao.
    pub hedge_band_base: f64,

    /// Legacy: Max step. Alias for max_step_tao.
    pub hedge_max_step: f64,

    /// Legacy: alpha_hedge (unused in new model).
    pub alpha_hedge: f64,

    /// Legacy: beta_hedge (unused in new model).
    pub beta_hedge: f64,
}

#[derive(Debug, Clone)]
pub struct ExitConfig {
    /// Master enable for the exit engine.
    pub enabled: bool,

    /// Max total TAO we are allowed to exit per tick.
    pub max_total_tao_per_tick: f64,

    /// Max TAO per single venue per tick.
    pub max_venue_tao_per_tick: f64,

    /// Do nothing if |q_global| is smaller than this.
    pub min_global_abs_tao: f64,

    /// Do not emit intents smaller than this (dust guard).
    pub min_intent_size_tao: f64,

    /// Minimum profit-only edge per TAO in USD (after fees & buffers).
    pub edge_min_usd: f64,

    /// Volatility buffer term (USD/TAO) scaled by vol_ratio_clipped.
    pub edge_vol_mult: f64,

    /// Scoring weight on basis term (USD/TAO).
    pub basis_weight: f64,

    /// Scoring weight on funding benefit term (USD/TAO).
    pub funding_weight: f64,
    /// Enable funding-aware exit logic (default: false).
    pub funding_enabled: bool,

    /// Funding horizon (seconds) used for approximating funding benefit.
    pub funding_horizon_sec: f64,

    /// Fragmentation penalty proxy (USD/TAO) applied when an exit opens/increases a new leg.
    pub fragmentation_penalty_per_tao: f64,

    /// Fraction of depth_near_mid we allow consuming for exits.
    pub depth_fraction: f64,

    /// Slippage model linear coefficient (USD per TAO).
    /// slippage = linear_coeff * size + quadratic_coeff * size^2
    pub slippage_linear_coeff: f64,

    /// Slippage model quadratic coefficient (USD per TAO^2).
    /// slippage = linear_coeff * size + quadratic_coeff * size^2
    pub slippage_quadratic_coeff: f64,

    /// Legacy slippage model multiplier against (notional/depth)*spread.
    /// Used when depth-based slippage model is desired.
    pub slippage_spread_mult: f64,

    /// Minimum depth_near_mid (USD) required to consider a venue for exits.
    pub min_depth_usd: f64,

    /// Volatility buffer multiplier applied to sigma_eff * fair_value.
    /// vol_buffer = vol_buffer_mult * sigma_eff * fair
    pub vol_buffer_mult: f64,

    /// Basis-risk penalty weight (USD/TAO per unit basis increase).
    /// Applied when an exit would increase |B_t|.
    pub basis_risk_penalty_weight: f64,

    /// Fragmentation reduction bonus (USD/TAO) when exit reduces venue count.
    /// Provides deterministic preference for consolidation when edges are similar.
    pub fragmentation_reduction_bonus: f64,
}

#[derive(Debug, Clone)]
pub struct ToxicityConfig {
    /// Scale for converting local vol ratio to toxicity feature.
    ///
    /// f1 = clip((local_vol / sigma_eff - 1) / vol_tox_scale, 0, 1)
    ///
    /// With the default values below a venue has to run roughly 25–30% hotter
    /// than the global volatility before entering Warning, and ~45–50% hotter
    /// before being Disabled.
    pub vol_tox_scale: f64,
    /// Toxicity threshold between Healthy and Warning.
    pub tox_med_threshold: f64,
    /// Toxicity threshold above which the venue is Disabled.
    pub tox_high_threshold: f64,

    // ----- Markout-based toxicity (v2) -----
    /// Time horizon (ms) after fill when we evaluate the markout.
    /// E.g. 5000 means we check fair/mid 5 seconds after each fill.
    pub markout_horizon_ms: i64,

    /// EWMA alpha for blending instantaneous markout toxicity into the running score.
    /// tox_new = (1 - alpha) * tox_old + alpha * tox_instant
    pub markout_alpha: f64,

    /// Scale for converting adverse markout (USD/TAO) to instantaneous toxicity [0,1].
    /// tox_instant = clamp((-markout) / markout_scale_usd_per_tao, 0, 1)
    pub markout_scale_usd_per_tao: f64,

    /// Maximum number of pending markout evaluations per venue.
    /// Older entries are dropped when this limit is exceeded.
    pub max_pending_per_venue: usize,
    /// Grace period (ms) before depth=0 triggers the toxicity fallback.
    /// This prevents transient empty-side snapshots from immediately disabling a venue.
    /// Set to 0 to restore legacy behavior.
    pub depth_fallback_grace_ms: i64,
    /// Catastrophic staleness threshold (ms). If a venue's last book update
    /// is older than this, force toxicity=1.0 (Disabled) regardless of shadow
    /// mode. Prevents stuck reconnect loops from leaving venues in Healthy state.
    /// Set to 0 to disable. Default: 120_000 (2 minutes).
    pub catastrophic_stale_ms: i64,
}

impl Default for Config {
    fn default() -> Self {
        // ------------------------------------------------------------------
        // World-model tuned "balanced" centre (exp07/exp08):
        //
        //   band_base      = 5.625     TAO
        //   mm_size_eta    = 0.10
        //   vol_ref        = 0.028125
        //   daily_loss_lim = 5000 USD
        //
        // Aggressive / Conservative profiles only change daily_loss_limit
        // on top of this centre; see Config::for_profile.
        // ------------------------------------------------------------------
        const BAND_BASE: f64 = 5.625;
        const MM_SIZE_ETA: f64 = 0.10;
        const VOL_REF: f64 = 0.028_125;
        const DAILY_LOSS_LIMIT_BAL: f64 = 5_000.0;

        // ----- Venue configs -----
        //
        // These are deliberately conservative, assuming ~300 USD / TAO:
        //  - base_order_size = 1 TAO  → ~300 USD notional,
        //  - max_order_size  = 20 TAO → ~6k  USD notional.
        // The risk engine and MM sizing logic will further scale these.
        let venues = vec![
            VenueConfig {
                id: "extended".to_string(),
                id_arc: Arc::from("extended"),
                name: "Extended".to_string(),
                tick_size: 0.01,
                base_order_size: 1.0,
                max_order_size: 20.0,
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
                maker_rebate_bps: 0.0,
                gamma: 0.10,
                k: 1.5,
                w_liq: 0.25,
                w_fund: 0.25,
                is_hedge_allowed: true,
                contributes_to_fv: true,
                fv_max_spread_bps: None,
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
                stale_ms_override: None,
                rate_limit_rps: Some(15.0), // Assumed Binance-compatible ~20 RPS; 75% capacity.
                rate_limit_burst: Some(20),
            },
            VenueConfig {
                id: "hyperliquid".to_string(),
                id_arc: Arc::from("hyperliquid"),
                name: "Hyperliquid".to_string(),
                tick_size: 0.01,
                base_order_size: 1.0,
                max_order_size: 20.0,
                maker_fee_bps: 1.5,
                taker_fee_bps: 5.0,
                maker_rebate_bps: 0.0,
                gamma: 0.09,
                k: 1.6,
                w_liq: 0.25,
                w_fund: 0.25,
                is_hedge_allowed: true,
                contributes_to_fv: true,
                fv_max_spread_bps: None,
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
                // Hyperliquid WebSocket P50 ~1195ms, P95 ~1444ms (obs_5000_ticks_report).
                // Default 1000ms causes 63.9% Stale / 1257 flaps in 83 min.
                stale_ms_override: Some(2_000),
                rate_limit_rps: Some(15.0), // 1200 wt/min; batched orders = wt 1; 75% capacity.
                rate_limit_burst: Some(20),
            },
            VenueConfig {
                id: "aster".to_string(),
                id_arc: Arc::from("aster"),
                name: "Aster".to_string(),
                tick_size: 0.01,
                base_order_size: 1.0,
                max_order_size: 20.0,
                // Aster Pro fee schedule: maker 0.005%, taker 0.04%.
                maker_fee_bps: 0.5,
                taker_fee_bps: 4.0,
                maker_rebate_bps: 0.0,
                gamma: 0.11,
                k: 1.4,
                w_liq: 0.20,
                w_fund: 0.20,
                is_hedge_allowed: true,
                contributes_to_fv: true,
                fv_max_spread_bps: None,
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
                stale_ms_override: None,
                rate_limit_rps: Some(15.0), // 1200 wt/min Binance-style; ~75% capacity.
                rate_limit_burst: Some(20),
            },
            VenueConfig {
                id: "lighter".to_string(),
                id_arc: Arc::from("lighter"),
                name: "Lighter".to_string(),
                tick_size: 0.01,
                base_order_size: 1.0,
                max_order_size: 20.0,
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
                maker_rebate_bps: 0.0,
                gamma: 0.10,
                k: 1.5,
                w_liq: 0.15,
                w_fund: 0.15,
                is_hedge_allowed: true,
                contributes_to_fv: true,
                fv_max_spread_bps: None,
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
                stale_ms_override: None,
                rate_limit_rps: Some(30.0), // 24k wt/min premium; sendTx wt 6; ~45% capacity.
                rate_limit_burst: Some(45),
            },
            VenueConfig {
                id: "paradex".to_string(),
                id_arc: Arc::from("paradex"),
                name: "Paradex".to_string(),
                tick_size: 0.01,
                base_order_size: 1.0,
                max_order_size: 20.0,
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
                maker_rebate_bps: 0.0,
                gamma: 0.10,
                k: 1.5,
                w_liq: 0.15,
                w_fund: 0.15,
                is_hedge_allowed: true,
                contributes_to_fv: true,
                fv_max_spread_bps: None,
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
                // Paradex BBO feed cadence: startup P95 ~1,546ms, steady-state P95 ~250ms.
                // 3,000ms gives ~2x headroom over worst startup case.
                stale_ms_override: Some(3_000),
                rate_limit_rps: Some(50.0), // 800 RPS documented; 50 is ~6% capacity.
                rate_limit_burst: Some(75),
            },
        ];

        // ----- Order book / observation config -----
        let book = BookConfig {
            depth_levels: 10,
            stale_ms: 1_000,
            min_healthy_for_kf: 2,
            max_mid_jump_pct: 0.02,
        };

        // ----- Kalman fair-value config -----
        let kalman = KalmanConfig {
            q_base: 1e-6,
            r_a: 1e-6,
            r_b: 1e2,
            r_min: 1e-8,
            r_max: 1e-2,
            p_init: 1.0,
        };

        // ----- Volatility & control scalars -----
        let volatility = VolatilityConfig {
            fv_vol_alpha_short: 0.2,
            fv_vol_alpha_long: 0.05,
            sigma_min: 0.001,
            // World-model tuned reference vol for vol_ratio.
            vol_ref: VOL_REF,
            vol_ratio_min: 0.25,
            vol_ratio_max: 4.0,
            spread_vol_mult_coeff: 1.0,
            size_vol_mult_coeff: 2.0,
            band_vol_mult_coeff: 1.0,
            // Timescale for vol_ref / sigma_min auto-scaling.
            // vol_ref = 0.028125 ≈ daily ETH vol (44.6% annualized).
            vol_ref_cadence_sec: 86400.0,
            sigma_min_cadence_sec: 86400.0,
        };

        // ----- Global risk config (Section 14) -----
        let risk = RiskConfig {
            // This is the *base* hard delta limit at vol_ratio ≈ 1.
            // The engine scales this by vol_ratio: high vol ⇒ smaller limit.
            delta_hard_limit_usd_base: 100_000.0,
            // Warning regime kicks in once |Δ| exceeds this fraction of limit.
            delta_warn_frac: 0.7,
            // Basis limit is intentionally smaller: we don’t want to run a big
            // basis book while market is volatile.
            basis_hard_limit_usd: 10_000.0,
            basis_warn_frac: 0.7,
            // Daily loss limit (realised + unrealised), as a positive threshold.
            // For Balanced profile we use the world-model tuned centre.
            daily_loss_limit: DAILY_LOSS_LIMIT_BAL,
            pnl_warn_frac: 0.5,
            // In Warning regime we widen spreads and cap sizes.
            spread_warn_mult: 1.5,
            q_warn_cap: 5.0,
            // Use ~50% of available margin, assuming up to 10x leverage for sizing.
            mm_margin_safety: 0.5, // use only half of available margin
            mm_max_leverage: 10.0, // allow up to 10x notionally for MM sizing
            // Start shrinking sizes as we get within 5σ of liq; 0 sizes inside 2σ.
            liq_warn_sigma: 5.0, // start shrinking sizes inside 5σ to liq
            liq_crit_sigma: 2.0, // treat as “too close” inside 2σ
        };

        // ----- Funding policy (staleness + avoid window) -----
        let funding = FundingPolicyConfig {
            stale_ms: 10 * 60 * 1000,   // 10 minutes
            avoid_window_ms: 60 * 1000, // 60 seconds
        };

        // ----- MM (Avellaneda–Stoikov + basis/funding) -----
        let mm = MmConfig {
            // Reservation price adjustment weights. These are deliberately
            // modest so we only lightly lean inventory based on basis/funding.
            basis_weight: 0.3,
            funding_weight: 0.3,
            funding_enabled: false,
            // Local minimum edge in USD, plus a vol-dependent buffer.
            edge_local_min: 0.5,
            edge_local_min_by_venue: BTreeMap::new(),
            edge_local_min_bid_by_venue: BTreeMap::new(),
            edge_local_min_ask_by_venue: BTreeMap::new(),
            edge_vol_mult: 0.2,
            hedge_cost_edge_mult: 0.0,
            // Inventory-risk parameter in J(Q) = eQ - 0.5 η Q².
            // World-model tuned η at the profile centre.
            size_eta: MM_SIZE_ETA,
            // 0 = pure global, 1 = pure per-venue target; we sit in the middle.
            lambda_inv: 0.3,
            // Quoting horizon in seconds used in the AS formulas.
            quote_horizon_sec: 30.0,
            // Funding-driven inventory skew: slope and clip.
            funding_skew_slope: 10_000.0,
            funding_skew_clip: 100.0,

            // Order management (Section 11)
            min_quote_lifetime_ms: 500,
            min_quote_lifetime_ms_by_venue: BTreeMap::new(),
            paradex_post_control_suppression_grace_ms: None,
            paradex_edge_under_min_grace_ms: None,
            paradex_edge_under_min_band_usd: None,
            hyperliquid_edge_under_min_grace_ms: None,
            hyperliquid_edge_under_min_band_usd: None,
            extended_edge_under_min_grace_ms: None,
            extended_edge_under_min_band_usd: None,
            paradex_supported_replace_snapshot_gap_grace_ms: None,
            supported_replace_snapshot_gap_grace_ms_by_venue: BTreeMap::new(),
            paradex_pending_place_grace_ms: None,
            pending_place_grace_ms_by_venue: BTreeMap::new(),
            price_tol_ticks: 1.0,
            price_tol_ticks_by_venue: BTreeMap::new(),
            size_tol_rel: 0.10,
            size_tol_rel_by_venue: BTreeMap::new(),
            max_quote_spread_abs_usd_by_venue: BTreeMap::new(),
            max_quote_spread_bps_by_venue: BTreeMap::new(),
            max_generated_quote_spread_bps_by_venue: BTreeMap::new(),
            max_quote_size_tao_by_venue: BTreeMap::new(),
            venue_role_by_venue: BTreeMap::new(),
            pre_soft_taper_global_position_tao_by_venue: BTreeMap::new(),
            pre_soft_taper_venue_position_tao_by_venue: BTreeMap::new(),
            pre_soft_taper_size_multiplier_by_venue: BTreeMap::new(),

            // Funding target inventory (Section 9)
            funding_target_rate_scale: 0.001, // 0.1% funding rate = full shift
            funding_target_max_tao: 5.0,      // max TAO shift from funding preference
            // Quote staleness guard (fail-fast)
            quote_max_age_ms: None, // None = use venue's effective stale_ms
        };

        // ----- Hedge engine (global LQ controller + allocation, Section 13) -----
        let hedge = HedgeConfig {
            // Deadband + LQ control (Section 13.1)
            // With ~300 USD / TAO and our default limits, this band corresponds
            // to ~6–9k USD of unhedged delta before the LQ controller kicks in.
            band_base_tao: BAND_BASE, // TAO band (balanced centre)
            band_vol_mult: 1.0,       // volatility scaling for the band

            // k_hedge = alpha / (alpha + beta); with alpha=beta=1 => k=0.5
            // Using k=0.5 as default for smoother hedging
            k_hedge: 0.5,
            max_step_tao: 20.0, // TAO per hedge step (global cap)

            // Per-venue allocation (Section 13.2)
            max_venue_tao_per_tick: 10.0,
            depth_fraction: 0.10,
            min_depth_usd: 500.0,
            funding_weight: 0.20,
            funding_enabled: false,
            basis_weight: 0.20,
            funding_horizon_sec: 30.0,
            slippage_buffer: 0.05, // USD/TAO
            guard_mult: 0.5,       // half-spread for guard price
            frag_penalty: 0.02,    // USD/TAO penalty for opening new leg

            // Liquidation-aware gating
            liq_warn_sigma: 5.0,     // start penalizing inside 5σ
            liq_crit_sigma: 2.0,     // hard-skip inside 2σ
            liq_penalty_scale: 0.10, // USD/TAO per sigma below warn

            // Margin constraints (Milestone F)
            margin_safety_buffer: 0.95, // use 95% of available margin headroom
            max_leverage: 10.0,         // max leverage for margin calculations

            // Multi-chunk allocation (Milestone F)
            chunk_size_tao: 0.0, // 0 = use default (max_venue_tao_per_tick / 4)
            chunk_convexity_cost_bps: 0.0, // 0 = no convexity, preserves existing behavior

            // Legacy aliases (for backwards compat)
            hedge_band_base: BAND_BASE,
            hedge_max_step: 20.0,
            alpha_hedge: 1.0,
            beta_hedge: 1.0,
        };

        // ----- Exit engine (cross-venue profit-only exits) -----
        let exit = ExitConfig {
            enabled: true,
            max_total_tao_per_tick: 10.0,
            max_venue_tao_per_tick: 6.0,
            min_global_abs_tao: 0.25,
            min_intent_size_tao: 0.01,
            edge_min_usd: 0.25,
            edge_vol_mult: 0.10,
            basis_weight: 0.20,
            funding_weight: 0.20,
            funding_enabled: false,
            funding_horizon_sec: 30.0,
            fragmentation_penalty_per_tao: 0.05,
            depth_fraction: 0.10,
            // Linear + quadratic slippage model coefficients
            slippage_linear_coeff: 0.01,     // USD per TAO
            slippage_quadratic_coeff: 0.001, // USD per TAO^2
            slippage_spread_mult: 1.00,      // legacy spread-based model
            min_depth_usd: 500.0,
            // Volatility buffer: vol_buffer = vol_buffer_mult * sigma_eff * fair
            vol_buffer_mult: 0.5,
            // Basis-risk penalty weight
            basis_risk_penalty_weight: 0.10,
            // Fragmentation reduction bonus (deterministic tie-break preference)
            fragmentation_reduction_bonus: 0.02,
        };

        // ----- Toxicity / venue health -----
        //
        // f1 = clip((local_vol / sigma_eff - 1) / vol_tox_scale, 0, 1)
        //
        // With these defaults we need a venue to run roughly 25–30% hotter
        // than the global volatility before entering Warning, and ~45–50%
        // hotter before being Disabled.
        //
        // Markout-based toxicity (v2):
        // - After each fill, we schedule an evaluation at t + markout_horizon_ms.
        // - At evaluation time, markout = (mid_now - fill_price) for buys,
        //   or (fill_price - mid_now) for sells.
        // - Adverse markout (negative) increases toxicity via EWMA.
        let toxicity = ToxicityConfig {
            vol_tox_scale: 0.5,
            tox_med_threshold: 0.6, // Warning only when volatility is clearly elevated
            tox_high_threshold: 0.9, // Disabled only when it is much higher still

            // Markout-based toxicity v2 defaults
            markout_horizon_ms: 5_000,      // evaluate 5s after fill
            markout_alpha: 0.1,             // EWMA blend factor
            markout_scale_usd_per_tao: 2.0, // $2 adverse markout → tox_instant = 1.0
            max_pending_per_venue: 100,     // bounded queue size
            depth_fallback_grace_ms: 500,   // tolerate brief empty-side snapshots
            catastrophic_stale_ms: 120_000, // 2 minutes — force Disabled for stuck reconnects
        };

        Config {
            version: "v0.1.6-worldmodel-exp08-presets",
            initial_q_tao: 0.0,
            venues,
            book,
            fill_agg_interval_ms: 1_000,
            main_loop_interval_ms: 250,
            hedge_loop_interval_ms: 500,
            risk_loop_interval_ms: 3_000,
            kalman,
            volatility,
            risk,
            funding,
            mm,
            hedge,
            exit,
            toxicity,
        }
    }
}

// --- Runtime config loader: profiles + env overrides -------------------------

impl Config {
    /// Build a Config using a given risk profile on top of the
    /// whitepaper-spec defaults.
    ///
    /// We treat `Config::default()` as the *Balanced* world-model centre.
    /// Other profiles only adjust a small set of knobs on top.
    pub fn for_profile(profile: RiskProfile) -> Self {
        let mut cfg = Config::default();

        // We treat Config::default() as the world-model "balanced" centre.
        // Profiles below are tuned using the Exp07/10/11/12 research pipeline.

        match profile {
            RiskProfile::Balanced => {
                // World-model tuned "balanced" centre (exp08).
                cfg.initial_q_tao = 0.0;
                cfg.hedge.band_base_tao = 5.625;
                cfg.hedge.hedge_band_base = 5.625; // legacy alias
                cfg.mm.size_eta = 0.10;
                cfg.volatility.vol_ref = 0.028_125;
                // Keep existing loss limit for now; empirical drawdown is safe.
                cfg.risk.daily_loss_limit = 5_000.0;
            }

            RiskProfile::Conservative => {
                // Conservative profile:
                // - Same world-model as Balanced
                // - Risk-scaled down using Exp12 (risk_scale ≈ 0.466)
                cfg.initial_q_tao = 0.0;

                // Tighten hedge band and MM risk parameter.
                cfg.hedge.band_base_tao = 2.621_25; // ≈ 5.625 * 0.466
                cfg.hedge.hedge_band_base = 2.621_25; // legacy alias
                cfg.mm.size_eta = 0.046_6; // ≈ 0.10  * 0.466
                cfg.volatility.vol_ref = 0.028_125;

                // Shrink delta & PnL limits proportionally.
                cfg.risk.delta_hard_limit_usd_base = 46_600.0; // ≈ 100k * 0.466
                cfg.risk.daily_loss_limit = 2_000.0; // matches Exp12 tuned preset
            }

            RiskProfile::Aggressive => {
                // Aggressive profile:
                // - Same world-model centre as Balanced
                // - Risk-scaled down using Exp12 (risk_scale ≈ 0.864)
                cfg.initial_q_tao = 0.0;

                // Slightly tighter than centre to bring empirical dd back under 8k.
                cfg.hedge.band_base_tao = 4.86; // ≈ 5.625 * 0.864
                cfg.hedge.hedge_band_base = 4.86; // legacy alias
                cfg.mm.size_eta = 0.086_4; // ≈ 0.10  * 0.864
                cfg.volatility.vol_ref = 0.028_125;

                // Scale delta limit proportionally.
                cfg.risk.delta_hard_limit_usd_base = 86_400.0; // ≈ 100k * 0.864

                // Keep loss limit as-is for now; world-model dd budget is 8k.
                cfg.risk.daily_loss_limit = 2_000.0;
            }
        }

        cfg
    }

    /// Build a Config from a profile, then apply environment overrides.
    ///
    /// This is designed for research / batch runs and future RL:
    ///
    ///   - PARAPHINA_INIT_Q_TAO        (f64, TAO)
    ///   - PARAPHINA_HEDGE_BAND_BASE   (f64, TAO)
    ///   - PARAPHINA_HEDGE_MAX_STEP    (f64, TAO)
    ///   - PARAPHINA_HEDGE_MIN_DEPTH_USD (f64, USD)
    ///   - PARAPHINA_HEDGE_DISABLED_VENUES (csv venue ids)
    ///   - PARAPHINA_FV_DISABLED_VENUES (csv venue ids)
    ///   - PARAPHINA_MM_SIZE_ETA       (f64)
    ///   - PARAPHINA_MM_EDGE_LOCAL_MIN (f64, USD per unit)
    ///   - PARAPHINA_MM_EDGE_LOCAL_MIN_<VENUE> (f64, USD per unit)
    ///   - PARAPHINA_MM_EDGE_LOCAL_MIN_<VENUE>_BID (f64, USD per unit)
    ///   - PARAPHINA_MM_EDGE_LOCAL_MIN_<VENUE>_ASK (f64, USD per unit)
    ///   - PARAPHINA_MM_HEDGE_COST_EDGE_MULT (f64, multiplier)
    ///   - PARAPHINA_MM_LAMBDA_INV     (f64, [0, 1])
    ///   - PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS (i64, ms)
    ///   - PARAPHINA_MM_PRICE_TOL_TICKS (f64, ticks)
    ///   - PARAPHINA_MM_SIZE_TOL_REL   (f64, relative size delta)
    ///   - PARAPHINA_MM_MAX_QUOTE_SPREAD_USD_<VENUE> (f64, USD)
    ///   - PARAPHINA_MM_MAX_QUOTE_SPREAD_BPS_<VENUE> (f64, bps)
    ///   - PARAPHINA_MM_MAX_GENERATED_SPREAD_BPS_<VENUE> (f64, bps)
    ///   - PARAPHINA_MM_VENUE_ROLE_<VENUE> (fill|probationary|anchor|noise)
    ///   - PARAPHINA_VOL_REF           (f64)
    ///   - PARAPHINA_DAILY_LOSS_LIMIT  (f64, USD; positive threshold)
    ///   - PARAPHINA_MAIN_LOOP_INTERVAL_MS  (i64, ms)
    ///   - PARAPHINA_HEDGE_LOOP_INTERVAL_MS (i64, ms)
    ///   - PARAPHINA_RISK_LOOP_INTERVAL_MS  (i64, ms)
    ///
    /// Any variable that fails to parse is ignored with a warning.
    pub fn from_env_or_profile(profile: RiskProfile) -> Self {
        use std::env;

        let mut cfg = Config::for_profile(profile);

        // Initial global inventory q0 in TAO.
        if let Ok(raw) = env::var("PARAPHINA_INIT_Q_TAO") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.initial_q_tao = v;
                    eprintln!("[config] PARAPHINA_INIT_Q_TAO = {v} (overrode default)");
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_INIT_Q_TAO = {:?} as f64; using default {}",
                        raw,
                        cfg.initial_q_tao
                    );
                }
            }
        }

        // Hedge band base (in TAO).
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_BAND_BASE") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.band_base_tao = v.max(0.0);
                    cfg.hedge.hedge_band_base = v.max(0.0); // legacy alias
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_BAND_BASE = {} (overrode default)",
                        cfg.hedge.band_base_tao
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_BAND_BASE = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.band_base_tao
                    );
                }
            }
        }

        // Hedge max step (in TAO).
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_MAX_STEP") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.max_step_tao = v.max(0.0);
                    cfg.hedge.hedge_max_step = v.max(0.0); // legacy alias
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_MAX_STEP = {} (overrode default)",
                        cfg.hedge.max_step_tao
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_MAX_STEP = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.max_step_tao
                    );
                }
            }
        }

        // Hedge minimum depth requirement (USD).
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_MIN_DEPTH_USD") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.min_depth_usd = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_MIN_DEPTH_USD = {} (overrode default)",
                        cfg.hedge.min_depth_usd
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_MIN_DEPTH_USD = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.min_depth_usd
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_HEDGE_DISABLED_VENUES") {
            let disabled: Vec<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_ascii_lowercase())
                .collect();
            if disabled.is_empty() {
                eprintln!(
                    "[config] WARN: PARAPHINA_HEDGE_DISABLED_VENUES was set but empty; leaving hedge venue defaults unchanged"
                );
            } else {
                for venue in &mut cfg.venues {
                    if disabled.iter().any(|id| id == &venue.id) {
                        venue.is_hedge_allowed = false;
                    }
                }
                eprintln!(
                    "[config] PARAPHINA_HEDGE_DISABLED_VENUES = {:?} (disabled hedging on matching venues)",
                    disabled
                );
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_FV_DISABLED_VENUES") {
            let disabled: Vec<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_ascii_lowercase())
                .collect();
            if disabled.is_empty() {
                eprintln!(
                    "[config] WARN: PARAPHINA_FV_DISABLED_VENUES was set but empty; leaving fair-value venue defaults unchanged"
                );
            } else {
                for venue in &mut cfg.venues {
                    if disabled.iter().any(|id| id == &venue.id) {
                        venue.contributes_to_fv = false;
                    }
                }
                eprintln!(
                    "[config] PARAPHINA_FV_DISABLED_VENUES = {:?} (disabled fair-value contribution on matching venues)",
                    disabled
                );
            }
        }

        // Hedge k_hedge (LQ controller gain).
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_K_HEDGE") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.k_hedge = v.clamp(0.0, 1.0);
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_K_HEDGE = {} (overrode default)",
                        cfg.hedge.k_hedge
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_K_HEDGE = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.k_hedge
                    );
                }
            }
        }

        // Hedge funding weight.
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_FUNDING_WEIGHT") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.funding_weight = v;
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_FUNDING_WEIGHT = {} (overrode default)",
                        cfg.hedge.funding_weight
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_FUNDING_WEIGHT = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.funding_weight
                    );
                }
            }
        }

        // Hedge basis weight.
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_BASIS_WEIGHT") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.basis_weight = v;
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_BASIS_WEIGHT = {} (overrode default)",
                        cfg.hedge.basis_weight
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_BASIS_WEIGHT = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.basis_weight
                    );
                }
            }
        }

        // MM size_eta (risk parameter in J(Q) = eQ - 0.5 η Q²).
        if let Ok(raw) = env::var("PARAPHINA_MM_SIZE_ETA") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    // Keep it strictly positive to avoid degeneracy.
                    cfg.mm.size_eta = v.max(1e-6);
                    eprintln!(
                        "[config] PARAPHINA_MM_SIZE_ETA = {} (overrode default)",
                        cfg.mm.size_eta
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_SIZE_ETA = {:?} as f64; using default {}",
                        raw,
                        cfg.mm.size_eta
                    );
                }
            }
        }

        // MM local edge floor in USD per unit.
        if let Ok(raw) = env::var("PARAPHINA_MM_EDGE_LOCAL_MIN") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.mm.edge_local_min = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_MM_EDGE_LOCAL_MIN = {} (overrode default)",
                        cfg.mm.edge_local_min
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_EDGE_LOCAL_MIN = {:?} as f64; using default {}",
                        raw,
                        cfg.mm.edge_local_min
                    );
                }
            }
        }

        // Optional hedge/forced-unwind cost pass-through into MM edge floors.
        if let Ok(raw) = env::var("PARAPHINA_MM_HEDGE_COST_EDGE_MULT") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.mm.hedge_cost_edge_mult = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_MM_HEDGE_COST_EDGE_MULT = {} (added hedge-cost edge floor multiplier)",
                        cfg.mm.hedge_cost_edge_mult
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_HEDGE_COST_EDGE_MULT = {:?} as f64; using default {}",
                        raw,
                        cfg.mm.hedge_cost_edge_mult
                    );
                }
            }
        }

        for venue in &cfg.venues {
            let venue_key = venue.id.to_ascii_uppercase();
            let venue_env = format!("PARAPHINA_MM_EDGE_LOCAL_MIN_{venue_key}");
            if let Ok(raw) = env::var(&venue_env) {
                match raw.parse::<f64>() {
                    Ok(v) => {
                        cfg.mm
                            .edge_local_min_by_venue
                            .insert(venue.id.clone(), v.max(0.0));
                        eprintln!(
                            "[config] {} = {} (overrode MM edge floor for {})",
                            venue_env,
                            cfg.mm
                                .edge_local_min_by_venue
                                .get(&venue.id)
                                .copied()
                                .unwrap_or(cfg.mm.edge_local_min),
                            venue.id
                        );
                    }
                    Err(_) => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as f64; leaving {} on default/parent edge floor",
                            venue_env,
                            raw,
                            venue.id
                        );
                    }
                }
            }

            for (side_suffix, side_name, target_map) in [
                ("BID", "bid", &mut cfg.mm.edge_local_min_bid_by_venue),
                ("ASK", "ask", &mut cfg.mm.edge_local_min_ask_by_venue),
            ] {
                let env_key = format!("PARAPHINA_MM_EDGE_LOCAL_MIN_{venue_key}_{side_suffix}");
                if let Ok(raw) = env::var(&env_key) {
                    match raw.parse::<f64>() {
                        Ok(v) => {
                            target_map.insert(venue.id.clone(), v.max(0.0));
                            eprintln!(
                                "[config] {} = {} (overrode MM {} edge floor for {})",
                                env_key,
                                target_map
                                    .get(&venue.id)
                                    .copied()
                                    .unwrap_or(cfg.mm.edge_local_min),
                                side_name,
                                venue.id
                            );
                        }
                        Err(_) => {
                            eprintln!(
                                "[config] WARN: could not parse {} = {:?} as f64; leaving {} {} edge floor on default/parent value",
                                env_key,
                                raw,
                                venue.id,
                                side_name
                            );
                        }
                    }
                }
            }
        }

        // MM inventory skew balance between global and venue-local inventory.
        if let Ok(raw) = env::var("PARAPHINA_MM_LAMBDA_INV") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.mm.lambda_inv = v.clamp(0.0, 1.0);
                    eprintln!(
                        "[config] PARAPHINA_MM_LAMBDA_INV = {} (overrode default)",
                        cfg.mm.lambda_inv
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_LAMBDA_INV = {:?} as f64; using default {}",
                        raw,
                        cfg.mm.lambda_inv
                    );
                }
            }
        }

        // MM order persistence before requoting passive orders.
        if let Ok(raw) = env::var("PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.mm.min_quote_lifetime_ms = v.max(1);
                    eprintln!(
                        "[config] PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS = {} (overrode default)",
                        cfg.mm.min_quote_lifetime_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS = {:?} as i64; using default {}",
                        raw,
                        cfg.mm.min_quote_lifetime_ms
                    );
                }
            }
        }

        // MM cancel/replace price tolerance in ticks.
        if let Ok(raw) = env::var("PARAPHINA_MM_PRICE_TOL_TICKS") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.mm.price_tol_ticks = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_MM_PRICE_TOL_TICKS = {} (overrode default)",
                        cfg.mm.price_tol_ticks
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_PRICE_TOL_TICKS = {:?} as f64; using default {}",
                        raw,
                        cfg.mm.price_tol_ticks
                    );
                }
            }
        }

        // MM cancel/replace size tolerance (relative).
        if let Ok(raw) = env::var("PARAPHINA_MM_SIZE_TOL_REL") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.mm.size_tol_rel = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_MM_SIZE_TOL_REL = {} (overrode default)",
                        cfg.mm.size_tol_rel
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MM_SIZE_TOL_REL = {:?} as f64; using default {}",
                        raw,
                        cfg.mm.size_tol_rel
                    );
                }
            }
        }

        for venue in &cfg.venues {
            let venue_key = venue.id.to_ascii_uppercase();

            let lifetime_key = format!("PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_{venue_key}");
            if let Ok(raw) = env::var(&lifetime_key) {
                match raw.parse::<i64>() {
                    Ok(v) => {
                        cfg.mm
                            .min_quote_lifetime_ms_by_venue
                            .insert(venue.id.clone(), v.max(1));
                        eprintln!(
                            "[config] {} = {} (overrode quote lifetime for {})",
                            lifetime_key,
                            cfg.mm
                                .min_quote_lifetime_ms_by_venue
                                .get(&venue.id)
                                .copied()
                                .unwrap_or(cfg.mm.min_quote_lifetime_ms),
                            venue.id
                        );
                    }
                    Err(_) => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as i64; leaving {} quote lifetime on default/global value",
                            lifetime_key,
                            raw,
                            venue.id
                        );
                    }
                }
            }

            let price_tol_key = format!("PARAPHINA_MM_PRICE_TOL_TICKS_{venue_key}");
            if let Ok(raw) = env::var(&price_tol_key) {
                match raw.parse::<f64>() {
                    Ok(v) => {
                        cfg.mm
                            .price_tol_ticks_by_venue
                            .insert(venue.id.clone(), v.max(0.0));
                        eprintln!(
                            "[config] {} = {} (overrode price tolerance for {})",
                            price_tol_key,
                            cfg.mm
                                .price_tol_ticks_by_venue
                                .get(&venue.id)
                                .copied()
                                .unwrap_or(cfg.mm.price_tol_ticks),
                            venue.id
                        );
                    }
                    Err(_) => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as f64; leaving {} price tolerance on default/global value",
                            price_tol_key,
                            raw,
                            venue.id
                        );
                    }
                }
            }

            let size_tol_key = format!("PARAPHINA_MM_SIZE_TOL_REL_{venue_key}");
            if let Ok(raw) = env::var(&size_tol_key) {
                match raw.parse::<f64>() {
                    Ok(v) => {
                        cfg.mm
                            .size_tol_rel_by_venue
                            .insert(venue.id.clone(), v.max(0.0));
                        eprintln!(
                            "[config] {} = {} (overrode size tolerance for {})",
                            size_tol_key,
                            cfg.mm
                                .size_tol_rel_by_venue
                                .get(&venue.id)
                                .copied()
                                .unwrap_or(cfg.mm.size_tol_rel),
                            venue.id
                        );
                    }
                    Err(_) => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as f64; leaving {} size tolerance on default/global value",
                            size_tol_key,
                            raw,
                            venue.id
                        );
                    }
                }
            }

            let spread_abs_key = format!("PARAPHINA_MM_MAX_QUOTE_SPREAD_USD_{venue_key}");
            if let Ok(raw) = env::var(&spread_abs_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        cfg.mm
                            .max_quote_spread_abs_usd_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set quote spread USD cap for {})",
                            spread_abs_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} spread USD cap disabled",
                            spread_abs_key, raw, venue.id
                        );
                    }
                }
            }

            let spread_bps_key = format!("PARAPHINA_MM_MAX_QUOTE_SPREAD_BPS_{venue_key}");
            if let Ok(raw) = env::var(&spread_bps_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        cfg.mm
                            .max_quote_spread_bps_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set quote spread bps cap for {})",
                            spread_bps_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} spread bps cap disabled",
                            spread_bps_key, raw, venue.id
                        );
                    }
                }
            }

            let generated_spread_bps_key =
                format!("PARAPHINA_MM_MAX_GENERATED_SPREAD_BPS_{venue_key}");
            if let Ok(raw) = env::var(&generated_spread_bps_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        cfg.mm
                            .max_generated_quote_spread_bps_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set generated quote spread bps cap for {})",
                            generated_spread_bps_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} generated quote spread cap disabled",
                            generated_spread_bps_key, raw, venue.id
                        );
                    }
                }
            }

            let max_quote_size_key = format!("PARAPHINA_MM_MAX_QUOTE_SIZE_TAO_{venue_key}");
            if let Ok(raw) = env::var(&max_quote_size_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        cfg.mm
                            .max_quote_size_tao_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set MM max quote size tao for {})",
                            max_quote_size_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} max quote size cap disabled",
                            max_quote_size_key, raw, venue.id
                        );
                    }
                }
            }

            let venue_role_key = format!("PARAPHINA_MM_VENUE_ROLE_{venue_key}");
            if let Ok(raw) = env::var(&venue_role_key) {
                match MmVenueRole::parse(&raw) {
                    Some(role) => {
                        cfg.mm.venue_role_by_venue.insert(venue.id.clone(), role);
                        eprintln!(
                            "[config] {} = {} (set MM venue role for {})",
                            venue_role_key,
                            role.as_str(),
                            venue.id
                        );
                    }
                    None => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as fill|probationary|anchor|noise; leaving {} on default fill role",
                            venue_role_key,
                            raw,
                            venue.id
                        );
                    }
                }
            }

            let taper_global_key =
                format!("PARAPHINA_MM_PRE_SOFT_TAPER_GLOBAL_POS_TAO_{venue_key}");
            if let Ok(raw) = env::var(&taper_global_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        cfg.mm
                            .pre_soft_taper_global_position_tao_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set MM pre-soft global taper threshold for {})",
                            taper_global_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} global pre-soft taper disabled",
                            taper_global_key, raw, venue.id
                        );
                    }
                }
            }

            let taper_venue_key = format!("PARAPHINA_MM_PRE_SOFT_TAPER_VENUE_POS_TAO_{venue_key}");
            if let Ok(raw) = env::var(&taper_venue_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        cfg.mm
                            .pre_soft_taper_venue_position_tao_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set MM pre-soft venue taper threshold for {})",
                            taper_venue_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} venue pre-soft taper disabled",
                            taper_venue_key, raw, venue.id
                        );
                    }
                }
            }

            let taper_mult_key = format!("PARAPHINA_MM_PRE_SOFT_TAPER_SIZE_MULT_{venue_key}");
            if let Ok(raw) = env::var(&taper_mult_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 && v < 1.0 => {
                        cfg.mm
                            .pre_soft_taper_size_multiplier_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set MM pre-soft taper size multiplier for {})",
                            taper_mult_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as 0<f64<1; leaving {} pre-soft taper size multiplier disabled",
                            taper_mult_key, raw, venue.id
                        );
                    }
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let base = cfg.mm.min_quote_lifetime_ms_for("paradex");
                    let grace_ms = v.max(base);
                    cfg.mm.paradex_post_control_suppression_grace_ms = Some(grace_ms);
                    eprintln!(
                        "[config] PARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS = {} (set Paradex post-control suppression grace)",
                        grace_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS = {:?} as i64; leaving Paradex post-control suppression grace unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_PARADEX_EDGE_UNDER_MIN_GRACE_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let base = cfg.mm.min_quote_lifetime_ms_for("paradex");
                    let grace_ms = v.max(base);
                    cfg.mm.paradex_edge_under_min_grace_ms = Some(grace_ms);
                    eprintln!(
                        "[config] PARAPHINA_PARADEX_EDGE_UNDER_MIN_GRACE_MS = {} (set Paradex edge-under-min queue-hold grace)",
                        grace_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_PARADEX_EDGE_UNDER_MIN_GRACE_MS = {:?} as i64; leaving Paradex edge-under-min queue-hold grace unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_PARADEX_EDGE_UNDER_MIN_BAND_USD") {
            match raw.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= 0.0 => {
                    cfg.mm.paradex_edge_under_min_band_usd = Some(v);
                    eprintln!(
                        "[config] PARAPHINA_PARADEX_EDGE_UNDER_MIN_BAND_USD = {} (set Paradex edge-under-min queue-hold band)",
                        v
                    );
                }
                _ => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_PARADEX_EDGE_UNDER_MIN_BAND_USD = {:?} as non-negative f64; leaving Paradex edge-under-min queue-hold band unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_GRACE_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let base = cfg.mm.min_quote_lifetime_ms_for("hyperliquid");
                    let grace_ms = v.max(base);
                    cfg.mm.hyperliquid_edge_under_min_grace_ms = Some(grace_ms);
                    eprintln!(
                        "[config] PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_GRACE_MS = {} (set Hyperliquid edge-under-min queue-hold grace)",
                        grace_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_GRACE_MS = {:?} as i64; leaving Hyperliquid edge-under-min queue-hold grace unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_BAND_USD") {
            match raw.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= 0.0 => {
                    cfg.mm.hyperliquid_edge_under_min_band_usd = Some(v);
                    eprintln!(
                        "[config] PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_BAND_USD = {} (set Hyperliquid edge-under-min queue-hold band)",
                        v
                    );
                }
                _ => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_BAND_USD = {:?} as non-negative f64; leaving Hyperliquid edge-under-min queue-hold band unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_EXTENDED_EDGE_UNDER_MIN_GRACE_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let base = cfg.mm.min_quote_lifetime_ms_for("extended");
                    let grace_ms = v.max(base);
                    cfg.mm.extended_edge_under_min_grace_ms = Some(grace_ms);
                    eprintln!(
                        "[config] PARAPHINA_EXTENDED_EDGE_UNDER_MIN_GRACE_MS = {} (set Extended edge-under-min queue-hold grace)",
                        grace_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_EXTENDED_EDGE_UNDER_MIN_GRACE_MS = {:?} as i64; leaving Extended edge-under-min queue-hold grace unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_EXTENDED_EDGE_UNDER_MIN_BAND_USD") {
            match raw.parse::<f64>() {
                Ok(v) if v.is_finite() && v >= 0.0 => {
                    cfg.mm.extended_edge_under_min_band_usd = Some(v);
                    eprintln!(
                        "[config] PARAPHINA_EXTENDED_EDGE_UNDER_MIN_BAND_USD = {} (set Extended edge-under-min queue-hold band)",
                        v
                    );
                }
                _ => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_EXTENDED_EDGE_UNDER_MIN_BAND_USD = {:?} as non-negative f64; leaving Extended edge-under-min queue-hold band unset",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_PARADEX_SUPPORTED_REPLACE_GAP_GRACE_MS") {
            match raw.parse::<i64>() {
                Ok(v) if v > 0 => {
                    cfg.mm.paradex_supported_replace_snapshot_gap_grace_ms = Some(v);
                    eprintln!(
                        "[config] PARAPHINA_PARADEX_SUPPORTED_REPLACE_GAP_GRACE_MS = {} (set Paradex supported-replace snapshot-gap grace)",
                        v
                    );
                }
                _ => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_PARADEX_SUPPORTED_REPLACE_GAP_GRACE_MS = {:?} as positive i64; leaving Paradex supported-replace snapshot-gap grace at default",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_PARADEX_PENDING_PLACE_GRACE_MS") {
            match raw.parse::<i64>() {
                Ok(v) if v > 0 => {
                    cfg.mm.paradex_pending_place_grace_ms = Some(v);
                    eprintln!(
                        "[config] PARAPHINA_PARADEX_PENDING_PLACE_GRACE_MS = {} (set Paradex pending-place self-cancel guard)",
                        v
                    );
                }
                _ => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_PARADEX_PENDING_PLACE_GRACE_MS = {:?} as positive i64; leaving Paradex pending-place self-cancel guard disabled",
                        raw
                    );
                }
            }
        }

        for venue in &mut cfg.venues {
            let venue_key = venue.id.to_ascii_uppercase();
            let supported_replace_gap_grace_key =
                format!("PARAPHINA_MM_SUPPORTED_REPLACE_GAP_GRACE_MS_{venue_key}");
            if let Ok(raw) = env::var(&supported_replace_gap_grace_key) {
                match raw.parse::<i64>() {
                    Ok(v) if v > 0 => {
                        cfg.mm
                            .supported_replace_snapshot_gap_grace_ms_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set supported-replace snapshot-gap grace for {})",
                            supported_replace_gap_grace_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive i64; leaving {} supported-replace snapshot-gap grace at default",
                            supported_replace_gap_grace_key, raw, venue.id
                        );
                    }
                }
            }

            let pending_place_grace_key =
                format!("PARAPHINA_MM_PENDING_PLACE_GRACE_MS_{venue_key}");
            if let Ok(raw) = env::var(&pending_place_grace_key) {
                match raw.parse::<i64>() {
                    Ok(v) if v > 0 => {
                        cfg.mm
                            .pending_place_grace_ms_by_venue
                            .insert(venue.id.clone(), v);
                        eprintln!(
                            "[config] {} = {} (set pending-place guard for {})",
                            pending_place_grace_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive i64; leaving {} pending-place guard disabled",
                            pending_place_grace_key, raw, venue.id
                        );
                    }
                }
            }

            let fv_max_spread_bps_key = format!("PARAPHINA_FV_MAX_SPREAD_BPS_{venue_key}");
            if let Ok(raw) = env::var(&fv_max_spread_bps_key) {
                match raw.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 => {
                        venue.fv_max_spread_bps = Some(v);
                        eprintln!(
                            "[config] {} = {} (set fair-value max spread gate for {})",
                            fv_max_spread_bps_key, v, venue.id
                        );
                    }
                    _ => {
                        eprintln!(
                            "[config] WARN: could not parse {} = {:?} as positive f64; leaving {} fair-value spread gate disabled",
                            fv_max_spread_bps_key, raw, venue.id
                        );
                    }
                }
            }
        }

        // Volatility reference σ_ref used for vol_ratio = sigma_eff / σ_ref.
        if let Ok(raw) = env::var("PARAPHINA_VOL_REF") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.volatility.vol_ref = v.max(1e-6);
                    eprintln!(
                        "[config] PARAPHINA_VOL_REF = {} (overrode default)",
                        cfg.volatility.vol_ref
                    );
                }
                Err(_) => {
                    eprintln!("[config] WARN: could not parse PARAPHINA_VOL_REF = {:?} as f64; using default {}",
                        raw,
                        cfg.volatility.vol_ref
                    );
                }
            }
        }

        // Timescale (seconds) that vol_ref was calibrated for.
        if let Ok(raw) = env::var("PARAPHINA_VOL_REF_CADENCE_SEC") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.volatility.vol_ref_cadence_sec = v.max(1e-3);
                    eprintln!(
                        "[config] PARAPHINA_VOL_REF_CADENCE_SEC = {} (overrode default)",
                        cfg.volatility.vol_ref_cadence_sec
                    );
                }
                Err(_) => {
                    eprintln!("[config] WARN: could not parse PARAPHINA_VOL_REF_CADENCE_SEC = {:?} as f64; using default {}",
                        raw,
                        cfg.volatility.vol_ref_cadence_sec
                    );
                }
            }
        }

        // Timescale (seconds) that sigma_min was calibrated for.
        if let Ok(raw) = env::var("PARAPHINA_SIGMA_MIN_CADENCE_SEC") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.volatility.sigma_min_cadence_sec = v.max(1e-3);
                    eprintln!(
                        "[config] PARAPHINA_SIGMA_MIN_CADENCE_SEC = {} (overrode default)",
                        cfg.volatility.sigma_min_cadence_sec
                    );
                }
                Err(_) => {
                    eprintln!("[config] WARN: could not parse PARAPHINA_SIGMA_MIN_CADENCE_SEC = {:?} as f64; using default {}",
                        raw,
                        cfg.volatility.sigma_min_cadence_sec
                    );
                }
            }
        }

        // Daily loss limit in absolute USD; engine treats this as |max drawdown|.
        if let Ok(raw) = env::var("PARAPHINA_DAILY_LOSS_LIMIT") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.risk.daily_loss_limit = v.abs();
                    eprintln!(
                        "[config] PARAPHINA_DAILY_LOSS_LIMIT = {} (overrode default)",
                        cfg.risk.daily_loss_limit
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_DAILY_LOSS_LIMIT = {:?} as f64; using default {}",
                        raw,
                        cfg.risk.daily_loss_limit
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_MAIN_LOOP_INTERVAL_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.main_loop_interval_ms = v.max(1);
                    eprintln!(
                        "[config] PARAPHINA_MAIN_LOOP_INTERVAL_MS = {} (overrode default)",
                        cfg.main_loop_interval_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_MAIN_LOOP_INTERVAL_MS = {:?} as i64; using default {}",
                        raw,
                        cfg.main_loop_interval_ms
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_HEDGE_LOOP_INTERVAL_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.hedge_loop_interval_ms = v.max(1);
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_LOOP_INTERVAL_MS = {} (overrode default)",
                        cfg.hedge_loop_interval_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_LOOP_INTERVAL_MS = {:?} as i64; using default {}",
                        raw,
                        cfg.hedge_loop_interval_ms
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_RISK_LOOP_INTERVAL_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.risk_loop_interval_ms = v.max(1);
                    eprintln!(
                        "[config] PARAPHINA_RISK_LOOP_INTERVAL_MS = {} (overrode default)",
                        cfg.risk_loop_interval_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_RISK_LOOP_INTERVAL_MS = {:?} as i64; using default {}",
                        raw,
                        cfg.risk_loop_interval_ms
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_FILL_AGG_INTERVAL_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.fill_agg_interval_ms = v.max(1);
                    eprintln!(
                        "[config] PARAPHINA_FILL_AGG_INTERVAL_MS = {} (overrode default)",
                        cfg.fill_agg_interval_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_FILL_AGG_INTERVAL_MS = {:?} as i64; using default {}",
                        raw,
                        cfg.fill_agg_interval_ms
                    );
                }
            }
        }

        // Hyperliquid state-level staleness override.
        // This sets VenueConfig.stale_ms_override for the hyperliquid venue,
        // which affects venue health gating and quote staleness guards.
        // NOT the connector watchdog (that's PARAPHINA_HL_STALE_MS).
        if let Ok(raw) = env::var("PARAPHINA_HL_STATE_STALE_MS_OVERRIDE") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let ms = v.max(0);
                    if let Some(venue) = cfg.venues.iter_mut().find(|v| v.id == "hyperliquid") {
                        venue.stale_ms_override = Some(ms);
                        eprintln!(
                            "[config] PARAPHINA_HL_STATE_STALE_MS_OVERRIDE = {} (set hyperliquid stale_ms_override)",
                            ms
                        );
                    }
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HL_STATE_STALE_MS_OVERRIDE = {:?} as i64; ignoring",
                        raw
                    );
                }
            }
        }

        // Extended state-level staleness override.
        // Same pattern as Hyperliquid: affects venue health gating and quote staleness guards.
        // NOT the connector watchdog (that's PARAPHINA_EXTENDED_STALE_MS).
        if let Ok(raw) = env::var("PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let ms = v.max(0);
                    if let Some(venue) = cfg.venues.iter_mut().find(|v| v.id == "extended") {
                        venue.stale_ms_override = Some(ms);
                        eprintln!(
                            "[config] PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE = {} (set extended stale_ms_override)",
                            ms
                        );
                    }
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE = {:?} as i64; ignoring",
                        raw
                    );
                }
            }
        }

        // Aster state-level staleness override.
        // Same pattern as Hyperliquid/Extended: affects venue health gating and quote staleness guards.
        // NOT the connector watchdog (that's PARAPHINA_ASTER_STALE_MS / PARAPHINA_ASTER_BRIDGE_WAIT_STALE_MS).
        if let Ok(raw) = env::var("PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let ms = v.max(0);
                    if let Some(venue) = cfg.venues.iter_mut().find(|v| v.id == "aster") {
                        venue.stale_ms_override = Some(ms);
                        eprintln!(
                            "[config] PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE = {} (set aster stale_ms_override)",
                            ms
                        );
                    }
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE = {:?} as i64; ignoring",
                        raw
                    );
                }
            }
        }

        // Lighter state-level staleness override.
        // Same pattern as Hyperliquid/Extended/Aster: affects venue health gating and quote staleness guards.
        // NOT the connector watchdog (that's PARAPHINA_LIGHTER_STALE_MS).
        if let Ok(raw) = env::var("PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let ms = v.max(0);
                    if let Some(venue) = cfg.venues.iter_mut().find(|v| v.id == "lighter") {
                        venue.stale_ms_override = Some(ms);
                        eprintln!(
                            "[config] PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE = {} (set lighter stale_ms_override)",
                            ms
                        );
                    }
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE = {:?} as i64; ignoring",
                        raw
                    );
                }
            }
        }

        // Paradex state-level staleness override.
        // Same pattern as Hyperliquid/Extended: affects venue health gating and quote staleness guards.
        if let Ok(raw) = env::var("PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    let ms = v.max(0);
                    if let Some(venue) = cfg.venues.iter_mut().find(|v| v.id == "paradex") {
                        venue.stale_ms_override = Some(ms);
                        eprintln!(
                            "[config] PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE = {} (set paradex stale_ms_override)",
                            ms
                        );
                    }
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE = {:?} as i64; ignoring",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_FUNDING_STALE_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.funding.stale_ms = v.max(0);
                    eprintln!(
                        "[config] PARAPHINA_FUNDING_STALE_MS = {} (overrode default)",
                        cfg.funding.stale_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_FUNDING_STALE_MS = {:?} as i64; using default {}",
                        raw, cfg.funding.stale_ms
                    );
                }
            }
        }

        // Catastrophic staleness threshold override.
        // If a venue's book data is older than this, force toxicity=1.0 (Disabled).
        if let Ok(raw) = env::var("PARAPHINA_CATASTROPHIC_STALE_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.toxicity.catastrophic_stale_ms = v.max(0);
                    eprintln!(
                        "[config] PARAPHINA_CATASTROPHIC_STALE_MS = {} (overrode default)",
                        cfg.toxicity.catastrophic_stale_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_CATASTROPHIC_STALE_MS = {:?} as i64; ignoring",
                        raw
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_DEPTH_FALLBACK_GRACE_MS") {
            match raw.parse::<u64>() {
                Ok(v) => {
                    cfg.toxicity.depth_fallback_grace_ms = v.min(i64::MAX as u64) as i64;
                    eprintln!(
                        "[config] PARAPHINA_DEPTH_FALLBACK_GRACE_MS = {} (overrode default)",
                        cfg.toxicity.depth_fallback_grace_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_DEPTH_FALLBACK_GRACE_MS = {:?} as u64; using default {}",
                        raw, cfg.toxicity.depth_fallback_grace_ms
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_FUNDING_AVOID_WINDOW_MS") {
            match raw.parse::<i64>() {
                Ok(v) => {
                    cfg.funding.avoid_window_ms = v.max(0);
                    eprintln!(
                        "[config] PARAPHINA_FUNDING_AVOID_WINDOW_MS = {} (overrode default)",
                        cfg.funding.avoid_window_ms
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_FUNDING_AVOID_WINDOW_MS = {:?} as i64; using default {}",
                        raw, cfg.funding.avoid_window_ms
                    );
                }
            }
        }

        if let Ok(raw) = env::var("PARAPHINA_ENABLE_FUNDING_MM") {
            let enabled = raw == "1" || raw.eq_ignore_ascii_case("true");
            cfg.mm.funding_enabled = enabled;
            eprintln!(
                "[config] PARAPHINA_ENABLE_FUNDING_MM = {} (overrode default)",
                cfg.mm.funding_enabled
            );
        }

        if let Ok(raw) = env::var("PARAPHINA_ENABLE_FUNDING_HEDGE") {
            let enabled = raw == "1" || raw.eq_ignore_ascii_case("true");
            cfg.hedge.funding_enabled = enabled;
            eprintln!(
                "[config] PARAPHINA_ENABLE_FUNDING_HEDGE = {} (overrode default)",
                cfg.hedge.funding_enabled
            );
        }

        if let Ok(raw) = env::var("PARAPHINA_ENABLE_FUNDING_EXIT") {
            let enabled = raw == "1" || raw.eq_ignore_ascii_case("true");
            cfg.exit.funding_enabled = enabled;
            eprintln!(
                "[config] PARAPHINA_ENABLE_FUNDING_EXIT = {} (overrode default)",
                cfg.exit.funding_enabled
            );
        }

        cfg
    }

    /// Backwards-compatible helper:
    /// pick risk profile from PARAPHINA_RISK_PROFILE (default Balanced),
    /// then apply all other env overrides.
    ///
    /// Allowed values (case-insensitive):
    ///   conservative | cons | c
    ///   balanced     | bal  | b   | "" (empty)
    ///   aggressive   | agg  | a
    pub fn from_env_or_default() -> Self {
        use std::env;

        let profile = match env::var("PARAPHINA_RISK_PROFILE") {
            Ok(s) => {
                let s_l = s.to_lowercase();
                match s_l.as_str() {
                    "conservative" | "cons" | "c" => RiskProfile::Conservative,
                    "aggressive" | "agg" | "a" => RiskProfile::Aggressive,
                    "balanced" | "bal" | "b" | "" => RiskProfile::Balanced,
                    other => {
                        eprintln!(
                            "[config] WARN: unknown PARAPHINA_RISK_PROFILE = {:?}; using Balanced",
                            other
                        );
                        RiskProfile::Balanced
                    }
                }
            }
            Err(_) => RiskProfile::Balanced,
        };

        Self::from_env_or_profile(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    /// Global lock for tests that touch env vars. Env vars are process-global,
    /// so parallel tests that modify them will race. Acquire this lock first.
    static ENV_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        ENV_TEST_LOCK.get_or_init(|| Mutex::new(()))
    }

    /// RAII guard that saves an env var's current value and restores it on Drop.
    /// Ensures cleanup even if the test panics.
    struct EnvGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl EnvGuard {
        fn new(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(val) => std::env::set_var(self.key, val),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// Test that PARAPHINA_HL_STATE_STALE_MS_OVERRIDE sets hyperliquid's
    /// stale_ms_override without affecting other venues.
    #[test]
    fn hl_state_stale_override_env_sets_hyperliquid_only() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_HL_STATE_STALE_MS_OVERRIDE";

        // Serialize env-var tests to avoid races.
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        // Clear any existing value for baseline test.
        env::remove_var(ENV_KEY);

        // Baseline: no override set.
        let cfg_baseline = Config::from_env_or_profile(RiskProfile::Balanced);
        let hl_baseline = cfg_baseline.venues.iter().find(|v| v.id == "hyperliquid");
        assert!(
            hl_baseline.is_some(),
            "hyperliquid venue must exist in default config"
        );
        assert_eq!(
            hl_baseline.unwrap().stale_ms_override,
            Some(2_000),
            "baseline hyperliquid stale_ms_override should be Some(2000)"
        );

        // Set override and reload config.
        env::set_var(ENV_KEY, "1500");
        let cfg_with_override = Config::from_env_or_profile(RiskProfile::Balanced);

        // Verify hyperliquid has override.
        let hl = cfg_with_override
            .venues
            .iter()
            .find(|v| v.id == "hyperliquid")
            .expect("hyperliquid venue must exist");
        assert_eq!(
            hl.stale_ms_override,
            Some(1500),
            "hyperliquid stale_ms_override should be 1500"
        );

        // Verify other venues are NOT affected (except paradex which has compiled default 3000).
        for venue in &cfg_with_override.venues {
            if venue.id != "hyperliquid" {
                let expected = if venue.id == "paradex" {
                    Some(3_000)
                } else {
                    None
                };
                assert_eq!(
                    venue.stale_ms_override, expected,
                    "venue {} stale_ms_override mismatch",
                    venue.id
                );
            }
        }
        // EnvGuard restores on drop.
    }

    /// Test that invalid values are ignored (no panic, no override).
    #[test]
    fn hl_state_stale_override_env_ignores_invalid() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_HL_STATE_STALE_MS_OVERRIDE";

        // Serialize env-var tests to avoid races.
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "not_a_number");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let hl = cfg
            .venues
            .iter()
            .find(|v| v.id == "hyperliquid")
            .expect("hyperliquid venue must exist");
        assert_eq!(
            hl.stale_ms_override,
            Some(2_000),
            "invalid env value should leave the compiled default (2000) unchanged"
        );
        // EnvGuard restores on drop.
    }

    /// Test that PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE sets extended's
    /// stale_ms_override without affecting other venues.
    #[test]
    fn extended_state_stale_override_env_sets_extended_only() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE";

        // Serialize env-var tests to avoid races.
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        // Clear any existing value for baseline test.
        env::remove_var(ENV_KEY);

        // Baseline: no override set.
        let cfg_baseline = Config::from_env_or_profile(RiskProfile::Balanced);
        let ext_baseline = cfg_baseline.venues.iter().find(|v| v.id == "extended");
        assert!(
            ext_baseline.is_some(),
            "extended venue must exist in default config"
        );
        assert_eq!(
            ext_baseline.unwrap().stale_ms_override,
            None,
            "baseline extended stale_ms_override should be None"
        );

        // Set override and reload config.
        env::set_var(ENV_KEY, "1500");
        let cfg_with_override = Config::from_env_or_profile(RiskProfile::Balanced);

        // Verify extended has override.
        let ext = cfg_with_override
            .venues
            .iter()
            .find(|v| v.id == "extended")
            .expect("extended venue must exist");
        assert_eq!(
            ext.stale_ms_override,
            Some(1500),
            "extended stale_ms_override should be 1500"
        );

        // Verify other venues are NOT affected (except hyperliquid which has compiled default 2000).
        for venue in &cfg_with_override.venues {
            if venue.id == "extended" {
                continue;
            }
            let expected = if venue.id == "hyperliquid" {
                Some(2_000)
            } else if venue.id == "paradex" {
                Some(3_000)
            } else {
                None
            };
            assert_eq!(
                venue.stale_ms_override, expected,
                "venue {} stale_ms_override mismatch",
                venue.id
            );
        }
        // EnvGuard restores on drop.
    }

    #[test]
    fn aster_default_fee_schedule_matches_current_fee_page() {
        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let aster = cfg
            .venues
            .iter()
            .find(|v| v.id == "aster")
            .expect("aster venue must exist");
        assert!((aster.maker_fee_bps - 0.5).abs() < 1e-9);
        assert!((aster.taker_fee_bps - 4.0).abs() < 1e-9);
        assert!((aster.maker_rebate_bps - 0.0).abs() < 1e-9);
    }

    /// Test that PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE sets aster's
    /// stale_ms_override without affecting other venues.
    #[test]
    fn aster_state_stale_override_env_sets_aster_only() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::remove_var(ENV_KEY);

        let cfg_baseline = Config::from_env_or_profile(RiskProfile::Balanced);
        let aster_baseline = cfg_baseline.venues.iter().find(|v| v.id == "aster");
        assert!(
            aster_baseline.is_some(),
            "aster venue must exist in default config"
        );
        assert_eq!(
            aster_baseline.unwrap().stale_ms_override,
            None,
            "baseline aster stale_ms_override should be None"
        );

        env::set_var(ENV_KEY, "12000");
        let cfg_with_override = Config::from_env_or_profile(RiskProfile::Balanced);

        let aster = cfg_with_override
            .venues
            .iter()
            .find(|v| v.id == "aster")
            .expect("aster venue must exist");
        assert_eq!(
            aster.stale_ms_override,
            Some(12_000),
            "aster stale_ms_override should be 12000"
        );

        for venue in &cfg_with_override.venues {
            if venue.id == "aster" {
                continue;
            }
            let expected = if venue.id == "hyperliquid" {
                Some(2_000)
            } else if venue.id == "paradex" {
                Some(3_000)
            } else {
                None
            };
            assert_eq!(
                venue.stale_ms_override, expected,
                "venue {} stale_ms_override mismatch",
                venue.id
            );
        }
    }

    /// Test that invalid values for Extended are ignored (no panic, no override).
    #[test]
    fn extended_state_stale_override_env_ignores_invalid() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE";

        // Serialize env-var tests to avoid races.
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "not_a_number");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let ext = cfg
            .venues
            .iter()
            .find(|v| v.id == "extended")
            .expect("extended venue must exist");
        assert_eq!(
            ext.stale_ms_override, None,
            "invalid env value should be ignored"
        );
        // EnvGuard restores on drop.
    }

    #[test]
    fn aster_state_stale_override_env_ignores_invalid() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "not_a_number");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let aster = cfg
            .venues
            .iter()
            .find(|v| v.id == "aster")
            .expect("aster venue must exist");
        assert_eq!(
            aster.stale_ms_override, None,
            "invalid env value should be ignored"
        );
    }

    /// Test that PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE sets lighter's
    /// stale_ms_override without affecting other venues.
    #[test]
    fn lighter_state_stale_override_env_sets_lighter_only() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::remove_var(ENV_KEY);

        let cfg_baseline = Config::from_env_or_profile(RiskProfile::Balanced);
        let lighter_baseline = cfg_baseline.venues.iter().find(|v| v.id == "lighter");
        assert!(
            lighter_baseline.is_some(),
            "lighter venue must exist in default config"
        );
        assert_eq!(
            lighter_baseline.unwrap().stale_ms_override,
            None,
            "baseline lighter stale_ms_override should be None"
        );

        env::set_var(ENV_KEY, "12000");
        let cfg_with_override = Config::from_env_or_profile(RiskProfile::Balanced);

        let lighter = cfg_with_override
            .venues
            .iter()
            .find(|v| v.id == "lighter")
            .expect("lighter venue must exist");
        assert_eq!(
            lighter.stale_ms_override,
            Some(12_000),
            "lighter stale_ms_override should be 12000"
        );

        for venue in &cfg_with_override.venues {
            if venue.id == "lighter" {
                continue;
            }
            let expected = if venue.id == "hyperliquid" {
                Some(2_000)
            } else if venue.id == "paradex" {
                Some(3_000)
            } else {
                None
            };
            assert_eq!(
                venue.stale_ms_override, expected,
                "venue {} stale_ms_override mismatch",
                venue.id
            );
        }
    }

    #[test]
    fn lighter_state_stale_override_env_ignores_invalid() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "not_a_number");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let lighter = cfg
            .venues
            .iter()
            .find(|v| v.id == "lighter")
            .expect("lighter venue must exist");
        assert_eq!(
            lighter.stale_ms_override, None,
            "invalid env value should be ignored"
        );
    }

    #[test]
    fn depth_fallback_grace_env_unset_keeps_default() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_DEPTH_FALLBACK_GRACE_MS";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::remove_var(ENV_KEY);

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(
            cfg.toxicity.depth_fallback_grace_ms, 500,
            "default depth_fallback_grace_ms should remain 500 when env is unset"
        );
    }

    #[test]
    fn depth_fallback_grace_env_set_applies() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_DEPTH_FALLBACK_GRACE_MS";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "1500");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(
            cfg.toxicity.depth_fallback_grace_ms, 1500,
            "PARAPHINA_DEPTH_FALLBACK_GRACE_MS should override depth_fallback_grace_ms"
        );
    }

    #[test]
    fn depth_fallback_grace_env_invalid_keeps_default() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_DEPTH_FALLBACK_GRACE_MS";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "not_a_number");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(
            cfg.toxicity.depth_fallback_grace_ms, 500,
            "invalid PARAPHINA_DEPTH_FALLBACK_GRACE_MS should leave default unchanged"
        );
    }

    #[test]
    fn mm_live_tuning_envs_override_defaults() {
        use std::env;

        const EDGE_KEY: &str = "PARAPHINA_MM_EDGE_LOCAL_MIN";
        const PARADEX_KEY: &str = "PARAPHINA_MM_EDGE_LOCAL_MIN_PARADEX";
        const PARADEX_BID_KEY: &str = "PARAPHINA_MM_EDGE_LOCAL_MIN_PARADEX_BID";
        const PARADEX_ASK_KEY: &str = "PARAPHINA_MM_EDGE_LOCAL_MIN_PARADEX_ASK";
        const HEDGE_COST_EDGE_MULT_KEY: &str = "PARAPHINA_MM_HEDGE_COST_EDGE_MULT";
        const LAMBDA_KEY: &str = "PARAPHINA_MM_LAMBDA_INV";
        const LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS";
        const HL_LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_HYPERLIQUID";
        const PARADEX_SUPPRESSION_GRACE_KEY: &str =
            "PARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS";
        const PARADEX_GAP_GRACE_KEY: &str = "PARAPHINA_PARADEX_SUPPORTED_REPLACE_GAP_GRACE_MS";
        const EXT_SUPPORTED_REPLACE_GAP_GRACE_KEY: &str =
            "PARAPHINA_MM_SUPPORTED_REPLACE_GAP_GRACE_MS_EXTENDED";
        const PARADEX_PENDING_PLACE_GRACE_KEY: &str = "PARAPHINA_PARADEX_PENDING_PLACE_GRACE_MS";
        const LIGHTER_PENDING_PLACE_GRACE_KEY: &str = "PARAPHINA_MM_PENDING_PLACE_GRACE_MS_LIGHTER";
        const PRICE_TOL_KEY: &str = "PARAPHINA_MM_PRICE_TOL_TICKS";
        const HL_PRICE_TOL_KEY: &str = "PARAPHINA_MM_PRICE_TOL_TICKS_HYPERLIQUID";
        const SIZE_TOL_KEY: &str = "PARAPHINA_MM_SIZE_TOL_REL";
        const HL_SIZE_TOL_KEY: &str = "PARAPHINA_MM_SIZE_TOL_REL_HYPERLIQUID";
        const EXT_SPREAD_USD_KEY: &str = "PARAPHINA_MM_MAX_QUOTE_SPREAD_USD_EXTENDED";
        const EXT_SPREAD_BPS_KEY: &str = "PARAPHINA_MM_MAX_QUOTE_SPREAD_BPS_EXTENDED";
        const ASTER_GEN_SPREAD_BPS_KEY: &str = "PARAPHINA_MM_MAX_GENERATED_SPREAD_BPS_ASTER";
        const ASTER_MAX_QUOTE_SIZE_KEY: &str = "PARAPHINA_MM_MAX_QUOTE_SIZE_TAO_ASTER";
        const HL_ROLE_KEY: &str = "PARAPHINA_MM_VENUE_ROLE_HYPERLIQUID";
        const EXT_ROLE_KEY: &str = "PARAPHINA_MM_VENUE_ROLE_EXTENDED";

        let _lock = env_lock().lock().unwrap();
        let _edge = EnvGuard::new(EDGE_KEY);
        let _paradex = EnvGuard::new(PARADEX_KEY);
        let _paradex_bid = EnvGuard::new(PARADEX_BID_KEY);
        let _paradex_ask = EnvGuard::new(PARADEX_ASK_KEY);
        let _hedge_cost_edge_mult = EnvGuard::new(HEDGE_COST_EDGE_MULT_KEY);
        let _lambda = EnvGuard::new(LAMBDA_KEY);
        let _lifetime = EnvGuard::new(LIFETIME_KEY);
        let _hl_lifetime = EnvGuard::new(HL_LIFETIME_KEY);
        let _hedge_cost_edge_mult = EnvGuard::new(HEDGE_COST_EDGE_MULT_KEY);
        let _paradex_suppression_grace = EnvGuard::new(PARADEX_SUPPRESSION_GRACE_KEY);
        let _paradex_gap_grace = EnvGuard::new(PARADEX_GAP_GRACE_KEY);
        let _ext_supported_replace_gap_grace = EnvGuard::new(EXT_SUPPORTED_REPLACE_GAP_GRACE_KEY);
        let _paradex_pending_place_grace = EnvGuard::new(PARADEX_PENDING_PLACE_GRACE_KEY);
        let _lighter_pending_place_grace = EnvGuard::new(LIGHTER_PENDING_PLACE_GRACE_KEY);
        let _price_tol = EnvGuard::new(PRICE_TOL_KEY);
        let _hl_price_tol = EnvGuard::new(HL_PRICE_TOL_KEY);
        let _size_tol = EnvGuard::new(SIZE_TOL_KEY);
        let _hl_size_tol = EnvGuard::new(HL_SIZE_TOL_KEY);
        let _ext_spread_usd = EnvGuard::new(EXT_SPREAD_USD_KEY);
        let _ext_spread_bps = EnvGuard::new(EXT_SPREAD_BPS_KEY);
        let _aster_gen_spread_bps = EnvGuard::new(ASTER_GEN_SPREAD_BPS_KEY);
        let _aster_max_quote_size = EnvGuard::new(ASTER_MAX_QUOTE_SIZE_KEY);
        let _hl_role = EnvGuard::new(HL_ROLE_KEY);
        let _ext_role = EnvGuard::new(EXT_ROLE_KEY);

        env::set_var(EDGE_KEY, "0.15");
        env::set_var(PARADEX_KEY, "0.09");
        env::set_var(PARADEX_BID_KEY, "0.03");
        env::set_var(PARADEX_ASK_KEY, "0.11");
        env::set_var(HEDGE_COST_EDGE_MULT_KEY, "0.6");
        env::set_var(LAMBDA_KEY, "0.8");
        env::set_var(LIFETIME_KEY, "1500");
        env::set_var(HL_LIFETIME_KEY, "3000");
        env::set_var(PARADEX_SUPPRESSION_GRACE_KEY, "900");
        env::set_var(PARADEX_GAP_GRACE_KEY, "4000");
        env::set_var(EXT_SUPPORTED_REPLACE_GAP_GRACE_KEY, "45000");
        env::set_var(PARADEX_PENDING_PLACE_GRACE_KEY, "8000");
        env::set_var(LIGHTER_PENDING_PLACE_GRACE_KEY, "7000");
        env::set_var(PRICE_TOL_KEY, "2.5");
        env::set_var(HL_PRICE_TOL_KEY, "4.0");
        env::set_var(SIZE_TOL_KEY, "0.25");
        env::set_var(HL_SIZE_TOL_KEY, "0.35");
        env::set_var(EXT_SPREAD_USD_KEY, "3.0");
        env::set_var(EXT_SPREAD_BPS_KEY, "15");
        env::set_var(ASTER_GEN_SPREAD_BPS_KEY, "10");
        env::set_var(ASTER_MAX_QUOTE_SIZE_KEY, "0.01");
        env::set_var(HL_ROLE_KEY, "probationary");
        env::set_var(EXT_ROLE_KEY, "noise");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert!((cfg.mm.edge_local_min - 0.15).abs() < 1e-9);
        assert!((cfg.mm.edge_local_min_bid_for("paradex") - 0.03).abs() < 1e-9);
        assert!((cfg.mm.edge_local_min_ask_for("paradex") - 0.11).abs() < 1e-9);
        assert!((cfg.mm.edge_local_min_bid_for("aster") - 0.15).abs() < 1e-9);
        assert!((cfg.mm.edge_local_min_ask_for("extended") - 0.15).abs() < 1e-9);
        assert!((cfg.mm.hedge_cost_edge_mult - 0.6).abs() < 1e-9);
        assert!((cfg.mm.lambda_inv - 0.8).abs() < 1e-9);
        assert_eq!(cfg.mm.min_quote_lifetime_ms, 1500);
        assert_eq!(cfg.mm.min_quote_lifetime_ms_for("hyperliquid"), 3000);
        assert_eq!(cfg.mm.min_quote_lifetime_ms_for("lighter"), 1500);
        assert_eq!(
            cfg.mm.post_control_suppression_grace_ms_for("paradex"),
            1500
        );
        assert_eq!(cfg.mm.post_control_suppression_grace_ms_for("aster"), 1500);
        assert_eq!(
            cfg.mm
                .supported_replace_snapshot_gap_grace_ms_for("paradex"),
            4000
        );
        assert_eq!(
            cfg.mm
                .supported_replace_snapshot_gap_grace_ms_for("extended"),
            45000
        );
        assert_eq!(
            cfg.mm.supported_replace_snapshot_gap_grace_ms_for("aster"),
            2000
        );
        assert_eq!(cfg.mm.pending_place_grace_ms_for("paradex"), 8000);
        assert_eq!(cfg.mm.pending_place_grace_ms_for("lighter"), 7000);
        assert_eq!(cfg.mm.pending_place_grace_ms_for("aster"), 0);
        assert!((cfg.mm.price_tol_ticks - 2.5).abs() < 1e-9);
        assert!((cfg.mm.price_tol_ticks_for("hyperliquid") - 4.0).abs() < 1e-9);
        assert!((cfg.mm.price_tol_ticks_for("aster") - 2.5).abs() < 1e-9);
        assert!((cfg.mm.size_tol_rel - 0.25).abs() < 1e-9);
        assert!((cfg.mm.size_tol_rel_for("hyperliquid") - 0.35).abs() < 1e-9);
        assert!((cfg.mm.size_tol_rel_for("paradex") - 0.25).abs() < 1e-9);
        assert_eq!(cfg.mm.max_quote_spread_abs_usd_for("extended"), Some(3.0));
        assert_eq!(cfg.mm.max_quote_spread_bps_for("extended"), Some(15.0));
        assert_eq!(
            cfg.mm.max_generated_quote_spread_bps_for("aster"),
            Some(10.0)
        );
        assert_eq!(cfg.mm.max_quote_size_tao_for("aster"), Some(0.01));
        assert_eq!(cfg.mm.max_quote_spread_abs_usd_for("hyperliquid"), None);
        assert_eq!(
            cfg.mm.venue_role_for("hyperliquid"),
            MmVenueRole::Probationary
        );
        assert_eq!(cfg.mm.venue_role_for("extended"), MmVenueRole::Noise);
        assert_eq!(cfg.mm.venue_role_for("aster"), MmVenueRole::Fill);
    }

    #[test]
    fn mm_live_tuning_envs_ignore_invalid_values() {
        use std::env;

        const LAMBDA_KEY: &str = "PARAPHINA_MM_LAMBDA_INV";
        const LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS";
        const HL_LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_HYPERLIQUID";
        const HEDGE_COST_EDGE_MULT_KEY: &str = "PARAPHINA_MM_HEDGE_COST_EDGE_MULT";
        const PARADEX_SUPPRESSION_GRACE_KEY: &str =
            "PARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS";
        const PARADEX_GAP_GRACE_KEY: &str = "PARAPHINA_PARADEX_SUPPORTED_REPLACE_GAP_GRACE_MS";
        const EXT_SUPPORTED_REPLACE_GAP_GRACE_KEY: &str =
            "PARAPHINA_MM_SUPPORTED_REPLACE_GAP_GRACE_MS_EXTENDED";
        const PARADEX_PENDING_PLACE_GRACE_KEY: &str = "PARAPHINA_PARADEX_PENDING_PLACE_GRACE_MS";
        const LIGHTER_PENDING_PLACE_GRACE_KEY: &str = "PARAPHINA_MM_PENDING_PLACE_GRACE_MS_LIGHTER";
        const PRICE_TOL_KEY: &str = "PARAPHINA_MM_PRICE_TOL_TICKS";
        const HL_PRICE_TOL_KEY: &str = "PARAPHINA_MM_PRICE_TOL_TICKS_HYPERLIQUID";
        const SIZE_TOL_KEY: &str = "PARAPHINA_MM_SIZE_TOL_REL";
        const HL_SIZE_TOL_KEY: &str = "PARAPHINA_MM_SIZE_TOL_REL_HYPERLIQUID";
        const EXT_SPREAD_USD_KEY: &str = "PARAPHINA_MM_MAX_QUOTE_SPREAD_USD_EXTENDED";
        const EXT_SPREAD_BPS_KEY: &str = "PARAPHINA_MM_MAX_QUOTE_SPREAD_BPS_EXTENDED";
        const ASTER_MAX_QUOTE_SIZE_KEY: &str = "PARAPHINA_MM_MAX_QUOTE_SIZE_TAO_ASTER";
        const HL_ROLE_KEY: &str = "PARAPHINA_MM_VENUE_ROLE_HYPERLIQUID";

        let _lock = env_lock().lock().unwrap();
        let _lambda = EnvGuard::new(LAMBDA_KEY);
        let _lifetime = EnvGuard::new(LIFETIME_KEY);
        let _hl_lifetime = EnvGuard::new(HL_LIFETIME_KEY);
        let _paradex_suppression_grace = EnvGuard::new(PARADEX_SUPPRESSION_GRACE_KEY);
        let _paradex_gap_grace = EnvGuard::new(PARADEX_GAP_GRACE_KEY);
        let _ext_supported_replace_gap_grace = EnvGuard::new(EXT_SUPPORTED_REPLACE_GAP_GRACE_KEY);
        let _paradex_pending_place_grace = EnvGuard::new(PARADEX_PENDING_PLACE_GRACE_KEY);
        let _lighter_pending_place_grace = EnvGuard::new(LIGHTER_PENDING_PLACE_GRACE_KEY);
        let _price_tol = EnvGuard::new(PRICE_TOL_KEY);
        let _hl_price_tol = EnvGuard::new(HL_PRICE_TOL_KEY);
        let _size_tol = EnvGuard::new(SIZE_TOL_KEY);
        let _hl_size_tol = EnvGuard::new(HL_SIZE_TOL_KEY);
        let _ext_spread_usd = EnvGuard::new(EXT_SPREAD_USD_KEY);
        let _ext_spread_bps = EnvGuard::new(EXT_SPREAD_BPS_KEY);
        let _aster_max_quote_size = EnvGuard::new(ASTER_MAX_QUOTE_SIZE_KEY);
        let _hl_role = EnvGuard::new(HL_ROLE_KEY);

        env::set_var(LAMBDA_KEY, "not_a_number");
        env::set_var(LIFETIME_KEY, "not_a_number");
        env::set_var(HL_LIFETIME_KEY, "not_a_number");
        env::set_var(HEDGE_COST_EDGE_MULT_KEY, "not_a_number");
        env::set_var(PARADEX_SUPPRESSION_GRACE_KEY, "not_a_number");
        env::set_var(PARADEX_GAP_GRACE_KEY, "not_a_number");
        env::set_var(EXT_SUPPORTED_REPLACE_GAP_GRACE_KEY, "not_a_number");
        env::set_var(PARADEX_PENDING_PLACE_GRACE_KEY, "not_a_number");
        env::set_var(LIGHTER_PENDING_PLACE_GRACE_KEY, "not_a_number");
        env::set_var(PRICE_TOL_KEY, "not_a_number");
        env::set_var(HL_PRICE_TOL_KEY, "not_a_number");
        env::set_var(SIZE_TOL_KEY, "not_a_number");
        env::set_var(HL_SIZE_TOL_KEY, "not_a_number");
        env::set_var(EXT_SPREAD_USD_KEY, "not_a_number");
        env::set_var(EXT_SPREAD_BPS_KEY, "0");
        env::set_var(ASTER_MAX_QUOTE_SIZE_KEY, "0");
        env::set_var(HL_ROLE_KEY, "not_a_role");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert!((cfg.mm.lambda_inv - 0.3).abs() < 1e-9);
        assert!((cfg.mm.hedge_cost_edge_mult - 0.0).abs() < 1e-9);
        assert_eq!(cfg.mm.min_quote_lifetime_ms, 500);
        assert_eq!(cfg.mm.min_quote_lifetime_ms_for("hyperliquid"), 500);
        assert_eq!(cfg.mm.post_control_suppression_grace_ms_for("paradex"), 500);
        assert_eq!(
            cfg.mm
                .supported_replace_snapshot_gap_grace_ms_for("paradex"),
            2000
        );
        assert_eq!(
            cfg.mm
                .supported_replace_snapshot_gap_grace_ms_for("extended"),
            2000
        );
        assert_eq!(cfg.mm.pending_place_grace_ms_for("paradex"), 0);
        assert_eq!(cfg.mm.pending_place_grace_ms_for("lighter"), 0);
        assert!((cfg.mm.price_tol_ticks - 1.0).abs() < 1e-9);
        assert!((cfg.mm.price_tol_ticks_for("hyperliquid") - 1.0).abs() < 1e-9);
        assert!((cfg.mm.size_tol_rel - 0.10).abs() < 1e-9);
        assert!((cfg.mm.size_tol_rel_for("hyperliquid") - 0.10).abs() < 1e-9);
        assert_eq!(cfg.mm.max_quote_spread_abs_usd_for("extended"), None);
        assert_eq!(cfg.mm.max_quote_spread_bps_for("extended"), None);
        assert_eq!(cfg.mm.max_quote_size_tao_for("aster"), None);
        assert_eq!(cfg.mm.venue_role_for("hyperliquid"), MmVenueRole::Fill);
    }

    #[test]
    fn paradex_post_control_suppression_grace_env_extends_beyond_base_lifetime() {
        use std::env;

        const PARADEX_LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_PARADEX";
        const PARADEX_SUPPRESSION_GRACE_KEY: &str =
            "PARAPHINA_PARADEX_POST_CONTROL_SUPPRESSION_GRACE_MS";

        let _lock = env_lock().lock().unwrap();
        let _paradex_lifetime = EnvGuard::new(PARADEX_LIFETIME_KEY);
        let _paradex_suppression_grace = EnvGuard::new(PARADEX_SUPPRESSION_GRACE_KEY);

        env::set_var(PARADEX_LIFETIME_KEY, "300");
        env::set_var(PARADEX_SUPPRESSION_GRACE_KEY, "900");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(cfg.mm.min_quote_lifetime_ms_for("paradex"), 300);
        assert_eq!(cfg.mm.post_control_suppression_grace_ms_for("paradex"), 900);
        assert_eq!(cfg.mm.post_control_suppression_grace_ms_for("aster"), 500);
    }

    #[test]
    fn paradex_edge_under_min_env_sets_grace_and_band() {
        use std::env;

        const PARADEX_LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_PARADEX";
        const PARADEX_EDGE_GRACE_KEY: &str = "PARAPHINA_PARADEX_EDGE_UNDER_MIN_GRACE_MS";
        const PARADEX_EDGE_BAND_KEY: &str = "PARAPHINA_PARADEX_EDGE_UNDER_MIN_BAND_USD";

        let _lock = env_lock().lock().unwrap();
        let _paradex_lifetime = EnvGuard::new(PARADEX_LIFETIME_KEY);
        let _paradex_edge_grace = EnvGuard::new(PARADEX_EDGE_GRACE_KEY);
        let _paradex_edge_band = EnvGuard::new(PARADEX_EDGE_BAND_KEY);

        env::set_var(PARADEX_LIFETIME_KEY, "300");
        env::set_var(PARADEX_EDGE_GRACE_KEY, "900");
        env::set_var(PARADEX_EDGE_BAND_KEY, "0.02");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(cfg.mm.edge_under_min_grace_ms_for("paradex"), 900);
        assert_eq!(cfg.mm.edge_under_min_grace_ms_for("aster"), 500);
        assert_eq!(cfg.mm.edge_under_min_band_usd_for("paradex"), Some(0.02));
        assert_eq!(cfg.mm.edge_under_min_band_usd_for("aster"), None);
    }

    #[test]
    fn hyperliquid_edge_under_min_env_sets_grace_and_band() {
        use std::env;

        const HYPERLIQUID_LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_HYPERLIQUID";
        const HYPERLIQUID_EDGE_GRACE_KEY: &str = "PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_GRACE_MS";
        const HYPERLIQUID_EDGE_BAND_KEY: &str = "PARAPHINA_HYPERLIQUID_EDGE_UNDER_MIN_BAND_USD";

        let _lock = env_lock().lock().unwrap();
        let _hyperliquid_lifetime = EnvGuard::new(HYPERLIQUID_LIFETIME_KEY);
        let _hyperliquid_edge_grace = EnvGuard::new(HYPERLIQUID_EDGE_GRACE_KEY);
        let _hyperliquid_edge_band = EnvGuard::new(HYPERLIQUID_EDGE_BAND_KEY);

        env::set_var(HYPERLIQUID_LIFETIME_KEY, "1500");
        env::set_var(HYPERLIQUID_EDGE_GRACE_KEY, "3000");
        env::set_var(HYPERLIQUID_EDGE_BAND_KEY, "0.04");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(cfg.mm.edge_under_min_grace_ms_for("hyperliquid"), 3000);
        assert_eq!(cfg.mm.edge_under_min_grace_ms_for("aster"), 500);
        assert_eq!(
            cfg.mm.edge_under_min_band_usd_for("hyperliquid"),
            Some(0.04)
        );
        assert_eq!(cfg.mm.edge_under_min_band_usd_for("aster"), None);
    }

    #[test]
    fn extended_edge_under_min_env_sets_grace_and_band() {
        use std::env;

        const EXTENDED_LIFETIME_KEY: &str = "PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_EXTENDED";
        const EXTENDED_EDGE_GRACE_KEY: &str = "PARAPHINA_EXTENDED_EDGE_UNDER_MIN_GRACE_MS";
        const EXTENDED_EDGE_BAND_KEY: &str = "PARAPHINA_EXTENDED_EDGE_UNDER_MIN_BAND_USD";

        let _lock = env_lock().lock().unwrap();
        let _extended_lifetime = EnvGuard::new(EXTENDED_LIFETIME_KEY);
        let _extended_edge_grace = EnvGuard::new(EXTENDED_EDGE_GRACE_KEY);
        let _extended_edge_band = EnvGuard::new(EXTENDED_EDGE_BAND_KEY);

        env::set_var(EXTENDED_LIFETIME_KEY, "500");
        env::set_var(EXTENDED_EDGE_GRACE_KEY, "3000");
        env::set_var(EXTENDED_EDGE_BAND_KEY, "0.04");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert_eq!(cfg.mm.edge_under_min_grace_ms_for("extended"), 3000);
        assert_eq!(cfg.mm.edge_under_min_grace_ms_for("aster"), 500);
        assert_eq!(cfg.mm.edge_under_min_band_usd_for("extended"), Some(0.04));
        assert_eq!(cfg.mm.edge_under_min_band_usd_for("aster"), None);
    }

    #[test]
    fn hedge_disabled_venues_env_disables_matching_venues_only() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_HEDGE_DISABLED_VENUES";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "paradex, lighter");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let by_id = |id: &str| {
            cfg.venues
                .iter()
                .find(|v| v.id == id)
                .expect("venue must exist")
                .is_hedge_allowed
        };

        assert!(!by_id("paradex"));
        assert!(!by_id("lighter"));
        assert!(by_id("extended"));
        assert!(by_id("hyperliquid"));
        assert!(by_id("aster"));
    }

    #[test]
    fn fv_disabled_venues_env_disables_matching_venues_only() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_FV_DISABLED_VENUES";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "paradex, lighter");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let by_id = |id: &str| {
            cfg.venues
                .iter()
                .find(|v| v.id == id)
                .expect("venue must exist")
                .contributes_to_fv
        };

        assert!(!by_id("paradex"));
        assert!(!by_id("lighter"));
        assert!(by_id("extended"));
        assert!(by_id("hyperliquid"));
        assert!(by_id("aster"));
    }

    #[test]
    fn fv_max_spread_bps_env_sets_only_matching_venue() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_FV_MAX_SPREAD_BPS_LIGHTER";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "0.5");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        let by_id = |id: &str| {
            cfg.venues
                .iter()
                .find(|v| v.id == id)
                .expect("venue must exist")
                .fv_max_spread_bps
        };

        assert_eq!(by_id("lighter"), Some(0.5));
        assert_eq!(by_id("hyperliquid"), None);
        assert_eq!(by_id("aster"), None);
        assert_eq!(by_id("paradex"), None);
        assert_eq!(by_id("extended"), None);
    }

    #[test]
    fn hedge_min_depth_env_overrides_default() {
        use std::env;

        const ENV_KEY: &str = "PARAPHINA_HEDGE_MIN_DEPTH_USD";

        let _lock = env_lock().lock().unwrap();
        let _guard = EnvGuard::new(ENV_KEY);

        env::set_var(ENV_KEY, "125.5");

        let cfg = Config::from_env_or_profile(RiskProfile::Balanced);
        assert!((cfg.hedge.min_depth_usd - 125.5).abs() < 1e-9);
    }
}
