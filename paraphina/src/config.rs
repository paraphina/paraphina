// src/config.rs
//
// Central configuration for the Paraphina engine.
// This is the single source of truth that maps directly onto the
// whitepaper parameters (venues, Kalman fair value, vols, risk,
// MM quoting, hedging, toxicity / venue health).
//
// It also carries a small number of "simulation environment"
// parameters such as the initial global inventory q0 in TAO.

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
    /// Kalman filter config for log fair value.
    pub kalman: KalmanConfig,
    /// Volatility / control-scalar config.
    pub volatility: VolatilityConfig,
    /// Global risk / limit config.
    pub risk: RiskConfig,
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
    /// Minimum lot size in TAO (orders smaller than this are rejected).
    pub lot_size_tao: f64,
    /// Size step/increment in TAO (orders must be multiples of this).
    pub size_step_tao: f64,
    /// Minimum notional value in USD (orders below this are skipped).
    pub min_notional_usd: f64,
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
pub struct MmConfig {
    /// Weight of basis in reservation price (β_b in spec).
    pub basis_weight: f64,
    /// Weight of funding in reservation price (β_f in spec).
    pub funding_weight: f64,
    /// Minimum per-unit edge for local MM quotes (in USD).
    pub edge_local_min: f64,
    /// Multiplier for volatility-based edge buffer.
    pub edge_vol_mult: f64,
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
    /// Price tolerance in ticks before triggering order replacement.
    pub price_tol_ticks: f64,
    /// Size tolerance (relative) before triggering order replacement.
    pub size_tol_rel: f64,

    // ----- Funding target inventory (Section 9) -----
    /// Scale for funding rate in target inventory calculation.
    /// phi(funding_8h) = clip(funding_8h / this, -1, 1) * FUNDING_TARGET_MAX_TAO
    pub funding_target_rate_scale: f64,
    /// Maximum TAO shift from funding preference per venue.
    pub funding_target_max_tao: f64,
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
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
            },
            VenueConfig {
                id: "hyperliquid".to_string(),
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
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
            },
            VenueConfig {
                id: "aster".to_string(),
                name: "Aster".to_string(),
                tick_size: 0.01,
                base_order_size: 1.0,
                max_order_size: 20.0,
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
                maker_rebate_bps: 0.0,
                gamma: 0.11,
                k: 1.4,
                w_liq: 0.20,
                w_fund: 0.20,
                is_hedge_allowed: true,
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
            },
            VenueConfig {
                id: "lighter".to_string(),
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
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
            },
            VenueConfig {
                id: "paradex".to_string(),
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
                lot_size_tao: 0.01,
                size_step_tao: 0.01,
                min_notional_usd: 10.0,
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

        // ----- MM (Avellaneda–Stoikov + basis/funding) -----
        let mm = MmConfig {
            // Reservation price adjustment weights. These are deliberately
            // modest so we only lightly lean inventory based on basis/funding.
            basis_weight: 0.3,
            funding_weight: 0.3,
            // Local minimum edge in USD, plus a vol-dependent buffer.
            edge_local_min: 0.5,
            edge_vol_mult: 0.2,
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
            price_tol_ticks: 1.0,
            size_tol_rel: 0.10,

            // Funding target inventory (Section 9)
            funding_target_rate_scale: 0.001, // 0.1% funding rate = full shift
            funding_target_max_tao: 5.0,      // max TAO shift from funding preference
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
        };

        Config {
            version: "v0.1.6-worldmodel-exp08-presets",
            initial_q_tao: 0.0,
            venues,
            book,
            kalman,
            volatility,
            risk,
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
    ///   - PARAPHINA_MM_SIZE_ETA       (f64)
    ///   - PARAPHINA_VOL_REF           (f64)
    ///   - PARAPHINA_DAILY_LOSS_LIMIT  (f64, USD; positive threshold)
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
