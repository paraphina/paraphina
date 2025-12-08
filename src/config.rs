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
    /// Toxicity scoring + venue health config.
    pub toxicity: ToxicityConfig,
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
    /// Max allowed daily PnL loss in absolute USD (realised + unrealised).
    /// Positive value; the engine interprets this as a loss threshold.
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
    /// Weight of basis in reservation price.
    pub basis_weight: f64,
    /// Weight of funding in reservation price.
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
}

#[derive(Debug, Clone)]
pub struct HedgeConfig {
    /// Base half-band in TAO (before vol scaling).
    pub hedge_band_base: f64,
    /// Max TAO we are allowed to move in one hedge action.
    pub hedge_max_step: f64,
    /// LQ controller weight α_hedge.
    pub alpha_hedge: f64,
    /// LQ controller weight β_hedge.
    pub beta_hedge: f64,
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
}

impl Default for Config {
    fn default() -> Self {
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
            // Hyper-search-optimal reference vol for current synthetic regime.
            vol_ref: 0.01,
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
            // Daily loss limit (realised + unrealised), in absolute USD.
            // Interpreted as a positive loss threshold by the engine.
            daily_loss_limit: 2_000.0,
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
            size_eta: 0.10,
            // 0 = pure global, 1 = pure per-venue target; we sit in the middle.
            lambda_inv: 0.5,
            // Quoting horizon in seconds used in the AS formulas.
            quote_horizon_sec: 30.0,
            // Funding-driven inventory skew: slope and clip.
            funding_skew_slope: 10_000.0,
            funding_skew_clip: 100.0,
        };

        // ----- Hedge engine (global LQ controller) -----
        let hedge = HedgeConfig {
            // With ~300 USD / TAO and our default limits, this band corresponds
            // to ~6–9k USD of unhedged delta before the LQ controller kicks in.
            hedge_band_base: 5.0, // TAO band
            hedge_max_step: 20.0, // TAO per hedge step
            alpha_hedge: 1.0,
            beta_hedge: 1.0,
        };

        // ----- Toxicity / venue health -----
        //
        // f1 = clip((local_vol / sigma_eff - 1) / vol_tox_scale, 0, 1)
        //
        // With these defaults we need a venue to run roughly 25–30% hotter
        // than the global volatility before entering Warning, and ~45–50%
        // hotter before being Disabled.
        let toxicity = ToxicityConfig {
            vol_tox_scale: 0.5,
            tox_med_threshold: 0.6, // Warning only when volatility is clearly elevated
            tox_high_threshold: 0.9, // Disabled only when it is much higher still
        };

        Config {
            version: "v0.1.3-hyper-optimised",
            initial_q_tao: 0.0,
            venues,
            book,
            kalman,
            volatility,
            risk,
            mm,
            hedge,
            toxicity,
        }
    }
}

// --- Runtime config loader: env overrides on top of defaults -----------------

impl Config {
    /// Build a Config from defaults, then apply environment overrides.
    ///
    /// This is designed for research / batch runs and future RL:
    ///
    ///   - PARAPHINA_INIT_Q_TAO       (f64, TAO)
    ///   - PARAPHINA_HEDGE_BAND_BASE  (f64, TAO)
    ///   - PARAPHINA_HEDGE_MAX_STEP   (f64, TAO)
    ///
    /// Any variable that fails to parse is ignored with a warning.
    pub fn from_env_or_default() -> Self {
        use std::env;

        let mut cfg = Config::default();

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
                    cfg.hedge.hedge_band_base = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_BAND_BASE = {} (overrode default)",
                        cfg.hedge.hedge_band_base
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_BAND_BASE = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.hedge_band_base
                    );
                }
            }
        }

        // Hedge max step (in TAO).
        if let Ok(raw) = env::var("PARAPHINA_HEDGE_MAX_STEP") {
            match raw.parse::<f64>() {
                Ok(v) => {
                    cfg.hedge.hedge_max_step = v.max(0.0);
                    eprintln!(
                        "[config] PARAPHINA_HEDGE_MAX_STEP = {} (overrode default)",
                        cfg.hedge.hedge_max_step
                    );
                }
                Err(_) => {
                    eprintln!(
                        "[config] WARN: could not parse PARAPHINA_HEDGE_MAX_STEP = {:?} as f64; using default {}",
                        raw,
                        cfg.hedge.hedge_max_step
                    );
                }
            }
        }

        cfg
    }
}
