// src/config.rs

#[derive(Debug, Clone)]
pub struct Config {
    /// Human-readable config / release version.
    pub version: &'static str,
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
    /// Toxicity / venue-health config.
    pub toxicity: ToxicityConfig,
}

#[derive(Debug, Clone)]
pub struct VenueConfig {
    pub id: String,
    pub name: String,
    pub tick_size: f64,
    pub base_order_size: f64,
    pub max_order_size: f64,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub maker_rebate_bps: f64,
    pub gamma: f64,
    pub k: f64,
    pub w_liq: f64,
    pub w_fund: f64,
    pub is_hedge_allowed: bool,
}

#[derive(Debug, Clone)]
pub struct BookConfig {
    pub depth_levels: usize,
    pub stale_ms: i64,
    pub min_healthy_for_kf: u32,
    pub max_mid_jump_pct: f64,
}

#[derive(Debug, Clone)]
pub struct KalmanConfig {
    pub q_base: f64,
    pub r_a: f64,
    pub r_b: f64,
    pub r_min: f64,
    pub r_max: f64,
    pub p_init: f64,
}

#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    pub fv_vol_alpha_short: f64,
    pub fv_vol_alpha_long: f64,
    pub sigma_min: f64,
    pub vol_ref: f64,
    pub vol_ratio_min: f64,
    pub vol_ratio_max: f64,
    pub spread_vol_mult_coeff: f64,
    pub size_vol_mult_coeff: f64,
    pub band_vol_mult_coeff: f64,
}

#[derive(Debug, Clone)]
pub struct RiskConfig {
    pub delta_hard_limit_usd_base: f64,
    pub delta_warn_frac: f64,
    pub basis_hard_limit_usd: f64,
    pub basis_warn_frac: f64,
    pub daily_loss_limit: f64,
    pub pnl_warn_frac: f64,
}

#[derive(Debug, Clone)]
pub struct MmConfig {
    pub basis_weight: f64,
    pub funding_weight: f64,
    pub edge_local_min: f64,
    pub edge_vol_mult: f64,
    pub lambda_inv: f64,
    pub quote_horizon_sec: f64,
    pub funding_skew_slope: f64,
    pub funding_skew_clip: f64,
}

#[derive(Debug, Clone)]
pub struct HedgeConfig {
    /// HEDGE_BAND_BASE in TAO.
    pub hedge_band_base: f64,
    /// HEDGE_MAX_STEP in TAO per hedge action.
    pub hedge_max_step: f64,
    /// LQ controller weights α, β.
    pub alpha_hedge: f64,
    pub beta_hedge: f64,
}

#[derive(Debug, Clone)]
pub struct ToxicityConfig {
    /// VOL_TOX_SCALE: how quickly relative vol turns into toxicity.
    pub vol_tox_scale: f64,
    /// FLOW_TOX_SCALE: scale for throughput normalisation (future use).
    pub flow_tox_scale: f64,
    /// Threshold between low and medium toxicity.
    pub tox_med_threshold: f64,
    /// Threshold for "high" toxicity → venue disabled.
    pub tox_high_threshold: f64,
    /// Weights for features f1..f5.
    pub w1: f64,
    pub w2: f64,
    pub w3: f64,
    pub w4: f64,
    pub w5: f64,
}

impl Default for Config {
    fn default() -> Self {
        // ----- Venue configs -----
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
                gamma: 0.1,
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
                w_liq: 0.2,
                w_fund: 0.2,
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
                gamma: 0.1,
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
                gamma: 0.1,
                k: 1.5,
                w_liq: 0.15,
                w_fund: 0.15,
                is_hedge_allowed: true,
            },
        ];

        let book = BookConfig {
            depth_levels: 10,
            stale_ms: 1_000,
            min_healthy_for_kf: 2,
            max_mid_jump_pct: 0.02,
        };

        let kalman = KalmanConfig {
            q_base: 1e-6,
            r_a: 1e-6,
            r_b: 1e2,
            r_min: 1e-8,
            r_max: 1e-2,
            p_init: 1.0,
        };

        let volatility = VolatilityConfig {
            fv_vol_alpha_short: 0.2,
            fv_vol_alpha_long: 0.05,
            sigma_min: 0.001,
            vol_ref: 0.02,
            vol_ratio_min: 0.25,
            vol_ratio_max: 4.0,
            spread_vol_mult_coeff: 1.0,
            size_vol_mult_coeff: 2.0,
            band_vol_mult_coeff: 1.0,
        };

        let risk = RiskConfig {
            delta_hard_limit_usd_base: 100_000.0,
            delta_warn_frac: 0.7,
            basis_hard_limit_usd: 10_000.0,
            basis_warn_frac: 0.7,
            daily_loss_limit: -5_000.0,
            pnl_warn_frac: 0.5,
        };

        let mm = MmConfig {
            basis_weight: 0.3,
            funding_weight: 0.3,
            edge_local_min: 0.5,
            edge_vol_mult: 0.2,
            lambda_inv: 0.5,
            quote_horizon_sec: 30.0,
            funding_skew_slope: 10_000.0,
            funding_skew_clip: 100.0,
        };

        let hedge = HedgeConfig {
            hedge_band_base: 5.0, // TAO band
            hedge_max_step: 20.0, // TAO per hedge step
            alpha_hedge: 1.0,
            beta_hedge: 1.0,
        };

        let toxicity = ToxicityConfig {
            vol_tox_scale: 1.0,      // VOL_TOX_SCALE
            flow_tox_scale: 1_000.0, // FLOW_TOX_SCALE (placeholder)
            tox_med_threshold: 0.4,
            tox_high_threshold: 0.8,
            // Feature weights; these sum to 1.0.
            w1: 0.5, // relative vol
            w2: 0.2, // neg markouts
            w3: 0.1, // imbalance
            w4: 0.1, // directional flow
            w5: 0.1, // throughput
        };

        Config {
            version: "v0.1.0-whitepaper-structure",
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
