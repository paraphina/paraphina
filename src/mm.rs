// src/mm.rs
//
// Avellaneda–Stoikov style quoting with the whitepaper's additions:
//
// - global + per-venue inventory tilt,
// - per-venue basis and funding terms in the reservation price,
// - hook for per-venue inventory targets,
// - simple volatility scaling of spreads,
// - conversion to abstract OrderIntent structs.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side};

#[derive(Debug, Clone)]
pub struct SideQuote {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct VenueQuote {
    pub venue_index: usize,
    pub venue_id: String,
    pub bid: Option<SideQuote>,
    pub ask: Option<SideQuote>,
}

/// Bounded funding-to-skew map φ from the whitepaper.
/// Here we use a simple linear map with clipping.
fn funding_to_skew(funding_8h: f64, slope: f64, clip_abs: f64) -> f64 {
    let raw = funding_8h * slope;
    let clip = clip_abs.abs();
    raw.max(-clip).min(clip)
}

/// Compute model-based quotes per venue.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Fallback fair value in case KF has not initialised yet.
    let s_t = state.fair_value.unwrap_or(250.0);

    // Effective sigma (short-horizon FV vol with floor).
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        cfg.volatility.sigma_min
    };

    // Volatility-driven global scalars from engine.
    let spread_mult_base = state.spread_mult;
    let size_mult_base = state.size_mult;

    let mut out = Vec::with_capacity(cfg.venues.len());

    // Global inventory q_t and common parameters.
    let q_t = state.q_global_tao;
    let tau = cfg.mm.quote_horizon_sec;
    let lambda_inv = cfg.mm.lambda_inv;

    // Pair each venue config with its dynamic state by index.
    for (idx, (vcfg, vstate)) in cfg.venues.iter().zip(state.venues.iter()).enumerate() {
        // Per-venue inventory q_v.
        let q_v = vstate.position_tao;

        // Per-venue basis term: mid - global fair.
        let basis_term = if let Some(mid) = vstate.mid {
            mid - s_t
        } else {
            0.0
        };

        // Per-venue funding term: mapped to a skew in price space.
        let funding_skew = funding_to_skew(
            vstate.funding_8h,
            cfg.mm.funding_skew_slope,
            cfg.mm.funding_skew_clip,
        );
        let funding_term = funding_skew * s_t;

        // Simple per-venue inventory target (for now, neutral).
        // This is where liquidity- or funding-driven targets plug in.
        let q_v_star = 0.0;

        // Reservation price (Section 9 skeleton from the whitepaper):
        //
        //   r_v(t) = S_t
        //            + w_B * basis_term_v
        //            + w_F * funding_term_v
        //            - γ_v * σ_eff^2 * τ [ q_t - λ^{-1}(q_v - q_v*) ].
        //
        let reservation = s_t
            + cfg.mm.basis_weight * basis_term
            + cfg.mm.funding_weight * funding_term
            - vcfg.gamma
                * sigma_eff.powi(2)
                * tau
                * (q_t - lambda_inv * (q_v - q_v_star));

        // Avellaneda–Stoikov half-spread at reference vol.
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());
        let delta_vol = delta_as * spread_mult_base;

        // Economic edge requirement.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;
        let min_half =
            ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0).max(delta_vol);

        // Vol-adjusted base size clamped to venue limits.
        let size =
            (vcfg.base_order_size * size_mult_base).min(vcfg.max_order_size).max(0.0);

        let bid = if size > 0.0 {
            Some(SideQuote {
                price: reservation - min_half,
                size,
            })
        } else {
            None
        };

        let ask = if size > 0.0 {
            Some(SideQuote {
                price: reservation + min_half,
                size,
            })
        } else {
            None
        };

        out.push(VenueQuote {
            venue_index: idx,
            venue_id: vcfg.id.clone(),
            bid,
            ask,
        });
    }

    out
}

/// Turn MM quotes into abstract order intents (one order per non-empty side).
pub fn mm_quotes_to_order_intents(quotes: &[VenueQuote]) -> Vec<OrderIntent> {
    let mut intents = Vec::new();

    for vq in quotes {
        if let Some(b) = &vq.bid {
            intents.push(OrderIntent {
                venue_index: vq.venue_index,
                venue_id: vq.venue_id.clone(),
                side: Side::Buy,
                price: b.price,
                size: b.size,
                purpose: OrderPurpose::Mm,
            });
        }

        if let Some(a) = &vq.ask {
            intents.push(OrderIntent {
                venue_index: vq.venue_index,
                venue_id: vq.venue_id.clone(),
                side: Side::Sell,
                price: a.price,
                size: a.size,
                purpose: OrderPurpose::Mm,
            });
        }
    }

    intents
}
