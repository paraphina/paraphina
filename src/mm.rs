// src/mm.rs
//
// Avellaneda–Stoikov style quoting with the whitepaper's additions:
// - global + per-venue inventory,
// - per-venue basis and funding terms in the reservation price,
// - per-venue inventory targets from liquidity + funding preferences,
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

/// Bounded funding-to-skew map φ from the whitepaper:
/// here we use a simple linear map with clipping.
///
/// skew = clip(funding_8h * funding_skew_slope, [-funding_skew_clip, +funding_skew_clip])
fn funding_to_skew(funding_8h: f64, slope: f64, clip_abs: f64) -> f64 {
    let raw = funding_8h * slope;
    raw.max(-clip_abs).min(clip_abs)
}

/// Compute model-based MM quotes per venue.
///
/// This implements the enhanced reservation price from Section 9.1 of the
/// whitepaper (basis + funding + per-venue inventory targets) plus a
/// simplified Avellaneda–Stoikov half-spread and edge check.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Fair value S_t and effective volatility σ_eff.
    let s_t = state.fair_value.unwrap_or(250.0);
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        0.02
    };

    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    // Global inventory q_t.
    let q_t = state.q_global_tao;

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        // Look up the matching venue state to get mid, funding, position, etc.
        let v_state = match state.venues.get(idx) {
            Some(vs) => vs,
            None => continue, // should not happen, but be defensive
        };

        // Per-venue inventory q_v.
        let q_v = v_state.position_tao;

        // Basis term b_{v,t} = m_v - S_t (0 if no mid yet).
        let basis_term = match v_state.mid {
            Some(mid) => mid - s_t,
            None => 0.0,
        };

        // Funding PnL per unit over the quoting horizon τ (in seconds).
        let tau = cfg.mm.quote_horizon_sec; // horizon in seconds
        let tau_hours = tau / 3600.0;
        let funding_term = v_state.funding_8h * (tau_hours / 8.0) * s_t;

        // Per-venue target inventory:
        // q_v^target = w_liq * q_t + w_fund * φ(funding_8h)
        let funding_skew = funding_to_skew(
            v_state.funding_8h,
            cfg.mm.funding_skew_slope,
            cfg.mm.funding_skew_clip,
        );
        let q_v_target = vcfg.w_liq * q_t + vcfg.w_fund * funding_skew;

        // How strongly to anchor to the per-venue target (λ_inv in [0,1]).
        let lambda_inv = cfg.mm.lambda_inv;

        // Enhanced reservation price:
        //
        // S_tilde^{(v)} =
        //   S_t
        //   + β_b * b_{v,t}
        //   + β_f * f_{v,t}
        //   - γ_v σ_eff^2 τ ( q_t - λ_inv (q_v - q_v_target) )
        //
        let reservation = s_t
            + cfg.mm.basis_weight * basis_term
            + cfg.mm.funding_weight * funding_term
            - vcfg.gamma * sigma_eff.powi(2) * tau
                * (q_t - lambda_inv * (q_v - q_v_target));

        // Classical Avellaneda–Stoikov half-spread:
        //
        // δ* = (1/γ) ln(1 + γ/k)
        //
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());
        let delta_vol = delta_as * spread_mult;

        // Economic edge requirement.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;

        // Minimum admissible half-spread.
        let min_half = ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0)
            .max(delta_vol);

        // Volatility-scaled base size, clipped to per-venue max.
        let size_raw = vcfg.base_order_size * size_mult;
        let size = size_raw
            .min(vcfg.max_order_size)
            .max(0.0);

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
