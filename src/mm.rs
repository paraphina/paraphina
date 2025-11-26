// src/mm.rs
//
// Market-making model:
// - Enhanced Avellaneda–Stoikov reservation price per venue
// - Basis & funding adjustments
// - Per-venue inventory targets
// - Conversion to abstract order intents
//
// This corresponds to the quoting logic in Sections 9 & 10 of the whitepaper.

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

/// Compute model-based quotes per venue using the enhanced AS model.
///
/// This uses:
/// - global inventory q_t from `GlobalState`
/// - per-venue inventory q_v,
/// - per-venue basis b_v = mid_v − S_t,
/// - per-venue funding via a simple φ(funding) skew,
/// - per-venue inventory targets q_v,target.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Fair value S_t; fall back to 250 if not initialised for some reason.
    let s_t = state.fair_value.unwrap_or(250.0);

    // Effective volatility with a small floor for demo.
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        0.02
    };

    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    // Global inventory q_t.
    let q_t = state.q_global_tao;

    // Horizon for AS and funding, in seconds.
    let tau = cfg.mm.quote_horizon_sec;
    let eight_hours_sec = 8.0 * 60.0 * 60.0;

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[idx];

        // Per-venue position q_v.
        let q_v = vstate.position_tao;

        // ----- Basis term: b_v = mid_v - S_t -----
        let mid_v = vstate.mid.unwrap_or(s_t);
        let b_v = mid_v - s_t;
        let basis_term = cfg.mm.basis_weight * b_v;

        // ----- Funding term: expected funding PnL per unit over horizon tau -----
        //
        // funding_8h is the 8h funding rate. Expected funding over horizon tau:
        // f_v ≈ funding_8h * (tau / 8h) * S_t
        let f_raw = vstate.funding_8h * (tau / eight_hours_sec) * s_t;
        let funding_term = cfg.mm.funding_weight * f_raw;

        // ----- Per-venue inventory target q_v,target -----
        //
        // φ(funding) = clip(slope * funding_8h, -clip, +clip).
        let phi = {
            let x = cfg.mm.funding_skew_slope * vstate.funding_8h;
            x.clamp(-cfg.mm.funding_skew_clip, cfg.mm.funding_skew_clip)
        };
        let q_target_v = vcfg.w_liq * q_t + vcfg.w_fund * phi;

        let lambda_inv = cfg.mm.lambda_inv;

        // Enhanced reservation price from the whitepaper:
        //
        // S_t
        // + β_b * b_v
        // + β_f * f_v
        // - γ_v * σ_eff^2 * τ * ( q_t - λ_inv (q_v - q_v,target) )
        let reservation = s_t
            + basis_term
            + funding_term
            - vcfg.gamma * sigma_eff.powi(2) * tau * (q_t - lambda_inv * (q_v - q_target_v));

        // ----- AS half-spread with vol scaling -----
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());
        let delta_vol = delta_as * spread_mult;

        // ----- Economic edge requirement -----
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;

        let min_half = ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0)
            .max(delta_vol);

        // ----- Size choice (simple; later we can plug in full Section 10 logic) -----
        let size = (vcfg.base_order_size * size_mult)
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
