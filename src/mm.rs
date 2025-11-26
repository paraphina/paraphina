// src/mm.rs
//
// Simplified Avellaneda–Stoikov style quotes + conversion to order intents.
// This version already respects:
//   - global inventory q_t (skews all venues),
//   - per-venue inventory q_v (skews each venue),
//   - a simple basis term b_v = mid_v - S_t (if mid is known).

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

/// Compute model-based quotes per venue (simplified AS + basis/inventory).
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Fair value and volatility with safe fallbacks.
    let s_t = state.fair_value.unwrap_or(250.0);
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        0.02
    };

    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[idx];

        // Global + per-venue inventory.
        let q_t = state.q_global_tao;
        let q_v = vstate.position_tao;

        let tau = cfg.mm.quote_horizon_sec;

        // Basis term: mid_v - S_t, if we have a mid.
        let basis_term = vstate.mid.map(|m| m - s_t).unwrap_or(0.0);

        // Funding term: left as 0 for now until we wire real funding.
        let funding_term = 0.0;

        // Per-venue target inventory q_v^target = w_liq * q_t + w_fund * φ(funding).
        // For now, φ(funding) = 0.0, so target = w_liq * q_t.
        let target_q_v = vcfg.w_liq * q_t;
        let lambda_inv = cfg.mm.lambda_inv;

        // Enhanced reservation price:
        //
        // \tilde{S}_t^{(v)} =
        //   S_t
        //   + β_b * basis_term
        //   + β_f * funding_term
        //   - γ_v σ_eff^2 τ ( q_t - λ_inv (q_v - q_v^target) )
        //
        let reservation = s_t
            + cfg.mm.basis_weight * basis_term
            + cfg.mm.funding_weight * funding_term
            - vcfg.gamma * sigma_eff.powi(2) * tau * (q_t - lambda_inv * (q_v - target_q_v));

        // AS half-spread.
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());
        let delta_vol = delta_as * spread_mult;

        // Economic edge requirement.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;
        let min_half = ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0).max(delta_vol);

        // Vol-scaled size with per-venue caps.
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
