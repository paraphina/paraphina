// src/mm.rs
//
// Avellaneda–Stoikov style quoting with the whitepaper's additions:
//
//  - global + per-venue inventory,
//  - per-venue basis and funding terms in the reservation price,
//  - per-venue inventory targets from liquidity + funding preferences,
//  - simple volatility scaling of spreads,
//  - conversion to abstract OrderIntent structs.
//
/// Avellaneda–Stoikov style market making with:
///  - global + per-venue inventory tilt,
///  - basis- and funding-aware reservation price,
///  - conversion to abstract order intents.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side, VenueStatus};

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
    raw.max(-clip_abs).min(clip_abs)
}

/// Compute model-based MM quotes per venue.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Fair value S_t and effective sigma.
    let s_t = state.fair_value.unwrap_or(250.0);

    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        // Fallback to config floor if sigma_eff not yet initialised.
        cfg.volatility.sigma_min
    };

    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let v_state = &state.venues[idx];

        // ------------- Toxicity / health gating -------------
        // If venue is not Healthy, do not quote there.
        if v_state.status != VenueStatus::Healthy {
            out.push(VenueQuote {
                venue_index: idx,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
            continue;
        }

        // ------------- Inventory terms -------------

        // Global inventory q_t (already in state).
        let q_t = state.q_global_tao;
        // Per-venue inventory q_v.
        let q_v = v_state.position_tao;
        // Per-venue inventory target q_v* (for now 0; later can depend on
        // liquidity, funding, venue preferences).
        let q_v_star = 0.0;

        let tau = cfg.mm.quote_horizon_sec;
        let lambda_inv = cfg.mm.lambda_inv;

        // ------------- Basis & funding terms -------------

        // b_v(t) = mid_v - S_t  (per-venue basis, USD per TAO).
        let basis_term = match v_state.mid {
            Some(mid) => mid - s_t,
            None => 0.0,
        };

        // Funding skew φ(f_v) in [-clip,+clip].
        let funding_skew = funding_to_skew(
            v_state.funding_8h,
            cfg.mm.funding_skew_slope,
            cfg.mm.funding_skew_clip,
        );
        // Turn funding skew into a price shift term. This is a simplified
        // version of the whitepaper: positive skew pushes the reservation
        // price up, negative skew down.
        let funding_term = funding_skew * s_t;

        // ------------- Reservation price r_v(t) -------------

        // Whitepaper Section 9 skeleton:
        //
        //   r_v(t) = S_t
        //         + w_B * b_v(t)
        //         + w_F * f_v(t)
        //         + v_v * σ_eff^2 * τ [ q_t - λ^{-1}(q_v - q_v*) ]
        //
        // where v_v is identified with gamma for that venue.
        let inventory_term =
            -vcfg.gamma * sigma_eff.powi(2) * tau * (q_t - lambda_inv * (q_v - q_v_star));

        let reservation = s_t
            + cfg.mm.basis_weight * basis_term
            + cfg.mm.funding_weight * funding_term
            + inventory_term;

        // ------------- Half-spread & economic edge -------------

        // Avellaneda–Stoikov half-spread.
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());
        let delta_vol = delta_as * spread_mult;

        // Economic edge requirement:
        //  - maker/taker fees and rebates,
        //  - volatility buffer proportional to σ_eff * S_t.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;

        let min_half = ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0).max(delta_vol);

        // ------------- Order sizes -------------

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
