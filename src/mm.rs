// src/mm.rs
//
// Avellaneda–Stoikov style quoting with the whitepaper's additions:
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

/// Compute model-based quotes per venue.
///
/// This is deliberately still “toy” on the funding/basis side:
///  - per-venue basis = mid_v - S_t (if we have a mid),
///  - funding term just passes through v.funding_8h for now,
/// but the structural hooks are in place to match the whitepaper later.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // If the global kill-switch is on, do not quote at all.
    if state.kill_switch {
        return Vec::new();
    }

    // Fair value and effective volatility.
    let s_t = state.fair_value.unwrap_or(250.0);
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        0.02
    };

    // Volatility-driven control scalars from the engine.
    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let v_state = &state.venues[idx];

        // Skip venues that are not Healthy: no quoting, just return empty sides.
        if v_state.status != VenueStatus::Healthy {
            out.push(VenueQuote {
                venue_index: idx,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
            continue;
        }

        // -------- Inventory inputs --------
        let q_t = state.q_global_tao;          // global inventory
        let q_v = v_state.position_tao;        // per-venue inventory

        // -------- Basis & funding hooks --------
        let mid_v = v_state.mid.unwrap_or(s_t);
        let basis_term = mid_v - s_t;          // very simple proxy for local basis

        // funding_8h is dimensionless (rate per 8h); we just pass it through
        // as a “signal” for now. Whitepaper can refine this into a dollar term.
        let funding_term = v_state.funding_8h;

        // Per-venue inventory target (right now just 0; later from liquidity/funding prefs).
        let target_q_v = 0.0;

        // λ^{-1} from the whitepaper (inventory coupling between global and per-venue).
        let lambda_inv = cfg.mm.lambda_inv;

        // Time horizon for the AS optimisation.
        let tau = cfg.mm.quote_horizon_sec;

        // -------- Reservation price r_v(t) --------
        //
        // r_v(t) = S_t
        //        + w_B * basis_term
        //        + w_F * funding_term
        //        - γ_v * σ_eff^2 * τ * [ q_t - λ^{-1}(q_v - q_v*) ].
        let reservation = s_t
            + cfg.mm.basis_weight * basis_term
            + cfg.mm.funding_weight * funding_term
            - vcfg.gamma * sigma_eff.powi(2) * tau * (q_t - lambda_inv * (q_v - target_q_v));

        // -------- Avellaneda–Stoikov half-spread --------
        //
        // δ_AS = (1 / γ_v) * ln(1 + γ_v / k_v).
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());

        // Volatility scaling of spreads.
        let delta_vol = delta_as * spread_mult;

        // -------- Economic edge requirement --------
        //
        // We want to cover:
        //  - maker fees (net of rebates),
        //  - a volatility “safety buffer” proportional to σ_eff * S_t,
        //  - a small local hard minimum edge.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;

        // Minimum half-spread we’re willing to quote.
        let min_half =
            ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0).max(delta_vol);

        // -------- Quote size --------
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
