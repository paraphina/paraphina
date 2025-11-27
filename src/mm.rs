// src/mm.rs
//
// Avellaneda–Stoikov style market making with:
//  - global + per-venue inventory tilt,
//  - basis- and funding-aware reservation price,
//  - conversion to abstract order intents.

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

/// Compute model-based quotes per venue.
///
/// This is the whitepaper Section 9 skeleton:
///   r_v(t) = S_t
///            + w_B * b_v(t)
///            + w_F * f_v(t)
///            - γ_v * σ_eff^2 * τ * [ q_t - λ^{-1}(q_v - q_v^*) ]
///
/// with an Avellaneda–Stoikov half-spread and economic edge floor.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // In Critical regime we do not post new MM orders (hedge engine may still act).
    if state.kill_switch {
        return Vec::new();
    }

    let s_t = state.fair_value.unwrap_or(250.0);

    // Effective volatility used both for inventory tilt and spread scaling.
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        0.02
    };

    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    debug_assert_eq!(
        cfg.venues.len(),
        state.venues.len(),
        "Config venues and state venues must be aligned"
    );

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let v_state = &state.venues[idx];

        // -------- Inventory terms (global + per-venue) --------
        let q_t = state.q_global_tao;
        let q_v = v_state.position_tao;

        // Target per-venue inventory. For now, neutral.
        let target_q_v = 0.0;
        let lambda_inv = cfg.mm.lambda_inv;

        // Time horizon (seconds) for AS and inventory-risk term.
        let tau = cfg.mm.quote_horizon_sec;

        // -------- Basis term b_v(t) --------
        // Work in price-space: b_v = mid_v - S_t, falling back to 0 if no mid.
        let basis_term = match v_state.mid {
            Some(mid) => mid - s_t,
            None => 0.0,
        };

        // -------- Funding term f_v(t) --------
        //
        // We map the 8h funding rate into an effective price tilt via:
        //   f_raw = funding_8h * funding_skew_slope
        //   f_v   = clip(f_raw, -funding_skew_clip, +funding_skew_clip)
        //
        // so a positive funding (we pay to be long) tilts us slightly short, and
        // negative funding tilts us slightly long, up to the configured clip.
        let f_raw = v_state.funding_8h * cfg.mm.funding_skew_slope;
        let funding_term = f_raw
            .max(-cfg.mm.funding_skew_clip)
            .min(cfg.mm.funding_skew_clip);

        // -------- Reservation price r_v(t) --------
        let inventory_term =
            vcfg.gamma * sigma_eff.powi(2) * tau * (q_t - lambda_inv * (q_v - target_q_v));

        let reservation = s_t
            + cfg.mm.basis_weight * basis_term
            + cfg.mm.funding_weight * funding_term
            - inventory_term;

        // -------- Half-spread (Avellaneda–Stoikov) --------
        //
        // δ_AS = (1/γ_v) * ln(1 + γ_v / k_v)
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());
        let delta_vol = delta_as * spread_mult;

        // -------- Economic edge floor --------
        //
        // We require that the total half-spread also covers:
        //   - net maker fee cost,
        //   - volatility buffer scaled by σ_eff,
        //   - local absolute edge floor.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = cfg.mm.edge_vol_mult * sigma_eff * s_t;

        let min_half = ((cfg.mm.edge_local_min + maker_cost + v_buf) / 2.0)
            .max(delta_vol);

        // -------- Quote size --------
        //
        // Base size is volatility-scaled and capped by per-venue max.
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
