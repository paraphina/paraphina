// src/mm.rs
//
// Avellaneda–Stoikov style quoting with the whitepaper's additions:
//  - global + per-venue inventory,
//  - per-venue basis and funding terms in the reservation price,
//  - per-venue inventory targets from liquidity + funding preferences,
//  - volatility scaling of spreads,
//  - edge-proportional size choice (Section 10),
//  - conversion to abstract OrderIntent structs.

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
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
/// Reservation price matches whitepaper 9.1, spread matches 9.2,
/// and size follows Section 10:
///
///   J(Q) = e Q - 0.5 η Q^2  ->  Q_raw = max(0, e / η)
///
/// then scaled and capped.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Global kill-switch disables all quoting.
    if state.kill_switch {
        return Vec::new();
    }

    let risk_cfg = &cfg.risk;

    // Fair value and effective volatility.
    let s_t = state.fair_value.unwrap_or(250.0);

    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        cfg.volatility.sigma_min.max(1e-6)
    };

    // Volatility-driven control scalars from the engine.
    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;

    let mm_cfg = &cfg.mm;
    let eta = mm_cfg.size_eta.max(1e-9);
    let regime = state.risk_regime;

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let v_state = &state.venues[idx];

        // Skip non-healthy venues entirely.
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
        let q_t = state.q_global_tao;     // global inventory
        let q_v = v_state.position_tao;   // per-venue inventory

        // -------- Basis & funding terms --------

        // Basis b_v = m_v - S_t.
        let mid_v = v_state.mid.unwrap_or(s_t);
        let b_v = mid_v - s_t;

        // Funding per unit over horizon τ:
        //   f_v ≈ funding_8h * (τ / 8h) * S_t.
        let funding_8h = v_state.funding_8h;
        let tau = mm_cfg.quote_horizon_sec;
        let eight_hours_sec = 8.0 * 60.0 * 60.0;
        let funding_factor = if eight_hours_sec > 0.0 {
            tau / eight_hours_sec
        } else {
            0.0
        };
        let f_v = funding_8h * funding_factor * s_t;

        // Funding→inventory skew map φ(funding_8h).
        let phi_raw = mm_cfg.funding_skew_slope * funding_8h;
        let phi = phi_raw
            .max(-mm_cfg.funding_skew_clip)
            .min(mm_cfg.funding_skew_clip);

        // Per-venue target inventory:
        //   q_v_target = w_liq * q_t + w_fund * φ.
        let q_v_target = vcfg.w_liq * q_t + vcfg.w_fund * phi;

        // λ_inv ∈ [0,1] controls how strongly we anchor to per-venue targets.
        let lambda_inv = mm_cfg.lambda_inv.clamp(0.0, 1.0);

        // Inventory tilt term inside the AS reservation price.
        let inventory_term = q_t - lambda_inv * (q_v - q_v_target);

        // -------- Enhanced reservation price (whitepaper 9.1) --------
        let reservation = s_t
            + mm_cfg.basis_weight * b_v
            + mm_cfg.funding_weight * f_v
            - vcfg.gamma * sigma_eff.powi(2) * tau * inventory_term;

        // -------- Avellaneda–Stoikov half-spread (whitepaper 9.2) --------
        // δ_AS = (1/γ_v) * ln(1 + γ_v / k_v).
        let delta_as = if vcfg.gamma > 0.0 && vcfg.k > 0.0 {
            (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln())
        } else {
            0.0
        };

        // Volatility scaling.
        let delta_vol = delta_as * spread_mult;

        // -------- Economic edge requirement --------
        //
        // Maker cost near mid:
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;

        // Volatility buffer.
        let v_buf = mm_cfg.edge_vol_mult * sigma_eff * s_t;

        // Minimum half-spread we're willing to quote:
        let min_half = ((mm_cfg.edge_local_min + maker_cost + v_buf) / 2.0)
            .max(delta_vol);

        // -------- Raw quotes, then tick snapping --------
        let tick = vcfg.tick_size.max(1e-9);

        let bid_raw = reservation - min_half;
        let ask_raw = reservation + min_half;

        let bid = (bid_raw / tick).floor() * tick;
        let ask = (ask_raw / tick).ceil() * tick;

        // -------- Helper for edge-based size choice (Section 10) --------
        let make_side = |price: f64, e: f64| -> Option<SideQuote> {
            if price <= 0.0 {
                return None;
            }

            // Require minimum per-unit edge.
            if e < mm_cfg.edge_local_min {
                return None;
            }

            // Unconstrained optimum from J(Q) = e Q - 0.5 η Q^2:
            //   Q_raw = e / η.
            let mut q_raw = e / eta;
            if q_raw <= 0.0 {
                return None;
            }

            // Volatility scaling.
            q_raw *= size_mult;

            // Per-venue hard max.
            let mut q = q_raw.min(vcfg.max_order_size);

            // ---------- Margin-based cap (Section 10, step 3) ----------
            //
            // notional_quote = |Q| * price_side
            // Q_margin_max   = margin_available * MM_MAX_LEVERAGE * MM_MARGIN_SAFETY / price_side
            let margin_avail = v_state.margin_available.max(0.0);
            if margin_avail > 0.0 {
                let q_margin_max = margin_avail
                    * risk_cfg.mm_max_leverage
                    * risk_cfg.mm_margin_safety
                    / (price + 1e-9);

                if q_margin_max > 0.0 {
                    q = q.min(q_margin_max);
                }
            }

            // ---------- Liquidation-distance shrink (Section 10, step 4) ----------
            let dist = v_state.dist_liq_sigma;
            let liq_warn = risk_cfg.liq_warn_sigma;
            let liq_crit = risk_cfg.liq_crit_sigma;

            if dist.is_finite() && dist < liq_warn {
                // Linear shrink from 1 at liq_warn to 0 at liq_crit.
                let num = (dist - liq_crit).max(0.0);
                let den = (liq_warn - liq_crit).max(1e-9);
                let k_liq = (num / den).min(1.0);
                q *= k_liq;

                // If we're inside Critical distance, k_liq will be 0.
            }

            // ---------- Risk-regime cap (Warning) ----------
            if let RiskRegime::Warning = regime {
                q = q.min(risk_cfg.q_warn_cap);
            }

            // Simple minimum: don't bother with dust.
            let min_size = (vcfg.base_order_size * 0.25).max(1e-6);
            if q < min_size {
                return None;
            }

            Some(SideQuote { price, size: q })
        };

        // -------- Per-side local edges and gating --------
        let e_bid = s_t - bid - maker_cost - v_buf;
        let e_ask = ask - s_t - maker_cost - v_buf;

        let bid_side = make_side(bid, e_bid);
        let ask_side = make_side(ask, e_ask);

        out.push(VenueQuote {
            venue_index: idx,
            venue_id: vcfg.id.clone(),
            bid: bid_side,
            ask: ask_side,
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
