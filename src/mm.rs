// src/mm.rs
//
// Avellaneda–Stoikov style quoting with the whitepaper's additions:
//  - global + per-venue inventory,
//  - per-venue basis and funding terms in the reservation price,
//  - (placeholder) per-venue inventory targets,
//  - volatility scaling of spreads,
//  - quadratic size objective J(Q) = e Q - 0.5 η Q^2,
//  - toxicity + liquidation-distance gating,
//  - simple Warning regime hooks,
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
/// Still “toy” on funding / inventory targets:
///  - basis term = mid_v - S_t,
///  - funding term just passes through v.funding_8h,
///  - per-venue target inventory is 0 for now.
///
/// But the structural pieces match the whitepaper:
///  - enhanced AS reservation price,
///  - volatility-scaled spreads,
///  - quadratic size choice with η,
///  - Warning regime widening + size cap,
///  - toxicity & liquidation-distance gating.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<VenueQuote> {
    // Global kill-switch: no quoting at all.
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

    let risk_cfg = &cfg.risk;
    let tox_cfg = &cfg.toxicity;
    let mm_cfg = &cfg.mm;

    // ----- Global volatility-driven scalars, then Warning regime tweaks -----
    let mut base_spread_mult = state.spread_mult;
    let base_size_mult = state.size_mult;

    // Warning regime: widen spreads as per SPREAD_WARN_MULT.
    if matches!(state.risk_regime, RiskRegime::Warning) {
        base_spread_mult *= risk_cfg.spread_warn_mult.max(1.0);
    }

    let mut out = Vec::with_capacity(cfg.venues.len());

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        let v_state = &state.venues[idx];

        // Skip venues that are fully disabled.
        if v_state.status == VenueStatus::Disabled {
            out.push(VenueQuote {
                venue_index: idx,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
            continue;
        }

        // Toxicity-based behaviour:
        //  - tox < tox_med_threshold  => "healthy"
        //  - tox in [med, high)       => medium toxic: shrink sizes
        //  - tox >= tox_high_threshold=> status would already be Disabled above
        let tox = v_state.toxicity;
        let medium_toxic = tox >= tox_cfg.tox_med_threshold;

        // ----- Start from global scalars, then apply per-venue liq gating -----
        let mut spread_mult = base_spread_mult;
        let mut size_mult = base_size_mult;

        // Liquidation-distance gating in sigma units.
        let mut dist_liq = v_state.dist_liq_sigma;
        if !dist_liq.is_finite() || dist_liq <= 0.0 {
            // Treat non-finite / non-positive as "far away" for now.
            dist_liq = f64::INFINITY;
        }

        let liq_warn = risk_cfg.liq_warn_sigma;
        let liq_crit = risk_cfg.liq_crit_sigma;

        // Inside critical zone: do not quote at all.
        if dist_liq <= liq_crit {
            out.push(VenueQuote {
                venue_index: idx,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
            continue;
        }

        // Warning zone between liq_crit and liq_warn:
        //  - linearly shrink sizes to 0 as dist→liq_crit,
        //  - widen spreads up to ~2x as we approach liq_crit.
        if dist_liq <= liq_warn && liq_warn > liq_crit {
            let t = ((dist_liq - liq_crit) / (liq_warn - liq_crit))
                .max(0.0)
                .min(1.0); // t=1 at warn edge, t→0 near critical

            size_mult *= t;
            spread_mult *= 1.0 + (1.0 - t); // between 1x and 2x
        }

        // -------- Inventory inputs --------
        let q_t = state.q_global_tao;   // global inventory
        let q_v = v_state.position_tao; // per-venue inventory

        // -------- Basis & funding terms --------
        let mid_v = v_state.mid.unwrap_or(s_t);
        let basis_term = mid_v - s_t;

        // funding_8h is dimensionless (rate per 8h); we treat it as a signal for now.
        let funding_term = v_state.funding_8h;

        // Placeholder per-venue inventory target.
        let target_q_v = 0.0;

        // λ_inv from the whitepaper (inventory coupling between global and per-venue).
        let lambda_inv = mm_cfg.lambda_inv;

        // Time horizon for the AS optimisation.
        let tau = mm_cfg.quote_horizon_sec;

        // -------- Reservation price S̃_v(t) --------
        //
        // S̃_v = S_t
        //      + β_b * basis_term
        //      + β_f * funding_term
        //      - γ_v * σ_eff^2 * τ * [ q_t - λ_inv (q_v - q_v^{target}) ].
        let reservation = s_t
            + mm_cfg.basis_weight * basis_term
            + mm_cfg.funding_weight * funding_term
            - vcfg.gamma * sigma_eff.powi(2) * tau * (q_t - lambda_inv * (q_v - target_q_v));

        // -------- Avellaneda–Stoikov half-spread --------
        //
        // δ_AS = (1 / γ_v) * ln(1 + γ_v / k_v).
        let delta_as = (1.0 / vcfg.gamma) * ((1.0 + vcfg.gamma / vcfg.k).ln());

        // Volatility + risk driven spreads.
        let delta_vol = delta_as * spread_mult;

        // -------- Economic edge requirement --------
        //
        // Maker cost near mid, plus volatility protection buffer.
        let maker_cost =
            (vcfg.maker_fee_bps - vcfg.maker_rebate_bps) / 10_000.0 * s_t;
        let v_buf = mm_cfg.edge_vol_mult * sigma_eff * s_t;

        // Minimum half-spread we’re willing to quote.
        let min_half =
            ((mm_cfg.edge_local_min + maker_cost + v_buf) / 2.0).max(delta_vol);

        // -------- Candidate prices (pre-tick) --------
        let mut bid_price = reservation - min_half;
        let mut ask_price = reservation + min_half;

        // Snap to tick grid.
        let tick = vcfg.tick_size.max(1e-6);
        bid_price = (bid_price / tick).floor() * tick;
        ask_price = (ask_price / tick).ceil() * tick;

        // -------- Per-unit edges for bid / ask --------
        //
        // e_bid = (S_t - bid) - maker_cost - v_buf
        // e_ask = (ask - S_t) - maker_cost - v_buf
        let e_bid = (s_t - bid_price) - maker_cost - v_buf;
        let e_ask = (ask_price - s_t) - maker_cost - v_buf;

        // -------- Size choice from quadratic objective --------
        //
        // J(Q) = e Q - 0.5 η Q^2  ⇒  Q_raw = e / η  for e > 0, else 0.
        let eta = mm_cfg.size_eta.max(1e-9);
        let size_mult_local = size_mult; // capture by value for the closure

        let size_from_edge = |e: f64| -> f64 {
            if e <= 0.0 {
                return 0.0;
            }

            // Unconstrained optimum.
            let mut q = e / eta;

            // Volatility & liquidation scaling.
            q *= size_mult_local;

            // Warning risk regime: cap per-order size.
            if matches!(state.risk_regime, RiskRegime::Warning) {
                q = q.min(risk_cfg.q_warn_cap);
            }

            // Medium toxicity: shrink sizes.
            if medium_toxic {
                q *= 0.5;
            }

            // Per-venue max.
            q = q.min(vcfg.max_order_size);

            // No negatives.
            if q < 0.0 {
                q = 0.0;
            }

            q
        };

        let bid_size = size_from_edge(e_bid);
        let ask_size = size_from_edge(e_ask);

        let bid = if bid_size > 0.0 {
            Some(SideQuote {
                price: bid_price,
                size: bid_size,
            })
        } else {
            None
        };

        let ask = if ask_size > 0.0 {
            Some(SideQuote {
                price: ask_price,
                size: ask_size,
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
