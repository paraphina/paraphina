// src/hedge.rs
//
// Global hedge engine for Paraphina.
//
// Responsibilities (whitepaper Sections 9–11):
//   - Look at global inventory q_t and dollar delta.
//   - Apply a volatility / band based LQ-style controller to decide
//     a desired change in TAO exposure ΔH.
//   - Allocate ΔH across a subset of venues based on expected
//     execution cost (spread + fees + toxicity penalty).
//   - Convert the resulting allocations into abstract hedge
//     OrderIntents consumed by main.rs.

use crate::config::{Config, VenueConfig};
use crate::state::{GlobalState, RiskRegime};
use crate::types::{OrderIntent, OrderPurpose, Side, VenueStatus};

/// One venue-level hedge slice in TAO.
#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub size: f64,
    pub est_price: f64,
}

/// Global hedge decision for this tick.
#[derive(Debug, Clone)]
pub struct HedgePlan {
    /// Desired change in global inventory q_t in TAO
    /// (positive = buy TAO, negative = sell TAO).
    pub desired_delta: f64,
    pub allocations: Vec<HedgeAllocation>,
}

/// Main hedge planner.
///
/// Returns `None` when no hedge action is required for this tick.
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    // 0) Global guards: no hedging without a fair value or when kill switch is active.
    if state.kill_switch {
        return None;
    }

    let _s_t = match state.fair_value {
        Some(v) if v > 0.0 => v,
        _ => return None,
    };

    let hedge_cfg = &cfg.hedge;

    // 1) Compute dynamic hedge band around 0 inventory.
    //
    //    band_t = hedge_band_base * band_mult(t)
    //
    // band_mult(t) is already computed in the engine as a function
    // of volatility (Section 6).
    let band_mult = state.band_mult.max(0.1);
    let band = (hedge_cfg.hedge_band_base.max(0.0) * band_mult).max(0.0);

    let q_t = state.q_global_tao;
    let q_abs = q_t.abs();

    // Inside the band => do nothing.
    if band <= 0.0 || q_abs <= band {
        return None;
    }

    // 2) LQ-style controller for desired ΔH in TAO.
    //
    // We hedge only the excess over the band, smoothed by alpha_hedge:
    //
    //   excess  = |q_t| - band
    //   ΔH_raw  = -sign(q_t) * excess * alpha_hedge
    //
    // Risk regime shrinks the step in Warning.
    let excess = q_abs - band;
    let direction = -q_t.signum(); // reduce inventory towards 0

    let alpha = hedge_cfg.alpha_hedge.clamp(0.0, 1.0);
    let mut desired_delta = direction * excess * alpha;

    let regime_scale = match state.risk_regime {
        RiskRegime::Normal => 1.0,
        RiskRegime::Warning => 0.5,
        // In HardLimit we keep full-strength hedging (we want to reduce risk,
        // not slow it down). Kill switch still disables hedging entirely.
        RiskRegime::HardLimit => 1.0,
    };

    desired_delta *= regime_scale;

    // Clip to per-tick max step.
    let max_step = hedge_cfg.hedge_max_step.max(0.0);
    if max_step > 0.0 && desired_delta.abs() > max_step {
        desired_delta = desired_delta.signum() * max_step;
    }

    // If the result is tiny, skip hedging.
    if desired_delta.abs() < 1e-6 {
        return None;
    }

    // 3) Build list of candidate venues with an expected per-unit
    //    execution cost (spread + fees, scaled by toxicity).
    //
    //    cost_v ≈ (half_spread_v + taker_fee_abs_v) * (1 + β * tox_v)
    //
    // We will allocate ΔH in proportion to 1 / cost_v.
    let mut candidates: Vec<(usize, &VenueConfig, f64, f64, f64)> = Vec::new();
    let beta_tox = hedge_cfg.beta_hedge.max(0.0);

    for (i, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[i];

        if !vcfg.is_hedge_allowed {
            continue;
        }
        if vstate.status != VenueStatus::Healthy {
            continue;
        }

        let mid = match vstate.mid {
            Some(m) if m > 0.0 => m,
            _ => continue,
        };

        // Local spread: prefer venue spread, otherwise fall back to a small synthetic.
        let spread = vstate
            .spread
            .unwrap_or(2.0 * vcfg.tick_size.max(1e-6).max(0.01));
        let half_spread = (spread / 2.0).max(vcfg.tick_size.max(1e-6));

        // Per-unit taker fee in USD.
        let fee_abs = (vcfg.taker_fee_bps / 10_000.0).abs() * mid;

        // Toxicity penalty (>= 1.0).
        let tox_penalty = 1.0 + beta_tox * vstate.toxicity.max(0.0);

        let cost = (half_spread + fee_abs) * tox_penalty;
        if cost <= 0.0 {
            continue;
        }

        let weight = 1.0 / cost;
        candidates.push((i, vcfg, mid, half_spread, weight));
    }

    if candidates.is_empty() {
        return None;
    }

    let weight_sum: f64 = candidates.iter().map(|(_, _, _, _, w)| w).sum();
    if weight_sum <= 0.0 {
        return None;
    }

    // 4) Turn desired_delta into per-venue hedge orders.
    let mut allocations: Vec<HedgeAllocation> = Vec::new();
    let side = if desired_delta > 0.0 {
        Side::Buy
    } else {
        Side::Sell
    };
    let total_size = desired_delta.abs();

    for (idx, vcfg, mid, half_spread, weight) in candidates {
        let frac = weight / weight_sum;
        let mut size = total_size * frac;

        // Clip to per-venue max order size.
        size = size.min(vcfg.max_order_size.max(0.0));
        if size < 1e-4 {
            continue;
        }

        // Round size to a sensible precision to avoid silly decimals.
        size = (size * 1_000_000.0).round() / 1_000_000.0;

        // Execution price: mid ± half_spread (we assume taker).
        let price = match side {
            Side::Buy => mid + half_spread,
            Side::Sell => mid - half_spread,
        };

        allocations.push(HedgeAllocation {
            venue_index: idx,
            venue_id: vcfg.id.clone(),
            side,
            size,
            est_price: price,
        });
    }

    if allocations.is_empty() {
        return None;
    }

    Some(HedgePlan {
        desired_delta,
        allocations,
    })
}

/// Convert a hedge plan into abstract OrderIntents.
pub fn hedge_plan_to_order_intents(plan: &HedgePlan) -> Vec<OrderIntent> {
    let mut out = Vec::new();

    for alloc in &plan.allocations {
        out.push(OrderIntent {
            venue_index: alloc.venue_index,
            venue_id: alloc.venue_id.clone(),
            side: alloc.side,
            price: alloc.est_price,
            size: alloc.size,
            purpose: OrderPurpose::Hedge,
        });
    }

    out
}

pub fn compute_hedge_orders(cfg: &Config, state: &GlobalState) -> Vec<OrderIntent> {
    match compute_hedge_plan(cfg, state) {
        Some(plan) => hedge_plan_to_order_intents(&plan),
        None => Vec::new(),
    }
}
