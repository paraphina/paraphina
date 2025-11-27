// src/hedge.rs
//
// Global hedge engine: decide desired change in net TAO exposure ΔH,
// then allocate that change across eligible venues based on an
// approximate cost model (fees + basis + rough liquidity penalty).
//
// This corresponds to the "global hedge allocator" in the whitepaper.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side, VenueStatus};

#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub size: f64,
    pub est_price: f64,
    pub est_cost_per_tao: f64,
}

#[derive(Debug, Clone)]
pub struct HedgePlan {
    /// Desired global change in TAO inventory (ΔH).
    /// If negative, we want to reduce q_t (sell); if positive, increase q_t (buy).
    pub desired_delta: f64,
    pub allocations: Vec<HedgeAllocation>,
}

/// Compute a one-step hedge plan:
///  1) Decide desired ΔH using an LQ-style rule + dead-band.
///  2) Allocate |ΔH| across eligible venues by approximate cost.
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    // If kill switch is active, don't open new hedges here.
    if state.kill_switch {
        return None;
    }

    let q_t = state.q_global_tao;

    // Volatility-scaled hedge band.
    let band_base = cfg.hedge.hedge_band_base;
    let band_mult = state.band_mult.max(0.25); // avoid zero
    let band = band_base * band_mult;

    if q_t.abs() <= band {
        // Inside dead-band: no hedge.
        return None;
    }

    // ----- Step 1: decide desired ΔH -----
    //
    // ΔH = -α * q_t  (simple linear rule towards 0),
    // then clipped to [-hedge_max_step, +hedge_max_step].
    let mut desired_delta = -cfg.hedge.alpha_hedge * q_t;

    // Ensure we move at least in the right direction if α is small.
    if desired_delta.abs() < 1e-6 {
        desired_delta = -cfg.hedge.beta_hedge * q_t.signum() * band;
    }

    let max_step = cfg.hedge.hedge_max_step.max(band);
    if desired_delta > max_step {
        desired_delta = max_step;
    } else if desired_delta < -max_step {
        desired_delta = -max_step;
    }

    let side = if desired_delta < 0.0 {
        // Negative ΔH => reduce q_t => sell.
        Side::Sell
    } else {
        // Positive ΔH => increase q_t => buy.
        Side::Buy
    };

    let total_size = desired_delta.abs();
    if total_size < 1e-6 {
        return None;
    }

    // ----- Step 2: build candidate venues -----

    let s_t = state.fair_value.unwrap_or(250.0);

    #[derive(Debug)]
    struct Candidate {
        venue_index: usize,
        venue_id: String,
        cost_per_tao: f64,
        max_size: f64,
        mid: f64,
        spread: f64,
    }

    let mut cands: Vec<Candidate> = Vec::new();

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }

        let vstate = &state.venues[idx];

        if vstate.status != VenueStatus::Healthy {
            continue;
        }

        let mid = match vstate.mid {
            Some(m) => m,
            None => continue,
        };

        if vstate.depth_near_mid <= 0.0 {
            continue;
        }

        // Approximate spread; fall back to several ticks if missing.
        let spread = vstate.spread.unwrap_or(10.0 * vcfg.tick_size.max(1e-9));

        let taker_fee_rate = vcfg.taker_fee_bps / 10_000.0;

        // trade_sign = +1 for buy, -1 for sell.
        let trade_sign = if matches!(side, Side::Buy) { 1.0 } else { -1.0 };

        // Basis term: mid - S_t.
        // For buys we prefer mid < S_t, for sells we prefer mid > S_t.
        // So we add trade_sign * basis into the cost (negative is good).
        let basis = mid - s_t;
        let basis_cost = trade_sign * basis;

        // Very rough liquidity penalty: smaller when depth is large.
        let liq_penalty = 0.5 * mid / (vstate.depth_near_mid + 1e-9);

        // Fee cost per TAO at mid.
        let fee_cost = taker_fee_rate * mid;

        let cost_per_tao = fee_cost + liq_penalty + basis_cost;

        // Max allocation on this venue per tick: a fraction of displayed depth.
        let max_size = 0.3 * vstate.depth_near_mid; // 30% of depth, arbitrary but safe

        if max_size <= 0.0 {
            continue;
        }

        cands.push(Candidate {
            venue_index: idx,
            venue_id: vcfg.id.clone(),
            cost_per_tao,
            max_size,
            mid,
            spread,
        });
    }

    if cands.is_empty() {
        return None;
    }

    // Sort by ascending cost: lowest-cost venues allocated first.
    cands.sort_by(|a, b| {
        a.cost_per_tao
            .partial_cmp(&b.cost_per_tao)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ----- Step 3: greedy allocation of |ΔH| across venues -----

    let mut remaining = total_size;
    let mut allocations: Vec<HedgeAllocation> = Vec::new();

    for c in &cands {
        if remaining <= 0.0 {
            break;
        }

        let sz = remaining.min(c.max_size);
        if sz <= 0.0 {
            continue;
        }

        // Approximate trade price: mid ± spread/2 depending on side.
        let est_price = if matches!(side, Side::Buy) {
            c.mid + 0.5 * c.spread
        } else {
            c.mid - 0.5 * c.spread
        };

        allocations.push(HedgeAllocation {
            venue_index: c.venue_index,
            venue_id: c.venue_id.clone(),
            side,
            size: sz,
            est_price,
            est_cost_per_tao: c.cost_per_tao,
        });

        remaining -= sz;
    }

    if allocations.is_empty() {
        return None;
    }

    Some(HedgePlan {
        desired_delta,
        allocations,
    })
}

/// Convert a HedgePlan into abstract order intents.
pub fn hedge_plan_to_order_intents(plan: &HedgePlan) -> Vec<OrderIntent> {
    let mut intents = Vec::new();

    for alloc in &plan.allocations {
        intents.push(OrderIntent {
            venue_index: alloc.venue_index,
            venue_id: alloc.venue_id.clone(),
            side: alloc.side,
            price: alloc.est_price,
            size: alloc.size,
            purpose: OrderPurpose::Hedge,
        });
    }

    intents
}
