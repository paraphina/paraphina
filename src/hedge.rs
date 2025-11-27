// src/hedge.rs
//
// Simple global hedge engine:
//  - computes desired global delta change ΔH in TAO,
//  - applies a dead-band around zero (hedge_band_base * band_mult),
//  - clips each hedge step to hedge_max_step,
//  - allocates the hedge to the cheapest healthy hedge venue,
//  - converts the plan into abstract OrderIntent structs.

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
}

#[derive(Debug, Clone)]
pub struct HedgePlan {
    /// Desired change in global inventory (TAO). Positive = buy TAO, negative = sell TAO.
    pub desired_delta: f64,
    pub allocations: Vec<HedgeAllocation>,
}

/// Compute a one-step hedge plan from the current state.
///
/// This is a simplified version of the whitepaper LQ controller:
///  - target global inventory is 0,
///  - dead band around 0 to avoid over-hedging,
///  - per-step clip via hedge_max_step,
///  - choose the single cheapest healthy hedge venue by taker fee.
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    // If the global kill switch is on, do not hedge.
    if state.kill_switch {
        return None;
    }

    // Hedge band scaled by the volatility band multiplier.
    let band_raw = cfg.hedge.hedge_band_base;
    let band_mult = state.band_mult.max(1e-9);
    let band = band_raw * band_mult;

    let q_t = state.q_global_tao;

    // Simple target: drive global inventory toward 0.
    let delta_to_target = -q_t;

    // Inside dead band? Do nothing.
    if delta_to_target.abs() <= band {
        return None;
    }

    // Clip per-step size.
    let mut desired_delta = delta_to_target;
    if desired_delta.abs() > cfg.hedge.hedge_max_step {
        desired_delta = cfg.hedge.hedge_max_step * desired_delta.signum();
    }

    // Determine hedge side and absolute size.
    let side = if desired_delta > 0.0 {
        Side::Buy
    } else {
        Side::Sell
    };
    let size_abs = desired_delta.abs();

    // Choose cheapest healthy hedge venue.
    let mut best: Option<(usize, f64)> = None;

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }

        let v_state = &state.venues[idx];
        if v_state.status != VenueStatus::Healthy {
            continue;
        }

        let mid = v_state.mid.or(state.fair_value).unwrap_or(250.0);

        // Approximate per-TAO cost using taker fee.
        let fee_bps = vcfg.taker_fee_bps;
        let cost_per_tao = fee_bps / 10_000.0 * mid;

        match best {
            None => best = Some((idx, cost_per_tao)),
            Some((_, best_cost)) if cost_per_tao < best_cost => {
                best = Some((idx, cost_per_tao))
            }
            _ => {}
        }
    }

    let (venue_index, _best_cost) = match best {
        Some(x) => x,
        None => {
            // No eligible hedge venues.
            return None;
        }
    };

    let vcfg = &cfg.venues[venue_index];
    let v_state = &state.venues[venue_index];
    let mid = v_state.mid.or(state.fair_value).unwrap_or(250.0);

    let allocation = HedgeAllocation {
        venue_index,
        venue_id: vcfg.id.clone(),
        side,
        size: size_abs,
        est_price: mid,
    };

    Some(HedgePlan {
        desired_delta,
        allocations: vec![allocation],
    })
}

/// Turn a hedge plan into abstract order intents.
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
