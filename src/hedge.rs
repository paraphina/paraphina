// src/hedge.rs
//
// Global hedge engine:
//
//  - Uses an LQ controller with (alpha_hedge, beta_hedge) to compute
//    a desired global hedge step ΔH_t (Section 13.1).
//  - Applies a volatility-scaled dead band around zero
//    (hedge_band_base * band_mult).
//  - Clips each hedge step to hedge_max_step.
//  - Allocates the hedge to the cheapest healthy hedge-allowed venue
//    using a simple per-unit cost (taker fee + liquidation-distance penalty).
//  - Converts the plan into abstract OrderIntent structs.

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
    /// Desired change in global inventory (TAO).
    /// Positive = buy TAO (reduce short), negative = sell TAO (reduce long).
    pub desired_delta: f64,
    pub allocations: Vec<HedgeAllocation>,
}

/// Compute a one-step hedge plan from the current state.
///
/// This follows the whitepaper's LQ controller more closely:
///
///   X_t           = q_global_tao
///   k_hedge       = α / (α + β)
///   ΔH_raw        = -k_hedge * X_t
///
/// with a dead band on |X_t| and a global max step on |ΔH|.
/// Allocation is still simplified to a single cheapest venue by
/// (taker fee + liquidation penalty).
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    // If the global kill switch is on, do not hedge.
    if state.kill_switch {
        return None;
    }

    let hedge_cfg = &cfg.hedge;
    let risk_cfg = &cfg.risk;

    // Global exposure X_t = q_t.
    let x_t = state.q_global_tao;

    // Volatility-scaled dead band around |X_t|.
    let band_mult = state.band_mult.max(1e-9);
    let band = hedge_cfg.hedge_band_base * band_mult;

    if x_t.abs() <= band {
        // Inside dead band: do nothing.
        return None;
    }

    // LQ controller gain k_hedge = α / (α + β).
    let alpha = hedge_cfg.alpha_hedge.max(0.0);
    let beta = hedge_cfg.beta_hedge.max(0.0);
    let denom = alpha + beta;
    let k_hedge = if denom > 0.0 { alpha / denom } else { 0.5 };

    // Raw desired hedge step (TAO).
    //
    // We define desired_delta as "change in position via hedge orders":
    // positive => buy TAO (reduce short), negative => sell TAO (reduce long).
    // To *reduce* |X_t|, we move opposite to X_t:
    //   if X_t > 0 (net long), desired_delta < 0 (sell).
    //   if X_t < 0 (net short), desired_delta > 0 (buy).
    let mut desired_delta = -k_hedge * x_t;

    // Global max step cap.
    let max_step = hedge_cfg.hedge_max_step.max(0.0);
    if max_step > 0.0 && desired_delta.abs() > max_step {
        desired_delta = max_step * desired_delta.signum();
    }

    // If the step is effectively zero, don't bother.
    if desired_delta.abs() < 1e-9 {
        return None;
    }

    // Determine hedge side and absolute size.
    let side = if desired_delta > 0.0 {
        Side::Buy
    } else {
        Side::Sell
    };
    let size_abs = desired_delta.abs();

    // -------- Venue selection: simple per-unit cost model --------
    //
    // For each hedge-allowed, Healthy venue, we compute:
    //
    //   cost_per_tao ≈ taker fee (in quote) + liquidation-distance penalty
    //
    // and pick the cheapest.
    let mut best: Option<(usize, f64)> = None;

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }

        let v_state = &state.venues[idx];
        if v_state.status != VenueStatus::Healthy {
            continue;
        }

        // Mid price proxy for hedge estimates.
        let mid = v_state
            .mid
            .or(state.fair_value)
            .unwrap_or(250.0);

        // Taker fee per TAO in quote currency.
        let fee_per_tao = vcfg.taker_fee_bps / 10_000.0 * mid;

        // Liquidation-distance penalty: venues closer to liquidation get penalised.
        let dist = v_state.dist_liq_sigma;
        let liq_penalty = if dist.is_finite() && dist < risk_cfg.liq_warn_sigma {
            // Linear penalty from 0 at liq_warn_sigma to ~1% of price at 0σ.
            let num = (risk_cfg.liq_warn_sigma - dist).max(0.0);
            let den = risk_cfg.liq_warn_sigma.max(1e-9);
            (num / den) * 0.01 * mid
        } else {
            0.0
        };

        let cost_per_tao = fee_per_tao + liq_penalty;

        match best {
            None => best = Some((idx, cost_per_tao)),
            Some((_, best_cost)) if cost_per_tao < best_cost => {
                best = Some((idx, cost_per_tao));
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
    let mid = v_state
        .mid
        .or(state.fair_value)
        .unwrap_or(250.0);

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
