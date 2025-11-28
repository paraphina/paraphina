// src/hedge.rs
//
// Global hedge engine for Paraphina.
//
// Responsibilities (matching the whitepaper):
//   1) Look at global inventory q_t (TAO) and dollar delta.
//   2) Define an effective hedge band around 0 inventory.
//   3) If |q_t| is inside the band -> no hedge.
//   4) If outside the band -> compute a desired ΔH_t (in TAO) using
//      a simple LQ-style controller with (alpha_hedge, beta_hedge)
//      and cap it by hedge_max_step.
//   5) Allocate that ΔH_t across hedge-eligible venues based on a
//      simple cost heuristic (currently: taker fee).
//   6) Convert the plan into abstract OrderIntents (side, size, price).

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side, VenueStatus};

/// A single hedge allocation on a specific venue.
#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    pub venue_id: String,
    pub venue_index: usize,
    pub side: Side,
    /// Size in TAO.
    pub size: f64,
    /// Estimated execution price (USD per TAO).
    pub est_price: f64,
}

/// High-level hedge plan for this tick.
#[derive(Debug, Clone)]
pub struct HedgePlan {
    /// Desired global ΔH in TAO (positive = buy TAO, negative = sell).
    pub desired_delta: f64,
    /// Concrete venue allocations that implement ΔH.
    pub allocations: Vec<HedgeAllocation>,
}

/// Compute a global hedge plan for the current state.
///
/// Returns:
///   - Some(HedgePlan) if we want to take hedge action,
///   - None if we remain inside the hedge band or cannot hedge safely.
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    let hedge_cfg = &cfg.hedge;

    // If band/step are misconfigured, do nothing.
    if hedge_cfg.hedge_band_base <= 0.0 || hedge_cfg.hedge_max_step <= 0.0 {
        return None;
    }

    // ---------- 1) Read global inventory & fair value ----------
    let q_t = state.q_global_tao;
    let s_t = match state.fair_value {
        Some(s) if s > 0.0 => s,
        _ => {
            // Without a meaningful fair value we can't reason about
            // dollar delta or hedge costs properly.
            return None;
        }
    };

    // Dollar delta in USD – used for the β_hedge term.
    let dollar_delta = state.dollar_delta_usd;

    // ---------- 2) Effective hedge band ----------
    //
    // Base band is in TAO (hedge_band_base).
    // We modulate it by the current band_mult, which already folds in
    // volatility & risk-regime information (Section 6 / 14).
    let mut band_eff = hedge_cfg.hedge_band_base * state.band_mult;

    // Clamp the effective band to something sensible.
    if band_eff < 1e-6 {
        band_eff = 1e-6;
    }

    // If we are inside the band, we intentionally do nothing.
    if q_t.abs() <= band_eff {
        return None;
    }

    // ---------- 3) LQ-style controller for desired ΔH ----------
    //
    // raw_step ≈ α * (|q_t| - band_eff) + β * (dollar_delta / S_t)
    // Then we apply the sign of q_t to decide buy/sell.
    let alpha = hedge_cfg.alpha_hedge.max(0.0);
    let beta = hedge_cfg.beta_hedge.max(0.0);

    // Inventory term: only the "excess" beyond the band.
    let inv_term = (q_t.abs() - band_eff).max(0.0);

    // Dollar term: convert dollar delta back into TAO units.
    let dollar_term_tao = dollar_delta / s_t;

    // Combine the two contributions.
    let controller_step = alpha * inv_term + beta * dollar_term_tao.abs();

    // Cap per-step hedge size.
    let step_capped = controller_step
        .min(hedge_cfg.hedge_max_step)
        .max(0.0);

    if step_capped <= 0.0 {
        return None;
    }

    // Direction: if q_t > 0 we are long TAO and want to SELL.
    //            if q_t < 0 we are short TAO and want to BUY.
    let sign = if q_t >= 0.0 { -1.0 } else { 1.0 };
    let desired_delta = sign * step_capped;

    // ---------- 4) Choose hedge venue(s) ----------
    //
    // For now we implement a simple cost-based choice:
    //   - Only venues with is_hedge_allowed = true
    //   - Only venues that are currently Healthy
    //   - Cost proxy = taker_fee_bps (lower is better)
    //
    // This is deliberately simple but gives us a clean hook to later
    // integrate a full cost model (fees + slippage + funding + basis).
    let mut best_idx: Option<usize> = None;
    let mut best_cost: f64 = f64::INFINITY;

    for (i, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }

        // Skip venues that are not Healthy from the toxicity engine.
        if state.venues[i].status != VenueStatus::Healthy {
            continue;
        }

        // Simple cost proxy: taker fee in bps (clip to non-negative).
        let fee_bps = vcfg.taker_fee_bps.max(0.0);

        if fee_bps < best_cost {
            best_cost = fee_bps;
            best_idx = Some(i);
        }
    }

    let best_idx = match best_idx {
        Some(i) => i,
        None => {
            // No viable hedge venue – better to do nothing than hedge
            // into a disabled / extremely expensive venue.
            return None;
        }
    };

    // ---------- 5) Build allocations ----------
    let vcfg = &cfg.venues[best_idx];
    let vstate = &state.venues[best_idx];

    let side = if desired_delta > 0.0 {
        Side::Buy
    } else {
        Side::Sell
    };

    let size = desired_delta.abs();

    // vstate.mid is Option<f64>; fall back to S_t if missing/non-positive.
    let est_price = match vstate.mid {
        Some(m) if m > 0.0 => m,
        _ => s_t,
    };

    let allocation = HedgeAllocation {
        venue_id: vcfg.id.clone(),
        venue_index: best_idx,
        side,
        size,
        est_price,
    };

    Some(HedgePlan {
        desired_delta,
        allocations: vec![allocation],
    })
}

/// Convert a HedgePlan into a list of abstract OrderIntents.
///
/// These intents are "taker-style" by design: they represent
/// immediate execution orders to rebalance risk.
pub fn hedge_plan_to_order_intents(plan: &HedgePlan) -> Vec<OrderIntent> {
    plan.allocations
        .iter()
        .map(|alloc| OrderIntent {
            venue_id: alloc.venue_id.clone(),
            venue_index: alloc.venue_index,
            side: alloc.side,
            size: alloc.size,
            price: alloc.est_price,
            purpose: OrderPurpose::Hedge,
        })
        .collect()
}
