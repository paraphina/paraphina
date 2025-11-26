// src/hedge.rs
//
// Global hedge engine (Section 13 of the whitepaper).
// - LQ control with dead band over global inventory q_t.
// - Volatility-scaled band.
// - Simple cost-aware allocation across hedge-allowed venues.
// - Conversion into abstract OrderIntent structs.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side};

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
    /// Desired global hedge step ΔH_t in TAO (same sign as q_t).
    pub desired_delta: f64,
    /// Per-venue allocations that approximately sum to |ΔH_t|.
    pub allocations: Vec<HedgeAllocation>,
}

/// Compute a single hedge step for the current state.
///
/// Implements:
///   - volatility-scaled dead band around q_t,
///   - LQ controller ΔH_t = k_hedge * q_t,
///   - global cap HEDGE_MAX_STEP,
///   - cost-aware allocation across hedge-allowed venues.
///
/// Returns None if no hedge is needed or if no hedge venue is available.
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    // Need a fair value to reason about dollar exposures.
    let s_t = state.fair_value?;

    // ---------- 1) Global exposure and dead band ----------
    let q_t = state.q_global_tao; // global inventory in TAO

    // Volatility-scaled band: band_vol = HEDGE_BAND_BASE * band_mult(t)
    let band_mult = state.band_mult;
    let band_vol = cfg.hedge.hedge_band_base * band_mult;

    if q_t.abs() <= band_vol {
        // Inside dead band: do nothing.
        return None;
    }

    // ---------- 2) LQ controller: ΔH_t = k * q_t ----------
    let alpha = cfg.hedge.alpha_hedge;
    let beta = cfg.hedge.beta_hedge;
    let k_hedge = if alpha + beta > 0.0 {
        alpha / (alpha + beta)
    } else {
        // Degenerate config; fall back to 0.5.
        0.5
    };

    // ΔH_t has the same sign as q_t and represents how much we *remove*
    // from the global exposure (we will trade in the opposite direction).
    let mut delta_h = k_hedge * q_t;

    // Global step cap: |ΔH_t| <= HEDGE_MAX_STEP
    let max_step = cfg.hedge.hedge_max_step;
    if delta_h.abs() > max_step {
        delta_h = max_step * delta_h.signum();
    }

    if delta_h.abs() < 1e-6 {
        return None;
    }

    // The size we actually want to trade in TAO.
    let desired_size = delta_h.abs();

    // Direction:
    //   q_t > 0 (net long)  -> sell ΔH_t to reduce;
    //   q_t < 0 (net short) -> buy  ΔH_t to reduce.
    let hedge_side = if q_t > 0.0 { Side::Sell } else { Side::Buy };

    // ---------- 3) Build candidate hedge venues ----------
    #[derive(Debug)]
    struct Candidate {
        venue_index: usize,
        venue_id: String,
        side: Side,
        est_price: f64,
        unit_cost: f64, // rough taker cost per TAO
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    for (idx, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }

        let v_state = match state.venues.get(idx) {
            Some(v) => v,
            None => continue,
        };

        // Approximate best bid / ask from mid ± spread/2.
        let mid = v_state.mid.unwrap_or(s_t);
        let spread = v_state.spread.unwrap_or(1.0).max(0.0);

        let est_price = match hedge_side {
            Side::Sell => mid - 0.5 * spread, // sell into bids
            Side::Buy => mid + 0.5 * spread,  // buy from asks
        }
        .max(0.01); // guard against nonsense

        // Simple taker fee cost per unit of TAO.
        let fee_per_unit =
            (vcfg.taker_fee_bps / 10_000.0) * est_price;

        candidates.push(Candidate {
            venue_index: idx,
            venue_id: vcfg.id.clone(),
            side: hedge_side,
            est_price,
            unit_cost: fee_per_unit,
        });
    }

    if candidates.is_empty() {
        // No hedge-allowed venues.
        return None;
    }

    // ---------- 4) Cost-aware allocation ----------
    //
    // For now we keep it simple and allocate the *entire* ΔH_t to the
    // cheapest venue by unit_cost. This is still consistent with the
    // whitepaper's spirit (global cost minimisation), and can be
    // upgraded later to a full knapsack-style allocator.

    candidates.sort_by(|a, b| a.unit_cost.partial_cmp(&b.unit_cost).unwrap());

    let best = &candidates[0];

    let allocation = HedgeAllocation {
        venue_index: best.venue_index,
        venue_id: best.venue_id.clone(),
        side: best.side,
        size: desired_size,
        est_price: best.est_price,
    };

    let plan = HedgePlan {
        desired_delta: delta_h,
        allocations: vec![allocation],
    };

    Some(plan)
}

/// Convert a hedge plan into abstract order intents.
///
/// One IOC hedge order per allocation.
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
