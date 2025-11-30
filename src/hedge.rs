// src/hedge.rs
//
// Global hedge engine for Paraphina.
//
// Responsibilities (whitepaper Sections 9–10):
//   - Look at current global inventory q_t (in TAO) and decide a *desired*
//     hedge change ΔH* using a simple LQ objective with a dead-band.
//   - Check kill-switch / risk regime / venue health and only hedge when allowed.
//   - Allocate ΔH* across a subset of venues based on expected cost
//     (fees + spread + a toxicity penalty).
//   - Convert that plan into abstract OrderIntents which main.rs will
//     synthetically “fill” as taker orders.

use crate::config::{Config, VenueConfig};
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side, VenueStatus};

/// Per-venue hedge allocation (abstract).
#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub size: f64,      // TAO
    pub est_price: f64, // USD / TAO
}

/// Global hedge plan (single step).
#[derive(Debug, Clone)]
pub struct HedgePlan {
    /// Desired global hedge change ΔH* in TAO (sign: + = buy, - = sell).
    pub desired_delta: f64,
    /// Per-venue allocations that sum (approximately) to desired_delta.
    pub allocations: Vec<HedgeAllocation>,
}

/// Main entry point: decide whether to hedge, and if so how.
pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    // ---------- Hard gates ----------
    if state.kill_switch {
        return None;
    }

    let s_t = state.fair_value?;
    if s_t <= 0.0 {
        return None;
    }

    // Current global inventory in TAO (already maintained by GlobalState).
    let q_t = state.q_global_tao;

    // Dynamic hedge band in TAO:
    //
    //   band_t ≈ hedge_band_base * band_mult(t)
    //
    // band_mult(t) already comes from volatility controls in the engine
    // (Section 6): higher vol => smaller band => more aggressive hedging.
    let hedge_cfg = &cfg.hedge;
    let mut band_tao = hedge_cfg.hedge_band_base * state.band_mult.max(0.25);
    if band_tao <= 0.0 {
        band_tao = hedge_cfg.hedge_band_base.max(1.0);
    }

    let q_abs = q_t.abs();

    // If we are inside the band, do nothing.
    if q_abs <= band_tao {
        return None;
    }

    // ---------- LQ objective for ΔH* ----------
    //
    // We use a simple quadratic cost:
    //
    //   J(ΔH) = α (q_t + ΔH)^2 + β (ΔH)^2
    //
    // Minimising gives the closed form:
    //
    //   ΔH* = - (α / (α + β)) q_t
    //
    // which smoothly pulls us toward 0 inventory. α controls how much we
    // care about ending close to 0; β controls how much we penalise trading.
    let alpha = hedge_cfg.alpha_hedge.max(1e-8);
    let beta = hedge_cfg.beta_hedge.max(1e-8);
    let gain = alpha / (alpha + beta);
    let mut desired_delta = -gain * q_t; // TAO

    // We *already* know |q_t| > band_tao here, so desired_delta ≠ 0
    // unless α is pathological.

    // Limit the size of a single hedge step:
    let max_step = hedge_cfg.hedge_max_step.max(band_tao);
    if desired_delta > max_step {
        desired_delta = max_step;
    } else if desired_delta < -max_step {
        desired_delta = -max_step;
    }

    // If after all that the step is tiny, skip it.
    if desired_delta.abs() < 1e-6 {
        return None;
    }

    // ---------- Candidate venues & cost model ----------
    //
    // We choose venues that:
    //   - are marked is_hedge_allowed in config,
    //   - are not Disabled,
    //   - have at least some notion of a mid.
    //
    // For each candidate we compute a simple per-TAO cost proxy:
    //
    //   cost_i ≈ taker_fee + half_spread + toxicity_penalty
    //
    // Then we allocate ΔH across venues with weights ∝ 1 / cost_i.

    let sign_side = if desired_delta > 0.0 {
        Side::Buy
    } else {
        Side::Sell
    };
    let total_abs = desired_delta.abs();

    #[derive(Debug)]
    struct Candidate {
        venue_index: usize,
        vcfg: &'static VenueConfig,
        cost: f64,
        side: Side,
        mid: f64,
        half_spread: f64,
        tox: f64,
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    for (i, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }

        let vstate = &state.venues[i];

        // Disabled venues are never used for hedging.
        if matches!(vstate.status, VenueStatus::Disabled) {
            continue;
        }

        // If we have no usable mid, skip: we can't price a hedge.
        let mid = vstate.mid.unwrap_or(s_t);
        if mid <= 0.0 {
            continue;
        }

        // Approximate half-spread: if we don't have a live spread, assume
        // something modest so we still hedge.
        let half_spread = vstate
            .spread
            .map(|sp| (sp / 2.0).max(0.01))
            .unwrap_or(0.05);

        // Fees expressed as USD/TAO:
        let taker_fee_usd = (vcfg.taker_fee_bps / 10_000.0) * mid;

        // Tox penalty (Section 7): higher toxicity => higher effective cost.
        let tox = vstate.toxicity;
        let tox_penalty = 0.25 * tox * mid; // 25% weight of 1σ move scaled by toxicity

        let base_cost = taker_fee_usd + half_spread + tox_penalty;

        // Ensure positive cost to avoid singular weights.
        let cost = base_cost.max(1e-6);

        candidates.push(Candidate {
            venue_index: i,
            vcfg: unsafe { &*(vcfg as *const VenueConfig) }, // keep lifetime simple for this small utility
            cost,
            side: sign_side,
            mid,
            half_spread,
            tox,
        });
    }

    if candidates.is_empty() {
        return None;
    }

    // ---------- Allocate ΔH across venues ----------
    //
    // Weight_i = 1 / cost_i
    // size_i   = |ΔH*| * Weight_i / Σ_j Weight_j
    let mut inv_cost_sum = 0.0;
    for c in &candidates {
        inv_cost_sum += 1.0 / c.cost;
    }

    if inv_cost_sum <= 0.0 {
        return None;
    }

    let mut allocations: Vec<HedgeAllocation> = Vec::new();
    let mut remaining = total_abs;

    for (idx, c) in candidates.iter().enumerate() {
        let weight = (1.0 / c.cost) / inv_cost_sum;
        let mut size = total_abs * weight;

        // Respect venue-level max_order_size.
        if size > c.vcfg.max_order_size {
            size = c.vcfg.max_order_size;
        }

        // Last venue gets whatever is left so rounding error doesn't leak.
        if idx == candidates.len() - 1 {
            size = remaining;
        }

        if size <= 0.0 {
            continue;
        }

        remaining -= size;
        if remaining < 0.0 {
            remaining = 0.0;
        }

        // Price: best guess taker level around the mid.
        let est_price = match c.side {
            Side::Buy => c.mid + c.half_spread,
            Side::Sell => c.mid - c.half_spread,
        };

        allocations.push(HedgeAllocation {
            venue_index: c.venue_index,
            venue_id: c.vcfg.id.clone(),
            side: c.side,
            size,
            est_price,
        });

        if remaining <= 1e-6 {
            break;
        }
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
///
/// These are interpreted by main.rs as *taker* orders (taker fees applied).
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
