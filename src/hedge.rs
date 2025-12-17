// src/hedge.rs
//
// Global hedge allocator for Paraphina (Milestone F / Whitepaper Section 13).
//
// Responsibilities:
//   - Compute desired signed hedge step ΔH using deadband + clipped proportional step (§13.1).
//   - Allocate |ΔH| across venues with a greedy knapsack-lite allocator (§13.2).
//   - Emit per-venue IOC OrderIntents (purpose Hedge) with guard prices.
//
// Key design principles:
//   - Deterministic: stable sorting and tie-break by venue_index.
//   - Config-driven: all weights/caps live in HedgeConfig.
//   - Gating: skip venues when Disabled, stale book, tox >= tox_high, missing mid/spread,
//     depth < min_depth_usd, or dist_liq_sigma <= liq_crit_sigma.
//   - Liquidation-aware: penalize venues closer to liquidation; hard-skip at crit.
//   - Respect constraints: total hedge ≤ max_step_tao (global) and ≤ per-venue caps.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus};

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
    /// Desired change in global inventory q_t in TAO.
    /// Sign convention: positive = sell TAO (reduce long), negative = buy TAO (reduce short).
    pub desired_delta: f64,
    pub allocations: Vec<HedgeAllocation>,
}

/// Candidate venue for hedge allocation.
#[derive(Debug, Clone)]
struct HedgeCandidate {
    venue_index: usize,
    venue_id: String,
    side: Side,
    /// Guard price (IOC limit).
    price: f64,
    /// Maximum size we can allocate to this venue.
    max_size: f64,
    /// Total cost per TAO (lower is better).
    total_cost: f64,
}

// --- Helpers ----------------------------------------------------------------

fn tick_floor(px: f64, tick: f64) -> f64 {
    (px / tick).floor() * tick
}

fn tick_ceil(px: f64, tick: f64) -> f64 {
    (px / tick).ceil() * tick
}

/// Round size down to the nearest lot size / size step.
fn round_to_lot_size(size: f64, lot_size: f64, size_step: f64) -> Option<f64> {
    if size <= 0.0 || !size.is_finite() {
        return None;
    }
    let effective_step = size_step.max(lot_size).max(1e-12);
    let rounded = (size / effective_step).floor() * effective_step;
    if rounded < lot_size || rounded <= 0.0 {
        return None;
    }
    Some(rounded)
}

/// Derive top-of-book from mid and spread.
fn top_of_book_from_mid(mid: f64, spread: f64) -> Option<(f64, f64)> {
    if !mid.is_finite() || !spread.is_finite() || mid <= 0.0 || spread <= 0.0 {
        return None;
    }
    let half = 0.5 * spread;
    let bid = mid - half;
    let ask = mid + half;
    if bid <= 0.0 || ask <= bid {
        return None;
    }
    Some((bid, ask))
}

/// Compute the liquidation penalty.
/// Grows as dist_liq_sigma approaches liq_warn_sigma.
/// Returns 0 if dist >= warn, large penalty as dist approaches crit.
fn compute_liq_penalty(dist: f64, warn: f64, crit: f64, scale: f64) -> f64 {
    if !dist.is_finite() || dist >= warn {
        return 0.0;
    }
    if dist <= crit {
        // Should be hard-skipped, but return large penalty as fallback.
        return f64::MAX;
    }
    // Linear penalty from 0 at warn to scale*(warn-crit) at crit.
    let distance_inside_warn = warn - dist;
    scale * distance_inside_warn
}

// --- Main allocator ---------------------------------------------------------

/// Main hedge planner implementing Whitepaper Section 13.
///
/// Returns `None` when no hedge action is required for this tick.
///
/// # Arguments
/// * `cfg` - Strategy configuration
/// * `state` - Current global state
/// * `now_ms` - Current timestamp in milliseconds (required for staleness gating)
pub fn compute_hedge_plan(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) -> Option<HedgePlan> {
    // 0) Global guards: no hedging without fair value or when kill switch is active.
    if state.kill_switch {
        return None;
    }

    let fair = match state.fair_value {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => return None,
    };

    let hedge_cfg = &cfg.hedge;

    // 1) Compute dynamic hedge band (Section 13.1).
    //
    //    band_vol = band_base_tao * (1 + band_vol_mult * vol_ratio_clipped)
    //
    let band_vol = hedge_cfg.band_base_tao.max(0.0)
        * (1.0 + hedge_cfg.band_vol_mult * state.vol_ratio_clipped.max(0.0));

    // X = global inventory (prefer TAO; if only USD delta exists, convert using fair).
    let x = state.q_global_tao;

    // 2) Deadband: if |X| <= band_vol then ΔH = 0.
    if x.abs() <= band_vol {
        return None;
    }

    // 3) Raw step: ΔH_raw = k_hedge * X.
    //    Then clamp to [-max_step_tao, +max_step_tao].
    let delta_raw = hedge_cfg.k_hedge * x;
    let max_step = hedge_cfg.max_step_tao.max(0.0);

    let delta_clamped = if max_step > 0.0 {
        delta_raw.clamp(-max_step, max_step)
    } else {
        delta_raw
    };

    // 4) Determine side and target.
    //    If ΔH > 0 (we have positive inventory, want to reduce) => Sell.
    //    If ΔH < 0 (we have negative inventory, want to reduce) => Buy.
    //
    //    Note: The whitepaper convention is ΔH reduces |X|, so:
    //      - X > 0 (long) => ΔH < 0 conceptually, but since ΔH = k * X, ΔH > 0.
    //        We sell to reduce the long position.
    //      - X < 0 (short) => ΔH < 0, we buy to cover.
    //
    //    Our convention: delta_clamped > 0 means we need to SELL (reduce long).
    if delta_clamped.abs() < 1e-9 {
        return None;
    }

    let side = if delta_clamped > 0.0 {
        Side::Sell
    } else {
        Side::Buy
    };
    let target = delta_clamped.abs();

    // 5) Build candidate list with per-venue costs.
    let candidates = build_candidates(cfg, state, now_ms, fair, side);

    if candidates.is_empty() {
        return None;
    }

    // 6) Greedy allocation (continuous knapsack-lite).
    let allocations = greedy_allocate(cfg, candidates, target);

    if allocations.is_empty() {
        return None;
    }

    Some(HedgePlan {
        desired_delta: delta_clamped,
        allocations,
    })
}

/// Build candidate venues for hedge allocation.
fn build_candidates(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    fair: f64,
    side: Side,
) -> Vec<HedgeCandidate> {
    let hedge_cfg = &cfg.hedge;
    let mut candidates: Vec<HedgeCandidate> = Vec::new();

    for (j, vcfg) in cfg.venues.iter().enumerate() {
        let v = &state.venues[j];

        // --- Gating ---

        // Skip if hedge not allowed on this venue.
        if !vcfg.is_hedge_allowed {
            continue;
        }

        // Skip Disabled venues.
        if matches!(v.status, VenueStatus::Disabled) {
            continue;
        }

        // Skip stale venues.
        let ts = match v.last_mid_update_ms {
            Some(t) => t,
            None => continue,
        };
        if now_ms > 0 && now_ms - ts > cfg.book.stale_ms {
            continue;
        }

        // Skip venues with high toxicity.
        if v.toxicity >= cfg.toxicity.tox_high_threshold {
            continue;
        }

        // Skip if missing mid/spread.
        let mid = match v.mid {
            Some(m) if m.is_finite() && m > 0.0 => m,
            _ => continue,
        };
        let spread = match v.spread {
            Some(s) if s.is_finite() && s > 0.0 => s,
            _ => continue,
        };

        // Skip if depth < min_depth_usd.
        let depth_usd = v.depth_near_mid.max(0.0);
        if depth_usd < hedge_cfg.min_depth_usd {
            continue;
        }

        // --- Liquidation check ---
        let dist = v.dist_liq_sigma;

        // Hard-skip if dist <= liq_crit_sigma.
        if dist.is_finite() && dist <= hedge_cfg.liq_crit_sigma {
            continue;
        }

        // --- Top-of-book and guard price ---
        let (bid, ask) = match top_of_book_from_mid(mid, spread) {
            Some(x) => x,
            None => continue,
        };

        let tick = vcfg.tick_size.max(1e-9);

        // Guard price: buy at ask + guard_mult*spread; sell at bid - guard_mult*spread.
        let (raw_px, guard_px) = match side {
            Side::Buy => {
                let guard = ask + hedge_cfg.guard_mult * spread;
                (ask, tick_ceil(guard, tick))
            }
            Side::Sell => {
                let guard = bid - hedge_cfg.guard_mult * spread;
                (bid, tick_floor(guard, tick))
            }
        };

        if !guard_px.is_finite() || guard_px <= 0.0 {
            continue;
        }

        // --- Per-venue cap u_j ---
        // u_j = min(max_venue_tao_per_tick, venue.max_order_size, depth_fraction * depth_usd / fair)
        let depth_cap = (hedge_cfg.depth_fraction * depth_usd / fair).max(0.0);
        let max_size = hedge_cfg
            .max_venue_tao_per_tick
            .min(vcfg.max_order_size)
            .min(depth_cap);

        if !max_size.is_finite() || max_size <= 0.0 {
            continue;
        }

        // --- Per-unit cost model (Section 13.2.1) ---

        // 1. Execution cost = 0.5 * spread + taker_fee + slippage_buffer
        let half_spread = 0.5 * spread;
        let taker_fee = (vcfg.taker_fee_bps / 10_000.0).abs() * mid;
        let exec_cost = half_spread + taker_fee + hedge_cfg.slippage_buffer;

        // 2. Funding benefit (directional vs funding_8h)
        //    Positive funding means shorts receive, longs pay.
        //    If side == Sell (reducing long / going short), we benefit from positive funding.
        //    If side == Buy (reducing short / going long), we benefit from negative funding.
        let horizon_frac = hedge_cfg.funding_horizon_sec / (8.0 * 60.0 * 60.0);
        let funding_benefit = match side {
            Side::Sell => v.funding_8h * horizon_frac * fair,
            Side::Buy => -v.funding_8h * horizon_frac * fair,
        };

        // 3. Basis edge (directional vs fair)
        //    For selling: want to sell at a premium => bid > fair is good (negative cost).
        //    For buying: want to buy at a discount => ask < fair is good.
        let basis_edge = match side {
            Side::Sell => raw_px - fair, // positive if bid > fair (good)
            Side::Buy => fair - raw_px,  // positive if ask < fair (good)
        };

        // 4. Liquidation penalty
        let liq_penalty = compute_liq_penalty(
            dist,
            hedge_cfg.liq_warn_sigma,
            hedge_cfg.liq_crit_sigma,
            hedge_cfg.liq_penalty_scale,
        );
        if !liq_penalty.is_finite() {
            continue; // Extreme penalty means skip.
        }

        // 5. Fragmentation penalty (opening a new position on this venue)
        let frag_penalty = if v.position_tao.abs() < 1e-9 {
            hedge_cfg.frag_penalty
        } else {
            0.0
        };

        // Total cost = exec_cost - funding_weight * funding_benefit
        //            - basis_weight * basis_edge + liq_penalty + frag_penalty
        let total_cost = exec_cost
            - hedge_cfg.funding_weight * funding_benefit
            - hedge_cfg.basis_weight * basis_edge
            + liq_penalty
            + frag_penalty;

        // Skip if total_cost not finite.
        if !total_cost.is_finite() {
            continue;
        }

        candidates.push(HedgeCandidate {
            venue_index: j,
            venue_id: vcfg.id.clone(),
            side,
            price: guard_px,
            max_size,
            total_cost,
        });
    }

    // Deterministic sort: by total_cost asc, then venue_index asc.
    candidates.sort_by(|a, b| {
        let cost_cmp = a
            .total_cost
            .partial_cmp(&b.total_cost)
            .unwrap_or(std::cmp::Ordering::Equal);
        if cost_cmp != std::cmp::Ordering::Equal {
            return cost_cmp;
        }
        a.venue_index.cmp(&b.venue_index)
    });

    candidates
}

/// Greedy knapsack-lite allocation.
fn greedy_allocate(
    cfg: &Config,
    candidates: Vec<HedgeCandidate>,
    target: f64,
) -> Vec<HedgeAllocation> {
    let mut allocations = Vec::new();
    let mut remaining = target;

    for c in candidates {
        if remaining <= 0.0 {
            break;
        }

        let vcfg = &cfg.venues[c.venue_index];

        // Size = min(max_size, remaining)
        let raw_size = c.max_size.min(remaining);

        // Apply lot size rounding.
        let rounded = match round_to_lot_size(raw_size, vcfg.lot_size_tao, vcfg.size_step_tao) {
            Some(s) => s,
            None => continue,
        };

        // Check min notional.
        let notional = rounded * c.price;
        if notional < vcfg.min_notional_usd {
            continue;
        }

        // Skip dust.
        if rounded < 1e-6 {
            continue;
        }

        allocations.push(HedgeAllocation {
            venue_index: c.venue_index,
            venue_id: c.venue_id.clone(),
            side: c.side,
            size: rounded,
            est_price: c.price,
        });

        remaining -= rounded;
    }

    allocations
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

/// Convenience function: compute hedge orders directly.
///
/// # Arguments
/// * `cfg` - Strategy configuration
/// * `state` - Current global state
/// * `now_ms` - Current timestamp in milliseconds (required for staleness gating)
pub fn compute_hedge_orders(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) -> Vec<OrderIntent> {
    match compute_hedge_plan(cfg, state, now_ms) {
        Some(plan) => hedge_plan_to_order_intents(&plan),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_round_to_lot_size() {
        // Basic rounding
        assert!(approx_eq(round_to_lot_size(1.55, 0.1, 0.1).unwrap(), 1.5));
        assert!(approx_eq(round_to_lot_size(1.59, 0.1, 0.1).unwrap(), 1.5));

        // Below lot size
        assert_eq!(round_to_lot_size(0.05, 0.1, 0.1), None);

        // Exact lot size
        assert!(approx_eq(round_to_lot_size(0.1, 0.1, 0.1).unwrap(), 0.1));

        // Zero and negative
        assert_eq!(round_to_lot_size(0.0, 0.1, 0.1), None);
        assert_eq!(round_to_lot_size(-1.0, 0.1, 0.1), None);
    }

    #[test]
    fn test_top_of_book_from_mid() {
        let (bid, ask) = top_of_book_from_mid(100.0, 0.5).unwrap();
        assert!(approx_eq(bid, 99.75));
        assert!(approx_eq(ask, 100.25));

        // Invalid inputs
        assert!(top_of_book_from_mid(0.0, 0.5).is_none());
        assert!(top_of_book_from_mid(100.0, 0.0).is_none());
        assert!(top_of_book_from_mid(100.0, -0.5).is_none());
    }

    #[test]
    fn test_liq_penalty() {
        let warn = 5.0;
        let crit = 2.0;
        let scale = 0.1;

        // Outside warn => 0
        assert!(approx_eq(compute_liq_penalty(6.0, warn, crit, scale), 0.0));
        assert!(approx_eq(compute_liq_penalty(5.0, warn, crit, scale), 0.0));

        // At warn boundary
        assert!(approx_eq(
            compute_liq_penalty(4.9, warn, crit, scale),
            scale * 0.1
        ));

        // Inside warn but outside crit
        assert!(approx_eq(
            compute_liq_penalty(3.5, warn, crit, scale),
            scale * 1.5
        ));

        // At/inside crit => MAX
        assert_eq!(compute_liq_penalty(2.0, warn, crit, scale), f64::MAX);
        assert_eq!(compute_liq_penalty(1.0, warn, crit, scale), f64::MAX);
    }
}
