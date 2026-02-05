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
//   - Deterministic: stable sorting and tie-break by venue_index, chunk_index.
//   - Config-driven: all weights/caps live in HedgeConfig.
//   - Gating: skip venues when Disabled, stale book, tox >= tox_high, missing mid/spread,
//     depth < min_depth_usd, or dist_liq_sigma <= liq_crit_sigma.
//   - Liquidation-aware: penalize venues closer to liquidation; hard-skip at crit.
//   - Margin-aware: enforce per-venue margin constraints when increasing exposure.
//   - Multi-chunk: generate multiple chunk candidates per venue, aggregate into single order.
//   - Respect constraints: total hedge ≤ max_step_tao (global) and ≤ per-venue caps.

use std::sync::Arc;

use crate::config::Config;
use crate::state::{funding_rate_for_decision, GlobalState, VenueState};
use crate::types::{
    OrderIntent, OrderPurpose, PlaceOrderIntent, Side, TimeInForce, TimestampMs, VenueStatus,
};

// ============================================================================
// MARGIN CONSTRAINT HELPERS (Section A of specification)
// ============================================================================

/// Check whether applying dq would increase absolute exposure.
///
/// # Arguments
/// * `q_old` - Current position in TAO (signed)
/// * `dq` - Proposed change in position (signed: positive = buy, negative = sell)
///
/// # Returns
/// `true` if the trade would increase |position|, `false` otherwise.
///
/// # Examples
/// - q_old=10, dq=5 → |10|=10, |15|=15 → increases (true)
/// - q_old=10, dq=-5 → |10|=10, |5|=5 → decreases (false)
/// - q_old=10, dq=-15 → |10|=10, |-5|=5 → decreases overall (false)
/// - q_old=0, dq=5 → |0|=0, |5|=5 → increases (true)
pub fn increases_abs_exposure(q_old: f64, dq: f64) -> bool {
    let abs_old = q_old.abs();
    let abs_new = (q_old + dq).abs();
    abs_new > abs_old
}

fn funding_rate_for_hedge(cfg: &Config, vstate: &VenueState, now_ms: TimestampMs) -> f64 {
    if !cfg.hedge.funding_enabled {
        return 0.0;
    }
    funding_rate_for_decision(&vstate.funding_state, now_ms, &cfg.funding, true).unwrap_or(0.0)
}

/// Compute the maximum absolute position allowed after the hedge.
///
/// The cap is computed as:
///   abs_limit = |q_old| + additional_abs_cap_tao
///
/// where:
///   additional_abs_cap_tao = (margin_available_usd * max_leverage * safety_buffer) / mark_price_usd
///
/// This allows exposure to grow by at most additional_abs_cap over current |q_old|.
/// Reducing exposure or crossing through zero is always allowed (within this formula).
///
/// # Arguments
/// * `q_old` - Current position in TAO (signed)
/// * `margin_available_usd` - Available margin in USD (≥0)
/// * `max_leverage` - Maximum leverage (>0)
/// * `safety_buffer` - Safety factor (0 < safety_buffer ≤ 1, default 0.95)
/// * `mark_price_usd` - Mark price in USD/TAO (>0)
///
/// # Returns
/// Maximum absolute position allowed after trade in TAO.
pub fn compute_abs_limit_after_trade(
    q_old: f64,
    margin_available_usd: f64,
    max_leverage: f64,
    safety_buffer: f64,
    mark_price_usd: f64,
) -> f64 {
    // Handle edge cases
    if mark_price_usd <= 0.0 || !mark_price_usd.is_finite() {
        return q_old.abs(); // No growth allowed
    }
    if margin_available_usd <= 0.0 || max_leverage <= 0.0 || safety_buffer <= 0.0 {
        return q_old.abs(); // No growth allowed
    }

    let additional_abs_cap_tao =
        (margin_available_usd * max_leverage * safety_buffer) / mark_price_usd;

    q_old.abs() + additional_abs_cap_tao
}

/// Cap a requested signed dq by the absolute limit constraint.
///
/// The constraint is: |q_old + dq_capped| <= abs_limit_after_trade
///
/// This correctly handles crossing through zero.
///
/// # Arguments
/// * `q_old` - Current position in TAO (signed)
/// * `desired_dq` - Desired change in position (signed)
/// * `abs_limit_after_trade` - Maximum absolute position allowed after trade
///
/// # Returns
/// Capped dq that respects the absolute limit constraint.
pub fn cap_dq_by_abs_limit(q_old: f64, desired_dq: f64, abs_limit_after_trade: f64) -> f64 {
    let q_new = q_old + desired_dq;

    // If already within limit, no capping needed
    if q_new.abs() <= abs_limit_after_trade {
        return desired_dq;
    }

    // Need to cap. Determine which side of zero we end up on.
    if q_new >= 0.0 {
        // Final position is positive (or zero)
        // q_old + dq_capped = abs_limit_after_trade
        let dq_capped = abs_limit_after_trade - q_old;
        // Ensure we don't flip sign of dq
        if desired_dq >= 0.0 {
            dq_capped.max(0.0)
        } else {
            // Selling, shouldn't increase positive position
            desired_dq
        }
    } else {
        // Final position is negative
        // q_old + dq_capped = -abs_limit_after_trade
        let dq_capped = -abs_limit_after_trade - q_old;
        // Ensure we don't flip sign of dq
        if desired_dq <= 0.0 {
            dq_capped.min(0.0)
        } else {
            // Buying, shouldn't increase negative position
            desired_dq
        }
    }
}

// ============================================================================
// MULTI-CHUNK ALLOCATION (Section C of specification)
// ============================================================================

/// An internal chunk candidate for allocation.
///
/// Multiple chunks may exist per venue; they are aggregated into a single
/// order per venue after allocation.
#[derive(Debug, Clone)]
pub struct ChunkCandidate {
    /// Venue index.
    pub venue_index: usize,
    /// Venue identifier (for deterministic tie-breaking). Uses Arc<str> for cheap cloning.
    pub venue_id: Arc<str>,
    /// Chunk index within this venue (for deterministic tie-breaking).
    pub chunk_index: usize,
    /// Signed size of this chunk in TAO.
    pub dq_signed: f64,
    /// Unit cost per TAO (lower is better).
    pub unit_cost: f64,
    /// Guard price for the order.
    pub guard_price: f64,
    /// Side of the trade.
    pub side: Side,
}

/// Build chunk candidates for a single venue.
///
/// # Arguments
/// * `venue_index` - Index of the venue
/// * `venue_id` - Venue identifier string
/// * `side` - Trade side
/// * `guard_price` - Guard price for orders
/// * `max_dq_abs` - Maximum absolute dq for this venue (after all caps)
/// * `chunk_size_abs` - Size of each chunk in TAO (must be > 0)
/// * `base_unit_cost` - Base cost per TAO for this venue
/// * `convexity_cost_per_chunk` - Additional cost per chunk (for spreading)
///
/// # Returns
/// Vector of chunk candidates for this venue.
#[allow(clippy::too_many_arguments)]
pub fn build_chunk_candidates(
    venue_index: usize,
    venue_id: Arc<str>,
    side: Side,
    guard_price: f64,
    max_dq_abs: f64,
    chunk_size_abs: f64,
    base_unit_cost: f64,
    convexity_cost_per_chunk: f64,
) -> Vec<ChunkCandidate> {
    let mut chunks = Vec::new();

    if !base_unit_cost.is_finite() || !convexity_cost_per_chunk.is_finite() {
        return chunks;
    }

    // Safety: ensure chunk_size is positive to avoid infinite loop
    let effective_chunk_size = chunk_size_abs.max(1e-9);

    // Compute how many full chunks we can make
    let num_full_chunks = (max_dq_abs / effective_chunk_size).floor() as usize;
    let remainder = max_dq_abs - (num_full_chunks as f64 * effective_chunk_size);

    // Generate full chunks
    for i in 0..num_full_chunks {
        let dq_signed = match side {
            Side::Buy => effective_chunk_size,
            Side::Sell => -effective_chunk_size,
        };
        let chunk_cost = base_unit_cost + (i as f64 * convexity_cost_per_chunk);

        chunks.push(ChunkCandidate {
            venue_index,
            venue_id: venue_id.clone(),
            chunk_index: i,
            dq_signed,
            unit_cost: chunk_cost,
            guard_price,
            side,
        });
    }

    // Generate partial chunk if significant remainder
    if remainder > 1e-9 {
        let dq_signed = match side {
            Side::Buy => remainder,
            Side::Sell => -remainder,
        };
        let chunk_cost = base_unit_cost + (num_full_chunks as f64 * convexity_cost_per_chunk);

        chunks.push(ChunkCandidate {
            venue_index,
            venue_id: venue_id.clone(),
            chunk_index: num_full_chunks,
            dq_signed,
            unit_cost: chunk_cost,
            guard_price,
            side,
        });
    }

    chunks
}

/// Greedy allocation of chunks to meet target hedge.
///
/// Sorting is deterministic with tie-break:
///   (unit_cost ASC, venue_id ASC, chunk_index ASC)
///
/// Returns a map from venue_index to (total_allocated_abs, guard_price, side).
pub fn greedy_allocate_chunks(
    mut all_chunks: Vec<ChunkCandidate>,
    target_dq_abs: f64,
) -> Vec<(usize, f64, f64, Side)> {
    // Sort deterministically: by unit_cost ASC, then venue_id ASC, then chunk_index ASC
    all_chunks.sort_by(|a, b| {
        // Primary: unit_cost ascending
        let cost_cmp = a
            .unit_cost
            .partial_cmp(&b.unit_cost)
            .unwrap_or(std::cmp::Ordering::Equal);
        if cost_cmp != std::cmp::Ordering::Equal {
            return cost_cmp;
        }
        // Secondary: venue_id ascending (lexicographic)
        let id_cmp = a.venue_id.cmp(&b.venue_id);
        if id_cmp != std::cmp::Ordering::Equal {
            return id_cmp;
        }
        // Tertiary: chunk_index ascending
        a.chunk_index.cmp(&b.chunk_index)
    });

    // Aggregate allocations by venue
    let mut venue_allocations: std::collections::BTreeMap<usize, (f64, f64, Side)> =
        std::collections::BTreeMap::new();

    let mut remaining = target_dq_abs;

    for chunk in all_chunks {
        if remaining <= 0.0 {
            break;
        }

        let chunk_abs = chunk.dq_signed.abs();
        let alloc_abs = chunk_abs.min(remaining);

        if alloc_abs < 1e-9 {
            continue;
        }

        let entry = venue_allocations.entry(chunk.venue_index).or_insert((
            0.0,
            chunk.guard_price,
            chunk.side,
        ));
        entry.0 += alloc_abs;

        remaining -= alloc_abs;
    }

    // Convert to vector for output
    venue_allocations
        .into_iter()
        .map(|(idx, (size, price, side))| (idx, size, price, side))
        .collect()
}

// ============================================================================
// PUBLIC TYPES
// ============================================================================

/// One venue-level hedge slice in TAO.
///
/// Note: `venue_id` uses `Arc<str>` for cheap cloning in hot paths.
#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
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

/// Hedge cost component breakdown (telemetry).
#[derive(Debug, Clone)]
pub struct HedgeCostComponents {
    pub exec_cost: f64,
    pub funding_benefit: f64,
    pub basis_edge: f64,
    pub liq_penalty: f64,
    pub frag_penalty: f64,
    pub total_cost: f64,
}

/// Compute hedge cost components for a hedge intent.
pub fn compute_hedge_cost_components(
    cfg: &Config,
    state: &GlobalState,
    intent: &OrderIntent,
) -> Option<HedgeCostComponents> {
    let intent = match intent {
        OrderIntent::Place(pi) if matches!(pi.purpose, OrderPurpose::Hedge) => pi,
        _ => return None,
    };

    let vcfg = cfg.venues.get(intent.venue_index)?;
    let v = state.venues.get(intent.venue_index)?;
    let fair = state.fair_value.unwrap_or(1.0).max(1.0);
    let mid = v.mid.unwrap_or(fair);
    let spread = v.spread.unwrap_or(0.0).max(0.0);

    let half_spread = 0.5 * spread;
    let taker_fee = (vcfg.taker_fee_bps / 10_000.0).abs() * mid;
    let exec_cost = half_spread + taker_fee + cfg.hedge.slippage_buffer;

    let horizon_frac = cfg.hedge.funding_horizon_sec / (8.0 * 60.0 * 60.0);
    let now_ms = crate::types::now_ms();
    let funding_8h = funding_rate_for_hedge(cfg, v, now_ms);
    let funding_benefit = match intent.side {
        Side::Sell => funding_8h * horizon_frac * fair,
        Side::Buy => -funding_8h * horizon_frac * fair,
    };

    let basis_edge = match intent.side {
        Side::Sell => intent.price - fair,
        Side::Buy => fair - intent.price,
    };

    let liq_penalty = compute_liq_penalty(
        v.dist_liq_sigma,
        cfg.hedge.liq_warn_sigma,
        cfg.hedge.liq_crit_sigma,
        cfg.hedge.liq_penalty_scale,
    );

    let frag_penalty = if v.position_tao.abs() < 1e-9 {
        cfg.hedge.frag_penalty
    } else {
        0.0
    };

    let total_cost = exec_cost
        - cfg.hedge.funding_weight * funding_benefit
        - cfg.hedge.basis_weight * basis_edge
        + liq_penalty
        + frag_penalty;

    if !exec_cost.is_finite()
        || !funding_benefit.is_finite()
        || !basis_edge.is_finite()
        || !liq_penalty.is_finite()
        || !frag_penalty.is_finite()
        || !total_cost.is_finite()
    {
        return None;
    }

    Some(HedgeCostComponents {
        exec_cost,
        funding_benefit,
        basis_edge,
        liq_penalty,
        frag_penalty,
        total_cost,
    })
}

/// Candidate venue for hedge allocation (internal).
#[derive(Debug, Clone)]
struct HedgeCandidate {
    venue_index: usize,
    venue_id: Arc<str>,
    /// Guard price (IOC limit).
    price: f64,
    /// Maximum size we can allocate to this venue.
    max_size: f64,
    /// Total cost per TAO (lower is better).
    total_cost: f64,
    /// Margin-constrained maximum size (may be tighter than max_size).
    margin_capped_size: f64,
}

// ============================================================================
// HELPERS
// ============================================================================

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

// ============================================================================
// MAIN ALLOCATOR
// ============================================================================

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

    let delta_clamped = delta_raw.clamp(-max_step, max_step);

    // 4) Determine side and target.
    if delta_clamped.abs() < 1e-9 {
        return None;
    }

    let side = if delta_clamped > 0.0 {
        Side::Sell
    } else {
        Side::Buy
    };
    let target = delta_clamped.abs();

    // 5) Build candidate list with per-venue costs and margin constraints.
    let candidates = build_candidates(cfg, state, now_ms, fair, side);

    if candidates.is_empty() {
        return None;
    }

    // 6) Build chunk candidates for all venues.
    let chunk_size = compute_chunk_size(cfg);
    let all_chunks = build_all_chunk_candidates(cfg, &candidates, side, chunk_size);

    if all_chunks.is_empty() {
        return None;
    }

    // 7) Greedy allocation of chunks.
    let venue_allocations = greedy_allocate_chunks(all_chunks, target);

    if venue_allocations.is_empty() {
        return None;
    }

    // 8) Convert to final allocations with lot-size rounding.
    let allocations = finalize_allocations(cfg, &candidates, venue_allocations);

    if allocations.is_empty() {
        return None;
    }

    Some(HedgePlan {
        desired_delta: delta_clamped,
        allocations,
    })
}

/// Compute the chunk size for multi-chunk allocation.
///
/// If hedge_chunk_size_tao is set (> 0), use it.
/// Otherwise, compute a default from per-venue caps: cap/4 with minimum 0.1 TAO.
fn compute_chunk_size(cfg: &Config) -> f64 {
    let hedge_cfg = &cfg.hedge;

    if hedge_cfg.chunk_size_tao > 0.0 {
        return hedge_cfg.chunk_size_tao;
    }

    // Default: max_venue_tao_per_tick / 4, minimum 0.1 TAO
    (hedge_cfg.max_venue_tao_per_tick / 4.0).max(0.1)
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
        let depth_cap = (hedge_cfg.depth_fraction * depth_usd / fair).max(0.0);
        let max_size = hedge_cfg
            .max_venue_tao_per_tick
            .min(vcfg.max_order_size)
            .min(depth_cap);

        if !max_size.is_finite() || max_size <= 0.0 {
            continue;
        }

        // --- Margin constraint (Section B) ---
        // Only applies when increasing absolute exposure.
        // Compute margin-based additional capacity.
        let q_old = v.position_tao;
        let margin_available = v.margin_available_usd;
        let max_leverage = hedge_cfg.max_leverage;
        let safety_buffer = hedge_cfg.margin_safety_buffer;

        let abs_limit = compute_abs_limit_after_trade(
            q_old,
            margin_available,
            max_leverage,
            safety_buffer,
            fair,
        );

        // Compute the margin-constrained max size for this direction.
        // For selling (dq < 0): desired_dq = -max_size
        // For buying (dq > 0): desired_dq = +max_size
        let desired_dq = match side {
            Side::Buy => max_size,
            Side::Sell => -max_size,
        };

        let capped_dq = cap_dq_by_abs_limit(q_old, desired_dq, abs_limit);
        let margin_capped_size = capped_dq.abs();

        // --- Per-unit cost model (Section 13.2.1) ---

        let half_spread = 0.5 * spread;
        let taker_fee = (vcfg.taker_fee_bps / 10_000.0).abs() * mid;
        let exec_cost = half_spread + taker_fee + hedge_cfg.slippage_buffer;

        let horizon_frac = hedge_cfg.funding_horizon_sec / (8.0 * 60.0 * 60.0);
        let funding_8h = funding_rate_for_hedge(cfg, v, now_ms);
        let funding_benefit = match side {
            Side::Sell => funding_8h * horizon_frac * fair,
            Side::Buy => -funding_8h * horizon_frac * fair,
        };

        let basis_edge = match side {
            Side::Sell => raw_px - fair,
            Side::Buy => fair - raw_px,
        };

        let liq_penalty = compute_liq_penalty(
            dist,
            hedge_cfg.liq_warn_sigma,
            hedge_cfg.liq_crit_sigma,
            hedge_cfg.liq_penalty_scale,
        );
        if !liq_penalty.is_finite() {
            continue;
        }

        let frag_penalty = if v.position_tao.abs() < 1e-9 {
            hedge_cfg.frag_penalty
        } else {
            0.0
        };

        let total_cost = exec_cost
            - hedge_cfg.funding_weight * funding_benefit
            - hedge_cfg.basis_weight * basis_edge
            + liq_penalty
            + frag_penalty;

        if !total_cost.is_finite() {
            continue;
        }

        candidates.push(HedgeCandidate {
            venue_index: j,
            venue_id: vcfg.id_arc.clone(),
            price: guard_px,
            max_size,
            total_cost,
            margin_capped_size,
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

/// Build all chunk candidates from all venue candidates.
fn build_all_chunk_candidates(
    cfg: &Config,
    candidates: &[HedgeCandidate],
    side: Side,
    chunk_size: f64,
) -> Vec<ChunkCandidate> {
    let hedge_cfg = &cfg.hedge;
    let convexity_cost = hedge_cfg.chunk_convexity_cost_bps / 10_000.0; // Convert bps to fraction

    let mut all_chunks = Vec::new();

    for c in candidates {
        // Use the minimum of max_size and margin_capped_size
        let effective_max = c.max_size.min(c.margin_capped_size);

        if effective_max <= 0.0 {
            continue;
        }

        let venue_chunks = build_chunk_candidates(
            c.venue_index,
            c.venue_id.clone(),
            side,
            c.price,
            effective_max,
            chunk_size,
            c.total_cost,
            convexity_cost,
        );

        all_chunks.extend(venue_chunks);
    }

    all_chunks
}

/// Convert venue allocations to final HedgeAllocations with lot-size rounding.
fn finalize_allocations(
    cfg: &Config,
    candidates: &[HedgeCandidate],
    venue_allocations: Vec<(usize, f64, f64, Side)>,
) -> Vec<HedgeAllocation> {
    let mut allocations = Vec::new();

    for (venue_index, size, guard_price, side) in venue_allocations {
        let vcfg = &cfg.venues[venue_index];

        // Apply lot size rounding
        let rounded = match round_to_lot_size(size, vcfg.lot_size_tao, vcfg.size_step_tao) {
            Some(s) => s,
            None => continue,
        };

        // Check min notional
        let notional = rounded * guard_price;
        if notional < vcfg.min_notional_usd {
            continue;
        }

        // Skip dust
        if rounded < 1e-6 {
            continue;
        }

        // Find venue_id from candidates
        let venue_id = candidates
            .iter()
            .find(|c| c.venue_index == venue_index)
            .map(|c| c.venue_id.clone())
            .unwrap_or_else(|| cfg.venues[venue_index].id_arc.clone());

        allocations.push(HedgeAllocation {
            venue_index,
            venue_id,
            side,
            size: rounded,
            est_price: guard_price,
        });
    }

    // Sort by venue_index for deterministic output
    allocations.sort_by_key(|a| a.venue_index);

    allocations
}

/// Convert a hedge plan into abstract OrderIntents.
pub fn hedge_plan_to_order_intents(plan: &HedgePlan) -> Vec<OrderIntent> {
    let mut out = Vec::new();
    hedge_plan_to_order_intents_into(plan, &mut out);
    out
}

/// Convert a hedge plan into abstract OrderIntents (buffer-reusing variant).
///
/// Clears `out` and pushes intents into it, reusing capacity.
/// Use this in hot paths to avoid per-tick allocations.
pub fn hedge_plan_to_order_intents_into(plan: &HedgePlan, out: &mut Vec<OrderIntent>) {
    out.clear();

    for alloc in &plan.allocations {
        out.push(OrderIntent::Place(PlaceOrderIntent {
            venue_index: alloc.venue_index,
            venue_id: alloc.venue_id.clone(),
            side: alloc.side,
            price: alloc.est_price,
            size: alloc.size,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
        }));
    }
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
    let mut out = Vec::new();
    compute_hedge_orders_into(cfg, state, now_ms, &mut out);
    out
}

/// Compute hedge orders directly (buffer-reusing variant).
///
/// Clears `out` and pushes intents into it, reusing capacity.
/// Use this in hot paths to avoid per-tick allocations.
///
/// # Arguments
/// * `cfg` - Strategy configuration
/// * `state` - Current global state
/// * `now_ms` - Current timestamp in milliseconds (required for staleness gating)
/// * `out` - Output buffer for order intents
pub fn compute_hedge_orders_into(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    out: &mut Vec<OrderIntent>,
) {
    out.clear();
    if let Some(plan) = compute_hedge_plan(cfg, state, now_ms) {
        for alloc in &plan.allocations {
            out.push(OrderIntent::Place(PlaceOrderIntent {
                venue_index: alloc.venue_index,
                venue_id: alloc.venue_id.clone(),
                side: alloc.side,
                price: alloc.est_price,
                size: alloc.size,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: None,
            }));
        }
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // -------------------------------------------------------------------------
    // Margin constraint helper tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_increases_abs_exposure() {
        // Positive position, buying more -> increases
        assert!(increases_abs_exposure(10.0, 5.0));

        // Positive position, selling some -> decreases
        assert!(!increases_abs_exposure(10.0, -5.0));

        // Positive position, selling all and going short -> may decrease overall
        assert!(!increases_abs_exposure(10.0, -15.0)); // |10| -> |-5| = 5 < 10

        // Negative position, selling more -> increases
        assert!(increases_abs_exposure(-10.0, -5.0));

        // Negative position, buying to cover -> decreases
        assert!(!increases_abs_exposure(-10.0, 5.0));

        // Zero position, any trade increases
        assert!(increases_abs_exposure(0.0, 5.0));
        assert!(increases_abs_exposure(0.0, -5.0));
    }

    #[test]
    fn test_compute_abs_limit_after_trade() {
        // Normal case
        // q_old=10, margin=1000, leverage=10, safety=0.95, price=100
        // additional_cap = (1000 * 10 * 0.95) / 100 = 95
        // abs_limit = 10 + 95 = 105
        let limit = compute_abs_limit_after_trade(10.0, 1000.0, 10.0, 0.95, 100.0);
        assert!(approx_eq(limit, 105.0));

        // Zero margin -> no additional capacity
        let limit = compute_abs_limit_after_trade(10.0, 0.0, 10.0, 0.95, 100.0);
        assert!(approx_eq(limit, 10.0));

        // Zero leverage -> no additional capacity
        let limit = compute_abs_limit_after_trade(10.0, 1000.0, 0.0, 0.95, 100.0);
        assert!(approx_eq(limit, 10.0));

        // Zero price -> no additional capacity (edge case)
        let limit = compute_abs_limit_after_trade(10.0, 1000.0, 10.0, 0.95, 0.0);
        assert!(approx_eq(limit, 10.0));
    }

    #[test]
    fn test_cap_dq_by_abs_limit_no_capping_needed() {
        // q_old=10, desired_dq=5, limit=20
        // q_new = 15, |15| <= 20, so no capping
        let capped = cap_dq_by_abs_limit(10.0, 5.0, 20.0);
        assert!(approx_eq(capped, 5.0));
    }

    #[test]
    fn test_cap_dq_by_abs_limit_positive_cap() {
        // q_old=10, desired_dq=15, limit=20
        // q_new = 25, |25| > 20, need to cap
        // capped_dq = 20 - 10 = 10
        let capped = cap_dq_by_abs_limit(10.0, 15.0, 20.0);
        assert!(approx_eq(capped, 10.0));
    }

    #[test]
    fn test_cap_dq_by_abs_limit_negative_cap() {
        // q_old=-10, desired_dq=-15, limit=20
        // q_new = -25, |-25| > 20, need to cap
        // capped_dq = -20 - (-10) = -10
        let capped = cap_dq_by_abs_limit(-10.0, -15.0, 20.0);
        assert!(approx_eq(capped, -10.0));
    }

    #[test]
    fn test_cap_dq_by_abs_limit_crossing_zero() {
        // q_old=10, desired_dq=-25, limit=12
        // q_new = -15, |-15| > 12, need to cap
        // But we're crossing zero and ending negative
        // capped_dq = -12 - 10 = -22 (but wait, that's wrong)
        // Actually: we want |q_old + dq| <= limit
        // If ending negative: q_old + dq = -limit => dq = -limit - q_old = -12 - 10 = -22
        // But -22 > -25 in magnitude terms... let's check the logic

        // Actually, crossing zero should be allowed more freely since it reduces exposure first
        // q_old=10, selling 10 gets us to 0, then selling another 12 is allowed (limit=12)
        // So total allowed = -22, but we only want -25
        // -25 brings us to -15, |-15| > 12, so cap to -12 from 10
        // dq = -22 to get from 10 to -12
        let capped = cap_dq_by_abs_limit(10.0, -25.0, 12.0);
        assert!(approx_eq(capped, -22.0));
    }

    #[test]
    fn test_cap_dq_reducing_exposure() {
        // q_old=10, desired_dq=-5, limit=8
        // q_new = 5, |5| <= 8, so no capping (reducing exposure)
        let capped = cap_dq_by_abs_limit(10.0, -5.0, 8.0);
        assert!(approx_eq(capped, -5.0));
    }

    // -------------------------------------------------------------------------
    // Multi-chunk tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_chunk_candidates_single_chunk() {
        let chunks = build_chunk_candidates(
            0,
            Arc::from("venue_a"),
            Side::Buy,
            100.0, // guard price
            5.0,   // max_dq_abs
            10.0,  // chunk_size (larger than max)
            1.0,   // base_unit_cost
            0.0,   // no convexity
        );

        // Only one partial chunk of 5.0
        assert_eq!(chunks.len(), 1);
        assert!(approx_eq(chunks[0].dq_signed, 5.0));
        assert!(approx_eq(chunks[0].unit_cost, 1.0));
    }

    #[test]
    fn test_build_chunk_candidates_multiple_chunks() {
        let chunks = build_chunk_candidates(
            0,
            Arc::from("venue_a"),
            Side::Sell,
            100.0, // guard price
            10.0,  // max_dq_abs
            3.0,   // chunk_size
            1.0,   // base_unit_cost
            0.1,   // convexity per chunk
        );

        // 10 / 3 = 3 full chunks + 1 TAO remainder
        assert_eq!(chunks.len(), 4);

        // Check chunks are properly sized
        assert!(approx_eq(chunks[0].dq_signed, -3.0)); // Sell = negative
        assert!(approx_eq(chunks[1].dq_signed, -3.0));
        assert!(approx_eq(chunks[2].dq_signed, -3.0));
        assert!(approx_eq(chunks[3].dq_signed, -1.0)); // remainder

        // Check convexity costs
        assert!(approx_eq(chunks[0].unit_cost, 1.0)); // chunk 0
        assert!(approx_eq(chunks[1].unit_cost, 1.1)); // chunk 1
        assert!(approx_eq(chunks[2].unit_cost, 1.2)); // chunk 2
        assert!(approx_eq(chunks[3].unit_cost, 1.3)); // chunk 3 (remainder)
    }

    #[test]
    fn test_greedy_allocate_chunks_deterministic() {
        let chunks = vec![
            ChunkCandidate {
                venue_index: 1,
                venue_id: Arc::from("b"),
                chunk_index: 0,
                dq_signed: 5.0,
                unit_cost: 1.0,
                guard_price: 100.0,
                side: Side::Buy,
            },
            ChunkCandidate {
                venue_index: 0,
                venue_id: Arc::from("a"),
                chunk_index: 0,
                dq_signed: 5.0,
                unit_cost: 1.0, // Same cost
                guard_price: 100.0,
                side: Side::Buy,
            },
        ];

        let allocs = greedy_allocate_chunks(chunks.clone(), 7.0);

        // With same cost, should prefer venue "a" (lexicographically first)
        assert_eq!(allocs.len(), 2);
        // After sorting by venue_id, "a" (venue 0) should come first in allocation
        // But allocation order depends on greedy selection, not final output order
        // The key is determinism
        let allocs2 = greedy_allocate_chunks(chunks, 7.0);
        assert_eq!(allocs.len(), allocs2.len());
    }

    #[test]
    fn test_greedy_allocate_chunks_prefers_lower_cost() {
        let chunks = vec![
            ChunkCandidate {
                venue_index: 0,
                venue_id: Arc::from("a"),
                chunk_index: 0,
                dq_signed: 5.0,
                unit_cost: 2.0, // Higher cost
                guard_price: 100.0,
                side: Side::Buy,
            },
            ChunkCandidate {
                venue_index: 1,
                venue_id: Arc::from("b"),
                chunk_index: 0,
                dq_signed: 5.0,
                unit_cost: 1.0, // Lower cost
                guard_price: 100.0,
                side: Side::Buy,
            },
        ];

        let allocs = greedy_allocate_chunks(chunks, 5.0);

        // Should allocate from venue 1 (lower cost) first
        assert_eq!(allocs.len(), 1);
        assert_eq!(allocs[0].0, 1); // venue_index
    }

    // -------------------------------------------------------------------------
    // Legacy helper tests
    // -------------------------------------------------------------------------

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
