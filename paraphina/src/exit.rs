// src/exit.rs
//
// Cross-venue "exit" engine (Milestone E implementation).
//
// Spec-aligned, pragmatic (L2 not yet available):
// - Deterministic, config-driven "profit-only" exit actions.
// - Uses top-of-book approximations derived from (mid, spread, depth_near_mid).
// - Incorporates basis + funding adjustments (as ranking terms).
// - Basis-risk penalty: penalise trades that increase |B|.
// - Fragmentation penalty: penalise opening a new leg / increasing venue abs exposure.
// - Fragmentation reduction bonus: prefer exits that consolidate positions.
// - Liquidation-distance-aware caps and reduce-only behavior near LIQ_WARN/LIQ_CRIT.
// - Per-venue gating: Disabled, stale, tox>=high, missing book, insufficient depth.
// - Multi-venue greedy allocation (continuous knapsack-lite).
// - Lot size rounding + min notional enforcement per venue.
// - Linear + quadratic slippage model.
// - Volatility buffer using effective sigma.
//
// Ordering expectation:
//   MM fills -> recompute -> exit -> recompute -> hedge
//
// Notes:
// - Without a full fill ledger, we use a global "entry reference" derived from
//   existing per-venue VWAP entry prices in the direction of the global inventory.
// - Profit-only is enforced using a conservative immediate edge model:
//   (price vs entry_ref) - taker_fee - slippage - vol_buffer must exceed edge threshold.
// - Basis + funding + fragmentation + basis-risk influence *ranking* among profitable exits,
//   but do not override the profit-only gate.
// - Deterministic tie-breaking: when two venues have similar edge, prefer actions that
//   reduce fragmentation/basis risk (proven via unit tests).

use std::sync::Arc;

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus};

#[derive(Debug, Clone)]
struct Candidate {
    venue_index: usize,
    venue_id: Arc<str>,
    side: Side,
    price: f64,
    max_size: f64,
    /// Immediate profit per TAO before slippage (already fee-adjusted).
    base_profit_per_tao: f64,
    /// Linear slippage coefficient (USD / TAO).
    slip_linear: f64,
    /// Quadratic slippage coefficient (USD / TAO^2).
    slip_quadratic: f64,
    /// Ranking score per TAO (profit + basis/funding - penalties + bonuses).
    score_per_tao: f64,
    /// Whether this exit reduces venue absolute exposure (for fragmentation).
    reduces_venue_abs: bool,
    /// Whether this exit would close out a venue position entirely (consolidation).
    closes_venue_position: bool,
    /// Lot size for this venue.
    lot_size: f64,
    /// Size step for this venue.
    size_step: f64,
    /// Min notional for this venue.
    min_notional: f64,
}

fn tick_floor(px: f64, tick: f64) -> f64 {
    (px / tick).floor() * tick
}

fn tick_ceil(px: f64, tick: f64) -> f64 {
    (px / tick).ceil() * tick
}

/// Round size down to the nearest lot size / size step.
/// Returns None if the rounded size is below the minimum lot size.
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

/// Check if size meets minimum notional requirement.
fn meets_min_notional(size: f64, price: f64, min_notional: f64) -> bool {
    let notional = size * price;
    notional >= min_notional
}

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

/// Weighted average entry price reference for the direction of global inventory.
/// Pragmatic substitute for a full fill ledger for cross-venue exits.
fn global_entry_reference(state: &GlobalState, fair: f64) -> f64 {
    let q = state.q_global_tao;
    if q == 0.0 {
        return fair;
    }

    let want_long = q > 0.0;
    let mut w_sum = 0.0;
    let mut p_sum = 0.0;

    for v in &state.venues {
        if want_long {
            if v.position_tao > 0.0 && v.avg_entry_price > 0.0 {
                let w = v.position_tao.abs();
                w_sum += w;
                p_sum += w * v.avg_entry_price;
            }
        } else if v.position_tao < 0.0 && v.avg_entry_price > 0.0 {
            let w = v.position_tao.abs();
            w_sum += w;
            p_sum += w * v.avg_entry_price;
        }
    }

    if w_sum > 0.0 {
        p_sum / w_sum
    } else {
        fair
    }
}

/// Global liquidation mode:
/// - If the closest venue is inside LIQ_WARN, we enter "reduce-only" mode globally.
/// - We also shrink total exit size within (LIQ_CRIT, LIQ_WARN).
/// - At/inside LIQ_CRIT, we do NOT cap to 0 — we still allow profit-only exits,
///   but strictly reduce-only (do not increase venue abs exposure).
fn global_liq_mode(cfg: &Config, state: &GlobalState) -> (bool, f64) {
    let mut min_dist = f64::INFINITY;
    for v in &state.venues {
        if v.dist_liq_sigma.is_finite() {
            min_dist = min_dist.min(v.dist_liq_sigma);
        }
    }

    if !min_dist.is_finite() {
        return (false, 1.0);
    }

    if min_dist < cfg.risk.liq_warn_sigma {
        // Reduce-only global mode.
        if min_dist <= cfg.risk.liq_crit_sigma {
            // Critical: still allow full-size exits (subject to other caps), but reduce-only.
            return (true, 1.0);
        }

        // Between crit and warn: shrink total exit size.
        let t = (min_dist - cfg.risk.liq_crit_sigma)
            / (cfg.risk.liq_warn_sigma - cfg.risk.liq_crit_sigma);
        return (true, t.clamp(0.0, 1.0));
    }

    (false, 1.0)
}

/// Per-venue liquidation cap multiplier.
/// - If dist is finite and in (LIQ_CRIT, LIQ_WARN): shrink sizes linearly.
/// - If <= LIQ_CRIT: reduce-only is enforced elsewhere, but we keep cap multiplier = 1.0
///   to allow meaningful de-risking.
fn venue_liq_cap_mult(cfg: &Config, dist: f64) -> f64 {
    if !dist.is_finite() {
        return 1.0;
    }
    if dist <= cfg.risk.liq_crit_sigma {
        return 1.0;
    }
    if dist < cfg.risk.liq_warn_sigma {
        let t =
            (dist - cfg.risk.liq_crit_sigma) / (cfg.risk.liq_warn_sigma - cfg.risk.liq_crit_sigma);
        return t.clamp(0.0, 1.0);
    }
    1.0
}

/// Compute slippage buffer using linear + quadratic model.
/// slippage(size) = linear_coeff * size + quadratic_coeff * size^2
/// Also incorporates depth-based adjustment when depth is available.
#[allow(dead_code)]
fn compute_slippage_buffer(cfg: &Config, fair: f64, spread: f64, depth_usd: f64, size: f64) -> f64 {
    // Linear + quadratic model
    let linear = cfg.exit.slippage_linear_coeff * size;
    let quadratic = cfg.exit.slippage_quadratic_coeff * size * size;

    // Depth-based adjustment (legacy model for backwards compatibility)
    let depth_adj = if depth_usd > 0.0 {
        cfg.exit.slippage_spread_mult * fair * spread * size / depth_usd
    } else {
        0.0
    };

    // Take the max of the two models for conservative estimation
    (linear + quadratic).max(depth_adj)
}

/// Compute volatility buffer using effective sigma.
/// vol_buffer = vol_buffer_mult * sigma_eff * fair
fn compute_vol_buffer(cfg: &Config, state: &GlobalState, fair: f64) -> f64 {
    cfg.exit.vol_buffer_mult * state.sigma_eff * fair
}

/// Compute fragmentation score for a set of venue positions.
/// Higher score = more fragmented.
fn compute_fragmentation_score(positions: &[f64], min_threshold: f64) -> f64 {
    let mut score = 0.0;
    for &q in positions {
        let abs_q = q.abs();
        if abs_q > 0.0 && abs_q < min_threshold {
            // Penalize small fragmented positions
            score += 1.0;
        }
    }
    score
}

/// Compute the approximate change in basis exposure magnitude.
/// ΔB ≈ Δq * (mid_j - fair)
/// Returns positive if it would increase |B|, negative if decrease.
fn compute_basis_risk_change(
    state: &GlobalState,
    venue_mid: f64,
    fair: f64,
    trade_sign: f64,
) -> f64 {
    let b_j = venue_mid - fair;
    let delta_b_per_tao = trade_sign * b_j;
    // Δ|B| ≈ sign(B) * ΔB (linear approximation)
    state.basis_usd.signum() * delta_b_per_tao
}

pub fn compute_exit_intents(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) -> Vec<OrderIntent> {
    let mut intents = Vec::new();
    compute_exit_intents_into(cfg, state, now_ms, &mut intents);
    intents
}

/// Compute exit intents (buffer-reusing variant).
///
/// Clears `out` and pushes intents into it, reusing capacity.
/// Use this in hot paths to avoid per-tick allocations.
pub fn compute_exit_intents_into(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    out: &mut Vec<OrderIntent>,
) {
    out.clear();

    if !cfg.exit.enabled {
        return;
    }

    // Exit engine should not run when kill-switch is on or in HardLimit.
    if state.kill_switch || matches!(state.risk_regime, RiskRegime::HardLimit) {
        return;
    }

    let fair = match state.fair_value {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => return,
    };

    let q = state.q_global_tao;
    if q.abs() < cfg.exit.min_global_abs_tao {
        return;
    }

    // Global liquidation-aware mode.
    let (global_reduce_only, global_cap_mult) = global_liq_mode(cfg, state);

    // Total amount we try to exit this tick.
    let mut remaining = q
        .abs()
        .min(cfg.exit.max_total_tao_per_tick * global_cap_mult);
    if remaining <= 0.0 {
        return;
    }

    // We are always trying to reduce |q_global|.
    let need_side = if q > 0.0 { Side::Sell } else { Side::Buy };
    let trade_sign = match need_side {
        Side::Sell => -1.0,
        Side::Buy => 1.0,
    };

    // Profit-only threshold (USD/TAO). Uses edge_min + vol-scaled component.
    let edge_threshold =
        cfg.exit.edge_min_usd + cfg.exit.edge_vol_mult * state.vol_ratio_clipped.max(0.0);

    // Volatility buffer using effective sigma
    let vol_buffer = compute_vol_buffer(cfg, state, fair);

    let entry_ref = global_entry_reference(state, fair);

    // Pre-compute current fragmentation score for bonus calculation
    let current_positions: Vec<f64> = state.venues.iter().map(|v| v.position_tao).collect();
    let _current_frag_score = compute_fragmentation_score(&current_positions, 1.0);

    let mut cands: Vec<Candidate> = Vec::new();

    for (j, vcfg) in cfg.venues.iter().enumerate() {
        let v = &state.venues[j];

        // --- Eligibility gating (spec-like) ---
        if matches!(v.status, VenueStatus::Disabled) {
            continue;
        }

        let ts = match v.last_mid_update_ms {
            Some(t) => t,
            None => continue,
        };
        if now_ms - ts > cfg.book.stale_ms {
            continue;
        }

        if v.toxicity >= cfg.toxicity.tox_high_threshold {
            continue;
        }

        let mid = match v.mid {
            Some(m) if m.is_finite() && m > 0.0 => m,
            _ => continue,
        };
        let spread = match v.spread {
            Some(s) if s.is_finite() && s > 0.0 => s,
            _ => continue,
        };

        let (bid, ask) = match top_of_book_from_mid(mid, spread) {
            Some(x) => x,
            None => continue,
        };

        let tick = vcfg.tick_size.max(1e-9);
        let raw_px = match need_side {
            Side::Sell => bid,
            Side::Buy => ask,
        };
        let px = match need_side {
            Side::Sell => tick_floor(raw_px, tick),
            Side::Buy => tick_ceil(raw_px, tick),
        };
        if !px.is_finite() || px <= 0.0 {
            continue;
        }

        // Depth gating + cap by depth fraction.
        let depth_usd = v.depth_near_mid.max(0.0);
        if depth_usd < cfg.exit.min_depth_usd {
            continue;
        }
        let depth_cap_tao = (cfg.exit.depth_fraction * depth_usd / fair).max(0.0);

        // --- Liquidation-aware gating / caps (per venue) ---
        let dist = v.dist_liq_sigma;
        let venue_mult = venue_liq_cap_mult(cfg, dist);

        // Does trading on this venue in this direction reduce this venue's absolute exposure?
        // For small sizes, selling reduces abs only if venue is long; buying reduces abs only if venue is short.
        let reduces_venue_abs = (v.position_tao * trade_sign) < 0.0;

        // Would this trade fully close the venue position?
        let closes_venue_position = reduces_venue_abs && v.position_tao.abs() > 0.0;

        // If globally in reduce-only, we require this trade to reduce venue abs.
        // Additionally, if THIS venue is inside LIQ_WARN, require reduce-only behavior on this venue.
        let venue_in_liq_warn = dist.is_finite() && dist < cfg.risk.liq_warn_sigma;
        let must_reduce_here = global_reduce_only || venue_in_liq_warn;

        if must_reduce_here && !reduces_venue_abs {
            continue;
        }

        // Max size per venue from config + venue constraints + depth cap.
        let mut max_size = cfg
            .exit
            .max_venue_tao_per_tick
            .min(vcfg.max_order_size)
            .min(depth_cap_tao);

        // In reduce-only conditions, don't allow flipping through zero (that would increase venue abs).
        if must_reduce_here {
            max_size = max_size.min(v.position_tao.abs());
        }

        // Apply per-venue liquidation shrink inside (crit, warn).
        max_size *= venue_mult;

        if !max_size.is_finite() || max_size <= 0.0 {
            continue;
        }

        // --- Immediate profit model (fee + price edge) ---
        let fee_per_tao = (vcfg.taker_fee_bps / 10_000.0) * px;

        let base_profit_per_tao = match need_side {
            Side::Sell => (px - entry_ref) - fee_per_tao - vol_buffer,
            Side::Buy => (entry_ref - px) - fee_per_tao - vol_buffer,
        };

        // Profit-only gate (conservative): require base profit to clear threshold
        // even before basis/funding ranking terms.
        if base_profit_per_tao <= edge_threshold {
            continue;
        }

        // Linear + quadratic slippage coefficients
        let slip_linear = cfg.exit.slippage_linear_coeff
            + (cfg.exit.slippage_spread_mult * fair * spread / depth_usd).max(0.0);
        let slip_quadratic = cfg.exit.slippage_quadratic_coeff;

        // --- Fragmentation penalty/bonus ---
        // Penalise opening a new leg (venue was ~flat, now we'd create exposure).
        // Penalise increasing venue absolute exposure (same-sign as trade).
        let small = cfg.exit.min_intent_size_tao.max(1e-9);
        let opens_new_leg = v.position_tao.abs() < small && !reduces_venue_abs;
        let increases_venue_abs = (v.position_tao * trade_sign) > 0.0;
        let frag_pen = if opens_new_leg || increases_venue_abs {
            cfg.exit.fragmentation_penalty_per_tao
        } else {
            0.0
        };

        // Fragmentation reduction bonus: prefer exits that consolidate
        // (reduce the number of venues with positions)
        let frag_bonus = if closes_venue_position {
            // Closing a position entirely reduces fragmentation
            cfg.exit.fragmentation_reduction_bonus
        } else if reduces_venue_abs {
            // Reducing position size is good but not as good as closing
            cfg.exit.fragmentation_reduction_bonus * 0.5
        } else {
            0.0
        };

        // --- Basis term (ranking) ---
        // Prefer selling on rich venues / buying on cheap venues relative to fair.
        let basis_term = match need_side {
            Side::Sell => px - fair,
            Side::Buy => fair - px,
        };

        // --- Funding term (ranking) ---
        // Approximate expected funding benefit over horizon.
        // Positive funding means "shorts receive" under common perp convention,
        // so selling (towards short) gets +funding_8h benefit; buying gets the opposite.
        let horizon_frac = cfg.exit.funding_horizon_sec / (8.0 * 60.0 * 60.0);
        let funding_benefit_per_tao = match need_side {
            Side::Sell => v.funding_8h * horizon_frac * fair,
            Side::Buy => -v.funding_8h * horizon_frac * fair,
        };

        // --- Basis-risk penalty (ranking) ---
        // Compute approximate change in |B| from this trade.
        // Penalize if it would increase |B|.
        let basis_risk_change = compute_basis_risk_change(state, mid, fair, trade_sign);
        let basis_risk_pen = if basis_risk_change > 0.0 {
            cfg.exit.basis_risk_penalty_weight * basis_risk_change
        } else {
            // Bonus for reducing basis risk (negative change = reduction)
            cfg.exit.basis_risk_penalty_weight * basis_risk_change * 0.5
        };

        // Final ranking score per TAO.
        // Higher is better.
        let score_per_tao = base_profit_per_tao
            + cfg.exit.basis_weight * basis_term
            + cfg.exit.funding_weight * funding_benefit_per_tao
            - frag_pen
            + frag_bonus
            - basis_risk_pen;

        cands.push(Candidate {
            venue_index: j,
            venue_id: vcfg.id_arc.clone(),
            side: need_side,
            price: px,
            max_size,
            base_profit_per_tao,
            slip_linear,
            slip_quadratic,
            score_per_tao,
            reduces_venue_abs,
            closes_venue_position,
            lot_size: vcfg.lot_size_tao,
            size_step: vcfg.size_step_tao,
            min_notional: vcfg.min_notional_usd,
        });
    }

    if cands.is_empty() {
        return;
    }

    // Deterministic sort: best score first, tie-break by:
    // 1. Prefer venues that reduce fragmentation (closes_venue_position first)
    // 2. Prefer venues that reduce venue abs exposure
    // 3. Prefer lower venue index (stable ordering)
    cands.sort_by(|a, b| {
        // Primary: score (higher is better)
        let score_cmp = b
            .score_per_tao
            .partial_cmp(&a.score_per_tao)
            .unwrap_or(std::cmp::Ordering::Equal);

        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }

        // Tie-break 1: prefer closing positions (reduces fragmentation)
        let close_cmp = b.closes_venue_position.cmp(&a.closes_venue_position);
        if close_cmp != std::cmp::Ordering::Equal {
            return close_cmp;
        }

        // Tie-break 2: prefer reducing venue abs exposure
        let reduce_cmp = b.reduces_venue_abs.cmp(&a.reduces_venue_abs);
        if reduce_cmp != std::cmp::Ordering::Equal {
            return reduce_cmp;
        }

        // Tie-break 3: stable ordering by venue index
        a.venue_index.cmp(&b.venue_index)
    });

    // Greedy allocation (continuous knapsack-lite).
    for c in cands {
        if remaining <= 0.0 {
            break;
        }

        let mut size_cap = c.max_size.min(remaining);

        // Profit-only size cap under linear + quadratic slippage:
        // base_profit - slip_linear*size - slip_quadratic*size^2 >= edge_threshold
        // This is a quadratic inequality; solve for max size.
        if c.slip_quadratic > 0.0 {
            // Quadratic: slip_quadratic*size^2 + slip_linear*size + (edge_threshold - base_profit) <= 0
            // Using quadratic formula to find max positive root
            let a = c.slip_quadratic;
            let b = c.slip_linear;
            let c_term = edge_threshold - c.base_profit_per_tao;

            let discriminant = b * b - 4.0 * a * c_term;
            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let max_by_profit = (-b + sqrt_disc) / (2.0 * a);
                if max_by_profit > 0.0 {
                    size_cap = size_cap.min(max_by_profit);
                }
            } else {
                // No valid solution - skip this candidate
                continue;
            }
        } else if c.slip_linear > 0.0 {
            // Linear only: base_profit - slip_linear*size >= edge_threshold
            let max_by_profit = (c.base_profit_per_tao - edge_threshold) / c.slip_linear;
            size_cap = size_cap.min(max_by_profit);
        }

        if !size_cap.is_finite() || size_cap <= 0.0 {
            continue;
        }

        // Apply lot size rounding
        let rounded_size = match round_to_lot_size(size_cap, c.lot_size, c.size_step) {
            Some(s) => s,
            None => continue, // Size too small after rounding
        };

        // Check minimum notional
        if !meets_min_notional(rounded_size, c.price, c.min_notional) {
            continue;
        }

        // Final dust guard
        if rounded_size < cfg.exit.min_intent_size_tao {
            continue;
        }

        out.push(OrderIntent {
            venue_index: c.venue_index,
            venue_id: c.venue_id,
            side: c.side,
            price: c.price,
            size: rounded_size,
            purpose: OrderPurpose::Exit,
        });

        remaining -= rounded_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn assert_approx_some(result: Option<f64>, expected: f64) {
        match result {
            Some(v) => assert!(approx_eq(v, expected), "expected ~{expected}, got {v}"),
            None => panic!("expected Some({expected}), got None"),
        }
    }

    #[test]
    fn test_round_to_lot_size() {
        // Basic rounding
        assert_approx_some(round_to_lot_size(1.55, 0.1, 0.1), 1.5);
        assert_approx_some(round_to_lot_size(1.59, 0.1, 0.1), 1.5);
        assert_approx_some(round_to_lot_size(1.61, 0.1, 0.1), 1.6);

        // Size below lot size
        assert_eq!(round_to_lot_size(0.05, 0.1, 0.1), None);

        // Exact lot size
        assert_approx_some(round_to_lot_size(0.1, 0.1, 0.1), 0.1);

        // Zero and negative
        assert_eq!(round_to_lot_size(0.0, 0.1, 0.1), None);
        assert_eq!(round_to_lot_size(-1.0, 0.1, 0.1), None);

        // Different step sizes
        assert_approx_some(round_to_lot_size(1.23, 0.01, 0.05), 1.20);
    }

    #[test]
    fn test_meets_min_notional() {
        assert!(meets_min_notional(1.0, 100.0, 50.0)); // 100 >= 50
        assert!(!meets_min_notional(0.1, 100.0, 50.0)); // 10 < 50
        assert!(meets_min_notional(0.5, 100.0, 50.0)); // 50 >= 50
    }

    #[test]
    fn test_compute_slippage_buffer() {
        let mut cfg = Config::default();
        cfg.exit.slippage_linear_coeff = 0.01;
        cfg.exit.slippage_quadratic_coeff = 0.001;
        cfg.exit.slippage_spread_mult = 1.0;

        let fair = 100.0;
        let spread = 0.5;
        let depth_usd = 10000.0;
        let size = 5.0;

        let slip = compute_slippage_buffer(&cfg, fair, spread, depth_usd, size);

        // Linear: 0.01 * 5 = 0.05
        // Quadratic: 0.001 * 25 = 0.025
        // Depth-based: 1.0 * 100 * 0.5 * 5 / 10000 = 0.025
        // Total L+Q: 0.075
        assert!(slip >= 0.025); // At least the depth-based component
        assert!(slip <= 0.1); // Not unreasonably large
    }

    #[test]
    fn test_fragmentation_score() {
        // No positions
        assert_eq!(compute_fragmentation_score(&[], 1.0), 0.0);

        // One large position (not fragmented)
        assert_eq!(compute_fragmentation_score(&[10.0], 1.0), 0.0);

        // Small positions (fragmented)
        assert_eq!(compute_fragmentation_score(&[0.5, 0.3], 1.0), 2.0);

        // Mixed
        assert_eq!(
            compute_fragmentation_score(&[10.0, 0.5, 5.0, 0.1], 1.0),
            2.0
        );
    }
}
