// src/exit.rs
//
// Cross-venue "exit" engine.
//
// Spec-aligned, pragmatic (L2 not yet available):
// - Deterministic, config-driven "profit-only" exit actions.
// - Uses top-of-book approximations derived from (mid, spread, depth_near_mid).
// - Incorporates basis + funding adjustments (as ranking terms).
// - Basis-risk penalty proxy (penalise trades that increase |B|).
// - Fragmentation penalty proxy (penalise opening a new leg / increasing venue abs exposure).
// - Liquidation-distance-aware caps and reduce-only behavior near LIQ_WARN/LIQ_CRIT.
// - Per-venue gating: Disabled, stale, tox>=high, missing book, insufficient depth.
// - Multi-venue greedy allocation (continuous knapsack-lite).
//
// Ordering expectation:
//   MM fills -> recompute -> exit -> recompute -> hedge
//
// Notes:
// - Without a full fill ledger, we use a global "entry reference" derived from
//   existing per-venue VWAP entry prices in the direction of the global inventory.
// - Profit-only is enforced using a conservative immediate edge model:
//   (price vs entry_ref) - taker_fee - slippage(size) must exceed edge threshold.
// - Basis + funding + fragmentation + basis-risk influence *ranking* among profitable exits,
//   but do not override the profit-only gate.

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus};

#[derive(Debug, Clone)]
struct Candidate {
    venue_index: usize,
    venue_id: String,
    side: Side,
    price: f64,
    max_size: f64,
    /// Immediate profit per TAO before linear slippage (already fee-adjusted).
    base_profit_per_tao: f64,
    /// Linear slippage coefficient (USD / TAO^2) so slippage(size) ~= slip_coef * size.
    slip_coef: f64,
    /// Ranking score per TAO (profit + basis/funding - penalties).
    score_per_tao: f64,
}

fn tick_floor(px: f64, tick: f64) -> f64 {
    (px / tick).floor() * tick
}

fn tick_ceil(px: f64, tick: f64) -> f64 {
    (px / tick).ceil() * tick
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

pub fn compute_exit_intents(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) -> Vec<OrderIntent> {
    if !cfg.exit.enabled {
        return Vec::new();
    }

    // Exit engine should not run when kill-switch is on or in HardLimit.
    if state.kill_switch || matches!(state.risk_regime, RiskRegime::HardLimit) {
        return Vec::new();
    }

    let fair = match state.fair_value {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => return Vec::new(),
    };

    let q = state.q_global_tao;
    if q.abs() < cfg.exit.min_global_abs_tao {
        return Vec::new();
    }

    // Global liquidation-aware mode.
    let (global_reduce_only, global_cap_mult) = global_liq_mode(cfg, state);

    // Total amount we try to exit this tick.
    let mut remaining = q
        .abs()
        .min(cfg.exit.max_total_tao_per_tick * global_cap_mult);
    if remaining <= 0.0 {
        return Vec::new();
    }

    // We are always trying to reduce |q_global|.
    let need_side = if q > 0.0 { Side::Sell } else { Side::Buy };
    let trade_sign = match need_side {
        Side::Sell => -1.0,
        Side::Buy => 1.0,
    };

    // Profit-only threshold (USD/TAO). Keep the existing semantics:
    // edge_min_usd + edge_vol_mult * vol_ratio_clipped.
    // (This matches your current tests/config expectations.)
    let edge_threshold =
        cfg.exit.edge_min_usd + cfg.exit.edge_vol_mult * state.vol_ratio_clipped.max(0.0);

    let entry_ref = global_entry_reference(state, fair);

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
            Side::Sell => (px - entry_ref) - fee_per_tao,
            Side::Buy => (entry_ref - px) - fee_per_tao,
        };

        // Profit-only gate (conservative): require base profit to clear threshold
        // even before basis/funding ranking terms.
        if base_profit_per_tao <= edge_threshold {
            continue;
        }

        // Linear slippage coefficient (USD/TAO^2).
        // We approximate market impact using spread and depth:
        // slip(size) ≈ (slippage_spread_mult * fair * spread / depth_usd) * size
        let slip_coef = (cfg.exit.slippage_spread_mult * fair * spread / depth_usd).max(0.0);

        // --- Fragmentation proxy ---
        // Penalise:
        //  - opening a new leg (venue was ~flat, now we'd create exposure),
        //  - increasing venue absolute exposure (same-sign as trade).
        //
        // Rewarding fragmentation reduction is intentionally omitted (conservative),
        // but ranking will naturally prefer reduce-only venues in stressed mode.
        let small = cfg.exit.min_intent_size_tao.max(1e-9);
        let opens_new_leg = v.position_tao.abs() < small && !reduces_venue_abs;
        let increases_venue_abs = (v.position_tao * trade_sign) > 0.0;
        let frag_pen = if opens_new_leg || increases_venue_abs {
            cfg.exit.fragmentation_penalty_per_tao
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

        // --- Basis-risk penalty proxy (ranking) ---
        // We approximate how this trade changes |B| using:
        //   B = Σ q_v * (mid_v - fair)
        // For a small trade Δq on venue j:
        //   ΔB ≈ Δq * (mid_j - fair)
        // We penalise if it would increase |B| in the local linear approximation:
        //   Δ|B| ≈ sign(B) * ΔB  (only when this is positive).
        //
        // Weight: reuse |basis_weight| as a conservative penalty scaler (no new config field).
        let b_j = mid - fair;
        let delta_b_per_tao = trade_sign * b_j; // ΔB per +1 TAO traded on this venue
        let approx_delta_abs_b_per_tao = state.basis_usd.signum() * delta_b_per_tao;
        let basis_risk_pen = cfg.exit.basis_weight.abs() * approx_delta_abs_b_per_tao.max(0.0);

        // Final ranking score per TAO.
        let score_per_tao = base_profit_per_tao
            + cfg.exit.basis_weight * basis_term
            + cfg.exit.funding_weight * funding_benefit_per_tao
            - frag_pen
            - basis_risk_pen;

        cands.push(Candidate {
            venue_index: j,
            venue_id: vcfg.id.clone(),
            side: need_side,
            price: px,
            max_size,
            base_profit_per_tao,
            slip_coef,
            score_per_tao,
        });
    }

    if cands.is_empty() {
        return Vec::new();
    }

    // Deterministic sort: best score first, tie-break by venue index.
    cands.sort_by(|a, b| {
        b.score_per_tao
            .partial_cmp(&a.score_per_tao)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.venue_index.cmp(&b.venue_index))
    });

    let mut intents = Vec::new();

    // Greedy allocation (continuous knapsack-lite).
    for c in cands {
        if remaining <= 0.0 {
            break;
        }

        let mut size_cap = c.max_size.min(remaining);

        // Profit-only size cap under linear slippage:
        //   base_profit - slip_coef*size >= edge_threshold
        if c.slip_coef > 0.0 {
            let max_by_profit = (c.base_profit_per_tao - edge_threshold) / c.slip_coef;
            size_cap = size_cap.min(max_by_profit);
        }

        if !size_cap.is_finite() || size_cap <= cfg.exit.min_intent_size_tao {
            continue;
        }

        // Clamp to avoid tiny dust.
        if size_cap < cfg.exit.min_intent_size_tao {
            continue;
        }

        intents.push(OrderIntent {
            venue_index: c.venue_index,
            venue_id: c.venue_id,
            side: c.side,
            price: c.price,
            size: size_cap,
            purpose: OrderPurpose::Exit,
        });

        remaining -= size_cap;
    }

    intents
}
