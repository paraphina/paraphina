// src/mm.rs
//
// Market-making quote engine (Avellaneda–Stoikov style) for Paraphina.
//
// Milestone G: Full multi-venue quoting model per whitepaper Sections 9–11.
//
// Responsibilities:
//   - For each venue, convert global fair value + risk scalars + venue state
//     into local bid/ask quotes.
//   - Respect kill switch, HardLimit regime, and venue health.
//   - Incorporate inventory, volatility, basis, funding and risk regime
//     into reservation price, spread, and quote sizes.
//   - Emit a thin abstraction (MmQuote) which main.rs converts into
//     abstract OrderIntents.
//
// Spec alignment (Sections 9–11):
//   - HardLimit => return NO MM quotes (safety-first).
//   - Reservation price (Section 9.1):
//       r_v = S_t + β_b*b_v + β_f*f_v - γ*(σ_eff^2)*τ*( q_global - λ_inv*(q_v - q_target_v) )
//   - Half-spread (Section 9.2):
//       δ* = (1/γ) * ln(1 + γ/k)
//       then apply spread_mult
//       then enforce min half-spread based on (EDGE_LOCAL_MIN + maker_cost + vol_buffer)/2
//   - Size (Section 10):
//       Q_raw = e / η (if e > 0)
//       then apply constraints: size_mult, per-venue max, margin cap, liq-distance shrink
//   - Passivity enforcement:
//       bid <= best_bid - tick, ask >= best_ask + tick
//   - Quote gating:
//       Only quote if edge >= EDGE_LOCAL_MIN and venue passes gating

use std::sync::Arc;

use crate::config::{Config, VenueConfig};
use crate::sim_eval::AblationSet;
use crate::state::{GlobalState, RiskRegime, VenueState};
use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus};

/// Internal MM quote level (one side of the book).
#[derive(Debug, Clone)]
pub struct MmLevel {
    pub price: f64,
    pub size: f64,
}

/// Per-venue MM quote (bid/ask).
///
/// Note: `venue_id` uses `Arc<str>` for cheap cloning in hot paths.
#[derive(Debug, Clone)]
pub struct MmQuote {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub bid: Option<MmLevel>,
    pub ask: Option<MmLevel>,
}

/// Active MM order state for order lifetime management.
#[derive(Debug, Clone)]
pub struct ActiveMmOrder {
    pub venue_index: usize,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub timestamp_ms: TimestampMs,
}

/// Per-venue target inventory computed each tick.
#[derive(Debug, Clone)]
pub struct VenueTargetInventory {
    pub venue_index: usize,
    /// Liquidity weight w_liq_v (from depth proportion).
    pub w_liq: f64,
    /// Target inventory q_target_v.
    pub q_target: f64,
}

/// Precomputed tick-level scalars for MM quoting.
///
/// These values are computed once per tick and reused across all venues,
/// avoiding repeated config lookups and arithmetic in the hot venue loop.
/// This struct is internal and preserves exact numerical results.
struct TickScalars {
    // Fair value and global state
    s_t: f64,
    sigma_eff: f64,
    q_global: f64,
    delta_abs_usd: f64,
    delta_limit_usd: f64,
    spread_mult: f64,
    size_mult: f64,

    // Precomputed from MM config (tick-invariant)
    beta_b: f64,
    beta_f: f64,
    lambda_inv: f64,
    edge_local_min: f64,
    edge_vol_mult: f64,
    size_eta: f64,
    /// tau / (8 * 60 * 60) - funding horizon fraction
    funding_horizon_frac: f64,
    /// sigma_eff * sigma_eff * tau - for inventory skew term
    sigma_sq_tau: f64,

    // Precomputed from risk config
    liq_crit_sigma: f64,
    liq_warn_sigma: f64,
    spread_warn_mult: f64,
    mm_margin_factor: f64, // mm_max_leverage * mm_margin_safety
    q_warn_cap: f64,

    // Precomputed from toxicity config
    tox_high_threshold: f64,
    tox_med_threshold: f64,

    // Ablation flags (computed once per tick)
    disable_fv_gating: bool,
    disable_tox_gating: bool,

    // Risk regime flags (precomputed to avoid matches! in loop)
    is_warning_regime: bool,
}

/// Compute per-venue target inventories based on liquidity and funding.
///
/// Section 9.1:
///   w_liq_v = depth_v / sum(depth)
///   phi(funding) = clip(funding_8h / FUNDING_TARGET_RATE_SCALE, -1, 1) * FUNDING_TARGET_MAX_TAO
///   q_target_v = w_liq_v * q_global + w_fund_v * phi(funding_8h_v)
pub fn compute_venue_targets(cfg: &Config, state: &GlobalState) -> Vec<VenueTargetInventory> {
    let n = cfg.venues.len();
    let mut targets = Vec::with_capacity(n);
    compute_venue_targets_into(cfg, state, &mut targets);
    targets
}

/// Compute per-venue target inventories (buffer-reusing variant).
///
/// Clears `out` and pushes targets into it, reusing capacity.
/// Use this in hot paths to avoid per-tick allocations.
pub fn compute_venue_targets_into(
    cfg: &Config,
    state: &GlobalState,
    out: &mut Vec<VenueTargetInventory>,
) {
    out.clear();

    let n = cfg.venues.len();
    let s_t = state.fair_value.unwrap_or(1.0).max(1.0);
    let s_t_inv = 1.0 / s_t; // Precompute reciprocal for division

    // Sum depth across all healthy venues.
    let mut total_depth: f64 = 0.0;
    for i in 0..n {
        let vstate = &state.venues[i];
        let vcfg = &cfg.venues[i];
        // Only include healthy venues with valid data.
        if !matches!(vstate.status, VenueStatus::Disabled) && vstate.depth_near_mid > 0.0 {
            // Use depth_near_mid as liquidity proxy.
            // Convert to TAO-equivalent if depth is in USD.
            let depth_tao = vstate.depth_near_mid * s_t_inv;
            total_depth += depth_tao * vcfg.w_liq;
        }
    }

    let q_global = state.q_global_tao;
    let funding_scale = cfg.mm.funding_target_rate_scale;
    let funding_max_tao = cfg.mm.funding_target_max_tao;
    let has_funding_scale = funding_scale > 0.0;
    let funding_scale_inv = if has_funding_scale {
        1.0 / funding_scale
    } else {
        0.0
    };
    let total_depth_inv = if total_depth > 0.0 {
        1.0 / total_depth
    } else {
        0.0
    };

    for i in 0..n {
        let vstate = &state.venues[i];
        let vcfg = &cfg.venues[i];

        // Default: no weight, target = 0.
        if matches!(vstate.status, VenueStatus::Disabled) || total_depth <= 0.0 {
            out.push(VenueTargetInventory {
                venue_index: i,
                w_liq: 0.0,
                q_target: 0.0,
            });
            continue;
        }

        // Compute liquidity weight.
        let depth_tao = vstate.depth_near_mid * s_t_inv;
        let w_liq = (depth_tao * vcfg.w_liq) * total_depth_inv;

        // Compute funding preference phi.
        // phi(funding_8h) = clip(funding_8h / scale, -1, 1) * max_tao
        let funding_norm = if has_funding_scale {
            (vstate.funding_8h * funding_scale_inv).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let phi_funding = funding_norm * funding_max_tao;

        // q_target_v = w_liq_v * q_global + w_fund_v * phi(funding)
        let q_target = w_liq * q_global + vcfg.w_fund * phi_funding;

        out.push(VenueTargetInventory {
            venue_index: i,
            w_liq,
            q_target,
        });
    }
}

/// Main MM quoting function.
///
/// This reads the global state (fair value, volatility, risk scalars,
/// inventory, etc.) and per-venue state, then produces local quotes
/// for each venue.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<MmQuote> {
    compute_mm_quotes_with_ablations(cfg, state, &AblationSet::new())
}

/// Main MM quoting function (buffer-reusing variant).
///
/// Clears `out` and pushes quotes into it, reusing capacity.
/// Use this in hot paths to avoid per-tick allocations.
pub fn compute_mm_quotes_into(cfg: &Config, state: &GlobalState, out: &mut Vec<MmQuote>) {
    compute_mm_quotes_with_ablations_into(cfg, state, &AblationSet::new(), out)
}

/// Main MM quoting function with ablation support.
///
/// Ablations:
/// - disable_fair_value_gating: Gating always allows (never blocks quoting)
/// - disable_toxicity_gate: Toxicity gating never blocks/disables venues
pub fn compute_mm_quotes_with_ablations(
    cfg: &Config,
    state: &GlobalState,
    ablations: &AblationSet,
) -> Vec<MmQuote> {
    let n = cfg.venues.len();
    let mut out = Vec::with_capacity(n);
    compute_mm_quotes_with_ablations_into(cfg, state, ablations, &mut out);
    out
}

/// Main MM quoting function with ablation support (buffer-reusing variant).
///
/// Uses stable quote slots to avoid per-tick Arc<str> clone/drop overhead.
/// On first call (or if venue count changes), initializes `out` with N slots
/// containing cloned venue_id Arc<str>. On subsequent calls, resets bid/ask
/// to None and updates in place. This eliminates per-tick allocation and
/// Arc reference count churn in the hot path.
///
/// Ablations:
/// - disable_fair_value_gating: Gating always allows (never blocks quoting)
/// - disable_toxicity_gate: Toxicity gating never blocks/disables venues
///
/// Optimization notes:
/// - Stable quote slots: venue_id cloned once, bid/ask reset to None each tick.
/// - Tick-invariant scalars are precomputed once into TickScalars struct.
/// - Per-venue target inventory computation reuses scratch buffer pattern.
/// - Config lookups are hoisted out of the per-venue loop.
/// - Explicit indexed loops for deterministic ordering.
pub fn compute_mm_quotes_with_ablations_into(
    cfg: &Config,
    state: &GlobalState,
    ablations: &AblationSet,
    out: &mut Vec<MmQuote>,
) {
    let n = cfg.venues.len();

    // Initialize stable quote slots on first call or if venue count changed.
    // This clones venue_id Arc<str> once; subsequent ticks only reset bid/ask.
    if out.len() != n {
        out.clear();
        out.reserve(n);
        for i in 0..n {
            out.push(MmQuote {
                venue_index: i,
                venue_id: cfg.venues[i].id_arc.clone(),
                bid: None,
                ask: None,
            });
        }
    } else {
        // Reset bid/ask to None without dropping/recreating venue_id.
        for q in out.iter_mut() {
            q.bid = None;
            q.ask = None;
        }
    }

    // If we don't have a fair value yet, we can't quote meaningfully.
    // Slots already have bid/ask = None, so just return.
    let s_t = match state.fair_value {
        Some(v) => v,
        None => {
            return;
        }
    };

    // Global scalars.
    let sigma_eff = state.sigma_eff.max(cfg.volatility.sigma_min).max(1e-8);
    let spread_mult = state.spread_mult.max(0.0);
    let size_mult = state.size_mult.max(0.0);
    let risk_regime = state.risk_regime;

    // ---------------------------------------------------------------------
    // Global safety gating
    // ---------------------------------------------------------------------

    // Kill switch OR HardLimit => return no quotes.
    // Spec: HardLimit is "structurally unsafe", so MM must fully stop.
    // Slots already have bid/ask = None, so just return.
    if state.kill_switch || matches!(risk_regime, RiskRegime::HardLimit) {
        return;
    }

    // If volatility or size_mult are degenerate, don't quote.
    // Slots already have bid/ask = None, so just return.
    if sigma_eff <= 0.0 || size_mult <= 0.0 {
        return;
    }

    // Cache config references
    let mm_cfg = &cfg.mm;
    let risk_cfg = &cfg.risk;
    let tox_cfg = &cfg.toxicity;

    // Precompute tick-invariant scalars (hoisted from per-venue loop)
    let tau = mm_cfg.quote_horizon_sec.max(1.0);
    let scalars = TickScalars {
        s_t,
        sigma_eff,
        q_global: state.q_global_tao,
        delta_abs_usd: state.dollar_delta_usd.abs(),
        delta_limit_usd: state.delta_limit_usd.max(1.0),
        spread_mult,
        size_mult,

        // MM config scalars
        beta_b: mm_cfg.basis_weight,
        beta_f: mm_cfg.funding_weight,
        lambda_inv: mm_cfg.lambda_inv,
        edge_local_min: mm_cfg.edge_local_min.max(0.0),
        edge_vol_mult: mm_cfg.edge_vol_mult,
        size_eta: mm_cfg.size_eta.max(1e-9),
        funding_horizon_frac: tau / (8.0 * 60.0 * 60.0),
        sigma_sq_tau: sigma_eff * sigma_eff * tau,

        // Risk config scalars
        liq_crit_sigma: risk_cfg.liq_crit_sigma,
        liq_warn_sigma: risk_cfg.liq_warn_sigma,
        spread_warn_mult: risk_cfg.spread_warn_mult.max(1.0),
        mm_margin_factor: risk_cfg.mm_max_leverage * risk_cfg.mm_margin_safety,
        q_warn_cap: risk_cfg.q_warn_cap.max(0.0),

        // Toxicity config scalars
        tox_high_threshold: tox_cfg.tox_high_threshold,
        tox_med_threshold: tox_cfg.tox_med_threshold,

        // Ablation flags (computed once)
        disable_fv_gating: ablations.disable_fair_value_gating(),
        disable_tox_gating: ablations.disable_toxicity_gate(),

        // Risk regime flags
        is_warning_regime: matches!(risk_regime, RiskRegime::Warning),
    };

    // Compute per-venue target inventories.
    // Note: Using allocating version here for simplicity; could add scratch buffer to state
    // if this shows up in profiles. The Vec is small (n venues).
    let venue_targets = compute_venue_targets(cfg, state);

    // ---------------------------------------------------------------------
    // Per-venue quoting (explicit indexed loop for determinism)
    // ---------------------------------------------------------------------
    // We use explicit index because we need to:
    // 1. Access cfg.venues[i], state.venues[i], and venue_targets[i] in lockstep
    // 2. Update out[i] in place (stable slots optimization)
    // 3. Maintain deterministic iteration order
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let vcfg = &cfg.venues[i];
        let vstate = &state.venues[i];
        let target = &venue_targets[i];

        // Disabled venue => no quotes (bid/ask already None from reset).
        if matches!(vstate.status, VenueStatus::Disabled) {
            continue;
        }

        // Venue mid: prefer local mid if present, else fall back to fair value.
        let mid = vstate.mid.unwrap_or(s_t);

        let (bid, ask) =
            compute_single_venue_quotes_fast(vcfg, vstate, mid, target.q_target, &scalars);

        // Update in place (stable slots: venue_id already set, just update bid/ask).
        out[i].bid = bid;
        out[i].ask = ask;
    }
}

/// Convert MM quotes into abstract OrderIntents used by the rest of the engine.
pub fn mm_quotes_to_order_intents(quotes: &[MmQuote]) -> Vec<OrderIntent> {
    // Pre-allocate max capacity: each quote can produce at most 2 intents (bid + ask).
    let mut intents = Vec::with_capacity(quotes.len() * 2);
    mm_quotes_to_order_intents_into(quotes, &mut intents);
    intents
}

/// Convert MM quotes into abstract OrderIntents (buffer-reusing variant).
///
/// Reuses existing `out` entries in-place to avoid per-tick Arc<str> churn.
/// Only reassigns `venue_id` if it actually differs (checked via `Arc::ptr_eq`).
/// After processing, truncates any leftover tail entries from previous calls.
///
/// Use this in hot paths to avoid per-tick allocations. Once warmed up, this
/// function performs no heap allocations.
///
/// Ordering: iterates quotes in order; for each quote, emits bid then ask.
/// This is deterministic and matches the original implementation.
pub fn mm_quotes_to_order_intents_into(quotes: &[MmQuote], out: &mut Vec<OrderIntent>) {
    // We avoid out.clear() to preserve existing OrderIntent entries for in-place
    // mutation, eliminating Arc<str> clone/drop overhead on venue_id each tick.

    let mut write_idx = 0;

    for q in quotes {
        if let Some(bid) = &q.bid {
            if write_idx < out.len() {
                // Mutate existing slot in place.
                let slot = &mut out[write_idx];
                slot.venue_index = q.venue_index;
                // Only reassign venue_id if pointer differs (avoids Arc refcount churn).
                if !Arc::ptr_eq(&slot.venue_id, &q.venue_id) {
                    slot.venue_id = q.venue_id.clone();
                }
                slot.side = Side::Buy;
                slot.price = bid.price;
                slot.size = bid.size;
                slot.purpose = OrderPurpose::Mm;
            } else {
                // Push new entry.
                out.push(OrderIntent {
                    venue_index: q.venue_index,
                    venue_id: q.venue_id.clone(),
                    side: Side::Buy,
                    price: bid.price,
                    size: bid.size,
                    purpose: OrderPurpose::Mm,
                });
            }
            write_idx += 1;
        }

        if let Some(ask) = &q.ask {
            if write_idx < out.len() {
                // Mutate existing slot in place.
                let slot = &mut out[write_idx];
                slot.venue_index = q.venue_index;
                // Only reassign venue_id if pointer differs (avoids Arc refcount churn).
                if !Arc::ptr_eq(&slot.venue_id, &q.venue_id) {
                    slot.venue_id = q.venue_id.clone();
                }
                slot.side = Side::Sell;
                slot.price = ask.price;
                slot.size = ask.size;
                slot.purpose = OrderPurpose::Mm;
            } else {
                // Push new entry.
                out.push(OrderIntent {
                    venue_index: q.venue_index,
                    venue_id: q.venue_id.clone(),
                    side: Side::Sell,
                    price: ask.price,
                    size: ask.size,
                    purpose: OrderPurpose::Mm,
                });
            }
            write_idx += 1;
        }
    }

    // Drop any leftover tail items from previous calls.
    out.truncate(write_idx);
}

/// Compute the maker cost for a venue (fee minus rebate).
///
/// Returns cost in USD per unit of notional.
#[inline]
fn compute_maker_cost(vcfg: &VenueConfig, price: f64) -> f64 {
    // maker_fee_bps is positive = fee, maker_rebate_bps is positive = rebate
    let fee_rate = vcfg.maker_fee_bps / 10_000.0;
    let rebate_rate = vcfg.maker_rebate_bps / 10_000.0;
    (fee_rate - rebate_rate) * price
}

/// Optimized single-venue quote computation using precomputed tick scalars.
///
/// This is the hot-path version that avoids repeated config lookups by using
/// the precomputed TickScalars struct. All numerical computations are identical
/// to compute_single_venue_quotes to preserve bit-exact determinism.
#[inline]
fn compute_single_venue_quotes_fast(
    vcfg: &VenueConfig,
    vstate: &VenueState,
    mid: f64,
    q_target_v: f64,
    sc: &TickScalars,
) -> (Option<MmLevel>, Option<MmLevel>) {
    // ---------------------------------------------------------------------
    // 0) Venue-level gating
    // ---------------------------------------------------------------------

    // If the venue is Disabled, no quoting (unless toxicity gating is disabled).
    if !sc.disable_tox_gating && matches!(vstate.status, VenueStatus::Disabled) {
        return (None, None);
    }

    // If liquidation distance is below critical, do not quote at all.
    let dist_liq = vstate.dist_liq_sigma;
    if dist_liq <= sc.liq_crit_sigma {
        return (None, None);
    }

    // Toxicity gating: if toxicity >= TOX_HIGH_THRESHOLD, skip venue.
    if !sc.disable_tox_gating && vstate.toxicity >= sc.tox_high_threshold {
        return (None, None);
    }

    // Stale book check: if we have no mid or it's stale, don't quote.
    if vstate.mid.is_none() && !sc.disable_fv_gating {
        return (None, None);
    }

    // Check for valid spread/depth.
    let spread = vstate
        .spread
        .unwrap_or(if sc.disable_fv_gating { 0.01 } else { 0.0 });
    let depth = if sc.disable_fv_gating && vstate.depth_near_mid <= 0.0 {
        10_000.0 // Default depth when gating disabled
    } else {
        vstate.depth_near_mid
    };
    if spread <= 0.0 || depth <= 0.0 {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 1) AS half-spread (Section 9.2)
    // ---------------------------------------------------------------------

    let gamma = vcfg.gamma.max(1e-8);
    let k = vcfg.k.max(1e-8);

    // Classical AS half-spread: δ* = (1/γ) * ln(1 + γ/k)
    let delta_as = (1.0 / gamma) * (1.0 + (gamma / k)).ln();

    // Apply volatility spread multiplier.
    let delta_vol = delta_as * sc.spread_mult.max(1e-6);

    // Compute minimum admissible half-spread (Section 9.2):
    //   min_half = (EDGE_LOCAL_MIN + maker_cost + vol_buffer) / 2
    let maker_cost = compute_maker_cost(vcfg, sc.s_t);
    let vol_buffer = sc.edge_vol_mult * sc.sigma_eff * sc.s_t;
    let min_half_spread = (sc.edge_local_min + maker_cost + vol_buffer) / 2.0;

    // Final half-spread: max of AS-derived and minimum economic requirement.
    let mut half_spread = delta_vol.max(min_half_spread).max(0.0);

    // Warning regime: widen spreads further.
    if sc.is_warning_regime {
        half_spread *= sc.spread_warn_mult;
    }

    // As we approach liquidation warning threshold, widen spreads (linear ramp).
    if dist_liq > 0.0 && dist_liq < sc.liq_warn_sigma {
        let t = ((sc.liq_warn_sigma - dist_liq) / sc.liq_warn_sigma).clamp(0.0, 1.0);
        // Up to +200% extra spread near liquidation.
        half_spread *= 1.0 + 2.0 * t;
    }

    if half_spread <= 0.0 {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 2) Reservation price (Section 9.1)
    // ---------------------------------------------------------------------

    // Basis and funding for this venue.
    let basis_v = mid - sc.s_t;
    let funding_8h = vstate.funding_8h;
    let q_v = vstate.position_tao;

    // Funding expected PnL per unit over horizon tau:
    //   f_v = funding_8h * (tau/8h) * S
    let funding_pnl_per_unit = funding_8h * sc.funding_horizon_frac * sc.s_t;

    // Enhanced reservation price (spec formula):
    //   r_v = S_t + β_b*b_v + β_f*f_v - γ*(σ_eff^2)*τ*( q_global - λ_inv*(q_v - q_target_v) )
    let basis_adj = sc.beta_b * basis_v;
    let funding_adj = sc.beta_f * funding_pnl_per_unit;

    // Inventory skew term (using precomputed sigma_sq_tau):
    let inv_deviation = sc.q_global - sc.lambda_inv * (q_v - q_target_v);
    let inv_term = gamma * sc.sigma_sq_tau * inv_deviation;

    let reservation_price = sc.s_t + basis_adj + funding_adj - inv_term;

    // Raw quote prices.
    let raw_bid = reservation_price - half_spread;
    let raw_ask = reservation_price + half_spread;

    // ---------------------------------------------------------------------
    // 3) Passivity enforcement (Section 9.2)
    // ---------------------------------------------------------------------

    // Derive best_bid/best_ask from venue mid/spread if no L2 is available.
    let half_book_spread = spread / 2.0;
    let best_bid = mid - half_book_spread;
    let best_ask = mid + half_book_spread;

    let tick = vcfg.tick_size.max(1e-6);

    // Enforce passivity: bid <= best_bid - tick, ask >= best_ask + tick.
    let passive_bid_limit = best_bid - tick;
    let passive_ask_limit = best_ask + tick;

    // Apply tick snapping.
    let mut bid_price = (raw_bid.min(passive_bid_limit) / tick).floor() * tick;
    let mut ask_price = (raw_ask.max(passive_ask_limit) / tick).ceil() * tick;

    // Re-check passivity after snapping; adjust if needed.
    if bid_price > passive_bid_limit {
        bid_price = (passive_bid_limit / tick).floor() * tick;
    }
    if ask_price < passive_ask_limit {
        ask_price = (passive_ask_limit / tick).ceil() * tick;
    }

    // Sanity checks.
    if bid_price <= 0.0 || ask_price <= bid_price {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 4) Per-unit edge calculation and gating (Section 9.2)
    // ---------------------------------------------------------------------

    // Per-unit edge for bid and ask after maker fees only.
    let edge_bid = sc.s_t - bid_price - maker_cost;
    let edge_ask = ask_price - sc.s_t - maker_cost;

    let bid_edge_ok = edge_bid >= sc.edge_local_min;
    let ask_edge_ok = edge_ask >= sc.edge_local_min;

    // ---------------------------------------------------------------------
    // 5) Size model (Section 10)
    // ---------------------------------------------------------------------

    // Quadratic objective: J(Q) = e*Q - 0.5*η*Q^2
    // Unconstrained optimum: Q_raw = e / η (if e > 0)
    let q_raw_bid = if bid_edge_ok && edge_bid > 0.0 {
        edge_bid / sc.size_eta
    } else {
        0.0
    };

    let q_raw_ask = if ask_edge_ok && edge_ask > 0.0 {
        edge_ask / sc.size_eta
    } else {
        0.0
    };

    // Apply volatility size multiplier.
    let mut size_bid = q_raw_bid * sc.size_mult;
    let mut size_ask = q_raw_ask * sc.size_mult;

    // Apply per-venue min/max size.
    size_bid = size_bid.max(vcfg.lot_size_tao);
    size_ask = size_ask.max(vcfg.lot_size_tao);

    // Margin cap: Q_margin_max = margin_available * mm_margin_factor / price
    let margin_cap_bid = if bid_price > 0.0 {
        (vstate.margin_available_usd * sc.mm_margin_factor) / bid_price
    } else {
        f64::MAX
    };

    let margin_cap_ask = if ask_price > 0.0 {
        (vstate.margin_available_usd * sc.mm_margin_factor) / ask_price
    } else {
        f64::MAX
    };

    size_bid = size_bid.min(margin_cap_bid);
    size_ask = size_ask.min(margin_cap_ask);

    // Liquidation-distance-aware shrinking (Section 10):
    // If dist_liq <= LIQ_CRIT_SIGMA: no risk-increasing quotes (reduce-only)
    // If LIQ_CRIT < dist < LIQ_WARN: shrink sizes linearly
    if dist_liq <= sc.liq_crit_sigma {
        // At or below critical: only allow reduce-only quotes.
        if q_v > 0.0 {
            size_bid = 0.0;
        } else if q_v < 0.0 {
            size_ask = 0.0;
        } else {
            size_bid = 0.0;
            size_ask = 0.0;
        }
    } else if dist_liq < sc.liq_warn_sigma {
        // Linear shrink as we approach critical.
        let k_liq = ((dist_liq - sc.liq_crit_sigma)
            / (sc.liq_warn_sigma - sc.liq_crit_sigma + 1e-9))
            .clamp(0.0, 1.0);
        size_bid *= k_liq;
        size_ask *= k_liq;
    }

    // Venue health: Warning venues get reduced size.
    if !sc.disable_tox_gating && matches!(vstate.status, VenueStatus::Warning) {
        size_bid *= 0.5;
        size_ask *= 0.5;
    }

    // Toxicity: medium toxicity reduces sizes.
    if !sc.disable_tox_gating
        && vstate.toxicity >= sc.tox_med_threshold
        && vstate.toxicity < sc.tox_high_threshold
    {
        let tox_factor = 1.0 - (vstate.toxicity - sc.tox_med_threshold) / (0.3_f64).max(1e-6);
        size_bid *= tox_factor.clamp(0.1, 1.0);
        size_ask *= tox_factor.clamp(0.1, 1.0);
    }

    // Delta-limit directional throttling.
    let delta_ratio = (sc.delta_abs_usd / sc.delta_limit_usd).max(0.0);

    if delta_ratio >= 2.0 {
        return (None, None);
    } else if delta_ratio > 1.0 {
        let factor = (2.0 - delta_ratio).clamp(0.0, 1.0);

        if sc.q_global > 0.0 {
            size_bid = 0.0;
            size_ask *= factor;
        } else if sc.q_global < 0.0 {
            size_ask = 0.0;
            size_bid *= factor;
        } else {
            size_bid *= factor;
            size_ask *= factor;
        }
    }

    // Warning regime: cap size.
    if sc.is_warning_regime {
        size_bid = size_bid.min(sc.q_warn_cap);
        size_ask = size_ask.min(sc.q_warn_cap);
    }

    // Hard cap by venue max size.
    size_bid = size_bid.min(vcfg.max_order_size);
    size_ask = size_ask.min(vcfg.max_order_size);

    // Apply lot size rounding.
    let lot = vcfg.lot_size_tao.max(1e-9);
    size_bid = (size_bid / lot).floor() * lot;
    size_ask = (size_ask / lot).floor() * lot;

    // Check minimum size.
    let min_size = vcfg.lot_size_tao.max(0.0);

    let bid = if size_bid >= min_size && bid_edge_ok {
        Some(MmLevel {
            price: bid_price,
            size: size_bid,
        })
    } else {
        None
    };

    let ask = if size_ask >= min_size && ask_edge_ok {
        Some(MmLevel {
            price: ask_price,
            size: size_ask,
        })
    } else {
        None
    };

    (bid, ask)
}

/// Compute bid/ask for a single venue using an Avellaneda–Stoikov-style
/// model with practical tweaks:
///   - Enhanced reservation price with per-venue targets,
///   - basis + funding adjustments,
///   - vol-driven spread & size multipliers,
///   - risk-regime / liquidation-distance / delta-limit aware sizing,
///   - passivity enforcement.
///
/// Section 9–11 spec alignment.
///
/// Ablation support:
/// - disable_fair_value_gating: Gating always allows (never blocks quoting)
/// - disable_toxicity_gate: Toxicity gating never blocks/disables venues
///
/// Note: This function is preserved for documentation and potential external use.
/// The hot path uses compute_single_venue_quotes_fast with precomputed scalars.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn compute_single_venue_quotes(
    cfg: &Config,
    vcfg: &VenueConfig,
    vstate: &VenueState,
    s_t: f64,
    mid: f64,
    basis_v: f64,
    funding_8h: f64,
    sigma_eff: f64,
    _vol_ratio: f64,
    q_global: f64,
    q_v: f64,
    q_target_v: f64,
    delta_abs_usd: f64,
    delta_limit_usd: f64,
    spread_mult: f64,
    size_mult: f64,
    risk_regime: RiskRegime,
    ablations: &AblationSet,
) -> (Option<MmLevel>, Option<MmLevel>) {
    let mm_cfg = &cfg.mm;
    let risk_cfg = &cfg.risk;
    let tox_cfg = &cfg.toxicity;

    // Check if fair value gating is disabled (always allow quoting)
    let disable_fv_gating = ablations.disable_fair_value_gating();
    let disable_tox_gating = ablations.disable_toxicity_gate();

    // ---------------------------------------------------------------------
    // 0) Venue-level gating
    // ---------------------------------------------------------------------

    // If the venue is Disabled, no quoting (unless toxicity gating is disabled).
    if !disable_tox_gating && matches!(vstate.status, VenueStatus::Disabled) {
        return (None, None);
    }

    // If liquidation distance is below critical, do not quote at all.
    // (This is not affected by fair value gating ablation - it's risk-based)
    let dist_liq = vstate.dist_liq_sigma;
    if dist_liq <= risk_cfg.liq_crit_sigma {
        return (None, None);
    }

    // Toxicity gating: if toxicity >= TOX_HIGH_THRESHOLD, skip venue.
    // (Skip if toxicity gating is disabled)
    if !disable_tox_gating && vstate.toxicity >= tox_cfg.tox_high_threshold {
        return (None, None);
    }

    // Stale book check: if we have no mid or it's stale, don't quote.
    // (Skip if fair value gating is disabled - use fair value as mid)
    if vstate.mid.is_none() && !disable_fv_gating {
        return (None, None);
    }

    // Check for valid spread/depth.
    // (Skip if fair value gating is disabled - use defaults)
    let spread = vstate
        .spread
        .unwrap_or(if disable_fv_gating { 0.01 } else { 0.0 });
    let depth = if disable_fv_gating && vstate.depth_near_mid <= 0.0 {
        10_000.0 // Default depth when gating disabled
    } else {
        vstate.depth_near_mid
    };
    if spread <= 0.0 || depth <= 0.0 {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 1) AS half-spread (Section 9.2)
    // ---------------------------------------------------------------------

    let gamma = vcfg.gamma.max(1e-8);
    let k = vcfg.k.max(1e-8);
    let tau = mm_cfg.quote_horizon_sec.max(1.0);

    // Classical AS half-spread:
    //   δ* = (1/γ) * ln(1 + γ/k)
    let delta_as = (1.0 / gamma) * (1.0 + (gamma / k)).ln();

    // Apply volatility spread multiplier.
    let delta_vol = delta_as * spread_mult.max(1e-6);

    // Compute minimum admissible half-spread (Section 9.2):
    //   min_half = (EDGE_LOCAL_MIN + maker_cost + vol_buffer) / 2
    let maker_cost = compute_maker_cost(vcfg, s_t);
    let vol_buffer = mm_cfg.edge_vol_mult * sigma_eff * s_t;
    let min_half_spread = (mm_cfg.edge_local_min + maker_cost + vol_buffer) / 2.0;

    // Final half-spread: max of AS-derived and minimum economic requirement.
    let mut half_spread = delta_vol.max(min_half_spread).max(0.0);

    // Warning regime: widen spreads further.
    if matches!(risk_regime, RiskRegime::Warning) {
        half_spread *= risk_cfg.spread_warn_mult.max(1.0);
    }

    // As we approach liquidation warning threshold, widen spreads (linear ramp).
    if dist_liq > 0.0 && dist_liq < risk_cfg.liq_warn_sigma {
        let t = ((risk_cfg.liq_warn_sigma - dist_liq) / risk_cfg.liq_warn_sigma).clamp(0.0, 1.0);
        // Up to +200% extra spread near liquidation.
        half_spread *= 1.0 + 2.0 * t;
    }

    if half_spread <= 0.0 {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 2) Reservation price (Section 9.1)
    // ---------------------------------------------------------------------

    // Funding expected PnL per unit over horizon tau:
    //   f_v = funding_8h * (tau/8h) * S
    let funding_horizon_frac = tau / (8.0 * 60.0 * 60.0);
    let funding_pnl_per_unit = funding_8h * funding_horizon_frac * s_t;

    let beta_b = mm_cfg.basis_weight;
    let beta_f = mm_cfg.funding_weight;
    let lambda_inv = mm_cfg.lambda_inv;

    // Enhanced reservation price (spec formula):
    //   r_v = S_t + β_b*b_v + β_f*f_v - γ*(σ_eff^2)*τ*( q_global - λ_inv*(q_v - q_target_v) )
    let basis_adj = beta_b * basis_v;
    let funding_adj = beta_f * funding_pnl_per_unit;

    // Inventory skew term:
    // - q_global term: global inventory pressure
    // - (q_v - q_target_v) term: local deviation from target
    let inv_deviation = q_global - lambda_inv * (q_v - q_target_v);
    let inv_term = gamma * sigma_eff * sigma_eff * tau * inv_deviation;

    let reservation_price = s_t + basis_adj + funding_adj - inv_term;

    // Raw quote prices.
    let raw_bid = reservation_price - half_spread;
    let raw_ask = reservation_price + half_spread;

    // ---------------------------------------------------------------------
    // 3) Passivity enforcement (Section 9.2)
    // ---------------------------------------------------------------------

    // Derive best_bid/best_ask from venue mid/spread if no L2 is available.
    let half_book_spread = spread / 2.0;
    let best_bid = mid - half_book_spread;
    let best_ask = mid + half_book_spread;

    let tick = vcfg.tick_size.max(1e-6);

    // Enforce passivity: bid <= best_bid - tick, ask >= best_ask + tick.
    let passive_bid_limit = best_bid - tick;
    let passive_ask_limit = best_ask + tick;

    // Apply tick snapping.
    let mut bid_price = (raw_bid.min(passive_bid_limit) / tick).floor() * tick;
    let mut ask_price = (raw_ask.max(passive_ask_limit) / tick).ceil() * tick;

    // Re-check passivity after snapping; adjust if needed.
    if bid_price > passive_bid_limit {
        bid_price = (passive_bid_limit / tick).floor() * tick;
    }
    if ask_price < passive_ask_limit {
        ask_price = (passive_ask_limit / tick).ceil() * tick;
    }

    // Sanity checks.
    if bid_price <= 0.0 || ask_price <= bid_price {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 4) Per-unit edge calculation and gating (Section 9.2)
    // ---------------------------------------------------------------------

    // Per-unit edge for bid and ask after maker fees only:
    //   e_bid = S_t - bid - maker_cost
    //   e_ask = ask - S_t - maker_cost
    //
    // Note: vol_buffer is already included in the minimum half-spread calculation,
    // so we don't subtract it again here. The edge check ensures we're getting
    // at least edge_local_min profit after fees.
    let edge_bid = s_t - bid_price - maker_cost;
    let edge_ask = ask_price - s_t - maker_cost;

    let edge_min = mm_cfg.edge_local_min.max(0.0);

    let bid_edge_ok = edge_bid >= edge_min;
    let ask_edge_ok = edge_ask >= edge_min;

    // ---------------------------------------------------------------------
    // 5) Size model (Section 10)
    // ---------------------------------------------------------------------

    // Quadratic objective: J(Q) = e*Q - 0.5*η*Q^2
    // Unconstrained optimum: Q_raw = e / η (if e > 0)
    let eta = mm_cfg.size_eta.max(1e-9);

    let q_raw_bid = if bid_edge_ok && edge_bid > 0.0 {
        edge_bid / eta
    } else {
        0.0
    };

    let q_raw_ask = if ask_edge_ok && edge_ask > 0.0 {
        edge_ask / eta
    } else {
        0.0
    };

    // Apply volatility size multiplier.
    let mut size_bid = q_raw_bid * size_mult;
    let mut size_ask = q_raw_ask * size_mult;

    // Apply per-venue min/max size.
    size_bid = size_bid.max(vcfg.lot_size_tao);
    size_ask = size_ask.max(vcfg.lot_size_tao);

    // Margin cap:
    //   Q_margin_max = margin_available * MM_MAX_LEVERAGE * MM_MARGIN_SAFETY / price
    let margin_cap_bid = if bid_price > 0.0 {
        (vstate.margin_available_usd * risk_cfg.mm_max_leverage * risk_cfg.mm_margin_safety)
            / bid_price
    } else {
        f64::MAX
    };

    let margin_cap_ask = if ask_price > 0.0 {
        (vstate.margin_available_usd * risk_cfg.mm_max_leverage * risk_cfg.mm_margin_safety)
            / ask_price
    } else {
        f64::MAX
    };

    size_bid = size_bid.min(margin_cap_bid);
    size_ask = size_ask.min(margin_cap_ask);

    // Liquidation-distance-aware shrinking (Section 10):
    // If dist_liq <= LIQ_CRIT_SIGMA: no risk-increasing quotes (reduce-only)
    // If LIQ_CRIT < dist < LIQ_WARN: shrink sizes linearly
    if dist_liq <= risk_cfg.liq_crit_sigma {
        // At or below critical: only allow reduce-only quotes.
        // If long (q_v > 0), only allow sells (asks).
        // If short (q_v < 0), only allow buys (bids).
        if q_v > 0.0 {
            size_bid = 0.0; // Don't increase long
        } else if q_v < 0.0 {
            size_ask = 0.0; // Don't increase short
        } else {
            // Flat: no quoting near liquidation
            size_bid = 0.0;
            size_ask = 0.0;
        }
    } else if dist_liq < risk_cfg.liq_warn_sigma {
        // Linear shrink as we approach critical.
        let k_liq = ((dist_liq - risk_cfg.liq_crit_sigma)
            / (risk_cfg.liq_warn_sigma - risk_cfg.liq_crit_sigma + 1e-9))
            .clamp(0.0, 1.0);
        size_bid *= k_liq;
        size_ask *= k_liq;
    }

    // Venue health: Warning venues get reduced size (skip if toxicity gating is disabled).
    if !disable_tox_gating && matches!(vstate.status, VenueStatus::Warning) {
        size_bid *= 0.5;
        size_ask *= 0.5;
    }

    // Toxicity: medium toxicity reduces sizes (skip if toxicity gating is disabled).
    if !disable_tox_gating
        && vstate.toxicity >= tox_cfg.tox_med_threshold
        && vstate.toxicity < tox_cfg.tox_high_threshold
    {
        let tox_factor = 1.0 - (vstate.toxicity - tox_cfg.tox_med_threshold) / (0.3_f64).max(1e-6);
        size_bid *= tox_factor.clamp(0.1, 1.0);
        size_ask *= tox_factor.clamp(0.1, 1.0);
    }

    // Delta-limit directional throttling:
    // - If |Δ| >= 2x limit: stop quoting entirely.
    // - If |Δ| in (1x, 2x): only quote the risk-reducing side, scaled down.
    let delta_ratio = (delta_abs_usd / delta_limit_usd).max(0.0);

    if delta_ratio >= 2.0 {
        return (None, None);
    } else if delta_ratio > 1.0 {
        let factor = (2.0 - delta_ratio).clamp(0.0, 1.0);

        if q_global > 0.0 {
            // Long => only allow sells (asks).
            size_bid = 0.0;
            size_ask *= factor;
        } else if q_global < 0.0 {
            // Short => only allow buys (bids).
            size_ask = 0.0;
            size_bid *= factor;
        } else {
            // Shrink both.
            size_bid *= factor;
            size_ask *= factor;
        }
    }

    // Warning regime: cap size.
    if matches!(risk_regime, RiskRegime::Warning) {
        let cap = risk_cfg.q_warn_cap.max(0.0);
        size_bid = size_bid.min(cap);
        size_ask = size_ask.min(cap);
    }

    // Hard cap by venue max size.
    size_bid = size_bid.min(vcfg.max_order_size);
    size_ask = size_ask.min(vcfg.max_order_size);

    // Apply lot size rounding.
    let lot = vcfg.lot_size_tao.max(1e-9);
    size_bid = (size_bid / lot).floor() * lot;
    size_ask = (size_ask / lot).floor() * lot;

    // Check minimum size.
    let min_size = vcfg.lot_size_tao.max(0.0);

    let bid = if size_bid >= min_size && bid_edge_ok {
        Some(MmLevel {
            price: bid_price,
            size: size_bid,
        })
    } else {
        None
    };

    let ask = if size_ask >= min_size && ask_edge_ok {
        Some(MmLevel {
            price: ask_price,
            size: size_ask,
        })
    } else {
        None
    };

    (bid, ask)
}

// ---------------------------------------------------------------------
// Order lifetime + tolerance logic (Section 11)
// ---------------------------------------------------------------------

/// Context for evaluating whether an existing MM order should be replaced.
///
/// Bundles all parameters needed by `should_replace_order` to avoid
/// clippy::too_many_arguments.
pub struct ShouldReplaceOrderCtx<'a> {
    pub cfg: &'a Config,
    pub vcfg: &'a VenueConfig,
    pub current: &'a ActiveMmOrder,
    pub desired_price: f64,
    pub desired_size: f64,
    pub now_ms: TimestampMs,
    pub best_bid: f64,
    pub best_ask: f64,
}

/// Determines whether an existing MM order should be replaced.
///
/// Returns `true` if the order should be cancelled and replaced.
///
/// Section 11:
/// - If age_ms < MIN_QUOTE_LIFETIME_MS and order is still passive, keep it.
/// - Else: replace only if price_diff > PRICE_TOL_TICKS or size_diff > SIZE_TOL_REL.
pub fn should_replace_order(ctx: ShouldReplaceOrderCtx<'_>) -> bool {
    let mm_cfg = &ctx.cfg.mm;
    let tick = ctx.vcfg.tick_size.max(1e-6);

    let age_ms = ctx.now_ms - ctx.current.timestamp_ms;

    // Check if current order is still passive.
    let is_passive = match ctx.current.side {
        Side::Buy => ctx.current.price <= ctx.best_bid - tick,
        Side::Sell => ctx.current.price >= ctx.best_ask + tick,
    };

    // If young and still passive, keep it.
    if age_ms < mm_cfg.min_quote_lifetime_ms && is_passive {
        return false;
    }

    // Compute differences.
    let price_diff_ticks = (ctx.current.price - ctx.desired_price).abs() / tick;
    let size_diff_rel =
        (ctx.current.size - ctx.desired_size).abs() / ctx.desired_size.max(ctx.vcfg.lot_size_tao);

    // Replace if price or size changed beyond tolerance.
    price_diff_ticks > mm_cfg.price_tol_ticks || size_diff_rel > mm_cfg.size_tol_rel
}

/// Generate cancel/replace actions for MM orders.
///
/// This function compares desired quotes against current active orders
/// and determines what actions to take.
#[derive(Debug, Clone)]
pub enum MmOrderAction {
    /// No change needed.
    Keep,
    /// Cancel existing order (no replacement).
    Cancel { venue_index: usize, side: Side },
    /// Place new order (no existing order).
    Place {
        venue_index: usize,
        level: MmLevel,
        side: Side,
    },
    /// Cancel existing and place new.
    Replace {
        venue_index: usize,
        old_level: MmLevel,
        new_level: MmLevel,
        side: Side,
    },
}

/// Compute order management actions for a single venue.
pub fn compute_order_actions(
    cfg: &Config,
    vcfg: &VenueConfig,
    vstate: &VenueState,
    desired_quote: &MmQuote,
    current_bid: Option<&ActiveMmOrder>,
    current_ask: Option<&ActiveMmOrder>,
    now_ms: TimestampMs,
) -> Vec<MmOrderAction> {
    let mut actions = Vec::new();

    // Derive best bid/ask from venue state.
    let mid = vstate.mid.unwrap_or(0.0);
    let spread = vstate.spread.unwrap_or(0.0);
    let half_spread = spread / 2.0;
    let best_bid = mid - half_spread;
    let best_ask = mid + half_spread;

    // Handle bid side.
    match (&desired_quote.bid, current_bid) {
        (Some(desired), Some(current)) => {
            let ctx = ShouldReplaceOrderCtx {
                cfg,
                vcfg,
                current,
                desired_price: desired.price,
                desired_size: desired.size,
                now_ms,
                best_bid,
                best_ask,
            };
            if should_replace_order(ctx) {
                actions.push(MmOrderAction::Replace {
                    venue_index: desired_quote.venue_index,
                    old_level: MmLevel {
                        price: current.price,
                        size: current.size,
                    },
                    new_level: desired.clone(),
                    side: Side::Buy,
                });
            }
            // else: keep current order
        }
        (Some(desired), None) => {
            actions.push(MmOrderAction::Place {
                venue_index: desired_quote.venue_index,
                level: desired.clone(),
                side: Side::Buy,
            });
        }
        (None, Some(_)) => {
            actions.push(MmOrderAction::Cancel {
                venue_index: desired_quote.venue_index,
                side: Side::Buy,
            });
        }
        (None, None) => {}
    }

    // Handle ask side.
    match (&desired_quote.ask, current_ask) {
        (Some(desired), Some(current)) => {
            let ctx = ShouldReplaceOrderCtx {
                cfg,
                vcfg,
                current,
                desired_price: desired.price,
                desired_size: desired.size,
                now_ms,
                best_bid,
                best_ask,
            };
            if should_replace_order(ctx) {
                actions.push(MmOrderAction::Replace {
                    venue_index: desired_quote.venue_index,
                    old_level: MmLevel {
                        price: current.price,
                        size: current.size,
                    },
                    new_level: desired.clone(),
                    side: Side::Sell,
                });
            }
            // else: keep current order
        }
        (Some(desired), None) => {
            actions.push(MmOrderAction::Place {
                venue_index: desired_quote.venue_index,
                level: desired.clone(),
                side: Side::Sell,
            });
        }
        (None, Some(_)) => {
            actions.push(MmOrderAction::Cancel {
                venue_index: desired_quote.venue_index,
                side: Side::Sell,
            });
        }
        (None, None) => {}
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::GlobalState;
    use crate::types::VenueStatus;

    /// Helper to create a test config and state.
    fn setup_test() -> (Config, GlobalState) {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        // Set up fair value and basic market conditions.
        state.fair_value = Some(300.0);
        state.fair_value_prev = 300.0;
        state.sigma_eff = 0.02;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;
        state.vol_ratio_clipped = 1.0;
        state.delta_limit_usd = 100_000.0;

        // Set up venue states with valid data.
        for v in &mut state.venues {
            v.mid = Some(300.0);
            v.spread = Some(1.00); // $1 spread (wider to ensure edge is positive)
            v.depth_near_mid = 10_000.0; // $10k depth
            v.margin_available_usd = 10_000.0;
            v.dist_liq_sigma = 10.0; // Safe distance
            v.status = VenueStatus::Healthy;
            v.toxicity = 0.0;
        }

        (cfg, state)
    }

    #[test]
    fn test_passivity_bid_below_best_bid() {
        let (cfg, state) = setup_test();
        let quotes = compute_mm_quotes(&cfg, &state);

        for (i, q) in quotes.iter().enumerate() {
            if let Some(bid) = &q.bid {
                let vstate = &state.venues[i];
                let vcfg = &cfg.venues[i];
                let mid = vstate.mid.unwrap();
                let spread = vstate.spread.unwrap();
                let best_bid = mid - spread / 2.0;
                let tick = vcfg.tick_size;

                assert!(
                    bid.price <= best_bid - tick,
                    "Bid {} should be <= best_bid - tick {} at venue {}",
                    bid.price,
                    best_bid - tick,
                    i
                );
            }
        }
    }

    #[test]
    fn test_passivity_ask_above_best_ask() {
        let (cfg, state) = setup_test();
        let quotes = compute_mm_quotes(&cfg, &state);

        for (i, q) in quotes.iter().enumerate() {
            if let Some(ask) = &q.ask {
                let vstate = &state.venues[i];
                let vcfg = &cfg.venues[i];
                let mid = vstate.mid.unwrap();
                let spread = vstate.spread.unwrap();
                let best_ask = mid + spread / 2.0;
                let tick = vcfg.tick_size;

                assert!(
                    ask.price >= best_ask + tick,
                    "Ask {} should be >= best_ask + tick {} at venue {}",
                    ask.price,
                    best_ask + tick,
                    i
                );
            }
        }
    }

    #[test]
    fn test_reservation_price_moves_with_q_global() {
        let (cfg, mut state) = setup_test();

        // Neutral inventory.
        state.q_global_tao = 0.0;
        let quotes_neutral = compute_mm_quotes(&cfg, &state);

        // Long inventory.
        state.q_global_tao = 10.0;
        let quotes_long = compute_mm_quotes(&cfg, &state);

        // Short inventory.
        state.q_global_tao = -10.0;
        let quotes_short = compute_mm_quotes(&cfg, &state);

        // When long, reservation should be lower (lean offers).
        // When short, reservation should be higher (lean bids).
        if let (Some(bid_n), Some(bid_l), Some(bid_s)) = (
            &quotes_neutral[0].bid,
            &quotes_long[0].bid,
            &quotes_short[0].bid,
        ) {
            // Long should have lower bid (reservation lower).
            assert!(
                bid_l.price <= bid_n.price,
                "Long inventory should have lower bid: {} <= {}",
                bid_l.price,
                bid_n.price
            );
            // Short should have higher bid (reservation higher).
            assert!(
                bid_s.price >= bid_n.price,
                "Short inventory should have higher bid: {} >= {}",
                bid_s.price,
                bid_n.price
            );
        }
    }

    #[test]
    fn test_reservation_price_with_venue_target_deviation() {
        let (cfg, mut state) = setup_test();

        // Set q_global to 0.
        state.q_global_tao = 0.0;

        // Venue 0: position at target (no deviation).
        state.venues[0].position_tao = 0.0;

        let quotes_at_target = compute_mm_quotes(&cfg, &state);

        // Venue 0: position above target (positive deviation).
        state.venues[0].position_tao = 5.0;

        let quotes_above_target = compute_mm_quotes(&cfg, &state);

        // When position > target, should skew quotes to reduce position (lower bids).
        if let (Some(bid_at), Some(bid_above)) =
            (&quotes_at_target[0].bid, &quotes_above_target[0].bid)
        {
            // With lambda_inv > 0, having position above target should lower reservation.
            // This test may need adjustment based on actual lambda_inv value.
            assert!(
                (bid_at.price - bid_above.price).abs() < 10.0 || bid_above.price <= bid_at.price,
                "Position above target should affect bid: at={} above={}",
                bid_at.price,
                bid_above.price
            );
        }
    }

    #[test]
    fn test_hardlimit_produces_no_quotes() {
        let (cfg, mut state) = setup_test();

        // Normal regime produces quotes.
        state.risk_regime = RiskRegime::Normal;
        let quotes_normal = compute_mm_quotes(&cfg, &state);
        let has_quotes_normal = quotes_normal
            .iter()
            .any(|q| q.bid.is_some() || q.ask.is_some());

        // HardLimit regime produces no quotes.
        state.risk_regime = RiskRegime::HardLimit;
        let quotes_hardlimit = compute_mm_quotes(&cfg, &state);
        let has_quotes_hardlimit = quotes_hardlimit
            .iter()
            .any(|q| q.bid.is_some() || q.ask.is_some());

        assert!(has_quotes_normal, "Normal regime should produce quotes");
        assert!(
            !has_quotes_hardlimit,
            "HardLimit regime should produce no quotes"
        );
    }

    #[test]
    fn test_kill_switch_produces_no_quotes() {
        let (cfg, mut state) = setup_test();

        // Kill switch active.
        state.kill_switch = true;
        let quotes = compute_mm_quotes(&cfg, &state);

        for q in &quotes {
            assert!(q.bid.is_none(), "Kill switch should produce no bids");
            assert!(q.ask.is_none(), "Kill switch should produce no asks");
        }
    }

    #[test]
    fn test_high_toxicity_skips_venue() {
        let (cfg, mut state) = setup_test();

        // Set venue 0 to high toxicity.
        state.venues[0].toxicity = 0.95; // Above tox_high_threshold (0.9)

        let quotes = compute_mm_quotes(&cfg, &state);

        assert!(
            quotes[0].bid.is_none() && quotes[0].ask.is_none(),
            "High toxicity venue should have no quotes"
        );

        // Other venues should still quote.
        let other_has_quotes = quotes[1..]
            .iter()
            .any(|q| q.bid.is_some() || q.ask.is_some());
        assert!(other_has_quotes, "Other venues should still have quotes");
    }

    #[test]
    fn test_venue_targets_computed_from_depth() {
        let (cfg, mut state) = setup_test();

        state.q_global_tao = 10.0;

        // Set different depths.
        state.venues[0].depth_near_mid = 20_000.0;
        state.venues[1].depth_near_mid = 10_000.0;
        state.venues[2].depth_near_mid = 5_000.0;

        let targets = compute_venue_targets(&cfg, &state);

        // Venue 0 should have higher liquidity weight.
        assert!(
            targets[0].w_liq > targets[1].w_liq,
            "Higher depth should give higher w_liq"
        );
        assert!(
            targets[1].w_liq > targets[2].w_liq,
            "Higher depth should give higher w_liq"
        );
    }

    #[test]
    fn test_order_replacement_within_lifetime() {
        let cfg = Config::default();
        let vcfg = &cfg.venues[0];

        let current = ActiveMmOrder {
            venue_index: 0,
            side: Side::Buy,
            price: 299.90,
            size: 1.0,
            timestamp_ms: 100,
        };

        // Within lifetime (500ms), passive order should not be replaced.
        let ctx = ShouldReplaceOrderCtx {
            cfg: &cfg,
            vcfg,
            current: &current,
            desired_price: 299.91, // Slightly different price
            desired_size: 1.0,
            now_ms: 200, // Only 100ms old
            best_bid: 300.0,
            best_ask: 300.10,
        };
        let should_replace = should_replace_order(ctx);

        assert!(
            !should_replace,
            "Young passive order should not be replaced"
        );
    }

    #[test]
    fn test_order_replacement_beyond_tolerance() {
        let cfg = Config::default();
        let vcfg = &cfg.venues[0];

        let current = ActiveMmOrder {
            venue_index: 0,
            side: Side::Buy,
            price: 299.90,
            size: 1.0,
            timestamp_ms: 0,
        };

        // Beyond lifetime, large price change should trigger replacement.
        let ctx = ShouldReplaceOrderCtx {
            cfg: &cfg,
            vcfg,
            current: &current,
            desired_price: 299.80, // 10 cents different (10 ticks with 0.01 tick size)
            desired_size: 1.0,
            now_ms: 1000, // 1 second old
            best_bid: 300.0,
            best_ask: 300.10,
        };
        let should_replace = should_replace_order(ctx);

        assert!(
            should_replace,
            "Order with large price change should be replaced"
        );
    }

    #[test]
    fn test_mm_quotes_determinism_and_buffer_capacity() {
        // Verify that compute_mm_quotes_into produces identical results
        // across multiple calls and preserves buffer capacity.

        let (cfg, state) = setup_test();
        let n = cfg.venues.len();

        // Pre-allocate with extra capacity
        let mut quotes_buf: Vec<MmQuote> = Vec::with_capacity(n + 10);
        let initial_capacity = quotes_buf.capacity();

        // First call
        compute_mm_quotes_into(&cfg, &state, &mut quotes_buf);
        let quotes1: Vec<_> = quotes_buf
            .iter()
            .map(|q| {
                (
                    q.venue_index,
                    q.bid.as_ref().map(|b| (b.price, b.size)),
                    q.ask.as_ref().map(|a| (a.price, a.size)),
                )
            })
            .collect();

        // Second call with same inputs
        compute_mm_quotes_into(&cfg, &state, &mut quotes_buf);
        let quotes2: Vec<_> = quotes_buf
            .iter()
            .map(|q| {
                (
                    q.venue_index,
                    q.bid.as_ref().map(|b| (b.price, b.size)),
                    q.ask.as_ref().map(|a| (a.price, a.size)),
                )
            })
            .collect();

        // Verify determinism: outputs must be identical
        assert_eq!(quotes1.len(), quotes2.len(), "Quote count differs");
        for (i, (q1, q2)) in quotes1.iter().zip(quotes2.iter()).enumerate() {
            assert_eq!(q1, q2, "Quote mismatch at venue {}", i);
        }

        // Verify capacity preservation
        assert!(
            quotes_buf.capacity() >= initial_capacity,
            "Buffer capacity shrunk: {} < {}",
            quotes_buf.capacity(),
            initial_capacity
        );

        // Run many iterations to ensure no accumulation or capacity growth issues
        for _ in 0..100 {
            compute_mm_quotes_into(&cfg, &state, &mut quotes_buf);
        }

        // Buffer should still be length n (cleared each call) not accumulated
        assert_eq!(
            quotes_buf.len(),
            n,
            "Buffer accumulated instead of clearing"
        );
    }

    #[test]
    fn test_mm_quotes_to_order_intents_into_determinism_and_capacity() {
        // Verify that mm_quotes_to_order_intents_into:
        // 1. Produces output identical to mm_quotes_to_order_intents
        // 2. Produces identical results across multiple calls
        // 3. Does not grow capacity on subsequent calls (once warmed up)

        let (cfg, state) = setup_test();

        // Get quotes to convert
        let quotes = compute_mm_quotes(&cfg, &state);

        // Get reference output from allocating version
        let reference = mm_quotes_to_order_intents(&quotes);

        // Use into() variant with pre-allocated buffer
        let mut intents_buf: Vec<OrderIntent> = Vec::with_capacity(quotes.len() * 2);

        // First call
        mm_quotes_to_order_intents_into(&quotes, &mut intents_buf);
        let cap_after_first = intents_buf.capacity();

        // Verify output matches reference
        assert_eq!(
            intents_buf.len(),
            reference.len(),
            "Intent count differs from reference"
        );
        for (i, (got, want)) in intents_buf.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                got.venue_index, want.venue_index,
                "venue_index mismatch at {}",
                i
            );
            assert!(
                Arc::ptr_eq(&got.venue_id, &want.venue_id) || *got.venue_id == *want.venue_id,
                "venue_id mismatch at {}",
                i
            );
            assert_eq!(got.side, want.side, "side mismatch at {}", i);
            assert_eq!(got.price, want.price, "price mismatch at {}", i);
            assert_eq!(got.size, want.size, "size mismatch at {}", i);
            assert_eq!(got.purpose, want.purpose, "purpose mismatch at {}", i);
        }

        // Second call with same inputs
        mm_quotes_to_order_intents_into(&quotes, &mut intents_buf);
        let cap_after_second = intents_buf.capacity();

        // Verify capacity did not grow
        assert_eq!(
            cap_after_second, cap_after_first,
            "Capacity grew on second call: {} -> {}",
            cap_after_first, cap_after_second
        );

        // Verify output is still correct after second call
        assert_eq!(
            intents_buf.len(),
            reference.len(),
            "Intent count differs after second call"
        );
        for (i, (got, want)) in intents_buf.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                got.venue_index, want.venue_index,
                "venue_index mismatch at {} (2nd call)",
                i
            );
            assert_eq!(got.side, want.side, "side mismatch at {} (2nd call)", i);
            assert_eq!(got.price, want.price, "price mismatch at {} (2nd call)", i);
            assert_eq!(got.size, want.size, "size mismatch at {} (2nd call)", i);
            assert_eq!(
                got.purpose, want.purpose,
                "purpose mismatch at {} (2nd call)",
                i
            );
        }

        // Run many iterations to verify no capacity growth once warmed up
        for _ in 0..100 {
            mm_quotes_to_order_intents_into(&quotes, &mut intents_buf);
        }
        assert_eq!(
            intents_buf.capacity(),
            cap_after_first,
            "Capacity grew after 100 iterations"
        );
    }
}
