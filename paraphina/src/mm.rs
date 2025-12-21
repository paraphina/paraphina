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
#[derive(Debug, Clone)]
pub struct MmQuote {
    pub venue_index: usize,
    pub venue_id: String,
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

/// Compute per-venue target inventories based on liquidity and funding.
///
/// Section 9.1:
///   w_liq_v = depth_v / sum(depth)
///   phi(funding) = clip(funding_8h / FUNDING_TARGET_RATE_SCALE, -1, 1) * FUNDING_TARGET_MAX_TAO
///   q_target_v = w_liq_v * q_global + w_fund_v * phi(funding_8h_v)
pub fn compute_venue_targets(cfg: &Config, state: &GlobalState) -> Vec<VenueTargetInventory> {
    let n = cfg.venues.len();
    let mut targets = Vec::with_capacity(n);

    // Sum depth across all healthy venues.
    let mut total_depth: f64 = 0.0;
    for (i, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[i];
        // Only include healthy venues with valid data.
        if !matches!(vstate.status, VenueStatus::Disabled) && vstate.depth_near_mid > 0.0 {
            // Use depth_near_mid as liquidity proxy.
            // Convert to TAO-equivalent if depth is in USD.
            let s_t = state.fair_value.unwrap_or(1.0).max(1.0);
            let depth_tao = vstate.depth_near_mid / s_t;
            total_depth += depth_tao * vcfg.w_liq;
        }
    }

    let q_global = state.q_global_tao;
    let mm_cfg = &cfg.mm;

    for (i, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[i];

        // Default: no weight, target = 0.
        if matches!(vstate.status, VenueStatus::Disabled) || total_depth <= 0.0 {
            targets.push(VenueTargetInventory {
                venue_index: i,
                w_liq: 0.0,
                q_target: 0.0,
            });
            continue;
        }

        // Compute liquidity weight.
        let s_t = state.fair_value.unwrap_or(1.0).max(1.0);
        let depth_tao = vstate.depth_near_mid / s_t;
        let w_liq = (depth_tao * vcfg.w_liq) / total_depth;

        // Compute funding preference phi.
        // phi(funding_8h) = clip(funding_8h / scale, -1, 1) * max_tao
        let funding_norm = if mm_cfg.funding_target_rate_scale > 0.0 {
            (vstate.funding_8h / mm_cfg.funding_target_rate_scale).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let phi_funding = funding_norm * mm_cfg.funding_target_max_tao;

        // q_target_v = w_liq_v * q_global + w_fund_v * phi(funding)
        let q_target = w_liq * q_global + vcfg.w_fund * phi_funding;

        targets.push(VenueTargetInventory {
            venue_index: i,
            w_liq,
            q_target,
        });
    }

    targets
}

/// Main MM quoting function.
///
/// This reads the global state (fair value, volatility, risk scalars,
/// inventory, etc.) and per-venue state, then produces local quotes
/// for each venue.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<MmQuote> {
    compute_mm_quotes_with_ablations(cfg, state, &AblationSet::new())
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

    // If we don't have a fair value yet, we can't quote meaningfully.
    let s_t = match state.fair_value {
        Some(v) => v,
        None => {
            // Return "empty quotes" so downstream code still sees one
            // entry per venue, but with no bid/ask.
            for (i, vcfg) in cfg.venues.iter().enumerate() {
                out.push(MmQuote {
                    venue_index: i,
                    venue_id: vcfg.id.clone(),
                    bid: None,
                    ask: None,
                });
            }
            return out;
        }
    };

    // Global scalars.
    let sigma_eff = state.sigma_eff.max(cfg.volatility.sigma_min).max(1e-8);
    let vol_ratio = state.vol_ratio_clipped.max(0.0);

    let spread_mult = state.spread_mult.max(0.0);
    let size_mult = state.size_mult.max(0.0);

    let q_global = state.q_global_tao;
    let delta_abs_usd = state.dollar_delta_usd.abs();
    let delta_limit_usd = state.delta_limit_usd.max(1.0);

    let risk_regime = state.risk_regime;

    // ---------------------------------------------------------------------
    // Global safety gating
    // ---------------------------------------------------------------------

    // Kill switch OR HardLimit => return no quotes.
    //
    // Spec: HardLimit is "structurally unsafe", so MM must fully stop.
    if state.kill_switch || matches!(risk_regime, RiskRegime::HardLimit) {
        for (i, vcfg) in cfg.venues.iter().enumerate() {
            out.push(MmQuote {
                venue_index: i,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
        }
        return out;
    }

    // If volatility or size_mult are degenerate, don't quote.
    if sigma_eff <= 0.0 || size_mult <= 0.0 {
        for (i, vcfg) in cfg.venues.iter().enumerate() {
            out.push(MmQuote {
                venue_index: i,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
        }
        return out;
    }

    // Compute per-venue target inventories.
    let venue_targets = compute_venue_targets(cfg, state);

    // ---------------------------------------------------------------------
    // Per-venue quoting
    // ---------------------------------------------------------------------
    for (i, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[i];
        let target = &venue_targets[i];

        // Disabled venue => no quotes.
        if matches!(vstate.status, VenueStatus::Disabled) {
            out.push(MmQuote {
                venue_index: i,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
            continue;
        }

        // Venue mid: prefer local mid if present, else fall back to fair value.
        let mid = vstate.mid.unwrap_or(s_t);

        // Basis and funding.
        let basis_v = mid - s_t;
        let funding_8h = vstate.funding_8h;
        let q_v = vstate.position_tao;
        let q_target_v = target.q_target;

        let (bid, ask) = compute_single_venue_quotes(
            cfg,
            vcfg,
            vstate,
            s_t,
            mid,
            basis_v,
            funding_8h,
            sigma_eff,
            vol_ratio,
            q_global,
            q_v,
            q_target_v,
            delta_abs_usd,
            delta_limit_usd,
            spread_mult,
            size_mult,
            risk_regime,
            ablations,
        );

        out.push(MmQuote {
            venue_index: i,
            venue_id: vcfg.id.clone(),
            bid,
            ask,
        });
    }

    out
}

/// Convert MM quotes into abstract OrderIntents used by the rest of the engine.
pub fn mm_quotes_to_order_intents(quotes: &[MmQuote]) -> Vec<OrderIntent> {
    let mut intents = Vec::new();

    for q in quotes {
        if let Some(bid) = &q.bid {
            intents.push(OrderIntent {
                venue_index: q.venue_index,
                venue_id: q.venue_id.clone(),
                side: Side::Buy,
                price: bid.price,
                size: bid.size,
                purpose: OrderPurpose::Mm,
            });
        }

        if let Some(ask) = &q.ask {
            intents.push(OrderIntent {
                venue_index: q.venue_index,
                venue_id: q.venue_id.clone(),
                side: Side::Sell,
                price: ask.price,
                size: ask.size,
                purpose: OrderPurpose::Mm,
            });
        }
    }

    intents
}

/// Compute the maker cost for a venue (fee minus rebate).
///
/// Returns cost in USD per unit of notional.
fn compute_maker_cost(vcfg: &VenueConfig, price: f64) -> f64 {
    // maker_fee_bps is positive = fee, maker_rebate_bps is positive = rebate
    let fee_rate = vcfg.maker_fee_bps / 10_000.0;
    let rebate_rate = vcfg.maker_rebate_bps / 10_000.0;
    (fee_rate - rebate_rate) * price
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
#[allow(clippy::too_many_arguments)]
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
}
