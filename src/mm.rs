// src/mm.rs
//
// Market-making quote engine (Avellaneda–Stoikov style) for Paraphina.
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
// Notes on "spec alignment":
//   - HardLimit => return NO MM quotes (safety-first).
//   - Reservation price:
//       r_v = S_t + BASIS_WEIGHT*basis_v + FUNDING_WEIGHT*E[funding] - gamma*sigma^2*tau*q_global
//     (basis_v = mid_v - S_t)
//   - Spread:
//       delta_AS = (1/gamma)*ln(1 + gamma/k) + 0.5*gamma*sigma^2*tau
//     then widened by vol/risk/liquidation proximity.
//   - Size:
//       base_size scaled by size_mult, then directional inventory tilt,
//       delta-limit directional throttling, and liq-distance shrinking.
//   - Side gating:
//       If a side’s implied edge vs S_t is below EDGE_LOCAL_MIN => don’t quote that side.

use crate::config::{Config, VenueConfig};
use crate::state::{GlobalState, RiskRegime, VenueState};
use crate::types::{OrderIntent, OrderPurpose, Side, VenueStatus};

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

/// Main MM quoting function.
///
/// This reads the global state (fair value, volatility, risk scalars,
/// inventory, etc.) and per-venue state, then produces local quotes
/// for each venue.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<MmQuote> {
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
    let sigma_eff = state.sigma_eff.max(cfg.volatility.sigma_min).max(1e-8); // avoid degenerate 0-vol modes
    let vol_ratio = state.vol_ratio_clipped.max(0.0);

    let spread_mult = state.spread_mult.max(0.0);
    let size_mult = state.size_mult.max(0.0);
    let band_mult = state.band_mult.max(0.0); // currently used conceptually

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

    // ---------------------------------------------------------------------
    // Per-venue quoting
    // ---------------------------------------------------------------------
    for (i, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[i];

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
            delta_abs_usd,
            delta_limit_usd,
            spread_mult,
            size_mult,
            band_mult,
            risk_regime,
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

/// Compute bid/ask for a single venue using an Avellaneda–Stoikov-style
/// model with practical tweaks:
///   - inventory-aware reservation price,
///   - basis + funding adjustments,
///   - vol-driven spread & size multipliers,
///   - risk-regime / liquidation-distance / delta-limit aware sizing.
///
/// IMPORTANT:
/// We avoid assuming additional fields (best bid/ask, depth) to keep this
/// drop-in compatible. Passivity enforcement belongs in the order manager
/// (or can be added once best bid/ask exist in VenueState).
#[allow(clippy::too_many_arguments)]
fn compute_single_venue_quotes(
    cfg: &Config,
    vcfg: &VenueConfig,
    vstate: &VenueState,
    s_t: f64,
    _mid: f64,
    basis_v: f64,
    funding_8h: f64,
    sigma_eff: f64,
    _vol_ratio: f64,
    q_global: f64,
    delta_abs_usd: f64,
    delta_limit_usd: f64,
    spread_mult: f64,
    size_mult: f64,
    _band_mult: f64,
    risk_regime: RiskRegime,
) -> (Option<MmLevel>, Option<MmLevel>) {
    let mm_cfg = &cfg.mm;
    let risk_cfg = &cfg.risk;

    // ---------------------------------------------------------------------
    // 0) Venue-level gating
    // ---------------------------------------------------------------------

    // If the venue is Disabled, no quoting. (Should be handled above, but keep safe.)
    if matches!(vstate.status, VenueStatus::Disabled) {
        return (None, None);
    }

    // If liquidation distance is below critical, do not quote at all.
    let dist_liq = vstate.dist_liq_sigma;
    if dist_liq <= risk_cfg.liq_crit_sigma {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 1) Spread: enhanced AS half-spread with vol/risk/liquidation widening
    // ---------------------------------------------------------------------

    let gamma = vcfg.gamma.max(1e-8);
    let k = vcfg.k.max(1e-8);

    let tau = mm_cfg.quote_horizon_sec.max(1.0); // quoting horizon in seconds

    // AS base distance component (no-vol term):
    //   (1/gamma) * ln(1 + gamma/k)
    let delta_no_vol = (1.0 / gamma) * (1.0 + (gamma / k)).ln();

    // AS diffusion component:
    //   0.5 * gamma * sigma^2 * tau
    let delta_diff = 0.5 * gamma * sigma_eff * sigma_eff * tau;

    // Start with AS half-spread core.
    let mut half_spread = (delta_no_vol + delta_diff).max(0.0);

    // Volatility buffer in price units (spec-style): EDGE_VOL_MULT * sigma * S
    let vol_buf = mm_cfg.edge_vol_mult * sigma_eff * s_t;

    // Enforce a minimum economic half-spread buffer.
    // (We treat edge_local_min as a price-unit minimum edge threshold.)
    half_spread = half_spread.max(mm_cfg.edge_local_min.max(0.0));
    half_spread += vol_buf.max(0.0);

    // Apply global spread multiplier (vol regime).
    half_spread *= spread_mult.max(1e-6);

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
    // 2) Reservation price: S + basis + funding - inventory skew
    // ---------------------------------------------------------------------

    // Funding expected PnL per unit over horizon tau:
    //   funding_8h * (tau/8h) * S
    let funding_horizon_frac = tau / (8.0 * 60.0 * 60.0);
    let funding_pnl_per_unit = funding_8h * funding_horizon_frac * s_t;

    let basis_adj = mm_cfg.basis_weight * basis_v;
    let funding_adj = mm_cfg.funding_weight * funding_pnl_per_unit;

    // Inventory skew (classic AS reservation price):
    //   r = S - gamma*sigma^2*tau*q
    //
    // Positive q (long) => lower reservation => lean offers.
    // Negative q (short) => higher reservation => lean bids.
    let inv_term = gamma * sigma_eff * sigma_eff * tau * q_global;

    // Use fair value as base (not mid), then embed basis/funding adjustments.
    let reservation_price = s_t + basis_adj + funding_adj - inv_term;

    // Raw quote prices.
    let raw_bid = reservation_price - half_spread;
    let raw_ask = reservation_price + half_spread;

    // Tick snap.
    let tick = vcfg.tick_size.max(1e-6);
    let bid_price = (raw_bid / tick).floor() * tick;
    let ask_price = (raw_ask / tick).ceil() * tick;

    // Sanity.
    if bid_price <= 0.0 || ask_price <= bid_price {
        return (None, None);
    }

    // ---------------------------------------------------------------------
    // 3) Side-level edge gating (spec: only quote if edge >= EDGE_LOCAL_MIN)
    // ---------------------------------------------------------------------

    // Approx edge vs fair value (ignoring explicit fee/slippage model here):
    let edge_bid = (s_t - bid_price).max(0.0);
    let edge_ask = (ask_price - s_t).max(0.0);

    // If implied edge below threshold, don’t quote that side.
    let edge_min = mm_cfg.edge_local_min.max(0.0);
    let bid_edge_ok = edge_bid >= edge_min;
    let ask_edge_ok = edge_ask >= edge_min;

    // ---------------------------------------------------------------------
    // 4) Size: volatility scaling + directional inventory tilt + risk caps
    // ---------------------------------------------------------------------

    // Start from venue base size and global volatility size multiplier.
    // (This keeps existing behaviour stable while making sizes risk-aware.)
    let base_size = (vcfg.base_order_size * size_mult).max(0.0);

    let mut size_bid = if bid_edge_ok { base_size } else { 0.0 };
    let mut size_ask = if ask_edge_ok { base_size } else { 0.0 };

    // Directional inventory tilt:
    // - If long (q_global > 0), reduce bid size (which increases long).
    // - If short (q_global < 0), reduce ask size (which increases short).
    let eta = mm_cfg.size_eta.max(1e-9);

    if q_global > 0.0 {
        // long => dampen bids
        let damp = 1.0 / (1.0 + eta * q_global.abs());
        size_bid *= damp;
    } else if q_global < 0.0 {
        // short => dampen asks
        let damp = 1.0 / (1.0 + eta * q_global.abs());
        size_ask *= damp;
    }

    // Venue health: Warning venues get reduced size.
    if matches!(vstate.status, VenueStatus::Warning) {
        size_bid *= 0.5;
        size_ask *= 0.5;
    }

    // Liquidation proximity sizing:
    // Between (liq_crit, liq_warn), smoothly shrink sizes to zero.
    if dist_liq < risk_cfg.liq_warn_sigma {
        let t = (dist_liq - risk_cfg.liq_crit_sigma)
            / (risk_cfg.liq_warn_sigma - risk_cfg.liq_crit_sigma);
        let t_clamped = t.clamp(0.0, 1.0);
        size_bid *= t_clamped;
        size_ask *= t_clamped;
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
            // long => only allow sells (asks)
            size_bid = 0.0;
            size_ask *= factor;
        } else if q_global < 0.0 {
            // short => only allow buys (bids)
            size_ask = 0.0;
            size_bid *= factor;
        } else {
            // Should be rare; shrink both.
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

    // If a side size is effectively zero, drop it.
    let bid = if size_bid > 0.0 {
        Some(MmLevel {
            price: bid_price,
            size: size_bid,
        })
    } else {
        None
    };

    let ask = if size_ask > 0.0 {
        Some(MmLevel {
            price: ask_price,
            size: size_ask,
        })
    } else {
        None
    };

    (bid, ask)
}
