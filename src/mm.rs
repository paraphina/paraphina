// src/mm.rs
//
// Market-making quote engine (Avellaneda–Stoikov style) for Paraphina.
//
// Responsibilities:
//   - For each venue, convert global fair value + risk scalars + venue state
//     into a local bid/ask quote.
//   - Respect kill switch and venue health.
//   - Incorporate inventory, volatility, basis, funding and risk regime
//     into reservation price and quote sizes.
//   - Expose a thin abstraction (MmQuote) which main.rs converts into
//     abstract OrderIntents.

use crate::config::{Config, VenueConfig};
use crate::state::{GlobalState, VenueState, RiskRegime};
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
/// for each venue as described in the whitepaper.
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
    let sigma_eff = state
        .sigma_eff
        .max(cfg.volatility.sigma_min)
        .max(1e-8); // avoid degenerate 0-vol modes

    let vol_ratio = state.vol_ratio_clipped;
    let spread_mult = state.spread_mult.max(0.0);
    let size_mult = state.size_mult.max(0.0);
    let band_mult = state.band_mult.max(0.0); // currently only used conceptually

    let q_global = state.q_global_tao;
    let delta_abs_usd = state.dollar_delta_usd.abs();
    let delta_limit_usd = state.delta_limit_usd.max(1.0);
    let risk_regime = state.risk_regime;

    // Kill switch: still return one entry per venue, but with no quotes.
    if state.kill_switch {
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

    for (i, vcfg) in cfg.venues.iter().enumerate() {
        let vstate = &state.venues[i];

        // Only quote on Healthy / Warning venues. Disabled ⇒ no quotes.
        if matches!(vstate.status, VenueStatus::Disabled) {
            out.push(MmQuote {
                venue_index: i,
                venue_id: vcfg.id.clone(),
                bid: None,
                ask: None,
            });
            continue;
        }

        // Local mid: prefer venue mid if present, otherwise fall back
        // to the global fair value S_t.
        let mid: f64 = vstate.mid.unwrap_or(s_t);

        // Local basis and funding.
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
///   - risk-regime / liq / delta-limit aware size scaling.
fn compute_single_venue_quotes(
    cfg: &Config,
    vcfg: &VenueConfig,
    vstate: &VenueState,
    s_t: f64,
    mid: f64,
    basis_v: f64,
    funding_8h: f64,
    sigma_eff: f64,
    vol_ratio: f64,
    q_global: f64,
    delta_abs_usd: f64,
    delta_limit_usd: f64,
    spread_mult: f64,
    size_mult: f64,
    _band_mult: f64,
    risk_regime: RiskRegime,
) -> (Option<MmLevel>, Option<MmLevel>) {
    // If volatility or size_mult are degenerate, don't quote.
    if sigma_eff <= 0.0 || size_mult <= 0.0 {
        return (None, None);
    }

    let mm_cfg = &cfg.mm;
    let risk_cfg = &cfg.risk;

    // ----- 1) Half-spread (Avellaneda–Stoikov core + vol / risk multipliers) -----

    // Avellaneda–Stoikov base half-spread (simplified):
    //
    //   δ* ≈ (γ σ² T) / 2   +   (1 / k)
    //
    let gamma = vcfg.gamma.max(1e-8);
    let k = vcfg.k.max(1e-8);
    let t_horizon = mm_cfg.quote_horizon_sec.max(1.0);

    let base_half_spread = (gamma * sigma_eff * sigma_eff * t_horizon) / 2.0 + 1.0 / k;

    // Local minimum edge, plus a vol-dependent buffer.
    let mut half_spread = base_half_spread.max(mm_cfg.edge_local_min);
    half_spread += mm_cfg.edge_vol_mult * vol_ratio;
    half_spread = half_spread.max(mm_cfg.edge_local_min);

    // Global spread multiplier (from vol regime).
    half_spread *= spread_mult.max(1e-6);

    // Warning risk regime: widen spreads.
    if matches!(risk_regime, RiskRegime::Warning) {
        half_spread *= risk_cfg.spread_warn_mult.max(1.0);
    }

    // Distance-to-liq adjustment: as we approach liquidation, widen more.
    let dist_liq = vstate.dist_liq_sigma;
    if dist_liq > 0.0 && dist_liq < risk_cfg.liq_warn_sigma {
        // Linear ramp: at liq_warn_sigma → 1x, near 0 → up to 3x spread.
        let t = ((risk_cfg.liq_warn_sigma - dist_liq) / risk_cfg.liq_warn_sigma)
            .clamp(0.0, 1.0);
        half_spread *= 1.0 + 2.0 * t;
    }

    // ----- 2) Reservation price: inventory + basis + funding adjustments -----

    // Global inventory skew:
    //
    //   r_v = mid - (γ σ² T / 2) * q_global  +  basis_term  +  funding_term
    //
    // Negative q_global (short) ⇒ reservation price above mid ⇒ stronger bids.
    // Positive q_global (long)  ⇒ reservation price below mid ⇒ lean offers.
    let inv_term = (gamma * sigma_eff * sigma_eff * t_horizon * q_global) / 2.0;

    // Basis adjustment (USD):
    let basis_adj = mm_cfg.basis_weight * basis_v;

    // Funding adjustment: approximate expected funding PnL over the quote horizon.
    // funding_8h is dimensionless (rate per 8h). Convert to horizon fraction.
    let funding_horizon_frac = mm_cfg.quote_horizon_sec / (8.0 * 60.0 * 60.0);
    let funding_pnl_per_unit = funding_8h * funding_horizon_frac * s_t;
    let funding_adj = mm_cfg.funding_weight * funding_pnl_per_unit;

    let reservation_price = mid + basis_adj + funding_adj - inv_term;

    // ----- 3) Size logic -----

    // Base size from venue config + global vol scaling.
    let mut size = vcfg.base_order_size * size_mult;

    // Global inventory penalty:
    //
    //   size = base_size / (1 + η |q_global|)
    //
    let size_scale_inv = 1.0 + mm_cfg.size_eta * q_global.abs();
    size /= size_scale_inv.max(1e-6);

    // Risk-regime size limits.
    if matches!(risk_regime, RiskRegime::Warning) {
        size = size.min(risk_cfg.q_warn_cap.max(0.0));
    }

    // Distance-to-liq: inside liq_crit_sigma we refuse to quote;
    // between (liq_crit, liq_warn) we smoothly shrink sizes to zero.
    if dist_liq <= risk_cfg.liq_crit_sigma {
        return (None, None);
    } else if dist_liq < risk_cfg.liq_warn_sigma {
        let t = (dist_liq - risk_cfg.liq_crit_sigma)
            / (risk_cfg.liq_warn_sigma - risk_cfg.liq_crit_sigma);
        let t_clamped = t.clamp(0.0, 1.0);
        size *= t_clamped;
    }

    // Venue health: Warning venues get half size.
    if matches!(vstate.status, VenueStatus::Warning) {
        size *= 0.5;
    }

    // Delta-limit proximity: as |Δ| approaches 2× limit, shrink to zero.
    let delta_ratio = (delta_abs_usd / delta_limit_usd).max(0.0);
    if delta_ratio >= 2.0 {
        return (None, None);
    } else if delta_ratio > 1.0 {
        // Linear ramp from 1.0 → 0.0 as ratio goes 1 → 2.
        let factor = (2.0 - delta_ratio).clamp(0.0, 1.0);
        size *= factor;
    }

    // Hard cap by venue max size.
    if size > vcfg.max_order_size {
        size = vcfg.max_order_size;
    }

    if size <= 0.0 {
        return (None, None);
    }

    // ----- 4) Price levels: tick-aligned bid / ask around reservation price -----

    let raw_bid = reservation_price - half_spread;
    let raw_ask = reservation_price + half_spread;

    let tick = vcfg.tick_size.max(1e-6);

    // Tick-aligned prices.
    let bid_price = (raw_bid / tick).floor() * tick;
    let ask_price = (raw_ask / tick).ceil() * tick;

    // Sanity checks.
    if bid_price <= 0.0 || ask_price <= bid_price {
        return (None, None);
    }

    let bid = MmLevel {
        price: bid_price,
        size,
    };

    let ask = MmLevel {
        price: ask_price,
        size,
    };

    (Some(bid), Some(ask))
}
