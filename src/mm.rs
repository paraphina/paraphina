// src/mm.rs
//
// Market-making quote engine:
//  - computes per-venue reservation prices and spreads
//  - applies volatility, risk and margin scalars
//  - returns abstract MM order intents

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{OrderIntent, OrderPurpose, Side};

#[derive(Debug, Clone)]
pub struct QuoteLevel {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct MmQuote {
    pub venue_index: usize,
    pub venue_id: String,
    pub bid: Option<QuoteLevel>,
    pub ask: Option<QuoteLevel>,
}

impl MmQuote {
    pub fn empty(venue_id: String, venue_index: usize) -> Self {
        Self {
            venue_index,
            venue_id,
            bid: None,
            ask: None,
        }
    }
}

/// Compute per-venue MM quotes according to the whitepaper-style model.
pub fn compute_mm_quotes(cfg: &Config, state: &GlobalState) -> Vec<MmQuote> {
    // Kill switch: no quoting at all.
    if state.kill_switch {
        return Vec::new();
    }

    let Some(s_t) = state.fair_value else {
        // No fair value -> no quotes.
        return Vec::new();
    };

    let sigma_eff = state.sigma_eff.max(cfg.volatility.sigma_min);
    let spread_mult = state.spread_mult;
    let size_mult = state.size_mult;
    let band_mult = state.band_mult;

    // Global risk-driven size scaling (delta + basis).
    let risk_size_scale = compute_risk_size_scale(cfg, state);

    cfg.venues
        .iter()
        .enumerate()
        .map(|(venue_index, vcfg)| {
            // Per-venue Avellaneda-style half-spread.
            let gamma = vcfg.gamma.max(1e-6);
            let k = vcfg.k.max(1e-6);
            let t_h = cfg.mm.quote_horizon_sec.max(1e-3);

            // Very simple AS-inspired decomposition:
            let vol_term = gamma * sigma_eff * s_t * t_h.sqrt();
            let intensity_term = (1.0 / gamma) * (1.0 + gamma / k).ln() / t_h;

            let mut half_spread = (vol_term + intensity_term).abs();

            // Add explicit local edges from config.
            half_spread += cfg.mm.edge_local_min + cfg.mm.edge_vol_mult * sigma_eff * s_t;

            // Vol- and risk-driven spread multipliers.
            half_spread *= spread_mult * band_mult;

            // Enforce at least one tick.
            half_spread = (half_spread / vcfg.tick_size).ceil().max(1.0) * vcfg.tick_size;

            if !half_spread.is_finite() || half_spread <= 0.0 {
                return MmQuote::empty(vcfg.id.clone(), venue_index);
            }

            // Base symmetric size (per-venue base size * global controls * risk scalar).
            let mut q = vcfg.base_order_size * size_mult * risk_size_scale;

            // Funding / basis driven skew on size:
            // this uses funding_skew_slope, funding_skew_clip and w_fund,
            // with basis ratio as a stand-in signal until real funding data is wired.
            let funding_skew = compute_funding_skew(cfg, state, venue_index);
            let funding_mult = (1.0 + funding_skew).clamp(0.5, 2.0);
            q *= funding_mult;

            // Venue hard caps.
            q = q.min(vcfg.max_order_size);

            // Margin-aware cap using mm_margin_safety & mm_max_leverage.
            q = cap_size_by_margin(cfg, state, s_t, q);

            if q <= 0.0 || !q.is_finite() {
                return MmQuote::empty(vcfg.id.clone(), venue_index);
            }

            // Per-venue reservation price with basis + (placeholder) funding terms.
            let r_v = compute_reservation_price(cfg, state, venue_index, s_t);

            let bid_price = round_down_to_tick(r_v - half_spread, vcfg.tick_size);
            let ask_price = round_up_to_tick(r_v + half_spread, vcfg.tick_size);

            MmQuote {
                venue_index,
                venue_id: vcfg.id.clone(),
                bid: Some(QuoteLevel {
                    price: bid_price,
                    size: q,
                }),
                ask: Some(QuoteLevel {
                    price: ask_price,
                    size: q,
                }),
            }
        })
        .collect()
}

/// Convert MM quotes into abstract order intents for the driver.
pub fn mm_quotes_to_order_intents(quotes: &[MmQuote]) -> Vec<OrderIntent> {
    let mut intents = Vec::new();

    for q in quotes {
        if let Some(bid) = &q.bid {
            intents.push(OrderIntent {
                venue_index: q.venue_index,
                venue_id: q.venue_id.clone(),
                side: Side::Buy,
                size: bid.size,
                price: bid.price,
                purpose: OrderPurpose::Mm,
            });
        }

        if let Some(ask) = &q.ask {
            intents.push(OrderIntent {
                venue_index: q.venue_index,
                venue_id: q.venue_id.clone(),
                side: Side::Sell,
                size: ask.size,
                price: ask.price,
                purpose: OrderPurpose::Mm,
            });
        }
    }

    intents
}

/// Global risk utilisation → smooth size multiplier in [0.1, 1.0].
fn compute_risk_size_scale(cfg: &Config, state: &GlobalState) -> f64 {
    let delta_util = if state.delta_limit_usd > 0.0 {
        (state.dollar_delta_usd.abs() / state.delta_limit_usd).min(1.0)
    } else {
        0.0
    };

    let basis_util = if cfg.risk.basis_hard_limit_usd > 0.0 {
        (state.basis_usd.abs() / cfg.risk.basis_hard_limit_usd).min(1.0)
    } else {
        0.0
    };

    let util = delta_util.max(basis_util);
    let shrink = cfg.mm.size_eta * util; // size_eta ∈ [0,1] recommended.
    (1.0 - shrink).clamp(0.1, 1.0)
}

/// Reservation price r_v(t) with basis + (placeholder) funding adjustments.
fn compute_reservation_price(cfg: &Config, state: &GlobalState, venue_index: usize, s_t: f64) -> f64 {
    let mm_cfg = &cfg.mm;
    let risk_cfg = &cfg.risk;
    let vcfg = &cfg.venues[venue_index];

    // Normalised basis in [-1, 1].
    let basis_ratio = if risk_cfg.basis_hard_limit_usd > 0.0 {
        (state.basis_usd / risk_cfg.basis_hard_limit_usd).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // Basis term scaled to be O(1% * basis_weight * w_liq) of price at max utilisation.
    let basis_shift = mm_cfg.basis_weight
        * vcfg.w_liq
        * basis_ratio
        * 0.01
        * s_t;

    // Funding term placeholder: wire real per-venue funding here later.
    let funding_signal = 0.0_f64;
    let funding_shift = mm_cfg.funding_weight
        * vcfg.w_fund
        * funding_signal
        * 0.01
        * s_t;

    let mut r_v = s_t + basis_shift + funding_shift;

    // Tiny static tilt based on (w_liq + w_fund) vs equal weighting;
    // this makes sure both weights are "live" without dominating behaviour.
    let equal_weight = 1.0 / (cfg.venues.len() as f64).max(1.0);
    let liq_fund_excess = (vcfg.w_liq + vcfg.w_fund) - 2.0 * equal_weight;
    r_v += liq_fund_excess * 0.001 * s_t;

    r_v.max(0.01)
}

/// Funding / basis → size skew per venue, using funding_skew_slope & funding_skew_clip.
fn compute_funding_skew(cfg: &Config, state: &GlobalState, venue_index: usize) -> f64 {
    let mm_cfg = &cfg.mm;
    let risk_cfg = &cfg.risk;
    let vcfg = &cfg.venues[venue_index];

    // Use basis_ratio as a stand-in signal for now; later this can be replaced
    // by actual expected funding PnL per venue.
    let basis_ratio = if risk_cfg.basis_hard_limit_usd > 0.0 {
        state.basis_usd / risk_cfg.basis_hard_limit_usd
    } else {
        0.0
    };

    // Combine with w_fund and funding_skew_slope, with a small scale factor
    // so typical utilisation produces modest size skews (≈10–20%).
    let signal = basis_ratio * vcfg.w_fund;
    let raw = signal * mm_cfg.funding_skew_slope * 1e-4;

    raw.clamp(-mm_cfg.funding_skew_clip, mm_cfg.funding_skew_clip)
}

/// Margin-aware per-order cap using mm_margin_safety * mm_max_leverage * delta_limit_usd
/// as a crude proxy for available notional.
fn cap_size_by_margin(cfg: &Config, state: &GlobalState, s_t: f64, q: f64) -> f64 {
    if s_t <= 0.0 || q <= 0.0 {
        return q.max(0.0);
    }

    let mm_risk = &cfg.risk;
    let base_limit = state.delta_limit_usd.max(0.0);

    if base_limit <= 0.0 {
        return q;
    }

    // Total notional budget we are willing to allocate to MM quotes.
    let notional_cap_total =
        mm_risk.mm_margin_safety.max(0.0) * mm_risk.mm_max_leverage.max(0.0) * base_limit;

    if notional_cap_total <= 0.0 {
        return q;
    }

    // Be conservative and allocate only a small fraction per quote.
    let venues = cfg.venues.len() as f64;
    let per_order_cap = notional_cap_total / (venues * 4.0).max(1.0); // ~5% total if fully utilised.

    let q_cap = (per_order_cap / s_t).max(0.0);

    q.min(q_cap)
}

fn round_down_to_tick(price: f64, tick: f64) -> f64 {
    if tick <= 0.0 {
        return price;
    }
    (price / tick).floor() * tick
}

fn round_up_to_tick(price: f64, tick: f64) -> f64 {
    if tick <= 0.0 {
        return price;
    }
    (price / tick).ceil() * tick
}
