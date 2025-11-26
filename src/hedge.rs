// src/hedge.rs
//
// Hedge engine skeleton: global LQ control with dead band (Section 13).

use crate::config::Config;
use crate::state::{GlobalState, RiskRegime};
use crate::types::{OrderIntent, OrderPurpose, Side};

#[derive(Debug, Clone, Copy)]
pub enum HedgeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: HedgeSide,
    pub size: f64,
    pub est_price: f64,
}

#[derive(Debug, Clone)]
pub struct HedgePlan {
    /// Desired change in global delta (trade size), in TAO.
    /// Positive = net buy (reduce short), negative = net sell (reduce long).
    pub desired_delta: f64,
    pub allocations: Vec<HedgeAllocation>,
}

pub fn compute_hedge_plan(cfg: &Config, state: &GlobalState) -> Option<HedgePlan> {
    if state.kill_switch || matches!(state.risk_regime, RiskRegime::Critical) {
        return None;
    }

    let s_t = state.fair_value?;
    let hedge_cfg = &cfg.hedge;

    let q_t = state.q_global_tao;

    // Vol-adjusted dead band: band_vol = HEDGE_BAND_BASE * band_mult(t).
    let band_vol = hedge_cfg.hedge_band_base * state.band_mult;

    if q_t.abs() <= band_vol {
        // Inside dead band: no hedge.
        return None;
    }

    // LQ controller:
    // ΔH_raw* = -k * X_t, where X_t = q_t, k = α/(α+β).
    let k_hedge = hedge_cfg.alpha_hedge / (hedge_cfg.alpha_hedge + hedge_cfg.beta_hedge);
    let delta_h_raw = -k_hedge * q_t;

    // Clamp per-step hedge size.
    let desired_delta = delta_h_raw
        .signum()
        * delta_h_raw.abs().min(hedge_cfg.hedge_max_step);

    if desired_delta.abs() < 1e-6 {
        return None;
    }

    // Simple allocation: send whole hedge to first hedge-allowed venue.
    let (idx, venue_cfg) = match cfg
        .venues
        .iter()
        .enumerate()
        .find(|(_, v)| v.is_hedge_allowed)
    {
        Some(x) => x,
        None => return None,
    };

    let venue_state = &state.venues[idx];
    let est_price = venue_state.mid.unwrap_or(s_t);

    let side = if desired_delta > 0.0 {
        HedgeSide::Buy
    } else {
        HedgeSide::Sell
    };

    let alloc = HedgeAllocation {
        venue_index: idx,
        venue_id: venue_cfg.id.clone(),
        side,
        size: desired_delta.abs(),
        est_price,
    };

    Some(HedgePlan {
        desired_delta,
        allocations: vec![alloc],
    })
}

/// Convert a hedge plan into order intents (one intent per allocation).
pub fn hedge_plan_to_order_intents(plan: &HedgePlan) -> Vec<OrderIntent> {
    plan.allocations
        .iter()
        .map(|a| OrderIntent {
            venue_index: a.venue_index,
            venue_id: a.venue_id.clone(),
            side: match a.side {
                HedgeSide::Buy => Side::Buy,
                HedgeSide::Sell => Side::Sell,
            },
            price: a.est_price,
            size: a.size,
            purpose: OrderPurpose::Hedge,
        })
        .collect()
}
