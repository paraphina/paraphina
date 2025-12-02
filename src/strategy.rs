// src/strategy.rs
//
// Strategy policy layer for Paraphina.
//
// This module takes a read-only snapshot of (cfg, state),
// computes MM quotes + a global hedge plan, converts them
// into OrderIntents, and returns everything as a StrategyStep.
//
// This is the pure "π(s)" policy that we will later plug into
// real execution backends and RL / self-calibration.

use crate::config::Config;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents, HedgePlan};
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents, MmQuote};
use crate::state::GlobalState;
use crate::types::OrderIntent;

/// One strategy decision at a single tick.
#[derive(Debug, Clone)]
pub struct StrategyStep {
    /// Per-venue MM quotes (bid/ask) before any fills.
    pub mm_quotes: Vec<MmQuote>,
    /// Optional global hedge plan (may be None if inside band or kill-switch).
    pub hedge_plan: Option<HedgePlan>,
    /// MM order intents derived from the MM quotes.
    pub mm_intents: Vec<OrderIntent>,
    /// Hedge order intents derived from the hedge plan.
    pub hedge_intents: Vec<OrderIntent>,
}

/// Compute the strategy's decision for this tick.
///
/// This is deliberately side-effect-free: it only reads `cfg` and `state`
/// and returns a StrategyStep. Execution / fills are handled elsewhere.
pub fn compute_strategy_step(cfg: &Config, state: &GlobalState) -> StrategyStep {
    // 1) Per-venue MM quotes.
    let mm_quotes = compute_mm_quotes(cfg, state);
    let mm_intents = mm_quotes_to_order_intents(&mm_quotes);

    // 2) Global hedge plan.
    let hedge_plan = compute_hedge_plan(cfg, state);
    let hedge_intents = hedge_plan
        .as_ref()
        .map(hedge_plan_to_order_intents)
        .unwrap_or_default();

    StrategyStep {
        mm_quotes,
        hedge_plan,
        mm_intents,
        hedge_intents,
    }
}
