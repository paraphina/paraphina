// src/strategy_core.rs
//
// Pure strategy computation for Paraphina (Milestone H).
//
// This module contains the canonical strategy function that:
// - Takes immutable state and config
// - Returns a deterministic list of Actions
// - Performs NO I/O (no file writes, no env reads, no printing)
//
// The strategy core calls existing strategy modules (mm, exit, hedge)
// and converts their outputs into Action types.
//
// Design principles:
// - Pure function: same inputs always produce same outputs
// - Deterministic: uses ActionIdGenerator for stable IDs
// - Composable: each strategy phase is independently testable

use crate::actions::{Action, ActionBatch, ActionBuilder, ActionIdGenerator, LogLevel};
use crate::config::Config;
use crate::exit;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::{GlobalState, KillReason, RiskRegime};
use crate::types::{OrderIntent, TimestampMs};

/// Input context for strategy computation.
///
/// Bundles all inputs needed to compute actions for one tick.
#[derive(Debug, Clone)]
pub struct StrategyInput<'a> {
    /// Strategy configuration.
    pub cfg: &'a Config,
    /// Current global state (immutable snapshot).
    pub state: &'a GlobalState,
    /// Current timestamp in milliseconds.
    pub now_ms: TimestampMs,
    /// Current tick index.
    pub tick_index: u64,
    /// Optional run seed for determinism.
    pub run_seed: Option<u64>,
}

/// Output from strategy computation.
///
/// Contains all actions to execute plus metadata for replay verification.
#[derive(Debug, Clone)]
pub struct StrategyOutput {
    /// The batch of actions to execute.
    pub batch: ActionBatch,
    /// Order intents for MM phase (for state updates).
    pub mm_intents: Vec<OrderIntent>,
    /// Order intents for exit phase (for state updates).
    pub exit_intents: Vec<OrderIntent>,
    /// Order intents for hedge phase (for state updates).
    pub hedge_intents: Vec<OrderIntent>,
}

/// Compute all strategy actions for a single tick.
///
/// This is the canonical pure function for the strategy core.
/// It does NOT modify state or perform any I/O.
///
/// # Arguments
/// * `input` - All inputs needed for strategy computation
///
/// # Returns
/// A `StrategyOutput` containing all actions and intents for this tick.
pub fn compute_actions(input: StrategyInput<'_>) -> StrategyOutput {
    let mut gen = ActionIdGenerator::new(input.tick_index);
    let mut batch = ActionBatch::new(input.now_ms, input.tick_index, input.cfg.version)
        .with_seed(input.run_seed);

    let mut mm_intents = Vec::new();
    let mut exit_intents = Vec::new();
    let mut hedge_intents = Vec::new();

    // Early exit if kill switch is active
    if input.state.kill_switch {
        // Log that we're in kill mode
        let mut builder = ActionBuilder::new(&mut gen);
        batch.push(builder.log(
            LogLevel::Warn,
            &format!(
                "Kill switch active (reason: {:?}), skipping strategy",
                input.state.kill_reason
            ),
        ));
        return StrategyOutput {
            batch,
            mm_intents,
            exit_intents,
            hedge_intents,
        };
    }

    // Phase 1: Market Making
    mm_intents = compute_mm_phase(input.cfg, input.state, &mut gen, &mut batch);

    // Phase 2: Exit (before hedge per spec)
    if input.cfg.exit.enabled {
        exit_intents =
            compute_exit_phase(input.cfg, input.state, input.now_ms, &mut gen, &mut batch);
    }

    // Phase 3: Hedge
    hedge_intents = compute_hedge_phase(input.cfg, input.state, input.now_ms, &mut gen, &mut batch);

    // Check for risk transitions that should trigger kill switch
    check_risk_transitions(input.state, &mut gen, &mut batch);

    StrategyOutput {
        batch,
        mm_intents,
        exit_intents,
        hedge_intents,
    }
}

/// Compute market-making actions.
fn compute_mm_phase(
    cfg: &Config,
    state: &GlobalState,
    gen: &mut ActionIdGenerator,
    batch: &mut ActionBatch,
) -> Vec<OrderIntent> {
    let mm_quotes = compute_mm_quotes(cfg, state);
    let intents = mm_quotes_to_order_intents(&mm_quotes);

    // Convert intents to actions
    let mut builder = ActionBuilder::new(gen);
    for intent in &intents {
        let action = builder.place_order(
            intent.venue_index,
            &intent.venue_id,
            intent.side,
            intent.price,
            intent.size,
            intent.purpose,
        );
        batch.push(action);
    }

    intents
}

/// Compute exit actions.
fn compute_exit_phase(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    gen: &mut ActionIdGenerator,
    batch: &mut ActionBatch,
) -> Vec<OrderIntent> {
    let intents = exit::compute_exit_intents(cfg, state, now_ms);

    // Convert intents to actions
    let mut builder = ActionBuilder::new(gen);
    for intent in &intents {
        let action = builder.place_order(
            intent.venue_index,
            &intent.venue_id,
            intent.side,
            intent.price,
            intent.size,
            intent.purpose,
        );
        batch.push(action);
    }

    intents
}

/// Compute hedge actions.
fn compute_hedge_phase(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    gen: &mut ActionIdGenerator,
    batch: &mut ActionBatch,
) -> Vec<OrderIntent> {
    let plan = compute_hedge_plan(cfg, state, now_ms);
    let intents = match plan {
        Some(ref p) => hedge_plan_to_order_intents(p),
        None => Vec::new(),
    };

    // Convert intents to actions
    let mut builder = ActionBuilder::new(gen);
    for intent in &intents {
        let action = builder.place_order(
            intent.venue_index,
            &intent.venue_id,
            intent.side,
            intent.price,
            intent.size,
            intent.purpose,
        );
        batch.push(action);
    }

    intents
}

/// Check for risk state transitions and emit appropriate actions.
fn check_risk_transitions(
    state: &GlobalState,
    gen: &mut ActionIdGenerator,
    batch: &mut ActionBatch,
) {
    let mut builder = ActionBuilder::new(gen);

    // If we're in HardLimit regime but kill switch isn't set,
    // emit a SetKillSwitch action (this handles the case where
    // the engine hasn't processed the risk transition yet)
    if matches!(state.risk_regime, RiskRegime::HardLimit) && !state.kill_switch {
        // Determine reason from state
        let reason = if state.daily_pnl_total <= -state.delta_limit_usd {
            KillReason::PnlHardBreach
        } else if state.dollar_delta_usd.abs() >= state.delta_limit_usd {
            KillReason::DeltaHardBreach
        } else if state.basis_usd.abs() >= state.basis_limit_hard_usd {
            KillReason::BasisHardBreach
        } else {
            KillReason::None
        };

        if reason != KillReason::None {
            batch.push(builder.set_kill_switch(true, reason));
        }
    }

    // Log regime if in warning
    if matches!(state.risk_regime, RiskRegime::Warning) {
        batch.push(builder.log(
            LogLevel::Warn,
            &format!(
                "Warning regime: delta={:.2} basis={:.2} pnl={:.2}",
                state.dollar_delta_usd, state.basis_usd, state.daily_pnl_total
            ),
        ));
    }
}

/// Simplified single-phase compute for testing.
///
/// Computes only MM actions (useful for unit tests).
pub fn compute_mm_actions_only(cfg: &Config, state: &GlobalState, tick_index: u64) -> Vec<Action> {
    let mut gen = ActionIdGenerator::new(tick_index);
    let mut batch = ActionBatch::new(0, tick_index, cfg.version);

    compute_mm_phase(cfg, state, &mut gen, &mut batch);

    batch.actions
}

/// Simplified single-phase compute for testing.
///
/// Computes only hedge actions (useful for unit tests).
pub fn compute_hedge_actions_only(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    tick_index: u64,
) -> Vec<Action> {
    let mut gen = ActionIdGenerator::new(tick_index);
    let mut batch = ActionBatch::new(now_ms, tick_index, cfg.version);

    compute_hedge_phase(cfg, state, now_ms, &mut gen, &mut batch);

    batch.actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VenueStatus;

    fn setup_test_state() -> (Config, GlobalState) {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        // Set up fair value and basic market conditions
        state.fair_value = Some(300.0);
        state.fair_value_prev = 300.0;
        state.sigma_eff = 0.02;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;
        state.vol_ratio_clipped = 1.0;
        state.delta_limit_usd = 100_000.0;

        // Set up venue states with valid data
        for v in &mut state.venues {
            v.mid = Some(300.0);
            v.spread = Some(1.00);
            v.depth_near_mid = 10_000.0;
            v.margin_available_usd = 10_000.0;
            v.dist_liq_sigma = 10.0;
            v.status = VenueStatus::Healthy;
            v.toxicity = 0.0;
            v.last_mid_update_ms = Some(1000);
        }

        (cfg, state)
    }

    #[test]
    fn test_compute_actions_deterministic() {
        let (cfg, state) = setup_test_state();

        let input1 = StrategyInput {
            cfg: &cfg,
            state: &state,
            now_ms: 1000,
            tick_index: 5,
            run_seed: Some(42),
        };

        let input2 = StrategyInput {
            cfg: &cfg,
            state: &state,
            now_ms: 1000,
            tick_index: 5,
            run_seed: Some(42),
        };

        let output1 = compute_actions(input1);
        let output2 = compute_actions(input2);

        // Same inputs should produce identical outputs
        assert_eq!(output1.batch.actions.len(), output2.batch.actions.len());
        assert_eq!(output1.batch.tick_index, output2.batch.tick_index);

        // Actions should be identical
        for (a1, a2) in output1
            .batch
            .actions
            .iter()
            .zip(output2.batch.actions.iter())
        {
            assert_eq!(a1, a2);
        }
    }

    #[test]
    fn test_compute_actions_kill_switch_active() {
        let (cfg, mut state) = setup_test_state();
        state.kill_switch = true;
        state.kill_reason = KillReason::PnlHardBreach;

        let input = StrategyInput {
            cfg: &cfg,
            state: &state,
            now_ms: 1000,
            tick_index: 0,
            run_seed: None,
        };

        let output = compute_actions(input);

        // Should only have a log action, no trading actions
        assert_eq!(output.batch.actions.len(), 1);
        match &output.batch.actions[0] {
            Action::Log(log) => {
                assert_eq!(log.level, LogLevel::Warn);
                assert!(log.message.contains("Kill switch"));
            }
            _ => panic!("Expected Log action"),
        }
    }

    #[test]
    fn test_compute_actions_produces_mm_intents() {
        let (cfg, state) = setup_test_state();

        let input = StrategyInput {
            cfg: &cfg,
            state: &state,
            now_ms: 1000,
            tick_index: 0,
            run_seed: None,
        };

        let output = compute_actions(input);

        // Should have MM intents (venues are healthy with valid data)
        assert!(!output.mm_intents.is_empty());
    }

    #[test]
    fn test_action_ids_unique_within_batch() {
        let (cfg, state) = setup_test_state();

        let input = StrategyInput {
            cfg: &cfg,
            state: &state,
            now_ms: 1000,
            tick_index: 0,
            run_seed: None,
        };

        let output = compute_actions(input);

        // Collect all action IDs
        let mut ids: Vec<String> = Vec::new();
        for action in &output.batch.actions {
            let id = match action {
                Action::PlaceOrder(a) => a.action_id.clone(),
                Action::CancelOrder(a) => a.action_id.clone(),
                Action::CancelAll(a) => a.action_id.clone(),
                Action::SetKillSwitch(a) => a.action_id.clone(),
                Action::Log(a) => a.action_id.clone(),
            };
            ids.push(id);
        }

        // Check uniqueness
        let original_len = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "Action IDs should be unique");
    }
}
