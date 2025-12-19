//! Paraphina core library.
//!
//! This crate exposes the core market-making engine, state, and strategy
//! runner. The binary (`src/main.rs`) is just a thin simulation / research
//! harness around these components.
//!
//! # Architecture (Milestone H)
//!
//! The codebase follows a clean separation between strategy logic and I/O:
//!
//! - **Strategy Core** (`strategy_core`): Pure, deterministic functions that
//!   compute what actions to take given state and config. No I/O.
//!
//! - **Actions** (`actions`): Data types representing strategy decisions
//!   (PlaceOrder, CancelOrder, etc.) with deterministic IDs.
//!
//! - **I/O Layer** (`io`): VenueAdapter trait and implementations that
//!   execute actions. Swappable for sim/live/replay modes.
//!
//! - **Gateway** (`gateway`): Legacy execution gateway (preserved for
//!   backwards compatibility with existing code).
//!
//! # RL-0 Foundations (per ROADMAP.md)
//!
//! The `rl` module provides interfaces and instrumentation for RL evolution:
//!
//! - **Observation**: Versioned, serializable state snapshot for policy input
//! - **Policy**: Trait for strategy decision-making (heuristic or learned)
//! - **ShadowRunner**: Parallel execution of baseline + shadow policies
//! - **RLTelemetry**: Logging of policy inputs/outputs and reward components
//!
//! # RL-1 Gym Environment (per ROADMAP.md)
//!
//! The `rl` module also provides Gym-style environments for RL training:
//!
//! - **SimEnv**: Single Gym-style environment (reset, step)
//! - **VecEnv**: Vectorised environments for parallel rollouts
//! - **DomainRandConfig**: Domain randomisation for training robustness

pub mod actions;
pub mod config;
pub mod engine;
pub mod exit;
pub mod gateway;
pub mod hedge;
pub mod io;
pub mod logging;
pub mod metrics;
pub mod mm;
pub mod rl;
pub mod state;
pub mod strategy;
pub mod strategy_core;
pub mod telemetry;
pub mod toxicity;
pub mod types;

// --- Re-exports for ergonomic external use ---------------------------------

pub use config::Config;
pub use engine::Engine;

// Legacy gateway (preserved for backwards compatibility)
pub use gateway::{ExecutionGateway, SimGateway};

// New I/O layer (Milestone H)
pub use io::{
    noop::{compare_batches, create_noop_adapters, ActionRecorder, NoopAdapter},
    sim::{create_sim_adapters, SimAdapter},
    ActionResult, Gateway, GatewayPolicy, RateLimitPolicy, RetryPolicy, VenueAdapter,
};

// Actions (Milestone H)
pub use actions::{
    Action, ActionBatch, ActionBuilder, ActionId, ActionIdGenerator, CancelAllAction,
    CancelOrderAction, LogAction, LogLevel, PlaceOrderAction, SetKillSwitchAction,
};

// Strategy core (Milestone H)
pub use strategy_core::{compute_actions, StrategyInput, StrategyOutput};

pub use hedge::{
    compute_hedge_orders, compute_hedge_plan, hedge_plan_to_order_intents, HedgeAllocation,
    HedgePlan,
};

pub use logging::{EventSink, FileSink, NoopSink};

pub use mm::{
    compute_mm_quotes, compute_order_actions, compute_venue_targets, mm_quotes_to_order_intents,
    should_replace_order, ActiveMmOrder, MmLevel, MmOrderAction, MmQuote, VenueTargetInventory,
};

pub use state::{GlobalState, KillReason, PendingMarkout, RiskRegime, VenueState};

pub use strategy::StrategyRunner;

pub use toxicity::update_toxicity_and_health;

pub use types::{FillEvent, OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus};

// RL-0 Foundations (per ROADMAP.md)
pub use rl::{
    EpisodeConfig, EpisodeSummary, HeuristicPolicy, NoopPolicy, Observation, Policy, PolicyAction,
    RLTelemetry, RewardComponents, RewardWeights, ShadowRunner, TerminationReason, TickRecord,
    VenueObservation, HEURISTIC_POLICY_VERSION, OBS_VERSION,
};

// RL-1 Gym Environment (per ROADMAP.md)
pub use rl::sim_env::{SimEnvConfig, StepInfo};
pub use rl::{DomainRandConfig, DomainRandSample, DomainRandSampler, SimEnv, StepResult, VecEnv};

// RL-2 Behaviour Cloning (per ROADMAP.md)
pub use rl::{
    decode_action, encode_action, ActionEncodingSpec, TrajectoryCollector, TrajectoryMetadata,
    TrajectoryRecord, TrajectoryWriter, ACTION_VERSION, TRAJECTORY_VERSION,
};

// --- PnL / basis unit tests -------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Side;

    fn mk_state() -> (Config, GlobalState) {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        (cfg, state)
    }

    /// Two venues with opposite positions and symmetric basis.
    /// We should end up:
    ///   - globally flat (q_t = 0),
    ///   - positive net & gross basis,
    ///   - zero unrealised PnL when S_t = entry price.
    #[test]
    fn basis_and_unrealised_pnl_two_venues() {
        let (cfg, mut state) = mk_state();

        // Set a fair value anchor and venue mids.
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;

        // Venue 0: +1 TAO long, mid 101.
        state.venues[0].mid = Some(101.0);
        state.apply_perp_fill(0, Side::Buy, 1.0, 100.0, 0.0);

        // Venue 1: -1 TAO short, mid 99.
        state.venues[1].mid = Some(99.0);
        state.apply_perp_fill(1, Side::Sell, 1.0, 100.0, 0.0);

        state.recompute_after_fills(&cfg);

        // Net inventory and dollar delta should be ~0.
        assert!((state.q_global_tao - 0.0).abs() < 1e-9);
        assert!((state.dollar_delta_usd - 0.0).abs() < 1e-6);

        // Basis: q0*(101-100) + q1*(99-100) = 1 + 1 = 2.
        assert!((state.basis_usd - 2.0).abs() < 1e-6);
        assert!((state.basis_gross_usd - 2.0).abs() < 1e-6);

        // Fair value equals entry ⇒ unrealised PnL ~0.
        assert!((state.daily_unrealised_pnl - 0.0).abs() < 1e-9);
        assert!((state.daily_pnl_total - state.daily_realised_pnl).abs() < 1e-9);
    }

    /// Long open + close with no fees.
    #[test]
    fn long_open_and_close_no_fees() {
        let (cfg, mut state) = mk_state();

        // Long 2 TAO at 100, close at 110.
        state.apply_perp_fill(0, Side::Buy, 2.0, 100.0, 0.0);
        state.apply_perp_fill(0, Side::Sell, 2.0, 110.0, 0.0);
        state.recompute_after_fills(&cfg);

        let expected = (110.0 - 100.0) * 2.0;

        assert!((state.q_global_tao - 0.0).abs() < 1e-9);
        assert!((state.daily_realised_pnl - expected).abs() < 1e-9);
        assert!((state.daily_unrealised_pnl - 0.0).abs() < 1e-9);
        assert!((state.daily_pnl_total - expected).abs() < 1e-9);
    }

    /// Short open + close with symmetric fees on entry and exit.
    #[test]
    fn short_open_and_close_with_fees() {
        let (cfg, mut state) = mk_state();

        let size = 3.0;
        let entry = 110.0;
        let exit = 100.0;
        let fee_open_bps = 5.0; // 5 bps on entry
        let fee_close_bps = 5.0; // 5 bps on exit

        // Open short, then close.
        state.apply_perp_fill(0, Side::Sell, size, entry, fee_open_bps);
        state.apply_perp_fill(0, Side::Buy, size, exit, fee_close_bps);
        state.recompute_after_fills(&cfg);

        let gross_pnl = (entry - exit) * size;

        let fee_open = (fee_open_bps / 10_000.0) * entry * size;
        let fee_close = (fee_close_bps / 10_000.0) * exit * size;
        let expected_net = gross_pnl - (fee_open + fee_close);

        assert!((state.q_global_tao - 0.0).abs() < 1e-9);
        assert!((state.daily_realised_pnl - expected_net).abs() < 1e-6);
        assert!((state.daily_unrealised_pnl - 0.0).abs() < 1e-9);
        assert!((state.daily_pnl_total - expected_net).abs() < 1e-6);
    }
}
