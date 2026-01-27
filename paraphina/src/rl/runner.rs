// src/rl/runner.rs
//
// Shadow mode runner for RL-0 (per ROADMAP.md and WHITEPAPER Appendix A.5).
//
// The shadow runner executes two policies in parallel:
// - Baseline policy: executes (current behavior)
// - Shadow policy: proposes in parallel; proposals are logged but NOT executed
//
// This enables:
// - Validating new policies without trading risk
// - Comparing policy decisions via replay
// - Counterfactual evaluation

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::engine::Engine;
use crate::execution_events::apply_execution_events;
use crate::exit;
use crate::gateway::ExecutionGateway;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::logging::EventSink;
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::GlobalState;
use crate::types::{FillEvent, OrderIntent, Side, TimestampMs};

use super::observation::Observation;
use super::policy::{HeuristicPolicy, Policy, PolicyAction};
use super::safety::SafetyLayer;
use super::telemetry::RLTelemetry;

/// Episode termination reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    /// Normal end of episode (max ticks reached).
    EndOfEpisode,
    /// Kill switch triggered.
    KillSwitch,
    /// Manual termination requested.
    Manual,
    /// Error during execution.
    Error,
}

/// Configuration for an RL episode.
#[derive(Debug, Clone)]
pub struct EpisodeConfig {
    /// Random seed for deterministic simulation.
    pub seed: u64,
    /// Episode ID for logging.
    pub episode_id: u64,
    /// Maximum number of ticks to run.
    pub max_ticks: u64,
    /// Enable shadow mode (run shadow policy in parallel).
    pub shadow_mode: bool,
    /// Verbosity level (0=quiet, 1=summary, 2=debug).
    pub verbosity: u8,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            episode_id: 0,
            max_ticks: 1000,
            shadow_mode: false,
            verbosity: 0,
        }
    }
}

impl EpisodeConfig {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_episode_id(mut self, episode_id: u64) -> Self {
        self.episode_id = episode_id;
        self
    }

    pub fn with_max_ticks(mut self, max_ticks: u64) -> Self {
        self.max_ticks = max_ticks;
        self
    }

    pub fn with_shadow_mode(mut self, enabled: bool) -> Self {
        self.shadow_mode = enabled;
        self
    }
}

/// Summary of a completed episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeSummary {
    /// Episode ID.
    pub episode_id: u64,
    /// Seed used.
    pub seed: u64,
    /// Termination reason.
    pub termination_reason: TerminationReason,
    /// Total ticks executed.
    pub total_ticks: u64,
    /// Final realized PnL.
    pub final_realised_pnl: f64,
    /// Final unrealized PnL.
    pub final_unrealised_pnl: f64,
    /// Final total PnL.
    pub final_pnl_total: f64,
    /// Final global inventory (TAO).
    pub final_q_global_tao: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
    /// Peak PnL observed.
    pub peak_pnl: f64,
    /// Kill switch triggered.
    pub kill_switch_triggered: bool,
}

/// Shadow mode runner with deterministic episode mechanics.
///
/// Supports:
/// - Deterministic seeding + episode reset
/// - Baseline policy execution (actual trades)
/// - Shadow policy proposals (logged but not executed)
/// - RL telemetry logging
pub struct ShadowRunner<'a, G, S>
where
    G: ExecutionGateway,
    S: EventSink,
{
    cfg: &'a Config,
    engine: Engine<'a>,
    state: GlobalState,
    gateway: G,
    sink: S,

    // Policies
    baseline_policy: Box<dyn Policy>,
    shadow_policy: Option<Box<dyn Policy>>,

    // RL telemetry
    rl_telemetry: RLTelemetry,

    // Episode state
    episode_config: EpisodeConfig,
    peak_pnl: f64,
}

impl<'a, G, S> ShadowRunner<'a, G, S>
where
    G: ExecutionGateway,
    S: EventSink,
{
    /// Create a new shadow runner with default heuristic baseline.
    pub fn new(cfg: &'a Config, gateway: G, sink: S) -> Self {
        let engine = Engine::new(cfg);
        let state = GlobalState::new(cfg);
        let baseline_policy = Box::new(HeuristicPolicy::new());

        Self {
            cfg,
            engine,
            state,
            gateway,
            sink,
            baseline_policy,
            shadow_policy: None,
            rl_telemetry: RLTelemetry::new(),
            episode_config: EpisodeConfig::default(),
            peak_pnl: 0.0,
        }
    }

    /// Set the baseline policy.
    pub fn with_baseline_policy(mut self, policy: Box<dyn Policy>) -> Self {
        self.baseline_policy = policy;
        self
    }

    /// Set the shadow policy (enables shadow mode).
    pub fn with_shadow_policy(mut self, policy: Box<dyn Policy>) -> Self {
        self.shadow_policy = Some(policy);
        self
    }

    /// Set the RL telemetry sink.
    pub fn with_rl_telemetry(mut self, telemetry: RLTelemetry) -> Self {
        self.rl_telemetry = telemetry;
        self
    }

    /// Reset the runner for a new episode.
    ///
    /// This provides deterministic episode initialization:
    /// - Resets state to initial conditions
    /// - Sets the random seed for reproducibility
    /// - Resets policies for episode start
    pub fn reset_episode(&mut self, seed: u64, episode_id: u64) {
        // Reset state
        self.state = GlobalState::new(self.cfg);
        self.peak_pnl = 0.0;

        // Update episode config
        self.episode_config.seed = seed;
        self.episode_config.episode_id = episode_id;

        // Reset policies
        self.baseline_policy.reset_episode(seed, episode_id);
        if let Some(shadow) = &mut self.shadow_policy {
            shadow.reset_episode(seed, episode_id);
        }

        // Reset telemetry
        self.rl_telemetry.reset_episode(episode_id, seed);

        // Apply initial position
        self.inject_initial_position();
    }

    /// Run a complete episode.
    pub fn run_episode(&mut self, config: EpisodeConfig) -> EpisodeSummary {
        self.episode_config = config.clone();
        self.reset_episode(config.seed, config.episode_id);

        // Log episode start
        let base_ms: TimestampMs = (config.seed % 10_000) as i64;
        self.rl_telemetry
            .log_episode_start(config.episode_id, config.seed, base_ms);

        let dt_ms: TimestampMs = 1_000;
        let mut termination_reason = TerminationReason::EndOfEpisode;
        let mut ticks_executed: u64 = 0;

        for tick in 0..config.max_ticks {
            let now_ms: TimestampMs = base_ms + (tick as i64) * dt_ms;
            ticks_executed = tick + 1;

            // Run one tick
            let should_terminate = self.tick(now_ms, tick, &config);

            // Update peak PnL
            if self.state.daily_pnl_total > self.peak_pnl {
                self.peak_pnl = self.state.daily_pnl_total;
            }

            if should_terminate {
                termination_reason = if self.state.kill_switch {
                    TerminationReason::KillSwitch
                } else {
                    TerminationReason::Manual
                };
                break;
            }
        }

        let max_drawdown = self.peak_pnl - self.state.daily_pnl_total;

        // Log episode end
        let end_ms = base_ms + (ticks_executed as i64) * dt_ms;
        self.rl_telemetry.log_episode_end(
            config.episode_id,
            config.seed,
            end_ms,
            termination_reason,
            self.state.daily_pnl_total,
            ticks_executed,
        );

        // Print summary
        self.print_episode_summary(termination_reason);

        EpisodeSummary {
            episode_id: config.episode_id,
            seed: config.seed,
            termination_reason,
            total_ticks: ticks_executed,
            final_realised_pnl: self.state.daily_realised_pnl,
            final_unrealised_pnl: self.state.daily_unrealised_pnl,
            final_pnl_total: self.state.daily_pnl_total,
            final_q_global_tao: self.state.q_global_tao,
            max_drawdown,
            peak_pnl: self.peak_pnl,
            kill_switch_triggered: self.state.kill_switch,
        }
    }

    /// Execute a single tick.
    ///
    /// Returns true if the episode should terminate.
    fn tick(&mut self, now_ms: TimestampMs, tick_index: u64, config: &EpisodeConfig) -> bool {
        // 1) Seed synthetic books + engine tick
        self.engine.seed_dummy_mids(&mut self.state, now_ms);
        self.engine.main_tick(&mut self.state, now_ms);

        // 2) Build observation
        let obs = Observation::from_state(&self.state, self.cfg, now_ms, tick_index);

        // 3) Get baseline policy action
        let baseline_action = self.baseline_policy.act(&obs);
        let safety = SafetyLayer::apply(&baseline_action, obs.venues.len());
        let baseline_applied = safety.applied.clone();

        // 4) Get shadow policy action (if enabled)
        if config.shadow_mode {
            if let Some(shadow) = &self.shadow_policy {
                let shadow_action = shadow.act(&obs);
                let shadow_safety = SafetyLayer::apply(&shadow_action, obs.venues.len());
                // Log shadow action (not executed)
                self.rl_telemetry.log_shadow_tick(
                    &obs,
                    &shadow_action,
                    Some(&shadow_safety),
                    self.cfg.version,
                );
            }
        }

        // 5) Log baseline tick
        self.rl_telemetry.log_tick(
            &obs,
            Some(&baseline_action),
            Some(&safety),
            self.cfg.version,
        );

        // 6) Execute baseline strategy (existing behavior)
        // Note: For HeuristicPolicy, baseline_action is identity, so we just run
        // the existing MM/exit/hedge logic as-is.
        self.execute_strategy(now_ms, tick_index, &baseline_applied);

        // 7) Log to event sink
        self.sink.log_tick(
            tick_index,
            self.cfg,
            &self.state,
            &[], // intents already processed
            &[], // fills logged in execute_strategy
        );

        if config.verbosity >= 1 {
            println!(
                "tick {}: q={:.4} Δ={:.2} basis={:.2} pnl={:.2} regime={:?} kill={}",
                tick_index,
                self.state.q_global_tao,
                self.state.dollar_delta_usd,
                self.state.basis_usd,
                self.state.daily_pnl_total,
                self.state.risk_regime,
                self.state.kill_switch
            );
        }

        // Check termination
        self.state.kill_switch
    }

    /// Execute the strategy for one tick.
    ///
    /// This follows the existing MM -> Exit -> Hedge ordering.
    /// The policy action can modify quotes/sizes, but for HeuristicPolicy
    /// it's an identity transformation.
    fn execute_strategy(
        &mut self,
        now_ms: TimestampMs,
        _tick_index: u64,
        _policy_action: &PolicyAction,
    ) {
        // Skip if kill switch active
        if self.state.kill_switch {
            return;
        }

        let mut all_fills = Vec::new();

        // 1) Market-making
        let mm_quotes = compute_mm_quotes(self.cfg, &self.state);
        let mm_intents = mm_quotes_to_order_intents(&mm_quotes);

        // TODO: Apply policy_action modifiers to mm_intents
        // For now, HeuristicPolicy returns identity, so no modification needed

        let mm_events = self.gateway.process_intents(self.cfg, &mm_intents, now_ms);
        let mm_fills = apply_execution_events(&mut self.state, &mm_events, now_ms);
        self.apply_fills(&mm_fills, now_ms);
        all_fills.extend(mm_fills);
        self.state.recompute_after_fills(self.cfg);

        // 2) Exit engine
        if self.cfg.exit.enabled {
            let exit_intents = exit::compute_exit_intents(self.cfg, &self.state, now_ms);
            if !exit_intents.is_empty() {
                let exit_events = self
                    .gateway
                    .process_intents(self.cfg, &exit_intents, now_ms);
                let exit_fills = apply_execution_events(&mut self.state, &exit_events, now_ms);
                self.apply_fills(&exit_fills, now_ms);
                all_fills.extend(exit_fills);
                self.state.recompute_after_fills(self.cfg);
            }
        }

        // 3) Hedge engine
        let mut hedge_intents: Vec<OrderIntent> = Vec::new();
        if let Some(plan) = compute_hedge_plan(self.cfg, &self.state, now_ms) {
            hedge_intents = hedge_plan_to_order_intents(&plan);
        }

        // TODO: Apply policy_action.hedge_scale and hedge_venue_weights

        if !hedge_intents.is_empty() {
            let hedge_events = self
                .gateway
                .process_intents(self.cfg, &hedge_intents, now_ms);
            let hedge_fills = apply_execution_events(&mut self.state, &hedge_events, now_ms);
            self.apply_fills(&hedge_fills, now_ms);
            all_fills.extend(hedge_fills);
            self.state.recompute_after_fills(self.cfg);
        }
    }

    /// Apply fills to state (single update point for PnL/positions).
    fn apply_fills(&mut self, fills: &[FillEvent], now_ms: TimestampMs) {
        for fill in fills {
            self.state.apply_fill_event(fill, now_ms, self.cfg);
        }
    }

    /// Inject initial position if configured.
    fn inject_initial_position(&mut self) {
        if self.cfg.venues.is_empty() {
            return;
        }

        let q0 = self.cfg.initial_q_tao;
        if q0.abs() < 1e-9 {
            return;
        }

        let venue_index = 0usize;
        let size_tao = q0.abs();
        let side = if q0 > 0.0 { Side::Buy } else { Side::Sell };

        let s_t = self.state.fair_value.unwrap_or(self.state.fair_value_prev);
        let entry_price = if s_t.is_finite() && s_t > 0.0 {
            s_t
        } else {
            250.0
        };

        self.state
            .apply_perp_fill(venue_index, side, size_tao, entry_price, 0.0);
        self.state.recompute_after_fills(self.cfg);
    }

    /// Print episode summary.
    fn print_episode_summary(&self, reason: TerminationReason) {
        let r = self.state.daily_realised_pnl;
        let u = self.state.daily_unrealised_pnl;
        let t = self.state.daily_pnl_total;

        let r_str = if r >= 0.0 {
            format!("+{:.2}", r)
        } else {
            format!("{:.2}", r)
        };
        let u_str = if u >= 0.0 {
            format!("+{:.2}", u)
        } else {
            format!("{:.2}", u)
        };
        let t_str = if t >= 0.0 {
            format!("+{:.2}", t)
        } else {
            format!("{:.2}", t)
        };

        println!();
        println!("=== Episode Summary ===");
        println!("Episode ID: {}", self.episode_config.episode_id);
        println!("Seed: {}", self.episode_config.seed);
        println!("Termination: {:?}", reason);
        println!(
            "Daily PnL (realised / unrealised / total): {} / {} / {}",
            r_str, u_str, t_str
        );
        println!(
            "Kill switch: {}",
            if self.state.kill_switch {
                "true"
            } else {
                "false"
            }
        );
        if self.state.kill_switch {
            println!("Kill reason: {:?}", self.state.kill_reason);
        }
        println!("Risk regime: {:?}", self.state.risk_regime);
    }

    /// Get reference to current state (for testing).
    pub fn state(&self) -> &GlobalState {
        &self.state
    }

    /// Flush telemetry.
    pub fn flush_telemetry(&mut self) {
        self.rl_telemetry.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::SimGateway;
    use crate::logging::NoopSink;
    use crate::rl::policy::NoopPolicy;

    #[test]
    fn test_episode_config_builder() {
        let config = EpisodeConfig::default()
            .with_seed(42)
            .with_episode_id(1)
            .with_max_ticks(500)
            .with_shadow_mode(true);

        assert_eq!(config.seed, 42);
        assert_eq!(config.episode_id, 1);
        assert_eq!(config.max_ticks, 500);
        assert!(config.shadow_mode);
    }

    #[test]
    fn test_shadow_runner_creation() {
        let cfg = Config::default();
        let gateway = SimGateway::new();
        let sink = NoopSink;

        let runner = ShadowRunner::new(&cfg, gateway, sink);
        assert!(!runner.state.kill_switch);
    }

    #[test]
    fn test_shadow_runner_reset_episode() {
        let cfg = Config::default();
        let gateway = SimGateway::new();
        let sink = NoopSink;

        let mut runner = ShadowRunner::new(&cfg, gateway, sink);

        // First episode
        runner.reset_episode(42, 1);
        assert_eq!(runner.episode_config.seed, 42);
        assert_eq!(runner.episode_config.episode_id, 1);

        // Second episode
        runner.reset_episode(123, 2);
        assert_eq!(runner.episode_config.seed, 123);
        assert_eq!(runner.episode_config.episode_id, 2);
    }

    #[test]
    fn test_shadow_runner_determinism() {
        let cfg = Config::default();

        // Run 1
        let gateway1 = SimGateway::new();
        let sink1 = NoopSink;
        let mut runner1 = ShadowRunner::new(&cfg, gateway1, sink1);
        let config1 = EpisodeConfig::default()
            .with_seed(42)
            .with_episode_id(1)
            .with_max_ticks(10);
        let summary1 = runner1.run_episode(config1);

        // Run 2 with same seed
        let gateway2 = SimGateway::new();
        let sink2 = NoopSink;
        let mut runner2 = ShadowRunner::new(&cfg, gateway2, sink2);
        let config2 = EpisodeConfig::default()
            .with_seed(42)
            .with_episode_id(1)
            .with_max_ticks(10);
        let summary2 = runner2.run_episode(config2);

        // Should produce identical results
        assert_eq!(summary1.total_ticks, summary2.total_ticks);
        assert!(
            (summary1.final_pnl_total - summary2.final_pnl_total).abs() < 1e-9,
            "PnL should be identical with same seed"
        );
        assert!(
            (summary1.final_q_global_tao - summary2.final_q_global_tao).abs() < 1e-9,
            "Inventory should be identical with same seed"
        );
    }

    #[test]
    fn test_shadow_runner_with_shadow_policy() {
        let cfg = Config::default();
        let gateway = SimGateway::new();
        let sink = NoopSink;

        let shadow_policy = Box::new(NoopPolicy::new());
        let mut runner = ShadowRunner::new(&cfg, gateway, sink).with_shadow_policy(shadow_policy);

        let config = EpisodeConfig::default()
            .with_seed(42)
            .with_episode_id(1)
            .with_max_ticks(5)
            .with_shadow_mode(true);

        let summary = runner.run_episode(config);

        // Should complete without error
        assert!(summary.total_ticks <= 5);
    }

    #[test]
    fn test_shadow_mode_does_not_affect_execution() {
        let cfg = Config::default();

        // Run without shadow mode
        let gateway1 = SimGateway::new();
        let sink1 = NoopSink;
        let mut runner1 = ShadowRunner::new(&cfg, gateway1, sink1);
        let config1 = EpisodeConfig::default()
            .with_seed(42)
            .with_episode_id(1)
            .with_max_ticks(10)
            .with_shadow_mode(false);
        let summary1 = runner1.run_episode(config1);

        // Run with shadow mode
        let gateway2 = SimGateway::new();
        let sink2 = NoopSink;
        let shadow_policy = Box::new(NoopPolicy::new());
        let mut runner2 =
            ShadowRunner::new(&cfg, gateway2, sink2).with_shadow_policy(shadow_policy);
        let config2 = EpisodeConfig::default()
            .with_seed(42)
            .with_episode_id(1)
            .with_max_ticks(10)
            .with_shadow_mode(true);
        let summary2 = runner2.run_episode(config2);

        // Results should be identical
        assert_eq!(summary1.total_ticks, summary2.total_ticks);
        assert!(
            (summary1.final_pnl_total - summary2.final_pnl_total).abs() < 1e-9,
            "Shadow mode should not affect execution"
        );
        assert!(
            (summary1.final_q_global_tao - summary2.final_q_global_tao).abs() < 1e-9,
            "Shadow mode should not affect inventory"
        );
    }
}
