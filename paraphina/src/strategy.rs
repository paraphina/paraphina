// src/strategy.rs
//
// High-level strategy runner around the Paraphina core engine.
//
// Ordering (spec):
//   1) Engine tick (FV/vol scalars/toxicity/risk)
//   2) MM intents -> fills -> recompute
//   3) Exit intents -> fills -> recompute
//   4) Hedge intents -> fills -> recompute
//
// main.rs constructs StrategyRunner and calls run_simulation(num_ticks).

use crate::config::Config;
use crate::engine::Engine;
use crate::exit;
use crate::gateway::ExecutionGateway;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::logging::EventSink;
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::GlobalState;
use crate::telemetry::TelemetrySink;
use crate::types::{FillEvent, OrderIntent, Side, TimestampMs};

use serde_json::json;

/// High-level strategy runner.
pub struct StrategyRunner<'a, G, S>
where
    G: ExecutionGateway,
    S: EventSink,
{
    pub cfg: &'a Config,
    pub engine: Engine<'a>,
    pub state: GlobalState,
    pub gateway: G,
    pub sink: S,
    pub telemetry: TelemetrySink,

    // research-harness helpers
    seed: Option<u64>,
    verbosity: u8,
}

impl<'a, G, S> StrategyRunner<'a, G, S>
where
    G: ExecutionGateway,
    S: EventSink,
{
    /// Construct a new strategy runner with a given gateway and event sink.
    pub fn new(cfg: &'a Config, gateway: G, sink: S) -> Self {
        let engine = Engine::new(cfg);
        let state = GlobalState::new(cfg);
        let telemetry = TelemetrySink::from_env();

        Self {
            cfg,
            engine,
            state,
            gateway,
            sink,
            telemetry,
            seed: None,
            verbosity: 0,
        }
    }

    /// Optional deterministic seed. Currently used to offset the synthetic timebase.
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }

    /// Verbosity: 0 (quiet) / 1 (summary) / 2 (debug).
    pub fn set_verbosity(&mut self, v: u8) {
        self.verbosity = v;
    }

    /// Record pending markouts for a batch of fills.
    ///
    /// Called immediately after fills are produced to schedule markout evaluations.
    fn record_markouts_for_fills(&mut self, fills: &[FillEvent], now_ms: TimestampMs) {
        let tox_cfg = &self.cfg.toxicity;
        let horizon_ms = tox_cfg.markout_horizon_ms;
        let max_pending = tox_cfg.max_pending_per_venue;

        // Get fair value for recording
        let fair = self
            .state
            .fair_value
            .unwrap_or(self.state.fair_value_prev)
            .max(1.0);

        for fill in fills {
            // Get venue mid at fill time
            let mid = self
                .state
                .venues
                .get(fill.venue_index)
                .and_then(|v| v.mid)
                .unwrap_or(fair);

            self.state
                .record_pending_markout(crate::state::PendingMarkoutRecord {
                    venue_index: fill.venue_index,
                    side: fill.side,
                    size_tao: fill.size,
                    price: fill.price,
                    now_ms,
                    fair,
                    mid,
                    horizon_ms,
                    max_pending,
                });
        }
    }

    /// Run synthetic simulation for `num_ticks`.
    pub fn run_simulation(&mut self, num_ticks: u64) {
        // Deterministic timebase offset (keeps runs stable given same seed).
        let base_ms: TimestampMs = self.seed.map(|s| (s % 10_000) as i64).unwrap_or(0);

        let dt_ms: TimestampMs = 1_000;

        // Apply initial position once (if configured).
        self.inject_initial_position();

        for tick in 0..num_ticks {
            let now_ms: TimestampMs = base_ms + (tick as i64) * dt_ms;

            // 1) Seed synthetic books + engine tick.
            self.engine.seed_dummy_mids(&mut self.state, now_ms);
            self.engine.main_tick(&mut self.state, now_ms);

            // 2) Market-making
            let mm_quotes = compute_mm_quotes(self.cfg, &self.state);
            let mut all_intents: Vec<OrderIntent> = mm_quotes_to_order_intents(&mm_quotes);

            let mut all_fills = Vec::new();

            let mm_fills = self
                .gateway
                .process_intents(self.cfg, &mut self.state, &all_intents);

            // Record pending markouts for MM fills (before extending all_fills)
            self.record_markouts_for_fills(&mm_fills, now_ms);
            all_fills.extend(mm_fills);

            // Recompute after MM fills
            self.state.recompute_after_fills(self.cfg);

            // 3) Exit engine BEFORE hedge (spec)
            let exit_intents = if self.cfg.exit.enabled {
                exit::compute_exit_intents(self.cfg, &self.state, now_ms)
            } else {
                Vec::new()
            };
            if !exit_intents.is_empty() {
                // include in overall intents log
                all_intents.extend(exit_intents.iter().cloned());

                let exit_fills =
                    self.gateway
                        .process_intents(self.cfg, &mut self.state, &exit_intents);

                // Record pending markouts for exit fills
                self.record_markouts_for_fills(&exit_fills, now_ms);
                all_fills.extend(exit_fills);

                self.state.recompute_after_fills(self.cfg);
            }

            // 4) Hedge engine (after exits)
            let mut hedge_intents: Vec<OrderIntent> = Vec::new();
            if let Some(plan) = compute_hedge_plan(self.cfg, &self.state, now_ms) {
                hedge_intents = hedge_plan_to_order_intents(&plan);
            }

            if !hedge_intents.is_empty() {
                all_intents.extend(hedge_intents.iter().cloned());

                let hedge_fills =
                    self.gateway
                        .process_intents(self.cfg, &mut self.state, &hedge_intents);

                // Record pending markouts for hedge fills
                self.record_markouts_for_fills(&hedge_fills, now_ms);
                all_fills.extend(hedge_fills);

                self.state.recompute_after_fills(self.cfg);
            }

            // Telemetry snapshot (must keep schema stable for research tooling)
            // Schema contract: see docs/TELEMETRY_SCHEMA_V1.md and schemas/telemetry_schema_v1.json
            // Required fields: schema_version, t, pnl_realised, pnl_unrealised, pnl_total,
            //                  risk_regime, kill_switch, kill_reason, q_global_tao,
            //                  dollar_delta_usd, basis_usd
            // Optional fields: fv_available, fair_value, sigma_eff,
            //                  healthy_venues_used_count, healthy_venues_used
            self.telemetry.log_json(&json!({
                // Schema version (required for contract validation)
                "schema_version": 1,
                // Required fields
                "t": tick,
                "pnl_realised": self.state.daily_realised_pnl,
                "pnl_unrealised": self.state.daily_unrealised_pnl,
                "pnl_total": self.state.daily_pnl_total,
                "risk_regime": format!("{:?}", self.state.risk_regime),
                "kill_switch": self.state.kill_switch,
                "kill_reason": format!("{:?}", self.state.kill_reason),
                "q_global_tao": self.state.q_global_tao,
                "dollar_delta_usd": self.state.dollar_delta_usd,
                "basis_usd": self.state.basis_usd,
                // Optional fields (Milestone D: FV gating & volatility telemetry)
                "fv_available": self.state.fv_available,
                "fair_value": self.state.fair_value,
                "sigma_eff": self.state.sigma_eff,
                "healthy_venues_used_count": self.state.healthy_venues_used_count,
                "healthy_venues_used": self.state.healthy_venues_used,
            }));

            // Human-readable / structured logs via sink
            self.sink
                .log_tick(tick, self.cfg, &self.state, &all_intents, &all_fills);

            if self.verbosity >= 1 {
                println!(
                    "tick {}: q={:.4} Δ={:.2} basis={:.2} pnl={:.2} regime={:?} kill={}",
                    tick,
                    self.state.q_global_tao,
                    self.state.dollar_delta_usd,
                    self.state.basis_usd,
                    self.state.daily_pnl_total,
                    self.state.risk_regime,
                    self.state.kill_switch
                );
            }

            if self.state.kill_switch {
                println!(
                    "Kill switch active at tick {} - stopping simulation early (reason: {:?}).",
                    tick, self.state.kill_reason
                );
                break;
            }
        }

        // End-of-run summary (format required by batch_runs/metrics.py parse_daily_summary)
        self.print_daily_summary();

        self.telemetry.flush();
    }

    /// Print end-of-run summary in the format expected by batch_runs/metrics.py.
    ///
    /// This is the research contract - do not change the format without versioning.
    fn print_daily_summary(&self) {
        let r = self.state.daily_realised_pnl;
        let u = self.state.daily_unrealised_pnl;
        let t = self.state.daily_pnl_total;

        // Format with sign prefix for clarity
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
        println!("=== Daily Summary ===");
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

        // Use fair value if present, else fallback.
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
}
