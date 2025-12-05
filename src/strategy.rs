// src/strategy.rs
//
// High-level strategy runner around the Paraphina core engine.
// This wires together:
//
//   - Engine (fair value, vols, risk, inventory),
//   - MM quote engine,
//   - Hedge engine,
//   - Execution gateway (real or synthetic),
//   - Telemetry sink (file / noop).
//
// `main.rs` simply constructs a `StrategyRunner` and calls
// `run_simulation(num_ticks)`.

use crate::config::Config;
use crate::engine::Engine;
use crate::gateway::ExecutionGateway;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::logging::EventSink;
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::GlobalState;
use crate::toxicity::update_toxicity_and_health;
use crate::types::{OrderIntent, Side, TimestampMs};

/// High-level strategy runner.
///
/// `'a` is the lifetime of the shared `Config` reference.
pub struct StrategyRunner<'a, G, S>
where
    G: ExecutionGateway,
    S: EventSink,
{
    pub cfg: &'a Config,
    pub engine: Engine,
    pub state: GlobalState,
    pub gateway: G,
    pub sink: S,
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
        Self {
            cfg,
            engine,
            state,
            gateway,
            sink,
        }
    }

    /// Run a simple synthetic simulation for `num_ticks` ticks.
    ///
    /// This is the same logic you were previously running in `main.rs`,
    /// now encapsulated here and wired to the logging sink.
    pub fn run_simulation(&mut self, num_ticks: u64) {
        let mut now_ms: TimestampMs = 0;
        let dt_ms: TimestampMs = 1_000;

        for tick in 0..num_ticks {
            println!("\n================ Tick {} =================", tick);

            // 1) Seed synthetic mids / books and run the core engine tick.
            self.engine.seed_dummy_mids(&mut self.state, now_ms);
            self.engine.main_tick(&mut self.state, now_ms);

            // Inject the initial synthetic position (demo only).
            if tick == 0 && self.state.q_global_tao.abs() < 1e-9 {
                self.inject_initial_position();
            }

            // 2) Update toxicity / venue health.
            update_toxicity_and_health(&mut self.state, self.cfg);

            // 3) Print global snapshot.
            self.print_state_snapshot();

            // 4) Compute MM quotes & hedge plan.
            let mm_quotes = compute_mm_quotes(self.cfg, &self.state);

            println!("\nPer-venue quotes:");
            for q in &mm_quotes {
                println!(
                    "  {:<10}: bid={:?}, ask={:?}",
                    q.venue_id, q.bid, q.ask
                );
            }

            let hedge_plan = compute_hedge_plan(self.cfg, &self.state);
            match &hedge_plan {
                Some(plan) => {
                    println!(
                        "\nHedge plan: desired_delta={:.4}, allocations={}",
                        plan.desired_delta,
                        plan.allocations.len()
                    );
                }
                None => println!("\nHedge plan: none"),
            }

            // 5) Convert to abstract order intents.
            let mut all_intents: Vec<OrderIntent> =
                mm_quotes_to_order_intents(&mm_quotes);
            if let Some(plan) = hedge_plan {
                let mut hedge_intents = hedge_plan_to_order_intents(&plan);
                all_intents.append(&mut hedge_intents);
            }

            println!("\nBefore MM + hedge fills:");
            self.print_inventory_and_pnl();

            println!("\nOrder intents (abstract):");
            for it in &all_intents {
                println!(
                    "  {:<10} {:?} {:>6.4} @ {:>8.4} ({:?})",
                    it.venue_id, it.side, it.size, it.price, it.purpose
                );
            }

            // 6) Send intents to the execution gateway and get realised fills.
            println!("\nSynthetic fills (SimGateway):");
            let fills =
                self.gateway
                    .process_intents(self.cfg, &mut self.state, &all_intents);

            // 7) Recompute inventory / basis / unrealised PnL after fills.
            self.state.recompute_after_fills(self.cfg);

            println!("\nAfter MM + hedge fills:");
            self.print_inventory_and_pnl();

            // 8) Log everything via the telemetry sink.
            self.sink
                .log_tick(tick, self.cfg, &self.state, &all_intents, &fills);

            // Advance synthetic clock.
            now_ms += dt_ms;
        }
    }

    /// Inject an initial synthetic long position on venue 0.
    /// This mirrors the previous demo behaviour where you start with
    /// a small inventory that the hedge engine can work against.
    fn inject_initial_position(&mut self) {
        if self.cfg.venues.is_empty() {
            return;
        }

        let venue_index = 0usize;
        let size_tao = 5.0_f64;
        let s_t = self.state.fair_value.unwrap_or(self.state.fair_value_prev);
        let entry_price = if s_t > 0.0 { s_t } else { 250.0 };

        println!(
            "Injecting synthetic initial position: venue={} side=Buy size={} @ {:.4}",
            self.cfg.venues[venue_index].id, size_tao, entry_price
        );

        self.state.apply_perp_fill(
            venue_index,
            Side::Buy,
            size_tao,
            entry_price,
            0.0,
        );
        self.state.recompute_after_fills(self.cfg);
    }

    fn print_state_snapshot(&self) {
        let s_t = self.state.fair_value.unwrap_or(self.state.fair_value_prev);

        println!("Fair value S_t: {:.4}", s_t);
        println!("Sigma_eff: {:.6}", self.state.sigma_eff);
        println!(
            "Vol ratio (clipped): {:.4}",
            self.state.vol_ratio_clipped
        );
        println!("Spread_mult: {:.4}", self.state.spread_mult);
        println!("Size_mult: {:.4}", self.state.size_mult);
        println!("Band_mult: {:.4}", self.state.band_mult);
        println!("Global q_t (TAO): {:.4}", self.state.q_global_tao);
        println!(
            "Dollar delta (USD): {:.4}",
            self.state.dollar_delta_usd
        );
        println!("Basis exposure (USD): {:.4}", self.state.basis_usd);
        println!(
            "Basis gross (USD): {:.4}",
            self.state.basis_gross_usd
        );
        println!("Delta limit (USD): {:.4}", self.state.delta_limit_usd);
        println!(
            "Basis warn / hard (USD): {:.4} / {:.4}",
            self.state.basis_limit_warn_usd, self.state.basis_limit_hard_usd
        );
        println!(
            "Daily PnL (realised): {:.4}",
            self.state.daily_realised_pnl
        );
        println!(
            "Daily PnL (unrealised): {:.4}",
            self.state.daily_unrealised_pnl
        );
        println!(
            "Daily PnL total: {:.4}",
            self.state.daily_pnl_total
        );
        println!(
            "Risk regime after tick: {:?}",
            self.state.risk_regime
        );
        println!("Kill switch: {}", self.state.kill_switch);

        println!("\nPer-venue toxicity & status:");
        for v in &self.state.venues {
            println!(
                "  {:<10}: toxicity={:.3}, status={:?}",
                v.id, v.toxicity, v.status
            );
        }
    }

    fn print_inventory_and_pnl(&self) {
        println!("  Global q_t (TAO): {:.4}", self.state.q_global_tao);
        println!(
            "  Dollar delta (USD): {:.4}",
            self.state.dollar_delta_usd
        );
        println!("  Basis exposure (USD): {:.4}", self.state.basis_usd);
        println!(
            "  Daily PnL (realised / unrealised / total): {:.4} / {:.4} / {:.4}",
            self.state.daily_realised_pnl,
            self.state.daily_unrealised_pnl,
            self.state.daily_pnl_total
        );
        println!("  Risk regime: {:?}", self.state.risk_regime);
    }
}
