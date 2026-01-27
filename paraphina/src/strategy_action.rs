// src/strategy_action.rs
//
// Action-based strategy runner around the Paraphina core engine.
//
// Ordering (spec):
//   1) Engine tick (FV/vol scalars/toxicity/risk)
//   2) MM intents -> actions -> fills -> recompute
//   3) Exit intents -> actions -> fills -> recompute
//   4) Hedge intents -> actions -> fills -> recompute
//
// main.rs constructs StrategyRunner and calls run_simulation(num_ticks).

use crate::actions::{intents_to_actions, Action, ActionBatch, ActionIdGenerator};
use crate::config::Config;
use crate::engine::Engine;
#[cfg(feature = "event_log")]
use crate::event_log::{EventLogRecord, EventLogWriter, SerializableExecutionEvent};
use crate::exit;
use crate::fill_batcher::FillBatcher;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::io::{ActionResult, Gateway};
use crate::logging::EventSink;
use crate::loop_scheduler::LoopScheduler;
use crate::mm::compute_mm_quotes;
use crate::order_management::{plan_mm_order_actions, MmOrderManagementPlan};
use crate::rl::observation::Observation;
use crate::state::{GlobalState, KillEvent, MmOpenOrder, OpenOrderRecord};
use crate::telemetry::{
    ensure_schema_v1, TelemetryBuilder, TelemetryInputs, TelemetrySink, SCHEMA_VERSION,
};
use crate::types::{ExecutionEvent, FillEvent, OrderIntent, Side, TimestampMs};
use crate::{
    RewardComponents, RewardWeights, ACTION_VERSION, HEURISTIC_POLICY_VERSION, OBS_VERSION,
};

use serde_json::json;

/// High-level action-based strategy runner.
pub struct StrategyRunner<'a, S>
where
    S: EventSink,
{
    pub cfg: &'a Config,
    pub engine: Engine<'a>,
    pub state: GlobalState,
    pub gateway: Gateway,
    pub sink: S,
    pub telemetry: TelemetrySink,
    pub telemetry_builder: TelemetryBuilder,
    pub fill_batcher: FillBatcher,
    #[cfg(feature = "event_log")]
    event_log: Option<EventLogWriter>,
    rl_bridge_enabled: bool,
    rl_prev_pnl: f64,
    rl_peak_pnl: f64,

    // research-harness helpers
    seed: Option<u64>,
    verbosity: u8,
}

fn build_cancel_all_intents(cfg: &Config) -> Vec<OrderIntent> {
    cfg.venues
        .iter()
        .enumerate()
        .map(|(venue_index, venue)| {
            OrderIntent::CancelAll(crate::types::CancelAllOrderIntent {
                venue_index: Some(venue_index),
                venue_id: Some(venue.id_arc.clone()),
            })
        })
        .collect()
}

impl<'a, S> StrategyRunner<'a, S>
where
    S: EventSink,
{
    /// Construct a new strategy runner with a given gateway and event sink.
    pub fn new(cfg: &'a Config, gateway: Gateway, sink: S) -> Self {
        let engine = Engine::new(cfg);
        let state = GlobalState::new(cfg);
        let telemetry = TelemetrySink::from_env();
        let telemetry_builder = TelemetryBuilder::new(cfg);
        let fill_batcher = FillBatcher::new(cfg.fill_agg_interval_ms);
        let rl_bridge_enabled = std::env::var("PARAPHINA_RL_TELEMETRY_BRIDGE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Self {
            cfg,
            engine,
            state,
            gateway,
            sink,
            telemetry,
            telemetry_builder,
            fill_batcher,
            #[cfg(feature = "event_log")]
            event_log: EventLogWriter::from_env(),
            rl_bridge_enabled,
            rl_prev_pnl: 0.0,
            rl_peak_pnl: 0.0,
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

    fn apply_fill_batch(&mut self, fills: &[FillEvent], now_ms: TimestampMs) {
        for fill in fills {
            self.state.apply_fill_event(fill, now_ms, self.cfg);
        }
    }

    fn apply_action_results(&mut self, results: &[(Action, ActionResult)], now_ms: TimestampMs) {
        for (action, result) in results {
            match action {
                Action::PlaceOrder(place) => match result {
                    ActionResult::Filled(_)
                    | ActionResult::Placed { .. }
                    | ActionResult::Acknowledged => {
                        let order_id = match result {
                            ActionResult::Filled(fill) => fill
                                .order_id
                                .clone()
                                .unwrap_or_else(|| place.client_order_id.clone()),
                            ActionResult::Placed { order_id } => order_id.clone(),
                            ActionResult::Acknowledged => place.client_order_id.clone(),
                            _ => place.client_order_id.clone(),
                        };
                        if let Some(v) = self.state.venues.get_mut(place.venue_index) {
                            v.upsert_open_order(OpenOrderRecord {
                                order_id: order_id.clone(),
                                client_order_id: Some(place.client_order_id.clone()),
                                side: place.side,
                                price: place.price,
                                size: place.size,
                                remaining: place.size,
                                timestamp_ms: now_ms,
                                purpose: place.purpose,
                                time_in_force: Some(place.time_in_force),
                                post_only: Some(place.post_only),
                                reduce_only: Some(place.reduce_only),
                            });
                            let order = MmOpenOrder {
                                price: place.price,
                                size: place.size,
                                timestamp_ms: now_ms,
                                order_id: order_id.clone(),
                            };
                            if place.purpose == crate::types::OrderPurpose::Mm {
                                match place.side {
                                    Side::Buy => v.mm_open_bid = Some(order),
                                    Side::Sell => v.mm_open_ask = Some(order),
                                }
                            }
                        }
                    }
                    _ => {}
                },
                Action::CancelOrder(cancel) => match result {
                    ActionResult::Cancelled { .. } | ActionResult::Acknowledged => {
                        if let Some(v) = self.state.venues.get_mut(cancel.venue_index) {
                            v.remove_open_order(&cancel.order_id);
                        }
                    }
                    _ => {}
                },
                Action::CancelAll(cancel_all) => match cancel_all.venue_index {
                    Some(venue_index) => {
                        if let Some(v) = self.state.venues.get_mut(venue_index) {
                            v.clear_open_orders();
                        }
                    }
                    None => {
                        for v in &mut self.state.venues {
                            v.clear_open_orders();
                        }
                    }
                },
                Action::SetKillSwitch(set) => {
                    if set.activate {
                        self.state.kill_switch = true;
                        self.state.kill_reason = set.reason;
                    }
                }
                Action::Log(_) => {}
            }
        }
    }

    fn action_results_to_events(results: &[(Action, ActionResult)]) -> Vec<ExecutionEvent> {
        let mut events = Vec::new();
        for (action, result) in results {
            match action {
                Action::PlaceOrder(place) => match result {
                    ActionResult::Filled(fill) => {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: place.venue_index,
                            venue_id: place.venue_id.as_str().into(),
                            order_id: place.client_order_id.clone(),
                            client_order_id: Some(place.client_order_id.clone()),
                            seq: None,
                            side: Some(place.side),
                            price: Some(place.price),
                            size: Some(place.size),
                            purpose: Some(place.purpose),
                        }));
                        events.push(ExecutionEvent::Fill(fill.clone()));
                    }
                    ActionResult::Placed { order_id } => {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: place.venue_index,
                            venue_id: place.venue_id.as_str().into(),
                            order_id: order_id.clone(),
                            client_order_id: Some(place.client_order_id.clone()),
                            seq: None,
                            side: Some(place.side),
                            price: Some(place.price),
                            size: Some(place.size),
                            purpose: Some(place.purpose),
                        }));
                    }
                    ActionResult::Rejected { reason } | ActionResult::Error { message: reason } => {
                        events.push(ExecutionEvent::OrderReject(crate::types::OrderReject {
                            venue_index: place.venue_index,
                            venue_id: place.venue_id.as_str().into(),
                            order_id: Some(place.client_order_id.clone()),
                            client_order_id: Some(place.client_order_id.clone()),
                            seq: None,
                            reason: reason.clone(),
                        }));
                    }
                    _ => {}
                },
                Action::CancelOrder(cancel) => match result {
                    ActionResult::Cancelled { order_id } => {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: cancel.venue_index,
                            venue_id: cancel.venue_id.as_str().into(),
                            order_id: order_id.clone(),
                            client_order_id: None,
                            seq: None,
                            side: None,
                            price: None,
                            size: None,
                            purpose: None,
                        }));
                    }
                    ActionResult::Rejected { reason } | ActionResult::Error { message: reason } => {
                        events.push(ExecutionEvent::OrderReject(crate::types::OrderReject {
                            venue_index: cancel.venue_index,
                            venue_id: cancel.venue_id.as_str().into(),
                            order_id: Some(cancel.order_id.clone()),
                            client_order_id: None,
                            seq: None,
                            reason: reason.clone(),
                        }));
                    }
                    _ => {}
                },
                Action::CancelAll(cancel_all) => {
                    if let ActionResult::CancelledAll { .. } = result {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: cancel_all.venue_index.unwrap_or(0),
                            venue_id: "all".into(),
                            order_id: "cancel_all".to_string(),
                            client_order_id: None,
                            seq: None,
                            side: None,
                            price: None,
                            size: None,
                            purpose: None,
                        }));
                    }
                }
                _ => {}
            }
        }
        events
    }

    #[cfg(feature = "event_log")]
    fn record_action_results(
        &mut self,
        phase: &str,
        tick: u64,
        now_ms: TimestampMs,
        results: &[(Action, ActionResult)],
    ) {
        let mut events = Vec::new();
        for (action, result) in results {
            match action {
                Action::PlaceOrder(place) => match result {
                    ActionResult::Filled(fill) => {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: place.venue_index,
                            venue_id: place.venue_id.as_str().into(),
                            order_id: place.client_order_id.clone(),
                            client_order_id: Some(place.client_order_id.clone()),
                            seq: None,
                            side: Some(place.side),
                            price: Some(place.price),
                            size: Some(place.size),
                            purpose: Some(place.purpose),
                        }));
                        events.push(ExecutionEvent::Fill(fill.clone()));
                    }
                    ActionResult::Placed { order_id } => {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: place.venue_index,
                            venue_id: place.venue_id.as_str().into(),
                            order_id: order_id.clone(),
                            client_order_id: Some(place.client_order_id.clone()),
                            seq: None,
                            side: Some(place.side),
                            price: Some(place.price),
                            size: Some(place.size),
                            purpose: Some(place.purpose),
                        }));
                    }
                    ActionResult::Rejected { reason } | ActionResult::Error { message: reason } => {
                        events.push(ExecutionEvent::OrderReject(crate::types::OrderReject {
                            venue_index: place.venue_index,
                            venue_id: place.venue_id.as_str().into(),
                            order_id: Some(place.client_order_id.clone()),
                            client_order_id: Some(place.client_order_id.clone()),
                            seq: None,
                            reason: reason.clone(),
                        }));
                    }
                    _ => {}
                },
                Action::CancelOrder(cancel) => match result {
                    ActionResult::Cancelled { .. } | ActionResult::Acknowledged => {
                        events.push(ExecutionEvent::OrderAck(crate::types::OrderAck {
                            venue_index: cancel.venue_index,
                            venue_id: cancel.venue_id.as_str().into(),
                            order_id: cancel.order_id.clone(),
                            client_order_id: None,
                            seq: None,
                            side: None,
                            price: None,
                            size: None,
                            purpose: None,
                        }));
                    }
                    ActionResult::Rejected { reason } | ActionResult::Error { message: reason } => {
                        events.push(ExecutionEvent::OrderReject(crate::types::OrderReject {
                            venue_index: cancel.venue_index,
                            venue_id: cancel.venue_id.as_str().into(),
                            order_id: Some(cancel.order_id.clone()),
                            client_order_id: None,
                            seq: None,
                            reason: reason.clone(),
                        }));
                    }
                    _ => {}
                },
                Action::CancelAll(_) => {}
                Action::SetKillSwitch(_) | Action::Log(_) => {}
            }
        }

        if !events.is_empty() {
            self.record_events(phase, tick, now_ms, &events);
        }
    }

    #[cfg(feature = "event_log")]
    fn record_events(
        &mut self,
        phase: &str,
        tick: u64,
        now_ms: TimestampMs,
        events: &[ExecutionEvent],
    ) {
        if let Some(writer) = self.event_log.as_mut() {
            for event in events {
                let record = EventLogRecord {
                    tick,
                    now_ms,
                    phase: phase.to_string(),
                    event: crate::event_log::EventLogPayload::Execution(
                        SerializableExecutionEvent::from(event),
                    ),
                };
                writer.log_event(&record);
            }
        }
    }

    /// Run synthetic simulation for `num_ticks`.
    pub fn run_simulation(&mut self, num_ticks: u64) {
        // Deterministic timebase offset (keeps runs stable given same seed).
        let base_ms: TimestampMs = self.seed.map(|s| (s % 10_000) as i64).unwrap_or(0);

        // Apply initial position once (if configured).
        self.inject_initial_position();

        // Align batcher so first tick flushes in simulation.
        self.fill_batcher
            .set_last_flush_ms(base_ms - self.cfg.fill_agg_interval_ms);

        let mut scheduler = LoopScheduler::new(
            base_ms,
            self.cfg.main_loop_interval_ms,
            self.cfg.hedge_loop_interval_ms,
            self.cfg.risk_loop_interval_ms,
        );

        let kill_best_effort = std::env::var("PARAPHINA_KILL_BEST_EFFORT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        for tick in 0..num_ticks {
            let now_ms: TimestampMs = scheduler.advance_main();
            let mut action_id_gen = ActionIdGenerator::new(tick);

            // 1) Seed synthetic books + engine tick.
            self.engine.seed_dummy_mids(&mut self.state, now_ms);
            let sim_snapshots = self.state.synthesize_account_snapshots(self.cfg, now_ms);
            self.state.apply_sim_account_snapshots(&sim_snapshots);
            #[cfg(feature = "event_log")]
            {
                let mut book_events = Vec::new();
                for (idx, v) in self.state.venues.iter().enumerate() {
                    if let (Some(mid), Some(spread)) = (v.mid, v.spread) {
                        book_events.push(ExecutionEvent::BookUpdate(crate::types::BookUpdate {
                            venue_index: idx,
                            venue_id: v.id.clone(),
                            mid,
                            spread,
                            depth_near_mid: v.depth_near_mid,
                            timestamp_ms: now_ms,
                        }));
                    }
                }
                self.record_events("book", tick, now_ms, &book_events);
            }
            self.engine.main_tick_without_risk(&mut self.state, now_ms);

            // Risk loop cadence: run after main_tick_without_risk so it sees the latest
            // fair value / vol / toxicity snapshot before MM quoting.
            if scheduler.risk_due(now_ms) {
                self.engine.update_risk_limits_and_regime(&mut self.state);
                scheduler.mark_risk_ran();
            }

            let mut all_intents: Vec<OrderIntent> = Vec::new();
            let mut last_exit_intent: Option<OrderIntent> = None;
            let mut last_hedge_intent: Option<OrderIntent> = None;
            let mut all_fills = Vec::new();
            let mut exec_events: Vec<ExecutionEvent> = Vec::new();
            let mut kill_event: Option<KillEvent> = None;

            let kill_transition = self.state.mark_kill_handled(tick);
            if self.state.kill_switch {
                if kill_transition {
                    let cancel_intents = build_cancel_all_intents(self.cfg);
                    if !cancel_intents.is_empty() {
                        all_intents.extend(cancel_intents.iter().cloned());
                        let mut cancel_batch =
                            ActionBatch::new(now_ms, tick, &self.cfg.version).with_seed(self.seed);
                        for action in intents_to_actions(&cancel_intents, &mut action_id_gen) {
                            cancel_batch.push(action);
                        }
                        let cancel_results = self.gateway.execute_batch(
                            self.cfg,
                            &mut self.state,
                            &cancel_batch,
                            now_ms,
                        );
                        self.apply_action_results(&cancel_results.results, now_ms);
                        exec_events.extend(Self::action_results_to_events(&cancel_results.results));
                        #[cfg(feature = "event_log")]
                        self.record_action_results("kill", tick, now_ms, &cancel_results.results);
                    }

                    if kill_best_effort {
                        if let Some(intent) = self
                            .state
                            .best_effort_kill_intent_exit_first(self.cfg, tick)
                        {
                            all_intents.push(intent.clone());
                            let mut best_batch = ActionBatch::new(now_ms, tick, &self.cfg.version)
                                .with_seed(self.seed);
                            for action in intents_to_actions(&[intent], &mut action_id_gen) {
                                best_batch.push(action);
                            }
                            let best_results = self.gateway.execute_batch(
                                self.cfg,
                                &mut self.state,
                                &best_batch,
                                now_ms,
                            );
                            self.apply_action_results(&best_results.results, now_ms);
                            exec_events
                                .extend(Self::action_results_to_events(&best_results.results));
                            #[cfg(feature = "event_log")]
                            self.record_action_results("kill", tick, now_ms, &best_results.results);
                            if !best_results.fills.is_empty() {
                                self.apply_fill_batch(&best_results.fills, now_ms);
                                all_fills.extend(best_results.fills);
                                self.state.recompute_after_fills(self.cfg);
                            }
                        }
                    }

                    kill_event = Some(self.state.build_kill_event(tick, self.cfg));
                }
            } else {
                // 2) Market-making
                let mm_quotes = compute_mm_quotes(self.cfg, &self.state);
                let mm_plan: MmOrderManagementPlan = plan_mm_order_actions(
                    self.cfg,
                    &self.state,
                    &mm_quotes,
                    now_ms,
                    &mut action_id_gen,
                );
                all_intents.extend(mm_plan.intents.iter().cloned());

                let mut mm_batch =
                    ActionBatch::new(now_ms, tick, &self.cfg.version).with_seed(self.seed);
                for action in intents_to_actions(&mm_plan.intents, &mut action_id_gen) {
                    mm_batch.push(action);
                }
                let mm_results =
                    self.gateway
                        .execute_batch(self.cfg, &mut self.state, &mm_batch, now_ms);
                self.apply_action_results(&mm_results.results, now_ms);
                exec_events.extend(Self::action_results_to_events(&mm_results.results));
                #[cfg(feature = "event_log")]
                self.record_action_results("mm", tick, now_ms, &mm_results.results);
                self.fill_batcher.push(now_ms, mm_results.fills);

                if self.fill_batcher.should_flush(now_ms) {
                    let batch_fills = self.fill_batcher.flush(now_ms);
                    self.apply_fill_batch(&batch_fills, now_ms);
                    all_fills.extend(batch_fills);
                    self.state.recompute_after_fills(self.cfg);

                    // 3) Exit engine BEFORE hedge (spec), on post-fill snapshot
                    let exit_intents = if self.cfg.exit.enabled {
                        exit::compute_exit_intents(self.cfg, &self.state, now_ms)
                    } else {
                        Vec::new()
                    };
                    if !exit_intents.is_empty() {
                        last_exit_intent = exit_intents.first().cloned();
                        all_intents.extend(exit_intents.iter().cloned());

                        let mut exit_batch =
                            ActionBatch::new(now_ms, tick, &self.cfg.version).with_seed(self.seed);
                        for action in intents_to_actions(&exit_intents, &mut action_id_gen) {
                            exit_batch.push(action);
                        }
                        let exit_results = self.gateway.execute_batch(
                            self.cfg,
                            &mut self.state,
                            &exit_batch,
                            now_ms,
                        );
                        self.apply_action_results(&exit_results.results, now_ms);
                        exec_events.extend(Self::action_results_to_events(&exit_results.results));
                        #[cfg(feature = "event_log")]
                        self.record_action_results("exit", tick, now_ms, &exit_results.results);
                        let exit_fills = exit_results.fills;

                        self.apply_fill_batch(&exit_fills, now_ms);
                        all_fills.extend(exit_fills);
                        self.state.recompute_after_fills(self.cfg);
                    }

                    // 4) Hedge engine (after exits)
                    let mut hedge_intents: Vec<OrderIntent> = Vec::new();
                    if scheduler.hedge_due(now_ms) {
                        if let Some(plan) = compute_hedge_plan(self.cfg, &self.state, now_ms) {
                            hedge_intents = hedge_plan_to_order_intents(&plan);
                        }
                        scheduler.mark_hedge_ran();
                    }

                    if !hedge_intents.is_empty() {
                        last_hedge_intent = hedge_intents.first().cloned();
                        all_intents.extend(hedge_intents.iter().cloned());

                        let mut hedge_batch =
                            ActionBatch::new(now_ms, tick, &self.cfg.version).with_seed(self.seed);
                        for action in intents_to_actions(&hedge_intents, &mut action_id_gen) {
                            hedge_batch.push(action);
                        }
                        let hedge_results = self.gateway.execute_batch(
                            self.cfg,
                            &mut self.state,
                            &hedge_batch,
                            now_ms,
                        );
                        self.apply_action_results(&hedge_results.results, now_ms);
                        exec_events.extend(Self::action_results_to_events(&hedge_results.results));
                        #[cfg(feature = "event_log")]
                        self.record_action_results("hedge", tick, now_ms, &hedge_results.results);
                        let hedge_fills = hedge_results.fills;

                        self.apply_fill_batch(&hedge_fills, now_ms);
                        all_fills.extend(hedge_fills);
                        self.state.recompute_after_fills(self.cfg);
                    }
                }
            }

            let mut record = self.telemetry_builder.build_record(TelemetryInputs {
                cfg: self.cfg,
                state: &self.state,
                tick,
                now_ms,
                intents: &all_intents,
                exec_events: &exec_events,
                fills: &all_fills,
                last_exit_intent: last_exit_intent.as_ref(),
                last_hedge_intent: last_hedge_intent.as_ref(),
                kill_event: kill_event.as_ref(),
                shadow_mode: false,
                execution_mode: "strategy",
                reconcile_drift: &[],
                max_orders_per_tick: 200,
            });

            if self.rl_bridge_enabled {
                let obs = Observation::from_state(&self.state, self.cfg, now_ms, tick);
                if obs.daily_pnl_total > self.rl_peak_pnl {
                    self.rl_peak_pnl = obs.daily_pnl_total;
                }
                let components =
                    RewardComponents::from_observation(&obs, self.rl_prev_pnl, self.rl_peak_pnl);
                let weights = RewardWeights::default();
                let reward_total = components.compute_reward(&weights);

                let delta_penalty = -weights.lambda_delta * components.delta_ratio;
                let basis_penalty = -weights.lambda_basis * components.basis_ratio;
                let drawdown_penalty = -weights.lambda_drawdown
                    * (components.drawdown_abs / weights.drawdown_budget.max(1.0));
                let tox_penalty = -weights.lambda_toxicity * components.mean_toxicity;
                let turnover_penalty = 0.0; // Not modeled in baseline sim
                let kill_penalty = if components.kill_triggered {
                    weights.kill_penalty
                } else {
                    0.0
                };

                if let serde_json::Value::Object(map) = &mut record {
                    map.insert("obs_version".to_string(), json!(OBS_VERSION));
                    map.insert("action_version".to_string(), json!(ACTION_VERSION));
                    map.insert(
                        "policy_version".to_string(),
                        json!(HEURISTIC_POLICY_VERSION),
                    );
                    map.insert("config_version_id".to_string(), json!(self.cfg.version));
                    map.insert("policy_action_raw".to_string(), serde_json::Value::Null);
                    map.insert("policy_action_applied".to_string(), serde_json::Value::Null);
                    map.insert("reward_total".to_string(), json!(reward_total));
                    map.insert("pnl_delta".to_string(), json!(components.delta_pnl));
                    map.insert("delta_penalty".to_string(), json!(delta_penalty));
                    map.insert("basis_penalty".to_string(), json!(basis_penalty));
                    map.insert("drawdown_penalty".to_string(), json!(drawdown_penalty));
                    map.insert("tox_penalty".to_string(), json!(tox_penalty));
                    map.insert("turnover_penalty".to_string(), json!(turnover_penalty));
                    map.insert("kill_penalty".to_string(), json!(kill_penalty));
                    map.insert("episode_idx".to_string(), json!(0));
                    map.insert("episode_step_idx".to_string(), json!(tick));
                    map.insert(
                        "episode_done".to_string(),
                        json!(self.state.kill_switch || tick + 1 == num_ticks),
                    );
                    map.insert(
                        "done_reason".to_string(),
                        json!(if self.state.kill_switch {
                            "KillSwitch"
                        } else if tick + 1 == num_ticks {
                            "EndOfEpisode"
                        } else {
                            "Running"
                        }),
                    );
                }

                self.rl_prev_pnl = obs.daily_pnl_total;
            }

            ensure_schema_v1(&mut record);
            debug_assert_eq!(
                record["schema_version"].as_i64(),
                Some(SCHEMA_VERSION),
                "telemetry schema_version must be 1"
            );

            self.telemetry.log_json(&record);

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
