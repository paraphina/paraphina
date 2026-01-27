// src/bin/replay.rs
//
// Replay a normalized event log and emit telemetry.

use std::env;
use std::path::PathBuf;

use paraphina::actions::ActionIdGenerator;
use paraphina::config::Config;
use paraphina::engine::Engine;
use paraphina::event_log::{read_event_log, EventLogPayload};
use paraphina::execution_events::apply_execution_events;
use paraphina::hedge::hedge_plan_to_order_intents;
use paraphina::mm::compute_mm_quotes;
use paraphina::order_management::plan_mm_order_actions;
use paraphina::state::GlobalState;
use paraphina::telemetry::{ensure_schema_v1, TelemetryBuilder, TelemetryInputs, TelemetrySink};
use paraphina::types::{ExecutionEvent, FillEvent, OrderIntent, TimestampMs};

fn parse_args() -> Result<PathBuf, String> {
    let mut args = env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--event-log" => {
                let val = args.next().ok_or("Missing value for --event-log")?;
                path = Some(PathBuf::from(val));
            }
            "--help" | "-h" => {
                println!("USAGE: replay --event-log <PATH>");
                std::process::exit(0);
            }
            _ => return Err(format!("Unknown argument: {arg}")),
        }
    }
    path.ok_or("Missing required --event-log <PATH>".to_string())
}

fn apply_fill_batch(
    state: &mut GlobalState,
    fills: &[FillEvent],
    now_ms: TimestampMs,
    cfg: &Config,
) {
    for fill in fills {
        state.apply_fill_event(fill, now_ms, cfg);
    }
}

#[allow(clippy::too_many_arguments)]
fn log_telemetry(
    builder: &mut TelemetryBuilder,
    telemetry: &mut TelemetrySink,
    state: &GlobalState,
    tick: u64,
    now_ms: TimestampMs,
    intents: &[OrderIntent],
    exec_events: &[ExecutionEvent],
    fills: &[FillEvent],
    last_exit_intent: Option<&OrderIntent>,
    last_hedge_intent: Option<&OrderIntent>,
    cfg: &Config,
) {
    let mut record = builder.build_record(TelemetryInputs {
        cfg,
        state,
        tick,
        now_ms,
        intents,
        exec_events,
        fills,
        last_exit_intent,
        last_hedge_intent,
        kill_event: None,
        shadow_mode: false,
        execution_mode: "replay",
        reconcile_drift: &[],
        max_orders_per_tick: 200,
    });
    ensure_schema_v1(&mut record);
    telemetry.log_json(&record);
}

fn main() -> Result<(), String> {
    let event_log_path = parse_args()?;
    let records = read_event_log(&event_log_path).map_err(|e| e.to_string())?;
    let has_live_events = records
        .iter()
        .any(|record| matches!(record.event, EventLogPayload::Tick))
        || {
            #[cfg(feature = "live")]
            {
                records.iter().any(|record| {
                    matches!(
                        record.event,
                        EventLogPayload::MarketData(_)
                            | EventLogPayload::Account(_)
                            | EventLogPayload::LiveExecution(_)
                            | EventLogPayload::OrderSnapshot(_)
                    )
                })
            }
            #[cfg(not(feature = "live"))]
            {
                false
            }
        };

    if has_live_events {
        #[cfg(feature = "live")]
        {
            let telemetry_path = std::env::var("PARAPHINA_TELEMETRY_PATH")
                .ok()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    event_log_path
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new("."))
                        .join("telemetry_replay.jsonl")
                });
            let cfg = Config::default();
            let _ = paraphina::live::runner::replay_event_log(
                &cfg,
                &event_log_path,
                &telemetry_path,
                None,
            );
            println!(
                "replay | live_event_log=true | telemetry_path={}",
                telemetry_path.display()
            );
            return Ok(());
        }
        #[cfg(not(feature = "live"))]
        {
            return Err("Live event log replay requires --features live".to_string());
        }
    }

    let cfg = Config::default();
    let engine = Engine::new(&cfg);
    let mut state = GlobalState::new(&cfg);
    let mut telemetry = TelemetrySink::from_env();
    let mut telemetry_builder = TelemetryBuilder::new(&cfg);

    let mut current_tick: Option<u64> = None;
    let mut current_now_ms: TimestampMs = 0;
    let mut book_events: Vec<ExecutionEvent> = Vec::new();
    let mut mm_events: Vec<ExecutionEvent> = Vec::new();
    let mut exit_events: Vec<ExecutionEvent> = Vec::new();
    let mut hedge_events: Vec<ExecutionEvent> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn flush_tick(
        cfg: &Config,
        engine: &Engine,
        state: &mut GlobalState,
        telemetry: &mut TelemetrySink,
        telemetry_builder: &mut TelemetryBuilder,
        tick: u64,
        now_ms: TimestampMs,
        book_events: &mut Vec<ExecutionEvent>,
        mm_events: &mut Vec<ExecutionEvent>,
        exit_events: &mut Vec<ExecutionEvent>,
        hedge_events: &mut Vec<ExecutionEvent>,
    ) {
        // Re-seed synthetic books for deterministic vol/return updates.
        engine.seed_dummy_mids(state, now_ms);
        // Apply book updates from the log (should match seeded values).
        let _ = apply_execution_events(state, book_events, now_ms);
        engine.main_tick_without_risk(state, now_ms);
        engine.update_risk_limits_and_regime(state);

        let mut last_exit_intent: Option<OrderIntent> = None;
        let mut last_hedge_intent: Option<OrderIntent> = None;
        let mut intents: Vec<OrderIntent> = Vec::new();

        let mm_quotes = compute_mm_quotes(cfg, state);
        let mut action_id_gen = ActionIdGenerator::new(tick);
        let mm_plan = plan_mm_order_actions(cfg, state, &mm_quotes, now_ms, &mut action_id_gen);
        intents.extend(mm_plan.intents.clone());

        let mm_fills = apply_execution_events(state, mm_events, now_ms);
        if !mm_fills.is_empty() {
            apply_fill_batch(state, &mm_fills, now_ms, cfg);
            state.recompute_after_fills(cfg);
        }

        if !exit_events.is_empty() && cfg.exit.enabled {
            let exit_intents = paraphina::exit::compute_exit_intents(cfg, state, now_ms);
            last_exit_intent = exit_intents.first().cloned();
            intents.extend(exit_intents);
        }

        let exit_fills = apply_execution_events(state, exit_events, now_ms);
        if !exit_fills.is_empty() {
            apply_fill_batch(state, &exit_fills, now_ms, cfg);
            state.recompute_after_fills(cfg);
        }

        if !hedge_events.is_empty() {
            if let Some(plan) = paraphina::hedge::compute_hedge_plan(cfg, state, now_ms) {
                let hedge_intents = hedge_plan_to_order_intents(&plan);
                last_hedge_intent = hedge_intents.first().cloned();
                intents.extend(hedge_intents);
            }
        }

        let hedge_fills = apply_execution_events(state, hedge_events, now_ms);
        if !hedge_fills.is_empty() {
            apply_fill_batch(state, &hedge_fills, now_ms, cfg);
            state.recompute_after_fills(cfg);
        }

        let mut exec_events = Vec::new();
        exec_events.extend(mm_events.iter().cloned());
        exec_events.extend(exit_events.iter().cloned());
        exec_events.extend(hedge_events.iter().cloned());
        let fills = exec_events
            .iter()
            .filter_map(|e| match e {
                ExecutionEvent::Fill(fill) => Some(fill.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        log_telemetry(
            telemetry_builder,
            telemetry,
            state,
            tick,
            now_ms,
            &intents,
            &exec_events,
            &fills,
            last_exit_intent.as_ref(),
            last_hedge_intent.as_ref(),
            cfg,
        );
        book_events.clear();
        mm_events.clear();
        exit_events.clear();
        hedge_events.clear();
    }

    for record in records {
        if current_tick.is_none() {
            current_tick = Some(record.tick);
            current_now_ms = record.now_ms;
        }
        if Some(record.tick) != current_tick {
            flush_tick(
                &cfg,
                &engine,
                &mut state,
                &mut telemetry,
                &mut telemetry_builder,
                current_tick.unwrap(),
                current_now_ms,
                &mut book_events,
                &mut mm_events,
                &mut exit_events,
                &mut hedge_events,
            );
            current_tick = Some(record.tick);
            current_now_ms = record.now_ms;
        }

        if let Some(event) = record.to_execution_event() {
            match record.phase.as_str() {
                "book" => book_events.push(event),
                "mm" => mm_events.push(event),
                "exit" => exit_events.push(event),
                "hedge" => hedge_events.push(event),
                _ => {}
            }
        }
    }

    if let Some(tick) = current_tick {
        flush_tick(
            &cfg,
            &engine,
            &mut state,
            &mut telemetry,
            &mut telemetry_builder,
            tick,
            current_now_ms,
            &mut book_events,
            &mut mm_events,
            &mut exit_events,
            &mut hedge_events,
        );
    }

    Ok(())
}
