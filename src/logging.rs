// src/logging.rs
//
// Telemetry / logging sinks for Paraphina.

use std::fs::File;
use std::io::{BufWriter, Write};

use serde::Serialize;

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{FillEvent, OrderIntent};

/// Simple interface for logging one tick of state + activity.
pub trait EventSink {
    fn log_tick(
        &mut self,
        tick: u64,
        cfg: &Config,
        state: &GlobalState,
        intents: &[OrderIntent],
        fills: &[FillEvent],
    );
}

/// A sink that does nothing (useful for tests or when you only
/// care about stdout).
#[derive(Debug, Default)]
pub struct NoopSink;

impl EventSink for NoopSink {
    fn log_tick(
        &mut self,
        _tick: u64,
        _cfg: &Config,
        _state: &GlobalState,
        _intents: &[OrderIntent],
        _fills: &[FillEvent],
    ) {
        // no-op
    }
}

/// File-backed JSONL sink: one JSON record per tick.
pub struct FileSink {
    writer: BufWriter<File>,
}

impl FileSink {
    /// Create a new file sink writing JSONL to `path`.
    pub fn create(path: &str) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

#[derive(Serialize)]
struct TickRecord {
    tick: u64,

    fair_value: f64,
    q_global_tao: f64,
    dollar_delta_usd: f64,
    basis_usd: f64,
    basis_gross_usd: f64,

    daily_realised_pnl: f64,
    daily_unrealised_pnl: f64,
    daily_pnl_total: f64,
}

impl EventSink for FileSink {
    fn log_tick(
        &mut self,
        tick: u64,
        _cfg: & Config,
        state: &GlobalState,
        _intents: &[OrderIntent],
        _fills: &[FillEvent],
    ) {
        let fair_value = state.fair_value.unwrap_or(state.fair_value_prev);

        let record = TickRecord {
            tick,
            fair_value,
            q_global_tao: state.q_global_tao,
            dollar_delta_usd: state.dollar_delta_usd,
            basis_usd: state.basis_usd,
            basis_gross_usd: state.basis_gross_usd,
            daily_realised_pnl: state.daily_realised_pnl,
            daily_unrealised_pnl: state.daily_unrealised_pnl,
            daily_pnl_total: state.daily_pnl_total,
        };

        if let Ok(line) = serde_json::to_string(&record) {
            let _ = writeln!(self.writer, "{line}");
        }
    }
}

/// Forward `EventSink` to the value stored inside a `Box`.
/// This is what makes `Box<dyn EventSink>` satisfy the `EventSink`
/// bound used by `StrategyRunner`.
impl<T> EventSink for Box<T>
where
    T: EventSink + ?Sized,
{
    fn log_tick(
        &mut self,
        tick: u64,
        cfg: &Config,
        state: &GlobalState,
        intents: &[OrderIntent],
        fills: &[FillEvent],
    ) {
        (**self).log_tick(tick, cfg, state, intents, fills);
    }
}
