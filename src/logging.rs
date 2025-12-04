// src/logging.rs
//
// Telemetry sinks for Paraphina.
// - EventSink: trait used by the strategy runner
// - NoopSink:  discards all events
// - FileSink:  writes one JSON-like line per tick for backtesting / RL

use std::fs::File;
use std::io::{self, BufWriter, Write};

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{FillEvent, OrderIntent};

/// Abstract sink for per-tick telemetry.
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

/// Sink that discards all events.
#[derive(Debug, Default, Clone, Copy)]
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
        // intentionally no-op
    }
}

/// JSONL file sink.
///
/// Each tick is written as a single JSON object on its own line.
/// We keep the payload small and encode the JSON manually to
/// avoid additional dependencies.
pub struct FileSink {
    writer: BufWriter<File>,
}

impl FileSink {
    /// Create a new sink writing to `path`.
    pub fn create(path: &str) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl EventSink for FileSink {
    fn log_tick(
        &mut self,
        tick: u64,
        _cfg: &Config,
        state: &GlobalState,
        intents: &[OrderIntent],
        fills: &[FillEvent],
    ) {
        // Minimal JSON; focuses on core risk metrics + counts.
        let fair_value = state.fair_value.unwrap_or(state.fair_value_prev);

        let line = format!(
            "{{\
                \"tick\":{},\
                \"fair_value\":{},\
                \"q_global_tao\":{},\
                \"dollar_delta_usd\":{},\
                \"basis_usd\":{},\
                \"basis_gross_usd\":{},\
                \"daily_realised_pnl\":{},\
                \"daily_unrealised_pnl\":{},\
                \"daily_pnl_total\":{},\
                \"risk_regime\":\"{:?}\",\
                \"num_intents\":{},\
                \"num_fills\":{}\
            }}\n",
            tick,
            fair_value,
            state.q_global_tao,
            state.dollar_delta_usd,
            state.basis_usd,
            state.basis_gross_usd,
            state.daily_realised_pnl,
            state.daily_unrealised_pnl,
            state.daily_pnl_total,
            state.risk_regime,
            intents.len(),
            fills.len(),
        );

        // If logging fails we don't want to crash the engine,
        // so we deliberately ignore I/O errors.
        let _ = self.writer.write_all(line.as_bytes());
        let _ = self.writer.flush();
    }
}
