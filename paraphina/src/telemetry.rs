//! telemetry.rs
//!
//! Lightweight JSONL telemetry sink for Paraphina.
//!
//! # Goals
//!
//! - Provide a simple, low-overhead way to write one JSON object per tick
//!   (or per event) to a file.
//! - Controlled entirely via environment variables so experiments can
//!   turn telemetry on/off without code changes.
//!
//! # Environment variables
//!
//! - `PARAPHINA_TELEMETRY_MODE`: `"off"` (default) disables telemetry,
//!   `"jsonl"` writes JSONL to `PARAPHINA_TELEMETRY_PATH`.
//! - `PARAPHINA_TELEMETRY_PATH`: Path to the JSONL file. Required when
//!   mode is `"jsonl"`.
//! - `PARAPHINA_TELEMETRY_APPEND`: Optional. When set to `"1"`/`"true"`/`"yes"`,
//!   appends to existing files instead of truncating. Default is truncate.
//!
//! # Schema Version Contract
//!
//! **Important:** This module is a generic JSONL writer and does **not**
//! auto-inject `schema_version`. Producers are responsible for including
//! `"schema_version": 1` in each record per the telemetry contract
//! (`docs/TELEMETRY_SCHEMA_V1.md`).
//!
//! Use [`ensure_schema_v1`] to validate/insert schema version on records.
//!
//! # Schema Version Contract
//!
//! **Important:** This module is a generic JSONL writer and does **not**
//! auto-inject `schema_version`. Producers are responsible for including
//! `"schema_version": 1` in each record per the telemetry contract
//! (`docs/TELEMETRY_SCHEMA_V1.md`).
//!
//! Use [`ensure_schema_v1`] to validate/insert schema version on records.
//!
//! # Usage (conceptual)
//!
//! In your main / engine loop, once per tick:
//!
//! ```ignore
//! use crate::telemetry::TelemetrySink;
//! use serde_json::json;
//!
//! fn main_loop() -> anyhow::Result<()> {
//!     let mut telemetry = TelemetrySink::from_env();
//!
//!     // inside tick loop:
//!     telemetry.log_json(&json!({
//!         "schema_version": 1,  // REQUIRED by telemetry contract
//!         "t": tick_index,
//!         "pnl_realised": global_state.pnl_realised,
//!         "pnl_unrealised": global_state.pnl_unrealised,
//!         "pnl_total": global_state.pnl_total(),
//!         "risk_regime": format!("{:?}", global_state.risk_regime),
//!         "kill_switch": global_state.kill_switch,
//!         // ... other required fields per docs/TELEMETRY_SCHEMA_V1.md
//!     }));
//!
//!     Ok(())
//! }
//! ```
//!
//! You decide exactly what fields to log; this module just handles
//! opening the file and appending JSON lines.
//!

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use serde::Serialize;
use serde_json::{self, Value as JsonValue};

use crate::config::Config;
use crate::exit::compute_exit_edge_components;
use crate::hedge::compute_hedge_cost_components;
use crate::mm::{compute_mm_quotes, compute_mm_reservation_components};
use crate::state::{GlobalState, KillEvent};
use crate::treasury::TreasuryGuidanceEngine;
use crate::types::{
    ExecutionEvent, FillEvent, OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus,
};
/// Current telemetry schema version.
pub const SCHEMA_VERSION: i64 = 1;

/// Ensure a JSON record has `schema_version: 1`.
///
/// This is a non-breaking helper to reduce mistakes when producing telemetry.
/// It is **not** automatically called by `TelemetrySink::log_json` to preserve
/// backwards compatibility and avoid overhead in hot paths.
///
/// # Behavior
///
/// - If `record` is a JSON Object:
///   - If `schema_version` is missing, inserts `"schema_version": 1`.
///   - If `schema_version` exists, leaves it unchanged.
/// - If `record` is not a JSON Object:
///   - In debug builds, panics with an assertion.
///   - In release builds, returns without modification.
///
/// # Example
///
/// ```ignore
/// use serde_json::json;
/// use paraphina::telemetry::ensure_schema_v1;
///
/// let mut record = json!({"t": 0, "pnl_total": 100.0});
/// ensure_schema_v1(&mut record);
/// assert_eq!(record["schema_version"], 1);
/// ```
pub fn ensure_schema_v1(record: &mut JsonValue) {
    match record {
        JsonValue::Object(map) => {
            map.entry("schema_version")
                .or_insert_with(|| JsonValue::Number(SCHEMA_VERSION.into()));
        }
        _ => {
            debug_assert!(
                false,
                "ensure_schema_v1: telemetry records should be JSON objects, got {:?}",
                record
            );
        }
    }
}

/// Telemetry mode, controlled by PARAPHINA_TELEMETRY_MODE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryMode {
    Off,
    Jsonl,
}

impl TelemetryMode {
    /// Parse mode from environment. Defaults to Off.
    pub fn from_env() -> Self {
        match env::var("PARAPHINA_TELEMETRY_MODE") {
            Ok(s) => match s.to_lowercase().as_str() {
                "jsonl" => TelemetryMode::Jsonl,
                // Treat any unknown value as Off for safety.
                _ => TelemetryMode::Off,
            },
            Err(_) => TelemetryMode::Off,
        }
    }
}

/// Configuration for the telemetry sink.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub mode: TelemetryMode,
    pub path: Option<PathBuf>,
    pub append: bool,
}

impl TelemetryConfig {
    /// Construct from environment variables.
    ///
    /// - mode = PARAPHINA_TELEMETRY_MODE (default Off)
    /// - path = PARAPHINA_TELEMETRY_PATH (required if mode == Jsonl)
    pub fn from_env() -> Self {
        let mode = TelemetryMode::from_env();

        let path = if mode == TelemetryMode::Jsonl {
            env::var("PARAPHINA_TELEMETRY_PATH").ok().map(PathBuf::from)
        } else {
            None
        };

        TelemetryConfig {
            mode,
            path,
            append: Self::append_from_env(),
        }
    }

    pub fn append_from_env() -> bool {
        env::var("PARAPHINA_TELEMETRY_APPEND")
            .ok()
            .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    }
}

/// A JSONL telemetry sink.
///
/// When mode == Off, all methods are no-ops.
/// When mode == Jsonl, we lazily open PARAPHINA_TELEMETRY_PATH on first use,
/// and append one JSON object per line.
pub struct TelemetrySink {
    mode: TelemetryMode,
    path: Option<PathBuf>,
    append: bool,
    writer: Option<BufWriter<File>>,
}

impl TelemetrySink {
    /// Construct a telemetry sink from environment configuration.
    ///
    /// This never fails: if configuration is invalid, it falls back to Off and
    /// logs nothing.
    pub fn from_env() -> Self {
        let cfg = TelemetryConfig::from_env();
        TelemetrySink {
            mode: cfg.mode,
            path: cfg.path,
            append: cfg.append,
            writer: None,
        }
    }

    /// Explicit constructor from a given config.
    pub fn from_config(cfg: TelemetryConfig) -> Self {
        TelemetrySink {
            mode: cfg.mode,
            path: cfg.path,
            append: cfg.append,
            writer: None,
        }
    }

    fn ensure_writer(&mut self) -> Option<&mut BufWriter<File>> {
        if self.mode != TelemetryMode::Jsonl {
            return None;
        }

        if self.writer.is_none() {
            let path = match &self.path {
                Some(p) => p.clone(),
                None => {
                    // Misconfigured: mode Jsonl but no path. Disable telemetry.
                    self.mode = TelemetryMode::Off;
                    return None;
                }
            };

            // Try to create parent directories if they don't exist.
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            let mut options = OpenOptions::new();
            options.create(true).write(true);
            if self.append {
                options.append(true);
            } else {
                options.truncate(true);
            }
            let file_res = options.open(&path);

            let file = match file_res {
                Ok(f) => f,
                Err(_) => {
                    // If we cannot open the file, disable telemetry silently
                    // to avoid panicking inside the trading loop.
                    self.mode = TelemetryMode::Off;
                    return None;
                }
            };

            self.writer = Some(BufWriter::new(file));
        }

        self.writer.as_mut()
    }

    /// Log a JSON value as a single line.
    ///
    /// If mode == Off or the writer cannot be opened, this is a no-op.
    ///
    /// Errors while writing are swallowed and cause telemetry to disable
    /// itself for the remainder of the process; they do not propagate to the
    /// trading logic.
    pub fn log_json(&mut self, value: &JsonValue) {
        if self.mode != TelemetryMode::Jsonl {
            return;
        }

        let writer = match self.ensure_writer() {
            Some(w) => w,
            None => return,
        };

        let line = match serde_json::to_string(value) {
            Ok(s) => s,
            Err(_) => return,
        };

        if writeln!(writer, "{}", line).is_err() {
            // Disable telemetry on write error.
            self.mode = TelemetryMode::Off;
            self.writer = None;
        }
    }

    /// Convenience helper: log a map-like structure as JSON.
    pub fn log_map<I, K, V>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<JsonValue>,
    {
        let mut obj = serde_json::Map::new();
        for (k, v) in iter {
            obj.insert(k.into(), v.into());
        }
        self.log_json(&JsonValue::Object(obj));
    }

    /// Flush the underlying writer, if any.
    pub fn flush(&mut self) {
        if let Some(writer) = self.writer.as_mut() {
            let _ = writer.flush();
        }
    }

    /// Consume this sync sink and convert it into an [`AsyncTelemetryWriter`]
    /// backed by a bounded channel and a background writer task.
    ///
    /// Only available with the `live` feature (requires tokio runtime).
    #[cfg(feature = "live")]
    pub fn into_async(mut self) -> AsyncTelemetryWriter {
        let cap: usize = std::env::var("PARAPHINA_TELEMETRY_CHANNEL_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        let (tx, rx) = tokio::sync::mpsc::channel::<String>(cap);
        let error_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag_clone = error_flag.clone();

        // Ensure writer is opened before moving to background task.
        let _ = self.ensure_writer();
        let writer = self.writer.take();
        let mode = self.mode;

        if mode != TelemetryMode::Jsonl || writer.is_none() {
            // Telemetry is off or could not open file. Return a no-op writer.
            error_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            return AsyncTelemetryWriter {
                sender: tx,
                error_flag,
            };
        }

        tokio::spawn(async move {
            async_telemetry_bg_task(rx, writer.unwrap(), flag_clone).await;
        });

        AsyncTelemetryWriter {
            sender: tx,
            error_flag,
        }
    }
}

// ---------------------------------------------------------------------------
// Async telemetry writer (live feature only)
// ---------------------------------------------------------------------------

/// Background task that receives serialized JSON lines via a bounded channel
/// and writes them to a `BufWriter<File>`. Flushes every 100 lines or every
/// 1 second, whichever comes first. On write error, sets the error flag and
/// stops writing (mirrors `TelemetrySink::log_json` disable-on-error behavior).
#[cfg(feature = "live")]
async fn async_telemetry_bg_task(
    mut rx: tokio::sync::mpsc::Receiver<String>,
    mut writer: BufWriter<File>,
    error_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::io::Write as _;
    let mut flush_interval = tokio::time::interval(std::time::Duration::from_secs(1));
    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut lines_since_flush: u64 = 0;

    loop {
        tokio::select! {
            line_opt = rx.recv() => {
                match line_opt {
                    Some(line) => {
                        if writeln!(writer, "{}", line).is_err() {
                            error_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[telemetry] async writer: write error, disabling");
                            // Drain remaining messages without writing.
                            while rx.recv().await.is_some() {}
                            return;
                        }
                        lines_since_flush += 1;
                        if lines_since_flush >= 100 {
                            let _ = writer.flush();
                            lines_since_flush = 0;
                        }
                    }
                    None => {
                        // Channel closed (sender dropped). Flush and exit.
                        let _ = writer.flush();
                        return;
                    }
                }
            }
            _ = flush_interval.tick() => {
                if lines_since_flush > 0 {
                    let _ = writer.flush();
                    lines_since_flush = 0;
                }
            }
        }
    }
}

/// A non-blocking telemetry writer that sends serialized JSON lines to a
/// background task via a bounded channel.
///
/// This is the live-trading replacement for `TelemetrySink::log_json` that
/// removes sync file I/O from the tick thread. Records are dropped silently
/// if the channel is full (best-effort telemetry, not critical path).
#[cfg(feature = "live")]
#[derive(Clone)]
pub struct AsyncTelemetryWriter {
    sender: tokio::sync::mpsc::Sender<String>,
    error_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "live")]
impl AsyncTelemetryWriter {
    /// Try to send a pre-serialized JSON line to the background writer.
    ///
    /// Returns `true` if the line was enqueued, `false` if the channel is full
    /// or telemetry has been disabled by a write error.
    pub fn try_send(&self, line: String) -> bool {
        if self.error_flag.load(std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        self.sender.try_send(line).is_ok()
    }
}

/// Handle that can hold either a synchronous sink or an async writer.
///
/// The `Sync` variant is used for replay/backtest/strategy paths that run
/// single-threaded without a tokio runtime. The `Async` variant is used for
/// the live/shadow trading hot path.
#[cfg(feature = "live")]
#[derive(Clone)]
pub enum TelemetrySinkHandle {
    Sync(std::sync::Arc<std::sync::Mutex<TelemetrySink>>),
    Async(AsyncTelemetryWriter),
}

#[cfg(feature = "live")]
impl TelemetrySinkHandle {
    /// Write a JSON record through whichever sink variant is active.
    pub fn log_json(&self, value: &JsonValue) {
        match self {
            TelemetrySinkHandle::Async(writer) => {
                if let Ok(line) = serde_json::to_string(value) {
                    let _ = writer.try_send(line);
                }
            }
            TelemetrySinkHandle::Sync(sink) => {
                if let Ok(mut guard) = sink.lock() {
                    guard.log_json(value);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TelemetryBuilder {
    prev_risk_regime: Option<String>,
    prev_kill_switch: bool,
    prev_delta_warn: bool,
    prev_delta_hard: bool,
    prev_basis_warn: bool,
    prev_basis_hard: bool,
    prev_pnl_warn: bool,
    prev_pnl_hard: bool,
    prev_venue_status: Vec<String>,
    prev_liq_warn: Vec<bool>,
    treasury: TreasuryGuidanceEngine,
}

pub struct TelemetryInputs<'a> {
    pub cfg: &'a Config,
    pub state: &'a GlobalState,
    pub tick: u64,
    pub now_ms: TimestampMs,
    pub intents: &'a [OrderIntent],
    pub exec_events: &'a [ExecutionEvent],
    pub fills: &'a [FillEvent],
    pub last_exit_intent: Option<&'a OrderIntent>,
    pub last_hedge_intent: Option<&'a OrderIntent>,
    pub kill_event: Option<&'a KillEvent>,
    pub shadow_mode: bool,
    pub execution_mode: &'a str,
    pub reconcile_drift: &'a [ReconcileDriftRecord],
    pub max_orders_per_tick: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReconcileDriftRecord {
    pub timestamp_ms: TimestampMs,
    pub venue_index: usize,
    pub venue_id: String,
    pub kind: String,
    pub internal: Option<f64>,
    pub venue: Option<f64>,
    pub diff: Option<f64>,
    pub tolerance: Option<f64>,
    pub source: String,
    pub available: bool,
}

impl TelemetryBuilder {
    pub fn new(cfg: &Config) -> Self {
        #[cfg(feature = "live")]
        {
            let venue_ids: Vec<&str> = cfg.venues.iter().map(|v| v.id.as_str()).collect();
            crate::live::venues::warn_if_noncanonical_venue_order(&venue_ids, "telemetry");
        }
        Self {
            prev_risk_regime: None,
            prev_kill_switch: false,
            prev_delta_warn: false,
            prev_delta_hard: false,
            prev_basis_warn: false,
            prev_basis_hard: false,
            prev_pnl_warn: false,
            prev_pnl_hard: false,
            prev_venue_status: vec!["Unknown".to_string(); cfg.venues.len()],
            prev_liq_warn: vec![false; cfg.venues.len()],
            treasury: TreasuryGuidanceEngine::new(cfg.venues.len()),
        }
    }

    pub fn build_record(&mut self, input: TelemetryInputs<'_>) -> JsonValue {
        let cfg = input.cfg;
        let state = input.state;
        let tick = input.tick;
        let now_ms = input.now_ms;

        let kill_reason = {
            let reason = format!("{:?}", state.kill_reason);
            if reason.is_empty() {
                "None".to_string()
            } else {
                reason
            }
        };

        let fair = state.fair_value.unwrap_or(state.fair_value_prev).max(1.0);
        let global_stale_ms = cfg.book.stale_ms;
        let healthy_venues_used = compute_healthy_venues_used(cfg, state, now_ms, global_stale_ms);
        let healthy_venues_used_count = healthy_venues_used.len();
        let mut record = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "t": tick,
            "pnl_realised": state.daily_realised_pnl,
            "pnl_unrealised": state.daily_unrealised_pnl,
            "pnl_total": state.daily_pnl_total,
            "risk_regime": format!("{:?}", state.risk_regime),
            "kill_switch": state.kill_switch,
            "kill_reason": kill_reason,
            "q_global_tao": state.q_global_tao,
            "dollar_delta_usd": state.dollar_delta_usd,
            "basis_usd": state.basis_usd,
            "basis_gross_usd": state.basis_gross_usd,
            "fv_available": state.fv_available,
            "fair_value": state.fair_value,
            "fv_short_vol": state.fv_short_vol,
            "fv_long_vol": state.fv_long_vol,
            "sigma_eff": state.sigma_eff,
            "kf_p": state.kf_p,
            "kf_x_hat": state.kf_x_hat,
            "kf_last_update_ms": state.kf_last_update_ms,
            "regime_ratio": state.vol_ratio_clipped,
            "shadow_vol_ratio": state.shadow_vol_ratio,
            "shadow_spread_mult": state.shadow_spread_mult,
            "shadow_size_mult": state.shadow_size_mult,
            "healthy_venues_used_count": healthy_venues_used_count,
            "healthy_venues_used": healthy_venues_used,
            "config_version_id": cfg.version,
            "execution_mode": input.execution_mode,
        });

        if !input.reconcile_drift.is_empty() {
            let drift = input
                .reconcile_drift
                .iter()
                .map(|rec| serde_json::to_value(rec).unwrap_or_default())
                .collect::<Vec<_>>();
            record.as_object_mut().expect("telemetry record").insert(
                "reconcile_drift".to_string(),
                serde_json::Value::Array(drift),
            );
        }

        if let serde_json::Value::Object(map) = &mut record {
            self.treasury.update(state, fair);
            map.insert(
                "treasury_guidance".to_string(),
                self.treasury.build_guidance(state, tick, now_ms),
            );
            let quote_levels = build_quote_levels(cfg, state, fair, now_ms);
            map.insert(
                "quote_levels".to_string(),
                serde_json::Value::Array(quote_levels),
            );

            let (orders, would_send_orders, would_send_truncated) = build_order_records(
                cfg,
                state,
                input.intents,
                input.exec_events,
                input.max_orders_per_tick,
            );
            map.insert("orders".to_string(), serde_json::Value::Array(orders));
            map.insert(
                "would_send_orders".to_string(),
                serde_json::Value::Array(would_send_orders),
            );
            map.insert(
                "would_send_orders_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(input.intents.len() as u64)),
            );
            map.insert(
                "would_send_orders_truncated".to_string(),
                serde_json::Value::Bool(would_send_truncated),
            );

            let fills = build_fill_records(state, input.fills, now_ms);
            map.insert("fills".to_string(), serde_json::Value::Array(fills));

            let exits = build_exit_records(
                cfg,
                state,
                input.intents,
                input.fills,
                input.last_exit_intent,
                now_ms,
            );
            map.insert("exits".to_string(), serde_json::Value::Array(exits));

            let (hedges, hedge_delta_h_t) = build_hedge_records(
                cfg,
                state,
                input.intents,
                input.fills,
                input.last_hedge_intent,
                now_ms,
            );
            map.insert("hedges".to_string(), serde_json::Value::Array(hedges));
            map.insert(
                "hedge_x_t".to_string(),
                serde_json::json!(state.q_global_tao),
            );
            map.insert(
                "hedge_delta_h_t".to_string(),
                serde_json::json!(hedge_delta_h_t),
            );

            let risk_events = self.build_risk_events(cfg, state, now_ms);
            map.insert(
                "risk_events".to_string(),
                serde_json::Value::Array(risk_events),
            );

            let venue_metrics = build_venue_metrics(cfg, state, now_ms, global_stale_ms);
            for (key, value) in venue_metrics {
                map.insert(key, value);
            }
        }

        if let Some(kill_event) = input.kill_event {
            if let serde_json::Value::Object(map) = &mut record {
                map.insert("kill_event".to_string(), serde_json::json!(kill_event));
            }
        }

        ensure_schema_v1(&mut record);
        record
    }

    fn build_risk_events(
        &mut self,
        cfg: &Config,
        state: &GlobalState,
        now_ms: TimestampMs,
    ) -> Vec<JsonValue> {
        let mut events = Vec::new();
        let risk_regime = format!("{:?}", state.risk_regime);
        if let Some(prev) = &self.prev_risk_regime {
            if prev != &risk_regime {
                events.push(serde_json::json!({
                    "event_type": "risk_regime_transition",
                    "from": prev,
                    "to": risk_regime,
                    "timestamp_ms": now_ms,
                }));
            }
        }

        if !self.prev_kill_switch && state.kill_switch {
            events.push(serde_json::json!({
                "event_type": "kill_switch_activation",
                "risk_regime": risk_regime,
                "timestamp_ms": now_ms,
            }));
        }

        let delta_abs = state.dollar_delta_usd.abs();
        let delta_warn = cfg.risk.delta_warn_frac * state.delta_limit_usd;
        let delta_hard = delta_abs >= state.delta_limit_usd;
        let delta_warn_breach = delta_abs >= delta_warn;
        if delta_warn_breach && !self.prev_delta_warn {
            events.push(serde_json::json!({
                "event_type": "delta_warn_breach",
                "value": delta_abs,
                "threshold": delta_warn,
                "timestamp_ms": now_ms,
            }));
        }
        if delta_hard && !self.prev_delta_hard {
            events.push(serde_json::json!({
                "event_type": "delta_hard_breach",
                "value": delta_abs,
                "threshold": state.delta_limit_usd,
                "timestamp_ms": now_ms,
            }));
        }

        let basis_abs = state.basis_usd.abs();
        let basis_hard = basis_abs >= state.basis_limit_hard_usd;
        let basis_warn = basis_abs >= state.basis_limit_warn_usd;
        if basis_warn && !self.prev_basis_warn {
            events.push(serde_json::json!({
                "event_type": "basis_warn_breach",
                "value": basis_abs,
                "threshold": state.basis_limit_warn_usd,
                "timestamp_ms": now_ms,
            }));
        }
        if basis_hard && !self.prev_basis_hard {
            events.push(serde_json::json!({
                "event_type": "basis_hard_breach",
                "value": basis_abs,
                "threshold": state.basis_limit_hard_usd,
                "timestamp_ms": now_ms,
            }));
        }

        let pnl = state.daily_pnl_total;
        let loss_limit = -cfg.risk.daily_loss_limit.abs();
        let pnl_warn = loss_limit * cfg.risk.pnl_warn_frac;
        let pnl_warn_breach = pnl <= pnl_warn;
        let pnl_hard_breach = pnl <= loss_limit;
        if pnl_warn_breach && !self.prev_pnl_warn {
            events.push(serde_json::json!({
                "event_type": "pnl_warn_breach",
                "value": pnl,
                "threshold": pnl_warn,
                "timestamp_ms": now_ms,
            }));
        }
        if pnl_hard_breach && !self.prev_pnl_hard {
            events.push(serde_json::json!({
                "event_type": "pnl_hard_breach",
                "value": pnl,
                "threshold": loss_limit,
                "timestamp_ms": now_ms,
            }));
        }

        for (idx, v) in state.venues.iter().enumerate() {
            #[allow(clippy::collapsible_if)]
            if self.prev_venue_status.get(idx).map(|s| s.as_str())
                != Some(&format!("{:?}", v.status))
            {
                if matches!(v.status, VenueStatus::Disabled) {
                    events.push(serde_json::json!({
                        "event_type": "venue_disabled",
                        "venue_index": idx,
                        "timestamp_ms": now_ms,
                    }));
                }
            }
            let liq_warn =
                v.dist_liq_sigma.is_finite() && v.dist_liq_sigma <= cfg.risk.liq_warn_sigma;
            if liq_warn && !self.prev_liq_warn.get(idx).copied().unwrap_or(false) {
                events.push(serde_json::json!({
                    "event_type": "liq_warn",
                    "venue_index": idx,
                    "value": v.dist_liq_sigma,
                    "threshold": cfg.risk.liq_warn_sigma,
                    "timestamp_ms": now_ms,
                }));
            }
        }

        self.prev_risk_regime = Some(risk_regime);
        self.prev_kill_switch = state.kill_switch;
        self.prev_delta_warn = delta_warn_breach;
        self.prev_delta_hard = delta_hard;
        self.prev_basis_warn = basis_warn;
        self.prev_basis_hard = basis_hard;
        self.prev_pnl_warn = pnl_warn_breach;
        self.prev_pnl_hard = pnl_hard_breach;
        self.prev_venue_status = state
            .venues
            .iter()
            .map(|v| format!("{:?}", v.status))
            .collect();
        self.prev_liq_warn = state
            .venues
            .iter()
            .map(|v| v.dist_liq_sigma.is_finite() && v.dist_liq_sigma <= cfg.risk.liq_warn_sigma)
            .collect();

        events
    }
}

fn compute_age_ms(now_ms: TimestampMs, last_mid_update_ms: Option<TimestampMs>) -> TimestampMs {
    match last_mid_update_ms {
        None => -1,
        Some(ts) => {
            if now_ms >= ts {
                now_ms - ts
            } else {
                0
            }
        }
    }
}

/// Detect if we're in fixture mode by checking if wall-clock `now_ms` is vastly
/// ahead of venue timestamps.
///
/// In live mode, `now_ms` is wall-clock time and should match the book timestamps.
/// In fixture mode, `now_ms` may be wall-clock while book timestamps are synthetic
/// (e.g., 1000, 1250, 1500...). We detect this by checking if `now_ms` is more than
/// 1 day ahead of the max observed `last_mid_update_ms`.
fn is_fixture_mode(state: &GlobalState, now_ms: TimestampMs) -> bool {
    const ONE_DAY_MS: i64 = 86_400_000;

    let max_update_ms = state
        .venues
        .iter()
        .filter_map(|v| v.last_mid_update_ms)
        .max();

    match max_update_ms {
        Some(max_ts) => now_ms - max_ts > ONE_DAY_MS,
        None => false,
    }
}

/// Compute the effective "now" for staleness calculations.
///
/// Returns the effective now_ms to use for age calculations:
/// - Live mode: returns the original `now_ms` (wall clock)
/// - Fixture mode: returns `max(last_mid_update_ms)` from venues (fixture timeline)
fn effective_now_for_staleness(state: &GlobalState, now_ms: TimestampMs) -> TimestampMs {
    if is_fixture_mode(state, now_ms) {
        // Fixture mode: use max venue timestamp as effective "now"
        state
            .venues
            .iter()
            .filter_map(|v| v.last_mid_update_ms)
            .max()
            .unwrap_or(now_ms)
    } else {
        now_ms
    }
}

/// Compute which venues are healthy for fair-value/Kalman contribution.
///
/// **Fail-closed semantics (Milestone E)**: A venue is considered healthy ONLY if:
/// 1. `venue.status == VenueStatus::Healthy`, AND
/// 2. `age_ms >= 0` (has received at least one book update), AND
/// 3. `age_ms <= venue_stale_ms` (book data is fresh within per-venue threshold).
///
/// This ensures that a venue with stale book data (even if `venue.status` hasn't
/// been updated yet) is never included in healthy_venues_used.
///
/// **Per-venue thresholds (Milestone F)**: Each venue can have a `stale_ms_override`
/// to use a different threshold than the global `book.stale_ms`. This is useful for
/// high-latency venues (e.g., Hyperliquid) that need a larger staleness window.
///
/// **Fixture mode handling**: When wall-clock `now_ms` is detected to be >1 day
/// ahead of venue timestamps (indicating fixture mode), the staleness gating is
/// disabled because fixtures feed data asynchronously and inter-venue timing
/// skew is expected. In fixture mode, we only check `venue.status`.
fn compute_healthy_venues_used(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    global_stale_ms: i64,
) -> Vec<usize> {
    let fixture_mode = is_fixture_mode(state, now_ms);
    let effective_now = effective_now_for_staleness(state, now_ms);
    let mut out = Vec::new();
    for (idx, venue) in state.venues.iter().enumerate() {
        if !matches!(venue.status, VenueStatus::Healthy) {
            continue;
        }
        if fixture_mode {
            // Fixture mode: skip staleness gating due to async timing skew.
            // Just check that venue has received at least one book update.
            if venue.last_mid_apply_ms.is_some() {
                out.push(idx);
            }
        } else {
            // Live mode: fail-closed staleness gating with per-venue threshold.
            let venue_stale_ms = cfg
                .venues
                .get(idx)
                .map(|v| v.effective_stale_ms(global_stale_ms))
                .unwrap_or(global_stale_ms);
            let age_ms = compute_age_ms(effective_now, venue.last_mid_apply_ms);
            if age_ms >= 0 && age_ms <= venue_stale_ms {
                out.push(idx);
            }
        }
    }
    out
}

fn build_quote_levels(
    cfg: &Config,
    state: &GlobalState,
    fair: f64,
    now_ms: TimestampMs,
) -> Vec<JsonValue> {
    let effective_now_ms = if now_ms > 0 {
        now_ms
    } else {
        crate::types::now_ms()
    };
    let components = compute_mm_reservation_components(cfg, state, Some(effective_now_ms));
    let quotes = compute_mm_quotes(cfg, state);
    #[allow(clippy::type_complexity)]
    let mut quote_by_venue: Vec<(Option<f64>, Option<f64>, Option<f64>, Option<f64>)> =
        vec![(None, None, None, None); cfg.venues.len()];
    for quote in quotes {
        let bid_price = quote.bid.as_ref().map(|b| b.price);
        let bid_size = quote.bid.as_ref().map(|b| b.size);
        let ask_price = quote.ask.as_ref().map(|a| a.price);
        let ask_size = quote.ask.as_ref().map(|a| a.size);
        quote_by_venue[quote.venue_index] = (bid_price, bid_size, ask_price, ask_size);
    }

    let mut out = Vec::new();
    for (idx, v) in state.venues.iter().enumerate() {
        let (bid_price, bid_size, ask_price, ask_size) = quote_by_venue[idx];
        let basis_adj = components.basis_adj_usd.get(idx).copied().unwrap_or(0.0);
        let funding_adj = components.funding_adj_usd.get(idx).copied().unwrap_or(0.0);
        let inv_term = components
            .inventory_term_usd
            .get(idx)
            .copied()
            .unwrap_or(0.0);
        let s_tilde = fair + basis_adj + funding_adj - inv_term;
        let delta_final = match (bid_price, ask_price) {
            (Some(b), Some(a)) if a >= b => (a - b) / 2.0,
            _ => 0.0,
        };
        let maker_cost = mm_maker_cost(&cfg.venues[idx], fair);
        let size_eta = cfg.mm.size_eta.max(1e-9);
        let spread_mult = state.spread_mult;
        let size_mult = state.size_mult;

        let (edge_bid, q_raw_bid, size_bid, margin_cap_bid, liq_factor_bid) = quote_diagnostics(
            cfg,
            v,
            fair,
            maker_cost,
            bid_price.unwrap_or(0.0),
            bid_size.unwrap_or(0.0),
            size_eta,
        );
        let (edge_ask, q_raw_ask, size_ask, margin_cap_ask, liq_factor_ask) = quote_diagnostics(
            cfg,
            v,
            fair,
            maker_cost,
            ask_price.unwrap_or(0.0),
            ask_size.unwrap_or(0.0),
            size_eta,
        );

        out.push(serde_json::json!({
            "venue_index": idx,
            "venue_id": v.id.as_ref(),
            "side": "Bid",
            "s_tilde": s_tilde,
            "basis_adj_usd": basis_adj,
            "funding_adj_usd": funding_adj,
            "inventory_term_usd": inv_term,
            "delta_final": delta_final,
            "spread_mult": spread_mult,
            "size_mult": size_mult,
            "edge_local": edge_bid,
            "size_raw": q_raw_bid,
            "size_final": size_bid,
            "size_margin_cap": margin_cap_bid,
            "size_liq_factor": liq_factor_bid,
            "price": bid_price.unwrap_or(0.0),
        }));
        out.push(serde_json::json!({
            "venue_index": idx,
            "venue_id": v.id.as_ref(),
            "side": "Ask",
            "s_tilde": s_tilde,
            "basis_adj_usd": basis_adj,
            "funding_adj_usd": funding_adj,
            "inventory_term_usd": inv_term,
            "delta_final": delta_final,
            "spread_mult": spread_mult,
            "size_mult": size_mult,
            "edge_local": edge_ask,
            "size_raw": q_raw_ask,
            "size_final": size_ask,
            "size_margin_cap": margin_cap_ask,
            "size_liq_factor": liq_factor_ask,
            "price": ask_price.unwrap_or(0.0),
        }));
    }
    out
}

fn mm_maker_cost(vcfg: &crate::config::VenueConfig, price: f64) -> f64 {
    let maker_fee = vcfg.maker_fee_bps / 10_000.0;
    let maker_rebate = vcfg.maker_rebate_bps / 10_000.0;
    (maker_fee - maker_rebate).max(0.0) * price
}

fn quote_diagnostics(
    cfg: &Config,
    v: &crate::state::VenueState,
    fair: f64,
    maker_cost: f64,
    price: f64,
    size_final: f64,
    size_eta: f64,
) -> (f64, f64, f64, f64, f64) {
    let edge = if price > 0.0 {
        if size_final > 0.0 {
            // side-specific edge defined relative to fair value
            if price <= fair {
                fair - price - maker_cost
            } else {
                price - fair - maker_cost
            }
        } else {
            0.0
        }
    } else {
        0.0
    };
    let q_raw = if edge > 0.0 { edge / size_eta } else { 0.0 };
    let margin_cap = if price > 0.0 {
        (v.margin_available_usd * cfg.risk.mm_max_leverage * cfg.risk.mm_margin_safety) / price
    } else {
        0.0
    };
    let dist = v.dist_liq_sigma;
    let liq_factor = if dist <= cfg.risk.liq_crit_sigma {
        0.0
    } else if dist < cfg.risk.liq_warn_sigma {
        ((dist - cfg.risk.liq_crit_sigma)
            / (cfg.risk.liq_warn_sigma - cfg.risk.liq_crit_sigma + 1e-9))
            .clamp(0.0, 1.0)
    } else {
        1.0
    };
    (edge, q_raw, size_final, margin_cap, liq_factor)
}

fn build_order_records(
    _cfg: &Config,
    state: &GlobalState,
    intents: &[OrderIntent],
    exec_events: &[ExecutionEvent],
    max_orders_per_tick: usize,
) -> (Vec<JsonValue>, Vec<JsonValue>, bool) {
    let mut orders = Vec::new();
    let mut would_send = Vec::new();

    for intent in intents {
        let (
            action,
            venue_index,
            venue_id,
            side,
            price,
            size,
            tif,
            post_only,
            reduce_only,
            purpose,
            order_id,
            client_order_id,
        ) = match intent {
            OrderIntent::Place(pi) => (
                "place",
                pi.venue_index as i64,
                pi.venue_id.to_string(),
                Some(format!("{:?}", pi.side)),
                Some(pi.price),
                Some(pi.size),
                Some(format!("{:?}", pi.time_in_force)),
                Some(pi.post_only),
                Some(pi.reduce_only),
                Some(format!("{:?}", pi.purpose)),
                None,
                pi.client_order_id.clone(),
            ),
            OrderIntent::Cancel(ci) => (
                "cancel",
                ci.venue_index as i64,
                ci.venue_id.to_string(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(ci.order_id.clone()),
                None,
            ),
            OrderIntent::Replace(ri) => (
                "replace",
                ri.venue_index as i64,
                ri.venue_id.to_string(),
                Some(format!("{:?}", ri.side)),
                Some(ri.price),
                Some(ri.size),
                Some(format!("{:?}", ri.time_in_force)),
                Some(ri.post_only),
                Some(ri.reduce_only),
                Some(format!("{:?}", ri.purpose)),
                Some(ri.order_id.clone()),
                ri.client_order_id.clone(),
            ),
            OrderIntent::CancelAll(ci) => (
                "cancel_all",
                ci.venue_index.map(|v| v as i64).unwrap_or(-1),
                ci.venue_id
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "ALL".to_string()),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        };

        let action_id = build_order_action_id(
            action,
            venue_index,
            side.as_ref(),
            price,
            size,
            client_order_id.as_ref(),
            order_id.as_ref(),
        );
        let record = serde_json::json!({
            "action": action,
            "status": "intent",
            "venue_index": venue_index,
            "venue_id": venue_id,
            "side": side,
            "price": price,
            "size": size,
            "tif": tif,
            "post_only": post_only,
            "reduce_only": reduce_only,
            "purpose": purpose,
            "risk_regime": format!("{:?}", state.risk_regime),
            "order_id": order_id,
            "client_order_id": client_order_id,
            "action_id": action_id,
        });
        orders.push(record.clone());
        would_send.push(record);
    }

    for event in exec_events {
        match event {
            ExecutionEvent::OrderAck(ack) => {
                let action = if ack.side.is_none() && ack.price.is_none() && ack.size.is_none() {
                    "cancel"
                } else {
                    "place"
                };
                let side_str = ack.side.map(|s| format!("{s:?}"));
                let action_id = build_order_action_id(
                    action,
                    ack.venue_index as i64,
                    side_str.as_ref(),
                    ack.price,
                    ack.size,
                    ack.client_order_id.as_ref(),
                    Some(&ack.order_id),
                );
                orders.push(serde_json::json!({
                    "action": action,
                    "status": "ack",
                    "venue_index": ack.venue_index as i64,
                    "venue_id": ack.venue_id.as_ref(),
                    "side": side_str,
                    "price": ack.price,
                    "size": ack.size,
                    "tif": Option::<String>::None,
                    "post_only": Option::<bool>::None,
                    "reduce_only": Option::<bool>::None,
                    "purpose": ack.purpose.map(|p| format!("{p:?}")),
                    "risk_regime": format!("{:?}", state.risk_regime),
                    "order_id": ack.order_id,
                    "client_order_id": ack.client_order_id,
                    "action_id": action_id,
                }));
            }
            ExecutionEvent::OrderReject(rej) => {
                let action_id = build_order_action_id(
                    "place",
                    rej.venue_index as i64,
                    None,
                    None,
                    None,
                    rej.client_order_id.as_ref(),
                    rej.order_id.as_ref(),
                );
                orders.push(serde_json::json!({
                    "action": "place",
                    "status": "reject",
                    "venue_index": rej.venue_index as i64,
                    "venue_id": rej.venue_id.as_ref(),
                    "side": Option::<String>::None,
                    "price": Option::<f64>::None,
                    "size": Option::<f64>::None,
                    "tif": Option::<String>::None,
                    "post_only": Option::<bool>::None,
                    "reduce_only": Option::<bool>::None,
                    "purpose": Option::<String>::None,
                    "risk_regime": format!("{:?}", state.risk_regime),
                    "order_id": rej.order_id,
                    "client_order_id": rej.client_order_id,
                    "reason": rej.reason,
                    "action_id": action_id,
                }));
            }
            _ => {}
        }
    }

    orders.sort_by_key(order_sort_key);

    let mut truncated = false;
    if would_send.len() > max_orders_per_tick {
        would_send.truncate(max_orders_per_tick);
        truncated = true;
    }
    would_send.sort_by_key(order_sort_key);

    (orders, would_send, truncated)
}

fn build_order_action_id(
    action: &str,
    venue_index: i64,
    side: Option<&String>,
    price: Option<f64>,
    size: Option<f64>,
    client_order_id: Option<&String>,
    order_id: Option<&String>,
) -> String {
    if let Some(id) = client_order_id {
        if !id.is_empty() {
            return id.to_string();
        }
    }
    if let Some(id) = order_id {
        if !id.is_empty() {
            return id.to_string();
        }
    }
    let side_str = side.map(|s| s.as_str()).unwrap_or("NA");
    let price_bits = price.unwrap_or(0.0).to_bits();
    let size_bits = size.unwrap_or(0.0).to_bits();
    format!("{action}:{venue_index}:{side_str}:{price_bits}:{size_bits}")
}

fn order_sort_key(value: &JsonValue) -> (String, i64, String, i64, i64, String, String) {
    let action_id = value
        .get("action_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let action = value.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let status = value.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let venue_index = value
        .get("venue_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);
    let side = value.get("side").and_then(|v| v.as_str()).unwrap_or("");
    let price = value
        .get("price")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .to_bits() as i64;
    let size = value
        .get("size")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .to_bits() as i64;
    (
        action_id,
        venue_index,
        side.to_string(),
        price,
        size,
        action.to_string(),
        status.to_string(),
    )
}

fn build_fill_records(
    state: &GlobalState,
    fills: &[FillEvent],
    now_ms: TimestampMs,
) -> Vec<JsonValue> {
    let mut out = Vec::new();
    for fill in fills {
        let record = find_fill_record(state, fill, now_ms);
        out.push(serde_json::json!({
            "fill_seq": record.as_ref().map(|r| r.fill_seq),
            "venue_index": fill.venue_index as i64,
            "venue_id": fill.venue_id.as_ref(),
            "order_id": fill.order_id,
            "client_order_id": fill.client_order_id,
            "side": format!("{:?}", fill.side),
            "price": fill.price,
            "size": fill.size,
            "purpose": format!("{:?}", fill.purpose),
            "fee_bps": record.as_ref().map(|r| r.fee_bps).unwrap_or(fill.fee_bps),
            "fill_time_ms": record.as_ref().map(|r| r.fill_time_ms).unwrap_or(now_ms),
            "pre_q_v": record.as_ref().and_then(|r| r.pre_position_tao),
            "post_q_v": record.as_ref().and_then(|r| r.post_position_tao),
            "pre_q_t": record.as_ref().and_then(|r| r.pre_q_global_tao),
            "post_q_t": record.as_ref().and_then(|r| r.post_q_global_tao),
            "realised_pnl_usd": record.as_ref().and_then(|r| r.realised_pnl_usd),
            "markout_pnl_short": record.as_ref().and_then(|r| r.markout_pnl_short),
        }));
    }
    out.sort_by_key(fill_sort_key);
    out
}

fn fill_sort_key(value: &JsonValue) -> (i64, i64, i64, String) {
    let seq = value.get("fill_seq").and_then(|v| v.as_i64()).unwrap_or(-1);
    let venue_index = value
        .get("venue_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);
    let price = value
        .get("price")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .to_bits() as i64;
    let side = value
        .get("side")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    (seq, venue_index, price, side)
}

fn find_fill_record<'a>(
    state: &'a GlobalState,
    fill: &FillEvent,
    now_ms: TimestampMs,
) -> Option<&'a crate::state::FillRecord> {
    let v = state.venues.get(fill.venue_index)?;
    v.recent_fills.iter().rev().find(|rec| {
        rec.fill_time_ms == now_ms
            && rec.side == fill.side
            && (rec.price - fill.price).abs() < 1e-12
            && (rec.size - fill.size).abs() < 1e-12
            && rec.order_id == fill.order_id
            && rec.client_order_id == fill.client_order_id
    })
}

fn exit_components_to_json(components: &crate::exit::ExitEdgeComponents) -> JsonValue {
    serde_json::json!({
        "edge_threshold": components.edge_threshold,
        "fee_per_tao": components.fee_per_tao,
        "slippage_buffer": components.slippage_buffer,
        "vol_buffer": components.vol_buffer,
        "basis_term": components.basis_term,
        "funding_benefit_per_tao": components.funding_benefit_per_tao,
        "frag_penalty": components.frag_penalty,
        "basis_risk_penalty": components.basis_risk_penalty,
    })
}

fn hedge_components_to_json(components: &crate::hedge::HedgeCostComponents) -> JsonValue {
    serde_json::json!({
        "exec_cost": components.exec_cost,
        "funding_benefit": components.funding_benefit,
        "basis_edge": components.basis_edge,
        "liq_penalty": components.liq_penalty,
        "frag_penalty": components.frag_penalty,
        "total_cost": components.total_cost,
    })
}

fn build_exit_records(
    cfg: &Config,
    state: &GlobalState,
    intents: &[OrderIntent],
    fills: &[FillEvent],
    last_exit_intent: Option<&OrderIntent>,
    now_ms: TimestampMs,
) -> Vec<JsonValue> {
    let mut out = Vec::new();
    let mut fill_sizes: std::collections::HashMap<(usize, Side), f64> =
        std::collections::HashMap::new();
    for fill in fills {
        if matches!(fill.purpose, OrderPurpose::Exit) {
            *fill_sizes
                .entry((fill.venue_index, fill.side))
                .or_insert(0.0) += fill.size;
        }
    }
    for intent in intents {
        if let OrderIntent::Place(pi) = intent {
            if !matches!(pi.purpose, OrderPurpose::Exit) {
                continue;
            }
            let components = compute_exit_edge_components(cfg, state, now_ms, intent);
            let components_json = components.as_ref().map(exit_components_to_json);
            let filled = fill_sizes
                .get(&(pi.venue_index, pi.side))
                .copied()
                .unwrap_or(0.0);
            let entry_fill_seqs = state
                .venues
                .get(pi.venue_index)
                .map(|v| {
                    v.recent_fills
                        .iter()
                        .filter(|f| matches!(f.purpose, OrderPurpose::Mm))
                        .take(5)
                        .map(|f| f.fill_seq)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let edge_raw = components
                .as_ref()
                .map(|c| c.edge_threshold + c.fee_per_tao + c.slippage_buffer + c.vol_buffer);
            let edge_funding_basis_adj = components.as_ref().map(|c| {
                c.edge_threshold + c.fee_per_tao + c.slippage_buffer + c.vol_buffer
                    - c.basis_term
                    - c.funding_benefit_per_tao
            });
            let edge_final = components.as_ref().map(|c| {
                c.edge_threshold + c.fee_per_tao + c.slippage_buffer + c.vol_buffer
                    - c.basis_term
                    - c.funding_benefit_per_tao
                    + c.frag_penalty
                    + c.basis_risk_penalty
            });
            out.push(serde_json::json!({
                "venue_index": pi.venue_index as i64,
                "venue_id": pi.venue_id.as_ref(),
                "side": format!("{:?}", pi.side),
                "intended_size": pi.size,
                "filled_size": filled,
                "entry_fill_seqs": entry_fill_seqs,
                "edge_components": components_json,
                "edge_raw": edge_raw,
                "edge_funding_basis_adj": edge_funding_basis_adj,
                "edge_final": edge_final,
                "risk_regime": format!("{:?}", state.risk_regime),
            }));
        }
    }
    if out.is_empty() {
        let _ = last_exit_intent;
    }
    out.sort_by(|a, b| {
        let venue_index = a.get("venue_index").and_then(|v| v.as_i64()).unwrap_or(-1);
        let venue_index_b = b.get("venue_index").and_then(|v| v.as_i64()).unwrap_or(-1);
        let side = a.get("side").and_then(|v| v.as_str()).unwrap_or("");
        let side_b = b.get("side").and_then(|v| v.as_str()).unwrap_or("");
        (venue_index, side).cmp(&(venue_index_b, side_b))
    });
    out
}

fn build_hedge_records(
    cfg: &Config,
    state: &GlobalState,
    intents: &[OrderIntent],
    fills: &[FillEvent],
    last_hedge_intent: Option<&OrderIntent>,
    _now_ms: TimestampMs,
) -> (Vec<JsonValue>, f64) {
    let mut out = Vec::new();
    let fair = state.fair_value.unwrap_or(state.fair_value_prev).max(1.0);
    let mut filled_by_venue: std::collections::HashMap<(usize, Side), f64> =
        std::collections::HashMap::new();
    for fill in fills {
        if matches!(fill.purpose, OrderPurpose::Hedge) {
            *filled_by_venue
                .entry((fill.venue_index, fill.side))
                .or_insert(0.0) += fill.size;
        }
    }
    let mut delta_h_t = 0.0;
    for intent in intents {
        if let OrderIntent::Place(pi) = intent {
            if !matches!(pi.purpose, OrderPurpose::Hedge) {
                continue;
            }
            let signed = match pi.side {
                Side::Buy => pi.size,
                Side::Sell => -pi.size,
            };
            delta_h_t += signed;
            let components = compute_hedge_cost_components(cfg, state, intent);
            let components_json = components.as_ref().map(hedge_components_to_json);
            let filled = filled_by_venue
                .get(&(pi.venue_index, pi.side))
                .copied()
                .unwrap_or(0.0);
            let venue = state.venues.get(pi.venue_index);
            out.push(serde_json::json!({
                "venue_index": pi.venue_index as i64,
                "venue_id": pi.venue_id.as_ref(),
                "side": format!("{:?}", pi.side),
                "delta_h_v": signed,
                "intended_size": pi.size,
                "filled_size": filled,
                "cost_components": components_json,
                "pre_q_v": venue.map(|v| v.position_tao),
                "post_q_v": venue.map(|v| v.position_tao + signed),
                "pre_q_t": state.q_global_tao,
                "post_q_t": state.q_global_tao + signed,
                "funding_8h": venue.map(|v| v.funding_8h).unwrap_or(0.0),
                "basis_usd": venue.map(|v| v.mid.unwrap_or(fair) - fair).unwrap_or(0.0),
                "dist_liq_sigma": venue.map(|v| v.dist_liq_sigma).unwrap_or(0.0),
            }));
        }
    }
    if out.is_empty() {
        let _ = last_hedge_intent;
    }
    out.sort_by(|a, b| {
        let venue_index = a.get("venue_index").and_then(|v| v.as_i64()).unwrap_or(-1);
        let venue_index_b = b.get("venue_index").and_then(|v| v.as_i64()).unwrap_or(-1);
        let side = a.get("side").and_then(|v| v.as_str()).unwrap_or("");
        let side_b = b.get("side").and_then(|v| v.as_str()).unwrap_or("");
        (venue_index, side).cmp(&(venue_index_b, side_b))
    });
    (out, delta_h_t)
}

/// Build venue metric arrays for telemetry output.
///
/// **Fail-closed semantics (Milestone E)**: `venue_status` reflects actual staleness:
/// - If `age_ms > venue_stale_ms` and venue is not Disabled, status is reported as "Stale"
/// - This ensures telemetry consumers can see when a venue is stale even if the
///   internal VenueStatus hasn't been updated yet by the health manager.
///
/// **Per-venue thresholds (Milestone F)**: Each venue can have a `stale_ms_override`
/// to use a different threshold than the global `book.stale_ms`.
///
/// **Fixture mode handling**: When wall-clock `now_ms` is detected to be >1 day
/// ahead of venue timestamps (indicating fixture mode), the staleness override is
/// disabled because fixtures feed data asynchronously and inter-venue timing skew
/// is expected. In fixture mode, we report the internal `venue.status` directly.
fn build_venue_metrics(
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    global_stale_ms: i64,
) -> Vec<(String, JsonValue)> {
    // Detect fixture mode to disable staleness override.
    let fixture_mode = is_fixture_mode(state, now_ms);
    // Use effective_now for age calculation in fixture mode.
    let effective_now = effective_now_for_staleness(state, now_ms);

    let mut venue_mid = Vec::new();
    let mut venue_spread = Vec::new();
    let mut venue_depth = Vec::new();
    let mut venue_status = Vec::new();
    let mut venue_toxicity = Vec::new();
    let mut venue_age_ms = Vec::new();
    let mut venue_age_event_ms = Vec::new();
    let mut venue_position = Vec::new();
    let mut venue_dist_liq_sigma = Vec::new();
    let mut venue_funding_8h = Vec::new();
    let mut venue_funding_rate_8h = Vec::new();
    let mut venue_funding_age_ms = Vec::new();
    let mut venue_funding_interval_sec = Vec::new();
    let mut venue_next_funding_ms = Vec::new();
    let mut venue_funding_source = Vec::new();
    let mut venue_funding_status = Vec::new();
    let mut venue_funding_settlement_price_kind = Vec::new();
    let mut venue_local_vol_short = Vec::new();
    let mut venue_local_vol_long = Vec::new();
    let mut venue_margin_balance = Vec::new();
    let mut venue_margin_available = Vec::new();
    let mut venue_margin_used = Vec::new();
    let mut venue_maker_volume = Vec::new();
    let mut venue_taker_volume = Vec::new();
    let mut venue_fill_rate = Vec::new();
    let mut venue_markout_ewma = Vec::new();

    for (idx, venue) in state.venues.iter().enumerate() {
        venue_mid.push(venue.mid.unwrap_or(0.0));
        venue_spread.push(venue.spread.unwrap_or(0.0));
        venue_depth.push(venue.depth_near_mid);
        let age_apply = compute_age_ms(effective_now, venue.last_mid_apply_ms);
        let age_event = compute_age_ms(effective_now, venue.last_mid_update_ms);
        // Fail-closed in live mode: report "Stale" if age exceeds per-venue threshold.
        // In fixture mode: skip staleness override due to async timing skew.
        let venue_stale_ms = cfg
            .venues
            .get(idx)
            .map(|v| v.effective_stale_ms(global_stale_ms))
            .unwrap_or(global_stale_ms);
        let effective_status = if matches!(venue.status, VenueStatus::Disabled) {
            "Disabled".to_string()
        } else if !fixture_mode && (age_apply < 0 || age_apply > venue_stale_ms) {
            // Fail-closed in live mode only: override to "Stale" if age exceeds threshold.
            "Stale".to_string()
        } else {
            format!("{:?}", venue.status)
        };
        venue_status.push(effective_status);
        venue_toxicity.push(venue.toxicity);
        venue_age_ms.push(age_apply);
        venue_age_event_ms.push(age_event);
        venue_position.push(venue.position_tao);
        venue_dist_liq_sigma.push(venue.dist_liq_sigma);
        venue_funding_8h.push(venue.funding_8h);
        let funding_state = &venue.funding_state;
        let funding_age = funding_state.age_ms(effective_now);
        let funding_status = funding_state.status_at(effective_now, cfg.funding.stale_ms);
        venue_funding_rate_8h.push(funding_state.rate_8h);
        venue_funding_age_ms.push(funding_age);
        venue_funding_interval_sec.push(funding_state.interval_sec);
        venue_next_funding_ms.push(funding_state.next_funding_ms);
        venue_funding_source.push(format!("{:?}", funding_state.source));
        venue_funding_status.push(format!("{:?}", funding_status));
        venue_funding_settlement_price_kind
            .push(format!("{:?}", funding_state.settlement_price_kind));
        venue_local_vol_short.push(venue.local_vol_short);
        venue_local_vol_long.push(venue.local_vol_long);
        venue_margin_balance.push(venue.margin_balance_usd);
        venue_margin_available.push(venue.margin_available_usd);
        venue_margin_used.push(venue.margin_used_usd);
        let (maker_volume, taker_volume) =
            venue
                .recent_fills
                .iter()
                .fold((0.0, 0.0), |acc, fill| match fill.purpose {
                    OrderPurpose::Mm => (acc.0 + fill.size, acc.1),
                    OrderPurpose::Exit | OrderPurpose::Hedge => (acc.0, acc.1 + fill.size),
                });
        venue_maker_volume.push(maker_volume);
        venue_taker_volume.push(taker_volume);
        let fills_count = venue.recent_fills.len() as f64;
        let open_orders = venue.open_orders.len() as f64;
        let fill_rate = if fills_count + open_orders > 0.0 {
            fills_count / (fills_count + open_orders)
        } else {
            0.0
        };
        venue_fill_rate.push(fill_rate);
        venue_markout_ewma.push(venue.markout_ewma_usd_per_tao);
    }

    vec![
        ("venue_mid_usd".to_string(), serde_json::json!(venue_mid)),
        (
            "venue_spread_usd".to_string(),
            serde_json::json!(venue_spread),
        ),
        (
            "venue_depth_near_mid_usd".to_string(),
            serde_json::json!(venue_depth),
        ),
        ("venue_status".to_string(), serde_json::json!(venue_status)),
        (
            "venue_toxicity".to_string(),
            serde_json::json!(venue_toxicity),
        ),
        ("venue_age_ms".to_string(), serde_json::json!(venue_age_ms)),
        (
            "venue_age_event_ms".to_string(),
            serde_json::json!(venue_age_event_ms),
        ),
        (
            "venue_position_tao".to_string(),
            serde_json::json!(venue_position),
        ),
        (
            "venue_dist_liq_sigma".to_string(),
            serde_json::json!(venue_dist_liq_sigma),
        ),
        (
            "venue_funding_8h".to_string(),
            serde_json::json!(venue_funding_8h),
        ),
        (
            "venue_funding_rate_8h".to_string(),
            serde_json::json!(venue_funding_rate_8h),
        ),
        (
            "venue_funding_age_ms".to_string(),
            serde_json::json!(venue_funding_age_ms),
        ),
        (
            "venue_funding_interval_sec".to_string(),
            serde_json::json!(venue_funding_interval_sec),
        ),
        (
            "venue_next_funding_ms".to_string(),
            serde_json::json!(venue_next_funding_ms),
        ),
        (
            "venue_funding_source".to_string(),
            serde_json::json!(venue_funding_source),
        ),
        (
            "venue_funding_status".to_string(),
            serde_json::json!(venue_funding_status),
        ),
        (
            "venue_funding_settlement_price_kind".to_string(),
            serde_json::json!(venue_funding_settlement_price_kind),
        ),
        (
            "venue_local_vol_short".to_string(),
            serde_json::json!(venue_local_vol_short),
        ),
        (
            "venue_local_vol_long".to_string(),
            serde_json::json!(venue_local_vol_long),
        ),
        (
            "venue_margin_balance_usd".to_string(),
            serde_json::json!(venue_margin_balance),
        ),
        (
            "venue_margin_available_usd".to_string(),
            serde_json::json!(venue_margin_available),
        ),
        (
            "venue_margin_used_usd".to_string(),
            serde_json::json!(venue_margin_used),
        ),
        (
            "venue_maker_volume".to_string(),
            serde_json::json!(venue_maker_volume),
        ),
        (
            "venue_taker_volume".to_string(),
            serde_json::json!(venue_taker_volume),
        ),
        (
            "venue_fill_rate".to_string(),
            serde_json::json!(venue_fill_rate),
        ),
        (
            "venue_markout_ewma_usd_per_tao".to_string(),
            serde_json::json!(venue_markout_ewma),
        ),
    ]
}

impl Drop for TelemetrySink {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::GlobalState;
    use serde_json::json;

    #[test]
    fn ensure_schema_v1_inserts_when_missing() {
        let mut record = json!({"t": 0, "pnl_total": 100.0});
        ensure_schema_v1(&mut record);
        assert_eq!(record["schema_version"], 1);
        // Other fields preserved
        assert_eq!(record["t"], 0);
        assert_eq!(record["pnl_total"], 100.0);
    }

    #[test]
    fn ensure_schema_v1_preserves_existing() {
        let mut record = json!({"schema_version": 1, "t": 5});
        ensure_schema_v1(&mut record);
        assert_eq!(record["schema_version"], 1);
        assert_eq!(record["t"], 5);
    }

    #[test]
    fn ensure_schema_v1_does_not_overwrite_version() {
        // Even if someone puts a different version, we don't overwrite
        let mut record = json!({"schema_version": 2, "t": 0});
        ensure_schema_v1(&mut record);
        // Existing value preserved (we only insert if missing)
        assert_eq!(record["schema_version"], 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "telemetry records should be JSON objects")]
    fn ensure_schema_v1_panics_on_non_object_debug() {
        let mut record = json!([1, 2, 3]);
        ensure_schema_v1(&mut record);
    }

    #[test]
    fn venue_metrics_reflect_book_updates() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        for (idx, venue) in state.venues.iter_mut().enumerate() {
            venue.mid = Some(100.0 + idx as f64);
            venue.spread = Some(0.5);
            venue.depth_near_mid = 10.0 + idx as f64;
            venue.last_mid_update_ms = Some(1_000);
            venue.last_mid_apply_ms = Some(1_000);
        }
        let stale_ms = cfg.book.stale_ms;
        let metrics = build_venue_metrics(&cfg, &state, 1_050, stale_ms);
        let mid = metrics
            .iter()
            .find(|(k, _)| k == "venue_mid_usd")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_mid_usd");
        let age = metrics
            .iter()
            .find(|(k, _)| k == "venue_age_ms")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_age_ms");
        let age_event = metrics
            .iter()
            .find(|(k, _)| k == "venue_age_event_ms")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_age_event_ms");
        let depth = metrics
            .iter()
            .find(|(k, _)| k == "venue_depth_near_mid_usd")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_depth_near_mid_usd");
        for idx in 0..cfg.venues.len() {
            assert!(mid[idx].as_f64().unwrap_or(0.0) > 0.0);
            assert!(age[idx].as_i64().unwrap_or(-1) >= 0);
            assert!(age_event[idx].as_i64().unwrap_or(-1) >= 0);
            assert!(depth[idx].as_f64().unwrap_or(0.0) > 0.0);
        }
    }

    #[test]
    fn healthy_venues_used_matches_statuses() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 1_000;
        let statuses = [
            VenueStatus::Disabled,
            VenueStatus::Healthy,
            VenueStatus::Healthy,
            VenueStatus::Disabled,
            VenueStatus::Healthy,
        ];
        for (idx, venue) in state.venues.iter_mut().enumerate() {
            let status = if idx < statuses.len() {
                statuses[idx]
            } else {
                VenueStatus::Disabled
            };
            venue.status = status;
            if matches!(venue.status, VenueStatus::Healthy) {
                venue.last_mid_update_ms = Some(900);
                venue.last_mid_apply_ms = Some(900);
            }
        }
        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 1,
            now_ms,
            intents: &[],
            exec_events: &[],
            fills: &[],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "shadow",
            reconcile_drift: &[],
            max_orders_per_tick: 0,
        });
        let used = record
            .get("healthy_venues_used")
            .and_then(|v| v.as_array())
            .expect("healthy_venues_used");
        let used_count = record
            .get("healthy_venues_used_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let indices: Vec<u64> = used.iter().filter_map(|v| v.as_u64()).collect();
        assert_eq!(indices, vec![1, 2, 4]);
        assert_eq!(used_count, 3);
    }

    #[test]
    fn venue_age_clamps_future_timestamp_and_keeps_missing() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 10_000;
        let stale_ms = cfg.book.stale_ms;

        state.venues[0].status = VenueStatus::Healthy;
        state.venues[0].last_mid_update_ms = Some(now_ms + 500);
        state.venues[0].last_mid_apply_ms = Some(now_ms + 500);
        state.venues[1].status = VenueStatus::Healthy;
        state.venues[1].last_mid_update_ms = None;
        state.venues[1].last_mid_apply_ms = None;

        let metrics = build_venue_metrics(&cfg, &state, now_ms, stale_ms);
        let age = metrics
            .iter()
            .find(|(k, _)| k == "venue_age_ms")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_age_ms");
        assert_eq!(age[0].as_i64().unwrap_or(-1), 0);
        assert_eq!(age[1].as_i64().unwrap_or(0), -1);
    }

    /// Milestone E: Fail-closed health semantics - a venue with age_ms > stale_ms
    /// MUST NOT be reported as Healthy and MUST NOT be included in healthy_venues_used.
    #[test]
    fn stale_venue_cannot_be_healthy() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let stale_ms = cfg.book.stale_ms;
        // Set venue 0 as fresh, venue 1 as stale (but internal status still Healthy).
        let now_ms = 10_000;

        // Venue 0: fresh (updated 50ms ago, well within stale_ms=1000)
        state.venues[0].status = VenueStatus::Healthy;
        state.venues[0].mid = Some(100.0);
        state.venues[0].last_mid_update_ms = Some(now_ms - 50);
        state.venues[0].last_mid_apply_ms = Some(now_ms - 50);

        // Venue 1: stale (updated 5000ms ago, beyond stale_ms=1000)
        // Internal status is still Healthy, but telemetry should report Stale.
        state.venues[1].status = VenueStatus::Healthy;
        state.venues[1].mid = Some(100.0);
        state.venues[1].last_mid_update_ms = Some(now_ms - 5000);
        state.venues[1].last_mid_apply_ms = Some(now_ms - 5000);

        // Venue 2: Disabled (should remain Disabled regardless of age)
        state.venues[2].status = VenueStatus::Disabled;
        state.venues[2].mid = Some(100.0);
        state.venues[2].last_mid_update_ms = Some(now_ms - 50);
        state.venues[2].last_mid_apply_ms = Some(now_ms - 50);

        // Check build_venue_metrics reports correct status.
        let metrics = build_venue_metrics(&cfg, &state, now_ms, stale_ms);
        let status = metrics
            .iter()
            .find(|(k, _)| k == "venue_status")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_status");

        assert_eq!(
            status[0].as_str(),
            Some("Healthy"),
            "Fresh venue should be Healthy"
        );
        assert_eq!(
            status[1].as_str(),
            Some("Stale"),
            "Stale venue MUST NOT be Healthy (fail-closed)"
        );
        assert_eq!(
            status[2].as_str(),
            Some("Disabled"),
            "Disabled venue should remain Disabled"
        );

        // Check compute_healthy_venues_used excludes stale venue.
        let healthy = compute_healthy_venues_used(&cfg, &state, now_ms, stale_ms);
        assert!(
            healthy.contains(&0),
            "Fresh healthy venue should be included"
        );
        assert!(
            !healthy.contains(&1),
            "Stale venue MUST NOT be in healthy_venues_used (fail-closed)"
        );
        assert!(
            !healthy.contains(&2),
            "Disabled venue should not be in healthy_venues_used"
        );
    }

    /// Milestone F: Per-venue stale thresholds allow high-latency venues to use
    /// a larger staleness window than the global default.
    #[test]
    fn per_venue_stale_threshold_override() {
        let mut cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 10_000;

        // Global stale_ms = 1000 (default).
        let global_stale_ms = cfg.book.stale_ms;
        assert_eq!(
            global_stale_ms, 1000,
            "Default global stale_ms should be 1000"
        );

        // Venue 0: no override, use global threshold.
        // Set last update 500ms ago => should be healthy.
        state.venues[0].status = VenueStatus::Healthy;
        state.venues[0].mid = Some(100.0);
        state.venues[0].last_mid_update_ms = Some(now_ms - 500);
        state.venues[0].last_mid_apply_ms = Some(now_ms - 500);

        // Venue 1: override to 3000ms (simulating high-latency venue).
        // Set last update 2000ms ago => would be stale with global, but healthy with override.
        cfg.venues[1].stale_ms_override = Some(3000);
        state.venues[1].status = VenueStatus::Healthy;
        state.venues[1].mid = Some(100.0);
        state.venues[1].last_mid_update_ms = Some(now_ms - 2000);
        state.venues[1].last_mid_apply_ms = Some(now_ms - 2000);

        // Venue 2: no override, 1500ms old => stale with global threshold.
        state.venues[2].status = VenueStatus::Healthy;
        state.venues[2].mid = Some(100.0);
        state.venues[2].last_mid_update_ms = Some(now_ms - 1500);
        state.venues[2].last_mid_apply_ms = Some(now_ms - 1500);

        // Check compute_healthy_venues_used respects per-venue threshold.
        let healthy = compute_healthy_venues_used(&cfg, &state, now_ms, global_stale_ms);
        assert!(
            healthy.contains(&0),
            "Venue 0: fresh (500ms < 1000ms) should be healthy"
        );
        assert!(
            healthy.contains(&1),
            "Venue 1: override 3000ms, age 2000ms should be healthy"
        );
        assert!(
            !healthy.contains(&2),
            "Venue 2: no override, age 1500ms > 1000ms should be stale"
        );

        // Check build_venue_metrics respects per-venue threshold.
        let metrics = build_venue_metrics(&cfg, &state, now_ms, global_stale_ms);
        let status = metrics
            .iter()
            .find(|(k, _)| k == "venue_status")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_status");

        assert_eq!(
            status[0].as_str(),
            Some("Healthy"),
            "Venue 0 should be Healthy"
        );
        assert_eq!(
            status[1].as_str(),
            Some("Healthy"),
            "Venue 1 with override should be Healthy despite 2000ms age"
        );
        assert_eq!(
            status[2].as_str(),
            Some("Stale"),
            "Venue 2 without override should be Stale at 1500ms age"
        );
    }
}
