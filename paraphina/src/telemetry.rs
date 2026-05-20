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

use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use serde::Serialize;
use serde_json::{self, Value as JsonValue};

use crate::config::Config;
use crate::exit::compute_exit_edge_components;
use crate::hedge::compute_hedge_cost_components;
use crate::mm::venue_utility_conversion_penalties_enabled;
use crate::mm::{
    compute_mm_quotes_with_now, compute_mm_reservation_components, compute_venue_targets,
    compute_venue_utility_decision, quote_spread_gate_reason, VenueUtilityDecision,
    HYPERLIQUID_TOUCH_CLIP_MAX_TICKS,
};
use crate::state::{
    funding_rate_for_decision, GlobalState, KillEvent, RiskRegime, VenueUtilityTier,
};
use crate::treasury::TreasuryGuidanceEngine;
use crate::types::{
    ExecutionEvent, FillEvent, OrderIntent, OrderPurpose, Side, TimestampMs, VenueStatus,
};
/// Current telemetry schema version.
pub const SCHEMA_VERSION: i64 = 1;
const HEDGE_SOURCE_FILL_CACHE_MAX_AGE_MS: TimestampMs = 8 * 60 * 60 * 1_000;
const HEDGE_SOURCE_FILL_CACHE_CAP: usize = 256;

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
    source_fill_cache: VecDeque<SourceFillAttribution>,
}

#[derive(Debug, Clone)]
struct SourceFillAttribution {
    source_decision_id: String,
    venue_index: usize,
    venue_id: String,
    order_id: Option<String>,
    client_order_id: Option<String>,
    side: Side,
    price: f64,
    size: f64,
    fill_time_ms: TimestampMs,
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
    pub account_position_syncs: &'a [AccountPositionSyncRecord],
    pub max_orders_per_tick: usize,
    pub venue_health_diagnostics: &'a [VenueHealthDiagnostics],
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VenueHealthDiagnostics {
    pub api_errors: u32,
    pub stale_count: u32,
    pub dev_breaches: u32,
    pub disable_reason: String,
    pub last_error_source: String,
    pub last_error_message: String,
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

#[derive(Debug, Clone, Serialize)]
pub struct AccountPositionSyncRecord {
    pub venue_index: usize,
    pub venue_id: String,
    pub snapshot_seq: u64,
    pub snapshot_timestamp_ms: Option<TimestampMs>,
    pub ingest_now_ms: TimestampMs,
    pub pre_position_tao: f64,
    pub post_position_tao: f64,
    pub position_delta_tao: f64,
    pub pre_margin_available_usd: f64,
    pub post_margin_available_usd: f64,
    pub source: &'static str,
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
            source_fill_cache: VecDeque::new(),
        }
    }

    fn prune_source_fill_cache(&mut self, now_ms: TimestampMs) {
        while self.source_fill_cache.front().is_some_and(|fill| {
            (now_ms - fill.fill_time_ms).max(0) > HEDGE_SOURCE_FILL_CACHE_MAX_AGE_MS
        }) {
            self.source_fill_cache.pop_front();
        }
    }

    fn observe_source_fills(
        &mut self,
        state: &GlobalState,
        fills: &[FillEvent],
        now_ms: TimestampMs,
    ) {
        self.prune_source_fill_cache(now_ms);
        for fill in fills {
            if !matches!(fill.purpose, OrderPurpose::Mm) {
                continue;
            }
            let Some(source_decision_id) = lookup_order_decision_id(
                state,
                fill.order_id.as_deref(),
                fill.client_order_id.as_deref(),
            ) else {
                continue;
            };
            let duplicate = self.source_fill_cache.iter().any(|cached| {
                cached.venue_index == fill.venue_index
                    && cached.order_id == fill.order_id
                    && cached.client_order_id == fill.client_order_id
                    && cached.side == fill.side
                    && (cached.price - fill.price).abs() < 1e-12
                    && (cached.size - fill.size).abs() < 1e-12
                    && cached.fill_time_ms == now_ms
            });
            if duplicate {
                continue;
            }
            self.source_fill_cache.push_back(SourceFillAttribution {
                source_decision_id,
                venue_index: fill.venue_index,
                venue_id: fill.venue_id.to_string(),
                order_id: fill.order_id.clone(),
                client_order_id: fill.client_order_id.clone(),
                side: fill.side,
                price: fill.price,
                size: fill.size,
                fill_time_ms: now_ms,
            });
            while self.source_fill_cache.len() > HEDGE_SOURCE_FILL_CACHE_CAP {
                self.source_fill_cache.pop_front();
            }
        }
    }

    pub fn build_record(&mut self, input: TelemetryInputs<'_>) -> JsonValue {
        let cfg = input.cfg;
        let state = input.state;
        let tick = input.tick;
        let now_ms = input.now_ms;
        self.observe_source_fills(state, input.fills, now_ms);

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
        let q_gross_tao: f64 = state
            .venues
            .iter()
            .map(|venue| venue.position_tao.abs())
            .sum();
        let q_max_abs_venue_tao: f64 = state
            .venues
            .iter()
            .map(|venue| venue.position_tao.abs())
            .fold(0.0_f64, f64::max);
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
            "q_gross_tao": q_gross_tao,
            "q_max_abs_venue_tao": q_max_abs_venue_tao,
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
            let account_position_syncs = input
                .account_position_syncs
                .iter()
                .map(|sync| serde_json::to_value(sync).unwrap_or_default())
                .collect::<Vec<_>>();
            let execution_visibility_gap =
                !input.account_position_syncs.is_empty() && input.fills.is_empty();
            let execution_visibility_gap_venues = input
                .account_position_syncs
                .iter()
                .map(|sync| sync.venue_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(serde_json::Value::String)
                .collect::<Vec<_>>();
            map.insert(
                "account_position_syncs".to_string(),
                serde_json::Value::Array(account_position_syncs),
            );
            map.insert(
                "account_position_sync_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(
                    input.account_position_syncs.len() as u64,
                )),
            );
            map.insert(
                "execution_visibility_gap".to_string(),
                serde_json::Value::Bool(execution_visibility_gap),
            );
            map.insert(
                "execution_visibility_gap_reason".to_string(),
                serde_json::Value::String(if execution_visibility_gap {
                    "account_position_sync_without_fill".to_string()
                } else {
                    String::new()
                }),
            );
            map.insert(
                "execution_visibility_gap_venues".to_string(),
                serde_json::Value::Array(execution_visibility_gap_venues),
            );

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

            let fills = build_fill_records(state, input.fills, now_ms, &self.source_fill_cache);
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
                &self.source_fill_cache,
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

            let venue_metrics = build_venue_metrics(
                cfg,
                state,
                now_ms,
                global_stale_ms,
                input.venue_health_diagnostics,
            );
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
        if cfg
            .venues
            .get(idx)
            .map(|v| !v.contributes_to_fv)
            .unwrap_or(false)
        {
            continue;
        }
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
    let targets = compute_venue_targets(cfg, state, Some(effective_now_ms));
    let quotes = compute_mm_quotes_with_now(cfg, state, Some(effective_now_ms));
    #[allow(clippy::type_complexity)]
    let mut quote_by_venue: Vec<(
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        bool,
        bool,
        bool,
        Option<&'static str>,
        &'static str,
        &'static str,
    )> = vec![
        (
            None,
            None,
            None,
            None,
            false,
            false,
            false,
            None,
            "not_quoted",
            "not_quoted"
        );
        cfg.venues.len()
    ];
    for quote in quotes {
        let bid_price = quote.bid.as_ref().map(|b| b.price);
        let bid_size = quote.bid.as_ref().map(|b| b.size);
        let ask_price = quote.ask.as_ref().map(|a| a.price);
        let ask_size = quote.ask.as_ref().map(|a| a.size);
        quote_by_venue[quote.venue_index] = (
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            quote.generated_spread_cap_applied,
            quote.generated_spread_cap_bid_suppressed,
            quote.generated_spread_cap_ask_suppressed,
            quote.touch_mode_kind,
            quote.bid_terminal_reason,
            quote.ask_terminal_reason,
        );
    }

    let mut out = Vec::new();
    let conversion_penalties_enabled = venue_utility_conversion_penalties_enabled();
    for (idx, v) in state.venues.iter().enumerate() {
        let (
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            generated_spread_cap_applied,
            generated_spread_cap_bid_suppressed,
            generated_spread_cap_ask_suppressed,
            touch_mode_kind,
            bid_terminal_reason,
            ask_terminal_reason,
        ) = quote_by_venue[idx];
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
        let q_target = targets
            .get(idx)
            .map(|target| target.q_target)
            .unwrap_or(0.0);
        let utility = compute_venue_utility_decision(
            &cfg.mm,
            state.q_global_tao,
            &cfg.venues[idx],
            v,
            conversion_penalties_enabled,
        );

        let bid_diag = quote_diagnostics(
            cfg,
            state,
            &cfg.venues[idx],
            v,
            fair,
            effective_now_ms,
            Side::Buy,
            q_target,
            utility,
            maker_cost,
            bid_price.unwrap_or(0.0),
            bid_size.unwrap_or(0.0),
            size_eta,
            generated_spread_cap_applied,
            generated_spread_cap_bid_suppressed,
            generated_spread_cap_ask_suppressed,
        );
        let ask_diag = quote_diagnostics(
            cfg,
            state,
            &cfg.venues[idx],
            v,
            fair,
            effective_now_ms,
            Side::Sell,
            q_target,
            utility,
            maker_cost,
            ask_price.unwrap_or(0.0),
            ask_size.unwrap_or(0.0),
            size_eta,
            generated_spread_cap_applied,
            generated_spread_cap_bid_suppressed,
            generated_spread_cap_ask_suppressed,
        );
        let best_bid = v.mid.zip(v.spread).map(|(mid, spread)| mid - spread / 2.0);
        let best_ask = v.mid.zip(v.spread).map(|(mid, spread)| mid + spread / 2.0);
        let tick = cfg.venues[idx].tick_size.max(1e-6);
        let bid_distance_to_touch_bps = match (bid_price, best_bid, v.mid) {
            (Some(price), Some(best_bid), Some(mid)) if mid > 0.0 => {
                ((best_bid - price).max(0.0) / mid) * 10_000.0
            }
            _ => 0.0,
        };
        let ask_distance_to_touch_bps = match (ask_price, best_ask, v.mid) {
            (Some(price), Some(best_ask), Some(mid)) if mid > 0.0 => {
                ((price - best_ask).max(0.0) / mid) * 10_000.0
            }
            _ => 0.0,
        };
        let bid_touch_offset_ticks = match (bid_price, best_bid) {
            (Some(price), Some(best_bid)) => ((best_bid - price).max(0.0) / tick).round(),
            _ => 0.0,
        };
        let ask_touch_offset_ticks = match (ask_price, best_ask) {
            (Some(price), Some(best_ask)) => ((price - best_ask).max(0.0) / tick).round(),
            _ => 0.0,
        };
        let touch_mode_applied = touch_mode_kind.is_some();

        out.push(build_quote_level_record(
            idx,
            v.id.as_ref(),
            "Bid",
            s_tilde,
            basis_adj,
            funding_adj,
            inv_term,
            delta_final,
            spread_mult,
            size_mult,
            bid_price.unwrap_or(0.0),
            bid_distance_to_touch_bps,
            bid_touch_offset_ticks,
            touch_mode_applied,
            touch_mode_kind,
            generated_spread_cap_applied,
            generated_spread_cap_bid_suppressed,
            generated_spread_cap_ask_suppressed,
            bid_terminal_reason,
            q_target,
            &bid_diag,
        ));
        out.push(build_quote_level_record(
            idx,
            v.id.as_ref(),
            "Ask",
            s_tilde,
            basis_adj,
            funding_adj,
            inv_term,
            delta_final,
            spread_mult,
            size_mult,
            ask_price.unwrap_or(0.0),
            ask_distance_to_touch_bps,
            ask_touch_offset_ticks,
            touch_mode_applied,
            touch_mode_kind,
            generated_spread_cap_applied,
            generated_spread_cap_bid_suppressed,
            generated_spread_cap_ask_suppressed,
            ask_terminal_reason,
            q_target,
            &ask_diag,
        ));
    }
    out
}

#[inline]
fn generated_spread_cap_mode(
    generated_spread_cap_applied: bool,
    generated_spread_cap_bid_suppressed: bool,
    generated_spread_cap_ask_suppressed: bool,
) -> &'static str {
    if !generated_spread_cap_applied {
        "none"
    } else if generated_spread_cap_bid_suppressed && generated_spread_cap_ask_suppressed {
        "suppress_both"
    } else if generated_spread_cap_bid_suppressed {
        "suppress_bid"
    } else if generated_spread_cap_ask_suppressed {
        "suppress_ask"
    } else {
        "compressed_only"
    }
}

#[allow(clippy::too_many_arguments)]
fn build_quote_level_record(
    venue_index: usize,
    venue_id: &str,
    side: &'static str,
    s_tilde: f64,
    basis_adj: f64,
    funding_adj: f64,
    inv_term: f64,
    delta_final: f64,
    spread_mult: f64,
    size_mult: f64,
    price: f64,
    distance_to_touch_bps: f64,
    touch_offset_ticks: f64,
    touch_mode_applied: bool,
    touch_mode_kind: Option<&'static str>,
    generated_spread_cap_applied: bool,
    generated_spread_cap_bid_suppressed: bool,
    generated_spread_cap_ask_suppressed: bool,
    engine_terminal_reason: &'static str,
    q_target: f64,
    diag: &QuoteTelemetryDiagnostics,
) -> JsonValue {
    let mut record = serde_json::Map::new();
    record.insert(
        "venue_index".to_string(),
        JsonValue::from(venue_index as u64),
    );
    record.insert(
        "venue_id".to_string(),
        JsonValue::from(venue_id.to_string()),
    );
    record.insert("side".to_string(), JsonValue::from(side));
    record.insert("s_tilde".to_string(), JsonValue::from(s_tilde));
    record.insert("basis_adj_usd".to_string(), JsonValue::from(basis_adj));
    record.insert("funding_adj_usd".to_string(), JsonValue::from(funding_adj));
    record.insert("inventory_term_usd".to_string(), JsonValue::from(inv_term));
    record.insert("delta_final".to_string(), JsonValue::from(delta_final));
    record.insert("spread_mult".to_string(), JsonValue::from(spread_mult));
    record.insert("size_mult".to_string(), JsonValue::from(size_mult));
    record.insert("edge_local".to_string(), JsonValue::from(diag.edge));
    record.insert("size_raw".to_string(), JsonValue::from(diag.q_raw));
    record.insert("size_final".to_string(), JsonValue::from(diag.size_final));
    record.insert(
        "size_margin_cap".to_string(),
        JsonValue::from(diag.margin_cap),
    );
    record.insert(
        "size_liq_factor".to_string(),
        JsonValue::from(diag.liq_factor),
    );
    record.insert("price".to_string(), JsonValue::from(price));
    record.insert("quote_state".to_string(), JsonValue::from(diag.quote_state));
    record.insert(
        "engine_terminal_reason".to_string(),
        JsonValue::from(engine_terminal_reason),
    );
    record.insert(
        "generated_spread_cap_applied".to_string(),
        JsonValue::from(generated_spread_cap_applied),
    );
    record.insert(
        "generated_spread_cap_bid_suppressed".to_string(),
        JsonValue::from(generated_spread_cap_bid_suppressed),
    );
    record.insert(
        "generated_spread_cap_ask_suppressed".to_string(),
        JsonValue::from(generated_spread_cap_ask_suppressed),
    );
    record.insert(
        "generated_spread_cap_mode".to_string(),
        JsonValue::from(generated_spread_cap_mode(
            generated_spread_cap_applied,
            generated_spread_cap_bid_suppressed,
            generated_spread_cap_ask_suppressed,
        )),
    );
    record.insert(
        "distance_to_touch_bps".to_string(),
        JsonValue::from(distance_to_touch_bps),
    );
    record.insert(
        "touch_offset_ticks".to_string(),
        JsonValue::from(touch_offset_ticks),
    );
    record.insert(
        "touch_mode_applied".to_string(),
        JsonValue::from(touch_mode_applied),
    );
    record.insert(
        "touch_mode_kind".to_string(),
        touch_mode_kind
            .map(JsonValue::from)
            .unwrap_or(JsonValue::Null),
    );
    record.insert(
        "touch_clip_applied".to_string(),
        JsonValue::from(matches!(touch_mode_kind, Some("hyperliquid_clip"))),
    );
    record.insert(
        "touch_clip_max_ticks".to_string(),
        if matches!(touch_mode_kind, Some("hyperliquid_clip")) {
            JsonValue::from(HYPERLIQUID_TOUCH_CLIP_MAX_TICKS)
        } else {
            JsonValue::Null
        },
    );
    record.insert(
        "suppression_reason".to_string(),
        diag.suppression_reason
            .map(JsonValue::from)
            .unwrap_or(JsonValue::Null),
    );
    record.insert("book_age_ms".to_string(), JsonValue::from(diag.book_age_ms));
    record.insert(
        "book_stale_threshold_ms".to_string(),
        JsonValue::from(diag.book_stale_threshold_ms),
    );
    record.insert(
        "edge_threshold".to_string(),
        JsonValue::from(diag.edge_threshold),
    );
    record.insert(
        "edge_threshold_base".to_string(),
        JsonValue::from(diag.base_edge_threshold),
    );
    record.insert(
        "hedge_cost_edge_floor".to_string(),
        JsonValue::from(diag.hedge_cost_edge_floor),
    );
    record.insert(
        "utility_extra_edge_usd".to_string(),
        JsonValue::from(diag.utility_extra_edge_usd),
    );
    record.insert(
        "utility_tier".to_string(),
        JsonValue::from(diag.utility_tier),
    );
    record.insert(
        "utility_reason".to_string(),
        JsonValue::from(diag.utility_reason),
    );
    record.insert(
        "utility_role".to_string(),
        JsonValue::from(diag.utility_role),
    );
    record.insert(
        "utility_size_multiplier".to_string(),
        JsonValue::from(diag.utility_size_multiplier),
    );
    record.insert("prune_stage".to_string(), JsonValue::from(diag.prune_stage));
    record.insert(
        "candidate_price_pre_utility".to_string(),
        JsonValue::from(diag.candidate_price_pre_utility),
    );
    record.insert(
        "candidate_edge_pre_utility".to_string(),
        JsonValue::from(diag.candidate_edge_pre_utility),
    );
    record.insert(
        "candidate_size_pre_margin".to_string(),
        JsonValue::from(diag.candidate_size_pre_margin),
    );
    record.insert(
        "candidate_size_post_margin".to_string(),
        JsonValue::from(diag.candidate_size_post_margin),
    );
    record.insert(
        "candidate_size_pre_utility".to_string(),
        JsonValue::from(diag.candidate_size_pre_utility),
    );
    record.insert(
        "candidate_size_post_utility".to_string(),
        JsonValue::from(diag.candidate_size_post_utility),
    );
    record.insert(
        "candidate_size_post_taper".to_string(),
        JsonValue::from(diag.candidate_size_post_taper),
    );
    record.insert(
        "candidate_notional_post_utility".to_string(),
        JsonValue::from(diag.candidate_notional_post_utility),
    );
    record.insert(
        "lot_size_tao".to_string(),
        JsonValue::from(diag.lot_size_tao),
    );
    record.insert("q_target_tao".to_string(), JsonValue::from(q_target));
    JsonValue::Object(record)
}

#[derive(Debug, Clone)]
struct QuoteTelemetryDiagnostics {
    edge: f64,
    q_raw: f64,
    size_final: f64,
    margin_cap: f64,
    liq_factor: f64,
    quote_state: &'static str,
    suppression_reason: Option<&'static str>,
    prune_stage: &'static str,
    book_age_ms: TimestampMs,
    book_stale_threshold_ms: i64,
    edge_threshold: f64,
    base_edge_threshold: f64,
    hedge_cost_edge_floor: f64,
    utility_extra_edge_usd: f64,
    utility_tier: &'static str,
    utility_reason: &'static str,
    utility_role: &'static str,
    utility_size_multiplier: f64,
    candidate_price_pre_utility: f64,
    candidate_edge_pre_utility: f64,
    candidate_size_pre_margin: f64,
    candidate_size_post_margin: f64,
    candidate_size_pre_utility: f64,
    candidate_size_post_utility: f64,
    candidate_size_post_taper: f64,
    candidate_notional_post_utility: f64,
    lot_size_tao: f64,
}

#[derive(Debug, Clone, Copy)]
struct QuotePruneDetails {
    suppression_reason: Option<&'static str>,
    prune_stage: &'static str,
    candidate_price_pre_utility: f64,
    candidate_edge_pre_utility: f64,
    candidate_size_pre_margin: f64,
    candidate_size_post_margin: f64,
    candidate_size_pre_utility: f64,
    candidate_size_post_utility: f64,
    candidate_size_post_taper: f64,
    candidate_notional_post_utility: f64,
}

fn mm_maker_cost(vcfg: &crate::config::VenueConfig, price: f64) -> f64 {
    let maker_fee = vcfg.maker_fee_bps / 10_000.0;
    let maker_rebate = vcfg.maker_rebate_bps / 10_000.0;
    (maker_fee - maker_rebate).max(0.0) * price
}

fn mm_hedge_cost_edge_floor(cfg: &Config, state: &GlobalState, fallback_price: f64) -> f64 {
    let mult = cfg.mm.hedge_cost_edge_mult.max(0.0);
    if mult <= 0.0 {
        return 0.0;
    }

    let mut best_cost = f64::INFINITY;
    for (i, vcfg) in cfg.venues.iter().enumerate() {
        if !vcfg.is_hedge_allowed {
            continue;
        }
        let Some(vstate) = state.venues.get(i) else {
            continue;
        };
        if !matches!(vstate.status, VenueStatus::Healthy) {
            continue;
        }
        let mid = vstate.mid.unwrap_or(fallback_price);
        if !mid.is_finite() || mid <= 0.0 {
            continue;
        }
        let spread = vstate.spread.unwrap_or(0.0).max(0.0);
        let taker_fee = (vcfg.taker_fee_bps / 10_000.0).abs() * mid;
        let cost = 0.5 * spread + taker_fee + cfg.hedge.slippage_buffer.max(0.0);
        if cost.is_finite() && cost >= 0.0 {
            best_cost = best_cost.min(cost);
        }
    }

    if best_cost.is_finite() {
        mult * best_cost
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn quote_diagnostics(
    cfg: &Config,
    state: &GlobalState,
    vcfg: &crate::config::VenueConfig,
    v: &crate::state::VenueState,
    fair: f64,
    now_ms: TimestampMs,
    side: Side,
    q_target_v: f64,
    utility: VenueUtilityDecision,
    maker_cost: f64,
    price: f64,
    size_final: f64,
    size_eta: f64,
    generated_spread_cap_applied: bool,
    generated_spread_cap_bid_suppressed: bool,
    generated_spread_cap_ask_suppressed: bool,
) -> QuoteTelemetryDiagnostics {
    let book_age_ms = compute_age_ms(now_ms, v.last_mid_update_ms);
    let book_stale_threshold_ms = cfg
        .mm
        .quote_max_age_ms
        .unwrap_or_else(|| vcfg.effective_stale_ms(cfg.book.stale_ms));
    let lot_size_tao = vcfg.lot_size_tao.max(0.0);
    let bid_edge_threshold = cfg.mm.edge_local_min_bid_for(&vcfg.id);
    let ask_edge_threshold = cfg.mm.edge_local_min_ask_for(&vcfg.id);
    let base_edge_threshold = match side {
        Side::Buy => bid_edge_threshold,
        Side::Sell => ask_edge_threshold,
    };
    let hedge_cost_edge_floor = mm_hedge_cost_edge_floor(cfg, state, fair);
    let edge_threshold = base_edge_threshold + hedge_cost_edge_floor + utility.extra_edge_usd;
    let reference_price = if price > 0.0 {
        price
    } else {
        v.mid.unwrap_or(fair).max(1e-9)
    };
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
    let margin_cap_estimate = if reference_price > 0.0 {
        (v.margin_available_usd * cfg.risk.mm_max_leverage * cfg.risk.mm_margin_safety)
            / reference_price
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
    let quote_state = if price > 0.0 && size_final > 0.0 {
        "active"
    } else {
        "suppressed"
    };
    let prune = quote_prune_details(
        cfg,
        state,
        vcfg,
        v,
        side,
        now_ms,
        q_target_v,
        utility,
        bid_edge_threshold,
        ask_edge_threshold,
        lot_size_tao,
        margin_cap_estimate,
        liq_factor,
        generated_spread_cap_applied,
        generated_spread_cap_bid_suppressed,
        generated_spread_cap_ask_suppressed,
    );

    QuoteTelemetryDiagnostics {
        edge,
        q_raw,
        size_final,
        margin_cap,
        liq_factor,
        quote_state,
        suppression_reason: prune.suppression_reason,
        prune_stage: if quote_state == "active" {
            "active"
        } else {
            prune.prune_stage
        },
        book_age_ms,
        book_stale_threshold_ms,
        edge_threshold,
        base_edge_threshold,
        hedge_cost_edge_floor,
        utility_extra_edge_usd: utility.extra_edge_usd,
        utility_tier: utility.tier.as_str(),
        utility_reason: utility.reason.as_str(),
        utility_role: utility.role.as_str(),
        utility_size_multiplier: utility.size_multiplier,
        candidate_price_pre_utility: prune.candidate_price_pre_utility,
        candidate_edge_pre_utility: prune.candidate_edge_pre_utility,
        candidate_size_pre_margin: prune.candidate_size_pre_margin,
        candidate_size_post_margin: prune.candidate_size_post_margin,
        candidate_size_pre_utility: prune.candidate_size_pre_utility,
        candidate_size_post_utility: prune.candidate_size_post_utility,
        candidate_size_post_taper: prune.candidate_size_post_taper,
        candidate_notional_post_utility: prune.candidate_notional_post_utility,
        lot_size_tao,
    }
}

#[inline]
fn telemetry_scale_size(
    size: f64,
    factor: f64,
    price: f64,
    vcfg: &crate::config::VenueConfig,
) -> Option<f64> {
    if factor >= 1.0 {
        return Some(size);
    }
    let scaled = (size * factor).min(vcfg.max_order_size).max(0.0);
    if scaled < vcfg.lot_size_tao || scaled * price < vcfg.min_notional_usd {
        None
    } else {
        Some(scaled)
    }
}

#[inline]
fn telemetry_inventory_reducing_side(q_global_tao: f64, venue_position_tao: f64) -> Option<Side> {
    let signed_inventory = if q_global_tao.abs() > 1e-9 {
        q_global_tao
    } else {
        venue_position_tao
    };
    if signed_inventory > 0.0 {
        Some(Side::Sell)
    } else if signed_inventory < 0.0 {
        Some(Side::Buy)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_update)]
fn quote_prune_details(
    cfg: &Config,
    state: &GlobalState,
    vcfg: &crate::config::VenueConfig,
    v: &crate::state::VenueState,
    side: Side,
    now_ms: TimestampMs,
    q_target_v: f64,
    utility: VenueUtilityDecision,
    bid_edge_threshold: f64,
    ask_edge_threshold: f64,
    lot_size_tao: f64,
    margin_cap_estimate: f64,
    liq_factor: f64,
    _generated_spread_cap_applied: bool,
    generated_spread_cap_bid_suppressed: bool,
    generated_spread_cap_ask_suppressed: bool,
) -> QuotePruneDetails {
    let blank = QuotePruneDetails {
        suppression_reason: None,
        prune_stage: "active",
        candidate_price_pre_utility: 0.0,
        candidate_edge_pre_utility: 0.0,
        candidate_size_pre_margin: 0.0,
        candidate_size_post_margin: 0.0,
        candidate_size_pre_utility: 0.0,
        candidate_size_post_utility: 0.0,
        candidate_size_post_taper: 0.0,
        candidate_notional_post_utility: 0.0,
    };
    if !state.fv_available {
        return QuotePruneDetails {
            suppression_reason: Some("global_fv_unavailable"),
            prune_stage: "global_fv_unavailable",
            ..blank
        };
    }
    if state.fair_value.is_none() {
        return QuotePruneDetails {
            suppression_reason: Some("fair_value_missing"),
            prune_stage: "fair_value_missing",
            ..blank
        };
    }
    if state.kill_switch {
        return QuotePruneDetails {
            suppression_reason: Some("kill_switch"),
            prune_stage: "kill_switch",
            ..blank
        };
    }
    if matches!(state.risk_regime, RiskRegime::HardLimit) {
        return QuotePruneDetails {
            suppression_reason: Some("risk_hard_limit"),
            prune_stage: "risk_hard_limit",
            ..blank
        };
    }
    if !state.sigma_eff.is_finite() || state.sigma_eff <= 0.0 || state.size_mult <= 0.0 {
        return QuotePruneDetails {
            suppression_reason: Some("degenerate_global_scalars"),
            prune_stage: "degenerate_global_scalars",
            ..blank
        };
    }
    if matches!(v.status, VenueStatus::Disabled) {
        return QuotePruneDetails {
            suppression_reason: Some("venue_disabled"),
            prune_stage: "venue_disabled",
            ..blank
        };
    }
    if v.last_mid_update_ms.is_none() {
        return QuotePruneDetails {
            suppression_reason: Some("book_missing"),
            prune_stage: "book_missing",
            ..blank
        };
    }
    let quote_max_age_ms = cfg
        .mm
        .quote_max_age_ms
        .unwrap_or_else(|| vcfg.effective_stale_ms(cfg.book.stale_ms));
    if compute_age_ms(now_ms, v.last_mid_update_ms) > quote_max_age_ms {
        return QuotePruneDetails {
            suppression_reason: Some("book_stale"),
            prune_stage: "book_stale",
            ..blank
        };
    }
    if v.mid.is_none() {
        return QuotePruneDetails {
            suppression_reason: Some("mid_missing"),
            prune_stage: "mid_missing",
            ..blank
        };
    }
    if v.spread.unwrap_or(0.0) <= 0.0 || v.depth_near_mid <= 0.0 {
        return QuotePruneDetails {
            suppression_reason: Some("invalid_book"),
            prune_stage: "invalid_book",
            ..blank
        };
    }
    if v.dist_liq_sigma <= cfg.risk.liq_crit_sigma {
        return QuotePruneDetails {
            suppression_reason: Some("liq_critical"),
            prune_stage: "liq_critical",
            ..blank
        };
    }
    if v.toxicity >= cfg.toxicity.tox_high_threshold {
        return QuotePruneDetails {
            suppression_reason: Some("toxicity_high"),
            prune_stage: "toxicity_high",
            ..blank
        };
    }
    let delta_ratio = (state.dollar_delta_usd.abs() / state.delta_limit_usd.max(1.0)).max(0.0);
    if delta_ratio >= 2.0 {
        return QuotePruneDetails {
            suppression_reason: Some("delta_limit_block"),
            prune_stage: "delta_limit_block",
            ..blank
        };
    }
    if delta_ratio > 1.0 {
        if state.q_global_tao > 0.0 && matches!(side, Side::Buy) {
            return QuotePruneDetails {
                suppression_reason: Some("delta_directional_block_long"),
                prune_stage: "delta_directional_block_long",
                ..blank
            };
        }
        if state.q_global_tao < 0.0 && matches!(side, Side::Sell) {
            return QuotePruneDetails {
                suppression_reason: Some("delta_directional_block_short"),
                prune_stage: "delta_directional_block_short",
                ..blank
            };
        }
    }
    if margin_cap_estimate > 0.0 && margin_cap_estimate < lot_size_tao {
        return QuotePruneDetails {
            suppression_reason: Some("margin_cap_below_lot"),
            prune_stage: "margin_cap_below_lot",
            ..blank
        };
    }
    if liq_factor > 0.0 && (liq_factor * vcfg.max_order_size) < lot_size_tao {
        return QuotePruneDetails {
            suppression_reason: Some("liq_shrink_below_lot"),
            prune_stage: "liq_shrink_below_lot",
            ..blank
        };
    }
    if matches!(state.risk_regime, RiskRegime::Warning)
        && cfg.risk.q_warn_cap > 0.0
        && cfg.risk.q_warn_cap < lot_size_tao
    {
        return QuotePruneDetails {
            suppression_reason: Some("warning_cap_below_lot"),
            prune_stage: "warning_cap_below_lot",
            ..blank
        };
    }
    if matches!(utility.tier, VenueUtilityTier::Suppressed) {
        return QuotePruneDetails {
            suppression_reason: Some("utility_suppressed"),
            prune_stage: "utility_suppressed",
            ..blank
        };
    }

    let fair = state.fair_value.unwrap_or(state.fair_value_prev).max(1.0);
    let sigma_eff = state.sigma_eff.max(cfg.volatility.sigma_min).max(1e-8);
    let tau = cfg.mm.quote_horizon_sec.max(1.0);
    let funding_8h =
        funding_rate_for_decision(&v.funding_state, now_ms, &cfg.funding, true).unwrap_or(0.0);
    let funding_pnl_per_unit = funding_8h * (tau / (8.0 * 60.0 * 60.0)) * fair;
    let basis_v = v.mid.unwrap_or(fair) - fair;
    let basis_adj = cfg.mm.basis_weight * basis_v;
    let funding_adj = cfg.mm.funding_weight * funding_pnl_per_unit;
    let inv_deviation = state.q_global_tao - cfg.mm.lambda_inv * (v.position_tao - q_target_v);
    let inventory_term = vcfg.gamma * sigma_eff * sigma_eff * tau * inv_deviation;
    let reservation_price = fair + basis_adj + funding_adj - inventory_term;
    let spread = v.spread.unwrap_or(0.0);
    let best_bid = v.mid.unwrap_or(fair) - spread / 2.0;
    let best_ask = v.mid.unwrap_or(fair) + spread / 2.0;
    let tick = vcfg.tick_size.max(1e-6);
    let maker_cost = mm_maker_cost(vcfg, fair);
    let vol_buffer = cfg.mm.edge_vol_mult * sigma_eff * fair;
    let hedge_cost_edge_floor = mm_hedge_cost_edge_floor(cfg, state, fair);
    let effective_bid_edge_threshold =
        bid_edge_threshold + hedge_cost_edge_floor + utility.extra_edge_usd;
    let effective_ask_edge_threshold =
        ask_edge_threshold + hedge_cost_edge_floor + utility.extra_edge_usd;
    let gamma = vcfg.gamma.max(1e-8);
    let k = vcfg.k.max(1e-8);
    let delta_as = (1.0 / gamma) * (1.0 + (gamma / k)).ln();
    let delta_vol = delta_as * state.spread_mult.max(1e-6);
    let mut bid_half_spread = delta_vol
        .max((effective_bid_edge_threshold + maker_cost + vol_buffer) / 2.0)
        .max(0.0);
    let mut ask_half_spread = delta_vol
        .max((effective_ask_edge_threshold + maker_cost + vol_buffer) / 2.0)
        .max(0.0);
    if matches!(state.risk_regime, RiskRegime::Warning) {
        let warn_mult = cfg.risk.spread_warn_mult.max(1.0);
        bid_half_spread *= warn_mult;
        ask_half_spread *= warn_mult;
    }
    if v.dist_liq_sigma > 0.0 && v.dist_liq_sigma < cfg.risk.liq_warn_sigma {
        let t = ((cfg.risk.liq_warn_sigma - v.dist_liq_sigma) / cfg.risk.liq_warn_sigma)
            .clamp(0.0, 1.0);
        let liq_mult = 1.0 + 2.0 * t;
        bid_half_spread *= liq_mult;
        ask_half_spread *= liq_mult;
    }
    let raw_bid = reservation_price - bid_half_spread;
    let raw_ask = reservation_price + ask_half_spread;
    let passive_bid_limit = best_bid - tick;
    let passive_ask_limit = best_ask + tick;
    let snapped_bid = {
        let mut px = (raw_bid.min(passive_bid_limit) / tick).floor() * tick;
        if px > passive_bid_limit {
            px = (passive_bid_limit / tick).floor() * tick;
        }
        px
    };
    let snapped_ask = {
        let mut px = (raw_ask.max(passive_ask_limit) / tick).ceil() * tick;
        if px < passive_ask_limit {
            px = (passive_ask_limit / tick).ceil() * tick;
        }
        px
    };
    let candidate_price = match side {
        Side::Buy => snapped_bid,
        Side::Sell => snapped_ask,
    };
    let edge_bid = fair - snapped_bid - maker_cost;
    let edge_ask = snapped_ask - fair - maker_cost;
    let candidate_edge = match side {
        Side::Buy => edge_bid,
        Side::Sell => edge_ask,
    };
    let side_edge_ok = match side {
        Side::Buy => edge_bid >= effective_bid_edge_threshold,
        Side::Sell => edge_ask >= effective_ask_edge_threshold,
    };
    if !side_edge_ok {
        return QuotePruneDetails {
            suppression_reason: Some("edge_below_min"),
            prune_stage: "edge_fail_pre_utility",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            ..blank
        };
    }

    let q_raw_bid = if edge_bid > 0.0 {
        edge_bid / cfg.mm.size_eta.max(1e-9)
    } else {
        0.0
    };
    let q_raw_ask = if edge_ask > 0.0 {
        edge_ask / cfg.mm.size_eta.max(1e-9)
    } else {
        0.0
    };
    let mut size_bid = q_raw_bid * state.size_mult.max(0.0);
    let mut size_ask = q_raw_ask * state.size_mult.max(0.0);
    size_bid = size_bid.max(vcfg.lot_size_tao);
    size_ask = size_ask.max(vcfg.lot_size_tao);
    let candidate_size_pre_margin = match side {
        Side::Buy => size_bid,
        Side::Sell => size_ask,
    };

    let mm_margin_factor = cfg.risk.mm_max_leverage * cfg.risk.mm_margin_safety;
    let margin_cap_bid = if snapped_bid > 0.0 {
        (v.margin_available_usd * mm_margin_factor) / snapped_bid
    } else {
        0.0
    };
    let margin_cap_ask = if snapped_ask > 0.0 {
        (v.margin_available_usd * mm_margin_factor) / snapped_ask
    } else {
        0.0
    };
    size_bid = size_bid.min(margin_cap_bid);
    size_ask = size_ask.min(margin_cap_ask);
    let candidate_size_post_margin = match side {
        Side::Buy => size_bid,
        Side::Sell => size_ask,
    };

    if candidate_size_post_margin < lot_size_tao {
        return QuotePruneDetails {
            suppression_reason: Some("margin_cap_below_lot"),
            prune_stage: "margin_cap_below_lot",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            ..blank
        };
    }
    if candidate_size_post_margin * candidate_price < vcfg.min_notional_usd {
        return QuotePruneDetails {
            suppression_reason: Some("margin_cap_below_min_notional"),
            prune_stage: "margin_cap_below_min_notional",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            ..blank
        };
    }

    if matches!(v.status, VenueStatus::Warning) {
        size_bid *= 0.5;
        size_ask *= 0.5;
    }
    if v.toxicity >= cfg.toxicity.tox_med_threshold && v.toxicity < cfg.toxicity.tox_high_threshold
    {
        let tox_factor = 1.0 - (v.toxicity - cfg.toxicity.tox_med_threshold) / (0.3_f64).max(1e-6);
        size_bid *= tox_factor.clamp(0.1, 1.0);
        size_ask *= tox_factor.clamp(0.1, 1.0);
    }
    if v.dist_liq_sigma < cfg.risk.liq_warn_sigma {
        let k_liq = ((v.dist_liq_sigma - cfg.risk.liq_crit_sigma)
            / (cfg.risk.liq_warn_sigma - cfg.risk.liq_crit_sigma + 1e-9))
            .clamp(0.0, 1.0);
        size_bid *= k_liq;
        size_ask *= k_liq;
    }
    if delta_ratio > 1.0 {
        let factor = (2.0 - delta_ratio).clamp(0.0, 1.0);
        if state.q_global_tao > 0.0 {
            size_bid = 0.0;
            size_ask *= factor;
        } else if state.q_global_tao < 0.0 {
            size_ask = 0.0;
            size_bid *= factor;
        } else {
            size_bid *= factor;
            size_ask *= factor;
        }
    }
    if matches!(state.risk_regime, RiskRegime::Warning) {
        size_bid = size_bid.min(cfg.risk.q_warn_cap.max(0.0));
        size_ask = size_ask.min(cfg.risk.q_warn_cap.max(0.0));
    }
    size_bid = size_bid.min(vcfg.max_order_size);
    size_ask = size_ask.min(vcfg.max_order_size);
    let lot = vcfg.lot_size_tao.max(1e-9);
    size_bid = (size_bid / lot).floor() * lot;
    size_ask = (size_ask / lot).floor() * lot;
    let generated_spread_cap_side_suppressed = match side {
        Side::Buy => generated_spread_cap_bid_suppressed,
        Side::Sell => generated_spread_cap_ask_suppressed,
    };
    if generated_spread_cap_side_suppressed {
        match side {
            Side::Buy => size_bid = 0.0,
            Side::Sell => size_ask = 0.0,
        }
    }
    let candidate_size_pre_utility = match side {
        Side::Buy => size_bid,
        Side::Sell => size_ask,
    };
    if generated_spread_cap_side_suppressed && candidate_size_pre_utility <= 0.0 {
        return QuotePruneDetails {
            suppression_reason: Some("generated_spread_cap"),
            prune_stage: "generated_spread_cap",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            candidate_size_pre_utility,
            ..blank
        };
    }
    if candidate_size_pre_utility < lot_size_tao
        || candidate_size_pre_utility * candidate_price < vcfg.min_notional_usd
    {
        return QuotePruneDetails {
            suppression_reason: Some("size_or_passivity_gated"),
            prune_stage: "size_or_passivity_gated",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            candidate_size_pre_utility,
            ..blank
        };
    }

    let inventory_reducing_side =
        telemetry_inventory_reducing_side(state.q_global_tao, v.position_tao);
    let mut candidate_size_post_utility = candidate_size_pre_utility;
    match utility.tier {
        VenueUtilityTier::Full => {}
        VenueUtilityTier::Reduced => {
            candidate_size_post_utility = telemetry_scale_size(
                candidate_size_post_utility,
                utility.size_multiplier,
                candidate_price,
                vcfg,
            )
            .unwrap_or(0.0);
        }
        VenueUtilityTier::AnchorOnly => {
            candidate_size_post_utility = telemetry_scale_size(
                candidate_size_post_utility,
                utility.size_multiplier,
                candidate_price,
                vcfg,
            )
            .unwrap_or(0.0);
            let allowed = match inventory_reducing_side {
                Some(Side::Buy) => matches!(side, Side::Buy),
                Some(Side::Sell) => matches!(side, Side::Sell),
                None => false,
            };
            if !allowed {
                return QuotePruneDetails {
                    suppression_reason: Some("size_or_passivity_gated"),
                    prune_stage: "anchor_inventory_gate",
                    candidate_price_pre_utility: candidate_price,
                    candidate_edge_pre_utility: candidate_edge,
                    candidate_size_pre_margin,
                    candidate_size_post_margin,
                    candidate_size_pre_utility,
                    candidate_size_post_utility: 0.0,
                    candidate_size_post_taper: 0.0,
                    candidate_notional_post_utility: 0.0,
                    ..blank
                };
            }
        }
        VenueUtilityTier::Suppressed => {}
    }
    if candidate_size_post_utility < lot_size_tao {
        return QuotePruneDetails {
            suppression_reason: Some("size_or_passivity_gated"),
            prune_stage: "utility_scale_below_lot",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            candidate_size_pre_utility,
            candidate_size_post_utility,
            ..blank
        };
    }
    if candidate_size_post_utility * candidate_price < vcfg.min_notional_usd {
        return QuotePruneDetails {
            suppression_reason: Some("size_or_passivity_gated"),
            prune_stage: "utility_scale_below_min_notional",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            candidate_size_pre_utility,
            candidate_size_post_utility,
            candidate_notional_post_utility: candidate_size_post_utility * candidate_price,
            ..blank
        };
    }

    let candidate_size_post_taper = if utility.pre_soft_taper_active {
        let taper_matches_side = matches!(
            (utility.pre_soft_taper_side, side),
            (Some(Side::Buy), Side::Buy) | (Some(Side::Sell), Side::Sell)
        );
        if taper_matches_side {
            telemetry_scale_size(
                candidate_size_post_utility,
                utility.pre_soft_taper_size_multiplier,
                candidate_price,
                vcfg,
            )
            .unwrap_or(0.0)
        } else {
            candidate_size_post_utility
        }
    } else {
        candidate_size_post_utility
    };
    if candidate_size_post_taper < lot_size_tao {
        return QuotePruneDetails {
            suppression_reason: Some("size_or_passivity_gated"),
            prune_stage: "pre_soft_taper_below_lot",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            candidate_size_pre_utility,
            candidate_size_post_utility,
            candidate_size_post_taper,
            ..blank
        };
    }
    if candidate_size_post_taper * candidate_price < vcfg.min_notional_usd {
        return QuotePruneDetails {
            suppression_reason: Some("size_or_passivity_gated"),
            prune_stage: "pre_soft_taper_below_min_notional",
            candidate_price_pre_utility: candidate_price,
            candidate_edge_pre_utility: candidate_edge,
            candidate_size_pre_margin,
            candidate_size_post_margin,
            candidate_size_pre_utility,
            candidate_size_post_utility,
            candidate_size_post_taper,
            candidate_notional_post_utility: candidate_size_post_taper * candidate_price,
            ..blank
        };
    }

    QuotePruneDetails {
        suppression_reason: None,
        prune_stage: "active",
        candidate_price_pre_utility: candidate_price,
        candidate_edge_pre_utility: candidate_edge,
        candidate_size_pre_margin,
        candidate_size_post_margin,
        candidate_size_pre_utility,
        candidate_size_post_utility,
        candidate_size_post_taper,
        candidate_notional_post_utility: candidate_size_post_taper * candidate_price,
    }
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
        let decision_id =
            lookup_order_decision_id(state, order_id.as_deref(), client_order_id.as_deref());
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
            "action_id": action_id,
            "decision_id": decision_id,
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
                let decision_id = lookup_order_decision_id(
                    state,
                    Some(&ack.order_id),
                    ack.client_order_id.as_deref(),
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
                    "action_id": action_id,
                    "decision_id": decision_id,
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
                let decision_id = lookup_order_decision_id(
                    state,
                    rej.order_id.as_deref(),
                    rej.client_order_id.as_deref(),
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
                    "reason": rej.reason,
                    "action_id": action_id,
                    "decision_id": decision_id,
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
    _client_order_id: Option<&String>,
    _order_id: Option<&String>,
) -> String {
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
    source_fill_cache: &VecDeque<SourceFillAttribution>,
) -> Vec<JsonValue> {
    let mut out = Vec::new();
    for fill in fills {
        let record = find_fill_record(state, fill, now_ms);
        let decision_id = lookup_order_decision_id(
            state,
            fill.order_id.as_deref(),
            fill.client_order_id.as_deref(),
        );
        let hedge_source = matches!(fill.purpose, OrderPurpose::Hedge).then(|| {
            let source =
                attribute_hedge_source_decision(state, fill.side, now_ms, source_fill_cache);
            if source.source_decision_id.is_some() {
                source
            } else {
                residual_hedge_fill_source_attribution(fill)
            }
        });
        out.push(serde_json::json!({
            "fill_seq": record.as_ref().map(|r| r.fill_seq),
            "venue_index": fill.venue_index as i64,
            "venue_id": fill.venue_id.as_ref(),
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
            "decision_id": decision_id,
            "source_decision_id": hedge_source
                .as_ref()
                .and_then(|source| source.source_decision_id.clone()),
            "source_fill_venue_index": hedge_source
                .as_ref()
                .and_then(|source| source.source_fill_venue_index)
                .map(|venue_index| venue_index as i64),
            "source_fill_venue_id": hedge_source
                .as_ref()
                .and_then(|source| source.source_fill_venue_id.clone()),
            "source_fill_age_ms": hedge_source
                .as_ref()
                .and_then(|source| source.source_fill_age_ms),
            "source_kind": hedge_source
                .as_ref()
                .and_then(|source| source.source_kind),
        }));
    }
    out.sort_by_key(fill_sort_key);
    out
}

#[cfg(feature = "live")]
fn lookup_order_decision_id(
    state: &GlobalState,
    order_id: Option<&str>,
    client_order_id: Option<&str>,
) -> Option<String> {
    state
        .live_order_state
        .decision_id_for_order(order_id, client_order_id)
        .map(str::to_string)
}

#[cfg(not(feature = "live"))]
fn lookup_order_decision_id(
    _state: &GlobalState,
    _order_id: Option<&str>,
    _client_order_id: Option<&str>,
) -> Option<String> {
    None
}

#[derive(Debug, Clone, Default)]
struct HedgeSourceAttribution {
    source_decision_id: Option<String>,
    source_fill_venue_index: Option<usize>,
    source_fill_venue_id: Option<String>,
    source_fill_age_ms: Option<TimestampMs>,
    source_kind: Option<&'static str>,
}

fn attribute_hedge_source_decision(
    state: &GlobalState,
    hedge_side: Side,
    now_ms: TimestampMs,
    source_fill_cache: &VecDeque<SourceFillAttribution>,
) -> HedgeSourceAttribution {
    let mm_fill_side = match hedge_side {
        Side::Buy => Side::Sell,
        Side::Sell => Side::Buy,
    };
    let mut best: Option<(TimestampMs, f64, HedgeSourceAttribution)> = None;

    for fill in source_fill_cache.iter().rev() {
        if fill.side != mm_fill_side {
            continue;
        }
        let source_fill_age_ms = Some((now_ms - fill.fill_time_ms).max(0));
        let candidate = HedgeSourceAttribution {
            source_decision_id: Some(fill.source_decision_id.clone()),
            source_fill_venue_index: Some(fill.venue_index),
            source_fill_venue_id: Some(fill.venue_id.clone()),
            source_fill_age_ms,
            source_kind: Some("mm_fill_cache"),
        };
        let replace_best = match &best {
            None => true,
            Some((best_fill_time_ms, best_size, _)) => {
                fill.fill_time_ms > *best_fill_time_ms
                    || (fill.fill_time_ms == *best_fill_time_ms && fill.size > *best_size)
            }
        };
        if replace_best {
            best = Some((fill.fill_time_ms, fill.size, candidate));
        }
    }

    for (venue_index, venue) in state.venues.iter().enumerate() {
        for fill in venue.recent_fills.iter().rev() {
            if !matches!(fill.purpose, OrderPurpose::Mm) || fill.side != mm_fill_side {
                continue;
            }
            let Some(source_decision_id) = lookup_order_decision_id(
                state,
                fill.order_id.as_deref(),
                fill.client_order_id.as_deref(),
            ) else {
                continue;
            };
            let source_fill_age_ms = Some((now_ms - fill.fill_time_ms).max(0));
            let candidate = HedgeSourceAttribution {
                source_decision_id: Some(source_decision_id),
                source_fill_venue_index: Some(venue_index),
                source_fill_venue_id: Some(state.venues[venue_index].id.to_string()),
                source_fill_age_ms,
                source_kind: Some("mm_recent_fill"),
            };
            let replace_best = match &best {
                None => true,
                Some((best_fill_time_ms, best_size, _)) => {
                    fill.fill_time_ms > *best_fill_time_ms
                        || (fill.fill_time_ms == *best_fill_time_ms && fill.size > *best_size)
                }
            };
            if replace_best {
                best = Some((fill.fill_time_ms, fill.size, candidate));
            }
        }
    }

    best.map(|(_, _, attribution)| attribution)
        .unwrap_or_else(|| residual_hedge_source_attribution(state, hedge_side))
}

fn residual_hedge_source_attribution(
    state: &GlobalState,
    hedge_side: Side,
) -> HedgeSourceAttribution {
    const EPS: f64 = 1e-9;
    let residual_direction = match hedge_side {
        Side::Buy => "short",
        Side::Sell => "long",
    };
    let matching_position = |position_tao: f64| match hedge_side {
        Side::Buy => position_tao < -EPS,
        Side::Sell => position_tao > EPS,
    };
    let best = state
        .venues
        .iter()
        .enumerate()
        .filter(|(_, venue)| matching_position(venue.position_tao))
        .max_by(|(_, lhs), (_, rhs)| {
            lhs.position_tao
                .abs()
                .partial_cmp(&rhs.position_tao.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let Some((venue_index, venue)) = best else {
        return HedgeSourceAttribution::default();
    };
    HedgeSourceAttribution {
        source_decision_id: Some(format!("residual:{}:{}", venue.id, residual_direction)),
        source_fill_venue_index: Some(venue_index),
        source_fill_venue_id: Some(venue.id.to_string()),
        source_fill_age_ms: None,
        source_kind: Some("residual_position"),
    }
}

fn residual_hedge_fill_source_attribution(fill: &FillEvent) -> HedgeSourceAttribution {
    let residual_direction = match fill.side {
        Side::Buy => "short",
        Side::Sell => "long",
    };
    HedgeSourceAttribution {
        source_decision_id: Some(format!(
            "residual:{}:{}",
            fill.venue_id.as_ref(),
            residual_direction
        )),
        source_fill_venue_index: Some(fill.venue_index),
        source_fill_venue_id: Some(fill.venue_id.to_string()),
        source_fill_age_ms: None,
        source_kind: Some("residual_fill"),
    }
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
    source_fill_cache: &VecDeque<SourceFillAttribution>,
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
            let source =
                attribute_hedge_source_decision(state, pi.side, _now_ms, source_fill_cache);
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
                "source_decision_id": source.source_decision_id,
                "source_fill_venue_index": source.source_fill_venue_index.map(|venue_index| venue_index as i64),
                "source_fill_venue_id": source.source_fill_venue_id,
                "source_fill_age_ms": source.source_fill_age_ms,
                "source_kind": source.source_kind,
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
    venue_health_diagnostics: &[VenueHealthDiagnostics],
) -> Vec<(String, JsonValue)> {
    // Detect fixture mode to disable staleness override.
    let fixture_mode = is_fixture_mode(state, now_ms);
    // Use effective_now for age calculation in fixture mode.
    let effective_now = effective_now_for_staleness(state, now_ms);
    let generated_quotes = compute_mm_quotes_with_now(cfg, state, Some(now_ms));
    #[allow(clippy::type_complexity)]
    let mut generated_quote_by_venue: Vec<(
        Option<f64>,
        Option<f64>,
        bool,
        bool,
        bool,
        Option<&'static str>,
    )> = vec![(None, None, false, false, false, None); cfg.venues.len()];
    for quote in generated_quotes {
        generated_quote_by_venue[quote.venue_index] = (
            quote.bid.as_ref().map(|level| level.price),
            quote.ask.as_ref().map(|level| level.price),
            quote.generated_spread_cap_applied,
            quote.generated_spread_cap_bid_suppressed,
            quote.generated_spread_cap_ask_suppressed,
            quote.touch_mode_kind,
        );
    }

    let mut venue_mid = Vec::new();
    let mut venue_spread = Vec::new();
    let mut venue_quote_spread_gate_reason = Vec::new();
    let mut venue_generated_quote_spread_bps = Vec::new();
    let mut venue_generated_quote_spread_cap_applied = Vec::new();
    let mut venue_generated_quote_spread_cap_bid_suppressed = Vec::new();
    let mut venue_generated_quote_spread_cap_ask_suppressed = Vec::new();
    let mut venue_generated_quote_spread_cap_mode = Vec::new();
    let mut venue_bid_distance_to_touch_bps = Vec::new();
    let mut venue_ask_distance_to_touch_bps = Vec::new();
    let mut venue_touch_mode_applied = Vec::new();
    let mut venue_touch_mode_kind = Vec::new();
    let mut venue_touch_clip_applied = Vec::new();
    let mut venue_touch_offset_ticks = Vec::new();
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
    let mut venue_utility_score = Vec::new();
    let mut venue_utility_tier = Vec::new();
    let mut venue_utility_reason = Vec::new();
    let mut venue_utility_role = Vec::new();
    let mut venue_utility_role_cap_applied = Vec::new();
    let mut venue_utility_mm_ack_ewma = Vec::new();
    let mut venue_utility_mm_fill_count_ewma = Vec::new();
    let mut venue_utility_mm_fillless_ack_pressure = Vec::new();
    let mut venue_utility_spread_gate_hit_ewma = Vec::new();
    let mut venue_pre_soft_taper_active = Vec::new();
    let mut venue_pre_soft_taper_side = Vec::new();
    let mut venue_pre_soft_taper_reason = Vec::new();
    let mut venue_pre_soft_taper_size_multiplier = Vec::new();
    let mut venue_toxicity_bootstrap_pending = Vec::new();
    let mut venue_health_api_errors = Vec::new();
    let mut venue_health_stale_count = Vec::new();
    let mut venue_health_dev_breaches = Vec::new();
    let mut venue_health_disable_reason = Vec::new();
    let mut venue_health_last_error_source = Vec::new();
    let mut venue_health_last_error_message = Vec::new();

    for (idx, venue) in state.venues.iter().enumerate() {
        let health_diag = venue_health_diagnostics
            .get(idx)
            .cloned()
            .unwrap_or_default();
        venue_mid.push(venue.mid.unwrap_or(0.0));
        venue_spread.push(venue.spread.unwrap_or(0.0));
        venue_quote_spread_gate_reason.push(quote_spread_gate_reason(
            &cfg.mm,
            &cfg.venues[idx].id,
            venue.mid,
            venue.spread,
        ));
        let (
            generated_bid,
            generated_ask,
            generated_cap_applied,
            generated_cap_bid_suppressed,
            generated_cap_ask_suppressed,
            touch_mode_kind,
        ) = generated_quote_by_venue[idx];
        let generated_spread_bps = match (generated_bid, generated_ask, venue.mid) {
            (Some(bid), Some(ask), Some(mid))
                if bid.is_finite() && ask.is_finite() && mid > 0.0 =>
            {
                ((ask - bid) / mid) * 10_000.0
            }
            _ => 0.0,
        };
        venue_generated_quote_spread_bps.push(generated_spread_bps);
        venue_generated_quote_spread_cap_applied.push(generated_cap_applied);
        venue_generated_quote_spread_cap_bid_suppressed.push(generated_cap_bid_suppressed);
        venue_generated_quote_spread_cap_ask_suppressed.push(generated_cap_ask_suppressed);
        venue_generated_quote_spread_cap_mode.push(generated_spread_cap_mode(
            generated_cap_applied,
            generated_cap_bid_suppressed,
            generated_cap_ask_suppressed,
        ));
        let best_bid = venue
            .mid
            .zip(venue.spread)
            .map(|(mid, spread)| mid - spread / 2.0);
        let best_ask = venue
            .mid
            .zip(venue.spread)
            .map(|(mid, spread)| mid + spread / 2.0);
        let tick = cfg.venues[idx].tick_size.max(1e-6);
        let bid_distance_to_touch_bps = match (generated_bid, best_bid, venue.mid) {
            (Some(price), Some(best_bid), Some(mid)) if mid > 0.0 => {
                ((best_bid - price).max(0.0) / mid) * 10_000.0
            }
            _ => 0.0,
        };
        let ask_distance_to_touch_bps = match (generated_ask, best_ask, venue.mid) {
            (Some(price), Some(best_ask), Some(mid)) if mid > 0.0 => {
                ((price - best_ask).max(0.0) / mid) * 10_000.0
            }
            _ => 0.0,
        };
        let bid_touch_offset_ticks = match (generated_bid, best_bid) {
            (Some(price), Some(best_bid)) => ((best_bid - price).max(0.0) / tick).round(),
            _ => 0.0,
        };
        let ask_touch_offset_ticks = match (generated_ask, best_ask) {
            (Some(price), Some(best_ask)) => ((price - best_ask).max(0.0) / tick).round(),
            _ => 0.0,
        };
        venue_bid_distance_to_touch_bps.push(bid_distance_to_touch_bps);
        venue_ask_distance_to_touch_bps.push(ask_distance_to_touch_bps);
        venue_touch_mode_applied.push(touch_mode_kind.is_some());
        venue_touch_mode_kind.push(touch_mode_kind);
        venue_touch_clip_applied.push(matches!(touch_mode_kind, Some("hyperliquid_clip")));
        venue_touch_offset_ticks.push(bid_touch_offset_ticks.max(ask_touch_offset_ticks));
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
        let utility = compute_venue_utility_decision(
            &cfg.mm,
            state.q_global_tao,
            &cfg.venues[idx],
            venue,
            venue_utility_conversion_penalties_enabled(),
        );
        venue_utility_score.push(utility.score);
        venue_utility_tier.push(utility.tier.as_str());
        venue_utility_reason.push(utility.reason.as_str());
        venue_utility_role.push(utility.role.as_str());
        venue_utility_role_cap_applied.push(utility.role_cap_applied);
        venue_utility_mm_ack_ewma.push(venue.utility.mm_ack_ewma);
        venue_utility_mm_fill_count_ewma.push(venue.utility.mm_fill_count_ewma);
        venue_utility_mm_fillless_ack_pressure.push(venue.utility.mm_fillless_ack_pressure);
        venue_utility_spread_gate_hit_ewma.push(venue.utility.spread_gate_hit_ewma);
        venue_pre_soft_taper_active.push(utility.pre_soft_taper_active);
        venue_pre_soft_taper_side.push(match utility.pre_soft_taper_side {
            Some(Side::Buy) => "buy",
            Some(Side::Sell) => "sell",
            None => "",
        });
        venue_pre_soft_taper_reason.push(utility.pre_soft_taper_reason.unwrap_or(""));
        venue_pre_soft_taper_size_multiplier.push(utility.pre_soft_taper_size_multiplier);
        venue_toxicity_bootstrap_pending.push(
            venue.last_mid_apply_ms.is_none() && venue.mid.is_none() && venue.depth_near_mid <= 0.0,
        );
        venue_health_api_errors.push(health_diag.api_errors);
        venue_health_stale_count.push(health_diag.stale_count);
        venue_health_dev_breaches.push(health_diag.dev_breaches);
        venue_health_disable_reason.push(health_diag.disable_reason);
        venue_health_last_error_source.push(health_diag.last_error_source);
        venue_health_last_error_message.push(health_diag.last_error_message);
    }

    vec![
        ("venue_mid_usd".to_string(), serde_json::json!(venue_mid)),
        (
            "venue_spread_usd".to_string(),
            serde_json::json!(venue_spread),
        ),
        (
            "venue_quote_spread_gate_reason".to_string(),
            serde_json::json!(venue_quote_spread_gate_reason),
        ),
        (
            "venue_generated_quote_spread_bps".to_string(),
            serde_json::json!(venue_generated_quote_spread_bps),
        ),
        (
            "venue_generated_quote_spread_cap_applied".to_string(),
            serde_json::json!(venue_generated_quote_spread_cap_applied),
        ),
        (
            "venue_generated_quote_spread_cap_bid_suppressed".to_string(),
            serde_json::json!(venue_generated_quote_spread_cap_bid_suppressed),
        ),
        (
            "venue_generated_quote_spread_cap_ask_suppressed".to_string(),
            serde_json::json!(venue_generated_quote_spread_cap_ask_suppressed),
        ),
        (
            "venue_generated_quote_spread_cap_mode".to_string(),
            serde_json::json!(venue_generated_quote_spread_cap_mode),
        ),
        (
            "venue_bid_distance_to_touch_bps".to_string(),
            serde_json::json!(venue_bid_distance_to_touch_bps),
        ),
        (
            "venue_ask_distance_to_touch_bps".to_string(),
            serde_json::json!(venue_ask_distance_to_touch_bps),
        ),
        (
            "venue_touch_mode_applied".to_string(),
            serde_json::json!(venue_touch_mode_applied),
        ),
        (
            "venue_touch_mode_kind".to_string(),
            serde_json::json!(venue_touch_mode_kind),
        ),
        (
            "venue_touch_clip_applied".to_string(),
            serde_json::json!(venue_touch_clip_applied),
        ),
        (
            "venue_touch_offset_ticks".to_string(),
            serde_json::json!(venue_touch_offset_ticks),
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
        (
            "venue_utility_score".to_string(),
            serde_json::json!(venue_utility_score),
        ),
        (
            "venue_utility_tier".to_string(),
            serde_json::json!(venue_utility_tier),
        ),
        (
            "venue_utility_reason".to_string(),
            serde_json::json!(venue_utility_reason),
        ),
        (
            "venue_utility_role".to_string(),
            serde_json::json!(venue_utility_role),
        ),
        (
            "venue_utility_role_cap_applied".to_string(),
            serde_json::json!(venue_utility_role_cap_applied),
        ),
        (
            "venue_utility_mm_ack_ewma".to_string(),
            serde_json::json!(venue_utility_mm_ack_ewma),
        ),
        (
            "venue_utility_mm_fill_count_ewma".to_string(),
            serde_json::json!(venue_utility_mm_fill_count_ewma),
        ),
        (
            "venue_utility_mm_fillless_ack_pressure".to_string(),
            serde_json::json!(venue_utility_mm_fillless_ack_pressure),
        ),
        (
            "venue_utility_spread_gate_hit_ewma".to_string(),
            serde_json::json!(venue_utility_spread_gate_hit_ewma),
        ),
        (
            "venue_pre_soft_taper_active".to_string(),
            serde_json::json!(venue_pre_soft_taper_active),
        ),
        (
            "venue_pre_soft_taper_side".to_string(),
            serde_json::json!(venue_pre_soft_taper_side),
        ),
        (
            "venue_pre_soft_taper_reason".to_string(),
            serde_json::json!(venue_pre_soft_taper_reason),
        ),
        (
            "venue_pre_soft_taper_size_multiplier".to_string(),
            serde_json::json!(venue_pre_soft_taper_size_multiplier),
        ),
        (
            "venue_toxicity_bootstrap_pending".to_string(),
            serde_json::json!(venue_toxicity_bootstrap_pending),
        ),
        (
            "venue_health_api_errors".to_string(),
            serde_json::json!(venue_health_api_errors),
        ),
        (
            "venue_health_stale_count".to_string(),
            serde_json::json!(venue_health_stale_count),
        ),
        (
            "venue_health_dev_breaches".to_string(),
            serde_json::json!(venue_health_dev_breaches),
        ),
        (
            "venue_health_disable_reason".to_string(),
            serde_json::json!(venue_health_disable_reason),
        ),
        (
            "venue_health_last_error_source".to_string(),
            serde_json::json!(venue_health_last_error_source),
        ),
        (
            "venue_health_last_error_message".to_string(),
            serde_json::json!(venue_health_last_error_message),
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
    use crate::config::{Config, MmVenueRole};
    use crate::state::GlobalState;
    use crate::types::{OrderAck, OrderReject, PlaceOrderIntent, TimeInForce};
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
        let metrics = build_venue_metrics(&cfg, &state, 1_050, stale_ms, &[]);
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
    fn venue_metrics_expose_spread_gate_reason() {
        let mut cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        cfg.mm
            .max_quote_spread_abs_usd_by_venue
            .insert("extended".to_string(), 2.0);
        state.venues[0].mid = Some(100.0);
        state.venues[0].spread = Some(3.0);
        state.venues[0].depth_near_mid = 10_000.0;
        state.venues[0].last_mid_update_ms = Some(1_000);
        state.venues[0].last_mid_apply_ms = Some(1_000);

        let metrics = build_venue_metrics(&cfg, &state, 1_050, cfg.book.stale_ms, &[]);
        let reasons = metrics
            .iter()
            .find(|(k, _)| k == "venue_quote_spread_gate_reason")
            .and_then(|(_, v)| v.as_array())
            .expect("venue_quote_spread_gate_reason");

        assert_eq!(reasons[0].as_str(), Some("spread_abs_cap"));
    }

    #[test]
    fn generated_spread_cap_mode_distinguishes_compression_from_side_suppression() {
        assert_eq!(generated_spread_cap_mode(false, false, false), "none");
        assert_eq!(
            generated_spread_cap_mode(true, false, false),
            "compressed_only"
        );
        assert_eq!(generated_spread_cap_mode(true, true, false), "suppress_bid");
        assert_eq!(generated_spread_cap_mode(true, false, true), "suppress_ask");
        assert_eq!(generated_spread_cap_mode(true, true, true), "suppress_both");
    }

    #[test]
    fn quote_diagnostics_does_not_false_flag_compressed_only_quotes_as_cap_suppressed() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fv_available = true;
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.sigma_eff = 0.01;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;

        let vcfg = &cfg.venues[1];
        {
            let venue = &mut state.venues[1];
            venue.mid = Some(100.0);
            venue.spread = Some(0.2);
            venue.depth_near_mid = 10_000.0;
            venue.last_mid_update_ms = Some(1_000);
            venue.last_mid_apply_ms = Some(1_000);
            venue.status = VenueStatus::Healthy;
            venue.toxicity = 0.0;
            venue.dist_liq_sigma = 10.0;
            venue.margin_available_usd = 10_000.0;
        }
        let venue = &state.venues[1];

        let utility = compute_venue_utility_decision(
            &cfg.mm,
            state.q_global_tao,
            vcfg,
            venue,
            venue_utility_conversion_penalties_enabled(),
        );
        let maker_cost = mm_maker_cost(vcfg, 100.0);

        let diag = quote_diagnostics(
            &cfg,
            &state,
            vcfg,
            venue,
            100.0,
            1_050,
            Side::Sell,
            0.0,
            utility,
            maker_cost,
            100.20,
            0.5,
            cfg.mm.size_eta.max(1e-9),
            true,
            false,
            false,
        );

        assert_eq!(diag.quote_state, "active");
        assert_eq!(diag.suppression_reason, None);
        assert_eq!(diag.prune_stage, "active");
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
            account_position_syncs: &[],
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
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
    fn order_telemetry_redacts_raw_order_identifiers() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let venue_id = cfg.venues[0].id_arc.clone();
        let intent = OrderIntent::Place(PlaceOrderIntent {
            venue_index: 0,
            venue_id: venue_id.clone(),
            side: Side::Buy,
            price: 101.25,
            size: 0.01,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: Some("raw-client-intent".to_string()),
            phase51_target_key: None,
        });
        let ack = ExecutionEvent::OrderAck(OrderAck {
            venue_index: 0,
            venue_id: venue_id.clone(),
            order_id: "raw-order-ack".to_string(),
            client_order_id: Some("raw-client-ack".to_string()),
            seq: Some(1),
            side: Some(Side::Buy),
            price: Some(101.25),
            size: Some(0.01),
            purpose: Some(OrderPurpose::Hedge),
        });
        let reject = ExecutionEvent::OrderReject(OrderReject {
            venue_index: 0,
            venue_id,
            order_id: Some("raw-order-reject".to_string()),
            client_order_id: Some("raw-client-reject".to_string()),
            seq: Some(2),
            purpose: Some(OrderPurpose::Hedge),
            reduce_only: Some(true),
            reason: "synthetic_test_reject".to_string(),
        });
        let fill = FillEvent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            order_id: Some("raw-order-fill".to_string()),
            client_order_id: Some("raw-client-fill".to_string()),
            seq: Some(3),
            side: Side::Buy,
            price: 101.25,
            size: 0.01,
            purpose: OrderPurpose::Hedge,
            fee_bps: 0.0,
        };

        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 1,
            now_ms: 1_000,
            intents: &[intent],
            exec_events: &[ack, reject],
            fills: &[fill],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: false,
            execution_mode: "live",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let text = serde_json::to_string(&record).expect("serialize");
        for raw in [
            "raw-client-intent",
            "raw-order-ack",
            "raw-client-ack",
            "raw-order-reject",
            "raw-client-reject",
            "raw-order-fill",
            "raw-client-fill",
        ] {
            assert!(!text.contains(raw), "telemetry leaked {raw}");
        }
        for key in ["orders", "would_send_orders", "fills"] {
            let records = record
                .get(key)
                .and_then(|value| value.as_array())
                .expect(key);
            assert!(!records.is_empty());
            for item in records {
                let object = item.as_object().expect("order record object");
                assert!(!object.contains_key("order_id"));
                assert!(!object.contains_key("client_order_id"));
            }
        }
        for key in ["orders", "would_send_orders"] {
            let records = record
                .get(key)
                .and_then(|value| value.as_array())
                .expect(key);
            for item in records {
                let object = item.as_object().expect("order record object");
                let action_id = object
                    .get("action_id")
                    .and_then(|value| value.as_str())
                    .expect("sanitized action_id");
                assert!(!action_id.contains("raw-"));
            }
        }
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

        let metrics = build_venue_metrics(&cfg, &state, now_ms, stale_ms, &[]);
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
        let metrics = build_venue_metrics(&cfg, &state, now_ms, stale_ms, &[]);
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
        let metrics = build_venue_metrics(&cfg, &state, now_ms, global_stale_ms, &[]);
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

    #[test]
    fn record_includes_account_position_syncs() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let now_ms = 10_000;
        let syncs = vec![AccountPositionSyncRecord {
            venue_index: 2,
            venue_id: "aster".to_string(),
            snapshot_seq: 7,
            snapshot_timestamp_ms: Some(now_ms - 5),
            ingest_now_ms: now_ms,
            pre_position_tao: 0.0,
            post_position_tao: -0.02,
            position_delta_tao: -0.02,
            pre_margin_available_usd: 85.0,
            post_margin_available_usd: 84.0,
            source: "account_snapshot",
        }];

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
            account_position_syncs: &syncs,
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        let sync_count = record
            .get("account_position_sync_count")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let sync_records = record
            .get("account_position_syncs")
            .and_then(|value| value.as_array())
            .expect("account_position_syncs");
        assert_eq!(sync_count, 1);
        assert_eq!(sync_records.len(), 1);
        assert_eq!(sync_records[0]["venue_id"], "aster");
        assert_eq!(sync_records[0]["position_delta_tao"], -0.02);
    }

    #[test]
    fn record_includes_venue_utility_fillless_ack_pressure() {
        let mut cfg = Config::default();
        cfg.mm
            .venue_role_by_venue
            .insert(cfg.venues[1].id.clone(), crate::config::MmVenueRole::Anchor);
        cfg.mm
            .pre_soft_taper_global_position_tao_by_venue
            .insert(cfg.venues[2].id.clone(), 0.01);
        cfg.mm
            .pre_soft_taper_size_multiplier_by_venue
            .insert(cfg.venues[2].id.clone(), 0.5);
        let mut state = GlobalState::new(&cfg);
        state.q_global_tao = -0.02;
        state.venues[1].utility.mm_fillless_ack_pressure = 4.25;

        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 1,
            now_ms: 10_000,
            intents: &[],
            exec_events: &[],
            fills: &[],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "shadow",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        let values = record
            .get("venue_utility_mm_fillless_ack_pressure")
            .and_then(|value| value.as_array())
            .expect("venue_utility_mm_fillless_ack_pressure");
        assert_eq!(values.len(), cfg.venues.len());
        assert_eq!(values[1].as_f64(), Some(4.25));

        let roles = record
            .get("venue_utility_role")
            .and_then(|value| value.as_array())
            .expect("venue_utility_role");
        assert_eq!(roles[1].as_str(), Some("anchor"));

        let role_caps = record
            .get("venue_utility_role_cap_applied")
            .and_then(|value| value.as_array())
            .expect("venue_utility_role_cap_applied");
        assert_eq!(role_caps[1].as_bool(), Some(true));

        let taper_active = record
            .get("venue_pre_soft_taper_active")
            .and_then(|value| value.as_array())
            .expect("venue_pre_soft_taper_active");
        assert_eq!(taper_active[2].as_bool(), Some(true));

        let taper_side = record
            .get("venue_pre_soft_taper_side")
            .and_then(|value| value.as_array())
            .expect("venue_pre_soft_taper_side");
        assert_eq!(taper_side[2].as_str(), Some("sell"));

        let taper_reason = record
            .get("venue_pre_soft_taper_reason")
            .and_then(|value| value.as_array())
            .expect("venue_pre_soft_taper_reason");
        assert_eq!(taper_reason[2].as_str(), Some("global_inventory"));
    }

    #[test]
    fn record_marks_execution_visibility_gap_for_sync_without_fill() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let now_ms = 10_000;
        let syncs = vec![AccountPositionSyncRecord {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            snapshot_seq: 11,
            snapshot_timestamp_ms: Some(now_ms - 3),
            ingest_now_ms: now_ms,
            pre_position_tao: 0.0,
            post_position_tao: -0.09,
            position_delta_tao: -0.09,
            pre_margin_available_usd: 70.0,
            post_margin_available_usd: 69.0,
            source: "account_snapshot",
        }];

        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 7,
            now_ms,
            intents: &[],
            exec_events: &[],
            fills: &[],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: false,
            execution_mode: "live",
            reconcile_drift: &[],
            account_position_syncs: &syncs,
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        assert_eq!(
            record
                .get("execution_visibility_gap")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            record
                .get("execution_visibility_gap_reason")
                .and_then(|value| value.as_str()),
            Some("account_position_sync_without_fill")
        );
        let venues = record
            .get("execution_visibility_gap_venues")
            .and_then(|value| value.as_array())
            .expect("execution_visibility_gap_venues");
        assert_eq!(venues.len(), 1);
        assert_eq!(venues[0].as_str(), Some("paradex"));
    }

    #[test]
    fn quote_levels_report_global_fv_suppression() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let now_ms = 10_000;
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
            account_position_syncs: &[],
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        let quote_levels = record
            .get("quote_levels")
            .and_then(|value| value.as_array())
            .expect("quote_levels");
        let first_quote = &quote_levels[0];
        assert_eq!(first_quote["quote_state"], "suppressed");
        assert_eq!(first_quote["suppression_reason"], "global_fv_unavailable");
    }

    #[test]
    fn quote_levels_report_side_specific_edge_thresholds() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_bid_by_venue
            .insert("paradex".to_string(), 0.05);
        let mut state = GlobalState::new(&cfg);
        let now_ms = 10_000;
        state.fair_value = Some(300.0);
        state.fair_value_prev = 300.0;
        state.fv_available = true;
        state.sigma_eff = 0.1;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;
        state.vol_ratio_clipped = 1.0;
        state.delta_limit_usd = 100_000.0;
        state.q_global_tao = -10.0;

        for venue in &mut state.venues {
            venue.mid = Some(300.0);
            venue.spread = Some(1.0);
            venue.depth_near_mid = 10_000.0;
            venue.margin_available_usd = 10_000.0;
            venue.dist_liq_sigma = 10.0;
            venue.status = VenueStatus::Healthy;
            venue.toxicity = 0.0;
            venue.last_mid_update_ms = Some(now_ms - 10);
            venue.last_mid_apply_ms = Some(now_ms - 10);
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
            shadow_mode: false,
            execution_mode: "live",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        let quote_levels = record
            .get("quote_levels")
            .and_then(|value| value.as_array())
            .expect("quote_levels");
        let paradex_bid = quote_levels
            .iter()
            .find(|entry| entry["venue_id"] == "paradex" && entry["side"] == "Bid")
            .expect("paradex bid quote level");
        let paradex_ask = quote_levels
            .iter()
            .find(|entry| entry["venue_id"] == "paradex" && entry["side"] == "Ask")
            .expect("paradex ask quote level");

        assert_eq!(paradex_bid["edge_threshold"].as_f64(), Some(0.05));
        assert_eq!(paradex_ask["edge_threshold"].as_f64(), Some(0.5));
        assert_eq!(paradex_bid["hedge_cost_edge_floor"].as_f64(), Some(0.0));
    }

    #[test]
    fn quote_levels_include_hedge_cost_edge_floor_when_enabled() {
        let mut cfg = Config::default();
        cfg.mm.hedge_cost_edge_mult = 0.5;
        cfg.mm.edge_local_min = 0.10;
        let mut state = GlobalState::new(&cfg);
        let now_ms = 10_000;
        state.fair_value = Some(300.0);
        state.fair_value_prev = 300.0;
        state.fv_available = true;
        state.sigma_eff = 0.1;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;
        state.vol_ratio_clipped = 1.0;
        state.delta_limit_usd = 100_000.0;

        for venue in &mut state.venues {
            venue.mid = Some(300.0);
            venue.spread = Some(1.0);
            venue.depth_near_mid = 10_000.0;
            venue.margin_available_usd = 10_000.0;
            venue.dist_liq_sigma = 10.0;
            venue.status = VenueStatus::Healthy;
            venue.toxicity = 0.0;
            venue.last_mid_update_ms = Some(now_ms - 10);
            venue.last_mid_apply_ms = Some(now_ms - 10);
        }

        let expected_floor = 0.5 * (0.5 * 1.0 + 0.0004 * 300.0 + cfg.hedge.slippage_buffer);
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
            shadow_mode: false,
            execution_mode: "live",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        let quote_levels = record
            .get("quote_levels")
            .and_then(|value| value.as_array())
            .expect("quote_levels");
        let aster_bid = quote_levels
            .iter()
            .find(|entry| entry["venue_id"] == "aster" && entry["side"] == "Bid")
            .expect("aster bid quote level");
        let hedge_floor = aster_bid["hedge_cost_edge_floor"]
            .as_f64()
            .expect("hedge_cost_edge_floor");
        let edge_threshold = aster_bid["edge_threshold"]
            .as_f64()
            .expect("edge_threshold");

        assert!((hedge_floor - expected_floor).abs() < 1e-9);
        assert!((edge_threshold - (cfg.mm.edge_local_min + expected_floor)).abs() < 1e-9);
    }

    #[test]
    fn quote_levels_report_utility_prune_stage_for_probationary_margin_limited_quote() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_bid_by_venue
            .insert("hyperliquid".to_string(), 0.01);
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("hyperliquid".to_string(), 0.01);
        cfg.mm
            .venue_role_by_venue
            .insert("hyperliquid".to_string(), MmVenueRole::Probationary);

        let mut state = GlobalState::new(&cfg);
        let now_ms = 10_000;
        state.fair_value = Some(1_000.0);
        state.fair_value_prev = 1_000.0;
        state.fv_available = true;
        state.sigma_eff = 0.1;
        state.spread_mult = 1.0;
        state.size_mult = 1.0;
        state.vol_ratio_clipped = 1.0;
        state.delta_limit_usd = 100_000.0;
        state.q_global_tao = 0.0;

        for venue in &mut state.venues {
            venue.mid = Some(1_000.0);
            venue.spread = Some(2.0);
            venue.depth_near_mid = 10_000.0;
            venue.margin_available_usd = 10_000.0;
            venue.dist_liq_sigma = 10.0;
            venue.status = VenueStatus::Healthy;
            venue.toxicity = 0.0;
            venue.last_mid_update_ms = Some(now_ms - 10);
            venue.last_mid_apply_ms = Some(now_ms - 10);
        }
        state.venues[1].margin_available_usd = 4.0;

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
            shadow_mode: false,
            execution_mode: "live",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 0,
            venue_health_diagnostics: &[],
        });

        let quote_levels = record
            .get("quote_levels")
            .and_then(|value| value.as_array())
            .expect("quote_levels");
        let hyperliquid_ask = quote_levels
            .iter()
            .find(|entry| entry["venue_id"] == "hyperliquid" && entry["side"] == "Ask")
            .expect("hyperliquid ask quote level");

        assert_eq!(hyperliquid_ask["utility_role"], "probationary");
        assert_eq!(hyperliquid_ask["utility_tier"], "reduced");
        assert_eq!(
            hyperliquid_ask["suppression_reason"],
            "size_or_passivity_gated"
        );
        assert_eq!(hyperliquid_ask["prune_stage"], "utility_scale_below_lot");
        assert!(
            hyperliquid_ask["candidate_size_pre_utility"]
                .as_f64()
                .expect("candidate_size_pre_utility")
                >= 0.01
        );
        assert_eq!(
            hyperliquid_ask["candidate_size_post_utility"].as_f64(),
            Some(0.0)
        );
    }

    #[cfg(feature = "live")]
    #[test]
    fn hedge_records_include_source_decision_id_from_recent_mm_fill() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let fill_time_ms = 9_980;
        let now_ms = 10_000;

        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.venues[0].mid = Some(100.0);
        state.venues[1].mid = Some(100.0);

        state.live_order_state.register_mm_decision_lineage(
            0,
            "mm_coid_1",
            Side::Sell,
            100.5,
            0.25,
            OrderPurpose::Mm,
            "mm_decision_1",
            fill_time_ms - 1,
        );

        let mm_fill = FillEvent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            order_id: None,
            client_order_id: Some("mm_coid_1".to_string()),
            seq: Some(1),
            side: Side::Sell,
            price: 100.5,
            size: 0.25,
            purpose: OrderPurpose::Mm,
            fee_bps: 0.0,
        };
        state.apply_fill_event(&mm_fill, fill_time_ms, &cfg);
        state.recompute_after_fills(&cfg);

        let hedge_intent = OrderIntent::Place(PlaceOrderIntent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            side: Side::Buy,
            price: 100.4,
            size: 0.25,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        });
        let hedge_fill = FillEvent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            order_id: None,
            client_order_id: None,
            seq: Some(2),
            side: Side::Buy,
            price: 100.4,
            size: 0.25,
            purpose: OrderPurpose::Hedge,
            fee_bps: 0.0,
        };

        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 7,
            now_ms,
            intents: &[hedge_intent],
            exec_events: &[],
            fills: &[hedge_fill],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let hedges = record
            .get("hedges")
            .and_then(|value| value.as_array())
            .expect("hedges");
        assert_eq!(hedges.len(), 1);
        assert_eq!(hedges[0]["source_decision_id"], "mm_decision_1");
        assert_eq!(hedges[0]["source_fill_venue_id"], cfg.venues[0].id.as_str());
        assert_eq!(
            hedges[0]["source_fill_age_ms"].as_i64(),
            Some(now_ms - fill_time_ms)
        );

        let fills = record
            .get("fills")
            .and_then(|value| value.as_array())
            .expect("fills");
        let hedge_fill_record = fills
            .iter()
            .find(|fill| fill["purpose"] == "Hedge")
            .expect("hedge fill record");
        assert_eq!(hedge_fill_record["source_decision_id"], "mm_decision_1");
    }

    #[cfg(feature = "live")]
    #[test]
    fn hedge_records_include_source_decision_id_from_cached_inferred_mm_fill() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mm_fill_time_ms = 10_000;
        let hedge_time_ms = mm_fill_time_ms + 120_000;

        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.venues[0].mid = Some(100.0);
        state.venues[1].mid = Some(100.0);

        state.live_order_state.register_mm_decision_lineage(
            0,
            "mm_coid_cached",
            Side::Sell,
            100.5,
            0.25,
            OrderPurpose::Mm,
            "mm_decision_cached",
            mm_fill_time_ms - 1,
        );

        let mm_fill = FillEvent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            order_id: None,
            client_order_id: Some("mm_coid_cached".to_string()),
            seq: Some(1),
            side: Side::Sell,
            price: 100.5,
            size: 0.25,
            purpose: OrderPurpose::Mm,
            fee_bps: 0.0,
        };

        let mut builder = TelemetryBuilder::new(&cfg);
        let _ = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 1,
            now_ms: mm_fill_time_ms,
            intents: &[],
            exec_events: &[],
            fills: &[mm_fill],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        assert!(
            state.venues[0].recent_fills.is_empty(),
            "test must prove the telemetry cache, not state.recent_fills"
        );

        let hedge_intent = OrderIntent::Place(PlaceOrderIntent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            side: Side::Buy,
            price: 100.4,
            size: 0.25,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        });
        let hedge_fill = FillEvent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            order_id: None,
            client_order_id: None,
            seq: Some(2),
            side: Side::Buy,
            price: 100.4,
            size: 0.25,
            purpose: OrderPurpose::Hedge,
            fee_bps: 0.0,
        };

        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 2,
            now_ms: hedge_time_ms,
            intents: &[hedge_intent],
            exec_events: &[],
            fills: &[hedge_fill],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let hedges = record
            .get("hedges")
            .and_then(|value| value.as_array())
            .expect("hedges");
        assert_eq!(hedges.len(), 1);
        assert_eq!(hedges[0]["source_decision_id"], "mm_decision_cached");
        assert_eq!(
            hedges[0]["source_fill_age_ms"].as_i64(),
            Some(hedge_time_ms - mm_fill_time_ms)
        );

        let fills = record
            .get("fills")
            .and_then(|value| value.as_array())
            .expect("fills");
        let hedge_fill_record = fills
            .iter()
            .find(|fill| fill["purpose"] == "Hedge")
            .expect("hedge fill record");
        assert_eq!(
            hedge_fill_record["source_decision_id"],
            "mm_decision_cached"
        );
        assert_eq!(hedge_fill_record["source_kind"], "mm_fill_cache");
    }

    #[cfg(feature = "live")]
    #[test]
    fn hedge_records_retain_source_decision_id_across_long_soak() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mm_fill_time_ms = 10_000;
        let hedge_time_ms = mm_fill_time_ms + 7_200_000;

        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.venues[0].mid = Some(100.0);
        state.venues[1].mid = Some(100.0);

        state.live_order_state.register_mm_decision_lineage(
            0,
            "mm_coid_long_soak",
            Side::Sell,
            100.5,
            0.25,
            OrderPurpose::Mm,
            "mm_decision_long_soak",
            mm_fill_time_ms - 1,
        );

        let mm_fill = FillEvent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            order_id: None,
            client_order_id: Some("mm_coid_long_soak".to_string()),
            seq: Some(1),
            side: Side::Sell,
            price: 100.5,
            size: 0.25,
            purpose: OrderPurpose::Mm,
            fee_bps: 0.0,
        };

        let mut builder = TelemetryBuilder::new(&cfg);
        let _ = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 1,
            now_ms: mm_fill_time_ms,
            intents: &[],
            exec_events: &[],
            fills: &[mm_fill],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let hedge_intent = OrderIntent::Place(PlaceOrderIntent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            side: Side::Buy,
            price: 100.4,
            size: 0.25,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        });
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 2,
            now_ms: hedge_time_ms,
            intents: &[hedge_intent],
            exec_events: &[],
            fills: &[],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let hedges = record
            .get("hedges")
            .and_then(|value| value.as_array())
            .expect("hedges");
        assert_eq!(hedges.len(), 1);
        assert_eq!(hedges[0]["source_decision_id"], "mm_decision_long_soak");
        assert_eq!(hedges[0]["source_kind"], "mm_fill_cache");
        assert_eq!(
            hedges[0]["source_fill_age_ms"].as_i64(),
            Some(hedge_time_ms - mm_fill_time_ms)
        );
    }

    #[test]
    fn hedge_records_fallback_to_residual_source_when_no_mm_fill_lineage_exists() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 10_000;

        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.venues[1].mid = Some(100.0);
        state.venues[1].position_tao = -0.01;

        let hedge_intent = OrderIntent::Place(PlaceOrderIntent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            side: Side::Buy,
            price: 100.4,
            size: 0.01,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        });

        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 7,
            now_ms,
            intents: &[hedge_intent],
            exec_events: &[],
            fills: &[],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let hedges = record
            .get("hedges")
            .and_then(|value| value.as_array())
            .expect("hedges");
        assert_eq!(hedges.len(), 1);
        let expected_source = format!("residual:{}:short", cfg.venues[1].id);
        assert_eq!(hedges[0]["source_decision_id"], expected_source.as_str());
        assert_eq!(hedges[0]["source_fill_venue_id"], cfg.venues[1].id.as_str());
        assert_eq!(hedges[0]["source_kind"], "residual_position");
    }

    #[test]
    fn hedge_fill_records_fallback_to_residual_source_after_position_flattened() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let now_ms = 10_000;

        let hedge_fill = FillEvent {
            venue_index: 1,
            venue_id: cfg.venues[1].id_arc.clone(),
            order_id: None,
            client_order_id: None,
            seq: Some(2),
            side: Side::Buy,
            price: 100.4,
            size: 0.01,
            purpose: OrderPurpose::Hedge,
            fee_bps: 0.0,
        };

        let mut builder = TelemetryBuilder::new(&cfg);
        let record = builder.build_record(TelemetryInputs {
            cfg: &cfg,
            state: &state,
            tick: 7,
            now_ms,
            intents: &[],
            exec_events: &[],
            fills: &[hedge_fill],
            last_exit_intent: None,
            last_hedge_intent: None,
            kill_event: None,
            shadow_mode: true,
            execution_mode: "paper",
            reconcile_drift: &[],
            account_position_syncs: &[],
            max_orders_per_tick: 16,
            venue_health_diagnostics: &[],
        });

        let fills = record
            .get("fills")
            .and_then(|value| value.as_array())
            .expect("fills");
        assert_eq!(fills.len(), 1);
        let expected_source = format!("residual:{}:short", cfg.venues[1].id);
        assert_eq!(fills[0]["source_decision_id"], expected_source.as_str());
        assert_eq!(fills[0]["source_fill_venue_id"], cfg.venues[1].id.as_str());
        assert_eq!(fills[0]["source_kind"], "residual_fill");
    }
}
