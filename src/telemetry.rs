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
//!         "t": tick_index,
//!         "fair": global_state.fair_value,
//!         "sigma_eff": global_state.sigma_eff,
//!         "inventory_tao": global_state.inventory_tao,
//!         "basis_usd": global_state.basis_usd,
//!         "pnl_realised": global_state.pnl_realised,
//!         "pnl_unrealised": global_state.pnl_unrealised,
//!         "pnl_total": global_state.pnl_total(),
//!         "risk_regime": format!("{:?}", global_state.risk_regime),
//!         "kill_switch": global_state.kill_switch,
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

use serde_json::{self, Value as JsonValue};

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

        TelemetryConfig { mode, path }
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
            writer: None,
        }
    }

    /// Explicit constructor from a given config.
    pub fn from_config(cfg: TelemetryConfig) -> Self {
        TelemetrySink {
            mode: cfg.mode,
            path: cfg.path,
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

            let file_res = OpenOptions::new().create(true).append(true).open(&path);

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
}

impl Drop for TelemetrySink {
    fn drop(&mut self) {
        self.flush();
    }
}
