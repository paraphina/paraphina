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

use serde_json::{self, Value as JsonValue};

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
