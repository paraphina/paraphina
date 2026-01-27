// src/event_log.rs
//
// Feature-gated event log writer/reader for deterministic replay.

#[cfg(feature = "event_log")]
use std::env;
use std::fs::File;
#[cfg(feature = "event_log")]
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
#[cfg(feature = "event_log")]
use std::io::{BufWriter, Write};
use std::path::Path;
#[cfg(feature = "event_log")]
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::types::{
    BalanceUpdate, BookUpdate, ExecutionEvent, FillEvent, FundingUpdate, OrderAck, OrderPurpose,
    OrderReject, Side,
};

#[cfg(feature = "live")]
use crate::live::types::{
    AccountEvent, ExecutionEvent as LiveExecutionEvent, MarketDataEvent, OrderSnapshot,
};

/// Serializable execution event for event logs (uses String for venue_id).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableExecutionEvent {
    BookUpdate {
        venue_index: usize,
        venue_id: String,
        mid: f64,
        spread: f64,
        depth_near_mid: f64,
        timestamp_ms: i64,
    },
    Fill {
        venue_index: usize,
        venue_id: String,
        #[serde(default)]
        order_id: Option<String>,
        #[serde(default)]
        client_order_id: Option<String>,
        #[serde(default)]
        seq: Option<u64>,
        side: Side,
        price: f64,
        size: f64,
        purpose: OrderPurpose,
        fee_bps: f64,
    },
    OrderAck {
        venue_index: usize,
        venue_id: String,
        order_id: String,
        #[serde(default)]
        client_order_id: Option<String>,
        #[serde(default)]
        seq: Option<u64>,
        side: Option<Side>,
        price: Option<f64>,
        size: Option<f64>,
        purpose: Option<OrderPurpose>,
    },
    OrderReject {
        venue_index: usize,
        venue_id: String,
        order_id: Option<String>,
        #[serde(default)]
        client_order_id: Option<String>,
        #[serde(default)]
        seq: Option<u64>,
        reason: String,
    },
    BalanceUpdate {
        venue_index: usize,
        venue_id: String,
        margin_balance_usd: f64,
        margin_used_usd: f64,
        margin_available_usd: f64,
    },
    FundingUpdate {
        venue_index: usize,
        venue_id: String,
        funding_8h: f64,
    },
}

impl SerializableExecutionEvent {
    pub fn to_execution_event(&self) -> ExecutionEvent {
        match self {
            SerializableExecutionEvent::BookUpdate {
                venue_index,
                venue_id,
                mid,
                spread,
                depth_near_mid,
                timestamp_ms,
            } => ExecutionEvent::BookUpdate(BookUpdate {
                venue_index: *venue_index,
                venue_id: venue_id.as_str().into(),
                mid: *mid,
                spread: *spread,
                depth_near_mid: *depth_near_mid,
                timestamp_ms: *timestamp_ms,
            }),
            SerializableExecutionEvent::Fill {
                venue_index,
                venue_id,
                order_id,
                client_order_id,
                seq,
                side,
                price,
                size,
                purpose,
                fee_bps,
            } => ExecutionEvent::Fill(FillEvent {
                venue_index: *venue_index,
                venue_id: venue_id.as_str().into(),
                order_id: order_id.clone(),
                client_order_id: client_order_id.clone(),
                seq: *seq,
                side: *side,
                price: *price,
                size: *size,
                purpose: *purpose,
                fee_bps: *fee_bps,
            }),
            SerializableExecutionEvent::OrderAck {
                venue_index,
                venue_id,
                order_id,
                client_order_id,
                seq,
                side,
                price,
                size,
                purpose,
            } => ExecutionEvent::OrderAck(OrderAck {
                venue_index: *venue_index,
                venue_id: venue_id.as_str().into(),
                order_id: order_id.clone(),
                client_order_id: client_order_id.clone(),
                seq: *seq,
                side: *side,
                price: *price,
                size: *size,
                purpose: *purpose,
            }),
            SerializableExecutionEvent::OrderReject {
                venue_index,
                venue_id,
                order_id,
                client_order_id,
                seq,
                reason,
            } => ExecutionEvent::OrderReject(OrderReject {
                venue_index: *venue_index,
                venue_id: venue_id.as_str().into(),
                order_id: order_id.clone(),
                client_order_id: client_order_id.clone(),
                seq: *seq,
                reason: reason.clone(),
            }),
            SerializableExecutionEvent::BalanceUpdate {
                venue_index,
                venue_id,
                margin_balance_usd,
                margin_used_usd,
                margin_available_usd,
            } => ExecutionEvent::BalanceUpdate(BalanceUpdate {
                venue_index: *venue_index,
                venue_id: venue_id.as_str().into(),
                margin_balance_usd: *margin_balance_usd,
                margin_used_usd: *margin_used_usd,
                margin_available_usd: *margin_available_usd,
            }),
            SerializableExecutionEvent::FundingUpdate {
                venue_index,
                venue_id,
                funding_8h,
            } => ExecutionEvent::FundingUpdate(FundingUpdate {
                venue_index: *venue_index,
                venue_id: venue_id.as_str().into(),
                funding_8h: *funding_8h,
            }),
        }
    }
}

impl From<&ExecutionEvent> for SerializableExecutionEvent {
    fn from(event: &ExecutionEvent) -> Self {
        match event {
            ExecutionEvent::BookUpdate(b) => SerializableExecutionEvent::BookUpdate {
                venue_index: b.venue_index,
                venue_id: b.venue_id.to_string(),
                mid: b.mid,
                spread: b.spread,
                depth_near_mid: b.depth_near_mid,
                timestamp_ms: b.timestamp_ms,
            },
            ExecutionEvent::Fill(f) => SerializableExecutionEvent::Fill {
                venue_index: f.venue_index,
                venue_id: f.venue_id.to_string(),
                order_id: f.order_id.clone(),
                client_order_id: f.client_order_id.clone(),
                seq: f.seq,
                side: f.side,
                price: f.price,
                size: f.size,
                purpose: f.purpose,
                fee_bps: f.fee_bps,
            },
            ExecutionEvent::OrderAck(a) => SerializableExecutionEvent::OrderAck {
                venue_index: a.venue_index,
                venue_id: a.venue_id.to_string(),
                order_id: a.order_id.clone(),
                client_order_id: a.client_order_id.clone(),
                seq: a.seq,
                side: a.side,
                price: a.price,
                size: a.size,
                purpose: a.purpose,
            },
            ExecutionEvent::OrderReject(r) => SerializableExecutionEvent::OrderReject {
                venue_index: r.venue_index,
                venue_id: r.venue_id.to_string(),
                order_id: r.order_id.clone(),
                client_order_id: r.client_order_id.clone(),
                seq: r.seq,
                reason: r.reason.clone(),
            },
            ExecutionEvent::BalanceUpdate(b) => SerializableExecutionEvent::BalanceUpdate {
                venue_index: b.venue_index,
                venue_id: b.venue_id.to_string(),
                margin_balance_usd: b.margin_balance_usd,
                margin_used_usd: b.margin_used_usd,
                margin_available_usd: b.margin_available_usd,
            },
            ExecutionEvent::FundingUpdate(f) => SerializableExecutionEvent::FundingUpdate {
                venue_index: f.venue_index,
                venue_id: f.venue_id.to_string(),
                funding_8h: f.funding_8h,
            },
        }
    }
}

/// One normalized event log payload (JSONL).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum EventLogPayload {
    Tick,
    Execution(SerializableExecutionEvent),
    #[cfg(feature = "live")]
    MarketData(MarketDataEvent),
    #[cfg(feature = "live")]
    Account(AccountEvent),
    #[cfg(feature = "live")]
    LiveExecution(LiveExecutionEvent),
    #[cfg(feature = "live")]
    OrderSnapshot(OrderSnapshot),
}

/// One normalized event record (JSONL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLogRecord {
    pub tick: u64,
    pub now_ms: i64,
    pub phase: String,
    pub event: EventLogPayload,
}

impl EventLogRecord {
    pub fn to_execution_event(&self) -> Option<ExecutionEvent> {
        match &self.event {
            EventLogPayload::Execution(ev) => Some(ev.to_execution_event()),
            _ => None,
        }
    }
}

/// Event log writer (JSONL), enabled via feature + env.
#[cfg(feature = "event_log")]
#[derive(Debug)]
pub struct EventLogWriter {
    path: PathBuf,
    writer: BufWriter<File>,
}

#[cfg(feature = "event_log")]
impl EventLogWriter {
    /// Initialize from env var `PARAPHINA_EVENT_LOG_PATH`.
    pub fn from_env() -> Option<Self> {
        let path = env::var("PARAPHINA_EVENT_LOG_PATH")
            .ok()
            .map(PathBuf::from)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()?;
        Some(Self {
            path,
            writer: BufWriter::new(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn log_event(&mut self, record: &EventLogRecord) {
        if let Ok(line) = serde_json::to_string(record) {
            let _ = writeln!(self.writer, "{}", line);
        }
    }
}

/// Read JSONL event log into memory.
pub fn read_event_log(path: &Path) -> std::io::Result<Vec<EventLogRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_str::<EventLogRecord>(&line) {
            out.push(record);
        }
    }
    Ok(out)
}
