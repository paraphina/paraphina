//! Live trading loop runner (feature-gated).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, oneshot};

use super::ops::{
    append_account_reconcile_audit, append_reconcile_drift_audit, default_audit_dir, HealthState,
    LiveMetrics,
};
use super::venue_health::VenueHealthManager;
use crate::actions::{intents_to_actions, ActionBatch, ActionIdGenerator};
use crate::config::Config;
use crate::engine::Engine;
#[cfg(feature = "event_log")]
use crate::event_log::read_event_log;
#[cfg(feature = "event_log")]
use crate::event_log::{EventLogPayload, EventLogRecord, EventLogWriter};
use crate::execution_events::apply_execution_events;
use crate::exit;
use crate::fill_batcher::FillBatcher;
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::loop_scheduler::LoopScheduler;
use crate::mm::{compute_mm_quotes, compute_mm_quotes_with_ablations};
use crate::order_management::plan_mm_order_actions;
use crate::sim_eval::AblationSet;
use crate::state::GlobalState;
use crate::state::VenueState;
use crate::telemetry::{
    ensure_schema_v1, ReconcileDriftRecord, TelemetryBuilder, TelemetryInputs, TelemetrySink,
};
#[cfg(feature = "event_log")]
use crate::telemetry::{TelemetryConfig, TelemetryMode};
use crate::types::{
    ExecutionEvent, OrderAck, OrderIntent, OrderPurpose, OrderReject, TimestampMs, VenueStatus,
};

use super::orderbook_l2::OrderBookL2;
use super::state_cache::{CanonicalCacheSnapshot, LiveStateCache};
use super::types::ExecutionEvent as LiveExecutionEvent;
use serde_json::json;
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct LiveOrderRequest {
    pub intents: Vec<OrderIntent>,
    pub action_batch: ActionBatch,
    pub now_ms: TimestampMs,
    pub response: oneshot::Sender<Vec<LiveExecutionEvent>>,
}

#[derive(Debug)]
pub struct LiveAccountRequest {
    pub venue_index: Option<usize>,
    pub now_ms: TimestampMs,
    pub response: oneshot::Sender<super::types::AccountSnapshot>,
}

#[derive(Debug)]
pub struct LiveChannels {
    pub market_rx: mpsc::Receiver<super::types::MarketDataEvent>,
    pub account_rx: mpsc::Receiver<super::types::AccountEvent>,
    pub exec_rx: Option<mpsc::Receiver<super::types::ExecutionEvent>>,
    pub account_reconcile_tx: Option<mpsc::Sender<LiveAccountRequest>>,
    pub order_tx: mpsc::Sender<LiveOrderRequest>,
    pub order_snapshot_rx: Option<mpsc::Receiver<super::types::OrderSnapshot>>,
}

#[derive(Clone)]
pub struct LiveRuntimeHooks {
    pub metrics: LiveMetrics,
    pub health: HealthState,
    pub telemetry: Option<LiveTelemetry>,
}

#[derive(Clone)]
pub struct LiveTelemetry {
    pub sink: Arc<Mutex<TelemetrySink>>,
    pub shadow_mode: bool,
    pub execution_mode: &'static str,
    pub max_orders_per_tick: usize,
    pub stats: Arc<Mutex<LiveTelemetryStats>>,
}

#[derive(Debug, Default)]
pub struct LiveTelemetryStats {
    pub ticks_total: u64,
    pub fv_available_ticks: u64,
    pub venue_staleness_events: u64,
    pub venue_disabled_events: u64,
    pub kill_events: u64,
    pub would_place_by_purpose: std::collections::HashMap<String, u64>,
    pub would_cancel_by_purpose: std::collections::HashMap<String, u64>,
    pub would_replace_by_purpose: std::collections::HashMap<String, u64>,
}

fn parse_reconcile_interval_ms() -> Option<i64> {
    let raw = std::env::var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS").ok()?;
    let normalized = raw.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "false" | "off" | "no") {
        return None;
    }
    if let Ok(ms) = raw.parse::<i64>() {
        return (ms > 0).then_some(ms);
    }
    None
}

fn account_snapshot_available(
    snapshot: &super::types::AccountSnapshot,
    now_ms: TimestampMs,
    max_age_ms: i64,
) -> bool {
    if snapshot.timestamp_ms <= 0 {
        return false;
    }
    now_ms.saturating_sub(snapshot.timestamp_ms) <= max_age_ms
}

#[derive(Debug, Clone, Copy)]
pub enum LiveRunMode {
    Realtime {
        interval_ms: u64,
        max_ticks: Option<u64>,
    },
    Step {
        start_ms: TimestampMs,
        step_ms: i64,
        ticks: u64,
    },
}

#[derive(Debug, Clone)]
pub struct LiveRunSummary {
    pub ticks_run: u64,
    pub kill_switch: bool,
    pub fv_available: bool,
    pub ready_market_count: usize,
    pub stale_market_count: usize,
    pub local_vol_short_avg: f64,
    pub local_vol_long_avg: f64,
}

#[derive(Debug, Clone)]
enum CanonicalEvent {
    Market(super::types::MarketDataEvent),
    Account(super::types::AccountEvent),
    Execution(super::types::ExecutionEvent),
    OrderSnapshot(super::types::OrderSnapshot),
}

#[derive(Debug, Clone)]
struct OrderedEvent {
    venue_id: String,
    source_seq: u64,
    event_ts_ms: i64,
    type_order: u8,
    event: CanonicalEvent,
}

#[derive(Debug)]
struct ExecutionEventDeduper {
    seen: std::collections::HashSet<String>,
    order: std::collections::VecDeque<String>,
    max_entries: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct MarketRxStats {
    drained: u64,
    l2_delta: u64,
    l2_snapshot: u64,
    trade: u64,
    funding_update: u64,
    out_market: u64,
    out_l2_delta: u64,
    out_l2_snapshot: u64,
    out_trade: u64,
    out_funding_update: u64,
    cap_hits: u64,
}

impl ExecutionEventDeduper {
    fn new(max_entries: usize) -> Self {
        Self {
            seen: std::collections::HashSet::new(),
            order: std::collections::VecDeque::new(),
            max_entries: max_entries.max(1),
        }
    }

    fn is_duplicate(&mut self, event: &super::types::ExecutionEvent) -> bool {
        let Some(key) = execution_event_key(event) else {
            return false;
        };
        if self.seen.contains(&key) {
            return true;
        }
        self.seen.insert(key.clone());
        self.order.push_back(key);
        while self.order.len() > self.max_entries {
            if let Some(old) = self.order.pop_front() {
                self.seen.remove(&old);
            }
        }
        false
    }
}

fn execution_event_key(event: &super::types::ExecutionEvent) -> Option<String> {
    match event {
        super::types::ExecutionEvent::OrderAccepted(ack) => {
            if let Some(cloid) = &ack.client_order_id {
                Some(format!("order_ack:client:{cloid}"))
            } else {
                Some(format!("order_ack:order:{}", ack.order_id))
            }
        }
        super::types::ExecutionEvent::OrderRejected(rej) => {
            if let Some(order_id) = &rej.order_id {
                Some(format!("order_reject:order:{order_id}"))
            } else {
                Some(format!("order_reject:seq:{}", rej.seq))
            }
        }
        super::types::ExecutionEvent::CancelAccepted(cancel) => {
            Some(format!("cancel_ack:order:{}", cancel.order_id))
        }
        super::types::ExecutionEvent::CancelRejected(rej) => {
            if let Some(order_id) = &rej.order_id {
                Some(format!("cancel_reject:order:{order_id}"))
            } else {
                Some(format!("cancel_reject:seq:{}", rej.seq))
            }
        }
        super::types::ExecutionEvent::Filled(fill) => {
            if let Some(fill_id) = &fill.fill_id {
                Some(format!("fill:id:{fill_id}"))
            } else if let Some(order_id) = &fill.order_id {
                Some(format!(
                    "fill:order:{}:seq:{}:px:{}:sz:{}",
                    order_id, fill.seq, fill.price, fill.size
                ))
            } else if let Some(cloid) = &fill.client_order_id {
                Some(format!(
                    "fill:client:{}:seq:{}:px:{}:sz:{}",
                    cloid, fill.seq, fill.price, fill.size
                ))
            } else {
                Some(format!(
                    "fill:seq:{}:px:{}:sz:{}",
                    fill.seq, fill.price, fill.size
                ))
            }
        }
        super::types::ExecutionEvent::CancelAllAccepted(ack) => Some(format!(
            "cancel_all:venue:{}:seq:{}",
            ack.venue_index, ack.seq
        )),
        super::types::ExecutionEvent::CancelAllRejected(rej) => Some(format!(
            "cancel_all_reject:venue:{}:seq:{}",
            rej.venue_index, rej.seq
        )),
        super::types::ExecutionEvent::OrderSnapshot(snapshot) => Some(format!(
            "order_snapshot:venue:{}:seq:{}",
            snapshot.venue_index, snapshot.seq
        )),
    }
}

fn sort_fills_for_flush(fills: &mut [crate::types::FillEvent]) {
    fills.sort_by(|a, b| {
        let mut ord = a.venue_index.cmp(&b.venue_index);
        if ord != Ordering::Equal {
            return ord;
        }
        ord = a.seq.cmp(&b.seq);
        if ord != Ordering::Equal {
            return ord;
        }
        ord = a.client_order_id.cmp(&b.client_order_id);
        if ord != Ordering::Equal {
            return ord;
        }
        ord = a.order_id.cmp(&b.order_id);
        if ord != Ordering::Equal {
            return ord;
        }
        ord = format!("{:?}", a.side).cmp(&format!("{:?}", b.side));
        if ord != Ordering::Equal {
            return ord;
        }
        ord = a.price.total_cmp(&b.price);
        if ord != Ordering::Equal {
            return ord;
        }
        a.size.total_cmp(&b.size)
    });
}

fn flush_batched_fills(
    batcher: &mut FillBatcher,
    cfg: &Config,
    state: &mut GlobalState,
    now_ms: TimestampMs,
    force: bool,
) -> bool {
    if batcher.pending_len() == 0 {
        return false;
    }
    let should_flush = if force {
        batcher.last_flush_ms() != now_ms
    } else {
        batcher.should_flush(now_ms)
    };
    if !should_flush {
        return false;
    }
    let mut fills = batcher.flush(now_ms);
    if fills.is_empty() {
        return false;
    }
    sort_fills_for_flush(&mut fills);
    for fill in &fills {
        state.apply_fill_event(fill, now_ms, cfg);
    }
    state.recompute_after_fills(cfg);
    true
}

fn drain_ordered_events(
    market_rx: &mut mpsc::Receiver<super::types::MarketDataEvent>,
    account_rx: &mut mpsc::Receiver<super::types::AccountEvent>,
    exec_rx: &mut Option<mpsc::Receiver<super::types::ExecutionEvent>>,
    order_snapshot_rx: &mut Option<mpsc::Receiver<super::types::OrderSnapshot>>,
    mut market_stats: Option<&mut MarketRxStats>,
    l2_delta_coalesce: bool,
    l2_snapshot_coalesce: bool,
    coalesce_ready_mask: u64,
    saw_l2_snapshot_mask_this_tick: &mut u64,
) -> Vec<OrderedEvent> {
    let mut out = Vec::new();
    let coalesce_deltas = l2_delta_coalesce || l2_snapshot_coalesce;
    let mut pending_deltas: Option<Vec<Vec<super::types::L2Delta>>> =
        coalesce_deltas.then(|| Vec::new());
    let mut last_snapshots: Option<Vec<Option<super::types::L2Snapshot>>> =
        l2_snapshot_coalesce.then(|| Vec::new());
    let venue_ready = |vi: usize, saw_mask: u64| -> bool {
        vi < 64 && ((coalesce_ready_mask | saw_mask) & (1u64 << vi)) != 0
    };
    let tick_delta_buffer_max: Option<usize> = std::env::var("PARAPHINA_L2_TICK_DELTA_BUFFER_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0);
    let mut buffer_disabled_mask: u64 = 0;
    let buffer_disabled = |vi: usize, mask: u64| vi < 64 && (mask & (1u64 << vi)) != 0;
    let count_out_market =
        |stats: &mut Option<&mut MarketRxStats>, event: &super::types::MarketDataEvent| {
            if let Some(stats) = stats.as_deref_mut() {
                stats.out_market += 1;
                match event {
                    super::types::MarketDataEvent::L2Snapshot(_) => stats.out_l2_snapshot += 1,
                    super::types::MarketDataEvent::L2Delta(_) => stats.out_l2_delta += 1,
                    super::types::MarketDataEvent::Trade(_) => stats.out_trade += 1,
                    super::types::MarketDataEvent::FundingUpdate(_) => stats.out_funding_update += 1,
                }
            }
        };

    while let Ok(event) = market_rx.try_recv() {
        if let super::types::MarketDataEvent::L2Snapshot(snapshot) = &event {
            let vi = snapshot.venue_index;
            if vi < 64 {
                *saw_l2_snapshot_mask_this_tick |= 1u64 << vi;
            }
        }
        if let Some(stats) = market_stats.as_deref_mut() {
            stats.drained += 1;
            match &event {
                super::types::MarketDataEvent::L2Snapshot(_) => stats.l2_snapshot += 1,
                super::types::MarketDataEvent::L2Delta(_) => stats.l2_delta += 1,
                super::types::MarketDataEvent::Trade(_) => stats.trade += 1,
                super::types::MarketDataEvent::FundingUpdate(_) => stats.funding_update += 1,
            }
        }
        if l2_snapshot_coalesce {
            if let super::types::MarketDataEvent::L2Snapshot(s) = event {
                let vi = s.venue_index;
                if vi < 64 {
                    if !venue_ready(vi, *saw_l2_snapshot_mask_this_tick) {
                        // Not ready: fall through to forward normally.
                    } else {
                    if let Some(last_snapshots) = last_snapshots.as_mut() {
                        if last_snapshots.len() <= vi {
                            last_snapshots.resize_with(vi + 1, || None);
                        }
                        let replace = match last_snapshots[vi].as_ref() {
                            Some(prev) => {
                                s.seq > prev.seq
                                    || (s.seq == prev.seq && s.timestamp_ms >= prev.timestamp_ms)
                            }
                            None => true,
                        };
                        let snapshot_seq = s.seq;
                        if replace {
                            last_snapshots[vi] = Some(s);
                        }
                        if let Some(pending_deltas) = pending_deltas.as_mut() {
                            if pending_deltas.len() <= vi {
                                pending_deltas.resize_with(vi + 1, Vec::new);
                            }
                            pending_deltas[vi].retain(|d| d.seq > snapshot_seq);
                        }
                    }
                    continue;
                    }
                }
                let event = super::types::MarketDataEvent::L2Snapshot(s);
                count_out_market(&mut market_stats, &event);
                if let Some(ordered) = ordered_event_for_market(event) {
                    out.push(ordered);
                }
                continue;
            }
        }
        if coalesce_deltas {
            if let super::types::MarketDataEvent::L2Delta(d) = event {
                let vi = d.venue_index;
                if vi < 64 {
                    if !venue_ready(vi, *saw_l2_snapshot_mask_this_tick) {
                        // UNREADY venue: buffer if cap not reached, else drop
                        if buffer_disabled(vi, buffer_disabled_mask) {
                            // Cap already reached: drop delta for unready venue
                            continue;
                        }
                        if let Some(pending_deltas) = pending_deltas.as_mut() {
                            if pending_deltas.len() <= vi {
                                pending_deltas.resize_with(vi + 1, Vec::new);
                            }
                            if let Some(max) = tick_delta_buffer_max {
                                if pending_deltas[vi].len() >= max {
                                    buffer_disabled_mask |= 1u64 << vi;
                                    if let Some(stats) = market_stats.as_deref_mut() {
                                        stats.cap_hits += 1;
                                    }
                                    // Cap reached: drop delta for unready venue
                                    continue;
                                }
                            }
                            pending_deltas[vi].push(d);
                        }
                        continue;
                    } else {
                    // READY venue: buffer if cap not reached, else emit immediately
                    if buffer_disabled(vi, buffer_disabled_mask) {
                        // Cap already reached: apply snapshot-dominance check, then emit if not dominated
                        if l2_snapshot_coalesce {
                            if let Some(last_snapshots) = last_snapshots.as_ref() {
                                if let Some(Some(snapshot)) = last_snapshots.get(vi) {
                                    if snapshot.seq >= d.seq {
                                        continue;
                                    }
                                }
                            }
                        }
                        let event = super::types::MarketDataEvent::L2Delta(d);
                        count_out_market(&mut market_stats, &event);
                        if let Some(ordered) = ordered_event_for_market(event) {
                            out.push(ordered);
                        }
                        continue;
                    }
                    if l2_snapshot_coalesce {
                        if let Some(last_snapshots) = last_snapshots.as_ref() {
                            if let Some(Some(snapshot)) = last_snapshots.get(vi) {
                                if snapshot.seq >= d.seq {
                                    continue;
                                }
                            }
                        }
                    }
                    if let Some(pending_deltas) = pending_deltas.as_mut() {
                        if pending_deltas.len() <= vi {
                            pending_deltas.resize_with(vi + 1, Vec::new);
                        }
                        if let Some(max) = tick_delta_buffer_max {
                            if pending_deltas[vi].len() >= max {
                                buffer_disabled_mask |= 1u64 << vi;
                                if let Some(stats) = market_stats.as_deref_mut() {
                                    stats.cap_hits += 1;
                                }
                                // Cap reached: emit immediately for ready venue
                                let event = super::types::MarketDataEvent::L2Delta(d);
                                count_out_market(&mut market_stats, &event);
                                if let Some(ordered) = ordered_event_for_market(event) {
                                    out.push(ordered);
                                }
                                continue;
                            }
                        }
                        pending_deltas[vi].push(d);
                    }
                    continue;
                    }
                }
                // venue_index >= 64: fall through to push normally
                let event = super::types::MarketDataEvent::L2Delta(d);
                count_out_market(&mut market_stats, &event);
                if let Some(ordered) = ordered_event_for_market(event) {
                    out.push(ordered);
                }
                continue;
            }
        }
        count_out_market(&mut market_stats, &event);
        if let Some(ordered) = ordered_event_for_market(event) {
            out.push(ordered);
        }
    }
    let mut pending_deltas = pending_deltas.unwrap_or_default();
    let emit_delta_list = |deltas: Vec<super::types::L2Delta>,
                           market_stats: &mut Option<&mut MarketRxStats>,
                           out: &mut Vec<OrderedEvent>| {
        if deltas.is_empty() {
            return;
        }
        let mut deltas = deltas;
        deltas.sort_by(|a, b| (a.seq, a.timestamp_ms).cmp(&(b.seq, b.timestamp_ms)));
        let mut merged: Vec<super::types::L2Delta> = Vec::with_capacity(deltas.len());
        for delta in deltas {
            if let Some(last) = merged.last_mut() {
                if last.seq == delta.seq {
                    last.changes.extend(delta.changes);
                    if delta.timestamp_ms > last.timestamp_ms {
                        last.timestamp_ms = delta.timestamp_ms;
                    }
                    continue;
                }
            }
            merged.push(delta);
        }
        for delta in merged {
            let event = super::types::MarketDataEvent::L2Delta(delta);
            count_out_market(market_stats, &event);
            if let Some(ordered) = ordered_event_for_market(event) {
                out.push(ordered);
            }
        }
    };

    if l2_snapshot_coalesce {
        if let Some(last_snapshots) = last_snapshots {
            for (vi, slot) in last_snapshots.into_iter().enumerate() {
                let Some(snapshot) = slot else { continue };
                let mut deltas = if vi < pending_deltas.len() {
                    std::mem::take(&mut pending_deltas[vi])
                } else {
                    Vec::new()
                };
                if !deltas.is_empty() {
                    deltas.sort_by(|a, b| (a.seq, a.timestamp_ms).cmp(&(b.seq, b.timestamp_ms)));
                    let mut merged: Vec<super::types::L2Delta> = Vec::with_capacity(deltas.len());
                    for delta in deltas {
                        if let Some(last) = merged.last_mut() {
                            if last.seq == delta.seq {
                                last.changes.extend(delta.changes);
                                if delta.timestamp_ms > last.timestamp_ms {
                                    last.timestamp_ms = delta.timestamp_ms;
                                }
                                continue;
                            }
                        }
                        merged.push(delta);
                    }
                    deltas = merged;
                    deltas.retain(|d| d.seq > snapshot.seq);
                }
                let mut contiguous = false;
                if let Some(first) = deltas.first() {
                    contiguous = first.seq == snapshot.seq + 1;
                    if contiguous {
                        let mut prev = first.seq;
                        for delta in deltas.iter().skip(1) {
                            if delta.seq != prev + 1 {
                                contiguous = false;
                                break;
                            }
                            prev = delta.seq;
                        }
                    }
                }
                if contiguous {
                    let mut book = OrderBookL2::new();
                    let mut ok = book
                        .apply_snapshot(&snapshot.bids, &snapshot.asks, snapshot.seq)
                        .is_ok();
                    if ok {
                        for delta in &deltas {
                            if book.apply_delta(&delta.changes, delta.seq).is_err() {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        let last_delta = deltas.last().unwrap();
                        let event = super::types::MarketDataEvent::L2Snapshot(
                            super::types::L2Snapshot {
                                venue_index: snapshot.venue_index,
                                venue_id: snapshot.venue_id.clone(),
                                seq: book.last_seq(),
                                timestamp_ms: snapshot.timestamp_ms.max(last_delta.timestamp_ms),
                                bids: book.bids().to_vec(),
                                asks: book.asks().to_vec(),
                            },
                        );
                        count_out_market(&mut market_stats, &event);
                        if let Some(ordered) = ordered_event_for_market(event) {
                            out.push(ordered);
                        }
                        continue;
                    }
                }
                let event = super::types::MarketDataEvent::L2Snapshot(snapshot);
                count_out_market(&mut market_stats, &event);
                if let Some(ordered) = ordered_event_for_market(event) {
                    out.push(ordered);
                }
                emit_delta_list(deltas, &mut market_stats, &mut out);
            }
        }
    }
    if coalesce_deltas {
        for (vi, deltas) in pending_deltas.into_iter().enumerate() {
            if vi < 64 && !venue_ready(vi, *saw_l2_snapshot_mask_this_tick) {
                continue;
            }
            emit_delta_list(deltas, &mut market_stats, &mut out);
        }
    }
    while let Ok(event) = account_rx.try_recv() {
        if let Some(ordered) = ordered_event_for_account(event) {
            out.push(ordered);
        }
    }
    if let Some(rx) = exec_rx.as_mut() {
        while let Ok(event) = rx.try_recv() {
            if let Some(ordered) = ordered_event_for_execution(event) {
                out.push(ordered);
            }
        }
    }
    if let Some(rx) = order_snapshot_rx.as_mut() {
        while let Ok(snapshot) = rx.try_recv() {
            out.push(OrderedEvent {
                venue_id: snapshot.venue_id.clone(),
                source_seq: snapshot.seq,
                event_ts_ms: snapshot.timestamp_ms,
                type_order: 3,
                event: CanonicalEvent::OrderSnapshot(snapshot),
            });
        }
    }

    out.sort_by(|a, b| {
        (&a.venue_id, a.source_seq, a.event_ts_ms, a.type_order).cmp(&(
            &b.venue_id,
            b.source_seq,
            b.event_ts_ms,
            b.type_order,
        ))
    });

    out
}

fn ordered_event_for_market(event: super::types::MarketDataEvent) -> Option<OrderedEvent> {
    let (venue_id, source_seq, event_ts_ms) = match &event {
        super::types::MarketDataEvent::L2Snapshot(s) => (s.venue_id.clone(), s.seq, s.timestamp_ms),
        super::types::MarketDataEvent::L2Delta(d) => (d.venue_id.clone(), d.seq, d.timestamp_ms),
        super::types::MarketDataEvent::Trade(t) => (t.venue_id.clone(), t.seq, t.timestamp_ms),
        super::types::MarketDataEvent::FundingUpdate(f) => {
            (f.venue_id.clone(), f.seq, f.timestamp_ms)
        }
    };
    Some(OrderedEvent {
        venue_id,
        source_seq,
        event_ts_ms,
        type_order: 0,
        event: CanonicalEvent::Market(event),
    })
}

fn ordered_event_for_account(event: super::types::AccountEvent) -> Option<OrderedEvent> {
    let (venue_id, source_seq, event_ts_ms) = match &event {
        super::types::AccountEvent::Snapshot(s) => (s.venue_id.clone(), s.seq, s.timestamp_ms),
    };
    Some(OrderedEvent {
        venue_id,
        source_seq,
        event_ts_ms,
        type_order: 1,
        event: CanonicalEvent::Account(event),
    })
}

fn ordered_event_for_execution(event: super::types::ExecutionEvent) -> Option<OrderedEvent> {
    let (venue_id, source_seq, event_ts_ms) = match &event {
        super::types::ExecutionEvent::OrderAccepted(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::OrderRejected(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::Filled(e) => (e.venue_id.clone(), e.seq, e.timestamp_ms),
        super::types::ExecutionEvent::CancelAccepted(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelRejected(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelAllAccepted(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelAllRejected(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::OrderSnapshot(e) => {
            (e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
    };
    Some(OrderedEvent {
        venue_id,
        source_seq,
        event_ts_ms,
        type_order: 2,
        event: CanonicalEvent::Execution(event),
    })
}

fn derive_position_tao(positions: &[super::types::PositionSnapshot]) -> f64 {
    positions.iter().map(|p| p.size).sum()
}

fn push_reconcile_drift(
    pending: &mut Vec<ReconcileDriftRecord>,
    audit_dir: &std::path::Path,
    record: ReconcileDriftRecord,
) {
    let _ = append_reconcile_drift_audit(audit_dir, &record);
    pending.push(record);
}

fn diff_exceeds(lhs: f64, rhs: f64, tol: f64) -> bool {
    (lhs - rhs).abs() > tol
}

fn apply_canary_intent_overrides(
    intents: &mut [OrderIntent],
    enforce_post_only: bool,
    enforce_reduce_only: bool,
) {
    if !enforce_post_only && !enforce_reduce_only {
        return;
    }
    for intent in intents {
        match intent {
            OrderIntent::Place(place) => {
                if enforce_post_only && place.purpose == OrderPurpose::Mm {
                    place.post_only = true;
                }
                if enforce_reduce_only && place.purpose != OrderPurpose::Mm {
                    place.reduce_only = true;
                }
            }
            OrderIntent::Replace(replace) => {
                if enforce_post_only && replace.purpose == OrderPurpose::Mm {
                    replace.post_only = true;
                }
                if enforce_reduce_only && replace.purpose != OrderPurpose::Mm {
                    replace.reduce_only = true;
                }
            }
            _ => {}
        }
    }
}

pub async fn run_live_loop(
    cfg: &Config,
    channels: LiveChannels,
    mode: LiveRunMode,
    hooks: Option<LiveRuntimeHooks>,
) -> LiveRunSummary {
    let engine = Engine::new(cfg);
    let mut state = GlobalState::new(cfg);
    let mut cache = LiveStateCache::new(cfg);
    let mut health_manager = VenueHealthManager::new(cfg);
    let mut telemetry_builder = TelemetryBuilder::new(cfg);
    let mut applied_book_logged: Vec<bool> = vec![false; cfg.venues.len()];

    let mut market_rx = channels.market_rx;
    let mut account_rx = channels.account_rx;
    let mut exec_rx = channels.exec_rx;
    let mut order_snapshot_rx = channels.order_snapshot_rx;
    let account_reconcile_tx = channels.account_reconcile_tx;
    let order_tx = channels.order_tx;

    let mut scheduler = LoopScheduler::new(
        now_ms(),
        cfg.main_loop_interval_ms,
        cfg.hedge_loop_interval_ms,
        cfg.risk_loop_interval_ms,
    );
    let mut fill_batcher = FillBatcher::new(cfg.fill_agg_interval_ms);
    fill_batcher.set_last_flush_ms(scheduler.next_main_ms() - cfg.fill_agg_interval_ms);

    let account_reconcile_ms = parse_reconcile_interval_ms();
    let account_poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(5_000);
    let account_snapshot_max_age_ms = account_poll_ms
        .saturating_mul(2)
        .max(cfg.main_loop_interval_ms.saturating_mul(2));
    let mut last_account_reconcile_ms: Option<TimestampMs> = None;
    let mut last_account_snapshot_ms: Vec<Option<TimestampMs>> = vec![None; cfg.venues.len()];

    let mut tick: u64 = 0;
    #[cfg(feature = "event_log")]
    let mut event_log = EventLogWriter::from_env();
    let kill_best_effort_flatten = std::env::var("PARAPHINA_KILL_BEST_EFFORT")
        .or_else(|_| std::env::var("PARAPHINA_LIVE_KILL_FLATTEN"))
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let disable_fv_gate = std::env::var("PARAPHINA_PAPER_DISABLE_FV_GATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let disable_health_gates = std::env::var("PARAPHINA_PAPER_DISABLE_HEALTH_GATES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let smoke_intents = std::env::var("PARAPHINA_PAPER_SMOKE_INTENTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let canary_enabled = std::env::var("PARAPHINA_CANARY_MODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let canary_max_position_tao = std::env::var("PARAPHINA_CANARY_MAX_POSITION_TAO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    let canary_max_open_orders = std::env::var("PARAPHINA_CANARY_MAX_OPEN_ORDERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let canary_stale_max_ticks = std::env::var("PARAPHINA_CANARY_STALE_MAX_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let canary_enforce_post_only = std::env::var("PARAPHINA_CANARY_ENFORCE_POST_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let canary_enforce_reduce_only = std::env::var("PARAPHINA_CANARY_ENFORCE_REDUCE_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let market_rx_stats_enabled = std::env::var("PARAPHINA_MARKET_RX_STATS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let market_rx_stats_every = std::env::var("PARAPHINA_MARKET_RX_STATS_EVERY_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1);
    let market_rx_stats_path = std::env::var_os("PARAPHINA_MARKET_RX_STATS_PATH");
    let l2_delta_coalesce = std::env::var("PARAPHINA_L2_DELTA_COALESCE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let l2_snapshot_coalesce = std::env::var("PARAPHINA_L2_SNAPSHOT_COALESCE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut canary_stale_ticks: u64 = 0;
    let pos_tol = std::env::var("PARAPHINA_RECONCILE_POS_TAO_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.01);
    let bal_tol = std::env::var("PARAPHINA_RECONCILE_BALANCE_USD_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    let order_tol = std::env::var("PARAPHINA_RECONCILE_ORDER_COUNT_TOL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut pending_drift_events: Vec<ReconcileDriftRecord> = Vec::new();
    let fv_ablations = if disable_fv_gate {
        AblationSet::from_ids(&vec!["disable_fair_value_gating".to_string()]).unwrap_or_default()
    } else {
        AblationSet::new()
    };
    let mut interval = match mode {
        LiveRunMode::Realtime { interval_ms, .. } => {
            Some(tokio::time::interval(Duration::from_millis(interval_ms)))
        }
        LiveRunMode::Step { .. } => None,
    };

    let mut deduper = ExecutionEventDeduper::new(50_000);

    let audit_dir = default_audit_dir();
    let mut last_now_ms: TimestampMs = 0;
    let mut last_snapshot: Option<super::state_cache::CanonicalCacheSnapshot> = None;
    let mut pending_events: Vec<OrderedEvent> = Vec::new();
    let mut saw_ready_once = false;
    let mut coalesce_ready_mask: u64 = 0;

    loop {
        let now_ms = match mode {
            LiveRunMode::Realtime { max_ticks, .. } => {
                if let Some(interval) = interval.as_mut() {
                    interval.tick().await;
                }
                if let Some(max) = max_ticks {
                    if tick >= max {
                        break;
                    }
                }
                now_ms()
            }
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            } => {
                if tick >= ticks {
                    break;
                }
                start_ms + step_ms.saturating_mul(tick as i64)
            }
        };
        last_now_ms = now_ms;

        #[cfg(feature = "event_log")]
        if let Some(writer) = event_log.as_mut() {
            writer.log_event(&EventLogRecord {
                tick,
                now_ms,
                phase: "tick".to_string(),
                event: EventLogPayload::Tick,
            });
        }

        if let (Some(interval_ms), Some(tx)) = (account_reconcile_ms, account_reconcile_tx.as_ref())
        {
            let should_reconcile = last_account_reconcile_ms
                .map(|prev| now_ms.saturating_sub(prev) >= interval_ms)
                .unwrap_or(true);
            if should_reconcile {
                last_account_reconcile_ms = Some(now_ms);
                for venue_index in 0..cfg.venues.len() {
                    let (response_tx, mut response_rx) = oneshot::channel();
                    let request = LiveAccountRequest {
                        venue_index: Some(venue_index),
                        now_ms,
                        response: response_tx,
                    };
                    if tx.try_send(request).is_ok() {
                        for _ in 0..1_000 {
                            match response_rx.try_recv() {
                                Ok(snapshot) => {
                                    if snapshot.timestamp_ms > 0 {
                                        if let Some(last) =
                                            last_account_snapshot_ms.get_mut(snapshot.venue_index)
                                        {
                                            *last = Some(snapshot.timestamp_ms);
                                        }
                                    }
                                    let (report, diff) =
                                        cache.reconcile_account_snapshot_with_diff(&snapshot);
                                    if let Some(diff) = diff {
                                        let _ = append_account_reconcile_audit(
                                            &audit_dir, now_ms, diff,
                                        );
                                    }
                                    if !report.account_ok {
                                        if let Some(hooks) = hooks.as_ref() {
                                            hooks.metrics.inc_error();
                                            hooks.metrics.inc_reconcile_mismatch();
                                        }
                                    }
                                    break;
                                }
                                Err(oneshot::error::TryRecvError::Closed) => break,
                                Err(oneshot::error::TryRecvError::Empty) => {
                                    tokio::task::yield_now().await
                                }
                            }
                        }
                    }
                }
            }
        } else if let Some(interval_ms) = account_reconcile_ms {
            let should_reconcile = last_account_reconcile_ms
                .map(|prev| now_ms.saturating_sub(prev) >= interval_ms)
                .unwrap_or(true);
            if should_reconcile {
                last_account_reconcile_ms = Some(now_ms);
                let has_fresh_snapshot = last_account_snapshot_ms
                    .iter()
                    .flatten()
                    .any(|ts| now_ms.saturating_sub(*ts) <= account_snapshot_max_age_ms);
                if has_fresh_snapshot {
                    push_reconcile_drift(
                        &mut pending_drift_events,
                        &audit_dir,
                        ReconcileDriftRecord {
                            timestamp_ms: now_ms,
                            venue_index: 0,
                            venue_id: "all".to_string(),
                            kind: "account_unavailable".to_string(),
                            internal: None,
                            venue: None,
                            diff: None,
                            tolerance: None,
                            source: "account_snapshot".to_string(),
                            available: false,
                        },
                    );
                }
            }
        }

        let mut would_send_intents: Vec<OrderIntent> = Vec::new();
        let mut tick_exec_events: Vec<ExecutionEvent> = Vec::new();
        let mut tick_fills: Vec<crate::types::FillEvent> = Vec::new();

        let mut market_rx_stats = market_rx_stats_enabled.then_some(MarketRxStats::default());
        let maybe_print_market_rx_stats = |tick: u64, stats: &Option<MarketRxStats>| {
            if let Some(stats) = stats.as_ref() {
                if tick % market_rx_stats_every == 0 {
                    let other = stats.drained.saturating_sub(
                        stats.l2_delta + stats.l2_snapshot + stats.trade + stats.funding_update,
                    );
                    if let Some(path) = &market_rx_stats_path {
                        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
                            let _ = writeln!(
                                f,
                                "market_rx_stats tick={} raw_drained={} raw_l2_delta={} raw_l2_snapshot={} raw_trade={} raw_funding_update={} out_market={} out_l2_delta={} out_l2_snapshot={} out_trade={} out_funding_update={} other={} cap_hits={}",
                                tick,
                                stats.drained,
                                stats.l2_delta,
                                stats.l2_snapshot,
                                stats.trade,
                                stats.funding_update,
                                stats.out_market,
                                stats.out_l2_delta,
                                stats.out_l2_snapshot,
                                stats.out_trade,
                                stats.out_funding_update,
                                other,
                                stats.cap_hits
                            );
                        }
                    } else {
                        eprintln!(
                            "market_rx_stats tick={} raw_drained={} raw_l2_delta={} raw_l2_snapshot={} raw_trade={} raw_funding_update={} out_market={} out_l2_delta={} out_l2_snapshot={} out_trade={} out_funding_update={} other={} cap_hits={}",
                            tick,
                            stats.drained,
                            stats.l2_delta,
                            stats.l2_snapshot,
                            stats.trade,
                            stats.funding_update,
                            stats.out_market,
                            stats.out_l2_delta,
                            stats.out_l2_snapshot,
                            stats.out_trade,
                            stats.out_funding_update,
                            other,
                            stats.cap_hits
                        );
                    }
                }
            }
        };

        let delta_coalesce_now = l2_delta_coalesce && saw_ready_once;
        let snapshot_coalesce_now = l2_snapshot_coalesce && saw_ready_once;
        let mut saw_l2_snapshot_mask_this_tick: u64 = 0;

        // Drain ingress channels, canonicalize ordering, then apply.
        pending_events.extend(drain_ordered_events(
            &mut market_rx,
            &mut account_rx,
            &mut exec_rx,
            &mut order_snapshot_rx,
            market_rx_stats.as_mut(),
            delta_coalesce_now,
            snapshot_coalesce_now,
            coalesce_ready_mask,
            &mut saw_l2_snapshot_mask_this_tick,
        ));
        let mut ordered_events = Vec::new();
        let mut future_events = Vec::new();
        for event in pending_events.drain(..) {
            if event.event_ts_ms <= now_ms {
                ordered_events.push(event);
            } else {
                future_events.push(event);
            }
        }
        pending_events = future_events;
        ordered_events.sort_by(|a, b| {
            (&a.venue_id, a.source_seq, a.event_ts_ms, a.type_order).cmp(&(
                &b.venue_id,
                b.source_seq,
                b.event_ts_ms,
                b.type_order,
            ))
        });

        for ordered in ordered_events {
            match ordered.event {
                CanonicalEvent::Market(event) => {
                    #[cfg(feature = "event_log")]
                    if let Some(writer) = event_log.as_mut() {
                        writer.log_event(&EventLogRecord {
                            tick,
                            now_ms,
                            phase: "market".to_string(),
                            event: EventLogPayload::MarketData(event.clone()),
                        });
                    }
                    if let Err(_) = cache.apply_market_event(&event) {
                        health_manager.record_api_error(match &event {
                            super::types::MarketDataEvent::L2Snapshot(s) => s.venue_index,
                            super::types::MarketDataEvent::L2Delta(d) => d.venue_index,
                            super::types::MarketDataEvent::Trade(t) => t.venue_index,
                            super::types::MarketDataEvent::FundingUpdate(f) => f.venue_index,
                        });
                    } else {
                        apply_market_event_to_core(&mut state, cfg, &event);
                        let venue_index = match &event {
                            super::types::MarketDataEvent::L2Snapshot(s) => s.venue_index,
                            super::types::MarketDataEvent::L2Delta(d) => d.venue_index,
                            super::types::MarketDataEvent::Trade(t) => t.venue_index,
                            super::types::MarketDataEvent::FundingUpdate(f) => f.venue_index,
                        };
                        if let Some(logged) = applied_book_logged.get_mut(venue_index) {
                            if !*logged {
                                if let Some(market) = cache.market.get(venue_index) {
                                    if let (Some(mid), Some(spread)) = (market.mid, market.spread) {
                                        if market.depth_near_mid > 0.0 {
                                            let venue_id = cfg
                                                .venues
                                                .get(venue_index)
                                                .map(|v| v.id.as_str())
                                                .unwrap_or("unknown");
                                            eprintln!(
                                                "APPLIED_BOOK venue={} venue_index={} mid={} spread={} depth_usd={}",
                                                venue_id,
                                                venue_index,
                                                mid,
                                                spread,
                                                market.depth_near_mid
                                            );
                                            *logged = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                CanonicalEvent::Account(event) => {
                    #[cfg(feature = "event_log")]
                    if let Some(writer) = event_log.as_mut() {
                        writer.log_event(&EventLogRecord {
                            tick,
                            now_ms,
                            phase: "account".to_string(),
                            event: EventLogPayload::Account(event.clone()),
                        });
                    }
                    if let Err(_) = cache.apply_account_event(&event) {
                        health_manager.record_api_error(match &event {
                            super::types::AccountEvent::Snapshot(s) => s.venue_index,
                        });
                    }
                    let snapshot = match &event {
                        super::types::AccountEvent::Snapshot(snapshot) => snapshot,
                    };
                    if snapshot.timestamp_ms > 0 {
                        if let Some(last) = last_account_snapshot_ms.get_mut(snapshot.venue_index) {
                            *last = Some(snapshot.timestamp_ms);
                        }
                    }
                    if !account_snapshot_available(snapshot, now_ms, account_snapshot_max_age_ms) {
                        continue;
                    }
                    if let Some(vstate) = state.venues.get(snapshot.venue_index) {
                        let pos_internal = vstate.position_tao;
                        let pos_venue = derive_position_tao(&snapshot.positions);
                        if diff_exceeds(pos_internal, pos_venue, pos_tol) {
                            push_reconcile_drift(
                                &mut pending_drift_events,
                                &audit_dir,
                                ReconcileDriftRecord {
                                    timestamp_ms: snapshot.timestamp_ms,
                                    venue_index: snapshot.venue_index,
                                    venue_id: snapshot.venue_id.clone(),
                                    kind: "position_tao".to_string(),
                                    internal: Some(pos_internal),
                                    venue: Some(pos_venue),
                                    diff: Some(pos_internal - pos_venue),
                                    tolerance: Some(pos_tol),
                                    source: "account_snapshot".to_string(),
                                    available: true,
                                },
                            );
                            if !state.kill_switch {
                                state.kill_switch = true;
                                state.kill_reason = crate::state::KillReason::ReconciliationDrift;
                            }
                        }
                        let bal_internal = vstate.margin_balance_usd;
                        let bal_venue = snapshot.margin.balance_usd;
                        if diff_exceeds(bal_internal, bal_venue, bal_tol) {
                            push_reconcile_drift(
                                &mut pending_drift_events,
                                &audit_dir,
                                ReconcileDriftRecord {
                                    timestamp_ms: snapshot.timestamp_ms,
                                    venue_index: snapshot.venue_index,
                                    venue_id: snapshot.venue_id.clone(),
                                    kind: "margin_balance_usd".to_string(),
                                    internal: Some(bal_internal),
                                    venue: Some(bal_venue),
                                    diff: Some(bal_internal - bal_venue),
                                    tolerance: Some(bal_tol),
                                    source: "account_snapshot".to_string(),
                                    available: true,
                                },
                            );
                            if !state.kill_switch {
                                state.kill_switch = true;
                                state.kill_reason = crate::state::KillReason::ReconciliationDrift;
                            }
                        }
                        let used_internal = vstate.margin_used_usd;
                        let used_venue = snapshot.margin.used_usd;
                        if diff_exceeds(used_internal, used_venue, bal_tol) {
                            push_reconcile_drift(
                                &mut pending_drift_events,
                                &audit_dir,
                                ReconcileDriftRecord {
                                    timestamp_ms: snapshot.timestamp_ms,
                                    venue_index: snapshot.venue_index,
                                    venue_id: snapshot.venue_id.clone(),
                                    kind: "margin_used_usd".to_string(),
                                    internal: Some(used_internal),
                                    venue: Some(used_venue),
                                    diff: Some(used_internal - used_venue),
                                    tolerance: Some(bal_tol),
                                    source: "account_snapshot".to_string(),
                                    available: true,
                                },
                            );
                            if !state.kill_switch {
                                state.kill_switch = true;
                                state.kill_reason = crate::state::KillReason::ReconciliationDrift;
                            }
                        }
                        let avail_internal = vstate.margin_available_usd;
                        let avail_venue = snapshot.margin.available_usd;
                        if diff_exceeds(avail_internal, avail_venue, bal_tol) {
                            push_reconcile_drift(
                                &mut pending_drift_events,
                                &audit_dir,
                                ReconcileDriftRecord {
                                    timestamp_ms: snapshot.timestamp_ms,
                                    venue_index: snapshot.venue_index,
                                    venue_id: snapshot.venue_id.clone(),
                                    kind: "margin_available_usd".to_string(),
                                    internal: Some(avail_internal),
                                    venue: Some(avail_venue),
                                    diff: Some(avail_internal - avail_venue),
                                    tolerance: Some(bal_tol),
                                    source: "account_snapshot".to_string(),
                                    available: true,
                                },
                            );
                            if !state.kill_switch {
                                state.kill_switch = true;
                                state.kill_reason = crate::state::KillReason::ReconciliationDrift;
                            }
                        }
                    }
                }
                CanonicalEvent::Execution(event) => {
                    if deduper.is_duplicate(&event) {
                        continue;
                    }
                    if let super::types::ExecutionEvent::OrderSnapshot(snapshot) = event {
                        #[cfg(feature = "event_log")]
                        if let Some(writer) = event_log.as_mut() {
                            writer.log_event(&EventLogRecord {
                                tick,
                                now_ms,
                                phase: "order_snapshot".to_string(),
                                event: EventLogPayload::OrderSnapshot(snapshot.clone()),
                            });
                        }
                        let internal = state
                            .live_order_state
                            .open_order_ids_by_venue(snapshot.venue_index);
                        let mut venue_orders = snapshot
                            .open_orders
                            .iter()
                            .map(|o| o.order_id.clone())
                            .collect::<Vec<_>>();
                        venue_orders.sort();
                        let diff_count = internal
                            .iter()
                            .filter(|id| !venue_orders.contains(*id))
                            .count()
                            + venue_orders
                                .iter()
                                .filter(|id| !internal.contains(*id))
                                .count();
                        if diff_count > order_tol {
                            push_reconcile_drift(
                                &mut pending_drift_events,
                                &audit_dir,
                                ReconcileDriftRecord {
                                    timestamp_ms: snapshot.timestamp_ms,
                                    venue_index: snapshot.venue_index,
                                    venue_id: snapshot.venue_id.clone(),
                                    kind: "open_orders".to_string(),
                                    internal: Some(internal.len() as f64),
                                    venue: Some(venue_orders.len() as f64),
                                    diff: Some(internal.len() as f64 - venue_orders.len() as f64),
                                    tolerance: Some(order_tol as f64),
                                    source: "order_snapshot".to_string(),
                                    available: true,
                                },
                            );
                            if !state.kill_switch {
                                state.kill_switch = true;
                                state.kill_reason = crate::state::KillReason::ReconciliationDrift;
                            }
                        }
                        state.live_order_state.reconcile(&snapshot, now_ms);
                    } else {
                        #[cfg(feature = "event_log")]
                        if let Some(writer) = event_log.as_mut() {
                            writer.log_event(&EventLogRecord {
                                tick,
                                now_ms,
                                phase: "execution".to_string(),
                                event: EventLogPayload::LiveExecution(event.clone()),
                            });
                        }
                        let core_events = live_events_to_core(&[event]);
                        tick_exec_events.extend(core_events.iter().cloned());
                        let fills = apply_execution_events(&mut state, &core_events, now_ms);
                        if !fills.is_empty() {
                            tick_fills.extend(fills.iter().cloned());
                        }
                        if !fills.is_empty() {
                            fill_batcher.push(now_ms, fills);
                        }
                    }
                }
                CanonicalEvent::OrderSnapshot(snapshot) => {
                    #[cfg(feature = "event_log")]
                    if let Some(writer) = event_log.as_mut() {
                        writer.log_event(&EventLogRecord {
                            tick,
                            now_ms,
                            phase: "order_snapshot".to_string(),
                            event: EventLogPayload::OrderSnapshot(snapshot.clone()),
                        });
                    }
                    state.live_order_state.reconcile(&snapshot, now_ms);
                }
            }
        }

        let snapshot = cache.snapshot(now_ms, cfg.main_loop_interval_ms * 2);
        if snapshot.ready_market_count() > 0 || saw_l2_snapshot_mask_this_tick != 0 {
            saw_ready_once = true;
        }
        coalesce_ready_mask |= saw_l2_snapshot_mask_this_tick;
        last_snapshot = Some(snapshot.clone());
        let mut disabled = health_manager.update_from_snapshot(cfg, &mut state, &snapshot);
        if disable_health_gates {
            for venue in &mut state.venues {
                venue.status = VenueStatus::Healthy;
            }
            disabled.clear();
        }
        let stale_count = snapshot.market.iter().filter(|m| m.is_stale).count() as u64;
        if !disabled.is_empty() {
            for venue_index in &disabled {
                let intent =
                    crate::types::OrderIntent::CancelAll(crate::types::CancelAllOrderIntent {
                        venue_index: Some(*venue_index),
                        venue_id: Some(cfg.venues[*venue_index].id_arc.clone()),
                    });
                would_send_intents.push(intent.clone());
                dispatch_cancel_all_and_apply(
                    cfg,
                    &mut state,
                    &order_tx,
                    now_ms,
                    now_ms.max(0) as u64,
                    intent,
                    hooks.as_ref(),
                    &audit_dir,
                )
                .await;
            }
        }
        if let Some(hooks) = hooks.as_ref() {
            let ready_count = snapshot.ready_market_count();
            let ready = ready_count == cfg.venues.len();
            hooks.health.set_ready(ready);
        }
        let cache_events = snapshot_to_core_events(&snapshot, &state);
        let _ = apply_execution_events(&mut state, &cache_events, now_ms);
        apply_account_snapshot_to_state(cfg, &snapshot, &mut state, now_ms);

        if canary_enabled && !state.kill_switch {
            if let Some(max_pos) = canary_max_position_tao {
                if state.q_global_tao.abs() > max_pos {
                    state.kill_switch = true;
                    state.kill_reason = crate::state::KillReason::CanaryLimitBreach;
                }
            }
            if let Some(max_orders) = canary_max_open_orders {
                let open_orders = state
                    .venues
                    .iter()
                    .map(|v| v.open_orders.len())
                    .sum::<usize>();
                if open_orders > max_orders {
                    state.kill_switch = true;
                    state.kill_reason = crate::state::KillReason::CanaryLimitBreach;
                }
            }
            if canary_stale_max_ticks > 0 {
                if stale_count > 0 {
                    canary_stale_ticks = canary_stale_ticks.saturating_add(1);
                } else {
                    canary_stale_ticks = 0;
                }
                if canary_stale_ticks >= canary_stale_max_ticks {
                    state.kill_switch = true;
                    state.kill_reason = crate::state::KillReason::StaleMarket;
                }
            }
        }

        engine.main_tick_without_risk(&mut state, now_ms);
        if scheduler.risk_due(now_ms) {
            engine.update_risk_limits_and_regime(&mut state);
            scheduler.mark_risk_ran();
        }

        if let Some(hooks) = hooks.as_ref() {
            hooks.metrics.inc_tick(now_ms);
        }

        let mut kill_transition = false;
        if state.kill_switch {
            let kill_transitioned = state.mark_kill_handled(tick);
            kill_transition = kill_transitioned;
            if kill_transitioned {
                handle_kill_switch(
                    cfg,
                    &mut state,
                    &order_tx,
                    now_ms,
                    tick,
                    kill_best_effort_flatten,
                    hooks.as_ref(),
                    &audit_dir,
                )
                .await;
            }
            let _ = flush_batched_fills(&mut fill_batcher, cfg, &mut state, now_ms, true);
        }

        if let Some(hooks) = hooks.as_ref() {
            if let Some(telemetry) = hooks.telemetry.as_ref() {
                update_live_telemetry_stats(
                    telemetry,
                    state.fv_available,
                    stale_count,
                    disabled.len() as u64,
                    kill_transition,
                    &would_send_intents,
                );
                pending_drift_events.sort_by(|a, b| {
                    (a.venue_index, &a.kind, &a.source).cmp(&(b.venue_index, &b.kind, &b.source))
                });
                emit_live_telemetry(
                    &mut telemetry_builder,
                    telemetry,
                    cfg,
                    &state,
                    now_ms,
                    tick,
                    &would_send_intents,
                    &tick_exec_events,
                    &tick_fills,
                    None,
                    None,
                    &pending_drift_events,
                    market_rx_stats.as_ref(),
                );
                pending_drift_events.clear();
            }
        }

        if state.kill_switch {
            maybe_print_market_rx_stats(tick, &market_rx_stats);
            break;
        }

        if snapshot.ready_market_count() == 0 && !smoke_intents {
            maybe_print_market_rx_stats(tick, &market_rx_stats);
            tick += 1;
            continue;
        }

        let mm_quotes = if disable_fv_gate {
            compute_mm_quotes_with_ablations(cfg, &state, &fv_ablations)
        } else {
            compute_mm_quotes(cfg, &state)
        };
        let mut action_id_gen = crate::actions::ActionIdGenerator::new(tick);
        let mm_plan = plan_mm_order_actions(cfg, &state, &mm_quotes, now_ms, &mut action_id_gen);
        let mut intents = mm_plan.intents.clone();
        apply_canary_intent_overrides(
            &mut intents,
            canary_enforce_post_only,
            canary_enforce_reduce_only,
        );
        if intents.is_empty() && smoke_intents {
            let fair = state.fair_value.unwrap_or(state.fair_value_prev).max(1.0);
            let vcfg = &cfg.venues[0];
            intents.push(OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 0,
                venue_id: vcfg.id_arc.clone(),
                side: crate::types::Side::Buy,
                price: fair,
                size: vcfg.base_order_size,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: crate::types::TimeInForce::Gtc,
                post_only: false,
                reduce_only: false,
                client_order_id: None,
            }));
        }
        if !intents.is_empty() {
            would_send_intents.extend(intents.iter().cloned());
        }

        if !intents.is_empty() {
            if let Some(hooks) = hooks.as_ref() {
                hooks.metrics.inc_orders(intents.len());
            }
            let mut action_id_gen = ActionIdGenerator::new(tick);
            let actions = intents_to_actions(&intents, &mut action_id_gen);
            let mut action_batch = ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
            for action in actions {
                action_batch.push(action);
            }
            let (response_tx, mut response_rx) = oneshot::channel();
            let request = LiveOrderRequest {
                intents,
                action_batch,
                now_ms,
                response: response_tx,
            };
            if order_tx.try_send(request).is_ok() {
                let mut events = None;
                for _ in 0..1_000 {
                    match response_rx.try_recv() {
                        Ok(val) => {
                            events = Some(val);
                            break;
                        }
                        Err(oneshot::error::TryRecvError::Closed) => break,
                        Err(oneshot::error::TryRecvError::Empty) => {
                            tokio::task::yield_now().await;
                        }
                    }
                }
                if let Some(events) = events {
                    for event in events {
                        if deduper.is_duplicate(&event) {
                            continue;
                        }
                        if let super::types::ExecutionEvent::OrderSnapshot(snapshot) = event {
                            let internal = state
                                .live_order_state
                                .open_order_ids_by_venue(snapshot.venue_index);
                            let mut venue_orders = snapshot
                                .open_orders
                                .iter()
                                .map(|o| o.order_id.clone())
                                .collect::<Vec<_>>();
                            venue_orders.sort();
                            let diff_count = internal
                                .iter()
                                .filter(|id| !venue_orders.contains(*id))
                                .count()
                                + venue_orders
                                    .iter()
                                    .filter(|id| !internal.contains(*id))
                                    .count();
                            if diff_count > order_tol {
                                push_reconcile_drift(
                                    &mut pending_drift_events,
                                    &audit_dir,
                                    ReconcileDriftRecord {
                                        timestamp_ms: snapshot.timestamp_ms,
                                        venue_index: snapshot.venue_index,
                                        venue_id: snapshot.venue_id.clone(),
                                        kind: "open_orders".to_string(),
                                        internal: Some(internal.len() as f64),
                                        venue: Some(venue_orders.len() as f64),
                                        diff: Some(
                                            internal.len() as f64 - venue_orders.len() as f64,
                                        ),
                                        tolerance: Some(order_tol as f64),
                                        source: "order_snapshot".to_string(),
                                        available: true,
                                    },
                                );
                                if !state.kill_switch {
                                    state.kill_switch = true;
                                    state.kill_reason =
                                        crate::state::KillReason::ReconciliationDrift;
                                }
                            }
                            state.live_order_state.reconcile(&snapshot, now_ms);
                            continue;
                        }
                        #[cfg(feature = "event_log")]
                        log_live_execution_event(&mut event_log, tick, now_ms, "gateway", &event);
                        let core_events = live_events_to_core(&[event]);
                        tick_exec_events.extend(core_events.iter().cloned());
                        let fills = apply_execution_events(&mut state, &core_events, now_ms);
                        if !fills.is_empty() {
                            tick_fills.extend(fills.iter().cloned());
                        }
                        if !fills.is_empty() {
                            fill_batcher.push(now_ms, fills);
                        }
                    }
                }
            }
        }

        let did_flush = flush_batched_fills(&mut fill_batcher, cfg, &mut state, now_ms, false);

        if did_flush {
            if cfg.exit.enabled {
                let mut exit_intents = exit::compute_exit_intents(cfg, &state, now_ms);
                apply_canary_intent_overrides(
                    &mut exit_intents,
                    canary_enforce_post_only,
                    canary_enforce_reduce_only,
                );
                if !exit_intents.is_empty() {
                    would_send_intents.extend(exit_intents.iter().cloned());
                    if let Some(hooks) = hooks.as_ref() {
                        hooks.metrics.inc_orders(exit_intents.len());
                    }
                    let mut action_id_gen = ActionIdGenerator::new(tick);
                    let actions = intents_to_actions(&exit_intents, &mut action_id_gen);
                    let mut action_batch =
                        ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
                    for action in actions {
                        action_batch.push(action);
                    }
                    let (response_tx, mut response_rx) = oneshot::channel();
                    let request = LiveOrderRequest {
                        intents: exit_intents,
                        action_batch,
                        now_ms,
                        response: response_tx,
                    };
                    if order_tx.try_send(request).is_ok() {
                        let mut events = None;
                        for _ in 0..1_000 {
                            match response_rx.try_recv() {
                                Ok(val) => {
                                    events = Some(val);
                                    break;
                                }
                                Err(oneshot::error::TryRecvError::Closed) => break,
                                Err(oneshot::error::TryRecvError::Empty) => {
                                    tokio::task::yield_now().await;
                                }
                            }
                        }
                        if let Some(events) = events {
                            let mut exit_fills = Vec::new();
                            for event in events {
                                if deduper.is_duplicate(&event) {
                                    continue;
                                }
                                if let super::types::ExecutionEvent::OrderSnapshot(snapshot) = event
                                {
                                    state.live_order_state.reconcile(&snapshot, now_ms);
                                    continue;
                                }
                                #[cfg(feature = "event_log")]
                                log_live_execution_event(
                                    &mut event_log,
                                    tick,
                                    now_ms,
                                    "gateway",
                                    &event,
                                );
                                let core_events = live_events_to_core(&[event]);
                                tick_exec_events.extend(core_events.iter().cloned());
                                let fills =
                                    apply_execution_events(&mut state, &core_events, now_ms);
                                if !fills.is_empty() {
                                    tick_fills.extend(fills.iter().cloned());
                                    exit_fills.extend(fills);
                                }
                            }
                            if !exit_fills.is_empty() {
                                apply_live_fills(cfg, &mut state, &exit_fills, now_ms);
                                state.recompute_after_fills(cfg);
                            }
                        }
                    }
                }
            }

            if scheduler.hedge_due(now_ms) {
                if let Some(plan) = compute_hedge_plan(cfg, &state, now_ms) {
                    let mut hedge_intents = hedge_plan_to_order_intents(&plan);
                    apply_canary_intent_overrides(
                        &mut hedge_intents,
                        canary_enforce_post_only,
                        canary_enforce_reduce_only,
                    );
                    if !hedge_intents.is_empty() {
                        would_send_intents.extend(hedge_intents.iter().cloned());
                        if let Some(hooks) = hooks.as_ref() {
                            hooks.metrics.inc_orders(hedge_intents.len());
                        }
                        let mut action_id_gen = ActionIdGenerator::new(tick);
                        let actions = intents_to_actions(&hedge_intents, &mut action_id_gen);
                        let mut action_batch =
                            ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
                        for action in actions {
                            action_batch.push(action);
                        }
                        let (response_tx, mut response_rx) = oneshot::channel();
                        let request = LiveOrderRequest {
                            intents: hedge_intents,
                            action_batch,
                            now_ms,
                            response: response_tx,
                        };
                        if order_tx.try_send(request).is_ok() {
                            let mut events = None;
                            for _ in 0..1_000 {
                                match response_rx.try_recv() {
                                    Ok(val) => {
                                        events = Some(val);
                                        break;
                                    }
                                    Err(oneshot::error::TryRecvError::Closed) => break,
                                    Err(oneshot::error::TryRecvError::Empty) => {
                                        tokio::task::yield_now().await;
                                    }
                                }
                            }
                            if let Some(events) = events {
                                let mut hedge_fills = Vec::new();
                                for event in events {
                                    if deduper.is_duplicate(&event) {
                                        continue;
                                    }
                                    if let super::types::ExecutionEvent::OrderSnapshot(snapshot) =
                                        event
                                    {
                                        state.live_order_state.reconcile(&snapshot, now_ms);
                                        continue;
                                    }
                                    #[cfg(feature = "event_log")]
                                    log_live_execution_event(
                                        &mut event_log,
                                        tick,
                                        now_ms,
                                        "gateway",
                                        &event,
                                    );
                                    let core_events = live_events_to_core(&[event]);
                                    tick_exec_events.extend(core_events.iter().cloned());
                                    let fills =
                                        apply_execution_events(&mut state, &core_events, now_ms);
                                    if !fills.is_empty() {
                                        tick_fills.extend(fills.iter().cloned());
                                        hedge_fills.extend(fills);
                                    }
                                }
                                if !hedge_fills.is_empty() {
                                    apply_live_fills(cfg, &mut state, &hedge_fills, now_ms);
                                    state.recompute_after_fills(cfg);
                                }
                            }
                        }
                    }
                }
                scheduler.mark_hedge_ran();
            }
        }

        if state.kill_switch {
            maybe_print_market_rx_stats(tick, &market_rx_stats);
            break;
        }

        maybe_print_market_rx_stats(tick, &market_rx_stats);
        tick += 1;
        if let LiveRunMode::Step { .. } = mode {
            tokio::task::yield_now().await;
        }
    }

    if let Some(hooks) = hooks.as_ref() {
        hooks.health.set_ready(hooks.health.is_ready());
    }

    let _ = flush_batched_fills(&mut fill_batcher, cfg, &mut state, last_now_ms, true);

    let (ready_market_count, stale_market_count) = if let Some(snapshot) = last_snapshot {
        let ready_market = snapshot.ready_market_count();
        let stale_market = snapshot.market.iter().filter(|m| m.is_stale).count();
        (ready_market, stale_market)
    } else {
        (0, 0)
    };

    let (local_vol_short_avg, local_vol_long_avg) = compute_local_vol_avgs(&state.venues);

    LiveRunSummary {
        ticks_run: tick,
        kill_switch: state.kill_switch,
        fv_available: state.fv_available,
        ready_market_count,
        stale_market_count,
        local_vol_short_avg,
        local_vol_long_avg,
    }
}

pub async fn handle_kill_switch(
    cfg: &Config,
    state: &mut GlobalState,
    order_tx: &mpsc::Sender<LiveOrderRequest>,
    now_ms: TimestampMs,
    tick: u64,
    best_effort_flatten: bool,
    hooks: Option<&LiveRuntimeHooks>,
    audit_dir: &PathBuf,
) {
    let kill_event = state.build_kill_event(tick, cfg);
    if let Ok(line) = serde_json::to_string(&kill_event) {
        println!("{line}");
        if audit_dir.exists() {
            let path = audit_dir.join("kill_events.jsonl");
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(file, "{}", line);
            }
        }
    }

    for (venue_index, venue) in cfg.venues.iter().enumerate() {
        let intent = OrderIntent::CancelAll(crate::types::CancelAllOrderIntent {
            venue_index: Some(venue_index),
            venue_id: Some(venue.id_arc.clone()),
        });
        dispatch_cancel_all_and_apply(cfg, state, order_tx, now_ms, tick, intent, hooks, audit_dir)
            .await;
    }

    if best_effort_flatten {
        if let Some(intent) = state.best_effort_kill_intent_exit_first(cfg, tick) {
            let mut action_id_gen = ActionIdGenerator::new(tick);
            let actions = intents_to_actions(&[intent.clone()], &mut action_id_gen);
            let mut action_batch = ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
            for action in actions {
                action_batch.push(action);
            }
            let (response_tx, mut response_rx) = oneshot::channel();
            let request = LiveOrderRequest {
                intents: vec![intent],
                action_batch,
                now_ms,
                response: response_tx,
            };
            let _ = order_tx.try_send(request);
            for _ in 0..1_000 {
                match response_rx.try_recv() {
                    Ok(events) => {
                        #[cfg(feature = "event_log")]
                        log_live_execution_events_env(tick, now_ms, "gateway", &events);
                        let core_events = live_events_to_core(&events);
                        let fills = apply_execution_events(state, &core_events, now_ms);
                        if !fills.is_empty() {
                            apply_live_fills(cfg, state, &fills, now_ms);
                            state.recompute_after_fills(cfg);
                        }
                        break;
                    }
                    Err(oneshot::error::TryRecvError::Closed) => break,
                    Err(oneshot::error::TryRecvError::Empty) => tokio::task::yield_now().await,
                }
            }
        }
    }
}

async fn dispatch_cancel_all_and_apply(
    cfg: &Config,
    state: &mut GlobalState,
    order_tx: &mpsc::Sender<LiveOrderRequest>,
    now_ms: TimestampMs,
    tick: u64,
    intent: OrderIntent,
    hooks: Option<&LiveRuntimeHooks>,
    _audit_dir: &PathBuf,
) {
    let mut action_id_gen = ActionIdGenerator::new(tick);
    let actions = intents_to_actions(&[intent.clone()], &mut action_id_gen);
    let mut action_batch = ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
    for action in actions {
        action_batch.push(action);
    }
    let (response_tx, mut response_rx) = oneshot::channel();
    let request = LiveOrderRequest {
        intents: vec![intent],
        action_batch,
        now_ms,
        response: response_tx,
    };
    let _ = order_tx.try_send(request);
    if let Some(hooks) = hooks {
        hooks.metrics.inc_cancel_all();
    }
    for _ in 0..1_000 {
        match response_rx.try_recv() {
            Ok(events) => {
                #[cfg(feature = "event_log")]
                log_live_execution_events_env(tick, now_ms, "gateway", &events);
                let core_events = live_events_to_core(&events);
                let fills = apply_execution_events(state, &core_events, now_ms);
                if !fills.is_empty() {
                    apply_live_fills(cfg, state, &fills, now_ms);
                    state.recompute_after_fills(cfg);
                }
                break;
            }
            Err(oneshot::error::TryRecvError::Closed) => break,
            Err(oneshot::error::TryRecvError::Empty) => tokio::task::yield_now().await,
        }
    }
}

fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as TimestampMs
}

fn snapshot_to_core_events(
    snapshot: &super::state_cache::CanonicalCacheSnapshot,
    state: &GlobalState,
) -> Vec<ExecutionEvent> {
    let mut events = Vec::new();
    for venue in &snapshot.account {
        if venue.is_stale {
            continue;
        }
        if let Some(v) = state.venues.get(venue.venue_index) {
            events.push(ExecutionEvent::BalanceUpdate(crate::types::BalanceUpdate {
                venue_index: venue.venue_index,
                venue_id: v.id.clone(),
                margin_balance_usd: venue.margin_balance_usd,
                margin_used_usd: venue.margin_used_usd,
                margin_available_usd: venue.margin_available_usd,
            }));
        }
    }
    events
}

fn apply_market_event_to_core(
    state: &mut GlobalState,
    cfg: &Config,
    event: &super::types::MarketDataEvent,
) {
    let max_levels = cfg.book.depth_levels.max(1) as usize;
    let alpha_short = cfg.volatility.fv_vol_alpha_short;
    let alpha_long = cfg.volatility.fv_vol_alpha_long;
    match event {
        super::types::MarketDataEvent::L2Snapshot(snapshot) => {
            if let Some(v) = state.venues.get_mut(snapshot.venue_index) {
                let _ = v.apply_l2_snapshot(
                    &snapshot.bids,
                    &snapshot.asks,
                    snapshot.seq,
                    snapshot.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                );
            }
        }
        super::types::MarketDataEvent::L2Delta(delta) => {
            if let Some(v) = state.venues.get_mut(delta.venue_index) {
                let _ = v.apply_l2_delta(
                    &delta.changes,
                    delta.seq,
                    delta.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                );
            }
        }
        super::types::MarketDataEvent::Trade(_)
        | super::types::MarketDataEvent::FundingUpdate(_) => {}
    }
}

fn compute_local_vol_avgs(venues: &[VenueState]) -> (f64, f64) {
    if venues.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_short = 0.0;
    let mut sum_long = 0.0;
    for v in venues {
        sum_short += v.local_vol_short;
        sum_long += v.local_vol_long;
    }
    let denom = venues.len() as f64;
    (sum_short / denom, sum_long / denom)
}

fn update_live_telemetry_stats(
    telemetry: &LiveTelemetry,
    fv_available: bool,
    stale_count: u64,
    disabled_count: u64,
    kill_transition: bool,
    would_send_intents: &[OrderIntent],
) {
    let mut guard = match telemetry.stats.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    guard.ticks_total += 1;
    if fv_available {
        guard.fv_available_ticks += 1;
    }
    guard.venue_staleness_events += stale_count;
    guard.venue_disabled_events += disabled_count;
    if kill_transition {
        guard.kill_events += 1;
    }
    for intent in would_send_intents {
        match intent {
            OrderIntent::Place(place) => {
                let key = format!("{:?}", place.purpose);
                *guard.would_place_by_purpose.entry(key).or_insert(0) += 1;
            }
            OrderIntent::Cancel(_) | OrderIntent::CancelAll(_) => {
                *guard
                    .would_cancel_by_purpose
                    .entry("unknown".to_string())
                    .or_insert(0) += 1;
            }
            OrderIntent::Replace(replace) => {
                let key = format!("{:?}", replace.purpose);
                *guard.would_replace_by_purpose.entry(key).or_insert(0) += 1;
            }
        }
    }
}

fn emit_live_telemetry(
    builder: &mut TelemetryBuilder,
    telemetry: &LiveTelemetry,
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    tick: u64,
    would_send_intents: &[OrderIntent],
    exec_events: &[ExecutionEvent],
    fills: &[crate::types::FillEvent],
    last_exit_intent: Option<&OrderIntent>,
    last_hedge_intent: Option<&OrderIntent>,
    reconcile_drift: &[ReconcileDriftRecord],
    market_rx_stats: Option<&MarketRxStats>,
) {
    let mut record = builder.build_record(TelemetryInputs {
        cfg,
        state,
        tick,
        now_ms,
        intents: would_send_intents,
        exec_events,
        fills,
        last_exit_intent,
        last_hedge_intent,
        kill_event: None,
        shadow_mode: telemetry.shadow_mode,
        execution_mode: telemetry.execution_mode,
        reconcile_drift,
        max_orders_per_tick: telemetry.max_orders_per_tick,
    });
    ensure_schema_v1(&mut record);
    if let Some(stats) = market_rx_stats {
        if let serde_json::Value::Object(ref mut map) = record {
            map.insert(
                "market_rx_stats".to_string(),
                json!({
                    "drained": stats.drained,
                    "l2_delta": stats.l2_delta,
                    "l2_snapshot": stats.l2_snapshot,
                    "trade": stats.trade,
                    "funding_update": stats.funding_update,
                    "out_market": stats.out_market,
                    "out_l2_delta": stats.out_l2_delta,
                    "out_l2_snapshot": stats.out_l2_snapshot,
                    "out_trade": stats.out_trade,
                    "out_funding_update": stats.out_funding_update,
                    "cap_hits": stats.cap_hits
                }),
            );
        }
    }
    if let Ok(mut guard) = telemetry.sink.lock() {
        guard.log_json(&record);
    }
}

fn live_events_to_core(events: &[LiveExecutionEvent]) -> Vec<ExecutionEvent> {
    let mut out = Vec::new();
    for event in events {
        match event {
            LiveExecutionEvent::OrderAccepted(ack) => {
                out.push(ExecutionEvent::OrderAck(OrderAck {
                    venue_index: ack.venue_index,
                    venue_id: ack.venue_id.as_str().into(),
                    order_id: ack.order_id.clone(),
                    client_order_id: ack.client_order_id.clone(),
                    seq: Some(ack.seq),
                    side: Some(ack.side),
                    price: Some(ack.price),
                    size: Some(ack.size),
                    purpose: Some(ack.purpose),
                }));
            }
            LiveExecutionEvent::OrderRejected(rej) => {
                out.push(ExecutionEvent::OrderReject(OrderReject {
                    venue_index: rej.venue_index,
                    venue_id: rej.venue_id.as_str().into(),
                    order_id: rej.order_id.clone(),
                    client_order_id: None,
                    seq: Some(rej.seq),
                    reason: rej.reason.clone(),
                }));
            }
            LiveExecutionEvent::CancelAccepted(cancel) => {
                out.push(ExecutionEvent::OrderAck(OrderAck {
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.as_str().into(),
                    order_id: cancel.order_id.clone(),
                    client_order_id: None,
                    seq: Some(cancel.seq),
                    side: None,
                    price: None,
                    size: None,
                    purpose: None,
                }));
            }
            LiveExecutionEvent::CancelRejected(rej) => {
                out.push(ExecutionEvent::OrderReject(OrderReject {
                    venue_index: rej.venue_index,
                    venue_id: rej.venue_id.as_str().into(),
                    order_id: rej.order_id.clone(),
                    client_order_id: None,
                    seq: Some(rej.seq),
                    reason: rej.reason.clone(),
                }));
            }
            LiveExecutionEvent::Filled(fill) => {
                out.push(ExecutionEvent::Fill(crate::types::FillEvent {
                    venue_index: fill.venue_index,
                    venue_id: fill.venue_id.as_str().into(),
                    order_id: fill.order_id.clone(),
                    client_order_id: fill.client_order_id.clone(),
                    seq: Some(fill.seq),
                    side: fill.side,
                    price: fill.price,
                    size: fill.size,
                    purpose: fill.purpose,
                    fee_bps: fill.fee_bps,
                }));
            }
            LiveExecutionEvent::CancelAllAccepted(cancel) => {
                out.push(ExecutionEvent::OrderAck(OrderAck {
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.as_str().into(),
                    order_id: "cancel_all".to_string(),
                    client_order_id: None,
                    seq: Some(cancel.seq),
                    side: None,
                    price: None,
                    size: None,
                    purpose: None,
                }));
            }
            LiveExecutionEvent::CancelAllRejected(rej) => {
                out.push(ExecutionEvent::OrderReject(OrderReject {
                    venue_index: rej.venue_index,
                    venue_id: rej.venue_id.as_str().into(),
                    order_id: None,
                    client_order_id: None,
                    seq: Some(rej.seq),
                    reason: rej.reason.clone(),
                }));
            }
            LiveExecutionEvent::OrderSnapshot(_) => {}
        }
    }
    out
}

fn apply_live_fills(
    cfg: &Config,
    state: &mut GlobalState,
    fills: &[crate::types::FillEvent],
    now_ms: TimestampMs,
) {
    for fill in fills {
        state.apply_fill_event(fill, now_ms, cfg);
    }
}

pub fn apply_account_snapshot_to_state(
    _cfg: &Config,
    snapshot: &super::state_cache::CanonicalCacheSnapshot,
    state: &mut GlobalState,
    now_ms: TimestampMs,
) {
    let sigma_eff = state.sigma_eff;
    for acct in &snapshot.account {
        if acct.is_stale {
            continue;
        }
        let Some(v) = state.venues.get_mut(acct.venue_index) else {
            continue;
        };
        v.position_tao = acct.position_tao;
        v.avg_entry_price = if acct.position_tao.abs() > 0.0 {
            acct.avg_entry_price
        } else {
            0.0
        };
        v.margin_balance_usd = acct.margin_balance_usd;
        v.margin_used_usd = acct.margin_used_usd;
        v.margin_available_usd = acct.margin_available_usd;
        v.price_liq = acct.price_liq;
        if let Some(funding) = acct.funding_8h {
            v.funding_8h = funding;
        }

        let price_liq = match acct.price_liq {
            Some(val) if val.is_finite() && val > 0.0 => val,
            _ => continue,
        };
        if !sigma_eff.is_finite() || sigma_eff <= 0.0 {
            continue;
        }
        let mid = v
            .mid
            .or_else(|| snapshot.market.get(acct.venue_index).and_then(|m| m.mid));
        let Some(mid) = mid else {
            continue;
        };
        let s_t = match state.fair_value {
            Some(fair) if fair.is_finite() && fair > 0.0 => fair,
            _ => {
                eprintln!(
                    "live_account_ingest | venue={} now_ms={} fair_value_missing=true use_mid_proxy=true",
                    v.id,
                    now_ms
                );
                mid
            }
        };
        if s_t.is_finite() && s_t > 0.0 {
            v.dist_liq_sigma = (mid - price_liq).abs() / (sigma_eff * s_t);
        } else if let Some(dist) = acct.dist_liq_sigma {
            v.dist_liq_sigma = dist;
        }
    }
}

#[cfg(feature = "event_log")]
fn log_live_execution_event(
    event_log: &mut Option<EventLogWriter>,
    tick: u64,
    now_ms: TimestampMs,
    phase: &str,
    event: &super::types::ExecutionEvent,
) {
    let Some(writer) = event_log.as_mut() else {
        return;
    };
    let payload = match event {
        super::types::ExecutionEvent::OrderSnapshot(snapshot) => {
            EventLogPayload::OrderSnapshot(snapshot.clone())
        }
        _ => EventLogPayload::LiveExecution(event.clone()),
    };
    writer.log_event(&EventLogRecord {
        tick,
        now_ms,
        phase: phase.to_string(),
        event: payload,
    });
}

#[cfg(feature = "event_log")]
fn log_live_execution_events_env(
    tick: u64,
    now_ms: TimestampMs,
    phase: &str,
    events: &[super::types::ExecutionEvent],
) {
    let Some(mut writer) = EventLogWriter::from_env() else {
        return;
    };
    for event in events {
        let payload = match event {
            super::types::ExecutionEvent::OrderSnapshot(snapshot) => {
                EventLogPayload::OrderSnapshot(snapshot.clone())
            }
            _ => EventLogPayload::LiveExecution(event.clone()),
        };
        writer.log_event(&EventLogRecord {
            tick,
            now_ms,
            phase: phase.to_string(),
            event: payload,
        });
    }
}

#[cfg(feature = "event_log")]
pub fn replay_event_log(
    cfg: &Config,
    event_log_path: &std::path::Path,
    telemetry_path: &std::path::Path,
    max_ticks: Option<u64>,
) -> LiveRunSummary {
    let records = read_event_log(event_log_path).unwrap_or_default();
    let engine = Engine::new(cfg);
    let mut state = GlobalState::new(cfg);
    let mut cache = LiveStateCache::new(cfg);
    let mut health_manager = VenueHealthManager::new(cfg);
    let mut fill_batcher = FillBatcher::new(cfg.fill_agg_interval_ms);
    let mut deduper = ExecutionEventDeduper::new(256);
    let mut scheduler: Option<LoopScheduler> = None;

    let telemetry_cfg = TelemetryConfig {
        mode: TelemetryMode::Jsonl,
        path: Some(telemetry_path.to_path_buf()),
        append: false,
    };
    let telemetry_sink = TelemetrySink::from_config(telemetry_cfg);
    let telemetry = LiveTelemetry {
        sink: Arc::new(Mutex::new(telemetry_sink)),
        shadow_mode: false,
        execution_mode: "replay",
        max_orders_per_tick: 200,
        stats: Arc::new(Mutex::new(LiveTelemetryStats::default())),
    };
    let mut telemetry_builder = TelemetryBuilder::new(cfg);

    let mut last_snapshot: Option<CanonicalCacheSnapshot> = None;
    let mut current_tick: Option<u64> = None;
    let mut current_now_ms: TimestampMs = 0;
    let mut ticks_run: u64 = 0;
    let mut tick_exec_events: Vec<ExecutionEvent> = Vec::new();
    let mut tick_fills: Vec<crate::types::FillEvent> = Vec::new();

    for record in records {
        match &record.event {
            EventLogPayload::Tick => {
                if let Some(tick) = current_tick {
                    last_snapshot = Some(flush_replay_tick(
                        cfg,
                        &engine,
                        &mut state,
                        &mut cache,
                        &mut health_manager,
                        &mut fill_batcher,
                        scheduler.as_mut(),
                        &mut telemetry_builder,
                        &telemetry,
                        tick,
                        current_now_ms,
                        &tick_exec_events,
                        &tick_fills,
                    ));
                    ticks_run += 1;
                    tick_exec_events.clear();
                    tick_fills.clear();
                    if let Some(limit) = max_ticks {
                        if ticks_run >= limit {
                            break;
                        }
                    }
                }
                current_tick = Some(record.tick);
                current_now_ms = record.now_ms;
                if scheduler.is_none() {
                    let sched = LoopScheduler::new(
                        current_now_ms,
                        cfg.main_loop_interval_ms,
                        cfg.hedge_loop_interval_ms,
                        cfg.risk_loop_interval_ms,
                    );
                    fill_batcher.set_last_flush_ms(sched.next_main_ms() - cfg.fill_agg_interval_ms);
                    scheduler = Some(sched);
                }
            }
            EventLogPayload::MarketData(event) => {
                let _ = cache.apply_market_event(event);
                apply_market_event_to_core(&mut state, cfg, event);
            }
            EventLogPayload::Account(event) => {
                let _ = cache.apply_account_event(event);
            }
            EventLogPayload::LiveExecution(event) => {
                if deduper.is_duplicate(event) {
                    continue;
                }
                let core_events = live_events_to_core(&[event.clone()]);
                tick_exec_events.extend(core_events.iter().cloned());
                let fills = apply_execution_events(&mut state, &core_events, current_now_ms);
                if !fills.is_empty() {
                    tick_fills.extend(fills.iter().cloned());
                    fill_batcher.push(current_now_ms, fills);
                }
            }
            EventLogPayload::OrderSnapshot(snapshot) => {
                state.live_order_state.reconcile(snapshot, current_now_ms);
            }
            EventLogPayload::Execution(event) => {
                let core_events = vec![event.to_execution_event()];
                tick_exec_events.extend(core_events.iter().cloned());
                let fills = apply_execution_events(&mut state, &core_events, current_now_ms);
                if !fills.is_empty() {
                    tick_fills.extend(fills.iter().cloned());
                    fill_batcher.push(current_now_ms, fills);
                }
            }
        }
    }

    if let Some(tick) = current_tick {
        last_snapshot = Some(flush_replay_tick(
            cfg,
            &engine,
            &mut state,
            &mut cache,
            &mut health_manager,
            &mut fill_batcher,
            scheduler.as_mut(),
            &mut telemetry_builder,
            &telemetry,
            tick,
            current_now_ms,
            &tick_exec_events,
            &tick_fills,
        ));
        ticks_run += 1;
    }

    let (ready_market_count, stale_market_count) = if let Some(snapshot) = last_snapshot {
        let ready_market = snapshot.ready_market_count();
        let stale_market = snapshot.market.iter().filter(|m| m.is_stale).count();
        (ready_market, stale_market)
    } else {
        (0, 0)
    };
    let (local_vol_short_avg, local_vol_long_avg) = compute_local_vol_avgs(&state.venues);
    LiveRunSummary {
        ticks_run,
        kill_switch: state.kill_switch,
        fv_available: state.fv_available,
        ready_market_count,
        stale_market_count,
        local_vol_short_avg,
        local_vol_long_avg,
    }
}

#[cfg(feature = "event_log")]
fn flush_replay_tick(
    cfg: &Config,
    engine: &Engine,
    state: &mut GlobalState,
    cache: &mut LiveStateCache,
    health_manager: &mut VenueHealthManager,
    fill_batcher: &mut FillBatcher,
    scheduler: Option<&mut LoopScheduler>,
    telemetry_builder: &mut TelemetryBuilder,
    telemetry: &LiveTelemetry,
    tick: u64,
    now_ms: TimestampMs,
    exec_events: &[ExecutionEvent],
    fills: &[crate::types::FillEvent],
) -> CanonicalCacheSnapshot {
    let snapshot = cache.snapshot(now_ms, cfg.main_loop_interval_ms * 2);
    let disabled = health_manager.update_from_snapshot(cfg, state, &snapshot);
    let cache_events = snapshot_to_core_events(&snapshot, state);
    let _ = apply_execution_events(state, &cache_events, now_ms);
    apply_account_snapshot_to_state(cfg, &snapshot, state, now_ms);
    engine.main_tick_without_risk(state, now_ms);
    if let Some(scheduler) = scheduler {
        if scheduler.risk_due(now_ms) {
            engine.update_risk_limits_and_regime(state);
            scheduler.mark_risk_ran();
        }
    }
    update_live_telemetry_stats(
        telemetry,
        state.fv_available,
        snapshot.market.iter().filter(|m| m.is_stale).count() as u64,
        disabled.len() as u64,
        false,
        &[],
    );
    emit_live_telemetry(
        telemetry_builder,
        telemetry,
        cfg,
        state,
        now_ms,
        tick,
        &[],
        exec_events,
        fills,
        None,
        None,
        &[],
        None,
    );
    let _ = flush_batched_fills(fill_batcher, cfg, state, now_ms, true);
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::types;
    use crate::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
    use crate::types::{OrderPurpose, Side};
    use tokio::sync::mpsc;

    #[test]
    fn dedupe_order_ack_by_client_order_id() {
        let mut deduper = ExecutionEventDeduper::new(10);
        let event = types::ExecutionEvent::OrderAccepted(types::OrderAccepted {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 10,
            timestamp_ms: 1_700_000_000_000,
            order_id: "oid_1".to_string(),
            client_order_id: Some("co_1".to_string()),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
        });
        assert!(!deduper.is_duplicate(&event));
        assert!(deduper.is_duplicate(&event));
    }

    #[test]
    fn dedupe_fill_by_fill_id() {
        let mut deduper = ExecutionEventDeduper::new(10);
        let event = types::ExecutionEvent::Filled(types::Fill {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 11,
            timestamp_ms: 1_700_000_000_100,
            order_id: Some("oid_1".to_string()),
            client_order_id: Some("co_1".to_string()),
            fill_id: Some("fill_1".to_string()),
            side: Side::Buy,
            price: 100.0,
            size: 0.5,
            purpose: OrderPurpose::Mm,
            fee_bps: 1.0,
        });
        assert!(!deduper.is_duplicate(&event));
        assert!(deduper.is_duplicate(&event));
    }

    #[test]
    fn coalesced_deltas_fold_into_snapshot() {
        let (market_tx, mut market_rx) = mpsc::channel(16);
        let (_account_tx, mut account_rx) = mpsc::channel(1);
        let mut exec_rx: Option<mpsc::Receiver<types::ExecutionEvent>> = None;
        let mut order_snapshot_rx: Option<mpsc::Receiver<types::OrderSnapshot>> = None;
        let coalesce_ready_mask: u64 = 0;
        let mut saw_l2_snapshot_mask_this_tick: u64 = 0;

        let snapshot = types::L2Snapshot {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 10,
            timestamp_ms: 1_700_000_000_000,
            bids: vec![BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        market_tx
            .try_send(types::MarketDataEvent::L2Snapshot(snapshot))
            .unwrap();
        for (seq, price, size) in [(11, 100.0, 2.0), (12, 101.0, 2.0), (13, 99.0, 1.0)] {
            let delta = types::L2Delta {
                venue_index: 0,
                venue_id: "TAO".to_string(),
                seq,
                timestamp_ms: 1_700_000_000_000 + seq as i64,
                changes: vec![BookLevelDelta {
                    side: BookSide::Bid,
                    price,
                    size,
                }],
            };
            market_tx
                .try_send(types::MarketDataEvent::L2Delta(delta))
                .unwrap();
        }
        drop(market_tx);

        let out = drain_ordered_events(
            &mut market_rx,
            &mut account_rx,
            &mut exec_rx,
            &mut order_snapshot_rx,
            None,
            true,
            true,
            coalesce_ready_mask,
            &mut saw_l2_snapshot_mask_this_tick,
        );

        let mut snapshots = 0;
        let mut deltas = 0;
        let mut snapshot_seq = None;
        for event in out {
            if let CanonicalEvent::Market(market) = event.event {
                match market {
                    types::MarketDataEvent::L2Snapshot(s) => {
                        snapshots += 1;
                        snapshot_seq = Some(s.seq);
                    }
                    types::MarketDataEvent::L2Delta(_) => {
                        deltas += 1;
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(snapshots, 1);
        assert_eq!(deltas, 0);
        assert_eq!(snapshot_seq, Some(13));
    }

    #[test]
    fn non_contiguous_deltas_do_not_fold() {
        let (market_tx, mut market_rx) = mpsc::channel(16);
        let (_account_tx, mut account_rx) = mpsc::channel(1);
        let mut exec_rx: Option<mpsc::Receiver<types::ExecutionEvent>> = None;
        let mut order_snapshot_rx: Option<mpsc::Receiver<types::OrderSnapshot>> = None;
        let coalesce_ready_mask: u64 = 0;
        let mut saw_l2_snapshot_mask_this_tick: u64 = 0;

        let snapshot = types::L2Snapshot {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 10,
            timestamp_ms: 1_700_000_000_000,
            bids: vec![BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        market_tx
            .try_send(types::MarketDataEvent::L2Snapshot(snapshot))
            .unwrap();
        for (seq, price, size) in [(12, 100.0, 2.0), (13, 99.0, 1.0)] {
            let delta = types::L2Delta {
                venue_index: 0,
                venue_id: "TAO".to_string(),
                seq,
                timestamp_ms: 1_700_000_000_000 + seq as i64,
                changes: vec![BookLevelDelta {
                    side: BookSide::Bid,
                    price,
                    size,
                }],
            };
            market_tx
                .try_send(types::MarketDataEvent::L2Delta(delta))
                .unwrap();
        }
        drop(market_tx);

        let out = drain_ordered_events(
            &mut market_rx,
            &mut account_rx,
            &mut exec_rx,
            &mut order_snapshot_rx,
            None,
            true,
            true,
            coalesce_ready_mask,
            &mut saw_l2_snapshot_mask_this_tick,
        );

        let mut snapshots = 0;
        let mut deltas = 0;
        let mut snapshot_seq = None;
        for event in out {
            if let CanonicalEvent::Market(market) = event.event {
                match market {
                    types::MarketDataEvent::L2Snapshot(s) => {
                        snapshots += 1;
                        snapshot_seq = Some(s.seq);
                    }
                    types::MarketDataEvent::L2Delta(_) => {
                        deltas += 1;
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(snapshots, 1);
        assert_eq!(deltas, 2);
        assert_eq!(snapshot_seq, Some(10));
    }

    #[test]
    fn snapshot_dominates_lower_or_equal_deltas() {
        let (market_tx, mut market_rx) = mpsc::channel(16);
        let (_account_tx, mut account_rx) = mpsc::channel(1);
        let mut exec_rx: Option<mpsc::Receiver<types::ExecutionEvent>> = None;
        let mut order_snapshot_rx: Option<mpsc::Receiver<types::OrderSnapshot>> = None;
        let coalesce_ready_mask: u64 = 0;
        let mut saw_l2_snapshot_mask_this_tick: u64 = 0;

        for seq in [9_u64, 10_u64] {
            let delta = types::L2Delta {
                venue_index: 0,
                venue_id: "TAO".to_string(),
                seq,
                timestamp_ms: 1_700_000_000_000 + seq as i64,
                changes: vec![BookLevelDelta {
                    side: BookSide::Bid,
                    price: 100.0 + seq as f64,
                    size: 1.0,
                }],
            };
            market_tx
                .try_send(types::MarketDataEvent::L2Delta(delta))
                .unwrap();
        }
        let snapshot = types::L2Snapshot {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 10,
            timestamp_ms: 1_700_000_000_000,
            bids: vec![BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        market_tx
            .try_send(types::MarketDataEvent::L2Snapshot(snapshot))
            .unwrap();
        let delta = types::L2Delta {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 11,
            timestamp_ms: 1_700_000_000_011,
            changes: vec![BookLevelDelta {
                side: BookSide::Bid,
                price: 99.0,
                size: 2.0,
            }],
        };
        market_tx
            .try_send(types::MarketDataEvent::L2Delta(delta))
            .unwrap();
        drop(market_tx);

        let out = drain_ordered_events(
            &mut market_rx,
            &mut account_rx,
            &mut exec_rx,
            &mut order_snapshot_rx,
            None,
            true,
            true,
            coalesce_ready_mask,
            &mut saw_l2_snapshot_mask_this_tick,
        );

        let mut snapshot_seq = None;
        let mut delta_seqs = Vec::new();
        for event in out {
            if let CanonicalEvent::Market(market) = event.event {
                match market {
                    types::MarketDataEvent::L2Snapshot(s) => {
                        snapshot_seq = Some(s.seq);
                    }
                    types::MarketDataEvent::L2Delta(d) => {
                        delta_seqs.push(d.seq);
                    }
                    _ => {}
                }
            }
        }
        delta_seqs.sort_unstable();
        assert_eq!(snapshot_seq, Some(11));
        assert!(delta_seqs.is_empty());
    }
}
