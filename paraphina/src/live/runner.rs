//! Live trading loop runner (feature-gated).

use std::collections::{BTreeMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, oneshot};

use super::ops::{
    append_account_reconcile_audit, append_reconcile_drift_audit, default_audit_dir, HealthState,
    LiveMetrics,
};
use super::venue_health::{VenueHealthErrorSource, VenueHealthManager};
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
use crate::mm::{
    compute_mm_quotes_with_ablations, compute_mm_quotes_with_now, compute_venue_utility_decision,
    quote_spread_gate_reason, venue_utility_conversion_penalties_enabled,
};
use crate::order_management::{plan_mm_order_actions, MmOrderDecisionSummary};
use crate::sim_eval::AblationSet;
use crate::state::{FundingState, GlobalState, MmOpenOrder, OpenOrderRecord, VenueState};
use crate::telemetry::{
    ensure_schema_v1, AccountPositionSyncRecord, ReconcileDriftRecord, TelemetryBuilder,
    TelemetryInputs, TelemetrySink, TelemetrySinkHandle, VenueHealthDiagnostics,
};
#[cfg(feature = "event_log")]
use crate::telemetry::{TelemetryConfig, TelemetryMode};
use crate::types::{
    ExecutionEvent, FundingSource, FundingStatus, OrderAck, OrderIntent, OrderPurpose, OrderReject,
    SettlementPriceKind, Side, TimeInForce, TimestampMs, VenueStatus,
};

use super::orderbook_l2::OrderBookL2;
use super::state_cache::{CanonicalCacheSnapshot, LiveStateCache};
use super::types::ExecutionEvent as LiveExecutionEvent;
use serde::Serialize;
use serde_json::json;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock};

/// How the order handler should respond to a request.
#[derive(Debug)]
pub enum ResponseMode {
    /// Reply via oneshot — caller blocks until response arrives (safety-critical paths).
    Oneshot(oneshot::Sender<Vec<LiveExecutionEvent>>),
    /// Fire-and-forget — results flow back via exec_tx on next tick (non-critical paths).
    FireAndForget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrderWaitOutcomeKind {
    Events,
    ChannelFull,
    HandlerDropped,
    Timeout,
}

#[derive(Debug)]
pub struct LiveOrderRequest {
    pub intents: Vec<OrderIntent>,
    pub action_batch: ActionBatch,
    pub now_ms: TimestampMs,
    pub response: ResponseMode,
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
    pub priority_order_tx: mpsc::Sender<LiveOrderRequest>,
    pub order_tx: mpsc::Sender<LiveOrderRequest>,
    pub order_snapshot_rx: Option<mpsc::Receiver<super::types::OrderSnapshot>>,
    /// Shared venue ages for cross-task health monitoring (Layer A + B).
    /// If `None`, age broadcasting is disabled (e.g. in tests / fixtures).
    pub shared_venue_ages: Option<super::shared_venue_ages::SharedVenueAges>,
}

#[derive(Clone)]
pub struct LiveRuntimeHooks {
    pub metrics: LiveMetrics,
    pub health: HealthState,
    pub telemetry: Option<LiveTelemetry>,
}

#[derive(Clone)]
pub struct LiveTelemetry {
    pub sink: TelemetrySinkHandle,
    pub shadow_mode: bool,
    pub execution_mode: &'static str,
    pub max_orders_per_tick: usize,
    pub stats: Arc<LiveTelemetryStats>,
}

/// Lock-free atomic counters for per-tick telemetry stats on the hot path.
/// The simple counters use `AtomicU64` so the tick thread never contends
/// with summary readers.  The purpose-tracking `HashMap`s live behind a
/// separate `Mutex` that is only acquired when intents are non-empty.
#[derive(Debug)]
pub struct LiveTelemetryStats {
    pub ticks_total: AtomicU64,
    pub fv_available_ticks: AtomicU64,
    pub venue_staleness_events: AtomicU64,
    pub venue_disabled_events: AtomicU64,
    pub kill_events: AtomicU64,
    /// Purpose-tracking maps — behind a separate Mutex to keep atomics lock-free.
    pub purpose_maps: Mutex<LiveTelemetryPurposeMaps>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SoftInventoryGovernorLimits {
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
}

impl SoftInventoryGovernorLimits {
    fn configured(self) -> bool {
        self.max_position_tao.is_some()
            || self.max_gross_position_tao.is_some()
            || self.max_abs_venue_position_tao.is_some()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct InventoryBrakeFractions {
    net_fraction: Option<f64>,
    gross_fraction: Option<f64>,
    venue_fraction: Option<f64>,
}

impl InventoryBrakeFractions {
    fn configured(self) -> bool {
        self.net_fraction.is_some()
            || self.gross_fraction.is_some()
            || self.venue_fraction.is_some()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct InventoryBrakeLimits {
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
}

impl InventoryBrakeLimits {
    fn configured(self) -> bool {
        self.max_position_tao.is_some()
            || self.max_gross_position_tao.is_some()
            || self.max_abs_venue_position_tao.is_some()
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct InventorySoftGovernorVenueStatus {
    venue_index: usize,
    venue_id: String,
    position_tao: f64,
    blocked_bid: bool,
    blocked_ask: bool,
    bid_reasons: Vec<String>,
    ask_reasons: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct InventorySoftGovernorStatus {
    configured: bool,
    triggered: bool,
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
    q_global_tao: f64,
    q_gross_tao: f64,
    q_max_abs_venue_tao: f64,
    global_reasons: Vec<String>,
    blocked_venues: Vec<InventorySoftGovernorVenueStatus>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct InventoryBrakeStatus {
    configured: bool,
    triggered: bool,
    sent: bool,
    grace_active: bool,
    grace_applied: bool,
    grace_deadline_ms: Option<TimestampMs>,
    grace_ticks_remaining: u32,
    net_fraction: Option<f64>,
    gross_fraction: Option<f64>,
    venue_fraction: Option<f64>,
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
    q_global_tao: f64,
    q_gross_tao: f64,
    q_max_abs_venue_tao: f64,
    projected_q_global_tao: f64,
    projected_q_gross_tao: f64,
    projected_q_max_abs_venue_tao: f64,
    global_reasons: Vec<String>,
    blocked_venues: Vec<InventorySoftGovernorVenueStatus>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EmergencyRequestClass {
    DisabledCancelAll,
    InventoryBrake,
    SoftUnwind,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct EmergencyRequestLatch {
    class: EmergencyRequestClass,
    venue_index: usize,
    venue_id: String,
    active_since_ms: TimestampMs,
    retry_latch_ms: TimestampMs,
    expires_at_ms: TimestampMs,
    max_expires_at_ms: TimestampMs,
    baseline_abs_position_tao: f64,
    baseline_open_orders: usize,
    last_progress_ms: TimestampMs,
    last_observed_abs_position_tao: f64,
    last_observed_open_orders: usize,
    extension_count: u32,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct EmergencyRequestLatchSet {
    disabled_cancel_all: Vec<EmergencyRequestLatch>,
    inventory_brake: Vec<EmergencyRequestLatch>,
    soft_unwind: Vec<EmergencyRequestLatch>,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct CanaryLimitStatus {
    breached: bool,
    net_breached: bool,
    gross_breached: bool,
    venue_breached: bool,
    open_orders_breached: bool,
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
    max_open_orders: Option<usize>,
    q_global_tao: f64,
    q_gross_tao: f64,
    q_max_abs_venue_tao: f64,
    open_order_count: usize,
    net_excess_tao: f64,
    gross_excess_tao: f64,
    venue_excess_tao: f64,
    open_order_excess: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct InventoryBrakeGraceConfig {
    grace_ms: TimestampMs,
    grace_ticks: u32,
    excess_fraction: f64,
}

impl InventoryBrakeGraceConfig {
    fn enabled(self) -> bool {
        self.grace_ms > 0 && self.grace_ticks > 0 && self.excess_fraction > 0.0
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct InventoryAttributionVenue {
    venue_index: usize,
    venue_id: String,
    position_tao: f64,
    tracked_mm_bid_live: bool,
    tracked_mm_ask_live: bool,
    mm_quote_intent_count: u64,
    mm_ack_count: u64,
    mm_fill_count: u64,
    mm_fill_abs_tao: f64,
    fill_count: u64,
    fill_delta_tao: f64,
    account_sync_count: u64,
    account_sync_delta_tao: f64,
    ack_count: u64,
    reject_count: u64,
    mm_reject_count: u64,
    benign_reduce_only_reject_count: u64,
    intent_place_count: u64,
    intent_replace_count: u64,
    intent_cancel_count: u64,
    intent_cancel_all_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct StartupPnlBaselineVenueStatus {
    venue_index: usize,
    venue_id: String,
    position_tao: f64,
    avg_entry_price: f64,
    fair_value: f64,
    unrealised_pnl_usd: f64,
    position_breach: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct StartupPnlBaselineConfig {
    enabled: bool,
    pnl_abs_limit_usd: f64,
    position_tol_tao: f64,
    max_wait_ticks: u64,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
struct StartupPnlBaselineStatus {
    enabled: bool,
    resolved: bool,
    waiting_for_accounts: bool,
    passed: bool,
    triggered: bool,
    timed_out: bool,
    reason: Option<String>,
    fresh_account_count: usize,
    required_account_count: usize,
    waited_ticks: u64,
    max_wait_ticks: u64,
    pnl_abs_limit_usd: f64,
    position_tol_tao: f64,
    daily_realised_pnl: f64,
    daily_unrealised_pnl: f64,
    daily_pnl_total: f64,
    violating_venues: Vec<StartupPnlBaselineVenueStatus>,
    pending_venues: Vec<String>,
}

/// HashMap-based purpose tracking that requires a Mutex (only locked when intents are non-empty).
#[derive(Debug, Default, Clone)]
pub struct LiveTelemetryPurposeMaps {
    pub would_place_by_purpose: std::collections::HashMap<String, u64>,
    pub would_cancel_by_purpose: std::collections::HashMap<String, u64>,
    pub would_replace_by_purpose: std::collections::HashMap<String, u64>,
}

static LIVE_CLIENT_ORDER_SEED: OnceLock<u64> = OnceLock::new();

impl Default for LiveTelemetryStats {
    fn default() -> Self {
        Self {
            ticks_total: AtomicU64::new(0),
            fv_available_ticks: AtomicU64::new(0),
            venue_staleness_events: AtomicU64::new(0),
            venue_disabled_events: AtomicU64::new(0),
            kill_events: AtomicU64::new(0),
            purpose_maps: Mutex::new(LiveTelemetryPurposeMaps::default()),
        }
    }
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

fn default_account_poll_ms() -> i64 {
    std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(5_000)
}

fn parse_positive_i64_env_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Option<i64> {
    keys.into_iter().find_map(|key| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|v| *v > 0)
    })
}

fn venue_specific_account_poll_ms(venue_id: &str, default_ms: i64) -> i64 {
    let normalized = venue_id
        .trim()
        .to_ascii_uppercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let generic_prefixed = format!("PARAPHINA_{normalized}_ACCOUNT_POLL_MS");
    let generic_plain = format!("{normalized}_ACCOUNT_POLL_MS");
    let alias = if normalized == "HYPERLIQUID" {
        parse_positive_i64_env_keys([
            "PARAPHINA_HL_ACCOUNT_POLL_MS",
            "HL_ACCOUNT_POLL_MS",
            generic_prefixed.as_str(),
            generic_plain.as_str(),
        ])
    } else {
        parse_positive_i64_env_keys([generic_prefixed.as_str(), generic_plain.as_str()])
    };
    alias.unwrap_or(default_ms)
}

fn account_snapshot_max_age_ms_for_venue(cfg: &Config, venue_id: &str) -> i64 {
    venue_specific_account_poll_ms(venue_id, default_account_poll_ms())
        .saturating_mul(2)
        .max(cfg.main_loop_interval_ms.saturating_mul(2))
}

fn account_snapshot_available(
    cfg: &Config,
    snapshot: &super::types::AccountSnapshot,
    now_ms: TimestampMs,
) -> bool {
    if snapshot.timestamp_ms <= 0 {
        return false;
    }
    now_ms.saturating_sub(snapshot.timestamp_ms)
        <= account_snapshot_max_age_ms_for_venue(cfg, &snapshot.venue_id)
}

fn venue_state_initialized(initialized: &[bool], venue_index: usize) -> bool {
    initialized.get(venue_index).copied().unwrap_or(false)
}

fn mark_venue_state_initialized(initialized: &mut [bool], venue_index: usize) {
    if let Some(flag) = initialized.get_mut(venue_index) {
        *flag = true;
    }
}

fn account_cache_snapshot_fresh(
    snapshot: &super::state_cache::VenueAccountSnapshot,
    now_ms: TimestampMs,
    max_age_ms: i64,
) -> bool {
    let Some(timestamp_ms) = snapshot.timestamp_ms else {
        return false;
    };
    timestamp_ms > 0 && now_ms.saturating_sub(timestamp_ms) <= max_age_ms
}

fn mark_initialized_account_venues(
    cfg: &Config,
    snapshot: &CanonicalCacheSnapshot,
    initialized: &mut [bool],
    now_ms: TimestampMs,
) {
    for acct in &snapshot.account {
        let fresh = account_cache_snapshot_fresh(
            acct,
            now_ms,
            account_snapshot_max_age_ms_for_venue(cfg, acct.venue_id.as_ref()),
        );
        if fresh {
            mark_venue_state_initialized(initialized, acct.venue_index);
        }
    }
}

fn update_applied_account_position_baselines(
    cfg: &Config,
    snapshot: &CanonicalCacheSnapshot,
    baselines: &mut [Option<f64>],
    now_ms: TimestampMs,
) {
    for acct in &snapshot.account {
        if !account_cache_snapshot_fresh(
            acct,
            now_ms,
            account_snapshot_max_age_ms_for_venue(cfg, acct.venue_id.as_ref()),
        ) {
            continue;
        }
        if let Some(slot) = baselines.get_mut(acct.venue_index) {
            *slot = Some(acct.position_tao);
        }
    }
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
    venue_index: usize,
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

/// Send an order request via the bounded channel and wait for the response with a
/// deterministic timeout.  Replaces the non-deterministic `yield_now() × 1000` busy-poll
/// that previously existed at six call-sites.
///
/// Returns `Some(events)` on success, `None` on channel-full / handler-dropped / timeout.
async fn send_order_and_wait(
    priority_order_tx: &mpsc::Sender<LiveOrderRequest>,
    intents: Vec<OrderIntent>,
    action_batch: ActionBatch,
    now_ms: TimestampMs,
    timeout_ms: u64,
    label: &str,
    tick: u64,
) -> Option<Vec<super::types::ExecutionEvent>> {
    match send_order_and_wait_with_status(
        priority_order_tx,
        intents,
        action_batch,
        now_ms,
        timeout_ms,
        label,
        tick,
    )
    .await
    {
        (OrderWaitOutcomeKind::Events, Some(events)) => Some(events),
        _ => None,
    }
}

async fn send_order_and_wait_with_status(
    priority_order_tx: &mpsc::Sender<LiveOrderRequest>,
    intents: Vec<OrderIntent>,
    action_batch: ActionBatch,
    now_ms: TimestampMs,
    timeout_ms: u64,
    label: &str,
    tick: u64,
) -> (
    OrderWaitOutcomeKind,
    Option<Vec<super::types::ExecutionEvent>>,
) {
    let (response_tx, response_rx) = oneshot::channel();
    let request = LiveOrderRequest {
        intents,
        action_batch,
        now_ms,
        response: ResponseMode::Oneshot(response_tx),
    };
    if let Err(_) = priority_order_tx.try_send(request) {
        eprintln!(
            "[runner] tick={} {}: priority_order_tx channel full (capacity={}), intents dropped",
            tick,
            label,
            priority_order_tx.max_capacity(),
        );
        return (OrderWaitOutcomeKind::ChannelFull, None);
    }
    match tokio::time::timeout(Duration::from_millis(timeout_ms), response_rx).await {
        Ok(Ok(events)) => (OrderWaitOutcomeKind::Events, Some(events)),
        Ok(Err(_)) => {
            eprintln!(
                "[runner] tick={} {}: oneshot closed (handler dropped sender)",
                tick, label,
            );
            (OrderWaitOutcomeKind::HandlerDropped, None)
        }
        Err(_) => {
            eprintln!(
                "[runner] tick={} {}: timeout after {}ms waiting for response",
                tick, label, timeout_ms,
            );
            (OrderWaitOutcomeKind::Timeout, None)
        }
    }
}

fn send_order_fire_and_forget(
    order_tx: &mpsc::Sender<LiveOrderRequest>,
    intents: Vec<OrderIntent>,
    action_batch: ActionBatch,
    now_ms: TimestampMs,
    label: &str,
    tick: u64,
) -> bool {
    let request = LiveOrderRequest {
        intents,
        action_batch,
        now_ms,
        response: ResponseMode::FireAndForget,
    };
    if let Err(_) = order_tx.try_send(request) {
        eprintln!(
            "[runner] tick={} {}: order_tx channel full (capacity={}), intents dropped",
            tick,
            label,
            order_tx.max_capacity(),
        );
        return false;
    }
    true
}

/// Send an account reconciliation request and wait for the response with a deterministic
/// timeout.  Separate from [`send_order_and_wait`] because account reconciliation uses a
/// different request type ([`LiveAccountRequest`]) and channel.
async fn send_account_and_wait(
    account_tx: &mpsc::Sender<LiveAccountRequest>,
    venue_index: usize,
    now_ms: TimestampMs,
    timeout_ms: u64,
    tick: u64,
) -> Option<super::types::AccountSnapshot> {
    let (response_tx, response_rx) = oneshot::channel();
    let request = LiveAccountRequest {
        venue_index: Some(venue_index),
        now_ms,
        response: response_tx,
    };
    if let Err(_) = account_tx.try_send(request) {
        eprintln!(
            "[runner] tick={} account_reconcile(venue={}): account_tx channel full, request dropped",
            tick, venue_index,
        );
        return None;
    }
    match tokio::time::timeout(Duration::from_millis(timeout_ms), response_rx).await {
        Ok(Ok(snapshot)) => Some(snapshot),
        Ok(Err(_)) => {
            eprintln!(
                "[runner] tick={} account_reconcile(venue={}): oneshot closed (handler dropped sender)",
                tick, venue_index,
            );
            None
        }
        Err(_) => {
            eprintln!(
                "[runner] tick={} account_reconcile(venue={}): timeout after {}ms",
                tick, venue_index, timeout_ms,
            );
            None
        }
    }
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
    tick_delta_buffer_max: Option<usize>,
) -> Vec<OrderedEvent> {
    let mut out = Vec::with_capacity(64);
    let coalesce_deltas = l2_delta_coalesce || l2_snapshot_coalesce;
    let mut pending_deltas: Option<Vec<Vec<super::types::L2Delta>>> =
        coalesce_deltas.then(|| Vec::new());
    let mut last_snapshots: Option<Vec<Option<super::types::L2Snapshot>>> =
        l2_snapshot_coalesce.then(|| Vec::new());
    let venue_ready = |vi: usize, saw_mask: u64| -> bool {
        vi < 64 && ((coalesce_ready_mask | saw_mask) & (1u64 << vi)) != 0
    };
    let mut buffer_disabled_mask: u64 = 0;
    let buffer_disabled = |vi: usize, mask: u64| vi < 64 && (mask & (1u64 << vi)) != 0;
    let count_out_market = |stats: &mut Option<&mut MarketRxStats>,
                            event: &super::types::MarketDataEvent| {
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
                                        || (s.seq == prev.seq
                                            && s.timestamp_ms >= prev.timestamp_ms)
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
    fn l2_deltas_strictly_increasing_by_seq_ts(deltas: &[super::types::L2Delta]) -> bool {
        if deltas.len() <= 1 {
            return true;
        }
        let mut prev = &deltas[0];
        for cur in deltas.iter().skip(1) {
            if (prev.seq, prev.timestamp_ms) >= (cur.seq, cur.timestamp_ms) {
                return false;
            }
            prev = cur;
        }
        true
    }
    let emit_delta_list = |deltas: Vec<super::types::L2Delta>,
                           market_stats: &mut Option<&mut MarketRxStats>,
                           out: &mut Vec<OrderedEvent>| {
        if deltas.is_empty() {
            return;
        }
        let mut deltas = deltas;
        if !l2_deltas_strictly_increasing_by_seq_ts(&deltas) {
            deltas.sort_by(|a, b| (a.seq, a.timestamp_ms).cmp(&(b.seq, b.timestamp_ms)));
        }
        deltas.dedup_by(|a, b| {
            if a.seq == b.seq {
                a.changes.extend(std::mem::take(&mut b.changes));
                if b.timestamp_ms > a.timestamp_ms {
                    a.timestamp_ms = b.timestamp_ms;
                }
                true
            } else {
                false
            }
        });
        for delta in deltas {
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
                    if !l2_deltas_strictly_increasing_by_seq_ts(&deltas) {
                        deltas
                            .sort_by(|a, b| (a.seq, a.timestamp_ms).cmp(&(b.seq, b.timestamp_ms)));
                    }
                    deltas.dedup_by(|a, b| {
                        if a.seq == b.seq {
                            a.changes.extend(std::mem::take(&mut b.changes));
                            if b.timestamp_ms > a.timestamp_ms {
                                a.timestamp_ms = b.timestamp_ms;
                            }
                            true
                        } else {
                            false
                        }
                    });
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
                        let event =
                            super::types::MarketDataEvent::L2Snapshot(super::types::L2Snapshot {
                                venue_index: snapshot.venue_index,
                                venue_id: snapshot.venue_id.clone(),
                                seq: book.last_seq(),
                                timestamp_ms: snapshot.timestamp_ms.max(last_delta.timestamp_ms),
                                bids: book.bids().to_vec(),
                                asks: book.asks().to_vec(),
                            });
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
                venue_index: snapshot.venue_index,
                venue_id: snapshot.venue_id.clone(),
                source_seq: snapshot.seq,
                event_ts_ms: snapshot.timestamp_ms,
                type_order: 3,
                event: CanonicalEvent::OrderSnapshot(snapshot),
            });
        }
    }

    out.sort_by(|a, b| {
        (a.venue_index, a.source_seq, a.event_ts_ms, a.type_order).cmp(&(
            b.venue_index,
            b.source_seq,
            b.event_ts_ms,
            b.type_order,
        ))
    });

    out
}

fn ordered_event_for_market(event: super::types::MarketDataEvent) -> Option<OrderedEvent> {
    let (venue_index, venue_id, source_seq, event_ts_ms) = match &event {
        super::types::MarketDataEvent::L2Snapshot(s) => {
            (s.venue_index, s.venue_id.clone(), s.seq, s.timestamp_ms)
        }
        super::types::MarketDataEvent::L2Delta(d) => {
            (d.venue_index, d.venue_id.clone(), d.seq, d.timestamp_ms)
        }
        super::types::MarketDataEvent::Trade(t) => {
            (t.venue_index, t.venue_id.clone(), t.seq, t.timestamp_ms)
        }
        super::types::MarketDataEvent::FundingUpdate(f) => {
            (f.venue_index, f.venue_id.clone(), f.seq, f.timestamp_ms)
        }
    };
    Some(OrderedEvent {
        venue_index,
        venue_id,
        source_seq,
        event_ts_ms,
        type_order: 0,
        event: CanonicalEvent::Market(event),
    })
}

fn ordered_event_for_account(event: super::types::AccountEvent) -> Option<OrderedEvent> {
    let (venue_index, venue_id, source_seq, event_ts_ms) = match &event {
        super::types::AccountEvent::Snapshot(s) => {
            (s.venue_index, s.venue_id.clone(), s.seq, s.timestamp_ms)
        }
    };
    Some(OrderedEvent {
        venue_index,
        venue_id,
        source_seq,
        event_ts_ms,
        type_order: 1,
        event: CanonicalEvent::Account(event),
    })
}

fn ordered_event_for_execution(event: super::types::ExecutionEvent) -> Option<OrderedEvent> {
    let (venue_index, venue_id, source_seq, event_ts_ms) = match &event {
        super::types::ExecutionEvent::OrderAccepted(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::OrderRejected(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::Filled(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelAccepted(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelRejected(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelAllAccepted(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::CancelAllRejected(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
        super::types::ExecutionEvent::OrderSnapshot(e) => {
            (e.venue_index, e.venue_id.clone(), e.seq, e.timestamp_ms)
        }
    };
    Some(OrderedEvent {
        venue_index,
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

const RECONCILE_TOL_EPSILON: f64 = 1e-9;
const KILL_CANCEL_ALL_TIMEOUT_MS_MIN: u64 = 5_000;
const KILL_CANCEL_ALL_TIMEOUT_MS_PER_VENUE: u64 = 1_000;
const KILL_FLATTEN_TIMEOUT_MS: u64 = 2_000;

fn diff_exceeds(lhs: f64, rhs: f64, tol: f64) -> bool {
    let diff = (lhs - rhs).abs();
    let tol_eps = RECONCILE_TOL_EPSILON.max(tol.abs() * 1e-9);
    diff > tol && (diff - tol) > tol_eps
}

fn kill_cancel_all_timeout_ms(venue_count: usize) -> u64 {
    let per_venue = (venue_count as u64).saturating_mul(KILL_CANCEL_ALL_TIMEOUT_MS_PER_VENUE);
    per_venue.max(KILL_CANCEL_ALL_TIMEOUT_MS_MIN)
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
    const EXTENDED_IDX: usize = 0;

    let engine = Engine::new(cfg);
    let mut state = GlobalState::new(cfg);
    let mut cache = LiveStateCache::new(cfg);
    let mut health_manager = VenueHealthManager::new(cfg);
    let mut telemetry_builder = TelemetryBuilder::new(cfg);
    let mut applied_book_logged: Vec<bool> = vec![false; cfg.venues.len()];

    let shared_venue_ages = channels.shared_venue_ages;
    let mut market_rx = channels.market_rx;
    let mut account_rx = channels.account_rx;
    let mut exec_rx = channels.exec_rx;
    let mut order_snapshot_rx = channels.order_snapshot_rx;
    let account_reconcile_tx = channels.account_reconcile_tx;
    let priority_order_tx = channels.priority_order_tx;
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
    let mut last_account_reconcile_ms: Option<TimestampMs> = None;
    let mut last_account_snapshot_ms: Vec<Option<TimestampMs>> = vec![None; cfg.venues.len()];
    let mut account_state_initialized: Vec<bool> = vec![false; cfg.venues.len()];
    let mut applied_account_position_baselines: Vec<Option<f64>> = vec![None; cfg.venues.len()];
    let mut order_state_initialized: Vec<bool> = vec![false; cfg.venues.len()];
    let mut soft_unwind_cooldown_until_ms: Option<TimestampMs> = None;
    let mut soft_unwind_response_backoff_until_ms: Option<TimestampMs> = None;
    let mut emergency_request_latches = EmergencyRequestLatchSet::default();
    let mut inventory_brake_grace_until_ms: Option<TimestampMs> = None;
    let mut inventory_brake_grace_remaining_ticks: u32 = 0;
    let mut inventory_brake_grace_available = true;

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
    let trade_mode = std::env::var("PARAPHINA_TRADE_MODE")
        .ok()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let skip_reconcile_kill = matches!(trade_mode.as_str(), "paper" | "p");
    let order_snapshot_fill_inference_enabled = !skip_reconcile_kill;
    let canary_enabled = std::env::var("PARAPHINA_CANARY_MODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let canary_max_position_tao = std::env::var("PARAPHINA_CANARY_MAX_POSITION_TAO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    let canary_max_gross_position_tao = std::env::var("PARAPHINA_CANARY_MAX_GROSS_POSITION_TAO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    let canary_max_abs_venue_position_tao =
        std::env::var("PARAPHINA_CANARY_MAX_ABS_VENUE_POSITION_TAO")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
    let inventory_brake_fractions = if canary_enabled {
        inventory_brake_fractions_from_env()
    } else {
        InventoryBrakeFractions::default()
    };
    let inventory_brake_limits = if canary_enabled {
        inventory_brake_limits(
            canary_max_position_tao,
            canary_max_gross_position_tao,
            canary_max_abs_venue_position_tao,
            inventory_brake_fractions,
        )
    } else {
        InventoryBrakeLimits::default()
    };
    let soft_inventory_governor_limits = if canary_enabled {
        SoftInventoryGovernorLimits {
            max_position_tao: parse_optional_positive_f64_env(
                "PARAPHINA_CANARY_SOFT_MAX_POSITION_TAO",
            ),
            max_gross_position_tao: parse_optional_positive_f64_env(
                "PARAPHINA_CANARY_SOFT_MAX_GROSS_POSITION_TAO",
            ),
            max_abs_venue_position_tao: parse_optional_positive_f64_env(
                "PARAPHINA_CANARY_SOFT_MAX_ABS_VENUE_POSITION_TAO",
            ),
        }
    } else {
        SoftInventoryGovernorLimits::default()
    };
    let soft_unwind_cooldown_ms = if canary_enabled && soft_inventory_governor_limits.configured() {
        parse_optional_positive_i64_env("PARAPHINA_CANARY_SOFT_UNWIND_COOLDOWN_MS").unwrap_or(5_000)
    } else {
        0
    };
    let soft_unwind_timeout_backoff_ms =
        if canary_enabled && soft_inventory_governor_limits.configured() {
            parse_optional_positive_i64_env("PARAPHINA_CANARY_SOFT_UNWIND_TIMEOUT_BACKOFF_MS")
                .unwrap_or(1_500)
        } else {
            0
        };
    let lighter_emergency_latch_ms = if canary_enabled {
        parse_optional_positive_i64_env("PARAPHINA_CANARY_LIGHTER_EMERGENCY_LATCH_MS")
            .unwrap_or_else(|| {
                venue_specific_account_poll_ms("lighter", default_account_poll_ms())
                    .max(soft_unwind_timeout_backoff_ms)
            })
    } else {
        0
    };
    let lighter_emergency_max_latch_ms = if canary_enabled {
        parse_optional_positive_i64_env("PARAPHINA_CANARY_LIGHTER_EMERGENCY_MAX_LATCH_MS")
            .unwrap_or_else(|| lighter_emergency_latch_ms.saturating_mul(3))
            .max(lighter_emergency_latch_ms)
    } else {
        0
    };
    let inventory_brake_grace_config = if canary_enabled {
        InventoryBrakeGraceConfig {
            grace_ms: parse_optional_positive_i64_env("PARAPHINA_CANARY_BRAKE_GRACE_MS")
                .unwrap_or(1_500),
            grace_ticks: std::env::var("PARAPHINA_CANARY_BRAKE_GRACE_TICKS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(3),
            excess_fraction: parse_optional_fraction_env(
                "PARAPHINA_CANARY_BRAKE_GRACE_EXCESS_FRACTION",
            )
            .unwrap_or(0.10),
        }
    } else {
        InventoryBrakeGraceConfig::default()
    };
    let canary_max_open_orders = std::env::var("PARAPHINA_CANARY_MAX_OPEN_ORDERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let canary_stale_max_ticks = std::env::var("PARAPHINA_CANARY_STALE_MAX_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let startup_pnl_baseline_cfg =
        startup_pnl_baseline_config_from_env(canary_enabled && trade_mode == "live");
    let canary_enforce_post_only = std::env::var("PARAPHINA_CANARY_ENFORCE_POST_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let canary_enforce_reduce_only = std::env::var("PARAPHINA_CANARY_ENFORCE_REDUCE_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let market_rx_stats_enabled = std::env::var("PARAPHINA_MARKET_RX_STATS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let ws_audit_enabled = std::env::var("PARAPHINA_WS_AUDIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let ext_apply_any = std::env::var("PARAPHINA_EXTENDED_APPLY_AGE_ON_ANY_L2")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let market_rx_stats_every = std::env::var("PARAPHINA_MARKET_RX_STATS_EVERY_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1);
    let market_rx_stats_path = std::env::var_os("PARAPHINA_MARKET_RX_STATS_PATH");
    let l2_delta_coalesce = std::env::var("PARAPHINA_L2_DELTA_COALESCE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let l2_snapshot_coalesce = std::env::var("PARAPHINA_L2_SNAPSHOT_COALESCE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut canary_stale_ticks: u64 = 0;
    let pos_tol = std::env::var("PARAPHINA_RECONCILE_POS_TAO_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.01);
    let reconcile_pos_soft_kill_streak = std::env::var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(2);
    let reconcile_pos_hard_kill_mult = std::env::var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 1.0)
        .unwrap_or(5.0);
    let reconcile_pos_kill_exempt_venues: HashSet<String> =
        std::env::var("PARAPHINA_RECONCILE_POS_KILL_EXEMPT_VENUES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|item| item.trim().to_ascii_lowercase())
                    .filter(|item| !item.is_empty())
                    .collect()
            })
            .unwrap_or_default();
    let _bal_tol = std::env::var("PARAPHINA_RECONCILE_BALANCE_USD_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    let order_tol = std::env::var("PARAPHINA_RECONCILE_ORDER_COUNT_TOL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut pending_drift_events: Vec<ReconcileDriftRecord> = Vec::new();
    let mut position_drift_streaks: Vec<usize> = vec![0; cfg.venues.len()];
    let mut startup_pnl_baseline_resolved = !startup_pnl_baseline_cfg.enabled;
    let mut startup_pnl_baseline_wait_ticks: u64 = 0;
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
    // Conditional quoting: per-venue tracking of last quoted state.
    let conditional_quoting = std::env::var("PARAPHINA_CONDITIONAL_QUOTING")
        .map(|v| v != "0")
        .unwrap_or(true);
    let max_quote_age_ms: i64 = std::env::var("PARAPHINA_MAX_QUOTE_AGE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);
    let mut last_quoted_mid: Vec<Option<f64>> = vec![None; cfg.venues.len()];
    let mut last_quoted_book_seq: Vec<u64> = vec![0; cfg.venues.len()];
    let mut last_quote_ts_ms: Vec<TimestampMs> = vec![0; cfg.venues.len()];

    let mut pending_events: Vec<OrderedEvent> = Vec::new();
    let mut saw_ready_once = false;
    let mut coalesce_ready_mask: u64 = 0;
    let tick_delta_buffer_max: Option<usize> = std::env::var("PARAPHINA_L2_TICK_DELTA_BUFFER_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0);
    let mut ext_seen_total: u64 = 0;
    let mut ext_seen_snapshot: u64 = 0;
    let mut ext_seen_delta: u64 = 0;
    let mut ext_deferred_future_total: u64 = 0;
    let mut ext_cache_ok_total: u64 = 0;
    let mut ext_cache_err_total: u64 = 0;
    let mut ext_core_apply_called_total: u64 = 0;
    let mut ext_mid_apply_set_total: u64 = 0;
    let mut ext_cache_err_sample: Option<String> = None;
    let mut last_runner_apply_audit = Instant::now();

    let min_inter_tick = Duration::from_millis(
        std::env::var("PARAPHINA_MIN_INTER_TICK_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(100),
    );
    let mut last_tick_instant = Instant::now();

    loop {
        let now_ms = match mode {
            LiveRunMode::Realtime { max_ticks, .. } => {
                // Event-driven wakeup: fire on interval OR when new market data arrives,
                // whichever comes first. min_inter_tick prevents overload from high-frequency venues.
                if let Some(interval) = interval.as_mut() {
                    tokio::select! {
                        _ = interval.tick() => {
                            // Scheduled tick — always proceed
                        }
                        result = market_rx.recv() => {
                            match result {
                                Some(event) => {
                                    // Push consumed event into pending_events for uniform drain processing.
                                    if let Some(ordered) = ordered_event_for_market(event) {
                                        pending_events.push(ordered);
                                    }
                                    // Check minimum inter-tick interval to prevent overload.
                                    if last_tick_instant.elapsed() < min_inter_tick {
                                        // Too soon — wait for the interval to fire naturally.
                                        interval.tick().await;
                                    } else {
                                        // Early wakeup: reset interval to prevent back-to-back
                                        // ticks when the dropped interval.tick() future's deadline
                                        // has already elapsed.
                                        interval.reset();
                                    }
                                }
                                None => {
                                    // Channel closed — proceed with regular interval
                                    interval.tick().await;
                                }
                            }
                        }
                    }
                }
                last_tick_instant = Instant::now();
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
        let tick_start = Instant::now();
        let mut audit_account_unavailable_after_drain = false;

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
                    if let Some(snapshot) =
                        send_account_and_wait(tx, venue_index, now_ms, 500, tick).await
                    {
                        if snapshot.timestamp_ms > 0 {
                            if let Some(last) =
                                last_account_snapshot_ms.get_mut(snapshot.venue_index)
                            {
                                *last = Some(snapshot.timestamp_ms);
                            }
                        }
                        let (report, diff) = cache.reconcile_account_snapshot_with_diff(&snapshot);
                        if let Some(diff) = diff {
                            let _ = append_account_reconcile_audit(&audit_dir, now_ms, diff);
                        }
                        if !report.account_ok {
                            if let Some(hooks) = hooks.as_ref() {
                                hooks.metrics.inc_error();
                                hooks.metrics.inc_reconcile_mismatch();
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
                audit_account_unavailable_after_drain = true;
            }
        }

        let reconcile_elapsed_us = tick_start.elapsed().as_micros() as u64;

        let mut would_send_intents: Vec<OrderIntent> = Vec::new();
        let mut tick_exec_events: Vec<ExecutionEvent> = Vec::new();
        let mut tick_fills: Vec<crate::types::FillEvent> = Vec::new();
        let mut tick_account_position_syncs: Vec<AccountPositionSyncRecord> = Vec::new();
        let mut inventory_soft_governor = InventorySoftGovernorStatus::default();
        let mut inventory_brake = InventoryBrakeStatus::default();
        let mut startup_pnl_baseline = StartupPnlBaselineStatus {
            enabled: startup_pnl_baseline_cfg.enabled,
            resolved: startup_pnl_baseline_resolved,
            passed: startup_pnl_baseline_resolved,
            pnl_abs_limit_usd: startup_pnl_baseline_cfg.pnl_abs_limit_usd,
            position_tol_tao: startup_pnl_baseline_cfg.position_tol_tao,
            max_wait_ticks: startup_pnl_baseline_cfg.max_wait_ticks,
            waited_ticks: startup_pnl_baseline_wait_ticks,
            ..StartupPnlBaselineStatus::default()
        };
        let mut last_exit_intent: Option<OrderIntent> = None;
        let mut last_hedge_intent: Option<OrderIntent> = None;

        let mut market_rx_stats = MarketRxStats::default();
        let maybe_print_market_rx_stats = |tick: u64, enabled: bool, stats: &MarketRxStats| {
            if !enabled {
                return;
            }
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
        };

        let delta_coalesce_now = l2_delta_coalesce && saw_ready_once;
        let snapshot_coalesce_now = l2_snapshot_coalesce && saw_ready_once;
        let mut saw_l2_snapshot_mask_this_tick: u64 = 0;

        // Pre-scan pending_events for L2Snapshots consumed by the select! wakeup arm.
        // Without this, snapshots received via market_rx.recv() (outside drain_ordered_events)
        // would not update saw_l2_snapshot_mask_this_tick, causing the venue to remain
        // permanently UNREADY in the coalescing logic and all its deltas to be silently dropped.
        for ev in &pending_events {
            if let CanonicalEvent::Market(super::types::MarketDataEvent::L2Snapshot(ref s)) =
                ev.event
            {
                if s.venue_index < 64 {
                    saw_l2_snapshot_mask_this_tick |= 1u64 << s.venue_index;
                }
            }
        }

        // Drain ingress channels, canonicalize ordering, then apply.
        pending_events.extend(drain_ordered_events(
            &mut market_rx,
            &mut account_rx,
            &mut exec_rx,
            &mut order_snapshot_rx,
            Some(&mut market_rx_stats),
            delta_coalesce_now,
            snapshot_coalesce_now,
            coalesce_ready_mask,
            &mut saw_l2_snapshot_mask_this_tick,
            tick_delta_buffer_max,
        ));
        let mut ordered_events = Vec::new();
        let mut future_events = Vec::new();
        for event in pending_events.drain(..) {
            if event.event_ts_ms <= now_ms {
                ordered_events.push(event);
            } else {
                if event.venue_index == EXTENDED_IDX {
                    ext_deferred_future_total = ext_deferred_future_total.saturating_add(1);
                }
                future_events.push(event);
            }
        }
        pending_events = future_events;
        ordered_events.sort_by(|a, b| {
            (a.venue_index, a.source_seq, a.event_ts_ms, a.type_order).cmp(&(
                b.venue_index,
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
                    let is_extended_event = match &event {
                        super::types::MarketDataEvent::L2Snapshot(s) => {
                            if s.venue_index == EXTENDED_IDX {
                                ext_seen_total = ext_seen_total.saturating_add(1);
                                ext_seen_snapshot = ext_seen_snapshot.saturating_add(1);
                                true
                            } else {
                                false
                            }
                        }
                        super::types::MarketDataEvent::L2Delta(d) => {
                            if d.venue_index == EXTENDED_IDX {
                                ext_seen_total = ext_seen_total.saturating_add(1);
                                ext_seen_delta = ext_seen_delta.saturating_add(1);
                                true
                            } else {
                                false
                            }
                        }
                        super::types::MarketDataEvent::Trade(t) => {
                            if t.venue_index == EXTENDED_IDX {
                                ext_seen_total = ext_seen_total.saturating_add(1);
                                true
                            } else {
                                false
                            }
                        }
                        super::types::MarketDataEvent::FundingUpdate(f) => {
                            if f.venue_index == EXTENDED_IDX {
                                ext_seen_total = ext_seen_total.saturating_add(1);
                                true
                            } else {
                                false
                            }
                        }
                    };
                    if let Err(e) = cache.apply_market_event(&event) {
                        let vi = match &event {
                            super::types::MarketDataEvent::L2Snapshot(s) => s.venue_index,
                            super::types::MarketDataEvent::L2Delta(d) => d.venue_index,
                            super::types::MarketDataEvent::Trade(t) => t.venue_index,
                            super::types::MarketDataEvent::FundingUpdate(f) => f.venue_index,
                        };
                        if vi == EXTENDED_IDX {
                            ext_cache_err_total = ext_cache_err_total.saturating_add(1);
                            if ext_cache_err_sample.is_none() {
                                let sample = format!("{e:?}").replace(' ', "_");
                                let sample: String = sample.chars().take(80).collect();
                                ext_cache_err_sample = Some(sample);
                            }
                        }
                        health_manager.record_api_error(
                            vi,
                            VenueHealthErrorSource::MarketCache,
                            &format!("{e:?}"),
                        );
                    } else {
                        if is_extended_event {
                            ext_cache_ok_total = ext_cache_ok_total.saturating_add(1);
                            ext_core_apply_called_total =
                                ext_core_apply_called_total.saturating_add(1);
                        }
                        let prev_ext_mid_apply_ms = if is_extended_event {
                            state
                                .venues
                                .get(EXTENDED_IDX)
                                .and_then(|v| v.last_mid_apply_ms)
                        } else {
                            None
                        };
                        apply_market_event_to_core(&mut state, cfg, &event, now_ms, ext_apply_any);
                        if is_extended_event {
                            let new_ext_mid_apply_ms = state
                                .venues
                                .get(EXTENDED_IDX)
                                .and_then(|v| v.last_mid_apply_ms);
                            if new_ext_mid_apply_ms == Some(now_ms)
                                && prev_ext_mid_apply_ms != Some(now_ms)
                            {
                                ext_mid_apply_set_total = ext_mid_apply_set_total.saturating_add(1);
                            }
                        }
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
                    if let Err(e) = cache.apply_account_event(&event) {
                        health_manager.record_api_error(
                            match &event {
                                super::types::AccountEvent::Snapshot(s) => s.venue_index,
                            },
                            VenueHealthErrorSource::AccountCache,
                            &format!("{e:?}"),
                        );
                    }
                    let snapshot = match &event {
                        super::types::AccountEvent::Snapshot(snapshot) => snapshot,
                    };
                    if snapshot.timestamp_ms > 0 {
                        if let Some(last) = last_account_snapshot_ms.get_mut(snapshot.venue_index) {
                            *last = Some(snapshot.timestamp_ms);
                        }
                    }
                    if !account_snapshot_available(cfg, snapshot, now_ms) {
                        continue;
                    }
                    if !venue_state_initialized(&account_state_initialized, snapshot.venue_index) {
                        continue;
                    }
                    if let Some(vstate) = state.venues.get(snapshot.venue_index) {
                        let pos_internal = vstate.position_tao;
                        let pos_venue = derive_position_tao(&snapshot.positions);
                        let pos_diff = pos_internal - pos_venue;
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
                                    diff: Some(pos_diff),
                                    tolerance: Some(pos_tol),
                                    source: "account_snapshot".to_string(),
                                    available: true,
                                },
                            );
                            let venue_position_changed_since_apply =
                                !applied_account_position_baselines
                                    .get(snapshot.venue_index)
                                    .copied()
                                    .flatten()
                                    .is_some_and(|applied| {
                                        !diff_exceeds(applied, pos_venue, pos_tol)
                                    });
                            let pos_kill_exempt = reconcile_pos_kill_exempt_venues
                                .contains(&snapshot.venue_id.to_ascii_lowercase());
                            // A fresh account-side position change should be synchronized into
                            // state first. Reconciliation kills are reserved for stable,
                            // already-applied venue positions that state still fails to match.
                            if pos_kill_exempt || venue_position_changed_since_apply {
                                if let Some(streak) =
                                    position_drift_streaks.get_mut(snapshot.venue_index)
                                {
                                    *streak = 0;
                                }
                            } else {
                                let drift_streak = if let Some(streak) =
                                    position_drift_streaks.get_mut(snapshot.venue_index)
                                {
                                    *streak = streak.saturating_add(1);
                                    *streak
                                } else {
                                    reconcile_pos_soft_kill_streak
                                };
                                let hard_kill = pos_tol <= 0.0
                                    || pos_diff.abs() >= pos_tol * reconcile_pos_hard_kill_mult;
                                if !skip_reconcile_kill {
                                    if !state.kill_switch
                                        && (hard_kill
                                            || drift_streak >= reconcile_pos_soft_kill_streak)
                                    {
                                        state.kill_switch = true;
                                        state.kill_reason =
                                            crate::state::KillReason::ReconciliationDrift;
                                    }
                                }
                            }
                        } else if let Some(streak) =
                            position_drift_streaks.get_mut(snapshot.venue_index)
                        {
                            *streak = 0;
                        }
                        // Margin fields are sourced directly from venue account snapshots and
                        // applied back into `state` after the event drain. Comparing a fresh
                        // snapshot against the pre-apply state here creates false-positive
                        // reconciliation kills on legitimate venue-side balance updates.
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
                        if venue_state_initialized(&order_state_initialized, snapshot.venue_index) {
                            let internal_before = state
                                .live_order_state
                                .open_order_ids_by_venue(snapshot.venue_index);
                            let mut venue_orders = snapshot
                                .open_orders
                                .iter()
                                .map(|o| o.order_id.clone())
                                .collect::<Vec<_>>();
                            venue_orders.sort();
                            let diff_count = internal_before
                                .iter()
                                .filter(|id| !venue_orders.contains(*id))
                                .count()
                                + venue_orders
                                    .iter()
                                    .filter(|id| !internal_before.contains(*id))
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
                                        internal: Some(internal_before.len() as f64),
                                        venue: Some(venue_orders.len() as f64),
                                        diff: Some(
                                            internal_before.len() as f64
                                                - venue_orders.len() as f64,
                                        ),
                                        tolerance: Some(order_tol as f64),
                                        source: "order_snapshot".to_string(),
                                        available: true,
                                    },
                                );
                            }
                        }
                        // Order snapshots are the venue-side source of truth; reconcile them
                        // into local state instead of hard-killing on transient replace skew.
                        let (core_events, fills) = infer_fills_from_order_snapshot(
                            cfg,
                            &mut state,
                            &snapshot,
                            now_ms,
                            order_snapshot_fill_inference_enabled,
                        );
                        if !core_events.is_empty() {
                            tick_exec_events.extend(core_events.iter().cloned());
                        }
                        if !fills.is_empty() {
                            tick_fills.extend(fills.iter().cloned());
                            fill_batcher.push(now_ms, fills);
                        }
                        state.live_order_state.reconcile(&snapshot, now_ms);
                        sync_venue_order_tracking_from_live_order_state(
                            &mut state,
                            snapshot.venue_index,
                        );
                        mark_venue_state_initialized(
                            &mut order_state_initialized,
                            snapshot.venue_index,
                        );
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
                    let (core_events, fills) = infer_fills_from_order_snapshot(
                        cfg,
                        &mut state,
                        &snapshot,
                        now_ms,
                        order_snapshot_fill_inference_enabled,
                    );
                    if !core_events.is_empty() {
                        tick_exec_events.extend(core_events.iter().cloned());
                    }
                    if !fills.is_empty() {
                        tick_fills.extend(fills.iter().cloned());
                        fill_batcher.push(now_ms, fills);
                    }
                    state.live_order_state.reconcile(&snapshot, now_ms);
                    sync_venue_order_tracking_from_live_order_state(
                        &mut state,
                        snapshot.venue_index,
                    );
                    mark_venue_state_initialized(
                        &mut order_state_initialized,
                        snapshot.venue_index,
                    );
                }
            }
        }
        if ws_audit_enabled && last_runner_apply_audit.elapsed() >= Duration::from_millis(1000) {
            let (age_apply_ms, age_event_ms) = if let Some(venue) = state.venues.get(EXTENDED_IDX) {
                let apply = venue
                    .last_mid_apply_ms
                    .map(|ts| now_ms.saturating_sub(ts))
                    .unwrap_or(-1);
                let event = venue
                    .last_mid_update_ms
                    .map(|ts| now_ms.saturating_sub(ts))
                    .unwrap_or(-1);
                (apply, event)
            } else {
                (-1, -1)
            };
            let err_sample = ext_cache_err_sample.as_deref().unwrap_or("none");
            eprintln!(
                concat!(
                    "WS_AUDIT venue=extended component=runner_apply reason=periodic interval_ms=1000 ",
                    "tick={} now_ms={} ext_seen={} ext_seen_snapshot={} ext_seen_delta={} ",
                    "ext_future={} cache_ok={} cache_err={} core_apply={} mid_apply_set={} ",
                    "age_apply_ms={} age_event_ms={} err_sample={}",
                ),
                tick,
                now_ms,
                ext_seen_total,
                ext_seen_snapshot,
                ext_seen_delta,
                ext_deferred_future_total,
                ext_cache_ok_total,
                ext_cache_err_total,
                ext_core_apply_called_total,
                ext_mid_apply_set_total,
                age_apply_ms,
                age_event_ms,
                err_sample
            );
            last_runner_apply_audit = Instant::now();
        }

        let event_drain_elapsed_us = tick_start.elapsed().as_micros() as u64;

        if audit_account_unavailable_after_drain {
            let has_fresh_snapshot = last_account_snapshot_ms
                .iter()
                .enumerate()
                .filter_map(|(venue_index, ts)| ts.map(|ts| (venue_index, ts)))
                .any(|(venue_index, ts)| {
                    let venue_id = cfg
                        .venues
                        .get(venue_index)
                        .map(|venue| venue.id.as_str())
                        .unwrap_or("");
                    now_ms.saturating_sub(ts)
                        <= account_snapshot_max_age_ms_for_venue(cfg, venue_id)
                });
            if !has_fresh_snapshot {
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

        last_snapshot = Some(cache.snapshot_per_venue(now_ms, &cfg.venues, cfg.book.stale_ms));
        let snapshot = last_snapshot.as_ref().unwrap();
        if snapshot.ready_market_count() > 0 || saw_l2_snapshot_mask_this_tick != 0 {
            saw_ready_once = true;
        }
        coalesce_ready_mask |= saw_l2_snapshot_mask_this_tick;
        let update = health_manager.update_from_snapshot(cfg, &mut state, snapshot);
        let mut disabled = update.disabled;
        if disable_health_gates {
            for venue in &mut state.venues {
                venue.status = VenueStatus::Healthy;
            }
            disabled.clear();
        }
        let venue_health_diagnostics = health_manager.diagnostics();
        if !disable_health_gates {
            for transition in update.transitions {
                let venue_id = cfg
                    .venues
                    .get(transition.venue_index)
                    .map(|venue| venue.id.as_str())
                    .unwrap_or("unknown");
                let diag = venue_health_diagnostics
                    .get(transition.venue_index)
                    .cloned()
                    .unwrap_or_default();
                eprintln!(
                    "VENUE_HEALTH_TRANSITION venue={} venue_index={} from={:?} to={:?} api_errors={} stale_count={} dev_breaches={} disable_reason={} last_error_source={} last_error_message={}",
                    venue_id,
                    transition.venue_index,
                    transition.from,
                    transition.to,
                    diag.api_errors,
                    diag.stale_count,
                    diag.dev_breaches,
                    diag.disable_reason,
                    diag.last_error_source,
                    diag.last_error_message,
                );
            }
        }
        // Update SharedVenueAges for Layer A (enforcer) and Layer B (REST monitor)
        // using local apply-age semantics.
        if let Some(ref ages) = shared_venue_ages {
            for (idx, venue) in state.venues.iter().enumerate() {
                let age = match venue.last_mid_apply_ms {
                    None => i64::MAX,
                    Some(ts) => (now_ms - ts).max(0),
                };
                ages.set_age(idx, age);
            }
            ages.mark_write(now_ms);
        }
        let stale_count = snapshot.market.iter().filter(|m| m.is_stale).count() as u64;
        refresh_emergency_request_latches(&mut emergency_request_latches, cfg, &state, now_ms);
        if !disabled.is_empty() {
            // Batch all disabled-venue cancel-all intents into a single channel send.
            let mut cancel_intents: Vec<OrderIntent> = Vec::with_capacity(disabled.len());
            for venue_index in &disabled {
                if emergency_request_latched(
                    &mut emergency_request_latches,
                    EmergencyRequestClass::DisabledCancelAll,
                    cfg,
                    &state,
                    now_ms,
                    *venue_index,
                ) {
                    continue;
                }
                let intent =
                    crate::types::OrderIntent::CancelAll(crate::types::CancelAllOrderIntent {
                        venue_index: Some(*venue_index),
                        venue_id: Some(cfg.venues[*venue_index].id_arc.clone()),
                    });
                would_send_intents.push(intent.clone());
                cancel_intents.push(intent);
            }
            if !cancel_intents.is_empty() {
                if let Some(hooks_ref) = hooks.as_ref() {
                    hooks_ref.metrics.inc_cancel_all();
                }
                let tick = now_ms.max(0) as u64;
                let mut sync_cancel_intents = cancel_intents;
                let async_cancel_requests =
                    take_emergency_single_flight_intents(cfg, &mut sync_cancel_intents);
                let _ = send_emergency_single_flight_requests(
                    cfg,
                    &priority_order_tx,
                    &mut emergency_request_latches,
                    &state,
                    async_cancel_requests,
                    EmergencyRequestClass::DisabledCancelAll,
                    now_ms,
                    tick,
                    lighter_emergency_latch_ms,
                    lighter_emergency_max_latch_ms,
                    "disabled_cancel_all",
                );
                let sync_cancel_venues = venue_indices_for_order_intents(&sync_cancel_intents);
                let (cancel_outcome, cancel_events) = if !sync_cancel_intents.is_empty() {
                    let action_batch = build_live_action_batch(
                        cfg,
                        &sync_cancel_intents,
                        now_ms,
                        tick.saturating_mul(2),
                    );
                    send_order_and_wait_with_status(
                        &priority_order_tx,
                        sync_cancel_intents,
                        action_batch,
                        now_ms,
                        1000,
                        "disabled_cancel_all",
                        tick,
                    )
                    .await
                } else {
                    (OrderWaitOutcomeKind::Events, None)
                };
                if matches!(
                    cancel_outcome,
                    OrderWaitOutcomeKind::Timeout | OrderWaitOutcomeKind::HandlerDropped
                ) {
                    for venue_index in sync_cancel_venues {
                        latch_emergency_request(
                            &mut emergency_request_latches,
                            EmergencyRequestClass::DisabledCancelAll,
                            cfg,
                            &state,
                            venue_index,
                            now_ms,
                            lighter_emergency_latch_ms,
                            lighter_emergency_max_latch_ms,
                        );
                    }
                }
                if let Some(events) = cancel_events {
                    #[cfg(feature = "event_log")]
                    log_live_execution_events_env(now_ms.max(0) as u64, now_ms, "gateway", &events);
                    let core_events = live_events_to_core(&events);
                    let fills = apply_execution_events(&mut state, &core_events, now_ms);
                    if !fills.is_empty() {
                        apply_live_fills(cfg, &mut state, &fills, now_ms);
                        state.recompute_after_fills(cfg);
                    }
                }
            }
        }
        if let Some(hooks) = hooks.as_ref() {
            let ready_count = snapshot.ready_market_count();
            let ready = ready_count == cfg.venues.len();
            hooks.health.set_ready(ready);
        }
        let cache_events = snapshot_to_core_events(snapshot, &state);
        let _ = apply_execution_events(&mut state, &cache_events, now_ms);
        let account_apply = apply_account_snapshot_to_state(cfg, snapshot, &mut state, now_ms);
        let account_position_changed = account_apply.position_changed;
        tick_account_position_syncs.extend(account_apply.position_syncs);
        let (account_fill_events, account_fills) = infer_fills_from_account_position_syncs(
            cfg,
            &mut state,
            &tick_account_position_syncs,
            now_ms,
        );
        if !account_fill_events.is_empty() {
            tick_exec_events.extend(account_fill_events);
        }
        if !account_fills.is_empty() {
            tick_fills.extend(account_fills);
        }
        mark_initialized_account_venues(cfg, snapshot, &mut account_state_initialized, now_ms);
        update_applied_account_position_baselines(
            cfg,
            snapshot,
            &mut applied_account_position_baselines,
            now_ms,
        );
        refresh_emergency_request_latches(&mut emergency_request_latches, cfg, &state, now_ms);

        if canary_enabled && !state.kill_switch {
            inventory_brake = evaluate_inventory_brake(
                cfg,
                &state,
                inventory_brake_fractions,
                inventory_brake_limits,
            );
            if inventory_brake.triggered {
                soft_unwind_cooldown_until_ms = None;
                let mut brake_intents =
                    build_inventory_brake_intents(cfg, &state, snapshot, now_ms, &inventory_brake);
                let removed_brake_venues = remove_latched_intents_for_class(
                    &mut brake_intents,
                    &mut emergency_request_latches,
                    EmergencyRequestClass::InventoryBrake,
                    cfg,
                    &state,
                    now_ms,
                );
                if !removed_brake_venues.is_empty() {
                    inventory_brake.sent = true;
                }
                apply_canary_intent_overrides(
                    &mut brake_intents,
                    canary_enforce_post_only,
                    canary_enforce_reduce_only,
                );
                if !brake_intents.is_empty() {
                    inventory_brake.sent = true;
                    last_hedge_intent = brake_intents
                        .iter()
                        .find(|intent| {
                            matches!(
                                intent,
                                OrderIntent::Place(place)
                                    if place.purpose == OrderPurpose::Hedge
                            )
                        })
                        .cloned();
                    normalize_live_client_order_ids(&mut brake_intents, tick);
                    would_send_intents.extend(brake_intents.iter().cloned());
                    if let Some(hooks) = hooks.as_ref() {
                        hooks.metrics.inc_orders(brake_intents.len());
                    }
                    let mut sync_brake_intents = brake_intents;
                    let async_brake_requests =
                        take_emergency_single_flight_intents(cfg, &mut sync_brake_intents);
                    let _ = send_emergency_single_flight_requests(
                        cfg,
                        &priority_order_tx,
                        &mut emergency_request_latches,
                        &state,
                        async_brake_requests,
                        EmergencyRequestClass::InventoryBrake,
                        now_ms,
                        tick,
                        lighter_emergency_latch_ms,
                        lighter_emergency_max_latch_ms,
                        "inventory_brake",
                    );
                    let sync_brake_venues = venue_indices_for_order_intents(&sync_brake_intents);
                    let (brake_outcome, brake_events) = if !sync_brake_intents.is_empty() {
                        let action_batch = build_live_action_batch(
                            cfg,
                            &sync_brake_intents,
                            now_ms,
                            tick.saturating_mul(2),
                        );
                        send_order_and_wait_with_status(
                            &priority_order_tx,
                            sync_brake_intents,
                            action_batch,
                            now_ms,
                            1_000,
                            "inventory_brake",
                            tick,
                        )
                        .await
                    } else {
                        (OrderWaitOutcomeKind::Events, None)
                    };
                    if matches!(
                        brake_outcome,
                        OrderWaitOutcomeKind::Timeout | OrderWaitOutcomeKind::HandlerDropped
                    ) {
                        for venue_index in sync_brake_venues {
                            latch_emergency_request(
                                &mut emergency_request_latches,
                                EmergencyRequestClass::InventoryBrake,
                                cfg,
                                &state,
                                venue_index,
                                now_ms,
                                lighter_emergency_latch_ms,
                                lighter_emergency_max_latch_ms,
                            );
                        }
                    }
                    if let Some(events) = brake_events {
                        let brake_fills = apply_priority_response_events(
                            cfg,
                            &mut state,
                            &mut deduper,
                            events,
                            now_ms,
                            order_snapshot_fill_inference_enabled,
                            &mut tick_exec_events,
                            &mut tick_fills,
                            &mut order_state_initialized,
                        );
                        if !brake_fills.is_empty() {
                            apply_live_fills(cfg, &mut state, &brake_fills, now_ms);
                            state.recompute_after_fills(cfg);
                        }
                    }
                }
            }
        }

        if canary_enabled && !state.kill_switch {
            let canary_limits = canary_limit_status(
                &state,
                canary_max_position_tao,
                canary_max_gross_position_tao,
                canary_max_abs_venue_position_tao,
                canary_max_open_orders,
            );
            if canary_limits.breached {
                let grace_eligible = inventory_brake_grace_allowed(
                    &canary_limits,
                    &inventory_brake,
                    inventory_brake_grace_config,
                );
                let mut grace_active = inventory_brake_grace_is_active(
                    inventory_brake_grace_until_ms,
                    inventory_brake_grace_remaining_ticks,
                    now_ms,
                );
                if !grace_active && inventory_brake_grace_available && grace_eligible {
                    inventory_brake_grace_until_ms =
                        Some(now_ms.saturating_add(inventory_brake_grace_config.grace_ms));
                    inventory_brake_grace_remaining_ticks =
                        inventory_brake_grace_config.grace_ticks;
                    inventory_brake_grace_available = false;
                    grace_active = inventory_brake_grace_is_active(
                        inventory_brake_grace_until_ms,
                        inventory_brake_grace_remaining_ticks,
                        now_ms,
                    );
                }

                if grace_active && grace_eligible {
                    inventory_brake.grace_applied = true;
                    inventory_brake_grace_remaining_ticks =
                        inventory_brake_grace_remaining_ticks.saturating_sub(1);
                } else {
                    inventory_brake_grace_until_ms = None;
                    inventory_brake_grace_remaining_ticks = 0;
                    state.kill_switch = true;
                    state.kill_reason = crate::state::KillReason::CanaryLimitBreach;
                }
            } else {
                inventory_brake_grace_until_ms = None;
                inventory_brake_grace_remaining_ticks = 0;
                inventory_brake_grace_available = true;
            }
            apply_inventory_brake_grace_telemetry(
                &mut inventory_brake,
                inventory_brake_grace_until_ms,
                inventory_brake_grace_remaining_ticks,
                now_ms,
            );
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
        if startup_pnl_baseline_cfg.enabled && !startup_pnl_baseline_resolved {
            startup_pnl_baseline = evaluate_startup_pnl_baseline(
                cfg,
                &state,
                snapshot,
                &account_state_initialized,
                now_ms,
                startup_pnl_baseline_wait_ticks,
                startup_pnl_baseline_cfg,
            );
            if startup_pnl_baseline.triggered {
                state.kill_switch = true;
                state.kill_reason = crate::state::KillReason::StartupPnlBaselineBreach;
                state.risk_regime = crate::state::RiskRegime::HardLimit;
                eprintln!(
                    "paraphina_live | startup_pnl_baseline_breach reason={} pnl_total={:.4} realised={:.4} unrealised={:.4} limit_usd={:.4} waited_ticks={} fresh_accounts={}/{} pending={:?} violating={:?}",
                    startup_pnl_baseline
                        .reason
                        .as_deref()
                        .unwrap_or("unknown"),
                    startup_pnl_baseline.daily_pnl_total,
                    startup_pnl_baseline.daily_realised_pnl,
                    startup_pnl_baseline.daily_unrealised_pnl,
                    startup_pnl_baseline.pnl_abs_limit_usd,
                    startup_pnl_baseline.waited_ticks,
                    startup_pnl_baseline.fresh_account_count,
                    startup_pnl_baseline.required_account_count,
                    startup_pnl_baseline.pending_venues,
                    startup_pnl_baseline.violating_venues,
                );
            } else if startup_pnl_baseline.waiting_for_accounts {
                startup_pnl_baseline_wait_ticks = startup_pnl_baseline_wait_ticks.saturating_add(1);
            } else if startup_pnl_baseline.passed {
                startup_pnl_baseline_resolved = true;
                eprintln!(
                    "paraphina_live | startup_pnl_baseline_ok pnl_total={:.4} realised={:.4} unrealised={:.4} fresh_accounts={}/{}",
                    startup_pnl_baseline.daily_pnl_total,
                    startup_pnl_baseline.daily_realised_pnl,
                    startup_pnl_baseline.daily_unrealised_pnl,
                    startup_pnl_baseline.fresh_account_count,
                    startup_pnl_baseline.required_account_count,
                );
            }
        }
        if scheduler.risk_due(now_ms) {
            if startup_pnl_baseline_cfg.enabled
                && !startup_pnl_baseline_resolved
                && startup_pnl_baseline.waiting_for_accounts
            {
                scheduler.mark_risk_ran();
            } else {
                engine.update_risk_limits_and_regime(&mut state);
                scheduler.mark_risk_ran();
            }
        }
        let engine_elapsed_us = tick_start.elapsed().as_micros() as u64;

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
                    &priority_order_tx,
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

        // Build tick timing snapshot for telemetry (total_us updated at tick end).
        let order_tx_pending = (priority_order_tx.max_capacity() - priority_order_tx.capacity())
            + (order_tx.max_capacity() - order_tx.capacity());
        let mut tick_timing = TickTiming {
            reconcile_us: reconcile_elapsed_us,
            event_drain_us: event_drain_elapsed_us,
            engine_us: engine_elapsed_us,
            submit_us: 0,
            total_us: tick_start.elapsed().as_micros() as u64,
            order_tx_pending,
        };
        let mut mm_order_management = MmOrderDecisionSummary::default();
        // Only emit tick timing in Realtime mode; Step mode uses non-deterministic
        // wall-clock values that break replay hash comparison.
        let emit_tick_timing = matches!(mode, LiveRunMode::Realtime { .. });

        pending_drift_events.sort_by(|a, b| {
            (a.venue_index, &a.kind, &a.source).cmp(&(b.venue_index, &b.kind, &b.source))
        });

        // Early exit: kill switch triggered before submission phase.
        if state.kill_switch {
            tick_timing.total_us = tick_start.elapsed().as_micros() as u64;
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
                        last_exit_intent.as_ref(),
                        last_hedge_intent.as_ref(),
                        &pending_drift_events,
                        &tick_account_position_syncs,
                        &inventory_soft_governor,
                        &inventory_brake,
                        &startup_pnl_baseline,
                        &emergency_request_latches,
                        &mm_order_management,
                        market_rx_stats_enabled.then_some(&market_rx_stats),
                        if emit_tick_timing {
                            Some(&tick_timing)
                        } else {
                            None
                        },
                        &venue_health_diagnostics,
                    );
                    pending_drift_events.clear();
                }
            }
            maybe_print_market_rx_stats(tick, market_rx_stats_enabled, &market_rx_stats);
            break;
        }

        if startup_pnl_baseline_cfg.enabled
            && !startup_pnl_baseline_resolved
            && startup_pnl_baseline.waiting_for_accounts
        {
            tick_timing.total_us = tick_start.elapsed().as_micros() as u64;
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
                        last_exit_intent.as_ref(),
                        last_hedge_intent.as_ref(),
                        &pending_drift_events,
                        &tick_account_position_syncs,
                        &inventory_soft_governor,
                        &inventory_brake,
                        &startup_pnl_baseline,
                        &emergency_request_latches,
                        &mm_order_management,
                        market_rx_stats_enabled.then_some(&market_rx_stats),
                        if emit_tick_timing {
                            Some(&tick_timing)
                        } else {
                            None
                        },
                        &venue_health_diagnostics,
                    );
                    pending_drift_events.clear();
                }
            }
            maybe_print_market_rx_stats(tick, market_rx_stats_enabled, &market_rx_stats);
            tick += 1;
            continue;
        }

        // Early exit: no ready markets — skip quoting/submission.
        if snapshot.ready_market_count() == 0 && !smoke_intents {
            tick_timing.total_us = tick_start.elapsed().as_micros() as u64;
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
                        last_exit_intent.as_ref(),
                        last_hedge_intent.as_ref(),
                        &pending_drift_events,
                        &tick_account_position_syncs,
                        &inventory_soft_governor,
                        &inventory_brake,
                        &startup_pnl_baseline,
                        &emergency_request_latches,
                        &mm_order_management,
                        market_rx_stats_enabled.then_some(&market_rx_stats),
                        if emit_tick_timing {
                            Some(&tick_timing)
                        } else {
                            None
                        },
                        &venue_health_diagnostics,
                    );
                    pending_drift_events.clear();
                }
            }
            maybe_print_market_rx_stats(tick, market_rx_stats_enabled, &market_rx_stats);
            tick += 1;
            continue;
        }

        // Conditional quoting: skip requote for venues with unchanged data.
        let mut should_quote = true;
        if conditional_quoting && !smoke_intents {
            let mut any_changed = false;
            for vm in &snapshot.market {
                let vi = vm.venue_index;
                if vm.is_stale {
                    continue;
                }
                let mid = vm.mid;
                let seq = vm.seq;
                let prev_mid = last_quoted_mid.get(vi).copied().flatten();
                let prev_seq = last_quoted_book_seq.get(vi).copied().unwrap_or(0);
                let prev_ts = last_quote_ts_ms.get(vi).copied().unwrap_or(0);
                // Force requote if max quote age exceeded.
                if now_ms.saturating_sub(prev_ts) > max_quote_age_ms {
                    any_changed = true;
                    break;
                }
                // Force requote if book sequence advanced (new data).
                if seq > prev_seq {
                    any_changed = true;
                    break;
                }
                // Force requote if mid changed meaningfully.
                if let (Some(cur), Some(prev)) = (mid, prev_mid) {
                    let tick_size = cfg.venues.get(vi).map(|v| v.tick_size).unwrap_or(0.01);
                    if (cur - prev).abs() > tick_size * 0.5 {
                        any_changed = true;
                        break;
                    }
                } else if mid.is_some() != prev_mid.is_some() {
                    any_changed = true;
                    break;
                }
            }
            // Also force quote if no live orders exist (need to establish quotes).
            let has_live_orders = has_tracked_mm_orders(&state);
            if !any_changed && has_live_orders {
                should_quote = false;
            }
        }

        let mut mm_quotes = if disable_fv_gate {
            compute_mm_quotes_with_ablations(cfg, &state, &fv_ablations)
        } else {
            // Use staleness-guarded quoting in live mode with current timestamp.
            compute_mm_quotes_with_now(cfg, &state, Some(now_ms))
        };
        apply_inventory_brake_to_quotes(&mut mm_quotes, &inventory_brake);
        inventory_soft_governor =
            apply_inventory_soft_governor(&state, &mut mm_quotes, soft_inventory_governor_limits);
        let soft_limit_triggered = inventory_soft_governor.triggered;
        let soft_unwind_cooldown_active = soft_unwind_cooldown_until_ms
            .map(|until_ms| now_ms < until_ms)
            .unwrap_or(false);
        let soft_unwind_response_backoff_active = soft_unwind_response_backoff_until_ms
            .map(|until_ms| now_ms < until_ms)
            .unwrap_or(false);
        if soft_unwind_cooldown_active {
            push_reason(
                &mut inventory_soft_governor.global_reasons,
                "soft_unwind_cooldown",
            );
            inventory_soft_governor.triggered = true;
        }
        if soft_unwind_response_backoff_active {
            push_reason(
                &mut inventory_soft_governor.global_reasons,
                "soft_unwind_response_backoff",
            );
            inventory_soft_governor.triggered = true;
        }
        let soft_unwind_state = soft_unwind_runtime_state(
            soft_limit_triggered,
            soft_unwind_cooldown_active,
            soft_unwind_response_backoff_active,
            soft_unwind_has_material_positions(cfg, &state),
            has_tracked_mm_orders(&state),
        );
        let reserve_priority_path = reserve_priority_path_for_inventory_control(
            soft_unwind_state,
            inventory_brake.triggered,
        );
        let pause_mm_quotes = soft_unwind_state.pause_mm_quotes || inventory_brake.triggered;
        let mut action_id_gen = crate::actions::ActionIdGenerator::new(tick);
        let mm_plan = plan_mm_order_actions(cfg, &state, &mm_quotes, now_ms, &mut action_id_gen);
        if should_quote && !pause_mm_quotes {
            mm_order_management = mm_plan.decision_summary.clone();
        }
        let mut intents = if should_quote && !pause_mm_quotes {
            mm_plan.intents.clone()
        } else {
            Vec::new() // Skip requote — data unchanged and live orders exist.
        };
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
            normalize_live_client_order_ids(&mut intents, tick);
            mm_order_management.bind_mm_intent_client_order_ids(&intents);
            register_mm_decision_lineage(&mut state, &mm_order_management, now_ms);
            would_send_intents.extend(intents.iter().cloned());
            // Update conditional quoting tracking for venues we're quoting.
            if conditional_quoting {
                for vm in &snapshot.market {
                    let vi = vm.venue_index;
                    if vi < last_quoted_mid.len() {
                        last_quoted_mid[vi] = vm.mid;
                        last_quoted_book_seq[vi] = vm.seq;
                        last_quote_ts_ms[vi] = now_ms;
                    }
                }
            }
        }

        let submit_start = tick_start.elapsed();
        if soft_unwind_state.refresh_cooldown && soft_unwind_cooldown_ms > 0 {
            soft_unwind_cooldown_until_ms = Some(now_ms.saturating_add(soft_unwind_cooldown_ms));
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
            let _ = send_order_fire_and_forget(
                &order_tx,
                intents,
                action_batch,
                now_ms,
                "mm_quote",
                tick,
            );
        }

        if soft_unwind_state.send_unwind && !inventory_brake.sent {
            let mut soft_unwind_intents = build_soft_unwind_intents(cfg, &state, snapshot, now_ms);
            let _ = remove_latched_intents_for_class(
                &mut soft_unwind_intents,
                &mut emergency_request_latches,
                EmergencyRequestClass::SoftUnwind,
                cfg,
                &state,
                now_ms,
            );
            apply_canary_intent_overrides(
                &mut soft_unwind_intents,
                canary_enforce_post_only,
                canary_enforce_reduce_only,
            );
            if !soft_unwind_intents.is_empty() {
                last_hedge_intent = soft_unwind_intents.first().cloned();
                normalize_live_client_order_ids(&mut soft_unwind_intents, tick);
                would_send_intents.extend(soft_unwind_intents.iter().cloned());
                if let Some(hooks) = hooks.as_ref() {
                    hooks.metrics.inc_orders(soft_unwind_intents.len());
                }
                let mut sync_soft_unwind_intents = soft_unwind_intents;
                let async_soft_unwind_requests =
                    take_emergency_single_flight_intents(cfg, &mut sync_soft_unwind_intents);
                let _ = send_emergency_single_flight_requests(
                    cfg,
                    &priority_order_tx,
                    &mut emergency_request_latches,
                    &state,
                    async_soft_unwind_requests,
                    EmergencyRequestClass::SoftUnwind,
                    now_ms,
                    tick,
                    lighter_emergency_latch_ms,
                    lighter_emergency_max_latch_ms,
                    "soft_unwind",
                );
                let sync_soft_unwind_venues =
                    venue_indices_for_order_intents(&sync_soft_unwind_intents);
                let (soft_unwind_outcome, soft_unwind_events) =
                    if !sync_soft_unwind_intents.is_empty() {
                        let action_batch = build_live_action_batch(
                            cfg,
                            &sync_soft_unwind_intents,
                            now_ms,
                            tick.saturating_mul(2),
                        );
                        send_order_and_wait_with_status(
                            &priority_order_tx,
                            sync_soft_unwind_intents,
                            action_batch,
                            now_ms,
                            1_000,
                            "soft_unwind",
                            tick,
                        )
                        .await
                    } else {
                        (OrderWaitOutcomeKind::Events, None)
                    };
                if matches!(
                    soft_unwind_outcome,
                    OrderWaitOutcomeKind::Timeout | OrderWaitOutcomeKind::HandlerDropped
                ) && soft_unwind_timeout_backoff_ms > 0
                {
                    soft_unwind_response_backoff_until_ms =
                        Some(now_ms.saturating_add(soft_unwind_timeout_backoff_ms));
                }
                if matches!(
                    soft_unwind_outcome,
                    OrderWaitOutcomeKind::Timeout | OrderWaitOutcomeKind::HandlerDropped
                ) {
                    for venue_index in sync_soft_unwind_venues {
                        latch_emergency_request(
                            &mut emergency_request_latches,
                            EmergencyRequestClass::SoftUnwind,
                            cfg,
                            &state,
                            venue_index,
                            now_ms,
                            lighter_emergency_latch_ms,
                            lighter_emergency_max_latch_ms,
                        );
                    }
                }
                if let Some(events) = soft_unwind_events {
                    let soft_unwind_fills = apply_priority_response_events(
                        cfg,
                        &mut state,
                        &mut deduper,
                        events,
                        now_ms,
                        order_snapshot_fill_inference_enabled,
                        &mut tick_exec_events,
                        &mut tick_fills,
                        &mut order_state_initialized,
                    );
                    if !soft_unwind_fills.is_empty() {
                        apply_live_fills(cfg, &mut state, &soft_unwind_fills, now_ms);
                        state.recompute_after_fills(cfg);
                    }
                    soft_unwind_response_backoff_until_ms = None;
                }
            }
        }

        let did_flush = flush_batched_fills(&mut fill_batcher, cfg, &mut state, now_ms, false);
        let run_exit_risk = !reserve_priority_path && (did_flush || account_position_changed);
        if run_exit_risk && cfg.exit.enabled {
            let mut exit_intents = exit::compute_exit_intents(cfg, &state, now_ms);
            apply_canary_intent_overrides(
                &mut exit_intents,
                canary_enforce_post_only,
                canary_enforce_reduce_only,
            );
            if !exit_intents.is_empty() {
                last_exit_intent = exit_intents.first().cloned();
                normalize_live_client_order_ids(&mut exit_intents, tick);
                would_send_intents.extend(exit_intents.iter().cloned());
                if let Some(hooks) = hooks.as_ref() {
                    hooks.metrics.inc_orders(exit_intents.len());
                }
                let mut action_id_gen = ActionIdGenerator::new(tick);
                let actions = intents_to_actions(&exit_intents, &mut action_id_gen);
                let mut action_batch = ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
                for action in actions {
                    action_batch.push(action);
                }
                if let Some(events) = send_order_and_wait(
                    &priority_order_tx,
                    exit_intents,
                    action_batch,
                    now_ms,
                    500,
                    "exit",
                    tick,
                )
                .await
                {
                    let mut exit_fills = Vec::new();
                    for event in events {
                        if deduper.is_duplicate(&event) {
                            continue;
                        }
                        if let super::types::ExecutionEvent::OrderSnapshot(snapshot) = event {
                            let (core_events, fills) = infer_fills_from_order_snapshot(
                                cfg,
                                &mut state,
                                &snapshot,
                                now_ms,
                                order_snapshot_fill_inference_enabled,
                            );
                            if !core_events.is_empty() {
                                tick_exec_events.extend(core_events.iter().cloned());
                            }
                            if !fills.is_empty() {
                                tick_fills.extend(fills.iter().cloned());
                                exit_fills.extend(fills);
                            }
                            state.live_order_state.reconcile(&snapshot, now_ms);
                            sync_venue_order_tracking_from_live_order_state(
                                &mut state,
                                snapshot.venue_index,
                            );
                            mark_venue_state_initialized(
                                &mut order_state_initialized,
                                snapshot.venue_index,
                            );
                            continue;
                        }
                        #[cfg(feature = "event_log")]
                        log_live_execution_event(&mut event_log, tick, now_ms, "gateway", &event);
                        let core_events = live_events_to_core(&[event]);
                        tick_exec_events.extend(core_events.iter().cloned());
                        let fills = apply_execution_events(&mut state, &core_events, now_ms);
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

        let hedge_band_tao = current_hedge_band_tao(cfg, &state);
        let hedge_due = scheduler.hedge_due(now_ms);
        let hedge_triggered_by_account =
            account_position_changed && state.q_global_tao.abs() > hedge_band_tao;
        if !reserve_priority_path && (hedge_due || hedge_triggered_by_account) {
            if let Some(plan) = compute_hedge_plan(cfg, &state, now_ms) {
                let mut hedge_intents = hedge_plan_to_order_intents(&plan);
                apply_canary_intent_overrides(
                    &mut hedge_intents,
                    canary_enforce_post_only,
                    canary_enforce_reduce_only,
                );
                if !hedge_intents.is_empty() {
                    last_hedge_intent = hedge_intents.first().cloned();
                    normalize_live_client_order_ids(&mut hedge_intents, tick);
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
                    if let Some(events) = send_order_and_wait(
                        &priority_order_tx,
                        hedge_intents,
                        action_batch,
                        now_ms,
                        500,
                        "hedge",
                        tick,
                    )
                    .await
                    {
                        let mut hedge_fills = Vec::new();
                        for event in events {
                            if deduper.is_duplicate(&event) {
                                continue;
                            }
                            if let super::types::ExecutionEvent::OrderSnapshot(snapshot) = event {
                                let (core_events, fills) = infer_fills_from_order_snapshot(
                                    cfg,
                                    &mut state,
                                    &snapshot,
                                    now_ms,
                                    order_snapshot_fill_inference_enabled,
                                );
                                if !core_events.is_empty() {
                                    tick_exec_events.extend(core_events.iter().cloned());
                                }
                                if !fills.is_empty() {
                                    tick_fills.extend(fills.iter().cloned());
                                    hedge_fills.extend(fills);
                                }
                                state.live_order_state.reconcile(&snapshot, now_ms);
                                sync_venue_order_tracking_from_live_order_state(
                                    &mut state,
                                    snapshot.venue_index,
                                );
                                mark_venue_state_initialized(
                                    &mut order_state_initialized,
                                    snapshot.venue_index,
                                );
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
                            let fills = apply_execution_events(&mut state, &core_events, now_ms);
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
            } else if state.q_global_tao.abs() > hedge_band_tao {
                eprintln!(
                    "[runner] tick={} hedge_due_no_plan q_global_tao={:.6} hedge_band_tao={:.6} vol_ratio={:.4} fair_value={:?} fv_available={} kill_switch={} trigger={} venue_status={:?} venue_position_tao={:?}",
                    tick,
                    state.q_global_tao,
                    hedge_band_tao,
                    state.vol_ratio_clipped,
                    state.fair_value,
                    state.fv_available,
                    state.kill_switch,
                    if hedge_due {
                        "schedule"
                    } else {
                        "account_position_change"
                    },
                    state
                        .venues
                        .iter()
                        .map(|venue| format!("{:?}", venue.status))
                        .collect::<Vec<_>>(),
                    state
                        .venues
                        .iter()
                        .map(|venue| venue.position_tao)
                        .collect::<Vec<_>>(),
                );
            }
            if hedge_due {
                scheduler.mark_hedge_ran();
            }
        }

        tick_timing.submit_us =
            tick_start.elapsed().as_micros() as u64 - submit_start.as_micros() as u64;

        update_venue_utility_state(
            cfg,
            &mut state,
            &would_send_intents,
            &tick_exec_events,
            &tick_fills,
            &tick_account_position_syncs,
        );

        // Kill switch triggered during submission phase.
        if state.kill_switch {
            tick_timing.total_us = tick_start.elapsed().as_micros() as u64;
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
                        last_exit_intent.as_ref(),
                        last_hedge_intent.as_ref(),
                        &pending_drift_events,
                        &tick_account_position_syncs,
                        &inventory_soft_governor,
                        &inventory_brake,
                        &startup_pnl_baseline,
                        &emergency_request_latches,
                        &mm_order_management,
                        market_rx_stats_enabled.then_some(&market_rx_stats),
                        if emit_tick_timing {
                            Some(&tick_timing)
                        } else {
                            None
                        },
                        &venue_health_diagnostics,
                    );
                    pending_drift_events.clear();
                }
            }
            maybe_print_market_rx_stats(tick, market_rx_stats_enabled, &market_rx_stats);
            break;
        }

        maybe_print_market_rx_stats(tick, market_rx_stats_enabled, &market_rx_stats);
        if let Some(hooks) = hooks.as_ref() {
            hooks.metrics.add_market_rx_stats(
                market_rx_stats.drained,
                market_rx_stats.out_market,
                market_rx_stats.cap_hits,
            );
        }
        // --- Tick timing: graduated overrun detection ---
        tick_timing.total_us = tick_start.elapsed().as_micros() as u64;
        if let LiveRunMode::Realtime { interval_ms, .. } = mode {
            let budget_us = (interval_ms as u64) * 1000;
            if tick_timing.total_us > budget_us {
                eprintln!(
                    "[runner] tick={} OVERRUN: {}us > {}us budget (reconcile={}us drain={}us engine={}us submit={}us order_tx_pending={})",
                    tick, tick_timing.total_us, budget_us,
                    tick_timing.reconcile_us, tick_timing.event_drain_us,
                    tick_timing.engine_us, tick_timing.submit_us, tick_timing.order_tx_pending,
                );
            } else if tick_timing.total_us > budget_us * 4 / 5 {
                eprintln!(
                    "[runner] tick={} WARN: {}us > 80% of {}us budget",
                    tick, tick_timing.total_us, budget_us,
                );
            }
        }

        // Normal path: emit telemetry with finalized timing (submit_us + total_us are accurate).
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
                    last_exit_intent.as_ref(),
                    last_hedge_intent.as_ref(),
                    &pending_drift_events,
                    &tick_account_position_syncs,
                    &inventory_soft_governor,
                    &inventory_brake,
                    &startup_pnl_baseline,
                    &emergency_request_latches,
                    &mm_order_management,
                    market_rx_stats_enabled.then_some(&market_rx_stats),
                    if emit_tick_timing {
                        Some(&tick_timing)
                    } else {
                        None
                    },
                    &venue_health_diagnostics,
                );
                pending_drift_events.clear();
            }
        }

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
    priority_order_tx: &mpsc::Sender<LiveOrderRequest>,
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

    // Batch all venue cancel-all intents into a single channel send.
    let mut cancel_intents: Vec<OrderIntent> = Vec::with_capacity(cfg.venues.len());
    for (venue_index, venue) in cfg.venues.iter().enumerate() {
        cancel_intents.push(OrderIntent::CancelAll(crate::types::CancelAllOrderIntent {
            venue_index: Some(venue_index),
            venue_id: Some(venue.id_arc.clone()),
        }));
    }
    if !cancel_intents.is_empty() {
        let mut action_id_gen = ActionIdGenerator::new(tick);
        let actions = intents_to_actions(&cancel_intents, &mut action_id_gen);
        let mut action_batch = ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
        for action in actions {
            action_batch.push(action);
        }
        if let Some(hooks_ref) = hooks {
            hooks_ref.metrics.inc_cancel_all();
        }
        if let Some(events) = send_order_and_wait(
            priority_order_tx,
            cancel_intents,
            action_batch,
            now_ms,
            kill_cancel_all_timeout_ms(cfg.venues.len()),
            "kill_cancel_all",
            tick,
        )
        .await
        {
            #[cfg(feature = "event_log")]
            log_live_execution_events_env(tick, now_ms, "gateway", &events);
            let core_events = live_events_to_core(&events);
            let fills = apply_execution_events(state, &core_events, now_ms);
            if !fills.is_empty() {
                apply_live_fills(cfg, state, &fills, now_ms);
                state.recompute_after_fills(cfg);
            }
        }
    }

    if best_effort_flatten {
        if let Some(intent) = state.best_effort_kill_intent_exit_first(cfg, tick) {
            let mut flatten_intents = vec![intent];
            normalize_live_client_order_ids(&mut flatten_intents, tick);
            let mut action_id_gen = ActionIdGenerator::new(tick);
            let actions = intents_to_actions(&flatten_intents, &mut action_id_gen);
            let mut action_batch = ActionBatch::new(now_ms, tick, &cfg.version).with_seed(None);
            for action in actions {
                action_batch.push(action);
            }
            if let Some(events) = send_order_and_wait(
                priority_order_tx,
                flatten_intents,
                action_batch,
                now_ms,
                KILL_FLATTEN_TIMEOUT_MS,
                "flatten",
                tick,
            )
            .await
            {
                #[cfg(feature = "event_log")]
                log_live_execution_events_env(tick, now_ms, "gateway", &events);
                let core_events = live_events_to_core(&events);
                let fills = apply_execution_events(state, &core_events, now_ms);
                if !fills.is_empty() {
                    apply_live_fills(cfg, state, &fills, now_ms);
                    state.recompute_after_fills(cfg);
                }
            }
        }
    }
}

fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as TimestampMs
}

fn is_numeric_id(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_hyperliquid_cloid(value: &str) -> bool {
    value.len() == 34
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

const LIGHTER_CLIENT_ORDER_INDEX_MAX: u64 = 0x0000_ffff_ffff_ffff;

fn lighter_numeric_client_order_id(seed: &str) -> String {
    if let Ok(parsed) = seed.parse::<u64>() {
        if (1..=LIGHTER_CLIENT_ORDER_INDEX_MAX).contains(&parsed) {
            return parsed.to_string();
        }
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in seed.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash & LIGHTER_CLIENT_ORDER_INDEX_MAX).max(1).to_string()
}

fn live_client_order_seed() -> u64 {
    *LIVE_CLIENT_ORDER_SEED.get_or_init(|| {
        (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis() as u64)
            & 0x0000_00ff_ffff_ffff
    })
}

fn scoped_live_client_order_id(seed: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in seed.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!(
        "co_{:010x}_{:012x}",
        live_client_order_seed(),
        hash & 0x0000_ffff_ffff_ffff
    )
}

fn hyperliquid_cloid(seed: &str) -> String {
    let mut hi: u64 = 0xcbf29ce484222325;
    let mut lo: u64 = 0x84222325cbf29ce4;
    for byte in seed.as_bytes() {
        let byte = u64::from(*byte);
        hi ^= byte;
        hi = hi.wrapping_mul(0x100000001b3);
        lo ^= byte.rotate_left(1);
        lo = lo.wrapping_mul(0x100000001b3).rotate_left(7);
    }
    format!("0x{hi:016x}{lo:016x}")
}

fn tracked_mm_open_order_count(state: &GlobalState) -> usize {
    state
        .venues
        .iter()
        .map(|venue| {
            usize::from(venue.mm_open_bid.is_some()) + usize::from(venue.mm_open_ask.is_some())
        })
        .sum()
}

fn live_open_order_count_for_venue(state: &GlobalState, venue_index: usize) -> usize {
    state
        .live_order_state
        .open_orders()
        .into_iter()
        .filter(|order| order.venue_index == venue_index)
        .count()
}

fn sync_venue_order_tracking_from_live_order_state(state: &mut GlobalState, venue_index: usize) {
    #[derive(Clone)]
    struct SyncedOrder {
        order_id: String,
        client_order_id: Option<String>,
        side: Side,
        price: f64,
        size: f64,
        purpose: OrderPurpose,
        updated_ms: TimestampMs,
    }

    let mut synced_orders = state
        .live_order_state
        .open_orders()
        .into_iter()
        .filter(|order| order.venue_index == venue_index)
        .filter_map(|order| {
            let side = order.side?;
            let price = order.price?;
            let size = order.remaining_qty.or(order.total_qty)?;
            let purpose = order.purpose?;
            let order_id = order
                .exchange_order_id
                .clone()
                .or_else(|| order.client_order_id.clone())?;
            Some(SyncedOrder {
                order_id,
                client_order_id: order.client_order_id.clone(),
                side,
                price,
                size,
                purpose,
                updated_ms: order.updated_ms,
            })
        })
        .collect::<Vec<_>>();

    synced_orders.sort_by(|a, b| {
        a.updated_ms
            .cmp(&b.updated_ms)
            .then_with(|| a.order_id.cmp(&b.order_id))
    });

    let Some(venue) = state.venues.get_mut(venue_index) else {
        return;
    };

    venue.open_orders = BTreeMap::new();
    venue.mm_open_bid = None;
    venue.mm_open_ask = None;

    for order in synced_orders {
        venue.open_orders.insert(
            order.order_id.clone(),
            OpenOrderRecord {
                order_id: order.order_id.clone(),
                client_order_id: order.client_order_id.clone(),
                side: order.side,
                price: order.price,
                size: order.size,
                remaining: order.size,
                timestamp_ms: order.updated_ms,
                purpose: order.purpose,
                time_in_force: None,
                post_only: None,
                reduce_only: None,
            },
        );

        if order.purpose != OrderPurpose::Mm {
            continue;
        }

        let tracked = MmOpenOrder {
            price: order.price,
            size: order.size,
            timestamp_ms: order.updated_ms,
            order_id: order.order_id,
        };
        match order.side {
            Side::Buy => venue.mm_open_bid = Some(tracked),
            Side::Sell => venue.mm_open_ask = Some(tracked),
        }
    }
}

fn live_mm_open_exposure_for_venue(state: &GlobalState, venue_index: usize) -> (f64, f64) {
    let mut bid_size = 0.0;
    let mut ask_size = 0.0;
    for order in state.live_order_state.open_orders() {
        if order.venue_index != venue_index {
            continue;
        }
        if order.purpose != Some(OrderPurpose::Mm) {
            continue;
        }
        let qty = order
            .remaining_qty
            .or(order.total_qty)
            .unwrap_or(0.0)
            .max(0.0);
        match order.side {
            Some(Side::Buy) => bid_size += qty,
            Some(Side::Sell) => ask_size += qty,
            _ => {}
        }
    }
    (bid_size, ask_size)
}

fn venue_index_for_order_intent(intent: &OrderIntent) -> Option<usize> {
    match intent {
        OrderIntent::Place(place) => Some(place.venue_index),
        OrderIntent::Cancel(cancel) => Some(cancel.venue_index),
        OrderIntent::Replace(replace) => Some(replace.venue_index),
        OrderIntent::CancelAll(cancel_all) => cancel_all.venue_index,
    }
}

fn intents_touch_venue(intents: &[OrderIntent], venue_index: usize) -> bool {
    intents
        .iter()
        .any(|intent| venue_index_for_order_intent(intent) == Some(venue_index))
}

fn remove_intents_for_venue(intents: &mut Vec<OrderIntent>, venue_index: usize) -> usize {
    let before = intents.len();
    intents.retain(|intent| venue_index_for_order_intent(intent) != Some(venue_index));
    before.saturating_sub(intents.len())
}

fn intents_for_venue(intents: &[OrderIntent], venue_index: usize) -> Vec<OrderIntent> {
    intents
        .iter()
        .filter(|intent| venue_index_for_order_intent(intent) == Some(venue_index))
        .cloned()
        .collect()
}

fn build_live_action_batch(
    cfg: &Config,
    intents: &[OrderIntent],
    now_ms: TimestampMs,
    batch_id: u64,
) -> ActionBatch {
    let mut action_id_gen = ActionIdGenerator::new(batch_id);
    let actions = intents_to_actions(intents, &mut action_id_gen);
    let mut action_batch = ActionBatch::new(now_ms, batch_id, &cfg.version).with_seed(None);
    for action in actions {
        action_batch.push(action);
    }
    action_batch
}

fn emergency_single_flight_enabled_for_venue_id(venue_id: &str) -> bool {
    matches!(
        venue_id.to_ascii_lowercase().as_str(),
        "lighter" | "aster" | "paradex"
    )
}

fn take_emergency_single_flight_intents(
    cfg: &Config,
    intents: &mut Vec<OrderIntent>,
) -> Vec<(usize, Vec<OrderIntent>)> {
    let mut out = Vec::new();
    for venue_index in venue_indices_for_order_intents(intents) {
        let Some(venue_cfg) = cfg.venues.get(venue_index) else {
            continue;
        };
        if !emergency_single_flight_enabled_for_venue_id(venue_cfg.id.as_str()) {
            continue;
        }
        let venue_intents = intents_for_venue(intents, venue_index);
        if venue_intents.is_empty() {
            continue;
        }
        let _ = remove_intents_for_venue(intents, venue_index);
        out.push((venue_index, venue_intents));
    }
    out
}

fn send_emergency_single_flight_requests(
    cfg: &Config,
    order_tx: &mpsc::Sender<LiveOrderRequest>,
    latches: &mut EmergencyRequestLatchSet,
    state: &GlobalState,
    requests: Vec<(usize, Vec<OrderIntent>)>,
    class: EmergencyRequestClass,
    now_ms: TimestampMs,
    tick: u64,
    latch_ms: TimestampMs,
    max_latch_ms: TimestampMs,
    label_prefix: &str,
) -> Vec<usize> {
    let mut sent = Vec::new();
    for (venue_index, intents) in requests {
        if intents.is_empty() {
            continue;
        }
        let action_batch = build_live_action_batch(
            cfg,
            &intents,
            now_ms,
            tick.saturating_mul(16)
                .saturating_add(venue_index as u64)
                .saturating_add(1),
        );
        let label = format!(
            "{label_prefix}_{}_single_flight",
            cfg.venues[venue_index].id
        );
        if send_order_fire_and_forget(order_tx, intents, action_batch, now_ms, &label, tick) {
            latch_emergency_request(
                latches,
                class,
                cfg,
                state,
                venue_index,
                now_ms,
                latch_ms,
                max_latch_ms,
            );
            sent.push(venue_index);
        }
    }
    sent
}

fn emergency_request_latch_entries_mut(
    latches: &mut EmergencyRequestLatchSet,
    class: EmergencyRequestClass,
) -> &mut Vec<EmergencyRequestLatch> {
    match class {
        EmergencyRequestClass::DisabledCancelAll => &mut latches.disabled_cancel_all,
        EmergencyRequestClass::InventoryBrake => &mut latches.inventory_brake,
        EmergencyRequestClass::SoftUnwind => &mut latches.soft_unwind,
    }
}

fn emergency_request_latch_entries(
    latches: &EmergencyRequestLatchSet,
    class: EmergencyRequestClass,
) -> &[EmergencyRequestLatch] {
    match class {
        EmergencyRequestClass::DisabledCancelAll => &latches.disabled_cancel_all,
        EmergencyRequestClass::InventoryBrake => &latches.inventory_brake,
        EmergencyRequestClass::SoftUnwind => &latches.soft_unwind,
    }
}

fn emergency_request_latch_progress_tao(cfg: &Config, venue_index: usize) -> f64 {
    cfg.venues
        .get(venue_index)
        .map(|venue| (venue.lot_size_tao.max(1e-9) * 0.5).max(1e-6))
        .unwrap_or(1e-6)
}

fn clear_emergency_request_latch_if_progressed(
    latch: &mut EmergencyRequestLatch,
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) -> bool {
    let current_abs_position_tao = state
        .venues
        .get(latch.venue_index)
        .map(|venue| venue.position_tao.abs())
        .unwrap_or(0.0);
    let current_open_orders = live_open_order_count_for_venue(state, latch.venue_index);

    let progress_tao = emergency_request_latch_progress_tao(cfg, latch.venue_index);
    let position_progress =
        current_abs_position_tao + progress_tao < latch.last_observed_abs_position_tao;
    let open_order_progress = current_open_orders < latch.last_observed_open_orders;
    if position_progress || open_order_progress {
        latch.last_progress_ms = now_ms;
    }
    latch.last_observed_abs_position_tao = current_abs_position_tao;
    latch.last_observed_open_orders = current_open_orders;

    let resolved_position = current_abs_position_tao <= progress_tao;
    let resolved_open_orders = current_open_orders == 0;
    let expired = now_ms >= latch.expires_at_ms;
    let max_expired = now_ms >= latch.max_expires_at_ms;
    let clear = match latch.class {
        EmergencyRequestClass::DisabledCancelAll => {
            resolved_open_orders || open_order_progress || max_expired
        }
        EmergencyRequestClass::InventoryBrake | EmergencyRequestClass::SoftUnwind => {
            (resolved_position && resolved_open_orders)
                || position_progress
                || open_order_progress
                || max_expired
        }
    };
    if clear {
        return true;
    }
    if expired {
        let still_pending = match latch.class {
            EmergencyRequestClass::DisabledCancelAll => !resolved_open_orders,
            EmergencyRequestClass::InventoryBrake | EmergencyRequestClass::SoftUnwind => {
                !resolved_position || !resolved_open_orders
            }
        };
        if still_pending {
            let next_expiry = now_ms
                .saturating_add(latch.retry_latch_ms.max(1))
                .min(latch.max_expires_at_ms);
            if next_expiry > now_ms {
                latch.expires_at_ms = next_expiry;
                latch.extension_count = latch.extension_count.saturating_add(1);
            }
        }
    }
    false
}

fn refresh_emergency_request_latches(
    latches: &mut EmergencyRequestLatchSet,
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) {
    latches.disabled_cancel_all.retain_mut(|latch| {
        !clear_emergency_request_latch_if_progressed(latch, cfg, state, now_ms)
    });
    latches.inventory_brake.retain_mut(|latch| {
        !clear_emergency_request_latch_if_progressed(latch, cfg, state, now_ms)
    });
    latches.soft_unwind.retain_mut(|latch| {
        !clear_emergency_request_latch_if_progressed(latch, cfg, state, now_ms)
    });
}

fn emergency_request_latched(
    latches: &mut EmergencyRequestLatchSet,
    class: EmergencyRequestClass,
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
    venue_index: usize,
) -> bool {
    refresh_emergency_request_latches(latches, cfg, state, now_ms);
    emergency_request_latch_entries(latches, class)
        .iter()
        .any(|latch| latch.venue_index == venue_index)
}

fn latch_emergency_request(
    latches: &mut EmergencyRequestLatchSet,
    class: EmergencyRequestClass,
    cfg: &Config,
    state: &GlobalState,
    venue_index: usize,
    now_ms: TimestampMs,
    latch_ms: TimestampMs,
    max_latch_ms: TimestampMs,
) {
    if latch_ms <= 0 {
        return;
    }
    let Some(venue_cfg) = cfg.venues.get(venue_index) else {
        return;
    };
    let baseline_abs_position_tao = state
        .venues
        .get(venue_index)
        .map(|venue| venue.position_tao.abs())
        .unwrap_or(0.0);
    let baseline_open_orders = live_open_order_count_for_venue(state, venue_index);
    let bounded_max_latch_ms = max_latch_ms.max(latch_ms);
    let entries = emergency_request_latch_entries_mut(latches, class);
    let replacement = EmergencyRequestLatch {
        class,
        venue_index,
        venue_id: venue_cfg.id.to_string(),
        active_since_ms: now_ms,
        retry_latch_ms: latch_ms,
        expires_at_ms: now_ms.saturating_add(latch_ms),
        max_expires_at_ms: now_ms.saturating_add(bounded_max_latch_ms),
        baseline_abs_position_tao,
        baseline_open_orders,
        last_progress_ms: now_ms,
        last_observed_abs_position_tao: baseline_abs_position_tao,
        last_observed_open_orders: baseline_open_orders,
        extension_count: 0,
    };
    if let Some(existing) = entries
        .iter_mut()
        .find(|entry| entry.venue_index == venue_index)
    {
        *existing = replacement;
    } else {
        entries.push(replacement);
    }
}

fn venue_indices_for_order_intents(intents: &[OrderIntent]) -> Vec<usize> {
    let mut venues = std::collections::BTreeSet::new();
    for intent in intents {
        if let Some(venue_index) = venue_index_for_order_intent(intent) {
            venues.insert(venue_index);
        }
    }
    venues.into_iter().collect()
}

fn remove_latched_intents_for_class(
    intents: &mut Vec<OrderIntent>,
    latches: &mut EmergencyRequestLatchSet,
    class: EmergencyRequestClass,
    cfg: &Config,
    state: &GlobalState,
    now_ms: TimestampMs,
) -> Vec<usize> {
    refresh_emergency_request_latches(latches, cfg, state, now_ms);
    let mut removed = Vec::new();
    for venue_index in venue_indices_for_order_intents(intents) {
        if emergency_request_latch_entries(latches, class)
            .iter()
            .any(|latch| latch.venue_index == venue_index)
        {
            let dropped = remove_intents_for_venue(intents, venue_index);
            if dropped > 0 {
                removed.push(venue_index);
            }
        }
    }
    removed
}

fn gross_position_tao(state: &GlobalState) -> f64 {
    state
        .venues
        .iter()
        .map(|venue| venue.position_tao.abs())
        .sum()
}

fn max_abs_venue_position_tao(state: &GlobalState) -> f64 {
    state
        .venues
        .iter()
        .map(|venue| venue.position_tao.abs())
        .fold(0.0_f64, f64::max)
}

fn projected_worsening_position_tao(state: &GlobalState, venue_index: usize) -> f64 {
    let Some(venue) = state.venues.get(venue_index) else {
        return 0.0;
    };
    let tracked_bid_size = venue
        .mm_open_bid
        .as_ref()
        .map(|order| order.size.max(0.0))
        .unwrap_or(0.0);
    let tracked_ask_size = venue
        .mm_open_ask
        .as_ref()
        .map(|order| order.size.max(0.0))
        .unwrap_or(0.0);
    let (live_bid_size, live_ask_size) = live_mm_open_exposure_for_venue(state, venue_index);
    let bid_size = tracked_bid_size.max(live_bid_size);
    let ask_size = tracked_ask_size.max(live_ask_size);
    let global_sign = state
        .q_global_tao
        .partial_cmp(&0.0)
        .unwrap_or(Ordering::Equal);
    match venue
        .position_tao
        .partial_cmp(&0.0)
        .unwrap_or(Ordering::Equal)
    {
        Ordering::Greater => venue.position_tao + bid_size,
        Ordering::Less => venue.position_tao - ask_size,
        Ordering::Equal => match global_sign {
            Ordering::Greater => bid_size,
            Ordering::Less => -ask_size,
            Ordering::Equal => {
                if bid_size >= ask_size {
                    bid_size
                } else {
                    -ask_size
                }
            }
        },
    }
}

fn projected_gross_position_tao(state: &GlobalState) -> f64 {
    state
        .venues
        .iter()
        .enumerate()
        .map(|(venue_index, _)| projected_worsening_position_tao(state, venue_index).abs())
        .sum()
}

fn projected_max_abs_venue_position_tao(state: &GlobalState) -> f64 {
    state
        .venues
        .iter()
        .enumerate()
        .map(|(venue_index, _)| projected_worsening_position_tao(state, venue_index).abs())
        .fold(0.0_f64, f64::max)
}

fn projected_net_position_tao(state: &GlobalState) -> f64 {
    state
        .venues
        .iter()
        .enumerate()
        .map(|(venue_index, _)| projected_worsening_position_tao(state, venue_index))
        .sum()
}

fn evaluate_inventory_brake(
    cfg: &Config,
    state: &GlobalState,
    fractions: InventoryBrakeFractions,
    limits: InventoryBrakeLimits,
) -> InventoryBrakeStatus {
    let q_gross_tao = gross_position_tao(state);
    let q_max_abs_venue_tao = max_abs_venue_position_tao(state);
    let projected_q_global_tao = projected_net_position_tao(state);
    let projected_q_gross_tao = projected_gross_position_tao(state);
    let projected_q_max_abs_venue_tao = projected_max_abs_venue_position_tao(state);
    let mut status = InventoryBrakeStatus {
        configured: limits.configured() && fractions.configured(),
        triggered: false,
        sent: false,
        grace_active: false,
        grace_applied: false,
        grace_deadline_ms: None,
        grace_ticks_remaining: 0,
        net_fraction: fractions.net_fraction,
        gross_fraction: fractions.gross_fraction,
        venue_fraction: fractions.venue_fraction,
        max_position_tao: limits.max_position_tao,
        max_gross_position_tao: limits.max_gross_position_tao,
        max_abs_venue_position_tao: limits.max_abs_venue_position_tao,
        q_global_tao: state.q_global_tao,
        q_gross_tao,
        q_max_abs_venue_tao,
        projected_q_global_tao,
        projected_q_gross_tao,
        projected_q_max_abs_venue_tao,
        global_reasons: Vec::new(),
        blocked_venues: Vec::new(),
    };
    if !status.configured {
        return status;
    }

    let net_long_breach = limits
        .max_position_tao
        .is_some_and(|limit| state.q_global_tao >= limit);
    let net_short_breach = limits
        .max_position_tao
        .is_some_and(|limit| state.q_global_tao <= -limit);
    let projected_net_long_breach = limits
        .max_position_tao
        .is_some_and(|limit| projected_q_global_tao >= limit);
    let projected_net_short_breach = limits
        .max_position_tao
        .is_some_and(|limit| projected_q_global_tao <= -limit);
    let gross_breach = limits
        .max_gross_position_tao
        .is_some_and(|limit| q_gross_tao >= limit);
    let projected_gross_breach = limits
        .max_gross_position_tao
        .is_some_and(|limit| projected_q_gross_tao >= limit);

    if net_long_breach {
        push_reason(&mut status.global_reasons, "net_long_brake");
    }
    if net_short_breach {
        push_reason(&mut status.global_reasons, "net_short_brake");
    }
    if projected_net_long_breach {
        push_reason(&mut status.global_reasons, "projected_net_long_brake");
    }
    if projected_net_short_breach {
        push_reason(&mut status.global_reasons, "projected_net_short_brake");
    }
    if gross_breach {
        push_reason(&mut status.global_reasons, "gross_brake");
    }
    if projected_gross_breach {
        push_reason(&mut status.global_reasons, "projected_gross_brake");
    }

    for (venue_index, venue) in state.venues.iter().enumerate() {
        let position_tao = venue.position_tao;
        let projected_position_tao = projected_worsening_position_tao(state, venue_index);
        let venue_breach = limits
            .max_abs_venue_position_tao
            .is_some_and(|limit| position_tao.abs() >= limit);
        let projected_venue_breach = limits
            .max_abs_venue_position_tao
            .is_some_and(|limit| projected_position_tao.abs() >= limit);
        let mut bid_reasons = Vec::new();
        let mut ask_reasons = Vec::new();

        if net_long_breach || projected_net_long_breach {
            push_reason(&mut bid_reasons, "net_long_brake");
        }
        if net_short_breach || projected_net_short_breach {
            push_reason(&mut ask_reasons, "net_short_brake");
        }
        if gross_breach || projected_gross_breach {
            if side_increases_abs_position(Side::Buy, projected_position_tao) {
                push_reason(&mut bid_reasons, "gross_brake");
            }
            if side_increases_abs_position(Side::Sell, projected_position_tao) {
                push_reason(&mut ask_reasons, "gross_brake");
            }
        }
        if venue_breach || projected_venue_breach {
            if side_increases_abs_position(Side::Buy, projected_position_tao) {
                push_reason(&mut bid_reasons, "venue_brake");
            }
            if side_increases_abs_position(Side::Sell, projected_position_tao) {
                push_reason(&mut ask_reasons, "venue_brake");
            }
        }

        let blocked_bid = !bid_reasons.is_empty();
        let blocked_ask = !ask_reasons.is_empty();
        if blocked_bid || blocked_ask {
            status
                .blocked_venues
                .push(InventorySoftGovernorVenueStatus {
                    venue_index,
                    venue_id: cfg.venues[venue_index].id.clone(),
                    position_tao,
                    blocked_bid,
                    blocked_ask,
                    bid_reasons,
                    ask_reasons,
                });
        }
    }

    status.triggered = !status.global_reasons.is_empty() || !status.blocked_venues.is_empty();
    status
}

fn apply_inventory_brake_to_quotes(
    quotes: &mut [crate::mm::MmQuote],
    status: &InventoryBrakeStatus,
) {
    if !status.triggered {
        return;
    }
    for blocked in &status.blocked_venues {
        let Some(quote) = quotes
            .iter_mut()
            .find(|quote| quote.venue_index == blocked.venue_index)
        else {
            continue;
        };
        if blocked.blocked_bid {
            quote.bid = None;
        }
        if blocked.blocked_ask {
            quote.ask = None;
        }
    }
}

fn canary_limit_status(
    state: &GlobalState,
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao_limit: Option<f64>,
    max_open_orders: Option<usize>,
) -> CanaryLimitStatus {
    let q_global_tao = state.q_global_tao;
    let q_gross_tao = gross_position_tao(state);
    let q_max_abs_venue_tao = max_abs_venue_position_tao(state);
    let open_order_count = tracked_mm_open_order_count(state);
    let net_excess_tao = max_position_tao
        .map(|limit| (q_global_tao.abs() - limit).max(0.0))
        .unwrap_or(0.0);
    let gross_excess_tao = max_gross_position_tao
        .map(|limit| (q_gross_tao - limit).max(0.0))
        .unwrap_or(0.0);
    let venue_excess_tao = max_abs_venue_position_tao_limit
        .map(|limit| (q_max_abs_venue_tao - limit).max(0.0))
        .unwrap_or(0.0);
    let open_order_excess = max_open_orders
        .map(|limit| open_order_count.saturating_sub(limit))
        .unwrap_or(0);
    let net_breached = net_excess_tao > 0.0;
    let gross_breached = gross_excess_tao > 0.0;
    let venue_breached = venue_excess_tao > 0.0;
    let open_orders_breached = open_order_excess > 0;

    CanaryLimitStatus {
        breached: net_breached || gross_breached || venue_breached || open_orders_breached,
        net_breached,
        gross_breached,
        venue_breached,
        open_orders_breached,
        max_position_tao,
        max_gross_position_tao,
        max_abs_venue_position_tao: max_abs_venue_position_tao_limit,
        max_open_orders,
        q_global_tao,
        q_gross_tao,
        q_max_abs_venue_tao,
        open_order_count,
        net_excess_tao,
        gross_excess_tao,
        venue_excess_tao,
        open_order_excess,
    }
}

fn canary_limit_breached(
    state: &GlobalState,
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao_limit: Option<f64>,
    max_open_orders: Option<usize>,
) -> bool {
    canary_limit_status(
        state,
        max_position_tao,
        max_gross_position_tao,
        max_abs_venue_position_tao_limit,
        max_open_orders,
    )
    .breached
}

fn has_tracked_mm_orders(state: &GlobalState) -> bool {
    tracked_mm_open_order_count(state) > 0
}

fn inventory_brake_grace_is_active(
    grace_until_ms: Option<TimestampMs>,
    grace_ticks_remaining: u32,
    now_ms: TimestampMs,
) -> bool {
    grace_ticks_remaining > 0 && grace_until_ms.is_some_and(|deadline| deadline >= now_ms)
}

fn inventory_brake_grace_allowed(
    limits: &CanaryLimitStatus,
    brake: &InventoryBrakeStatus,
    config: InventoryBrakeGraceConfig,
) -> bool {
    if !config.enabled() || !brake.sent || !limits.breached || limits.open_orders_breached {
        return false;
    }

    let within_fraction = |excess: f64, limit: Option<f64>| -> bool {
        excess <= 0.0
            || limit.is_some_and(|limit| excess <= (limit * config.excess_fraction) + 1e-12)
    };

    (!limits.net_breached || within_fraction(limits.net_excess_tao, limits.max_position_tao))
        && (!limits.gross_breached
            || within_fraction(limits.gross_excess_tao, limits.max_gross_position_tao))
        && (!limits.venue_breached
            || within_fraction(limits.venue_excess_tao, limits.max_abs_venue_position_tao))
}

fn apply_inventory_brake_grace_telemetry(
    brake: &mut InventoryBrakeStatus,
    grace_until_ms: Option<TimestampMs>,
    grace_ticks_remaining: u32,
    now_ms: TimestampMs,
) {
    brake.grace_active =
        inventory_brake_grace_is_active(grace_until_ms, grace_ticks_remaining, now_ms);
    brake.grace_deadline_ms = if brake.grace_active {
        grace_until_ms
    } else {
        None
    };
    brake.grace_ticks_remaining = grace_ticks_remaining;
}

fn normalize_live_client_order_ids(intents: &mut [OrderIntent], tick: u64) {
    let mut lighter_sequence: u64 = 0;
    let mut generic_sequence: u64 = 0;
    for intent in intents {
        match intent {
            OrderIntent::Place(place)
                if place.venue_id.as_ref().eq_ignore_ascii_case("lighter") =>
            {
                let seed = place.client_order_id.clone().unwrap_or_else(|| {
                    format!(
                        "lighter:{}:{}:{:?}:{}",
                        tick, place.venue_index, place.purpose, lighter_sequence
                    )
                });
                place.client_order_id = Some(lighter_numeric_client_order_id(&seed));
                lighter_sequence = lighter_sequence.wrapping_add(1);
            }
            OrderIntent::Replace(replace)
                if replace.venue_id.as_ref().eq_ignore_ascii_case("lighter") =>
            {
                let seed = replace.client_order_id.clone().unwrap_or_else(|| {
                    format!(
                        "lighter:{}:{}:{:?}:{}",
                        tick, replace.venue_index, replace.purpose, lighter_sequence
                    )
                });
                replace.client_order_id = Some(lighter_numeric_client_order_id(&seed));
                lighter_sequence = lighter_sequence.wrapping_add(1);
            }
            OrderIntent::Place(place)
                if place.venue_id.as_ref().eq_ignore_ascii_case("hyperliquid") =>
            {
                let seed = place.client_order_id.clone().unwrap_or_else(|| {
                    format!(
                        "hyperliquid:{}:{}:{:?}",
                        tick, place.venue_index, place.purpose
                    )
                });
                if !is_hyperliquid_cloid(&seed) {
                    place.client_order_id = Some(hyperliquid_cloid(&seed));
                }
            }
            OrderIntent::Replace(replace)
                if replace
                    .venue_id
                    .as_ref()
                    .eq_ignore_ascii_case("hyperliquid") =>
            {
                let seed = replace.client_order_id.clone().unwrap_or_else(|| {
                    format!(
                        "hyperliquid:{}:{}:{:?}",
                        tick, replace.venue_index, replace.purpose
                    )
                });
                if !is_hyperliquid_cloid(&seed) {
                    replace.client_order_id = Some(hyperliquid_cloid(&seed));
                }
            }
            OrderIntent::Place(place) => {
                if let Some(client_order_id) = place.client_order_id.as_mut() {
                    if client_order_id.starts_with("co_") {
                        *client_order_id = scoped_live_client_order_id(client_order_id);
                    }
                } else {
                    let seed = format!(
                        "generic:{}:{}:{:?}:{}",
                        tick, place.venue_index, place.purpose, generic_sequence
                    );
                    place.client_order_id = Some(scoped_live_client_order_id(&seed));
                    generic_sequence = generic_sequence.wrapping_add(1);
                }
            }
            OrderIntent::Replace(replace) => {
                if let Some(client_order_id) = replace.client_order_id.as_mut() {
                    if client_order_id.starts_with("co_") {
                        *client_order_id = scoped_live_client_order_id(client_order_id);
                    }
                } else {
                    let seed = format!(
                        "generic:{}:{}:{:?}:{}",
                        tick, replace.venue_index, replace.purpose, generic_sequence
                    );
                    replace.client_order_id = Some(scoped_live_client_order_id(&seed));
                    generic_sequence = generic_sequence.wrapping_add(1);
                }
            }
            _ => {}
        }
    }
}

fn register_mm_decision_lineage(
    state: &mut GlobalState,
    mm_order_management: &MmOrderDecisionSummary,
    now_ms: TimestampMs,
) {
    for record in &mm_order_management.decision_records {
        if !matches!(record.outcome.as_str(), "place" | "replace") {
            continue;
        }
        let (Some(client_order_id), Some(price), Some(size)) = (
            record.client_order_id.as_deref(),
            record.desired_price,
            record.desired_size,
        ) else {
            continue;
        };
        let side = match record.side.as_str() {
            "Buy" => crate::types::Side::Buy,
            "Sell" => crate::types::Side::Sell,
            _ => continue,
        };
        state.live_order_state.register_mm_decision_lineage(
            record.venue_index,
            client_order_id,
            side,
            price,
            size,
            crate::types::OrderPurpose::Mm,
            &record.decision_id,
            now_ms,
        );
    }
}

fn parse_optional_positive_f64_env(key: &str) -> Option<f64> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

fn parse_optional_fraction_env(key: &str) -> Option<f64> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
}

fn parse_optional_positive_i64_env(key: &str) -> Option<i64> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
}

fn startup_pnl_baseline_config_from_env(canary_enabled: bool) -> StartupPnlBaselineConfig {
    if !canary_enabled {
        return StartupPnlBaselineConfig::default();
    }
    StartupPnlBaselineConfig {
        enabled: std::env::var("PARAPHINA_CANARY_STARTUP_PNL_BASELINE_GUARD")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true),
        pnl_abs_limit_usd: parse_optional_positive_f64_env(
            "PARAPHINA_CANARY_STARTUP_PNL_LIMIT_USD",
        )
        .unwrap_or(1.0),
        position_tol_tao: parse_optional_positive_f64_env(
            "PARAPHINA_CANARY_STARTUP_POSITION_TAO_TOL",
        )
        .unwrap_or(0.0025),
        max_wait_ticks: std::env::var("PARAPHINA_CANARY_STARTUP_PNL_MAX_WAIT_TICKS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(40),
    }
}

fn inventory_brake_fractions_from_env() -> InventoryBrakeFractions {
    let default_fraction =
        parse_optional_fraction_env("PARAPHINA_CANARY_INVENTORY_BRAKE_FRACTION").unwrap_or(0.75);
    InventoryBrakeFractions {
        net_fraction: Some(
            parse_optional_fraction_env("PARAPHINA_CANARY_INVENTORY_BRAKE_NET_FRACTION")
                .unwrap_or(default_fraction),
        ),
        gross_fraction: Some(
            parse_optional_fraction_env("PARAPHINA_CANARY_INVENTORY_BRAKE_GROSS_FRACTION")
                .unwrap_or(default_fraction),
        ),
        venue_fraction: Some(
            parse_optional_fraction_env("PARAPHINA_CANARY_INVENTORY_BRAKE_VENUE_FRACTION")
                .unwrap_or(default_fraction),
        ),
    }
}

fn inventory_brake_limits(
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
    fractions: InventoryBrakeFractions,
) -> InventoryBrakeLimits {
    InventoryBrakeLimits {
        max_position_tao: max_position_tao
            .zip(fractions.net_fraction)
            .map(|(limit, fraction)| (limit * fraction).max(0.0)),
        max_gross_position_tao: max_gross_position_tao
            .zip(fractions.gross_fraction)
            .map(|(limit, fraction)| (limit * fraction).max(0.0)),
        max_abs_venue_position_tao: max_abs_venue_position_tao
            .zip(fractions.venue_fraction)
            .map(|(limit, fraction)| (limit * fraction).max(0.0)),
    }
}

fn side_increases_abs_position(side: Side, position_tao: f64) -> bool {
    match position_tao.partial_cmp(&0.0).unwrap_or(Ordering::Equal) {
        Ordering::Greater => matches!(side, Side::Buy),
        Ordering::Less => matches!(side, Side::Sell),
        Ordering::Equal => true,
    }
}

fn push_reason(reasons: &mut Vec<String>, reason: &'static str) {
    if !reasons.iter().any(|existing| existing == reason) {
        reasons.push(reason.to_string());
    }
}

fn apply_inventory_soft_governor(
    state: &GlobalState,
    quotes: &mut [crate::mm::MmQuote],
    limits: SoftInventoryGovernorLimits,
) -> InventorySoftGovernorStatus {
    let q_gross_tao = gross_position_tao(state);
    let q_max_abs_venue_tao = max_abs_venue_position_tao(state);
    let mut status = InventorySoftGovernorStatus {
        configured: limits.configured(),
        triggered: false,
        max_position_tao: limits.max_position_tao,
        max_gross_position_tao: limits.max_gross_position_tao,
        max_abs_venue_position_tao: limits.max_abs_venue_position_tao,
        q_global_tao: state.q_global_tao,
        q_gross_tao,
        q_max_abs_venue_tao,
        global_reasons: Vec::new(),
        blocked_venues: Vec::new(),
    };

    if !status.configured {
        return status;
    }

    let net_long_breach = limits
        .max_position_tao
        .is_some_and(|limit| state.q_global_tao >= limit);
    let net_short_breach = limits
        .max_position_tao
        .is_some_and(|limit| state.q_global_tao <= -limit);
    let gross_breach = limits
        .max_gross_position_tao
        .is_some_and(|limit| q_gross_tao >= limit);

    if net_long_breach {
        status.global_reasons.push("net_long_soft_cap".to_string());
    }
    if net_short_breach {
        status.global_reasons.push("net_short_soft_cap".to_string());
    }
    if gross_breach {
        status.global_reasons.push("gross_soft_cap".to_string());
    }

    for quote in quotes.iter_mut() {
        let Some(venue_state) = state.venues.get(quote.venue_index) else {
            continue;
        };
        let position_tao = venue_state.position_tao;
        let venue_breach = limits
            .max_abs_venue_position_tao
            .is_some_and(|limit| position_tao.abs() >= limit);
        let mut bid_reasons = Vec::new();
        let mut ask_reasons = Vec::new();

        if gross_breach {
            if quote.bid.is_some() && side_increases_abs_position(Side::Buy, position_tao) {
                push_reason(&mut bid_reasons, "gross_soft_cap");
            }
            if quote.ask.is_some() && side_increases_abs_position(Side::Sell, position_tao) {
                push_reason(&mut ask_reasons, "gross_soft_cap");
            }
        }
        if net_long_breach && quote.bid.is_some() {
            push_reason(&mut bid_reasons, "net_long_soft_cap");
        }
        if net_short_breach && quote.ask.is_some() {
            push_reason(&mut ask_reasons, "net_short_soft_cap");
        }
        if venue_breach {
            if quote.bid.is_some() && side_increases_abs_position(Side::Buy, position_tao) {
                push_reason(&mut bid_reasons, "venue_soft_cap");
            }
            if quote.ask.is_some() && side_increases_abs_position(Side::Sell, position_tao) {
                push_reason(&mut ask_reasons, "venue_soft_cap");
            }
        }

        let blocked_bid = !bid_reasons.is_empty();
        let blocked_ask = !ask_reasons.is_empty();
        if blocked_bid {
            quote.bid = None;
        }
        if blocked_ask {
            quote.ask = None;
        }
        if blocked_bid || blocked_ask {
            status
                .blocked_venues
                .push(InventorySoftGovernorVenueStatus {
                    venue_index: quote.venue_index,
                    venue_id: quote.venue_id.to_string(),
                    position_tao,
                    blocked_bid,
                    blocked_ask,
                    bid_reasons,
                    ask_reasons,
                });
        }
    }

    status.triggered = !status.global_reasons.is_empty() || !status.blocked_venues.is_empty();
    status
}

fn signed_fill_delta_tao(side: Side, size: f64) -> f64 {
    match side {
        Side::Buy => size,
        Side::Sell => -size,
    }
}

fn soft_unwind_has_material_positions(cfg: &Config, state: &GlobalState) -> bool {
    state.venues.iter().enumerate().any(|(venue_index, venue)| {
        venue.position_tao.abs()
            >= cfg
                .venues
                .get(venue_index)
                .map(|cfg| cfg.lot_size_tao.max(1e-9))
                .unwrap_or(1e-9)
    })
}

fn snap_size_down_to_lot(size: f64, lot_size: f64) -> f64 {
    if !size.is_finite() || !lot_size.is_finite() || lot_size <= 0.0 {
        return 0.0;
    }
    let lots = (size / lot_size).floor();
    if !lots.is_finite() || lots <= 0.0 {
        0.0
    } else {
        lots * lot_size
    }
}

fn fresh_account_or_state_position_tao(
    cfg: &Config,
    state: &GlobalState,
    snapshot: &CanonicalCacheSnapshot,
    venue_index: usize,
    now_ms: TimestampMs,
) -> f64 {
    snapshot
        .account
        .get(venue_index)
        .filter(|acct| {
            account_cache_snapshot_fresh(
                acct,
                now_ms,
                account_snapshot_max_age_ms_for_venue(cfg, acct.venue_id.as_ref()),
            )
        })
        .map(|acct| acct.position_tao)
        .unwrap_or_else(|| {
            state
                .venues
                .get(venue_index)
                .map(|venue| venue.position_tao)
                .unwrap_or_default()
        })
}

fn fresh_account_position_tao(
    cfg: &Config,
    snapshot: &CanonicalCacheSnapshot,
    venue_index: usize,
    now_ms: TimestampMs,
) -> Option<f64> {
    snapshot
        .account
        .get(venue_index)
        .filter(|acct| {
            account_cache_snapshot_fresh(
                acct,
                now_ms,
                account_snapshot_max_age_ms_for_venue(cfg, acct.venue_id.as_ref()),
            )
        })
        .map(|acct| acct.position_tao)
}

fn evaluate_startup_pnl_baseline(
    cfg: &Config,
    state: &GlobalState,
    snapshot: &CanonicalCacheSnapshot,
    account_state_initialized: &[bool],
    now_ms: TimestampMs,
    waited_ticks: u64,
    guard_cfg: StartupPnlBaselineConfig,
) -> StartupPnlBaselineStatus {
    let mut status = StartupPnlBaselineStatus {
        enabled: guard_cfg.enabled,
        waited_ticks,
        max_wait_ticks: guard_cfg.max_wait_ticks,
        pnl_abs_limit_usd: guard_cfg.pnl_abs_limit_usd,
        position_tol_tao: guard_cfg.position_tol_tao,
        daily_realised_pnl: state.daily_realised_pnl,
        daily_unrealised_pnl: state.daily_unrealised_pnl,
        daily_pnl_total: state.daily_pnl_total,
        required_account_count: cfg.venues.len(),
        ..StartupPnlBaselineStatus::default()
    };
    if !guard_cfg.enabled {
        return status;
    }

    let fair_value = state.fair_value.unwrap_or(state.fair_value_prev).max(1.0);
    let mut pending_venues = Vec::new();
    let mut fresh_account_count = 0usize;
    let mut violating_venues = Vec::new();

    for (venue_index, venue_cfg) in cfg.venues.iter().enumerate() {
        let acct = snapshot.account.get(venue_index);
        let fresh = acct
            .filter(|acct| {
                account_cache_snapshot_fresh(
                    acct,
                    now_ms,
                    account_snapshot_max_age_ms_for_venue(cfg, acct.venue_id.as_ref()),
                )
            })
            .is_some();
        if fresh && venue_state_initialized(account_state_initialized, venue_index) {
            fresh_account_count += 1;
        } else {
            pending_venues.push(venue_cfg.id.clone());
        }
        let Some(acct) = acct else {
            continue;
        };
        if !fresh {
            continue;
        }
        let unrealised_pnl_usd = acct.position_tao * (fair_value - acct.avg_entry_price);
        let position_breach = acct.position_tao.abs() > guard_cfg.position_tol_tao;
        if position_breach || unrealised_pnl_usd <= -guard_cfg.pnl_abs_limit_usd {
            violating_venues.push(StartupPnlBaselineVenueStatus {
                venue_index,
                venue_id: acct.venue_id.to_string(),
                position_tao: acct.position_tao,
                avg_entry_price: acct.avg_entry_price,
                fair_value,
                unrealised_pnl_usd,
                position_breach,
            });
        }
    }

    status.fresh_account_count = fresh_account_count;
    status.pending_venues = pending_venues;
    status.violating_venues = violating_venues;

    let full_account_coverage = fresh_account_count >= status.required_account_count;
    let partial_account_coverage_timed_out = !full_account_coverage
        && waited_ticks >= guard_cfg.max_wait_ticks
        && fresh_account_count > 0;
    if !full_account_coverage && !partial_account_coverage_timed_out {
        status.waiting_for_accounts = true;
        if waited_ticks >= guard_cfg.max_wait_ticks {
            status.triggered = true;
            status.timed_out = true;
            status.reason = Some("awaiting_account_snapshots_timeout".to_string());
        } else {
            status.reason = Some("awaiting_account_snapshots".to_string());
        }
        return status;
    }

    if state.daily_pnl_total <= -guard_cfg.pnl_abs_limit_usd {
        status.triggered = true;
        status.reason = Some("startup_inherited_pnl_breach".to_string());
        return status;
    }

    if !status.violating_venues.is_empty() {
        status.triggered = true;
        status.reason = Some("startup_position_baseline_breach".to_string());
        return status;
    }

    status.resolved = true;
    status.passed = true;
    status.reason = Some(if partial_account_coverage_timed_out {
        "startup_baseline_partial_account_coverage_ok".to_string()
    } else {
        "startup_baseline_ok".to_string()
    });
    status
}

fn clamp_aster_emergency_target_position_tao(
    cfg: &Config,
    snapshot: &CanonicalCacheSnapshot,
    venue_index: usize,
    now_ms: TimestampMs,
    target_position_tao: f64,
) -> Option<f64> {
    let venue = cfg.venues.get(venue_index)?;
    if !venue.id.eq_ignore_ascii_case("aster") {
        return Some(target_position_tao);
    }
    let lot_size = venue.lot_size_tao.max(1e-9);
    let account_position_tao = fresh_account_position_tao(cfg, snapshot, venue_index, now_ms)?;
    if account_position_tao.abs() < lot_size {
        return None;
    }
    if target_position_tao.signum() == 0.0
        || account_position_tao.signum() != target_position_tao.signum()
    {
        return None;
    }
    let confirmed_abs = snap_size_down_to_lot(account_position_tao.abs(), lot_size);
    if confirmed_abs < lot_size {
        return None;
    }
    let max_send_abs = snap_size_down_to_lot((confirmed_abs - lot_size).max(0.0), lot_size);
    if max_send_abs < lot_size {
        return None;
    }
    let clamped_abs = target_position_tao.abs().min(max_send_abs);
    if clamped_abs < lot_size {
        return None;
    }
    Some(target_position_tao.signum() * clamped_abs)
}

fn is_benign_reduce_only_reject(reject: &OrderReject) -> bool {
    reject.venue_id.eq_ignore_ascii_case("aster")
        && reject.reduce_only == Some(true)
        && matches!(
            reject.purpose,
            Some(OrderPurpose::Hedge) | Some(OrderPurpose::Exit)
        )
        && reject.reason.contains("ReduceOnly Order is rejected")
}

fn soft_unwind_target_position_tao(
    cfg: &Config,
    state: &GlobalState,
    snapshot: &CanonicalCacheSnapshot,
    venue_index: usize,
    now_ms: TimestampMs,
) -> f64 {
    let lot_size = cfg
        .venues
        .get(venue_index)
        .map(|venue| venue.lot_size_tao.max(1e-9))
        .unwrap_or(1e-9);
    let position_source =
        fresh_account_or_state_position_tao(cfg, state, snapshot, venue_index, now_ms);
    let snapped_abs = snap_size_down_to_lot(position_source.abs(), lot_size);
    if snapped_abs < lot_size {
        return 0.0;
    }
    let conservative_abs = if snapped_abs > lot_size {
        snap_size_down_to_lot(snapped_abs - lot_size, lot_size).max(lot_size)
    } else {
        snapped_abs
    };
    position_source.signum() * conservative_abs
}

fn inventory_brake_target_position_tao(
    cfg: &Config,
    state: &GlobalState,
    snapshot: &CanonicalCacheSnapshot,
    venue_index: usize,
    now_ms: TimestampMs,
) -> f64 {
    let lot_size = cfg
        .venues
        .get(venue_index)
        .map(|venue| venue.lot_size_tao.max(1e-9))
        .unwrap_or(1e-9);
    let position_source =
        fresh_account_or_state_position_tao(cfg, state, snapshot, venue_index, now_ms);
    let projected_position_source = projected_worsening_position_tao(state, venue_index);
    // Size the brake to the live projected same-side exposure so a last worsening fill racing
    // the cancel does not leave the venue one lot beyond the hard cap.
    let target_position_source = if position_source.signum() != 0.0
        && projected_position_source.signum() == position_source.signum()
        && projected_position_source.abs() > position_source.abs()
    {
        projected_position_source
    } else {
        position_source
    };
    let snapped_abs = snap_size_down_to_lot(target_position_source.abs(), lot_size);
    if snapped_abs < lot_size {
        return 0.0;
    }
    target_position_source.signum() * snapped_abs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SoftUnwindRuntimeState {
    pause_mm_quotes: bool,
    send_unwind: bool,
    refresh_cooldown: bool,
}

fn soft_unwind_runtime_state(
    soft_limit_triggered: bool,
    cooldown_active: bool,
    response_backoff_active: bool,
    has_material_positions: bool,
    has_live_mm_orders: bool,
) -> SoftUnwindRuntimeState {
    let send_unwind = !response_backoff_active
        && (soft_limit_triggered || cooldown_active)
        && (has_material_positions || has_live_mm_orders);
    SoftUnwindRuntimeState {
        pause_mm_quotes: soft_limit_triggered || cooldown_active || response_backoff_active,
        send_unwind,
        refresh_cooldown: soft_limit_triggered,
    }
}

fn reserve_priority_path_for_inventory_control(
    soft_unwind_state: SoftUnwindRuntimeState,
    inventory_brake_triggered: bool,
) -> bool {
    soft_unwind_state.pause_mm_quotes || inventory_brake_triggered
}

fn aggressive_unwind_price(
    cfg: &Config,
    state: &GlobalState,
    venue_index: usize,
    side: Side,
) -> Option<f64> {
    let venue_cfg = cfg.venues.get(venue_index)?;
    let venue = state.venues.get(venue_index)?;
    let fair = state.fair_value.unwrap_or(state.fair_value_prev).max(1.0);
    let reference_mid = venue.mid.unwrap_or(fair).max(venue_cfg.tick_size.max(1e-9));
    let spread = venue
        .spread
        .filter(|spread| spread.is_finite() && *spread > 0.0)
        .unwrap_or(venue_cfg.tick_size.max(1e-9) * 2.0);
    let half_spread = 0.5 * spread;
    let best_bid = (reference_mid - half_spread).max(venue_cfg.tick_size.max(1e-9));
    let best_ask = (reference_mid + half_spread).max(venue_cfg.tick_size.max(1e-9));
    let cushion = half_spread.max(venue_cfg.tick_size.max(1e-9));
    Some(match side {
        Side::Buy => best_ask + cushion,
        Side::Sell => (best_bid - cushion).max(venue_cfg.tick_size.max(1e-9)),
    })
}

fn inventory_brake_slippage_bps_for_venue(venue_id: &str) -> f64 {
    let venue_key = venue_id.to_ascii_uppercase();
    parse_optional_positive_f64_env(
        format!("PARAPHINA_CANARY_BRAKE_UNWIND_SLIPPAGE_BPS_{venue_key}").as_str(),
    )
    .or_else(|| parse_optional_positive_f64_env("PARAPHINA_CANARY_BRAKE_UNWIND_SLIPPAGE_BPS"))
    .unwrap_or(50.0)
}

fn snap_aggressive_price_to_tick(price: f64, tick_size: f64, side: Side) -> Option<f64> {
    if !price.is_finite() || price <= 0.0 || !tick_size.is_finite() || tick_size <= 0.0 {
        return None;
    }
    let ticks = price / tick_size;
    let snapped_ticks = match side {
        Side::Buy => ticks.ceil(),
        Side::Sell => ticks.floor(),
    };
    Some((snapped_ticks * tick_size).max(tick_size))
}

fn inventory_brake_unwind_price(
    cfg: &Config,
    state: &GlobalState,
    venue_index: usize,
    side: Side,
) -> Option<f64> {
    let venue_cfg = cfg.venues.get(venue_index)?;
    let venue = state.venues.get(venue_index)?;
    let fair = state
        .fair_value
        .unwrap_or(state.fair_value_prev)
        .max(venue_cfg.tick_size.max(1e-9));
    let reference_mid = venue.mid.unwrap_or(fair).max(venue_cfg.tick_size.max(1e-9));
    let bump =
        (inventory_brake_slippage_bps_for_venue(venue_cfg.id.as_str()) / 10_000.0).max(0.0005);
    let raw_price = match side {
        Side::Buy => reference_mid * (1.0 + bump),
        Side::Sell => reference_mid * (1.0 - bump),
    };
    snap_aggressive_price_to_tick(raw_price, venue_cfg.tick_size.max(1e-9), side)
}

fn build_soft_unwind_intents(
    cfg: &Config,
    state: &GlobalState,
    snapshot: &CanonicalCacheSnapshot,
    now_ms: TimestampMs,
) -> Vec<OrderIntent> {
    let mut intents = Vec::new();

    for (venue_index, venue) in state.venues.iter().enumerate() {
        if let Some(order) = venue.mm_open_bid.as_ref() {
            intents.push(OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index,
                venue_id: cfg.venues[venue_index].id_arc.clone(),
                order_id: order.order_id.clone(),
            }));
        }
        if let Some(order) = venue.mm_open_ask.as_ref() {
            intents.push(OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index,
                venue_id: cfg.venues[venue_index].id_arc.clone(),
                order_id: order.order_id.clone(),
            }));
        }
    }

    for venue_index in 0..state.venues.len() {
        let lot_size = cfg.venues[venue_index].lot_size_tao.max(1e-9);
        let Some(unwind_position_tao) = clamp_aster_emergency_target_position_tao(
            cfg,
            snapshot,
            venue_index,
            now_ms,
            soft_unwind_target_position_tao(cfg, state, snapshot, venue_index, now_ms),
        ) else {
            continue;
        };
        if unwind_position_tao.abs() < lot_size {
            continue;
        }
        let side = if unwind_position_tao > 0.0 {
            Side::Sell
        } else {
            Side::Buy
        };
        let Some(price) = aggressive_unwind_price(cfg, state, venue_index, side) else {
            continue;
        };
        intents.push(OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index,
            venue_id: cfg.venues[venue_index].id_arc.clone(),
            side,
            price,
            size: unwind_position_tao.abs(),
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
        }));
    }

    intents
}

fn build_inventory_brake_intents(
    cfg: &Config,
    state: &GlobalState,
    snapshot: &CanonicalCacheSnapshot,
    now_ms: TimestampMs,
    brake: &InventoryBrakeStatus,
) -> Vec<OrderIntent> {
    let mut intents = Vec::new();

    for blocked in &brake.blocked_venues {
        let Some(venue) = state.venues.get(blocked.venue_index) else {
            continue;
        };
        if blocked.blocked_bid {
            if let Some(order) = venue.mm_open_bid.as_ref() {
                intents.push(OrderIntent::Cancel(crate::types::CancelOrderIntent {
                    venue_index: blocked.venue_index,
                    venue_id: cfg.venues[blocked.venue_index].id_arc.clone(),
                    order_id: order.order_id.clone(),
                }));
            }
        }
        if blocked.blocked_ask {
            if let Some(order) = venue.mm_open_ask.as_ref() {
                intents.push(OrderIntent::Cancel(crate::types::CancelOrderIntent {
                    venue_index: blocked.venue_index,
                    venue_id: cfg.venues[blocked.venue_index].id_arc.clone(),
                    order_id: order.order_id.clone(),
                }));
            }
        }
    }

    for venue_index in 0..state.venues.len() {
        let lot_size = cfg.venues[venue_index].lot_size_tao.max(1e-9);
        let Some(unwind_position_tao) = clamp_aster_emergency_target_position_tao(
            cfg,
            snapshot,
            venue_index,
            now_ms,
            inventory_brake_target_position_tao(cfg, state, snapshot, venue_index, now_ms),
        ) else {
            continue;
        };
        if unwind_position_tao.abs() < lot_size {
            continue;
        }
        let side = if unwind_position_tao > 0.0 {
            Side::Sell
        } else {
            Side::Buy
        };
        let Some(price) = inventory_brake_unwind_price(cfg, state, venue_index, side) else {
            continue;
        };
        intents.push(OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index,
            venue_id: cfg.venues[venue_index].id_arc.clone(),
            side,
            price,
            size: unwind_position_tao.abs(),
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
        }));
    }

    intents
}

fn build_inventory_attribution(
    cfg: &Config,
    state: &GlobalState,
    intents: &[OrderIntent],
    exec_events: &[ExecutionEvent],
    fills: &[crate::types::FillEvent],
    account_position_syncs: &[AccountPositionSyncRecord],
) -> Vec<InventoryAttributionVenue> {
    let mut out = cfg
        .venues
        .iter()
        .enumerate()
        .map(|(venue_index, venue)| InventoryAttributionVenue {
            venue_index,
            venue_id: venue.id.clone(),
            position_tao: state
                .venues
                .get(venue_index)
                .map(|venue_state| venue_state.position_tao)
                .unwrap_or_default(),
            tracked_mm_bid_live: state
                .venues
                .get(venue_index)
                .and_then(|venue_state| venue_state.mm_open_bid.as_ref())
                .is_some(),
            tracked_mm_ask_live: state
                .venues
                .get(venue_index)
                .and_then(|venue_state| venue_state.mm_open_ask.as_ref())
                .is_some(),
            ..InventoryAttributionVenue::default()
        })
        .collect::<Vec<_>>();

    for intent in intents {
        match intent {
            OrderIntent::Place(place) => {
                if let Some(venue) = out.get_mut(place.venue_index) {
                    venue.intent_place_count += 1;
                    if matches!(place.purpose, OrderPurpose::Mm) {
                        venue.mm_quote_intent_count += 1;
                    }
                }
            }
            OrderIntent::Replace(replace) => {
                if let Some(venue) = out.get_mut(replace.venue_index) {
                    venue.intent_replace_count += 1;
                    if matches!(replace.purpose, OrderPurpose::Mm) {
                        venue.mm_quote_intent_count += 1;
                    }
                }
            }
            OrderIntent::Cancel(cancel) => {
                if let Some(venue) = out.get_mut(cancel.venue_index) {
                    venue.intent_cancel_count += 1;
                }
            }
            OrderIntent::CancelAll(cancel_all) => {
                if let Some(venue_index) = cancel_all.venue_index {
                    if let Some(venue) = out.get_mut(venue_index) {
                        venue.intent_cancel_all_count += 1;
                    }
                } else {
                    for venue in &mut out {
                        venue.intent_cancel_all_count += 1;
                    }
                }
            }
        }
    }

    for event in exec_events {
        match event {
            ExecutionEvent::OrderAck(ack) => {
                if let Some(venue) = out.get_mut(ack.venue_index) {
                    venue.ack_count += 1;
                    if matches!(ack.purpose, Some(OrderPurpose::Mm)) {
                        venue.mm_ack_count += 1;
                    }
                }
            }
            ExecutionEvent::OrderReject(reject) => {
                if let Some(venue) = out.get_mut(reject.venue_index) {
                    venue.reject_count += 1;
                    if matches!(reject.purpose, Some(OrderPurpose::Mm)) {
                        venue.mm_reject_count += 1;
                    }
                    if is_benign_reduce_only_reject(reject) {
                        venue.benign_reduce_only_reject_count += 1;
                    }
                }
            }
            _ => {}
        }
    }

    for fill in fills {
        if let Some(venue) = out.get_mut(fill.venue_index) {
            venue.fill_count += 1;
            venue.fill_delta_tao += signed_fill_delta_tao(fill.side, fill.size);
            if matches!(fill.purpose, OrderPurpose::Mm) {
                venue.mm_fill_count += 1;
                venue.mm_fill_abs_tao += fill.size.abs();
            }
        }
    }

    for sync in account_position_syncs {
        if let Some(venue) = out.get_mut(sync.venue_index) {
            venue.account_sync_count += 1;
            venue.account_sync_delta_tao += sync.position_delta_tao;
        }
    }

    out
}

const VENUE_UTILITY_DECAY: f64 = 0.98;
const VENUE_UTILITY_FILL_CREDIT_DECAY: f64 = 0.995;
const VENUE_UTILITY_FILLLESS_ACK_DECAY: f64 = 0.995;

fn update_venue_utility_state(
    cfg: &Config,
    state: &mut GlobalState,
    intents: &[OrderIntent],
    exec_events: &[ExecutionEvent],
    fills: &[crate::types::FillEvent],
    account_position_syncs: &[AccountPositionSyncRecord],
) {
    let attribution = build_inventory_attribution(
        cfg,
        state,
        intents,
        exec_events,
        fills,
        account_position_syncs,
    );

    for (venue_index, venue_attr) in attribution.iter().enumerate() {
        let spread_gate_hit = quote_spread_gate_reason(
            &cfg.mm,
            &cfg.venues[venue_index].id,
            state.venues[venue_index].mid,
            state.venues[venue_index].spread,
        )
        .is_some();

        {
            let utility = &mut state.venues[venue_index].utility;
            utility.mm_ack_ewma =
                utility.mm_ack_ewma * VENUE_UTILITY_DECAY + venue_attr.mm_ack_count as f64;
            utility.mm_reject_ewma =
                utility.mm_reject_ewma * VENUE_UTILITY_DECAY + venue_attr.mm_reject_count as f64;
            utility.mm_fill_count_ewma =
                utility.mm_fill_count_ewma * VENUE_UTILITY_DECAY + venue_attr.mm_fill_count as f64;
            utility.mm_fill_base_ewma =
                utility.mm_fill_base_ewma * VENUE_UTILITY_DECAY + venue_attr.mm_fill_abs_tao;
            utility.mm_fill_credit_ewma = utility.mm_fill_credit_ewma
                * VENUE_UTILITY_FILL_CREDIT_DECAY
                + venue_attr.mm_fill_count as f64;
            utility.mm_fillless_ack_pressure = if venue_attr.mm_fill_count > 0 {
                0.0
            } else {
                utility.mm_fillless_ack_pressure * VENUE_UTILITY_FILLLESS_ACK_DECAY
                    + venue_attr.mm_ack_count as f64
            };
            utility.spread_gate_hit_ewma = utility.spread_gate_hit_ewma * VENUE_UTILITY_DECAY
                + if spread_gate_hit { 1.0 } else { 0.0 };
        }

        let decision = compute_venue_utility_decision(
            &cfg.mm,
            state.q_global_tao,
            &cfg.venues[venue_index],
            &state.venues[venue_index],
            venue_utility_conversion_penalties_enabled(),
        );
        let utility = &mut state.venues[venue_index].utility;
        utility.score = decision.score;
        utility.tier = decision.tier;
        utility.reason = decision.reason;
    }
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
    now_ms: TimestampMs,
    ext_apply_any: bool,
) {
    const EXTENDED_IDX: usize = 0;
    let max_levels = cfg.book.depth_levels.max(1) as usize;
    let alpha_short = cfg.volatility.fv_vol_alpha_short;
    let alpha_long = cfg.volatility.fv_vol_alpha_long;
    match event {
        super::types::MarketDataEvent::L2Snapshot(snapshot) => {
            if let Some(v) = state.venues.get_mut(snapshot.venue_index) {
                let is_extended = snapshot.venue_index == EXTENDED_IDX;
                let current = if is_extended { Some(v.clone()) } else { None };
                let mut candidate = current.clone().unwrap_or_else(|| v.clone());
                if let Ok(metrics) = candidate.apply_l2_snapshot(
                    &snapshot.bids,
                    &snapshot.asks,
                    snapshot.seq,
                    snapshot.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                ) {
                    if is_extended
                        && current.as_ref().is_some_and(|prev| {
                            should_freeze_extended_top_of_book(cfg, prev, &candidate)
                        })
                    {
                        // Keep Extended's internal book/sequence in sync, but do not let a
                        // distorted top of book overwrite the last good quoted state.
                        v.orderbook_l2 = candidate.orderbook_l2;
                        v.last_book_update_ms = candidate.last_book_update_ms;
                        eprintln!(
                            "WARN: Extended core book update frozen mid={} spread={}",
                            candidate.mid.unwrap_or(0.0),
                            candidate.spread.unwrap_or(0.0)
                        );
                        return;
                    }
                    *v = candidate;
                    if ext_apply_any && snapshot.venue_index == EXTENDED_IDX {
                        v.last_mid_apply_ms = Some(now_ms);
                    } else if metrics.mid.is_some() && metrics.spread.is_some() {
                        v.last_mid_apply_ms = Some(now_ms);
                    }
                }
            }
        }
        super::types::MarketDataEvent::L2Delta(delta) => {
            if let Some(v) = state.venues.get_mut(delta.venue_index) {
                let is_extended = delta.venue_index == EXTENDED_IDX;
                let current = if is_extended { Some(v.clone()) } else { None };
                let mut candidate = current.clone().unwrap_or_else(|| v.clone());
                if let Ok(metrics) = candidate.apply_l2_delta(
                    &delta.changes,
                    delta.seq,
                    delta.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                ) {
                    if is_extended
                        && current.as_ref().is_some_and(|prev| {
                            should_freeze_extended_top_of_book(cfg, prev, &candidate)
                        })
                    {
                        // Keep Extended's internal book/sequence in sync, but do not let a
                        // distorted top of book overwrite the last good quoted state.
                        v.orderbook_l2 = candidate.orderbook_l2;
                        v.last_book_update_ms = candidate.last_book_update_ms;
                        eprintln!(
                            "WARN: Extended core book update frozen mid={} spread={}",
                            candidate.mid.unwrap_or(0.0),
                            candidate.spread.unwrap_or(0.0)
                        );
                        return;
                    }
                    *v = candidate;
                    if ext_apply_any && delta.venue_index == EXTENDED_IDX {
                        v.last_mid_apply_ms = Some(now_ms);
                    } else if metrics.mid.is_some() && metrics.spread.is_some() {
                        v.last_mid_apply_ms = Some(now_ms);
                    }
                }
            }
        }
        super::types::MarketDataEvent::Trade(_) => {}
        super::types::MarketDataEvent::FundingUpdate(update) => {
            if let Some(v) = state.venues.get_mut(update.venue_index) {
                let received_ms = update.received_ms.unwrap_or(now_ms);
                v.funding_state = FundingState {
                    rate_8h: update.funding_rate_8h,
                    rate_native: update.funding_rate_native,
                    interval_sec: update.interval_sec,
                    as_of_ms_exchange: Some(update.timestamp_ms).filter(|v| *v > 0),
                    received_ms: Some(received_ms).filter(|v| *v > 0),
                    next_funding_ms: update.next_funding_ms,
                    settlement_price_kind: update
                        .settlement_price_kind
                        .unwrap_or(SettlementPriceKind::Unknown),
                    source: update.source,
                    status: update
                        .funding_rate_8h
                        .map(|_| FundingStatus::Healthy)
                        .unwrap_or(FundingStatus::Unknown),
                };
                if let Some(rate) = update.funding_rate_8h {
                    v.funding_8h = rate;
                }
            }
        }
    }
}

fn should_freeze_extended_top_of_book(
    cfg: &Config,
    current: &VenueState,
    candidate: &VenueState,
) -> bool {
    let Some(mid) = candidate.mid.filter(|v| v.is_finite() && *v > 0.0) else {
        return false;
    };
    let Some(spread) = candidate.spread.filter(|v| v.is_finite()) else {
        return false;
    };
    if spread <= 0.0 {
        return true;
    }

    // Extended occasionally emits transient books with wide spreads or large
    // one-tick price jumps. Preserve book continuity, but do not let those
    // top-of-book outliers contaminate local vol/toxicity and disable the venue.
    let rel_limit = cfg.book.max_mid_jump_pct.abs().clamp(0.005, 0.01);
    if spread / mid > rel_limit {
        return true;
    }
    if let Some(prev_mid) = current.mid.filter(|v| v.is_finite() && *v > 0.0) {
        if ((mid - prev_mid) / prev_mid).abs() > rel_limit {
            return true;
        }
    }
    false
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
    // Lock-free atomic counter updates — no Mutex contention on the hot path.
    telemetry
        .stats
        .ticks_total
        .fetch_add(1, AtomicOrdering::Relaxed);
    if fv_available {
        telemetry
            .stats
            .fv_available_ticks
            .fetch_add(1, AtomicOrdering::Relaxed);
    }
    if stale_count > 0 {
        telemetry
            .stats
            .venue_staleness_events
            .fetch_add(stale_count, AtomicOrdering::Relaxed);
    }
    if disabled_count > 0 {
        telemetry
            .stats
            .venue_disabled_events
            .fetch_add(disabled_count, AtomicOrdering::Relaxed);
    }
    if kill_transition {
        telemetry
            .stats
            .kill_events
            .fetch_add(1, AtomicOrdering::Relaxed);
    }
    // Only acquire the Mutex for the purpose-tracking HashMaps when there are intents.
    if !would_send_intents.is_empty() {
        if let Ok(mut maps) = telemetry.stats.purpose_maps.lock() {
            for intent in would_send_intents {
                match intent {
                    OrderIntent::Place(place) => {
                        let key = format!("{:?}", place.purpose);
                        *maps.would_place_by_purpose.entry(key).or_insert(0) += 1;
                    }
                    OrderIntent::Cancel(_) | OrderIntent::CancelAll(_) => {
                        *maps
                            .would_cancel_by_purpose
                            .entry("unknown".to_string())
                            .or_insert(0) += 1;
                    }
                    OrderIntent::Replace(replace) => {
                        let key = format!("{:?}", replace.purpose);
                        *maps.would_replace_by_purpose.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
    }
}

/// Per-tick timing breakdown (microseconds) for telemetry and overrun detection.
#[derive(Debug, Clone, Default)]
struct TickTiming {
    /// Wall-clock time spent in account reconciliation phase.
    reconcile_us: u64,
    /// Cumulative time through event drain + processing phase.
    event_drain_us: u64,
    /// Cumulative time through engine tick + risk update phase.
    engine_us: u64,
    /// Wall-clock time spent blocking on order submission (mm + exit + hedge).
    submit_us: u64,
    /// Total tick wall-clock time (set at tick end).
    total_us: u64,
    /// Approximate pending requests across the priority and MM order channels.
    order_tx_pending: usize,
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
    account_position_syncs: &[AccountPositionSyncRecord],
    inventory_soft_governor: &InventorySoftGovernorStatus,
    inventory_brake: &InventoryBrakeStatus,
    startup_pnl_baseline: &StartupPnlBaselineStatus,
    emergency_request_latches: &EmergencyRequestLatchSet,
    mm_order_management: &MmOrderDecisionSummary,
    market_rx_stats: Option<&MarketRxStats>,
    tick_timing: Option<&TickTiming>,
    venue_health_diagnostics: &[VenueHealthDiagnostics],
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
        account_position_syncs,
        max_orders_per_tick: telemetry.max_orders_per_tick,
        venue_health_diagnostics,
    });
    ensure_schema_v1(&mut record);
    if let serde_json::Value::Object(ref mut map) = record {
        map.insert(
            "inventory_soft_governor".to_string(),
            serde_json::to_value(inventory_soft_governor).unwrap_or_default(),
        );
        map.insert(
            "inventory_brake".to_string(),
            serde_json::to_value(inventory_brake).unwrap_or_default(),
        );
        map.insert(
            "startup_pnl_baseline".to_string(),
            serde_json::to_value(startup_pnl_baseline).unwrap_or_default(),
        );
        map.insert(
            "inventory_attribution".to_string(),
            serde_json::to_value(build_inventory_attribution(
                cfg,
                state,
                would_send_intents,
                exec_events,
                fills,
                account_position_syncs,
            ))
            .unwrap_or_default(),
        );
        map.insert(
            "emergency_request_latches".to_string(),
            serde_json::to_value(emergency_request_latches).unwrap_or_default(),
        );
        if telemetry.execution_mode != "replay" {
            map.insert(
                "mm_order_management".to_string(),
                serde_json::to_value(mm_order_management).unwrap_or_default(),
            );
        }
    }
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
    if let Some(timing) = tick_timing {
        if let serde_json::Value::Object(ref mut map) = record {
            map.insert(
                "tick_timing".to_string(),
                json!({
                    "reconcile_us": timing.reconcile_us,
                    "event_drain_us": timing.event_drain_us,
                    "engine_us": timing.engine_us,
                    "submit_us": timing.submit_us,
                    "total_us": timing.total_us,
                    "total_ms": timing.total_us / 1000,
                    "order_tx_pending": timing.order_tx_pending,
                }),
            );
        }
    }
    telemetry.sink.log_json(&record);
}

fn current_hedge_band_tao(cfg: &Config, state: &GlobalState) -> f64 {
    cfg.hedge.band_base_tao.max(0.0)
        * (1.0 + cfg.hedge.band_vol_mult * state.vol_ratio_clipped.max(0.0))
}

fn apply_priority_response_events(
    cfg: &Config,
    state: &mut GlobalState,
    deduper: &mut ExecutionEventDeduper,
    events: Vec<super::types::ExecutionEvent>,
    now_ms: TimestampMs,
    order_snapshot_fill_inference_enabled: bool,
    tick_exec_events: &mut Vec<ExecutionEvent>,
    tick_fills: &mut Vec<crate::types::FillEvent>,
    order_state_initialized: &mut [bool],
) -> Vec<crate::types::FillEvent> {
    let mut response_fills = Vec::new();
    for event in events {
        if deduper.is_duplicate(&event) {
            continue;
        }
        if let super::types::ExecutionEvent::OrderSnapshot(snapshot) = event {
            let (core_events, fills) = infer_fills_from_order_snapshot(
                cfg,
                state,
                &snapshot,
                now_ms,
                order_snapshot_fill_inference_enabled,
            );
            if !core_events.is_empty() {
                tick_exec_events.extend(core_events.iter().cloned());
            }
            if !fills.is_empty() {
                tick_fills.extend(fills.iter().cloned());
                response_fills.extend(fills);
            }
            state.live_order_state.reconcile(&snapshot, now_ms);
            sync_venue_order_tracking_from_live_order_state(state, snapshot.venue_index);
            mark_venue_state_initialized(order_state_initialized, snapshot.venue_index);
            continue;
        }
        let core_events = live_events_to_core(&[event]);
        tick_exec_events.extend(core_events.iter().cloned());
        let fills = apply_execution_events(state, &core_events, now_ms);
        if !fills.is_empty() {
            tick_fills.extend(fills.iter().cloned());
            response_fills.extend(fills);
        }
    }
    response_fills
}

fn inferred_passive_fill_fee_bps(cfg: &Config, venue_index: usize, purpose: OrderPurpose) -> f64 {
    cfg.venues
        .get(venue_index)
        .map(|venue| match purpose {
            OrderPurpose::Mm => venue.maker_fee_bps - venue.maker_rebate_bps,
            OrderPurpose::Exit | OrderPurpose::Hedge => venue.taker_fee_bps,
        })
        .unwrap_or(0.0)
}

fn infer_fills_from_order_snapshot(
    cfg: &Config,
    state: &mut GlobalState,
    snapshot: &super::types::OrderSnapshot,
    now_ms: TimestampMs,
    fill_inference_enabled: bool,
) -> (Vec<ExecutionEvent>, Vec<crate::types::FillEvent>) {
    if !fill_inference_enabled {
        return (Vec::new(), Vec::new());
    }
    let inferred_live_events = state
        .live_order_state
        .reconcile_with_fill_inference(snapshot)
        .into_iter()
        .map(|fill| {
            LiveExecutionEvent::Filled(super::types::Fill {
                venue_index: fill.venue_index,
                venue_id: fill.venue_id,
                seq: fill.seq,
                timestamp_ms: snapshot.timestamp_ms,
                order_id: fill.order_id,
                client_order_id: fill.client_order_id,
                fill_id: None,
                side: fill.side,
                price: fill.price,
                size: fill.size,
                purpose: fill.purpose,
                fee_bps: inferred_passive_fill_fee_bps(cfg, fill.venue_index, fill.purpose),
            })
        })
        .collect::<Vec<_>>();
    if inferred_live_events.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let core_events = live_events_to_core(&inferred_live_events);
    let fills = apply_execution_events(state, &core_events, now_ms);
    (core_events, fills)
}

fn infer_fills_from_account_position_syncs(
    cfg: &Config,
    state: &mut GlobalState,
    account_position_syncs: &[AccountPositionSyncRecord],
    now_ms: TimestampMs,
) -> (Vec<ExecutionEvent>, Vec<crate::types::FillEvent>) {
    let inferred_live_events = account_position_syncs
        .iter()
        .flat_map(|sync| {
            state.live_order_state.infer_fills_from_position_delta(
                sync.venue_index,
                &sync.venue_id,
                sync.position_delta_tao,
                sync.snapshot_seq,
                now_ms,
            )
        })
        .map(|fill| {
            LiveExecutionEvent::Filled(super::types::Fill {
                venue_index: fill.venue_index,
                venue_id: fill.venue_id,
                seq: fill.seq,
                timestamp_ms: now_ms,
                order_id: fill.order_id,
                client_order_id: fill.client_order_id,
                fill_id: None,
                side: fill.side,
                price: fill.price,
                size: fill.size,
                purpose: fill.purpose,
                fee_bps: inferred_passive_fill_fee_bps(cfg, fill.venue_index, fill.purpose),
            })
        })
        .collect::<Vec<_>>();
    if inferred_live_events.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let core_events = live_events_to_core(&inferred_live_events);
    let fills = apply_execution_events(state, &core_events, now_ms);
    (core_events, fills)
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
                    client_order_id: rej.client_order_id.clone(),
                    seq: Some(rej.seq),
                    purpose: rej.purpose,
                    reduce_only: rej.reduce_only,
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
                    purpose: None,
                    reduce_only: None,
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
                    purpose: None,
                    reduce_only: None,
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
    cfg: &Config,
    snapshot: &super::state_cache::CanonicalCacheSnapshot,
    state: &mut GlobalState,
    now_ms: TimestampMs,
) -> AccountSnapshotApplyResult {
    let sigma_eff = state.sigma_eff;
    let mut position_changed = false;
    let mut position_syncs = Vec::new();
    for acct in &snapshot.account {
        if !account_cache_snapshot_fresh(
            acct,
            now_ms,
            account_snapshot_max_age_ms_for_venue(cfg, acct.venue_id.as_ref()),
        ) {
            continue;
        }
        let Some(v) = state.venues.get_mut(acct.venue_index) else {
            continue;
        };
        let pre_position_tao = v.position_tao;
        let pre_margin_available_usd = v.margin_available_usd;
        if (pre_position_tao - acct.position_tao).abs() > 1e-9 {
            position_changed = true;
            position_syncs.push(AccountPositionSyncRecord {
                venue_index: acct.venue_index,
                venue_id: acct.venue_id.to_string(),
                snapshot_seq: acct.seq,
                snapshot_timestamp_ms: acct.timestamp_ms,
                ingest_now_ms: now_ms,
                pre_position_tao,
                post_position_tao: acct.position_tao,
                position_delta_tao: acct.position_tao - pre_position_tao,
                pre_margin_available_usd,
                post_margin_available_usd: acct.margin_available_usd,
                source: "account_snapshot",
            });
        }
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
            v.funding_state = FundingState {
                rate_8h: Some(funding),
                rate_native: Some(funding),
                interval_sec: None,
                as_of_ms_exchange: acct.timestamp_ms.filter(|v| *v > 0),
                received_ms: Some(now_ms).filter(|v| *v > 0),
                next_funding_ms: None,
                settlement_price_kind: SettlementPriceKind::Unknown,
                source: FundingSource::AccountSnapshot,
                status: FundingStatus::Healthy,
            };
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
    AccountSnapshotApplyResult {
        position_changed,
        position_syncs,
    }
}

#[derive(Debug, Clone, Default)]
pub struct AccountSnapshotApplyResult {
    pub position_changed: bool,
    pub position_syncs: Vec<AccountPositionSyncRecord>,
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
    let ext_apply_any = std::env::var("PARAPHINA_EXTENDED_APPLY_AGE_ON_ANY_L2")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
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
        sink: TelemetrySinkHandle::Sync(Arc::new(Mutex::new(telemetry_sink))),
        shadow_mode: false,
        execution_mode: "replay",
        max_orders_per_tick: 200,
        stats: Arc::new(LiveTelemetryStats::default()),
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
                apply_market_event_to_core(&mut state, cfg, event, current_now_ms, ext_apply_any);
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
                sync_venue_order_tracking_from_live_order_state(&mut state, snapshot.venue_index);
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
    let snapshot = cache.snapshot_per_venue(now_ms, &cfg.venues, cfg.book.stale_ms);
    let update = health_manager.update_from_snapshot(cfg, state, &snapshot);
    let disabled = update.disabled;
    let venue_health_diagnostics = health_manager.diagnostics();
    let cache_events = snapshot_to_core_events(&snapshot, state);
    let _ = apply_execution_events(state, &cache_events, now_ms);
    let account_apply = apply_account_snapshot_to_state(cfg, &snapshot, state, now_ms);
    engine.main_tick_without_risk(state, now_ms);
    if let Some(scheduler) = scheduler {
        if scheduler.risk_due(now_ms) {
            engine.update_risk_limits_and_regime(state);
            scheduler.mark_risk_ran();
        }
    }
    let startup_pnl_baseline_cfg = startup_pnl_baseline_config_from_env(false);
    let startup_pnl_baseline = StartupPnlBaselineStatus {
        enabled: startup_pnl_baseline_cfg.enabled,
        resolved: !startup_pnl_baseline_cfg.enabled,
        passed: !startup_pnl_baseline_cfg.enabled,
        pnl_abs_limit_usd: startup_pnl_baseline_cfg.pnl_abs_limit_usd,
        position_tol_tao: startup_pnl_baseline_cfg.position_tol_tao,
        max_wait_ticks: startup_pnl_baseline_cfg.max_wait_ticks,
        ..StartupPnlBaselineStatus::default()
    };
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
        &account_apply.position_syncs,
        &InventorySoftGovernorStatus::default(),
        &InventoryBrakeStatus::default(),
        &startup_pnl_baseline,
        &EmergencyRequestLatchSet::default(),
        &MmOrderDecisionSummary::default(),
        None,
        None, // replay path: no tick timing (matches Step mode live path)
        &venue_health_diagnostics,
    );
    let _ = flush_batched_fills(fill_batcher, cfg, state, now_ms, true);
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::order_state::OrderStatus;
    use crate::live::state_cache::{
        CanonicalCacheSnapshot, VenueAccountSnapshot, VenueMarketSnapshot,
    };
    use crate::live::types;
    use crate::mm::{MmLevel, MmQuote};
    use crate::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
    use crate::state::{MmOpenOrder, OpenOrderRecord};
    use crate::telemetry::{TelemetryBuilder, TelemetryInputs};
    use crate::types::{FundingSource, OrderPurpose, SettlementPriceKind, Side, TimeInForce};
    use std::sync::{Mutex, OnceLock};
    use tokio::sync::mpsc;

    static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&str]) -> Self {
            Self {
                saved: keys
                    .iter()
                    .map(|key| ((*key).to_string(), std::env::var(key).ok()))
                    .collect(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..) {
                if let Some(value) = value {
                    std::env::set_var(&key, value);
                } else {
                    std::env::remove_var(&key);
                }
            }
        }
    }

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
            None,
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
    fn funding_update_flows_into_telemetry() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms: TimestampMs = 1_700_000_000_000;
        let update = types::FundingUpdate {
            venue_index: 0,
            venue_id: "TAO".to_string(),
            seq: 1,
            timestamp_ms: now_ms - 1_000,
            received_ms: Some(now_ms),
            funding_rate_8h: Some(0.001),
            funding_rate_native: Some(0.001),
            interval_sec: Some(8 * 60 * 60),
            next_funding_ms: Some(now_ms + 3_600_000),
            settlement_price_kind: Some(SettlementPriceKind::Mark),
            source: FundingSource::MarketDataRest,
        };
        apply_market_event_to_core(
            &mut state,
            &cfg,
            &types::MarketDataEvent::FundingUpdate(update),
            now_ms,
            false,
        );
        assert_eq!(state.venues[0].funding_state.rate_8h, Some(0.001));

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
        let funding_rates = record
            .get("venue_funding_rate_8h")
            .and_then(|v| v.as_array())
            .expect("venue_funding_rate_8h");
        let funding_status = record
            .get("venue_funding_status")
            .and_then(|v| v.as_array())
            .expect("venue_funding_status");
        assert_eq!(funding_rates[0].as_f64().unwrap_or(0.0), 0.001);
        assert_eq!(funding_status[0].as_str().unwrap_or(""), "Healthy");
    }

    #[test]
    fn current_hedge_band_scales_with_vol_ratio() {
        let mut cfg = Config::default();
        cfg.hedge.band_base_tao = 0.10;
        cfg.hedge.band_vol_mult = 0.5;
        let mut state = GlobalState::new(&cfg);
        state.vol_ratio_clipped = 2.0;

        assert!((current_hedge_band_tao(&cfg, &state) - 0.20).abs() < 1e-9);
    }

    #[test]
    fn account_snapshot_apply_reports_position_changes() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 1_700_000_000_000;
        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: vec![VenueMarketSnapshot {
                venue_index: 0,
                venue_id: "extended".into(),
                seq: 1,
                timestamp_ms: Some(now_ms),
                mid: Some(100.0),
                spread: Some(1.0),
                depth_near_mid: 1_000.0,
                is_stale: false,
            }],
            account: vec![VenueAccountSnapshot {
                venue_index: 0,
                venue_id: "extended".into(),
                seq: 1,
                timestamp_ms: Some(now_ms),
                position_tao: 0.25,
                avg_entry_price: 100.0,
                funding_8h: None,
                margin_balance_usd: 50.0,
                margin_used_usd: 5.0,
                margin_available_usd: 45.0,
                price_liq: Some(80.0),
                dist_liq_sigma: Some(10.0),
                is_stale: false,
            }],
        };

        let applied = apply_account_snapshot_to_state(&cfg, &snapshot, &mut state, now_ms);
        assert!(applied.position_changed);
        assert_eq!(applied.position_syncs.len(), 1);
        assert_eq!(applied.position_syncs[0].pre_position_tao, 0.0);
        assert_eq!(applied.position_syncs[0].post_position_tao, 0.25);
        assert_eq!(state.venues[0].position_tao, 0.25);
        assert!(
            !apply_account_snapshot_to_state(&cfg, &snapshot, &mut state, now_ms).position_changed
        );
    }

    #[test]
    fn account_position_syncs_infer_fills_without_reapplying_inventory() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 1_700_000_000_000;

        state.live_order_state.apply_execution_event(
            &ExecutionEvent::OrderAck(crate::types::OrderAck {
                venue_index: 0,
                venue_id: "extended".into(),
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                seq: Some(1),
                side: Some(Side::Sell),
                price: Some(100.0),
                size: Some(0.5),
                purpose: Some(OrderPurpose::Mm),
            }),
            now_ms,
        );
        state.live_order_state.apply_execution_event(
            &ExecutionEvent::OrderAck(crate::types::OrderAck {
                venue_index: 0,
                venue_id: "extended".into(),
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                seq: Some(2),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            now_ms + 100,
        );
        state.venues[0].position_tao = -0.2;

        let syncs = vec![AccountPositionSyncRecord {
            venue_index: 0,
            venue_id: "extended".to_string(),
            snapshot_seq: 3,
            snapshot_timestamp_ms: Some(now_ms + 200),
            ingest_now_ms: now_ms + 200,
            pre_position_tao: 0.0,
            post_position_tao: -0.2,
            position_delta_tao: -0.2,
            pre_margin_available_usd: 100.0,
            post_margin_available_usd: 100.0,
            source: "account_snapshot",
        }];

        let (events, fills) =
            infer_fills_from_account_position_syncs(&cfg, &mut state, &syncs, now_ms + 200);

        assert_eq!(events.len(), 1);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].order_id.as_deref(), Some("oid_1"));
        assert_eq!(fills[0].client_order_id.as_deref(), Some("co_1"));
        assert_eq!(fills[0].side, Side::Sell);
        assert!((fills[0].size - 0.2).abs() < 1e-9);
        assert!((state.venues[0].position_tao + 0.2).abs() < 1e-9);
        let live_order = state
            .live_order_state
            .open_orders()
            .into_iter()
            .find(|order| order.client_order_id.as_deref() == Some("co_1"))
            .expect("tracked order");
        assert_eq!(live_order.status, OrderStatus::PartiallyFilled);
        assert_eq!(live_order.remaining_qty, Some(0.3));
    }

    #[test]
    fn order_snapshot_fill_inference_can_be_disabled_for_paper_mode() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 1_700_000_000_000;

        state.live_order_state.apply_execution_event(
            &ExecutionEvent::OrderAck(crate::types::OrderAck {
                venue_index: 0,
                venue_id: "extended".into(),
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                seq: Some(1),
                side: Some(Side::Buy),
                price: Some(100.0),
                size: Some(0.5),
                purpose: Some(OrderPurpose::Mm),
            }),
            now_ms,
        );

        let snapshot = types::OrderSnapshot {
            venue_index: 0,
            venue_id: "extended".to_string(),
            seq: 2,
            timestamp_ms: now_ms + 1_000,
            open_orders: vec![types::OpenOrderSnapshot {
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                side: Side::Buy,
                price: 100.0,
                size: 0.4,
                purpose: Some(OrderPurpose::Mm),
            }],
        };

        let (events, fills) =
            infer_fills_from_order_snapshot(&cfg, &mut state, &snapshot, now_ms + 1_000, false);

        assert!(events.is_empty());
        assert!(fills.is_empty());
        let live_order = state
            .live_order_state
            .open_orders()
            .into_iter()
            .find(|order| order.client_order_id.as_deref() == Some("co_1"))
            .expect("tracked order");
        assert_eq!(live_order.remaining_qty, Some(0.5));
    }

    fn extended_snapshot(
        seq: u64,
        timestamp_ms: TimestampMs,
        bid_px: f64,
        ask_px: f64,
    ) -> types::MarketDataEvent {
        types::MarketDataEvent::L2Snapshot(types::L2Snapshot {
            venue_index: 0,
            venue_id: "extended".to_string(),
            seq,
            timestamp_ms,
            bids: vec![BookLevel {
                price: bid_px,
                size: 5.0,
            }],
            asks: vec![BookLevel {
                price: ask_px,
                size: 5.0,
            }],
        })
    }

    #[test]
    fn extended_wide_spread_snapshot_freezes_last_good_top() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        apply_market_event_to_core(
            &mut state,
            &cfg,
            &extended_snapshot(1, 1_000, 2080.0, 2080.1),
            1_000,
            false,
        );
        let good_mid = state.venues[0].mid.expect("good mid");
        let good_spread = state.venues[0].spread.expect("good spread");
        let good_prev_ln_mid = state.venues[0].prev_ln_mid;

        apply_market_event_to_core(
            &mut state,
            &cfg,
            &extended_snapshot(2, 1_100, 1999.5, 2081.7),
            1_100,
            false,
        );

        assert_eq!(state.venues[0].mid, Some(good_mid));
        assert_eq!(state.venues[0].spread, Some(good_spread));
        assert_eq!(state.venues[0].prev_ln_mid, good_prev_ln_mid);
        assert_eq!(state.venues[0].last_book_update_ms, Some(1_100));
        assert_eq!(
            state.venues[0]
                .orderbook_l2
                .best_bid()
                .expect("best bid after freeze")
                .price,
            1999.5
        );
        assert_eq!(
            state.venues[0]
                .orderbook_l2
                .best_ask()
                .expect("best ask after freeze")
                .price,
            2081.7
        );
    }

    #[test]
    fn extended_large_mid_jump_snapshot_freezes_last_good_top() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(2080.05);
        state.fair_value_prev = 2080.05;

        apply_market_event_to_core(
            &mut state,
            &cfg,
            &extended_snapshot(1, 1_000, 2080.0, 2080.1),
            1_000,
            false,
        );
        let good_mid = state.venues[0].mid.expect("good mid");
        let good_spread = state.venues[0].spread.expect("good spread");

        apply_market_event_to_core(
            &mut state,
            &cfg,
            &extended_snapshot(2, 1_100, 2111.45, 2111.55),
            1_100,
            false,
        );

        assert_eq!(state.venues[0].mid, Some(good_mid));
        assert_eq!(state.venues[0].spread, Some(good_spread));
        assert_eq!(state.venues[0].last_book_update_ms, Some(1_100));
        assert_eq!(
            state.venues[0]
                .orderbook_l2
                .best_bid()
                .expect("best bid after freeze")
                .price,
            2111.45
        );
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
            None,
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
            None,
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

    #[test]
    fn lighter_client_order_ids_are_normalized_to_numeric_strings() {
        let mut intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 3,
                venue_id: "lighter".into(),
                side: Side::Buy,
                price: 100.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("co_42_v3_mm_0".to_string()),
            }),
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 3,
                venue_id: "lighter".into(),
                side: Side::Sell,
                price: 101.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: None,
            }),
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 2,
                venue_id: "aster".into(),
                side: Side::Buy,
                price: 99.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("co_42_v2_mm_0".to_string()),
            }),
        ];

        normalize_live_client_order_ids(&mut intents, 42);

        let OrderIntent::Place(first_lighter) = &intents[0] else {
            panic!("expected lighter place intent");
        };
        let OrderIntent::Place(second_lighter) = &intents[1] else {
            panic!("expected lighter place intent");
        };
        let OrderIntent::Place(aster) = &intents[2] else {
            panic!("expected aster place intent");
        };

        let first_id = first_lighter
            .client_order_id
            .as_deref()
            .expect("lighter client_order_id");
        let second_id = second_lighter
            .client_order_id
            .as_deref()
            .expect("lighter client_order_id");
        assert!(is_numeric_id(first_id));
        assert!(is_numeric_id(second_id));
        assert!(
            first_id.parse::<u64>().expect("first lighter id") <= LIGHTER_CLIENT_ORDER_INDEX_MAX
        );
        assert!(
            second_id.parse::<u64>().expect("second lighter id") <= LIGHTER_CLIENT_ORDER_INDEX_MAX
        );
        assert_ne!(first_id, second_id);
        let aster_id = aster
            .client_order_id
            .as_deref()
            .expect("aster client_order_id");
        assert!(aster_id.starts_with("co_"));
        assert_ne!(aster_id, "co_42_v2_mm_0");
        assert_ne!(aster_id, first_id);
        assert_ne!(aster_id, second_id);
        assert!(aster_id.len() <= 27);
    }

    #[test]
    fn hyperliquid_client_order_ids_are_normalized_to_hex_cloids() {
        let mut intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 0,
                venue_id: "hyperliquid".into(),
                side: Side::Buy,
                price: 100.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("co_42_v0_mm_0".to_string()),
            }),
            OrderIntent::Replace(crate::types::ReplaceOrderIntent {
                venue_index: 0,
                venue_id: "hyperliquid".into(),
                order_id: "123".to_string(),
                side: Side::Sell,
                price: 101.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("co_42_v0_mm_1".to_string()),
            }),
        ];

        normalize_live_client_order_ids(&mut intents, 42);

        let OrderIntent::Place(place) = &intents[0] else {
            panic!("expected hyperliquid place intent");
        };
        let OrderIntent::Replace(replace) = &intents[1] else {
            panic!("expected hyperliquid replace intent");
        };

        let place_id = place
            .client_order_id
            .as_deref()
            .expect("hyperliquid place client_order_id");
        let replace_id = replace
            .client_order_id
            .as_deref()
            .expect("hyperliquid replace client_order_id");
        assert!(is_hyperliquid_cloid(place_id));
        assert!(is_hyperliquid_cloid(replace_id));
        assert_ne!(place_id, "co_42_v0_mm_0");
        assert_ne!(replace_id, "co_42_v0_mm_1");
    }

    #[test]
    fn generic_live_client_order_ids_are_generated_when_missing() {
        let mut intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 2,
                venue_id: "aster".into(),
                side: Side::Buy,
                price: 100.0,
                size: 1.0,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: None,
            }),
            OrderIntent::Replace(crate::types::ReplaceOrderIntent {
                venue_index: 4,
                venue_id: "paradex".into(),
                order_id: "co_old".to_string(),
                side: Side::Sell,
                price: 101.0,
                size: 1.0,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: None,
            }),
        ];

        normalize_live_client_order_ids(&mut intents, 42);

        let OrderIntent::Place(aster) = &intents[0] else {
            panic!("expected aster place intent");
        };
        let OrderIntent::Replace(paradex) = &intents[1] else {
            panic!("expected paradex replace intent");
        };

        let aster_id = aster
            .client_order_id
            .as_deref()
            .expect("aster client order id");
        let paradex_id = paradex
            .client_order_id
            .as_deref()
            .expect("paradex client order id");
        assert!(aster_id.starts_with("co_"));
        assert!(paradex_id.starts_with("co_"));
        assert_ne!(aster_id, paradex_id);
        assert!(aster_id.len() <= 27);
        assert!(paradex_id.len() <= 27);
    }

    #[test]
    fn oversized_numeric_lighter_client_order_ids_are_rehashed_into_range() {
        let mut intents = vec![OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index: 3,
            venue_id: "lighter".into(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some((LIGHTER_CLIENT_ORDER_INDEX_MAX + 1).to_string()),
        })];

        normalize_live_client_order_ids(&mut intents, 7);

        let OrderIntent::Place(lighter) = &intents[0] else {
            panic!("expected lighter place intent");
        };
        let client_order_id = lighter
            .client_order_id
            .as_deref()
            .expect("lighter client_order_id");
        assert!(is_numeric_id(client_order_id));
        assert_ne!(
            client_order_id,
            (LIGHTER_CLIENT_ORDER_INDEX_MAX + 1).to_string()
        );
        assert!(
            client_order_id
                .parse::<u64>()
                .expect("lighter client order id")
                <= LIGHTER_CLIENT_ORDER_INDEX_MAX
        );
    }

    #[test]
    fn tracked_mm_order_checks_ignore_stale_open_order_history() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        for idx in 0..32 {
            state.venues[0].upsert_open_order(OpenOrderRecord {
                order_id: format!("stale_{idx}"),
                client_order_id: None,
                side: Side::Buy,
                price: 100.0 + idx as f64,
                size: 1.0,
                remaining: 1.0,
                timestamp_ms: 1_700_000_000_000,
                purpose: OrderPurpose::Mm,
                time_in_force: None,
                post_only: None,
                reduce_only: None,
            });
        }

        assert_eq!(tracked_mm_open_order_count(&state), 0);
        assert!(!has_tracked_mm_orders(&state));

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 100.0,
            size: 1.0,
            timestamp_ms: 1_700_000_000_001,
            order_id: "mm_bid".to_string(),
        });
        state.venues[0].mm_open_ask = Some(MmOpenOrder {
            price: 101.0,
            size: 1.0,
            timestamp_ms: 1_700_000_000_001,
            order_id: "mm_ask".to_string(),
        });

        assert_eq!(tracked_mm_open_order_count(&state), 2);
        assert!(has_tracked_mm_orders(&state));
    }

    #[test]
    fn order_snapshot_sync_clears_stale_mm_tracking_slots() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 1_700_000_000_000;
        let venue_index = 2;

        state.live_order_state.apply_execution_event(
            &ExecutionEvent::OrderAck(crate::types::OrderAck {
                venue_index,
                venue_id: "aster".into(),
                order_id: "oid_mm_bid".to_string(),
                client_order_id: Some("co_mm_bid".to_string()),
                seq: Some(1),
                side: Some(Side::Buy),
                price: Some(2_200.0),
                size: Some(0.01),
                purpose: Some(OrderPurpose::Mm),
            }),
            now_ms,
        );
        sync_venue_order_tracking_from_live_order_state(&mut state, venue_index);
        assert!(state.venues[venue_index].mm_open_bid.is_some());
        assert_eq!(state.venues[venue_index].open_orders.len(), 1);
        assert!(has_tracked_mm_orders(&state));

        let snapshot = types::OrderSnapshot {
            venue_index,
            venue_id: "aster".to_string(),
            seq: 2,
            timestamp_ms: now_ms + 1,
            open_orders: Vec::new(),
        };
        state.live_order_state.reconcile(&snapshot, now_ms + 1);
        sync_venue_order_tracking_from_live_order_state(&mut state, venue_index);

        assert!(state.venues[venue_index].mm_open_bid.is_none());
        assert!(state.venues[venue_index].mm_open_ask.is_none());
        assert!(state.venues[venue_index].open_orders.is_empty());
        assert!(!has_tracked_mm_orders(&state));
    }

    #[test]
    fn canary_limit_breach_detects_gross_and_single_venue_inventory() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.venues[0].position_tao = -0.26;
        state.venues[2].position_tao = 0.08;
        state.venues[3].position_tao = 0.24;
        state.venues[4].position_tao = -0.04;
        state.fair_value = Some(2_340.0);
        state.fair_value_prev = 2_340.0;
        state.recompute_after_fills(&cfg);

        assert!(state.q_global_tao.abs() < 0.12);
        assert!((gross_position_tao(&state) - 0.62).abs() < 1e-9);
        assert!((max_abs_venue_position_tao(&state) - 0.26).abs() < 1e-9);
        assert!(canary_limit_breached(
            &state,
            Some(0.12),
            Some(0.20),
            Some(0.15),
            Some(10),
        ));
        assert!(!canary_limit_breached(
            &state,
            Some(0.12),
            Some(0.70),
            Some(0.30),
            Some(10),
        ));
    }

    #[test]
    fn soft_inventory_governor_blocks_only_risk_increasing_sides() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.venues[2].position_tao = -0.04;
        state.venues[3].position_tao = 0.10;
        state.venues[4].position_tao = -0.04;
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;
        state.recompute_after_fills(&cfg);

        let mk_quote = |venue_index: usize, venue_id: &str| MmQuote {
            venue_index,
            venue_id: venue_id.into(),
            bid: Some(MmLevel {
                price: 2_349.0,
                size: 0.01,
            }),
            ask: Some(MmLevel {
                price: 2_351.0,
                size: 0.01,
            }),
            generated_spread_cap_applied: false,
        };

        let mut quotes = vec![
            mk_quote(0, "extended"),
            mk_quote(1, "hyperliquid"),
            mk_quote(2, "aster"),
            mk_quote(3, "lighter"),
            mk_quote(4, "paradex"),
        ];
        let status = apply_inventory_soft_governor(
            &state,
            &mut quotes,
            SoftInventoryGovernorLimits {
                max_position_tao: Some(0.06),
                max_gross_position_tao: Some(0.08),
                max_abs_venue_position_tao: Some(0.04),
            },
        );

        assert!(status.configured);
        assert!(status.triggered);
        assert_eq!(status.global_reasons, vec!["gross_soft_cap".to_string()]);

        assert!(quotes[0].bid.is_none());
        assert!(quotes[0].ask.is_none());
        assert!(quotes[1].bid.is_none());
        assert!(quotes[1].ask.is_none());
        assert!(quotes[2].bid.is_some());
        assert!(quotes[2].ask.is_none());
        assert!(quotes[3].bid.is_none());
        assert!(quotes[3].ask.is_some());
        assert!(quotes[4].bid.is_some());
        assert!(quotes[4].ask.is_none());
    }

    #[test]
    fn inventory_brake_triggers_before_hard_limit_on_distributed_short() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 2_000;
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;

        state.venues[2].position_tao = -0.045;
        state.venues[2].mid = Some(2_351.0);
        state.venues[2].spread = Some(0.2);
        state.venues[2].mm_open_ask = Some(MmOpenOrder {
            price: 2_351.2,
            size: 0.01,
            timestamp_ms: 1_000,
            order_id: "aster_ask".to_string(),
        });
        state.venues[3].position_tao = -0.025;
        state.venues[3].mid = Some(2_350.5);
        state.venues[3].spread = Some(0.2);
        state.venues[3].mm_open_ask = Some(MmOpenOrder {
            price: 2_350.7,
            size: 0.01,
            timestamp_ms: 1_000,
            order_id: "lighter_ask".to_string(),
        });
        state.venues[4].position_tao = -0.02;
        state.venues[4].mid = Some(2_350.8);
        state.venues[4].spread = Some(0.2);
        state.venues[4].mm_open_ask = Some(MmOpenOrder {
            price: 2_351.0,
            size: 0.01,
            timestamp_ms: 1_000,
            order_id: "paradex_ask".to_string(),
        });
        state.recompute_after_fills(&cfg);

        assert!((state.q_global_tao + 0.09).abs() < 1e-9);
        assert!(!canary_limit_breached(
            &state,
            Some(0.12),
            Some(0.20),
            Some(0.08),
            Some(10),
        ));

        let fractions = InventoryBrakeFractions {
            net_fraction: Some(0.75),
            gross_fraction: Some(0.75),
            venue_fraction: Some(0.75),
        };
        let brake = evaluate_inventory_brake(
            &cfg,
            &state,
            fractions,
            inventory_brake_limits(Some(0.12), Some(0.20), Some(0.08), fractions),
        );

        assert!(brake.configured);
        assert!(brake.triggered);
        assert!(brake
            .global_reasons
            .iter()
            .any(|reason| reason == "net_short_brake"));
        assert!(brake.blocked_venues.iter().all(|venue| !venue.blocked_bid));
        let blocked_asks = brake
            .blocked_venues
            .iter()
            .filter(|venue| venue.blocked_ask)
            .map(|venue| venue.venue_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            blocked_asks,
            vec!["extended", "hyperliquid", "aster", "lighter", "paradex"]
        );

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms),
                    position_tao: state.venues[venue_index].position_tao,
                    avg_entry_price: 2_350.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let intents = build_inventory_brake_intents(&cfg, &state, &snapshot, now_ms, &brake);
        let cancel_orders = intents
            .iter()
            .filter_map(|intent| match intent {
                OrderIntent::Cancel(cancel) => Some((cancel.venue_index, cancel.order_id.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            cancel_orders,
            vec![(2, "aster_ask"), (3, "lighter_ask"), (4, "paradex_ask")]
        );

        let unwind_places = intents
            .iter()
            .filter_map(|intent| match intent {
                OrderIntent::Place(place) => {
                    Some((place.venue_index, place.side, place.size, place.price))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            unwind_places,
            vec![
                (2, Side::Buy, 0.03, 2362.76),
                (3, Side::Buy, 0.03, 2362.26),
                (4, Side::Buy, 0.03, 2362.56),
            ]
        );
    }

    #[test]
    fn inventory_brake_projects_live_mm_open_order_exposure() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        state.venues[2].position_tao = -0.034;
        state.venues[3].position_tao = -0.02;
        state.venues[4].position_tao = -0.02;
        state.recompute_after_fills(&cfg);

        assert!(!canary_limit_breached(
            &state,
            Some(0.12),
            Some(0.20),
            Some(0.08),
            Some(10),
        ));

        seed_open_orders_with_side(&mut state, 2, 4, Side::Sell, 10_000);
        seed_open_orders_with_side(&mut state, 3, 7, Side::Sell, 10_000);
        seed_open_orders_with_side(&mut state, 4, 4, Side::Sell, 10_000);

        let fractions = InventoryBrakeFractions {
            net_fraction: Some(0.75),
            gross_fraction: Some(0.75),
            venue_fraction: Some(0.75),
        };
        let brake = evaluate_inventory_brake(
            &cfg,
            &state,
            fractions,
            inventory_brake_limits(Some(0.12), Some(0.20), Some(0.08), fractions),
        );

        assert!(brake.triggered);
        assert!(brake
            .global_reasons
            .iter()
            .any(|reason| reason == "projected_net_short_brake"));
        assert!(brake
            .global_reasons
            .iter()
            .any(|reason| reason == "projected_gross_brake"));
        assert!((brake.projected_q_global_tao + 0.224).abs() < 1e-9);
        assert!((brake.projected_q_gross_tao - 0.224).abs() < 1e-9);
        assert!((brake.projected_q_max_abs_venue_tao - 0.09).abs() < 1e-9);
        let lighter = brake
            .blocked_venues
            .iter()
            .find(|venue| venue.venue_id == "lighter")
            .expect("lighter brake venue");
        assert!(lighter.blocked_ask);
        assert!(!lighter.blocked_bid);
    }

    #[test]
    fn inventory_brake_grace_allows_small_position_excess_after_sent_brake() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;
        state.venues[2].position_tao = -0.03;
        state.venues[3].position_tao = -0.02;
        state.venues[4].position_tao = -0.08;
        state.recompute_after_fills(&cfg);

        let limits = canary_limit_status(&state, Some(0.12), Some(0.20), Some(0.08), Some(10));
        assert!(limits.breached);
        assert!(limits.net_breached);
        assert!(!limits.gross_breached);
        assert!(!limits.venue_breached);
        assert!((limits.net_excess_tao - 0.01).abs() < 1e-9);

        let brake = InventoryBrakeStatus {
            configured: true,
            triggered: true,
            sent: true,
            ..InventoryBrakeStatus::default()
        };
        let config = InventoryBrakeGraceConfig {
            grace_ms: 1_500,
            grace_ticks: 3,
            excess_fraction: 0.10,
        };

        assert!(inventory_brake_grace_allowed(&limits, &brake, config));
    }

    #[test]
    fn inventory_brake_grace_rejects_open_order_breach() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        for venue in &mut state.venues {
            venue.mm_open_bid = Some(MmOpenOrder {
                price: 2_349.0,
                size: 0.01,
                timestamp_ms: 1_000,
                order_id: "bid".to_string(),
            });
            venue.mm_open_ask = Some(MmOpenOrder {
                price: 2_351.0,
                size: 0.01,
                timestamp_ms: 1_000,
                order_id: "ask".to_string(),
            });
        }
        state.recompute_after_fills(&cfg);

        let limits = canary_limit_status(&state, Some(0.12), Some(0.20), Some(0.08), Some(5));
        assert!(limits.breached);
        assert!(limits.open_orders_breached);
        assert_eq!(limits.open_order_excess, 5);

        let brake = InventoryBrakeStatus {
            configured: true,
            triggered: true,
            sent: true,
            ..InventoryBrakeStatus::default()
        };
        let config = InventoryBrakeGraceConfig {
            grace_ms: 1_500,
            grace_ticks: 3,
            excess_fraction: 0.10,
        };

        assert!(!inventory_brake_grace_allowed(&limits, &brake, config));
    }

    #[test]
    fn inventory_brake_grace_rejects_large_venue_excess() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.venues[4].position_tao = -0.10;
        state.recompute_after_fills(&cfg);

        let limits = canary_limit_status(&state, Some(0.12), Some(0.20), Some(0.08), Some(10));
        assert!(limits.breached);
        assert!(limits.venue_breached);
        assert!((limits.venue_excess_tao - 0.02).abs() < 1e-9);

        let brake = InventoryBrakeStatus {
            configured: true,
            triggered: true,
            sent: true,
            ..InventoryBrakeStatus::default()
        };
        let config = InventoryBrakeGraceConfig {
            grace_ms: 1_500,
            grace_ticks: 3,
            excess_fraction: 0.10,
        };

        assert!(!inventory_brake_grace_allowed(&limits, &brake, config));
    }

    #[test]
    fn inventory_attribution_summarizes_fills_syncs_and_intents() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.venues[3].position_tao = 0.08;
        state.venues[3].mm_open_ask = Some(MmOpenOrder {
            price: 101.0,
            size: 0.01,
            timestamp_ms: 1_000,
            order_id: "mm_ask".to_string(),
        });
        state.venues[4].position_tao = -0.04;

        let intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: 3,
                venue_id: cfg.venues[3].id_arc.clone(),
                side: Side::Sell,
                price: 101.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("co_1".to_string()),
            }),
            OrderIntent::Replace(crate::types::ReplaceOrderIntent {
                venue_index: 4,
                venue_id: cfg.venues[4].id_arc.clone(),
                side: Side::Buy,
                price: 99.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                order_id: "oid_4".to_string(),
                client_order_id: Some("co_4".to_string()),
            }),
            OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index: 2,
                venue_id: cfg.venues[2].id_arc.clone(),
                order_id: "oid_cancel".to_string(),
            }),
        ];
        let exec_events = vec![
            ExecutionEvent::OrderAck(OrderAck {
                venue_index: 3,
                venue_id: cfg.venues[3].id_arc.clone(),
                order_id: "oid_ack".to_string(),
                client_order_id: Some("co_1".to_string()),
                seq: Some(7),
                side: Some(Side::Sell),
                price: Some(101.0),
                size: Some(0.01),
                purpose: Some(OrderPurpose::Mm),
            }),
            ExecutionEvent::OrderReject(OrderReject {
                venue_index: 4,
                venue_id: cfg.venues[4].id_arc.clone(),
                order_id: Some("oid_reject".to_string()),
                client_order_id: Some("co_4".to_string()),
                seq: Some(8),
                purpose: Some(OrderPurpose::Mm),
                reduce_only: Some(false),
                reason: "bad_price".to_string(),
            }),
        ];
        let fills = vec![crate::types::FillEvent {
            venue_index: 3,
            venue_id: cfg.venues[3].id_arc.clone(),
            order_id: Some("oid_fill".to_string()),
            client_order_id: Some("co_1".to_string()),
            seq: Some(9),
            side: Side::Sell,
            price: 101.0,
            size: 0.02,
            purpose: OrderPurpose::Mm,
            fee_bps: 0.0,
        }];
        let account_syncs = vec![AccountPositionSyncRecord {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            snapshot_seq: 11,
            snapshot_timestamp_ms: Some(1_995),
            ingest_now_ms: 2_000,
            pre_position_tao: 0.0,
            post_position_tao: -0.04,
            position_delta_tao: -0.04,
            pre_margin_available_usd: 70.0,
            post_margin_available_usd: 69.0,
            source: "account_snapshot",
        }];

        let attribution = build_inventory_attribution(
            &cfg,
            &state,
            &intents,
            &exec_events,
            &fills,
            &account_syncs,
        );

        let lighter = attribution
            .iter()
            .find(|venue| venue.venue_id == "lighter")
            .expect("lighter attribution");
        assert_eq!(lighter.position_tao, 0.08);
        assert!(lighter.tracked_mm_ask_live);
        assert_eq!(lighter.intent_place_count, 1);
        assert_eq!(lighter.ack_count, 1);
        assert_eq!(lighter.fill_count, 1);
        assert_eq!(lighter.fill_delta_tao, -0.02);

        let paradex = attribution
            .iter()
            .find(|venue| venue.venue_id == "paradex")
            .expect("paradex attribution");
        assert_eq!(paradex.position_tao, -0.04);
        assert_eq!(paradex.intent_replace_count, 1);
        assert_eq!(paradex.reject_count, 1);
        assert_eq!(paradex.account_sync_count, 1);
        assert_eq!(paradex.account_sync_delta_tao, -0.04);

        let aster = attribution
            .iter()
            .find(|venue| venue.venue_id == "aster")
            .expect("aster attribution");
        assert_eq!(aster.intent_cancel_count, 1);
    }

    #[test]
    fn venue_utility_fillless_ack_pressure_resets_on_fill() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let venue_index = 1;

        update_venue_utility_state(
            &cfg,
            &mut state,
            &[],
            &[ExecutionEvent::OrderAck(OrderAck {
                venue_index,
                venue_id: cfg.venues[venue_index].id_arc.clone(),
                order_id: "oid_ack".to_string(),
                client_order_id: Some("co_ack".to_string()),
                seq: Some(1),
                side: Some(Side::Buy),
                price: Some(100.0),
                size: Some(0.01),
                purpose: Some(OrderPurpose::Mm),
            })],
            &[],
            &[],
        );

        assert!(
            state.venues[venue_index].utility.mm_fillless_ack_pressure > 0.0,
            "acknowledged MM churn should accumulate fillless pressure"
        );

        update_venue_utility_state(
            &cfg,
            &mut state,
            &[],
            &[],
            &[crate::types::FillEvent {
                venue_index,
                venue_id: cfg.venues[venue_index].id_arc.clone(),
                order_id: Some("oid_fill".to_string()),
                client_order_id: Some("co_ack".to_string()),
                seq: Some(2),
                side: Side::Buy,
                price: 100.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                fee_bps: 0.0,
            }],
            &[],
        );

        assert_eq!(
            state.venues[venue_index].utility.mm_fillless_ack_pressure, 0.0,
            "real MM fills should reset fillless pressure"
        );
        assert!(
            state.venues[venue_index].utility.mm_fill_credit_ewma > 0.0,
            "real MM fills should still accrue fill credit"
        );
    }

    #[test]
    fn soft_unwind_intents_cancel_mm_and_flatten_positions() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 2_000;
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;

        state.venues[0].position_tao = -0.04;
        state.venues[0].mid = Some(2_350.0);
        state.venues[0].spread = Some(0.2);
        state.venues[2].position_tao = -0.01;
        state.venues[2].mid = Some(2_351.0);
        state.venues[2].spread = Some(0.2);
        state.venues[3].position_tao = 0.09;
        state.venues[3].mid = Some(2_349.5);
        state.venues[3].spread = Some(0.2);
        state.venues[4].position_tao = -0.04;
        state.venues[4].mid = Some(2_350.5);
        state.venues[4].spread = Some(0.4);
        state.venues[1].mm_open_bid = Some(MmOpenOrder {
            price: 2_349.0,
            size: 0.01,
            timestamp_ms: 1_000,
            order_id: "hl_bid".to_string(),
        });
        state.venues[3].mm_open_ask = Some(MmOpenOrder {
            price: 2_350.0,
            size: 0.01,
            timestamp_ms: 1_000,
            order_id: "lighter_ask".to_string(),
        });

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms),
                    position_tao: state.venues[venue_index].position_tao,
                    avg_entry_price: 2_350.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let intents = build_soft_unwind_intents(&cfg, &state, &snapshot, now_ms);
        let cancel_orders = intents
            .iter()
            .filter_map(|intent| match intent {
                OrderIntent::Cancel(cancel) => Some((cancel.venue_index, cancel.order_id.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(cancel_orders, vec![(1, "hl_bid"), (3, "lighter_ask")]);

        let unwind_places = intents
            .iter()
            .filter_map(|intent| match intent {
                OrderIntent::Place(place) => Some(place),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(unwind_places.len(), 4);
        assert!(unwind_places.iter().all(|place| {
            place.purpose == OrderPurpose::Hedge
                && place.reduce_only
                && place.time_in_force == TimeInForce::Ioc
                && !place.post_only
        }));

        let extended = unwind_places
            .iter()
            .find(|place| place.venue_index == 0)
            .expect("extended unwind");
        assert_eq!(extended.side, Side::Buy);
        assert_eq!(extended.size, 0.03);

        let lighter = unwind_places
            .iter()
            .find(|place| place.venue_index == 3)
            .expect("lighter unwind");
        assert_eq!(lighter.side, Side::Sell);
        assert_eq!(lighter.size, 0.08);
    }

    #[test]
    fn soft_unwind_prefers_fresh_account_position_over_internal_state() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 2_000;
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;
        state.venues[2].position_tao = -0.04;
        state.venues[2].mid = Some(2_351.0);
        state.venues[2].spread = Some(0.2);

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms),
                    position_tao: if venue_index == 2 { 0.03 } else { 0.0 },
                    avg_entry_price: 2_350.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let intents = build_soft_unwind_intents(&cfg, &state, &snapshot, now_ms);
        let aster = intents
            .iter()
            .filter_map(|intent| match intent {
                OrderIntent::Place(place) if place.venue_index == 2 => Some(place),
                _ => None,
            })
            .next()
            .expect("aster unwind");
        assert_eq!(aster.side, Side::Sell);
        assert_eq!(aster.size, 0.01);
    }

    #[test]
    fn soft_unwind_falls_back_to_internal_position_when_snapshot_is_stale() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 12_000;
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;
        state.venues[3].position_tao = 0.09;
        state.venues[3].mid = Some(2_349.5);
        state.venues[3].spread = Some(0.2);

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms - 11_000),
                    position_tao: 0.0,
                    avg_entry_price: 2_350.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let lighter = build_soft_unwind_intents(&cfg, &state, &snapshot, now_ms)
            .into_iter()
            .find_map(|intent| match intent {
                OrderIntent::Place(place) if place.venue_index == 3 => Some(place),
                _ => None,
            })
            .expect("lighter unwind from internal fallback");
        assert_eq!(lighter.side, Side::Sell);
        assert_eq!(lighter.size, 0.08);
    }

    #[test]
    fn soft_unwind_skips_aster_when_account_snapshot_is_stale() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 12_000;
        state.fair_value = Some(2_350.0);
        state.fair_value_prev = 2_350.0;
        state.venues[2].position_tao = -0.04;
        state.venues[2].mid = Some(2_351.0);
        state.venues[2].spread = Some(0.2);

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms - 11_000),
                    position_tao: if venue_index == 2 { -0.04 } else { 0.0 },
                    avg_entry_price: 2_350.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let aster = build_soft_unwind_intents(&cfg, &state, &snapshot, now_ms)
            .into_iter()
            .find_map(|intent| match intent {
                OrderIntent::Place(place) if place.venue_index == 2 => Some(place),
                _ => None,
            });
        assert!(aster.is_none());
    }

    #[test]
    fn inventory_attribution_treats_aster_reduce_only_hedge_reject_as_benign() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let aster = 2;
        let exec_events = vec![ExecutionEvent::OrderReject(OrderReject {
            venue_index: aster,
            venue_id: cfg.venues[aster].id_arc.clone(),
            order_id: Some("co_aster_hedge".to_string()),
            client_order_id: Some("co_aster_hedge".to_string()),
            seq: Some(1),
            purpose: Some(OrderPurpose::Hedge),
            reduce_only: Some(true),
            reason: "{\"code\":-2022,\"msg\":\"ReduceOnly Order is rejected.\"}".to_string(),
        })];

        let attribution = build_inventory_attribution(&cfg, &state, &[], &exec_events, &[], &[]);
        let aster_attr = attribution
            .iter()
            .find(|venue| venue.venue_id == "aster")
            .expect("aster attribution");
        assert_eq!(aster_attr.reject_count, 1);
        assert_eq!(aster_attr.mm_reject_count, 0);
        assert_eq!(aster_attr.benign_reduce_only_reject_count, 1);
    }

    #[test]
    fn startup_pnl_baseline_waits_for_fresh_accounts() {
        let cfg = Config::default();
        let state = GlobalState::new(&cfg);
        let now_ms = 5_000;
        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 0,
                    timestamp_ms: None,
                    position_tao: 0.0,
                    avg_entry_price: 0.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: true,
                })
                .collect(),
        };

        let status = evaluate_startup_pnl_baseline(
            &cfg,
            &state,
            &snapshot,
            &vec![false; cfg.venues.len()],
            now_ms,
            0,
            StartupPnlBaselineConfig {
                enabled: true,
                pnl_abs_limit_usd: 1.0,
                position_tol_tao: 0.0025,
                max_wait_ticks: 40,
            },
        );

        assert!(status.enabled);
        assert!(status.waiting_for_accounts);
        assert!(!status.triggered);
        assert_eq!(status.fresh_account_count, 0);
        assert_eq!(status.required_account_count, cfg.venues.len());
    }

    #[test]
    fn startup_pnl_baseline_flags_inherited_pnl_from_small_aster_carry() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 5_000;
        state.fair_value = Some(2_095.0);
        state.fair_value_prev = 2_095.0;
        state.venues[2].position_tao = 0.01;
        state.venues[2].avg_entry_price = 3_960.35;
        state.recompute_after_fills(&cfg);

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms),
                    position_tao: if venue_index == 2 { 0.01 } else { 0.0 },
                    avg_entry_price: if venue_index == 2 { 3_960.35 } else { 0.0 },
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let status = evaluate_startup_pnl_baseline(
            &cfg,
            &state,
            &snapshot,
            &vec![true; cfg.venues.len()],
            now_ms,
            0,
            StartupPnlBaselineConfig {
                enabled: true,
                pnl_abs_limit_usd: 1.0,
                position_tol_tao: 0.0025,
                max_wait_ticks: 40,
            },
        );

        assert!(status.triggered);
        assert_eq!(
            status.reason.as_deref(),
            Some("startup_inherited_pnl_breach")
        );
        assert!((status.daily_pnl_total + 18.6535).abs() < 1e-6);
        assert_eq!(status.violating_venues.len(), 1);
        assert_eq!(status.violating_venues[0].venue_id, "aster");
    }

    #[test]
    fn startup_pnl_baseline_passes_clean_flat_state() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 5_000;
        state.fair_value = Some(2_095.0);
        state.fair_value_prev = 2_095.0;
        state.recompute_after_fills(&cfg);
        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: 1,
                    timestamp_ms: Some(now_ms),
                    position_tao: 0.0,
                    avg_entry_price: 0.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let status = evaluate_startup_pnl_baseline(
            &cfg,
            &state,
            &snapshot,
            &vec![true; cfg.venues.len()],
            now_ms,
            0,
            StartupPnlBaselineConfig {
                enabled: true,
                pnl_abs_limit_usd: 1.0,
                position_tol_tao: 0.0025,
                max_wait_ticks: 40,
            },
        );

        assert!(status.passed);
        assert!(status.resolved);
        assert!(!status.triggered);
    }

    #[test]
    fn startup_pnl_baseline_allows_partial_account_coverage_after_timeout() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 5_000;
        state.fair_value = Some(2_095.0);
        state.fair_value_prev = 2_095.0;
        state.recompute_after_fills(&cfg);
        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: cfg
                .venues
                .iter()
                .enumerate()
                .map(|(venue_index, venue)| VenueAccountSnapshot {
                    venue_index,
                    venue_id: venue.id_arc.clone(),
                    seq: if venue_index < 2 { 1 } else { 0 },
                    timestamp_ms: if venue_index < 2 { Some(now_ms) } else { None },
                    position_tao: 0.0,
                    avg_entry_price: 0.0,
                    funding_8h: None,
                    margin_balance_usd: 100.0,
                    margin_used_usd: 0.0,
                    margin_available_usd: 100.0,
                    price_liq: None,
                    dist_liq_sigma: None,
                    is_stale: false,
                })
                .collect(),
        };

        let mut initialized = vec![false; cfg.venues.len()];
        initialized[0] = true;
        initialized[1] = true;
        let status = evaluate_startup_pnl_baseline(
            &cfg,
            &state,
            &snapshot,
            &initialized,
            now_ms,
            40,
            StartupPnlBaselineConfig {
                enabled: true,
                pnl_abs_limit_usd: 1.0,
                position_tol_tao: 0.0025,
                max_wait_ticks: 40,
            },
        );

        assert!(status.passed);
        assert!(status.resolved);
        assert!(!status.triggered);
        assert_eq!(
            status.reason.as_deref(),
            Some("startup_baseline_partial_account_coverage_ok")
        );
    }

    #[test]
    fn account_snapshot_apply_respects_lighter_poll_override() {
        let _guard = ENV_MUTEX
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "PARAPHINA_LIVE_ACCOUNT_POLL_MS",
            "PARAPHINA_LIGHTER_ACCOUNT_POLL_MS",
            "LIGHTER_ACCOUNT_POLL_MS",
        ]);
        std::env::set_var("PARAPHINA_LIVE_ACCOUNT_POLL_MS", "1000");
        std::env::set_var("PARAPHINA_LIGHTER_ACCOUNT_POLL_MS", "3000");

        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let now_ms = 5_000;
        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market: Vec::new(),
            account: vec![VenueAccountSnapshot {
                venue_index: 3,
                venue_id: "lighter".into(),
                seq: 1,
                timestamp_ms: Some(now_ms - 2_500),
                position_tao: 0.12,
                avg_entry_price: 2_350.0,
                funding_8h: None,
                margin_balance_usd: 100.0,
                margin_used_usd: 10.0,
                margin_available_usd: 90.0,
                price_liq: None,
                dist_liq_sigma: None,
                is_stale: false,
            }],
        };

        let applied = apply_account_snapshot_to_state(&cfg, &snapshot, &mut state, now_ms);
        assert!(applied.position_changed);
        assert_eq!(state.venues[3].position_tao, 0.12);
        assert_eq!(applied.position_syncs.len(), 1);
    }

    #[test]
    fn soft_unwind_runtime_state_holds_quotes_flat_during_cooldown() {
        let state = soft_unwind_runtime_state(false, true, false, false, false);
        assert!(state.pause_mm_quotes);
        assert!(!state.send_unwind);
        assert!(!state.refresh_cooldown);

        let state = soft_unwind_runtime_state(true, false, false, true, true);
        assert!(state.pause_mm_quotes);
        assert!(state.send_unwind);
        assert!(state.refresh_cooldown);
    }

    #[test]
    fn soft_unwind_runtime_state_pauses_without_resend_during_response_backoff() {
        let state = soft_unwind_runtime_state(true, false, true, true, true);
        assert!(state.pause_mm_quotes);
        assert!(!state.send_unwind);
        assert!(state.refresh_cooldown);
    }

    #[test]
    fn reserve_priority_path_for_inventory_control_blocks_on_pause_or_brake() {
        let clear = SoftUnwindRuntimeState {
            pause_mm_quotes: false,
            send_unwind: false,
            refresh_cooldown: false,
        };
        assert!(!reserve_priority_path_for_inventory_control(clear, false));
        assert!(reserve_priority_path_for_inventory_control(clear, true));

        let paused = SoftUnwindRuntimeState {
            pause_mm_quotes: true,
            send_unwind: false,
            refresh_cooldown: true,
        };
        assert!(reserve_priority_path_for_inventory_control(paused, false));
        assert!(reserve_priority_path_for_inventory_control(paused, true));
    }

    fn seed_open_orders_with_side(
        state: &mut GlobalState,
        venue_index: usize,
        count: usize,
        side: Side,
        now_ms: TimestampMs,
    ) {
        let venue_id = state.venues[venue_index].id.to_string();
        let snapshot = types::OrderSnapshot {
            venue_index,
            venue_id,
            seq: now_ms as u64,
            timestamp_ms: now_ms,
            open_orders: (0..count)
                .map(|idx| types::OpenOrderSnapshot {
                    order_id: format!("oid_{venue_index}_{idx}"),
                    client_order_id: Some(format!("co_{venue_index}_{idx}")),
                    side,
                    price: 2_300.0 + idx as f64,
                    size: 0.01,
                    purpose: Some(OrderPurpose::Mm),
                })
                .collect(),
        };
        state.live_order_state.reconcile(&snapshot, now_ms);
    }

    fn seed_open_orders(
        state: &mut GlobalState,
        venue_index: usize,
        count: usize,
        now_ms: TimestampMs,
    ) {
        seed_open_orders_with_side(state, venue_index, count, Side::Buy, now_ms);
    }

    #[test]
    fn lighter_soft_unwind_latch_clears_on_position_progress() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mut latches = EmergencyRequestLatchSet::default();
        let lighter = 3;
        let now_ms = 10_000;

        state.venues[lighter].position_tao = 0.02;
        latch_emergency_request(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            lighter,
            now_ms,
            3_000,
            9_000,
        );
        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 100,
            lighter,
        ));

        state.venues[lighter].position_tao = 0.01;
        assert!(!emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 200,
            lighter,
        ));
        assert!(latches.soft_unwind.is_empty());
    }

    #[test]
    fn lighter_disabled_cancel_all_latch_clears_on_open_order_progress() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mut latches = EmergencyRequestLatchSet::default();
        let lighter = 3;
        let now_ms = 20_000;

        seed_open_orders(&mut state, lighter, 2, now_ms);
        latch_emergency_request(
            &mut latches,
            EmergencyRequestClass::DisabledCancelAll,
            &cfg,
            &state,
            lighter,
            now_ms,
            3_000,
            9_000,
        );
        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::DisabledCancelAll,
            &cfg,
            &state,
            now_ms + 100,
            lighter,
        ));

        seed_open_orders(&mut state, lighter, 1, now_ms + 200);
        assert!(!emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::DisabledCancelAll,
            &cfg,
            &state,
            now_ms + 300,
            lighter,
        ));
        assert!(latches.disabled_cancel_all.is_empty());
    }

    #[test]
    fn remove_intents_for_venue_drops_only_latched_lighter_orders() {
        let cfg = Config::default();
        let lighter = 3;
        let aster = 2;
        let mut intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: lighter,
                venue_id: cfg.venues[lighter].id_arc.clone(),
                side: Side::Buy,
                price: 2_300.0,
                size: 0.01,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("co_lighter".to_string()),
            }),
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: aster,
                venue_id: cfg.venues[aster].id_arc.clone(),
                side: Side::Buy,
                price: 2_300.0,
                size: 0.01,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("co_aster".to_string()),
            }),
        ];

        assert!(intents_touch_venue(&intents, lighter));
        assert_eq!(remove_intents_for_venue(&mut intents, lighter), 1);
        assert_eq!(intents.len(), 1);
        assert_eq!(venue_index_for_order_intent(&intents[0]), Some(aster));
    }

    #[test]
    fn intents_for_venue_collects_only_target_orders() {
        let cfg = Config::default();
        let lighter = 3;
        let aster = 2;
        let intents = vec![
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: lighter,
                venue_id: cfg.venues[lighter].id_arc.clone(),
                side: Side::Buy,
                price: 2_300.0,
                size: 0.01,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("co_lighter".to_string()),
            }),
            OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: aster,
                venue_id: cfg.venues[aster].id_arc.clone(),
                side: Side::Sell,
                price: 2_301.0,
                size: 0.02,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("co_aster".to_string()),
            }),
        ];

        let lighter_only = intents_for_venue(&intents, lighter);
        assert_eq!(lighter_only.len(), 1);
        assert_eq!(
            venue_index_for_order_intent(&lighter_only[0]),
            Some(lighter)
        );
    }

    #[test]
    fn lighter_soft_unwind_latch_extends_when_no_progress() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mut latches = EmergencyRequestLatchSet::default();
        let lighter = 3;
        let now_ms = 30_000;

        state.venues[lighter].position_tao = 0.02;
        seed_open_orders(&mut state, lighter, 2, now_ms);
        latch_emergency_request(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            lighter,
            now_ms,
            3_000,
            9_000,
        );

        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 3_100,
            lighter,
        ));
        let latch = latches.soft_unwind.first().expect("extended latch");
        assert_eq!(latch.extension_count, 1);
        assert_eq!(latch.expires_at_ms, now_ms + 6_100);
        assert_eq!(latch.max_expires_at_ms, now_ms + 9_000);
    }

    #[test]
    fn lighter_soft_unwind_latch_clears_after_bounded_max_age() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mut latches = EmergencyRequestLatchSet::default();
        let lighter = 3;
        let now_ms = 40_000;

        state.venues[lighter].position_tao = 0.02;
        seed_open_orders(&mut state, lighter, 2, now_ms);
        latch_emergency_request(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            lighter,
            now_ms,
            3_000,
            9_000,
        );

        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 3_100,
            lighter,
        ));
        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 6_200,
            lighter,
        ));
        assert!(!emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 9_000,
            lighter,
        ));
        assert!(latches.soft_unwind.is_empty());
    }

    #[test]
    fn emergency_latches_track_multiple_venues_per_class() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mut latches = EmergencyRequestLatchSet::default();
        let aster = 2;
        let lighter = 3;
        let now_ms = 50_000;

        state.venues[aster].position_tao = 0.02;
        state.venues[lighter].position_tao = 0.03;
        latch_emergency_request(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            aster,
            now_ms,
            3_000,
            9_000,
        );
        latch_emergency_request(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            lighter,
            now_ms,
            3_000,
            9_000,
        );

        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 100,
            aster,
        ));
        assert!(emergency_request_latched(
            &mut latches,
            EmergencyRequestClass::SoftUnwind,
            &cfg,
            &state,
            now_ms + 100,
            lighter,
        ));
        assert_eq!(latches.soft_unwind.len(), 2);
    }

    #[test]
    fn emergency_single_flight_targets_fill_venues() {
        assert!(emergency_single_flight_enabled_for_venue_id("lighter"));
        assert!(emergency_single_flight_enabled_for_venue_id("LIGHTER"));
        assert!(emergency_single_flight_enabled_for_venue_id("aster"));
        assert!(emergency_single_flight_enabled_for_venue_id("paradex"));
        assert!(!emergency_single_flight_enabled_for_venue_id("hyperliquid"));
        assert!(!emergency_single_flight_enabled_for_venue_id("extended"));
    }

    #[test]
    fn reconcile_diff_at_tolerance_boundary_does_not_exceed() {
        assert!(!diff_exceeds(-0.03, -0.04, 0.01));
        assert!(!diff_exceeds(0.03, 0.04, 0.01));
    }

    #[test]
    fn kill_cancel_all_timeout_scales_with_venue_count() {
        assert_eq!(
            kill_cancel_all_timeout_ms(0),
            KILL_CANCEL_ALL_TIMEOUT_MS_MIN
        );
        assert_eq!(
            kill_cancel_all_timeout_ms(3),
            KILL_CANCEL_ALL_TIMEOUT_MS_MIN
        );
        assert_eq!(kill_cancel_all_timeout_ms(5), 5_000);
        assert_eq!(kill_cancel_all_timeout_ms(7), 7_000);
    }
}
