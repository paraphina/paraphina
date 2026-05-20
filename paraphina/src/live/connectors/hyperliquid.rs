//! Hyperliquid connector (feature-gated).
//!
//! Uses official Hyperliquid WS/REST endpoints and subscription types.

pub const STUB_CONNECTOR: bool = false;
pub const SUPPORTS_MARKET: bool = true;
pub const SUPPORTS_ACCOUNT: bool = true;
pub const SUPPORTS_EXECUTION: bool = true;

const HL_STALE_MS_DEFAULT: u64 = 10_000;
const HL_WATCHDOG_TICK_MS: u64 = 200;
const HL_SNAPSHOT_COOLDOWN_MS: u64 = 8_000;
const HL_INTERNAL_PUB_Q: usize = 256;
const HL_DELTA_BOOTSTRAP_BUF: usize = 1024;
const HL_DECODE_WARN_INTERVAL_MS: u64 = 10_000;
const HL_WS_AUDIT_INTERVAL_MS: u64 = 1_000;
const HL_TS_MAX_PAST_SKEW_MS_DEFAULT: u64 = 1_500;
const HL_TS_MAX_FUTURE_SKEW_MS_DEFAULT: u64 = 250;

/// Maximum time to wait for WS connect (TCP + TLS + upgrade).
/// Prevents `connect_async` from hanging indefinitely on unresponsive hosts.
/// Evidence: obs_5000_ticks_report — 76-min dead period, no reconnect logs.
const HL_WS_CONNECT_TIMEOUT_MS_DEFAULT: u64 = 15_000;

/// Maximum time to wait for a single WS frame before treating connection as dead.
/// Prevents idle ESTABLISHED sockets from blocking the reconnect loop.
const HL_WS_READ_TIMEOUT_MS_DEFAULT: u64 = 30_000;
const HL_WS_POST_RESPONSE_TIMEOUT_MS_DEFAULT: u64 = 15_000;
const HL_WS_POST_CHANNEL_CAPACITY: usize = 256;
const HL_WS_POST_MAX_INFLIGHT_DEFAULT: usize = 32;

static MONO_START: OnceLock<Instant> = OnceLock::new();
static HL_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static HL_RECONNECT_COUNTS: OnceLock<StdMutex<BTreeMap<&'static str, u64>>> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn hl_stale_ms() -> u64 {
    std::env::var("PARAPHINA_HL_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(HL_STALE_MS_DEFAULT)
}

fn hl_internal_pub_q() -> usize {
    std::env::var("PARAPHINA_HL_INTERNAL_PUB_Q")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(HL_INTERNAL_PUB_Q)
}

fn hl_ws_connect_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_HL_WS_CONNECT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(HL_WS_CONNECT_TIMEOUT_MS_DEFAULT),
    )
}

fn hl_ws_read_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_HL_WS_READ_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(HL_WS_READ_TIMEOUT_MS_DEFAULT),
    )
}

fn hl_ws_post_response_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_HL_WS_POST_RESPONSE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(HL_WS_POST_RESPONSE_TIMEOUT_MS_DEFAULT),
    )
}

fn hl_ws_post_max_inflight() -> usize {
    std::env::var("PARAPHINA_HL_WS_POST_MAX_INFLIGHT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(HL_WS_POST_MAX_INFLIGHT_DEFAULT)
}

fn env_bool(var: &str) -> bool {
    std::env::var(var)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn env_bool_default(var: &str, default: bool) -> bool {
    std::env::var(var)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn hl_sync_control_http_fallback_enabled() -> bool {
    env_bool_default("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED", true)
}

fn hl_cancel_all_http_fallback_enabled() -> bool {
    env_bool_default("PARAPHINA_HL_CANCEL_ALL_HTTP_FALLBACK_ENABLED", true)
}

fn hl_ws_audit_enabled() -> bool {
    *HL_WS_AUDIT_ENABLED.get_or_init(|| env_bool("PARAPHINA_WS_AUDIT"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperliquidActionTransport {
    Http,
    WsPost,
}

impl HyperliquidActionTransport {
    fn from_env() -> Self {
        match std::env::var("PARAPHINA_HL_ACTION_TRANSPORT")
            .unwrap_or_else(|_| "http".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ws_post" | "ws-post" | "ws" => Self::WsPost,
            _ => Self::Http,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::WsPost => "ws_post",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HlTimestampPolicy {
    enabled: bool,
    max_past_skew_ms: u64,
    max_future_skew_ms: u64,
}

impl HlTimestampPolicy {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("PARAPHINA_HL_TS_HARDENING"),
            max_past_skew_ms: std::env::var("PARAPHINA_HL_TS_MAX_PAST_SKEW_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(HL_TS_MAX_PAST_SKEW_MS_DEFAULT),
            max_future_skew_ms: std::env::var("PARAPHINA_HL_TS_MAX_FUTURE_SKEW_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(HL_TS_MAX_FUTURE_SKEW_MS_DEFAULT),
        }
    }
}

fn hl_audit_reconnect(reason: &'static str) {
    if !hl_ws_audit_enabled() {
        return;
    }
    let mut counts = HL_RECONNECT_COUNTS
        .get_or_init(|| StdMutex::new(BTreeMap::new()))
        .lock()
        .expect("hyperliquid reconnect audit mutex poisoned");
    let count = counts
        .entry(reason)
        .and_modify(|value| *value += 1)
        .or_insert(1);
    eprintln!(
        "WS_AUDIT venue=hyperliquid reconnect_reason={} count={}",
        reason, *count
    );
}

/// Wrap a future with a timeout, returning an anyhow error on expiration.
/// Testable helper — used for both connect and read timeouts.
async fn with_timeout<T>(
    duration: Duration,
    label: &str,
    fut: impl std::future::Future<Output = T>,
) -> anyhow::Result<T> {
    tokio::time::timeout(duration, fut)
        .await
        .map_err(|_| anyhow::anyhow!("Hyperliquid {label} timed out after {duration:?}"))
}

fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

fn atomic_max_u64(cell: &AtomicU64, value: u64) {
    let mut observed = cell.load(Ordering::Relaxed);
    while value > observed {
        match cell.compare_exchange_weak(observed, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => observed = actual,
        }
    }
}

#[derive(Debug, Default)]
struct HlForwardAudit {
    send_block_max_ms: AtomicU64,
    send_block_gt_5ms: AtomicU64,
    send_block_gt_50ms: AtomicU64,
    send_block_gt_250ms: AtomicU64,
    forward_send_count: AtomicU64,
    forward_send_err_count: AtomicU64,
    coalesced_drop_count: AtomicU64,
    pending_take_count: AtomicU64,
    ts_missing_or_zero_count: AtomicU64,
    ts_clamped_past_skew_count: AtomicU64,
    ts_clamped_future_skew_count: AtomicU64,
    ts_policy_applied_count: AtomicU64,
    ts_kept_exchange_count: AtomicU64,
    ts_past_skew_max_ms: AtomicU64,
    ts_future_skew_max_ms: AtomicU64,
}

#[derive(Debug, Default, Clone, Copy)]
struct HlForwardAuditSnapshot {
    send_block_max_ms: u64,
    send_block_gt_5ms: u64,
    send_block_gt_50ms: u64,
    send_block_gt_250ms: u64,
    forward_send_count: u64,
    forward_send_err_count: u64,
    coalesced_drop_count: u64,
    pending_take_count: u64,
    ts_missing_or_zero_count: u64,
    ts_clamped_past_skew_count: u64,
    ts_clamped_future_skew_count: u64,
    ts_policy_applied_count: u64,
    ts_kept_exchange_count: u64,
    ts_past_skew_max_ms: u64,
    ts_future_skew_max_ms: u64,
}

impl HlForwardAudit {
    fn observe_send_block_ms(&self, block_ms: u64) {
        self.forward_send_count.fetch_add(1, Ordering::Relaxed);
        atomic_max_u64(&self.send_block_max_ms, block_ms);
        if block_ms > 5 {
            self.send_block_gt_5ms.fetch_add(1, Ordering::Relaxed);
        }
        if block_ms > 50 {
            self.send_block_gt_50ms.fetch_add(1, Ordering::Relaxed);
        }
        if block_ms > 250 {
            self.send_block_gt_250ms.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn observe_send_err(&self) {
        self.forward_send_err_count.fetch_add(1, Ordering::Relaxed);
    }

    fn observe_coalesced_drop(&self, count: u64) {
        if count > 0 {
            self.coalesced_drop_count
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    fn observe_pending_take(&self) {
        self.pending_take_count.fetch_add(1, Ordering::Relaxed);
    }

    fn observe_ts_missing_or_zero(&self) {
        self.ts_missing_or_zero_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn observe_ts_clamped_past_skew(&self) {
        self.ts_clamped_past_skew_count
            .fetch_add(1, Ordering::Relaxed);
        self.ts_policy_applied_count.fetch_add(1, Ordering::Relaxed);
    }

    fn observe_ts_clamped_future_skew(&self) {
        self.ts_clamped_future_skew_count
            .fetch_add(1, Ordering::Relaxed);
        self.ts_policy_applied_count.fetch_add(1, Ordering::Relaxed);
    }

    fn observe_ts_kept_exchange(&self) {
        self.ts_kept_exchange_count.fetch_add(1, Ordering::Relaxed);
    }

    fn observe_ts_policy_applied(&self) {
        self.ts_policy_applied_count.fetch_add(1, Ordering::Relaxed);
    }

    fn observe_ts_skew(&self, past_skew_ms: u64, future_skew_ms: u64) {
        atomic_max_u64(&self.ts_past_skew_max_ms, past_skew_ms);
        atomic_max_u64(&self.ts_future_skew_max_ms, future_skew_ms);
    }

    fn snapshot_and_reset(&self) -> HlForwardAuditSnapshot {
        HlForwardAuditSnapshot {
            send_block_max_ms: self.send_block_max_ms.swap(0, Ordering::Relaxed),
            send_block_gt_5ms: self.send_block_gt_5ms.swap(0, Ordering::Relaxed),
            send_block_gt_50ms: self.send_block_gt_50ms.swap(0, Ordering::Relaxed),
            send_block_gt_250ms: self.send_block_gt_250ms.swap(0, Ordering::Relaxed),
            forward_send_count: self.forward_send_count.swap(0, Ordering::Relaxed),
            forward_send_err_count: self.forward_send_err_count.swap(0, Ordering::Relaxed),
            coalesced_drop_count: self.coalesced_drop_count.swap(0, Ordering::Relaxed),
            pending_take_count: self.pending_take_count.swap(0, Ordering::Relaxed),
            ts_missing_or_zero_count: self.ts_missing_or_zero_count.swap(0, Ordering::Relaxed),
            ts_clamped_past_skew_count: self.ts_clamped_past_skew_count.swap(0, Ordering::Relaxed),
            ts_clamped_future_skew_count: self
                .ts_clamped_future_skew_count
                .swap(0, Ordering::Relaxed),
            ts_policy_applied_count: self.ts_policy_applied_count.swap(0, Ordering::Relaxed),
            ts_kept_exchange_count: self.ts_kept_exchange_count.swap(0, Ordering::Relaxed),
            ts_past_skew_max_ms: self.ts_past_skew_max_ms.swap(0, Ordering::Relaxed),
            ts_future_skew_max_ms: self.ts_future_skew_max_ms.swap(0, Ordering::Relaxed),
        }
    }
}

fn apply_hl_l2_timestamp_policy(
    exchange_ts_ms: TimestampMs,
    rx_now_ms: TimestampMs,
    policy: HlTimestampPolicy,
    audit: &HlForwardAudit,
) -> TimestampMs {
    if exchange_ts_ms <= 0 {
        audit.observe_ts_missing_or_zero();
        if policy.enabled {
            audit.observe_ts_policy_applied();
            return rx_now_ms;
        }
        return exchange_ts_ms;
    }

    let past_skew_ms = if rx_now_ms > exchange_ts_ms {
        (rx_now_ms - exchange_ts_ms) as u64
    } else {
        0
    };
    let future_skew_ms = if exchange_ts_ms > rx_now_ms {
        (exchange_ts_ms - rx_now_ms) as u64
    } else {
        0
    };
    audit.observe_ts_skew(past_skew_ms, future_skew_ms);

    if policy.enabled {
        if past_skew_ms > policy.max_past_skew_ms {
            audit.observe_ts_clamped_past_skew();
            return rx_now_ms;
        }
        if future_skew_ms > policy.max_future_skew_ms {
            audit.observe_ts_clamped_future_skew();
            return rx_now_ms;
        }
        audit.observe_ts_kept_exchange();
    }

    exchange_ts_ms
}

fn apply_hl_l2_event_ts_policy(
    event: MarketDataEvent,
    rx_now_ms: TimestampMs,
    policy: HlTimestampPolicy,
    audit: &HlForwardAudit,
) -> MarketDataEvent {
    match event {
        MarketDataEvent::L2Snapshot(mut snapshot) => {
            snapshot.timestamp_ms =
                apply_hl_l2_timestamp_policy(snapshot.timestamp_ms, rx_now_ms, policy, audit);
            MarketDataEvent::L2Snapshot(snapshot)
        }
        MarketDataEvent::L2Delta(mut delta) => {
            delta.timestamp_ms =
                apply_hl_l2_timestamp_policy(delta.timestamp_ms, rx_now_ms, policy, audit);
            MarketDataEvent::L2Delta(delta)
        }
        other => other,
    }
}

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_parsed_ns: AtomicU64,
    last_published_ns: AtomicU64,
    last_snapshot_resync_ns: AtomicU64,
    /// FIX A3: Tracks the last time an l2Book event was decoded into a publishable
    /// MarketDataEvent. Used by the watchdog to detect "WS alive but no book data"
    /// scenarios where heartbeats keep last_ws_rx_ns fresh but no book updates flow.
    last_book_event_ns: AtomicU64,
}

impl Freshness {
    fn reset_for_new_connection(&self) {
        self.last_ws_rx_ns.store(0, Ordering::Relaxed);
        self.last_data_rx_ns.store(0, Ordering::Relaxed);
        self.last_parsed_ns.store(0, Ordering::Relaxed);
        self.last_published_ns.store(0, Ordering::Relaxed);
        self.last_snapshot_resync_ns.store(0, Ordering::Relaxed);
        self.last_book_event_ns.store(0, Ordering::Relaxed);
    }

    fn anchor_with_connect_start(&self, connect_start_ns: u64) -> u64 {
        // FIX A3: Use last_book_event_ns as the primary watchdog anchor.
        // This ensures the watchdog fires when book data stops flowing,
        // even if non-book WS messages (heartbeats) keep last_ws_rx_ns fresh.
        let last_book = self.last_book_event_ns.load(Ordering::Relaxed);
        let last_pub = self.last_published_ns.load(Ordering::Relaxed);
        let anchor = last_book.max(last_pub);
        if anchor == 0 {
            connect_start_ns
        } else {
            anchor
        }
    }
}

#[derive(Debug)]
struct HyperliquidPostRequest {
    id: u64,
    action_label: &'static str,
    batch_kind: &'static str,
    batch_size: usize,
    payload: serde_json::Value,
    response_tx: oneshot::Sender<anyhow::Result<()>>,
}

#[derive(Debug)]
struct HyperliquidPostPending {
    action_label: &'static str,
    batch_kind: &'static str,
    batch_size: usize,
    sent_at: Instant,
    response_tx: oneshot::Sender<anyhow::Result<()>>,
}

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex as StdMutex, OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use k256::ecdsa::{RecoveryId, Signature, SigningKey};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha3::{Digest, Keccak256};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::types::{
    FundingSource, OrderIntent, OrderPurpose, SettlementPriceKind, Side, TimeInForce, TimestampMs,
};

use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, FundingUpdate,
    LiquidationSnapshot, MarginSnapshot, MarketDataEvent, Phase51ForwardRefreshNativeRole,
    PositionSnapshot, TopOfBook,
};
use crate::live::gateway::{
    BoxFuture, LiveGatewayError, LiveRestCancelAllRequest, LiveRestCancelRequest, LiveRestClient,
    LiveRestPlaceRequest, LiveRestReplaceRequest, LiveRestResponse, LiveResult, TransportHint,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperliquidNetwork {
    Mainnet,
    Testnet,
}

#[derive(Debug, Clone)]
pub struct HyperliquidConfig {
    pub network: HyperliquidNetwork,
    /// Ordered list of WebSocket URLs for failover rotation.
    pub ws_urls: Vec<String>,
    /// Ordered list of REST (exchange) URLs for failover rotation.
    pub rest_urls: Vec<String>,
    /// Ordered list of REST info URLs for failover rotation.
    pub info_urls: Vec<String>,
    pub coin: String,
    pub n_sig_figs: u32,
    pub n_levels: u32,
    pub venue_index: usize,
    pub paper_mode: bool,
    pub private_key_hex: Option<String>,
    pub vault_address: Option<String>,
}

impl HyperliquidConfig {
    /// Return the current ws_url (first entry or default).
    pub fn ws_url(&self) -> &str {
        self.ws_urls.first().map(|s| s.as_str()).unwrap_or("")
    }
    /// Return the current rest_url (first entry or default).
    pub fn rest_url(&self) -> &str {
        self.rest_urls.first().map(|s| s.as_str()).unwrap_or("")
    }
    /// Return the current info_url (first entry or default).
    pub fn info_url(&self) -> &str {
        self.info_urls.first().map(|s| s.as_str()).unwrap_or("")
    }
}

impl HyperliquidConfig {
    /// Parse a comma-separated list of URLs from the given env var, or fall back
    /// to the singular env var wrapped in a vec, or the default.
    fn urls_from_env(plural_var: &str, singular_var: &str, default: &str) -> Vec<String> {
        // Prefer the plural form (comma-separated list) if set.
        if let Ok(raw) = std::env::var(plural_var) {
            let urls: Vec<String> = raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !urls.is_empty() {
                return urls;
            }
        }
        // Fall back to the singular form for backward compatibility.
        if let Ok(single) = std::env::var(singular_var) {
            if !single.trim().is_empty() {
                return vec![single.trim().to_string()];
            }
        }
        vec![default.to_string()]
    }
}

impl HyperliquidConfig {
    pub fn from_env() -> Self {
        let network = match std::env::var("HL_NETWORK")
            .unwrap_or_else(|_| "mainnet".to_string())
            .to_lowercase()
            .as_str()
        {
            "testnet" => HyperliquidNetwork::Testnet,
            _ => HyperliquidNetwork::Mainnet,
        };
        let (default_ws, default_rest, default_info) = match network {
            HyperliquidNetwork::Mainnet => (
                "wss://api.hyperliquid.xyz/ws",
                "https://api.hyperliquid.xyz/exchange",
                "https://api.hyperliquid.xyz/info",
            ),
            HyperliquidNetwork::Testnet => (
                "wss://api.hyperliquid-testnet.xyz/ws",
                "https://api.hyperliquid-testnet.xyz/exchange",
                "https://api.hyperliquid-testnet.xyz/info",
            ),
        };
        let ws_urls = Self::urls_from_env("HL_WS_URLS", "HL_WS_URL", default_ws);
        let rest_urls = Self::urls_from_env("HL_REST_URLS", "HL_REST_URL", default_rest);
        let info_urls = Self::urls_from_env("HL_INFO_URLS", "HL_INFO_URL", default_info);
        let coin = std::env::var("HL_COIN").unwrap_or_else(|_| "TAO".to_string());
        let n_sig_figs = std::env::var("HL_L2_SIGFIGS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(5);
        let n_levels = std::env::var("HL_L2_LEVELS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(20);
        let paper_mode = std::env::var("HL_PAPER_MODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let private_key_hex = std::env::var("HL_PRIVATE_KEY").ok();
        let vault_address = std::env::var("HL_VAULT_ADDRESS").ok();
        Self {
            network,
            ws_urls,
            rest_urls,
            info_urls,
            coin,
            n_sig_figs,
            n_levels,
            venue_index: 0,
            paper_mode,
            private_key_hex,
            vault_address,
        }
    }
}

#[derive(Debug)]
pub struct HyperliquidConnector {
    cfg: HyperliquidConfig,
    http: Client,
    market_tx: mpsc::Sender<MarketDataEvent>,
    exec_tx: mpsc::Sender<ExecutionEvent>,
    account_tx: Option<mpsc::Sender<AccountEvent>>,
    asset_meta: tokio::sync::Mutex<Option<HyperliquidAssetMeta>>,
    account_role: tokio::sync::Mutex<Option<HyperliquidUserRole>>,
    account_abstraction: tokio::sync::Mutex<Option<HyperliquidAccountAbstraction>>,
    freshness: Arc<Freshness>,
    /// Cached signing key parsed once at initialization (avoids per-call hex decode + key construction).
    signing_key: Option<SigningKey>,
    /// Current endpoint index for round-robin rotation across ws_urls/rest_urls/info_urls.
    endpoint_index: std::sync::atomic::AtomicUsize,
    action_transport: HyperliquidActionTransport,
    action_nonce: AtomicU64,
    post_request_seq: AtomicU64,
    post_inflight: AtomicU64,
    post_request_tx: mpsc::Sender<HyperliquidPostRequest>,
    post_request_rx: tokio::sync::Mutex<mpsc::Receiver<HyperliquidPostRequest>>,
}

impl HyperliquidConnector {
    pub fn new(
        cfg: HyperliquidConfig,
        market_tx: mpsc::Sender<MarketDataEvent>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
    ) -> Self {
        let action_transport = HyperliquidActionTransport::from_env();
        let (post_request_tx, post_request_rx) = mpsc::channel(HL_WS_POST_CHANNEL_CAPACITY);
        let signing_key = cfg.private_key_hex.as_ref().and_then(|key_hex| {
            let trimmed = key_hex.trim_start_matches("0x");
            match hex::decode(trimmed) {
                Ok(key_bytes) => match SigningKey::from_slice(&key_bytes) {
                    Ok(sk) => Some(sk),
                    Err(e) => {
                        eprintln!("[hl] WARN: failed to parse signing key: {e}");
                        None
                    }
                },
                Err(e) => {
                    eprintln!("[hl] WARN: failed to decode signing key hex: {e}");
                    None
                }
            }
        });
        Self {
            cfg,
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("hl http client build"),
            market_tx,
            exec_tx,
            account_tx: None,
            asset_meta: tokio::sync::Mutex::new(None),
            account_role: tokio::sync::Mutex::new(None),
            account_abstraction: tokio::sync::Mutex::new(None),
            freshness: Arc::new(Freshness::default()),
            signing_key,
            endpoint_index: std::sync::atomic::AtomicUsize::new(0),
            action_transport,
            action_nonce: AtomicU64::new(0),
            post_request_seq: AtomicU64::new(0),
            post_inflight: AtomicU64::new(0),
            post_request_tx,
            post_request_rx: tokio::sync::Mutex::new(post_request_rx),
        }
    }

    pub fn with_account_tx(mut self, account_tx: mpsc::Sender<AccountEvent>) -> Self {
        self.account_tx = Some(account_tx);
        self
    }

    pub fn uses_ws_post_actions(&self) -> bool {
        self.action_transport == HyperliquidActionTransport::WsPost
    }

    pub fn action_transport_label(&self) -> &'static str {
        self.action_transport.label()
    }

    /// Return the current endpoint URLs without rotating.
    fn current_endpoints(&self) -> (String, String, String) {
        let idx = self
            .endpoint_index
            .load(std::sync::atomic::Ordering::Relaxed);
        let ws = self.cfg.ws_urls[idx % self.cfg.ws_urls.len()].clone();
        let rest = self.cfg.rest_urls[idx % self.cfg.rest_urls.len()].clone();
        let info = self.cfg.info_urls[idx % self.cfg.info_urls.len()].clone();
        (ws, rest, info)
    }

    /// Rotate to the next endpoint in the list. Returns the new (ws, rest, info) triple.
    ///
    /// NOTE (pool staleness): The shared `reqwest::Client` may hold idle pooled
    /// connections to the previous host.  These stale connections are evicted
    /// naturally by reqwest on first failed use — no explicit pool flush is
    /// needed.  The first REST request to the new endpoint will establish a
    /// fresh connection.
    fn rotate_endpoint(&self) -> (String, String, String) {
        let old_idx = self
            .endpoint_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let new_idx = old_idx + 1;
        let ws = self.cfg.ws_urls[new_idx % self.cfg.ws_urls.len()].clone();
        let rest = self.cfg.rest_urls[new_idx % self.cfg.rest_urls.len()].clone();
        let info = self.cfg.info_urls[new_idx % self.cfg.info_urls.len()].clone();
        eprintln!(
            "INFO: HL endpoint rotated: ws={ws}, rest={rest}, info={info} (index {new_idx}; \
             note: stale pooled connections to previous host will be evicted on first failed use)"
        );
        (ws, rest, info)
    }

    /// Return the current info URL (for REST calls).
    fn current_info_url(&self) -> String {
        let idx = self
            .endpoint_index
            .load(std::sync::atomic::Ordering::Relaxed);
        self.cfg.info_urls[idx % self.cfg.info_urls.len()].clone()
    }

    /// Return the current REST (exchange) URL.
    fn current_rest_url(&self) -> String {
        let idx = self
            .endpoint_index
            .load(std::sync::atomic::Ordering::Relaxed);
        self.cfg.rest_urls[idx % self.cfg.rest_urls.len()].clone()
    }

    /// Return the current WS URL (for connections).
    fn current_ws_url(&self) -> String {
        let idx = self
            .endpoint_index
            .load(std::sync::atomic::Ordering::Relaxed);
        self.cfg.ws_urls[idx % self.cfg.ws_urls.len()].clone()
    }

    /// Lightweight HTTP health probe: GET the current info_url with a 3-second
    /// timeout. Returns `true` if the endpoint responds with HTTP 200.
    /// Used before WS reconnection to avoid wasting the WS connect timeout on
    /// a known-dead endpoint.
    async fn probe_info_endpoint(&self) -> bool {
        let info_url = self.current_info_url();
        let probe_timeout = Duration::from_secs(3);
        match tokio::time::timeout(
            probe_timeout,
            self.http
                .post(&info_url)
                .json(&serde_json::json!({"type": "meta"}))
                .send(),
        )
        .await
        {
            Ok(Ok(resp)) if resp.status().is_success() => true,
            Ok(Ok(resp)) => {
                eprintln!(
                    "WARN: HL health probe non-200: status={} url={info_url}",
                    resp.status()
                );
                false
            }
            Ok(Err(err)) => {
                eprintln!("WARN: HL health probe request error: {err} url={info_url}");
                false
            }
            Err(_) => {
                eprintln!("WARN: HL health probe timed out (3s) url={info_url}");
                false
            }
        }
    }

    pub async fn run_public_ws(&self) {
        use rand::Rng;

        let mut backoff = Duration::from_secs(1);
        let mut consecutive_failures: u32 = 0;

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            // After 3+ consecutive failures, probe the info endpoint before
            // attempting a WS connection.  On probe failure, rotate to the
            // next endpoint immediately (avoids wasting the WS connect timeout
            // on a known-dead endpoint).
            if consecutive_failures >= 3 {
                if !self.probe_info_endpoint().await {
                    let (ws, _rest, info) = self.rotate_endpoint();
                    eprintln!("INFO: HL public WS probe failed, rotated to ws={ws} info={info}");
                }
            }

            let session_start = std::time::Instant::now();

            // Layer C: session-level timeout catches ALL hang scenarios.
            let max_session = Duration::from_secs(
                std::env::var("PARAPHINA_WS_MAX_SESSION_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(86_400), // 24h — Layer A enforcer handles stuck connections
            );
            let result = tokio::time::timeout(max_session, self.public_ws_once()).await;
            match result {
                Ok(Err(err)) => {
                    consecutive_failures += 1;
                    let level = if consecutive_failures >= 20 {
                        "ERROR"
                    } else if consecutive_failures >= 5 {
                        "WARN"
                    } else {
                        "INFO"
                    };
                    eprintln!(
                        "{level}: Hyperliquid public WS error (consecutive_failures={consecutive_failures}): {err}"
                    );
                }
                Err(_timeout) => {
                    // Session hung for >max_session — force restart.
                    hl_audit_reconnect("session_timeout");
                    eprintln!(
                        "ERROR: Hyperliquid public WS session timeout ({}s) — force reconnect",
                        max_session.as_secs()
                    );
                    consecutive_failures += 1;
                }
                Ok(Ok(())) => {}
            }

            // FIX: Reset backoff and failure counter if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                if consecutive_failures > 0 {
                    eprintln!(
                        "INFO: Hyperliquid WS session was healthy for {:?}; \
                         resetting backoff and failure counter (was {})",
                        session_duration, consecutive_failures
                    );
                }
                consecutive_failures = 0;
                backoff = Duration::from_secs(1);
            }

            // Escalating backoff caps: give upstream more time to recover
            let max_backoff = match consecutive_failures {
                0..=10 => Duration::from_secs(30),
                11..=20 => Duration::from_secs(60),
                _ => Duration::from_secs(120),
            };

            // Add jitter to prevent thundering-herd reconnection storms.
            let jitter = Duration::from_millis(
                rand::thread_rng().gen_range(0..=backoff.as_millis().max(1) as u64 / 4),
            );
            tokio::time::sleep(backoff + jitter).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        let freshness = self.freshness.clone();
        let connect_timeout = hl_ws_connect_timeout();
        let read_timeout = hl_ws_read_timeout();
        let ws_url = self.current_ws_url();
        eprintln!(
            "INFO: Hyperliquid public WS connecting url={ws_url} connect_timeout={connect_timeout:?}",
        );
        // FIX A1: connect_async can hang indefinitely if the remote host is unreachable
        // or accepts the TCP connection but never completes TLS/upgrade.
        let (ws_stream, _) = with_timeout(
            connect_timeout,
            "public WS connect",
            connect_async(ws_url.as_str()),
        )
        .await?
        // Unwrap the inner Result from connect_async
        .map_err(|e| anyhow::anyhow!("Hyperliquid public WS connect error: {e}"))?;
        eprintln!("INFO: Hyperliquid public WS connected url={ws_url}",);
        let (mut write, mut read) = ws_stream.split();
        let sub = json!({
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": self.cfg.coin,
                "nSigFigs": self.cfg.n_sig_figs,
                "nLevels": self.cfg.n_levels
            }
        });
        write.send(Message::Text(sub.to_string())).await?;
        eprintln!(
            "INFO: Hyperliquid public WS subscribed coin={} nSigFigs={} nLevels={}",
            self.cfg.coin, self.cfg.n_sig_figs, self.cfg.n_levels
        );
        // FIX: Application-level heartbeat per Hyperliquid docs.
        // "The server will close any connection if it hasn't sent a message to it
        //  in the last 60 seconds." Send {"method":"ping"} every 30s to prevent
        //  server-side idle disconnection.
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_HL_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Skip the first immediate tick
        ping_timer.tick().await;
        let connect_start_ns = mono_now_ns();
        freshness.reset_for_new_connection();
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        let stale_ms = hl_stale_ms();
        if std::env::var_os("HL_FIXTURE_DIR").is_some() {
            eprintln!("INFO: Hyperliquid fixture mode detected; freshness watchdog disabled");
        } else {
            let watchdog_stale_ms = stale_ms;
            let watchdog_freshness = self.freshness.clone();
            tokio::spawn(async move {
                let mut iv = tokio::time::interval(Duration::from_millis(HL_WATCHDOG_TICK_MS));
                iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    iv.tick().await;
                    let now = mono_now_ns();
                    let anchor = watchdog_freshness.anchor_with_connect_start(connect_start_ns);
                    if anchor != 0 && age_ms(now, anchor) > watchdog_stale_ms {
                        let _ = stale_tx.send(());
                        break;
                    }
                }
            });
        }
        let hl_internal_pub_q = hl_internal_pub_q();
        let (tx_int, mut rx_int) = tokio::sync::mpsc::channel::<MarketDataEvent>(hl_internal_pub_q);
        let pending_latest = Arc::new(tokio::sync::Mutex::new(None::<MarketDataEvent>));
        let forward_market_tx = self.market_tx.clone();
        let forward_freshness = self.freshness.clone();
        let forward_pending = pending_latest.clone();
        let forward_audit = Arc::new(HlForwardAudit::default());
        let hl_ts_policy = HlTimestampPolicy::from_env();
        let forward_audit_task = forward_audit.clone();
        tokio::spawn(async move {
            while let Some(mut event) = rx_int.recv().await {
                let mut coalesced_drops = 0u64;
                while let Ok(next) = rx_int.try_recv() {
                    event = next;
                    coalesced_drops = coalesced_drops.saturating_add(1);
                }
                if coalesced_drops > 0 {
                    forward_audit_task.observe_coalesced_drop(coalesced_drops);
                }
                if let Some(pending) = forward_pending.lock().await.take() {
                    event = pending;
                    forward_audit_task.observe_pending_take();
                }
                let send_started = Instant::now();
                let send_ok = forward_market_tx.send(event).await.is_ok();
                forward_audit_task.observe_send_block_ms(send_started.elapsed().as_millis() as u64);
                if send_ok {
                    forward_freshness
                        .last_published_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                } else {
                    forward_audit_task.observe_send_err();
                }
            }
        });
        fn maybe_emit_hl_pubq_audit(
            enabled: bool,
            queue_cap: usize,
            ts_policy_enabled: bool,
            freshness: &Freshness,
            forward_audit: &HlForwardAudit,
            tx_int: &mpsc::Sender<MarketDataEvent>,
            pending_latest: &tokio::sync::Mutex<Option<MarketDataEvent>>,
            last_emit: &mut Instant,
            queued_hiwater: &mut usize,
            pending_latest_present: &mut u8,
            pending_overwrite: &mut u64,
            pending_lock_fail: &mut u64,
            ts_zero_count: &mut u64,
            try_send_ok: &mut u64,
            try_send_full: &mut u64,
        ) {
            if !enabled {
                return;
            }
            let queued_len = queue_cap.saturating_sub(tx_int.capacity());
            *queued_hiwater = (*queued_hiwater).max(queued_len);
            if let Ok(guard) = pending_latest.try_lock() {
                *pending_latest_present = u8::from(guard.is_some());
            }
            let emit_since_ms = last_emit.elapsed().as_millis() as u64;
            if emit_since_ms < HL_WS_AUDIT_INTERVAL_MS {
                return;
            }
            let now_ns = mono_now_ns();
            let ws_rx_age_ms = age_ms(now_ns, freshness.last_ws_rx_ns.load(Ordering::Relaxed));
            let data_rx_age_ms = age_ms(now_ns, freshness.last_data_rx_ns.load(Ordering::Relaxed));
            let pub_age_ms = age_ms(now_ns, freshness.last_published_ns.load(Ordering::Relaxed));
            let book_age_ms = age_ms(now_ns, freshness.last_book_event_ns.load(Ordering::Relaxed));
            let pub_minus_book_age_ms = pub_age_ms.saturating_sub(book_age_ms);
            let forward = forward_audit.snapshot_and_reset();
            eprintln!(
                "WS_AUDIT venue=hyperliquid component=hl_pubq reason=periodic interval_ms=1000 \
queue_cap={} queued_len={} queued_hiwater={} pending_latest_present={} pending_overwrite={} \
pending_lock_fail={} ts_zero_count={} ws_rx_age_ms={} data_rx_age_ms={} pub_age_ms={} book_age_ms={} \
pub_minus_book_age_ms={} send_block_max_ms={} send_block_gt_5ms={} send_block_gt_50ms={} \
send_block_gt_250ms={} forward_send_count={} forward_send_err_count={} coalesced_drop_count={} pending_take_count={} \
ts_missing_or_zero_count={} ts_clamped_past_skew_count={} ts_clamped_future_skew_count={} \
ts_policy_enabled={} ts_policy_applied_count={} ts_kept_exchange_count={} ts_past_skew_max_ms={} ts_future_skew_max_ms={} \
try_send_ok={} try_send_full={} emit_since_ms={}",
                queue_cap,
                queued_len,
                (*queued_hiwater).max(queued_len),
                *pending_latest_present,
                *pending_overwrite,
                *pending_lock_fail,
                *ts_zero_count,
                ws_rx_age_ms,
                data_rx_age_ms,
                pub_age_ms,
                book_age_ms,
                pub_minus_book_age_ms,
                forward.send_block_max_ms,
                forward.send_block_gt_5ms,
                forward.send_block_gt_50ms,
                forward.send_block_gt_250ms,
                forward.forward_send_count,
                forward.forward_send_err_count,
                forward.coalesced_drop_count,
                forward.pending_take_count,
                forward.ts_missing_or_zero_count,
                forward.ts_clamped_past_skew_count,
                forward.ts_clamped_future_skew_count,
                u8::from(ts_policy_enabled),
                forward.ts_policy_applied_count,
                forward.ts_kept_exchange_count,
                forward.ts_past_skew_max_ms,
                forward.ts_future_skew_max_ms,
                *try_send_ok,
                *try_send_full,
                emit_since_ms,
            );
            *last_emit = Instant::now();
            *queued_hiwater = queued_len;
            *pending_overwrite = 0;
            *pending_lock_fail = 0;
            *ts_zero_count = 0;
            *try_send_ok = 0;
            *try_send_full = 0;
        }
        let hl_pubq_audit_enabled = hl_ws_audit_enabled();
        let mut hl_pubq_last_emit = Instant::now();
        let mut hl_pubq_queued_hiwater: usize = 0;
        let mut hl_pubq_pending_latest_present: u8 = 0;
        let mut hl_pubq_pending_overwrite: u64 = 0;
        let mut hl_pubq_pending_lock_fail: u64 = 0;
        let mut hl_pubq_ts_zero_count: u64 = 0;
        let mut hl_pubq_try_send_ok: u64 = 0;
        let mut hl_pubq_try_send_full: u64 = 0;
        let mut try_publish = |event: MarketDataEvent| -> anyhow::Result<()> {
            let event_ts_zero = match &event {
                MarketDataEvent::L2Snapshot(snapshot) => snapshot.timestamp_ms == 0,
                MarketDataEvent::L2Delta(delta) => delta.timestamp_ms == 0,
                _ => false,
            };
            match tx_int.try_send(event) {
                Ok(()) => {
                    if hl_pubq_audit_enabled {
                        if event_ts_zero {
                            hl_pubq_ts_zero_count = hl_pubq_ts_zero_count.saturating_add(1);
                        }
                        hl_pubq_try_send_ok = hl_pubq_try_send_ok.saturating_add(1);
                        maybe_emit_hl_pubq_audit(
                            hl_pubq_audit_enabled,
                            hl_internal_pub_q,
                            hl_ts_policy.enabled,
                            freshness.as_ref(),
                            forward_audit.as_ref(),
                            &tx_int,
                            pending_latest.as_ref(),
                            &mut hl_pubq_last_emit,
                            &mut hl_pubq_queued_hiwater,
                            &mut hl_pubq_pending_latest_present,
                            &mut hl_pubq_pending_overwrite,
                            &mut hl_pubq_pending_lock_fail,
                            &mut hl_pubq_ts_zero_count,
                            &mut hl_pubq_try_send_ok,
                            &mut hl_pubq_try_send_full,
                        );
                    }
                    Ok(())
                }
                Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                    if hl_pubq_audit_enabled {
                        if event_ts_zero {
                            hl_pubq_ts_zero_count = hl_pubq_ts_zero_count.saturating_add(1);
                        }
                        hl_pubq_try_send_full = hl_pubq_try_send_full.saturating_add(1);
                    }
                    if let Ok(mut guard) = pending_latest.try_lock() {
                        if hl_pubq_audit_enabled && guard.is_some() {
                            hl_pubq_pending_overwrite = hl_pubq_pending_overwrite.saturating_add(1);
                        }
                        *guard = Some(event);
                        if hl_pubq_audit_enabled {
                            hl_pubq_pending_latest_present = 1;
                        }
                    } else if hl_pubq_audit_enabled {
                        hl_pubq_pending_lock_fail = hl_pubq_pending_lock_fail.saturating_add(1);
                    }
                    if hl_pubq_audit_enabled {
                        maybe_emit_hl_pubq_audit(
                            hl_pubq_audit_enabled,
                            hl_internal_pub_q,
                            hl_ts_policy.enabled,
                            freshness.as_ref(),
                            forward_audit.as_ref(),
                            &tx_int,
                            pending_latest.as_ref(),
                            &mut hl_pubq_last_emit,
                            &mut hl_pubq_queued_hiwater,
                            &mut hl_pubq_pending_latest_present,
                            &mut hl_pubq_pending_overwrite,
                            &mut hl_pubq_pending_lock_fail,
                            &mut hl_pubq_ts_zero_count,
                            &mut hl_pubq_try_send_ok,
                            &mut hl_pubq_try_send_full,
                        );
                    }
                    Ok(())
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => Err(anyhow::anyhow!(
                    "Hyperliquid public WS internal publish queue closed"
                )),
            }
        };
        let mut tracker = L2SeqTracker::new();
        let mut l2_seq_fallback: u64 = 0;
        let mut first_book_update_logged = false;
        let mut first_message_logged = false;
        let mut logged_non_utf8_binary = false;
        let mut first_decoded_top_logged = false;
        let mut decode_miss_count = 0usize;
        let mut have_baseline = false;
        let mut delta_buf: VecDeque<MarketDataEvent> = VecDeque::new();
        // Bounded sampling for non-book messages to diagnose staleness issues.
        let mut non_book_msg_count: u64 = 0;
        const NON_BOOK_LOG_LIMIT: u64 = 5;
        // Rate-limited logging for snapshot decode failures and skipped levels.
        let mut last_snapshot_fail_warn_ns: u64 = 0;
        let mut last_skipped_levels_warn_ns: u64 = 0;
        loop {
            tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    hl_audit_reconnect("stale_watchdog");
                    eprintln!("WARN: Hyperliquid public WS watchdog: no publishable book update for {stale_ms}ms — reconnecting");
                    anyhow::bail!("Hyperliquid public WS stale: freshness exceeded {stale_ms}ms");
                }
                // FIX: Send application-level ping to prevent server-side idle disconnection.
                _ = ping_timer.tick() => {
                    if let Err(e) = write.send(Message::Text(r#"{"method":"ping"}"#.to_string())).await {
                        hl_audit_reconnect("ping_send_fail");
                        eprintln!("WARN: Hyperliquid public WS ping send failed: {e} — reconnecting");
                        anyhow::bail!("Hyperliquid public WS ping send failed: {e}");
                    }
                }
                // FIX A2: read timeout prevents idle ESTABLISHED sockets from blocking forever.
                read_result = tokio::time::timeout(read_timeout, read.next()) => {
                    let maybe = match read_result {
                        Ok(m) => m,
                        Err(_) => {
                            hl_audit_reconnect("read_timeout");
                            eprintln!(
                                "WARN: Hyperliquid public WS read timeout ({read_timeout:?}) — no frame received, reconnecting"
                            );
                            anyhow::bail!("Hyperliquid public WS read timeout after {read_timeout:?}");
                        }
                    };
                    let Some(msg) = maybe else { break; };
                    let msg = msg?;
                    freshness
                        .last_ws_rx_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    let payload = match msg {
                        Message::Text(text) => text,
                        Message::Binary(bytes) => match String::from_utf8(bytes) {
                            Ok(text) => text,
                            Err(_) => {
                                if !logged_non_utf8_binary {
                                    eprintln!(
                                        "WARN: Hyperliquid public WS non-utf8 binary frame url={ws_url}",
                                    );
                                    logged_non_utf8_binary = true;
                                }
                                continue;
                            }
                        },
                        _ => continue,
                    };
                    if !first_message_logged {
                        eprintln!("INFO: Hyperliquid public WS first message received");
                        first_message_logged = true;
                    }
                    let value = match serde_json::from_str::<serde_json::Value>(&payload) {
                        Ok(value) => value,
                        Err(err) => {
                            let snippet: String = payload.chars().take(160).collect();
                            eprintln!(
                                "WARN: Hyperliquid public WS parse error: {err} url={ws_url} snippet={snippet}",
                            );
                            continue;
                        }
                    };
                    let channel = value.get("channel").and_then(|v| v.as_str()).unwrap_or("");
                    if channel == "subscriptionResponse" || channel == "pong" {
                        continue;
                    }
                    if channel == "l2Book" {
                        freshness
                            .last_data_rx_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        if let Some(top) = decode_l2book_top(&value) {
                            if !first_decoded_top_logged {
                                eprintln!(
                                    "FIRST_DECODED_TOP venue=hyperliquid bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                    top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                                );
                                first_decoded_top_logged = true;
                            }
                        } else if decode_miss_count < 3 {
                            decode_miss_count += 1;
                            log_decode_miss(
                                "Hyperliquid",
                                &value,
                                &payload,
                                decode_miss_count,
                                ws_url.as_str(),
                            );
                        }
                        // Use resilient snapshot decoder that skips malformed levels
                        let decode_result = decode_l2book_snapshot_resilient(
                            &value,
                            self.cfg.venue_index,
                            self.cfg.coin.as_str(),
                            &mut l2_seq_fallback,
                        );

                        // Log skipped levels (rate-limited)
                        let total_skipped = decode_result.bid_skipped + decode_result.ask_skipped;
                        if total_skipped > 0 {
                            let now_ns = mono_now_ns();
                            if age_ms(now_ns, last_skipped_levels_warn_ns) >= HL_DECODE_WARN_INTERVAL_MS {
                                last_skipped_levels_warn_ns = now_ns;
                                eprintln!(
                                    "WARN: Hyperliquid l2Book skipped {} malformed levels (bids: {}/{}, asks: {}/{})",
                                    total_skipped,
                                    decode_result.bid_skipped,
                                    decode_result.bid_total,
                                    decode_result.ask_skipped,
                                    decode_result.ask_total
                                );
                            }
                        }

                        if let Some(snapshot) = decode_result.event {
                            let rx_now_ms = now_ms();
                            let snapshot = apply_hl_l2_event_ts_policy(
                                snapshot,
                                rx_now_ms,
                                hl_ts_policy,
                                forward_audit.as_ref(),
                            );
                            // Update freshness ONLY when we produce a publishable event
                            let now_ns = mono_now_ns();
                            freshness
                                .last_parsed_ns
                                .store(now_ns, Ordering::Relaxed);
                            // FIX A3: Track book-specific events for the watchdog
                            freshness
                                .last_book_event_ns
                                .store(now_ns, Ordering::Relaxed);
                            have_baseline = true;
                            while let Some(buffered) = delta_buf.pop_front() {
                                try_publish(buffered)?;
                            }
                            try_publish(snapshot)?;
                        } else {
                            // Snapshot decode failed - emit bounded warning
                            let now_ns = mono_now_ns();
                            if age_ms(now_ns, last_snapshot_fail_warn_ns) >= HL_DECODE_WARN_INTERVAL_MS {
                                last_snapshot_fail_warn_ns = now_ns;
                                let keys = value
                                    .as_object()
                                    .map(|obj| {
                                        let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                                        keys.sort();
                                        format!("[{}]", keys.join(","))
                                    })
                                    .unwrap_or_else(|| "[non-object]".to_string());
                                let snippet: String = payload.chars().take(200).collect();
                                eprintln!(
                                    "WARN: Hyperliquid l2Book snapshot decode failed reason={:?} keys={} bids={}/{} asks={}/{} snippet={}",
                                    decode_result.failure_reason.unwrap_or("unknown"),
                                    keys,
                                    decode_result.bid_total.saturating_sub(decode_result.bid_skipped),
                                    decode_result.bid_total,
                                    decode_result.ask_total.saturating_sub(decode_result.ask_skipped),
                                    decode_result.ask_total,
                                    snippet
                                );
                            }
                        }
                        continue;
                    }
                    // Non-l2Book message received - log bounded samples for staleness diagnosis.
                    // This helps identify if WS is alive but not receiving book updates.
                    non_book_msg_count += 1;
                    if non_book_msg_count <= NON_BOOK_LOG_LIMIT {
                        let snippet: String = payload.chars().take(120).collect();
                        eprintln!(
                            "WARN: Hyperliquid public WS non-book message after subscribe channel={} count={} snippet={}",
                            channel, non_book_msg_count, snippet
                        );
                    } else if non_book_msg_count == NON_BOOK_LOG_LIMIT + 1 {
                        eprintln!(
                            "WARN: Hyperliquid public WS suppressing further non-book message logs (count={})",
                            non_book_msg_count
                        );
                    }
                    // NOTE: In public_ws_once this parser path is effectively unreachable:
                    // l2Book frames are handled above and continue early, while
                    // parse_l2_message_value() also gates on channel=="l2Book".
                    // It is kept for fixture/tooling paths that call parse_l2_message* directly.
                    if let Some(parsed) = parse_l2_message_value(&value, self.cfg.venue_index) {
                        let now_ns = mono_now_ns();
                        freshness
                            .last_parsed_ns
                            .store(now_ns, Ordering::Relaxed);
                        // FIX A3: Also update book event tracker for non-l2Book parseable events
                        freshness
                            .last_book_event_ns
                            .store(now_ns, Ordering::Relaxed);
                        let outcome = tracker.on_message(parsed);
                        if let Some(seq) = outcome.refresh_snapshot {
                        if let Some(snapshot) = self.refresh_snapshot(seq).await {
                            if matches!(&snapshot, MarketDataEvent::L2Snapshot(_)) {
                                have_baseline = true;
                                while let Some(buffered) = delta_buf.pop_front() {
                                    try_publish(buffered)?;
                                }
                            }
                            try_publish(snapshot)?;
                        }
                        }
                        if let Some(event) = outcome.event {
                            let event = apply_hl_l2_event_ts_policy(
                                event,
                                now_ms(),
                                hl_ts_policy,
                                forward_audit.as_ref(),
                            );
                            if !first_book_update_logged {
                                eprintln!("INFO: Hyperliquid public WS first book update");
                                first_book_update_logged = true;
                            }
                        match event {
                            MarketDataEvent::L2Delta(_) if !have_baseline => {
                                delta_buf.push_back(event);
                                if delta_buf.len() > HL_DELTA_BOOTSTRAP_BUF {
                                    delta_buf.clear();
                                    if let Some(snapshot) = self.refresh_snapshot(0).await {
                                    if matches!(&snapshot, MarketDataEvent::L2Snapshot(_)) {
                                            have_baseline = true;
                                            while let Some(buffered) = delta_buf.pop_front() {
                                                try_publish(buffered)?;
                                            }
                                        }
                                        try_publish(snapshot)?;
                                    }
                                }
                            }
                            MarketDataEvent::L2Snapshot(_) => {
                                have_baseline = true;
                                while let Some(buffered) = delta_buf.pop_front() {
                                    try_publish(buffered)?;
                                }
                                try_publish(event)?;
                            }
                            _ => {
                                try_publish(event)?;
                            }
                        }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub async fn run_private_ws(&self) {
        use rand::Rng;

        if self.cfg.private_key_hex.is_none() {
            return;
        }
        let mut backoff = Duration::from_secs(1);
        let mut consecutive_reconnects: u32 = 0;

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            // After 3+ consecutive reconnects, probe the info endpoint.
            // On probe failure, rotate to the next endpoint.
            if consecutive_reconnects >= 3 {
                if !self.probe_info_endpoint().await {
                    let (ws, _rest, info) = self.rotate_endpoint();
                    eprintln!("INFO: HL private WS probe failed, rotated to ws={ws} info={info}");
                }
            }

            let session_start = std::time::Instant::now();

            if let Err(err) = self.private_ws_once().await {
                consecutive_reconnects += 1;
                eprintln!(
                    "Hyperliquid private WS error (consecutive_reconnects={consecutive_reconnects}): {err}"
                );
            }

            // FIX: Reset backoff and reconnect counter if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                if consecutive_reconnects > 0 {
                    eprintln!(
                        "INFO: Hyperliquid private WS session was healthy for {:?}; \
                         resetting backoff and reconnect counter (was {})",
                        session_duration, consecutive_reconnects
                    );
                }
                consecutive_reconnects = 0;
                backoff = Duration::from_secs(1);
            }

            // Add jitter to prevent thundering-herd reconnection storms.
            let jitter = Duration::from_millis(
                rand::thread_rng().gen_range(0..=backoff.as_millis().max(1) as u64 / 4),
            );
            tokio::time::sleep(backoff + jitter).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn private_ws_once(&self) -> anyhow::Result<()> {
        let connect_timeout = hl_ws_connect_timeout();
        let read_timeout = hl_ws_read_timeout();
        let ws_url = self.current_ws_url();

        let (ws_stream, _) = with_timeout(
            connect_timeout,
            "private WS connect",
            connect_async(ws_url.as_str()),
        )
        .await?
        .map_err(|e| anyhow::anyhow!("Hyperliquid private WS connect error: {e}"))?;
        let (mut write, mut read) = ws_stream.split();
        let account_user = if self.account_tx.is_some() {
            self.cfg.vault_address.as_deref()
        } else {
            None
        };
        for subscription in build_private_subscriptions(account_user) {
            write.send(Message::Text(subscription.to_string())).await?;
        }
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_HL_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ping_timer.tick().await;
        loop {
            tokio::select! {
                _ = ping_timer.tick() => {
                    if let Err(err) = write.send(Message::Text(r#"{"method":"ping"}"#.to_string())).await {
                        eprintln!("WARN: Hyperliquid private WS ping send failed: {err} — reconnecting");
                        anyhow::bail!("Hyperliquid private WS ping send failed: {err}");
                    }
                }
                read_result = tokio::time::timeout(read_timeout, read.next()) => {
                    let msg = match read_result {
                        Ok(Some(msg)) => msg?,
                        Ok(None) => break,
                        Err(_) => {
                            eprintln!(
                                "WARN: Hyperliquid private WS read timeout ({read_timeout:?}) — reconnecting"
                            );
                            anyhow::bail!("Hyperliquid private WS read timeout after {read_timeout:?}");
                        }
                    };
                    if let Message::Text(text) = msg {
                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                            let channel = value.get("channel").and_then(|v| v.as_str()).unwrap_or("");
                            if channel == "subscriptionResponse" || channel == "pong" {
                                continue;
                            }
                            for event in translate_private_events(&value) {
                                let _ = self.exec_tx.send(event).await;
                            }
                            if let Some(account_tx) = self.account_tx.as_ref() {
                                if let Some(event) = translate_account_event(
                                    &value,
                                    Some(self.cfg.coin.as_str()),
                                    self.cfg.venue_index,
                                ) {
                                    let _ = account_tx.send(event).await;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn run_post_ws(&self) {
        use rand::Rng;

        if !self.uses_ws_post_actions() {
            eprintln!("INFO: Hyperliquid post WS not enabled; transport=http");
            return;
        }

        let mut backoff = Duration::from_secs(1);
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = Instant::now();
            if let Err(err) = self.post_ws_once().await {
                eprintln!("Hyperliquid post WS error: {err}");
            }

            if session_start.elapsed() >= healthy_threshold {
                backoff = Duration::from_secs(1);
            }

            let jitter = Duration::from_millis(
                rand::thread_rng().gen_range(0..=backoff.as_millis().max(1) as u64 / 4),
            );
            tokio::time::sleep(backoff + jitter).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn post_ws_once(&self) -> anyhow::Result<()> {
        let connect_timeout = hl_ws_connect_timeout();
        let read_timeout = hl_ws_read_timeout();
        let ws_url = self.current_ws_url();

        let (ws_stream, _) = with_timeout(
            connect_timeout,
            "post WS connect",
            connect_async(ws_url.as_str()),
        )
        .await?
        .map_err(|e| anyhow::anyhow!("Hyperliquid post WS connect error: {e}"))?;
        let (mut write, mut read) = ws_stream.split();
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_HL_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ping_timer.tick().await;

        let max_inflight = hl_ws_post_max_inflight();
        let mut receiver = self.post_request_rx.lock().await;
        let mut pending: BTreeMap<u64, HyperliquidPostPending> = BTreeMap::new();

        loop {
            tokio::select! {
                _ = ping_timer.tick() => {
                    if let Err(err) = write.send(Message::Text(r#"{"method":"ping"}"#.to_string())).await {
                        self.fail_pending_post_requests(&mut pending, format!("Hyperliquid post WS ping send failed: {err}"));
                        anyhow::bail!("Hyperliquid post WS ping send failed: {err}");
                    }
                }
                maybe_req = receiver.recv() => {
                    let Some(req) = maybe_req else {
                        self.fail_pending_post_requests(&mut pending, "Hyperliquid post WS request channel closed".to_string());
                        break;
                    };
                    if pending.len() >= max_inflight {
                        let _ = req.response_tx.send(Err(anyhow::anyhow!(
                            "Hyperliquid post WS inflight limit exceeded: {} >= {}",
                            pending.len(),
                            max_inflight
                        )));
                        continue;
                    }
                    let envelope = json!({
                        "method": "post",
                        "id": req.id,
                        "request": {
                            "type": "action",
                            "payload": req.payload,
                        }
                    });
                    if let Err(err) = write.send(Message::Text(envelope.to_string())).await {
                        let _ = req.response_tx.send(Err(anyhow::anyhow!(
                            "Hyperliquid post WS send failed: {err}"
                        )));
                        self.fail_pending_post_requests(&mut pending, format!("Hyperliquid post WS send failed: {err}"));
                        anyhow::bail!("Hyperliquid post WS send failed: {err}");
                    }
                    let inflight = self.post_inflight.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "HL_POST_SUBMIT submit_path=ws_post post_id={} action_label={} batch_kind={} batch_size={} post_inflight={}",
                        req.id,
                        req.action_label,
                        req.batch_kind,
                        req.batch_size,
                        inflight
                    );
                    pending.insert(req.id, HyperliquidPostPending {
                        action_label: req.action_label,
                        batch_kind: req.batch_kind,
                        batch_size: req.batch_size,
                        sent_at: Instant::now(),
                        response_tx: req.response_tx,
                    });
                }
                read_result = tokio::time::timeout(read_timeout, read.next()) => {
                    let msg = match read_result {
                        Ok(Some(msg)) => msg?,
                        Ok(None) => {
                            self.fail_pending_post_requests(&mut pending, "Hyperliquid post WS closed".to_string());
                            break;
                        }
                        Err(_) => {
                            self.fail_pending_post_requests(&mut pending, format!("Hyperliquid post WS read timeout after {read_timeout:?}"));
                            anyhow::bail!("Hyperliquid post WS read timeout after {read_timeout:?}");
                        }
                    };
                    let Message::Text(text) = msg else {
                        continue;
                    };
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                        continue;
                    };
                    let channel = value.get("channel").and_then(|v| v.as_str()).unwrap_or("");
                    if channel == "pong" || channel == "subscriptionResponse" {
                        continue;
                    }
                    if channel != "post" {
                        continue;
                    }
                    let Some((id, response_type, response_result)) = parse_ws_post_response(&value) else {
                        continue;
                    };
                    let Some(pending_req) = pending.remove(&id) else {
                        continue;
                    };
                    let inflight = self.post_inflight.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                    let latency_ms = pending_req.sent_at.elapsed().as_millis();
                    eprintln!(
                        "HL_POST_RESPONSE submit_path=ws_post post_id={} action_label={} batch_kind={} batch_size={} response_type={} post_latency_ms={} post_inflight={}",
                        id,
                        pending_req.action_label,
                        pending_req.batch_kind,
                        pending_req.batch_size,
                        response_type,
                        latency_ms,
                        inflight
                    );
                    if let Err(err) = &response_result {
                        eprintln!(
                            "HL_POST_ACTION_ERR submit_path=ws_post post_id={} action_label={} batch_kind={} batch_size={} detail={}",
                            id,
                            pending_req.action_label,
                            pending_req.batch_kind,
                            pending_req.batch_size,
                            err
                        );
                    }
                    let _ = pending_req.response_tx.send(response_result);
                }
            }
        }

        Ok(())
    }

    pub async fn run_account_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let Some(account_tx) = self.account_tx.as_ref() else {
                continue;
            };
            match self.fetch_account_snapshot().await {
                Ok(snapshot) => {
                    let _ = account_tx.send(snapshot).await;
                }
                Err(err) => {
                    eprintln!("Hyperliquid account polling error: {err}");
                }
            }
        }
    }

    /// REST-based book polling fallback: when the WS has been stale for longer
    /// than `PARAPHINA_HL_REST_FALLBACK_STALE_MS` (default 15s), fetch the book
    /// via REST API every `PARAPHINA_HL_REST_FALLBACK_POLL_MS` (default 2s).
    /// This provides a completely independent data path that survives WS failures.
    pub async fn run_rest_book_fallback(&self) {
        let stale_threshold_ms: u64 = std::env::var("PARAPHINA_HL_REST_FALLBACK_STALE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15_000);
        let poll_interval_ms: u64 = std::env::var("PARAPHINA_HL_REST_FALLBACK_POLL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_000);
        let mut interval = tokio::time::interval(Duration::from_millis(poll_interval_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut active_logged = false;
        let mut inactive_logged = false;
        // FIX D1: Startup grace period.  During the first 60s of the process,
        // `last_book_event_ns == 0` is expected (WS hasn't delivered yet).
        // After the grace period, treat 0 as stale — this catches the bug where
        // `reset_for_new_connection()` sets `last_book_event_ns` to 0 on reconnect,
        // permanently disabling the fallback.
        let startup_grace_ns: u64 = std::env::var("PARAPHINA_HL_REST_STARTUP_GRACE_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(60_000)
            * 1_000_000; // convert ms → ns
        loop {
            interval.tick().await;
            let now = mono_now_ns();
            // Check if WS is stale: use the book event freshness as the indicator.
            let last_book = self.freshness.last_book_event_ns.load(Ordering::Relaxed);
            let ws_stale = if last_book == 0 {
                // FIX D1: Only give WS grace during actual startup.
                // After the grace period, treat zero as stale (reconnect scenario).
                now >= startup_grace_ns
            } else {
                age_ms(now, last_book) > stale_threshold_ms
            };
            if !ws_stale {
                if active_logged && !inactive_logged {
                    eprintln!("INFO: Hyperliquid REST book fallback deactivated (WS recovered)");
                    inactive_logged = true;
                    active_logged = false;
                }
                continue;
            }
            if !active_logged {
                eprintln!(
                    "WARN: Hyperliquid REST book fallback activated (WS stale for >{}ms)",
                    stale_threshold_ms
                );
                active_logged = true;
                inactive_logged = false;
            }
            match fetch_l2_snapshot(&self.http, &self.cfg, &self.current_info_url()).await {
                Ok(snapshot) => {
                    if self.market_tx.send(snapshot).await.is_err() {
                        eprintln!("WARN: Hyperliquid REST book fallback: market_tx closed");
                    }
                }
                Err(err) => {
                    eprintln!("WARN: Hyperliquid REST book fallback error: {err}");
                }
            }
        }
    }

    pub async fn run_funding_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut seq: u64 = 0;
        loop {
            interval.tick().await;
            match fetch_public_funding(&self.http, &self.cfg, &self.current_info_url()).await {
                Ok(mut update) => {
                    seq = seq.wrapping_add(1);
                    update.seq = seq;
                    // FIX: Log channel send failures instead of silently ignoring
                    if let Err(err) = self
                        .market_tx
                        .send(MarketDataEvent::FundingUpdate(update))
                        .await
                    {
                        eprintln!("Hyperliquid funding send failed: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Hyperliquid funding polling error: {err}");
                }
            }
        }
    }

    async fn refresh_snapshot(&self, seq: u64) -> Option<MarketDataEvent> {
        let now = mono_now_ns();
        let last = self
            .freshness
            .last_snapshot_resync_ns
            .load(Ordering::Relaxed);
        if last != 0 && age_ms(now, last) < HL_SNAPSHOT_COOLDOWN_MS {
            return None;
        }
        self.freshness
            .last_snapshot_resync_ns
            .store(now, Ordering::Relaxed);
        if let Ok(snapshot) =
            fetch_l2_snapshot(&self.http, &self.cfg, &self.current_info_url()).await
        {
            return Some(snapshot);
        } else {
            eprintln!("Hyperliquid snapshot refresh failed at seq={seq}");
        }
        None
    }

    pub async fn place_order(
        &self,
        intent: &OrderIntent,
        now_ms: TimestampMs,
    ) -> anyhow::Result<()> {
        if self.cfg.paper_mode {
            eprintln!("Hyperliquid paper mode: {:?}", intent);
            return Ok(());
        }
        let asset_meta = self.get_asset_meta().await?;
        let action = build_action(intent, asset_meta)?;
        self.submit_signed_action(action, now_ms, "order", "place_single", 1)
            .await
    }

    pub async fn cancel_order(
        &self,
        intent: &OrderIntent,
        now_ms: TimestampMs,
    ) -> anyhow::Result<()> {
        if self.cfg.paper_mode {
            eprintln!("Hyperliquid paper mode cancel: {:?}", intent);
            return Ok(());
        }
        let asset_index = self.get_asset_meta().await?.index;
        let action = build_cancel_action(intent, asset_index)?;
        self.submit_signed_action(action, now_ms, "cancel", "cancel_single", 1)
            .await
    }

    pub async fn cancel_all(&self, now_ms: TimestampMs) -> anyhow::Result<()> {
        if self.cfg.paper_mode {
            eprintln!("Hyperliquid paper mode cancel_all");
            return Ok(());
        }
        let asset_index = self.get_asset_meta().await?.index;
        let action = build_cancel_all_action(asset_index);
        self.submit_signed_action(action, now_ms, "cancel_all", "cancel_all", 1)
            .await
    }

    async fn submit_signed_action_with_hint(
        &self,
        action: serde_json::Value,
        now_ms: TimestampMs,
        action_label: &'static str,
        batch_kind: &'static str,
        batch_size: usize,
        hint: TransportHint,
    ) -> anyhow::Result<()> {
        let nonce = self.next_action_nonce(now_ms);
        let payload = self.build_signed_payload(action, nonce).await?;
        match self.action_transport_for(action_label, hint) {
            HyperliquidActionTransport::Http => {
                if self.action_transport == HyperliquidActionTransport::WsPost {
                    eprintln!(
                        "HL_ACTION_FALLBACK submit_path=http_control_fallback request_class={} action_label={} batch_kind={} batch_size={}",
                        self.transport_hint_label(hint),
                        action_label,
                        batch_kind,
                        batch_size,
                    );
                }
                self.submit_signed_action_http(payload, action_label).await
            }
            HyperliquidActionTransport::WsPost => {
                self.submit_signed_action_ws_post(payload, action_label, batch_kind, batch_size)
                    .await
            }
        }
    }

    async fn submit_signed_action(
        &self,
        action: serde_json::Value,
        now_ms: TimestampMs,
        action_label: &'static str,
        batch_kind: &'static str,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        self.submit_signed_action_with_hint(
            action,
            now_ms,
            action_label,
            batch_kind,
            batch_size,
            TransportHint::Default,
        )
        .await
    }

    fn action_transport_for(
        &self,
        action_label: &'static str,
        hint: TransportHint,
    ) -> HyperliquidActionTransport {
        if self.action_transport != HyperliquidActionTransport::WsPost {
            return self.action_transport;
        }
        if action_label == "cancel_all" && hl_cancel_all_http_fallback_enabled() {
            return HyperliquidActionTransport::Http;
        }
        if hint.is_hyperliquid_sync_control() && hl_sync_control_http_fallback_enabled() {
            return HyperliquidActionTransport::Http;
        }
        self.action_transport
    }

    fn transport_hint_label(&self, hint: TransportHint) -> &'static str {
        match hint {
            TransportHint::Default => "default",
            TransportHint::HyperliquidSyncControl => "sync_control",
        }
    }

    fn next_action_nonce(&self, requested_ms: TimestampMs) -> u64 {
        let requested = if requested_ms > 0 {
            requested_ms as u64
        } else {
            now_ms() as u64
        };
        let mut observed = self.action_nonce.load(Ordering::Relaxed);
        loop {
            let nonce = observed.max(requested);
            match self.action_nonce.compare_exchange_weak(
                observed,
                nonce.saturating_add(1),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return nonce,
                Err(actual) => observed = actual,
            }
        }
    }

    async fn build_signed_payload(
        &self,
        action: serde_json::Value,
        nonce: u64,
    ) -> anyhow::Result<serde_json::Value> {
        self.build_signed_payload_with_vault_override(action, nonce, None)
            .await
    }

    async fn build_signed_payload_with_vault_override(
        &self,
        action: serde_json::Value,
        nonce: u64,
        vault_address_override: Option<&str>,
    ) -> anyhow::Result<serde_json::Value> {
        let sk = self
            .signing_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HL_PRIVATE_KEY is required for live orders"))?;
        let vault_address = if let Some(value) = vault_address_override {
            Some(normalize_hl_address(value)?)
        } else {
            self.effective_vault_address()
                .await?
                .map(|value| normalize_hl_address(&value))
                .transpose()?
        };
        let signature = sign_action(
            &action,
            vault_address.as_deref(),
            nonce as TimestampMs,
            matches!(self.cfg.network, HyperliquidNetwork::Mainnet),
            sk,
        )?;
        let mut payload = json!({
            "action": action,
            "nonce": nonce,
            "signature": signature,
        });
        if let Some(vault_address) = vault_address {
            payload["vaultAddress"] = json!(vault_address);
        }
        Ok(payload)
    }

    async fn submit_signed_action_http(
        &self,
        payload: serde_json::Value,
        action_label: &'static str,
    ) -> anyhow::Result<()> {
        self.submit_signed_action_http_json(payload, action_label)
            .await
            .map(|_| ())
    }

    async fn submit_signed_action_http_json(
        &self,
        payload: serde_json::Value,
        action_label: &'static str,
    ) -> anyhow::Result<serde_json::Value> {
        let rest_url = self.current_rest_url();
        let resp = self
            .http
            .post(rest_url.as_str())
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!(
                "Hyperliquid {action_label} failed status_code={} sanitized_body=true",
                status.as_u16()
            );
        }
        let body = resp.text().await.unwrap_or_default();
        let value: serde_json::Value = serde_json::from_str(&body).map_err(|_| {
            anyhow::anyhow!("Hyperliquid {action_label} invalid_json_response sanitized_body=true")
        })?;
        if let Some(error) = hyperliquid_exchange_response_error(&value) {
            anyhow::bail!(
                "Hyperliquid {action_label} rejected sanitized_reason={}",
                ws_post_exchange_error_reason(&error)
            );
        }
        Ok(value)
    }

    pub async fn reserve_request_weight(&self, weight: u64) -> anyhow::Result<serde_json::Value> {
        let action = build_reserve_request_weight_action(weight)?;
        let nonce = self.next_action_nonce(0);
        let payload = self.build_signed_payload(action, nonce).await?;
        self.submit_signed_action_http_json(payload, "reserve_request_weight")
            .await
    }

    pub async fn reserve_request_weight_for_vault_address(
        &self,
        weight: u64,
        vault_address: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let action = build_reserve_request_weight_action(weight)?;
        let nonce = self.next_action_nonce(0);
        let payload = self
            .build_signed_payload_with_vault_override(action, nonce, Some(vault_address))
            .await?;
        self.submit_signed_action_http_json(payload, "reserve_request_weight")
            .await
    }

    async fn submit_signed_action_ws_post(
        &self,
        payload: serde_json::Value,
        action_label: &'static str,
        batch_kind: &'static str,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        let post_id = self.post_request_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let (response_tx, response_rx) = oneshot::channel();
        self.post_request_tx
            .send(HyperliquidPostRequest {
                id: post_id,
                action_label,
                batch_kind,
                batch_size,
                payload,
                response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Hyperliquid post WS request channel closed"))?;
        match tokio::time::timeout(hl_ws_post_response_timeout(), response_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!(
                "Hyperliquid post WS response channel closed for action_label={action_label}"
            )),
            Err(_) => Err(anyhow::anyhow!(
                "Hyperliquid post WS response timeout after {:?} action_label={action_label}",
                hl_ws_post_response_timeout()
            )),
        }
    }

    fn fail_pending_post_requests(
        &self,
        pending: &mut BTreeMap<u64, HyperliquidPostPending>,
        message: String,
    ) {
        let count = pending.len() as u64;
        if count > 0 {
            self.post_inflight.fetch_sub(count, Ordering::Relaxed);
        }
        for (_id, pending_req) in std::mem::take(pending) {
            let _ = pending_req
                .response_tx
                .send(Err(anyhow::anyhow!(message.clone())));
        }
    }

    async fn get_asset_meta(&self) -> anyhow::Result<HyperliquidAssetMeta> {
        {
            let guard = self.asset_meta.lock().await;
            if let Some(meta) = *guard {
                return Ok(meta);
            }
        }
        let meta = fetch_asset_meta(&self.http, &self.cfg, &self.current_info_url()).await?;
        let mut guard = self.asset_meta.lock().await;
        *guard = Some(meta);
        Ok(meta)
    }

    async fn effective_vault_address(&self) -> anyhow::Result<Option<String>> {
        let Some(address) = self.cfg.vault_address.clone() else {
            return Ok(None);
        };

        {
            let guard = self.account_role.lock().await;
            if let Some(role) = *guard {
                return Ok(role.requires_vault_address().then_some(address));
            }
        }

        let role = fetch_user_role(&self.http, &self.current_info_url(), &address)
            .await?
            .unwrap_or(HyperliquidUserRole::Missing);
        let mut guard = self.account_role.lock().await;
        *guard = Some(role);
        Ok(role.requires_vault_address().then_some(address))
    }

    pub async fn fetch_account_snapshot(&self) -> anyhow::Result<AccountEvent> {
        let user =
            self.cfg.vault_address.clone().ok_or_else(|| {
                anyhow::anyhow!("HL_VAULT_ADDRESS is required for account polling")
            })?;
        let info_url = self.current_info_url();
        let value = post_info_request_json(
            &self.http,
            &info_url,
            &build_account_snapshot_request(&user),
        )
        .await
        .map_err(|err| anyhow::anyhow!("hyperliquid account snapshot error: {err}"))?;
        let mut snapshot = parse_account_snapshot_with_meta(
            &value,
            Some(self.cfg.coin.as_str()),
            self.cfg.venue_index,
        )
        .ok_or_else(|| anyhow::anyhow!("invalid account snapshot response"))?;

        // Unified-account / portfolio-margin API users expose collateral on the spot
        // clearinghouse endpoint; the perp clearinghouse balance fields are not meaningful for
        // sizing.
        if matches!(
            self.cached_user_abstraction(&user).await?,
            Some(HyperliquidAccountAbstraction::UnifiedAccount)
                | Some(HyperliquidAccountAbstraction::PortfolioMargin)
        ) {
            let spot_value = post_info_request_json(
                &self.http,
                &info_url,
                &build_spot_clearinghouse_state_request(&user),
            )
            .await
            .map_err(|err| anyhow::anyhow!("hyperliquid spot collateral snapshot error: {err}"))?;
            let spot_snapshot = parse_spot_collateral_snapshot(&spot_value).ok_or_else(|| {
                anyhow::anyhow!("invalid hyperliquid spot collateral snapshot response")
            })?;
            snapshot.timestamp_ms = snapshot.timestamp_ms.max(now_ms());
            snapshot.balances = spot_snapshot.balances;
            snapshot.margin = spot_snapshot.margin;
        }

        Ok(AccountEvent::Snapshot(snapshot))
    }

    async fn cached_user_abstraction(
        &self,
        user: &str,
    ) -> anyhow::Result<Option<HyperliquidAccountAbstraction>> {
        {
            let guard = self.account_abstraction.lock().await;
            if let Some(mode) = *guard {
                return Ok(Some(mode));
            }
        }

        let mode = fetch_user_abstraction(&self.http, &self.current_info_url(), user).await?;
        if let Some(mode) = mode {
            let mut guard = self.account_abstraction.lock().await;
            *guard = Some(mode);
        }
        Ok(mode)
    }
}

impl LiveRestClient for HyperliquidConnector {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let intent = OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: req.venue_index,
                venue_id: req.venue_id.as_str().into(),
                side: req.side,
                price: req.price,
                size: req.size,
                purpose: req.purpose,
                time_in_force: req.time_in_force,
                post_only: req.post_only,
                reduce_only: req.reduce_only,
                client_order_id: Some(req.client_order_id.clone()),
                phase51_target_key: None,
            });
            HyperliquidConnector::place_order(self, &intent, 0)
                .await
                .map(|_| LiveRestResponse {
                    order_id: None,
                    client_order_id: Some(req.client_order_id),
                })
                .map_err(map_rest_error)
        })
    }

    fn place_order_with_hint(
        &self,
        req: LiveRestPlaceRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let intent = OrderIntent::Place(crate::types::PlaceOrderIntent {
                venue_index: req.venue_index,
                venue_id: req.venue_id.as_str().into(),
                side: req.side,
                price: req.price,
                size: req.size,
                purpose: req.purpose,
                time_in_force: req.time_in_force,
                post_only: req.post_only,
                reduce_only: req.reduce_only,
                client_order_id: Some(req.client_order_id.clone()),
                phase51_target_key: None,
            });
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: Some(req.client_order_id.clone()),
                    client_order_id: Some(req.client_order_id),
                });
            }
            let asset_meta = self.get_asset_meta().await.map_err(map_rest_error)?;
            let action = build_action(&intent, asset_meta).map_err(map_rest_error)?;
            self.submit_signed_action_with_hint(action, 0, "order", "place_single", 1, hint)
                .await
                .map(|_| LiveRestResponse {
                    order_id: Some(req.client_order_id.clone()),
                    client_order_id: Some(req.client_order_id),
                })
                .map_err(map_rest_error)
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let intent = OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index: req.venue_index,
                venue_id: req.venue_id.as_str().into(),
                order_id: req.order_id.clone(),
            });
            HyperliquidConnector::cancel_order(self, &intent, 0)
                .await
                .map(|_| LiveRestResponse {
                    order_id: Some(req.order_id),
                    client_order_id: None,
                })
                .map_err(map_rest_error)
        })
    }

    fn cancel_order_with_hint(
        &self,
        req: LiveRestCancelRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let intent = OrderIntent::Cancel(crate::types::CancelOrderIntent {
                venue_index: req.venue_index,
                venue_id: req.venue_id.as_str().into(),
                order_id: req.order_id.clone(),
            });
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: Some(req.order_id),
                    client_order_id: None,
                });
            }
            let asset_index = self.get_asset_meta().await.map_err(map_rest_error)?.index;
            let action = build_cancel_action(&intent, asset_index).map_err(map_rest_error)?;
            self.submit_signed_action_with_hint(action, 0, "cancel", "cancel_single", 1, hint)
                .await
                .map(|_| LiveRestResponse {
                    order_id: Some(req.order_id),
                    client_order_id: None,
                })
                .map_err(map_rest_error)
        })
    }

    fn cancel_all(
        &self,
        _req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            HyperliquidConnector::cancel_all(self, 0)
                .await
                .map(|_| LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                })
                .map_err(map_rest_error)
        })
    }

    fn cancel_all_with_hint(
        &self,
        _req: LiveRestCancelAllRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                });
            }
            let asset_index = self.get_asset_meta().await.map_err(map_rest_error)?.index;
            let action = build_cancel_all_action(asset_index);
            self.submit_signed_action_with_hint(action, 0, "cancel_all", "cancel_all", 1, hint)
                .await
                .map(|_| LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                })
                .map_err(map_rest_error)
        })
    }

    fn replace_order(
        &self,
        req: LiveRestReplaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.replace_order_with_hint(req, TransportHint::Default)
    }

    fn replace_order_with_hint(
        &self,
        req: LiveRestReplaceRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if !req.post_only
                || req.reduce_only
                || req.purpose != OrderPurpose::Mm
                || !is_hyperliquid_cloid(&req.order_id)
            {
                return Err(LiveGatewayError::fatal(
                    "hyperliquid native replace unsupported for request",
                ));
            }
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: Some(req.client_order_id.clone()),
                    client_order_id: Some(req.client_order_id),
                });
            }
            let asset_meta = self.get_asset_meta().await.map_err(map_rest_error)?;
            let action = build_modify_action(&req, asset_meta).map_err(map_rest_error)?;
            self.submit_signed_action_with_hint(action, 0, "modify", "modify_alo", 1, hint)
                .await
                .map(|_| LiveRestResponse {
                    order_id: Some(req.client_order_id.clone()),
                    client_order_id: Some(req.client_order_id),
                })
                .map_err(map_rest_error)
        })
    }

    fn place_batch(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            if self.cfg.paper_mode {
                return reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.client_order_id),
                            client_order_id: None,
                        })
                    })
                    .collect();
            }

            let asset_meta = match self.get_asset_meta().await {
                Ok(asset_meta) => asset_meta,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let intents: Vec<crate::types::PlaceOrderIntent> = reqs
                .iter()
                .map(|req| crate::types::PlaceOrderIntent {
                    venue_index: req.venue_index,
                    venue_id: req.venue_id.as_str().into(),
                    side: req.side,
                    price: req.price,
                    size: req.size,
                    purpose: req.purpose,
                    time_in_force: req.time_in_force,
                    post_only: req.post_only,
                    reduce_only: req.reduce_only,
                    client_order_id: Some(req.client_order_id.clone()),
                    phase51_target_key: None,
                })
                .collect();
            let intent_refs: Vec<&crate::types::PlaceOrderIntent> = intents.iter().collect();
            let action = match build_batch_action(&intent_refs, asset_meta) {
                Ok(action) => action,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            match self
                .submit_signed_action(action, 0, "order_batch", "place_alo", reqs.len())
                .await
            {
                Ok(()) => reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.client_order_id),
                            client_order_id: None,
                        })
                    })
                    .collect(),
                Err(err) => {
                    let mapped = map_rest_error(err);
                    vec![Err(mapped); reqs.len()]
                }
            }
        })
    }

    fn place_batch_with_hint(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            if self.cfg.paper_mode {
                return reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.client_order_id),
                            client_order_id: None,
                        })
                    })
                    .collect();
            }

            let asset_meta = match self.get_asset_meta().await {
                Ok(asset_meta) => asset_meta,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let intents: Vec<crate::types::PlaceOrderIntent> = reqs
                .iter()
                .map(|req| crate::types::PlaceOrderIntent {
                    venue_index: req.venue_index,
                    venue_id: req.venue_id.as_str().into(),
                    side: req.side,
                    price: req.price,
                    size: req.size,
                    purpose: req.purpose,
                    time_in_force: req.time_in_force,
                    post_only: req.post_only,
                    reduce_only: req.reduce_only,
                    client_order_id: Some(req.client_order_id.clone()),
                    phase51_target_key: None,
                })
                .collect();
            let intent_refs: Vec<&crate::types::PlaceOrderIntent> = intents.iter().collect();
            let action = match build_batch_action(&intent_refs, asset_meta) {
                Ok(action) => action,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            match self
                .submit_signed_action_with_hint(
                    action,
                    0,
                    "order_batch",
                    "place_alo",
                    reqs.len(),
                    hint,
                )
                .await
            {
                Ok(()) => reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.client_order_id),
                            client_order_id: None,
                        })
                    })
                    .collect(),
                Err(err) => {
                    let mapped = map_rest_error(err);
                    vec![Err(mapped); reqs.len()]
                }
            }
        })
    }

    fn cancel_batch(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            if self.cfg.paper_mode {
                return reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.order_id),
                            client_order_id: None,
                        })
                    })
                    .collect();
            }

            let asset_index = match self.get_asset_meta().await {
                Ok(asset_meta) => asset_meta.index,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let intents: Vec<crate::types::CancelOrderIntent> = reqs
                .iter()
                .map(|req| crate::types::CancelOrderIntent {
                    venue_index: req.venue_index,
                    venue_id: req.venue_id.as_str().into(),
                    order_id: req.order_id.clone(),
                })
                .collect();
            let intent_refs: Vec<&crate::types::CancelOrderIntent> = intents.iter().collect();
            let (action, kind) = match build_batch_cancel_action(&intent_refs, asset_index) {
                Ok(result) => result,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let batch_kind = match kind {
                HyperliquidCancelBatchKind::Oid => "cancel_oid",
                HyperliquidCancelBatchKind::Cloid => "cancel_by_cloid",
            };
            match self
                .submit_signed_action(action, 0, "cancel_batch", batch_kind, reqs.len())
                .await
            {
                Ok(()) => reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.order_id),
                            client_order_id: None,
                        })
                    })
                    .collect(),
                Err(err) => {
                    let mapped = map_rest_error(err);
                    vec![Err(mapped); reqs.len()]
                }
            }
        })
    }

    fn cancel_batch_with_hint(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            if self.cfg.paper_mode {
                return reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.order_id),
                            client_order_id: None,
                        })
                    })
                    .collect();
            }

            let asset_index = match self.get_asset_meta().await {
                Ok(asset_meta) => asset_meta.index,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let intents: Vec<crate::types::CancelOrderIntent> = reqs
                .iter()
                .map(|req| crate::types::CancelOrderIntent {
                    venue_index: req.venue_index,
                    venue_id: req.venue_id.as_str().into(),
                    order_id: req.order_id.clone(),
                })
                .collect();
            let intent_refs: Vec<&crate::types::CancelOrderIntent> = intents.iter().collect();
            let (action, kind) = match build_batch_cancel_action(&intent_refs, asset_index) {
                Ok(result) => result,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let batch_kind = match kind {
                HyperliquidCancelBatchKind::Oid => "cancel_oid",
                HyperliquidCancelBatchKind::Cloid => "cancel_by_cloid",
            };
            match self
                .submit_signed_action_with_hint(
                    action,
                    0,
                    "cancel_batch",
                    batch_kind,
                    reqs.len(),
                    hint,
                )
                .await
            {
                Ok(()) => reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.order_id),
                            client_order_id: None,
                        })
                    })
                    .collect(),
                Err(err) => {
                    let mapped = map_rest_error(err);
                    vec![Err(mapped); reqs.len()]
                }
            }
        })
    }

    fn replace_batch(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.replace_batch_with_hint(reqs, TransportHint::Default)
    }

    fn replace_batch_with_hint(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            if reqs.iter().any(|req| {
                !req.post_only
                    || req.reduce_only
                    || req.purpose != OrderPurpose::Mm
                    || !is_hyperliquid_cloid(&req.order_id)
            }) {
                let err = LiveGatewayError::fatal(
                    "hyperliquid native replace batch unsupported for request",
                );
                return vec![Err(err); reqs.len()];
            }
            if self.cfg.paper_mode {
                return reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.client_order_id.clone()),
                            client_order_id: Some(req.client_order_id),
                        })
                    })
                    .collect();
            }
            let asset_meta = match self.get_asset_meta().await {
                Ok(asset_meta) => asset_meta,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            let action = match build_batch_modify_action(&reqs, asset_meta) {
                Ok(action) => action,
                Err(err) => return vec![Err(map_rest_error(err)); reqs.len()],
            };
            match self
                .submit_signed_action_with_hint(
                    action,
                    0,
                    "modify_batch",
                    "modify_alo",
                    reqs.len(),
                    hint,
                )
                .await
            {
                Ok(()) => reqs
                    .into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.client_order_id.clone()),
                            client_order_id: Some(req.client_order_id),
                        })
                    })
                    .collect(),
                Err(err) => {
                    let mapped = map_rest_error(err);
                    vec![Err(mapped); reqs.len()]
                }
            }
        })
    }
}

fn map_rest_error(err: anyhow::Error) -> LiveGatewayError {
    let msg = err.to_string();
    let lower = msg.to_lowercase();
    let sanitized = || {
        format!(
            "Hyperliquid live gateway error sanitized_reason={}",
            hyperliquid_rest_error_reason(&msg)
        )
    };
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(sanitized());
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(sanitized());
    }
    if lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(sanitized());
    }
    if lower.contains("timeout") || lower.contains("tempor") || lower.contains("retry") {
        return LiveGatewayError::retryable(sanitized());
    }
    LiveGatewayError::fatal(sanitized())
}

fn hyperliquid_rest_error_reason(raw: &str) -> &'static str {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("invalid") && lower.contains("order id") {
        "invalid_order_id"
    } else if lower.contains("timeout") {
        "timeout"
    } else if lower.contains("tempor") || lower.contains("retry") {
        "retryable"
    } else if lower.contains("reduce") && lower.contains("only") {
        "reduce_only_violation"
    } else {
        ws_post_exchange_error_reason(raw)
    }
}

#[derive(Debug, Clone)]
pub struct ParsedL2Message {
    pub event: MarketDataEvent,
    pub seq: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2SeqTracker {
    last_seq: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct L2SeqOutcome {
    pub event: Option<MarketDataEvent>,
    pub refresh_snapshot: Option<u64>,
}

impl L2SeqTracker {
    pub fn new() -> Self {
        Self { last_seq: None }
    }

    pub fn on_message(&mut self, msg: ParsedL2Message) -> L2SeqOutcome {
        let mut refresh_snapshot = None;
        if let Some(prev) = self.last_seq {
            if msg.seq > prev + 1 {
                refresh_snapshot = Some(msg.seq);
            } else if msg.seq <= prev {
                return L2SeqOutcome {
                    event: None,
                    refresh_snapshot: None,
                };
            }
        }
        self.last_seq = Some(msg.seq);
        L2SeqOutcome {
            event: Some(msg.event),
            refresh_snapshot,
        }
    }
}

pub fn parse_l2_message(text: &str, venue_index: usize) -> Option<ParsedL2Message> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    parse_l2_message_value(&value, venue_index)
}

fn parse_l2_message_value(
    value: &serde_json::Value,
    venue_index: usize,
) -> Option<ParsedL2Message> {
    let channel = value.get("channel")?.as_str()?;
    if channel != "l2Book" {
        return None;
    }
    let data = value.get("data")?;
    let seq = data.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let coin = data
        .get("coin")
        .and_then(|v| v.as_str())
        .unwrap_or("UNKNOWN");
    let venue_id = coin.to_string();
    let timestamp_ms = data.get("time").and_then(|v| v.as_i64()).unwrap_or(0);
    if let Some(levels) = data.get("levels") {
        let bids = parse_levels(levels.get(0)?)?;
        let asks = parse_levels(levels.get(1)?)?;
        let snapshot = super::super::types::L2Snapshot {
            venue_index,
            venue_id,
            seq,
            timestamp_ms,
            bids,
            asks,
        };
        return Some(ParsedL2Message {
            event: MarketDataEvent::L2Snapshot(snapshot),
            seq,
        });
    }
    if let Some(changes) = data.get("changes") {
        let deltas = parse_deltas(changes)?;
        let delta = super::super::types::L2Delta {
            venue_index,
            venue_id,
            seq,
            timestamp_ms,
            changes: deltas,
        };
        return Some(ParsedL2Message {
            event: MarketDataEvent::L2Delta(delta),
            seq,
        });
    }
    None
}

#[derive(Debug, Clone)]
pub struct HyperliquidFixtureFeed {
    messages: Vec<String>,
}

impl HyperliquidFixtureFeed {
    pub fn from_files(paths: &[PathBuf]) -> std::io::Result<Self> {
        let mut messages = Vec::new();
        for path in paths {
            messages.push(std::fs::read_to_string(path)?);
        }
        Ok(Self { messages })
    }

    pub fn from_dir(dir: &Path) -> std::io::Result<Self> {
        let snapshot = dir.join("ws_l2_snapshot.json");
        let delta = dir.join("ws_l2_delta.json");
        Self::from_files(&[snapshot, delta])
    }

    pub async fn run_ticks(
        &self,
        market_tx: mpsc::Sender<MarketDataEvent>,
        venue_index: usize,
        start_ms: TimestampMs,
        step_ms: i64,
        ticks: u64,
    ) {
        let pace_ticks = std::env::var("PARAPHINA_PAPER_USE_WALLCLOCK_TS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let sleep_duration = Duration::from_millis(step_ms.max(1) as u64);
        let mut seq: u64 = 1;
        for tick in 0..ticks {
            let now_ms = start_ms + step_ms.saturating_mul(tick as i64);
            for raw in &self.messages {
                if let Some(parsed) = parse_l2_message(raw, venue_index) {
                    let event = override_market_event(parsed.event, seq, now_ms);
                    let event = apply_fixture_tick_variation(event, tick);
                    seq = seq.wrapping_add(1);
                    let _ = market_tx.send(event).await;
                }
            }
            if pace_ticks {
                tokio::time::sleep(sleep_duration).await;
            } else {
                tokio::task::yield_now().await;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HyperliquidAccountFixtureFeed {
    snapshots: Vec<String>,
}

impl HyperliquidAccountFixtureFeed {
    pub fn from_dir(dir: &Path) -> std::io::Result<Self> {
        let snapshot = dir.join("rest_account_snapshot.json");
        let snapshots = vec![std::fs::read_to_string(snapshot)?];
        Ok(Self { snapshots })
    }

    pub async fn run_ticks(
        &self,
        account_tx: mpsc::Sender<AccountEvent>,
        start_ms: TimestampMs,
        step_ms: i64,
        ticks: u64,
    ) {
        let pace_ticks = std::env::var("PARAPHINA_PAPER_USE_WALLCLOCK_TS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let sleep_duration = Duration::from_millis(step_ms.max(1) as u64);
        let mut seq: u64 = 1;
        for tick in 0..ticks {
            let now_ms = start_ms + step_ms.saturating_mul(tick as i64);
            for raw in &self.snapshots {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
                    if let Some(mut snapshot) = parse_account_snapshot(&value) {
                        snapshot.seq = seq;
                        snapshot.timestamp_ms = now_ms;
                        seq = seq.wrapping_add(1);
                        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;
                    }
                }
            }
            if pace_ticks {
                tokio::time::sleep(sleep_duration).await;
            } else {
                tokio::task::yield_now().await;
            }
        }
    }
}

fn override_market_event(
    event: MarketDataEvent,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> MarketDataEvent {
    match event {
        MarketDataEvent::L2Snapshot(mut snap) => {
            snap.seq = seq;
            snap.timestamp_ms = timestamp_ms;
            MarketDataEvent::L2Snapshot(snap)
        }
        MarketDataEvent::L2Delta(mut delta) => {
            delta.seq = seq;
            delta.timestamp_ms = timestamp_ms;
            MarketDataEvent::L2Delta(delta)
        }
        MarketDataEvent::Trade(mut trade) => {
            trade.seq = seq;
            trade.timestamp_ms = timestamp_ms;
            MarketDataEvent::Trade(trade)
        }
        MarketDataEvent::FundingUpdate(mut funding) => {
            funding.seq = seq;
            funding.timestamp_ms = timestamp_ms;
            if funding.received_ms.is_none() {
                funding.received_ms = Some(timestamp_ms);
            }
            MarketDataEvent::FundingUpdate(funding)
        }
    }
}

fn apply_fixture_tick_variation(event: MarketDataEvent, tick: u64) -> MarketDataEvent {
    match event {
        MarketDataEvent::L2Delta(mut delta) => {
            if let Some(change) = delta
                .changes
                .iter_mut()
                .find(|level| matches!(level.side, BookSide::Ask))
            {
                // Alternate best-ask removal/restoration to create deterministic mid changes.
                change.size = if tick % 2 == 0 { 0.0 } else { 1.5 };
            }
            MarketDataEvent::L2Delta(delta)
        }
        _ => event,
    }
}

fn parse_levels(levels: &serde_json::Value) -> Option<Vec<BookLevel>> {
    let mut out = Vec::new();
    for level in levels.as_array()? {
        let level = parse_level_entry(level)?;
        out.push(level);
    }
    Some(out)
}

/// Result of resilient level parsing: parsed levels + count of skipped invalid entries.
#[derive(Debug, Clone)]
struct ResilientLevelsResult {
    levels: Vec<BookLevel>,
    skipped_count: usize,
    total_count: usize,
}

/// Parse levels resiliently: skip invalid entries instead of failing entirely.
/// Returns None only if the input is not an array.
fn parse_levels_resilient(levels: &serde_json::Value) -> Option<ResilientLevelsResult> {
    let arr = levels.as_array()?;
    let total_count = arr.len();
    let mut out = Vec::with_capacity(total_count);
    let mut skipped_count = 0;
    for level in arr {
        if let Some(parsed) = parse_level_entry(level) {
            out.push(parsed);
        } else {
            skipped_count += 1;
        }
    }
    Some(ResilientLevelsResult {
        levels: out,
        skipped_count,
        total_count,
    })
}

fn parse_level_entry(level: &serde_json::Value) -> Option<BookLevel> {
    if let Some(items) = level.as_array() {
        let price = parse_f64_value(items.get(0)?)?;
        let size = parse_f64_value(items.get(1)?)?;
        return Some(BookLevel { price, size });
    }
    let obj = level.as_object()?;
    let price = parse_f64_value(obj.get("px")?)?;
    let size = parse_f64_value(obj.get("sz")?)?;
    Some(BookLevel { price, size })
}

fn parse_deltas(changes: &serde_json::Value) -> Option<Vec<BookLevelDelta>> {
    let mut out = Vec::new();
    for change in changes.as_array()? {
        let side_raw = change.get(0)?.as_str()?;
        let side = match side_raw {
            "b" | "bid" | "Bid" => BookSide::Bid,
            "a" | "ask" | "Ask" => BookSide::Ask,
            _ => return None,
        };
        let price = parse_f64_value(change.get(1)?)?;
        let size = parse_f64_value(change.get(2)?)?;
        out.push(BookLevelDelta { side, price, size });
    }
    Some(out)
}

fn parse_f64_value(value: &serde_json::Value) -> Option<f64> {
    if let Some(raw) = value.as_f64() {
        return Some(raw);
    }
    if let Some(raw) = value.as_str() {
        return raw.parse::<f64>().ok();
    }
    None
}

fn parse_i64_value(value: &serde_json::Value) -> Option<i64> {
    if let Some(raw) = value.as_i64() {
        return Some(raw);
    }
    if let Some(raw) = value.as_f64() {
        return Some(raw as i64);
    }
    if let Some(raw) = value.as_str() {
        return raw.parse::<i64>().ok();
    }
    None
}

fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as TimestampMs
}

fn decode_l2book_top(value: &serde_json::Value) -> Option<TopOfBook> {
    let data = value.get("data")?;
    let levels = data.get("levels")?;
    let bids = levels.get(0)?.as_array()?;
    let asks = levels.get(1)?.as_array()?;
    let bid = bids.first()?.as_object()?;
    let ask = asks.first()?.as_object()?;
    let bid_px = parse_f64_value(bid.get("px")?)?;
    let bid_sz = parse_f64_value(bid.get("sz")?)?;
    let ask_px = parse_f64_value(ask.get("px")?)?;
    let ask_sz = parse_f64_value(ask.get("sz")?)?;
    if bid_sz <= 0.0 || ask_sz <= 0.0 {
        return None;
    }
    if !bid_px.is_finite() || !bid_sz.is_finite() || !ask_px.is_finite() || !ask_sz.is_finite() {
        return None;
    }
    let timestamp_ms = data.get("time").and_then(|v| v.as_i64());
    Some(TopOfBook {
        best_bid_px: bid_px,
        best_bid_sz: bid_sz,
        best_ask_px: ask_px,
        best_ask_sz: ask_sz,
        timestamp_ms,
    })
}

/// Result of decode_l2book_snapshot with diagnostic info for logging.
#[derive(Debug)]
struct SnapshotDecodeResult {
    event: Option<MarketDataEvent>,
    bid_skipped: usize,
    ask_skipped: usize,
    bid_total: usize,
    ask_total: usize,
    /// If event is None, this explains why.
    failure_reason: Option<&'static str>,
}

fn decode_l2book_snapshot_resilient(
    value: &serde_json::Value,
    venue_index: usize,
    default_coin: &str,
    fallback_seq: &mut u64,
) -> SnapshotDecodeResult {
    let data = match value.get("data") {
        Some(d) => d,
        None => {
            return SnapshotDecodeResult {
                event: None,
                bid_skipped: 0,
                ask_skipped: 0,
                bid_total: 0,
                ask_total: 0,
                failure_reason: Some("missing 'data' field"),
            };
        }
    };
    let levels = match data.get("levels") {
        Some(l) => l,
        None => {
            return SnapshotDecodeResult {
                event: None,
                bid_skipped: 0,
                ask_skipped: 0,
                bid_total: 0,
                ask_total: 0,
                failure_reason: Some("missing 'levels' field"),
            };
        }
    };
    let bid_levels_value = match levels.get(0) {
        Some(b) => b,
        None => {
            return SnapshotDecodeResult {
                event: None,
                bid_skipped: 0,
                ask_skipped: 0,
                bid_total: 0,
                ask_total: 0,
                failure_reason: Some("missing bid levels at index 0"),
            };
        }
    };
    let ask_levels_value = match levels.get(1) {
        Some(a) => a,
        None => {
            return SnapshotDecodeResult {
                event: None,
                bid_skipped: 0,
                ask_skipped: 0,
                bid_total: 0,
                ask_total: 0,
                failure_reason: Some("missing ask levels at index 1"),
            };
        }
    };

    // Parse levels resiliently (skip invalid entries)
    let bid_result = match parse_levels_resilient(bid_levels_value) {
        Some(r) => r,
        None => {
            return SnapshotDecodeResult {
                event: None,
                bid_skipped: 0,
                ask_skipped: 0,
                bid_total: 0,
                ask_total: 0,
                failure_reason: Some("bid levels not an array"),
            };
        }
    };
    let ask_result = match parse_levels_resilient(ask_levels_value) {
        Some(r) => r,
        None => {
            return SnapshotDecodeResult {
                event: None,
                bid_skipped: bid_result.skipped_count,
                ask_skipped: 0,
                bid_total: bid_result.total_count,
                ask_total: 0,
                failure_reason: Some("ask levels not an array"),
            };
        }
    };

    // Require at least 1 valid bid AND 1 valid ask to produce a snapshot
    if bid_result.levels.is_empty() || ask_result.levels.is_empty() {
        return SnapshotDecodeResult {
            event: None,
            bid_skipped: bid_result.skipped_count,
            ask_skipped: ask_result.skipped_count,
            bid_total: bid_result.total_count,
            ask_total: ask_result.total_count,
            failure_reason: Some("no valid levels on at least one side"),
        };
    }

    let seq = data.get("seq").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        *fallback_seq = fallback_seq.wrapping_add(1);
        *fallback_seq
    });
    let timestamp_ms = data.get("time").and_then(|v| v.as_i64()).unwrap_or(0);
    let venue_id = data
        .get("coin")
        .and_then(|v| v.as_str())
        .unwrap_or(default_coin)
        .to_string();

    SnapshotDecodeResult {
        event: Some(MarketDataEvent::L2Snapshot(
            super::super::types::L2Snapshot {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                bids: bid_result.levels,
                asks: ask_result.levels,
            },
        )),
        bid_skipped: bid_result.skipped_count,
        ask_skipped: ask_result.skipped_count,
        bid_total: bid_result.total_count,
        ask_total: ask_result.total_count,
        failure_reason: None,
    }
}

fn log_decode_miss(venue: &str, value: &serde_json::Value, payload: &str, count: usize, url: &str) {
    let keys = value
        .as_object()
        .map(|obj| {
            let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
            keys.sort();
            format!("[{}]", keys.join(","))
        })
        .unwrap_or_else(|| "[non-object]".to_string());
    let snippet: String = payload.chars().take(160).collect();
    eprintln!(
        "WARN: {venue} WS decode miss keys={keys} snippet={snippet} (count={count}) url={url}",
    );
}

pub async fn fetch_l2_snapshot(
    client: &Client,
    cfg: &HyperliquidConfig,
    info_url: &str,
) -> anyhow::Result<MarketDataEvent> {
    let payload = json!({
        "type": "l2Book",
        "coin": cfg.coin,
        "nSigFigs": cfg.n_sig_figs,
        "nLevels": cfg.n_levels
    });
    let resp = client.post(info_url).json(&payload).send().await?;
    let value: serde_json::Value = resp.json().await?;
    let seq = value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = value.get("time").and_then(|v| v.as_i64()).unwrap_or(0);
    let bids_value = value
        .get("levels")
        .and_then(|v| v.get(0))
        .ok_or_else(|| anyhow::anyhow!("missing bids"))?;
    let asks_value = value
        .get("levels")
        .and_then(|v| v.get(1))
        .ok_or_else(|| anyhow::anyhow!("missing asks"))?;
    let bids = parse_levels(bids_value).ok_or_else(|| anyhow::anyhow!("invalid bids"))?;
    let asks = parse_levels(asks_value).ok_or_else(|| anyhow::anyhow!("invalid asks"))?;
    let snapshot = super::super::types::L2Snapshot {
        venue_index: cfg.venue_index,
        venue_id: cfg.coin.clone(),
        seq,
        timestamp_ms,
        bids,
        asks,
    };
    Ok(MarketDataEvent::L2Snapshot(snapshot))
}

async fn fetch_asset_meta(
    client: &Client,
    cfg: &HyperliquidConfig,
    info_url: &str,
) -> anyhow::Result<HyperliquidAssetMeta> {
    let payload = json!({ "type": "meta" });
    let resp = client.post(info_url).json(&payload).send().await?;
    let value: serde_json::Value = resp.json().await?;
    let universe = value
        .get("universe")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing universe in meta response"))?;
    for (idx, entry) in universe.iter().enumerate() {
        if entry
            .get("name")
            .and_then(|v| v.as_str())
            .map(|name| name.eq_ignore_ascii_case(&cfg.coin))
            .unwrap_or(false)
        {
            let sz_decimals = entry
                .get("szDecimals")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing szDecimals for coin {}", cfg.coin))?
                as u32;
            return Ok(HyperliquidAssetMeta {
                index: idx as u32,
                sz_decimals,
            });
        }
    }
    anyhow::bail!("coin {} not found in Hyperliquid universe", cfg.coin);
}

fn build_account_snapshot_request(user: &str) -> serde_json::Value {
    json!({ "type": "clearinghouseState", "user": user })
}

fn build_private_subscriptions(account_user: Option<&str>) -> Vec<serde_json::Value> {
    let mut subscriptions = vec![
        json!({
            "method": "subscribe",
            "subscription": { "type": "userFills" }
        }),
        json!({
            "method": "subscribe",
            "subscription": { "type": "userEvents" }
        }),
    ];
    if let Some(user) = account_user {
        subscriptions.push(json!({
            "method": "subscribe",
            "subscription": { "type": "clearinghouseState", "user": user }
        }));
    }
    subscriptions
}

fn build_spot_clearinghouse_state_request(user: &str) -> serde_json::Value {
    json!({ "type": "spotClearinghouseState", "user": user })
}

pub async fn fetch_user_rate_limit(
    client: &Client,
    info_url: &str,
    user: &str,
) -> anyhow::Result<serde_json::Value> {
    post_info_request_json(
        client,
        info_url,
        &json!({
            "type": "userRateLimit",
            "user": user,
        }),
    )
    .await
}

pub async fn fetch_clearinghouse_state(
    client: &Client,
    info_url: &str,
    user: &str,
) -> anyhow::Result<serde_json::Value> {
    post_info_request_json(client, info_url, &build_account_snapshot_request(user)).await
}

pub async fn fetch_user_abstraction_raw(
    client: &Client,
    info_url: &str,
    user: &str,
) -> anyhow::Result<serde_json::Value> {
    post_info_request_json(
        client,
        info_url,
        &json!({
            "type": "userAbstraction",
            "user": user,
        }),
    )
    .await
}

pub async fn fetch_spot_clearinghouse_state(
    client: &Client,
    info_url: &str,
    user: &str,
) -> anyhow::Result<serde_json::Value> {
    post_info_request_json(
        client,
        info_url,
        &build_spot_clearinghouse_state_request(user),
    )
    .await
}

pub fn summarize_clearinghouse_state(value: &serde_json::Value) -> serde_json::Value {
    json!({
        "marginSummary": value.get("marginSummary").cloned().unwrap_or(serde_json::Value::Null),
        "crossMarginSummary": value.get("crossMarginSummary").cloned().unwrap_or(serde_json::Value::Null),
        "withdrawable": value.get("withdrawable").cloned().unwrap_or(serde_json::Value::Null),
    })
}

pub fn summarize_spot_clearinghouse_state(value: &serde_json::Value) -> serde_json::Value {
    let usdc = value
        .get("balances")
        .and_then(|balances| balances.as_array())
        .and_then(|balances| {
            balances
                .iter()
                .find(|balance| balance.get("coin").and_then(|coin| coin.as_str()) == Some("USDC"))
        })
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    json!({
        "usdc": usdc,
        "tokenToAvailableAfterMaintenance": value
            .get("tokenToAvailableAfterMaintenance")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    })
}

async fn post_info_request_json(
    client: &Client,
    info_url: &str,
    payload: &serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let resp = client.post(info_url).json(payload).send().await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("http {}: {}", status, body);
    }
    Ok(serde_json::from_str(&body)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperliquidAccountAbstraction {
    Standard,
    UnifiedAccount,
    PortfolioMargin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperliquidUserRole {
    Missing,
    User,
    Agent,
    Vault,
    SubAccount,
}

impl HyperliquidUserRole {
    fn requires_vault_address(self) -> bool {
        matches!(self, Self::Vault | Self::SubAccount)
    }
}

async fn fetch_user_abstraction(
    client: &Client,
    info_url: &str,
    user: &str,
) -> anyhow::Result<Option<HyperliquidAccountAbstraction>> {
    let value = post_info_request_json(
        client,
        info_url,
        &json!({
            "type": "userAbstraction",
            "user": user,
        }),
    )
    .await?;
    Ok(parse_user_abstraction(&value))
}

fn parse_user_abstraction(value: &serde_json::Value) -> Option<HyperliquidAccountAbstraction> {
    match value.as_str()? {
        "standard" => Some(HyperliquidAccountAbstraction::Standard),
        "unifiedAccount" => Some(HyperliquidAccountAbstraction::UnifiedAccount),
        "portfolioMargin" => Some(HyperliquidAccountAbstraction::PortfolioMargin),
        _ => None,
    }
}

async fn fetch_user_role(
    client: &Client,
    info_url: &str,
    user: &str,
) -> anyhow::Result<Option<HyperliquidUserRole>> {
    let value = post_info_request_json(
        client,
        info_url,
        &json!({
            "type": "userRole",
            "user": user,
        }),
    )
    .await?;
    Ok(parse_user_role(&value))
}

fn parse_user_role(value: &serde_json::Value) -> Option<HyperliquidUserRole> {
    match value.get("role")?.as_str()? {
        "missing" => Some(HyperliquidUserRole::Missing),
        "user" => Some(HyperliquidUserRole::User),
        "agent" => Some(HyperliquidUserRole::Agent),
        "vault" => Some(HyperliquidUserRole::Vault),
        "subAccount" => Some(HyperliquidUserRole::SubAccount),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SpotCollateralSnapshot {
    balances: Vec<BalanceSnapshot>,
    margin: MarginSnapshot,
}

fn parse_spot_collateral_snapshot(data: &serde_json::Value) -> Option<SpotCollateralSnapshot> {
    let balances = data.get("balances")?.as_array()?;
    let usdc = balances.iter().find(|entry| {
        entry
            .get("token")
            .and_then(|v| v.as_u64())
            .map(|token| token == 0)
            .unwrap_or(false)
            || entry
                .get("coin")
                .and_then(|v| v.as_str())
                .map(|coin| coin.eq_ignore_ascii_case("USDC"))
                .unwrap_or(false)
    })?;

    let total = usdc.get("total").and_then(parse_f64_value)?;
    let hold = usdc.get("hold").and_then(parse_f64_value).unwrap_or(0.0);
    let available = data
        .get("tokenToAvailableAfterMaintenance")
        .and_then(|v| v.as_array())
        .and_then(|entries| {
            entries.iter().find_map(|entry| {
                let pair = entry.as_array()?;
                if pair.len() != 2 {
                    return None;
                }
                let token = pair[0].as_u64()?;
                if token != 0 {
                    return None;
                }
                parse_f64_value(&pair[1])
            })
        })
        .unwrap_or_else(|| (total - hold).max(0.0));
    let used = (total - available).max(0.0);

    Some(SpotCollateralSnapshot {
        balances: vec![BalanceSnapshot {
            asset: "USDC".to_string(),
            total,
            available,
        }],
        margin: MarginSnapshot {
            balance_usd: total,
            used_usd: used,
            available_usd: available,
        },
    })
}

async fn fetch_public_funding(
    client: &Client,
    cfg: &HyperliquidConfig,
    info_url: &str,
) -> anyhow::Result<FundingUpdate> {
    let payload = json!({ "type": "metaAndAssetCtxs" });
    let resp = client.post(info_url).json(&payload).send().await?;
    let value: serde_json::Value = resp.json().await?;
    parse_public_funding(&value, cfg)
        .ok_or_else(|| anyhow::anyhow!("invalid public funding response for coin={}", cfg.coin))
}

fn parse_public_funding(
    value: &serde_json::Value,
    cfg: &HyperliquidConfig,
) -> Option<FundingUpdate> {
    let now_ms = now_ms();
    let mut universe: Option<&Vec<serde_json::Value>> = None;
    let mut ctxs: Option<&Vec<serde_json::Value>> = None;
    let mut as_of_ms: Option<i64> = None;

    if let Some(obj) = value.as_object() {
        universe = obj.get("universe").and_then(|v| v.as_array());
        ctxs = obj.get("assetCtxs").and_then(|v| v.as_array());
        as_of_ms = obj.get("time").and_then(parse_i64_value);
    } else if let Some(arr) = value.as_array() {
        if let Some(meta) = arr.get(0).and_then(|v| v.as_object()) {
            universe = meta.get("universe").and_then(|v| v.as_array());
            as_of_ms = meta.get("time").and_then(parse_i64_value);
        }
        if let Some(ctx) = arr.get(1) {
            ctxs = ctx
                .get("assetCtxs")
                .and_then(|v| v.as_array())
                .or_else(|| ctx.as_array());
        }
    }

    let universe = universe?;
    let ctxs = ctxs?;
    let idx = universe.iter().position(|entry| {
        entry
            .get("name")
            .and_then(|v| v.as_str())
            .map(|name| name.eq_ignore_ascii_case(&cfg.coin))
            .unwrap_or(false)
    })?;
    let ctx = ctxs.get(idx)?;

    let funding_rate = ctx
        .get("funding8h")
        .or_else(|| ctx.get("funding"))
        .or_else(|| ctx.get("fundingRate"))
        .or_else(|| ctx.get("fundingRate8h"))
        .and_then(parse_f64_value);

    let interval_sec = ctx
        .get("fundingIntervalSec")
        .or_else(|| ctx.get("fundingInterval"))
        .and_then(parse_i64_value)
        .and_then(|v| if v > 0 { Some(v as u64) } else { None })
        .or(Some(8 * 60 * 60));

    let next_funding_ms = ctx
        .get("nextFundingTime")
        .or_else(|| ctx.get("nextFundingTimestamp"))
        .or_else(|| ctx.get("nextFundingMs"))
        .and_then(parse_i64_value);

    Some(FundingUpdate {
        venue_index: cfg.venue_index,
        venue_id: cfg.coin.clone(),
        seq: 0,
        timestamp_ms: as_of_ms.unwrap_or(now_ms),
        received_ms: Some(now_ms),
        funding_rate_8h: funding_rate,
        funding_rate_native: funding_rate,
        interval_sec,
        next_funding_ms,
        settlement_price_kind: Some(SettlementPriceKind::Mark), // HL settles funding at mark price
        source: FundingSource::MarketDataRest,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlLimitOrderType {
    tif: String,
}

#[derive(Debug, Clone, Copy)]
struct HyperliquidAssetMeta {
    index: u32,
    sz_decimals: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HlTriggerOrderType {
    is_market: bool,
    trigger_px: String,
    tpsl: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum HlOrderType {
    Limit { limit: HlLimitOrderType },
    Trigger { trigger: HlTriggerOrderType },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlOrderWire {
    a: u32,
    b: bool,
    p: String,
    s: String,
    r: bool,
    t: HlOrderType,
    #[serde(skip_serializing_if = "Option::is_none")]
    c: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlOrderAction {
    #[serde(rename = "type")]
    action_type: String,
    orders: Vec<HlOrderWire>,
    grouping: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    builder: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlCancelOidWire {
    a: u32,
    o: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlCancelOidAction {
    #[serde(rename = "type")]
    action_type: String,
    cancels: Vec<HlCancelOidWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlCancelCloidWire {
    asset: u32,
    cloid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlCancelCloidAction {
    #[serde(rename = "type")]
    action_type: String,
    cancels: Vec<HlCancelCloidWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlCancelAllAction {
    #[serde(rename = "type")]
    action_type: String,
    asset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlNoopAction {
    #[serde(rename = "type")]
    action_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlReserveRequestWeightAction {
    #[serde(rename = "type")]
    action_type: String,
    weight: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlModifyAction {
    #[serde(rename = "type")]
    action_type: String,
    oid: serde_json::Value,
    order: HlOrderWire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlBatchModifyWire {
    oid: serde_json::Value,
    order: HlOrderWire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HlBatchModifyAction {
    #[serde(rename = "type")]
    action_type: String,
    modifies: Vec<HlBatchModifyWire>,
}

fn build_hl_order_wire(
    place: &crate::types::PlaceOrderIntent,
    asset_meta: HyperliquidAssetMeta,
) -> HlOrderWire {
    let tif = if place.post_only {
        "Alo"
    } else {
        match place.time_in_force {
            TimeInForce::Gtc => "Gtc",
            TimeInForce::Ioc => "Ioc",
        }
    };
    let price = quantize_hl_price(place.price, place.side, asset_meta);
    HlOrderWire {
        a: asset_meta.index,
        b: place.side == Side::Buy,
        // Match Hyperliquid SDK wire formatting after snapping to the venue's
        // valid price grid (5 significant figures and max venue decimals).
        p: hl_float_to_wire(price),
        s: hl_float_to_wire(place.size),
        r: place.reduce_only,
        t: HlOrderType::Limit {
            limit: HlLimitOrderType {
                tif: tif.to_string(),
            },
        },
        c: place.client_order_id.clone(),
    }
}

fn hl_float_to_wire(value: f64) -> String {
    let mut wire = format!("{:.8}", value);
    while wire.ends_with('0') {
        wire.pop();
    }
    if wire.ends_with('.') {
        wire.pop();
    }
    if wire == "-0" {
        "0".to_string()
    } else {
        wire
    }
}

fn max_hl_price_decimals(sz_decimals: u32, asset_index: u32) -> u32 {
    let max_decimals: u32 = if asset_index < 10_000 { 6 } else { 8 };
    max_decimals.saturating_sub(sz_decimals)
}

fn quantize_hl_price(value: f64, side: Side, asset_meta: HyperliquidAssetMeta) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }

    let price_decimals = max_hl_price_decimals(asset_meta.sz_decimals, asset_meta.index);
    let decimal_step = 10f64.powi(-(price_decimals as i32));
    let magnitude = value.abs().log10().floor() as i32;
    let sig_step = 10f64.powi(magnitude - 5 + 1);
    let step = decimal_step.max(sig_step);
    let scaled = value / step;
    let snapped = match side {
        Side::Buy => scaled.floor() * step,
        Side::Sell => scaled.ceil() * step,
    };

    if snapped == -0.0 {
        0.0
    } else {
        snapped
    }
}

fn build_action(
    intent: &OrderIntent,
    asset_meta: HyperliquidAssetMeta,
) -> anyhow::Result<serde_json::Value> {
    let OrderIntent::Place(place) = intent else {
        anyhow::bail!("intent not a place order");
    };
    let action = HlOrderAction {
        action_type: "order".to_string(),
        orders: vec![build_hl_order_wire(place, asset_meta)],
        grouping: "na".to_string(),
        builder: None,
    };
    Ok(serde_json::to_value(action)?)
}

/// Build a batch order action with N orders in a single API call.
/// Weight cost: 1 + floor(N/40). For N<=39, cost is weight 1 (same as a single order).
fn build_batch_action(
    places: &[&crate::types::PlaceOrderIntent],
    asset_meta: HyperliquidAssetMeta,
) -> anyhow::Result<serde_json::Value> {
    let action = HlOrderAction {
        action_type: "order".to_string(),
        orders: places
            .iter()
            .map(|place| build_hl_order_wire(place, asset_meta))
            .collect(),
        grouping: "na".to_string(),
        builder: None,
    };
    Ok(serde_json::to_value(action)?)
}

fn build_modify_oid_value(order_id: &str) -> anyhow::Result<serde_json::Value> {
    if is_hyperliquid_cloid(order_id) {
        return Ok(serde_json::Value::String(order_id.to_string()));
    }
    Ok(serde_json::Value::Number(parse_hl_oid(order_id)?.into()))
}

fn build_modify_action(
    req: &LiveRestReplaceRequest,
    asset_meta: HyperliquidAssetMeta,
) -> anyhow::Result<serde_json::Value> {
    let place = crate::types::PlaceOrderIntent {
        venue_index: req.venue_index,
        venue_id: req.venue_id.as_str().into(),
        side: req.side,
        price: req.price,
        size: req.size,
        purpose: req.purpose,
        time_in_force: req.time_in_force,
        post_only: req.post_only,
        reduce_only: req.reduce_only,
        client_order_id: Some(req.client_order_id.clone()),
        phase51_target_key: None,
    };
    let action = HlModifyAction {
        action_type: "modify".to_string(),
        oid: build_modify_oid_value(&req.order_id)?,
        order: build_hl_order_wire(&place, asset_meta),
    };
    Ok(serde_json::to_value(action)?)
}

fn build_batch_modify_action(
    reqs: &[LiveRestReplaceRequest],
    asset_meta: HyperliquidAssetMeta,
) -> anyhow::Result<serde_json::Value> {
    let modifies = reqs
        .iter()
        .map(|req| {
            let place = crate::types::PlaceOrderIntent {
                venue_index: req.venue_index,
                venue_id: req.venue_id.as_str().into(),
                side: req.side,
                price: req.price,
                size: req.size,
                purpose: req.purpose,
                time_in_force: req.time_in_force,
                post_only: req.post_only,
                reduce_only: req.reduce_only,
                client_order_id: Some(req.client_order_id.clone()),
                phase51_target_key: None,
            };
            Ok(HlBatchModifyWire {
                oid: build_modify_oid_value(&req.order_id)?,
                order: build_hl_order_wire(&place, asset_meta),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(serde_json::to_value(HlBatchModifyAction {
        action_type: "batchModify".to_string(),
        modifies,
    })?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperliquidCancelBatchKind {
    Oid,
    Cloid,
}

fn parse_hl_oid(order_id: &str) -> anyhow::Result<u64> {
    order_id.parse::<u64>().map_err(|_err| {
        anyhow::anyhow!("invalid Hyperliquid numeric order id sanitized_reason=invalid_order_id")
    })
}

/// Build a batch cancel action with N cancels in a single API call.
fn build_batch_cancel_action(
    cancels: &[&crate::types::CancelOrderIntent],
    asset_index: u32,
) -> anyhow::Result<(serde_json::Value, HyperliquidCancelBatchKind)> {
    let Some(first) = cancels.first() else {
        anyhow::bail!("empty cancel batch");
    };
    let kind = if is_hyperliquid_cloid(&first.order_id) {
        HyperliquidCancelBatchKind::Cloid
    } else {
        HyperliquidCancelBatchKind::Oid
    };
    let mixed_kinds = cancels.iter().any(|cancel| {
        let item_kind = if is_hyperliquid_cloid(&cancel.order_id) {
            HyperliquidCancelBatchKind::Cloid
        } else {
            HyperliquidCancelBatchKind::Oid
        };
        item_kind != kind
    });
    if mixed_kinds {
        anyhow::bail!("mixed hyperliquid cancel batch id kinds");
    }

    let action = match kind {
        HyperliquidCancelBatchKind::Oid => {
            let cancel_items: anyhow::Result<Vec<HlCancelOidWire>> = cancels
                .iter()
                .map(|c| {
                    Ok(HlCancelOidWire {
                        a: asset_index,
                        o: parse_hl_oid(&c.order_id)?,
                    })
                })
                .collect();
            serde_json::to_value(HlCancelOidAction {
                action_type: "cancel".to_string(),
                cancels: cancel_items?,
            })?
        }
        HyperliquidCancelBatchKind::Cloid => {
            let cancel_items: Vec<HlCancelCloidWire> = cancels
                .iter()
                .map(|c| HlCancelCloidWire {
                    asset: asset_index,
                    cloid: c.order_id.clone(),
                })
                .collect();
            serde_json::to_value(HlCancelCloidAction {
                action_type: "cancelByCloid".to_string(),
                cancels: cancel_items,
            })?
        }
    };
    Ok((action, kind))
}

fn is_hyperliquid_cloid(value: &str) -> bool {
    value.len() == 34
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn build_cancel_action(
    intent: &OrderIntent,
    asset_index: u32,
) -> anyhow::Result<serde_json::Value> {
    let OrderIntent::Cancel(cancel) = intent else {
        anyhow::bail!("intent not a cancel order");
    };
    if is_hyperliquid_cloid(&cancel.order_id) {
        return Ok(serde_json::to_value(HlCancelCloidAction {
            action_type: "cancelByCloid".to_string(),
            cancels: vec![HlCancelCloidWire {
                asset: asset_index,
                cloid: cancel.order_id.clone(),
            }],
        })?);
    }
    Ok(serde_json::to_value(HlCancelOidAction {
        action_type: "cancel".to_string(),
        cancels: vec![HlCancelOidWire {
            a: asset_index,
            o: parse_hl_oid(&cancel.order_id)?,
        }],
    })?)
}

fn build_cancel_all_action(asset_index: u32) -> serde_json::Value {
    serde_json::to_value(HlCancelAllAction {
        action_type: "cancelAll".to_string(),
        asset: asset_index,
    })
    .expect("serialize hyperliquid cancelAll action")
}

fn build_noop_action() -> serde_json::Value {
    serde_json::to_value(HlNoopAction {
        action_type: "noop".to_string(),
    })
    .expect("serialize hyperliquid noop action")
}

pub fn reserve_request_weight_cost_micros(weight: u64) -> u64 {
    weight.saturating_mul(500)
}

pub fn format_usdc_micros(micros: u64) -> String {
    let whole = micros / 1_000_000;
    let frac = micros % 1_000_000;
    if frac == 0 {
        return whole.to_string();
    }
    let mut frac_str = format!("{frac:06}");
    while frac_str.ends_with('0') {
        frac_str.pop();
    }
    format!("{whole}.{frac_str}")
}

pub fn build_reserve_request_weight_action(weight: u64) -> anyhow::Result<serde_json::Value> {
    if weight == 0 {
        anyhow::bail!("reserveRequestWeight weight must be positive");
    }
    Ok(serde_json::to_value(HlReserveRequestWeightAction {
        action_type: "reserveRequestWeight".to_string(),
        weight,
    })?)
}

pub fn hyperliquid_exchange_response_error(value: &serde_json::Value) -> Option<String> {
    let status = value.get("status").and_then(|field| field.as_str());
    if matches!(status, Some("err" | "error")) {
        return Some(
            exchange_response_detail(value.get("response")).unwrap_or_else(|| value.to_string()),
        );
    }

    let response = value.get("response")?;
    let response_status = response.get("status").and_then(|field| field.as_str());
    if matches!(response_status, Some("err" | "error")) {
        return Some(
            exchange_response_detail(Some(response)).unwrap_or_else(|| response.to_string()),
        );
    }
    let response_type = response.get("type").and_then(|field| field.as_str());
    if matches!(response_type, Some("err" | "error")) {
        return Some(
            exchange_response_detail(Some(response)).unwrap_or_else(|| response.to_string()),
        );
    }
    None
}

fn exchange_response_detail(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?;
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    for key in ["message", "error", "payload", "response"] {
        if let Some(text) = value.get(key).and_then(|field| field.as_str()) {
            return Some(text.to_string());
        }
    }
    Some(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::LiveGatewayErrorKind;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        value: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                value: std::env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = self.value.as_deref() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn hl_internal_pub_q_reads_env_override() {
        let _guard = ENV_MUTEX.lock().expect("env mutex");
        let _restore = EnvVarGuard::new("PARAPHINA_HL_INTERNAL_PUB_Q");
        std::env::set_var("PARAPHINA_HL_INTERNAL_PUB_Q", "1024");
        assert_eq!(hl_internal_pub_q(), 1024);
    }

    #[tokio::test]
    async fn hyperliquid_cancel_all_smoke() {
        use tiny_http::{Response, Server};

        let server = Server::http("127.0.0.1:0").expect("bind server");
        let addr = server.server_addr();
        let rest_url = format!("http://{}", addr);
        let info_url = rest_url.clone();
        std::thread::spawn(move || {
            for mut request in server.incoming_requests().take(3) {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if body.contains(r#""type":"meta""#) {
                    let resp =
                        Response::from_string(r#"{"universe":[{"name":"TAO","szDecimals":4}]}"#);
                    let _ = request.respond(resp);
                } else if body.contains(r#""type":"userRole""#) {
                    let resp = Response::from_string(r#"{"role":"missing"}"#);
                    let _ = request.respond(resp);
                } else {
                    let resp = Response::from_string(r#"{"status":"ok"}"#);
                    let _ = request.respond(resp);
                }
            }
        });

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["wss://example".to_string()],
            rest_urls: vec![rest_url],
            info_urls: vec![info_url],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: Some("0xdeadbeef".to_string()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);
        connector.cancel_all(1_234).await.expect("cancel_all");
    }

    #[test]
    fn next_action_nonce_is_monotonic_and_fast_forwards() {
        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["wss://example".to_string()],
            rest_urls: vec!["https://example".to_string()],
            info_urls: vec!["https://example".to_string()],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: true,
            private_key_hex: None,
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);

        let first = connector.next_action_nonce(10_000);
        let second = connector.next_action_nonce(10_000);
        let jumped = connector.next_action_nonce((second + 25) as TimestampMs);

        assert_eq!(second, first + 1);
        assert_eq!(jumped, second + 25);
    }

    fn test_signing_key() -> SigningKey {
        let key_bytes =
            hex::decode("e908f86dbb4d55ac876378565aafeabc187f6690f046459397b17d9b9a19688e")
                .expect("decode signing key");
        SigningKey::from_slice(&key_bytes).expect("signing key")
    }

    fn verifying_key_to_eth_address(verifying_key: &k256::ecdsa::VerifyingKey) -> String {
        let encoded = verifying_key.to_encoded_point(false);
        let digest = keccak_bytes(&encoded.as_bytes()[1..]);
        format!("0x{}", hex::encode(&digest[12..]))
    }

    fn signature_to_hex(signature: &serde_json::Value) -> String {
        let r = signature["r"].as_str().expect("r").trim_start_matches("0x");
        let s = signature["s"].as_str().expect("s").trim_start_matches("0x");
        let v = signature["v"].as_u64().expect("v");
        format!("0x{r}{s}{v:02x}")
    }

    #[test]
    fn hl_float_to_wire_matches_official_sdk_examples() {
        assert_eq!(hl_float_to_wire(0.0), "0");
        assert_eq!(hl_float_to_wire(-0.0), "0");
        assert_eq!(hl_float_to_wire(0.00076), "0.00076");
        assert_eq!(hl_float_to_wire(0.00000001), "0.00000001");
        assert_eq!(hl_float_to_wire(87654321.1234), "87654321.1234");
        assert_eq!(hl_float_to_wire(987654321.0), "987654321");
        assert_eq!(hl_float_to_wire(2062.62), "2062.62");
        assert_eq!(hl_float_to_wire(0.01), "0.01");
    }

    #[test]
    fn build_hl_order_wire_canonicalizes_live_like_price_and_size() {
        let place = crate::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: std::sync::Arc::<str>::from("hyperliquid"),
            side: Side::Buy,
            price: 2062.62,
            size: 0.01,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some("0xc8f74824e72973753f7a01a83e322717".to_string()),
            phase51_target_key: None,
        };

        let wire = build_hl_order_wire(
            &place,
            HyperliquidAssetMeta {
                index: 1,
                sz_decimals: 4,
            },
        );
        assert_eq!(wire.p, "2062.6");
        assert_eq!(wire.s, "0.01");
        match wire.t {
            HlOrderType::Limit { limit } => assert_eq!(limit.tif, "Alo"),
            other => panic!("unexpected order type: {other:?}"),
        }
    }

    #[test]
    fn build_modify_action_serializes_cloid_and_new_cloid_wire() {
        let req = LiveRestReplaceRequest {
            venue_index: 0,
            venue_id: "hyperliquid".to_string(),
            order_id: "0xc8f74824e72973753f7a01a83e322717".to_string(),
            side: Side::Buy,
            price: 2062.62,
            size: 0.01,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        };
        let action = build_modify_action(
            &req,
            HyperliquidAssetMeta {
                index: 1,
                sz_decimals: 4,
            },
        )
        .expect("modify action");
        assert_eq!(action["type"], "modify");
        assert_eq!(action["oid"], req.order_id);
        assert_eq!(action["order"]["c"], req.client_order_id);
        assert_eq!(action["order"]["t"]["limit"]["tif"], "Alo");
    }

    #[test]
    fn build_batch_modify_action_serializes_multiple_modifies() {
        let reqs = vec![
            LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "hyperliquid".to_string(),
                order_id: "0x11111111111111111111111111111111".to_string(),
                side: Side::Buy,
                price: 100.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            },
            LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "hyperliquid".to_string(),
                order_id: "0x22222222222222222222222222222222".to_string(),
                side: Side::Sell,
                price: 101.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            },
        ];
        let action = build_batch_modify_action(
            &reqs,
            HyperliquidAssetMeta {
                index: 1,
                sz_decimals: 4,
            },
        )
        .expect("batch modify action");
        assert_eq!(action["type"], "batchModify");
        assert_eq!(action["modifies"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(action["modifies"][0]["oid"], reqs[0].order_id);
        assert_eq!(action["modifies"][1]["order"]["c"], reqs[1].client_order_id);
    }

    #[test]
    fn quantize_hl_price_snaps_eth_perp_buy_down_to_valid_tick() {
        let asset_meta = HyperliquidAssetMeta {
            index: 1,
            sz_decimals: 4,
        };
        assert_eq!(
            hl_float_to_wire(quantize_hl_price(2068.22, Side::Buy, asset_meta)),
            "2068.2"
        );
        assert_eq!(
            hl_float_to_wire(quantize_hl_price(2067.92, Side::Buy, asset_meta)),
            "2067.9"
        );
    }

    #[test]
    fn quantize_hl_price_snaps_eth_perp_sell_up_to_valid_tick() {
        let asset_meta = HyperliquidAssetMeta {
            index: 1,
            sz_decimals: 4,
        };
        assert_eq!(
            hl_float_to_wire(quantize_hl_price(2069.36, Side::Sell, asset_meta)),
            "2069.4"
        );
        assert_eq!(
            hl_float_to_wire(quantize_hl_price(2069.62, Side::Sell, asset_meta)),
            "2069.7"
        );
    }

    #[test]
    fn sign_action_matches_official_limit_order_vector() {
        let wallet = test_signing_key();
        let action = json!({
            "type": "order",
            "orders": [{
                "a": 1,
                "b": true,
                "p": "2000.0",
                "s": "3.5",
                "r": false,
                "t": { "limit": { "tif": "Ioc" } }
            }],
            "grouping": "na"
        });

        let signature = sign_action(&action, None, 1_583_838, true, &wallet).expect("signature");
        assert_eq!(
            signature_to_hex(&signature),
            "0x77957e58e70f43b6b68581f2dc42011fc384538a2e5b7bf42d5b936f19fbb67360721a8598727230f67080efee48c812a6a4442013fd3b0eed509171bef9f23f1c"
        );
    }

    #[test]
    fn sign_action_matches_official_limit_order_cloid_vector() {
        let wallet = test_signing_key();
        let action = json!({
            "type": "order",
            "orders": [{
                "a": 1,
                "b": true,
                "p": "2000.0",
                "s": "3.5",
                "r": false,
                "t": { "limit": { "tif": "Ioc" } },
                "c": "0x1e60610f0b3d420597c88c1fed2ad5ee"
            }],
            "grouping": "na"
        });

        let signature = sign_action(&action, None, 1_583_838, true, &wallet).expect("signature");
        assert_eq!(
            signature_to_hex(&signature),
            "0xd3e894092eb27098077145714630a77bbe3836120ee29df7d935d8510b03a08f456de5ec1be82aa65fc6ecda9ef928b0445e212517a98858cfaa251c4cd7552b1c"
        );
    }

    #[test]
    fn sign_action_matches_official_cancel_vector() {
        let wallet = test_signing_key();
        let action = json!({
            "type": "cancel",
            "cancels": [{
                "a": 1,
                "o": 82382
            }]
        });

        let signature = sign_action(&action, None, 1_583_838, true, &wallet).expect("signature");
        assert_eq!(
            signature_to_hex(&signature),
            "0x02f76cc5b16e0810152fa0e14e7b219f49c361e3325f771544c6f54e157bf9fa17ed0afc11a98596be85d5cd9f86600aad515337318f7ab346e5ccc1b03425d51b"
        );
    }

    #[test]
    fn sign_action_recovery_matches_signing_key_address() {
        let wallet = test_signing_key();
        let verifying_key = wallet.verifying_key();
        let expected_address = verifying_key_to_eth_address(verifying_key);
        let action = json!({
            "type": "order",
            "orders": [{
                "a": 1,
                "b": true,
                "p": "2000.0",
                "s": "3.5",
                "r": false,
                "t": { "limit": { "tif": "Ioc" } }
            }],
            "grouping": "na"
        });

        let signature = sign_action(&action, None, 1_583_838, true, &wallet).expect("signature");
        let recovered = recover_action_signer_address(&action, None, 1_583_838, true, &signature)
            .expect("recover signer");
        assert_eq!(recovered, expected_address);
    }

    #[test]
    fn reserve_request_weight_action_serializes_and_costs_exactly() {
        let action = build_reserve_request_weight_action(4_149).expect("reserve action");
        assert_eq!(action["type"], "reserveRequestWeight");
        assert_eq!(action["weight"], 4_149);
        assert!(serialize_action_for_hl_signing(&action).is_ok());
        assert_eq!(reserve_request_weight_cost_micros(4_149), 2_074_500);
        assert_eq!(format_usdc_micros(2_074_500), "2.0745");
        assert_eq!(format_usdc_micros(500), "0.0005");
    }

    #[test]
    fn reserve_request_weight_action_rejects_zero_weight() {
        let err = build_reserve_request_weight_action(0).expect_err("zero weight must fail");
        assert!(err
            .to_string()
            .contains("reserveRequestWeight weight must be positive"));
    }

    #[test]
    fn map_rest_error_sanitizes_raw_exchange_payloads() {
        let err = map_rest_error(anyhow::anyhow!(
            "Hyperliquid ws_post action exchange_errors=Post only order would have immediately matched, bbo was 2124.4@2124.5 payload={{\"order_id\":\"123\"}}"
        ));
        assert_eq!(err.kind, LiveGatewayErrorKind::PostOnlyReject);
        assert!(err
            .message
            .contains("sanitized_reason=post_only_would_match"));
        assert!(!err.message.contains("2124.4"));
        assert!(!err.message.contains("payload"));
        assert!(!err.message.contains("order_id"));
    }

    #[test]
    fn parse_hl_oid_rejects_invalid_ids_without_echoing_raw_value() {
        let err = parse_hl_oid("raw_order_id=123 client_order_id=abc")
            .expect_err("invalid order id must fail")
            .to_string();
        assert!(err.contains("sanitized_reason=invalid_order_id"));
        assert!(!err.contains("raw_order_id"));
        assert!(!err.contains("client_order_id"));
        assert!(!err.contains("123"));
        assert!(!err.contains("abc"));
    }

    #[test]
    fn reserve_request_weight_signature_recovers_signer() {
        let wallet = test_signing_key();
        let verifying_key = wallet.verifying_key();
        let expected_address = verifying_key_to_eth_address(verifying_key);
        let action = build_reserve_request_weight_action(4_149).expect("reserve action");

        let signature = sign_action(&action, None, 1_583_838, true, &wallet).expect("signature");
        let recovered = recover_action_signer_address(&action, None, 1_583_838, true, &signature)
            .expect("recover signer");
        assert_eq!(recovered, expected_address);
    }

    #[test]
    fn reserve_request_weight_signature_with_vault_recovers_signer() {
        let wallet = test_signing_key();
        let verifying_key = wallet.verifying_key();
        let expected_address = verifying_key_to_eth_address(verifying_key);
        let vault_address = "0x000000000000000000000000000000000000dEaD";
        let action = build_reserve_request_weight_action(4_149).expect("reserve action");

        let without_vault =
            sign_action(&action, None, 1_583_838, true, &wallet).expect("signature");
        let with_vault =
            sign_action(&action, Some(vault_address), 1_583_838, true, &wallet).expect("signature");
        assert_ne!(with_vault, without_vault);
        let recovered = recover_action_signer_address(
            &action,
            Some(vault_address),
            1_583_838,
            true,
            &with_vault,
        )
        .expect("recover signer");
        assert_eq!(recovered, expected_address);
    }

    #[test]
    fn exchange_response_error_detects_status_err() {
        let value = json!({
            "status": "err",
            "response": "Must deposit before performing actions. User: 0xabc"
        });

        assert_eq!(
            hyperliquid_exchange_response_error(&value).as_deref(),
            Some("Must deposit before performing actions. User: 0xabc")
        );
    }

    #[test]
    fn exchange_response_error_detects_nested_error_payload() {
        let value = json!({
            "status": "ok",
            "response": {
                "type": "error",
                "payload": "429 Too Many Requests"
            }
        });

        assert_eq!(
            hyperliquid_exchange_response_error(&value).as_deref(),
            Some("429 Too Many Requests")
        );
    }

    #[test]
    fn exchange_response_error_allows_status_ok() {
        let value = json!({
            "status": "ok",
            "response": {
                "type": "default"
            }
        });

        assert!(hyperliquid_exchange_response_error(&value).is_none());
    }

    #[test]
    fn parse_ws_post_response_maps_error_payloads() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 42,
                "response": {
                    "type": "error",
                    "payload": "429 Too Many Requests"
                }
            }
        });
        let (id, response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        assert_eq!(id, 42);
        assert_eq!(response_type, "error");
        let err = result.expect_err("error response");
        let err = err.to_string();
        assert!(err.contains("sanitized_reason=rate_limited"));
        assert!(!err.contains("429 Too Many Requests"));
        assert!(!err.contains("payload"));
    }

    #[test]
    fn parse_ws_post_response_maps_top_level_err_payloads() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 7,
                "response": {
                    "type": "action",
                    "payload": {
                        "status": "err",
                        "response": "User or API Wallet does not exist"
                    }
                }
            }
        });
        let (id, response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        assert_eq!(id, 7);
        assert_eq!(response_type, "action");
        let err = result.expect_err("top level err response");
        let err = err.to_string();
        assert!(err.contains("top_level_err sanitized_reason=exchange_error"));
        assert!(!err.contains("User or API Wallet does not exist"));
        assert!(!err.contains("payload"));
    }

    #[test]
    fn parse_ws_post_response_sanitizes_action_decode_failures() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 11,
                "response": {
                    "type": "action",
                    "payload": "raw_order_id=123 client_order_id=abc"
                }
            }
        });
        let (id, response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        assert_eq!(id, 11);
        assert_eq!(response_type, "action");
        let err = result.expect_err("decode failure").to_string();
        assert!(err.contains("action decode_failed sanitized_context=true"));
        assert!(!err.contains("raw_order_id"));
        assert!(!err.contains("client_order_id"));
        assert!(!err.contains("123"));
        assert!(!err.contains("abc"));
    }

    #[test]
    fn parse_ws_post_response_sanitizes_logged_response_type() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 12,
                "response": {
                    "type": "raw_order_id=123 client_order_id=abc",
                    "payload": null
                }
            }
        });
        let (id, response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        assert_eq!(id, 12);
        assert_eq!(response_type, "other");
        result.expect("non-action unknown response type should not fail");
    }

    #[test]
    fn parse_ws_post_response_accepts_resting_order_status() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 8,
                "response": {
                    "type": "action",
                    "payload": {
                        "status": "ok",
                        "response": {
                            "type": "order",
                            "data": {
                                "statuses": [
                                    { "resting": { "oid": 12345 } }
                                ]
                            }
                        }
                    }
                }
            }
        });
        let (id, response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        assert_eq!(id, 8);
        assert_eq!(response_type, "action");
        result.expect("resting status should succeed");
    }

    #[test]
    fn parse_ws_post_response_maps_nested_exchange_errors() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 9,
                "response": {
                    "type": "action",
                    "payload": {
                        "status": "ok",
                        "response": {
                            "type": "order",
                            "data": {
                                "statuses": [
                                    { "error": "BadAloPx" }
                                ]
                            }
                        }
                    }
                }
            }
        });
        let (id, response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        assert_eq!(id, 9);
        assert_eq!(response_type, "action");
        let err = result.expect_err("nested exchange error");
        let msg = err.to_string();
        assert!(msg.contains("exchange_errors_count=1"));
        assert!(msg.contains("exchange_reasons=bad_alo_px:1"));
        assert!(msg.contains("response_type=order"));
        assert!(!msg.contains("BadAloPx"));
        assert!(!msg.contains("\"statuses\""));
        assert!(!msg.contains("payload"));
    }

    #[test]
    fn parse_ws_post_response_sanitizes_post_only_reject_payload() {
        let value = json!({
            "channel": "post",
            "data": {
                "id": 10,
                "response": {
                    "type": "action",
                    "payload": {
                        "status": "ok",
                        "response": {
                            "type": "order",
                            "data": {
                                "statuses": [
                                    { "error": "Post only order would have immediately matched, bbo was 2124.4@2124.5. asset=1" }
                                ]
                            }
                        }
                    }
                }
            }
        });
        let (_id, _response_type, result) =
            parse_ws_post_response(&value).expect("parse ws post response");
        let msg = result.expect_err("post-only exchange error").to_string();
        assert!(msg.contains("exchange_errors_count=1"));
        assert!(msg.contains("exchange_reasons=post_only_would_match:1"));
        assert!(!msg.contains("bbo"));
        assert!(!msg.contains("2124.4"));
        assert!(!msg.contains("payload"));
    }

    #[tokio::test]
    async fn submit_signed_action_uses_ws_post_for_orders_when_enabled() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_transport = EnvVarGuard::new("PARAPHINA_HL_ACTION_TRANSPORT");
        let _guard_timeout = EnvVarGuard::new("PARAPHINA_HL_WS_POST_RESPONSE_TIMEOUT_MS");
        std::env::set_var("PARAPHINA_HL_ACTION_TRANSPORT", "ws_post");
        std::env::set_var("PARAPHINA_HL_WS_POST_RESPONSE_TIMEOUT_MS", "1000");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ws listener");
        let addr = listener.local_addr().expect("listener addr");
        let observed = Arc::new(Mutex::new(None::<serde_json::Value>));
        let observed_server = observed.clone();

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept ws client");
            let mut ws = tokio_tungstenite::accept_async(stream)
                .await
                .expect("accept websocket");
            while let Some(message) = ws.next().await {
                let message = message.expect("ws frame");
                if let Message::Text(text) = message {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).expect("ws post payload");
                    if value.get("method").and_then(|v| v.as_str()) != Some("post") {
                        continue;
                    }
                    *observed_server.lock().expect("observed lock") = Some(value.clone());
                    let id = value["id"].as_u64().expect("post id");
                    let response = json!({
                        "channel": "post",
                        "data": {
                            "id": id,
                            "response": {
                                "type": "action",
                                "payload": {
                                    "status": "ok",
                                    "response": {
                                        "type": "order",
                                        "data": {
                                            "statuses": [
                                                { "resting": { "oid": 777 } }
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    });
                    ws.send(Message::Text(response.to_string()))
                        .await
                        .expect("send post response");
                    break;
                }
            }
        });

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec![format!("ws://{}", addr)],
            rest_urls: vec!["http://127.0.0.1:1/exchange".to_string()],
            info_urls: vec!["http://127.0.0.1:1/info".to_string()],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = Arc::new(HyperliquidConnector::new(cfg, market_tx, exec_tx));
        assert!(connector.uses_ws_post_actions());
        let worker = {
            let connector = connector.clone();
            tokio::spawn(async move { connector.run_post_ws().await })
        };

        connector
            .submit_signed_action(
                json!({
                    "type": "order",
                    "orders": [{
                        "a": 7,
                        "b": true,
                        "p": "1",
                        "s": "1",
                        "r": false,
                        "t": { "limit": { "tif": "Alo" } }
                    }],
                    "grouping": "na"
                }),
                12_345,
                "order_batch",
                "place_alo",
                1,
            )
            .await
            .expect("ws post action");

        server.await.expect("ws server task");
        let observed = observed
            .lock()
            .expect("observed lock")
            .clone()
            .expect("captured ws post payload");
        assert_eq!(observed["method"], "post");
        assert_eq!(observed["request"]["type"], "action");
        assert_eq!(observed["request"]["payload"]["action"]["type"], "order");
        assert!(observed["request"]["payload"]["nonce"].as_u64().is_some());
        assert!(observed["request"]["payload"]["signature"]["r"]
            .as_str()
            .is_some());
        assert!(observed["id"].as_u64().is_some());

        worker.abort();
    }

    #[tokio::test]
    async fn submit_signed_action_routes_cancel_all_to_http_when_ws_post_enabled() {
        use std::sync::{Arc, Mutex};
        use tiny_http::{Response, Server};

        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_transport = EnvVarGuard::new("PARAPHINA_HL_ACTION_TRANSPORT");
        std::env::set_var("PARAPHINA_HL_ACTION_TRANSPORT", "ws_post");

        let server = Server::http("127.0.0.1:0").expect("bind server");
        let addr = server.server_addr();
        let rest_url = format!("http://{}", addr);
        let observed = Arc::new(Mutex::new(None::<serde_json::Value>));
        let observed_server = observed.clone();
        std::thread::spawn(move || {
            for mut request in server.incoming_requests().take(2) {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                let value: serde_json::Value =
                    serde_json::from_str(&body).expect("http payload json");
                if value.get("type").and_then(|value| value.as_str()) == Some("meta") {
                    let _ = request.respond(Response::from_string(
                        r#"{"universe":[{"name":"TAO","szDecimals":4}]}"#,
                    ));
                } else {
                    *observed_server.lock().expect("observed lock") = Some(value);
                    let _ = request.respond(Response::from_string(r#"{"status":"ok"}"#));
                }
            }
        });

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["ws://127.0.0.1:1".to_string()],
            rest_urls: vec![rest_url.clone()],
            info_urls: vec![rest_url],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);

        connector
            .submit_signed_action(
                json!({ "type": "cancelAll", "asset": 7 }),
                12_345,
                "cancel_all",
                "cancel_all",
                1,
            )
            .await
            .expect("http control fallback");

        let observed = observed
            .lock()
            .expect("observed lock")
            .clone()
            .expect("captured http payload");
        assert_eq!(observed["action"]["type"], "cancelAll");
        assert!(observed["nonce"].as_u64().is_some());
        assert!(observed["signature"]["r"].as_str().is_some());
    }

    #[tokio::test]
    async fn submit_signed_action_routes_sync_control_orders_to_http_when_ws_post_enabled() {
        use std::sync::{Arc, Mutex};
        use tiny_http::{Response, Server};

        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_transport = EnvVarGuard::new("PARAPHINA_HL_ACTION_TRANSPORT");
        let _guard_sync_fallback =
            EnvVarGuard::new("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED");
        std::env::set_var("PARAPHINA_HL_ACTION_TRANSPORT", "ws_post");
        std::env::set_var("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED", "1");

        let server = Server::http("127.0.0.1:0").expect("bind server");
        let addr = server.server_addr();
        let rest_url = format!("http://{}", addr);
        let observed = Arc::new(Mutex::new(None::<serde_json::Value>));
        let observed_server = observed.clone();
        std::thread::spawn(move || {
            for mut request in server.incoming_requests().take(2) {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                let value: serde_json::Value =
                    serde_json::from_str(&body).expect("http payload json");
                if value.get("type").and_then(|value| value.as_str()) == Some("meta") {
                    let _ = request.respond(Response::from_string(
                        r#"{"universe":[{"name":"TAO","szDecimals":4}]}"#,
                    ));
                } else {
                    *observed_server.lock().expect("observed lock") = Some(value);
                    let _ = request.respond(Response::from_string(r#"{"status":"ok"}"#));
                }
            }
        });

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["ws://127.0.0.1:1".to_string()],
            rest_urls: vec![rest_url.clone()],
            info_urls: vec![rest_url],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);

        connector
            .submit_signed_action_with_hint(
                json!({
                    "type": "order",
                    "orders": [{
                        "a": 7,
                        "b": true,
                        "p": "1",
                        "s": "1",
                        "r": false,
                        "t": { "limit": { "tif": "Alo" } }
                    }],
                    "grouping": "na"
                }),
                12_345,
                "order_batch",
                "place_alo",
                1,
                TransportHint::HyperliquidSyncControl,
            )
            .await
            .expect("http sync-control fallback");

        let observed = observed
            .lock()
            .expect("observed lock")
            .clone()
            .expect("captured http payload");
        assert_eq!(observed["action"]["type"], "order");
        assert!(observed["nonce"].as_u64().is_some());
        assert!(observed["signature"]["r"].as_str().is_some());
    }

    #[test]
    fn action_transport_routes_sync_control_to_http_when_ws_post_enabled() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_transport = EnvVarGuard::new("PARAPHINA_HL_ACTION_TRANSPORT");
        let _guard_sync_fallback =
            EnvVarGuard::new("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED");
        let _guard_cancel_all_fallback =
            EnvVarGuard::new("PARAPHINA_HL_CANCEL_ALL_HTTP_FALLBACK_ENABLED");
        std::env::set_var("PARAPHINA_HL_ACTION_TRANSPORT", "ws_post");
        std::env::set_var("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED", "1");
        std::env::set_var("PARAPHINA_HL_CANCEL_ALL_HTTP_FALLBACK_ENABLED", "1");

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["ws://127.0.0.1:1".to_string()],
            rest_urls: vec!["http://127.0.0.1:1/exchange".to_string()],
            info_urls: vec!["http://127.0.0.1:1/info".to_string()],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);

        assert_eq!(
            connector.action_transport_for("order_batch", TransportHint::Default),
            HyperliquidActionTransport::WsPost
        );
        assert_eq!(
            connector.action_transport_for("order_batch", TransportHint::HyperliquidSyncControl),
            HyperliquidActionTransport::Http
        );
        assert_eq!(
            connector.action_transport_for("cancel_all", TransportHint::Default),
            HyperliquidActionTransport::Http
        );
    }

    #[test]
    fn action_transport_uses_ws_post_for_sync_control_when_http_fallback_disabled() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_transport = EnvVarGuard::new("PARAPHINA_HL_ACTION_TRANSPORT");
        let _guard_sync_fallback =
            EnvVarGuard::new("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED");
        let _guard_cancel_all_fallback =
            EnvVarGuard::new("PARAPHINA_HL_CANCEL_ALL_HTTP_FALLBACK_ENABLED");
        std::env::set_var("PARAPHINA_HL_ACTION_TRANSPORT", "ws_post");
        std::env::set_var("PARAPHINA_HL_SYNC_CONTROL_HTTP_FALLBACK_ENABLED", "0");
        std::env::set_var("PARAPHINA_HL_CANCEL_ALL_HTTP_FALLBACK_ENABLED", "1");

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["ws://127.0.0.1:1".to_string()],
            rest_urls: vec!["http://127.0.0.1:1/exchange".to_string()],
            info_urls: vec!["http://127.0.0.1:1/info".to_string()],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);

        assert_eq!(
            connector.action_transport_for("order_batch", TransportHint::HyperliquidSyncControl),
            HyperliquidActionTransport::WsPost
        );
        assert_eq!(
            connector.action_transport_for("cancel_all", TransportHint::Default),
            HyperliquidActionTransport::Http
        );
    }

    #[test]
    fn cancel_action_uses_oid_for_numeric_order_ids() {
        let intent = OrderIntent::Cancel(crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "123456".to_string(),
        });
        let action = build_cancel_action(&intent, 7).expect("cancel action");
        assert_eq!(action["type"], "cancel");
        assert_eq!(action["cancels"][0]["a"], 7);
        assert_eq!(action["cancels"][0]["o"], 123456);
    }

    #[test]
    fn cancel_action_uses_cloid_path_for_hex_client_ids() {
        let intent = OrderIntent::Cancel(crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
        });
        let action = build_cancel_action(&intent, 7).expect("cancel action");
        assert_eq!(action["type"], "cancelByCloid");
        assert_eq!(action["cancels"][0]["asset"], 7);
        assert_eq!(
            action["cancels"][0]["cloid"],
            "0x1234567890abcdef1234567890abcdef"
        );
    }

    #[test]
    fn batch_cancel_action_uses_oid_shape_for_numeric_order_ids() {
        let cancel_a = crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "123456".to_string(),
        };
        let cancel_b = crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "654321".to_string(),
        };
        let (action, kind) =
            build_batch_cancel_action(&[&cancel_a, &cancel_b], 7).expect("batch cancel action");
        assert_eq!(action["type"], "cancel");
        assert_eq!(action["cancels"][0]["a"], 7);
        assert_eq!(action["cancels"][0]["o"], 123456);
        assert_eq!(kind, HyperliquidCancelBatchKind::Oid);
    }

    #[test]
    fn batch_cancel_action_uses_cloid_shape_for_hex_client_ids() {
        let cancel_a = crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
        };
        let cancel_b = crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "0xfedcba0987654321fedcba0987654321".to_string(),
        };
        let (action, kind) =
            build_batch_cancel_action(&[&cancel_a, &cancel_b], 7).expect("batch cancel action");
        assert_eq!(action["type"], "cancelByCloid");
        assert_eq!(action["cancels"][0]["asset"], 7);
        assert_eq!(
            action["cancels"][0]["cloid"],
            "0x1234567890abcdef1234567890abcdef"
        );
        assert_eq!(kind, HyperliquidCancelBatchKind::Cloid);
    }

    #[test]
    fn batch_cancel_action_rejects_mixed_id_kinds() {
        let cancel_a = crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "123456".to_string(),
        };
        let cancel_b = crate::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "0xfedcba0987654321fedcba0987654321".to_string(),
        };
        let err = build_batch_cancel_action(&[&cancel_a, &cancel_b], 7).expect_err("mixed ids");
        assert!(err
            .to_string()
            .contains("mixed hyperliquid cancel batch id kinds"));
    }

    #[test]
    fn build_noop_action_has_expected_shape() {
        let action = build_noop_action();
        assert_eq!(action["type"], "noop");
    }

    #[test]
    fn serialize_action_for_hl_signing_supports_noop() {
        let encoded = serialize_action_for_hl_signing(&build_noop_action())
            .expect("noop should serialize for signing");
        assert!(!encoded.is_empty());
    }

    #[tokio::test]
    async fn cancel_batch_with_hint_routes_sync_control_cloid_batches_to_cancel_by_cloid() {
        use tiny_http::{Response, Server};

        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_transport = EnvVarGuard::new("PARAPHINA_HL_ACTION_TRANSPORT");
        std::env::set_var("PARAPHINA_HL_ACTION_TRANSPORT", "ws_post");

        let server = Server::http("127.0.0.1:0").expect("bind server");
        let addr = server.server_addr();
        let rest_url = format!("http://{}", addr);
        let observed = Arc::new(Mutex::new(None::<serde_json::Value>));
        let observed_server = observed.clone();
        std::thread::spawn(move || {
            for mut request in server.incoming_requests().take(2) {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                let value: serde_json::Value =
                    serde_json::from_str(&body).expect("http payload json");
                if value.get("type").and_then(|value| value.as_str()) == Some("meta") {
                    let _ = request.respond(Response::from_string(
                        r#"{"universe":[{"name":"TAO","szDecimals":4}]}"#,
                    ));
                } else {
                    *observed_server.lock().expect("observed lock") = Some(value);
                    let _ = request.respond(Response::from_string(r#"{"status":"ok"}"#));
                }
            }
        });

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["ws://127.0.0.1:1".to_string()],
            rest_urls: vec![rest_url.clone()],
            info_urls: vec![rest_url],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: false,
            private_key_hex: Some(
                "0000000000000000000000000000000000000000000000000000000000000001".to_string(),
            ),
            vault_address: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);

        let results = connector
            .cancel_batch_with_hint(
                vec![
                    LiveRestCancelRequest {
                        venue_index: 0,
                        venue_id: "hyperliquid".to_string(),
                        order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
                    },
                    LiveRestCancelRequest {
                        venue_index: 0,
                        venue_id: "hyperliquid".to_string(),
                        order_id: "0xfedcba0987654321fedcba0987654321".to_string(),
                    },
                ],
                TransportHint::HyperliquidSyncControl,
            )
            .await;

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.is_ok()));

        let observed = observed
            .lock()
            .expect("observed lock")
            .clone()
            .expect("captured http payload");
        assert_eq!(observed["action"]["type"], "cancelByCloid");
        assert_eq!(observed["action"]["cancels"][0]["asset"], 0);
        assert_eq!(
            observed["action"]["cancels"][0]["cloid"],
            "0x1234567890abcdef1234567890abcdef"
        );
        assert!(observed["nonce"].as_u64().is_some());
        assert!(observed["signature"]["r"].as_str().is_some());
    }

    #[test]
    fn freshness_reset_and_anchor_behavior() {
        let freshness = Freshness::default();
        freshness.last_parsed_ns.store(123, Ordering::Relaxed);
        freshness.last_published_ns.store(456, Ordering::Relaxed);
        freshness.last_book_event_ns.store(789, Ordering::Relaxed);
        freshness.reset_for_new_connection();
        assert_eq!(freshness.last_parsed_ns.load(Ordering::Relaxed), 0);
        assert_eq!(freshness.last_published_ns.load(Ordering::Relaxed), 0);
        assert_eq!(freshness.last_snapshot_resync_ns.load(Ordering::Relaxed), 0);
        assert_eq!(
            freshness.last_book_event_ns.load(Ordering::Relaxed),
            0,
            "last_book_event_ns must be reset on new connection"
        );

        // After reset with no book events, anchor falls back to connect_start_ns
        let connect_start_ns = 1_000;
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, connect_start_ns);

        // FIX A3: anchor now uses last_book_event_ns (not last_parsed_ns)
        freshness.last_book_event_ns.store(2_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 2_000);

        freshness.last_published_ns.store(3_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 3_000);
    }

    #[test]
    fn parse_public_funding_fixture() {
        let raw = include_str!(
            "../../../../tests/fixtures/hyperliquid/public_funding_meta_and_asset_ctxs.json"
        );
        let value: serde_json::Value = serde_json::from_str(raw).expect("fixture json");
        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_urls: vec!["wss://example".to_string()],
            rest_urls: vec!["https://example".to_string()],
            info_urls: vec!["https://example".to_string()],
            coin: "TAO".to_string(),
            n_sig_figs: 5,
            n_levels: 5,
            venue_index: 0,
            paper_mode: true,
            private_key_hex: None,
            vault_address: None,
        };
        let update = parse_public_funding(&value, &cfg).expect("funding update");
        assert_eq!(update.funding_rate_8h, Some(0.001));
        assert_eq!(update.interval_sec, Some(28_800));
        assert_eq!(update.next_funding_ms, Some(1_700_003_600_000));
        assert_eq!(update.source, FundingSource::MarketDataRest);
        // Fix C: HL settles funding at mark price (docs: https://hyperliquid.gitbook.io)
        assert_eq!(
            update.settlement_price_kind,
            Some(SettlementPriceKind::Mark),
            "Hyperliquid settlement must be Mark"
        );
    }

    #[test]
    fn account_snapshot_request_and_clearinghouse_shape_parse() {
        let request = build_account_snapshot_request("0xdeadbeef");
        assert_eq!(request["type"], "clearinghouseState");
        assert_eq!(request["user"], "0xdeadbeef");

        let payload = serde_json::json!({
            "marginSummary": {
                "accountValue": "100.0",
                "totalMarginUsed": "12.5",
                "totalRawUsd": "100.0"
            },
            "withdrawable": "87.5",
            "assetPositions": [{
                "position": {
                    "coin": "ETH",
                    "szi": "0.25",
                    "entryPx": "2500.0",
                    "liqPx": "2100.0"
                }
            }],
            "time": 1_700_000_000_123i64
        });
        let snapshot =
            parse_account_snapshot_with_meta(&payload, Some("ETH"), 7).expect("snapshot");
        assert_eq!(snapshot.venue_index, 7);
        assert_eq!(snapshot.venue_id, "ETH");
        assert_eq!(snapshot.timestamp_ms, 1_700_000_000_123);
        assert_eq!(snapshot.positions.len(), 1);
        assert_eq!(snapshot.positions[0].symbol, "ETH");
        assert!((snapshot.margin.balance_usd - 100.0).abs() < 1e-9);
        assert!((snapshot.margin.available_usd - 87.5).abs() < 1e-9);
        assert_eq!(snapshot.liquidation.price_liq, Some(2100.0));
    }

    #[test]
    fn private_subscriptions_include_clearinghouse_state_when_account_user_is_present() {
        let subscriptions = build_private_subscriptions(Some("0xdeadbeef"));
        assert_eq!(subscriptions.len(), 3);
        assert_eq!(subscriptions[0]["subscription"]["type"], "userFills");
        assert_eq!(subscriptions[1]["subscription"]["type"], "userEvents");
        assert_eq!(
            subscriptions[2]["subscription"]["type"],
            "clearinghouseState"
        );
        assert_eq!(subscriptions[2]["subscription"]["user"], "0xdeadbeef");
    }

    #[test]
    fn private_subscriptions_skip_clearinghouse_state_without_account_user() {
        let subscriptions = build_private_subscriptions(None);
        assert_eq!(subscriptions.len(), 2);
        assert_eq!(subscriptions[0]["subscription"]["type"], "userFills");
        assert_eq!(subscriptions[1]["subscription"]["type"], "userEvents");
    }

    fn hyperliquid_fill_payload() -> serde_json::Value {
        serde_json::json!({
            "side": "B",
            "px": "101.25",
            "sz": "0.5",
            "feeBps": "1.25",
            "purpose": "Mm"
        })
    }

    fn parsed_hyperliquid_user_fill(fill: serde_json::Value) -> crate::live::types::Fill {
        let msg = serde_json::json!({
            "channel": "userFills",
            "seq": 42,
            "data": {
                "timestamp": 1_700_000_000_123i64,
                "fills": [fill]
            }
        });
        let mut events = translate_private_events(&msg);
        assert_eq!(events.len(), 1);
        match events.remove(0) {
            ExecutionEvent::Filled(fill) => fill,
            _ => panic!("expected fill event"),
        }
    }

    #[test]
    fn hyperliquid_user_fill_crossed_true_populates_native_role() {
        let mut payload = hyperliquid_fill_payload();
        payload["crossed"] = serde_json::json!(true);

        let fill = parsed_hyperliquid_user_fill(payload);

        assert_eq!(fill.venue_id, "hyperliquid");
        assert_eq!(
            fill.phase51_native_role,
            Some(Phase51ForwardRefreshNativeRole::Hyperliquid { crossed: true })
        );
        assert_eq!(fill.phase51_lighter_native_limit, None);
    }

    #[test]
    fn hyperliquid_user_fill_crossed_false_populates_native_role() {
        let mut payload = hyperliquid_fill_payload();
        payload["crossed"] = serde_json::json!(false);

        let fill = parsed_hyperliquid_user_fill(payload);

        assert_eq!(fill.venue_id, "hyperliquid");
        assert_eq!(
            fill.phase51_native_role,
            Some(Phase51ForwardRefreshNativeRole::Hyperliquid { crossed: false })
        );
        assert_eq!(fill.phase51_lighter_native_limit, None);
    }

    #[test]
    fn hyperliquid_user_fill_missing_crossed_leaves_native_role_none() {
        let fill = parsed_hyperliquid_user_fill(hyperliquid_fill_payload());

        assert_eq!(fill.venue_id, "hyperliquid");
        assert_eq!(fill.phase51_native_role, None);
        assert_eq!(fill.phase51_lighter_native_limit, None);
    }

    #[test]
    fn hyperliquid_user_fill_non_bool_crossed_leaves_native_role_none() {
        for crossed in [
            serde_json::Value::Null,
            serde_json::json!("true"),
            serde_json::json!(1),
            serde_json::json!({ "value": true }),
            serde_json::json!([true]),
        ] {
            let mut payload = hyperliquid_fill_payload();
            payload["crossed"] = crossed;

            let fill = parsed_hyperliquid_user_fill(payload);

            assert_eq!(fill.phase51_native_role, None);
            assert_eq!(fill.phase51_lighter_native_limit, None);
        }
    }

    #[test]
    fn hyperliquid_user_fill_auxiliary_fields_do_not_create_native_role() {
        let mut payload = hyperliquid_fill_payload();
        payload["timestamp"] = serde_json::json!(1_700_000_000_124i64);
        payload["oid"] = serde_json::json!("synthetic-order-handle");
        payload["cloid"] = serde_json::json!("synthetic-client-handle");
        payload["tid"] = serde_json::json!("synthetic-fill-handle");

        let fill = parsed_hyperliquid_user_fill(payload);

        assert_eq!(fill.venue_id, "hyperliquid");
        assert_eq!(fill.phase51_native_role, None);
        assert_eq!(fill.phase51_lighter_native_limit, None);
    }

    #[test]
    fn translate_account_event_parses_wrapped_clearinghouse_state_payload() {
        let msg = serde_json::json!({
            "channel": "clearinghouseState",
            "data": {
                "user": "0xdeadbeef",
                "dex": "",
                "clearinghouseState": {
                    "marginSummary": {
                        "accountValue": "100.0",
                        "totalMarginUsed": "12.5",
                        "totalRawUsd": "100.0"
                    },
                    "withdrawable": "87.5",
                    "assetPositions": [{
                        "position": {
                            "coin": "ETH",
                            "szi": "0.25",
                            "entryPx": "2500.0",
                            "liqPx": "2100.0"
                        }
                    }],
                    "time": 1_700_000_000_123i64
                }
            }
        });
        let event = translate_account_event(&msg, Some("ETH"), 7).expect("account event");
        let AccountEvent::Snapshot(snapshot) = event;
        assert_eq!(snapshot.venue_index, 7);
        assert_eq!(snapshot.venue_id, "ETH");
        assert_eq!(snapshot.timestamp_ms, 1_700_000_000_123);
        assert_eq!(snapshot.positions.len(), 1);
        assert_eq!(snapshot.positions[0].symbol, "ETH");
    }

    #[test]
    fn translate_account_event_stamps_local_time_when_clearinghouse_time_missing() {
        let msg = serde_json::json!({
            "channel": "clearinghouseState",
            "data": {
                "user": "0xdeadbeef",
                "dex": "",
                "clearinghouseState": {
                    "marginSummary": {
                        "accountValue": "50.0",
                        "totalMarginUsed": "0.0",
                        "totalRawUsd": "50.0"
                    },
                    "withdrawable": "50.0",
                    "assetPositions": []
                }
            }
        });
        let before = now_ms();
        let event = translate_account_event(&msg, Some("ETH"), 3).expect("account event");
        let AccountEvent::Snapshot(snapshot) = event;
        let after = now_ms();
        assert_eq!(snapshot.venue_index, 3);
        assert_eq!(snapshot.venue_id, "ETH");
        assert!(snapshot.timestamp_ms >= before);
        assert!(snapshot.timestamp_ms <= after);
    }

    #[test]
    fn user_abstraction_parse_recognizes_supported_modes() {
        assert_eq!(
            parse_user_abstraction(&serde_json::json!("standard")),
            Some(HyperliquidAccountAbstraction::Standard)
        );
        assert_eq!(
            parse_user_abstraction(&serde_json::json!("unifiedAccount")),
            Some(HyperliquidAccountAbstraction::UnifiedAccount)
        );
        assert_eq!(
            parse_user_abstraction(&serde_json::json!("portfolioMargin")),
            Some(HyperliquidAccountAbstraction::PortfolioMargin)
        );
        assert_eq!(parse_user_abstraction(&serde_json::json!("other")), None);
    }

    #[test]
    fn user_role_parse_identifies_when_vault_address_is_required() {
        assert_eq!(
            parse_user_role(&serde_json::json!({"role": "user"})),
            Some(HyperliquidUserRole::User)
        );
        assert_eq!(
            parse_user_role(&serde_json::json!({"role": "vault"})),
            Some(HyperliquidUserRole::Vault)
        );
        assert_eq!(
            parse_user_role(&serde_json::json!({"role": "subAccount"})),
            Some(HyperliquidUserRole::SubAccount)
        );
        assert!(!HyperliquidUserRole::User.requires_vault_address());
        assert!(HyperliquidUserRole::Vault.requires_vault_address());
        assert!(HyperliquidUserRole::SubAccount.requires_vault_address());
    }

    #[test]
    fn spot_collateral_snapshot_parse_uses_usdc_available_after_maintenance() {
        let payload = serde_json::json!({
            "balances": [
                {
                    "coin": "USDC",
                    "token": 0,
                    "total": "94.8",
                    "hold": "0.0",
                    "entryNtl": "0.0"
                }
            ],
            "tokenToAvailableAfterMaintenance": [[0, "91.3"]]
        });

        let snapshot = parse_spot_collateral_snapshot(&payload).expect("spot collateral");
        assert_eq!(
            snapshot.balances,
            vec![BalanceSnapshot {
                asset: "USDC".to_string(),
                total: 94.8,
                available: 91.3,
            }]
        );
        assert!((snapshot.margin.balance_usd - 94.8).abs() < 1e-9);
        assert!((snapshot.margin.available_usd - 91.3).abs() < 1e-9);
        assert!((snapshot.margin.used_usd - 3.5).abs() < 1e-9);
    }

    #[test]
    fn resilient_snapshot_decode_skips_malformed_deeper_levels() {
        // Message where top level is valid but a deeper level has malformed data
        let json_str = r#"{
            "channel": "l2Book",
            "data": {
                "coin": "ETH",
                "time": 1700000000000,
                "seq": 123,
                "levels": [
                    [
                        {"px": "2000.5", "sz": "10.0"},
                        {"px": "invalid_price", "sz": "5.0"},
                        {"px": "1999.0", "sz": "20.0"}
                    ],
                    [
                        {"px": "2001.0", "sz": "8.0"},
                        {"px": "2002.0", "sz": "bad_size"},
                        {"px": "2003.0", "sz": "15.0"}
                    ]
                ]
            }
        }"#;
        let value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let mut fallback_seq = 0u64;

        let result = decode_l2book_snapshot_resilient(&value, 0, "ETH", &mut fallback_seq);

        // Should produce a valid snapshot despite malformed levels
        assert!(
            result.event.is_some(),
            "Should produce snapshot when at least 1 valid level per side"
        );
        assert_eq!(result.bid_skipped, 1, "Should skip 1 malformed bid level");
        assert_eq!(result.ask_skipped, 1, "Should skip 1 malformed ask level");
        assert_eq!(result.bid_total, 3, "Total bid levels should be 3");
        assert_eq!(result.ask_total, 3, "Total ask levels should be 3");
        assert!(result.failure_reason.is_none());

        // Verify the snapshot contains valid levels
        if let Some(MarketDataEvent::L2Snapshot(snap)) = result.event {
            assert_eq!(snap.bids.len(), 2, "Should have 2 valid bid levels");
            assert_eq!(snap.asks.len(), 2, "Should have 2 valid ask levels");
            // First bid should be the valid top level
            assert!((snap.bids[0].price - 2000.5).abs() < 0.01);
            assert!((snap.bids[0].size - 10.0).abs() < 0.01);
        } else {
            panic!("Expected L2Snapshot event");
        }
    }

    #[test]
    fn resilient_snapshot_decode_fails_when_all_levels_one_side_malformed() {
        // Message where ALL bid levels are malformed
        let json_str = r#"{
            "channel": "l2Book",
            "data": {
                "coin": "ETH",
                "time": 1700000000000,
                "seq": 456,
                "levels": [
                    [
                        {"px": "not_a_number", "sz": "also_bad"},
                        {"px": null, "sz": "5.0"}
                    ],
                    [
                        {"px": "2001.0", "sz": "8.0"},
                        {"px": "2002.0", "sz": "15.0"}
                    ]
                ]
            }
        }"#;
        let value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let mut fallback_seq = 0u64;

        let result = decode_l2book_snapshot_resilient(&value, 0, "ETH", &mut fallback_seq);

        // Should fail because no valid bid levels
        assert!(
            result.event.is_none(),
            "Should fail when all levels on one side are malformed"
        );
        assert_eq!(result.bid_skipped, 2, "Both bid levels should be skipped");
        assert_eq!(result.ask_skipped, 0, "No ask levels should be skipped");
        assert_eq!(result.bid_total, 2);
        assert_eq!(result.ask_total, 2);
        assert_eq!(
            result.failure_reason,
            Some("no valid levels on at least one side")
        );
    }

    #[test]
    fn resilient_snapshot_decode_handles_empty_levels() {
        // Message with empty bid levels array
        let json_str = r#"{
            "channel": "l2Book",
            "data": {
                "coin": "ETH",
                "time": 1700000000000,
                "seq": 789,
                "levels": [
                    [],
                    [{"px": "2001.0", "sz": "8.0"}]
                ]
            }
        }"#;
        let value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let mut fallback_seq = 0u64;

        let result = decode_l2book_snapshot_resilient(&value, 0, "ETH", &mut fallback_seq);

        // Should fail because no bid levels at all
        assert!(result.event.is_none());
        assert_eq!(result.bid_total, 0);
        assert_eq!(result.ask_total, 1);
        assert_eq!(
            result.failure_reason,
            Some("no valid levels on at least one side")
        );
    }

    #[test]
    fn resilient_levels_parser_works_with_array_format() {
        // Some venues use [price, size] array format
        let json_str = r#"[["2000.5", "10.0"], ["bad", "data"], ["1999.0", "20.0"]]"#;
        let value: serde_json::Value = serde_json::from_str(json_str).unwrap();

        let result = parse_levels_resilient(&value);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.levels.len(), 2, "Should have 2 valid levels");
        assert_eq!(r.skipped_count, 1, "Should skip 1 malformed level");
        assert_eq!(r.total_count, 3);
    }

    #[test]
    fn parse_level_entry_handles_both_formats() {
        // Object format with px/sz
        let obj_json = r#"{"px": "100.5", "sz": "10.0", "n": 5}"#;
        let obj_value: serde_json::Value = serde_json::from_str(obj_json).unwrap();
        let obj_result = parse_level_entry(&obj_value);
        assert!(obj_result.is_some());
        let level = obj_result.unwrap();
        assert!((level.price - 100.5).abs() < 0.01);
        assert!((level.size - 10.0).abs() < 0.01);

        // Array format [price, size]
        let arr_json = r#"["200.25", "5.5"]"#;
        let arr_value: serde_json::Value = serde_json::from_str(arr_json).unwrap();
        let arr_result = parse_level_entry(&arr_value);
        assert!(arr_result.is_some());
        let level = arr_result.unwrap();
        assert!((level.price - 200.25).abs() < 0.01);
        assert!((level.size - 5.5).abs() < 0.01);

        // Malformed entry
        let bad_json = r#"{"px": "not_a_number", "sz": "10.0"}"#;
        let bad_value: serde_json::Value = serde_json::from_str(bad_json).unwrap();
        let bad_result = parse_level_entry(&bad_value);
        assert!(bad_result.is_none());
    }

    // ───────── Fix A5: Deterministic timeout tests ─────────
    // NOTE: Uses very short real timeouts (1 ms) since `test-util`
    // (start_paused) is not available in this workspace's tokio features.

    #[tokio::test]
    async fn with_timeout_fires_on_pending_future() {
        // A future that never resolves must trigger the timeout.
        let result = with_timeout(
            Duration::from_millis(1),
            "test_connect",
            std::future::pending::<()>(),
        )
        .await;
        assert!(result.is_err(), "expected timeout error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("timed out"),
            "error should mention timeout: {msg}"
        );
    }

    #[tokio::test]
    async fn with_timeout_passes_through_on_immediate_resolve() {
        let result = with_timeout(Duration::from_secs(5), "test_connect", async { 42u64 }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn read_timeout_path_returns_error() {
        // Simulate a WS read that hangs forever — the read timeout should fire.
        let read_timeout = Duration::from_millis(1);
        let result = tokio::time::timeout(read_timeout, std::future::pending::<Option<()>>()).await;
        assert!(
            result.is_err(),
            "read loop should time out on a hanging stream"
        );
    }

    #[tokio::test]
    async fn connect_timeout_returns_error_msg() {
        // with_timeout label must appear in the error message for operator diagnostics.
        let err = with_timeout(
            Duration::from_millis(1),
            "public WS connect",
            std::future::pending::<()>(),
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("public WS connect"), "msg: {msg}");
        assert!(msg.contains("1ms"), "msg should include duration: {msg}");
    }

    #[test]
    fn freshness_book_event_ns_independent_of_parsed_ns() {
        // Verify that last_book_event_ns and last_parsed_ns are independent:
        // non-book WS messages that update last_parsed_ns should NOT affect
        // the watchdog anchor (which uses last_book_event_ns).
        let freshness = Freshness::default();

        // Simulate: connect happened at 1000
        let connect_start_ns = 1_000;

        // Initially anchor = connect_start_ns (no book events)
        assert_eq!(
            freshness.anchor_with_connect_start(connect_start_ns),
            connect_start_ns
        );

        // Simulate: a non-book message updates last_parsed_ns but NOT last_book_event_ns
        freshness.last_parsed_ns.store(5_000, Ordering::Relaxed);

        // Anchor should still be connect_start_ns because no BOOK events happened
        assert_eq!(
            freshness.anchor_with_connect_start(connect_start_ns),
            connect_start_ns,
            "non-book parsed events must not advance watchdog anchor"
        );

        // Simulate: a book event updates last_book_event_ns
        freshness.last_book_event_ns.store(6_000, Ordering::Relaxed);

        // NOW the anchor should advance
        assert_eq!(
            freshness.anchor_with_connect_start(connect_start_ns),
            6_000,
            "book events must advance watchdog anchor"
        );
    }

    #[test]
    fn hl_forward_audit_snapshot_resets_interval_counters() {
        let stats = HlForwardAudit::default();
        stats.observe_send_block_ms(2);
        stats.observe_send_block_ms(9);
        stats.observe_send_block_ms(80);
        stats.observe_send_block_ms(300);
        stats.observe_send_err();
        stats.observe_coalesced_drop(3);
        stats.observe_pending_take();
        stats.observe_pending_take();
        stats.observe_ts_missing_or_zero();
        stats.observe_ts_clamped_past_skew();
        stats.observe_ts_clamped_future_skew();
        stats.observe_ts_kept_exchange();
        stats.observe_ts_skew(1_250, 450);

        let first = stats.snapshot_and_reset();
        assert_eq!(first.send_block_max_ms, 300);
        assert_eq!(first.send_block_gt_5ms, 3);
        assert_eq!(first.send_block_gt_50ms, 2);
        assert_eq!(first.send_block_gt_250ms, 1);
        assert_eq!(first.forward_send_count, 4);
        assert_eq!(first.forward_send_err_count, 1);
        assert_eq!(first.coalesced_drop_count, 3);
        assert_eq!(first.pending_take_count, 2);
        assert_eq!(first.ts_missing_or_zero_count, 1);
        assert_eq!(first.ts_clamped_past_skew_count, 1);
        assert_eq!(first.ts_clamped_future_skew_count, 1);
        assert_eq!(first.ts_policy_applied_count, 2);
        assert_eq!(first.ts_kept_exchange_count, 1);
        assert_eq!(first.ts_past_skew_max_ms, 1_250);
        assert_eq!(first.ts_future_skew_max_ms, 450);

        let second = stats.snapshot_and_reset();
        assert_eq!(second.send_block_max_ms, 0);
        assert_eq!(second.send_block_gt_5ms, 0);
        assert_eq!(second.send_block_gt_50ms, 0);
        assert_eq!(second.send_block_gt_250ms, 0);
        assert_eq!(second.forward_send_count, 0);
        assert_eq!(second.forward_send_err_count, 0);
        assert_eq!(second.coalesced_drop_count, 0);
        assert_eq!(second.pending_take_count, 0);
        assert_eq!(second.ts_missing_or_zero_count, 0);
        assert_eq!(second.ts_clamped_past_skew_count, 0);
        assert_eq!(second.ts_clamped_future_skew_count, 0);
        assert_eq!(second.ts_policy_applied_count, 0);
        assert_eq!(second.ts_kept_exchange_count, 0);
        assert_eq!(second.ts_past_skew_max_ms, 0);
        assert_eq!(second.ts_future_skew_max_ms, 0);
    }

    #[test]
    fn hl_timestamp_policy_defaults_to_disabled_when_env_unset() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard_enabled = EnvVarGuard::new("PARAPHINA_HL_TS_HARDENING");
        let _guard_past = EnvVarGuard::new("PARAPHINA_HL_TS_MAX_PAST_SKEW_MS");
        let _guard_future = EnvVarGuard::new("PARAPHINA_HL_TS_MAX_FUTURE_SKEW_MS");
        std::env::remove_var("PARAPHINA_HL_TS_HARDENING");
        std::env::remove_var("PARAPHINA_HL_TS_MAX_PAST_SKEW_MS");
        std::env::remove_var("PARAPHINA_HL_TS_MAX_FUTURE_SKEW_MS");

        let policy = HlTimestampPolicy::from_env();
        assert!(!policy.enabled);
        assert_eq!(policy.max_past_skew_ms, HL_TS_MAX_PAST_SKEW_MS_DEFAULT);
        assert_eq!(policy.max_future_skew_ms, HL_TS_MAX_FUTURE_SKEW_MS_DEFAULT);
    }

    #[test]
    fn hl_timestamp_policy_disabled_preserves_exchange_timestamp() {
        let audit = HlForwardAudit::default();
        let policy = HlTimestampPolicy {
            enabled: false,
            max_past_skew_ms: 10,
            max_future_skew_ms: 10,
        };

        assert_eq!(apply_hl_l2_timestamp_policy(0, 20_000, policy, &audit), 0);
        assert_eq!(
            apply_hl_l2_timestamp_policy(15_000, 20_000, policy, &audit),
            15_000
        );

        let snapshot = audit.snapshot_and_reset();
        assert_eq!(snapshot.ts_missing_or_zero_count, 1);
        assert_eq!(snapshot.ts_clamped_past_skew_count, 0);
        assert_eq!(snapshot.ts_clamped_future_skew_count, 0);
        assert_eq!(snapshot.ts_policy_applied_count, 0);
        assert_eq!(snapshot.ts_kept_exchange_count, 0);
        assert_eq!(snapshot.ts_past_skew_max_ms, 5_000);
    }

    #[test]
    fn hl_timestamp_policy_enabled_clamps_zero_past_and_future_skew() {
        let audit = HlForwardAudit::default();
        let policy = HlTimestampPolicy {
            enabled: true,
            max_past_skew_ms: 1_000,
            max_future_skew_ms: 250,
        };

        assert_eq!(
            apply_hl_l2_timestamp_policy(0, 20_000, policy, &audit),
            20_000
        );
        assert_eq!(
            apply_hl_l2_timestamp_policy(14_000, 20_500, policy, &audit),
            20_500
        );
        assert_eq!(
            apply_hl_l2_timestamp_policy(21_000, 20_500, policy, &audit),
            20_500
        );
        assert_eq!(
            apply_hl_l2_timestamp_policy(20_450, 20_500, policy, &audit),
            20_450
        );

        let snapshot = audit.snapshot_and_reset();
        assert_eq!(snapshot.ts_missing_or_zero_count, 1);
        assert_eq!(snapshot.ts_clamped_past_skew_count, 1);
        assert_eq!(snapshot.ts_clamped_future_skew_count, 1);
        assert_eq!(snapshot.ts_policy_applied_count, 3);
        assert_eq!(snapshot.ts_kept_exchange_count, 1);
        assert_eq!(snapshot.ts_past_skew_max_ms, 6_500);
        assert_eq!(snapshot.ts_future_skew_max_ms, 500);
    }
}

#[derive(Debug, Deserialize)]
struct HyperliquidWsPostEnvelope {
    data: HyperliquidWsPostData,
}

#[derive(Debug, Deserialize)]
struct HyperliquidWsPostData {
    id: u64,
    response: HyperliquidWsPostActionResponse,
}

#[derive(Debug, Deserialize)]
struct HyperliquidWsPostActionResponse {
    #[serde(rename = "type")]
    response_type: String,
    payload: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "status", content = "response")]
enum HyperliquidExchangeResponseStatus {
    #[serde(rename = "ok")]
    Ok(HyperliquidExchangeResponse),
    #[serde(rename = "err")]
    Err(serde_json::Value),
}

#[derive(Debug, Deserialize, Clone)]
struct HyperliquidExchangeResponse {
    #[serde(rename = "type")]
    response_type: String,
    data: Option<HyperliquidExchangeDataStatuses>,
}

#[derive(Debug, Deserialize, Clone)]
struct HyperliquidExchangeDataStatuses {
    statuses: Vec<HyperliquidExchangeDataStatus>,
}

#[derive(Debug, Deserialize, Clone)]
struct HyperliquidRestingOrder {
    oid: u64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct HyperliquidFilledOrder {
    total_sz: String,
    avg_px: String,
    oid: u64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
enum HyperliquidExchangeDataStatus {
    Success,
    WaitingForFill,
    WaitingForTrigger,
    Error(String),
    Resting(HyperliquidRestingOrder),
    Filled(HyperliquidFilledOrder),
}

fn normalize_hl_address(address: &str) -> anyhow::Result<String> {
    let trimmed = address.trim();
    let hex_part = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    if hex_part.len() != 40 || !hex_part.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("invalid Hyperliquid address format: {address}");
    }
    Ok(format!("0x{}", hex_part.to_ascii_lowercase()))
}

fn decode_hl_address_bytes(address: &str) -> anyhow::Result<[u8; 20]> {
    let normalized = normalize_hl_address(address)?;
    let mut out = [0u8; 20];
    hex::decode_to_slice(&normalized[2..], &mut out)
        .map_err(|err| anyhow::anyhow!("invalid Hyperliquid address hex: {err}"))?;
    Ok(out)
}

fn keccak_bytes(input: impl AsRef<[u8]>) -> [u8; 32] {
    let digest = Keccak256::digest(input.as_ref());
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn abi_word_from_u64(value: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..].copy_from_slice(&value.to_be_bytes());
    out
}

fn abi_word_from_address(address: [u8; 20]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(&address);
    out
}

fn serialize_action_for_hl_signing(action: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    let action_type = action
        .get("type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("Hyperliquid action missing type"))?;
    match action_type {
        "order" => {
            let parsed: HlOrderAction = serde_json::from_value(action.clone())?;
            Ok(rmp_serde::to_vec_named(&parsed)?)
        }
        "cancel" => {
            let parsed: HlCancelOidAction = serde_json::from_value(action.clone())?;
            Ok(rmp_serde::to_vec_named(&parsed)?)
        }
        "cancelByCloid" => {
            let parsed: HlCancelCloidAction = serde_json::from_value(action.clone())?;
            Ok(rmp_serde::to_vec_named(&parsed)?)
        }
        "cancelAll" => {
            let parsed: HlCancelAllAction = serde_json::from_value(action.clone())?;
            Ok(rmp_serde::to_vec_named(&parsed)?)
        }
        "noop" => {
            let parsed: HlNoopAction = serde_json::from_value(action.clone())?;
            Ok(rmp_serde::to_vec_named(&parsed)?)
        }
        "reserveRequestWeight" => {
            let parsed: HlReserveRequestWeightAction = serde_json::from_value(action.clone())?;
            Ok(rmp_serde::to_vec_named(&parsed)?)
        }
        other => anyhow::bail!("unsupported Hyperliquid action type for signing parity: {other}"),
    }
}

fn action_hash(
    action: &serde_json::Value,
    vault_address: Option<&str>,
    nonce: TimestampMs,
    expires_after: Option<TimestampMs>,
) -> anyhow::Result<[u8; 32]> {
    let mut packed = serialize_action_for_hl_signing(action)?;
    packed.extend_from_slice(&(nonce as u64).to_be_bytes());
    if let Some(vault_address) = vault_address {
        packed.push(1);
        packed.extend_from_slice(&decode_hl_address_bytes(vault_address)?);
    } else {
        packed.push(0);
    }
    if let Some(expires_after) = expires_after {
        packed.push(0);
        packed.extend_from_slice(&(expires_after as u64).to_be_bytes());
    }
    Ok(keccak_bytes(packed))
}

fn l1_domain_separator() -> [u8; 32] {
    let type_hash = keccak_bytes(
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
    );
    let name_hash = keccak_bytes("Exchange");
    let version_hash = keccak_bytes("1");
    let chain_id = abi_word_from_u64(1337);
    let verifying_contract = abi_word_from_address([0u8; 20]);
    let mut encoded = Vec::with_capacity(32 * 5);
    encoded.extend_from_slice(&type_hash);
    encoded.extend_from_slice(&name_hash);
    encoded.extend_from_slice(&version_hash);
    encoded.extend_from_slice(&chain_id);
    encoded.extend_from_slice(&verifying_contract);
    keccak_bytes(encoded)
}

fn l1_struct_hash(connection_id: [u8; 32], is_mainnet: bool) -> [u8; 32] {
    let type_hash = keccak_bytes("Agent(string source,bytes32 connectionId)");
    let source_hash = keccak_bytes(if is_mainnet { "a" } else { "b" });
    let mut encoded = Vec::with_capacity(32 * 3);
    encoded.extend_from_slice(&type_hash);
    encoded.extend_from_slice(&source_hash);
    encoded.extend_from_slice(&connection_id);
    keccak_bytes(encoded)
}

fn eip712_l1_digest(connection_id: [u8; 32], is_mainnet: bool) -> [u8; 32] {
    let domain_separator = l1_domain_separator();
    let struct_hash = l1_struct_hash(connection_id, is_mainnet);
    let mut encoded = Vec::with_capacity(2 + 32 + 32);
    encoded.extend_from_slice(b"\x19\x01");
    encoded.extend_from_slice(&domain_separator);
    encoded.extend_from_slice(&struct_hash);
    keccak_bytes(encoded)
}

fn format_hl_signature(signature: &Signature, recovery_id: RecoveryId) -> serde_json::Value {
    let (r, s) = signature.split_bytes();
    let v: u8 = 27 + recovery_id.to_byte();
    json!({
        "r": format!("0x{}", hex::encode(r)),
        "s": format!("0x{}", hex::encode(s)),
        "v": v,
    })
}

fn sign_action(
    action: &serde_json::Value,
    vault_address: Option<&str>,
    nonce: TimestampMs,
    is_mainnet: bool,
    signing_key: &SigningKey,
) -> anyhow::Result<serde_json::Value> {
    let connection_id = action_hash(action, vault_address, nonce, None)?;
    let digest = eip712_l1_digest(connection_id, is_mainnet);
    let (sig, recid) = signing_key.sign_prehash_recoverable(&digest)?;
    Ok(format_hl_signature(&sig, recid))
}

#[cfg(test)]
fn recover_action_signer_address(
    action: &serde_json::Value,
    vault_address: Option<&str>,
    nonce: TimestampMs,
    is_mainnet: bool,
    signature: &serde_json::Value,
) -> anyhow::Result<String> {
    let connection_id = action_hash(action, vault_address, nonce, None)?;
    let digest = eip712_l1_digest(connection_id, is_mainnet);
    let r = signature
        .get("r")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("signature missing r"))?;
    let s = signature
        .get("s")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("signature missing s"))?;
    let v = signature
        .get("v")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| anyhow::anyhow!("signature missing v"))?;
    let r_bytes = hex::decode(r.trim_start_matches("0x"))?;
    let s_bytes = hex::decode(s.trim_start_matches("0x"))?;
    let mut sig_bytes = [0u8; 64];
    sig_bytes[..32].copy_from_slice(&r_bytes);
    sig_bytes[32..].copy_from_slice(&s_bytes);
    let signature = Signature::try_from(sig_bytes.as_slice())?;
    let recovery_id = RecoveryId::try_from((v as u8).saturating_sub(27))?;
    let verifying_key =
        k256::ecdsa::VerifyingKey::recover_from_prehash(&digest, &signature, recovery_id)?;
    let pubkey = verifying_key.to_encoded_point(false);
    let pubkey_bytes = pubkey.as_bytes();
    let hashed = keccak_bytes(&pubkey_bytes[1..]);
    Ok(format!("0x{}", hex::encode(&hashed[12..])))
}

fn parse_ws_post_response(value: &serde_json::Value) -> Option<(u64, String, anyhow::Result<()>)> {
    let envelope: HyperliquidWsPostEnvelope = serde_json::from_value(value.clone()).ok()?;
    let id = envelope.data.id;
    let raw_response_type = envelope.data.response.response_type;
    let response_type = ws_post_sanitized_label(&raw_response_type).to_string();
    let payload = envelope.data.response.payload;

    if raw_response_type.eq_ignore_ascii_case("error") {
        return Some((
            id,
            response_type,
            Err(anyhow::anyhow!(
                "Hyperliquid ws_post failed sanitized_reason={}",
                ws_post_sanitized_payload_reason(&payload)
            )),
        ));
    }

    if !raw_response_type.eq_ignore_ascii_case("action") {
        return Some((id, response_type, Ok(())));
    }

    let action_status: HyperliquidExchangeResponseStatus =
        match serde_json::from_value(payload.clone()) {
            Ok(parsed) => parsed,
            Err(_err) => {
                return Some((
                    id,
                    response_type,
                    Err(anyhow::anyhow!(
                        "Hyperliquid ws_post action decode_failed sanitized_context=true"
                    )),
                ));
            }
        };

    let result = match action_status {
        HyperliquidExchangeResponseStatus::Err(detail) => Err(anyhow::anyhow!(
            "Hyperliquid ws_post action top_level_err sanitized_reason={}",
            ws_post_sanitized_payload_reason(&detail)
        )),
        HyperliquidExchangeResponseStatus::Ok(exchange_response) => {
            if let Some(exchange_data) = exchange_response.data {
                let mut errors = Vec::new();
                for status in exchange_data.statuses {
                    match status {
                        HyperliquidExchangeDataStatus::Error(detail) => errors.push(detail),
                        HyperliquidExchangeDataStatus::Success
                        | HyperliquidExchangeDataStatus::WaitingForFill
                        | HyperliquidExchangeDataStatus::WaitingForTrigger
                        | HyperliquidExchangeDataStatus::Resting(_)
                        | HyperliquidExchangeDataStatus::Filled(_) => {}
                    }
                }
                if errors.is_empty() {
                    Ok(())
                } else {
                    let reason_counts = ws_post_exchange_error_reason_counts(&errors);
                    Err(anyhow::anyhow!(
                        "Hyperliquid ws_post action exchange_errors_count={} exchange_reasons={} response_type={}",
                        errors.len(),
                        reason_counts,
                        ws_post_sanitized_label(&exchange_response.response_type)
                    ))
                }
            } else {
                Ok(())
            }
        }
    };

    Some((id, response_type, result))
}

fn ws_post_sanitized_label(raw: &str) -> &'static str {
    let lower = raw.to_ascii_lowercase();
    match lower.as_str() {
        "order" => "order",
        "action" => "action",
        "error" => "error",
        "ok" => "ok",
        "err" => "err",
        _ => "other",
    }
}

fn ws_post_sanitized_payload_reason(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::String(raw) => ws_post_exchange_error_reason(raw),
        serde_json::Value::Object(_) => "object_payload",
        serde_json::Value::Array(_) => "array_payload",
        serde_json::Value::Null => "null_payload",
        serde_json::Value::Bool(_) => "bool_payload",
        serde_json::Value::Number(_) => "number_payload",
    }
}

fn ws_post_exchange_error_reason(raw: &str) -> &'static str {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("post only") && lower.contains("immediately matched") {
        "post_only_would_match"
    } else if lower.contains("badalopx") {
        "bad_alo_px"
    } else if lower.contains("insufficient") && lower.contains("margin") {
        "insufficient_margin"
    } else if lower.contains("rate") && lower.contains("limit")
        || lower.contains("too many requests")
        || lower.contains("429")
    {
        "rate_limited"
    } else {
        "exchange_error"
    }
}

fn ws_post_exchange_error_reason_counts(errors: &[String]) -> String {
    let mut counts = std::collections::BTreeMap::<&'static str, usize>::new();
    for err in errors {
        *counts
            .entry(ws_post_exchange_error_reason(err))
            .or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(reason, count)| format!("{reason}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn translate_private_events(msg: &serde_json::Value) -> Vec<ExecutionEvent> {
    let mut out = Vec::new();
    let channel = msg.get("channel").and_then(|v| v.as_str()).unwrap_or("");
    let seq = msg.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let data = match msg.get("data") {
        Some(val) => val,
        None => return out,
    };

    match channel {
        "userEvents" => {
            if let Some(event) = parse_user_event(data, seq) {
                out.push(event);
            }
        }
        "userFills" => {
            if let Some(fills) = data.get("fills").and_then(|v| v.as_array()) {
                for fill in fills {
                    if let Some(event) = parse_user_fill(fill, data, seq) {
                        out.push(event);
                    }
                }
            }
        }
        _ => {}
    }

    out
}

pub fn translate_account_event(
    msg: &serde_json::Value,
    default_venue_id: Option<&str>,
    venue_index: usize,
) -> Option<AccountEvent> {
    let channel = msg.get("channel").and_then(|v| v.as_str()).unwrap_or("");
    if channel != "userState" && channel != "clearinghouseState" {
        return None;
    }
    let data = msg.get("data")?;
    let payload = match channel {
        "clearinghouseState" => data.get("clearinghouseState").unwrap_or(data),
        "userState" => data.get("userState").unwrap_or(data),
        _ => data,
    };
    let mut snapshot = parse_account_snapshot_with_meta(payload, default_venue_id, venue_index)?;
    if snapshot.timestamp_ms <= 0 {
        snapshot.timestamp_ms = now_ms();
    }
    Some(AccountEvent::Snapshot(snapshot))
}

fn parse_user_event(data: &serde_json::Value, seq: u64) -> Option<ExecutionEvent> {
    let event_type = data.get("event").and_then(|v| v.as_str()).unwrap_or("");
    let status = data.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let order = data.get("order")?;
    let order_id = order.get("oid")?.as_str()?.to_string();
    let client_order_id = order
        .get("cloid")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let timestamp_ms = order
        .get("timestamp")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("timestamp").and_then(|v| v.as_i64()))
        .unwrap_or(0);
    let venue_id = order
        .get("coin")
        .or_else(|| data.get("coin"))
        .and_then(|v| v.as_str())
        .unwrap_or("UNKNOWN")
        .to_string();
    let venue_index = 0;

    if matches!(event_type, "cancel") || matches!(status, "canceled" | "cancelled") {
        return Some(ExecutionEvent::CancelAccepted(
            super::super::types::CancelAccepted {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                order_id,
            },
        ));
    }

    if matches!(status, "rejected") {
        let reason = data
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("rejected")
            .to_string();
        return Some(ExecutionEvent::OrderRejected(
            super::super::types::OrderRejected {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                order_id: Some(order_id),
                client_order_id: None,
                purpose: None,
                reduce_only: None,
                reason,
            },
        ));
    }

    let side = parse_side(order.get("side")?)?;
    let price = order
        .get("limitPx")
        .or_else(|| order.get("px"))
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let size = order
        .get("sz")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let purpose = parse_purpose(order.get("purpose")).unwrap_or(OrderPurpose::Mm);

    Some(ExecutionEvent::OrderAccepted(
        super::super::types::OrderAccepted {
            venue_index,
            venue_id,
            seq,
            timestamp_ms,
            order_id,
            client_order_id,
            side,
            price,
            size,
            purpose,
        },
    ))
}

fn parse_user_fill(
    fill: &serde_json::Value,
    data: &serde_json::Value,
    seq: u64,
) -> Option<ExecutionEvent> {
    let order_id = fill
        .get("oid")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let client_order_id = fill
        .get("cloid")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let fill_id = fill
        .get("tid")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let timestamp_ms = fill
        .get("timestamp")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("timestamp").and_then(|v| v.as_i64()))
        .unwrap_or(0);
    let venue_id = "hyperliquid".to_string();
    let venue_index = 0;
    let side = parse_side(fill.get("side")?)?;
    let price = fill
        .get("px")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let size = fill
        .get("sz")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let fee_bps = fill
        .get("feeBps")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let purpose = parse_purpose(fill.get("purpose")).unwrap_or(OrderPurpose::Mm);
    let phase51_native_role = fill
        .get("crossed")
        .and_then(|v| v.as_bool())
        .map(|crossed| Phase51ForwardRefreshNativeRole::Hyperliquid { crossed });

    Some(ExecutionEvent::Filled(super::super::types::Fill {
        venue_index,
        venue_id,
        seq,
        timestamp_ms,
        order_id,
        client_order_id,
        fill_id,
        phase51_target_key: None,
        phase51_native_role,
        phase51_lighter_native_limit: None,
        side,
        price,
        size,
        purpose,
        fee_bps,
    }))
}

fn parse_side(value: &serde_json::Value) -> Option<Side> {
    let raw = value.as_str()?;
    match raw {
        "B" | "b" | "buy" | "Buy" => Some(Side::Buy),
        "S" | "s" | "sell" | "Sell" => Some(Side::Sell),
        _ => None,
    }
}

fn parse_purpose(value: Option<&serde_json::Value>) -> Option<OrderPurpose> {
    let raw = value.and_then(|v| v.as_str())?;
    match raw {
        "Mm" | "mm" | "MM" => Some(OrderPurpose::Mm),
        "Exit" | "exit" => Some(OrderPurpose::Exit),
        "Hedge" | "hedge" => Some(OrderPurpose::Hedge),
        _ => None,
    }
}

pub fn parse_account_snapshot(data: &serde_json::Value) -> Option<AccountSnapshot> {
    parse_account_snapshot_with_meta(data, None, 0)
}

fn parse_account_snapshot_with_meta(
    data: &serde_json::Value,
    default_venue_id: Option<&str>,
    venue_index: usize,
) -> Option<AccountSnapshot> {
    if data.get("marginSummary").is_some() || data.get("assetPositions").is_some() {
        return parse_clearinghouse_account_snapshot(data, default_venue_id, venue_index);
    }

    let seq = data.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = data.get("time").and_then(parse_i64_value).unwrap_or(0);
    let venue_id = data
        .get("coin")
        .and_then(|v| v.as_str())
        .or(default_venue_id)
        .unwrap_or("TAO");

    let positions = data
        .get("positions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pos| {
                    let symbol = pos
                        .get("coin")
                        .or_else(|| pos.get("symbol"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(venue_id);
                    let size = pos.get("size").and_then(parse_f64_value)?;
                    let entry_price = pos
                        .get("entryPx")
                        .or_else(|| pos.get("entry_price"))
                        .and_then(parse_f64_value)?;
                    Some(PositionSnapshot {
                        symbol: symbol.to_string(),
                        size,
                        entry_price,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let balances = data
        .get("balances")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|bal| {
                    let asset = bal.get("asset")?.as_str()?.to_string();
                    let total = bal.get("total").and_then(parse_f64_value)?;
                    let available = bal.get("available").and_then(parse_f64_value)?;
                    Some(BalanceSnapshot {
                        asset,
                        total,
                        available,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let margin = data.get("margin")?;
    let margin = MarginSnapshot {
        balance_usd: margin.get("balance").and_then(parse_f64_value)?,
        used_usd: margin.get("used").and_then(parse_f64_value)?,
        available_usd: margin.get("available").and_then(parse_f64_value)?,
    };

    let liquidation = data.get("liquidation")?;
    let liquidation = LiquidationSnapshot {
        price_liq: liquidation.get("priceLiq").and_then(parse_f64_value),
        dist_liq_sigma: liquidation.get("distLiqSigma").and_then(parse_f64_value),
    };

    let funding_8h = data.get("funding8h").and_then(parse_f64_value);

    Some(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        positions,
        balances,
        funding_8h,
        margin,
        liquidation,
    })
}

fn parse_clearinghouse_account_snapshot(
    data: &serde_json::Value,
    default_venue_id: Option<&str>,
    venue_index: usize,
) -> Option<AccountSnapshot> {
    let margin_summary = data
        .get("marginSummary")
        .or_else(|| data.get("crossMarginSummary"))?;
    let balance_usd = margin_summary
        .get("accountValue")
        .and_then(parse_f64_value)?;
    let used_usd = margin_summary
        .get("totalMarginUsed")
        .and_then(parse_f64_value)
        .unwrap_or(0.0);
    let available_usd = data
        .get("withdrawable")
        .and_then(parse_f64_value)
        .unwrap_or_else(|| (balance_usd - used_usd).max(0.0));
    let venue_id = data
        .get("coin")
        .and_then(|v| v.as_str())
        .or(default_venue_id)
        .unwrap_or("TAO");
    let positions = data
        .get("assetPositions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| {
                    let pos = entry.get("position").unwrap_or(entry);
                    let symbol = pos
                        .get("coin")
                        .or_else(|| pos.get("symbol"))
                        .and_then(|v| v.as_str())?;
                    let size = pos
                        .get("szi")
                        .or_else(|| pos.get("size"))
                        .and_then(parse_f64_value)?;
                    let entry_price = pos
                        .get("entryPx")
                        .or_else(|| pos.get("entry_price"))
                        .and_then(parse_f64_value)?;
                    Some(PositionSnapshot {
                        symbol: symbol.to_string(),
                        size,
                        entry_price,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let price_liq = data
        .get("assetPositions")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter().find_map(|entry| {
                let pos = entry.get("position").unwrap_or(entry);
                pos.get("liqPx").and_then(parse_f64_value)
            })
        });

    Some(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq: data.get("seq").and_then(|v| v.as_u64()).unwrap_or(0),
        timestamp_ms: data.get("time").and_then(parse_i64_value).unwrap_or(0),
        positions,
        balances: vec![BalanceSnapshot {
            asset: "USD".to_string(),
            total: balance_usd,
            available: available_usd,
        }],
        funding_8h: None,
        margin: MarginSnapshot {
            balance_usd,
            used_usd,
            available_usd,
        },
        liquidation: LiquidationSnapshot {
            price_liq,
            dist_liq_sigma: None,
        },
    })
}
