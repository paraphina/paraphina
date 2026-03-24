//! Aster connector (public WS market data + fixtures, feature-gated).

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex as StdMutex, OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use reqwest::Client;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::super::gateway::{
    BoxFuture, LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest,
    LiveRestCancelRequest, LiveRestClient, LiveRestPlaceRequest, LiveRestResponse, LiveResult,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, FundingUpdate,
    LiquidationSnapshot, MarginSnapshot, MarketDataEvent, OpenOrderSnapshot, OrderSnapshot,
    PositionSnapshot, TopOfBook,
};
use crate::live::MarketPublisher;
use crate::types::{FundingSource, SettlementPriceKind, Side, TimeInForce, TimestampMs};

#[cfg(feature = "live_aster")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_aster")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_aster")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_aster")]
pub const SUPPORTS_EXECUTION: bool = true;

// FIX: Normalized default to 10,000ms to match other venues (was 1,800ms)
const ASTER_STALE_MS_DEFAULT: u64 = 10_000;
const ASTER_WATCHDOG_TICK_MS: u64 = 200;
const ASTER_MARKET_PUB_QUEUE_CAP_LIVE: usize = 256;
const ASTER_MARKET_PUB_QUEUE_CAP_FIXTURE: usize = 4096;
const ASTER_MARKET_PUB_DRAIN_MAX: usize = 64;

static MONO_START: OnceLock<Instant> = OnceLock::new();
static ASTER_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static ASTER_RECONNECT_COUNTS: OnceLock<StdMutex<BTreeMap<&'static str, u64>>> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// FIX: Configurable stale threshold via env var, normalized default to 10,000ms
fn aster_stale_ms() -> u64 {
    std::env::var("PARAPHINA_ASTER_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(ASTER_STALE_MS_DEFAULT)
}

#[allow(dead_code)]
fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

fn env_is_true(key: &str) -> bool {
    std::env::var(key)
        .map(|value| value.eq_ignore_ascii_case("true") || value == "1")
        .unwrap_or(false)
}

fn is_aster_fixture_mode_now() -> bool {
    env_is_true("ASTER_FIXTURE_MODE")
        || std::env::var_os("ASTER_FIXTURE_DIR").is_some()
        || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
}

fn aster_ws_audit_enabled() -> bool {
    *ASTER_WS_AUDIT_ENABLED.get_or_init(|| {
        std::env::var("PARAPHINA_WS_AUDIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn aster_audit_reconnect(reason: &'static str) {
    if !aster_ws_audit_enabled() {
        return;
    }
    let mut counts = ASTER_RECONNECT_COUNTS
        .get_or_init(|| StdMutex::new(BTreeMap::new()))
        .lock()
        .expect("aster reconnect audit mutex poisoned");
    let count = counts
        .entry(reason)
        .and_modify(|value| *value += 1)
        .or_insert(1);
    eprintln!(
        "WS_AUDIT venue=aster reconnect_reason={} count={}",
        reason, *count
    );
}

#[derive(Debug, Clone, Copy, Default)]
struct FreshnessAges {
    ws_rx_age_ms: u64,
    data_rx_age_ms: u64,
    parsed_age_ms: u64,
    pub_age_ms: u64,
    book_age_ms: u64,
    anchor_age_ms: u64,
}

fn aster_freshness_ages(freshness: &Freshness, connect_start_ns: u64) -> FreshnessAges {
    let now = mono_now_ns();
    let ws_rx = freshness.last_ws_rx_ns.load(Ordering::Relaxed);
    let data_rx = freshness.last_data_rx_ns.load(Ordering::Relaxed);
    let parsed = freshness.last_parsed_ns.load(Ordering::Relaxed);
    let published = freshness.last_published_ns.load(Ordering::Relaxed);
    let book = freshness.last_book_event_ns.load(Ordering::Relaxed);
    let anchor = freshness.anchor_with_connect_start(connect_start_ns);
    FreshnessAges {
        ws_rx_age_ms: if ws_rx == 0 { 0 } else { age_ms(now, ws_rx) },
        data_rx_age_ms: if data_rx == 0 {
            0
        } else {
            age_ms(now, data_rx)
        },
        parsed_age_ms: if parsed == 0 { 0 } else { age_ms(now, parsed) },
        pub_age_ms: if published == 0 {
            0
        } else {
            age_ms(now, published)
        },
        book_age_ms: if book == 0 { 0 } else { age_ms(now, book) },
        anchor_age_ms: if anchor == 0 { 0 } else { age_ms(now, anchor) },
    }
}

fn aster_audit_book_recovery(stage: &'static str, phase: &'static str, fields: &[(&str, String)]) {
    if !aster_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=aster component=book_recovery stage={} phase={}",
        stage, phase
    );
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(value);
    }
    eprintln!("{line}");
}

fn aster_audit_funding_ws(stage: &'static str, fields: &[(&str, String)]) {
    if !aster_ws_audit_enabled() {
        return;
    }
    let mut line = format!("WS_AUDIT venue=aster component=funding_ws stage={stage}");
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(value);
    }
    eprintln!("{line}");
}

#[derive(Debug, Clone, Copy)]
pub struct AsterPublicRestBudgetState {
    pub fail_streak: u32,
    pub next_allowed_ns: u64,
    pub last_http_status: Option<u16>,
    pub last_weight_1m: Option<u64>,
    pub last_cooldown_ms: u64,
    pub last_failure_class: &'static str,
    pub shared_cooldown_until_ns: u64,
    pub shared_cooldown_ms: u64,
    pub shared_http_status: Option<u16>,
    pub shared_weight_1m: Option<u64>,
    pub shared_consumer: &'static str,
    pub shared_priority: &'static str,
    pub recovery_active: bool,
}

impl Default for AsterPublicRestBudgetState {
    fn default() -> Self {
        Self {
            fail_streak: 0,
            next_allowed_ns: 0,
            last_http_status: None,
            last_weight_1m: None,
            last_cooldown_ms: 0,
            last_failure_class: "",
            shared_cooldown_until_ns: 0,
            shared_cooldown_ms: 0,
            shared_http_status: None,
            shared_weight_1m: None,
            shared_consumer: "",
            shared_priority: "",
            recovery_active: false,
        }
    }
}

pub type AsterPublicRestBudgetHandle = Arc<StdMutex<AsterPublicRestBudgetState>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsterProbeBudgetDecision {
    Allow,
    Suppressed {
        reason: &'static str,
        cooldown_ms: u64,
        fail_streak: u32,
        shared_consumer: &'static str,
        shared_priority: &'static str,
    },
}

fn classify_snapshot_failure(status: Option<u16>) -> &'static str {
    match status {
        Some(418) => "ip_banned",
        Some(429) => "rate_limited",
        Some(code) if (500..=599).contains(&code) => "upstream_5xx",
        Some(code) if (400..=499).contains(&code) => "http_4xx",
        Some(_) => "http_other",
        None => "transport_or_parse",
    }
}

fn snapshot_backoff_ms(status: Option<u16>, fail_streak: u32, jitter_seed_ns: u64) -> u64 {
    let (base_ms, max_ms) = match status {
        Some(418) => (30_000_u64, 120_000_u64),
        Some(429) => (1_000_u64, 15_000_u64),
        Some(code) if (500..=599).contains(&code) => (500_u64, 5_000_u64),
        Some(code) if (400..=499).contains(&code) => (750_u64, 5_000_u64),
        Some(_) => (500_u64, 3_000_u64),
        None => (500_u64, 3_000_u64),
    };
    let exp = base_ms.saturating_mul(1_u64 << fail_streak.saturating_sub(1).min(6));
    let capped = exp.min(max_ms);
    let jitter_window = (capped / 3).max(1);
    capped
        .saturating_add(jitter_seed_ns % (jitter_window + 1))
        .min(max_ms)
}

fn shared_public_rest_backoff_ms(
    status: Option<u16>,
    fail_streak: u32,
    jitter_seed_ns: u64,
) -> u64 {
    match status {
        Some(418) | Some(429) => snapshot_backoff_ms(status, fail_streak.max(1), jitter_seed_ns),
        _ => 0,
    }
}

fn snapshot_weight_1m(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers.iter().find_map(|(name, value)| {
        if name.as_str().eq_ignore_ascii_case("x-mbx-used-weight-1m") {
            value.to_str().ok().and_then(|raw| raw.parse::<u64>().ok())
        } else {
            None
        }
    })
}

pub fn aster_probe_budget_decision(
    handle: &AsterPublicRestBudgetHandle,
) -> AsterProbeBudgetDecision {
    let state = *handle
        .lock()
        .expect("aster public rest budget mutex poisoned");
    let now_ns = mono_now_ns();
    if state.shared_cooldown_until_ns > now_ns {
        return AsterProbeBudgetDecision::Suppressed {
            reason: "shared_cooldown",
            cooldown_ms: age_ms(state.shared_cooldown_until_ns, now_ns),
            fail_streak: state.fail_streak,
            shared_consumer: state.shared_consumer,
            shared_priority: state.shared_priority,
        };
    }
    if state.recovery_active || state.fail_streak > 0 {
        return AsterProbeBudgetDecision::Suppressed {
            reason: "recovery_active",
            cooldown_ms: 0,
            fail_streak: state.fail_streak,
            shared_consumer: state.shared_consumer,
            shared_priority: state.shared_priority,
        };
    }
    AsterProbeBudgetDecision::Allow
}

pub fn aster_note_probe_rate_limit(
    handle: &AsterPublicRestBudgetHandle,
    status: u16,
    weight_1m: Option<u64>,
) -> AsterPublicRestBudgetState {
    let mut state = handle
        .lock()
        .expect("aster public rest budget mutex poisoned");
    let cooldown_ms = shared_public_rest_backoff_ms(Some(status), state.fail_streak, mono_now_ns());
    if cooldown_ms > 0 {
        state.shared_cooldown_ms = cooldown_ms;
        state.shared_cooldown_until_ns =
            mono_now_ns().saturating_add(cooldown_ms.saturating_mul(1_000_000));
    }
    state.shared_http_status = Some(status);
    state.shared_weight_1m = weight_1m;
    state.shared_consumer = "probe";
    state.shared_priority = "optional";
    *state
}

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_parsed_ns: AtomicU64,
    last_published_ns: AtomicU64,
    /// Tracks the last time a book event (snapshot or delta) was decoded into a
    /// publishable MarketDataEvent. Used by the watchdog to detect "WS alive but
    /// no book data" scenarios where non-book messages keep last_ws_rx_ns fresh.
    last_book_event_ns: AtomicU64,
}

impl Freshness {
    fn reset_for_new_connection(&self) {
        self.last_ws_rx_ns.store(0, Ordering::Relaxed);
        self.last_data_rx_ns.store(0, Ordering::Relaxed);
        self.last_parsed_ns.store(0, Ordering::Relaxed);
        self.last_published_ns.store(0, Ordering::Relaxed);
        self.last_book_event_ns.store(0, Ordering::Relaxed);
    }

    fn anchor_with_connect_start(&self, connect_start_ns: u64) -> u64 {
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

#[derive(Debug, Clone)]
pub struct AsterConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub market: String,
    pub depth_limit: usize,
    pub venue_index: usize,
    pub venue_id: String,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub recv_window: Option<u64>,
    pub record_dir: Option<PathBuf>,
}

impl AsterConfig {
    pub fn from_env() -> Self {
        let ws_url = std::env::var("ASTER_WS_URL")
            .unwrap_or_else(|_| "wss://fstream.asterdex.com/ws".to_string());
        let rest_url = std::env::var("ASTER_REST_URL")
            .unwrap_or_else(|_| "https://fapi.asterdex.com".to_string());
        let market = std::env::var("ASTER_MARKET").unwrap_or_else(|_| "BTCUSDT".to_string());
        let depth_limit = std::env::var("ASTER_DEPTH_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100);
        let venue_id = std::env::var("ASTER_VENUE").unwrap_or_else(|_| "ASTER".to_string());
        let api_key = std::env::var("ASTER_API_KEY").ok();
        let api_secret = std::env::var("ASTER_API_SECRET").ok();
        let recv_window = std::env::var("ASTER_RECV_WINDOW")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .or(Some(5_000));
        Self {
            ws_url,
            rest_url,
            market,
            depth_limit,
            venue_index: 0,
            venue_id,
            api_key,
            api_secret,
            recv_window,
            record_dir: None,
        }
    }

    pub fn with_record_dir(mut self, dir: PathBuf) -> Self {
        self.record_dir = Some(dir);
        self
    }

    pub fn has_auth(&self) -> bool {
        self.api_key.is_some() && self.api_secret.is_some()
    }

    fn stream_symbol(&self) -> String {
        self.market.to_ascii_lowercase()
    }
}

#[derive(Debug)]
pub struct AsterConnector {
    cfg: AsterConfig,
    http: Client,
    market_publisher: MarketPublisher,
    recorder: Option<Mutex<AsterRecorder>>,
    freshness: Arc<Freshness>,
    snapshot_recovery: AsterPublicRestBudgetHandle,
    is_fixture: bool,
}

impl AsterConnector {
    pub fn new(cfg: AsterConfig, market_tx: mpsc::Sender<MarketDataEvent>) -> Self {
        let recorder = cfg
            .record_dir
            .as_ref()
            .and_then(|dir| AsterRecorder::new(dir).ok())
            .map(Mutex::new);
        let is_fixture = is_aster_fixture_mode_now();
        let cap = if is_aster_fixture_mode_now() {
            ASTER_MARKET_PUB_QUEUE_CAP_FIXTURE
        } else {
            ASTER_MARKET_PUB_QUEUE_CAP_LIVE
        };
        let freshness = Arc::new(Freshness::default());
        let market_publisher = MarketPublisher::new(
            cap,
            ASTER_MARKET_PUB_DRAIN_MAX,
            market_tx.clone(),
            Some(Arc::new(move || is_fixture || is_aster_fixture_mode_now())),
            Arc::new(|event: &MarketDataEvent| {
                matches!(
                    event,
                    MarketDataEvent::L2Delta(_) | MarketDataEvent::L2Snapshot(_)
                )
            }),
            None,
            "aster market_tx closed",
            "aster market publish queue closed",
        );
        let connector = Self {
            cfg,
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("aster http client build"),
            market_publisher,
            recorder,
            freshness,
            snapshot_recovery: Arc::new(StdMutex::new(AsterPublicRestBudgetState::default())),
            is_fixture,
        };
        connector
    }

    fn snapshot_recovery_state(&self) -> AsterPublicRestBudgetState {
        *self
            .snapshot_recovery
            .lock()
            .expect("aster snapshot recovery mutex poisoned")
    }

    pub fn public_rest_budget(&self) -> AsterPublicRestBudgetHandle {
        self.snapshot_recovery.clone()
    }

    fn mark_snapshot_recovery_active(&self) {
        let mut recovery = self
            .snapshot_recovery
            .lock()
            .expect("aster snapshot recovery mutex poisoned");
        recovery.recovery_active = true;
        recovery.shared_consumer = "snapshot";
        recovery.shared_priority = "critical";
    }

    fn mark_book_live(&self) {
        let mut recovery = self
            .snapshot_recovery
            .lock()
            .expect("aster snapshot recovery mutex poisoned");
        recovery.recovery_active = false;
        recovery.shared_consumer = "snapshot";
        recovery.shared_priority = "critical";
    }

    fn record_snapshot_failure(
        &self,
        status: Option<u16>,
        weight_1m: Option<u64>,
    ) -> AsterPublicRestBudgetState {
        let mut recovery = self
            .snapshot_recovery
            .lock()
            .expect("aster snapshot recovery mutex poisoned");
        recovery.fail_streak = recovery.fail_streak.saturating_add(1);
        recovery.last_http_status = status;
        recovery.last_weight_1m = weight_1m;
        recovery.last_failure_class = classify_snapshot_failure(status);
        recovery.last_cooldown_ms =
            snapshot_backoff_ms(status, recovery.fail_streak, mono_now_ns());
        recovery.next_allowed_ns =
            mono_now_ns().saturating_add(recovery.last_cooldown_ms.saturating_mul(1_000_000));
        recovery.recovery_active = true;
        recovery.shared_consumer = "snapshot";
        recovery.shared_priority = "critical";
        let shared_cooldown_ms =
            shared_public_rest_backoff_ms(status, recovery.fail_streak, mono_now_ns());
        if shared_cooldown_ms > 0 {
            recovery.shared_cooldown_ms = shared_cooldown_ms;
            recovery.shared_cooldown_until_ns =
                mono_now_ns().saturating_add(shared_cooldown_ms.saturating_mul(1_000_000));
            recovery.shared_http_status = status;
            recovery.shared_weight_1m = weight_1m;
        }
        *recovery
    }

    fn reset_snapshot_recovery(&self, weight_1m: Option<u64>) {
        let mut recovery = self
            .snapshot_recovery
            .lock()
            .expect("aster snapshot recovery mutex poisoned");
        recovery.fail_streak = 0;
        recovery.next_allowed_ns = 0;
        recovery.last_http_status = Some(200);
        recovery.last_weight_1m = weight_1m;
        recovery.last_cooldown_ms = 0;
        recovery.last_failure_class = "";
        recovery.shared_consumer = "snapshot";
        recovery.shared_priority = "critical";
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        let book_event = matches!(
            &event,
            MarketDataEvent::L2Delta(_) | MarketDataEvent::L2Snapshot(_)
        );
        let result = self.market_publisher.publish_market(event).await;
        if result.is_ok() && book_event {
            self.freshness
                .last_published_ns
                .store(mono_now_ns(), Ordering::Relaxed);
        }
        result
    }

    pub async fn run_public_ws(&self) {
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
                        "{level}: Aster public WS error (consecutive_failures={consecutive_failures}): {err}"
                    );
                }
                Err(_timeout) => {
                    aster_audit_reconnect("session_timeout");
                    eprintln!(
                        "ERROR: Aster public WS session timeout ({}s) — force reconnect",
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
                        "INFO: Aster WS session was healthy for {:?}; \
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

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    pub async fn run_funding_polling(&self, poll_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(poll_ms.max(250)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut seq: u64 = 0;
        loop {
            interval.tick().await;
            match fetch_public_funding(&self.http, &self.cfg).await {
                Ok(mut update) => {
                    seq = seq.wrapping_add(1);
                    update.seq = seq;
                    if let Err(err) = self
                        .market_publisher
                        .publish_market(MarketDataEvent::FundingUpdate(update))
                        .await
                    {
                        eprintln!("Aster funding publish error: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Aster funding polling error: {err}");
                }
            }
        }
    }

    pub async fn run_mark_price_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let mut consecutive_failures = 0u32;
        let healthy_threshold = Duration::from_secs(60);
        loop {
            let session_start = Instant::now();
            match self.mark_price_ws_once().await {
                Err(err) => {
                    consecutive_failures = consecutive_failures.saturating_add(1);
                    aster_audit_funding_ws(
                        "session_error",
                        &[
                            ("consecutive_failures", consecutive_failures.to_string()),
                            ("err", err.to_string().replace(' ', "_")),
                        ],
                    );
                    let level = if consecutive_failures <= 3 {
                        "WARN"
                    } else {
                        "ERROR"
                    };
                    eprintln!(
                        "{level}: Aster funding WS error (consecutive_failures={consecutive_failures}): {err}"
                    );
                }
                Ok(()) => {}
            }

            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                if consecutive_failures > 0 {
                    eprintln!(
                        "INFO: Aster funding WS session was healthy for {:?}; resetting backoff and failure counter (was {})",
                        session_duration, consecutive_failures
                    );
                }
                consecutive_failures = 0;
                backoff = Duration::from_secs(1);
            }

            let max_backoff = match consecutive_failures {
                0..=10 => Duration::from_secs(30),
                11..=20 => Duration::from_secs(60),
                _ => Duration::from_secs(120),
            };
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    async fn mark_price_ws_once(&self) -> anyhow::Result<()> {
        let stream = format!("{}@markPrice@1s", self.cfg.stream_symbol());
        let ws_url = format!("{}/{}", self.cfg.ws_url.trim_end_matches('/'), stream);
        eprintln!("INFO: Aster funding WS connecting url={}", ws_url);
        let (ws_stream, _) =
            tokio::time::timeout(Duration::from_secs(15), connect_async(ws_url.as_str()))
                .await
                .map_err(|_| anyhow::anyhow!("Aster funding WS connect timeout (15s)"))?
                .map_err(|e| anyhow::anyhow!("Aster funding WS connect error: {e}"))?;
        eprintln!("INFO: Aster funding WS connected url={}", ws_url);
        aster_audit_funding_ws("connected", &[("url", ws_url.clone())]);
        let (mut write, mut read) = ws_stream.split();
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_ASTER_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ping_timer.tick().await;
        let mut seq: u64 = 0;
        let mut parse_error_count: u64 = 0;
        let mut publish_error_count: u64 = 0;

        loop {
            tokio::select! {
                _ = ping_timer.tick() => {
                    if let Err(err) = write.send(Message::Ping(vec![].into())).await {
                        anyhow::bail!("Aster funding WS ping send failed: {err}");
                    }
                }
                read_result = tokio::time::timeout(Duration::from_secs(30), read.next()) => {
                    let maybe = match read_result {
                        Ok(msg) => msg,
                        Err(_) => {
                            anyhow::bail!("Aster funding WS read timeout after 30s");
                        }
                    };
                    let Some(msg) = maybe else {
                        return Ok(());
                    };
                    match msg? {
                        Message::Text(text) => {
                            match parse_mark_price_update(&text, &self.cfg) {
                                Ok(Some(mut update)) => {
                                    seq = seq.wrapping_add(1);
                                    update.seq = seq;
                                    if let Err(err) = self.publish_market(MarketDataEvent::FundingUpdate(update)).await {
                                        publish_error_count = publish_error_count.saturating_add(1);
                                        aster_audit_funding_ws(
                                            "publish_error",
                                            &[
                                                ("count", publish_error_count.to_string()),
                                                ("err", err.to_string().replace(' ', "_")),
                                            ],
                                        );
                                        eprintln!("Aster funding WS publish error: {err}");
                                    }
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    parse_error_count = parse_error_count.saturating_add(1);
                                    aster_audit_funding_ws(
                                        "parse_error",
                                        &[
                                            ("count", parse_error_count.to_string()),
                                            ("err", err.to_string().replace(' ', "_")),
                                        ],
                                    );
                                    if parse_error_count <= 3 || parse_error_count % 25 == 0 {
                                        eprintln!("WARN: Aster funding WS parse error: {err}");
                                    }
                                }
                            }
                        }
                        Message::Binary(bytes) => {
                            let text = String::from_utf8(bytes)
                                .map_err(|_| anyhow::anyhow!("Aster funding WS non-utf8 binary frame"))?;
                            match parse_mark_price_update(&text, &self.cfg) {
                                Ok(Some(mut update)) => {
                                    seq = seq.wrapping_add(1);
                                    update.seq = seq;
                                    if let Err(err) = self.publish_market(MarketDataEvent::FundingUpdate(update)).await {
                                        publish_error_count = publish_error_count.saturating_add(1);
                                        aster_audit_funding_ws(
                                            "publish_error",
                                            &[
                                                ("count", publish_error_count.to_string()),
                                                ("err", err.to_string().replace(' ', "_")),
                                            ],
                                        );
                                        eprintln!("Aster funding WS publish error: {err}");
                                    }
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    parse_error_count = parse_error_count.saturating_add(1);
                                    aster_audit_funding_ws(
                                        "parse_error",
                                        &[
                                            ("count", parse_error_count.to_string()),
                                            ("err", err.to_string().replace(' ', "_")),
                                        ],
                                    );
                                    if parse_error_count <= 3 || parse_error_count % 25 == 0 {
                                        eprintln!("WARN: Aster funding WS parse error: {err}");
                                    }
                                }
                            }
                        }
                        Message::Ping(payload) => {
                            write.send(Message::Pong(payload)).await?;
                        }
                        Message::Close(_) => {
                            eprintln!("Aster funding WS closed; reconnecting url={}", self.cfg.ws_url);
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        // Use URL-path style connection (stream embedded in URL) instead of
        // JSON SUBSCRIBE.  This is more reliable for Binance-style futures APIs
        // and avoids issues with subscribe ACK frames and silent subscription
        // failures.
        let stream = format!("{}@depth@100ms", self.cfg.stream_symbol());
        let ws_url = format!("{}/{}", self.cfg.ws_url.trim_end_matches('/'), stream);
        eprintln!("INFO: Aster public WS connecting url={}", ws_url);
        let (ws_stream, _) =
            tokio::time::timeout(Duration::from_secs(15), connect_async(ws_url.as_str()))
                .await
                .map_err(|_| anyhow::anyhow!("Aster public WS connect timeout (15s)"))?
                .map_err(|e| anyhow::anyhow!("Aster public WS connect error: {e}"))?;
        eprintln!("INFO: Aster public WS connected url={}", ws_url);
        let (mut write, mut read) = ws_stream.split();

        const MAX_BUFFERED_UPDATES: usize = 1024;
        let mut buffered_updates: Vec<AsterDepthUpdate> = Vec::new();
        let mut last_update_id: Option<u64> = None;
        // Tracks the snapshot's lastUpdateId after applying a REST snapshot,
        // while waiting for a WS delta that bridges it.  Set when the snapshot
        // has been applied but no buffered delta bridged yet.
        let mut snapshot_last_id: Option<u64> = None;
        let mut snapshot_future: Option<SnapshotFuture<'_>> = Some(Box::pin(self.fetch_snapshot()));
        let mut last_gap_log = Instant::now() - Duration::from_secs(60);
        let mut last_snapshot_err_log = Instant::now() - Duration::from_secs(60);
        let mut first_decoded_top_logged = false;
        let mut first_size_raw_logged = false;
        let mut decode_miss_count = 0usize;
        let mut logged_non_utf8_binary = false;
        let mut emit_seq: u64 = 0;
        let mut next_seq = || {
            emit_seq = emit_seq.wrapping_add(1);
            emit_seq
        };
        let mut last_applied_at = Instant::now();
        let mut last_watchdog_trigger_at: Option<Instant> = None;
        let mut watchdog = tokio::time::interval(Duration::from_millis(250));
        watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        const STALE_MS: u64 = 2_000;
        const COOLDOWN_MS: u64 = 7_000;
        // WS-level ping timer to prevent idle connection drops.
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_ASTER_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ping_timer.tick().await; // skip first immediate tick
        let connect_start_ns = mono_now_ns();
        self.freshness.reset_for_new_connection();
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        let fixture_mode = env_is_true("ASTER_FIXTURE_MODE")
            || std::env::var_os("ASTER_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some();
        // FIX: Use configurable stale threshold (normalized to 10,000ms default)
        let stale_ms = aster_stale_ms();
        let mut _stale_tx_guard = None;
        if fixture_mode {
            _stale_tx_guard = Some(stale_tx);
        } else {
            let freshness = Arc::clone(&self.freshness);
            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(ASTER_WATCHDOG_TICK_MS));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    interval.tick().await;
                    let now = mono_now_ns();
                    let anchor = freshness.anchor_with_connect_start(connect_start_ns);
                    if anchor != 0 && age_ms(now, anchor) > stale_ms {
                        let _ = stale_tx.send(());
                        break;
                    }
                }
            });
        }

        loop {
            if last_update_id.is_none() {
                let future = snapshot_future.get_or_insert_with(|| Box::pin(self.fetch_snapshot()));
                tokio::select! {
                    biased;
                    _ = &mut stale_rx => {
                        aster_audit_reconnect("stale_watchdog");
                        anyhow::bail!("Aster public WS stale: freshness exceeded {}ms", stale_ms);
                    }
                    snapshot = future => {
                        match snapshot {
                            Ok((snapshot_raw, snapshot)) => {
                        snapshot_future = None;
                        if let Some(recorder) = self.recorder.as_ref() {
                            let mut guard = recorder.lock().await;
                            guard.record_snapshot(&snapshot_raw)?;
                        }
                        if let Ok(value) = serde_json::from_str::<Value>(&snapshot_raw) {
                            if let Some((top, bid_raw, ask_raw)) =
                                decode_top_of_book_with_raw(&value)
                            {
                                if !first_decoded_top_logged {
                                    eprintln!(
                                        "FIRST_DECODED_TOP venue=aster bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                        top.best_bid_px,
                                        top.best_bid_sz,
                                        top.best_ask_px,
                                        top.best_ask_sz
                                    );
                                    first_decoded_top_logged = true;
                                }
                                if !first_size_raw_logged {
                                    eprintln!(
                                        "ASTER_SIZE_RAW bid_sz_raw={} ask_sz_raw={} parsed_bid_sz={} parsed_ask_sz={}",
                                        bid_raw,
                                        ask_raw,
                                        top.best_bid_sz,
                                        top.best_ask_sz
                                    );
                                    first_size_raw_logged = true;
                                }
                            } else if decode_miss_count < 3 {
                                decode_miss_count += 1;
                                log_decode_miss(
                                    "Aster",
                                    &value,
                                    &snapshot_raw,
                                    decode_miss_count,
                                    self.cfg.ws_url.as_str(),
                                );
                            }
                        }
                        let snapshot_event =
                            MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
                                venue_index: self.cfg.venue_index,
                                venue_id: self.cfg.venue_id.clone(),
                                seq: next_seq(),
                                timestamp_ms: now_ms(),
                                bids: snapshot.bids,
                                asks: snapshot.asks,
                            });
                        {
                            let now_ns = mono_now_ns();
                            self.freshness
                                .last_parsed_ns
                                .store(now_ns, Ordering::Relaxed);
                            self.freshness
                                .last_book_event_ns
                                .store(now_ns, Ordering::Relaxed);
                        }
                        if self.publish_market(snapshot_event).await.is_ok() {
                            last_applied_at = Instant::now();
                        }

                        let snap_id = snapshot.last_update_id;
                        let mut next_last = snap_id;
                        let mut gap = false;
                        let mut any_applied = false;
                        let mut applied_count = 0usize;
                        let buffered_before = buffered_updates.len();
                        for update in buffered_updates.drain(..) {
                            // Stream is per-symbol; avoid dropping on formatting mismatch.
                            match seq_decision_lenient(next_last, &update) {
                                SeqDecision::Apply => {
                                    next_last = update.end_id;
                                    any_applied = true;
                                    applied_count = applied_count.saturating_add(1);
                                    {
                                        let now_ns = mono_now_ns();
                                        self.freshness
                                            .last_parsed_ns
                                            .store(now_ns, Ordering::Relaxed);
                                        self.freshness
                                            .last_book_event_ns
                                            .store(now_ns, Ordering::Relaxed);
                                    }
                                    if self
                                        .publish_market(delta_event_from_update(
                                            &update,
                                            self.cfg.venue_index,
                                            &self.cfg.venue_id,
                                            next_seq(),
                                        ))
                                        .await
                                        .is_ok()
                                    {
                                        last_applied_at = Instant::now();
                                    }
                                }
                                SeqDecision::Stale => {}
                                SeqDecision::Gap => {
                                    aster_audit_reconnect("seq_gap");
                                    aster_audit_book_recovery(
                                        "seq_gap",
                                        "buffered_drain",
                                        &[
                                            ("snap_id", snap_id.to_string()),
                                            ("next_last", next_last.to_string()),
                                            ("update_start", update.start_id.to_string()),
                                            ("update_end", update.end_id.to_string()),
                                            (
                                                "update_prev",
                                                update
                                                    .prev_id
                                                    .map(|value| value.to_string())
                                                    .unwrap_or_else(|| "none".to_string()),
                                            ),
                                            ("buffered_before", buffered_before.to_string()),
                                        ],
                                    );
                                    if last_gap_log.elapsed() > Duration::from_secs(10) {
                                        eprintln!(
                                            "WARN: Aster loop1 seq gap in buffered drain; snap_id={} next_last={} update_start={} update_end={} update_prev={:?}",
                                            snap_id, next_last, update.start_id, update.end_id, update.prev_id
                                        );
                                        last_gap_log = Instant::now();
                                    }
                                    gap = true;
                                    break;
                                }
                            }
                        }
                        if gap {
                            buffered_updates.clear();
                            last_update_id = None;
                            snapshot_last_id = None;
                            snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                            continue;
                        }
                        if any_applied {
                            // At least one buffered delta bridged the snapshot —
                            // transition to steady-state delta mode (loop 2).
                            last_update_id = Some(next_last);
                            snapshot_last_id = None;
                            aster_audit_book_recovery(
                                "snapshot_applied",
                                "buffered_drain",
                                &[
                                    ("snap_id", snap_id.to_string()),
                                    ("buffered_before", buffered_before.to_string()),
                                    ("applied_count", applied_count.to_string()),
                                    ("next_last", next_last.to_string()),
                                ],
                            );
                            self.mark_book_live();
                        } else {
                            // All buffered deltas were stale (none bridged).
                            // Stay in loop 1 but remember the snapshot ID so
                            // incoming WS frames can be checked for a bridge.
                            snapshot_last_id = Some(snap_id);
                            aster_audit_book_recovery(
                                "snapshot_wait_bridge",
                                "loop1",
                                &[
                                    ("snap_id", snap_id.to_string()),
                                    ("buffered_before", buffered_before.to_string()),
                                ],
                            );
                            eprintln!(
                                "INFO: Aster snapshot applied (snap_id={}), waiting for bridge delta on WS",
                                snap_id
                            );
                        }
                            }
                            Err(err) => {
                                let recovery = self.snapshot_recovery_state();
                                aster_audit_book_recovery(
                                    "snapshot_fetch_failed",
                                    "loop1",
                                    &[
                                        ("buffered_before", buffered_updates.len().to_string()),
                                        ("err_len", err.to_string().len().to_string()),
                                        ("fail_streak", recovery.fail_streak.to_string()),
                                        ("cooldown_ms", recovery.last_cooldown_ms.to_string()),
                                        (
                                            "http_status",
                                            recovery
                                                .last_http_status
                                                .map(|value| value.to_string())
                                                .unwrap_or_else(|| "0".to_string()),
                                        ),
                                        (
                                            "weight_1m",
                                            recovery
                                                .last_weight_1m
                                                .map(|value| value.to_string())
                                                .unwrap_or_else(|| "0".to_string()),
                                        ),
                                        (
                                            "failure_class",
                                            recovery.last_failure_class.to_string(),
                                        ),
                                    ],
                                );
                                if last_snapshot_err_log.elapsed() > Duration::from_secs(30) {
                                    let url = format!(
                                        "{}/fapi/v1/depth?symbol={}&limit={}",
                                        self.cfg.rest_url, self.cfg.market, self.cfg.depth_limit
                                    );
                                    eprintln!(
                                        "WARN: Aster snapshot fetch failed; url={} failure_class={} http_status={:?} fail_streak={} cooldown_ms={} weight_1m={:?} err={}",
                                        url,
                                        recovery.last_failure_class,
                                        recovery.last_http_status,
                                        recovery.fail_streak,
                                        recovery.last_cooldown_ms,
                                        recovery.last_weight_1m,
                                        err
                                    );
                                    last_snapshot_err_log = Instant::now();
                                }
                                snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                                buffered_updates.clear();
                                last_update_id = None;
                                snapshot_last_id = None;
                                continue;
                            }
                        }
                    }
                    _ = watchdog.tick() => {
                        let now = Instant::now();
                        let stale = now.duration_since(last_applied_at);
                        let cooldown_ok = last_watchdog_trigger_at
                            .map(|last| now.duration_since(last) >= Duration::from_millis(COOLDOWN_MS))
                            .unwrap_or(true);
                        if stale > Duration::from_millis(STALE_MS)
                            && (cooldown_ok || stale >= Duration::from_millis(15_000))
                        {
                            aster_audit_reconnect("stale_watchdog");
                            let ages = aster_freshness_ages(&self.freshness, connect_start_ns);
                            aster_audit_book_recovery(
                                "stale_watchdog",
                                "loop1",
                                &[
                                    ("stale_ms", stale.as_millis().to_string()),
                                    ("buffered_before", buffered_updates.len().to_string()),
                                    ("cooldown_ok", if cooldown_ok { "1" } else { "0" }.to_string()),
                                    ("anchor_age_ms", ages.anchor_age_ms.to_string()),
                                    ("ws_rx_age_ms", ages.ws_rx_age_ms.to_string()),
                                    ("data_rx_age_ms", ages.data_rx_age_ms.to_string()),
                                    ("parsed_age_ms", ages.parsed_age_ms.to_string()),
                                    ("pub_age_ms", ages.pub_age_ms.to_string()),
                                    ("book_age_ms", ages.book_age_ms.to_string()),
                                ],
                            );
                            eprintln!(
                                "WARN: Aster WS stale; resyncing url={} stale_ms={}",
                                self.cfg.ws_url,
                                stale.as_millis()
                            );
                            buffered_updates.clear();
                            last_update_id = None;
                            snapshot_last_id = None;
                            snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                            last_watchdog_trigger_at = Some(now);
                            continue;
                        }
                    }
                    msg = read.next() => {
                        let Some(msg) = msg else {
                            return Ok(());
                        };
                        let msg = msg?;
                        self.freshness
                            .last_ws_rx_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        // Helper: extract text payload from Text or Binary frames.
                        let text_payload = match &msg {
                            Message::Text(text) => Some(text.clone()),
                            Message::Binary(bytes) => {
                                match String::from_utf8(bytes.clone()) {
                                    Ok(text) => Some(text),
                                    Err(_) => {
                                        if !logged_non_utf8_binary {
                                            eprintln!(
                                                "WARN: Aster public WS non-utf8 binary frame url={}",
                                                self.cfg.ws_url
                                            );
                                            logged_non_utf8_binary = true;
                                        }
                                        None
                                    }
                                }
                            }
                            _ => None,
                        };
                        match msg {
                            Message::Ping(payload) => {
                                write.send(Message::Pong(payload)).await?;
                            }
                            Message::Close(_) => {
                                eprintln!("Aster WS closed; reconnecting url={}", self.cfg.ws_url);
                                return Ok(());
                            }
                            _ => {}
                        }
                        if let Some(text) = text_payload {
                            if let Some(recorder) = self.recorder.as_ref() {
                                let mut guard = recorder.lock().await;
                                let _ = guard.record_ws_frame(&text);
                            }
                            self.freshness
                                .last_data_rx_ns
                                .store(mono_now_ns(), Ordering::Relaxed);
                            if let Some(update) = parse_depth_update(&text) {
                                self.freshness
                                    .last_parsed_ns
                                    .store(mono_now_ns(), Ordering::Relaxed);

                                // If we have a snapshot applied but are waiting
                                // for a bridge delta, check incoming frames
                                // directly instead of just buffering.
                                if let Some(sid) = snapshot_last_id {
                                    match seq_decision_lenient(sid, &update) {
                                        SeqDecision::Apply => {
                                            // Bridge found! Transition to loop 2.
                                            let now_ns = mono_now_ns();
                                            self.freshness
                                                .last_parsed_ns
                                                .store(now_ns, Ordering::Relaxed);
                                            self.freshness
                                                .last_book_event_ns
                                                .store(now_ns, Ordering::Relaxed);
                                            if self
                                                .publish_market(delta_event_from_update(
                                                    &update,
                                                    self.cfg.venue_index,
                                                    &self.cfg.venue_id,
                                                    next_seq(),
                                                ))
                                                .await
                                                .is_ok()
                                        {
                                            last_applied_at = Instant::now();
                                        }
                                        last_update_id = Some(update.end_id);
                                        snapshot_last_id = None;
                                        aster_audit_book_recovery(
                                            "bridge_found",
                                            "bridge_wait",
                                            &[
                                                ("snap_id", sid.to_string()),
                                                ("update_start", update.start_id.to_string()),
                                                ("update_end", update.end_id.to_string()),
                                                (
                                                    "update_prev",
                                                    update
                                                        .prev_id
                                                        .map(|value| value.to_string())
                                                        .unwrap_or_else(|| "none".to_string()),
                                                ),
                                            ],
                                        );
                                        eprintln!(
                                            "INFO: Aster bridge delta found after snapshot; snap_id={} delta_end={}",
                                            sid, update.end_id
                                            );
                                        self.mark_book_live();
                                        }
                                        SeqDecision::Stale => {
                                            // Still behind the snapshot, keep waiting.
                                        }
                                        SeqDecision::Gap => {
                                            aster_audit_reconnect("seq_gap");
                                            aster_audit_book_recovery(
                                                "seq_gap",
                                                "bridge_wait",
                                                &[
                                                    ("snap_id", sid.to_string()),
                                                    ("update_start", update.start_id.to_string()),
                                                    ("update_end", update.end_id.to_string()),
                                                    (
                                                        "update_prev",
                                                        update
                                                            .prev_id
                                                            .map(|value| value.to_string())
                                                            .unwrap_or_else(|| "none".to_string()),
                                                    ),
                                                ],
                                            );
                                            // WS jumped past the snapshot — re-fetch.
                                            if last_gap_log.elapsed() > Duration::from_secs(10) {
                                                eprintln!(
                                                    "WARN: Aster loop1 bridge-wait gap; snap_id={} update_start={} update_end={} — re-fetching snapshot",
                                                    sid, update.start_id, update.end_id
                                                );
                                                last_gap_log = Instant::now();
                                            }
                                            snapshot_last_id = None;
                                            buffered_updates.clear();
                                            snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                                        }
                                    }
                                } else {
                                    // No snapshot yet — just buffer.
                                    buffered_updates.push(update);
                                    if buffered_updates.len() > MAX_BUFFERED_UPDATES {
                                        aster_audit_book_recovery(
                                            "buffer_overflow",
                                            "pre_snapshot",
                                            &[("buffered_before", buffered_updates.len().to_string())],
                                        );
                                        eprintln!(
                                            "Aster WS buffer overflow; resyncing url={}",
                                            self.cfg.ws_url
                                        );
                                        buffered_updates.clear();
                                        snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                                    }
                                }
                            }
                        }
                    }
                }
                continue;
            }

            tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    aster_audit_reconnect("stale_watchdog");
                    let ages = aster_freshness_ages(&self.freshness, connect_start_ns);
                    aster_audit_book_recovery(
                        "stale_watchdog",
                        "global_watchdog",
                        &[
                            ("anchor_age_ms", ages.anchor_age_ms.to_string()),
                            ("ws_rx_age_ms", ages.ws_rx_age_ms.to_string()),
                            ("data_rx_age_ms", ages.data_rx_age_ms.to_string()),
                            ("parsed_age_ms", ages.parsed_age_ms.to_string()),
                            ("pub_age_ms", ages.pub_age_ms.to_string()),
                            ("book_age_ms", ages.book_age_ms.to_string()),
                        ],
                    );
                    anyhow::bail!("Aster public WS stale: freshness exceeded {}ms", stale_ms);
                }
                _ = ping_timer.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        aster_audit_reconnect("ping_send_fail");
                        eprintln!("WARN: Aster public WS ping send failed: {e} — reconnecting");
                        anyhow::bail!("Aster public WS ping send failed: {e}");
                    }
                    continue;
                }
                read_result = tokio::time::timeout(Duration::from_secs(30), read.next()) => {
                    let maybe = match read_result {
                        Ok(m) => m,
                        Err(_) => {
                            aster_audit_reconnect("read_timeout");
                            let ages = aster_freshness_ages(&self.freshness, connect_start_ns);
                            aster_audit_book_recovery(
                                "read_timeout",
                                "steady_state",
                                &[
                                    ("ws_rx_age_ms", ages.ws_rx_age_ms.to_string()),
                                    ("data_rx_age_ms", ages.data_rx_age_ms.to_string()),
                                    ("parsed_age_ms", ages.parsed_age_ms.to_string()),
                                    ("pub_age_ms", ages.pub_age_ms.to_string()),
                                    ("book_age_ms", ages.book_age_ms.to_string()),
                                ],
                            );
                            eprintln!(
                                "WARN: Aster public WS read timeout (30s) — no frame received, reconnecting"
                            );
                            anyhow::bail!("Aster public WS read timeout after 30s");
                        }
                    };
                    let Some(msg) = maybe else {
                        return Ok(());
                    };
                    let msg = msg?;
                    self.freshness
                        .last_ws_rx_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    match msg {
                        Message::Text(text) => {
                            if let Some(recorder) = self.recorder.as_ref() {
                                let mut guard = recorder.lock().await;
                                let _ = guard.record_ws_frame(&text);
                            }
                            self.freshness
                                .last_data_rx_ns
                                .store(mono_now_ns(), Ordering::Relaxed);
                            let Some(update) = parse_depth_update(&text) else {
                                continue;
                            };
                            self.freshness
                                .last_parsed_ns
                                .store(mono_now_ns(), Ordering::Relaxed);
                            if !symbol_matches(&update.symbol, &self.cfg.market) {
                                continue;
                            }
                            let current_last = last_update_id.unwrap_or_default();
                            let decision = seq_decision_lenient(current_last, &update);
                            match decision {
                                SeqDecision::Apply => {
                                    last_update_id = Some(update.end_id);
                                    {
                                        let now_ns = mono_now_ns();
                                        self.freshness
                                            .last_parsed_ns
                                            .store(now_ns, Ordering::Relaxed);
                                        self.freshness
                                            .last_book_event_ns
                                            .store(now_ns, Ordering::Relaxed);
                                    }
                                    if self
                                        .publish_market(delta_event_from_update(
                                            &update,
                                            self.cfg.venue_index,
                                            &self.cfg.venue_id,
                                            next_seq(),
                                        ))
                                        .await
                                        .is_ok()
                                    {
                                        last_applied_at = Instant::now();
                                    }
                                }
                                SeqDecision::Stale => {}
                                SeqDecision::Gap => {
                                    aster_audit_reconnect("seq_gap");
                                    aster_audit_book_recovery(
                                        "seq_gap",
                                        "steady_state",
                                        &[
                                            ("current_last", current_last.to_string()),
                                            ("update_start", update.start_id.to_string()),
                                            ("update_end", update.end_id.to_string()),
                                            (
                                                "update_prev",
                                                update
                                                    .prev_id
                                                    .map(|value| value.to_string())
                                                    .unwrap_or_else(|| "none".to_string()),
                                            ),
                                        ],
                                    );
                                    if last_gap_log.elapsed() > Duration::from_secs(30) {
                                        eprintln!(
                                            "Aster WS seq gap; resyncing last={} prev={:?} start={} end={} url={}",
                                            current_last,
                                            update.prev_id,
                                            update.start_id,
                                            update.end_id,
                                            self.cfg.ws_url
                                        );
                                        last_gap_log = Instant::now();
                                    }
                                    buffered_updates.clear();
                                    last_update_id = None;
                                    snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                                }
                            }
                        }
                        Message::Binary(bytes) => match String::from_utf8(bytes) {
                            Ok(text) => {
                                if let Some(recorder) = self.recorder.as_ref() {
                                    let mut guard = recorder.lock().await;
                                    let _ = guard.record_ws_frame(&text);
                                }
                                self.freshness
                                    .last_data_rx_ns
                                    .store(mono_now_ns(), Ordering::Relaxed);
                                let Some(update) = parse_depth_update(&text) else {
                                    continue;
                                };
                                self.freshness
                                    .last_parsed_ns
                                    .store(mono_now_ns(), Ordering::Relaxed);
                                if !symbol_matches(&update.symbol, &self.cfg.market) {
                                    continue;
                                }
                                let current_last = last_update_id.unwrap_or_default();
                                match seq_decision_lenient(current_last, &update) {
                                    SeqDecision::Apply => {
                                        last_update_id = Some(update.end_id);
                                        {
                                            let now_ns = mono_now_ns();
                                            self.freshness
                                                .last_parsed_ns
                                                .store(now_ns, Ordering::Relaxed);
                                            self.freshness
                                                .last_book_event_ns
                                                .store(now_ns, Ordering::Relaxed);
                                        }
                                        if self
                                            .publish_market(delta_event_from_update(
                                                &update,
                                                self.cfg.venue_index,
                                                &self.cfg.venue_id,
                                                next_seq(),
                                            ))
                                            .await
                                            .is_ok()
                                        {
                                            last_applied_at = Instant::now();
                                        }
                                    }
                                    SeqDecision::Stale => {}
                                    SeqDecision::Gap => {
                                        aster_audit_reconnect("seq_gap");
                                        aster_audit_book_recovery(
                                            "seq_gap",
                                            "steady_state",
                                            &[
                                                ("current_last", current_last.to_string()),
                                                ("update_start", update.start_id.to_string()),
                                                ("update_end", update.end_id.to_string()),
                                                (
                                                    "update_prev",
                                                    update
                                                        .prev_id
                                                        .map(|value| value.to_string())
                                                        .unwrap_or_else(|| "none".to_string()),
                                                ),
                                            ],
                                        );
                                        if last_gap_log.elapsed() > Duration::from_secs(30) {
                                            eprintln!(
                                                "Aster WS seq gap; resyncing last={} prev={:?} start={} end={} url={}",
                                                current_last,
                                                update.prev_id,
                                                update.start_id,
                                                update.end_id,
                                                self.cfg.ws_url
                                            );
                                            last_gap_log = Instant::now();
                                        }
                                        buffered_updates.clear();
                                        last_update_id = None;
                                        snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                                    }
                                }
                            }
                            Err(_) => {
                                if !logged_non_utf8_binary {
                                    eprintln!(
                                        "WARN: Aster public WS non-utf8 binary frame url={}",
                                        self.cfg.ws_url
                                    );
                                    logged_non_utf8_binary = true;
                                }
                            }
                        },
                        Message::Ping(payload) => {
                            write.send(Message::Pong(payload)).await?;
                        }
                        Message::Close(_) => {
                            eprintln!("Aster WS closed; reconnecting url={}", self.cfg.ws_url);
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                _ = watchdog.tick() => {
                    let now = Instant::now();
                    let stale = now.duration_since(last_applied_at);
                    let cooldown_ok = last_watchdog_trigger_at
                        .map(|last| now.duration_since(last) >= Duration::from_millis(COOLDOWN_MS))
                        .unwrap_or(true);
                    if stale > Duration::from_millis(STALE_MS)
                        && (cooldown_ok || stale >= Duration::from_millis(15_000))
                    {
                        aster_audit_reconnect("stale_watchdog");
                        eprintln!(
                            "WARN: Aster WS stale; resyncing url={} stale_ms={}",
                            self.cfg.ws_url,
                            stale.as_millis()
                        );
                        buffered_updates.clear();
                        last_update_id = None;
                        snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                        last_watchdog_trigger_at = Some(now);
                        continue;
                    }
                }
            }
        }
    }

    async fn fetch_snapshot(&self) -> anyhow::Result<(String, AsterDepthSnapshot)> {
        self.mark_snapshot_recovery_active();
        let url = format!(
            "{}/fapi/v1/depth?symbol={}&limit={}",
            self.cfg.rest_url, self.cfg.market, self.cfg.depth_limit
        );
        let recovery = self.snapshot_recovery_state();
        let now_ns = mono_now_ns();
        let next_allowed_ns = recovery
            .next_allowed_ns
            .max(recovery.shared_cooldown_until_ns);
        if next_allowed_ns > now_ns {
            let wait_ms = age_ms(next_allowed_ns, now_ns);
            aster_audit_book_recovery(
                "snapshot_backoff_wait",
                "fetch",
                &[
                    ("cooldown_ms", wait_ms.to_string()),
                    ("fail_streak", recovery.fail_streak.to_string()),
                    (
                        "http_status",
                        recovery
                            .last_http_status
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "0".to_string()),
                    ),
                    (
                        "weight_1m",
                        recovery
                            .last_weight_1m
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "0".to_string()),
                    ),
                    ("shared_consumer", recovery.shared_consumer.to_string()),
                    ("shared_priority", recovery.shared_priority.to_string()),
                ],
            );
            tokio::time::sleep(Duration::from_millis(wait_ms.max(1))).await;
        }

        let resp = match self
            .http
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                let recovery = self.record_snapshot_failure(None, None);
                return Err(anyhow::anyhow!(
                    "aster snapshot request failed class={} fail_streak={} cooldown_ms={} err={}",
                    recovery.last_failure_class,
                    recovery.fail_streak,
                    recovery.last_cooldown_ms,
                    err
                ));
            }
        };
        let status = resp.status();
        let headers = resp.headers().clone();
        let weight_1m = snapshot_weight_1m(&headers);
        if !status.is_success() {
            let recovery = self.record_snapshot_failure(Some(status.as_u16()), weight_1m);
            let body = resp.text().await.unwrap_or_default();
            let body_snippet: String = body.chars().take(160).collect();
            return Err(anyhow::anyhow!(
                "aster snapshot http_status={} class={} fail_streak={} cooldown_ms={} weight_1m={} body={}",
                status.as_u16(),
                recovery.last_failure_class,
                recovery.fail_streak,
                recovery.last_cooldown_ms,
                recovery
                    .last_weight_1m
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "0".to_string()),
                body_snippet
            ));
        }

        let raw = match resp.text().await {
            Ok(raw) => raw,
            Err(err) => {
                let recovery = self.record_snapshot_failure(None, weight_1m);
                return Err(anyhow::anyhow!(
                    "aster snapshot body read failed class={} fail_streak={} cooldown_ms={} err={}",
                    recovery.last_failure_class,
                    recovery.fail_streak,
                    recovery.last_cooldown_ms,
                    err
                ));
            }
        };
        let value: Value = match serde_json::from_str(&raw) {
            Ok(value) => value,
            Err(err) => {
                let recovery = self.record_snapshot_failure(None, weight_1m);
                return Err(anyhow::anyhow!(
                    "aster snapshot json parse failed class={} fail_streak={} cooldown_ms={} err={}",
                    recovery.last_failure_class,
                    recovery.fail_streak,
                    recovery.last_cooldown_ms,
                    err
                ));
            }
        };
        let snapshot = match parse_depth_snapshot(&value) {
            Some(snapshot) => snapshot,
            None => {
                let recovery = self.record_snapshot_failure(None, weight_1m);
                return Err(anyhow::anyhow!(
                    "aster snapshot parse failed class={} fail_streak={} cooldown_ms={}",
                    recovery.last_failure_class,
                    recovery.fail_streak,
                    recovery.last_cooldown_ms
                ));
            }
        };
        self.reset_snapshot_recovery(weight_1m);
        Ok((raw, snapshot))
    }
}

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
pub struct AsterRestClient {
    cfg: AsterConfig,
    http: Client,
    timestamp_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync>,
    price_tick_size: Arc<Mutex<Option<f64>>>,
    poll_seq: Arc<AtomicU64>,
}

impl AsterRestClient {
    pub fn new(cfg: AsterConfig) -> Self {
        Self {
            cfg,
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("aster rest http client build"),
            timestamp_fn: Arc::new(now_ms),
            price_tick_size: Arc::new(Mutex::new(None)),
            poll_seq: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn with_timestamp_fn(
        mut self,
        timestamp_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync>,
    ) -> Self {
        self.timestamp_fn = timestamp_fn;
        self
    }

    pub fn has_auth(&self) -> bool {
        self.cfg.has_auth()
    }

    async fn resolve_price_tick_size(&self) -> LiveResult<f64> {
        {
            let guard = self.price_tick_size.lock().await;
            if let Some(tick_size) = *guard {
                return Ok(tick_size);
            }
        }

        let url = format!(
            "{}/fapi/v1/exchangeInfo",
            self.cfg.rest_url.trim_end_matches('/')
        );
        let resp = self
            .http
            .get(&url)
            .query(&[("symbol", self.cfg.market.as_str())])
            .send()
            .await
            .map_err(|err| {
                LiveGatewayError::retryable(format!("aster exchangeInfo error: {err}"))
            })?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("aster exchangeInfo parse error: {err}"))
        })?;
        let tick_size =
            parse_exchange_info_price_tick_size(&value, &self.cfg.market).ok_or_else(|| {
                LiveGatewayError::fatal(format!(
                    "aster exchangeInfo missing price tick size for {}",
                    self.cfg.market
                ))
            })?;
        let mut guard = self.price_tick_size.lock().await;
        Ok(*guard.get_or_insert(tick_size))
    }

    fn signed_query(&self, mut params: Vec<(String, String)>) -> Result<String, LiveGatewayError> {
        let api_secret = self
            .cfg
            .api_secret
            .as_ref()
            .ok_or_else(|| LiveGatewayError::fatal("aster api secret missing"))?;
        let timestamp = (self.timestamp_fn)();
        params.push(("timestamp".to_string(), timestamp.to_string()));
        if let Some(recv_window) = self.cfg.recv_window {
            params.push(("recvWindow".to_string(), recv_window.to_string()));
        }
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical = canonical_query(&params);
        // Signing per Aster REST API docs: https://asterdex.org/docs (HMAC SHA256 + X-MBX-APIKEY + timestamp/signature).
        let signature = sign_query(api_secret, &canonical);
        Ok(format!("{canonical}&signature={signature}"))
    }

    async fn send_signed_request(
        &self,
        method: Method,
        path: &str,
        params: Vec<(String, String)>,
    ) -> LiveResult<reqwest::Response> {
        let api_key = self
            .cfg
            .api_key
            .as_ref()
            .ok_or_else(|| LiveGatewayError::fatal("aster api key missing"))?;
        let query = self.signed_query(params)?;
        let url = format!("{}{}?{}", self.cfg.rest_url, path, query);
        let resp = self
            .http
            .request(method, url)
            .header("X-MBX-APIKEY", api_key)
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("rest_error: {err}")))?;
        Ok(resp)
    }

    async fn fetch_account_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<AccountSnapshot> {
        let resp = self
            .send_signed_request(Method::GET, "/fapi/v2/account", Vec::new())
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body)
            .map_err(|err| LiveGatewayError::fatal(format!("aster account parse error: {err}")))?;
        let mut snapshot =
            parse_account_snapshot(&value, venue_id, venue_index).ok_or_else(|| {
                LiveGatewayError::fatal("aster account snapshot missing required fields")
            })?;
        // Treat freshness as the time we successfully polled the venue. Aster's
        // `updateTime` reflects the last account mutation and can remain unchanged
        // for long periods, which would make a fresh poll look stale downstream.
        snapshot.timestamp_ms = (self.timestamp_fn)();
        Ok(snapshot)
    }

    async fn fetch_open_order_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<OrderSnapshot> {
        let params = vec![("symbol".to_string(), self.cfg.market.clone())];
        let resp = self
            .send_signed_request(Method::GET, "/fapi/v1/openOrders", params)
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("aster open orders parse error: {err}"))
        })?;
        Ok(OrderSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: self.poll_seq.fetch_add(1, Ordering::Relaxed),
            timestamp_ms: (self.timestamp_fn)(),
            open_orders: parse_open_orders(&value, &self.cfg.market),
        })
    }

    pub async fn run_account_polling(
        self: Arc<Self>,
        account_tx: mpsc::Sender<AccountEvent>,
        venue_id: String,
        venue_index: usize,
        poll_ms: u64,
    ) {
        let mut interval = tokio::time::interval(Duration::from_millis(poll_ms.max(250)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            match self.fetch_account_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;
                }
                Err(err) => {
                    eprintln!("Aster account snapshot error: {}", err.message);
                }
            }
        }
    }

    pub async fn run_order_polling(
        self: Arc<Self>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
        venue_id: String,
        venue_index: usize,
        poll_ms: u64,
    ) {
        let mut interval = tokio::time::interval(Duration::from_millis(poll_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            match self.fetch_open_order_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    let _ = exec_tx.send(ExecutionEvent::OrderSnapshot(snapshot)).await;
                }
                Err(err) => {
                    eprintln!("Aster open order snapshot error: {}", err.message);
                }
            }
        }
    }

    // Note: funding polling lives on AsterConnector (market publisher).
}

impl LiveRestClient for AsterRestClient {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let price_tick_size = self.resolve_price_tick_size().await?;
            let price = snap_price_to_tick(req.price, price_tick_size, req.side, req.post_only);
            let mut params = vec![
                ("symbol".to_string(), self.cfg.market.clone()),
                ("side".to_string(), map_side(req.side).to_string()),
                ("type".to_string(), "LIMIT".to_string()),
                (
                    "timeInForce".to_string(),
                    map_time_in_force(req.time_in_force, req.post_only).to_string(),
                ),
                ("price".to_string(), format_f64(price)),
                ("quantity".to_string(), format_f64(req.size)),
                ("newClientOrderId".to_string(), req.client_order_id.clone()),
            ];
            if req.reduce_only {
                params.push(("reduceOnly".to_string(), "true".to_string()));
            }
            let resp = self
                .send_signed_request(Method::POST, "/fapi/v1/order", params)
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(map_rest_error(status.as_u16(), &body));
            }
            let order_id = parse_order_id(&body).or(Some(req.client_order_id));
            Ok(LiveRestResponse { order_id })
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let mut params = vec![("symbol".to_string(), self.cfg.market.clone())];
            if is_numeric_order_id(&req.order_id) {
                params.push(("orderId".to_string(), req.order_id.clone()));
            } else {
                params.push(("origClientOrderId".to_string(), req.order_id.clone()));
            }
            let resp = self
                .send_signed_request(Method::DELETE, "/fapi/v1/order", params)
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                if is_aster_unknown_order(&body) {
                    return Ok(LiveRestResponse {
                        order_id: Some(req.order_id),
                    });
                }
                return Err(map_rest_error(status.as_u16(), &body));
            }
            Ok(LiveRestResponse { order_id: None })
        })
    }

    fn cancel_all(
        &self,
        _req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let params = vec![("symbol".to_string(), self.cfg.market.clone())];
            let resp = self
                .send_signed_request(Method::DELETE, "/fapi/v1/allOpenOrders", params)
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(map_rest_error(status.as_u16(), &body));
            }
            Ok(LiveRestResponse { order_id: None })
        })
    }
}

fn canonical_query(params: &[(String, String)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", encode_component(k), encode_component(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn encode_component(raw: &str) -> String {
    raw.as_bytes()
        .iter()
        .map(|b| {
            let c = *b as char;
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
                c.to_string()
            } else {
                format!("%{:02X}", b)
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn sign_query(secret: &str, query: &str) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
    mac.update(query.as_bytes());
    let bytes = mac.finalize().into_bytes();
    hex::encode(bytes)
}

fn map_side(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

fn map_time_in_force(time_in_force: TimeInForce, post_only: bool) -> &'static str {
    if post_only {
        return "GTX";
    }
    match time_in_force {
        TimeInForce::Ioc => "IOC",
        TimeInForce::Gtc => "GTC",
    }
}

fn format_f64(value: f64) -> String {
    if !value.is_finite() {
        return "0".to_string();
    }
    let mut formatted = format!("{value:.12}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    if formatted == "-0" {
        formatted = "0".to_string();
    }
    formatted
}

fn snap_price_to_tick(price: f64, tick_size: f64, side: Side, post_only: bool) -> f64 {
    if !price.is_finite() || !tick_size.is_finite() || tick_size <= 0.0 {
        return price;
    }
    let ticks = price / tick_size;
    let epsilon = 1e-9;
    let snapped_ticks = if post_only {
        match side {
            Side::Buy => (ticks + epsilon).floor(),
            Side::Sell => (ticks - epsilon).ceil(),
        }
    } else {
        ticks.round()
    };
    snapped_ticks * tick_size
}

fn is_numeric_order_id(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
}

fn is_aster_unknown_order(body: &str) -> bool {
    body.contains("\"code\":-2011") || body.contains("Unknown order sent")
}

fn parse_exchange_info_price_tick_size(value: &Value, market: &str) -> Option<f64> {
    let symbols = value.get("symbols")?.as_array()?;
    let entry = symbols
        .iter()
        .find(|symbol| symbol.get("symbol").and_then(|v| v.as_str()) == Some(market))
        .or_else(|| symbols.first())?;
    let filters = entry.get("filters")?.as_array()?;
    let price_filter = filters
        .iter()
        .find(|filter| filter.get("filterType").and_then(|v| v.as_str()) == Some("PRICE_FILTER"))?;
    let tick_size = price_filter
        .get("tickSize")
        .or_else(|| price_filter.get("tick_size"))
        .and_then(parse_f64)
        .filter(|tick| *tick > 0.0);
    tick_size.or_else(|| {
        entry
            .get("pricePrecision")
            .and_then(|v| v.as_u64())
            .map(|precision| 10f64.powi(-(precision as i32)))
            .filter(|tick| *tick > 0.0)
    })
}

fn parse_order_id(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    if let Some(order_id) = value.get("orderId") {
        if let Some(raw) = order_id.as_i64() {
            return Some(raw.to_string());
        }
        if let Some(raw) = order_id.as_str() {
            return Some(raw.to_string());
        }
    }
    value
        .get("clientOrderId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn parse_open_orders(value: &Value, market: &str) -> Vec<OpenOrderSnapshot> {
    let list = value.as_array().cloned().unwrap_or_default();
    list.into_iter()
        .filter_map(|order| {
            let symbol = order.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
            if !symbol_matches(symbol, market) {
                return None;
            }
            let status = order.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if !(status.eq_ignore_ascii_case("NEW")
                || status.eq_ignore_ascii_case("PARTIALLY_FILLED")
                || status.eq_ignore_ascii_case("OPEN"))
            {
                return None;
            }
            let order_id = order.get("orderId").and_then(|v| {
                v.as_i64()
                    .map(|raw| raw.to_string())
                    .or_else(|| v.as_str().map(|raw| raw.to_string()))
            })?;
            let client_order_id = order
                .get("clientOrderId")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string());
            let side = match order.get("side").and_then(|v| v.as_str())? {
                side if side.eq_ignore_ascii_case("BUY") => Side::Buy,
                side if side.eq_ignore_ascii_case("SELL") => Side::Sell,
                _ => return None,
            };
            let price = order.get("price").and_then(parse_f64)?;
            let orig_qty = order
                .get("origQty")
                .or_else(|| order.get("orig_quantity"))
                .and_then(parse_f64)?;
            let executed_qty = order
                .get("executedQty")
                .or_else(|| order.get("executed_quantity"))
                .and_then(parse_f64)
                .unwrap_or(0.0);
            let size = (orig_qty - executed_qty).max(0.0);
            (size > 0.0).then_some(OpenOrderSnapshot {
                order_id,
                client_order_id,
                side,
                price,
                size,
                purpose: None,
            })
        })
        .collect()
}

fn parse_account_snapshot(
    value: &Value,
    venue_id: &str,
    venue_index: usize,
) -> Option<AccountSnapshot> {
    let seq = value
        .get("updateTime")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let timestamp_ms = value
        .get("updateTime")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let positions = value
        .get("positions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pos| {
                    let symbol = pos.get("symbol")?.as_str()?.to_string();
                    let size = parse_f64(pos.get("positionAmt")?)?;
                    let entry_price = parse_f64(pos.get("entryPrice")?)?;
                    Some(PositionSnapshot {
                        symbol,
                        size,
                        entry_price,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let balances = value
        .get("assets")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|bal| {
                    let asset = bal.get("asset")?.as_str()?.to_string();
                    let total = parse_f64(bal.get("walletBalance")?)?;
                    let available = parse_f64(bal.get("availableBalance")?)?;
                    Some(BalanceSnapshot {
                        asset,
                        total,
                        available,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let margin = MarginSnapshot {
        balance_usd: value
            .get("totalWalletBalance")
            .and_then(parse_f64)
            .unwrap_or(0.0),
        used_usd: value
            .get("totalPositionInitialMargin")
            .and_then(parse_f64)
            .unwrap_or(0.0),
        available_usd: value
            .get("availableBalance")
            .and_then(parse_f64)
            .unwrap_or(0.0),
    };
    let liquidation = LiquidationSnapshot {
        price_liq: None,
        dist_liq_sigma: None,
    };

    Some(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        positions,
        balances,
        funding_8h: None,
        margin,
        liquidation,
    })
}

async fn fetch_public_funding(client: &Client, cfg: &AsterConfig) -> anyhow::Result<FundingUpdate> {
    let path =
        std::env::var("ASTER_FUNDING_PATH").unwrap_or_else(|_| "/fapi/v1/premiumIndex".to_string());
    let url = format!("{}{}", cfg.rest_url.trim_end_matches('/'), path);
    let resp = client
        .get(url)
        .query(&[("symbol", cfg.market.clone())])
        .send()
        .await?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = resp.text().await?;
    if !status.is_success() {
        let weight_suffix = snapshot_weight_1m(&headers)
            .map(|weight| format!(" weight_1m={weight}"))
            .unwrap_or_default();
        let body_snippet = body.chars().take(160).collect::<String>();
        anyhow::bail!(
            "HTTP status {} for {}{} body={}",
            status.as_u16(),
            path,
            weight_suffix,
            body_snippet.replace('\n', " ")
        );
    }
    let value: Value = serde_json::from_str(&body).map_err(|err| {
        anyhow::anyhow!(
            "aster funding parse error: {err} body={}",
            body.chars()
                .take(160)
                .collect::<String>()
                .replace('\n', " ")
        )
    })?;
    parse_public_funding(&value, cfg)
        .ok_or_else(|| anyhow::anyhow!("invalid public funding response"))
}

fn parse_public_funding(value: &Value, cfg: &AsterConfig) -> Option<FundingUpdate> {
    let data = value
        .get("data")
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let rate_native = data
        .get("fundingRate")
        .or_else(|| data.get("funding_rate"))
        .or_else(|| data.get("lastFundingRate"))
        .and_then(parse_f64);
    let interval_sec = data
        .get("fundingIntervalSec")
        .or_else(|| data.get("funding_interval_sec"))
        .or_else(|| data.get("fundingInterval"))
        .and_then(parse_i64_value)
        .and_then(|v| if v > 0 { Some(v as u64) } else { None })
        .or_else(|| {
            // Aster settles funding every 8h. Default to 28800s when API doesn't provide it.
            // Ref: https://docs.asterdex.com/astherusex-orderbook-perp-guide/mark-price-oracle
            if rate_native.is_some() {
                Some(28_800)
            } else {
                None
            }
        });
    let next_funding_ms = data
        .get("nextFundingTime")
        .or_else(|| data.get("next_funding_time"))
        .or_else(|| data.get("nextFundingTimestamp"))
        .and_then(parse_i64_value);
    let as_of_ms = data
        .get("time")
        .or_else(|| data.get("timestamp"))
        .or_else(|| data.get("ts"))
        .and_then(parse_i64_value)
        .unwrap_or_else(now_ms);
    let rate_8h = match (rate_native, interval_sec) {
        (Some(rate), Some(sec)) if sec > 0 => Some(rate * (8.0 * 60.0 * 60.0 / sec as f64)),
        (Some(rate), None) => Some(rate),
        _ => None,
    };
    Some(FundingUpdate {
        venue_index: cfg.venue_index,
        venue_id: cfg.venue_id.clone(),
        seq: 0,
        timestamp_ms: as_of_ms,
        received_ms: Some(now_ms()),
        funding_rate_8h: rate_8h,
        funding_rate_native: rate_native,
        interval_sec,
        next_funding_ms,
        // Aster uses mark price for funding: Mark = Median(Price1, Price2, ContractPrice).
        // Ref: https://docs.asterdex.com/astherusex-orderbook-perp-guide/mark-price-oracle
        settlement_price_kind: Some(SettlementPriceKind::Mark),
        source: FundingSource::MarketDataRest,
    })
}

fn parse_mark_price_update(text: &str, cfg: &AsterConfig) -> anyhow::Result<Option<FundingUpdate>> {
    let value: Value = serde_json::from_str(text)?;
    let data = value.get("data").unwrap_or(&value);
    if data.is_null() {
        return Ok(None);
    }
    let event_type = data.get("e").and_then(|v| v.as_str()).unwrap_or_default();
    if !event_type.is_empty() && event_type != "markPriceUpdate" {
        return Ok(None);
    }
    let symbol = data
        .get("s")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing symbol"))?;
    if !symbol_matches(symbol, &cfg.market) {
        return Ok(None);
    }
    let rate_native = data
        .get("r")
        .or_else(|| data.get("fundingRate"))
        .or_else(|| data.get("lastFundingRate"))
        .and_then(parse_f64);
    let next_funding_ms = data
        .get("T")
        .or_else(|| data.get("nextFundingTime"))
        .or_else(|| data.get("nextFundingTimestamp"))
        .and_then(parse_i64_value);
    let timestamp_ms = data
        .get("E")
        .or_else(|| data.get("time"))
        .or_else(|| data.get("timestamp"))
        .and_then(parse_i64_value)
        .unwrap_or_else(now_ms);
    if rate_native.is_none() && next_funding_ms.is_none() {
        return Ok(None);
    }
    Ok(Some(FundingUpdate {
        venue_index: cfg.venue_index,
        venue_id: cfg.venue_id.clone(),
        seq: 0,
        timestamp_ms,
        received_ms: Some(now_ms()),
        funding_rate_8h: rate_native,
        funding_rate_native: rate_native,
        interval_sec: rate_native.map(|_| 28_800),
        next_funding_ms,
        settlement_price_kind: Some(SettlementPriceKind::Mark),
        source: FundingSource::MarketDataWs,
    }))
}

fn map_rest_error(status: u16, body: &str) -> LiveGatewayError {
    let lower = body.to_lowercase();
    if status == 429 || status == 418 || lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(body);
    }
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(body);
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(body);
    }
    if status >= 500 || lower.contains("timeout") || lower.contains("tempor") {
        return LiveGatewayError::retryable(body);
    }
    LiveGatewayError {
        kind: LiveGatewayErrorKind::Fatal,
        message: body.to_string(),
    }
}

#[derive(Debug, Clone)]
struct AsterDepthSnapshot {
    last_update_id: u64,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

#[derive(Debug, Clone)]
struct AsterDepthUpdate {
    symbol: String,
    event_time: Option<TimestampMs>,
    start_id: u64,
    end_id: u64,
    prev_id: Option<u64>,
    bids: Vec<BookLevelDelta>,
    asks: Vec<BookLevelDelta>,
}

#[derive(Debug, Clone)]
struct AsterSeqState {
    last_update_id: u64,
    venue_index: usize,
    venue_id: String,
}

impl AsterSeqState {
    fn new(last_update_id: u64, venue_index: usize, venue_id: &str) -> Self {
        Self {
            last_update_id,
            venue_index,
            venue_id: venue_id.to_string(),
        }
    }

    fn apply_update(
        &mut self,
        update: &AsterDepthUpdate,
    ) -> anyhow::Result<Option<MarketDataEvent>> {
        if let Some(prev) = update.prev_id {
            if prev != self.last_update_id && !seq_bridge_ok(self.last_update_id, update) {
                return Err(anyhow::anyhow!(
                    "aster seq mismatch prev_id={} last={}",
                    prev,
                    self.last_update_id
                ));
            }
        }
        if update.end_id <= self.last_update_id {
            return Ok(None);
        }
        if update.start_id > self.last_update_id + 1 {
            return Err(anyhow::anyhow!(
                "aster seq gap last={} next_start={}",
                self.last_update_id,
                update.start_id
            ));
        }
        self.last_update_id = update.end_id;
        let mut changes = Vec::with_capacity(update.bids.len() + update.asks.len());
        changes.extend(update.bids.iter().cloned());
        changes.extend(update.asks.iter().cloned());
        let event = MarketDataEvent::L2Delta(super::super::types::L2Delta {
            venue_index: self.venue_index,
            venue_id: self.venue_id.clone(),
            seq: update.end_id,
            timestamp_ms: update.event_time.unwrap_or_else(now_ms),
            changes,
        });
        Ok(Some(event))
    }
}

type SnapshotFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<(String, AsterDepthSnapshot)>> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeqDecision {
    Apply,
    Stale,
    Gap,
}

fn seq_bridge_ok(last_update_id: u64, update: &AsterDepthUpdate) -> bool {
    let want = last_update_id + 1;
    update.start_id <= want && update.end_id >= want
}

fn seq_decision_lenient(last_update_id: u64, update: &AsterDepthUpdate) -> SeqDecision {
    if let Some(prev) = update.prev_id {
        if update.end_id <= last_update_id {
            return SeqDecision::Stale;
        }
        if prev == last_update_id {
            return SeqDecision::Apply;
        }
        if seq_bridge_ok(last_update_id, update) {
            return SeqDecision::Apply;
        }
        if prev < last_update_id {
            return SeqDecision::Stale;
        }
        return SeqDecision::Gap;
    }
    if update.end_id <= last_update_id {
        SeqDecision::Stale
    } else if update.start_id > last_update_id + 1 {
        SeqDecision::Gap
    } else {
        SeqDecision::Apply
    }
}

fn delta_event_from_update(
    update: &AsterDepthUpdate,
    venue_index: usize,
    venue_id: &str,
    seq: u64,
) -> MarketDataEvent {
    let mut changes = Vec::with_capacity(update.bids.len() + update.asks.len());
    changes.extend(update.bids.iter().cloned());
    changes.extend(update.asks.iter().cloned());
    MarketDataEvent::L2Delta(super::super::types::L2Delta {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms: update.event_time.unwrap_or_else(now_ms),
        changes,
    })
}

#[derive(Debug)]
struct AsterRecorder {
    dir: PathBuf,
}

impl AsterRecorder {
    fn new(dir: &PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self { dir: dir.clone() })
    }

    fn record_snapshot(&mut self, raw: &str) -> std::io::Result<()> {
        let path = self.dir.join("rest_snapshot.json");
        std::fs::write(path, raw)
    }

    fn record_ws_frame(&mut self, raw: &str) -> std::io::Result<()> {
        let path = self.dir.join("ws_frames.jsonl");
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        use std::io::Write;
        file.write_all(raw.as_bytes())?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

fn parse_depth_snapshot(value: &Value) -> Option<AsterDepthSnapshot> {
    let last_update_id = value.get("lastUpdateId")?.as_u64()?;
    let bids = parse_levels_from_value(value.get("bids")?)?;
    let asks = parse_levels_from_value(value.get("asks")?)?;
    Some(AsterDepthSnapshot {
        last_update_id,
        bids,
        asks,
    })
}

fn decode_top_of_book_with_raw(value: &Value) -> Option<(TopOfBook, String, String)> {
    let bids_raw = value.get("bids")?.as_array()?;
    let asks_raw = value.get("asks")?.as_array()?;
    let bid_entry = bids_raw.first()?;
    let ask_entry = asks_raw.first()?;
    let bid_items = bid_entry.as_array()?;
    let ask_items = ask_entry.as_array()?;
    let bid_sz_raw = bid_items.get(1)?.to_string();
    let ask_sz_raw = ask_items.get(1)?.to_string();
    let bids = parse_levels_from_value(value.get("bids")?)?;
    let asks = parse_levels_from_value(value.get("asks")?)?;
    let timestamp_ms = value.get("E").and_then(|v| v.as_i64());
    let top = TopOfBook::from_levels(&bids, &asks, timestamp_ms)?;
    Some((top, bid_sz_raw, ask_sz_raw))
}

fn log_decode_miss(venue: &str, value: &Value, payload: &str, count: usize, url: &str) {
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

fn parse_depth_update(text: &str) -> Option<AsterDepthUpdate> {
    let value: Value = serde_json::from_str(text).ok()?;
    let payload = value.get("data").unwrap_or(&value);
    let event = payload.get("e").and_then(|v| v.as_str()).unwrap_or("");
    if event != "depthUpdate" {
        return None;
    }
    let symbol = payload.get("s")?.as_str()?.to_string();
    let start_id = payload.get("U")?.as_u64()?;
    let end_id = payload.get("u")?.as_u64()?;
    let prev_id = payload.get("pu").and_then(|v| v.as_u64());
    let event_time = payload
        .get("E")
        .and_then(|v| v.as_i64())
        .map(|v| v as TimestampMs);
    let bids = parse_deltas_from_value(payload.get("b")?, BookSide::Bid)?;
    let asks = parse_deltas_from_value(payload.get("a")?, BookSide::Ask)?;
    Some(AsterDepthUpdate {
        symbol,
        event_time,
        start_id,
        end_id,
        prev_id,
        bids,
        asks,
    })
}

fn parse_levels_from_value(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        out.push(BookLevel { price, size });
    }
    Some(out)
}

fn parse_deltas_from_value(value: &Value, side: BookSide) -> Option<Vec<BookLevelDelta>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        out.push(BookLevelDelta { side, price, size });
    }
    Some(out)
}

fn parse_level_pair(value: &Value) -> Option<(f64, f64)> {
    let items = value.as_array()?;
    if items.len() < 2 {
        return None;
    }
    let price = parse_f64(&items[0])?;
    let size = parse_f64(&items[1])?;
    Some((price, size))
}

fn parse_f64(value: &Value) -> Option<f64> {
    if let Some(v) = value.as_f64() {
        return Some(v);
    }
    if let Some(s) = value.as_str() {
        return s.parse::<f64>().ok();
    }
    None
}

fn parse_i64_value(value: &Value) -> Option<i64> {
    if let Some(v) = value.as_i64() {
        return Some(v);
    }
    if let Some(v) = value.as_f64() {
        return Some(v as i64);
    }
    if let Some(s) = value.as_str() {
        return s.parse::<i64>().ok();
    }
    None
}

fn symbol_matches(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as TimestampMs
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FixtureSnapshot {
    seq: u64,
    timestamp_ms: TimestampMs,
    bids: Vec<[f64; 2]>,
    asks: Vec<[f64; 2]>,
    venue_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureDelta {
    seq: u64,
    timestamp_ms: TimestampMs,
    side: String,
    price: f64,
    size: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixturePosition {
    symbol: String,
    size: f64,
    entry_price: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureBalance {
    asset: String,
    total: f64,
    available: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureMargin {
    balance_usd: f64,
    used_usd: f64,
    available_usd: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureLiquidation {
    price_liq: Option<f64>,
    dist_liq_sigma: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureAccountSnapshot {
    seq: u64,
    timestamp_ms: TimestampMs,
    positions: Vec<FixturePosition>,
    balances: Vec<FixtureBalance>,
    funding_8h: Option<f64>,
    margin: FixtureMargin,
    liquidation: FixtureLiquidation,
}

#[derive(Debug, Clone)]
pub struct AsterFixtureFeed {
    snapshot: FixtureSnapshot,
    deltas: Vec<FixtureDelta>,
    account: FixtureAccountSnapshot,
}

impl AsterFixtureFeed {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let snapshot_path = dir.join("snapshot.json");
        let deltas_path = dir.join("deltas.jsonl");
        let account_path = dir.join("account_snapshot.json");

        let snapshot = read_json::<FixtureSnapshot>(&snapshot_path)?;
        let deltas = read_json_lines::<FixtureDelta>(&deltas_path)?;
        let account = read_json::<FixtureAccountSnapshot>(&account_path)?;
        Ok(Self {
            snapshot,
            deltas,
            account,
        })
    }

    pub async fn run_ticks(
        &self,
        market_tx: mpsc::Sender<MarketDataEvent>,
        account_tx: mpsc::Sender<AccountEvent>,
        venue_id: &str,
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
            let snapshot = snapshot_event(&self.snapshot, venue_id, venue_index, seq, now_ms);
            seq = seq.wrapping_add(1);
            let _ = market_tx.send(snapshot).await;
            for delta in &self.deltas {
                let delta_event = delta_event(delta, venue_id, venue_index, seq, now_ms);
                seq = seq.wrapping_add(1);
                let _ = market_tx.send(delta_event).await;
            }
            let account = account_event(&self.account, venue_id, venue_index, seq, now_ms);
            seq = seq.wrapping_add(1);
            let _ = account_tx.send(account).await;
            if pace_ticks {
                tokio::time::sleep(sleep_duration).await;
            } else {
                tokio::task::yield_now().await;
            }
        }
    }
}

fn snapshot_event(
    snapshot: &FixtureSnapshot,
    venue_id: &str,
    venue_index: usize,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> MarketDataEvent {
    let bids = parse_levels(&snapshot.bids);
    let asks = parse_levels(&snapshot.asks);
    MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    })
}

fn delta_event(
    delta: &FixtureDelta,
    venue_id: &str,
    venue_index: usize,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> MarketDataEvent {
    let side = match delta.side.as_str() {
        "bid" | "Bid" | "BID" => BookSide::Bid,
        "ask" | "Ask" | "ASK" => BookSide::Ask,
        _ => BookSide::Bid,
    };
    MarketDataEvent::L2Delta(super::super::types::L2Delta {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        changes: vec![BookLevelDelta {
            side,
            price: delta.price,
            size: delta.size,
        }],
    })
}

fn account_event(
    account: &FixtureAccountSnapshot,
    venue_id: &str,
    venue_index: usize,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> AccountEvent {
    let positions = account
        .positions
        .iter()
        .map(|pos| PositionSnapshot {
            symbol: pos.symbol.clone(),
            size: pos.size,
            entry_price: pos.entry_price,
        })
        .collect();
    let balances = account
        .balances
        .iter()
        .map(|bal| BalanceSnapshot {
            asset: bal.asset.clone(),
            total: bal.total,
            available: bal.available,
        })
        .collect();
    AccountEvent::Snapshot(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        positions,
        balances,
        funding_8h: account.funding_8h,
        margin: MarginSnapshot {
            balance_usd: account.margin.balance_usd,
            used_usd: account.margin.used_usd,
            available_usd: account.margin.available_usd,
        },
        liquidation: LiquidationSnapshot {
            price_liq: account.liquidation.price_liq,
            dist_liq_sigma: account.liquidation.dist_liq_sigma,
        },
    })
}

fn parse_levels(levels: &[[f64; 2]]) -> Vec<BookLevel> {
    levels
        .iter()
        .map(|level| BookLevel {
            price: level[0],
            size: level[1],
        })
        .collect()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("fixture_read_error path={} err={}", path.display(), err))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("fixture_parse_error path={} err={}", path.display(), err))
}

fn read_json_lines<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("fixture_read_error path={} err={}", path.display(), err))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let item: T = serde_json::from_str(line)
            .map_err(|err| format!("fixture_parse_error path={} err={}", path.display(), err))?;
        out.push(item);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::{DELETE, GET, POST};
    use httpmock::MockServer;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/aster");
        let feed = AsterFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
    }

    #[test]
    fn parse_public_funding_fixture() {
        let raw = include_str!("../../../../tests/fixtures/aster/public_premium_index.json");
        let value: Value = serde_json::from_str(raw).expect("fixture json");
        let cfg = AsterConfig {
            ws_url: "wss://example".to_string(),
            rest_url: "https://example".to_string(),
            market: "BTCUSDT".to_string(),
            depth_limit: 100,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: None,
            api_secret: None,
            recv_window: Some(5_000),
            record_dir: None,
        };
        let update = parse_public_funding(&value, &cfg).expect("funding update");
        assert_eq!(update.funding_rate_8h, Some(0.0002));
        assert_eq!(update.next_funding_ms, Some(1_700_003_600_000));
        assert_eq!(update.source, FundingSource::MarketDataRest);
        // Fix: Aster interval_sec must be 28800 (8h funding period)
        assert_eq!(
            update.interval_sec,
            Some(28_800),
            "Aster interval_sec must be 28800"
        );
        // Fix C: Aster settles funding at mark price.
        assert_eq!(
            update.settlement_price_kind,
            Some(SettlementPriceKind::Mark),
            "Aster settlement must be Mark"
        );
    }

    #[test]
    fn parse_mark_price_update_fixture() {
        let raw = r#"{
            "e":"markPriceUpdate",
            "E":1700000000123,
            "s":"BTCUSDT",
            "p":"43210.1",
            "i":"43205.0",
            "P":"43204.5",
            "r":"0.000125",
            "T":1700003600000
        }"#;
        let cfg = AsterConfig {
            ws_url: "wss://example".to_string(),
            rest_url: "https://example".to_string(),
            market: "BTCUSDT".to_string(),
            depth_limit: 100,
            venue_index: 2,
            venue_id: "ASTER".to_string(),
            api_key: None,
            api_secret: None,
            recv_window: Some(5_000),
            record_dir: None,
        };
        let update = parse_mark_price_update(raw, &cfg)
            .expect("parse ok")
            .expect("funding update");
        assert_eq!(update.venue_index, 2);
        assert_eq!(update.venue_id, "ASTER");
        assert_eq!(update.timestamp_ms, 1_700_000_000_123);
        assert_eq!(update.next_funding_ms, Some(1_700_003_600_000));
        assert_eq!(update.funding_rate_native, Some(0.000125));
        assert_eq!(update.funding_rate_8h, Some(0.000125));
        assert_eq!(update.interval_sec, Some(28_800));
        assert_eq!(
            update.settlement_price_kind,
            Some(SettlementPriceKind::Mark)
        );
        assert_eq!(update.source, FundingSource::MarketDataWs);
    }

    #[test]
    fn delta_applies_to_snapshot_levels() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/aster");
        let feed = AsterFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let mut bids = feed.snapshot.bids.clone();
        let delta = feed.deltas.first().expect("delta");
        let side = delta.side.to_lowercase();
        if side == "bid" {
            if let Some(level) = bids.iter_mut().find(|level| level[0] == delta.price) {
                level[1] = delta.size;
            } else {
                bids.push([delta.price, delta.size]);
            }
            assert!(bids.iter().any(|level| level[0] == delta.price));
        }
    }

    #[test]
    fn seq_gap_triggers_refresh_marker() {
        let gap = FixtureDelta {
            seq: 7,
            timestamp_ms: 1_000,
            side: "ask".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let next = FixtureDelta {
            seq: 9,
            timestamp_ms: 1_010,
            side: "ask".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let mut last_seq = gap.seq;
        let gap_detected = next.seq > last_seq + 1;
        last_seq = next.seq;
        assert!(gap_detected);
        assert_eq!(last_seq, 9);
    }

    #[test]
    fn deterministic_serialization_roundtrip() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/aster");
        let feed = AsterFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let raw = serde_json::to_string(&feed.snapshot).expect("serialize");
        let reparsed: FixtureSnapshot = serde_json::from_str(&raw).expect("reparse");
        assert_eq!(feed.snapshot.seq, reparsed.seq);
        assert_eq!(feed.snapshot.bids.len(), reparsed.bids.len());
    }

    #[test]
    fn live_snapshot_fixture_parses() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/aster_live_recording");
        let raw =
            std::fs::read_to_string(fixture_dir.join("rest_snapshot.json")).expect("snapshot raw");
        let value: Value = serde_json::from_str(&raw).expect("snapshot json");
        let snapshot = parse_depth_snapshot(&value).expect("parse snapshot");
        assert!(snapshot.last_update_id > 0);
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[test]
    fn live_ws_replay_is_deterministic_and_monotonic() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/aster_live_recording");
        let snapshot_raw =
            std::fs::read_to_string(fixture_dir.join("rest_snapshot.json")).expect("snapshot raw");
        let snapshot_value: Value = serde_json::from_str(&snapshot_raw).expect("snapshot json");
        let snapshot = parse_depth_snapshot(&snapshot_value).expect("parse snapshot");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");

        let collect_events = |snapshot_id: u64| -> Vec<MarketDataEvent> {
            let mut state = AsterSeqState::new(snapshot_id, 0, "ASTER");
            let mut events = Vec::new();
            for line in frames.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Some(update) = parse_depth_update(trimmed) {
                    let outcome = state.apply_update(&update).expect("seq ok");
                    if let Some(event) = outcome {
                        events.push(event);
                    }
                }
            }
            events
        };

        let events_a = collect_events(snapshot.last_update_id);
        let events_b = collect_events(snapshot.last_update_id);
        assert_eq!(events_a, events_b);

        let mut last_ts: Option<TimestampMs> = None;
        for event in events_a {
            let ts = match event {
                MarketDataEvent::L2Delta(delta) => delta.timestamp_ms,
                MarketDataEvent::L2Snapshot(snapshot) => snapshot.timestamp_ms,
                MarketDataEvent::Trade(trade) => trade.timestamp_ms,
                MarketDataEvent::FundingUpdate(update) => update.timestamp_ms,
            };
            if let Some(prev) = last_ts {
                assert!(ts >= prev);
            }
            last_ts = Some(ts);
        }
    }

    #[test]
    fn signing_matches_known_vector() {
        let query = "price=100&quantity=0.1&recvWindow=5000&side=BUY&symbol=BTCUSDT&timeInForce=GTC&timestamp=1700000000000&type=LIMIT";
        let signature = sign_query("testsecret", query);
        assert_eq!(
            signature,
            "7ce35481df1c771813dfdf305ecf8a94804816bdc818eeb0404e79a58c887f66"
        );
    }

    #[test]
    fn seq_gap_triggers_resync_decision() {
        let update = AsterDepthUpdate {
            symbol: "BTCUSDT".to_string(),
            event_time: None,
            start_id: 110,
            end_id: 111,
            prev_id: Some(109),
            bids: vec![BookLevelDelta {
                side: BookSide::Bid,
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![BookLevelDelta {
                side: BookSide::Ask,
                price: 101.0,
                size: 1.0,
            }],
        };
        let decision = seq_decision_lenient(100, &update);
        assert_eq!(decision, SeqDecision::Gap);
    }

    #[test]
    fn seq_bridge_allows_lock_on_after_snapshot() {
        let update = AsterDepthUpdate {
            symbol: "BTCUSDT".to_string(),
            event_time: None,
            start_id: 95,
            end_id: 105,
            prev_id: Some(90),
            bids: vec![BookLevelDelta {
                side: BookSide::Bid,
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![BookLevelDelta {
                side: BookSide::Ask,
                price: 101.0,
                size: 1.0,
            }],
        };
        let decision = seq_decision_lenient(100, &update);
        assert_eq!(decision, SeqDecision::Apply);

        let mut state = AsterSeqState::new(100, 0, "ASTER");
        let event = state.apply_update(&update).expect("apply update");
        assert!(event.is_some());
        assert_eq!(state.last_update_id, 105);
    }

    #[tokio::test]
    async fn rest_place_order_post_only_is_signed() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

        let exchange_info = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v1/exchangeInfo")
                    .query_param("symbol", "BTCUSDT");
                then.status(200).json_body(serde_json::json!({
                    "symbols": [{
                        "symbol": "BTCUSDT",
                        "filters": [{
                            "filterType": "PRICE_FILTER",
                            "tickSize": "0.1"
                        }]
                    }]
                }));
            })
            .await;
        let expected_signature = "4b0927aa17b493de48e207d2e891485c491aefb6c6ed0bd374259b42a21a1284";
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/fapi/v1/order")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("symbol", "BTCUSDT")
                    .query_param("side", "BUY")
                    .query_param("type", "LIMIT")
                    .query_param("timeInForce", "GTX")
                    .query_param("price", "100")
                    .query_param("quantity", "0.1")
                    .query_param("newClientOrderId", "co_post_only")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000000000")
                    .query_param("signature", expected_signature);
                then.status(200).body("{\"orderId\": 12345}");
            })
            .await;

        let _ = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "aster".to_string(),
                side: Side::Buy,
                price: 100.0,
                size: 0.1,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_post_only".to_string(),
            })
            .await
            .expect("place order");

        exchange_info.assert_async().await;
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fetch_public_funding_http_error_surfaces_status() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 100,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: None,
            api_secret: None,
            recv_window: Some(5_000),
            record_dir: None,
        };
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v1/premiumIndex")
                    .query_param("symbol", "BTCUSDT");
                then.status(429)
                    .header("x-mbx-used-weight-1m", "2400")
                    .body("{\"code\":-1003,\"msg\":\"Too many requests\"}");
            })
            .await;

        let err = fetch_public_funding(&Client::new(), &cfg)
            .await
            .expect_err("expected rate limit failure");
        let msg = err.to_string();
        assert!(msg.contains("HTTP status 429"), "msg={msg}");
        assert!(msg.contains("weight_1m=2400"), "msg={msg}");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_place_order_ioc_reduce_only_is_signed() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

        let exchange_info = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v1/exchangeInfo")
                    .query_param("symbol", "BTCUSDT");
                then.status(200).json_body(serde_json::json!({
                    "symbols": [{
                        "symbol": "BTCUSDT",
                        "filters": [{
                            "filterType": "PRICE_FILTER",
                            "tickSize": "0.1"
                        }]
                    }]
                }));
            })
            .await;
        let expected_signature = "fb231bb1595dd627ceab277d9d9b6f9ff238ad515830ba44ea5717e01ff578ad";
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/fapi/v1/order")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("symbol", "BTCUSDT")
                    .query_param("side", "SELL")
                    .query_param("type", "LIMIT")
                    .query_param("timeInForce", "IOC")
                    .query_param("price", "101")
                    .query_param("quantity", "0.2")
                    .query_param("newClientOrderId", "co_ioc_ro")
                    .query_param("reduceOnly", "true")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000000000")
                    .query_param("signature", expected_signature);
                then.status(200).body("{\"orderId\": 67890}");
            })
            .await;

        let _ = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "aster".to_string(),
                side: Side::Sell,
                price: 101.0,
                size: 0.2,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: "co_ioc_ro".to_string(),
            })
            .await
            .expect("place order");

        exchange_info.assert_async().await;
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_place_order_formats_decimal_fields_without_float_noise() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "ETHUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

        let exchange_info = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v1/exchangeInfo")
                    .query_param("symbol", "ETHUSDT");
                then.status(200).json_body(serde_json::json!({
                    "symbols": [{
                        "symbol": "ETHUSDT",
                        "filters": [{
                            "filterType": "PRICE_FILTER",
                            "tickSize": "0.01"
                        }]
                    }]
                }));
            })
            .await;
        let canonical = "newClientOrderId=co_decimal&price=2073.54&quantity=0.01&recvWindow=5000&side=BUY&symbol=ETHUSDT&timeInForce=GTX&timestamp=1700000000000&type=LIMIT";
        let expected_signature = sign_query("testsecret", canonical);
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/fapi/v1/order")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("symbol", "ETHUSDT")
                    .query_param("side", "BUY")
                    .query_param("type", "LIMIT")
                    .query_param("timeInForce", "GTX")
                    .query_param("price", "2073.54")
                    .query_param("quantity", "0.01")
                    .query_param("newClientOrderId", "co_decimal")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000000000")
                    .query_param("signature", expected_signature);
                then.status(200).body("{\"orderId\": 13579}");
            })
            .await;

        let _ = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "aster".to_string(),
                side: Side::Buy,
                price: 2073.5399999999995,
                size: 0.010000000000000002,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_decimal".to_string(),
            })
            .await
            .expect("place order");

        exchange_info.assert_async().await;
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_place_order_rounds_post_only_prices_to_exchange_tick() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "ETHUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

        let exchange_info = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v1/exchangeInfo")
                    .query_param("symbol", "ETHUSDT");
                then.status(200).json_body(serde_json::json!({
                    "symbols": [{
                        "symbol": "ETHUSDT",
                        "filters": [{
                            "filterType": "PRICE_FILTER",
                            "tickSize": "0.1"
                        }]
                    }]
                }));
            })
            .await;
        let canonical = "newClientOrderId=co_tick&price=2074&quantity=0.01&recvWindow=5000&side=BUY&symbol=ETHUSDT&timeInForce=GTX&timestamp=1700000000000&type=LIMIT";
        let expected_signature = sign_query("testsecret", canonical);
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/fapi/v1/order")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("symbol", "ETHUSDT")
                    .query_param("side", "BUY")
                    .query_param("type", "LIMIT")
                    .query_param("timeInForce", "GTX")
                    .query_param("price", "2074")
                    .query_param("quantity", "0.01")
                    .query_param("newClientOrderId", "co_tick")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000000000")
                    .query_param("signature", expected_signature);
                then.status(200).body("{\"orderId\": 24680}");
            })
            .await;

        let _ = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "aster".to_string(),
                side: Side::Buy,
                price: 2074.08,
                size: 0.01,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_tick".to_string(),
            })
            .await
            .expect("place order");

        exchange_info.assert_async().await;
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_cancel_all_is_signed() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

        let expected_signature = "c848f23c14e1e39ab9b87af2e2b433ebc78ab2393952b62660e5229c0c979fdf";
        let mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/fapi/v1/allOpenOrders")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("symbol", "BTCUSDT")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000000000")
                    .query_param("signature", expected_signature);
                then.status(200).body("{}");
            })
            .await;

        let _ = client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 0,
                venue_id: "aster".to_string(),
            })
            .await
            .expect("cancel_all");

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_cancel_order_uses_numeric_order_id_when_available() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "ETHUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

        let canonical = "orderId=12345&recvWindow=5000&symbol=ETHUSDT&timestamp=1700000000000";
        let expected_signature = sign_query("testsecret", canonical);
        let mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/fapi/v1/order")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("symbol", "ETHUSDT")
                    .query_param("orderId", "12345")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000000000")
                    .query_param("signature", expected_signature);
                then.status(200).body("{}");
            })
            .await;

        let _ = client
            .cancel_order(LiveRestCancelRequest {
                venue_index: 0,
                venue_id: "aster".to_string(),
                order_id: "12345".to_string(),
            })
            .await
            .expect("cancel order");

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn fetch_account_snapshot_uses_poll_time_for_freshness() {
        let server = MockServer::start_async().await;
        let cfg = AsterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "ETHUSDT".to_string(),
            depth_limit: 10,
            venue_index: 2,
            venue_id: "ASTER".to_string(),
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = AsterRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_123_456));

        let canonical = "recvWindow=5000&timestamp=1700000123456";
        let expected_signature = sign_query("testsecret", canonical);
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v2/account")
                    .header("X-MBX-APIKEY", "test-key")
                    .query_param("recvWindow", "5000")
                    .query_param("timestamp", "1700000123456")
                    .query_param("signature", expected_signature);
                then.status(200).json_body(serde_json::json!({
                    "updateTime": 1_700_000_000_000u64,
                    "totalWalletBalance": "100.0",
                    "totalPositionInitialMargin": "10.0",
                    "availableBalance": "90.0",
                    "positions": [{
                        "symbol": "ETHUSDT",
                        "positionAmt": "0.14",
                        "entryPrice": "2091.67",
                        "updateTime": 1_700_000_000_000u64
                    }],
                    "assets": [{
                        "asset": "USDT",
                        "walletBalance": "100.0",
                        "availableBalance": "90.0"
                    }]
                }));
            })
            .await;

        let snapshot = client
            .fetch_account_snapshot("ASTER", 2)
            .await
            .expect("account snapshot");

        mock.assert_async().await;
        assert_eq!(snapshot.seq, 1_700_000_000_000u64);
        assert_eq!(snapshot.timestamp_ms, 1_700_000_123_456);
        assert_eq!(snapshot.positions.len(), 1);
        assert!((snapshot.positions[0].size - 0.14).abs() < 1e-9);
    }

    #[test]
    fn parse_open_orders_keeps_remaining_size_and_client_order_id() {
        let value = serde_json::json!([
            {
                "symbol": "ETHUSDT",
                "status": "PARTIALLY_FILLED",
                "orderId": 12345,
                "clientOrderId": "co_aster_1",
                "side": "BUY",
                "price": "2090.5",
                "origQty": "0.50",
                "executedQty": "0.20"
            },
            {
                "symbol": "BTCUSDT",
                "status": "NEW",
                "orderId": 999,
                "side": "SELL",
                "price": "60000",
                "origQty": "1.0",
                "executedQty": "0"
            }
        ]);

        let open_orders = parse_open_orders(&value, "ETHUSDT");

        assert_eq!(open_orders.len(), 1);
        assert_eq!(open_orders[0].order_id, "12345");
        assert_eq!(
            open_orders[0].client_order_id.as_deref(),
            Some("co_aster_1")
        );
        assert_eq!(open_orders[0].side, Side::Buy);
        assert!((open_orders[0].price - 2090.5).abs() < 1e-9);
        assert!((open_orders[0].size - 0.30).abs() < 1e-9);
        assert_eq!(open_orders[0].purpose, None);
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
        assert_eq!(
            freshness.last_book_event_ns.load(Ordering::Relaxed),
            0,
            "last_book_event_ns must be reset on new connection"
        );

        let connect_start_ns = 1_000;
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, connect_start_ns);

        // Non-book parsed events must NOT advance watchdog anchor
        freshness.last_parsed_ns.store(2_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(
            anchor, connect_start_ns,
            "non-book parsed events must not advance watchdog anchor"
        );

        freshness.last_book_event_ns.store(3_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 3_000);

        freshness.last_published_ns.store(4_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 4_000);
    }

    #[test]
    fn snapshot_failure_classification_is_rate_limit_aware() {
        assert_eq!(classify_snapshot_failure(Some(429)), "rate_limited");
        assert_eq!(classify_snapshot_failure(Some(418)), "ip_banned");
        assert_eq!(classify_snapshot_failure(Some(503)), "upstream_5xx");
        assert_eq!(classify_snapshot_failure(Some(404)), "http_4xx");
        assert_eq!(classify_snapshot_failure(None), "transport_or_parse");
    }

    #[test]
    fn snapshot_backoff_grows_and_caps_for_rate_limits() {
        let first = snapshot_backoff_ms(Some(429), 1, 0);
        let third = snapshot_backoff_ms(Some(429), 3, 0);
        let late = snapshot_backoff_ms(Some(429), 12, 0);
        assert!(first >= 1_000);
        assert!(third > first);
        assert!(late <= 15_000);
    }
}
