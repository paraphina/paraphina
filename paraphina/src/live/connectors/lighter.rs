//! Lighter connector (feature-gated).

pub const STUB_CONNECTOR: bool = false;
pub const SUPPORTS_MARKET: bool = true;
pub const SUPPORTS_ACCOUNT: bool = true;
pub const SUPPORTS_EXECUTION: bool = true;

const LIGHTER_MARKET_PUB_QUEUE_CAP: usize = 256;
const LIGHTER_MARKET_PUB_DRAIN_MAX: usize = 64;
const LIGHTER_STALE_MS_DEFAULT: u64 = 10_000;
const LIGHTER_TICKER_BACKSTOP_AFTER_MS_DEFAULT: u64 = 1_200;
const LIGHTER_WATCHDOG_TICK_MS: u64 = 200;
const LIGHTER_DECODE_WARN_INTERVAL_MS: u64 = 10_000;
const LIGHTER_WS_CONNECT_TIMEOUT_MS_DEFAULT: u64 = 15_000;
const LIGHTER_WS_READ_TIMEOUT_MS_DEFAULT: u64 = 30_000;
const LIGHTER_PING_INTERVAL_MS_DEFAULT: u64 = 30_000;
const LIGHTER_ACCOUNT_LOG_INTERVAL_MS: u64 = 30_000;
const LIGHTER_ACCOUNT_RATE_LIMIT_BASE_BACKOFF_MS: u64 = 2_000;
const LIGHTER_ACCOUNT_RATE_LIMIT_MAX_BACKOFF_MS: u64 = 10_000;
const LIGHTER_EMERGENCY_IOC_TIMEOUT_MS_DEFAULT: u64 = 1_000;
const PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV: &str =
    "PARAPHINA_PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION";
const PHASE51_FORWARD_REFRESH_CAPTURE_ENABLED_ENV: &str =
    "PARAPHINA_PHASE51_FORWARD_REFRESH_CAPTURE_ENABLED";
const PHASE51_FORWARD_REFRESH_CAPTURE_LIVE_NATIVE_ROLE_CANARY_APPROVED_ENV: &str =
    "PARAPHINA_PHASE51_FORWARD_REFRESH_CAPTURE_LIVE_NATIVE_ROLE_CANARY_APPROVED";
const PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY_ENV: &str =
    "PARAPHINA_PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY";
const PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV: &str =
    "PARAPHINA_PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY";
const PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE_ENV: &str =
    "PARAPHINA_PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE";
const PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE_DEFAULT: f64 = 0.01;
/// Maximum consecutive delta decode failures before forcing a reconnect to
/// obtain a fresh full snapshot.  Protects against book drift from missed deltas.
const LIGHTER_MAX_CONSECUTIVE_DELTA_FAILURES: usize = 10;
const LIGHTER_CLIENT_ORDER_INDEX_MAX: u64 = (1u64 << 48) - 1;

static MONO_START: OnceLock<Instant> = OnceLock::new();
static LIGHTER_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static LIGHTER_TS_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static LIGHTER_PING_SENT_COUNT: AtomicU64 = AtomicU64::new(0);
static LIGHTER_PING_SEND_FAIL_COUNT: AtomicU64 = AtomicU64::new(0);
static LIGHTER_RECONNECT_COUNTS: OnceLock<StdMutex<BTreeMap<&'static str, u64>>> = OnceLock::new();
static LIGHTER_ACCOUNT_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static LIGHTER_ACCOUNT_LAST_LOG_MS: AtomicU64 = AtomicU64::new(0);
static LIGHTER_ACCOUNT_ERROR_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static LIGHTER_ACCOUNT_ERROR_LAST_LOG_MS: AtomicU64 = AtomicU64::new(0);
static LIGHTER_PHASE51_PASSIVE_PRESSURE_TAP_ROWS: AtomicU64 = AtomicU64::new(0);

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn lighter_stale_ms() -> u64 {
    std::env::var("PARAPHINA_LIGHTER_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(LIGHTER_STALE_MS_DEFAULT)
}

fn lighter_ws_connect_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_LIGHTER_WS_CONNECT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(LIGHTER_WS_CONNECT_TIMEOUT_MS_DEFAULT),
    )
}

fn lighter_ws_read_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_LIGHTER_WS_READ_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(LIGHTER_WS_READ_TIMEOUT_MS_DEFAULT),
    )
}

fn lighter_ping_interval_ms() -> u64 {
    std::env::var("PARAPHINA_LIGHTER_PING_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LIGHTER_PING_INTERVAL_MS_DEFAULT)
}

fn lighter_emergency_ioc_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_LIGHTER_EMERGENCY_IOC_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(LIGHTER_EMERGENCY_IOC_TIMEOUT_MS_DEFAULT),
    )
}

fn lighter_emergency_ioc_request(req: &LiveRestPlaceRequest) -> bool {
    req.reduce_only
        && matches!(req.time_in_force, TimeInForce::Ioc)
        && matches!(req.purpose, OrderPurpose::Exit | OrderPurpose::Hedge)
}

fn phase51_lighter_strict_maker_only_env_value_enabled(value: &str) -> bool {
    let value = value.trim();
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
}

fn phase51_lighter_strict_maker_only_observation_enabled() -> bool {
    std::env::var(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV)
        .map(|value| phase51_lighter_strict_maker_only_env_value_enabled(&value))
        .unwrap_or(false)
}

fn phase51_lighter_true_env(name: &str) -> bool {
    std::env::var(name)
        .map(|value| phase51_lighter_strict_maker_only_env_value_enabled(&value))
        .unwrap_or(false)
}

pub fn phase51_lighter_account_all_trades_source_owner_enabled() -> bool {
    phase51_lighter_true_env(PHASE51_FORWARD_REFRESH_CAPTURE_ENABLED_ENV)
        && phase51_lighter_true_env(
            PHASE51_FORWARD_REFRESH_CAPTURE_LIVE_NATIVE_ROLE_CANARY_APPROVED_ENV,
        )
        && phase51_lighter_true_env(PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY_ENV)
}

fn phase51_lighter_baseline_cleanup_only_enabled() -> bool {
    std::env::var(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV)
        .map(|value| phase51_lighter_strict_maker_only_env_value_enabled(&value))
        .unwrap_or(false)
}

fn phase51_lighter_sanitized_live_error_context_enabled() -> bool {
    phase51_lighter_strict_maker_only_observation_enabled()
        || phase51_lighter_baseline_cleanup_only_enabled()
}

fn phase51_lighter_baseline_cleanup_max_size() -> f64 {
    std::env::var(PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE_DEFAULT)
}

fn phase51_lighter_baseline_cleanup_market_allowed(market: &str) -> bool {
    market.eq_ignore_ascii_case("ETH-USD") || market.eq_ignore_ascii_case("ETH")
}

fn phase51_lighter_baseline_cleanup_place_rejection(
    configured_market: &str,
    req: &LiveRestPlaceRequest,
) -> Option<&'static str> {
    if !phase51_lighter_baseline_cleanup_only_enabled() {
        return None;
    }
    if !phase51_lighter_baseline_cleanup_market_allowed(configured_market) {
        return Some("lighter: baseline cleanup-only rejects non ETH market");
    }
    if !matches!(req.purpose, OrderPurpose::Exit | OrderPurpose::Hedge)
        || !matches!(req.time_in_force, TimeInForce::Ioc)
        || req.post_only
        || !req.reduce_only
    {
        return Some(
            "lighter: baseline cleanup-only rejects order-creating request \
             (requires purpose=Exit/Hedge, time_in_force=Ioc, post_only=false, reduce_only=true)",
        );
    }
    if !req.size.is_finite()
        || req.size <= 0.0
        || req.size > phase51_lighter_baseline_cleanup_max_size() + 1e-12
    {
        return Some("lighter: baseline cleanup-only rejects size outside cleanup cap");
    }
    None
}

fn phase51_lighter_strict_maker_only_place_rejection(
    req: &LiveRestPlaceRequest,
) -> Option<&'static str> {
    if !phase51_lighter_strict_maker_only_observation_enabled() {
        return None;
    }
    if req.purpose != OrderPurpose::Mm
        || !matches!(req.time_in_force, TimeInForce::Gtc)
        || !req.post_only
        || req.reduce_only
    {
        return Some(
            "lighter: strict maker-only observation rejects order-creating request \
             (requires purpose=Mm, time_in_force=Gtc, post_only=true, reduce_only=false)",
        );
    }
    None
}

fn phase51_lighter_place_error_context(
    market_id: u64,
    client_order_index: u64,
    price: i64,
    base_amount: i64,
    time_in_force: TimeInForce,
    post_only: bool,
    reduce_only: bool,
) -> String {
    if phase51_lighter_strict_maker_only_observation_enabled() {
        return format!(
            "lighter place context market_id={} tif={:?} post_only={} reduce_only={} strict_maker_only_observation=true",
            market_id, time_in_force, post_only, reduce_only
        );
    }
    if phase51_lighter_baseline_cleanup_only_enabled() {
        return format!(
            "lighter place context market_id={} tif={:?} post_only={} reduce_only={} baseline_cleanup_only=true",
            market_id, time_in_force, post_only, reduce_only
        );
    }
    format!(
        "lighter place context market_id={} client_order_index={} price={} base_amount={} tif={:?} post_only={} reduce_only={}",
        market_id, client_order_index, price, base_amount, time_in_force, post_only, reduce_only
    )
}

fn phase51_lighter_replace_error_context(
    market_id: u64,
    identity_label: &str,
    raw_order_id: u64,
    price: i64,
    base_amount: i64,
    requested_client_order_id: &str,
) -> String {
    if phase51_lighter_strict_maker_only_observation_enabled() {
        return format!(
            "lighter replace context market_id={} identity_kind={} strict_maker_only_observation=true",
            market_id, identity_label
        );
    }
    if phase51_lighter_baseline_cleanup_only_enabled() {
        return format!(
            "lighter replace context market_id={} identity_kind={} baseline_cleanup_only=true",
            market_id, identity_label
        );
    }
    format!(
        "lighter replace context market_id={} {}={} price={} base_amount={} client_order_id={}",
        market_id, identity_label, raw_order_id, price, base_amount, requested_client_order_id
    )
}

fn phase51_lighter_strict_maker_only_retryable_error(
    label: &str,
    err: impl std::fmt::Display,
) -> LiveGatewayError {
    if phase51_lighter_strict_maker_only_observation_enabled() {
        return LiveGatewayError::retryable(format!(
            "lighter: strict maker-only observation {label} failed"
        ));
    }
    if phase51_lighter_baseline_cleanup_only_enabled() {
        return LiveGatewayError::retryable(format!(
            "lighter: baseline cleanup-only {label} failed"
        ));
    }
    LiveGatewayError::retryable(format!("{label}: {err}"))
}

fn phase51_lighter_strict_maker_only_err_with_context(
    err: LiveGatewayError,
    context: &str,
) -> LiveGatewayError {
    if phase51_lighter_strict_maker_only_observation_enabled() {
        return LiveGatewayError {
            kind: err.kind,
            message: format!(
                "lighter: strict maker-only observation operation failed [{}]",
                context
            ),
        };
    }
    if phase51_lighter_baseline_cleanup_only_enabled() {
        return LiveGatewayError {
            kind: err.kind,
            message: format!(
                "lighter: baseline cleanup-only operation failed [{}]",
                context
            ),
        };
    }
    err_with_context(err, context)
}

fn phase51_lighter_strict_maker_only_cancel_err_with_context(
    err: LiveGatewayError,
    operation: &str,
    context: &str,
) -> LiveGatewayError {
    if phase51_lighter_strict_maker_only_observation_enabled() {
        return LiveGatewayError {
            kind: err.kind,
            message: format!(
                "lighter: strict maker-only observation {operation} failed [{}]",
                context
            ),
        };
    }
    if phase51_lighter_baseline_cleanup_only_enabled() {
        return LiveGatewayError {
            kind: err.kind,
            message: format!(
                "lighter: baseline cleanup-only {operation} failed [{}]",
                context
            ),
        };
    }
    err
}

fn phase51_lighter_orderbooks_endpoint_family(endpoint: &str) -> &'static str {
    if endpoint.eq_ignore_ascii_case("/api/v1/orderBooks")
        || endpoint.eq_ignore_ascii_case("/api/v1/orderbooks")
    {
        "orderBooks"
    } else {
        "unknown"
    }
}

fn phase51_lighter_strict_orderbooks_attempt_log(endpoint: &str, attempt_index: usize) -> String {
    format!(
        "INFO: Lighter resolving market id attempt endpoint_family={} attempt_index={} strict_maker_only_observation=true",
        phase51_lighter_orderbooks_endpoint_family(endpoint),
        attempt_index
    )
}

fn phase51_lighter_strict_orderbooks_failure_log(
    endpoint: &str,
    attempt_index: usize,
    status: reqwest::StatusCode,
    reason: &str,
) -> String {
    format!(
        "WARN: Lighter orderBooks fetch failed endpoint_family={} attempt_index={} status={} reason={} strict_maker_only_observation=true",
        phase51_lighter_orderbooks_endpoint_family(endpoint),
        attempt_index,
        status,
        reason
    )
}

fn phase51_lighter_strict_orderbooks_error_log(endpoint: &str, attempt_index: usize) -> String {
    format!(
        "WARN: Lighter orderBooks fetch error endpoint_family={} attempt_index={} reason=request_error strict_maker_only_observation=true",
        phase51_lighter_orderbooks_endpoint_family(endpoint),
        attempt_index
    )
}

fn phase51_lighter_orderbooks_source_label(source: &str) -> &'static str {
    if source == "env:LIGHTER_MARKET_ID" {
        "env_lighter_market_id"
    } else {
        "orderbooks_discovery"
    }
}

fn phase51_lighter_market_decimals_missing_error(
    field: &str,
    market_id: u64,
    source: &str,
) -> anyhow::Error {
    if phase51_lighter_strict_maker_only_observation_enabled() {
        return anyhow::anyhow!(
            "Lighter market decimals missing field={} source_label={} strict_maker_only_observation=true",
            field,
            phase51_lighter_orderbooks_source_label(source)
        );
    }
    anyhow::anyhow!(
        "Lighter {} decimals missing for market_id={} source_url={}",
        field,
        market_id,
        source
    )
}

fn lighter_ticker_backstop_enabled() -> bool {
    env_is_true("PARAPHINA_LIGHTER_TICKER_BACKSTOP_ENABLED")
}

fn lighter_ticker_backstop_after_ms() -> u64 {
    std::env::var("PARAPHINA_LIGHTER_TICKER_BACKSTOP_AFTER_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LIGHTER_TICKER_BACKSTOP_AFTER_MS_DEFAULT)
}

pub fn lighter_account_poll_ms(default_ms: u64) -> u64 {
    std::env::var("PARAPHINA_LIGHTER_ACCOUNT_POLL_MS")
        .ok()
        .or_else(|| std::env::var("LIGHTER_ACCOUNT_POLL_MS").ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_ms)
}

fn env_is_true(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn lighter_ws_readonly_enabled() -> bool {
    env_is_true("PARAPHINA_LIGHTER_WS_READONLY")
        || std::env::var("PARAPHINA_TRADE_MODE")
            .map(|v| v.eq_ignore_ascii_case("shadow"))
            .unwrap_or(false)
}

fn lighter_public_ws_url(base: &str, readonly: bool) -> String {
    if !readonly || base.contains("readonly=") {
        return base.to_string();
    }
    if base.contains('?') {
        format!("{base}&readonly=true")
    } else {
        format!("{base}?readonly=true")
    }
}

/// Wrap a future with a timeout, returning an anyhow error on expiration.
async fn with_timeout<T>(
    duration: Duration,
    label: &str,
    fut: impl std::future::Future<Output = T>,
) -> anyhow::Result<T> {
    tokio::time::timeout(duration, fut)
        .await
        .map_err(|_| anyhow::anyhow!("Lighter {label} timed out after {duration:?}"))
}

fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

fn age_ms_or_connect_age(now_ns: u64, then_ns: u64, connect_start_ns: u64) -> u64 {
    if then_ns == 0 {
        age_ms(now_ns, connect_start_ns)
    } else {
        age_ms(now_ns, then_ns)
    }
}

fn lighter_audit_freshness(
    reason: &'static str,
    freshness: &Freshness,
    connect_start_ns: u64,
    stale_ms: u64,
) {
    if !lighter_ws_audit_enabled() {
        return;
    }
    let now_ns = mono_now_ns();
    let last_ws = freshness.last_ws_rx_ns.load(Ordering::Relaxed);
    let last_data = freshness.last_data_rx_ns.load(Ordering::Relaxed);
    let last_ticker = freshness.last_ticker_ns.load(Ordering::Relaxed);
    let last_parsed = freshness.last_parsed_ns.load(Ordering::Relaxed);
    let last_published = freshness.last_published_ns.load(Ordering::Relaxed);
    let last_book = freshness.last_book_event_ns.load(Ordering::Relaxed);
    let anchor = freshness.anchor_with_connect_start(connect_start_ns);
    eprintln!(
        "WS_AUDIT venue=lighter component=freshness reason={} stale_ms={} connect_age_ms={} age_anchor_ms={} age_ws_rx_ms={} age_data_rx_ms={} age_ticker_ms={} age_parsed_ms={} age_book_event_ms={} age_published_ms={}",
        reason,
        stale_ms,
        age_ms(now_ns, connect_start_ns),
        age_ms(now_ns, anchor),
        age_ms_or_connect_age(now_ns, last_ws, connect_start_ns),
        age_ms_or_connect_age(now_ns, last_data, connect_start_ns),
        age_ms_or_connect_age(now_ns, last_ticker, connect_start_ns),
        age_ms_or_connect_age(now_ns, last_parsed, connect_start_ns),
        age_ms_or_connect_age(now_ns, last_book, connect_start_ns),
        age_ms_or_connect_age(now_ns, last_published, connect_start_ns),
    );
}

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_ticker_ns: AtomicU64,
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
        self.last_ticker_ns.store(0, Ordering::Relaxed);
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

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex as StdMutex, OnceLock,
};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::types::{
    FundingSource, OrderIntent, OrderPurpose, SettlementPriceKind, Side, TimeInForce, TimestampMs,
};

use super::super::gateway::{
    LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest, LiveRestCancelRequest,
    LiveRestClient, LiveRestPlaceRequest, LiveRestReplaceRequest, LiveRestResponse,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, FundingUpdate,
    LiquidationSnapshot, MarginSnapshot, MarketDataEvent, Phase51ForwardRefreshNativeRole,
    Phase51ForwardRefreshSourceOwnerFill, Phase51ForwardRefreshSourceOwnerPfillObservation,
    PositionSnapshot, TopOfBook,
};
use super::lighter_nonce::{load_last_nonce, store_last_nonce, LighterNonceManager};
use super::lighter_signer::{
    LighterSignerClient, SignCancelAllRequest, SignCancelOrderRequest, SignCreateOrderRequest,
    SignModifyOrderRequest, SignedTx,
};
use crate::live::{live_market_pub_drain_max, live_market_pub_queue_cap, MarketPublisher};

#[derive(Debug, Clone)]
pub struct LighterConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub market: String,
    pub venue_id: String,
    pub venue_index: usize,
    pub paper_mode: bool,
    pub api_key_index: Option<u64>,
    pub account_index: Option<u64>,
    pub api_private_key_hex: Option<String>,
    pub auth_token: Option<String>,
    pub nonce_path: Option<PathBuf>,
    pub signer_url: Option<String>,
}

impl LighterConfig {
    pub fn from_env() -> Self {
        let network = std::env::var("LIGHTER_NETWORK")
            .unwrap_or_else(|_| "mainnet".to_string())
            .to_lowercase();
        let (default_rest, default_ws) = match network.as_str() {
            "testnet" => (
                "https://testnet.zklighter.elliot.ai",
                "wss://testnet.zklighter.elliot.ai/stream",
            ),
            _ => (
                "https://mainnet.zklighter.elliot.ai",
                "wss://mainnet.zklighter.elliot.ai/stream",
            ),
        };
        let ws_url = std::env::var("LIGHTER_WS_URL").unwrap_or_else(|_| default_ws.to_string());
        let rest_url = std::env::var("LIGHTER_HTTP_BASE_URL")
            .or_else(|_| std::env::var("LIGHTER_REST_URL"))
            .unwrap_or_else(|_| default_rest.to_string());
        let market = std::env::var("LIGHTER_MARKET").unwrap_or_else(|_| "BTC-USD".to_string());
        let venue_id = std::env::var("LIGHTER_VENUE").unwrap_or_else(|_| "LIGHTER".to_string());
        let paper_mode = std::env::var("LIGHTER_PAPER_MODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let api_key_index = std::env::var("LIGHTER_API_KEY_INDEX")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        let account_index = std::env::var("LIGHTER_ACCOUNT_INDEX")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        let api_private_key_hex = std::env::var("LIGHTER_API_PRIVATE_KEY_HEX").ok();
        let auth_token = std::env::var("LIGHTER_AUTH_TOKEN").ok();
        let nonce_path = std::env::var("LIGHTER_NONCE_PATH")
            .ok()
            .map(std::path::PathBuf::from);
        let signer_url = std::env::var("LIGHTER_SIGNER_URL").ok();
        Self {
            ws_url,
            rest_url,
            market,
            venue_id,
            venue_index: 0,
            paper_mode,
            api_key_index,
            account_index,
            api_private_key_hex,
            auth_token,
            nonce_path,
            signer_url,
        }
    }

    pub fn has_auth(&self) -> bool {
        self.api_key_index.is_some()
            && self.account_index.is_some()
            && self
                .api_private_key_hex
                .as_ref()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
    }

    pub fn has_signer(&self) -> bool {
        self.signer_url
            .as_ref()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    }
}

fn api_base(cfg: &LighterConfig) -> String {
    format!("{}/api/v1", cfg.rest_url.trim_end_matches('/'))
}

fn account_url(cfg: &LighterConfig) -> String {
    format!("{}/account", api_base(cfg))
}

fn sendtx_url(cfg: &LighterConfig) -> String {
    format!("{}/sendTx", api_base(cfg))
}

#[derive(Debug, Clone)]
struct Phase51LighterPassivePressureTapConfig {
    enabled: bool,
    output_path: PathBuf,
    max_rows: u64,
}

impl Phase51LighterPassivePressureTapConfig {
    fn from_env() -> Self {
        let enabled = std::env::var("PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_ENABLED")
            .ok()
            .and_then(|raw| parse_lighter_bool_env(&raw))
            .unwrap_or(false);
        let output_path =
            std::env::var("PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_OUTPUT_PATH")
                .ok()
                .filter(|raw| !raw.trim().is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(
                        "/home/ubuntu/source_owner_inbox/phase51/lighter_passive_pressure_observations.jsonl",
                    )
                });
        let max_rows = std::env::var("PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_MAX_ROWS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|rows| *rows > 0)
            .unwrap_or(5_000);
        Self {
            enabled,
            output_path,
            max_rows,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Phase51LighterPassivePressureObservation {
    schema_version: u32,
    producer: &'static str,
    venue_id: &'static str,
    source_endpoint_family: String,
    observed_at_ms: u64,
    http_status_success: bool,
    no_live_flag: bool,
    approved_for_live: bool,
    approved_for_canary: bool,
    approved_for_capital_escalation: bool,
    live_orders_allowed: bool,
    capital_change_allowed: bool,
    risk_limit_relaxation_allowed: bool,
    phase51_validators_run: bool,
    blocker_cleared: bool,
    raw_request_persisted: bool,
    raw_response_body_persisted: bool,
    unsanitized_headers_persisted: bool,
    secret_material_persisted: bool,
    native_limit_event_time_status: &'static str,
    pressure_dimensions_complete_from_passive_response: bool,
    required_missing_dimensions: Vec<&'static str>,
    active_order_headroom_account: Option<i64>,
    sendtx_per_minute_limit: Option<i64>,
    sendtx_per_minute_remaining: Option<i64>,
    rest_requests_per_minute_limit: Option<i64>,
    rest_requests_per_minute_remaining: Option<i64>,
    weighted_requests_per_minute_limit: Option<i64>,
    weighted_requests_per_minute_remaining: Option<i64>,
    pressure_field_sources: BTreeMap<&'static str, &'static str>,
    sanitized_pressure_header_names: Vec<String>,
    generic_rate_limit_header_present: bool,
}

fn parse_lighter_bool_env(raw: &str) -> Option<bool> {
    let text = raw.trim();
    if text == "1" || text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("yes") {
        Some(true)
    } else if text == "0" || text.eq_ignore_ascii_case("false") || text.eq_ignore_ascii_case("no") {
        Some(false)
    } else {
        None
    }
}

fn phase51_lighter_normalized_key(raw: &str) -> String {
    raw.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn phase51_lighter_json_i64(value: &serde_json::Value) -> Option<i64> {
    if value.is_boolean() || value.is_null() {
        return None;
    }
    if let Some(value) = value.as_i64() {
        return Some(value);
    }
    value.as_str()?.trim().parse::<i64>().ok()
}

fn phase51_lighter_find_i64(value: &serde_json::Value, candidate_keys: &[&str]) -> Option<i64> {
    if let Some(obj) = value.as_object() {
        for (key, item) in obj {
            let normalized = phase51_lighter_normalized_key(key);
            if candidate_keys
                .iter()
                .any(|candidate| *candidate == normalized)
            {
                if let Some(value) = phase51_lighter_json_i64(item) {
                    return Some(value);
                }
            }
        }
        for item in obj.values() {
            if let Some(value) = phase51_lighter_find_i64(item, candidate_keys) {
                return Some(value);
            }
        }
    } else if let Some(items) = value.as_array() {
        for item in items {
            if let Some(value) = phase51_lighter_find_i64(item, candidate_keys) {
                return Some(value);
            }
        }
    }
    None
}

fn phase51_lighter_body_pressure_value(
    body: Option<&serde_json::Value>,
    field: &'static str,
) -> Option<i64> {
    let body = body?;
    match field {
        "active_order_headroom_account" => phase51_lighter_find_i64(
            body,
            &[
                "activeorderheadroomaccount",
                "activeordersheadroomaccount",
                "activeorderheadroom",
                "activeordersheadroom",
            ],
        ),
        "sendtx_per_minute_limit" => phase51_lighter_find_i64(
            body,
            &[
                "sendtxperminutelimit",
                "sendtxperminute",
                "sendtxlimit",
                "sendtxbatchlimit",
                "sendtxsendtxbatchperminutelimit",
            ],
        ),
        "sendtx_per_minute_remaining" => phase51_lighter_find_i64(
            body,
            &[
                "sendtxperminuteremaining",
                "sendtxremaining",
                "sendtxbatchremaining",
                "sendtxsendtxbatchperminuteremaining",
            ],
        ),
        "rest_requests_per_minute_limit" => phase51_lighter_find_i64(
            body,
            &[
                "restrequestsperminutelimit",
                "restrequestslimit",
                "standardrequestsperminute",
                "standardrequestsperminutelimit",
                "restlimit",
            ],
        ),
        "rest_requests_per_minute_remaining" => phase51_lighter_find_i64(
            body,
            &[
                "restrequestsperminuteremaining",
                "restrequestsremaining",
                "standardrequestsperminuteremaining",
                "restremaining",
            ],
        ),
        "weighted_requests_per_minute_limit" => phase51_lighter_find_i64(
            body,
            &[
                "weightedrequestsperminutelimit",
                "premiumweightedrequests",
                "premiumweightedrequestsperminute",
                "premiumweightedrequestsperminutelimit",
                "weightedlimit",
            ],
        ),
        "weighted_requests_per_minute_remaining" => phase51_lighter_find_i64(
            body,
            &[
                "weightedrequestsperminuteremaining",
                "premiumweightedrequestsremaining",
                "premiumweightedrequestsperminuteremaining",
                "weightedremaining",
            ],
        ),
        _ => None,
    }
}

fn phase51_lighter_header_i64(value: &reqwest::header::HeaderValue) -> Option<i64> {
    value.to_str().ok()?.trim().parse::<i64>().ok()
}

fn phase51_lighter_header_pressure_field(normalized: &str) -> Option<&'static str> {
    let has_limit = normalized.contains("limit") && !normalized.contains("remaining");
    let has_remaining = normalized.contains("remaining");
    if !(has_limit || has_remaining) {
        return None;
    }
    if normalized.contains("sendtx") || normalized.contains("sendtxbatch") {
        return if has_remaining {
            Some("sendtx_per_minute_remaining")
        } else {
            Some("sendtx_per_minute_limit")
        };
    }
    if normalized.contains("weighted") || normalized.contains("weight") {
        return if has_remaining {
            Some("weighted_requests_per_minute_remaining")
        } else {
            Some("weighted_requests_per_minute_limit")
        };
    }
    if normalized.contains("rest") {
        return if has_remaining {
            Some("rest_requests_per_minute_remaining")
        } else {
            Some("rest_requests_per_minute_limit")
        };
    }
    if normalized.contains("active")
        && normalized.contains("order")
        && (normalized.contains("headroom") || normalized.contains("remaining"))
    {
        return Some("active_order_headroom_account");
    }
    None
}

fn phase51_lighter_build_passive_pressure_observation(
    source_endpoint_family: &str,
    http_status_success: bool,
    headers: &reqwest::header::HeaderMap,
    body: Option<&serde_json::Value>,
) -> Phase51LighterPassivePressureObservation {
    const FIELDS: [&str; 7] = [
        "active_order_headroom_account",
        "sendtx_per_minute_limit",
        "sendtx_per_minute_remaining",
        "rest_requests_per_minute_limit",
        "rest_requests_per_minute_remaining",
        "weighted_requests_per_minute_limit",
        "weighted_requests_per_minute_remaining",
    ];

    let mut values: BTreeMap<&'static str, Option<i64>> =
        FIELDS.into_iter().map(|field| (field, None)).collect();
    let mut sources: BTreeMap<&'static str, &'static str> = BTreeMap::new();
    for field in FIELDS {
        if let Some(value) = phase51_lighter_body_pressure_value(body, field) {
            values.insert(field, Some(value));
            sources.insert(field, "response_body_exact_field");
        }
    }

    let mut sanitized_header_names = Vec::new();
    let mut generic_rate_limit_header_present = false;
    for (name, value) in headers {
        let raw_name = name.as_str();
        let normalized = phase51_lighter_normalized_key(raw_name);
        if normalized.contains("auth")
            || normalized.contains("cookie")
            || normalized.contains("credential")
            || normalized.contains("jwt")
            || normalized.contains("password")
            || normalized.contains("secret")
            || normalized.contains("session")
            || normalized.contains("signature")
            || normalized.contains("token")
        {
            continue;
        }
        if (normalized.contains("rate")
            || normalized.contains("limit")
            || normalized.contains("remaining"))
            && phase51_lighter_header_pressure_field(&normalized).is_none()
        {
            generic_rate_limit_header_present = true;
        }
        let Some(field) = phase51_lighter_header_pressure_field(&normalized) else {
            continue;
        };
        sanitized_header_names.push(raw_name.to_string());
        if values.get(field).and_then(|value| *value).is_some() {
            continue;
        }
        if let Some(numeric) = phase51_lighter_header_i64(value) {
            values.insert(field, Some(numeric));
            sources.insert(field, "response_header_explicit_pressure_field");
        }
    }
    sanitized_header_names.sort();
    sanitized_header_names.dedup();

    let has_active = values
        .get("active_order_headroom_account")
        .and_then(|value| *value)
        .is_some();
    let has_sendtx = values
        .get("sendtx_per_minute_limit")
        .and_then(|value| *value)
        .is_some()
        && values
            .get("sendtx_per_minute_remaining")
            .and_then(|value| *value)
            .is_some();
    let has_rest = values
        .get("rest_requests_per_minute_limit")
        .and_then(|value| *value)
        .is_some()
        && values
            .get("rest_requests_per_minute_remaining")
            .and_then(|value| *value)
            .is_some();
    let has_weighted = values
        .get("weighted_requests_per_minute_limit")
        .and_then(|value| *value)
        .is_some()
        && values
            .get("weighted_requests_per_minute_remaining")
            .and_then(|value| *value)
            .is_some();
    let complete = has_active && has_sendtx && (has_rest || has_weighted);
    let mut missing = Vec::new();
    if !has_active {
        missing.push("active_order_headroom_account");
    }
    if !has_sendtx {
        missing.push("sendtx_per_minute_limit/sendtx_per_minute_remaining");
    }
    if !has_rest && !has_weighted {
        missing.push("rest_requests_per_minute_pair_or_weighted_requests_per_minute_pair");
    }

    Phase51LighterPassivePressureObservation {
        schema_version: 1,
        producer: "paraphina_lighter_passive_sendtx_pressure_tap",
        venue_id: "lighter",
        source_endpoint_family: source_endpoint_family.to_string(),
        observed_at_ms: now_ms(),
        http_status_success,
        no_live_flag: true,
        approved_for_live: false,
        approved_for_canary: false,
        approved_for_capital_escalation: false,
        live_orders_allowed: false,
        capital_change_allowed: false,
        risk_limit_relaxation_allowed: false,
        phase51_validators_run: false,
        blocker_cleared: false,
        raw_request_persisted: false,
        raw_response_body_persisted: false,
        unsanitized_headers_persisted: false,
        secret_material_persisted: false,
        native_limit_event_time_status: "PASSIVE_SENDTX_RESPONSE_OBSERVED",
        pressure_dimensions_complete_from_passive_response: complete,
        required_missing_dimensions: missing,
        active_order_headroom_account: values
            .get("active_order_headroom_account")
            .and_then(|value| *value),
        sendtx_per_minute_limit: values
            .get("sendtx_per_minute_limit")
            .and_then(|value| *value),
        sendtx_per_minute_remaining: values
            .get("sendtx_per_minute_remaining")
            .and_then(|value| *value),
        rest_requests_per_minute_limit: values
            .get("rest_requests_per_minute_limit")
            .and_then(|value| *value),
        rest_requests_per_minute_remaining: values
            .get("rest_requests_per_minute_remaining")
            .and_then(|value| *value),
        weighted_requests_per_minute_limit: values
            .get("weighted_requests_per_minute_limit")
            .and_then(|value| *value),
        weighted_requests_per_minute_remaining: values
            .get("weighted_requests_per_minute_remaining")
            .and_then(|value| *value),
        pressure_field_sources: sources,
        sanitized_pressure_header_names: sanitized_header_names,
        generic_rate_limit_header_present,
    }
}

fn phase51_lighter_maybe_emit_passive_pressure_observation(
    source_endpoint_family: &str,
    http_status_success: bool,
    headers: &reqwest::header::HeaderMap,
    body: Option<&serde_json::Value>,
) -> bool {
    let cfg = Phase51LighterPassivePressureTapConfig::from_env();
    if !cfg.enabled {
        return false;
    }
    let row_index = LIGHTER_PHASE51_PASSIVE_PRESSURE_TAP_ROWS.fetch_add(1, Ordering::Relaxed);
    if row_index >= cfg.max_rows {
        return false;
    }
    if cfg.output_path.is_symlink() {
        return false;
    }
    if let Some(parent) = cfg.output_path.parent() {
        if parent.is_symlink() || std::fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    let observation = phase51_lighter_build_passive_pressure_observation(
        source_endpoint_family,
        http_status_success,
        headers,
        body,
    );
    let Ok(encoded) = serde_json::to_string(&observation) else {
        return false;
    };
    let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.output_path)
    else {
        return false;
    };
    writeln!(file, "{encoded}").is_ok()
}

#[cfg(test)]
fn phase51_lighter_passive_pressure_tap_reset_for_tests() {
    LIGHTER_PHASE51_PASSIVE_PRESSURE_TAP_ROWS.store(0, Ordering::Relaxed);
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn now_timestamp_ms_nonzero() -> TimestampMs {
    now_ms().max(1) as TimestampMs
}

fn lighter_ws_audit_enabled() -> bool {
    *LIGHTER_WS_AUDIT_ENABLED.get_or_init(|| {
        std::env::var("PARAPHINA_WS_AUDIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn lighter_audit_reconnect(reason: &'static str) {
    if !lighter_ws_audit_enabled() {
        return;
    }
    let mut counts = LIGHTER_RECONNECT_COUNTS
        .get_or_init(|| StdMutex::new(BTreeMap::new()))
        .lock()
        .expect("lighter reconnect audit mutex poisoned");
    let count = counts
        .entry(reason)
        .and_modify(|value| *value += 1)
        .or_insert(1);
    eprintln!(
        "WS_AUDIT venue=lighter reconnect_reason={} count={}",
        reason, *count
    );
}

fn lighter_audit_seq_gap(last_nonce: Option<u64>, begin_nonce: Option<u64>, nonce: Option<u64>) {
    if !lighter_ws_audit_enabled() {
        return;
    }
    let fmt = |value: Option<u64>| {
        value
            .map(|v| v.to_string())
            .unwrap_or_else(|| "na".to_string())
    };
    eprintln!(
        "WS_AUDIT venue=lighter reconnect_reason=seq_gap last_nonce={} begin_nonce={} nonce={}",
        fmt(last_nonce),
        fmt(begin_nonce),
        fmt(nonce),
    );
}

fn lighter_audit_ticker_backstop(
    action: &str,
    source_kind: &str,
    top: &TopOfBook,
    venue_nonce: Option<u64>,
    order_book_age_ms: u64,
    threshold_ms: u64,
) {
    if !lighter_ws_audit_enabled() {
        return;
    }
    let nonce = venue_nonce
        .map(|value| value.to_string())
        .unwrap_or_else(|| "na".to_string());
    eprintln!(
        "WS_AUDIT venue=lighter component=ticker_backstop action={} source_kind={} nonce={} bid_px={} bid_sz={} ask_px={} ask_sz={} order_book_age_ms={} threshold_ms={}",
        action,
        source_kind,
        nonce,
        top.best_bid_px,
        top.best_bid_sz,
        top.best_ask_px,
        top.best_ask_sz,
        order_book_age_ms,
        threshold_ms,
    );
}

fn lighter_should_skip_reconnect_sleep(
    session_duration: Duration,
    healthy_threshold: Duration,
) -> bool {
    session_duration >= healthy_threshold
}

fn decode_market_timestamp_ms(value: &serde_json::Value, context: &str) -> TimestampMs {
    let raw = value
        .get("timestamp")
        .or_else(|| value.get("ts"))
        .and_then(|v| v.as_i64());
    if let Some(ts) = raw {
        if ts > 0 {
            return ts;
        }
    }
    let fallback = now_timestamp_ms_nonzero();
    if lighter_ws_audit_enabled() {
        let fallback_count = LIGHTER_TS_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if fallback_count <= 3 || fallback_count % 100 == 0 {
            let raw_value = raw
                .map(|v| v.to_string())
                .unwrap_or_else(|| "missing".to_string());
            eprintln!(
                "WS_AUDIT venue=lighter lighter_ts_fallback_count={} context={} raw_ts={} fallback_ts_ms={}",
                fallback_count, context, raw_value, fallback
            );
        }
    }
    fallback
}

fn scale_to_i64(value: f64, decimals: u32, label: &str) -> anyhow::Result<i64> {
    if !value.is_finite() {
        anyhow::bail!("lighter: non-finite {label}");
    }
    if decimals > 18 {
        anyhow::bail!("lighter: unsupported {label} decimals={decimals}");
    }
    let factor = 10_f64.powi(decimals as i32);
    let scaled = (value * factor).round();
    if scaled < 0.0 || scaled > i64::MAX as f64 {
        anyhow::bail!("lighter: {label} out of range");
    }
    Ok(scaled as i64)
}

pub struct LighterConnector {
    cfg: LighterConfig,
    http: Client,
    market_publisher: MarketPublisher,
    exec_tx: mpsc::Sender<ExecutionEvent>,
    account_tx: Option<mpsc::Sender<AccountEvent>>,
    nonce: Arc<LighterNonceManager>,
    nonce_path: Option<PathBuf>,
    signer: Option<LighterSignerClient>,
    freshness: Arc<Freshness>,
    market_seq: Arc<AtomicU64>,
    resolved_market: Arc<Mutex<Option<LighterResolvedMarket>>>,
}

impl LighterConnector {
    pub fn new(
        cfg: LighterConfig,
        market_tx: mpsc::Sender<MarketDataEvent>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
    ) -> Self {
        let is_fixture = std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some();
        let queue_cap = if is_fixture {
            LIGHTER_MARKET_PUB_QUEUE_CAP
        } else {
            live_market_pub_queue_cap(LIGHTER_MARKET_PUB_QUEUE_CAP)
        };
        let drain_max = live_market_pub_drain_max(LIGHTER_MARKET_PUB_DRAIN_MAX);
        let market_publisher = MarketPublisher::new(
            queue_cap,
            drain_max,
            "lighter",
            market_tx.clone(),
            Some(Arc::new(move || is_fixture)),
            Arc::new(|event: &MarketDataEvent| {
                matches!(
                    event,
                    MarketDataEvent::L2Delta(_) | MarketDataEvent::L2Snapshot(_)
                )
            }),
            None,
            "lighter market_tx closed",
            "lighter market publish queue closed",
        );
        let nonce_path = cfg.nonce_path.clone();
        let nonce = if let Some(path) = nonce_path.as_ref() {
            match load_last_nonce(path) {
                Ok(Some(last)) => Arc::new(LighterNonceManager::new(Some(last))),
                Ok(None) => Arc::new(LighterNonceManager::new(None)),
                Err(err) => {
                    eprintln!(
                        "Lighter nonce load failed path={} err={}",
                        path.display(),
                        err
                    );
                    Arc::new(LighterNonceManager::new(None))
                }
            }
        } else {
            Arc::new(LighterNonceManager::new(None))
        };
        let signer = cfg
            .signer_url
            .as_ref()
            .map(|url| LighterSignerClient::new(url.clone()));
        let freshness = Arc::new(Freshness::default());
        let market_seq = Arc::new(AtomicU64::new(0));
        let resolved_market = Arc::new(Mutex::new(None));
        Self {
            cfg,
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("lighter http client build"),
            market_publisher,
            exec_tx,
            account_tx: None,
            nonce,
            nonce_path,
            signer,
            freshness,
            market_seq,
            resolved_market,
        }
    }

    pub fn with_account_tx(mut self, account_tx: mpsc::Sender<AccountEvent>) -> Self {
        self.account_tx = Some(account_tx);
        self
    }

    pub fn has_auth(&self) -> bool {
        self.cfg.has_auth()
    }

    pub fn has_signer(&self) -> bool {
        self.cfg.has_signer()
    }

    fn next_nonce(&self) -> u64 {
        let now_ms = now_ms();
        let nonce = self.nonce.next(now_ms);
        if let Some(path) = self.nonce_path.as_ref() {
            if let Err(err) = store_last_nonce(path, nonce) {
                eprintln!(
                    "Lighter nonce persist failed path={} err={}",
                    path.display(),
                    err
                );
            }
        }
        nonce
    }

    fn next_market_seq(&self) -> u64 {
        self.market_seq
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
    }

    async fn resolve_market_decimals(&self, market_id: u64) -> anyhow::Result<(u32, u32)> {
        let info = self.resolve_market_info().await?;
        if info.market_id != market_id {
            anyhow::bail!("Lighter market_id not found for decimals: {}", market_id);
        }
        let price = info.price_decimals.ok_or_else(|| {
            phase51_lighter_market_decimals_missing_error("price", market_id, &info.source)
        })?;
        let size = info.size_decimals.ok_or_else(|| {
            phase51_lighter_market_decimals_missing_error("size", market_id, &info.source)
        })?;
        Ok((price, size))
    }

    async fn submit_sendtx(
        &self,
        signed: SignedTx,
    ) -> super::super::gateway::LiveResult<LiveRestResponse> {
        let tx_info = serde_json::to_string(&signed.tx_info)
            .map_err(|err| LiveGatewayError::retryable(format!("sendtx_serialize_error: {err}")))?;
        let resp = self
            .http
            .post(sendtx_url(&self.cfg))
            .form(&[
                ("tx_type", signed.tx_type.to_string()),
                ("tx_info", tx_info),
            ])
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("sendtx_error: {err}")))?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = resp.text().await.unwrap_or_default();
        let passive_body = serde_json::from_str::<serde_json::Value>(&body).ok();
        phase51_lighter_maybe_emit_passive_pressure_observation(
            "sendTx",
            status.is_success(),
            &headers,
            passive_body.as_ref(),
        );
        if !status.is_success() {
            return Err(map_rest_error(&body));
        }
        let value = match passive_body {
            Some(value) => value,
            None => serde_json::from_str::<serde_json::Value>(&body)
                .map_err(|err| LiveGatewayError::retryable(format!("sendtx_parse_error: {err}")))?,
        };
        let order_id = value
            .get("order_id")
            .or_else(|| value.get("orderId"))
            .and_then(|v| v.as_str())
            .map(|v| v.to_string());
        Ok(LiveRestResponse {
            order_id,
            client_order_id: None,
        })
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        self.market_publisher.publish_market(event).await
    }

    async fn resolve_market_id_and_symbol(&self) -> anyhow::Result<(String, u64)> {
        let info = self.resolve_market_info().await?;
        Ok((info.symbol, info.market_id))
    }

    async fn resolve_market_info(&self) -> anyhow::Result<LighterResolvedMarket> {
        if let Some(info) = self.resolved_market.lock().await.clone() {
            return Ok(info);
        }

        let market_id_env = std::env::var("LIGHTER_MARKET_ID")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        let market_symbol_env = std::env::var("LIGHTER_MARKET").ok();
        let (orderbooks, source_url) = fetch_lighter_orderbooks_with_fallbacks(
            &self.http,
            &self.cfg.rest_url,
            &self.cfg.ws_url,
        )
        .await?;

        let resolved = if let Some(market_id) = market_id_env {
            let matched = orderbooks
                .iter()
                .find(|info| info.market_id == market_id)
                .cloned();
            let symbol = market_symbol_env
                .clone()
                .or_else(|| matched.as_ref().map(|info| info.symbol.clone()))
                .unwrap_or_else(|| self.cfg.market.clone());
            if phase51_lighter_strict_maker_only_observation_enabled() {
                eprintln!(
                    "INFO: Lighter resolving market id source_label={} strict_maker_only_observation=true",
                    phase51_lighter_orderbooks_source_label("env:LIGHTER_MARKET_ID")
                );
                eprintln!(
                    "INFO: Lighter market id resolved source_label={} strict_maker_only_observation=true",
                    phase51_lighter_orderbooks_source_label("env:LIGHTER_MARKET_ID")
                );
            } else {
                eprintln!(
                    "INFO: Lighter resolving market id symbol={} source_url=env:LIGHTER_MARKET_ID",
                    symbol
                );
                eprintln!(
                    "INFO: Lighter market id resolved symbol={} market_id={} source_url=env:LIGHTER_MARKET_ID",
                    symbol, market_id
                );
            }
            let resolved = LighterResolvedMarket {
                symbol,
                market_id,
                price_decimals: matched.as_ref().and_then(|info| info.price_decimals),
                size_decimals: matched.as_ref().and_then(|info| info.size_decimals),
                source: "env:LIGHTER_MARKET_ID".to_string(),
            };
            if matched.is_some() {
                let mut cache = self.resolved_market.lock().await;
                *cache = Some(resolved.clone());
            }
            resolved
        } else {
            let symbol = market_symbol_env.unwrap_or_else(|| self.cfg.market.clone());
            if phase51_lighter_strict_maker_only_observation_enabled() {
                eprintln!(
                    "INFO: Lighter resolving market id source_label={} strict_maker_only_observation=true",
                    phase51_lighter_orderbooks_source_label(&source_url)
                );
            } else {
                eprintln!(
                    "INFO: Lighter resolving market id symbol={} source_url={}",
                    symbol, source_url
                );
            }
            let normalized = normalize_lighter_symbol(&symbol);
            let found = orderbooks
                .iter()
                .find(|info| normalize_lighter_symbol(&info.symbol) == normalized)
                .cloned();
            let Some(info) = found else {
                if phase51_lighter_strict_maker_only_observation_enabled() {
                    eprintln!(
                        "WARN: Lighter market id not found reason=market_symbol_unmatched source_label={} strict_maker_only_observation=true",
                        phase51_lighter_orderbooks_source_label(&source_url)
                    );
                } else {
                    let available: Vec<String> = orderbooks
                        .iter()
                        .take(15)
                        .map(|info| info.symbol.clone())
                        .collect();
                    eprintln!(
                        "WARN: Lighter market id not found requested={} available_symbols={:?}",
                        normalized, available
                    );
                }
                anyhow::bail!(
                    "LIGHTER_MARKET not found in orderBooks response: {}",
                    symbol
                );
            };
            if phase51_lighter_strict_maker_only_observation_enabled() {
                eprintln!(
                    "INFO: Lighter market id resolved source_label={} strict_maker_only_observation=true",
                    phase51_lighter_orderbooks_source_label(&source_url)
                );
            } else {
                eprintln!(
                    "INFO: Lighter market id resolved symbol={} market_id={} source_url={}",
                    info.symbol, info.market_id, source_url
                );
            }
            let resolved = LighterResolvedMarket {
                symbol: info.symbol,
                market_id: info.market_id,
                price_decimals: info.price_decimals,
                size_decimals: info.size_decimals,
                source: source_url,
            };
            let mut cache = self.resolved_market.lock().await;
            *cache = Some(resolved.clone());
            resolved
        };

        Ok(resolved)
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let mut consecutive_failures: u32 = 0;
        let mut subscribe_failures = 0usize;
        let mut logged_subscribe_failure = false;

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = Instant::now();

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
                    let msg = err.to_string();
                    if msg.contains("Lighter subscribe failed") {
                        lighter_audit_reconnect("subscribe_error");
                        subscribe_failures += 1;
                        if subscribe_failures >= 3 && !logged_subscribe_failure {
                            eprintln!(
                                "WARN: Lighter subscribe failed {} times; backing off",
                                subscribe_failures
                            );
                            logged_subscribe_failure = true;
                        }
                    } else {
                        subscribe_failures = 0;
                        logged_subscribe_failure = false;
                        let level = if consecutive_failures >= 20 {
                            "ERROR"
                        } else if consecutive_failures >= 5 {
                            "WARN"
                        } else {
                            "INFO"
                        };
                        eprintln!(
                            "{level}: Lighter public WS error (consecutive_failures={consecutive_failures}): {err}"
                        );
                    }
                }
                Err(_timeout) => {
                    lighter_audit_reconnect("session_timeout");
                    eprintln!(
                        "ERROR: Lighter public WS session timeout ({}s) — force reconnect",
                        max_session.as_secs()
                    );
                    consecutive_failures += 1;
                }
                Ok(Ok(())) => {}
            }

            // FIX: Reset backoff and failure counter if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            let skip_reconnect_sleep =
                lighter_should_skip_reconnect_sleep(session_duration, healthy_threshold);
            if session_duration >= healthy_threshold {
                if consecutive_failures > 0 {
                    eprintln!(
                        "INFO: Lighter WS session was healthy for {:?}; \
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

            if skip_reconnect_sleep {
                continue;
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        let connect_timeout = lighter_ws_connect_timeout();
        let read_timeout = lighter_ws_read_timeout();
        let ping_interval_ms = lighter_ping_interval_ms();
        let ticker_backstop_enabled = lighter_ticker_backstop_enabled();
        let ticker_backstop_after_ms = lighter_ticker_backstop_after_ms();
        let readonly = lighter_ws_readonly_enabled();
        let ws_url = lighter_public_ws_url(&self.cfg.ws_url, readonly);

        let (market_symbol, market_id) = self.resolve_market_id_and_symbol().await?;

        // FIX: Setup freshness watchdog (mirrors pattern in HL/Paradex/Extended/Aster)
        let connect_start_ns = mono_now_ns();
        self.freshness.reset_for_new_connection();
        let stale_ms = lighter_stale_ms();

        let fixture_mode = std::env::var_os("LIGHTER_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some();

        // Spawn watchdog task that signals when connection is stale
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        if !fixture_mode {
            let watchdog_freshness = self.freshness.clone();
            tokio::spawn(async move {
                let mut iv = tokio::time::interval(Duration::from_millis(LIGHTER_WATCHDOG_TICK_MS));
                iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    iv.tick().await;
                    let now = mono_now_ns();
                    let anchor = watchdog_freshness.anchor_with_connect_start(connect_start_ns);
                    if anchor != 0 && age_ms(now, anchor) > stale_ms {
                        lighter_audit_freshness(
                            "stale_watchdog_trigger",
                            watchdog_freshness.as_ref(),
                            connect_start_ns,
                            stale_ms,
                        );
                        let _ = stale_tx.send(());
                        break;
                    }
                }
            });
        }

        eprintln!(
            "INFO: Lighter public WS connecting readonly={} url={}",
            readonly, ws_url
        );
        let (ws_stream, _) = with_timeout(
            connect_timeout,
            "public WS connect",
            connect_async(ws_url.as_str()),
        )
        .await?
        .map_err(|e| anyhow::anyhow!("Lighter public WS connect error: {e}"))?;
        eprintln!("INFO: Lighter public WS connected url={}", ws_url);
        let (mut write, mut read) = ws_stream.split();
        let channel = build_order_book_channel(market_id);
        let sub = json!({
            "type": "subscribe",
            "channel": channel,
        });
        write.send(Message::Text(sub.to_string())).await?;
        if ticker_backstop_enabled {
            let ticker_sub = json!({
                "type": "subscribe",
                "channel": build_ticker_channel(market_id),
            });
            write.send(Message::Text(ticker_sub.to_string())).await?;
        }
        let mut subscribed = false;
        let mut first_message_logs = 0usize;
        let mut fallback_tracker = LighterSeqTracker::new();
        let mut last_book_nonce: Option<u64> = None;
        let mut last_ticker_backstop_applied_nonce: Option<u64> = None;
        let mut first_book_update_logged = false;
        let mut first_ticker_logged = false;
        let mut first_message_logged = false;
        let mut logged_non_utf8_binary = false;
        let mut first_decoded_top_logged = false;
        let mut first_json_pong_logged = false;
        let mut decode_miss_count = 0usize;
        // Tracks whether the initial full snapshot has been applied to the L2 book.
        // After this, all subsequent order_book messages are applied as L2Delta (not L2Snapshot),
        // because Lighter sends only state-changes after the subscription snapshot.
        // See: https://apidocs.lighter.xyz/docs/websocket-reference#order-book
        let mut initial_snapshot_applied = false;
        let mut consecutive_delta_failures: usize = 0;
        let mut seq_fallback: u64 = 0;
        // Rate-limited logging for decode failures
        let mut last_decode_fail_warn_ns: u64 = 0;
        let ping_enabled = ping_interval_ms > 0;
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms.max(1)));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        if ping_enabled {
            ping_timer.tick().await;
        }
        loop {
            let msg = tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    lighter_audit_reconnect("stale_watchdog");
                    anyhow::bail!("Lighter public WS stale: freshness exceeded {stale_ms}ms");
                }
                _ = ping_timer.tick(), if ping_enabled => {
                    match write.send(Message::Ping(vec![b'p'].into())).await {
                        Ok(()) => {
                            if lighter_ws_audit_enabled() {
                                let sent = LIGHTER_PING_SENT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                                if sent <= 3 || sent % 100 == 0 {
                                    eprintln!(
                                        "WS_AUDIT venue=lighter lighter_ping_sent_count={} interval_ms={}",
                                        sent, ping_interval_ms
                                    );
                                }
                            }
                        }
                        Err(err) => {
                            lighter_audit_reconnect("ping_send_fail");
                            if lighter_ws_audit_enabled() {
                                let fail = LIGHTER_PING_SEND_FAIL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                                eprintln!(
                                    "WS_AUDIT venue=lighter lighter_ping_send_fail_count={} err={}",
                                    fail, err
                                );
                            }
                            anyhow::bail!("Lighter public WS ping send failed: {err}");
                        }
                    }
                    continue;
                }
                read_result = tokio::time::timeout(read_timeout, read.next()) => {
                    let maybe = match read_result {
                        Ok(m) => m,
                        Err(_) => {
                            lighter_audit_freshness(
                                "read_timeout",
                                self.freshness.as_ref(),
                                connect_start_ns,
                                stale_ms,
                            );
                            lighter_audit_reconnect("read_timeout");
                            eprintln!(
                                "WARN: Lighter public WS read timeout ({read_timeout:?}) — no frame received, reconnecting"
                            );
                            anyhow::bail!("Lighter public WS read timeout after {read_timeout:?}");
                        }
                    };
                    match maybe {
                        Some(Ok(msg)) => msg,
                        Some(Err(err)) => {
                            eprintln!("Lighter public WS read error: {err}");
                            break;
                        }
                        None => {
                            eprintln!("Lighter public WS stream ended");
                            break;
                        }
                    }
                }
            };

            // Update WS receive timestamp
            self.freshness
                .last_ws_rx_ns
                .store(mono_now_ns(), Ordering::Relaxed);

            let payload = match msg {
                Message::Ping(payload) => {
                    let _ = write.send(Message::Pong(payload)).await;
                    continue;
                }
                Message::Pong(_) => continue,
                Message::Close(frame) => {
                    eprintln!("Lighter public WS closed frame={frame:?}");
                    break;
                }
                Message::Text(text) => text,
                Message::Binary(bytes) => match String::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(_) => {
                        if !logged_non_utf8_binary {
                            eprintln!(
                                "WARN: Lighter public WS non-utf8 binary frame url={}",
                                self.cfg.ws_url
                            );
                            logged_non_utf8_binary = true;
                        }
                        continue;
                    }
                },
                _ => continue,
            };

            // Update freshness on data message
            self.freshness
                .last_data_rx_ns
                .store(mono_now_ns(), Ordering::Relaxed);

            if !first_message_logged {
                eprintln!("INFO: Lighter public WS first message received");
                first_message_logged = true;
            }
            let value = match serde_json::from_str::<serde_json::Value>(&payload) {
                Ok(value) => value,
                Err(err) => {
                    let snippet: String = payload.chars().take(160).collect();
                    eprintln!(
                        "WARN: Lighter public WS parse error: {err} url={} snippet={}",
                        self.cfg.ws_url, snippet
                    );
                    continue;
                }
            };
            if first_message_logs < 2 {
                let keys = value
                    .as_object()
                    .map(|obj| {
                        let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                        keys.sort();
                        format!("[{}]", keys.join(","))
                    })
                    .unwrap_or_else(|| "[non-object]".to_string());
                let snippet: String = payload.chars().take(160).collect();
                eprintln!("INFO: Lighter WS first msg keys={keys} snippet={snippet}");
                first_message_logs += 1;
            }
            if let Some(pong) = json_ping_response(&value) {
                if !first_json_pong_logged {
                    eprintln!("INFO: Lighter sent JSON pong");
                    first_json_pong_logged = true;
                }
                let _ = write.send(Message::Text(pong.to_string())).await;
                continue;
            }
            if !subscribed {
                if let Some(code) = lighter_error_code(&value) {
                    if code == 30005 {
                        lighter_audit_reconnect("subscribe_error");
                        anyhow::bail!("Lighter subscribe failed: invalid channel");
                    }
                }
            }
            if let Some(ticker) = decode_ticker_channel_message(&value) {
                let now_ns = mono_now_ns();
                self.freshness
                    .last_ticker_ns
                    .store(now_ns, Ordering::Relaxed);
                let order_book_age_ms = age_ms_or_connect_age(
                    now_ns,
                    self.freshness.last_book_event_ns.load(Ordering::Relaxed),
                    connect_start_ns,
                );
                if !first_ticker_logged {
                    lighter_audit_ticker_backstop(
                        "first_ticker",
                        "ticker",
                        &ticker.top,
                        ticker.venue_nonce,
                        order_book_age_ms,
                        ticker_backstop_after_ms,
                    );
                    first_ticker_logged = true;
                }
                if lighter_should_apply_ticker_backstop(
                    initial_snapshot_applied,
                    ticker_backstop_enabled,
                    order_book_age_ms,
                    ticker_backstop_after_ms,
                    ticker.venue_nonce,
                    last_ticker_backstop_applied_nonce,
                ) {
                    let event = build_ticker_backstop_event(
                        &ticker.top,
                        self.cfg.venue_index,
                        &self.cfg.venue_id,
                        self.next_market_seq(),
                        ticker.timestamp_ms,
                    );
                    if self.publish_market(event).await.is_err() {
                        lighter_audit_reconnect("ticker_backstop_publish_fail");
                        anyhow::bail!("Lighter ticker backstop publish failed");
                    }
                    self.freshness
                        .last_parsed_ns
                        .store(now_ns, Ordering::Relaxed);
                    self.freshness
                        .last_book_event_ns
                        .store(now_ns, Ordering::Relaxed);
                    self.freshness
                        .last_published_ns
                        .store(now_ns, Ordering::Relaxed);
                    last_ticker_backstop_applied_nonce = ticker.venue_nonce;
                    lighter_audit_ticker_backstop(
                        "applied",
                        "ticker",
                        &ticker.top,
                        ticker.venue_nonce,
                        order_book_age_ms,
                        ticker_backstop_after_ms,
                    );
                }
                continue;
            }
            if !subscribed && is_lighter_book_message(&value) {
                subscribed = true;
                eprintln!(
                    "INFO: Lighter subscribe ok channel=order_book/{} symbol={} market_id={}",
                    market_id, market_symbol, market_id
                );
            }
            if subscribed {
                // Update data receive timestamp when we get book-related messages
                self.freshness
                    .last_data_rx_ns
                    .store(mono_now_ns(), Ordering::Relaxed);

                if let Some(top) = decode_order_book_top(&value) {
                    if !first_decoded_top_logged {
                        eprintln!(
                            "FIRST_DECODED_TOP venue=lighter bid_px={} bid_sz={} ask_px={} ask_sz={}",
                            top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                        );
                        first_decoded_top_logged = true;
                    }
                } else if !initial_snapshot_applied
                    && decode_miss_count < 3
                    && has_lighter_book_fields(&value)
                {
                    // Startup decode miss: decode_order_book_top() fails on the
                    // snapshot-format message before initial_snapshot_applied is set.
                    // This is expected — the full L2 path handles it correctly.
                    // Log once at INFO (not WARN) for diagnostics.
                    decode_miss_count += 1;
                    if decode_miss_count == 1 {
                        eprintln!(
                            "INFO: Lighter startup: decode_order_book_top miss on snapshot-format \
                             message (expected, snapshot will be processed by L2 path)"
                        );
                    }
                }
                // === L2 decode: snapshot for first message, delta for subsequent ===
                //
                // Lighter sends a full snapshot on subscription, then only state-changes
                // (deltas) afterwards. Treating deltas as snapshots would replace the
                // entire book with partial data, causing intermittent zero-depth and
                // Disabled flapping.
                //
                // See: https://apidocs.lighter.xyz/docs/websocket-reference#order-book
                let decoded = if !initial_snapshot_applied {
                    // First book message: decode as full L2Snapshot.
                    decode_order_book_channel_message(
                        &value,
                        self.cfg.venue_index,
                        &self.cfg.venue_id,
                        &mut seq_fallback,
                    )
                    .or_else(|| {
                        decode_order_book_snapshot(
                            &value,
                            self.cfg.venue_index,
                            &self.cfg.venue_id,
                            &mut seq_fallback,
                        )
                        .map(|parsed| LighterBookMessage {
                            event: parsed.event,
                            seq: parsed.seq,
                            venue_nonce: lighter_order_book_nonce(&value),
                            venue_begin_nonce: lighter_order_book_begin_nonce(&value),
                        })
                    })
                } else {
                    // Subsequent messages: decode as L2Delta (state changes only).
                    decode_order_book_channel_delta(
                        &value,
                        self.cfg.venue_index,
                        &self.cfg.venue_id,
                        &mut seq_fallback,
                    )
                };

                if let Some(parsed) = decoded {
                    consecutive_delta_failures = 0;
                    if initial_snapshot_applied {
                        if lighter_has_continuity_gap(
                            last_book_nonce,
                            parsed.venue_begin_nonce,
                            parsed.venue_nonce,
                        ) {
                            lighter_audit_reconnect("seq_gap");
                            lighter_audit_seq_gap(
                                last_book_nonce,
                                parsed.venue_begin_nonce,
                                parsed.venue_nonce,
                            );
                            anyhow::bail!(
                                "Lighter order_book continuity gap last_nonce={:?} begin_nonce={:?} nonce={:?}",
                                last_book_nonce,
                                parsed.venue_begin_nonce,
                                parsed.venue_nonce
                            );
                        }
                    }
                    if let Some(nonce) = parsed.venue_nonce {
                        last_book_nonce = Some(nonce);
                    }
                    let timestamp_ms = market_event_timestamp_ms(&parsed.event);
                    let event =
                        override_market_event(parsed.event, self.next_market_seq(), timestamp_ms);
                    let publish_result = self.publish_market(event).await;
                    if publish_result.is_err() {
                        lighter_audit_reconnect(if initial_snapshot_applied {
                            "publish_fail"
                        } else {
                            "snapshot_publish_fail"
                        });
                        anyhow::bail!(
                            "Lighter public WS market publish failed while applying {}",
                            if initial_snapshot_applied {
                                "delta"
                            } else {
                                "snapshot"
                            }
                        );
                    }
                    let now_ns = mono_now_ns();
                    self.freshness
                        .last_parsed_ns
                        .store(now_ns, Ordering::Relaxed);
                    self.freshness
                        .last_book_event_ns
                        .store(now_ns, Ordering::Relaxed);
                    self.freshness
                        .last_published_ns
                        .store(now_ns, Ordering::Relaxed);
                    if !initial_snapshot_applied {
                        initial_snapshot_applied = true;
                        eprintln!(
                            "INFO: Lighter L2 initial snapshot applied, switching to delta mode"
                        );
                    }
                    if !first_book_update_logged {
                        eprintln!("INFO: Lighter public WS first book update");
                        first_book_update_logged = true;
                    }
                    continue;
                }

                // Decode failed — handle as bounded warning + delta-failure tracking
                if has_lighter_book_fields(&value) {
                    if initial_snapshot_applied {
                        consecutive_delta_failures += 1;
                        if consecutive_delta_failures >= LIGHTER_MAX_CONSECUTIVE_DELTA_FAILURES {
                            lighter_audit_reconnect("decode_fail_loop");
                            anyhow::bail!(
                                "Lighter: {} consecutive delta decode failures — \
                                 forcing reconnect for fresh snapshot",
                                consecutive_delta_failures
                            );
                        }
                    }
                    let now_ns = mono_now_ns();
                    if age_ms(now_ns, last_decode_fail_warn_ns) >= LIGHTER_DECODE_WARN_INTERVAL_MS {
                        last_decode_fail_warn_ns = now_ns;
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
                            "WARN: Lighter order_book decode failed (mode={}) keys={} snippet={}",
                            if initial_snapshot_applied {
                                "delta"
                            } else {
                                "snapshot"
                            },
                            keys,
                            snippet
                        );
                    }
                }
            }
            // Try parse_l2_message_value as fallback only for legacy/fixture payloads
            // that are not already handled by the live order_book channel path.
            if !has_lighter_book_fields(&value) {
                if let Some(parsed) =
                    parse_l2_message_value(&value, &self.cfg.venue_id, self.cfg.venue_index)
                {
                    let outcome = fallback_tracker.on_message(parsed);
                    if let Some(event) = outcome.event {
                        let timestamp_ms = market_event_timestamp_ms(&event);
                        let event =
                            override_market_event(event, self.next_market_seq(), timestamp_ms);
                        let now_ns = mono_now_ns();
                        self.freshness
                            .last_parsed_ns
                            .store(now_ns, Ordering::Relaxed);
                        if !first_book_update_logged {
                            eprintln!("INFO: Lighter public WS first book update");
                            first_book_update_logged = true;
                        }
                        if self.publish_market(event).await.is_ok() {
                            self.freshness
                                .last_book_event_ns
                                .store(now_ns, Ordering::Relaxed);
                            self.freshness
                                .last_published_ns
                                .store(now_ns, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn run_account_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut rate_limit_backoff_ms: u64 = 0;
        loop {
            interval.tick().await;
            if rate_limit_backoff_ms > 0 {
                tokio::time::sleep(Duration::from_millis(rate_limit_backoff_ms)).await;
            }
            let Some(account_tx) = self.account_tx.as_ref() else {
                continue;
            };
            match fetch_account_snapshot(&self.http, &self.cfg).await {
                Ok(snapshot) => {
                    rate_limit_backoff_ms = 0;
                    let _ = account_tx.send(snapshot).await;
                }
                Err(err) => {
                    let err_text = err.to_string();
                    if err_text.contains("http 429") {
                        rate_limit_backoff_ms = if rate_limit_backoff_ms == 0 {
                            interval_ms.max(LIGHTER_ACCOUNT_RATE_LIMIT_BASE_BACKOFF_MS)
                        } else {
                            (rate_limit_backoff_ms * 2)
                                .min(LIGHTER_ACCOUNT_RATE_LIMIT_MAX_BACKOFF_MS)
                        };
                    } else {
                        rate_limit_backoff_ms = 0;
                    }
                    maybe_log_lighter_account_error(&err_text, interval_ms, rate_limit_backoff_ms);
                }
            }
        }
    }

    pub async fn run_funding_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut seq: u64 = 0;

        // FIX: Retry initialization with exponential backoff instead of fatal exit
        let mut init_backoff = Duration::from_secs(5);
        const MAX_INIT_BACKOFF_SECS: u64 = 60;
        let (market_symbol, market_id) = loop {
            match self.resolve_market_id_and_symbol().await {
                Ok(val) => break val,
                Err(err) => {
                    eprintln!(
                        "Lighter funding polling init error (retry in {:?}): {err}",
                        init_backoff
                    );
                    tokio::time::sleep(init_backoff).await;
                    init_backoff =
                        (init_backoff * 2).min(Duration::from_secs(MAX_INIT_BACKOFF_SECS));
                    // Continue loop to retry
                }
            }
        };

        eprintln!(
            "INFO: Lighter funding polling initialized symbol={} market_id={}",
            market_symbol, market_id
        );

        loop {
            interval.tick().await;
            match fetch_public_funding(&self.http, &self.cfg, market_id, &market_symbol).await {
                Ok(mut update) => {
                    seq = seq.wrapping_add(1);
                    update.seq = seq;
                    // FIX: Log channel send failures instead of silently ignoring
                    if let Err(err) = self
                        .market_publisher
                        .publish_market(MarketDataEvent::FundingUpdate(update))
                        .await
                    {
                        eprintln!("Lighter funding publish error: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Lighter funding polling error: {err}");
                }
            }
        }
    }

    pub async fn run_account_fixture(
        &self,
        fixture_dir: &std::path::Path,
        start_ms: i64,
        step_ms: i64,
        ticks: u64,
    ) {
        let Some(account_tx) = self.account_tx.as_ref() else {
            return;
        };
        let snapshot_path = fixture_dir.join("rest_account_snapshot.json");
        let raw = match std::fs::read_to_string(snapshot_path) {
            Ok(val) => val,
            Err(_) => return,
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            return;
        };
        let Some(mut snapshot) = parse_account_snapshot(&value, &self.cfg.venue_id) else {
            return;
        };
        let mut seq: u64 = 1;
        for tick in 0..ticks {
            snapshot.seq = seq;
            snapshot.timestamp_ms = start_ms + step_ms.saturating_mul(tick as i64);
            seq = seq.wrapping_add(1);
            let _ = account_tx
                .send(AccountEvent::Snapshot(snapshot.clone()))
                .await;
            tokio::task::yield_now().await;
        }
    }

    pub async fn run_private_ws(&self) {
        let mut backoff = Duration::from_secs(1);

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = Instant::now();

            if let Err(err) = self.private_ws_once().await {
                eprintln!("Lighter private WS error: {err}");
            }

            // FIX: Reset backoff if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                eprintln!(
                    "INFO: Lighter private WS session was healthy for {:?}; resetting backoff",
                    session_duration
                );
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    pub async fn run_phase51_account_all_trades_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = Instant::now();

            if self.phase51_account_all_trades_ws_once().await.is_err() {
                eprintln!(
                    "WARN: Lighter Phase 5.1 account_all_trades WS reconnecting after sanitized transport error"
                );
            }

            if session_start.elapsed() >= healthy_threshold {
                eprintln!(
                    "INFO: Lighter Phase 5.1 account_all_trades WS session was healthy; resetting backoff"
                );
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn private_ws_once(&self) -> anyhow::Result<()> {
        let connect_timeout = lighter_ws_connect_timeout();
        let read_timeout = lighter_ws_read_timeout();

        let (ws_stream, _) = with_timeout(
            connect_timeout,
            "private WS connect",
            connect_async(self.cfg.ws_url.as_str()),
        )
        .await?
        .map_err(|e| anyhow::anyhow!("Lighter private WS connect error: {e}"))?;
        let (mut write, mut read) = ws_stream.split();
        let phase51_account_all_trades_enabled =
            phase51_lighter_account_all_trades_source_owner_enabled();
        let phase51_account_index = if phase51_account_all_trades_enabled {
            self.cfg.account_index
        } else {
            None
        };
        if let Some(account_index) = phase51_account_index {
            let sub = json!({
                "type": "subscribe",
                "channel": format!("account_all_trades/{account_index}"),
            });
            write.send(Message::Text(sub.to_string().into())).await?;
        }
        loop {
            let msg = match tokio::time::timeout(read_timeout, read.next()).await {
                Ok(Some(msg)) => msg?,
                Ok(None) => break,
                Err(_) => {
                    eprintln!(
                        "WARN: Lighter private WS read timeout ({read_timeout:?}) — reconnecting"
                    );
                    anyhow::bail!("Lighter private WS read timeout after {read_timeout:?}");
                }
            };
            if let Message::Text(text) = msg {
                for event in translate_private_events(
                    &text,
                    self.cfg.venue_index,
                    &self.cfg.venue_id,
                    phase51_account_index,
                ) {
                    let _ = self.exec_tx.send(event).await;
                }
            }
        }
        Ok(())
    }

    async fn phase51_account_all_trades_ws_once(&self) -> anyhow::Result<()> {
        let connect_timeout = lighter_ws_connect_timeout();
        let read_timeout = lighter_ws_read_timeout();
        let account_index = self.cfg.account_index.ok_or_else(|| {
            anyhow::anyhow!("missing Lighter account index for Phase 5.1 source-owner stream")
        })?;

        let (ws_stream, _) = with_timeout(
            connect_timeout,
            "Phase 5.1 account_all_trades WS connect",
            connect_async(self.cfg.ws_url.as_str()),
        )
        .await?
        .map_err(|_| anyhow::anyhow!("Lighter Phase 5.1 account_all_trades WS connect error"))?;
        let (mut write, mut read) = ws_stream.split();
        let sub = json!({
            "type": "subscribe",
            "channel": format!("account_all_trades/{account_index}"),
        });
        write
            .send(Message::Text(sub.to_string().into()))
            .await
            .map_err(|_| {
                anyhow::anyhow!("Lighter Phase 5.1 account_all_trades WS subscribe error")
            })?;

        loop {
            let msg = match tokio::time::timeout(read_timeout, read.next()).await {
                Ok(Some(msg)) => msg.map_err(|_| {
                    anyhow::anyhow!("Lighter Phase 5.1 account_all_trades WS read error")
                })?,
                Ok(None) => break,
                Err(_) => {
                    anyhow::bail!("Lighter Phase 5.1 account_all_trades WS read timeout");
                }
            };
            if let Message::Text(text) = msg {
                for event in translate_phase51_account_all_trades_source_owner_events(
                    &text,
                    self.cfg.venue_index,
                    &self.cfg.venue_id,
                    Some(account_index),
                ) {
                    let _ = self.exec_tx.send(event).await;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LighterFixtureFeed {
    messages: Vec<String>,
}

impl LighterFixtureFeed {
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
        start_ms: i64,
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
                if let Some(parsed) = parse_l2_message(raw, "LIGHTER", venue_index) {
                    let event = override_market_event(parsed.event, seq, now_ms);
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
pub struct ParsedL2Message {
    pub event: MarketDataEvent,
    pub seq: u64,
}

#[derive(Debug, Clone)]
struct LighterBookMessage {
    event: MarketDataEvent,
    seq: u64,
    venue_nonce: Option<u64>,
    venue_begin_nonce: Option<u64>,
}

#[derive(Debug, Clone)]
struct LighterTickerMessage {
    top: TopOfBook,
    timestamp_ms: TimestampMs,
    venue_nonce: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LighterSeqTracker {
    last_seq: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct LighterSeqOutcome {
    pub event: Option<MarketDataEvent>,
}

impl LighterSeqTracker {
    pub fn new() -> Self {
        Self { last_seq: None }
    }

    pub fn on_message(&mut self, msg: ParsedL2Message) -> LighterSeqOutcome {
        if let Some(prev) = self.last_seq {
            if msg.seq < prev {
                return LighterSeqOutcome { event: None };
            }
        }
        self.last_seq = Some(msg.seq);
        LighterSeqOutcome {
            event: Some(msg.event),
        }
    }
}

pub fn parse_l2_message(text: &str, venue_id: &str, venue_index: usize) -> Option<ParsedL2Message> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    parse_l2_message_value(&value, venue_id, venue_index)
}

fn parse_l2_message_value(
    value: &serde_json::Value,
    venue_id: &str,
    venue_index: usize,
) -> Option<ParsedL2Message> {
    let msg_type = value.get("type")?.as_str()?;
    let seq = value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = decode_market_timestamp_ms(value, "parse_l2_message_value");
    match msg_type {
        "l2_snapshot" => {
            let bids = parse_levels(value.get("bids")?)?;
            let asks = parse_levels(value.get("asks")?)?;
            let snapshot = super::super::types::L2Snapshot {
                venue_index,
                venue_id: venue_id.to_string(),
                seq,
                timestamp_ms,
                bids,
                asks,
            };
            Some(ParsedL2Message {
                event: MarketDataEvent::L2Snapshot(snapshot),
                seq,
            })
        }
        "l2_delta" => {
            let changes = parse_deltas(value.get("changes")?)?;
            let delta = super::super::types::L2Delta {
                venue_index,
                venue_id: venue_id.to_string(),
                seq,
                timestamp_ms,
                changes,
            };
            Some(ParsedL2Message {
                event: MarketDataEvent::L2Delta(delta),
                seq,
            })
        }
        _ => None,
    }
}

fn market_event_timestamp_ms(event: &MarketDataEvent) -> TimestampMs {
    match event {
        MarketDataEvent::L2Snapshot(snapshot) => snapshot.timestamp_ms,
        MarketDataEvent::L2Delta(delta) => delta.timestamp_ms,
        MarketDataEvent::Trade(trade) => trade.timestamp_ms,
        MarketDataEvent::FundingUpdate(funding) => funding.timestamp_ms,
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

fn build_ticker_backstop_event(
    top: &TopOfBook,
    venue_index: usize,
    venue_id: &str,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> MarketDataEvent {
    MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        bids: vec![BookLevel {
            price: top.best_bid_px,
            size: top.best_bid_sz,
        }],
        asks: vec![BookLevel {
            price: top.best_ask_px,
            size: top.best_ask_sz,
        }],
    })
}

fn parse_levels(levels: &serde_json::Value) -> Option<Vec<BookLevel>> {
    let mut out = Vec::new();
    for level in levels.as_array()? {
        let price = parse_f64_value(level.get(0)?)?;
        let size = parse_f64_value(level.get(1)?)?;
        out.push(BookLevel { price, size });
    }
    Some(out)
}

fn parse_level_from_object(value: &serde_json::Value) -> Option<BookLevel> {
    let obj = value.as_object()?;
    let price = obj.get("price").or_else(|| obj.get("px"))?;
    let size = obj.get("size").or_else(|| obj.get("sz"))?;
    Some(BookLevel {
        price: parse_f64_value(price)?,
        size: parse_f64_value(size)?,
    })
}

fn parse_deltas(changes: &serde_json::Value) -> Option<Vec<BookLevelDelta>> {
    let mut out = Vec::new();
    for change in changes.as_array()? {
        let side_raw = change.get("side")?.as_str()?;
        let side = match side_raw {
            "bid" => BookSide::Bid,
            "ask" => BookSide::Ask,
            _ => return None,
        };
        let price = parse_f64_value(change.get("price")?)?;
        let size = parse_f64_value(change.get("size")?)?;
        out.push(BookLevelDelta { side, price, size });
    }
    Some(out)
}

fn lighter_order_book_value(value: &serde_json::Value) -> Option<&serde_json::Value> {
    value
        .get("order_book")
        .or_else(|| value.get("orderBook"))
        .or_else(|| value.get("data").and_then(|v| v.get("order_book")))
        .or_else(|| value.get("data").and_then(|v| v.get("orderBook")))
}

fn lighter_ticker_value(value: &serde_json::Value) -> Option<&serde_json::Value> {
    value
        .get("ticker")
        .or_else(|| value.get("data").and_then(|v| v.get("ticker")))
}

fn lighter_order_book_nonce(value: &serde_json::Value) -> Option<u64> {
    lighter_order_book_value(value)
        .and_then(|book| book.get("nonce"))
        .or_else(|| value.get("nonce"))
        .or_else(|| value.get("seq"))
        .and_then(|v| v.as_u64())
}

fn lighter_order_book_begin_nonce(value: &serde_json::Value) -> Option<u64> {
    lighter_order_book_value(value)
        .and_then(|book| book.get("begin_nonce").or_else(|| book.get("beginNonce")))
        .or_else(|| value.get("begin_nonce"))
        .or_else(|| value.get("beginNonce"))
        .and_then(|v| v.as_u64())
}

fn lighter_ticker_nonce(value: &serde_json::Value) -> Option<u64> {
    value
        .get("nonce")
        .or_else(|| lighter_ticker_value(value).and_then(|ticker| ticker.get("nonce")))
        .and_then(|v| v.as_u64())
}

fn lighter_has_continuity_gap(
    last_nonce: Option<u64>,
    begin_nonce: Option<u64>,
    nonce: Option<u64>,
) -> bool {
    let Some(previous) = last_nonce else {
        return false;
    };
    if let Some(begin) = begin_nonce {
        return begin != previous;
    }
    if let Some(current) = nonce {
        return current <= previous;
    }
    false
}

fn decode_order_book_snapshot(
    value: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    seq_fallback: &mut u64,
) -> Option<ParsedL2Message> {
    let order_book = lighter_order_book_value(value)?;
    let bids = match order_book.get("bids") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    let asks = match order_book.get("asks") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    let seq = lighter_order_book_nonce(value).unwrap_or_else(|| {
        *seq_fallback = seq_fallback.wrapping_add(1);
        *seq_fallback
    });
    let timestamp_ms = decode_market_timestamp_ms(value, "decode_order_book_snapshot");
    let snapshot = super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    };
    Some(ParsedL2Message {
        event: MarketDataEvent::L2Snapshot(snapshot),
        seq,
    })
}

fn decode_order_book_top(value: &serde_json::Value) -> Option<TopOfBook> {
    let order_book = value
        .get("order_book")
        .or_else(|| value.get("orderBook"))
        .or_else(|| value.get("data").and_then(|v| v.get("order_book")))
        .or_else(|| value.get("data").and_then(|v| v.get("orderBook")))?;
    let bids = match order_book.get("bids") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    let asks = match order_book.get("asks") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    TopOfBook::from_levels(
        &bids,
        &asks,
        value
            .get("timestamp")
            .or_else(|| value.get("ts"))
            .and_then(|v| v.as_i64()),
    )
}

fn decode_ticker_channel_message(value: &serde_json::Value) -> Option<LighterTickerMessage> {
    let channel = value.get("channel").and_then(|v| v.as_str())?;
    if !channel.starts_with("ticker:") {
        return None;
    }
    let ticker = lighter_ticker_value(value)?;
    let bid = parse_level_from_object(ticker.get("b")?)?;
    let ask = parse_level_from_object(ticker.get("a")?)?;
    let timestamp_ms = decode_market_timestamp_ms(value, "decode_ticker_channel_message");
    let top = TopOfBook::from_levels(&[bid], &[ask], Some(timestamp_ms))?;
    Some(LighterTickerMessage {
        top,
        timestamp_ms,
        venue_nonce: lighter_ticker_nonce(value),
    })
}

fn lighter_should_apply_ticker_backstop(
    initial_snapshot_applied: bool,
    enabled: bool,
    order_book_age_ms: u64,
    threshold_ms: u64,
    ticker_nonce: Option<u64>,
    last_applied_nonce: Option<u64>,
) -> bool {
    if !enabled || !initial_snapshot_applied || order_book_age_ms < threshold_ms {
        return false;
    }
    match ticker_nonce {
        Some(nonce) => Some(nonce) != last_applied_nonce,
        None => last_applied_nonce.is_none(),
    }
}

fn json_ping_response(value: &serde_json::Value) -> Option<&'static str> {
    let msg_type = value.get("type").and_then(|v| v.as_str());
    if msg_type == Some("ping") {
        Some(r#"{"type":"pong"}"#)
    } else {
        None
    }
}

/// Decode the first subscribed Lighter order_book message as a full snapshot.
///
/// Lighter sends a complete snapshot on subscription, then only state changes
/// after that. The websocket continuity keys are `begin_nonce` and `nonce`,
/// while `offset` is server-local and may jump on reconnection.
fn decode_order_book_channel_message(
    value: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    seq_fallback: &mut u64,
) -> Option<LighterBookMessage> {
    let channel = value.get("channel").and_then(|v| v.as_str())?;
    if !channel.starts_with("order_book:") {
        return None;
    }
    let order_book = lighter_order_book_value(value)?;
    let bids = match order_book.get("bids") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    let asks = match order_book.get("asks") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    let venue_nonce = lighter_order_book_nonce(value);
    let venue_begin_nonce = lighter_order_book_begin_nonce(value);
    let seq = venue_nonce.unwrap_or_else(|| {
        *seq_fallback = seq_fallback.wrapping_add(1);
        *seq_fallback
    });
    let timestamp_ms = decode_market_timestamp_ms(value, "decode_order_book_channel_message");
    let snapshot = super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    };
    Some(LighterBookMessage {
        event: MarketDataEvent::L2Snapshot(snapshot),
        seq,
        venue_nonce,
        venue_begin_nonce,
    })
}

/// Decode a Lighter order_book channel message as an L2Delta (post-subscription).
///
/// After the initial subscription snapshot, Lighter sends "state changes" only —
/// each message contains changed levels, NOT the full book. Empty `bids`/`asks`
/// arrays mean "no changes on that side", not "side is empty".
///
/// Levels with size == 0.0 represent removals; non-zero sizes are upserts.
/// See: <https://apidocs.lighter.xyz/docs/websocket-reference#order-book>
fn decode_order_book_channel_delta(
    value: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    seq_fallback: &mut u64,
) -> Option<LighterBookMessage> {
    let channel = value.get("channel").and_then(|v| v.as_str())?;
    if !channel.starts_with("order_book:") {
        return None;
    }
    let order_book = lighter_order_book_value(value)?;
    let mut changes: Vec<BookLevelDelta> = Vec::new();
    if let Some(bids_val) = order_book.get("bids") {
        if let Some(levels) = parse_levels_from_objects(bids_val) {
            for level in levels {
                changes.push(BookLevelDelta {
                    side: BookSide::Bid,
                    price: level.price,
                    size: level.size,
                });
            }
        }
    }
    if let Some(asks_val) = order_book.get("asks") {
        if let Some(levels) = parse_levels_from_objects(asks_val) {
            for level in levels {
                changes.push(BookLevelDelta {
                    side: BookSide::Ask,
                    price: level.price,
                    size: level.size,
                });
            }
        }
    }
    let venue_nonce = lighter_order_book_nonce(value);
    let venue_begin_nonce = lighter_order_book_begin_nonce(value);
    let seq = venue_nonce.unwrap_or_else(|| {
        *seq_fallback = seq_fallback.wrapping_add(1);
        *seq_fallback
    });
    let timestamp_ms = decode_market_timestamp_ms(value, "decode_order_book_channel_delta");
    let delta = super::super::types::L2Delta {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        changes,
    };
    Some(LighterBookMessage {
        event: MarketDataEvent::L2Delta(delta),
        seq,
        venue_nonce,
        venue_begin_nonce,
    })
}

fn lighter_error_code(value: &serde_json::Value) -> Option<i64> {
    value
        .get("error")
        .and_then(|err| err.get("code"))
        .and_then(|v| v.as_i64())
}

fn is_lighter_book_message(value: &serde_json::Value) -> bool {
    matches!(
        value.get("type").and_then(|v| v.as_str()),
        Some("l2_snapshot") | Some("l2_delta") | Some("update/order_book")
    ) || (value.get("bids").is_some() && value.get("asks").is_some())
        || value.get("levels").is_some()
        || value.get("order_book").is_some()
}

fn has_lighter_book_fields(value: &serde_json::Value) -> bool {
    value.get("order_book").is_some()
        || value.get("orderBook").is_some()
        || value
            .get("data")
            .and_then(|v| v.get("order_book"))
            .is_some()
        || value.get("data").and_then(|v| v.get("orderBook")).is_some()
}

fn build_order_book_channel(market_id: u64) -> String {
    format!("order_book/{market_id}")
}

fn build_ticker_channel(market_id: u64) -> String {
    format!("ticker/{market_id}")
}

async fn fetch_lighter_orderbooks_with_fallbacks(
    http: &Client,
    rest_url: &str,
    ws_url: &str,
) -> anyhow::Result<(Vec<LighterOrderBookInfo>, String)> {
    let mut bases = Vec::new();
    if let Ok(val) = std::env::var("LIGHTER_HTTP_BASE_URL") {
        if !val.trim().is_empty() {
            bases.push(val);
        }
    }
    if let Ok(val) = std::env::var("LIGHTER_REST_URL") {
        if !val.trim().is_empty() {
            bases.push(val);
        }
    }
    if !rest_url.trim().is_empty() {
        bases.push(rest_url.to_string());
    }
    if let Some(derived) = derive_https_base_from_ws(ws_url) {
        bases.push(derived);
    }
    bases.push("https://api.lighter.xyz".to_string());
    let endpoints = ["/api/v1/orderBooks", "/api/v1/orderbooks"];
    let strict_maker_only = phase51_lighter_strict_maker_only_observation_enabled();
    for (base_index, base) in bases.into_iter().enumerate() {
        let base = base.trim_end_matches('/').to_string();
        for (endpoint_index, endpoint) in endpoints.iter().enumerate() {
            let attempt_index = base_index
                .saturating_mul(endpoints.len())
                .saturating_add(endpoint_index);
            let url = format!("{base}{endpoint}");
            if strict_maker_only {
                eprintln!(
                    "{}",
                    phase51_lighter_strict_orderbooks_attempt_log(endpoint, attempt_index)
                );
            } else {
                eprintln!("INFO: Lighter resolving market id attempt url={}", url);
            }
            match http.get(url.clone()).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    let parsed = serde_json::from_str::<serde_json::Value>(&body).ok();
                    if status.is_success() {
                        if let Some(value) = parsed {
                            let data = parse_lighter_orderbooks(&value);
                            if !data.is_empty() {
                                return Ok((data, url));
                            }
                        }
                    }
                    if strict_maker_only {
                        eprintln!(
                            "{}",
                            phase51_lighter_strict_orderbooks_failure_log(
                                endpoint,
                                attempt_index,
                                status,
                                "non_success_or_empty",
                            )
                        );
                    } else {
                        let snippet: String = body.chars().take(160).collect();
                        eprintln!(
                            "WARN: Lighter orderBooks fetch failed status={} url={} snippet={}",
                            status, url, snippet
                        );
                    }
                }
                Err(err) => {
                    if strict_maker_only {
                        eprintln!(
                            "{}",
                            phase51_lighter_strict_orderbooks_error_log(endpoint, attempt_index)
                        );
                    } else {
                        eprintln!(
                            "WARN: Lighter orderBooks fetch error url={} err={}",
                            url, err
                        );
                    }
                }
            }
        }
    }
    anyhow::bail!("Lighter orderBooks discovery failed")
}

fn parse_lighter_orderbooks(value: &serde_json::Value) -> Vec<LighterOrderBookInfo> {
    let empty: Vec<serde_json::Value> = Vec::new();
    let list = value
        .as_array()
        .or_else(|| value.get("data").and_then(|v| v.as_array()))
        .or_else(|| value.get("order_books").and_then(|v| v.as_array()))
        .or_else(|| value.get("orderBooks").and_then(|v| v.as_array()))
        .unwrap_or(&empty);
    list.iter()
        .filter_map(|entry| {
            let symbol = entry
                .get("symbol")
                .or_else(|| entry.get("market"))
                .and_then(|v| v.as_str())?
                .to_string();
            let market_id = entry
                .get("market_id")
                .or_else(|| entry.get("marketId"))
                .or_else(|| entry.get("id"))
                .and_then(|v| v.as_u64())?;
            let price_decimals = parse_optional_u32(
                entry,
                &[
                    "price_decimals",
                    "priceDecimals",
                    "supported_price_decimals",
                    "supportedPriceDecimals",
                    "price_precision",
                    "pricePrecision",
                ],
            );
            let size_decimals = parse_optional_u32(
                entry,
                &[
                    "size_decimals",
                    "sizeDecimals",
                    "supported_size_decimals",
                    "supportedSizeDecimals",
                    "size_precision",
                    "sizePrecision",
                ],
            );
            Some(LighterOrderBookInfo {
                symbol,
                market_id,
                price_decimals,
                size_decimals,
            })
        })
        .collect()
}

fn parse_optional_u32(value: &serde_json::Value, keys: &[&str]) -> Option<u32> {
    for key in keys {
        if let Some(val) = value.get(*key).and_then(|v| v.as_u64()) {
            return u32::try_from(val).ok();
        }
    }
    None
}

fn parse_levels_from_objects(value: &serde_json::Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if entry.as_object().is_some() {
            out.push(parse_level_from_object(entry)?);
            continue;
        }
        if let Some(items) = entry.as_array() {
            if items.len() < 2 {
                continue;
            }
            let price = parse_f64_value(&items[0])?;
            let size = parse_f64_value(&items[1])?;
            out.push(BookLevel { price, size });
        }
    }
    Some(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LighterOrderBookInfo {
    symbol: String,
    market_id: u64,
    price_decimals: Option<u32>,
    size_decimals: Option<u32>,
}

#[derive(Debug, Clone)]
struct LighterResolvedMarket {
    symbol: String,
    market_id: u64,
    price_decimals: Option<u32>,
    size_decimals: Option<u32>,
    source: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LighterAccountQuery {
    by: &'static str,
    value: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct LighterSendTxResponse {
    order_id: Option<String>,
}

fn derive_https_base_from_ws(ws_url: &str) -> Option<String> {
    let ws_url = ws_url.trim();
    let host = ws_url
        .strip_prefix("wss://")
        .or_else(|| ws_url.strip_prefix("ws://"))?;
    let host = host.split('/').next()?;
    if host.is_empty() {
        None
    } else {
        Some(format!("https://{host}"))
    }
}

fn normalize_lighter_symbol(symbol: &str) -> String {
    let mut upper = symbol.trim().to_uppercase();
    for suffix in ["-USD-PERP", "-PERP", "-USD"] {
        if upper.ends_with(suffix) {
            upper = upper.trim_end_matches(suffix).to_string();
            break;
        }
    }
    upper
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeInForce;
    use httpmock::Method::{GET, POST};
    use httpmock::MockServer;
    use reqwest::header::{HeaderMap, HeaderValue};
    use std::fs;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::tempdir;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn set_env(key: &str, val: &str) {
        std::env::set_var(key, val);
    }

    fn unset_env(key: &str) {
        std::env::remove_var(key);
    }

    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&str]) -> Self {
            let saved = keys
                .iter()
                .map(|key| ((*key).to_string(), std::env::var(*key).ok()))
                .collect::<Vec<_>>();
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.iter() {
                match value {
                    Some(val) => std::env::set_var(key, val),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    fn passive_tap_env_keys() -> [&'static str; 3] {
        [
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_ENABLED",
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_OUTPUT_PATH",
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_MAX_ROWS",
        ]
    }

    fn passive_tap_headers(entries: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (key, value) in entries {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).expect("header name"),
                HeaderValue::from_str(value).expect("header value"),
            );
        }
        headers
    }

    #[test]
    fn phase51_lighter_passive_pressure_tap_disabled_emits_no_file() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let keys = passive_tap_env_keys();
        let _guard = EnvGuard::new(&keys);
        for key in keys {
            unset_env(key);
        }
        phase51_lighter_passive_pressure_tap_reset_for_tests();
        let tmp = tempdir().expect("tempdir");
        let output = tmp.path().join("pressure.jsonl");
        set_env(
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_OUTPUT_PATH",
            output.to_str().expect("path"),
        );

        let emitted = phase51_lighter_maybe_emit_passive_pressure_observation(
            "sendTx",
            true,
            &HeaderMap::new(),
            Some(&json!({"sendtx_per_minute_limit": 4000})),
        );

        assert!(!emitted);
        assert!(!output.exists());
    }

    #[test]
    fn phase51_lighter_passive_pressure_tap_writes_complete_sanitized_metadata() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let keys = passive_tap_env_keys();
        let _guard = EnvGuard::new(&keys);
        for key in keys {
            unset_env(key);
        }
        phase51_lighter_passive_pressure_tap_reset_for_tests();
        let tmp = tempdir().expect("tempdir");
        let output = tmp.path().join("pressure.jsonl");
        set_env(
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_ENABLED",
            "true",
        );
        set_env(
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_OUTPUT_PATH",
            output.to_str().expect("path"),
        );
        let body = json!({
            "order_id": "raw-order-value",
            "api_key": "secret-value",
            "active_order_headroom_account": 12,
            "sendtx_per_minute_limit": 4000,
            "sendtx_per_minute_remaining": 3998,
            "weighted_requests_per_minute_limit": 24000,
            "weighted_requests_per_minute_remaining": 23999
        });
        let headers =
            passive_tap_headers(&[("authorization", "secret"), ("x-sendtx-limit", "9999")]);

        let emitted = phase51_lighter_maybe_emit_passive_pressure_observation(
            "sendTx",
            true,
            &headers,
            Some(&body),
        );

        assert!(emitted);
        let content = fs::read_to_string(&output).expect("output");
        assert!(!content.contains("raw-order-value"));
        assert!(!content.contains("secret-value"));
        assert!(!content.contains("authorization"));
        let row: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("line")).expect("json");
        assert_eq!(row["venue_id"], "lighter");
        assert_eq!(row["source_endpoint_family"], "sendTx");
        assert_eq!(row["active_order_headroom_account"], 12);
        assert_eq!(row["sendtx_per_minute_limit"], 4000);
        assert_eq!(row["sendtx_per_minute_remaining"], 3998);
        assert_eq!(row["weighted_requests_per_minute_limit"], 24000);
        assert_eq!(row["weighted_requests_per_minute_remaining"], 23999);
        assert_eq!(
            row["pressure_dimensions_complete_from_passive_response"],
            true
        );
        assert_eq!(row["blocker_cleared"], false);
        assert_eq!(row["raw_response_body_persisted"], false);
        assert_eq!(row["unsanitized_headers_persisted"], false);
        assert_eq!(row["secret_material_persisted"], false);
        assert_eq!(row["phase51_validators_run"], false);
    }

    #[test]
    fn phase51_lighter_passive_pressure_tap_does_not_treat_generic_rate_limit_as_source_truth() {
        let headers = passive_tap_headers(&[
            ("x-ratelimit-limit", "24000"),
            ("x-ratelimit-remaining", "23999"),
        ]);

        let observation =
            phase51_lighter_build_passive_pressure_observation("sendTx", true, &headers, None);

        assert!(observation.generic_rate_limit_header_present);
        assert!(observation.rest_requests_per_minute_limit.is_none());
        assert!(observation.rest_requests_per_minute_remaining.is_none());
        assert!(!observation.pressure_dimensions_complete_from_passive_response);
        assert_eq!(
            observation.required_missing_dimensions,
            vec![
                "active_order_headroom_account",
                "sendtx_per_minute_limit/sendtx_per_minute_remaining",
                "rest_requests_per_minute_pair_or_weighted_requests_per_minute_pair",
            ]
        );
    }

    #[test]
    fn phase51_lighter_passive_pressure_tap_is_bounded() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let keys = passive_tap_env_keys();
        let _guard = EnvGuard::new(&keys);
        for key in keys {
            unset_env(key);
        }
        phase51_lighter_passive_pressure_tap_reset_for_tests();
        let tmp = tempdir().expect("tempdir");
        let output = tmp.path().join("pressure.jsonl");
        set_env(
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_ENABLED",
            "true",
        );
        set_env(
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_OUTPUT_PATH",
            output.to_str().expect("path"),
        );
        set_env(
            "PARAPHINA_PHASE51_LIGHTER_PASSIVE_PRESSURE_TAP_MAX_ROWS",
            "1",
        );

        assert!(phase51_lighter_maybe_emit_passive_pressure_observation(
            "sendTx",
            true,
            &HeaderMap::new(),
            Some(&json!({})),
        ));
        assert!(!phase51_lighter_maybe_emit_passive_pressure_observation(
            "sendTx",
            true,
            &HeaderMap::new(),
            Some(&json!({})),
        ));
        let content = fs::read_to_string(&output).expect("output");
        assert_eq!(content.lines().count(), 1);
    }

    fn temp_nonce_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "lighter_connector_nonce_{}_{}_{}.json",
            label,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn freshness_reset_and_anchor_behavior() {
        let freshness = Freshness::default();
        freshness.last_parsed_ns.store(123, Ordering::Relaxed);
        freshness.last_published_ns.store(456, Ordering::Relaxed);
        freshness.last_book_event_ns.store(789, Ordering::Relaxed);

        // After reset, all timestamps should be 0
        freshness.reset_for_new_connection();
        assert_eq!(freshness.last_parsed_ns.load(Ordering::Relaxed), 0);
        assert_eq!(freshness.last_published_ns.load(Ordering::Relaxed), 0);
        assert_eq!(freshness.last_ws_rx_ns.load(Ordering::Relaxed), 0);
        assert_eq!(freshness.last_data_rx_ns.load(Ordering::Relaxed), 0);
        assert_eq!(
            freshness.last_book_event_ns.load(Ordering::Relaxed),
            0,
            "last_book_event_ns must be reset on new connection"
        );

        // With anchor == 0, should use connect_start_ns
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

        // Book events advance the anchor
        freshness.last_book_event_ns.store(3_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 3_000);

        // last_published_ns also advances the anchor
        freshness.last_published_ns.store(4_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 4_000);
    }

    #[test]
    fn freshness_watchdog_would_stale_with_no_data() {
        let freshness = Freshness::default();
        freshness.reset_for_new_connection();

        // Simulate connect_start_ns at time 0
        let connect_start_ns: u64 = 0;

        // With no data, anchor should be connect_start_ns
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(
            anchor, connect_start_ns,
            "anchor should be connect_start_ns when no data"
        );

        // Simulate "now" being 15 seconds later (15_000_000_000 ns)
        let now_ns: u64 = 15_000_000_000;
        let age = age_ms(now_ns, anchor);
        assert_eq!(age, 15_000, "age should be 15000ms");

        // With default stale_ms of 10_000, this should trigger stale
        assert!(
            age > LIGHTER_STALE_MS_DEFAULT,
            "should be stale when age ({age}ms) > stale_ms ({LIGHTER_STALE_MS_DEFAULT}ms)"
        );
    }

    #[test]
    fn freshness_stays_fresh_with_data() {
        let freshness = Freshness::default();
        freshness.reset_for_new_connection();

        // Simulate connect_start_ns at time 0
        let connect_start_ns: u64 = 0;

        // Simulate receiving a book event at 5 seconds
        let data_time_ns: u64 = 5_000_000_000;
        freshness
            .last_book_event_ns
            .store(data_time_ns, Ordering::Relaxed);

        // Check anchor uses last_book_event_ns
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, data_time_ns, "anchor should be last_book_event_ns");

        // Simulate "now" being 8 seconds (8_000_000_000 ns)
        let now_ns: u64 = 8_000_000_000;
        let age = age_ms(now_ns, anchor);
        assert_eq!(age, 3_000, "age should be 3000ms from last data");

        // With default stale_ms of 10_000, this should NOT trigger stale
        assert!(
            age < LIGHTER_STALE_MS_DEFAULT,
            "should NOT be stale when age ({age}ms) < stale_ms ({LIGHTER_STALE_MS_DEFAULT}ms)"
        );
    }

    #[test]
    fn lighter_account_poll_ms_prefers_specific_override() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "PARAPHINA_LIVE_ACCOUNT_POLL_MS",
            "PARAPHINA_LIGHTER_ACCOUNT_POLL_MS",
            "LIGHTER_ACCOUNT_POLL_MS",
        ]);
        unset_env("LIGHTER_ACCOUNT_POLL_MS");
        set_env("PARAPHINA_LIVE_ACCOUNT_POLL_MS", "1000");
        set_env("PARAPHINA_LIGHTER_ACCOUNT_POLL_MS", "3000");
        assert_eq!(lighter_account_poll_ms(1_000), 3_000);
        unset_env("PARAPHINA_LIGHTER_ACCOUNT_POLL_MS");
        set_env("LIGHTER_ACCOUNT_POLL_MS", "2500");
        assert_eq!(lighter_account_poll_ms(1_000), 2_500);
        unset_env("LIGHTER_ACCOUNT_POLL_MS");
        assert_eq!(lighter_account_poll_ms(1_000), 1_000);
    }

    #[test]
    fn order_book_channel_formats() {
        assert_eq!(build_order_book_channel(42), "order_book/42");
        assert_eq!(build_ticker_channel(42), "ticker/42");
    }

    #[test]
    fn lighter_reconnect_sleep_is_skipped_after_healthy_session() {
        assert!(lighter_should_skip_reconnect_sleep(
            Duration::from_secs(65),
            Duration::from_secs(60),
        ));
        assert!(!lighter_should_skip_reconnect_sleep(
            Duration::from_secs(30),
            Duration::from_secs(60),
        ));
    }

    #[test]
    fn decode_order_book_update_top() {
        let value = serde_json::json!({
            "type": "update/order_book",
            "order_book": {
                "bids": [{"price":"100.0","size":"2.0"}],
                "asks": [{"price":"101.0","size":"3.0"}]
            }
        });
        let top = decode_order_book_top(&value).expect("top");
        assert_eq!(top.best_bid_px, 100.0);
        assert_eq!(top.best_bid_sz, 2.0);
        assert_eq!(top.best_ask_px, 101.0);
        assert_eq!(top.best_ask_sz, 3.0);
    }

    #[test]
    fn decode_ticker_channel_message_parses_top_and_nonce() {
        let value = serde_json::json!({
            "channel": "ticker:42",
            "nonce": 9182249734u64,
            "timestamp": 1700000000123i64,
            "ticker": {
                "s": "ETH",
                "a": {"price": "2064.48", "size": "0.4950"},
                "b": {"price": "2064.30", "size": "1.0392"},
                "last_updated_at": 1774883844921166u64
            },
            "type": "update/ticker"
        });
        let parsed = decode_ticker_channel_message(&value).expect("ticker");
        assert_eq!(parsed.venue_nonce, Some(9182249734));
        assert_eq!(parsed.timestamp_ms, 1_700_000_000_123);
        assert_eq!(parsed.top.best_bid_px, 2064.30);
        assert_eq!(parsed.top.best_bid_sz, 1.0392);
        assert_eq!(parsed.top.best_ask_px, 2064.48);
        assert_eq!(parsed.top.best_ask_sz, 0.4950);
    }

    #[test]
    fn ticker_backstop_requires_threshold_and_new_nonce() {
        assert!(!lighter_should_apply_ticker_backstop(
            false,
            true,
            2_000,
            1_200,
            Some(10),
            None,
        ));
        assert!(!lighter_should_apply_ticker_backstop(
            true,
            true,
            1_000,
            1_200,
            Some(10),
            None,
        ));
        assert!(lighter_should_apply_ticker_backstop(
            true,
            true,
            1_500,
            1_200,
            Some(10),
            None,
        ));
        assert!(!lighter_should_apply_ticker_backstop(
            true,
            true,
            1_500,
            1_200,
            Some(10),
            Some(10),
        ));
    }

    #[test]
    fn build_ticker_backstop_event_emits_one_level_snapshot() {
        let top = TopOfBook {
            best_bid_px: 2064.30,
            best_bid_sz: 1.0392,
            best_ask_px: 2064.48,
            best_ask_sz: 0.4950,
            timestamp_ms: Some(1_700_000_000_123),
        };
        let event = build_ticker_backstop_event(&top, 3, "LIGHTER", 77, 1_700_000_000_123);
        match event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.venue_index, 3);
                assert_eq!(snapshot.venue_id, "LIGHTER");
                assert_eq!(snapshot.seq, 77);
                assert_eq!(snapshot.timestamp_ms, 1_700_000_000_123);
                assert_eq!(snapshot.bids.len(), 1);
                assert_eq!(snapshot.asks.len(), 1);
                assert_eq!(snapshot.bids[0].price, 2064.30);
                assert_eq!(snapshot.bids[0].size, 1.0392);
                assert_eq!(snapshot.asks[0].price, 2064.48);
                assert_eq!(snapshot.asks[0].size, 0.4950);
            }
            other => panic!("expected L2Snapshot, got {other:?}"),
        }
    }

    #[test]
    fn lighter_has_auth_false_when_missing_any_component() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "LIGHTER_API_KEY_INDEX",
            "LIGHTER_ACCOUNT_INDEX",
            "LIGHTER_API_PRIVATE_KEY_HEX",
        ]);
        unset_env("LIGHTER_API_KEY_INDEX");
        unset_env("LIGHTER_ACCOUNT_INDEX");
        unset_env("LIGHTER_API_PRIVATE_KEY_HEX");

        set_env("LIGHTER_ACCOUNT_INDEX", "2");
        set_env("LIGHTER_API_PRIVATE_KEY_HEX", "deadbeef");
        assert!(!LighterConfig::from_env().has_auth());

        unset_env("LIGHTER_API_KEY_INDEX");
        unset_env("LIGHTER_ACCOUNT_INDEX");
        unset_env("LIGHTER_API_PRIVATE_KEY_HEX");

        set_env("LIGHTER_API_KEY_INDEX", "1");
        set_env("LIGHTER_API_PRIVATE_KEY_HEX", "deadbeef");
        assert!(!LighterConfig::from_env().has_auth());

        unset_env("LIGHTER_API_KEY_INDEX");
        unset_env("LIGHTER_ACCOUNT_INDEX");
        unset_env("LIGHTER_API_PRIVATE_KEY_HEX");

        set_env("LIGHTER_API_KEY_INDEX", "1");
        set_env("LIGHTER_ACCOUNT_INDEX", "2");
        assert!(!LighterConfig::from_env().has_auth());
    }

    #[test]
    fn lighter_has_auth_true_when_all_present() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "LIGHTER_API_KEY_INDEX",
            "LIGHTER_ACCOUNT_INDEX",
            "LIGHTER_API_PRIVATE_KEY_HEX",
        ]);
        unset_env("LIGHTER_API_KEY_INDEX");
        unset_env("LIGHTER_ACCOUNT_INDEX");
        unset_env("LIGHTER_API_PRIVATE_KEY_HEX");

        set_env("LIGHTER_API_KEY_INDEX", "1");
        set_env("LIGHTER_ACCOUNT_INDEX", "2");
        set_env("LIGHTER_API_PRIVATE_KEY_HEX", "deadbeef");
        assert!(LighterConfig::from_env().has_auth());
    }

    #[test]
    fn lighter_has_auth_false_when_private_key_empty() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "LIGHTER_API_KEY_INDEX",
            "LIGHTER_ACCOUNT_INDEX",
            "LIGHTER_API_PRIVATE_KEY_HEX",
        ]);
        unset_env("LIGHTER_API_KEY_INDEX");
        unset_env("LIGHTER_ACCOUNT_INDEX");
        unset_env("LIGHTER_API_PRIVATE_KEY_HEX");

        set_env("LIGHTER_API_KEY_INDEX", "1");
        set_env("LIGHTER_ACCOUNT_INDEX", "2");
        set_env("LIGHTER_API_PRIVATE_KEY_HEX", "   ");
        assert!(!LighterConfig::from_env().has_auth());
    }

    #[tokio::test]
    async fn lighter_connector_nonce_increases() {
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let first = connector.next_nonce();
        let second = connector.next_nonce();
        assert!(second > first);
    }

    #[tokio::test]
    async fn lighter_nonce_uses_persisted_value() {
        let path = temp_nonce_path("persisted");
        let _ = fs::remove_file(&path);
        store_last_nonce(&path, 100).expect("store");
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: Some(path.clone()),
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        assert_eq!(connector.nonce.get(), 100);
        let first = connector.next_nonce();
        assert!(first > 100);
        let _ = fs::remove_file(&path);
    }

    #[tokio::test]
    async fn lighter_account_snapshot_hits_api_v1_account() {
        let req = Client::new()
            .get("https://example.invalid/api/v1/account")
            .query(&LighterAccountQuery {
                by: "index",
                value: "123".to_string(),
            })
            .build()
            .expect("request");
        assert_eq!(req.url().query(), Some("by=index&value=123"));

        let payload = serde_json::json!({
            "code": 200,
            "total": 1,
            "accounts": [{
                "index": 123,
                "account_index": 123,
                "available_balance": "990.0",
                "collateral": "1000.0",
                "transaction_time": 1_700_000_000_123_000i64,
                "positions": [{
                    "symbol": "ETH",
                    "sign": -1,
                    "position": "2.5",
                    "avg_entry_price": "2500.0",
                    "liquidation_price": "2100.0"
                }],
                "assets": [{
                    "symbol": "ETH",
                    "asset_id": 1,
                    "balance": "0.5",
                    "locked_balance": "0.1"
                }]
            }]
        });
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 3,
            paper_mode: true,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: Some("t".to_string()),
            nonce_path: None,
            signer_url: None,
        };
        let snapshot = parse_account_snapshot_with_meta(&payload, &cfg.venue_id, cfg.venue_index)
            .expect("snapshot");
        assert_eq!(snapshot.venue_id, "LIGHTER");
        assert_eq!(snapshot.venue_index, 3);
        assert_eq!(snapshot.timestamp_ms, 1_700_000_000_123);
        assert_eq!(snapshot.positions.len(), 1);
        assert!((snapshot.positions[0].size + 2.5).abs() < 1e-9);
        assert_eq!(snapshot.liquidation.price_liq, Some(2100.0));
        assert!(snapshot
            .balances
            .iter()
            .any(|bal| bal.asset == "USDC" && (bal.available - 990.0).abs() < 1e-9));
    }

    #[tokio::test]
    async fn lighter_fetch_account_snapshot_uses_poll_time_for_freshness() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/api/v1/account")
                    .query_param("by", "index")
                    .query_param("value", "123")
                    .header("authorization", "Bearer t");
                then.status(200).json_body(serde_json::json!({
                    "accounts": [{
                        "account_index": 123,
                        "available_balance": "990.0",
                        "collateral": "1000.0",
                        "transaction_time": 1_700_000_000_123_000i64,
                        "positions": [],
                        "assets": []
                    }]
                }));
            })
            .await;

        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 2,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: Some("t".to_string()),
            nonce_path: None,
            signer_url: None,
        };
        let client = Client::new();
        let before = now_ms();
        let event = fetch_account_snapshot(&client, &cfg)
            .await
            .expect("account snapshot");
        let after = now_ms();
        match event {
            AccountEvent::Snapshot(snapshot) => {
                assert!(snapshot.timestamp_ms >= before as i64);
                assert!(snapshot.timestamp_ms <= after.saturating_add(1_000) as i64);
                assert_ne!(snapshot.timestamp_ms, 1_700_000_000_123i64);
            }
        }
    }

    #[tokio::test]
    async fn lighter_place_order_calls_signer_then_sendtx() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "LIGHTER_MARKET_ID",
            "LIGHTER_MARKET",
            PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV,
        ]);
        unset_env("LIGHTER_MARKET_ID");
        unset_env("LIGHTER_MARKET");
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "true");
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "BTC-USD",
                            "market_id": 7,
                            "price_decimals": 2,
                            "size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"create_order\"")
                    .body_contains("\"account_index\":123")
                    .body_contains("\"api_key_index\":1")
                    .body_contains("\"market_index\":7")
                    .body_contains("\"client_order_index\":42")
                    .body_contains("\"price\":10012")
                    .body_contains("\"base_amount\":1234")
                    .body_contains("\"post_only\":1");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":14,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=14")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200)
                    .json_body(serde_json::json!({"order_id":"abc"}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Buy,
            price: 100.12,
            size: 1.234,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: "42".to_string(),
        };
        let resp = connector.place_order(req).await.expect("place");
        assert_eq!(resp.order_id.as_deref(), Some("abc"));
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_ioc_order_uses_nil_order_expiry() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "LIGHTER_MARKET_ID",
            "LIGHTER_MARKET",
            PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV,
        ]);
        unset_env("LIGHTER_MARKET_ID");
        unset_env("LIGHTER_MARKET");
        unset_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV);
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "ETH",
                            "market_id": 0,
                            "supported_price_decimals": 2,
                            "supported_size_decimals": 4
                        }
                    ]
                }));
            })
            .await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"create_order\"")
                    .body_contains("\"time_in_force\":\"Ioc\"")
                    .body_contains("\"reduce_only\":1")
                    .body_contains("\"order_expiry\":0");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":14,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=14")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200)
                    .json_body(serde_json::json!({"order_id":"ioc-cleanup"}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Buy,
            price: 2150.0,
            size: 0.79,
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: "42".to_string(),
        };
        let resp = connector.place_order(req).await.expect("place");
        assert_eq!(resp.order_id.as_deref(), Some("ioc-cleanup"));
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[test]
    fn phase51_lighter_strict_maker_only_env_parser_contract() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV]);

        unset_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV);
        assert!(!phase51_lighter_strict_maker_only_observation_enabled());

        for value in ["", "false", "0", "no", "invalid"] {
            set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, value);
            assert!(
                !phase51_lighter_strict_maker_only_observation_enabled(),
                "{value:?} must not enable strict maker-only observation"
            );
        }

        for value in ["true", "1", "yes", "TRUE", "YeS"] {
            set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, value);
            assert!(
                phase51_lighter_strict_maker_only_observation_enabled(),
                "{value:?} must enable strict maker-only observation"
            );
        }
    }

    #[test]
    fn phase51_lighter_account_all_trades_source_owner_gate_is_explicit() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            PHASE51_FORWARD_REFRESH_CAPTURE_ENABLED_ENV,
            PHASE51_FORWARD_REFRESH_CAPTURE_LIVE_NATIVE_ROLE_CANARY_APPROVED_ENV,
            PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY_ENV,
        ]);

        unset_env(PHASE51_FORWARD_REFRESH_CAPTURE_ENABLED_ENV);
        unset_env(PHASE51_FORWARD_REFRESH_CAPTURE_LIVE_NATIVE_ROLE_CANARY_APPROVED_ENV);
        unset_env(PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY_ENV);
        assert!(!phase51_lighter_account_all_trades_source_owner_enabled());

        set_env(PHASE51_FORWARD_REFRESH_CAPTURE_ENABLED_ENV, "true");
        set_env(
            PHASE51_FORWARD_REFRESH_CAPTURE_LIVE_NATIVE_ROLE_CANARY_APPROVED_ENV,
            "true",
        );
        assert!(!phase51_lighter_account_all_trades_source_owner_enabled());

        set_env(PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY_ENV, "false");
        assert!(!phase51_lighter_account_all_trades_source_owner_enabled());

        set_env(PHASE51_LIGHTER_NATIVE_ROLE_STRICT_CANARY_ENV, "yes");
        assert!(phase51_lighter_account_all_trades_source_owner_enabled());
    }

    #[test]
    fn phase51_lighter_strict_maker_only_contexts_are_sanitized_when_enabled() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV]);
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "yes");

        let place =
            phase51_lighter_place_error_context(7, 42, 10012, 1234, TimeInForce::Gtc, true, false);
        assert!(place.contains("strict_maker_only_observation=true"));
        assert!(!place.contains("client_order_index"));
        assert!(!place.contains("42"));
        assert!(!place.contains("price"));
        assert!(!place.contains("base_amount"));

        let replace = phase51_lighter_replace_error_context(
            7,
            "order_index",
            123456,
            10012,
            1234,
            "client-raw",
        );
        assert!(replace.contains("strict_maker_only_observation=true"));
        assert!(!replace.contains("123456"));
        assert!(!replace.contains("client-raw"));
        assert!(!replace.contains("client_order_id"));
        assert!(!replace.contains("price"));
        assert!(!replace.contains("base_amount"));

        unset_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV);
        let disabled =
            phase51_lighter_place_error_context(7, 42, 10012, 1234, TimeInForce::Gtc, true, false);
        assert!(disabled.contains("client_order_index"));
        assert!(disabled.contains("price"));
        assert!(disabled.contains("base_amount"));
    }

    #[test]
    fn phase51_lighter_strict_maker_only_orderbooks_logs_are_sanitized() {
        let attempt = phase51_lighter_strict_orderbooks_attempt_log("/api/v1/orderBooks", 3);
        assert!(attempt.contains("endpoint_family=orderBooks"));
        assert!(attempt.contains("attempt_index=3"));
        assert!(attempt.contains("strict_maker_only_observation=true"));
        assert!(!attempt.contains("/api/v1"));
        assert!(!attempt.contains("http"));
        assert!(!attempt.contains("token"));

        let failure = phase51_lighter_strict_orderbooks_failure_log(
            "/api/v1/orderBooks",
            4,
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            "non_success_or_empty",
        );
        assert!(failure.contains("status=429"));
        assert!(failure.contains("reason=non_success_or_empty"));
        assert!(!failure.contains("snippet"));
        assert!(!failure.contains("response"));
        assert!(!failure.contains("body"));
        assert!(!failure.contains("/api/v1"));

        let request_error = phase51_lighter_strict_orderbooks_error_log("/api/v1/orderBooks", 5);
        assert!(request_error.contains("reason=request_error"));
        assert!(!request_error.contains("err="));
        assert!(!request_error.contains("http"));

        assert_eq!(
            phase51_lighter_orderbooks_source_label(
                "https://example.invalid/api/v1/orderBooks?token=secret"
            ),
            "orderbooks_discovery"
        );
        assert_eq!(
            phase51_lighter_orderbooks_source_label("env:LIGHTER_MARKET_ID"),
            "env_lighter_market_id"
        );
    }

    #[test]
    fn phase51_lighter_strict_maker_only_decimal_errors_are_sanitized() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV]);
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "true");

        let strict = phase51_lighter_market_decimals_missing_error(
            "price",
            7,
            "https://example.invalid/api/v1/orderBooks?token=secret",
        )
        .to_string();
        assert!(strict.contains("strict_maker_only_observation=true"));
        assert!(strict.contains("source_label=orderbooks_discovery"));
        assert!(!strict.contains("https://"));
        assert!(!strict.contains("source_url"));
        assert!(!strict.contains("token"));
        assert!(!strict.contains("secret"));
        assert!(!strict.contains("market_id"));

        unset_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV);
        let disabled = phase51_lighter_market_decimals_missing_error(
            "price",
            7,
            "https://example.invalid/api/v1/orderBooks",
        )
        .to_string();
        assert!(disabled.contains("source_url=https://example.invalid/api/v1/orderBooks"));
        assert!(disabled.contains("market_id=7"));
    }

    #[test]
    fn phase51_lighter_strict_maker_only_rejection_contract() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV]);
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "true");
        let mut req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 0.01,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: "42".to_string(),
        };

        assert!(phase51_lighter_strict_maker_only_place_rejection(&req).is_none());

        req.post_only = false;
        assert!(phase51_lighter_strict_maker_only_place_rejection(&req).is_some());

        req.post_only = true;
        req.time_in_force = TimeInForce::Ioc;
        assert!(phase51_lighter_strict_maker_only_place_rejection(&req).is_some());

        req.time_in_force = TimeInForce::Gtc;
        req.reduce_only = true;
        assert!(phase51_lighter_strict_maker_only_place_rejection(&req).is_some());

        req.reduce_only = false;
        req.purpose = OrderPurpose::Exit;
        assert!(phase51_lighter_strict_maker_only_place_rejection(&req).is_some());

        unset_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV);
        assert!(phase51_lighter_strict_maker_only_place_rejection(&req).is_none());
    }

    #[test]
    fn phase51_lighter_baseline_cleanup_only_env_parser_contract() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV]);

        unset_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV);
        assert!(!phase51_lighter_baseline_cleanup_only_enabled());

        for value in ["", "false", "0", "no", "invalid"] {
            set_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV, value);
            assert!(
                !phase51_lighter_baseline_cleanup_only_enabled(),
                "{value:?} must not enable baseline cleanup-only"
            );
        }

        for value in ["true", "1", "yes", "TRUE", "YeS"] {
            set_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV, value);
            assert!(
                phase51_lighter_baseline_cleanup_only_enabled(),
                "{value:?} must enable baseline cleanup-only"
            );
        }
    }

    #[test]
    fn phase51_lighter_baseline_cleanup_only_rejection_contract() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV,
            PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE_ENV,
        ]);
        set_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV, "true");
        unset_env(PHASE51_LIGHTER_BASELINE_CLEANUP_MAX_SIZE_ENV);

        let mut req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Sell,
            price: 2150.0,
            size: 0.01,
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: "42".to_string(),
        };

        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH-USD", &req).is_none());
        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH", &req).is_none());
        assert!(phase51_lighter_baseline_cleanup_place_rejection("BTC-USD", &req).is_some());

        req.purpose = OrderPurpose::Mm;
        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH-USD", &req).is_some());
        req.purpose = OrderPurpose::Exit;

        req.time_in_force = TimeInForce::Gtc;
        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH-USD", &req).is_some());
        req.time_in_force = TimeInForce::Ioc;

        req.post_only = true;
        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH-USD", &req).is_some());
        req.post_only = false;

        req.reduce_only = false;
        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH-USD", &req).is_some());
        req.reduce_only = true;

        req.size = 0.010001;
        assert!(phase51_lighter_baseline_cleanup_place_rejection("ETH-USD", &req).is_some());

        unset_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV);
        assert!(phase51_lighter_baseline_cleanup_place_rejection("BTC-USD", &req).is_none());
    }

    #[test]
    fn phase51_lighter_baseline_cleanup_only_contexts_are_sanitized_when_enabled() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV]);
        set_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV, "1");

        let place =
            phase51_lighter_place_error_context(0, 42, 215012, 100, TimeInForce::Ioc, false, true);
        assert!(place.contains("baseline_cleanup_only=true"));
        assert!(!place.contains("client_order_index"));
        assert!(!place.contains("42"));
        assert!(!place.contains("price"));
        assert!(!place.contains("base_amount"));

        let replace =
            phase51_lighter_replace_error_context(0, "order_index", 123456, 215012, 100, "raw");
        assert!(replace.contains("baseline_cleanup_only=true"));
        assert!(!replace.contains("123456"));
        assert!(!replace.contains("raw"));
        assert!(!replace.contains("client_order_id"));
        assert!(!replace.contains("price"));
        assert!(!replace.contains("base_amount"));
    }

    #[tokio::test]
    async fn phase51_lighter_baseline_cleanup_only_rejects_mm_before_signing() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV]);
        set_env(PHASE51_LIGHTER_BASELINE_CLEANUP_ONLY_ENV, "yes");
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "http://127.0.0.1:9".to_string(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Buy,
            price: 2150.0,
            size: 0.01,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: "42".to_string(),
        };

        let err = connector
            .place_order(req)
            .await
            .expect_err("baseline cleanup-only gate must reject before signing");
        assert!(matches!(
            err.kind,
            crate::live::gateway::LiveGatewayErrorKind::Fatal
        ));
        assert!(err.message.contains("baseline cleanup-only"));
    }

    #[tokio::test]
    async fn phase51_lighter_strict_maker_only_rejects_reduce_only_ioc_before_signing() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV]);
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "true");
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "http://127.0.0.1:9".to_string(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Buy,
            price: 2150.0,
            size: 0.01,
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: "42".to_string(),
        };

        let err = connector
            .place_order(req)
            .await
            .expect_err("strict maker-only gate must reject before signing");
        assert!(matches!(
            err.kind,
            crate::live::gateway::LiveGatewayErrorKind::Fatal
        ));
        assert!(err.message.contains("strict maker-only observation"));
    }

    #[test]
    fn emergency_ioc_timeout_only_applies_to_reduce_only_exit_and_hedge() {
        let mut req = LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            side: Side::Buy,
            price: 2150.0,
            size: 0.04,
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: "42".to_string(),
        };

        assert!(lighter_emergency_ioc_request(&req));
        req.purpose = OrderPurpose::Hedge;
        assert!(lighter_emergency_ioc_request(&req));
        req.purpose = OrderPurpose::Mm;
        assert!(!lighter_emergency_ioc_request(&req));
        req.purpose = OrderPurpose::Exit;
        req.reduce_only = false;
        assert!(!lighter_emergency_ioc_request(&req));
        req.reduce_only = true;
        req.time_in_force = TimeInForce::Gtc;
        assert!(!lighter_emergency_ioc_request(&req));
    }

    #[tokio::test]
    async fn resolve_market_id_uses_cfg_market_when_env_missing() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("LIGHTER_MARKET_ID");
        std::env::remove_var("LIGHTER_MARKET");
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "ETH",
                            "market_id": 0,
                            "supported_price_decimals": 2,
                            "supported_size_decimals": 4
                        }
                    ]
                }));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let (symbol, market_id) = connector
            .resolve_market_id_and_symbol()
            .await
            .expect("market resolution");
        assert_eq!(symbol, "ETH");
        assert_eq!(market_id, 0);
    }

    #[tokio::test]
    async fn lighter_place_order_rejects_client_order_id_above_u48() {
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(2),
            account_index: Some(718392),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some("http://127.0.0.1:9".to_string()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let err = connector
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "LIGHTER".to_string(),
                side: Side::Buy,
                price: 2081.57,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: (LIGHTER_CLIENT_ORDER_INDEX_MAX + 1).to_string(),
            })
            .await
            .expect_err("place should fail");
        assert_eq!(err.kind, LiveGatewayErrorKind::Fatal);
        assert!(err.message.contains("exceeds uint48 max"));
    }

    #[test]
    fn lighter_err_with_context_preserves_kind_and_appends_context() {
        let err = err_with_context(
            LiveGatewayError::fatal("{\"code\":20001,\"message\":\"invalid param \"}"),
            "lighter place context market_id=0 client_order_index=42 price=208157 base_amount=100 tif=Gtc post_only=true reduce_only=false",
        );
        assert_eq!(err.kind, LiveGatewayErrorKind::Fatal);
        assert!(err.message.contains("\"code\":20001"));
        assert!(err.message.contains("market_id=0"));
        assert!(err.message.contains("client_order_index=42"));
        assert!(err.message.contains("price=208157"));
        assert!(err.message.contains("base_amount=100"));
    }

    #[test]
    fn lighter_private_fill_uses_connector_venue_identity() {
        let text = serde_json::json!({
            "type": "fill",
            "seq": 44,
            "ts": 1_700_000_123_456i64,
            "order_id": "oid-123",
            "client_order_id": "coid-456",
            "fill_id": "fill-789",
            "side": "sell",
            "price": 2081.57,
            "size": 0.01,
            "purpose": "Mm",
            "fee_bps": 1.5
        })
        .to_string();
        let event = translate_private_event(&text, 3, "lighter").expect("fill");
        match event {
            ExecutionEvent::Filled(fill) => {
                assert_eq!(fill.venue_index, 3);
                assert_eq!(fill.venue_id, "lighter");
                assert_eq!(fill.seq, 44);
                assert_eq!(fill.timestamp_ms, 1_700_000_123_456);
                assert_eq!(fill.order_id.as_deref(), Some("oid-123"));
                assert_eq!(fill.client_order_id.as_deref(), Some("coid-456"));
                assert_eq!(fill.fill_id.as_deref(), Some("fill-789"));
                assert_eq!(fill.side, Side::Sell);
                assert!((fill.price - 2081.57).abs() < 1e-9);
                assert!((fill.size - 0.01).abs() < 1e-9);
                assert_eq!(fill.purpose, OrderPurpose::Mm);
                assert!((fill.fee_bps - 1.5).abs() < 1e-9);
                assert!(fill.phase51_native_role.is_none());
                assert!(fill.phase51_lighter_native_limit.is_none());
            }
            other => panic!("expected fill event, got {other:?}"),
        }
    }

    #[test]
    fn lighter_private_fill_complete_native_role_fields_populate_phase51_native_role() {
        for is_maker_ask in [true, false] {
            let text = serde_json::json!({
                "type": "fill",
                "seq": 47,
                "ts": 1_700_000_123_459i64,
                "order_id": "oid-native",
                "client_order_id": "coid-native",
                "fill_id": "fill-native",
                "side": "sell",
                "price": 2081.57,
                "size": 0.01,
                "purpose": "Mm",
                "fee_bps": 1.5,
                "account_index": 123_i64,
                "is_maker_ask": is_maker_ask,
                "ask_account_id": 456_i64,
                "bid_account_id": 789_i64
            })
            .to_string();
            let event = translate_private_event(&text, 3, "lighter").expect("fill");
            match event {
                ExecutionEvent::Filled(fill) => {
                    assert_eq!(
                        fill.phase51_native_role,
                        Some(Phase51ForwardRefreshNativeRole::Lighter {
                            account_index: 123,
                            is_maker_ask,
                            ask_account_id: 456,
                            bid_account_id: 789,
                        })
                    );
                    assert!(fill.phase51_lighter_native_limit.is_none());
                }
                other => panic!("expected fill event, got {other:?}"),
            }
        }
    }

    #[test]
    fn lighter_account_all_trades_emit_target_linkable_source_owner_fill_for_account_side() {
        for (account_is_ask, is_maker_ask) in [(true, true), (false, false)] {
            let text = serde_json::json!({
                "type": "update/account_all_trades",
                "channel": "account_all_trades/123",
                "trades": [{
                    "trade_id": 99_u64,
                    "timestamp": 1_700_000_123_462i64,
                    "ask_id_str": "ask-order",
                    "bid_id_str": "bid-order",
                    "ask_client_id_str": "ask-client",
                    "bid_client_id_str": "bid-client",
                    "ask_account_id": if account_is_ask { 123_i64 } else { 456_i64 },
                    "bid_account_id": if account_is_ask { 456_i64 } else { 123_i64 },
                    "is_maker_ask": is_maker_ask,
                }]
            })
            .to_string();

            let events = translate_private_events(&text, 3, "lighter", Some(123));
            assert_eq!(events.len(), 1);
            match &events[0] {
                ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(fill) => {
                    assert_eq!(fill.venue_index, 3);
                    assert_eq!(fill.venue_id, "lighter");
                    assert_eq!(fill.seq, 1_700_000_123_462_000);
                    assert_eq!(fill.timestamp_ms, 1_700_000_123_462);
                    assert_eq!(
                        fill.order_id(),
                        Some(if account_is_ask {
                            "ask-order"
                        } else {
                            "bid-order"
                        })
                    );
                    assert_eq!(
                        fill.client_order_id(),
                        Some(if account_is_ask {
                            "ask-client"
                        } else {
                            "bid-client"
                        })
                    );
                    assert_eq!(
                        fill.phase51_native_role,
                        Some(Phase51ForwardRefreshNativeRole::Lighter {
                            account_index: 123,
                            is_maker_ask,
                            ask_account_id: if account_is_ask { 123 } else { 456 },
                            bid_account_id: if account_is_ask { 456 } else { 123 },
                        })
                    );
                    assert!(fill.phase51_target_key.is_none());
                    assert!(fill.phase51_lighter_native_limit.is_none());
                    assert!(fill.phase51_source_owner_pfill_observation.is_none());
                }
                other => panic!("expected source-owner fill, got {other:?}"),
            }
        }
    }

    #[test]
    fn lighter_account_all_trades_populates_pfill_observation_only_from_explicit_fields() {
        let text = serde_json::json!({
            "type": "update/account_all_trades",
            "channel": "account_all_trades/123",
            "trades": [{
                "timestamp": 1_700_000_123_466i64,
                "side": "sell",
                "price": "2100.50",
                "size": "0.01",
                "ask_id_str": "ask-order",
                "ask_client_id_str": "ask-client",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }]
        })
        .to_string();

        let events = translate_phase51_account_all_trades_source_owner_events(
            &text,
            3,
            "lighter",
            Some(123),
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(fill) => {
                let observation = fill
                    .phase51_source_owner_pfill_observation
                    .as_ref()
                    .expect("explicit pfill observation");
                assert_eq!(observation.source_event_type, "LIGHTER_TRADES_JSON");
                assert_eq!(observation.side, Side::Sell);
                assert!((observation.price - 2_100.5).abs() < 1e-9);
                assert!((observation.size - 0.01).abs() < 1e-12);
                assert_eq!(observation.event_time_ms, 1_700_000_123_466);
                assert_eq!(observation.fill_count, 1);
                assert_eq!(observation.outcome_status, "OBSERVED_FILLED");
                assert_eq!(observation.p_fill_outcome, 1.0);
            }
            other => panic!("expected source-owner fill, got {other:?}"),
        }
    }

    #[test]
    fn lighter_account_all_trades_missing_or_mismatched_pfill_fields_leave_observation_none() {
        for trade in [
            serde_json::json!({
                "timestamp": 1_700_000_123_467i64,
                "price": "2100.50",
                "size": "0.01",
                "ask_id_str": "ask-order",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }),
            serde_json::json!({
                "timestamp": 1_700_000_123_467i64,
                "side": "sell",
                "size": "0.01",
                "ask_id_str": "ask-order",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }),
            serde_json::json!({
                "timestamp": 1_700_000_123_467i64,
                "side": "sell",
                "price": "2100.50",
                "ask_id_str": "ask-order",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }),
            serde_json::json!({
                "side": "sell",
                "price": "2100.50",
                "size": "0.01",
                "ask_id_str": "ask-order",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }),
            serde_json::json!({
                "timestamp": 1_700_000_123_467i64,
                "side": "buy",
                "price": "2100.50",
                "size": "0.01",
                "ask_id_str": "ask-order",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }),
        ] {
            let text = serde_json::json!({
                "type": "update/account_all_trades",
                "channel": "account_all_trades/123",
                "trades": [trade],
            })
            .to_string();
            let events = translate_phase51_account_all_trades_source_owner_events(
                &text,
                3,
                "lighter",
                Some(123),
            );
            assert_eq!(events.len(), 1);
            match &events[0] {
                ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(fill) => {
                    assert!(fill.phase51_source_owner_pfill_observation.is_none());
                }
                other => panic!("expected source-owner fill, got {other:?}"),
            }
        }
    }

    #[test]
    fn lighter_account_all_trades_source_owner_path_ignores_generic_private_events() {
        let generic_fill = serde_json::json!({
            "type": "fill",
            "seq": 48,
            "ts": 1_700_000_123_460i64,
            "order_id": "raw-order",
            "client_order_id": "raw-client",
            "fill_id": "raw-fill",
            "side": "buy",
            "price": 2200.0,
            "size": 0.01,
            "purpose": "mm",
            "account_index": 123_i64,
            "is_maker_ask": true,
            "ask_account_id": 123_i64,
            "bid_account_id": 456_i64,
        })
        .to_string();
        assert!(matches!(
            translate_private_events(&generic_fill, 3, "lighter", Some(123)).first(),
            Some(ExecutionEvent::Filled(_))
        ));
        assert!(translate_phase51_account_all_trades_source_owner_events(
            &generic_fill,
            3,
            "lighter",
            Some(123)
        )
        .is_empty());

        let account_all_trades = serde_json::json!({
            "type": "update/account_all_trades",
            "channel": "account_all_trades/123",
            "trades": [{
                "timestamp": 1_700_000_123_462i64,
                "ask_id_str": "ask-order",
                "ask_client_id_str": "ask-client",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }]
        })
        .to_string();
        assert!(matches!(
            translate_phase51_account_all_trades_source_owner_events(
                &account_all_trades,
                3,
                "lighter",
                Some(123)
            )
            .first(),
            Some(ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(_))
        ));
    }

    #[test]
    fn lighter_account_all_trades_require_exact_native_role_fields_and_account_side() {
        let missing_role = serde_json::json!({
            "type": "update/account_all_trades",
            "channel": "account_all_trades:123",
            "trades": [{
                "trade_id": 100_u64,
                "timestamp": 1_700_000_123_463i64,
                "ask_id_str": "ask-order",
                "ask_client_id_str": "ask-client",
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
            }]
        })
        .to_string();
        assert!(translate_private_events(&missing_role, 3, "lighter", Some(123)).is_empty());

        let unrelated_account = serde_json::json!({
            "type": "update/account_all_trades",
            "channel": "account_all_trades:123",
            "trades": [{
                "trade_id": 101_u64,
                "timestamp": 1_700_000_123_464i64,
                "ask_id_str": "ask-order",
                "ask_client_id_str": "ask-client",
                "ask_account_id": 456_i64,
                "bid_account_id": 789_i64,
                "is_maker_ask": true,
            }]
        })
        .to_string();
        assert!(translate_private_events(&unrelated_account, 3, "lighter", Some(123)).is_empty());

        let missing_handle = serde_json::json!({
            "type": "update/account_all_trades",
            "channel": "account_all_trades/123",
            "trades": [{
                "timestamp": 1_700_000_123_465i64,
                "ask_account_id": 123_i64,
                "bid_account_id": 456_i64,
                "is_maker_ask": true,
            }]
        })
        .to_string();
        assert!(translate_private_events(&missing_handle, 3, "lighter", Some(123)).is_empty());
    }

    #[test]
    fn lighter_private_fill_incomplete_native_role_fields_leave_none() {
        for missing_field in [
            "account_index",
            "is_maker_ask",
            "ask_account_id",
            "bid_account_id",
        ] {
            let mut payload = serde_json::json!({
                "type": "fill",
                "seq": 48,
                "ts": 1_700_000_123_460i64,
                "order_id": "oid-incomplete",
                "client_order_id": "coid-incomplete",
                "fill_id": "fill-incomplete",
                "side": "sell",
                "price": 2081.57,
                "size": 0.01,
                "purpose": "Mm",
                "account_index": 123_i64,
                "is_maker_ask": true,
                "ask_account_id": 456_i64,
                "bid_account_id": 789_i64
            });
            payload
                .as_object_mut()
                .expect("object")
                .remove(missing_field);

            let event = translate_private_event(&payload.to_string(), 3, "lighter").expect("fill");
            match event {
                ExecutionEvent::Filled(fill) => {
                    assert!(fill.phase51_native_role.is_none());
                    assert!(fill.phase51_lighter_native_limit.is_none());
                }
                other => panic!("expected fill event, got {other:?}"),
            }
        }
    }

    #[test]
    fn lighter_private_fill_non_exact_native_role_types_leave_none() {
        for (field, value) in [
            ("account_index", serde_json::json!("123")),
            ("account_index", serde_json::json!(123.5)),
            ("is_maker_ask", serde_json::json!("true")),
            ("ask_account_id", serde_json::json!("456")),
            ("ask_account_id", serde_json::json!(456.5)),
            ("bid_account_id", serde_json::json!("789")),
            ("bid_account_id", serde_json::json!(789.5)),
            ("is_maker_ask", serde_json::json!(1)),
        ] {
            let mut payload = serde_json::json!({
                "type": "fill",
                "seq": 49,
                "ts": 1_700_000_123_461i64,
                "order_id": "oid-non-exact",
                "client_order_id": "coid-non-exact",
                "fill_id": "fill-non-exact",
                "side": "sell",
                "price": 2081.57,
                "size": 0.01,
                "purpose": "Mm",
                "account_index": 123_i64,
                "is_maker_ask": true,
                "ask_account_id": 456_i64,
                "bid_account_id": 789_i64
            });
            payload
                .as_object_mut()
                .expect("object")
                .insert(field.to_string(), value);

            let event = translate_private_event(&payload.to_string(), 3, "lighter").expect("fill");
            match event {
                ExecutionEvent::Filled(fill) => {
                    assert!(fill.phase51_native_role.is_none());
                    assert!(fill.phase51_lighter_native_limit.is_none());
                }
                other => panic!("expected fill event, got {other:?}"),
            }
        }
    }

    #[test]
    fn lighter_private_fill_order_fields_do_not_create_native_role() {
        let text = serde_json::json!({
            "type": "fill",
            "seq": 50,
            "ts": 1_700_000_123_462i64,
            "order_id": "maker-looking-order",
            "client_order_id": "coid-maker-looking",
            "fill_id": "fill-maker-looking",
            "side": "sell",
            "price": 2081.57,
            "size": 0.01,
            "purpose": "Mm",
            "fee_bps": -1.0
        })
        .to_string();
        let event = translate_private_event(&text, 3, "lighter").expect("fill");
        match event {
            ExecutionEvent::Filled(fill) => {
                assert!(fill.phase51_native_role.is_none());
                assert!(fill.phase51_lighter_native_limit.is_none());
            }
            other => panic!("expected fill event, got {other:?}"),
        }
    }

    #[test]
    fn lighter_private_events_accept_uppercase_side_variants() {
        let ack = serde_json::json!({
            "type": "order_ack",
            "seq": 45,
            "ts": 1_700_000_123_457i64,
            "order_id": "oid-ack",
            "client_order_id": "coid-ack",
            "side": "BUY",
            "price": 2080.0,
            "size": 0.01,
            "purpose": "Mm"
        })
        .to_string();
        let event = translate_private_event(&ack, 3, "lighter").expect("order ack");
        match event {
            ExecutionEvent::OrderAccepted(ack) => {
                assert_eq!(ack.side, Side::Buy);
            }
            other => panic!("expected order ack event, got {other:?}"),
        }

        let fill = serde_json::json!({
            "type": "fill",
            "seq": 46,
            "ts": 1_700_000_123_458i64,
            "order_id": "oid-fill",
            "client_order_id": "coid-fill",
            "fill_id": "fill-790",
            "side": "SELL",
            "price": 2082.0,
            "size": 0.01,
            "purpose": "Mm"
        })
        .to_string();
        let event = translate_private_event(&fill, 3, "lighter").expect("fill");
        match event {
            ExecutionEvent::Filled(fill) => {
                assert_eq!(fill.side, Side::Sell);
            }
            other => panic!("expected fill event, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn lighter_cancel_order_calls_signer_then_sendtx() {
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "BTC-USD",
                            "market_id": 7,
                            "price_decimals": 2,
                            "size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"cancel_order\"")
                    .body_contains("\"account_index\":123")
                    .body_contains("\"api_key_index\":1")
                    .body_contains("\"market_index\":7")
                    .body_contains("\"client_order_index\":55");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":15,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=15")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200).json_body(serde_json::json!({}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestCancelRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            order_id: "55".to_string(),
        };
        let resp = connector.cancel_order(req).await.expect("cancel");
        assert!(resp.order_id.is_none());
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_cancel_order_uses_exchange_order_index_for_large_ids() {
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "BTC",
                            "market_id": 7,
                            "supported_price_decimals": 2,
                            "supported_size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let large_order_index = LIGHTER_CLIENT_ORDER_INDEX_MAX + 1;
        let sign = signer
            .mock_async(move |when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"cancel_order\"")
                    .body_contains("\"account_index\":123")
                    .body_contains("\"api_key_index\":1")
                    .body_contains("\"market_index\":7")
                    .body_contains(&format!("\"order_index\":{}", large_order_index));
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":15,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=15")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200).json_body(serde_json::json!({}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestCancelRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            order_id: large_order_index.to_string(),
        };
        let resp = connector.cancel_order(req).await.expect("cancel");
        assert!(resp.order_id.is_none());
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn phase51_lighter_strict_maker_only_sanitizes_cancel_sendtx_failure() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[
            "LIGHTER_MARKET_ID",
            "LIGHTER_MARKET",
            PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV,
        ]);
        unset_env("LIGHTER_MARKET_ID");
        unset_env("LIGHTER_MARKET");
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "yes");
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "BTC-USD",
                            "market_id": 7,
                            "price_decimals": 2,
                            "size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"cancel_order\"");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":15,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST).path("/api/v1/sendTx");
                then.status(429).body(
                    "rate limit response with raw order_id, client_order_id, token, signature",
                );
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestCancelRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            order_id: "55".to_string(),
        };

        let err = connector
            .cancel_order(req)
            .await
            .expect_err("strict-mode cancel sendTx failure should be sanitized");
        assert_eq!(err.kind, LiveGatewayErrorKind::RateLimited);
        assert!(err.message.contains("strict maker-only observation"));
        assert!(err.message.contains("cancel submit_sendtx"));
        assert!(!err.message.contains("raw order_id"));
        assert!(!err.message.contains("client_order_id"));
        assert!(!err.message.contains("token"));
        assert!(!err.message.contains("signature"));
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_replace_order_calls_signer_then_sendtx() {
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "BTC-USD",
                            "market_id": 7,
                            "price_decimals": 2,
                            "size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"modify_order\"")
                    .body_contains("\"account_index\":123")
                    .body_contains("\"api_key_index\":1")
                    .body_contains("\"market_index\":7")
                    .body_contains("\"client_order_index\":55")
                    .body_contains("\"price\":10022")
                    .body_contains("\"base_amount\":1250");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":17,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=17")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200).json_body(serde_json::json!({}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestReplaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            order_id: "55".to_string(),
            side: Side::Buy,
            price: 100.22,
            size: 1.25,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: "77".to_string(),
        };
        let resp = connector.replace_order(req).await.expect("replace");
        assert_eq!(resp.order_id.as_deref(), Some("55"));
        assert_eq!(resp.client_order_id.as_deref(), Some("77"));
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_replace_order_uses_exchange_order_index_for_large_ids() {
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let orderbooks = api
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "order_books": [
                        {
                            "symbol": "BTC-USD",
                            "market_id": 7,
                            "price_decimals": 2,
                            "size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let large_order_index = LIGHTER_CLIENT_ORDER_INDEX_MAX + 99;
        let sign = signer
            .mock_async(move |when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"modify_order\"")
                    .body_contains(&format!("\"order_index\":{}", large_order_index));
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":17,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=17")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200).json_body(serde_json::json!({}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestReplaceRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            order_id: large_order_index.to_string(),
            side: Side::Sell,
            price: 100.24,
            size: 1.25,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: "78".to_string(),
        };
        let resp = connector.replace_order(req).await.expect("replace");
        let expected_order_id = large_order_index.to_string();
        assert_eq!(resp.order_id.as_deref(), Some(expected_order_id.as_str()));
        assert_eq!(resp.client_order_id.as_deref(), Some("78"));
        orderbooks.assert_hits_async(1).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_replace_order_rejects_non_mm_or_reduce_only_paths() {
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            market: "ETH-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some("http://127.0.0.1:9".to_string()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);

        let err = connector
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "LIGHTER".to_string(),
                order_id: "55".to_string(),
                side: Side::Buy,
                price: 2081.57,
                size: 0.01,
                purpose: OrderPurpose::Exit,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: "79".to_string(),
            })
            .await
            .expect_err("replace should fail");
        assert_eq!(err.kind, LiveGatewayErrorKind::Fatal);
        assert!(err
            .message
            .contains("native replace requires mm gtc post_only non_reduce_only"));
    }

    #[tokio::test]
    async fn lighter_cancel_all_calls_signer_then_sendtx() {
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"cancel_all\"")
                    .body_contains("\"account_index\":123")
                    .body_contains("\"api_key_index\":1")
                    .body_contains("\"cancel_all_time\":0");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":16,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("tx_type=16")
                    .body_contains("tx_info=%7B%22signed%22%3Atrue%7D");
                then.status(200).json_body(serde_json::json!({}));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestCancelAllRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
        };
        let resp = connector.cancel_all(req).await.expect("cancel_all");
        assert!(resp.order_id.is_none());
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn phase51_lighter_strict_maker_only_sanitizes_cancel_all_sendtx_failure() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _env = EnvGuard::new(&[PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV]);
        set_env(PHASE51_LIGHTER_STRICT_MAKER_ONLY_OBSERVATION_ENV, "1");
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"cancel_all\"");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":16,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST).path("/api/v1/sendTx");
                then.status(500).body(
                    "temporary response with raw order_id, client_order_id, token, signature",
                );
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: api.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: false,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: None,
            nonce_path: None,
            signer_url: Some(signer.base_url()),
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let req = LiveRestCancelAllRequest {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
        };

        let err = connector
            .cancel_all(req)
            .await
            .expect_err("strict-mode cancel-all sendTx failure should be sanitized");
        assert_eq!(err.kind, LiveGatewayErrorKind::Retryable);
        assert!(err.message.contains("strict maker-only observation"));
        assert!(err.message.contains("cancel_all submit_sendtx"));
        assert!(!err.message.contains("raw order_id"));
        assert!(!err.message.contains("client_order_id"));
        assert!(!err.message.contains("token"));
        assert!(!err.message.contains("signature"));
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_market_id_by_symbol() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("LIGHTER_MARKET_ID");
        std::env::set_var("LIGHTER_MARKET", "BTC-USD-PERP");
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "code": 200,
                    "order_books": [
                        {"symbol":"BTC","market_id":1},
                        {"symbol":"ETH","market_id":2}
                    ]
                }));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let (symbol, market_id) = connector
            .resolve_market_id_and_symbol()
            .await
            .expect("resolve");
        assert_eq!(symbol, "BTC");
        assert_eq!(market_id, 1);
        std::env::remove_var("LIGHTER_MARKET");
    }

    #[tokio::test]
    async fn resolve_market_id_is_cached_after_first_lookup() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("LIGHTER_MARKET_ID");
        std::env::set_var("LIGHTER_MARKET", "BTC-USD");
        let server = MockServer::start_async().await;
        let orderbooks = server
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "code": 200,
                    "order_books": [
                        {
                            "symbol": "BTC-USD",
                            "market_id": 7,
                            "price_decimals": 2,
                            "size_decimals": 3
                        }
                    ]
                }));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);

        let first = connector
            .resolve_market_id_and_symbol()
            .await
            .expect("first resolve");
        let second = connector
            .resolve_market_id_and_symbol()
            .await
            .expect("second resolve");
        let decimals = connector
            .resolve_market_decimals(first.1)
            .await
            .expect("decimals");

        assert_eq!(first, ("BTC-USD".to_string(), 7));
        assert_eq!(second, first);
        assert_eq!(decimals, (2, 3));
        orderbooks.assert_hits_async(1).await;
        std::env::remove_var("LIGHTER_MARKET");
    }

    #[tokio::test]
    async fn resolve_market_id_from_env() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("LIGHTER_MARKET_ID", "2");
        std::env::remove_var("LIGHTER_MARKET");
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/api/v1/orderBooks");
                then.status(200).json_body(serde_json::json!({
                    "code": 200,
                    "order_books": [
                        {"symbol":"BTC","market_id":1},
                        {"symbol":"ETH","market_id":2}
                    ]
                }));
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = LighterConnector::new(cfg, market_tx, exec_tx);
        let (symbol, market_id) = connector
            .resolve_market_id_and_symbol()
            .await
            .expect("resolve");
        assert_eq!(symbol, "ETH");
        assert_eq!(market_id, 2);
        std::env::remove_var("LIGHTER_MARKET_ID");
    }

    #[test]
    fn order_book_snapshot_uses_timestamp() {
        let value = serde_json::json!({
            "type": "update/order_book",
            "timestamp": 1700000000123i64,
            "order_book": {
                "bids": [{"price":"100","size":"2"}],
                "asks": [{"price":"101","size":"3"}]
            }
        });
        let mut seq = 0u64;
        let parsed = decode_order_book_snapshot(&value, 3, "LIGHTER", &mut seq).expect("snap");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.timestamp_ms, 1_700_000_000_123);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_snapshot_zero_timestamp_falls_back_to_now() {
        let value = serde_json::json!({
            "type": "update/order_book",
            "timestamp": 0i64,
            "order_book": {
                "bids": [{"price":"100","size":"2"}],
                "asks": [{"price":"101","size":"3"}]
            }
        });
        let before_ms = now_timestamp_ms_nonzero();
        let mut seq = 0u64;
        let parsed = decode_order_book_snapshot(&value, 3, "LIGHTER", &mut seq).expect("snap");
        let after_ms = now_timestamp_ms_nonzero();
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert!(snapshot.timestamp_ms > 0);
                assert!(snapshot.timestamp_ms >= before_ms);
                assert!(snapshot.timestamp_ms <= after_ms + 2_000);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_snapshot_allows_empty_bids() {
        let value = serde_json::json!({
            "type": "update/order_book",
            "order_book": {
                "bids": [],
                "asks": [{"price":"101.0","size":"3.0"}]
            }
        });
        let mut seq = 0u64;
        let parsed = decode_order_book_snapshot(&value, 3, "LIGHTER", &mut seq).expect("snap");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert!(snapshot.bids.is_empty());
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_snapshot_allows_empty_asks() {
        let value = serde_json::json!({
            "type": "update/order_book",
            "order_book": {
                "bids": [{"price":"100.0","size":"2.0"}],
                "asks": []
            }
        });
        let mut seq = 0u64;
        let parsed = decode_order_book_snapshot(&value, 3, "LIGHTER", &mut seq).expect("snap");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 1);
                assert!(snapshot.asks.is_empty());
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_snapshot_allows_zero_size_level() {
        let value = serde_json::json!({
            "type": "update/order_book",
            "order_book": {
                "bids": [{"price":"100.0","size":"0.00000"}],
                "asks": [{"price":"101.0","size":"1.00000"}]
            }
        });
        let mut seq = 0u64;
        let parsed = decode_order_book_snapshot(&value, 3, "LIGHTER", &mut seq).expect("snap");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 1);
                assert_eq!(snapshot.bids[0].size, 0.0);
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_channel_snapshot_allows_empty_asks() {
        let value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 42,
            "timestamp": 1700000000123i64,
            "order_book": {
                "bids": [{"price":"100.0","size":"2.0"}],
                "asks": []
            }
        });
        let mut seq = 0u64;
        let parsed =
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq).expect("snapshot");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 1);
                assert_eq!(snapshot.asks.len(), 0);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_channel_snapshot_allows_empty_bids() {
        let value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 43,
            "timestamp": 1700000000456i64,
            "order_book": {
                "bids": [],
                "asks": [{"price":"101.0","size":"3.0"}]
            }
        });
        let mut seq = 0u64;
        let parsed =
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq).expect("snapshot");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 0);
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn order_book_channel_snapshot_allows_zero_size() {
        let value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 44,
            "timestamp": 1700000000789i64,
            "order_book": {
                "bids": [{"price":"100.0","size":"0.00000"}],
                "asks": [{"price":"101.0","size":"1.00000"}]
            }
        });
        let mut seq = 0u64;
        let parsed =
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq).expect("snapshot");
        match parsed.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 1);
                assert_eq!(snapshot.bids[0].size, 0.0);
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn json_ping_triggers_pong_response() {
        let value = serde_json::json!({
            "type": "ping"
        });
        assert_eq!(json_ping_response(&value), Some(r#"{"type":"pong"}"#));
    }

    #[test]
    fn order_book_channel_snapshot_decoder_emits_snapshot() {
        // First subscription snapshot with full book
        let first_value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 100,
            "timestamp": 1700000000123i64,
            "order_book": {
                "bids": [{"price":"100.0","size":"2.0"}],
                "asks": [{"price":"101.0","size":"3.0"}]
            }
        });
        // Another fresh snapshot used to verify full replacement semantics
        let second_value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 101,
            "timestamp": 1700000000456i64,
            "order_book": {
                "bids": [],
                "asks": [{"price":"101.0","size":"1.0"}]
            }
        });
        let mut seq = 0u64;
        let first =
            decode_order_book_channel_message(&first_value, 3, "LIGHTER", &mut seq).expect("first");
        match first.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 1);
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot for first message"),
        }
        // A fresh snapshot must replace the prior book state fully.
        let second = decode_order_book_channel_message(&second_value, 3, "LIGHTER", &mut seq)
            .expect("second");
        match second.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                // Bids should be empty (not accumulated from first message)
                assert_eq!(snapshot.bids.len(), 0, "stale bids must not persist");
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot for second decode"),
        }
    }

    #[test]
    fn lighter_seq_tracker_allows_equal_seq() {
        let mut tracker = LighterSeqTracker::new();
        let base_event = MarketDataEvent::L2Delta(crate::live::types::L2Delta {
            venue_index: 0,
            venue_id: "LIGHTER".to_string(),
            seq: 7,
            timestamp_ms: 0,
            changes: Vec::new(),
        });
        let first = ParsedL2Message {
            event: base_event.clone(),
            seq: 7,
        };
        let second = ParsedL2Message {
            event: base_event,
            seq: 7,
        };
        let first_outcome = tracker.on_message(first);
        let second_outcome = tracker.on_message(second);
        assert!(first_outcome.event.is_some());
        assert!(second_outcome.event.is_some());
    }

    #[test]
    fn order_book_channel_seq_always_increments() {
        let value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 10,
            "timestamp": 1700000000123i64,
            "order_book": {
                "bids": [{"price":"100.0","size":"2.0"}],
                "asks": [{"price":"101.0","size":"3.0"}]
            }
        });
        let mut seq = 0u64;
        let first =
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq).expect("first");
        let second =
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq).expect("second");
        assert!(second.seq > first.seq);
    }

    /// Verify that `decode_order_book_channel_delta` produces L2Delta, not L2Snapshot.
    #[test]
    fn decode_channel_delta_produces_l2_delta() {
        let value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 200,
            "timestamp": 1700000001000i64,
            "order_book": {
                "code": 0,
                "bids": [{"price":"100.0","size":"5.0"}],
                "asks": [{"price":"101.0","size":"2.0"}]
            }
        });
        let mut seq = 10u64;
        let parsed = decode_order_book_channel_delta(&value, 3, "LIGHTER", &mut seq)
            .expect("should decode delta");
        assert_eq!(parsed.seq, 11);
        match parsed.event {
            MarketDataEvent::L2Delta(delta) => {
                assert_eq!(delta.venue_index, 3);
                assert_eq!(delta.timestamp_ms, 1700000001000);
                assert_eq!(delta.changes.len(), 2);
                assert_eq!(
                    delta.changes[0],
                    BookLevelDelta {
                        side: BookSide::Bid,
                        price: 100.0,
                        size: 5.0
                    }
                );
                assert_eq!(
                    delta.changes[1],
                    BookLevelDelta {
                        side: BookSide::Ask,
                        price: 101.0,
                        size: 2.0
                    }
                );
            }
            _ => panic!("expected L2Delta, got {:?}", parsed.event),
        }
    }

    /// Verify that a delta message with empty asks produces changes for bids only.
    /// This is the core fix: previously, empty asks treated as snapshot would wipe all asks.
    #[test]
    fn decode_channel_delta_empty_asks_does_not_wipe_book() {
        let value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 300,
            "timestamp": 1700000002000i64,
            "order_book": {
                "code": 0,
                "bids": [{"price":"99.0","size":"1.0"}, {"price":"98.0","size":"0.0"}],
                "asks": []
            }
        });
        let mut seq = 0u64;
        let parsed = decode_order_book_channel_delta(&value, 3, "LIGHTER", &mut seq)
            .expect("should decode delta with empty asks");
        match parsed.event {
            MarketDataEvent::L2Delta(delta) => {
                // Only bid changes — no ask changes because asks array is empty
                assert_eq!(delta.changes.len(), 2, "should have 2 bid changes only");
                assert!(
                    delta.changes.iter().all(|c| c.side == BookSide::Bid),
                    "all changes should be bids"
                );
                // Second bid has size 0.0 — this is a removal
                assert_eq!(delta.changes[1].size, 0.0, "zero-size = removal");
            }
            _ => panic!("expected L2Delta"),
        }
    }

    /// Verify that a delta message with empty bids produces changes for asks only.
    #[test]
    fn decode_channel_delta_empty_bids_preserves_asks() {
        let value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 400,
            "timestamp": 1700000003000i64,
            "order_book": {
                "code": 0,
                "bids": [],
                "asks": [{"price":"110.0","size":"3.0"}]
            }
        });
        let mut seq = 5u64;
        let parsed =
            decode_order_book_channel_delta(&value, 0, "LIGHTER", &mut seq).expect("should decode");
        match parsed.event {
            MarketDataEvent::L2Delta(delta) => {
                assert_eq!(delta.changes.len(), 1);
                assert_eq!(delta.changes[0].side, BookSide::Ask);
            }
            _ => panic!("expected L2Delta"),
        }
    }

    #[test]
    fn decode_channel_snapshot_tracks_venue_nonce_fields() {
        let value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 100,
            "timestamp": 1700000001000i64,
            "order_book": {
                "code": 0,
                "bids": [{"price":"100.0","size":"10.0"}],
                "asks": [{"price":"101.0","size":"8.0"}],
                "nonce": 9182390020u64,
                "begin_nonce": 9182389998u64
            }
        });
        let mut seq = 0u64;
        let parsed = decode_order_book_channel_message(&value, 0, "L", &mut seq)
            .expect("snapshot should decode");
        assert_eq!(parsed.seq, 9182390020);
        assert_eq!(parsed.venue_nonce, Some(9182390020));
        assert_eq!(parsed.venue_begin_nonce, Some(9182389998));
    }

    #[test]
    fn decode_channel_delta_tracks_venue_nonce_fields() {
        let value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 101,
            "timestamp": 1700000002000i64,
            "order_book": {
                "code": 0,
                "bids": [{"price":"100.0","size":"12.0"}],
                "asks": [],
                "nonce": 9182390025u64,
                "begin_nonce": 9182390020u64
            }
        });
        let mut seq = 0u64;
        let parsed =
            decode_order_book_channel_delta(&value, 0, "L", &mut seq).expect("delta should decode");
        assert_eq!(parsed.seq, 9182390025);
        assert_eq!(parsed.venue_nonce, Some(9182390025));
        assert_eq!(parsed.venue_begin_nonce, Some(9182390020));
    }

    #[test]
    fn lighter_continuity_gap_detects_begin_nonce_mismatch() {
        assert!(lighter_has_continuity_gap(Some(100), Some(99), Some(105)));
        assert!(!lighter_has_continuity_gap(Some(100), Some(100), Some(105)));
    }

    #[test]
    fn lighter_continuity_gap_detects_nonce_regression_without_begin_nonce() {
        assert!(lighter_has_continuity_gap(Some(100), None, Some(100)));
        assert!(lighter_has_continuity_gap(Some(100), None, Some(99)));
        assert!(!lighter_has_continuity_gap(Some(100), None, Some(101)));
    }

    /// Integration test: snapshot then delta flow works with OrderBookL2.
    /// Simulates the Lighter WS lifecycle: first message is snapshot, subsequent are deltas.
    #[test]
    fn snapshot_then_delta_preserves_book_integrity() {
        use crate::live::orderbook_l2::{DepthConfig, OrderBookL2};

        let mut book = OrderBookL2::new();

        // 1. Initial snapshot: full book
        let snapshot_value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 100,
            "timestamp": 1000i64,
            "order_book": {
                "code": 0,
                "bids": [
                    {"price":"100.0","size":"10.0"},
                    {"price":"99.0","size":"5.0"}
                ],
                "asks": [
                    {"price":"101.0","size":"8.0"},
                    {"price":"102.0","size":"3.0"}
                ]
            }
        });
        let mut seq = 0u64;
        let snap_parsed = decode_order_book_channel_message(&snapshot_value, 0, "L", &mut seq)
            .expect("snapshot should decode");
        match snap_parsed.event {
            MarketDataEvent::L2Snapshot(snap) => {
                book.apply_snapshot(&snap.bids, &snap.asks, snap.seq)
                    .expect("snapshot apply");
            }
            _ => panic!("expected L2Snapshot"),
        }
        let m1 = book.compute_mid_spread_depth(DepthConfig {
            levels: 10,
            include_imbalance: false,
        });
        assert!(m1.mid.is_some());
        assert!(m1.depth_near_mid > 0.0);

        // 2. Delta: bid-side only update (asks empty = no ask changes)
        let delta_value = serde_json::json!({
            "channel": "order_book:0",
            "offset": 101,
            "timestamp": 2000i64,
            "order_book": {
                "code": 0,
                "bids": [{"price":"100.0","size":"12.0"}],  // update bid size
                "asks": []  // no ask changes!
            }
        });
        let delta_parsed = decode_order_book_channel_delta(&delta_value, 0, "L", &mut seq)
            .expect("delta should decode");
        match delta_parsed.event {
            MarketDataEvent::L2Delta(delta) => {
                book.apply_delta(&delta.changes, delta.seq)
                    .expect("delta apply");
            }
            _ => panic!("expected L2Delta"),
        }
        let m2 = book.compute_mid_spread_depth(DepthConfig {
            levels: 10,
            include_imbalance: false,
        });
        // After delta with empty asks: asks should still be there (not wiped)
        assert!(
            m2.mid.is_some(),
            "mid should exist after delta with empty asks"
        );
        assert!(
            m2.depth_near_mid > 0.0,
            "depth should be non-zero — asks must be preserved"
        );
        // Verify the bid was updated
        let best_bid = book.best_bid().expect("best bid should exist");
        assert!(
            (best_bid.size - 12.0).abs() < 1e-9,
            "bid size should be updated to 12.0"
        );
        // Verify asks are still intact
        let best_ask = book
            .best_ask()
            .expect("best ask should exist after delta with empty asks");
        assert!(
            (best_ask.price - 101.0).abs() < 1e-9,
            "ask price should be 101.0"
        );
        assert!(
            (best_ask.size - 8.0).abs() < 1e-9,
            "ask size should be 8.0 (unchanged)"
        );

        // 3. Delta: ask-side removal (bids empty = no bid changes)
        let delta_value2 = serde_json::json!({
            "channel": "order_book:0",
            "offset": 102,
            "timestamp": 3000i64,
            "order_book": {
                "code": 0,
                "bids": [],
                "asks": [{"price":"102.0","size":"0.0"}]  // remove 102.0 level
            }
        });
        let delta_parsed2 = decode_order_book_channel_delta(&delta_value2, 0, "L", &mut seq)
            .expect("delta2 should decode");
        match delta_parsed2.event {
            MarketDataEvent::L2Delta(delta) => {
                book.apply_delta(&delta.changes, delta.seq)
                    .expect("delta2 apply");
            }
            _ => panic!("expected L2Delta"),
        }
        let m3 = book.compute_mid_spread_depth(DepthConfig {
            levels: 10,
            include_imbalance: false,
        });
        assert!(m3.mid.is_some(), "mid should still exist");
        // 102.0 was removed, but 101.0 still exists
        let best_ask2 = book.best_ask().expect("best ask should still exist");
        assert!((best_ask2.price - 101.0).abs() < 1e-9);
    }

    /// Funding fixture regression for Lighter mark-price settlement.
    #[test]
    fn parse_public_funding_rates_fixture() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/lighter/public_funding_rates.json"
        );
        let raw = fs::read_to_string(fixture_path).expect("fixture exists");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");

        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            market: "ETH".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 3,
            paper_mode: true,
            api_key_index: None,
            account_index: None,
            api_private_key_hex: None,
            auth_token: None,
            nonce_path: None,
            signer_url: None,
        };

        // Test: find ETH by market_id=0
        let update = parse_public_funding(&value, &cfg, 0, "ETH").expect("parse ok");
        assert_eq!(update.venue_index, 3);
        assert_eq!(update.venue_id, "LIGHTER");

        // rate = 0.00001234 (hourly), interval_sec = 3600
        // rate_8h = 0.00001234 * 8 = 0.00009872
        let rate_native = update.funding_rate_native.expect("rate_native present");
        assert!(
            (rate_native - 0.00001234).abs() < 1e-10,
            "rate_native mismatch"
        );

        let interval = update.interval_sec.expect("interval_sec present");
        assert_eq!(interval, 3600);

        let rate_8h = update.funding_rate_8h.expect("rate_8h present");
        assert!((rate_8h - 0.00009872).abs() < 1e-10, "rate_8h mismatch");

        // Test: find BTC by market_id=1
        let update_btc = parse_public_funding(&value, &cfg, 1, "BTC").expect("parse btc ok");
        let rate_btc = update_btc.funding_rate_native.expect("btc rate present");
        assert!((rate_btc - 0.00000567).abs() < 1e-10);

        // Test: find by symbol fallback (market_id=999 doesn't exist, but symbol "ETH" does)
        let update_symbol =
            parse_public_funding(&value, &cfg, 999, "ETH").expect("symbol fallback");
        let rate_symbol = update_symbol.funding_rate_native.expect("rate present");
        assert!((rate_symbol - 0.00001234).abs() < 1e-10);

        // Fix C: Lighter settles at mark price.
        assert_eq!(
            update.settlement_price_kind,
            Some(SettlementPriceKind::Mark),
            "Lighter settlement must be Mark"
        );

        // Test: non-existent market returns None
        let none_result = parse_public_funding(&value, &cfg, 999, "NOSUCH");
        assert!(none_result.is_none());
    }

    #[test]
    fn order_book_channel_stale_bids_do_not_accumulate() {
        use crate::orderbook_l2::OrderBookL2;

        // Message 1: best_bid=110, best_ask=111
        let msg1 = serde_json::json!({
            "channel": "order_book:1",
            "timestamp": 1700000000100i64,
            "order_book": {
                "bids": [{"price":"110.0","size":"1.0"}],
                "asks": [{"price":"111.0","size":"1.0"}]
            }
        });
        // Message 2: best_bid drops to 100, best_ask=101 (market moved down)
        let msg2 = serde_json::json!({
            "channel": "order_book:1",
            "timestamp": 1700000000200i64,
            "order_book": {
                "bids": [{"price":"100.0","size":"1.0"}],
                "asks": [{"price":"101.0","size":"1.0"}]
            }
        });

        let mut seq = 0u64;
        let mut book = OrderBookL2::new();

        // Apply first message
        let parsed1 =
            decode_order_book_channel_message(&msg1, 0, "LIGHTER", &mut seq).expect("msg1");
        match parsed1.event {
            MarketDataEvent::L2Snapshot(snap) => {
                book.apply_snapshot(&snap.bids, &snap.asks, snap.seq)
                    .unwrap();
            }
            _ => panic!("expected snapshot"),
        }
        assert_eq!(book.best_bid().unwrap().price, 110.0);
        assert_eq!(book.best_ask().unwrap().price, 111.0);
        let spread1 = book.best_ask().unwrap().price - book.best_bid().unwrap().price;
        assert!(spread1 > 0.0, "spread must be positive");

        // Apply second message
        let parsed2 =
            decode_order_book_channel_message(&msg2, 0, "LIGHTER", &mut seq).expect("msg2");
        match parsed2.event {
            MarketDataEvent::L2Snapshot(snap) => {
                book.apply_snapshot(&snap.bids, &snap.asks, snap.seq)
                    .unwrap();
            }
            _ => panic!("expected snapshot for second message too"),
        }

        // CRITICAL: The old bid at 110 must NOT be present.
        // If it were (due to delta semantics), we'd have:
        //   best_bid = 110 (stale), best_ask = 101 → spread = -9 (CROSSED!)
        assert_eq!(
            book.best_bid().unwrap().price,
            100.0,
            "stale bid at 110 must be gone"
        );
        assert_eq!(book.best_ask().unwrap().price, 101.0);
        let spread2 = book.best_ask().unwrap().price - book.best_bid().unwrap().price;
        assert!(spread2 > 0.0, "spread must be positive after update");
        assert_eq!(spread2, 1.0, "spread should be 101 - 100 = 1");
    }
}

fn translate_private_events(
    text: &str,
    venue_index: usize,
    venue_id: &str,
    account_index: Option<u64>,
) -> Vec<ExecutionEvent> {
    if let Some(event) = translate_private_event(text, venue_index, venue_id) {
        return vec![event];
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    translate_account_all_trades_events(&value, venue_index, venue_id, account_index)
}

fn translate_phase51_account_all_trades_source_owner_events(
    text: &str,
    venue_index: usize,
    venue_id: &str,
    account_index: Option<u64>,
) -> Vec<ExecutionEvent> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    translate_account_all_trades_events(&value, venue_index, venue_id, account_index)
}

pub fn translate_private_event(
    text: &str,
    venue_index: usize,
    venue_id: &str,
) -> Option<ExecutionEvent> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let msg_type = value.get("type")?.as_str()?;
    let seq = value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = value.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let venue_id = venue_id.to_string();
    match msg_type {
        "order_ack" => Some(ExecutionEvent::OrderAccepted(
            super::super::types::OrderAccepted {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                order_id: value.get("order_id")?.as_str()?.to_string(),
                client_order_id: value
                    .get("client_order_id")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string()),
                side: parse_side(value.get("side")?)?,
                price: value.get("price")?.as_f64()?,
                size: value.get("size")?.as_f64()?,
                purpose: parse_purpose(value.get("purpose"))?,
            },
        )),
        "cancel_ack" => Some(ExecutionEvent::CancelAccepted(
            super::super::types::CancelAccepted {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                order_id: value.get("order_id")?.as_str()?.to_string(),
            },
        )),
        "fill" => {
            let phase51_native_role = phase51_lighter_native_role_from_fill(&value);
            Some(ExecutionEvent::Filled(super::super::types::Fill {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                order_id: value
                    .get("order_id")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string()),
                client_order_id: value
                    .get("client_order_id")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string()),
                fill_id: value
                    .get("fill_id")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string()),
                phase51_target_key: None,
                phase51_native_role,
                phase51_lighter_native_limit: None,
                side: parse_side(value.get("side")?)?,
                price: value.get("price")?.as_f64()?,
                size: value.get("size")?.as_f64()?,
                purpose: parse_purpose(value.get("purpose"))?,
                fee_bps: value.get("fee_bps").and_then(|v| v.as_f64()).unwrap_or(0.0),
            }))
        }
        _ => None,
    }
}

fn translate_account_all_trades_events(
    value: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    account_index: Option<u64>,
) -> Vec<ExecutionEvent> {
    let account_index = match account_index.and_then(|idx| i64::try_from(idx).ok()) {
        Some(idx) => idx,
        None => return Vec::new(),
    };
    let msg_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let channel = value
        .get("channel")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let account_all_trades_message = msg_type.contains("account_all_trades")
        || channel.starts_with("account_all_trades:")
        || channel.starts_with("account_all_trades/");
    if !account_all_trades_message {
        return Vec::new();
    }
    let Some(trades) = value.get("trades") else {
        return Vec::new();
    };
    let mut trade_values = Vec::new();
    collect_lighter_trade_values(trades, &mut trade_values);
    trade_values
        .into_iter()
        .enumerate()
        .filter_map(|(trade_index, trade)| {
            translate_account_all_trade(trade, venue_index, venue_id, account_index, trade_index)
        })
        .collect()
}

fn collect_lighter_trade_values<'a>(
    value: &'a serde_json::Value,
    out: &mut Vec<&'a serde_json::Value>,
) {
    if let Some(items) = value.as_array() {
        for item in items {
            collect_lighter_trade_values(item, out);
        }
        return;
    }
    let Some(object) = value.as_object() else {
        return;
    };
    if object.contains_key("is_maker_ask")
        && object.contains_key("ask_account_id")
        && object.contains_key("bid_account_id")
    {
        out.push(value);
        return;
    }
    for item in object.values() {
        collect_lighter_trade_values(item, out);
    }
}

fn translate_account_all_trade(
    trade: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    account_index: i64,
    trade_index: usize,
) -> Option<ExecutionEvent> {
    let ask_account_id = phase51_lighter_i64_field(trade, "ask_account_id")?;
    let bid_account_id = phase51_lighter_i64_field(trade, "bid_account_id")?;
    let is_maker_ask = trade.get("is_maker_ask")?.as_bool()?;
    let account_is_ask = ask_account_id == account_index;
    let account_is_bid = bid_account_id == account_index;
    if account_is_ask == account_is_bid {
        return None;
    }
    let (order_id, client_order_id) = if account_is_ask {
        (
            phase51_lighter_string_handle(trade, "ask_id_str")
                .or_else(|| phase51_lighter_string_handle(trade, "ask_id")),
            phase51_lighter_string_handle(trade, "ask_client_id_str")
                .or_else(|| phase51_lighter_string_handle(trade, "ask_client_id")),
        )
    } else {
        (
            phase51_lighter_string_handle(trade, "bid_id_str")
                .or_else(|| phase51_lighter_string_handle(trade, "bid_id")),
            phase51_lighter_string_handle(trade, "bid_client_id_str")
                .or_else(|| phase51_lighter_string_handle(trade, "bid_client_id")),
        )
    };
    if order_id.is_none() && client_order_id.is_none() {
        return None;
    }
    let native_role = Phase51ForwardRefreshNativeRole::Lighter {
        account_index,
        is_maker_ask,
        ask_account_id,
        bid_account_id,
    };
    let explicit_timestamp_ms = phase51_lighter_i64_field(trade, "timestamp")
        .or_else(|| phase51_lighter_i64_field(trade, "transaction_time"));
    let timestamp_ms = explicit_timestamp_ms.unwrap_or(0);
    let seq = phase51_lighter_source_owner_seq(timestamp_ms, trade_index);
    let pfill_observation = phase51_lighter_source_owner_pfill_observation(
        trade,
        account_is_ask,
        explicit_timestamp_ms,
    );
    let mut source_owner_fill = Phase51ForwardRefreshSourceOwnerFill::new(
        venue_index,
        venue_id,
        seq,
        timestamp_ms,
        order_id,
        client_order_id,
        Some(native_role),
    );
    if let Some(observation) = pfill_observation {
        source_owner_fill.set_phase51_source_owner_pfill_observation(observation);
    }
    Some(ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
        source_owner_fill,
    ))
}

fn phase51_lighter_source_owner_seq(timestamp_ms: i64, trade_index: usize) -> u64 {
    let base = u64::try_from(timestamp_ms.max(0)).unwrap_or(0);
    base.saturating_mul(1_000)
        .saturating_add(u64::try_from(trade_index).unwrap_or(u64::MAX).min(999))
}

fn phase51_lighter_i64_field(value: &serde_json::Value, key: &str) -> Option<i64> {
    let raw = value.get(key)?;
    if let Some(val) = raw.as_i64() {
        return Some(val);
    }
    let val = raw.as_u64()?;
    i64::try_from(val).ok()
}

fn phase51_lighter_string_handle(value: &serde_json::Value, key: &str) -> Option<String> {
    let raw = value.get(key)?;
    if let Some(text) = raw.as_str() {
        let trimmed = text.trim();
        return (!trimmed.is_empty()).then(|| trimmed.to_string());
    }
    if let Some(val) = raw.as_u64() {
        return Some(val.to_string());
    }
    raw.as_i64()
        .filter(|val| *val >= 0)
        .map(|val| val.to_string())
}

fn phase51_lighter_source_owner_pfill_observation(
    trade: &serde_json::Value,
    account_is_ask: bool,
    explicit_timestamp_ms: Option<i64>,
) -> Option<Phase51ForwardRefreshSourceOwnerPfillObservation> {
    let side = parse_side(trade.get("side")?)?;
    let expected_side = if account_is_ask {
        Side::Sell
    } else {
        Side::Buy
    };
    if side != expected_side {
        return None;
    }
    let price = phase51_lighter_positive_f64_field(trade, "price")?;
    let size = phase51_lighter_positive_f64_field(trade, "size")?;
    Phase51ForwardRefreshSourceOwnerPfillObservation::lighter_trade_observed_fill(
        side,
        price,
        size,
        explicit_timestamp_ms?,
    )
}

fn phase51_lighter_positive_f64_field(value: &serde_json::Value, key: &str) -> Option<f64> {
    let parsed = parse_f64_value(value.get(key)?)?;
    parsed
        .is_finite()
        .then_some(parsed)
        .filter(|val| *val > 0.0)
}

fn phase51_lighter_native_role_from_fill(
    value: &serde_json::Value,
) -> Option<Phase51ForwardRefreshNativeRole> {
    Some(Phase51ForwardRefreshNativeRole::Lighter {
        account_index: value.get("account_index")?.as_i64()?,
        is_maker_ask: value.get("is_maker_ask")?.as_bool()?,
        ask_account_id: value.get("ask_account_id")?.as_i64()?,
        bid_account_id: value.get("bid_account_id")?.as_i64()?,
    })
}

fn parse_side(value: &serde_json::Value) -> Option<Side> {
    let raw = value.as_str()?;
    if raw.eq_ignore_ascii_case("buy") || raw == "B" {
        Some(Side::Buy)
    } else if raw.eq_ignore_ascii_case("sell") || raw == "S" {
        Some(Side::Sell)
    } else {
        None
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

async fn fetch_account_snapshot(
    client: &Client,
    cfg: &LighterConfig,
) -> anyhow::Result<AccountEvent> {
    if !cfg.has_auth() {
        return Err(anyhow::anyhow!(
            "lighter: missing auth (set LIGHTER_API_KEY_INDEX, LIGHTER_ACCOUNT_INDEX, LIGHTER_API_PRIVATE_KEY_HEX)"
        ));
    }
    let account_index = cfg
        .account_index
        .ok_or_else(|| anyhow::anyhow!("lighter: missing auth account_index"))?;
    let query = LighterAccountQuery {
        by: "index",
        value: account_index.to_string(),
    };
    let mut req = client.get(account_url(cfg)).query(&query);
    if let Some(token) = cfg.auth_token.as_ref() {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("lighter account snapshot http {}: {}", status, body);
    }
    let value: serde_json::Value = serde_json::from_str(&body)?;
    let mut snapshot = parse_account_snapshot_with_meta(&value, &cfg.venue_id, cfg.venue_index)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "invalid account snapshot: {}",
                summarize_account_payload(&value)
            )
        })?;
    // Treat freshness as the time we successfully polled Lighter. The account
    // payload timestamps reflect exchange-side mutation times and can remain
    // unchanged across repeated polls, which would otherwise make fresh account
    // truth look stale downstream.
    snapshot.timestamp_ms = now_timestamp_ms_nonzero();
    maybe_log_lighter_account_snapshot(&snapshot, &value);
    Ok(AccountEvent::Snapshot(snapshot))
}

/// Rate-limit log state for funding fetch errors.
static LIGHTER_FUNDING_ERROR_LOG_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn should_log_account_event(last_log_ms: &AtomicU64, log_count: &AtomicU64, now_ms: u64) -> bool {
    let count = log_count.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 3 {
        last_log_ms.store(now_ms, Ordering::Relaxed);
        return true;
    }
    let last = last_log_ms.load(Ordering::Relaxed);
    if now_ms.saturating_sub(last) >= LIGHTER_ACCOUNT_LOG_INTERVAL_MS {
        last_log_ms.store(now_ms, Ordering::Relaxed);
        return true;
    }
    false
}

fn summarize_account_payload(value: &serde_json::Value) -> String {
    let top_keys = value
        .as_object()
        .map(|obj| {
            let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
            keys.sort();
            keys.join(",")
        })
        .unwrap_or_else(|| "non_object".to_string());
    let account = value
        .get("accounts")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .unwrap_or(value);
    let account_keys = account
        .as_object()
        .map(|obj| {
            let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
            keys.sort();
            keys.join(",")
        })
        .unwrap_or_else(|| "non_object".to_string());
    format!("top_keys=[{top_keys}] account_keys=[{account_keys}]")
}

fn maybe_log_lighter_account_snapshot(snapshot: &AccountSnapshot, value: &serde_json::Value) {
    let positions = snapshot
        .positions
        .iter()
        .filter(|pos| pos.size.abs() > 1e-9)
        .map(|pos| format!("{}:{:.4}", pos.symbol, pos.size))
        .collect::<Vec<_>>();
    let account = value
        .get("accounts")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .unwrap_or(value);
    let total_order_count = account
        .get("total_order_count")
        .and_then(parse_i64_value)
        .unwrap_or(0);
    let pending_order_count = account
        .get("pending_order_count")
        .and_then(parse_i64_value)
        .unwrap_or(0);
    let should_log = !positions.is_empty()
        || total_order_count > 0
        || pending_order_count > 0
        || should_log_account_event(
            &LIGHTER_ACCOUNT_LAST_LOG_MS,
            &LIGHTER_ACCOUNT_LOG_COUNT,
            now_ms(),
        );
    if !should_log {
        return;
    }
    let positions_summary = if positions.is_empty() {
        "flat".to_string()
    } else {
        positions.join("|")
    };
    eprintln!(
        "INFO: Lighter account snapshot seq={} ts={} positions={} collateral_usd={:.6} available_usd={:.6} total_order_count={} pending_order_count={}",
        snapshot.seq,
        snapshot.timestamp_ms,
        positions_summary,
        snapshot.margin.balance_usd,
        snapshot.margin.available_usd,
        total_order_count,
        pending_order_count
    );
}

fn maybe_log_lighter_account_error(err: &str, interval_ms: u64, backoff_ms: u64) {
    if !should_log_account_event(
        &LIGHTER_ACCOUNT_ERROR_LAST_LOG_MS,
        &LIGHTER_ACCOUNT_ERROR_LOG_COUNT,
        now_ms(),
    ) {
        return;
    }
    eprintln!(
        "Lighter account polling error: {} (interval_ms={} backoff_ms={})",
        err, interval_ms, backoff_ms
    );
}

async fn fetch_public_funding(
    client: &Client,
    cfg: &LighterConfig,
    market_id: u64,
    market_symbol: &str,
) -> anyhow::Result<FundingUpdate> {
    // Default to /api/v1/funding-rates which is the working public endpoint.
    // The old /api/v1/marketStats returns HTTP 403 on mainnet CloudFront.
    let path = std::env::var("LIGHTER_FUNDING_PATH")
        .unwrap_or_else(|_| "/api/v1/funding-rates".to_string());
    let base = cfg.rest_url.trim_end_matches('/');
    let mut url = format!("{base}{path}");
    let query = format!("market_id={market_id}&symbol={market_symbol}");
    if url.contains('?') {
        url.push('&');
        url.push_str(&query);
    } else {
        url.push('?');
        url.push_str(&query);
    }
    let resp = client.get(&url).send().await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        // Rate-limited logging: only log first 3 errors, then every 100th
        let count =
            LIGHTER_FUNDING_ERROR_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 3 || count % 100 == 0 {
            let snippet: String = body.chars().take(160).collect();
            eprintln!(
                "WARN: Lighter funding fetch failed status={} url={} snippet={} (count={})",
                status,
                url,
                snippet,
                count + 1
            );
        }
        anyhow::bail!("Lighter funding fetch failed: HTTP {}", status);
    }

    let value: serde_json::Value =
        match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(err) => {
                let count = LIGHTER_FUNDING_ERROR_LOG_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count < 3 || count % 100 == 0 {
                    let snippet: String = body.chars().take(160).collect();
                    eprintln!(
                    "WARN: Lighter funding JSON parse failed url={} err={} snippet={} (count={})",
                    url, err, snippet, count + 1
                );
                }
                anyhow::bail!("Lighter funding JSON parse failed: {}", err);
            }
        };

    parse_public_funding(&value, cfg, market_id, market_symbol)
        .ok_or_else(|| anyhow::anyhow!("invalid public funding response"))
}

fn parse_public_funding(
    value: &serde_json::Value,
    cfg: &LighterConfig,
    market_id: u64,
    market_symbol: &str,
) -> Option<FundingUpdate> {
    // Try to extract from funding_rates[] array (the /api/v1/funding-rates endpoint format)
    if let Some(funding_rates) = value.get("funding_rates").and_then(|v| v.as_array()) {
        // First try to find by market_id (preferred)
        let entry = funding_rates
            .iter()
            .find(|e| e.get("market_id").and_then(|v| v.as_u64()) == Some(market_id))
            .or_else(|| {
                // Fallback: match by symbol (normalized)
                let target = normalize_lighter_symbol(market_symbol);
                funding_rates.iter().find(|e| {
                    e.get("symbol")
                        .and_then(|v| v.as_str())
                        .map(normalize_lighter_symbol)
                        == Some(target.clone())
                })
            });

        if let Some(data) = entry {
            return parse_funding_entry(data, cfg);
        }
        // No matching entry found in funding_rates array
        return None;
    }

    // Fallback: legacy marketStats-style response (single market object)
    let data = value
        .get("data")
        .or_else(|| value.get("market"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);

    parse_funding_entry(data, cfg)
}

/// Parse a single funding entry (either from funding_rates[] or legacy single-market response).
fn parse_funding_entry(data: &serde_json::Value, cfg: &LighterConfig) -> Option<FundingUpdate> {
    let rate_native = data
        .get("rate")
        .or_else(|| data.get("funding_rate"))
        .or_else(|| data.get("fundingRate"))
        .or_else(|| data.get("funding_rate_1h"))
        .or_else(|| data.get("fundingRate1h"))
        .or_else(|| data.get("funding_8h"))
        .and_then(parse_f64_value);

    // For /api/v1/funding-rates, the rate is hourly (1h).
    // Default to 3600s (1 hour) if not explicitly provided.
    let interval_sec = data
        .get("funding_interval_sec")
        .or_else(|| data.get("fundingIntervalSec"))
        .or_else(|| data.get("funding_interval"))
        .or_else(|| data.get("fundingInterval"))
        .and_then(parse_i64_value)
        .and_then(|v| if v > 0 { Some(v as u64) } else { None })
        .or_else(|| {
            // If rate field is present but no interval, assume hourly (Lighter default)
            if rate_native.is_some() {
                Some(3600)
            } else {
                None
            }
        });

    let next_funding_ms = data
        .get("next_funding_time")
        .or_else(|| data.get("nextFundingTime"))
        .or_else(|| data.get("nextFundingTimestamp"))
        .or_else(|| data.get("next_funding_ms"))
        .and_then(parse_i64_value);

    let as_of_ms = data
        .get("timestamp")
        .or_else(|| data.get("ts"))
        .and_then(parse_i64_value)
        .unwrap_or_else(|| now_ms() as i64);

    // Convert to 8h rate: rate_8h = rate_native * (8h / interval_sec)
    let rate_8h = match (rate_native, interval_sec) {
        (Some(rate), Some(sec)) if sec > 0 => Some(rate * (8.0 * 60.0 * 60.0 / sec as f64)),
        (Some(rate), None) => Some(rate), // Assume already 8h if no interval
        _ => None,
    };

    Some(FundingUpdate {
        venue_index: cfg.venue_index,
        venue_id: cfg.venue_id.clone(),
        seq: 0,
        timestamp_ms: as_of_ms,
        received_ms: Some(now_ms() as i64),
        funding_rate_8h: rate_8h,
        funding_rate_native: rate_native,
        interval_sec,
        next_funding_ms,
        // Lighter uses mark price for funding settlement. Mark = (Impact Bid + Impact Ask) / 2.
        // Ref: https://docs.lighter.xyz/perpetual-futures/fair-price-marking
        settlement_price_kind: Some(SettlementPriceKind::Mark),
        source: FundingSource::MarketDataRest,
    })
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

pub fn parse_account_snapshot(data: &serde_json::Value, venue_id: &str) -> Option<AccountSnapshot> {
    parse_account_snapshot_with_meta(data, venue_id, 0)
}

fn parse_account_snapshot_with_meta(
    data: &serde_json::Value,
    venue_id: &str,
    venue_index: usize,
) -> Option<AccountSnapshot> {
    let account = data
        .get("accounts")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .unwrap_or(data);
    let seq = account
        .get("seq")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            account
                .get("transaction_time")
                .and_then(parse_i64_value)
                .map(|v| v.max(0) as u64)
        })
        .or_else(|| {
            account
                .get("created_at")
                .and_then(parse_i64_value)
                .map(|v| v.max(0) as u64)
        })
        .unwrap_or(0);
    let timestamp_ms = normalize_lighter_timestamp_ms(
        account
            .get("ts")
            .or_else(|| account.get("transaction_time"))
            .or_else(|| account.get("created_at"))
            .and_then(parse_i64_value)
            .unwrap_or(0),
    );
    let positions = account
        .get("positions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pos| {
                    let symbol = pos.get("symbol")?.as_str()?;
                    let size = if let Some(size) = pos.get("size").and_then(parse_f64_value) {
                        size
                    } else {
                        let base = pos.get("position").and_then(parse_f64_value)?;
                        let sign = pos.get("sign").and_then(parse_f64_value).unwrap_or(1.0);
                        base * sign.signum()
                    };
                    let entry_price = pos
                        .get("entry_px")
                        .or_else(|| pos.get("avg_entry_price"))
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

    let balances = if let Some(arr) = account.get("balances").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|bal| {
                let asset = bal.get("asset")?.as_str()?;
                let total = bal.get("total").and_then(parse_f64_value)?;
                let available = bal.get("available").and_then(parse_f64_value)?;
                Some(BalanceSnapshot {
                    asset: asset.to_string(),
                    total,
                    available,
                })
            })
            .collect::<Vec<_>>()
    } else {
        let mut parsed = account
            .get("assets")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|asset| {
                        let symbol = asset.get("symbol")?.as_str()?;
                        let total = asset.get("balance").and_then(parse_f64_value)?;
                        let locked = asset
                            .get("locked_balance")
                            .and_then(parse_f64_value)
                            .unwrap_or(0.0);
                        Some(BalanceSnapshot {
                            asset: symbol.to_string(),
                            total,
                            available: (total - locked).max(0.0),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let (Some(total), Some(available)) = (
            account.get("collateral").and_then(parse_f64_value),
            account.get("available_balance").and_then(parse_f64_value),
        ) {
            if !parsed
                .iter()
                .any(|balance| balance.asset.eq_ignore_ascii_case("USDC"))
            {
                parsed.push(BalanceSnapshot {
                    asset: "USDC".to_string(),
                    total,
                    available,
                });
            }
        }
        parsed
    };

    let (margin, liquidation, funding_8h) = if let Some(margin) = account.get("margin") {
        let liquidation = account.get("liquidation")?;
        (
            MarginSnapshot {
                balance_usd: margin.get("balance").and_then(parse_f64_value)?,
                used_usd: margin.get("used").and_then(parse_f64_value)?,
                available_usd: margin.get("available").and_then(parse_f64_value)?,
            },
            LiquidationSnapshot {
                price_liq: liquidation.get("price_liq").and_then(parse_f64_value),
                dist_liq_sigma: liquidation.get("dist_liq_sigma").and_then(parse_f64_value),
            },
            account.get("funding_8h").and_then(parse_f64_value),
        )
    } else {
        let balance_usd = account.get("collateral").and_then(parse_f64_value)?;
        let available_usd = account.get("available_balance").and_then(parse_f64_value)?;
        let price_liq = account
            .get("positions")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find_map(|pos| pos.get("liquidation_price").and_then(parse_f64_value))
            });
        (
            MarginSnapshot {
                balance_usd,
                used_usd: (balance_usd - available_usd).max(0.0),
                available_usd,
            },
            LiquidationSnapshot {
                price_liq,
                dist_liq_sigma: None,
            },
            None,
        )
    };

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

fn normalize_lighter_timestamp_ms(raw: i64) -> i64 {
    if raw > 10_000_000_000_000 {
        raw / 1_000
    } else {
        raw
    }
}

impl LiveRestClient for LighterConnector {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>>
    {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                });
            }
            if let Some(reason) = phase51_lighter_strict_maker_only_place_rejection(&req) {
                return Err(LiveGatewayError::fatal(reason));
            }
            if let Some(reason) =
                phase51_lighter_baseline_cleanup_place_rejection(&self.cfg.market, &req)
            {
                return Err(LiveGatewayError::fatal(reason));
            }
            if !self.has_auth() {
                return Err(LiveGatewayError::fatal(
                    "lighter: missing auth (set LIGHTER_API_KEY_INDEX, LIGHTER_ACCOUNT_INDEX, LIGHTER_API_PRIVATE_KEY_HEX)",
                ));
            }
            if !self.has_signer() {
                return Err(LiveGatewayError::fatal(
                    "lighter: signer unavailable (set LIGHTER_SIGNER_URL)",
                ));
            }
            let signer = self.signer.as_ref().ok_or_else(|| {
                LiveGatewayError::fatal("lighter: signer unavailable (set LIGHTER_SIGNER_URL)")
            })?;
            let account_index = self
                .cfg
                .account_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth account_index"))?;
            let api_key_index = self
                .cfg
                .api_key_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth api_key_index"))?;
            let client_order_index = req.client_order_id.parse::<u64>().map_err(|_| {
                LiveGatewayError::fatal(
                    "lighter: client_order_id must be numeric for signer bridge",
                )
            })?;
            if client_order_index > LIGHTER_CLIENT_ORDER_INDEX_MAX {
                if phase51_lighter_strict_maker_only_observation_enabled() {
                    return Err(LiveGatewayError::fatal(
                        "lighter: client_order_id exceeds uint48 max",
                    ));
                }
                return Err(LiveGatewayError::fatal(format!(
                    "lighter: client_order_id exceeds uint48 max client_order_id={} max={}",
                    req.client_order_id, LIGHTER_CLIENT_ORDER_INDEX_MAX
                )));
            }
            let (_, market_id) = self.resolve_market_id_and_symbol().await.map_err(|err| {
                LiveGatewayError::fatal(format!("lighter market_id error: {err}"))
            })?;
            let (price_decimals, size_decimals) = self
                .resolve_market_decimals(market_id)
                .await
                .map_err(|err| LiveGatewayError::fatal(format!("lighter decimals error: {err}")))?;
            let price = scale_to_i64(req.price, price_decimals, "price")
                .map_err(|err| LiveGatewayError::fatal(format!("{err}")))?;
            let base_amount = scale_to_i64(req.size, size_decimals, "size")
                .map_err(|err| LiveGatewayError::fatal(format!("{err}")))?;
            let expired_at = now_ms().saturating_add(60_000);
            let order_expiry = match req.time_in_force {
                TimeInForce::Ioc => Some(0),
                _ => None,
            };
            let _intent = OrderIntent::Place(crate::types::PlaceOrderIntent {
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
            let sign_req = SignCreateOrderRequest {
                op: "create_order".to_string(),
                account_index,
                api_key_index,
                nonce: self.next_nonce(),
                market_index: market_id,
                client_order_index,
                base_amount,
                price,
                is_ask: if req.side == Side::Sell { 1 } else { 0 },
                order_type: "limit".to_string(),
                time_in_force: format!("{:?}", req.time_in_force),
                post_only: if req.post_only { 1 } else { 0 },
                reduce_only: if req.reduce_only { 1 } else { 0 },
                trigger_price: None,
                order_expiry,
                expired_at,
            };
            let emergency_ioc_timeout =
                lighter_emergency_ioc_request(&req).then(lighter_emergency_ioc_timeout);
            let signed = if let Some(timeout_duration) = emergency_ioc_timeout {
                match tokio::time::timeout(timeout_duration, signer.sign_create_order(sign_req)).await
                {
                    Ok(result) => result,
                    Err(_) => {
                        if phase51_lighter_sanitized_live_error_context_enabled() {
                            eprintln!(
                                "WARN: Lighter emergency IOC signer timeout timeout_ms={} sanitized_context=true",
                                timeout_duration.as_millis()
                            );
                        } else {
                            eprintln!(
                                "WARN: Lighter emergency IOC signer timeout client_order_id={} timeout_ms={}",
                                req.client_order_id,
                                timeout_duration.as_millis()
                            );
                        }
                        return Err(LiveGatewayError::retryable(format!(
                            "lighter emergency_ioc signer_timeout after {}ms",
                            timeout_duration.as_millis()
                        )));
                    }
                }
            } else {
                signer.sign_create_order(sign_req).await
            }
            .map_err(|err| {
                phase51_lighter_strict_maker_only_retryable_error("signer_error", err)
            })?;
            let context = phase51_lighter_place_error_context(
                market_id,
                client_order_index,
                price,
                base_amount,
                req.time_in_force,
                req.post_only,
                req.reduce_only,
            );
            let resp = if let Some(timeout_duration) = emergency_ioc_timeout {
                match tokio::time::timeout(timeout_duration, self.submit_sendtx(signed)).await {
                    Ok(result) => result,
                    Err(_) => {
                        if phase51_lighter_sanitized_live_error_context_enabled() {
                            eprintln!(
                                "WARN: Lighter emergency IOC sendtx timeout timeout_ms={} sanitized_context=true",
                                timeout_duration.as_millis()
                            );
                        } else {
                            eprintln!(
                                "WARN: Lighter emergency IOC sendtx timeout client_order_id={} timeout_ms={}",
                                req.client_order_id,
                                timeout_duration.as_millis()
                            );
                        }
                        return Err(err_with_context(
                            LiveGatewayError::retryable(format!(
                                "lighter emergency_ioc sendtx_timeout after {}ms",
                                timeout_duration.as_millis()
                            )),
                            &context,
                        ));
                    }
                }
            } else {
                self.submit_sendtx(signed).await
            }
            .map_err(|err| phase51_lighter_strict_maker_only_err_with_context(err, &context))?;
            Ok(resp)
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>>
    {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                });
            }
            if !self.has_auth() {
                return Err(LiveGatewayError::fatal(
                    "lighter: missing auth (set LIGHTER_API_KEY_INDEX, LIGHTER_ACCOUNT_INDEX, LIGHTER_API_PRIVATE_KEY_HEX)",
                ));
            }
            if !self.has_signer() {
                return Err(LiveGatewayError::fatal(
                    "lighter: signer unavailable (set LIGHTER_SIGNER_URL)",
                ));
            }
            let signer = self.signer.as_ref().ok_or_else(|| {
                LiveGatewayError::fatal("lighter: signer unavailable (set LIGHTER_SIGNER_URL)")
            })?;
            let account_index = self
                .cfg
                .account_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth account_index"))?;
            let api_key_index = self
                .cfg
                .api_key_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth api_key_index"))?;
            let raw_order_id = req.order_id.parse::<u64>().map_err(|_| {
                LiveGatewayError::fatal("lighter: order_id must be numeric for signer bridge")
            })?;
            let (order_index, client_order_index) =
                if raw_order_id <= LIGHTER_CLIENT_ORDER_INDEX_MAX {
                    (None, Some(raw_order_id))
                } else {
                    (Some(raw_order_id), None)
                };
            let (_, market_id) = self.resolve_market_id_and_symbol().await.map_err(|err| {
                LiveGatewayError::fatal(format!("lighter market_id error: {err}"))
            })?;
            let identity_label = if order_index.is_some() {
                "order_index"
            } else {
                "client_order_index"
            };
            let sanitized_cancel_context = if phase51_lighter_baseline_cleanup_only_enabled() {
                format!(
                    "lighter cancel context market_id={} identity_kind={} baseline_cleanup_only=true",
                    market_id, identity_label
                )
            } else {
                format!(
                    "lighter cancel context market_id={} identity_kind={} strict_maker_only_observation=true",
                    market_id, identity_label
                )
            };
            let expired_at = now_ms().saturating_add(60_000);
            let sign_req = SignCancelOrderRequest {
                op: "cancel_order".to_string(),
                account_index,
                api_key_index,
                nonce: self.next_nonce(),
                market_index: market_id,
                order_index,
                client_order_index,
                expired_at,
            };
            let signed = signer.sign_cancel_order(sign_req).await.map_err(|err| {
                phase51_lighter_strict_maker_only_retryable_error("cancel signer_error", err)
            })?;
            let resp = self.submit_sendtx(signed).await.map_err(|err| {
                phase51_lighter_strict_maker_only_cancel_err_with_context(
                    err,
                    "cancel submit_sendtx",
                    &sanitized_cancel_context,
                )
            })?;
            Ok(resp)
        })
    }

    fn replace_order(
        &self,
        req: LiveRestReplaceRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>>
    {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: req.client_order_id.into(),
                });
            }
            if phase51_lighter_strict_maker_only_observation_enabled()
                && (req.purpose != OrderPurpose::Mm
                    || !matches!(req.time_in_force, TimeInForce::Gtc)
                    || !req.post_only
                    || req.reduce_only)
            {
                return Err(LiveGatewayError::fatal(
                    "lighter: strict maker-only observation rejects replace request \
                     (requires purpose=Mm, time_in_force=Gtc, post_only=true, reduce_only=false)",
                ));
            }
            if phase51_lighter_baseline_cleanup_only_enabled() {
                return Err(LiveGatewayError::fatal(
                    "lighter: baseline cleanup-only rejects replace request",
                ));
            }
            if req.purpose != OrderPurpose::Mm
                || !matches!(req.time_in_force, TimeInForce::Gtc)
                || !req.post_only
                || req.reduce_only
            {
                return Err(LiveGatewayError::fatal(
                    "lighter: native replace requires mm gtc post_only non_reduce_only",
                ));
            }
            if !self.has_auth() {
                return Err(LiveGatewayError::fatal(
                    "lighter: missing auth (set LIGHTER_API_KEY_INDEX, LIGHTER_ACCOUNT_INDEX, LIGHTER_API_PRIVATE_KEY_HEX)",
                ));
            }
            if !self.has_signer() {
                return Err(LiveGatewayError::fatal(
                    "lighter: signer unavailable (set LIGHTER_SIGNER_URL)",
                ));
            }
            let signer = self.signer.as_ref().ok_or_else(|| {
                LiveGatewayError::fatal("lighter: signer unavailable (set LIGHTER_SIGNER_URL)")
            })?;
            let account_index = self
                .cfg
                .account_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth account_index"))?;
            let api_key_index = self
                .cfg
                .api_key_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth api_key_index"))?;
            let raw_order_id = req.order_id.parse::<u64>().map_err(|_| {
                LiveGatewayError::fatal("lighter: replace order_id must be numeric")
            })?;
            let requested_order_id = req.order_id.clone();
            let requested_client_order_id = req.client_order_id.clone();
            let (order_index, client_order_index) =
                if raw_order_id <= LIGHTER_CLIENT_ORDER_INDEX_MAX {
                    (None, Some(raw_order_id))
                } else {
                    (Some(raw_order_id), None)
                };
            let (_, market_id) = self.resolve_market_id_and_symbol().await.map_err(|err| {
                LiveGatewayError::fatal(format!("lighter market_id error: {err}"))
            })?;
            let (price_decimals, size_decimals) = self
                .resolve_market_decimals(market_id)
                .await
                .map_err(|err| LiveGatewayError::fatal(format!("lighter decimals error: {err}")))?;
            let price = scale_to_i64(req.price, price_decimals, "price")
                .map_err(|err| LiveGatewayError::fatal(format!("{err}")))?;
            let base_amount = scale_to_i64(req.size, size_decimals, "size")
                .map_err(|err| LiveGatewayError::fatal(format!("{err}")))?;
            let expired_at = now_ms().saturating_add(60_000);
            let sign_req = SignModifyOrderRequest {
                op: "modify_order".to_string(),
                account_index,
                api_key_index,
                nonce: self.next_nonce(),
                market_index: market_id,
                order_index,
                client_order_index,
                base_amount,
                price,
                trigger_price: None,
                expired_at,
            };
            let identity_label = if order_index.is_some() {
                "order_index"
            } else {
                "client_order_index"
            };
            let context = phase51_lighter_replace_error_context(
                market_id,
                identity_label,
                raw_order_id,
                price,
                base_amount,
                &requested_client_order_id,
            );
            let signed = signer.sign_modify_order(sign_req).await.map_err(|err| {
                err_with_context(
                    phase51_lighter_strict_maker_only_retryable_error("signer_error", err),
                    &context,
                )
            })?;
            if phase51_lighter_strict_maker_only_observation_enabled() {
                eprintln!(
                    "INFO: Lighter native replace submit market_id={} identity_kind={} strict_maker_only_observation=true",
                    market_id, identity_label
                );
            } else {
                eprintln!(
                    "INFO: Lighter native replace submit market_id={} {}={} client_order_id={} price={} base_amount={}",
                    market_id, identity_label, raw_order_id, requested_client_order_id, price, base_amount
                );
            }
            let mut resp = self
                .submit_sendtx(signed)
                .await
                .map_err(|err| phase51_lighter_strict_maker_only_err_with_context(err, &context))?;
            resp.order_id = resp.order_id.or(Some(requested_order_id));
            resp.client_order_id = Some(requested_client_order_id);
            Ok(resp)
        })
    }

    fn cancel_all(
        &self,
        _req: LiveRestCancelAllRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>>
    {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                });
            }
            if !self.has_auth() {
                return Err(LiveGatewayError::fatal(
                    "lighter: missing auth (set LIGHTER_API_KEY_INDEX, LIGHTER_ACCOUNT_INDEX, LIGHTER_API_PRIVATE_KEY_HEX)",
                ));
            }
            if !self.has_signer() {
                return Err(LiveGatewayError::fatal(
                    "lighter: signer unavailable (set LIGHTER_SIGNER_URL)",
                ));
            }
            let signer = self.signer.as_ref().ok_or_else(|| {
                LiveGatewayError::fatal("lighter: signer unavailable (set LIGHTER_SIGNER_URL)")
            })?;
            let account_index = self
                .cfg
                .account_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth account_index"))?;
            let api_key_index = self
                .cfg
                .api_key_index
                .ok_or_else(|| LiveGatewayError::fatal("lighter: missing auth api_key_index"))?;
            let sign_req = SignCancelAllRequest {
                op: "cancel_all".to_string(),
                account_index,
                api_key_index,
                nonce: self.next_nonce(),
                cancel_all_time_in_force: 0,
                // Lighter expects a nil cancel_all_time sentinel, not wall-clock time.
                cancel_all_time: 0,
                expired_at: 0,
            };
            let signed = signer.sign_cancel_all(sign_req).await.map_err(|err| {
                phase51_lighter_strict_maker_only_retryable_error("cancel_all signer_error", err)
            })?;
            let resp = self.submit_sendtx(signed).await.map_err(|err| {
                let context = if phase51_lighter_baseline_cleanup_only_enabled() {
                    "lighter cancel_all context baseline_cleanup_only=true"
                } else {
                    "lighter cancel_all context strict_maker_only_observation=true"
                };
                phase51_lighter_strict_maker_only_cancel_err_with_context(
                    err,
                    "cancel_all submit_sendtx",
                    context,
                )
            })?;
            Ok(resp)
        })
    }
}

fn map_rest_error(body: &str) -> LiveGatewayError {
    let lower = body.to_lowercase();
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(body);
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(body);
    }
    if lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(body);
    }
    if lower.contains("timeout") || lower.contains("tempor") {
        return LiveGatewayError::retryable(body);
    }
    LiveGatewayError {
        kind: LiveGatewayErrorKind::Fatal,
        message: body.to_string(),
    }
}

fn err_with_context(err: LiveGatewayError, context: &str) -> LiveGatewayError {
    LiveGatewayError {
        kind: err.kind,
        message: format!("{} [{}]", err.message, context),
    }
}
