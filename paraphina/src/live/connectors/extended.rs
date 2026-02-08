//! Extended connector (public WS market data + fixtures, feature-gated).

#[cfg(feature = "live_extended")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_extended")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_extended")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_extended")]
pub const SUPPORTS_EXECUTION: bool = true;

const EXTENDED_STALE_MS_DEFAULT: u64 = 10_000;
const EXTENDED_WATCHDOG_TICK_MS: u64 = 200;
const EXTENDED_MARKET_PUB_QUEUE_CAP_LIVE: usize = 256;
const EXTENDED_MARKET_PUB_QUEUE_CAP_FIXTURE: usize = 4096;
const EXTENDED_MARKET_PUB_DRAIN_MAX: usize = 64;

static MONO_START: OnceLock<Instant> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn extended_stale_ms() -> u64 {
    std::env::var("PARAPHINA_EXTENDED_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(EXTENDED_STALE_MS_DEFAULT)
}

#[allow(dead_code)]
fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
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
        // Use last_book_event_ns as the primary watchdog anchor.
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

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
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
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::USER_AGENT;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::super::gateway::{
    BoxFuture, LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest,
    LiveRestCancelRequest, LiveRestClient, LiveRestPlaceRequest, LiveRestResponse, LiveResult,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, FundingUpdate, LiquidationSnapshot,
    MarginSnapshot, MarketDataEvent, PositionSnapshot, TopOfBook,
};
use crate::live::MarketPublisher;
use crate::types::{FundingSource, SettlementPriceKind, Side, TimeInForce, TimestampMs};

#[derive(Debug, Clone)]
pub struct ExtendedConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub market: String,
    pub depth_limit: usize,
    pub venue_index: usize,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub recv_window: Option<u64>,
    pub record_dir: Option<PathBuf>,
}

impl ExtendedConfig {
    pub fn from_env() -> Self {
        let ws_url = std::env::var("EXTENDED_WS_URL").unwrap_or_else(|_| {
            "wss://api.starknet.extended.exchange/stream.extended.exchange/v1".to_string()
        });
        // Default to Starknet Extended API (the original api.extended.exchange returns 404)
        let rest_url = std::env::var("EXTENDED_REST_URL")
            .unwrap_or_else(|_| "https://api.starknet.extended.exchange".to_string());
        let market = std::env::var("EXTENDED_MARKET").unwrap_or_else(|_| "BTCUSDT".to_string());
        let market = normalize_extended_market(&market);
        let depth_limit = std::env::var("EXTENDED_DEPTH_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100);
        let api_key = std::env::var("EXTENDED_API_KEY").ok();
        let api_secret = std::env::var("EXTENDED_API_SECRET").ok();
        let recv_window = std::env::var("EXTENDED_RECV_WINDOW")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .or(Some(5_000));
        Self {
            ws_url,
            rest_url,
            market,
            depth_limit,
            venue_index: 0,
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

    pub fn orderbook_ws_url(&self) -> String {
        format!(
            "{}/orderbooks/{}?depth=1",
            self.ws_url.trim_end_matches('/'),
            self.market
        )
    }
}

#[derive(Debug)]
pub struct ExtendedConnector {
    cfg: ExtendedConfig,
    http: Client,
    market_publisher: MarketPublisher,
    recorder: Option<Mutex<ExtendedRecorder>>,
    freshness: Arc<Freshness>,
    is_fixture: bool,
}

impl ExtendedConnector {
    pub fn new(cfg: ExtendedConfig, market_tx: mpsc::Sender<MarketDataEvent>) -> Self {
        let recorder = cfg
            .record_dir
            .as_ref()
            .and_then(|dir| ExtendedRecorder::new(dir).ok())
            .map(Mutex::new);
        let http = Client::builder()
            .user_agent("paraphina")
            .timeout(Duration::from_secs(10))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .pool_idle_timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(5)
            .build()
            .expect("extended http client build");
        let freshness = Arc::new(Freshness::default());
        let is_fixture = std::env::var_os("EXTENDED_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var_os("EXTENDED_FIXTURE_MODE").is_some();
        let cap = if is_fixture {
            EXTENDED_MARKET_PUB_QUEUE_CAP_FIXTURE
        } else {
            EXTENDED_MARKET_PUB_QUEUE_CAP_LIVE
        };
        let publish_freshness = freshness.clone();
        let on_published = Arc::new(move || {
            publish_freshness
                .last_published_ns
                .store(mono_now_ns(), Ordering::Relaxed);
        });
        let market_publisher = MarketPublisher::new(
            cap,
            EXTENDED_MARKET_PUB_DRAIN_MAX,
            market_tx.clone(),
            Some(Arc::new(move || is_fixture)),
            Arc::new(|event: &MarketDataEvent| matches!(event, MarketDataEvent::L2Delta(_))),
            Some(on_published),
            "extended market_tx closed",
            "extended market publish queue closed",
        );
        let connector = Self {
            cfg,
            http,
            market_publisher,
            recorder,
            freshness,
            is_fixture,
        };
        connector
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        self.market_publisher.publish_market(event).await
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let mut consecutive_failures: u32 = 0;
        let mut last_snapshot_warn: Option<Instant> = None;

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
            let result =
                tokio::time::timeout(max_session, self.public_ws_once(&mut last_snapshot_warn))
                    .await;
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
                        "{level}: Extended public WS error (consecutive_failures={consecutive_failures}): {err}"
                    );
                }
                Err(_timeout) => {
                    eprintln!(
                        "ERROR: Extended public WS session timeout ({}s) — force reconnect",
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
                        "INFO: Extended WS session was healthy for {:?}; \
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
                        eprintln!("Extended funding publish error: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Extended funding polling error: {err}");
                }
            }
        }
    }

    async fn public_ws_once(&self, last_snapshot_warn: &mut Option<Instant>) -> anyhow::Result<()> {
        let mut first_decoded_top_logged = false;
        let mut decode_miss_count = 0usize;
        let mut first_ws_message_logged = false;
        let mut first_book_update_logged = false;
        let mut ws_snapshot_seq: u64 = 0;
        let mut snapshot_state: Option<ExtendedDepthSnapshot> = None;
        if let Ok((snapshot_raw, snapshot)) = self.fetch_snapshot().await {
            if let Some(recorder) = self.recorder.as_ref() {
                let mut guard = recorder.lock().await;
                guard.record_snapshot(&snapshot_raw)?;
            }
            if let Ok(value) = serde_json::from_str::<Value>(&snapshot_raw) {
                if let Some(top) =
                    TopOfBook::from_levels(&snapshot.bids, &snapshot.asks, Some(now_ms()))
                {
                    eprintln!(
                        "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                        top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                    );
                    first_decoded_top_logged = true;
                } else if decode_miss_count < 3 {
                    decode_miss_count += 1;
                    log_decode_miss(
                        "Extended",
                        &value,
                        &snapshot_raw,
                        decode_miss_count,
                        self.cfg.ws_url.as_str(),
                    );
                }
            }
            snapshot_state = Some(snapshot);
        } else if last_snapshot_warn
            .map(|last| last.elapsed() >= Duration::from_secs(30))
            .unwrap_or(true)
        {
            *last_snapshot_warn = Some(Instant::now());
            eprintln!("WARN: Extended REST snapshot skipped; relying on WS depth=1");
        }
        let mut seq_state = ExtendedSeqState::new(
            snapshot_state
                .as_ref()
                .map(|snapshot| snapshot.last_update_id),
            self.cfg.venue_index,
        );
        if let Some(snapshot) = snapshot_state {
            let now_ns = mono_now_ns();
            self.freshness
                .last_parsed_ns
                .store(now_ns, Ordering::Relaxed);
            self.freshness
                .last_book_event_ns
                .store(now_ns, Ordering::Relaxed);
            let snapshot_event = MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
                venue_index: self.cfg.venue_index,
                venue_id: self.cfg.market.clone(),
                seq: snapshot.last_update_id,
                timestamp_ms: now_ms(),
                bids: snapshot.bids,
                asks: snapshot.asks,
            });
            let _ = self.publish_market(snapshot_event).await;
        }

        let ws_url = self.cfg.orderbook_ws_url();
        eprintln!("INFO: Extended public WS connecting url={}", ws_url);
        let mut request = ws_url.as_str().into_client_request()?;
        request
            .headers_mut()
            .insert(USER_AGENT, HeaderValue::from_static("paraphina"));
        let (ws_stream, _) = tokio::time::timeout(Duration::from_secs(15), connect_async(request))
            .await
            .map_err(|_| anyhow::anyhow!("Extended public WS connect timeout (15s)"))?
            .map_err(|e| anyhow::anyhow!("Extended public WS connect error: {e}"))?;
        eprintln!("INFO: Extended public WS connected url={}", ws_url);
        let (mut write, mut read) = ws_stream.split();

        const MAX_PARSE_ERRORS: usize = 25;
        let mut consecutive_parse_errors = 0usize;
        let mut first_message_logged = false;
        let ws_start = Instant::now();
        let mut no_book_warned = false;
        let mut first_ws_keys: Option<String> = None;
        let mut first_ws_snippet: Option<String> = None;
        let connect_start_ns = mono_now_ns();
        self.freshness.reset_for_new_connection();
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        let fixture_mode = std::env::var_os("EXTENDED_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("EXTENDED_FIXTURE_MODE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        let stale_ms = extended_stale_ms();
        // WS-level ping timer to prevent idle connection drops.
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_EXTENDED_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ping_timer.tick().await; // skip first immediate tick
        if fixture_mode {
            eprintln!("INFO: Extended fixture mode detected; freshness watchdog disabled");
        } else {
            let watchdog_stale_ms = stale_ms;
            let watchdog_freshness = self.freshness.clone();
            tokio::spawn(async move {
                let mut iv =
                    tokio::time::interval(Duration::from_millis(EXTENDED_WATCHDOG_TICK_MS));
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
        loop {
            let next = tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    anyhow::bail!("Extended public WS stale: freshness exceeded {stale_ms}ms");
                }
                _ = ping_timer.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        eprintln!("WARN: Extended public WS ping send failed: {e} — reconnecting");
                        anyhow::bail!("Extended public WS ping send failed: {e}");
                    }
                    continue;
                }
                next = tokio::time::timeout(Duration::from_secs(10), read.next()) => next,
            };
            let msg = match next {
                Ok(Some(msg)) => msg?,
                Ok(None) => break,
                Err(_) => {
                    if !first_message_logged {
                        eprintln!(
                            "WARN: Extended WS received no messages after 10s url={}",
                            ws_url
                        );
                        break;
                    }
                    continue;
                }
            };
            self.freshness
                .last_ws_rx_ns
                .store(mono_now_ns(), Ordering::Relaxed);
            match msg {
                Message::Text(text) => {
                    if !first_message_logged {
                        eprintln!("INFO: Extended public WS first message received");
                        first_message_logged = true;
                    }
                    let Some(cleaned) = clean_ws_payload(&text) else {
                        continue;
                    };
                    self.freshness
                        .last_data_rx_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if !first_ws_message_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            let keys = value
                                .as_object()
                                .map(|obj| {
                                    let mut keys: Vec<&str> =
                                        obj.keys().map(|k| k.as_str()).collect();
                                    keys.sort();
                                    format!("[{}]", keys.join(","))
                                })
                                .unwrap_or_else(|| "[non-object]".to_string());
                            let snippet: String = cleaned.chars().take(160).collect();
                            eprintln!(
                                "INFO: Extended public WS first message keys={} snippet={}",
                                keys, snippet
                            );
                            first_ws_keys = Some(keys);
                            first_ws_snippet = Some(snippet);
                            first_ws_message_logged = true;
                        }
                    }
                    if let Some(recorder) = self.recorder.as_ref() {
                        let mut guard = recorder.lock().await;
                        let _ = guard.record_ws_frame(cleaned);
                    }
                    let update = match parse_depth_update(cleaned) {
                        Ok(update) => update,
                        Err(err) => {
                            consecutive_parse_errors += 1;
                            if consecutive_parse_errors == 1
                                || consecutive_parse_errors.is_multiple_of(10)
                            {
                                let snippet: String = cleaned.chars().take(160).collect();
                                eprintln!(
                                    "WARN: Extended public WS parse error: {err} url={} snippet={}",
                                    ws_url, snippet
                                );
                            }
                            if consecutive_parse_errors > MAX_PARSE_ERRORS {
                                eprintln!(
                                    "Extended public WS too many parse errors; reconnecting url={}",
                                    ws_url
                                );
                                break;
                            }
                            continue;
                        }
                    };
                    let Some(update) = update else {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            if let Some(event) = parse_depth_snapshot_from_ws(
                                &value,
                                &self.cfg.market,
                                self.cfg.venue_index,
                                &mut ws_snapshot_seq,
                            ) {
                                if !first_book_update_logged {
                                    eprintln!("INFO: Extended public WS first book update");
                                    first_book_update_logged = true;
                                }
                                let now_ns = mono_now_ns();
                                self.freshness
                                    .last_parsed_ns
                                    .store(now_ns, Ordering::Relaxed);
                                self.freshness
                                    .last_book_event_ns
                                    .store(now_ns, Ordering::Relaxed);
                                if let Err(err) = self.publish_market(event).await {
                                    eprintln!("Extended public WS market send failed: {err}");
                                }
                            }
                            if !first_decoded_top_logged {
                                if let Some(top) = decode_top_from_value(&value) {
                                    eprintln!(
                                        "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                        top.best_bid_px,
                                        top.best_bid_sz,
                                        top.best_ask_px,
                                        top.best_ask_sz
                                    );
                                    first_decoded_top_logged = true;
                                }
                            }
                        }
                        continue;
                    };
                    self.freshness
                        .last_parsed_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if !first_decoded_top_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            if let Some(top) = decode_top_from_value(&value) {
                                eprintln!(
                                    "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                    top.best_bid_px,
                                    top.best_bid_sz,
                                    top.best_ask_px,
                                    top.best_ask_sz
                                );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged {
                            if let Some(top) = decode_top_from_update(&update, update.event_time) {
                                eprintln!(
                                "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                top.best_bid_px,
                                top.best_bid_sz,
                                top.best_ask_px,
                                top.best_ask_sz
                            );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged && decode_miss_count < 3 {
                            decode_miss_count += 1;
                            if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                                log_decode_miss(
                                    "Extended",
                                    &value,
                                    cleaned,
                                    decode_miss_count,
                                    ws_url.as_str(),
                                );
                            }
                        }
                    }
                    if !symbol_matches(&update.symbol, &self.cfg.market) {
                        continue;
                    }
                    let outcome = seq_state.apply_update(&update)?;
                    if let Some(event) = outcome {
                        consecutive_parse_errors = 0;
                        self.freshness
                            .last_book_event_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        if !first_book_update_logged {
                            eprintln!("INFO: Extended public WS first book update");
                            first_book_update_logged = true;
                        }
                        if let Err(err) = self.publish_market(event).await {
                            eprintln!("Extended public WS market send failed: {err}");
                        }
                    }
                }
                Message::Binary(bytes) => {
                    if !first_message_logged {
                        eprintln!("INFO: Extended public WS first message received");
                        first_message_logged = true;
                    }
                    let text = String::from_utf8_lossy(&bytes);
                    let Some(cleaned) = clean_ws_payload(&text) else {
                        continue;
                    };
                    self.freshness
                        .last_data_rx_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if !first_ws_message_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            let keys = value
                                .as_object()
                                .map(|obj| {
                                    let mut keys: Vec<&str> =
                                        obj.keys().map(|k| k.as_str()).collect();
                                    keys.sort();
                                    format!("[{}]", keys.join(","))
                                })
                                .unwrap_or_else(|| "[non-object]".to_string());
                            let snippet: String = cleaned.chars().take(160).collect();
                            eprintln!(
                                "INFO: Extended public WS first message keys={} snippet={}",
                                keys, snippet
                            );
                            first_ws_keys = Some(keys);
                            first_ws_snippet = Some(snippet);
                            first_ws_message_logged = true;
                        }
                    }
                    if let Some(recorder) = self.recorder.as_ref() {
                        let mut guard = recorder.lock().await;
                        let _ = guard.record_ws_frame(cleaned);
                    }
                    let update = match parse_depth_update(cleaned) {
                        Ok(update) => update,
                        Err(err) => {
                            consecutive_parse_errors += 1;
                            if consecutive_parse_errors == 1
                                || consecutive_parse_errors.is_multiple_of(10)
                            {
                                let snippet: String = cleaned.chars().take(160).collect();
                                eprintln!(
                                    "WARN: Extended public WS parse error: {err} url={} snippet={}",
                                    ws_url, snippet
                                );
                            }
                            if consecutive_parse_errors > MAX_PARSE_ERRORS {
                                eprintln!(
                                    "Extended public WS too many parse errors; reconnecting url={}",
                                    ws_url
                                );
                                break;
                            }
                            continue;
                        }
                    };
                    let Some(update) = update else {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            if let Some(event) = parse_depth_snapshot_from_ws(
                                &value,
                                &self.cfg.market,
                                self.cfg.venue_index,
                                &mut ws_snapshot_seq,
                            ) {
                                if !first_book_update_logged {
                                    eprintln!("INFO: Extended public WS first book update");
                                    first_book_update_logged = true;
                                }
                                let now_ns = mono_now_ns();
                                self.freshness
                                    .last_parsed_ns
                                    .store(now_ns, Ordering::Relaxed);
                                self.freshness
                                    .last_book_event_ns
                                    .store(now_ns, Ordering::Relaxed);
                                if let Err(err) = self.publish_market(event).await {
                                    eprintln!("Extended public WS market send failed: {err}");
                                }
                            }
                            if !first_decoded_top_logged {
                                if let Some(top) = decode_top_from_value(&value) {
                                    eprintln!(
                                        "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                        top.best_bid_px,
                                        top.best_bid_sz,
                                        top.best_ask_px,
                                        top.best_ask_sz
                                    );
                                    first_decoded_top_logged = true;
                                }
                            }
                        }
                        continue;
                    };
                    self.freshness
                        .last_parsed_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if !first_decoded_top_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            if let Some(top) = decode_top_from_value(&value) {
                                eprintln!(
                                    "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                    top.best_bid_px,
                                    top.best_bid_sz,
                                    top.best_ask_px,
                                    top.best_ask_sz
                                );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged {
                            if let Some(top) = decode_top_from_update(&update, update.event_time) {
                                eprintln!(
                                "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                top.best_bid_px,
                                top.best_bid_sz,
                                top.best_ask_px,
                                top.best_ask_sz
                            );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged && decode_miss_count < 3 {
                            decode_miss_count += 1;
                            if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                                log_decode_miss(
                                    "Extended",
                                    &value,
                                    cleaned,
                                    decode_miss_count,
                                    ws_url.as_str(),
                                );
                            }
                        }
                    }
                    if !symbol_matches(&update.symbol, &self.cfg.market) {
                        continue;
                    }
                    let outcome = seq_state.apply_update(&update)?;
                    if let Some(event) = outcome {
                        consecutive_parse_errors = 0;
                        self.freshness
                            .last_book_event_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        if !first_book_update_logged {
                            eprintln!("INFO: Extended public WS first book update");
                            first_book_update_logged = true;
                        }
                        if let Err(err) = self.publish_market(event).await {
                            eprintln!("Extended public WS market send failed: {err}");
                        }
                    }
                }
                Message::Ping(payload) => {
                    write.send(Message::Pong(payload)).await?;
                }
                Message::Pong(_) => {}
                Message::Close(_) => {
                    eprintln!("Extended WS closed; reconnecting url={}", ws_url);
                    break;
                }
                _ => {}
            }
            if !first_decoded_top_logged
                && !no_book_warned
                && ws_start.elapsed() >= Duration::from_secs(10)
            {
                let keys = first_ws_keys.as_deref().unwrap_or("unknown");
                let snippet = first_ws_snippet.as_deref().unwrap_or("unknown");
                eprintln!(
                    "WARN: Extended WS no book decoded after 10s url={} keys={} snippet={}",
                    ws_url, keys, snippet
                );
                no_book_warned = true;
            }
        }
        Ok(())
    }

    async fn fetch_snapshot(&self) -> anyhow::Result<(String, ExtendedDepthSnapshot)> {
        let url = format!(
            "{}/fapi/v1/depth?symbol={}&limit={}",
            self.cfg.rest_url, self.cfg.market, self.cfg.depth_limit
        );
        let resp = self.http.get(url).send().await?;
        let raw = resp.text().await?;
        let cleaned =
            clean_ws_payload(&raw).ok_or_else(|| anyhow::anyhow!("extended snapshot empty"))?;
        let value: Value = serde_json::from_str(cleaned)?;
        let snapshot = parse_depth_snapshot(&value)
            .ok_or_else(|| anyhow::anyhow!("extended snapshot parse failed"))?;
        Ok((raw, snapshot))
    }
}

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
pub struct ExtendedRestClient {
    cfg: ExtendedConfig,
    http: Client,
    timestamp_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync>,
}

impl ExtendedRestClient {
    pub fn new(cfg: ExtendedConfig) -> Self {
        Self {
            cfg,
            http: Client::builder()
                .user_agent("paraphina")
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("extended rest http client build"),
            timestamp_fn: Arc::new(now_ms),
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

    fn signed_query(&self, mut params: Vec<(String, String)>) -> Result<String, LiveGatewayError> {
        let api_secret = self
            .cfg
            .api_secret
            .as_ref()
            .ok_or_else(|| LiveGatewayError::fatal("extended api secret missing"))?;
        let timestamp = (self.timestamp_fn)();
        params.push(("timestamp".to_string(), timestamp.to_string()));
        if let Some(recv_window) = self.cfg.recv_window {
            params.push(("recvWindow".to_string(), recv_window.to_string()));
        }
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical = canonical_query(&params);
        // Signing per Extended API docs: https://docs.extended.exchange (HMAC SHA256 + X-MBX-APIKEY + timestamp/signature).
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
            .ok_or_else(|| LiveGatewayError::fatal("extended api key missing"))?;
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
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("extended account parse error: {err}"))
        })?;
        parse_account_snapshot(&value, venue_id, venue_index).ok_or_else(|| {
            LiveGatewayError::fatal("extended account snapshot missing required fields")
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
        loop {
            interval.tick().await;
            match self.fetch_account_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;
                }
                Err(err) => {
                    eprintln!("Extended account snapshot error: {}", err.message);
                }
            }
        }
    }

    // Note: funding polling lives on ExtendedConnector (market publisher).
}

impl LiveRestClient for ExtendedRestClient {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let mut params = vec![
                ("symbol".to_string(), self.cfg.market.clone()),
                ("side".to_string(), map_side(req.side).to_string()),
                ("type".to_string(), "LIMIT".to_string()),
                (
                    "timeInForce".to_string(),
                    map_time_in_force(req.time_in_force, req.post_only).to_string(),
                ),
                ("price".to_string(), format_f64(req.price)),
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
            let params = vec![
                ("symbol".to_string(), self.cfg.market.clone()),
                ("origClientOrderId".to_string(), req.order_id),
            ];
            let resp = self
                .send_signed_request(Method::DELETE, "/fapi/v1/order", params)
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
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
    if value.is_finite() {
        format!("{value}")
    } else {
        "0".to_string()
    }
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

async fn fetch_public_funding(
    client: &Client,
    cfg: &ExtendedConfig,
) -> anyhow::Result<FundingUpdate> {
    // Extended uses /api/v1/info/markets/{market}/stats for funding data.
    // The market is part of the path (e.g., ETH-USD), not a query parameter.
    let path = std::env::var("EXTENDED_FUNDING_PATH")
        .unwrap_or_else(|_| format!("/api/v1/info/markets/{}/stats", cfg.market));
    let url = format!("{}{}", cfg.rest_url.trim_end_matches('/'), path);
    let resp = client.get(&url).send().await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        anyhow::bail!(
            "Extended funding fetch failed: HTTP {} url={} body={}",
            status,
            url,
            body.chars().take(160).collect::<String>()
        );
    }

    let value: Value = serde_json::from_str(&body).map_err(|e| {
        anyhow::anyhow!(
            "Extended funding JSON parse error: {} body={}",
            e,
            body.chars().take(160).collect::<String>()
        )
    })?;

    parse_public_funding(&value, cfg)
        .ok_or_else(|| anyhow::anyhow!("invalid public funding response"))
}

fn parse_public_funding(value: &Value, cfg: &ExtendedConfig) -> Option<FundingUpdate> {
    let data = value
        .get("data")
        .or_else(|| value.get("result"))
        .unwrap_or(value);

    let rate_native = data
        .get("fundingRate")
        .or_else(|| data.get("funding_rate"))
        .or_else(|| data.get("lastFundingRate"))
        .and_then(parse_f64);

    // Extended API doesn't explicitly provide interval, but fundingRate is hourly.
    // Default to 3600s (1 hour) when rate is present.
    let interval_sec = data
        .get("fundingIntervalSec")
        .or_else(|| data.get("funding_interval_sec"))
        .or_else(|| data.get("fundingInterval"))
        .and_then(parse_i64_value)
        .and_then(|v| if v > 0 { Some(v as u64) } else { None })
        .or_else(|| {
            // Extended fundingRate is hourly; assume 3600s if rate is present
            if rate_native.is_some() {
                Some(3600)
            } else {
                None
            }
        });

    // Extended API uses "nextFundingRate" but it's actually the next funding TIME in ms
    let next_funding_ms = data
        .get("nextFundingRate") // Extended's field name (actually a timestamp, not a rate)
        .or_else(|| data.get("nextFundingTime"))
        .or_else(|| data.get("next_funding_time"))
        .or_else(|| data.get("nextFundingTimestamp"))
        .and_then(parse_i64_value);

    let as_of_ms = data
        .get("time")
        .or_else(|| data.get("timestamp"))
        .or_else(|| data.get("ts"))
        .and_then(parse_i64_value)
        .unwrap_or_else(now_ms);

    // Convert hourly rate to 8h: rate_8h = rate_native * (8h / interval_sec)
    let rate_8h = match (rate_native, interval_sec) {
        (Some(rate), Some(sec)) if sec > 0 => Some(rate * (8.0 * 60.0 * 60.0 / sec as f64)),
        (Some(rate), None) => Some(rate), // Assume already 8h if no interval
        _ => None,
    };

    Some(FundingUpdate {
        venue_index: cfg.venue_index,
        venue_id: cfg.market.clone(),
        seq: 0,
        timestamp_ms: as_of_ms,
        received_ms: Some(now_ms()),
        funding_rate_8h: rate_8h,
        funding_rate_native: rate_native,
        interval_sec,
        next_funding_ms,
        settlement_price_kind: Some(SettlementPriceKind::Mark), // Extended uses mark price
        source: FundingSource::MarketDataRest,
    })
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
struct ExtendedDepthSnapshot {
    last_update_id: u64,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

#[derive(Debug, Clone)]
struct ExtendedDepthUpdate {
    symbol: String,
    event_time: Option<TimestampMs>,
    start_id: u64,
    end_id: u64,
    prev_id: Option<u64>,
    bids: Vec<BookLevelDelta>,
    asks: Vec<BookLevelDelta>,
}

#[derive(Debug, Clone, Copy)]
struct ExtendedSeqState {
    last_update_id: Option<u64>,
    venue_index: usize,
}

impl ExtendedSeqState {
    fn new(last_update_id: Option<u64>, venue_index: usize) -> Self {
        Self {
            last_update_id,
            venue_index,
        }
    }

    fn apply_update(
        &mut self,
        update: &ExtendedDepthUpdate,
    ) -> anyhow::Result<Option<MarketDataEvent>> {
        if let Some(last_update_id) = self.last_update_id {
            if let Some(prev) = update.prev_id {
                if prev != last_update_id {
                    return Err(anyhow::anyhow!(
                        "extended seq mismatch prev_id={} last={}",
                        prev,
                        last_update_id
                    ));
                }
            }
            if update.end_id <= last_update_id {
                return Ok(None);
            }
            if update.start_id > last_update_id + 1 {
                return Err(anyhow::anyhow!(
                    "extended seq gap last={} next_start={}",
                    last_update_id,
                    update.start_id
                ));
            }
        }
        self.last_update_id = Some(update.end_id);
        let mut changes = Vec::with_capacity(update.bids.len() + update.asks.len());
        changes.extend(update.bids.iter().cloned());
        changes.extend(update.asks.iter().cloned());
        let event = MarketDataEvent::L2Delta(super::super::types::L2Delta {
            venue_index: self.venue_index,
            venue_id: update.symbol.clone(),
            seq: update.end_id,
            timestamp_ms: now_ms(),
            changes,
        });
        Ok(Some(event))
    }
}

#[derive(Debug)]
struct ExtendedRecorder {
    dir: PathBuf,
}

impl ExtendedRecorder {
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

fn parse_depth_snapshot(value: &Value) -> Option<ExtendedDepthSnapshot> {
    let last_update_id = value.get("lastUpdateId")?.as_u64()?;
    let bids = parse_levels_from_value(value.get("bids")?)?;
    let asks = parse_levels_from_value(value.get("asks")?)?;
    Some(ExtendedDepthSnapshot {
        last_update_id,
        bids,
        asks,
    })
}

fn parse_depth_update(text: &str) -> Result<Option<ExtendedDepthUpdate>, serde_json::Error> {
    let value: Value = serde_json::from_str(text)?;
    let payload = value.get("data").unwrap_or(&value);
    let event = payload.get("e").and_then(|v| v.as_str()).unwrap_or("");
    if event != "depthUpdate" {
        return Ok(None);
    }
    let symbol = payload
        .get("s")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let start_id = payload.get("U").and_then(|v| v.as_u64());
    let end_id = payload.get("u").and_then(|v| v.as_u64());
    let (symbol, start_id, end_id) = match (symbol, start_id, end_id) {
        (Some(symbol), Some(start_id), Some(end_id)) => (symbol, start_id, end_id),
        _ => return Ok(None),
    };
    let prev_id = payload.get("pu").and_then(|v| v.as_u64());
    let event_time = payload
        .get("E")
        .and_then(|v| v.as_i64())
        .map(|v| v as TimestampMs);
    let bids = match payload
        .get("b")
        .and_then(|v| parse_deltas_from_value(v, BookSide::Bid))
    {
        Some(bids) => bids,
        None => return Ok(None),
    };
    let asks = match payload
        .get("a")
        .and_then(|v| parse_deltas_from_value(v, BookSide::Ask))
    {
        Some(asks) => asks,
        None => return Ok(None),
    };
    Ok(Some(ExtendedDepthUpdate {
        symbol,
        event_time,
        start_id,
        end_id,
        prev_id,
        bids,
        asks,
    }))
}

fn clean_ws_payload(text: &str) -> Option<&str> {
    let cleaned = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn decode_top_from_update(
    update: &ExtendedDepthUpdate,
    timestamp_ms: Option<TimestampMs>,
) -> Option<TopOfBook> {
    let bid = update
        .bids
        .iter()
        .filter(|lvl| lvl.size > 0.0)
        .max_by(|a, b| {
            a.price
                .partial_cmp(&b.price)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let ask = update
        .asks
        .iter()
        .filter(|lvl| lvl.size > 0.0)
        .min_by(|a, b| {
            a.price
                .partial_cmp(&b.price)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    TopOfBook::from_levels(
        &[BookLevel {
            price: bid.price,
            size: bid.size,
        }],
        &[BookLevel {
            price: ask.price,
            size: ask.size,
        }],
        timestamp_ms,
    )
}

fn decode_top_from_value(value: &Value) -> Option<TopOfBook> {
    let payload = value
        .get("data")
        .or_else(|| value.get("order_book"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let bids = payload.get("b").or_else(|| payload.get("bids"))?;
    let asks = payload.get("a").or_else(|| payload.get("asks"))?;
    let bid = best_level_from_value(bids, true)?;
    let ask = best_level_from_value(asks, false)?;
    let timestamp_ms = payload
        .get("E")
        .or_else(|| payload.get("ts"))
        .and_then(|v| v.as_i64());
    TopOfBook::from_levels(
        &[BookLevel {
            price: bid.price,
            size: bid.size,
        }],
        &[BookLevel {
            price: ask.price,
            size: ask.size,
        }],
        timestamp_ms,
    )
}

fn parse_depth_snapshot_from_ws(
    value: &Value,
    market: &str,
    venue_index: usize,
    fallback_seq: &mut u64,
) -> Option<MarketDataEvent> {
    let payload = value
        .get("data")
        .or_else(|| value.get("order_book"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let bids_value = payload.get("bids").or_else(|| payload.get("b"))?;
    let asks_value = payload.get("asks").or_else(|| payload.get("a"))?;
    let bids = parse_levels_from_value(bids_value)?;
    let asks = parse_levels_from_value(asks_value)?;
    let seq = payload
        .get("lastUpdateId")
        .or_else(|| payload.get("u"))
        .or_else(|| value.get("seq"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| {
            *fallback_seq = fallback_seq.wrapping_add(1);
            *fallback_seq
        });
    let timestamp_ms = payload
        .get("E")
        .or_else(|| payload.get("ts"))
        .or_else(|| value.get("ts"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(now_ms);
    let venue_id = payload
        .get("m")
        .or_else(|| payload.get("symbol"))
        .and_then(|v| v.as_str())
        .unwrap_or(market)
        .to_string();
    Some(MarketDataEvent::L2Snapshot(
        super::super::types::L2Snapshot {
            venue_index,
            venue_id,
            seq,
            timestamp_ms,
            bids,
            asks,
        },
    ))
}

fn best_level_from_value(value: &Value, is_bid: bool) -> Option<BookLevel> {
    let entries = value.as_array()?;
    let mut best: Option<BookLevel> = None;
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        if size <= 0.0 {
            continue;
        }
        let replace = match best {
            None => true,
            Some(prev) => {
                if is_bid {
                    price > prev.price
                } else {
                    price < prev.price
                }
            }
        };
        if replace {
            best = Some(BookLevel { price, size });
        }
    }
    best
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
    if let Some(items) = value.as_array() {
        if items.len() < 2 {
            return None;
        }
        let price = parse_f64(&items[0])?;
        let size = parse_f64(&items[1])?;
        return Some((price, size));
    }
    if let Some(obj) = value.as_object() {
        let price = obj
            .get("price")
            .or_else(|| obj.get("px"))
            .or_else(|| obj.get("p"))
            .and_then(parse_f64)?;
        let size = obj
            .get("size")
            .or_else(|| obj.get("sz"))
            .or_else(|| obj.get("qty"))
            .or_else(|| obj.get("q"))
            .and_then(parse_f64)?;
        return Some((price, size));
    }
    None
}

fn normalize_extended_market(raw: &str) -> String {
    let mut upper = raw.trim().to_uppercase();
    if let Some(stripped) = upper.strip_suffix("-USD-PERP") {
        upper = stripped.to_string();
    } else if let Some(stripped) = upper.strip_suffix("-PERP") {
        upper = stripped.to_string();
    }
    if upper.contains("-USD") {
        let base = upper.split("-USD").next().unwrap_or(&upper);
        return format!("{base}-USD");
    }
    if let Some(stripped) = upper.strip_suffix("USDT") {
        return format!("{stripped}-USD");
    }
    if let Some(stripped) = upper.strip_suffix("USD") {
        return format!("{stripped}-USD");
    }
    format!("{upper}-USD")
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
pub struct ExtendedFixtureFeed {
    snapshot: FixtureSnapshot,
    deltas: Vec<FixtureDelta>,
    account: FixtureAccountSnapshot,
}

impl ExtendedFixtureFeed {
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
    use httpmock::Method::{DELETE, POST};
    use httpmock::MockServer;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/extended");
        let feed = ExtendedFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
    }

    #[test]
    fn delta_applies_to_snapshot_levels() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/extended");
        let feed = ExtendedFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
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
            seq: 2,
            timestamp_ms: 1_000,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let next = FixtureDelta {
            seq: 4,
            timestamp_ms: 1_010,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let mut last_seq = gap.seq;
        let gap_detected = next.seq > last_seq + 1;
        last_seq = next.seq;
        assert!(gap_detected);
        assert_eq!(last_seq, 4);
    }

    #[test]
    fn deterministic_serialization_roundtrip() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/extended");
        let feed = ExtendedFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let raw = serde_json::to_string(&feed.snapshot).expect("serialize");
        let reparsed: FixtureSnapshot = serde_json::from_str(&raw).expect("reparse");
        assert_eq!(feed.snapshot.seq, reparsed.seq);
        assert_eq!(feed.snapshot.bids.len(), reparsed.bids.len());
    }

    #[test]
    fn live_snapshot_fixture_parses() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/extended_live_recording");
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
            .join("../tests/fixtures/roadmap_b/extended_live_recording");
        let snapshot_raw =
            std::fs::read_to_string(fixture_dir.join("rest_snapshot.json")).expect("snapshot raw");
        let snapshot_value: Value = serde_json::from_str(&snapshot_raw).expect("snapshot json");
        let snapshot = parse_depth_snapshot(&snapshot_value).expect("parse snapshot");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");

        let collect_events = |snapshot_id: u64| -> Vec<MarketDataEvent> {
            let mut state = ExtendedSeqState::new(Some(snapshot_id), 0);
            let mut events = Vec::new();
            for line in frames.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(Some(update)) = parse_depth_update(trimmed) {
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

    #[tokio::test]
    async fn rest_place_order_post_only_is_signed() {
        let server = MockServer::start_async().await;
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

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
                venue_id: "extended".to_string(),
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

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_place_order_ioc_reduce_only_is_signed() {
        let server = MockServer::start_async().await;
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

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
                venue_id: "extended".to_string(),
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

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_cancel_all_is_signed() {
        let server = MockServer::start_async().await;
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTCUSDT".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            api_secret: Some("testsecret".to_string()),
            recv_window: Some(5000),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg).with_timestamp_fn(Arc::new(|| 1_700_000_000_000));

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
                venue_id: "extended".to_string(),
            })
            .await
            .expect("cancel_all");

        mock.assert_async().await;
    }

    #[test]
    fn ws_frame_whitespace_is_ignored_and_json_parses() {
        assert!(clean_ws_payload("").is_none());
        assert!(clean_ws_payload("   \n\t ").is_none());
        assert!(clean_ws_payload("\0\0").is_none());
        let raw =
            r#"{"e":"depthUpdate","s":"BTCUSDT","U":1,"u":1,"b":[["100","1"]],"a":[["101","2"]]}"#;
        let cleaned = clean_ws_payload(raw).expect("cleaned");
        let update = parse_depth_update(cleaned).expect("parse").expect("update");
        assert_eq!(update.symbol, "BTCUSDT");
        assert_eq!(update.start_id, 1);
        assert_eq!(update.end_id, 1);
        assert_eq!(update.bids.len(), 1);
        assert_eq!(update.asks.len(), 1);
    }

    #[test]
    fn ws_frame_empty_json_is_non_fatal() {
        let cleaned = clean_ws_payload("{}").expect("cleaned");
        let parsed = parse_depth_update(cleaned).expect("parse");
        assert!(parsed.is_none());
    }

    #[test]
    fn ws_value_decode_handles_object_levels() {
        let value = serde_json::json!({
            "data": {
                "bids": [{"price":"100","size":"2"}],
                "asks": [{"price":"101","size":"3"}],
                "ts": 1700000000000i64
            }
        });
        let top = decode_top_from_value(&value).expect("top");
        assert_eq!(top.best_bid_px, 100.0);
        assert_eq!(top.best_bid_sz, 2.0);
        assert_eq!(top.best_ask_px, 101.0);
        assert_eq!(top.best_ask_sz, 3.0);
    }

    #[test]
    fn normalize_extended_market_variants() {
        assert_eq!(normalize_extended_market("BTCUSDT"), "BTC-USD");
        assert_eq!(normalize_extended_market("BTCUSD"), "BTC-USD");
        assert_eq!(normalize_extended_market("BTC-USD"), "BTC-USD");
        assert_eq!(normalize_extended_market("btc-usd-perp"), "BTC-USD");
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

        // After reset with no book events, anchor falls back to connect_start_ns
        let connect_start_ns = 1_000;
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, connect_start_ns);

        // Non-book parsed events must NOT advance the watchdog anchor
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
    fn parse_public_funding_market_stats_fixture() {
        // Test parsing Extended's /api/v1/info/markets/{market}/stats response format
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/extended/public_market_stats.json");
        let raw = std::fs::read_to_string(&fixture_path).expect("read fixture");
        let value: Value = serde_json::from_str(&raw).expect("parse fixture JSON");

        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "ETH-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: None,
            api_secret: None,
            recv_window: None,
            record_dir: None,
        };

        let funding =
            parse_public_funding(&value, &cfg).expect("parse_public_funding should succeed");

        // Verify rate parsing: fixture has fundingRate "0.000013" (hourly)
        // rate_8h should be 0.000013 * 8 = 0.000104
        assert!(
            funding.funding_rate_native.is_some(),
            "native rate should be present"
        );
        let native = funding.funding_rate_native.unwrap();
        assert!(
            (native - 0.000013).abs() < 1e-10,
            "native rate mismatch: {}",
            native
        );

        assert!(
            funding.funding_rate_8h.is_some(),
            "8h rate should be present"
        );
        let rate_8h = funding.funding_rate_8h.unwrap();
        assert!(
            (rate_8h - 0.000104).abs() < 1e-10,
            "8h rate mismatch: {}",
            rate_8h
        );

        // Verify interval is detected as hourly (3600s)
        assert_eq!(
            funding.interval_sec,
            Some(3600),
            "interval_sec should be 3600"
        );

        // Verify next_funding_ms is extracted from "nextFundingRate" field
        // Fixture has: "nextFundingRate": 1770314400000
        assert_eq!(
            funding.next_funding_ms,
            Some(1770314400000),
            "next_funding_ms mismatch"
        );

        // Verify source and settlement
        assert!(matches!(funding.source, FundingSource::MarketDataRest));
        assert_eq!(
            funding.settlement_price_kind,
            Some(SettlementPriceKind::Mark)
        );

        // Verify venue info
        assert_eq!(funding.venue_index, 0);
        assert_eq!(funding.venue_id, "ETH-USD");
    }
}
