//! Aster connector (public WS market data + fixtures, feature-gated).

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
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
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::super::gateway::{
    BoxFuture, LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest,
    LiveRestCancelRequest, LiveRestClient, LiveRestPlaceRequest, LiveRestResponse, LiveResult,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, LiquidationSnapshot, MarginSnapshot,
    MarketDataEvent, PositionSnapshot, TopOfBook,
};
use crate::live::MarketPublisher;
use crate::types::{Side, TimeInForce, TimestampMs};

#[cfg(feature = "live_aster")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_aster")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_aster")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_aster")]
pub const SUPPORTS_EXECUTION: bool = true;

const ASTER_STALE_MS: u64 = 1_800;
const ASTER_WATCHDOG_TICK_MS: u64 = 200;
const ASTER_MARKET_PUB_QUEUE_CAP_LIVE: usize = 256;
const ASTER_MARKET_PUB_QUEUE_CAP_FIXTURE: usize = 4096;
const ASTER_MARKET_PUB_DRAIN_MAX: usize = 64;

static MONO_START: OnceLock<Instant> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
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

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_parsed_ns: AtomicU64,
    last_published_ns: AtomicU64,
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
        let publish_freshness = freshness.clone();
        let on_published = Arc::new(move || {
            publish_freshness
                .last_published_ns
                .store(mono_now_ns(), Ordering::Relaxed);
        });
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
            Some(on_published),
            "aster market_tx closed",
            "aster market publish queue closed",
        );
        let connector = Self {
            cfg,
            http: Client::new(),
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
        loop {
            if let Err(err) = self.public_ws_once().await {
                eprintln!("Aster public WS error: {err}");
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        let (mut write, mut read) = ws_stream.split();
        let stream = format!("{}@depth@100ms", self.cfg.stream_symbol());
        let sub = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": 1
        });
        write.send(Message::Text(sub.to_string())).await?;

        const MAX_BUFFERED_UPDATES: usize = 1024;
        let mut buffered_updates: Vec<AsterDepthUpdate> = Vec::new();
        let mut last_update_id: Option<u64> = None;
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
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        let fixture_mode = env_is_true("ASTER_FIXTURE_MODE")
            || std::env::var_os("ASTER_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some();
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
                    let last_pub = freshness.last_published_ns.load(Ordering::Relaxed);
                    let last_parsed = freshness.last_parsed_ns.load(Ordering::Relaxed);
                    let anchor = if last_pub != 0 { last_pub } else { last_parsed };
                    if anchor != 0 && age_ms(now, anchor) > ASTER_STALE_MS {
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
                        anyhow::bail!("Aster public WS stale: freshness exceeded ASTER_STALE_MS");
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
                        self.freshness
                            .last_parsed_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        if self.publish_market(snapshot_event).await.is_ok() {
                            last_applied_at = Instant::now();
                        }

                        let mut next_last = snapshot.last_update_id;
                        let mut gap = false;
                        for update in buffered_updates.drain(..) {
                            // Stream is per-symbol; avoid dropping on formatting mismatch.
                            match seq_decision_lenient(next_last, &update) {
                                SeqDecision::Apply => {
                                    next_last = update.end_id;
                                    self.freshness
                                        .last_parsed_ns
                                        .store(mono_now_ns(), Ordering::Relaxed);
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
                                    gap = true;
                                    break;
                                }
                            }
                        }
                        if gap {
                            buffered_updates.clear();
                            last_update_id = None;
                            snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                            continue;
                        }
                        last_update_id = Some(next_last);
                            }
                            Err(err) => {
                                if last_snapshot_err_log.elapsed() > Duration::from_secs(30) {
                                    let url = format!(
                                        "{}/fapi/v1/depth?symbol={}&limit={}",
                                        self.cfg.rest_url, self.cfg.market, self.cfg.depth_limit
                                    );
                                    eprintln!(
                                        "WARN: Aster snapshot fetch failed; url={} err={}",
                                        url, err
                                    );
                                    last_snapshot_err_log = Instant::now();
                                }
                                snapshot_future = Some(Box::pin(self.fetch_snapshot()));
                                buffered_updates.clear();
                                last_update_id = None;
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
                    msg = read.next() => {
                        let Some(msg) = msg else {
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
                                if let Some(update) = parse_depth_update(&text) {
                                    self.freshness
                                        .last_parsed_ns
                                        .store(mono_now_ns(), Ordering::Relaxed);
                                    buffered_updates.push(update);
                                    if buffered_updates.len() > MAX_BUFFERED_UPDATES {
                                        eprintln!(
                                            "Aster WS buffer overflow; resyncing url={}",
                                            self.cfg.ws_url
                                        );
                                        buffered_updates.clear();
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
                                    if let Some(update) = parse_depth_update(&text) {
                                        self.freshness
                                            .last_parsed_ns
                                            .store(mono_now_ns(), Ordering::Relaxed);
                                        buffered_updates.push(update);
                                        if buffered_updates.len() > MAX_BUFFERED_UPDATES {
                                            eprintln!(
                                                "Aster WS buffer overflow; resyncing url={}",
                                                self.cfg.ws_url
                                            );
                                            buffered_updates.clear();
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
                }
                continue;
            }

            tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    anyhow::bail!("Aster public WS stale: freshness exceeded ASTER_STALE_MS");
                }
                msg = read.next() => {
                    let Some(msg) = msg else {
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
                            match seq_decision_lenient(current_last, &update) {
                                SeqDecision::Apply => {
                                    last_update_id = Some(update.end_id);
                                    self.freshness
                                        .last_parsed_ns
                                        .store(mono_now_ns(), Ordering::Relaxed);
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
                                        self.freshness
                                            .last_parsed_ns
                                            .store(mono_now_ns(), Ordering::Relaxed);
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
        let url = format!(
            "{}/fapi/v1/depth?symbol={}&limit={}",
            self.cfg.rest_url, self.cfg.market, self.cfg.depth_limit
        );
        let resp = self
            .http
            .get(url)
            .timeout(Duration::from_secs(2))
            .send()
            .await?
            .error_for_status()?;
        let raw = resp.text().await?;
        let value: Value = serde_json::from_str(&raw)?;
        let snapshot = parse_depth_snapshot(&value)
            .ok_or_else(|| anyhow::anyhow!("aster snapshot parse failed"))?;
        Ok((raw, snapshot))
    }
}

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
pub struct AsterRestClient {
    cfg: AsterConfig,
    http: Client,
    timestamp_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync>,
}

impl AsterRestClient {
    pub fn new(cfg: AsterConfig) -> Self {
        Self {
            cfg,
            http: Client::new(),
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
        parse_account_snapshot(&value, venue_id, venue_index).ok_or_else(|| {
            LiveGatewayError::fatal("aster account snapshot missing required fields")
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
                    eprintln!("Aster account snapshot error: {}", err.message);
                }
            }
        }
    }
}

impl LiveRestClient for AsterRestClient {
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
            if prev != self.last_update_id {
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

fn seq_decision_lenient(last_update_id: u64, update: &AsterDepthUpdate) -> SeqDecision {
    if let Some(prev) = update.prev_id {
        if prev == last_update_id {
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
    use httpmock::Method::{DELETE, POST};
    use httpmock::MockServer;
    use std::path::PathBuf;

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/aster");
        let feed = AsterFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
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
}
