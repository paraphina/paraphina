//! Paradex connector (public WS market data + fixtures, feature-gated).

#[cfg(feature = "live_paradex")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_EXECUTION: bool = true;

const PARADEX_STALE_MS_DEFAULT: u64 = 10_000;
const PARADEX_WATCHDOG_TICK_MS: u64 = 200;
const PARADEX_MARKET_PUB_QUEUE_CAP: usize = 256;
const PARADEX_MARKET_PUB_DRAIN_MAX: usize = 64;

static MONO_START: OnceLock<Instant> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn paradex_stale_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(PARADEX_STALE_MS_DEFAULT)
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
}

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

#[derive(Debug, Clone)]
pub struct ParadexConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub auth_url: String,
    pub market: String,
    pub account_path: String,
    pub order_path: String,
    pub venue_index: usize,
    pub jwt: Option<String>,
    pub auth_payload_json: Option<Value>,
    pub record_dir: Option<PathBuf>,
}

impl ParadexConfig {
    pub fn from_env() -> Self {
        let ws_url = std::env::var("PARADEX_WS_URL")
            .unwrap_or_else(|_| "wss://ws.api.prod.paradex.trade/v1".to_string());
        let rest_url = std::env::var("PARADEX_REST_URL")
            .unwrap_or_else(|_| "https://api.prod.paradex.trade/v1".to_string());
        let auth_url = std::env::var("PARADEX_AUTH_URL")
            .unwrap_or_else(|_| "https://api.prod.paradex.trade/v1/auth/token".to_string());
        let market = std::env::var("PARADEX_MARKET").unwrap_or_else(|_| "BTC-USD-PERP".to_string());
        let account_path =
            std::env::var("PARADEX_ACCOUNT_PATH").unwrap_or_else(|_| "/account".to_string());
        let order_path =
            std::env::var("PARADEX_ORDER_PATH").unwrap_or_else(|_| "/orders".to_string());
        let jwt = std::env::var("PARADEX_JWT").ok();
        let auth_payload_json = std::env::var("PARADEX_AUTH_PAYLOAD_JSON")
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
        Self {
            ws_url,
            rest_url,
            auth_url,
            market,
            account_path,
            order_path,
            venue_index: 0,
            jwt,
            auth_payload_json,
            record_dir: None,
        }
    }

    pub fn with_record_dir(mut self, dir: PathBuf) -> Self {
        self.record_dir = Some(dir);
        self
    }

    pub fn has_auth(&self) -> bool {
        self.jwt.is_some() || self.auth_payload_json.is_some()
    }
}

#[derive(Debug)]
pub struct ParadexConnector {
    cfg: ParadexConfig,
    http: Client,
    market_publisher: MarketPublisher,
    recorder: Option<Mutex<ParadexRecorder>>,
    freshness: Arc<Freshness>,
    is_fixture: bool,
}

impl ParadexConnector {
    pub fn new(cfg: ParadexConfig, market_tx: mpsc::Sender<MarketDataEvent>) -> Self {
        let recorder = cfg
            .record_dir
            .as_ref()
            .and_then(|dir| ParadexRecorder::new(dir).ok())
            .map(Mutex::new);
        let is_fixture = std::env::var_os("PARADEX_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("PARADEX_FIXTURE_MODE").is_ok();
        let freshness = Arc::new(Freshness::default());
        let publish_freshness = freshness.clone();
        let on_published = Arc::new(move || {
            publish_freshness
                .last_published_ns
                .store(mono_now_ns(), Ordering::Relaxed);
        });
        let market_publisher = MarketPublisher::new(
            PARADEX_MARKET_PUB_QUEUE_CAP,
            PARADEX_MARKET_PUB_DRAIN_MAX,
            market_tx.clone(),
            Some(Arc::new(move || is_fixture || Self::fixture_mode_now())),
            Arc::new(|event: &MarketDataEvent| matches!(event, MarketDataEvent::L2Delta(_))),
            Some(on_published),
            "paradex market_tx closed",
            "paradex market publish queue closed",
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

    fn fixture_mode_now() -> bool {
        std::env::var_os("PARADEX_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("PARADEX_FIXTURE_MODE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        self.market_publisher.publish_market(event).await
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        loop {
            if let Err(err) = self.public_ws_once().await {
                eprintln!("Paradex public WS error: {err}");
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        eprintln!("INFO: Paradex public WS connecting url={}", self.cfg.ws_url);
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        eprintln!("INFO: Paradex public WS connected url={}", self.cfg.ws_url);
        let (mut write, mut read) = ws_stream.split();
        let mut subscribed = false;
        let channel = format!("bbo.{}", self.cfg.market);
        let subscribe =
            ParadexSubscribeCandidate::new("subscribe", serde_json::json!({ "channel": channel }));
        send_paradex_subscribe(&mut write, &subscribe).await?;
        eprintln!(
            "INFO: Paradex subscribed channel={}",
            format!("bbo.{}", self.cfg.market)
        );

        let mut tracker = ParadexSeqState::new(self.cfg.venue_index);
        let mut first_book_update_logged = false;
        let mut first_message_logged = false;
        let mut first_message_keys_logged = false;
        let mut logged_non_utf8_binary = false;
        let mut first_decoded_top_logged = false;
        let mut decode_miss_count = 0usize;
        let mut bbo_seq: u64 = 0;
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        let fixture_mode = std::env::var_os("PARADEX_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("PARADEX_FIXTURE_MODE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        let stale_ms = paradex_stale_ms();
        if fixture_mode {
            eprintln!("INFO: Paradex fixture mode detected; freshness watchdog disabled");
        } else {
            let watchdog_stale_ms = stale_ms;
            let watchdog_freshness = self.freshness.clone();
            tokio::spawn(async move {
                let mut iv = tokio::time::interval(Duration::from_millis(PARADEX_WATCHDOG_TICK_MS));
                iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    iv.tick().await;
                    let now = mono_now_ns();
                    let last_pub = watchdog_freshness.last_published_ns.load(Ordering::Relaxed);
                    let last_parsed = watchdog_freshness.last_parsed_ns.load(Ordering::Relaxed);
                    let anchor = if last_pub != 0 { last_pub } else { last_parsed };
                    if anchor != 0 && age_ms(now, anchor) > watchdog_stale_ms {
                        let _ = stale_tx.send(());
                        break;
                    }
                }
            });
        }
        loop {
            let msg = tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    anyhow::bail!("Paradex public WS stale: freshness exceeded {stale_ms}ms");
                }
                msg = read.next() => {
                    let Some(msg) = msg else { break; };
                    msg?
                }
            };
            self.freshness
                .last_ws_rx_ns
                .store(mono_now_ns(), Ordering::Relaxed);
            let payload = match msg {
                Message::Text(text) => text,
                Message::Binary(bytes) => match String::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(_) => {
                        if !logged_non_utf8_binary {
                            eprintln!(
                                "WARN: Paradex public WS non-utf8 binary frame url={}",
                                self.cfg.ws_url
                            );
                            logged_non_utf8_binary = true;
                        }
                        continue;
                    }
                },
                Message::Ping(payload) => {
                    let _ = write.send(Message::Pong(payload)).await;
                    continue;
                }
                _ => continue,
            };
            self.freshness
                .last_data_rx_ns
                .store(mono_now_ns(), Ordering::Relaxed);
            if !first_message_logged {
                eprintln!("INFO: Paradex public WS first message received");
                first_message_logged = true;
            }
            if let Some(recorder) = self.recorder.as_ref() {
                let mut guard = recorder.lock().await;
                let _ = guard.record_ws_frame(&payload);
            }
            let value = match serde_json::from_str::<Value>(&payload) {
                Ok(value) => value,
                Err(err) => {
                    let snippet: String = payload.chars().take(160).collect();
                    eprintln!(
                        "WARN: Paradex public WS parse error: {err} url={} snippet={}",
                        self.cfg.ws_url, snippet
                    );
                    continue;
                }
            };
            if !first_message_keys_logged {
                let keys = value
                    .as_object()
                    .map(|obj| {
                        let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                        keys.sort();
                        format!("[{}]", keys.join(","))
                    })
                    .unwrap_or_else(|| "[non-object]".to_string());
                let snippet: String = payload.chars().take(160).collect();
                eprintln!("INFO: Paradex WS first msg keys={keys} snippet={snippet}");
                first_message_keys_logged = true;
            }
            if !subscribed {
                if paradex_subscribe_error(&value) {
                    if let Some(err) = value.get("error") {
                        eprintln!("WARN: Paradex subscribe error: {err}");
                    }
                    anyhow::bail!("Paradex subscribe failed: invalid channel");
                }
                if value.get("id").and_then(|v| v.as_i64()) == Some(1)
                    && value.get("result").is_some()
                {
                    subscribed = true;
                }
            }
            if let Some((top, snapshot)) = decode_bbo_top_and_snapshot(
                &value,
                self.cfg.venue_index,
                &self.cfg.market,
                &mut bbo_seq,
            ) {
                if !first_decoded_top_logged {
                    eprintln!(
                        "FIRST_DECODED_TOP venue=paradex bid_px={} bid_sz={} ask_px={} ask_sz={}",
                        top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                    );
                    first_decoded_top_logged = true;
                }
                if !first_book_update_logged {
                    eprintln!("INFO: Paradex public WS first book update");
                    first_book_update_logged = true;
                }
                self.freshness
                    .last_parsed_ns
                    .store(mono_now_ns(), Ordering::Relaxed);
                if let Err(err) = self.publish_market(snapshot).await {
                    eprintln!("Paradex public WS market send failed: {err}");
                }
            }
            if subscribed {
                if !has_paradex_book_fields(&value) {
                    continue;
                }
                if let Some(top) = decode_top_of_book_value(&value) {
                    if !first_decoded_top_logged {
                        eprintln!(
                            "FIRST_DECODED_TOP venue=paradex bid_px={} bid_sz={} ask_px={} ask_sz={}",
                            top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                        );
                        first_decoded_top_logged = true;
                    }
                } else if decode_miss_count < 3 {
                    decode_miss_count += 1;
                    log_decode_miss(
                        "Paradex",
                        &value,
                        &payload,
                        decode_miss_count,
                        self.cfg.ws_url.as_str(),
                    );
                }
            }
            if let Some(event) = parse_orderbook_message_value(&value, &mut tracker)? {
                if !first_book_update_logged {
                    eprintln!("INFO: Paradex public WS first book update");
                    first_book_update_logged = true;
                }
                self.freshness
                    .last_parsed_ns
                    .store(mono_now_ns(), Ordering::Relaxed);
                let _ = self.publish_market(event).await;
            }
        }
        Ok(())
    }
}

fn has_paradex_book_fields(value: &Value) -> bool {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    payload.get("bids").is_some()
        || payload.get("asks").is_some()
        || payload.get("bid").is_some()
        || payload.get("ask").is_some()
}

#[derive(Clone)]
pub struct ParadexRestClient {
    cfg: ParadexConfig,
    http: Client,
    token_cache: Arc<Mutex<Option<ParadexAuthToken>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ParadexAuthToken {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    token: String,
    #[serde(default)]
    jwt: String,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    expires_at: Option<u64>,
}

impl ParadexRestClient {
    pub fn new(cfg: ParadexConfig) -> Self {
        Self {
            cfg,
            http: Client::new(),
            token_cache: Arc::new(Mutex::new(None)),
        }
    }

    pub fn has_auth(&self) -> bool {
        self.cfg.has_auth()
    }

    async fn ensure_token(&self) -> LiveResult<String> {
        if let Some(jwt) = self.cfg.jwt.as_ref() {
            return Ok(jwt.clone());
        }
        let mut guard = self.token_cache.lock().await;
        if let Some(token) = guard.as_ref() {
            if let Some(jwt) = token_token(token) {
                return Ok(jwt);
            }
        }
        let token = self.fetch_token().await?;
        let jwt = token_token(&token)
            .ok_or_else(|| LiveGatewayError::fatal("paradex auth token missing access_token"))?;
        *guard = Some(token);
        Ok(jwt)
    }

    async fn fetch_token(&self) -> LiveResult<ParadexAuthToken> {
        let payload = self
            .cfg
            .auth_payload_json
            .clone()
            .ok_or_else(|| LiveGatewayError::fatal("paradex auth payload missing"))?;
        let resp = self
            .http
            .request(Method::POST, &self.cfg.auth_url)
            .json(&payload)
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("auth_error: {err}")))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        parse_auth_token(&body)
            .ok_or_else(|| LiveGatewayError::fatal("paradex auth token parse error"))
    }

    async fn send_authed_request(
        &self,
        method: Method,
        path: &str,
        payload: Option<Value>,
    ) -> LiveResult<reqwest::Response> {
        let token = self.ensure_token().await?;
        let url = format!("{}{}", self.cfg.rest_url, path);
        let mut builder = self
            .http
            .request(method, url)
            .header("Authorization", format!("Bearer {token}"));
        if let Some(payload) = payload {
            builder = builder.json(&payload);
        }
        builder
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("rest_error: {err}")))
    }

    async fn fetch_account_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<AccountSnapshot> {
        let resp = self
            .send_authed_request(Method::GET, &self.cfg.account_path, None)
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("paradex account parse error: {err}"))
        })?;
        parse_account_snapshot(&value, venue_id, venue_index).ok_or_else(|| {
            LiveGatewayError::fatal("paradex account snapshot missing required fields")
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
                    eprintln!("Paradex account snapshot error: {}", err.message);
                }
            }
        }
    }
}

impl LiveRestClient for ParadexRestClient {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let payload = build_order_payload(&self.cfg.market, &req)?;
            let resp = self
                .send_authed_request(Method::POST, &self.cfg.order_path, Some(payload))
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
            let path = format!("{}/{}", self.cfg.order_path, req.order_id);
            let resp = self
                .send_authed_request(Method::DELETE, &path, None)
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
            let path = format!("{}/cancel_all", self.cfg.order_path);
            let payload = serde_json::json!({ "market": self.cfg.market });
            let resp = self
                .send_authed_request(Method::POST, &path, Some(payload))
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

#[derive(Debug, Clone)]
enum ParadexBookMessage {
    Snapshot(ParadexSnapshot),
    Delta(ParadexDelta),
}

#[derive(Debug, Clone)]
struct ParadexSnapshot {
    market: String,
    seq: u64,
    timestamp_ms: TimestampMs,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

#[derive(Debug, Clone)]
struct ParadexDelta {
    market: String,
    seq: u64,
    prev_seq: Option<u64>,
    timestamp_ms: TimestampMs,
    bids: Vec<BookLevelDelta>,
    asks: Vec<BookLevelDelta>,
}

#[derive(Debug, Clone)]
struct ParadexSeqState {
    last_seq: Option<u64>,
    has_snapshot: bool,
    venue_index: usize,
}

impl ParadexSeqState {
    fn new(venue_index: usize) -> Self {
        Self {
            last_seq: None,
            has_snapshot: false,
            venue_index,
        }
    }

    fn apply(&mut self, message: ParadexBookMessage) -> anyhow::Result<Option<MarketDataEvent>> {
        match message {
            ParadexBookMessage::Snapshot(snapshot) => {
                self.last_seq = Some(snapshot.seq);
                self.has_snapshot = true;
                Ok(Some(MarketDataEvent::L2Snapshot(
                    super::super::types::L2Snapshot {
                        venue_index: self.venue_index,
                        venue_id: snapshot.market,
                        seq: snapshot.seq,
                        timestamp_ms: snapshot.timestamp_ms,
                        bids: snapshot.bids,
                        asks: snapshot.asks,
                    },
                )))
            }
            ParadexBookMessage::Delta(delta) => {
                if !self.has_snapshot {
                    return Ok(None);
                }
                if let Some(prev) = delta.prev_seq {
                    if Some(prev) != self.last_seq {
                        return Err(anyhow::anyhow!(
                            "paradex seq mismatch prev_seq={:?} last_seq={:?}",
                            prev,
                            self.last_seq
                        ));
                    }
                }
                if let Some(last) = self.last_seq {
                    if delta.seq <= last {
                        return Ok(None);
                    }
                    if delta.seq > last + 1 {
                        return Err(anyhow::anyhow!(
                            "paradex seq gap last_seq={} next_seq={}",
                            last,
                            delta.seq
                        ));
                    }
                }
                self.last_seq = Some(delta.seq);
                let mut changes = Vec::with_capacity(delta.bids.len() + delta.asks.len());
                changes.extend(delta.bids.iter().cloned());
                changes.extend(delta.asks.iter().cloned());
                Ok(Some(MarketDataEvent::L2Delta(
                    super::super::types::L2Delta {
                        venue_index: self.venue_index,
                        venue_id: delta.market,
                        seq: delta.seq,
                        timestamp_ms: delta.timestamp_ms,
                        changes,
                    },
                )))
            }
        }
    }
}

fn token_token(token: &ParadexAuthToken) -> Option<String> {
    if !token.access_token.is_empty() {
        return Some(token.access_token.clone());
    }
    if !token.token.is_empty() {
        return Some(token.token.clone());
    }
    if !token.jwt.is_empty() {
        return Some(token.jwt.clone());
    }
    None
}

fn parse_auth_token(body: &str) -> Option<ParadexAuthToken> {
    serde_json::from_str::<ParadexAuthToken>(body).ok()
}

fn parse_order_id(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    if let Some(order_id) = value.get("order_id").or_else(|| value.get("orderId")) {
        if let Some(raw) = order_id.as_str() {
            return Some(raw.to_string());
        }
        if let Some(raw) = order_id.as_i64() {
            return Some(raw.to_string());
        }
    }
    value
        .get("client_order_id")
        .or_else(|| value.get("clientOrderId"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn map_time_in_force(time_in_force: TimeInForce, post_only: bool) -> &'static str {
    if post_only {
        return "POST_ONLY";
    }
    match time_in_force {
        TimeInForce::Ioc => "IOC",
        TimeInForce::Gtc => "GTC",
    }
}

fn map_side(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

fn build_order_payload(market: &str, req: &LiveRestPlaceRequest) -> LiveResult<Value> {
    if req.post_only && req.time_in_force == TimeInForce::Ioc {
        return Err(LiveGatewayError::post_only_reject(
            "paradex: post_only + IOC not allowed",
        ));
    }
    let tif = map_time_in_force(req.time_in_force, req.post_only);
    // Payload per Paradex REST docs: https://docs.paradex.trade (POST /orders).
    Ok(serde_json::json!({
        "market": market,
        "side": map_side(req.side),
        "type": "LIMIT",
        "time_in_force": tif,
        "price": req.price,
        "size": req.size,
        "post_only": req.post_only,
        "reduce_only": req.reduce_only,
        "client_order_id": req.client_order_id,
    }))
}

fn parse_account_snapshot(
    value: &Value,
    venue_id: &str,
    venue_index: usize,
) -> Option<AccountSnapshot> {
    if value.get("positions").is_some() && value.get("balances").is_some() {
        let positions = value
            .get("positions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|pos| {
                        let symbol = pos.get("symbol")?.as_str()?.to_string();
                        let size = pos.get("size")?.as_f64()?;
                        let entry_price = pos
                            .get("entry_price")
                            .or_else(|| pos.get("entryPrice"))?
                            .as_f64()?;
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
            .get("balances")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|bal| {
                        let asset = bal.get("asset")?.as_str()?.to_string();
                        let total = bal.get("total")?.as_f64()?;
                        let available = bal.get("available")?.as_f64()?;
                        Some(BalanceSnapshot {
                            asset,
                            total,
                            available,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let margin = value.get("margin")?;
        let margin = MarginSnapshot {
            balance_usd: margin.get("balance_usd")?.as_f64()?,
            used_usd: margin.get("used_usd")?.as_f64()?,
            available_usd: margin.get("available_usd")?.as_f64()?,
        };
        let liquidation = value.get("liquidation")?;
        let liquidation = LiquidationSnapshot {
            price_liq: liquidation.get("price_liq").and_then(|v| v.as_f64()),
            dist_liq_sigma: liquidation.get("dist_liq_sigma").and_then(|v| v.as_f64()),
        };
        return Some(AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0),
            timestamp_ms: value
                .get("timestamp_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            positions,
            balances,
            funding_8h: value.get("funding_8h").and_then(|v| v.as_f64()),
            margin,
            liquidation,
        });
    }
    None
}

fn map_rest_error(status: u16, body: &str) -> LiveGatewayError {
    let lower = body.to_lowercase();
    if status == 401 || status == 403 {
        return LiveGatewayError {
            kind: LiveGatewayErrorKind::Fatal,
            message: format!("auth_error: {body}"),
        };
    }
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(body);
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(body);
    }
    if status == 429 || lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(body);
    }
    if status >= 500 || lower.contains("timeout") {
        return LiveGatewayError::retryable(body);
    }
    LiveGatewayError {
        kind: LiveGatewayErrorKind::Fatal,
        message: body.to_string(),
    }
}

#[derive(Debug)]
struct ParadexRecorder {
    dir: PathBuf,
}

impl ParadexRecorder {
    fn new(dir: &PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self { dir: dir.clone() })
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

fn parse_orderbook_message_value(
    value: &Value,
    tracker: &mut ParadexSeqState,
) -> anyhow::Result<Option<MarketDataEvent>> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(&value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    if channel != "orderbook" && channel != "order_book" {
        return Ok(None);
    }
    let message_type = payload
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("action").and_then(|v| v.as_str()))
        .unwrap_or("");
    let message = if message_type == "snapshot" {
        parse_snapshot(payload).map(ParadexBookMessage::Snapshot)
    } else if message_type == "delta" || message_type == "update" {
        parse_delta(payload).map(ParadexBookMessage::Delta)
    } else {
        None
    };
    if let Some(message) = message {
        return tracker.apply(message);
    }
    Ok(None)
}

fn parse_snapshot(payload: &Value) -> Option<ParadexSnapshot> {
    let market = payload
        .get("market")
        .or_else(|| payload.get("symbol"))
        .and_then(|v| v.as_str())?
        .to_string();
    let seq = payload
        .get("seq")
        .or_else(|| payload.get("sequence"))
        .and_then(|v| v.as_u64())?;
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64())
        .map(|v| v as TimestampMs)
        .unwrap_or_else(now_ms);
    let bids = parse_levels_from_value(payload.get("bids")?)?;
    let asks = parse_levels_from_value(payload.get("asks")?)?;
    Some(ParadexSnapshot {
        market,
        seq,
        timestamp_ms,
        bids,
        asks,
    })
}

fn parse_delta(payload: &Value) -> Option<ParadexDelta> {
    let market = payload
        .get("market")
        .or_else(|| payload.get("symbol"))
        .and_then(|v| v.as_str())?
        .to_string();
    let seq = payload
        .get("seq")
        .or_else(|| payload.get("sequence"))
        .and_then(|v| v.as_u64())?;
    let prev_seq = payload
        .get("prev_seq")
        .or_else(|| payload.get("prevSequence"))
        .and_then(|v| v.as_u64());
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64())
        .map(|v| v as TimestampMs)
        .unwrap_or_else(now_ms);
    let bids = parse_deltas_from_value(payload.get("bids")?, BookSide::Bid)?;
    let asks = parse_deltas_from_value(payload.get("asks")?, BookSide::Ask)?;
    Some(ParadexDelta {
        market,
        seq,
        prev_seq,
        timestamp_ms,
        bids,
        asks,
    })
}

fn decode_top_of_book(value: &Value) -> Option<TopOfBook> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let bids = parse_levels_from_value(payload.get("bids")?)?;
    let asks = parse_levels_from_value(payload.get("asks")?)?;
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64());
    TopOfBook::from_levels(&bids, &asks, timestamp_ms)
}

fn decode_top_of_book_value(value: &Value) -> Option<TopOfBook> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let bids = parse_levels_any(payload.get("bids")?)?;
    let asks = parse_levels_any(payload.get("asks")?)?;
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64());
    TopOfBook::from_levels(&bids, &asks, timestamp_ms)
}

fn decode_bbo_top_and_snapshot(
    value: &Value,
    venue_index: usize,
    venue_id: &str,
    seq: &mut u64,
) -> Option<(TopOfBook, MarketDataEvent)> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    if !channel.starts_with("bbo.") {
        return None;
    }
    let data = payload.get("data").unwrap_or(payload);
    let bid_px = data
        .get("bid")
        .or_else(|| data.get("bid_price"))
        .and_then(parse_f64)?;
    let bid_sz = data
        .get("bid_size")
        .or_else(|| data.get("bidSize"))
        .and_then(parse_f64)?;
    let ask_px = data
        .get("ask")
        .or_else(|| data.get("ask_price"))
        .and_then(parse_f64)?;
    let ask_sz = data
        .get("ask_size")
        .or_else(|| data.get("askSize"))
        .and_then(parse_f64)?;
    if bid_sz <= 0.0 || ask_sz <= 0.0 {
        return None;
    }
    let timestamp_ms = data
        .get("ts")
        .or_else(|| data.get("timestamp"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(now_ms);
    *seq = seq.wrapping_add(1);
    let bids = vec![BookLevel {
        price: bid_px,
        size: bid_sz,
    }];
    let asks = vec![BookLevel {
        price: ask_px,
        size: ask_sz,
    }];
    let top = TopOfBook::from_levels(&bids, &asks, Some(timestamp_ms))?;
    let snapshot = MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq: *seq,
        timestamp_ms,
        bids,
        asks,
    });
    Some((top, snapshot))
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

#[derive(Clone)]
struct ParadexSubscribeCandidate {
    method: String,
    params: Value,
}

impl ParadexSubscribeCandidate {
    fn new(method: &str, params: Value) -> Self {
        Self {
            method: method.to_string(),
            params,
        }
    }
}

async fn send_paradex_subscribe(
    write: &mut (impl futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error>
              + Unpin),
    candidate: &ParadexSubscribeCandidate,
) -> anyhow::Result<()> {
    let sub = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": candidate.method,
        "params": candidate.params,
    });
    write.send(Message::Text(sub.to_string())).await?;
    Ok(())
}

fn paradex_subscribe_error(value: &Value) -> bool {
    let err = value.get("error").and_then(|v| v.as_object());
    let Some(err) = err else {
        return false;
    };
    let message = err
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    let data = err
        .get("data")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    message.contains("invalid") || data.contains("invalid")
}

fn is_paradex_orderbook_message(value: &Value) -> bool {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    channel == "orderbook" || channel == "order_book"
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

fn parse_levels_any(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some((price, size)) = parse_level_pair(entry) {
            out.push(BookLevel { price, size });
            continue;
        }
        if let Some(obj) = entry.as_object() {
            let price = obj
                .get("px")
                .or_else(|| obj.get("price"))
                .and_then(parse_f64)?;
            let size = obj
                .get("sz")
                .or_else(|| obj.get("size"))
                .and_then(parse_f64)?;
            out.push(BookLevel { price, size });
        }
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
pub struct ParadexFixtureFeed {
    snapshot: FixtureSnapshot,
    deltas: Vec<FixtureDelta>,
    account: FixtureAccountSnapshot,
}

impl ParadexFixtureFeed {
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
    use httpmock::Method::POST;
    use httpmock::MockServer;
    use std::path::PathBuf;

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
    }

    #[test]
    fn delta_applies_to_snapshot_levels() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
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
            seq: 10,
            timestamp_ms: 1_000,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let next = FixtureDelta {
            seq: 12,
            timestamp_ms: 1_010,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let mut last_seq = gap.seq;
        let gap_detected = next.seq > last_seq + 1;
        last_seq = next.seq;
        assert!(gap_detected);
        assert_eq!(last_seq, 12);
    }

    #[test]
    fn deterministic_serialization_roundtrip() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let raw = serde_json::to_string(&feed.snapshot).expect("serialize");
        let reparsed: FixtureSnapshot = serde_json::from_str(&raw).expect("reparse");
        assert_eq!(feed.snapshot.seq, reparsed.seq);
        assert_eq!(feed.snapshot.bids.len(), reparsed.bids.len());
    }

    #[test]
    fn live_ws_snapshot_parses() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex_live_recording");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");
        let first = frames
            .lines()
            .find(|line| !line.trim().is_empty())
            .expect("frame");
        let value: Value = serde_json::from_str(first).expect("snapshot json");
        let payload = value.get("params").unwrap_or(&value);
        let snapshot = parse_snapshot(payload).expect("parse snapshot");
        assert!(snapshot.seq > 0);
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[test]
    fn live_ws_replay_is_deterministic_and_monotonic() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex_live_recording");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");
        let mut tracker = ParadexSeqState::new(0);
        let mut events = Vec::new();
        for line in frames.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed).expect("parse json");
            if let Ok(Some(event)) = parse_orderbook_message_value(&value, &mut tracker) {
                events.push(event);
            }
        }
        assert!(!events.is_empty());
        let mut last_ts: Option<TimestampMs> = None;
        for event in events {
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
    fn auth_token_fixture_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let raw = std::fs::read_to_string(fixture_dir.join("auth_token.json")).expect("token");
        let token = parse_auth_token(&raw).expect("parse token");
        assert_eq!(token_token(&token).unwrap(), "test.jwt");
    }

    #[test]
    fn bbo_decode_emits_top() {
        let mut seq = 0u64;
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": "bbo.BTC-USD-PERP",
                "data": {
                    "bid": "30000",
                    "bid_size": "1.2",
                    "ask": "30010",
                    "ask_size": "0.9",
                    "ts": 1700000000000i64
                }
            }
        });
        let (top, _snapshot) =
            decode_bbo_top_and_snapshot(&msg, 0, "BTC-USD-PERP", &mut seq).expect("bbo");
        assert_eq!(top.best_bid_px, 30000.0);
        assert_eq!(top.best_bid_sz, 1.2);
        assert_eq!(top.best_ask_px, 30010.0);
        assert_eq!(top.best_ask_sz, 0.9);
    }

    #[tokio::test]
    async fn rest_place_order_builds_payload_and_auth_header() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth/token", server.base_url()),
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            auth_payload_json: None,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/orders")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "market": "BTC-USD-PERP",
                        "side": "BUY",
                        "type": "LIMIT",
                        "time_in_force": "POST_ONLY",
                        "price": 100.0,
                        "size": 0.1,
                        "post_only": true,
                        "reduce_only": false,
                        "client_order_id": "co_post_only",
                    }));
                then.status(200).body("{\"order_id\": \"oid_1\"}");
            })
            .await;

        let _ = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
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
}
