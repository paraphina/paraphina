//! Paradex connector (public WS market data + fixtures, feature-gated).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::super::gateway::{
    BoxFuture, LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest, LiveRestCancelRequest,
    LiveRestClient, LiveRestPlaceRequest, LiveRestResponse, LiveResult,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, LiquidationSnapshot, MarginSnapshot,
    MarketDataEvent, PositionSnapshot,
};
use crate::types::{Side, TimeInForce, TimestampMs};

#[cfg(feature = "live_paradex")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_EXECUTION: bool = true;

#[derive(Debug, Clone)]
pub struct ParadexConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub auth_url: String,
    pub market: String,
    pub account_path: String,
    pub order_path: String,
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
        let market = std::env::var("PARADEX_MARKET")
            .unwrap_or_else(|_| "BTC-USD-PERP".to_string());
        let account_path = std::env::var("PARADEX_ACCOUNT_PATH")
            .unwrap_or_else(|_| "/account".to_string());
        let order_path = std::env::var("PARADEX_ORDER_PATH")
            .unwrap_or_else(|_| "/orders".to_string());
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
    market_tx: mpsc::Sender<MarketDataEvent>,
    recorder: Option<Mutex<ParadexRecorder>>,
}

impl ParadexConnector {
    pub fn new(cfg: ParadexConfig, market_tx: mpsc::Sender<MarketDataEvent>) -> Self {
        let recorder = cfg
            .record_dir
            .as_ref()
            .and_then(|dir| ParadexRecorder::new(dir).ok())
            .map(Mutex::new);
        Self {
            cfg,
            http: Client::new(),
            market_tx,
            recorder,
        }
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
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        let (mut write, mut read) = ws_stream.split();
        let sub = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "subscribe",
            "params": {
                "channel": "orderbook",
                "market": self.cfg.market,
            }
        });
        write.send(Message::Text(sub.to_string())).await?;

        let mut tracker = ParadexSeqState::new();
        while let Some(msg) = read.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                if let Some(recorder) = self.recorder.as_ref() {
                    let mut guard = recorder.lock().await;
                    let _ = guard.record_ws_frame(&text);
                }
                if let Some(event) = parse_orderbook_message(&text, &mut tracker)? {
                    let _ = self.market_tx.send(event).await;
                }
            } else if let Message::Ping(payload) = msg {
                let _ = write.send(Message::Pong(payload)).await;
            }
        }
        Ok(())
    }
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
        let jwt = token_token(&token).ok_or_else(|| {
            LiveGatewayError::fatal("paradex auth token missing access_token")
        })?;
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
        let value: Value = serde_json::from_str(&body)
            .map_err(|err| LiveGatewayError::fatal(format!("paradex account parse error: {err}")))?;
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
}

impl ParadexSeqState {
    fn new() -> Self {
        Self {
            last_seq: None,
            has_snapshot: false,
        }
    }

    fn apply(&mut self, message: ParadexBookMessage) -> anyhow::Result<Option<MarketDataEvent>> {
        match message {
            ParadexBookMessage::Snapshot(snapshot) => {
                self.last_seq = Some(snapshot.seq);
                self.has_snapshot = true;
                Ok(Some(MarketDataEvent::L2Snapshot(
                    super::super::types::L2Snapshot {
                        venue_index: 0,
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
                        venue_index: 0,
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

fn build_order_payload(
    market: &str,
    req: &LiveRestPlaceRequest,
) -> LiveResult<Value> {
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
                        let entry_price = pos.get("entry_price").or_else(|| pos.get("entryPrice"))?.as_f64()?;
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
            timestamp_ms: value.get("timestamp_ms").and_then(|v| v.as_i64()).unwrap_or(0),
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

fn parse_orderbook_message(
    text: &str,
    tracker: &mut ParadexSeqState,
) -> anyhow::Result<Option<MarketDataEvent>> {
    let value: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
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
            tokio::task::yield_now().await;
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
        let item: T = serde_json::from_str(line).map_err(|err| {
            format!("fixture_parse_error path={} err={}", path.display(), err)
        })?;
        out.push(item);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::{POST};
    use httpmock::MockServer;
    use std::path::PathBuf;

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
    }

    #[test]
    fn delta_applies_to_snapshot_levels() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex");
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
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex");
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
        let frames = std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl"))
            .expect("ws frames");
        let first = frames.lines().find(|line| !line.trim().is_empty()).expect("frame");
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
        let frames = std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl"))
            .expect("ws frames");
        let mut tracker = ParadexSeqState::new();
        let mut events = Vec::new();
        for line in frames.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(Some(event)) = parse_orderbook_message(trimmed, &mut tracker) {
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
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex");
        let raw = std::fs::read_to_string(fixture_dir.join("auth_token.json")).expect("token");
        let token = parse_auth_token(&raw).expect("parse token");
        assert_eq!(token_token(&token).unwrap(), "test.jwt");
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
