//! Lighter connector (feature-gated).

pub const STUB_CONNECTOR: bool = false;
pub const SUPPORTS_MARKET: bool = true;
pub const SUPPORTS_ACCOUNT: bool = true;
pub const SUPPORTS_EXECUTION: bool = true;

const LIGHTER_MARKET_PUB_QUEUE_CAP: usize = 256;
const LIGHTER_MARKET_PUB_DRAIN_MAX: usize = 64;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs};

use super::super::gateway::{
    LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest, LiveRestCancelRequest,
    LiveRestClient, LiveRestPlaceRequest, LiveRestResponse,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, LiquidationSnapshot,
    MarginSnapshot, MarketDataEvent, PositionSnapshot, TopOfBook,
};
use super::lighter_nonce::{load_last_nonce, store_last_nonce, LighterNonceManager};
use super::lighter_signer::{
    LighterSignerClient, SignCancelAllRequest, SignCancelOrderRequest, SignCreateOrderRequest,
    SignedTx,
};
use crate::live::MarketPublisher;

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

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
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

#[derive(Debug)]
pub struct LighterConnector {
    cfg: LighterConfig,
    http: Client,
    market_publisher: MarketPublisher,
    exec_tx: mpsc::Sender<ExecutionEvent>,
    account_tx: Option<mpsc::Sender<AccountEvent>>,
    nonce: Arc<LighterNonceManager>,
    nonce_path: Option<PathBuf>,
    signer: Option<LighterSignerClient>,
}

impl LighterConnector {
    pub fn new(
        cfg: LighterConfig,
        market_tx: mpsc::Sender<MarketDataEvent>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
    ) -> Self {
        let is_fixture = std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some();
        let market_publisher = MarketPublisher::new(
            LIGHTER_MARKET_PUB_QUEUE_CAP,
            LIGHTER_MARKET_PUB_DRAIN_MAX,
            market_tx.clone(),
            Some(Arc::new(move || is_fixture)),
            Arc::new(|event: &MarketDataEvent| matches!(event, MarketDataEvent::L2Delta(_))),
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
        Self {
            cfg,
            http: Client::new(),
            market_publisher,
            exec_tx,
            account_tx: None,
            nonce,
            nonce_path,
            signer,
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

    async fn resolve_market_decimals(&self, market_id: u64) -> anyhow::Result<(u32, u32)> {
        let (orderbooks, source_url) = fetch_lighter_orderbooks_with_fallbacks(
            &self.http,
            &self.cfg.rest_url,
            &self.cfg.ws_url,
        )
        .await?;
        let Some(info) = orderbooks.iter().find(|info| info.market_id == market_id) else {
            anyhow::bail!("Lighter market_id not found for decimals: {}", market_id);
        };
        let price = info.price_decimals.ok_or_else(|| {
            anyhow::anyhow!(
                "Lighter price decimals missing for market_id={} source_url={}",
                market_id,
                source_url
            )
        })?;
        let size = info.size_decimals.ok_or_else(|| {
            anyhow::anyhow!(
                "Lighter size decimals missing for market_id={} source_url={}",
                market_id,
                source_url
            )
        })?;
        Ok((price, size))
    }

    async fn submit_sendtx(
        &self,
        signed: SignedTx,
    ) -> super::super::gateway::LiveResult<LiveRestResponse> {
        let resp = self
            .http
            .post(sendtx_url(&self.cfg))
            .json(&serde_json::json!({
                "tx_type": signed.tx_type,
                "tx_info": signed.tx_info,
            }))
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("sendtx_error: {err}")))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(&body));
        }
        let value = serde_json::from_str::<serde_json::Value>(&body)
            .map_err(|err| LiveGatewayError::retryable(format!("sendtx_parse_error: {err}")))?;
        let order_id = value
            .get("order_id")
            .or_else(|| value.get("orderId"))
            .and_then(|v| v.as_str())
            .map(|v| v.to_string());
        Ok(LiveRestResponse { order_id })
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        self.market_publisher.publish_market(event).await
    }

    async fn resolve_market_id_and_symbol(&self) -> anyhow::Result<(String, u64)> {
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

        if let Some(market_id) = market_id_env {
            let symbol = market_symbol_env.clone().or_else(|| {
                orderbooks
                    .iter()
                    .find(|info| info.market_id == market_id)
                    .map(|info| info.symbol.clone())
            });
            let symbol = symbol.unwrap_or_else(|| "UNKNOWN".to_string());
            eprintln!(
                "INFO: Lighter resolving market id symbol={} source_url=env:LIGHTER_MARKET_ID",
                symbol
            );
            eprintln!(
                "INFO: Lighter market id resolved symbol={} market_id={} source_url=env:LIGHTER_MARKET_ID",
                symbol, market_id
            );
            return Ok((symbol, market_id));
        }

        let symbol = market_symbol_env.unwrap_or_else(|| "BTC-USD".to_string());
        eprintln!(
            "INFO: Lighter resolving market id symbol={} source_url={}",
            symbol, source_url
        );
        let normalized = normalize_lighter_symbol(&symbol);
        let found = orderbooks
            .iter()
            .find(|info| normalize_lighter_symbol(&info.symbol) == normalized)
            .cloned();
        let Some(info) = found else {
            let available: Vec<String> = orderbooks
                .iter()
                .take(15)
                .map(|info| info.symbol.clone())
                .collect();
            eprintln!(
                "WARN: Lighter market id not found requested={} available_symbols={:?}",
                normalized, available
            );
            anyhow::bail!(
                "LIGHTER_MARKET not found in orderBooks response: {}",
                symbol
            );
        };
        eprintln!(
            "INFO: Lighter market id resolved symbol={} market_id={} source_url={}",
            info.symbol, info.market_id, source_url
        );
        Ok((info.symbol, info.market_id))
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let mut subscribe_failures = 0usize;
        let mut logged_subscribe_failure = false;
        loop {
            if let Err(err) = self.public_ws_once().await {
                let msg = err.to_string();
                if msg.contains("Lighter subscribe failed") {
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
                    eprintln!("Lighter public WS error: {err}");
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        let (market_symbol, market_id) = self.resolve_market_id_and_symbol().await?;
        eprintln!("INFO: Lighter public WS connecting url={}", self.cfg.ws_url);
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        eprintln!("INFO: Lighter public WS connected url={}", self.cfg.ws_url);
        let (mut write, mut read) = ws_stream.split();
        let channel = build_order_book_channel(market_id);
        let sub = json!({
            "type": "subscribe",
            "channel": channel,
        });
        write.send(Message::Text(sub.to_string())).await?;
        let mut subscribed = false;
        let mut first_message_logs = 0usize;
        let mut tracker = LighterSeqTracker::new();
        let mut first_book_update_logged = false;
        let mut first_message_logged = false;
        let mut logged_non_utf8_binary = false;
        let mut first_decoded_top_logged = false;
        let mut first_json_pong_logged = false;
        let mut decode_miss_count = 0usize;
        let mut seq_fallback: u64 = 0;
        loop {
            let msg = match read.next().await {
                Some(Ok(msg)) => msg,
                Some(Err(err)) => {
                    eprintln!("Lighter public WS read error: {err}");
                    break;
                }
                None => {
                    eprintln!("Lighter public WS stream ended");
                    break;
                }
            };
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
                        anyhow::bail!("Lighter subscribe failed: invalid channel");
                    }
                }
            }
            if !subscribed && is_lighter_book_message(&value) {
                subscribed = true;
                eprintln!(
                    "INFO: Lighter subscribe ok channel=order_book/{} symbol={} market_id={}",
                    market_id, market_symbol, market_id
                );
            }
            if subscribed {
                if let Some(top) = decode_order_book_top(&value) {
                    if !first_decoded_top_logged {
                        eprintln!(
                            "FIRST_DECODED_TOP venue=lighter bid_px={} bid_sz={} ask_px={} ask_sz={}",
                            top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                        );
                        first_decoded_top_logged = true;
                    }
                } else if decode_miss_count < 3 && has_lighter_book_fields(&value) {
                    decode_miss_count += 1;
                    log_decode_miss(
                        "Lighter",
                        &value,
                        &payload,
                        decode_miss_count,
                        self.cfg.ws_url.as_str(),
                    );
                }
                if let Some(parsed) = decode_order_book_channel_message(
                    &value,
                    self.cfg.venue_index,
                    &self.cfg.venue_id,
                    &mut seq_fallback,
                ) {
                    let outcome = tracker.on_message(parsed);
                    if let Some(event) = outcome.event {
                        if !first_book_update_logged {
                            eprintln!("INFO: Lighter public WS first book update");
                            first_book_update_logged = true;
                        }
                        if let Err(err) = self.publish_market(event).await {
                            eprintln!("Lighter public WS market send failed: {err}");
                        }
                    }
                    continue;
                }
                if let Some(parsed) = decode_order_book_snapshot(
                    &value,
                    self.cfg.venue_index,
                    &self.cfg.venue_id,
                    &mut seq_fallback,
                ) {
                    let outcome = tracker.on_message(parsed);
                    if let Some(event) = outcome.event {
                        if !first_book_update_logged {
                            eprintln!("INFO: Lighter public WS first book update");
                            first_book_update_logged = true;
                        }
                        if let Err(err) = self.publish_market(event).await {
                            eprintln!("Lighter public WS market send failed: {err}");
                        }
                    }
                }
            }
            if let Some(parsed) =
                parse_l2_message_value(&value, &self.cfg.venue_id, self.cfg.venue_index)
            {
                let outcome = tracker.on_message(parsed);
                if let Some(event) = outcome.event {
                    if !first_book_update_logged {
                        eprintln!("INFO: Lighter public WS first book update");
                        first_book_update_logged = true;
                    }
                    if let Err(err) = self.publish_market(event).await {
                        eprintln!("Lighter public WS market send failed: {err}");
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn run_account_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        loop {
            interval.tick().await;
            let Some(account_tx) = self.account_tx.as_ref() else {
                continue;
            };
            match fetch_account_snapshot(&self.http, &self.cfg).await {
                Ok(snapshot) => {
                    let _ = account_tx.send(snapshot).await;
                }
                Err(err) => {
                    eprintln!("Lighter account polling error: {err}");
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
        loop {
            if let Err(err) = self.private_ws_once().await {
                eprintln!("Lighter private WS error: {err}");
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn private_ws_once(&self) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        let (_write, mut read) = ws_stream.split();
        while let Some(msg) = read.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                if let Some(event) = translate_private_event(&text) {
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
    let timestamp_ms = value
        .get("timestamp")
        .or_else(|| value.get("ts"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
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
            MarketDataEvent::FundingUpdate(funding)
        }
    }
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

fn parse_f64_value(value: &serde_json::Value) -> Option<f64> {
    if let Some(raw) = value.as_f64() {
        return Some(raw);
    }
    if let Some(raw) = value.as_str() {
        return raw.parse::<f64>().ok();
    }
    None
}

fn decode_order_book_snapshot(
    value: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    seq_fallback: &mut u64,
) -> Option<ParsedL2Message> {
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
    let seq = value
        .get("seq")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| {
            *seq_fallback = seq_fallback.wrapping_add(1);
            *seq_fallback
        });
    let timestamp_ms = value
        .get("timestamp")
        .or_else(|| value.get("ts"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
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

fn json_ping_response(value: &serde_json::Value) -> Option<&'static str> {
    let msg_type = value.get("type").and_then(|v| v.as_str());
    if msg_type == Some("ping") {
        Some(r#"{"type":"pong"}"#)
    } else {
        None
    }
}

/// Decode Lighter order_book channel messages.
///
/// **IMPORTANT**: Lighter sends full orderbook state in every message, not incremental deltas.
/// Therefore, we ALWAYS emit L2Snapshot to ensure stale price levels are replaced, not accumulated.
/// Previously, this function emitted L2Delta after the first snapshot, which caused stale bid/ask
/// levels to persist and resulted in crossed books (negative spread).
fn decode_order_book_channel_message(
    value: &serde_json::Value,
    venue_index: usize,
    venue_id: &str,
    seq_fallback: &mut u64,
) -> Option<ParsedL2Message> {
    let channel = value.get("channel").and_then(|v| v.as_str())?;
    if !channel.starts_with("order_book:") {
        return None;
    }
    let order_book = value.get("order_book")?;
    let bids = match order_book.get("bids") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    let asks = match order_book.get("asks") {
        Some(value) => parse_levels_from_objects(value)?,
        None => Vec::new(),
    };
    *seq_fallback = seq_fallback.wrapping_add(1);
    let seq = *seq_fallback;
    let timestamp_ms = value
        .get("timestamp")
        .or_else(|| value.get("ts"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    // Always emit L2Snapshot since Lighter sends full book state each message.
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
    for base in bases {
        let base = base.trim_end_matches('/').to_string();
        for endpoint in endpoints {
            let url = format!("{base}{endpoint}");
            eprintln!("INFO: Lighter resolving market id attempt url={}", url);
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
                    let snippet: String = body.chars().take(160).collect();
                    eprintln!(
                        "WARN: Lighter orderBooks fetch failed status={} url={} snippet={}",
                        status, url, snippet
                    );
                }
                Err(err) => {
                    eprintln!(
                        "WARN: Lighter orderBooks fetch error url={} err={}",
                        url, err
                    );
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
        if let Some(obj) = entry.as_object() {
            let price = obj.get("price").or_else(|| obj.get("px"))?;
            let size = obj.get("size").or_else(|| obj.get("sz"))?;
            out.push(BookLevel {
                price: parse_f64_value(price)?,
                size: parse_f64_value(size)?,
            });
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

#[derive(Debug, Serialize, Deserialize)]
struct LighterAccountQuery {
    account_index: u64,
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
    use std::fs;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn order_book_channel_formats() {
        assert_eq!(build_order_book_channel(42), "order_book/42");
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
        let server = MockServer::start_async().await;
        let payload = serde_json::json!({
            "seq": 1,
            "ts": 2,
            "positions": [],
            "margin": { "balance": 1000.0, "used": 10.0, "available": 990.0 },
            "liquidation": { "price_liq": 50.0, "dist_liq_sigma": 1.2 },
            "balances": [{ "asset": "USD", "total": 1000.0, "available": 990.0 }]
        });
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/api/v1/account")
                    .query_param("account_index", "123")
                    .header("Authorization", "Bearer t");
                then.status(200).json_body(payload);
            })
            .await;
        let cfg = LighterConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            market: "BTC-USD".to_string(),
            venue_id: "LIGHTER".to_string(),
            venue_index: 0,
            paper_mode: true,
            api_key_index: Some(1),
            account_index: Some(123),
            api_private_key_hex: Some("deadbeef".to_string()),
            auth_token: Some("t".to_string()),
            nonce_path: None,
            signer_url: None,
        };
        let client = Client::new();
        let event = fetch_account_snapshot(&client, &cfg)
            .await
            .expect("snapshot");
        let AccountEvent::Snapshot(snapshot) = event;
        assert_eq!(snapshot.venue_id, "LIGHTER");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_place_order_calls_signer_then_sendtx() {

          let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
          std::env::remove_var("LIGHTER_MARKET_ID");
          std::env::remove_var("LIGHTER_MARKET");
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
                    .body_contains("\"base_amount\":1234");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":14,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("\"tx_type\":14")
                    .body_contains("\"signed\":true");
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
            post_only: false,
            reduce_only: false,
            client_order_id: "42".to_string(),
        };
        let resp = connector.place_order(req).await.expect("place");
        assert_eq!(resp.order_id.as_deref(), Some("abc"));
        orderbooks.assert_hits_async(2).await;
        sign.assert_async().await;
        sendtx.assert_async().await;
    }

    #[tokio::test]
    async fn lighter_cancel_order_calls_signer_then_sendtx() {
        let api = MockServer::start_async().await;
        let signer = MockServer::start_async().await;
        let sign = signer
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/sign")
                    .body_contains("\"op\":\"cancel_order\"")
                    .body_contains("\"account_index\":123")
                    .body_contains("\"api_key_index\":1")
                    .body_contains("\"order_index\":55");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":15,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("\"tx_type\":15")
                    .body_contains("\"signed\":true");
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
        sign.assert_async().await;
        sendtx.assert_async().await;
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
                    .body_contains("\"api_key_index\":1");
                then.status(200)
                    .json_body(serde_json::json!({"tx_type":16,"tx_info":{"signed":true}}));
            })
            .await;
        let sendtx = api
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/api/v1/sendTx")
                    .body_contains("\"tx_type\":16")
                    .body_contains("\"signed\":true");
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
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq)
                .expect("snapshot");
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
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq)
                .expect("snapshot");
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
            decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq)
                .expect("snapshot");
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
    fn order_book_channel_always_emits_snapshot() {
        // First message with full book
        let first_value = serde_json::json!({
            "channel": "order_book:1",
            "offset": 100,
            "timestamp": 1700000000123i64,
            "order_book": {
                "bids": [{"price":"100.0","size":"2.0"}],
                "asks": [{"price":"101.0","size":"3.0"}]
            }
        });
        // Second message with different full book (no bids)
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
        let first = decode_order_book_channel_message(&first_value, 3, "LIGHTER", &mut seq)
            .expect("first");
        match first.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.bids.len(), 1);
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot for first message"),
        }
        // Second message MUST also be snapshot (not delta) to avoid stale level accumulation
        let second = decode_order_book_channel_message(&second_value, 3, "LIGHTER", &mut seq)
            .expect("second");
        match second.event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                // Bids should be empty (not accumulated from first message)
                assert_eq!(snapshot.bids.len(), 0, "stale bids must not persist");
                assert_eq!(snapshot.asks.len(), 1);
            }
            _ => panic!("expected snapshot for second message (always snapshot, never delta)"),
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
        let first = decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq)
            .expect("first");
        let second = decode_order_book_channel_message(&value, 3, "LIGHTER", &mut seq)
            .expect("second");
        assert!(second.seq > first.seq);
    }

    /// Regression test: Lighter sends full orderbook state each message.
    /// If we treated these as deltas, stale bid levels would accumulate and cause crossed books.
    /// This test verifies that when the best bid drops from 110 to 100, the old 110 bid is removed.
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
        let parsed1 = decode_order_book_channel_message(&msg1, 0, "LIGHTER", &mut seq)
            .expect("msg1");
        match parsed1.event {
            MarketDataEvent::L2Snapshot(snap) => {
                book.apply_snapshot(&snap.bids, &snap.asks, snap.seq).unwrap();
            }
            _ => panic!("expected snapshot"),
        }
        assert_eq!(book.best_bid().unwrap().price, 110.0);
        assert_eq!(book.best_ask().unwrap().price, 111.0);
        let spread1 = book.best_ask().unwrap().price - book.best_bid().unwrap().price;
        assert!(spread1 > 0.0, "spread must be positive");

        // Apply second message
        let parsed2 = decode_order_book_channel_message(&msg2, 0, "LIGHTER", &mut seq)
            .expect("msg2");
        match parsed2.event {
            MarketDataEvent::L2Snapshot(snap) => {
                book.apply_snapshot(&snap.bids, &snap.asks, snap.seq).unwrap();
            }
            _ => panic!("expected snapshot for second message too"),
        }

        // CRITICAL: The old bid at 110 must NOT be present.
        // If it were (due to delta semantics), we'd have:
        //   best_bid = 110 (stale), best_ask = 101 → spread = -9 (CROSSED!)
        assert_eq!(book.best_bid().unwrap().price, 100.0, "stale bid at 110 must be gone");
        assert_eq!(book.best_ask().unwrap().price, 101.0);
        let spread2 = book.best_ask().unwrap().price - book.best_bid().unwrap().price;
        assert!(spread2 > 0.0, "spread must be positive after update");
        assert_eq!(spread2, 1.0, "spread should be 101 - 100 = 1");
    }
}

pub fn translate_private_event(text: &str) -> Option<ExecutionEvent> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let msg_type = value.get("type")?.as_str()?;
    let seq = value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = value.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let venue_id = "LIGHTER".to_string();
    let venue_index = 0;
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
        "fill" => Some(ExecutionEvent::Filled(super::super::types::Fill {
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
            side: parse_side(value.get("side")?)?,
            price: value.get("price")?.as_f64()?,
            size: value.get("size")?.as_f64()?,
            purpose: parse_purpose(value.get("purpose"))?,
            fee_bps: value.get("fee_bps").and_then(|v| v.as_f64()).unwrap_or(0.0),
        })),
        _ => None,
    }
}

fn parse_side(value: &serde_json::Value) -> Option<Side> {
    let raw = value.as_str()?;
    match raw {
        "buy" | "Buy" | "B" => Some(Side::Buy),
        "sell" | "Sell" | "S" => Some(Side::Sell),
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
    let query = LighterAccountQuery { account_index };
    let mut req = client.get(account_url(cfg)).query(&query);
    if let Some(token) = cfg.auth_token.as_ref() {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await?;
    let value: serde_json::Value = resp.json().await?;
    let snapshot = parse_account_snapshot(&value, &cfg.venue_id)
        .ok_or_else(|| anyhow::anyhow!("invalid account snapshot"))?;
    Ok(AccountEvent::Snapshot(snapshot))
}

pub fn parse_account_snapshot(data: &serde_json::Value, venue_id: &str) -> Option<AccountSnapshot> {
    let seq = data.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = data.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let venue_index = 0;
    let positions = data
        .get("positions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pos| {
                    let symbol = pos.get("symbol")?.as_str()?;
                    let size = pos.get("size")?.as_f64()?;
                    let entry_price = pos.get("entry_px")?.as_f64()?;
                    Some(PositionSnapshot {
                        symbol: symbol.to_string(),
                        size,
                        entry_price,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let margin = data.get("margin")?;
    let margin = MarginSnapshot {
        balance_usd: margin.get("balance")?.as_f64()?,
        used_usd: margin.get("used")?.as_f64()?,
        available_usd: margin.get("available")?.as_f64()?,
    };
    let liquidation = data.get("liquidation")?;
    let liquidation = LiquidationSnapshot {
        price_liq: liquidation.get("price_liq").and_then(|v| v.as_f64()),
        dist_liq_sigma: liquidation.get("dist_liq_sigma").and_then(|v| v.as_f64()),
    };
    let funding_8h = data.get("funding_8h").and_then(|v| v.as_f64());
    let balances = data
        .get("balances")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|bal| {
                    let asset = bal.get("asset")?.as_str()?;
                    let total = bal.get("total")?.as_f64()?;
                    let available = bal.get("available")?.as_f64()?;
                    Some(BalanceSnapshot {
                        asset: asset.to_string(),
                        total,
                        available,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
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

impl LiveRestClient for LighterConnector {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>>
    {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse { order_id: None });
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
                reduce_only: if req.reduce_only { 1 } else { 0 },
                trigger_price: None,
                order_expiry: None,
                expired_at,
            };
            let signed = signer
                .sign_create_order(sign_req)
                .await
                .map_err(|err| LiveGatewayError::retryable(format!("signer_error: {err}")))?;
            let resp = self.submit_sendtx(signed).await?;
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
                return Ok(LiveRestResponse { order_id: None });
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
            let order_index = req.order_id.parse::<u64>().map_err(|_| {
                LiveGatewayError::fatal("lighter: order_id must be numeric for signer bridge")
            })?;
            let expired_at = now_ms().saturating_add(60_000);
            let sign_req = SignCancelOrderRequest {
                op: "cancel_order".to_string(),
                account_index,
                api_key_index,
                nonce: self.next_nonce(),
                order_index: Some(order_index),
                client_order_index: None,
                expired_at,
            };
            let signed = signer
                .sign_cancel_order(sign_req)
                .await
                .map_err(|err| LiveGatewayError::retryable(format!("signer_error: {err}")))?;
            let resp = self.submit_sendtx(signed).await?;
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
                return Ok(LiveRestResponse { order_id: None });
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
            let now_ms = now_ms();
            let sign_req = SignCancelAllRequest {
                op: "cancel_all".to_string(),
                account_index,
                api_key_index,
                nonce: self.next_nonce(),
                cancel_all_time_in_force: 0,
                cancel_all_time: now_ms,
                expired_at: now_ms.saturating_add(60_000),
            };
            let signed = signer
                .sign_cancel_all(sign_req)
                .await
                .map_err(|err| LiveGatewayError::retryable(format!("signer_error: {err}")))?;
            let resp = self.submit_sendtx(signed).await?;
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
