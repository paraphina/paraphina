//! Lighter connector (feature-gated).

pub const STUB_CONNECTOR: bool = false;
pub const SUPPORTS_MARKET: bool = true;
pub const SUPPORTS_ACCOUNT: bool = true;
pub const SUPPORTS_EXECUTION: bool = true;

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::types::{OrderIntent, OrderPurpose, Side, TimestampMs};

use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, LiquidationSnapshot,
    MarginSnapshot, MarketDataEvent, PositionSnapshot,
};
use super::super::gateway::{
    LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest, LiveRestCancelRequest,
    LiveRestClient, LiveRestPlaceRequest, LiveRestResponse,
};

#[derive(Debug, Clone)]
pub struct LighterConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub venue_id: String,
    pub paper_mode: bool,
}

impl LighterConfig {
    pub fn from_env() -> Self {
        let ws_url =
            std::env::var("LIGHTER_WS_URL").unwrap_or_else(|_| "wss://api.lighter.xyz/ws".to_string());
        let rest_url = std::env::var("LIGHTER_HTTP_BASE_URL")
            .or_else(|_| std::env::var("LIGHTER_REST_URL"))
            .unwrap_or_else(|_| "https://api.lighter.xyz".to_string());
        let venue_id = std::env::var("LIGHTER_VENUE").unwrap_or_else(|_| "LIGHTER".to_string());
        let paper_mode = std::env::var("LIGHTER_PAPER_MODE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        Self {
            ws_url,
            rest_url,
            venue_id,
            paper_mode,
        }
    }
}

#[derive(Debug)]
pub struct LighterConnector {
    cfg: LighterConfig,
    http: Client,
    market_tx: mpsc::Sender<MarketDataEvent>,
    exec_tx: mpsc::Sender<ExecutionEvent>,
    account_tx: Option<mpsc::Sender<AccountEvent>>,
}

impl LighterConnector {
    pub fn new(
        cfg: LighterConfig,
        market_tx: mpsc::Sender<MarketDataEvent>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
    ) -> Self {
        Self {
            cfg,
            http: Client::new(),
            market_tx,
            exec_tx,
            account_tx: None,
        }
    }

    pub fn with_account_tx(mut self, account_tx: mpsc::Sender<AccountEvent>) -> Self {
        self.account_tx = Some(account_tx);
        self
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        loop {
            if let Err(err) = self.public_ws_once().await {
                eprintln!("Lighter public WS error: {err}");
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        let (mut write, mut read) = ws_stream.split();
        let sub = json!({
            "type": "subscribe",
            "channel": "l2",
            "venue": self.cfg.venue_id
        });
        write.send(Message::Text(sub.to_string())).await?;
        let mut tracker = LighterSeqTracker::new();
        while let Some(msg) = read.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                if let Some(parsed) = parse_l2_message(&text, &self.cfg.venue_id) {
                    let outcome = tracker.on_message(parsed);
                    if let Some(event) = outcome.event {
                        let _ = self.market_tx.send(event).await;
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
        start_ms: i64,
        step_ms: i64,
        ticks: u64,
    ) {
        let mut seq: u64 = 1;
        for tick in 0..ticks {
            let now_ms = start_ms + step_ms.saturating_mul(tick as i64);
            for raw in &self.messages {
                if let Some(parsed) = parse_l2_message(raw, "LIGHTER") {
                    let event = override_market_event(parsed.event, seq, now_ms);
                    seq = seq.wrapping_add(1);
                    let _ = market_tx.send(event).await;
                }
            }
            tokio::task::yield_now().await;
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
            if msg.seq <= prev {
                return LighterSeqOutcome { event: None };
            }
        }
        self.last_seq = Some(msg.seq);
        LighterSeqOutcome {
            event: Some(msg.event),
        }
    }
}

pub fn parse_l2_message(text: &str, venue_id: &str) -> Option<ParsedL2Message> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let msg_type = value.get("type")?.as_str()?;
    let seq = value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = value.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
    let venue_index = 0;
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

fn override_market_event(event: MarketDataEvent, seq: u64, timestamp_ms: TimestampMs) -> MarketDataEvent {
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
        let price = level.get(0)?.as_f64()?;
        let size = level.get(1)?.as_f64()?;
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
        let price = change.get("price")?.as_f64()?;
        let size = change.get("size")?.as_f64()?;
        out.push(BookLevelDelta { side, price, size });
    }
    Some(out)
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
                client_order_id: value.get("client_order_id").and_then(|v| v.as_str()).map(|v| v.to_string()),
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
            order_id: value.get("order_id").and_then(|v| v.as_str()).map(|v| v.to_string()),
            client_order_id: value.get("client_order_id").and_then(|v| v.as_str()).map(|v| v.to_string()),
            fill_id: value.get("fill_id").and_then(|v| v.as_str()).map(|v| v.to_string()),
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
    let resp = client
        .get(format!("{}/account", cfg.rest_url))
        .send()
        .await?;
    let value: serde_json::Value = resp.json().await?;
    let snapshot = parse_account_snapshot(&value, &cfg.venue_id)
        .ok_or_else(|| anyhow::anyhow!("invalid account snapshot"))?;
    Ok(AccountEvent::Snapshot(snapshot))
}

pub fn parse_account_snapshot(
    data: &serde_json::Value,
    venue_id: &str,
) -> Option<AccountSnapshot> {
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
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse { order_id: None });
            }
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
            });
            let payload = json!({ "intent": format!("{intent:?}") });
            let resp = self
                .http
                .post(format!("{}/orders", self.cfg.rest_url))
                .json(&payload)
                .send()
                .await
                .map_err(|err| LiveGatewayError::retryable(format!("rest_error: {err}")))?;
            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(map_rest_error(&body));
            }
            Ok(LiveRestResponse { order_id: None })
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse { order_id: None });
            }
            let payload = json!({ "order_id": req.order_id });
            let resp = self
                .http
                .post(format!("{}/cancel", self.cfg.rest_url))
                .json(&payload)
                .send()
                .await
                .map_err(|err| LiveGatewayError::retryable(format!("rest_error: {err}")))?;
            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(map_rest_error(&body));
            }
            Ok(LiveRestResponse { order_id: None })
        })
    }

    fn cancel_all(
        &self,
        _req: LiveRestCancelAllRequest,
    ) -> super::super::gateway::BoxFuture<'_, super::super::gateway::LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if self.cfg.paper_mode {
                return Ok(LiveRestResponse { order_id: None });
            }
            let resp = self
                .http
                .post(format!("{}/cancel_all", self.cfg.rest_url))
                .send()
                .await
                .map_err(|err| LiveGatewayError::retryable(format!("rest_error: {err}")))?;
            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(map_rest_error(&body));
            }
            Ok(LiveRestResponse { order_id: None })
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
