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

static MONO_START: OnceLock<Instant> = OnceLock::new();

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

fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_parsed_ns: AtomicU64,
    last_published_ns: AtomicU64,
    last_snapshot_resync_ns: AtomicU64,
}

impl Freshness {
    fn reset_for_new_connection(&self) {
        self.last_ws_rx_ns.store(0, Ordering::Relaxed);
        self.last_data_rx_ns.store(0, Ordering::Relaxed);
        self.last_parsed_ns.store(0, Ordering::Relaxed);
        self.last_published_ns.store(0, Ordering::Relaxed);
        self.last_snapshot_resync_ns
            .store(0, Ordering::Relaxed);
    }

    fn anchor_with_connect_start(&self, connect_start_ns: u64) -> u64 {
        let last_pub = self.last_published_ns.load(Ordering::Relaxed);
        let last_parsed = self.last_parsed_ns.load(Ordering::Relaxed);
        let anchor = last_pub.max(last_parsed);
        if anchor == 0 {
            connect_start_ns
        } else {
            anchor
        }
    }
}

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use k256::ecdsa::SigningKey;
use reqwest::Client;
use serde::Serialize;
use serde_json::json;
use sha3::{Digest, Keccak256};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::types::{
    FundingSource, OrderIntent, OrderPurpose, SettlementPriceKind, Side, TimeInForce, TimestampMs,
};

use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, FundingUpdate,
    LiquidationSnapshot, MarginSnapshot, MarketDataEvent, PositionSnapshot, TopOfBook,
};
use crate::live::gateway::{
    BoxFuture, LiveGatewayError, LiveRestCancelAllRequest, LiveRestCancelRequest, LiveRestClient,
    LiveRestPlaceRequest, LiveRestResponse, LiveResult,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperliquidNetwork {
    Mainnet,
    Testnet,
}

#[derive(Debug, Clone)]
pub struct HyperliquidConfig {
    pub network: HyperliquidNetwork,
    pub ws_url: String,
    pub rest_url: String,
    pub info_url: String,
    pub coin: String,
    pub n_sig_figs: u32,
    pub n_levels: u32,
    pub venue_index: usize,
    pub paper_mode: bool,
    pub private_key_hex: Option<String>,
    pub vault_address: Option<String>,
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
        let ws_url = std::env::var("HL_WS_URL").unwrap_or_else(|_| default_ws.to_string());
        let rest_url = std::env::var("HL_REST_URL").unwrap_or_else(|_| default_rest.to_string());
        let info_url = std::env::var("HL_INFO_URL").unwrap_or_else(|_| default_info.to_string());
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
            ws_url,
            rest_url,
            info_url,
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
    asset_index: tokio::sync::Mutex<Option<u32>>,
    freshness: Arc<Freshness>,
}

impl HyperliquidConnector {
    pub fn new(
        cfg: HyperliquidConfig,
        market_tx: mpsc::Sender<MarketDataEvent>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
    ) -> Self {
        Self {
            cfg,
            http: Client::new(),
            market_tx,
            exec_tx,
            account_tx: None,
            asset_index: tokio::sync::Mutex::new(None),
            freshness: Arc::new(Freshness::default()),
        }
    }

    pub fn with_account_tx(mut self, account_tx: mpsc::Sender<AccountEvent>) -> Self {
        self.account_tx = Some(account_tx);
        self
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = std::time::Instant::now();

            if let Err(err) = self.public_ws_once().await {
                eprintln!("Hyperliquid public WS error: {err}");
            }

            // FIX: Reset backoff if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                eprintln!(
                    "INFO: Hyperliquid WS session was healthy for {:?}; resetting backoff",
                    session_duration
                );
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        let freshness = self.freshness.clone();
        eprintln!(
            "INFO: Hyperliquid public WS connecting url={}",
            self.cfg.ws_url
        );
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        eprintln!(
            "INFO: Hyperliquid public WS connected url={}",
            self.cfg.ws_url
        );
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
        let (tx_int, mut rx_int) = tokio::sync::mpsc::channel::<MarketDataEvent>(HL_INTERNAL_PUB_Q);
        let pending_latest = Arc::new(tokio::sync::Mutex::new(None::<MarketDataEvent>));
        let forward_market_tx = self.market_tx.clone();
        let forward_freshness = self.freshness.clone();
        let forward_pending = pending_latest.clone();
        tokio::spawn(async move {
            while let Some(mut event) = rx_int.recv().await {
                while let Ok(next) = rx_int.try_recv() {
                    event = next;
                }
                if let Some(pending) = forward_pending.lock().await.take() {
                    event = pending;
                }
                if forward_market_tx.send(event).await.is_ok() {
                    forward_freshness
                        .last_published_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                }
            }
        });
        let try_publish = |event: MarketDataEvent| -> anyhow::Result<()> {
            match tx_int.try_send(event) {
                Ok(()) => Ok(()),
                Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                    if let Ok(mut guard) = pending_latest.try_lock() {
                        *guard = Some(event);
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
        loop {
            tokio::select! {
                biased;
                _ = &mut stale_rx => {
                    anyhow::bail!("Hyperliquid public WS stale: freshness exceeded {stale_ms}ms");
                }
                maybe = read.next() => {
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
                                        "WARN: Hyperliquid public WS non-utf8 binary frame url={}",
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
                        eprintln!("INFO: Hyperliquid public WS first message received");
                        first_message_logged = true;
                    }
                    let value = match serde_json::from_str::<serde_json::Value>(&payload) {
                        Ok(value) => value,
                        Err(err) => {
                            let snippet: String = payload.chars().take(160).collect();
                            eprintln!(
                                "WARN: Hyperliquid public WS parse error: {err} url={} snippet={}",
                                self.cfg.ws_url, snippet
                            );
                            continue;
                        }
                    };
                    let channel = value.get("channel").and_then(|v| v.as_str()).unwrap_or("");
                    if channel == "subscriptionResponse" {
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
                                self.cfg.ws_url.as_str(),
                            );
                        }
                        if let Some(snapshot) = decode_l2book_snapshot(
                            &value,
                            self.cfg.venue_index,
                            self.cfg.coin.as_str(),
                            &mut l2_seq_fallback,
                        ) {
                            freshness
                                .last_parsed_ns
                                .store(mono_now_ns(), Ordering::Relaxed);
                            have_baseline = true;
                            while let Some(buffered) = delta_buf.pop_front() {
                                try_publish(buffered)?;
                            }
                            try_publish(snapshot)?;
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
                    if let Some(parsed) = parse_l2_message_value(&value, self.cfg.venue_index) {
                        freshness
                            .last_parsed_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
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
        if self.cfg.private_key_hex.is_none() {
            return;
        }
        let mut backoff = Duration::from_secs(1);

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = std::time::Instant::now();

            if let Err(err) = self.private_ws_once().await {
                eprintln!("Hyperliquid private WS error: {err}");
            }

            // FIX: Reset backoff if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                eprintln!(
                    "INFO: Hyperliquid private WS session was healthy for {:?}; resetting backoff",
                    session_duration
                );
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn private_ws_once(&self) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
        let (mut write, mut read) = ws_stream.split();
        let sub_fills = json!({
            "method": "subscribe",
            "subscription": { "type": "userFills" }
        });
        write.send(Message::Text(sub_fills.to_string())).await?;
        let sub_orders = json!({
            "method": "subscribe",
            "subscription": { "type": "userEvents" }
        });
        write.send(Message::Text(sub_orders.to_string())).await?;
        while let Some(msg) = read.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    for event in translate_private_events(&value) {
                        let _ = self.exec_tx.send(event).await;
                    }
                    if let Some(account_tx) = self.account_tx.as_ref() {
                        if let Some(event) = translate_account_event(&value) {
                            let _ = account_tx.send(event).await;
                        }
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
                    eprintln!("Hyperliquid account polling error: {err}");
                }
            }
        }
    }

    pub async fn run_funding_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        let mut seq: u64 = 0;
        loop {
            interval.tick().await;
            match fetch_public_funding(&self.http, &self.cfg).await {
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
        if let Ok(snapshot) = fetch_l2_snapshot(&self.http, &self.cfg).await {
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
        let asset_index = self.get_asset_index().await?;
        let action = build_action(intent, asset_index)?;
        let nonce = now_ms;
        let signature = sign_action(&action, nonce, &self.cfg)?;
        let payload = json!({
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": self.cfg.vault_address,
        });
        let resp = self
            .http
            .post(self.cfg.rest_url.as_str())
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Hyperliquid order failed: {status} {body}");
        }
        Ok(())
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
        let asset_index = self.get_asset_index().await?;
        let action = build_cancel_action(intent, asset_index)?;
        let nonce = now_ms;
        let signature = sign_action(&action, nonce, &self.cfg)?;
        let payload = json!({
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": self.cfg.vault_address,
        });
        let resp = self
            .http
            .post(self.cfg.rest_url.as_str())
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Hyperliquid cancel failed: {status} {body}");
        }
        Ok(())
    }

    pub async fn cancel_all(&self, now_ms: TimestampMs) -> anyhow::Result<()> {
        if self.cfg.paper_mode {
            eprintln!("Hyperliquid paper mode cancel_all");
            return Ok(());
        }
        let asset_index = self.get_asset_index().await?;
        let action = build_cancel_all_action(asset_index);
        let nonce = now_ms;
        let signature = sign_action(&action, nonce, &self.cfg)?;
        let payload = json!({
            "action": action,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": self.cfg.vault_address,
        });
        let resp = self
            .http
            .post(self.cfg.rest_url.as_str())
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Hyperliquid cancel_all failed: {status} {body}");
        }
        Ok(())
    }

    async fn get_asset_index(&self) -> anyhow::Result<u32> {
        {
            let guard = self.asset_index.lock().await;
            if let Some(idx) = *guard {
                return Ok(idx);
            }
        }
        let idx = fetch_asset_index(&self.http, &self.cfg).await?;
        let mut guard = self.asset_index.lock().await;
        *guard = Some(idx);
        Ok(idx)
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
            });
            HyperliquidConnector::place_order(self, &intent, 0)
                .await
                .map(|_| LiveRestResponse { order_id: None })
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
                .map(|_| LiveRestResponse { order_id: None })
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
                .map(|_| LiveRestResponse { order_id: None })
                .map_err(map_rest_error)
        })
    }
}

fn map_rest_error(err: anyhow::Error) -> LiveGatewayError {
    let msg = err.to_string();
    let lower = msg.to_lowercase();
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(msg);
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(msg);
    }
    if lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(msg);
    }
    if lower.contains("timeout") || lower.contains("tempor") || lower.contains("retry") {
        return LiveGatewayError::retryable(msg);
    }
    LiveGatewayError::fatal(msg)
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

fn decode_l2book_snapshot(
    value: &serde_json::Value,
    venue_index: usize,
    default_coin: &str,
    fallback_seq: &mut u64,
) -> Option<MarketDataEvent> {
    let data = value.get("data")?;
    let levels = data.get("levels")?;
    let bids = parse_levels(levels.get(0)?)?;
    let asks = parse_levels(levels.get(1)?)?;
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

async fn fetch_l2_snapshot(
    client: &Client,
    cfg: &HyperliquidConfig,
) -> anyhow::Result<MarketDataEvent> {
    let payload = json!({
        "type": "l2Book",
        "coin": cfg.coin,
        "nSigFigs": cfg.n_sig_figs,
        "nLevels": cfg.n_levels
    });
    let resp = client
        .post(cfg.info_url.as_str())
        .json(&payload)
        .send()
        .await?;
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

async fn fetch_asset_index(client: &Client, cfg: &HyperliquidConfig) -> anyhow::Result<u32> {
    let payload = json!({ "type": "meta" });
    let resp = client
        .post(cfg.info_url.as_str())
        .json(&payload)
        .send()
        .await?;
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
            return Ok(idx as u32);
        }
    }
    anyhow::bail!("coin {} not found in Hyperliquid universe", cfg.coin);
}

async fn fetch_account_snapshot(
    client: &Client,
    cfg: &HyperliquidConfig,
) -> anyhow::Result<AccountEvent> {
    let user = cfg
        .vault_address
        .clone()
        .ok_or_else(|| anyhow::anyhow!("HL_VAULT_ADDRESS is required for account polling"))?;
    let payload = json!({ "type": "userState", "user": user });
    let resp = client
        .post(cfg.info_url.as_str())
        .json(&payload)
        .send()
        .await?;
    let value: serde_json::Value = resp.json().await?;
    let snapshot = parse_account_snapshot(&value)
        .ok_or_else(|| anyhow::anyhow!("invalid account snapshot response"))?;
    Ok(AccountEvent::Snapshot(snapshot))
}

async fn fetch_public_funding(
    client: &Client,
    cfg: &HyperliquidConfig,
) -> anyhow::Result<FundingUpdate> {
    let payload = json!({ "type": "metaAndAssetCtxs" });
    let resp = client
        .post(cfg.info_url.as_str())
        .json(&payload)
        .send()
        .await?;
    let value: serde_json::Value = resp.json().await?;
    parse_public_funding(&value, cfg).ok_or_else(|| {
        anyhow::anyhow!("invalid public funding response for coin={}", cfg.coin)
    })
}

fn parse_public_funding(value: &serde_json::Value, cfg: &HyperliquidConfig) -> Option<FundingUpdate> {
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
        settlement_price_kind: Some(SettlementPriceKind::Unknown),
        source: FundingSource::MarketDataRest,
    })
}

fn build_action(intent: &OrderIntent, asset_index: u32) -> anyhow::Result<serde_json::Value> {
    let OrderIntent::Place(place) = intent else {
        anyhow::bail!("intent not a place order");
    };
    let tif = if place.post_only {
        "Alo"
    } else {
        match place.time_in_force {
            TimeInForce::Gtc => "Gtc",
            TimeInForce::Ioc => "Ioc",
        }
    };
    let order = json!({
        "a": asset_index,
        "b": place.side == Side::Buy,
        "p": format!("{:.8}", place.price),
        "s": format!("{:.8}", place.size),
        "r": place.reduce_only,
        "t": { "limit": { "tif": tif } },
        "c": place.client_order_id,
    });
    Ok(json!({
        "type": "order",
        "orders": [order],
        "grouping": "na",
    }))
}

fn build_cancel_action(
    intent: &OrderIntent,
    asset_index: u32,
) -> anyhow::Result<serde_json::Value> {
    let OrderIntent::Cancel(cancel) = intent else {
        anyhow::bail!("intent not a cancel order");
    };
    Ok(json!({
        "type": "cancel",
        "cancels": [{
            "a": asset_index,
            "o": cancel.order_id,
        }]
    }))
}

fn build_cancel_all_action(asset_index: u32) -> serde_json::Value {
    json!({
        "type": "cancelAll",
        "asset": asset_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn hyperliquid_cancel_all_smoke() {
        use tiny_http::{Response, Server};

        let server = Server::http("127.0.0.1:0").expect("bind server");
        let addr = server.server_addr();
        let rest_url = format!("http://{}", addr);
        let info_url = rest_url.clone();
        std::thread::spawn(move || {
            for mut request in server.incoming_requests().take(2) {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if body.contains(r#""type":"meta""#) {
                    let resp = Response::from_string(r#"{"universe":[{"name":"TAO"}]}"#);
                    let _ = request.respond(resp);
                } else {
                    let resp = Response::from_string(r#"{"status":"ok"}"#);
                    let _ = request.respond(resp);
                }
            }
        });

        let cfg = HyperliquidConfig {
            network: HyperliquidNetwork::Testnet,
            ws_url: "wss://example".to_string(),
            rest_url,
            info_url,
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
    fn freshness_reset_and_anchor_behavior() {
        let freshness = Freshness::default();
        freshness
            .last_parsed_ns
            .store(123, Ordering::Relaxed);
        freshness
            .last_published_ns
            .store(456, Ordering::Relaxed);
        freshness.reset_for_new_connection();
        assert_eq!(
            freshness.last_parsed_ns.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            freshness.last_published_ns.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            freshness.last_snapshot_resync_ns.load(Ordering::Relaxed),
            0
        );

        let connect_start_ns = 1_000;
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, connect_start_ns);

        freshness
            .last_parsed_ns
            .store(2_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 2_000);

        freshness
            .last_published_ns
            .store(3_000, Ordering::Relaxed);
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
            ws_url: "wss://example".to_string(),
            rest_url: "https://example".to_string(),
            info_url: "https://example".to_string(),
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
    }
}

#[derive(Debug, Serialize)]
struct SignedPayload {
    action: serde_json::Value,
    nonce: u64,
}

fn sign_action(
    action: &serde_json::Value,
    nonce: TimestampMs,
    cfg: &HyperliquidConfig,
) -> anyhow::Result<serde_json::Value> {
    let key_hex = cfg
        .private_key_hex
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HL_PRIVATE_KEY is required for live orders"))?;
    let key_bytes = hex::decode(key_hex.trim_start_matches("0x"))?;
    let signing_key = SigningKey::from_slice(&key_bytes)?;
    let payload = SignedPayload {
        action: action.clone(),
        nonce: nonce as u64,
    };
    let packed = rmp_serde::to_vec_named(&payload)?;
    let digest = Keccak256::new().chain_update(&packed);
    let (sig, recid) = signing_key.sign_digest_recoverable(digest)?;
    let (r, s) = sig.split_bytes();
    let v: u8 = recid.to_byte();
    Ok(json!({
        "r": format!("0x{}", hex::encode(r)),
        "s": format!("0x{}", hex::encode(s)),
        "v": v,
    }))
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

pub fn translate_account_event(msg: &serde_json::Value) -> Option<AccountEvent> {
    let channel = msg.get("channel").and_then(|v| v.as_str()).unwrap_or("");
    if channel != "userState" {
        return None;
    }
    let data = msg.get("data")?;
    parse_account_snapshot(data).map(AccountEvent::Snapshot)
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
    let venue_id = fill
        .get("coin")
        .or_else(|| data.get("coin"))
        .and_then(|v| v.as_str())
        .unwrap_or("UNKNOWN")
        .to_string();
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

    Some(ExecutionEvent::Filled(super::super::types::Fill {
        venue_index,
        venue_id,
        seq,
        timestamp_ms,
        order_id,
        client_order_id,
        fill_id,
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
    let seq = data.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = data.get("time").and_then(|v| v.as_i64()).unwrap_or(0);
    let venue_id = data.get("coin").and_then(|v| v.as_str()).unwrap_or("TAO");
    let venue_index = 0;

    let positions = data
        .get("positions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pos| {
                    let symbol = pos.get("coin").and_then(|v| v.as_str()).unwrap_or("TAO");
                    let size = pos.get("size")?.as_str()?.parse::<f64>().ok()?;
                    let entry_price = pos.get("entryPx")?.as_str()?.parse::<f64>().ok()?;
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
                    let total = bal.get("total")?.as_str()?.parse::<f64>().ok()?;
                    let available = bal.get("available")?.as_str()?.parse::<f64>().ok()?;
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
    let margin_balance = margin.get("balance")?.as_str()?.parse::<f64>().ok()?;
    let margin_used = margin.get("used")?.as_str()?.parse::<f64>().ok()?;
    let margin_available = margin.get("available")?.as_str()?.parse::<f64>().ok()?;
    let margin = MarginSnapshot {
        balance_usd: margin_balance,
        used_usd: margin_used,
        available_usd: margin_available,
    };

    let liquidation = data.get("liquidation")?;
    let price_liq = liquidation
        .get("priceLiq")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok());
    let dist_liq_sigma = liquidation
        .get("distLiqSigma")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok());
    let liquidation = LiquidationSnapshot {
        price_liq,
        dist_liq_sigma,
    };

    let funding_8h = data
        .get("funding8h")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok());

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
